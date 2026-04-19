import re
import random
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertModel, BertTokenizerFast
from TorchCRF import CRF


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
LABELS = ["O", "B-ENT", "I-ENT"]
LABEL2ID = {v: i for i, v in enumerate(LABELS)}
ID2LABEL = {i: v for v, i in LABEL2ID.items()}


@dataclass
class DomainConfig:
    domain_name: str
    source_file: str
    output_csv: str
    type_rules: List[Tuple[str, List[str], List[str]]]
    default_type: str
    seed_entities: List[str] = field(default_factory=list)
    regex_entities: List[str] = field(default_factory=list)
    ngram_min_freq: int = 2
    ngram_min_len: int = 2
    ngram_max_len: int = 8
    use_ngrams: bool = True
    max_candidate_terms: int = 1200
    max_len: int = 128
    batch_size: int = 16
    epochs: int = 3
    lr: float = 3e-4
    seed: int = 42
    keep_default_type: bool = True
    min_mention_count: int = 1
    min_mention_count_default: int = 2
    min_entity_len: int = 2
    max_entity_len: int = 30
    append_only: bool = False
    allowed_entity_types: List[str] = field(default_factory=list)
    use_seed_entities: bool = True
    use_regex_entities: bool = True
    use_type_rules: bool = True
    merge_lexicon_entities: bool = True
    pmi_min: float = 3.5
    entropy_min: float = 0.8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def clean_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def split_sentences(text: str, max_chars: int) -> List[str]:
    chunks: List[str] = []
    for blk in text.split("\n"):
        blk = blk.strip()
        if not blk:
            continue
        segs = re.split(r"(?<=[銆傦紒锛燂紱;.!?])", blk)
        for seg in segs:
            seg = seg.strip()
            if not seg:
                continue
            if len(seg) <= max_chars:
                chunks.append(seg)
            else:
                for i in range(0, len(seg), max_chars):
                    p = seg[i : i + max_chars].strip()
                    if p:
                        chunks.append(p)
    return chunks


def build_vocab(text: str, seed_entities: List[str]) -> List[str]:
    chars = set(text)
    for e in seed_entities:
        chars.update(e)
    chars = [c for c in chars if c not in {"\n", "\r", "\t"}]
    return SPECIAL_TOKENS + sorted(chars)


def extract_parentheses_alias(text: str) -> List[str]:
    results = []
    pattern = re.compile(r"([\u4e00-\u9fffA-Za-z0-9_+\-/.]{2,30})[（(]([A-Za-z0-9_+\-/.]{2,20})[）)]")
    for m in pattern.finditer(text):
        left = m.group(1).strip()
        right = m.group(2).strip()
        if left:
            results.append(left)
        if right:
            results.append(right)
    return results


def extract_heading_terms(text: str) -> List[str]:
    terms = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^(?:第?[一二三四五六七八九十百千万\d]+(?:[、.．]|[)）]))\s*([\u4e00-\u9fffA-Za-z0-9_+\-/.]{2,30})", s)
        if m:
            terms.append(m.group(1))
    return terms


def extract_colon_terms(text: str) -> List[str]:
    terms = []
    for m in re.finditer(r"([\u4e00-\u9fffA-Za-z0-9_+\-/.]{2,30})(?:：|:)", text):
        terms.append(m.group(1))
    return terms


def extract_regex_terms(text: str, patterns: List[str]) -> List[str]:
    out = []
    for p in patterns:
        for m in re.finditer(p, text):
            t = m.group(0).strip()
            if t:
                out.append(t)
    return out


def extract_chinese_ngrams(text: str, min_len: int, max_len: int, min_freq: int) -> List[str]:
    zh = re.sub(r"[^\u4e00-\u9fff]", "", text)
    cnt = Counter()
    for n in range(min_len, max_len + 1):
        for i in range(0, max(0, len(zh) - n + 1)):
            cnt[zh[i : i + n]] += 1
    stop = {
        "我们", "你们", "他们", "一个", "一种", "以及", "可以", "进行", "如果", "这个", "那个",
        "通过", "使用", "主要", "包括", "用于", "实现", "结构", "系统", "数据", "方法", "过程",
        "问题", "方式", "不同", "存在", "之间", "其中", "常见",
    }
    terms = []
    for k, v in cnt.items():
        if v < min_freq:
            continue
        if k in stop:
            continue
        if len(set(k)) <= 1:
            continue
        if len(k) > 12:
            continue
        terms.append(k)
    return terms


def filter_terms(terms: List[str], allow_short_terms: set | None = None) -> List[str]:
    allow_short_terms = allow_short_terms or set()
    uniq = []
    seen = set()
    for t in terms:
        s = t.strip()
        s = re.sub(r"\s+", " ", s)
        s = s.strip("，。；：、（）()[]{}\"' ")
        if len(s) < 2 and s not in allow_short_terms:
            continue
        if re.fullmatch(r"[\W_]+", s):
            continue
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    uniq.sort(key=lambda x: (-len(x), x))
    return uniq


def _calc_entropy(counter: Counter) -> float:
    total = float(sum(counter.values()))
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in counter.values():
        p = float(v) / total
        ent -= p * math.log(p + 1e-12, 2)
    return ent


def extract_statistical_terms(
    text: str,
    min_len: int,
    max_len: int,
    min_freq: int,
    pmi_min: float,
    entropy_min: float,
) -> List[str]:
    zh = re.sub(r"[^\u4e00-\u9fff]", "", text)
    if len(zh) < min_len:
        return []

    counts: Dict[str, int] = defaultdict(int)
    left_ctx: Dict[str, Counter] = defaultdict(Counter)
    right_ctx: Dict[str, Counter] = defaultdict(Counter)

    n_chars = len(zh)
    for n in range(1, max_len + 1):
        for i in range(0, n_chars - n + 1):
            g = zh[i : i + n]
            counts[g] += 1
            if n >= min_len:
                if i > 0:
                    left_ctx[g][zh[i - 1]] += 1
                if i + n < n_chars:
                    right_ctx[g][zh[i + n]] += 1

    scored: List[Tuple[str, float]] = []
    total = float(n_chars)
    for term, c in counts.items():
        n = len(term)
        if n < min_len or n > max_len:
            continue
        if c < min_freq:
            continue
        if len(set(term)) <= 1:
            continue

        pmi = float("inf")
        for k in range(1, n):
            left = term[:k]
            right = term[k:]
            c_left = max(1, counts.get(left, 0))
            c_right = max(1, counts.get(right, 0))
            cur = math.log((c * total) / (c_left * c_right) + 1e-12, 2)
            pmi = min(pmi, cur)
        if pmi < pmi_min:
            continue

        le = _calc_entropy(left_ctx.get(term, Counter()))
        re_ = _calc_entropy(right_ctx.get(term, Counter()))
        if min(le, re_) < entropy_min:
            continue

        score = c * (1.0 + 0.2 * n) + pmi + min(le, re_)
        scored.append((term, score))

    scored.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))
    return [t for t, _ in scored]


def extract_english_terms(text: str) -> List[str]:
    terms: List[str] = []
    patterns = [
        r"\b[A-Za-z][A-Za-z0-9_+\-/.]{1,40}\b",
        r"\b[A-Za-z][A-Za-z0-9_+\-/.]{1,30}(?:\s+[A-Za-z][A-Za-z0-9_+\-/.]{1,30}){1,2}\b",
        r"\b-XX:[A-Za-z0-9:+._-]+\b",
        r"\b-X(?:ms|mx|ss)\b",
        r"\bO\([^)]+\)\b",
    ]
    for p in patterns:
        for m in re.finditer(p, text):
            t = m.group(0).strip()
            if t:
                terms.append(t)
    return terms


def _normalize_term_for_count(term: str) -> str:
    s = term.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip("，。；：、（）()[]{}\"' ")
    return s


GENERIC_NOISE_TERMS = {
    "主要", "常见", "可以", "用于", "包括", "其中", "存在", "方式", "进行", "实现", "定义", "负责", "相关",
    "问题", "过程", "场景", "效率", "关系", "一个", "一种", "这个", "那个", "通过", "通常", "一般", "支持",
    "功能", "系统中", "网络中", "数据", "结构", "操作", "方法",
}

ENTITY_HINT_CHARS = set("协议算法模型系统结构网络地址端口缓存索引事务线程进程内存文件命令函数接口类对象方法异常回收队列栈堆图树表锁路径页段帧报文字段版本状态库机器层码")
ENTITY_SUFFIXES = (
    "协议", "算法", "模型", "系统", "结构", "机制", "方法", "地址", "端口", "报文", "字段", "状态",
    "命令", "工具", "框架", "中间件", "函数", "接口", "对象", "线程", "进程", "内存", "事务", "索引",
    "数据库", "队列", "链表", "数组", "树", "图", "堆", "栈", "锁", "路径", "回收", "加载器", "类",
    "池", "模式", "日志", "参数", "视图", "分片", "复制", "异常", "错误", "操作",
)
BAD_EDGE_CHARS = set("的一是在和及与为于但由将把被其该等较更所向对并而从到给度址")


def _is_potential_entity_surface(term: str, occ: int) -> bool:
    s = term.strip()
    if not s:
        return False
    if re.search(r"[，。；：、（）()]", s):
        return False
    if s in GENERIC_NOISE_TERMS:
        return False
    if re.fullmatch(r"[0-9.]+", s):
        return False
    if re.match(r"^\d+(?:\.\d+)*$", s):
        return False

    is_en = bool(re.fullmatch(r"[A-Za-z0-9_+\-/. ]+", s))
    is_zh = bool(re.fullmatch(r"[\u4e00-\u9fff]+", s))

    if is_en:
        low = s.lower()
        if low in {"the", "and", "or", "for", "with", "from", "into", "that", "this", "used", "using"}:
            return False
        if s.islower() and len(s) <= 4:
            return False
        if len(s) <= 2 and not s.isupper():
            return False
        return True

    if is_zh:
        if s[0] in BAD_EDGE_CHARS or s[-1] in BAD_EDGE_CHARS:
            return False
        if s.startswith(("的", "和", "与", "及", "并", "在", "对", "由", "将", "把", "被", "其", "该", "每个", "所有", "一个", "一种")):
            return False
        if s.endswith(("的", "了", "和", "及", "并", "在", "中", "上", "下", "时", "后", "前")):
            return False
        if re.search(r"(是一种|用于|可以|包括|主要|常见|存在|负责|定义)", s):
            return False
        if any(x in s for x in ("和", "与", "及", "于", "但", "为", "其", "该", "所有", "一个", "一种", "不同")):
            if not s.endswith(ENTITY_SUFFIXES):
                return False

        has_hint = any(ch in ENTITY_HINT_CHARS for ch in s)
        if len(s) <= 2 and not has_hint and occ < 4:
            return False
        if len(s) >= 3:
            return has_hint or occ >= 2
        return has_hint or occ >= 2

    return False


def _build_candidate_terms_legacy(text: str, cfg: DomainConfig) -> List[str]:
    seed_set = set(cfg.seed_entities)
    alias_terms = extract_parentheses_alias(text)
    heading_terms = extract_heading_terms(text)
    colon_terms = extract_colon_terms(text)
    regex_terms = extract_regex_terms(text, cfg.regex_entities) if cfg.use_regex_entities else []
    ngram_terms: List[str] = []
    if cfg.use_ngrams:
        ngram_terms = extract_chinese_ngrams(text, cfg.ngram_min_len, cfg.ngram_max_len, cfg.ngram_min_freq)
    english_terms = re.findall(r"\b[A-Za-z][A-Za-z0-9_+\-/.]{1,30}\b", text)

    raw_terms: List[str] = []
    if cfg.use_seed_entities:
        raw_terms.extend(cfg.seed_entities)
    raw_terms.extend(alias_terms)
    raw_terms.extend(heading_terms)
    raw_terms.extend(colon_terms)
    raw_terms.extend(regex_terms)
    raw_terms.extend(ngram_terms)
    raw_terms.extend(english_terms)
    filtered = filter_terms(raw_terms, allow_short_terms=seed_set)
    freq = Counter(filtered)

    heading_set = set(heading_terms)
    colon_set = set(colon_terms)
    alias_set = set(alias_terms)
    regex_set = set(regex_terms)

    keep = []
    for t in filtered:
        is_seed = t in seed_set
        is_heading_like = t in heading_set or t in colon_set or t in alias_set
        is_regex = t in regex_set
        cnt = freq[t]
        is_en = bool(re.fullmatch(r"[A-Za-z0-9_+\-/.]+", t))
        is_zh = bool(re.fullmatch(r"[\u4e00-\u9fff]+", t))

        if is_seed:
            keep.append(t)
            continue

        if is_zh:
            if len(t) < 2 or len(t) > 12:
                continue
            if is_heading_like and cnt >= 1:
                keep.append(t)
                continue
            if cnt >= max(2, cfg.ngram_min_freq):
                keep.append(t)
                continue
            continue

        if is_en:
            if is_regex or is_heading_like:
                keep.append(t)
                continue
            if cnt >= 2:
                keep.append(t)
                continue
            if len(t) <= 5 and t.isupper():
                keep.append(t)
                continue
            continue

        if is_regex or is_heading_like:
            keep.append(t)

    keep = filter_terms(keep, allow_short_terms=seed_set)
    return keep[: cfg.max_candidate_terms]


def build_candidate_terms(text: str, cfg: DomainConfig) -> List[str]:
    legacy_mode = cfg.use_seed_entities or cfg.use_regex_entities or cfg.use_type_rules
    if legacy_mode:
        return _build_candidate_terms_legacy(text, cfg)

    seed_terms = cfg.seed_entities if cfg.use_seed_entities else []
    seed_set = set(seed_terms)
    alias_terms = extract_parentheses_alias(text)
    heading_terms = extract_heading_terms(text)
    colon_terms = extract_colon_terms(text)
    regex_terms = extract_regex_terms(text, cfg.regex_entities) if cfg.use_regex_entities else []
    ngram_terms = extract_chinese_ngrams(text, cfg.ngram_min_len, cfg.ngram_max_len, cfg.ngram_min_freq) if cfg.use_ngrams else []
    stat_terms = extract_statistical_terms(
        text,
        min_len=cfg.ngram_min_len,
        max_len=cfg.ngram_max_len,
        min_freq=max(2, cfg.ngram_min_freq),
        pmi_min=cfg.pmi_min,
        entropy_min=cfg.entropy_min,
    )
    english_terms = extract_english_terms(text)

    raw_terms: List[str] = []
    raw_terms.extend(seed_terms)
    raw_terms.extend(alias_terms)
    raw_terms.extend(heading_terms)
    raw_terms.extend(colon_terms)
    raw_terms.extend(regex_terms)
    raw_terms.extend(ngram_terms)
    raw_terms.extend(stat_terms)
    raw_terms.extend(english_terms)

    term_freq: Counter = Counter()
    for t in raw_terms:
        s = _normalize_term_for_count(t)
        if s:
            term_freq[s] += 1

    filtered = filter_terms(raw_terms, allow_short_terms=seed_set)

    heading_set = set(heading_terms)
    colon_set = set(colon_terms)
    alias_set = set(alias_terms)
    regex_set = set(regex_terms)
    stat_set = set(stat_terms)

    scored: List[Tuple[str, float]] = []
    for t in filtered:
        is_seed = t in seed_set
        is_heading_like = t in heading_set or t in colon_set or t in alias_set
        is_regex = t in regex_set
        is_stat = t in stat_set
        cnt = term_freq.get(t, 0)
        occ = text.count(t)
        if not _is_potential_entity_surface(t, occ):
            continue
        is_en = bool(re.fullmatch(r"[A-Za-z0-9_+\-/.]+", t))
        is_zh = bool(re.fullmatch(r"[\u4e00-\u9fff]+", t))

        if is_seed:
            scored.append((t, 1000.0))
            continue

        if is_zh:
            if len(t) < 2 or len(t) > 12:
                continue
            if occ <= 0 and not is_heading_like and not is_stat:
                continue
            score = 2.0 * len(t) + occ + cnt
            if is_heading_like:
                score += 6.0
            if is_regex:
                score += 3.0
            if is_stat:
                score += 3.0
            scored.append((t, score))
            continue

        if is_en:
            if len(t) < 2:
                continue
            if occ <= 0 and not is_regex and not is_heading_like:
                continue
            score = 1.5 * len(t) + occ + cnt
            if is_regex:
                score += 5.0
            if is_heading_like:
                score += 3.0
            if t.isupper() and len(t) <= 8:
                score += 2.0
            scored.append((t, score))
            continue

        if is_regex or is_heading_like or is_stat:
            score = len(t) + occ + cnt + (3.0 if is_heading_like else 0.0)
            scored.append((t, score))

    scored.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))
    keep = [t for t, _ in scored]
    keep = filter_terms(keep, allow_short_terms=seed_set)
    return keep[: cfg.max_candidate_terms]


def label_sentence_chars(sentence: str, terms: List[str]) -> List[int]:
    labels = [LABEL2ID["O"]] * len(sentence)
    for term in terms:
        start = 0
        while True:
            idx = sentence.find(term, start)
            if idx < 0:
                break
            end = idx + len(term)
            if all(labels[i] == LABEL2ID["O"] for i in range(idx, end)):
                labels[idx] = LABEL2ID["B-ENT"]
                for j in range(idx + 1, end):
                    labels[j] = LABEL2ID["I-ENT"]
            start = idx + 1
    return labels


class CharNERDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer: BertTokenizerFast, max_len: int):
        self.items = []
        self._build_cache(texts, labels, tokenizer, max_len)

    def _build_cache(self, texts: List[str], labels: List[List[int]], tokenizer: BertTokenizerFast, max_len: int) -> None:
        for text, y in zip(texts, labels):
            chars = list(text)
            enc = tokenizer(
                chars,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_attention_mask=True,
                return_tensors="pt",
            )
            word_ids = enc.word_ids(batch_index=0)
            token_labels = []
            for wid in word_ids:
                if wid is None or wid >= len(y):
                    token_labels.append(-100)
                else:
                    token_labels.append(y[wid])
            self.items.append(
                {
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "labels": torch.tensor(token_labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


class BertBiLstmCrf(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int):
        super().__init__()
        cfg = BertConfig(
            vocab_size=vocab_size,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            max_position_embeddings=512,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.bert = BertModel(cfg)
        self.bilstm = nn.LSTM(
            input_size=cfg.hidden_size,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(256, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        o = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = o.last_hidden_state
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        emissions = self.classifier(x)
        mask = attention_mask.bool()
        if labels is not None:
            y = labels.clone()
            y[y < 0] = 0
            train_mask = mask & (labels >= 0)
            llh = self.crf(emissions, y, mask=train_mask)
            return -llh.mean()
        return self.crf.viterbi_decode(emissions, mask=mask)


def train_model(model: nn.Module, loader: DataLoader, epochs: int, lr: float, device: torch.device) -> None:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            loss = model(ids, mask, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()


def extract_entities_from_tags(text: str, tags: List[int]) -> List[Tuple[str, int, int]]:
    entities: List[Tuple[str, int, int]] = []
    n = min(len(text), len(tags))
    i = 0
    while i < n:
        tag = ID2LABEL.get(tags[i], "O")
        if tag == "B-ENT":
            j = i + 1
            while j < n and ID2LABEL.get(tags[j], "O") == "I-ENT":
                j += 1
            ent = text[i:j].strip()
            if ent:
                entities.append((text[i:j], i, j))
            i = j
        else:
            i += 1
    return entities


def predict_entities(model: nn.Module, tokenizer: BertTokenizerFast, text: str, max_len: int, device: torch.device) -> List[Tuple[str, int, int]]:
    sents = split_sentences(text, max_len - 20)
    out: List[Tuple[str, int, int]] = []
    offset = 0
    model.eval()
    for sent in sents:
        chars = list(sent)
        enc = tokenizer(
            chars,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            decoded = model(ids, mask)[0]
        valid = int(mask[0].sum().item()) - 2
        decoded = decoded[1 : 1 + max(0, valid)]
        ents = extract_entities_from_tags(sent[:len(decoded)], decoded)
        for e, s, t in ents:
            out.append((e, offset + s, offset + t))
        offset += len(sent)
    return out


def lexicon_entities(text: str, terms: List[str]) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    for term in terms:
        start = 0
        while True:
            idx = text.find(term, start)
            if idx < 0:
                break
            out.append((text[idx : idx + len(term)], idx, idx + len(term)))
            start = idx + 1
    return out


def remove_contained_entities(entities: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    if not entities:
        return []
    ordered = sorted(entities, key=lambda x: (x[1], -(x[2] - x[1])))
    kept: List[Tuple[str, int, int]] = []
    for e, s, t in ordered:
        contained = False
        for _, ks, kt in kept:
            if ks <= s and t <= kt:
                contained = True
                break
        if not contained:
            kept.append((e, s, t))
    kept.sort(key=lambda x: (x[1], x[2], x[0]))
    return kept


def normalize_main_key(entity: str) -> str:
    s = entity.strip()
    s = re.sub(r"\s+", "", s)
    s = s.strip("，。；：、（）()[]{}\"' ")
    if re.fullmatch(r"[A-Za-z0-9_+\-/.]+", s):
        return s.lower()
    return s


def sanitize_mention(entity: str) -> str:
    s = entity.strip()
    s = re.sub(r"^(?:第?[一二三四五六七八九十百千万\d]+(?:[、.．]|[)）]))\s*", "", s)
    s = re.sub(r"^\d+(?:\.\d+)*\s*", "", s)
    s = re.sub(r"^(和|及|与|并且|其中)", "", s)
    s = re.sub(r"^(主要包括|包括|例如|比如)", "", s)
    s = re.sub(r"[，。；：].*$", "", s)
    s = re.sub(r"^(?:的|该|其|每个|所有|一个|一种)", "", s)
    for marker in ["是一种", "是指", "用于", "常用于", "通常用于", "其特点是", "其特征是"]:
        if marker in s:
            left = s.split(marker, 1)[0].strip()
            if len(left) >= 1:
                s = left
                break
    s = s.strip("，。；：、（）()[]{}\"' ")
    if s in GENERIC_NOISE_TERMS:
        return ""
    return s


def _candidate_types(cfg: DomainConfig) -> List[str]:
    if cfg.allowed_entity_types:
        return cfg.allowed_entity_types
    if cfg.type_rules:
        return [t for t, _, _ in cfg.type_rules]
    return [cfg.default_type]


def _type_tokens(type_name: str) -> List[str]:
    s = type_name.strip()
    s = re.sub(r"(实体|相关|类别|清单|去重后|核心)$", "", s)
    parts = re.split(r"[与和及、/,\s]+", s)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def _infer_type_from_heading(title: str, cfg: DomainConfig) -> str | None:
    title_norm = re.sub(r"\s+", "", title).lower()
    best_type = None
    best_score = 0
    for t in _candidate_types(cfg):
        t_norm = re.sub(r"\s+", "", t).lower()
        core = re.sub(r"(实体|相关|类别|清单)$", "", t_norm)
        score = 0
        if core and core in title_norm:
            score += 4
        for tok in _type_tokens(t):
            tok_norm = re.sub(r"\s+", "", tok).lower()
            if tok_norm and tok_norm in title_norm:
                score += 2
        if score > best_score:
            best_score = score
            best_type = t
    return best_type if best_score > 0 else None


def build_heading_type_lookup(text: str, cfg: DomainConfig) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    patterns = [
        r"(?m)^[ \t]*(?:第?[一二三四五六七八九十百千万0-9]+[、.．])\s*([^\n]{1,40})$",
        r"(?m)^[ \t]*(?:[（(]?[一二三四五六七八九十百千万0-9]+[)）])\s*([^\n]{1,40})$",
    ]
    seen = set()
    for p in patterns:
        for m in re.finditer(p, text):
            pos = int(m.start(1))
            title = m.group(1).strip()
            k = (pos, title)
            if k in seen:
                continue
            seen.add(k)
            t = _infer_type_from_heading(title, cfg)
            if t:
                out.append((pos, t))
    out.sort(key=lambda x: x[0])
    return out


def _type_cues(type_name: str) -> List[str]:
    cues = set(_type_tokens(type_name))
    t = type_name.lower()
    generic = {
        "协议": ["协议", "报文", "端口", "握手", "请求头", "状态码"],
        "算法": ["算法", "复杂度", "搜索", "排序", "最短路径", "回收"],
        "命令": ["命令", "工具", "诊断", "排查", "调试", "终端"],
        "中间件": ["中间件", "框架", "组件", "代理"],
        "数据库": ["表", "索引", "事务", "引擎", "SQL", "锁"],
        "系统": ["内核", "进程", "线程", "调度", "内存", "文件系统", "设备"],
        "数据结构": ["树", "图", "链表", "数组", "堆", "栈", "队列", "哈希"],
        "并发": ["线程", "并发", "锁", "同步", "可见性", "原子性"],
        "异常": ["异常", "错误", "溢出", "故障", "死锁"],
        "配置": ["参数", "配置", "选项", "开关"],
        "场景": ["场景", "应用", "用于", "常用于", "问题"],
        "概念": ["概念", "特性", "性质", "原理", "机制", "模型"],
        "地址": ["地址", "域名", "URI", "URL", "IP", "MAC"],
        "版本": ["版本", "状态码", "字节", "位", "RTT"],
        "层级": ["层", "模型", "分层"],
        "锁": ["锁", "互斥", "排他", "共享"],
    }
    for k, vals in generic.items():
        if k in t:
            for v in vals:
                cues.add(v)
    return [c for c in cues if c]


def classify_entity(
    entity: str,
    cfg: DomainConfig,
    source_text: str | None = None,
    start_pos: int | None = None,
    heading_lookup: List[Tuple[int, str]] | None = None,
) -> str:
    e = entity.strip()
    e_low = e.lower()

    if cfg.use_type_rules and cfg.type_rules:
        for type_name, keywords, patterns in cfg.type_rules:
            for k in keywords:
                if not k:
                    continue
                k_low = k.lower()
                is_ascii_kw = bool(re.fullmatch(r"[A-Za-z0-9_+\-/. ]+", k))
                if is_ascii_kw:
                    if e_low == k_low:
                        return type_name
                    continue
                if k in e:
                    return type_name
            for p in patterns:
                if re.search(p, e):
                    return type_name
        return cfg.default_type

    if not source_text:
        return cfg.default_type

    near_heading_type = None
    if heading_lookup and start_pos is not None:
        positions = [p for p, _ in heading_lookup]
        idx = bisect_right(positions, int(start_pos)) - 1
        if idx >= 0:
            near_heading_type = heading_lookup[idx][1]

    left = max(0, (start_pos or 0) - 24)
    right = min(len(source_text), (start_pos or 0) + len(e) + 24)
    ctx = source_text[left:right].lower()

    best_type = cfg.default_type
    best_score = -1.0
    for t in _candidate_types(cfg):
        score = 0.0
        for cue in _type_cues(t):
            cue_low = cue.lower()
            if not cue_low:
                continue
            if cue_low in e_low:
                score += 3.0
            if cue_low in ctx:
                score += 1.0
        if near_heading_type == t:
            score += 4.0
        if score > best_score:
            best_score = score
            best_type = t

    if best_score <= 0 and near_heading_type:
        return near_heading_type
    return best_type if best_score > 0 else cfg.default_type


def normalize_entity_type(entity_type: str, cfg: DomainConfig) -> str:
    t = (entity_type or "").strip()
    if not cfg.allowed_entity_types:
        return t if t else cfg.default_type
    allowed = set(cfg.allowed_entity_types)
    if t in allowed:
        return t
    if cfg.default_type in allowed:
        return cfg.default_type
    return cfg.allowed_entity_types[0]


def aggregate_to_main_entities(
    entities: List[Tuple[str, int, int]],
    cfg: DomainConfig,
    source_file: str,
    seed_entities: List[str],
    source_text: str,
    heading_lookup: List[Tuple[int, str]] | None = None,
) -> pd.DataFrame:
    seed_keys = {normalize_main_key(s) for s in seed_entities}
    grouped: Dict[str, Dict[str, object]] = defaultdict(dict)
    for e, s, _ in entities:
        mention = sanitize_mention(e)
        if not mention:
            continue
        key = normalize_main_key(mention)
        if not key:
            continue
        if key not in grouped:
            grouped[key] = {
                "main_entity": mention,
                "entity_type": normalize_entity_type(
                    classify_entity(
                        mention,
                        cfg,
                        source_text=source_text,
                        start_pos=int(s),
                        heading_lookup=heading_lookup,
                    ),
                    cfg,
                ),
                "mentions": set(),
                "mention_count": 0,
                "first_position": int(s),
            }
        grouped[key]["mentions"].add(mention)
        grouped[key]["mention_count"] += 1
        if int(s) < int(grouped[key]["first_position"]):
            grouped[key]["first_position"] = int(s)
            grouped[key]["entity_type"] = normalize_entity_type(
                classify_entity(
                    mention,
                    cfg,
                    source_text=source_text,
                    start_pos=int(s),
                    heading_lookup=heading_lookup,
                ),
                cfg,
            )

    rows = []
    effective_min_count = cfg.min_mention_count if cfg.use_seed_entities else 1
    effective_min_default = cfg.min_mention_count_default if cfg.use_seed_entities else 1
    effective_keep_default = cfg.keep_default_type if cfg.use_seed_entities else True
    for _, v in grouped.items():
        main_entity = str(v["main_entity"])
        main_key = normalize_main_key(main_entity)
        entity_type = str(v["entity_type"])
        mention_count = int(v["mention_count"])
        if main_key not in seed_keys and not _is_potential_entity_surface(main_entity, mention_count):
            continue
        if len(main_entity.strip()) < cfg.min_entity_len and main_key not in seed_keys:
            continue
        if len(main_entity.strip()) > cfg.max_entity_len and main_key not in seed_keys:
            continue
        if mention_count < effective_min_count and main_key not in seed_keys:
            continue
        if not effective_keep_default and entity_type == cfg.default_type and main_key not in seed_keys:
            continue
        if entity_type == cfg.default_type and mention_count < effective_min_default and main_key not in seed_keys:
            continue

        mentions = sorted(v["mentions"], key=lambda x: (-len(x), x))
        rows.append(
            {
                "main_entity": main_entity,
                "entity_type": entity_type,
                "mention_count": mention_count,
                "mentions": " | ".join(mentions[:10]),
                "source_file": source_file,
                "first_position": int(v.get("first_position", -1)),
            }
        )
    df = pd.DataFrame(
        rows,
        columns=["main_entity", "entity_type", "mention_count", "mentions", "source_file", "first_position"],
    )
    if not df.empty:
        df = df.sort_values(by=["first_position", "main_entity"], ascending=[True, True]).reset_index(drop=True)
    return df


def merge_append_only(existing_df: pd.DataFrame, new_df: pd.DataFrame, cfg: DomainConfig, source_text: str) -> pd.DataFrame:
    required_cols = ["main_entity", "entity_type", "mention_count", "mentions", "source_file", "first_position"]
    if existing_df.empty:
        return new_df.copy()
    if new_df.empty:
        return existing_df.copy()

    for c in required_cols:
        if c not in existing_df.columns:
            existing_df[c] = ""
        if c not in new_df.columns:
            new_df[c] = ""

    existing_df = existing_df[required_cols].copy()
    new_df = new_df[required_cols].copy()
    existing_df["mention_count"] = pd.to_numeric(existing_df["mention_count"], errors="coerce").fillna(0).astype(int)
    new_df["mention_count"] = pd.to_numeric(new_df["mention_count"], errors="coerce").fillna(0).astype(int)
    existing_df["first_position"] = pd.to_numeric(existing_df["first_position"], errors="coerce").fillna(-1).astype(int)
    new_df["first_position"] = pd.to_numeric(new_df["first_position"], errors="coerce").fillna(-1).astype(int)

    rows = {}
    order = []
    unknown_pos = 10**12

    def infer_first_position(main_entity: str, mentions: str, given_pos: int) -> int:
        if isinstance(given_pos, int) and given_pos >= 0:
            return given_pos
        candidates = [str(main_entity).strip()]
        candidates.extend([m.strip() for m in str(mentions).split("|") if m.strip()])
        best = unknown_pos
        for c in candidates:
            idx = source_text.find(c)
            if idx >= 0:
                best = min(best, idx)
        return best

    def key_of(entity: str) -> str:
        return normalize_main_key(str(entity))

    for _, r in existing_df.iterrows():
        k = key_of(r["main_entity"])
        if not k:
            continue
        if k not in rows:
            rows[k] = {
                "main_entity": str(r["main_entity"]),
                "entity_type": normalize_entity_type(str(r["entity_type"]), cfg),
                "mention_count": int(r["mention_count"]),
                "mentions": str(r["mentions"]),
                "source_file": str(r["source_file"]),
                "first_position": infer_first_position(str(r["main_entity"]), str(r["mentions"]), int(r["first_position"])),
            }
            order.append(k)

    for _, r in new_df.iterrows():
        k = key_of(r["main_entity"])
        if not k:
            continue
        if k not in rows:
            rows[k] = {
                "main_entity": str(r["main_entity"]),
                "entity_type": normalize_entity_type(str(r["entity_type"]), cfg),
                "mention_count": int(r["mention_count"]),
                "mentions": str(r["mentions"]),
                "source_file": str(r["source_file"]),
                "first_position": infer_first_position(str(r["main_entity"]), str(r["mentions"]), int(r["first_position"])),
            }
            order.append(k)
            continue

        old = rows[k]
        old_mentions = [m.strip() for m in str(old["mentions"]).split("|") if m.strip()]
        new_mentions = [m.strip() for m in str(r["mentions"]).split("|") if m.strip()]
        merged_mentions = []
        seen = set()
        for m in old_mentions + new_mentions:
            if m not in seen:
                seen.add(m)
                merged_mentions.append(m)
        old["mentions"] = " | ".join(merged_mentions[:20])
        old["mention_count"] = max(int(old["mention_count"]), int(r["mention_count"]))
        old_pos = int(old.get("first_position", unknown_pos))
        new_pos = infer_first_position(str(r["main_entity"]), str(r["mentions"]), int(r["first_position"]))
        old["first_position"] = min(old_pos if old_pos >= 0 else unknown_pos, new_pos if new_pos >= 0 else unknown_pos)
        new_type = normalize_entity_type(str(r["entity_type"]), cfg)
        old_type = normalize_entity_type(str(old["entity_type"]), cfg)
        fallback_type = normalize_entity_type("", cfg)
        if old_type == fallback_type and new_type != fallback_type:
            old["entity_type"] = new_type
        else:
            old["entity_type"] = old_type
        if str(old["source_file"]).strip() == "" and str(r["source_file"]).strip():
            old["source_file"] = str(r["source_file"])

    merged_rows = [rows[k] for k in order]
    out = pd.DataFrame(merged_rows, columns=required_cols)
    if not out.empty:
        out = out.sort_values(
            by=["first_position", "main_entity"],
            ascending=[True, True],
        ).reset_index(drop=True)
    return out


def run_bert_bilstm_crf_ner(cfg: DomainConfig) -> None:
    set_seed(cfg.seed)

    data_dir = Path("original_data")
    out_dir = Path("entity_output")
    model_dir = Path("model_artifacts")
    out_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    source_path = data_dir / cfg.source_file
    text = clean_text(read_text(source_path))

    print(
        f"[{cfg.domain_name}] mode "
        f"use_seed_entities={cfg.use_seed_entities} "
        f"use_regex_entities={cfg.use_regex_entities} "
        f"use_type_rules={cfg.use_type_rules} "
        f"merge_lexicon_entities={cfg.merge_lexicon_entities}"
    )
    terms = build_candidate_terms(text, cfg)
    print(f"[{cfg.domain_name}] candidate_terms={len(terms)}")

    vocab = build_vocab(text, terms[:5000])
    vocab_path = model_dir / f"{cfg.domain_name}_vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for t in vocab:
            f.write(t + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)

    sents = split_sentences(text, cfg.max_len - 20)
    train_texts: List[str] = []
    train_labels: List[List[int]] = []
    for s in sents:
        if len(s) < 4:
            continue
        y = label_sentence_chars(s, terms)
        train_texts.append(s)
        train_labels.append(y)
    print(f"[{cfg.domain_name}] train_samples={len(train_texts)}")

    ds = CharNERDataset(train_texts, train_labels, tokenizer, cfg.max_len)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertBiLstmCrf(vocab_size=len(vocab), num_labels=len(LABELS)).to(device)
    train_model(model, loader, cfg.epochs, cfg.lr, device)

    model_entities = predict_entities(model, tokenizer, text, cfg.max_len, device)
    legacy_mode = cfg.use_seed_entities or cfg.use_regex_entities or cfg.use_type_rules
    merged = list(model_entities)
    if cfg.merge_lexicon_entities:
        dict_entities = lexicon_entities(text, terms)
        merged.extend(dict_entities)

    # Deduplicate mentions by span + text.
    seen = set()
    uniq_entities: List[Tuple[str, int, int]] = []
    for e, s, t in merged:
        k = (e, s, t)
        if k in seen:
            continue
        seen.add(k)
        uniq_entities.append((e, s, t))
    if not legacy_mode:
        uniq_entities = remove_contained_entities(uniq_entities)
        if cfg.merge_lexicon_entities:
            term_set = set(terms)
            filtered_entities: List[Tuple[str, int, int]] = []
            for e, s, t in uniq_entities:
                m = sanitize_mention(e)
                if not m:
                    continue
                if m in term_set:
                    filtered_entities.append((m, int(s), int(s) + len(m)))
                    continue
                occ = text.count(m)
                if len(m) >= 3 and occ >= 2 and _is_potential_entity_surface(m, occ):
                    filtered_entities.append((m, int(s), int(t)))
            uniq_entities = remove_contained_entities(filtered_entities)

    heading_lookup = build_heading_type_lookup(text, cfg)
    seed_entities = cfg.seed_entities if cfg.use_seed_entities else []
    result_df = aggregate_to_main_entities(
        uniq_entities,
        cfg,
        cfg.source_file,
        seed_entities,
        source_text=text,
        heading_lookup=heading_lookup,
    )
    out_path = out_dir / cfg.output_csv
    if cfg.append_only and out_path.exists():
        try:
            old_df = pd.read_csv(out_path)
        except Exception:
            old_df = pd.DataFrame()
        result_df = merge_append_only(old_df, result_df, cfg, text)

    if cfg.allowed_entity_types:
        result_df["entity_type"] = result_df["entity_type"].astype(str).apply(lambda x: normalize_entity_type(x, cfg))

    if "first_position" in result_df.columns:
        result_df = result_df.sort_values(by=["first_position", "main_entity"], ascending=[True, True]).reset_index(drop=True)
        result_df = result_df.drop(columns=["first_position"])

    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[{cfg.domain_name}] saved={out_path} rows={len(result_df)}")


def run_domain_ner(cfg: DomainConfig) -> None:
    # Backward-compatible alias.
    run_bert_bilstm_crf_ner(cfg)


