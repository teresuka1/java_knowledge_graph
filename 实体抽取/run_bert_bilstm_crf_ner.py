import os
import re
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel, BertTokenizerFast
from TorchCRF import CRF

# ---------------------------
# Config
# ---------------------------
SEED = 42
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-4
DATA_DIR = Path("original_data")
OUT_DIR = Path("entity_output")
MODEL_DIR = Path("model_artifacts")

TXT_FILES = [
    "cn.txt",
    "data_structure.txt",
    "java.txt",
    "JVM.txt",
    "mysql.txt",
    "os.txt",
]

LABELS = ["O", "B-TERM", "I-TERM"]
LABEL2ID = {v: i for i, v in enumerate(LABELS)}
ID2LABEL = {i: v for v, i in LABEL2ID.items()}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_text_robust(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            txt = raw.decode(enc)
            if txt:
                return txt
        except Exception:
            pass
    return raw.decode("utf-8", errors="ignore")


def clean_text(t: str) -> str:
    t = t.replace("\u3000", " ")
    t = re.sub(r"\r\n?", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t


def split_sentences(text: str, max_chars: int = 110) -> List[str]:
    chunks = []
    for blk in text.split("\n"):
        blk = blk.strip()
        if not blk:
            continue
        segs = re.split(r"(?<=[。！？；;.!?])", blk)
        for s in segs:
            s = s.strip()
            if not s:
                continue
            if len(s) <= max_chars:
                chunks.append(s)
            else:
                for i in range(0, len(s), max_chars):
                    part = s[i : i + max_chars].strip()
                    if part:
                        chunks.append(part)
    return chunks


def build_char_vocab(all_texts: List[str], min_freq: int = 1) -> List[str]:
    cnt = Counter()
    for t in all_texts:
        cnt.update(list(t))
    chars = [c for c, n in cnt.items() if n >= min_freq and c not in ["\n", "\r", "\t"]]
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    return special + sorted(chars)


def build_candidate_terms(all_texts: List[str]) -> List[str]:
    whole = "\n".join(all_texts)

    # 1) English / acronym terms
    eng_terms = re.findall(r"\b[A-Za-z][A-Za-z0-9_+./-]{1,30}\b", whole)

    # 2) Chinese n-grams with frequency
    only_zh = re.sub(r"[^\u4e00-\u9fff]", "", whole)
    zh_counter = Counter()
    for n in range(2, 7):
        for i in range(0, max(0, len(only_zh) - n + 1)):
            g = only_zh[i : i + n]
            zh_counter[g] += 1

    stop = set([
        "我们", "你们", "他们", "一个", "一种", "以及", "可以", "进行", "如果", "这个", "那个", "通过", "使用",
        "主要", "常见", "包括", "用于", "实现", "结构", "系统", "数据", "方法", "过程", "问题", "方式"
    ])
    zh_terms = [
        k
        for k, v in zh_counter.items()
        if v >= 3 and k not in stop and len(set(k)) > 1
    ]

    # 3) Heading-like terms after numbering
    heading_terms = []
    for line in whole.split("\n"):
        m = re.match(r"^\s*[\d一二三四五六七八九十]+(?:\.[\d]+)*\s*([\u4e00-\u9fffA-Za-z0-9_+./-]{2,30})", line.strip())
        if m:
            heading_terms.append(m.group(1))

    # 4) Terms before colon: often concept names in notes
    colon_terms = re.findall(r"([\u4e00-\u9fff]{2,20})(?:：|:)", whole)

    all_terms = eng_terms + zh_terms + heading_terms + colon_terms
    all_terms = [t.strip() for t in all_terms if 2 <= len(t.strip()) <= 20]

    freq = Counter(all_terms)
    heading_set = set(heading_terms)
    colon_set = set(colon_terms)
    terms = []
    for k, v in freq.items():
        if re.search(r"[A-Za-z]", k):
            terms.append(k)
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", k):
            if k in heading_set or k in colon_set or v >= 2:
                terms.append(k)

    # remove highly generic pure Chinese short terms
    filtered = []
    for t in terms:
        if re.fullmatch(r"[\u4e00-\u9fff]+", t):
            if t in stop:
                continue
            if len(t) == 2 and freq[t] < 3 and t not in heading_set and t not in colon_set:
                continue
            if len(t) > 12:
                continue
        if re.fullmatch(r"[\W_]+", t):
            continue
        filtered.append(t)

    filtered = sorted(set(filtered), key=lambda x: (-len(x), x))
    return filtered[:5000]


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
                labels[idx] = LABEL2ID["B-TERM"]
                for j in range(idx + 1, end):
                    labels[j] = LABEL2ID["I-TERM"]
            start = idx + 1
    return labels


@dataclass
class Sample:
    text: str
    labels: List[int]


class NERDataset(Dataset):
    def __init__(self, samples: List[Sample], tokenizer: BertTokenizerFast, max_len: int = MAX_LEN):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        chars = list(s.text)
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        labels = []
        for wid in word_ids:
            if wid is None or wid >= len(s.labels):
                labels.append(-100)
            else:
                labels.append(s.labels[wid])

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return item


class BertBiLstmCrf(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int):
        super().__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            max_position_embeddings=512,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.bert = BertModel(config)
        self.bilstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(256, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state
        seq, _ = self.bilstm(seq)
        seq = self.dropout(seq)
        emissions = self.classifier(seq)

        mask = attention_mask.bool()

        if labels is not None:
            valid_labels = labels.clone()
            valid_labels[valid_labels < 0] = 0
            train_mask = mask & (labels >= 0)
            llh = self.crf(emissions, valid_labels, mask=train_mask)
            nll = -llh.mean()
            return nll

        decoded = self.crf.viterbi_decode(emissions, mask=mask)
        return decoded


def train_model(model, loader, device):
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    for ep in range(EPOCHS):
        losses = []
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

        print(f"Epoch {ep+1}/{EPOCHS} loss={np.mean(losses):.4f}")


def extract_entities_from_labels(text: str, pred_labels: List[int]) -> List[Tuple[str, int, int, str]]:
    ents = []
    i = 0
    n = min(len(text), len(pred_labels))
    while i < n:
        tag = ID2LABEL.get(pred_labels[i], "O")
        if tag == "B-TERM":
            j = i + 1
            while j < n and ID2LABEL.get(pred_labels[j], "O") == "I-TERM":
                j += 1
            ent = text[i:j]
            if ent.strip():
                ents.append((ent, i, j, "TERM"))
            i = j
        else:
            i += 1
    return ents


def extract_entities_by_terms(text: str, terms: List[str]) -> List[Tuple[str, int, int, str]]:
    ents = []
    for term in terms:
        start = 0
        while True:
            idx = text.find(term, start)
            if idx < 0:
                break
            end = idx + len(term)
            ents.append((term, idx, end, "TERM"))
            start = idx + 1
    return ents


def classify_entity(entity: str, source_file: str) -> str:
    e = entity.strip()
    up = e.upper()

    protocol_set = {
        "TCP", "UDP", "HTTP", "HTTPS", "FTP", "SMTP", "POP3", "IMAP",
        "SSH", "DNS", "RTP", "RTCP", "DHCP", "SNMP", "NTP", "ARP",
        "ICMP", "IGMP", "NAT", "OSPF", "RIP", "BGP", "IPV4", "IPV6",
        "TLS", "SSL", "MAC"
    }
    algorithm_set = {
        "DIJKSTRA", "DFS", "BFS", "AVL", "KMP", "KRUSKAL", "PRIM",
        "QUICKSORT", "MERGESORT", "HEAPSORT", "LRU", "LFU", "A*"
    }
    middleware_set = {
        "REDIS", "KAFKA", "RABBITMQ", "ROCKETMQ", "NGINX", "TOMCAT",
        "ZOOKEEPER", "DUBBO", "MYBATIS", "SHARDING-JDBC", "ELASTICSEARCH"
    }
    command_set = {
        "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
        "EXPLAIN", "COMMIT", "ROLLBACK", "GRANT", "REVOKE", "SHOW",
        "PING", "TRACERT", "NETSTAT", "PS", "TOP", "KILL", "CHMOD", "CHOWN"
    }
    jvm_set = {
        "JVM", "JDK", "JRE", "JIT", "GC", "EDEN", "SURVIVOR", "METASPACE",
        "CMS", "G1", "ZGC", "SHENANDOAH", "CLASSLOADER"
    }

    data_structure_keywords = ["数组", "链表", "栈", "队列", "哈希", "二叉树", "红黑树", "B树", "B+树", "堆", "图", "Trie"]
    algorithm_keywords = ["算法", "排序", "查找", "最短路径", "动态规划", "贪心", "回溯", "递归", "深度优先", "广度优先"]
    concept_keywords = [
        "原理", "机制", "模型", "事务", "索引", "锁", "并发", "一致性", "隔离级别",
        "线程", "进程", "内存", "缓存", "虚拟内存", "页表", "调度", "死锁",
        "类加载", "垃圾回收", "字节码"
    ]

    source = source_file.lower()

    if up in protocol_set or "协议" in e:
        return "协议"
    if up in algorithm_set or any(k in e for k in algorithm_keywords):
        return "算法"
    if up in command_set or "命令" in e:
        return "命令"
    if up in middleware_set or "中间件" in e:
        return "中间件"
    if up in jvm_set or source == "jvm.txt":
        if any(k in e for k in ["JVM", "GC", "类加载", "字节码", "堆", "栈帧", "方法区", "元空间"]):
            return "JVM"
    if any(k in e for k in data_structure_keywords) or source == "data_structure.txt":
        if any(k in e for k in data_structure_keywords):
            return "数据结构"
    if any(k in e for k in ["MySQL", "SQL", "InnoDB", "B+Tree", "索引", "事务", "MVCC", "锁"]):
        return "数据库"
    if any(k in e for k in ["TCP", "IP", "路由", "子网", "网关", "MAC", "端口", "拥塞控制", "三次握手"]):
        return "网络"
    if any(k in e for k in ["线程", "进程", "调度", "中断", "系统调用", "页", "内核", "死锁"]):
        return "操作系统"
    if any(k in e for k in concept_keywords):
        return "概念"

    # File-level fallback to keep categories specific instead of collapsing to TERM/概念.
    if source == "cn.txt":
        return "网络"
    if source == "mysql.txt":
        return "数据库"
    if source == "os.txt":
        return "操作系统"
    if source == "jvm.txt":
        return "JVM"
    if source == "data_structure.txt":
        return "数据结构"
    if source == "java.txt":
        return "Java基础"
    return "概念"


def predict_text_entities(model, tokenizer, text: str, device) -> List[Tuple[str, int, int, str]]:
    sentences = split_sentences(text)
    all_ents = []
    offset = 0

    for sent in sentences:
        chars = list(sent)
        enc = tokenizer(
            chars,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        model.eval()
        with torch.no_grad():
            decoded = model(input_ids=input_ids, attention_mask=mask)[0]

        valid_len = int(mask[0].sum().item()) - 2  # drop [CLS] [SEP]
        decoded = decoded[1 : 1 + max(0, valid_len)]

        ents = extract_entities_from_labels(sent[:len(decoded)], decoded)
        for e, s, t, tp in ents:
            all_ents.append((e, offset + s, offset + t, tp))

        offset += len(sent)

    return all_ents


def main():
    set_seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    data: Dict[str, str] = {}
    all_texts = []
    for fn in TXT_FILES:
        p = DATA_DIR / fn
        txt = clean_text(read_text_robust(p))
        data[fn] = txt
        all_texts.append(txt)

    vocab = build_char_vocab(all_texts)
    vocab_path = MODEL_DIR / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for t in vocab:
            f.write(t + "\n")

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)

    terms = build_candidate_terms(all_texts)
    print(f"Candidate terms: {len(terms)}")

    samples = []
    for txt in all_texts:
        for sent in split_sentences(txt):
            if len(sent) < 4:
                continue
            labels = label_sentence_chars(sent, terms)
            samples.append(Sample(text=sent, labels=labels))

    print(f"Train samples: {len(samples)}")

    train_ds = NERDataset(samples, tokenizer, MAX_LEN)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertBiLstmCrf(vocab_size=len(vocab), num_labels=len(LABELS)).to(device)

    train_model(model, loader, device)

    for fn, txt in data.items():
        model_ents = predict_text_entities(model, tokenizer, txt, device)
        lexicon_ents = extract_entities_by_terms(txt, terms)
        ents = model_ents + lexicon_ents

        # Deduplicate by (entity,start,end)
        seen = set()
        rows = []
        for ent, s, e, tp in ents:
            key = (ent, s, e)
            if key in seen:
                continue
            seen.add(key)
            # entity is always sliced from original text, preserving Chinese/English exactly as in source.
            cat = classify_entity(ent, fn)
            rows.append({
                "entity": ent,
                "entity_type": cat,
                "start": s,
                "end": e,
                "source_file": fn,
            })

        df = pd.DataFrame(rows)
        out_name = Path(fn).stem + ".csv"
        out_path = OUT_DIR / out_name
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {out_path} ({len(df)} entities)")


if __name__ == "__main__":
    main()
