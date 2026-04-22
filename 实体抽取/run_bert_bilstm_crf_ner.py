"""弱监督 NER 训练脚本（BERT + BiLSTM + CRF）。

脚本流程：
1. 汇总多领域文本，自动挖掘候选术语并生成伪标签训练数据。
2. 训练字符级序列标注模型，对原始文本执行分句推理。
3. 融合模型实体与词典实体，完成规则分类后导出 CSV。
"""

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
# 配置参数
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


# 作用：固定随机种子，减少不同机器和多次运行间的随机波动。
# 说明：同时设置系统随机库、数值计算库和深度学习库（含显卡后端）的随机状态。
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 作用：鲁棒读取文本文件，尽量避免编码差异导致的读取失败。
# 说明：按常见文本编码顺序逐个尝试解码。
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


# 作用：统一文本格式，降低后续分句与规则匹配噪声。
# 说明：处理全角空格、换行符和多余空白。
def clean_text(t: str) -> str:
    t = t.replace("\u3000", " ")
    t = re.sub(r"\r\n?", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t


# 作用：把长文本切成可训练/可推理片段，避免超长序列。
# 说明：先按标点切分，超长片段再按最大长度参数二次切分。
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


# 作用：构建字符级词表，供分词器使用自定义词表文件。
# 说明：过滤控制字符，仅保留满足最小频次的字符。
def build_char_vocab(all_texts: List[str], min_freq: int = 1) -> List[str]:
    cnt = Counter()
    for t in all_texts:
        cnt.update(list(t))
    chars = [c for c, n in cnt.items() if n >= min_freq and c not in ["\n", "\r", "\t"]]
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    return special + sorted(chars)


# 作用：挖掘弱监督候选术语（伪标签词典）。
# 说明：融合英文术语、中文字串片段、标题词、冒号前术语并做筛选。
def build_candidate_terms(all_texts: List[str]) -> List[str]:
    whole = "\n".join(all_texts)

    # 1）英文术语/缩写术语
    eng_terms = re.findall(r"\b[A-Za-z][A-Za-z0-9_+./-]{1,30}\b", whole)

    # 2）带频次统计的中文字串片段
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

    # 3）编号标题后的标题术语
    heading_terms = []
    for line in whole.split("\n"):
        m = re.match(r"^\s*[\d一二三四五六七八九十]+(?:\.[\d]+)*\s*([\u4e00-\u9fffA-Za-z0-9_+./-]{2,30})", line.strip())
        if m:
            heading_terms.append(m.group(1))

    # 4）冒号前术语：在学习笔记中通常是概念名称
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

    # 去掉过于泛化的中文短词
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


# 作用：将候选术语命中位置转换为字符级序列标注标签。
# 说明：标签集合为“外部、实体开头、实体内部”，重叠区域采用先到先得。
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
# 作用：表示一个训练样本（句子文本 + 字符级标签）。
class Sample:
    text: str
    labels: List[int]


# 作用：数据集封装层，负责分词编码与标签对齐。
# 说明：特殊标记和无效位置标为 -100，不参与损失计算。
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


# 作用：定义命名实体识别主模型（预训练编码器 + 双向循环层 + 条件随机场）。
# 说明：编码器负责上下文表示，循环层强化时序依赖，随机场层约束标签转移。
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


# 作用：执行训练循环并更新参数。
# 说明：使用自适应优化器，按全局训练轮数和学习率训练并输出每轮损失。
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


# 作用：将预测标签序列解码为实体区间列表。
# 返回：[(实体文本, 起始位置, 结束位置, 实体类型), ...]
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


# 作用：词典直匹配补充，扫描候选词在全文中的出现位置。
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


# 启发式实体分类器：结合关键词和来源文件线索判定实体类别。
# 作用：基于规则将实体映射到业务类别。
# 说明：结合关键词集合与来源文件名做类型判定和回退。
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

    # 文件级回退策略：尽量保持类别具体，不统一退化为“术语/概念”。
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


# 作用：执行全文推理并恢复绝对坐标。
# 说明：按分句预测后，通过偏移量将句内坐标映射回原文。
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


# 作用：脚本主入口。
# 流程：加载数据 -> 构建词表/候选词 -> 训练 -> 推理融合 -> 导出结果文件。
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

        # 按（实体文本，起始位置，结束位置）去重
        seen = set()
        rows = []
        for ent, s, e, tp in ents:
            key = (ent, s, e)
            if key in seen:
                continue
            seen.add(key)
            # 实体文本始终从原文切片得到，保留原始中英文写法。
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
