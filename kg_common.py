from __future__ import annotations

import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

os.environ.setdefault("PANDAS_NO_USE_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_USE_BOTTLENECK", "1")
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd


ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-./]+")
SENTENCE_SPLIT_RE = re.compile(r"[。！？；]\s*")
HEADING_RE = re.compile(
    r"^\s*(?:"
    r"第?[一二三四五六七八九十百千万零两\d]+(?:章|节|部分)?"
    r"|[一二三四五六七八九十百千万零两]+"
    r"|[0-9]+(?:\.[0-9]+)*"
    r"|[A-Za-z]"
    r")(?:[、.．)]|\s+)\s*(.+?)\s*$"
)


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u3000", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_name(text: str) -> str:
    text = clean_text(text).strip()
    text = text.strip("`'\"“”‘’[]{}【】<>《》()（）")
    text = re.sub(r"\s*([()/._:+\-])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip("：:，,；;。.")


def build_loose_key(text: str) -> str:
    text = normalize_name(text).lower()
    return re.sub(r"[\s\-_/.:：，,；;（）()\[\]【】]+", "", text)


def dedup_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def split_pipe_list(value: object) -> List[str]:
    if pd.isna(value):
        return []
    parts = [normalize_name(part) for part in str(value).split("|")]
    return [part for part in parts if part]


def split_sentences(text: str) -> List[str]:
    pieces: List[str] = []
    for block in clean_text(text).split("\n"):
        block = block.strip()
        if not block:
            continue
        parts = SENTENCE_SPLIT_RE.split(block)
        if len(parts) == 1:
            parts = [block]
        for part in parts:
            sentence = part.strip()
            if sentence:
                pieces.append(sentence)
    return pieces


def extract_heading(line: str) -> str | None:
    match = HEADING_RE.match(clean_text(line))
    if not match:
        return None
    heading = normalize_name(match.group(1))
    return heading[:80] if heading else None


def parse_sections(text: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    current_title = "全文"
    buffer: List[str] = []

    def flush() -> None:
        content = "\n".join(line for line in buffer if line).strip()
        if content:
            sections.append((current_title, content))
        buffer.clear()

    for raw_line in clean_text(text).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        heading = extract_heading(line)
        if heading:
            flush()
            current_title = heading
        else:
            buffer.append(line)
    flush()

    if not sections:
        fallback = clean_text(text)
        if fallback:
            sections.append(("全文", fallback))
    return sections


def contains_alias(text: str, alias: str) -> bool:
    alias = normalize_name(alias)
    if not alias:
        return False
    if alias in text:
        return True
    if ASCII_TOKEN_RE.fullmatch(alias):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    return alias.lower() in text.lower()


def find_all_alias_spans(text: str, aliases: Sequence[str]) -> List[Tuple[int, int, str]]:
    hits: List[Tuple[int, int, str]] = []
    sentence = clean_text(text)
    for alias in sorted({normalize_name(item) for item in aliases if item}, key=lambda item: (-len(item), item)):
        if len(alias) < 2:
            continue
        if ASCII_TOKEN_RE.fullmatch(alias):
            pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])", re.IGNORECASE)
            spans = [(match.start(), match.end(), sentence[match.start() : match.end()]) for match in pattern.finditer(sentence)]
        else:
            spans = []
            start = 0
            while True:
                index = sentence.find(alias, start)
                if index < 0:
                    break
                spans.append((index, index + len(alias), sentence[index : index + len(alias)]))
                start = index + 1
        hits.extend(spans)
    hits.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: List[Tuple[int, int, str]] = []
    occupied: List[Tuple[int, int]] = []
    for start, end, surface in hits:
        if any(not (end <= left or start >= right) for left, right in occupied):
            continue
        selected.append((start, end, surface))
        occupied.append((start, end))
    return selected


def domain_from_source(source_file: str, fallback: str = "") -> str:
    source = normalize_name(source_file)
    if source.lower().endswith(".txt"):
        return source[:-4]
    if source.lower().endswith(".csv"):
        return source[:-4]
    return source or fallback
