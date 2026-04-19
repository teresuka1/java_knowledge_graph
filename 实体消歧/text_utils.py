from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd

from constants import ASCII_TOKEN_RE, HEADING_RE, SENTENCE_SPLIT_RE


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def normalize_name(text: str) -> str:
    text = clean_text(text or "").strip()
    text = text.strip("`'\"“”‘’[]{}【】<>《》")
    text = re.sub(r"\s*([()/._:+\-])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text


def build_loose_key(text: str) -> str:
    text = normalize_name(text).lower()
    return re.sub(r"[\s\-_/.:：,，;；()（）\[\]【】]+", "", text)


def split_mentions(value: object) -> List[str]:
    if pd.isna(value):
        return []
    parts = [normalize_name(part) for part in str(value).split("|")]
    return [part for part in parts if part]


def build_alias_forms(main_entity: str, mentions: Sequence[str]) -> set[str]:
    alias_forms: set[str] = set()
    for item in [main_entity, *mentions]:
        normalized = normalize_name(item)
        if not normalized:
            continue
        alias_forms.add(normalized)
        alias_forms.add(build_loose_key(normalized))
        compact = normalized.replace(" ", "")
        if compact:
            alias_forms.add(compact)
            alias_forms.add(build_loose_key(compact))
        if ASCII_TOKEN_RE.fullmatch(normalized):
            alias_forms.add(normalized.lower())
            alias_forms.add(normalized.upper())
    return {item for item in alias_forms if item}


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
        fallback = clean_text(text).strip()
        if fallback:
            sections.append(("全文", fallback))
    return sections


def extract_heading(line: str) -> str | None:
    match = HEADING_RE.match(line)
    if not match:
        return None
    heading = normalize_name(match.group(1))
    return heading[:80] if heading else None


def split_sentences(text: str) -> List[str]:
    pieces: List[str] = []
    for block in text.split("\n"):
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


def contains_alias(text: str, alias: str) -> bool:
    if not alias:
        return False
    if alias in text:
        return True
    if ASCII_TOKEN_RE.fullmatch(alias):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    lowered = text.lower()
    return alias.lower() in lowered
