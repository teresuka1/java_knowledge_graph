from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, List

from constants import ASCII_TOKEN_RE, GENERIC_SUFFIXES
from text_utils import build_loose_key, normalize_name


def bounded_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def variant_relation_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0

    left_norm = normalize_name(left)
    right_norm = normalize_name(right)
    if left_norm == right_norm:
        return 1.0
    if build_loose_key(left_norm) == build_loose_key(right_norm):
        return 1.0

    if ASCII_TOKEN_RE.fullmatch(left_norm) and ASCII_TOKEN_RE.fullmatch(right_norm):
        return 0.0

    left_compact = left_norm.replace(" ", "")
    right_compact = right_norm.replace(" ", "")
    short, long = sorted([left_compact, right_compact], key=len)
    if not short or not long:
        return 0.0

    if long.startswith("非") and short == long[1:]:
        return 0.0

    if long.startswith(short):
        suffix = long[len(short) :]
        if suffix in GENERIC_SUFFIXES:
            return 0.90
    if long.endswith(short):
        prefix = long[: len(long) - len(short)]
        if prefix in {"全", "纯"}:
            return 0.82

    if re.search(r"[\u4e00-\u9fff]", short) and abs(len(long) - len(short)) <= 1:
        if SequenceMatcher(None, short, long).ratio() >= 0.88:
            return 0.92
    return 0.0


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def split_type_tokens(text: str) -> List[str]:
    normalized = normalize_name(text)
    ascii_tokens = re.findall(r"[A-Za-z]+", normalized)
    chinese_tokens = re.findall(r"[\u4e00-\u9fff]{1,6}", normalized)
    return ascii_tokens + chinese_tokens


def token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(split_type_tokens(left))
    right_tokens = set(split_type_tokens(right))
    return jaccard(left_tokens, right_tokens)


def deduplicate_preserve_order(items: Iterable) -> List:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
