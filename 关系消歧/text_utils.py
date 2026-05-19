from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Iterable, List, Sequence, TypeVar

T = TypeVar("T")


EMPTY_MARKERS = {"", "nan", "none", "null", "n/a"}


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u3000", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_entity(value: object) -> str:
    text = normalize_text(value)
    if text.lower() in EMPTY_MARKERS:
        return ""
    text = text.strip("`'\"“”‘’[]{}【】<>《》")
    text = re.sub(r"\s*([()/._:+\-])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def entity_key(value: object) -> str:
    text = normalize_entity(value).casefold()
    text = re.sub(r"[\s\-_/.:：,，;；()（）\[\]【】{}<>《》\"'`]+", "", text)
    return text


def relation_key(value: object) -> str:
    text = normalize_text(value).casefold()
    text = text.strip("`'\"“”‘’[]{}【】<>《》")
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", "", text)
    return text.strip("_")


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = normalize_text(value)
        if text.lower() in EMPTY_MARKERS:
            return default
        parsed = float(text)
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def dedupe_preserve_order(items: Iterable[T]) -> List[T]:
    seen: set[T] = set()
    result: List[T] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def compact_join(items: Iterable[object], limit: int = 8, sep: str = " | ") -> str:
    values = [normalize_text(item) for item in items]
    values = [item for item in dedupe_preserve_order(values) if item]
    if len(values) > limit:
        values = [*values[:limit], f"...(+{len(values) - limit})"]
    return sep.join(values)


def choose_display(values: Sequence[str], confidences: Sequence[float] | None = None) -> str:
    cleaned = [normalize_entity(value) for value in values if normalize_entity(value)]
    if not cleaned:
        return ""
    weights: Counter[str] = Counter()
    for index, value in enumerate(cleaned):
        weight = 1.0
        if confidences and index < len(confidences):
            weight += max(confidences[index], 0.0)
        weights[value] += weight

    def sort_key(item: tuple[str, float]) -> tuple[float, int, int, str]:
        value, weight = item
        ascii_only = 1 if re.fullmatch(r"[A-Za-z0-9_+\-./]+", value) else 0
        return (weight, len(value), ascii_only, value)

    return max(weights.items(), key=sort_key)[0]


def shorten(value: object, limit: int = 120) -> str:
    text = normalize_text(value).replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 4] + " ..."
