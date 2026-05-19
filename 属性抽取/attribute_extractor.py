from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("PANDAS_NO_USE_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_USE_BOTTLENECK", "1")
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd

from kg_common import clean_text, dedup_preserve_order, find_all_alias_spans, normalize_name, parse_sections, read_text, split_pipe_list, split_sentences


FEATURE_KEYWORDS = (
    "不可变",
    "可变",
    "线程安全",
    "有序",
    "无序",
    "可重复",
    "不重复",
    "高并发",
    "高可用",
)


@dataclass
class EntityNode:
    canonical: str
    aliases: List[str]
    entity_type: str
    source_file: str


def load_entities(csv_path: Path) -> List[EntityNode]:
    dataframe = pd.read_csv(csv_path, encoding="utf-8-sig")
    if dataframe.empty:
        return []

    rows: List[EntityNode] = []
    for _, row in dataframe.iterrows():
        canonical = normalize_name(str(row.get("main_entity", "")))
        if not canonical:
            continue
        aliases = [canonical]
        aliases.extend(split_pipe_list(row.get("aliases")))
        aliases.extend(split_pipe_list(row.get("mentions")))
        rows.append(
            EntityNode(
                canonical=canonical,
                aliases=dedup_preserve_order(alias for alias in aliases if normalize_name(alias)),
                entity_type=normalize_name(str(row.get("entity_type", ""))),
                source_file=normalize_name(str(row.get("source_file", ""))),
            )
        )
    return rows


def discover_pairs(raw_dir: Path, entity_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for raw_path in sorted(raw_dir.glob("*.txt")):
        csv_path = entity_dir / f"{raw_path.stem}.csv"
        if csv_path.exists():
            pairs.append((raw_path, csv_path))
    return pairs


def normalize_attribute_name(label: str) -> str:
    label = normalize_name(label).strip("：:")
    mapping = {
        "端口": "默认端口",
        "端口号": "默认端口",
        "默认端口号": "默认端口",
        "默认端口": "默认端口",
        "默认存储引擎": "默认存储引擎",
        "存储引擎": "默认存储引擎",
        "特点": "特性",
        "特征": "特性",
        "性质": "特性",
        "作用": "用途",
        "用途": "用途",
        "定义": "定义",
        "概念": "定义",
        "底层结构": "底层结构",
        "数据结构": "底层结构",
    }
    return mapping.get(label, label or "属性")


def find_entity_mentions(sentence: str, entities: Sequence[EntityNode]) -> List[Tuple[EntityNode, int, int, str]]:
    mentions: List[Tuple[EntityNode, int, int, str]] = []
    for entity in entities:
        for start, end, surface in find_all_alias_spans(sentence, entity.aliases):
            mentions.append((entity, start, end, surface))
    mentions.sort(key=lambda item: (item[1], -(item[2] - item[1])))
    occupied: List[Tuple[int, int]] = []
    selected: List[Tuple[EntityNode, int, int, str]] = []
    for entity, start, end, surface in mentions:
        if any(not (end <= left or start >= right) for left, right in occupied):
            continue
        selected.append((entity, start, end, surface))
        occupied.append((start, end))
    return selected


def count_entities_in_text(text: str, entities: Sequence[EntityNode]) -> int:
    total = 0
    for entity in entities:
        if any(alias and alias in text for alias in entity.aliases[:3]):
            total += 1
    return total


def extract_from_colon_pattern(
    sentence: str,
    mentions: Sequence[Tuple[EntityNode, int, int, str]],
    section_title: str,
    source_file: str,
    entities: Sequence[EntityNode],
) -> List[dict]:
    rows: List[dict] = []
    for entity, start, end, _ in mentions:
        if start > 4:
            continue
        colon_positions = [pos for pos in (sentence.find("：", end), sentence.find(":", end)) if pos >= 0]
        if not colon_positions:
            continue
        colon_pos = min(colon_positions)
        label = normalize_attribute_name(sentence[end:colon_pos])
        value = normalize_name(sentence[colon_pos + 1 :])
        if not label or not value or len(label) > 18 or len(value) > 120:
            continue
        if count_entities_in_text(value, entities) > 2:
            continue
        rows.append(
            {
                "entity": entity.canonical,
                "attribute_name": label,
                "attribute_value": value,
                "section_title": section_title,
                "source_file": source_file,
                "evidence": sentence,
                "confidence": 0.82,
                "method": "attribute_extraction_colon_pattern",
            }
        )
    return rows


def extract_from_definition_pattern(
    sentence: str,
    mentions: Sequence[Tuple[EntityNode, int, int, str]],
    section_title: str,
    source_file: str,
    entities: Sequence[EntityNode],
) -> List[dict]:
    rows: List[dict] = []
    for entity, start, end, _ in mentions:
        if start > 4:
            continue
        window = sentence[end : min(len(sentence), end + 80)]
        match = re.search(r"^[^，,。；;]{0,8}是([^。；;]{4,80})$", window)
        if not match:
            continue
        value = normalize_name(match.group(1))
        if not value or count_entities_in_text(value, entities) > 2:
            continue
        rows.append(
            {
                "entity": entity.canonical,
                "attribute_name": "定义",
                "attribute_value": value,
                "section_title": section_title,
                "source_file": source_file,
                "evidence": sentence,
                "confidence": 0.76,
                "method": "attribute_extraction_definition_pattern",
            }
        )
    return rows


def extract_from_feature_pattern(
    sentence: str,
    mentions: Sequence[Tuple[EntityNode, int, int, str]],
    section_title: str,
    source_file: str,
) -> List[dict]:
    rows: List[dict] = []
    for entity, start, end, _ in mentions:
        if start > 8:
            continue
        tail = sentence[end:]
        for keyword in FEATURE_KEYWORDS:
            if keyword not in tail:
                continue
            rows.append(
                {
                    "entity": entity.canonical,
                    "attribute_name": "特性",
                    "attribute_value": keyword,
                    "section_title": section_title,
                    "source_file": source_file,
                    "evidence": sentence,
                    "confidence": 0.78,
                    "method": "attribute_extraction_feature_pattern",
                }
            )
    return rows


def deduplicate_rows(rows: Sequence[dict]) -> List[dict]:
    best: Dict[Tuple[str, str, str, str], dict] = {}
    for row in rows:
        key = (row["entity"], row["attribute_name"], row["attribute_value"], row["source_file"])
        current = best.get(key)
        if current is None or float(row["confidence"]) >= float(current["confidence"]):
            best[key] = dict(row)
    output = list(best.values())
    output.sort(key=lambda item: (-float(item["confidence"]), item["entity"], item["attribute_name"]))
    return output


def extract_attributes(text: str, entities: Sequence[EntityNode], source_file: str) -> List[dict]:
    rows: List[dict] = []
    for section_title, section_text in parse_sections(text):
        for sentence in split_sentences(section_text):
            normalized_sentence = clean_text(sentence)
            mentions = find_entity_mentions(normalized_sentence, entities)
            if not mentions:
                continue
            rows.extend(extract_from_colon_pattern(normalized_sentence, mentions, section_title, source_file, entities))
            rows.extend(extract_from_definition_pattern(normalized_sentence, mentions, section_title, source_file, entities))
            rows.extend(extract_from_feature_pattern(normalized_sentence, mentions, section_title, source_file))
    return deduplicate_rows(rows)


def save_rows(rows: Sequence[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(output_path, index=False, encoding="utf-8-sig")
