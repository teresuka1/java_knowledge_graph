from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("PANDAS_NO_USE_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_USE_BOTTLENECK", "1")
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd

from kg_common import build_loose_key, clean_text, normalize_name


RELATION_NAME_MAP = {
    "implemented_by": "implements",
    "compare_with": "related_to",
}

ATTRIBUTE_VALUE_HINTS = {
    "public",
    "private",
    "protected",
    "default",
    "static",
    "final",
    "abstract",
    "immutable",
    "null",
    "true",
    "false",
    "可变",
    "不可变",
    "有序",
    "无序",
    "线程安全",
    "单继承",
}

ATTRIBUTE_NAME_MAP = {
    "public": "修饰符",
    "private": "修饰符",
    "protected": "修饰符",
    "default": "默认值",
    "static": "修饰符",
    "final": "修饰符",
    "abstract": "修饰符",
    "immutable": "特性",
    "可变": "特性",
    "不可变": "特性",
    "有序": "特性",
    "无序": "特性",
    "线程安全": "特性",
    "单继承": "继承特性",
}

GENERIC_TAILS = {
    "父类",
    "子类",
    "实例",
    "实现类",
    "特点",
    "特性",
    "作用",
    "用途",
    "概念",
    "定义",
    "修饰符",
    "默认值",
}


def discover_relation_files(relation_dir: Path) -> List[Path]:
    return sorted(path for path in relation_dir.glob("*.csv") if path.name != "all_relations.csv")


def save_rows(rows: Sequence[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(output_path, index=False, encoding="utf-8-sig")


def canonicalize_relation_name(relation: str) -> str:
    relation = normalize_name(relation)
    return RELATION_NAME_MAP.get(relation, relation)


def should_symmetrize(relation: str) -> bool:
    return relation in {"alias_of", "related_to"}


def infer_attribute_candidate(row: pd.Series) -> dict | None:
    relation = canonicalize_relation_name(str(row.get("relation", "")))
    head = normalize_name(str(row.get("head", "")))
    tail = normalize_name(str(row.get("tail", "")))
    if not head or not tail:
        return None

    tail_key = build_loose_key(tail)
    if relation in {"is_a", "has_part"} and (tail in GENERIC_TAILS or tail in ATTRIBUTE_VALUE_HINTS):
        attribute_name = ATTRIBUTE_NAME_MAP.get(tail, "特性" if tail in ATTRIBUTE_VALUE_HINTS else "定义")
    elif relation == "is_a" and tail_key.isascii() and tail_key.islower() and len(tail_key) <= 24:
        attribute_name = "特性"
    else:
        return None

    return {
        "entity": head,
        "attribute_name": attribute_name,
        "attribute_value": tail,
        "section_title": normalize_name(str(row.get("section_title", ""))),
        "source_file": normalize_name(str(row.get("source_file", ""))),
        "evidence": clean_text(str(row.get("evidence", ""))),
        "confidence": round(min(0.99, float(row.get("confidence", 0.70)) + 0.03), 4),
        "method": "relation_disambiguation_reclassified",
        "source_relation": relation,
    }


def normalize_relation_row(row: pd.Series) -> dict | None:
    head = normalize_name(str(row.get("head", "")))
    tail = normalize_name(str(row.get("tail", "")))
    relation = canonicalize_relation_name(str(row.get("relation", "")))
    if not head or not tail or not relation or head == tail:
        return None

    if should_symmetrize(relation):
        ordered = sorted([head, tail], key=lambda item: (len(item), item))
        head, tail = ordered[0], ordered[1]

    return {
        "head": head,
        "relation": relation,
        "tail": tail,
        "head_type": normalize_name(str(row.get("head_type", ""))),
        "tail_type": normalize_name(str(row.get("tail_type", ""))),
        "evidence": clean_text(str(row.get("evidence", ""))),
        "pattern_name": normalize_name(str(row.get("pattern_name", ""))),
        "section_title": normalize_name(str(row.get("section_title", ""))),
        "source_file": normalize_name(str(row.get("source_file", ""))),
        "confidence": round(float(row.get("confidence", 0.0)), 4),
        "method": "relation_disambiguation_normalized",
    }


def deduplicate_relations(rows: Sequence[dict]) -> List[dict]:
    best: Dict[Tuple[str, str, str, str], dict] = {}
    for row in rows:
        key = (row["head"], row["relation"], row["tail"], row["source_file"])
        current = best.get(key)
        if current is None or float(row["confidence"]) >= float(current["confidence"]):
            best[key] = dict(row)
    output = list(best.values())
    output.sort(key=lambda item: (-float(item["confidence"]), item["head"], item["tail"]))
    return output


def deduplicate_attributes(rows: Sequence[dict]) -> List[dict]:
    best: Dict[Tuple[str, str, str, str], dict] = {}
    for row in rows:
        key = (row["entity"], row["attribute_name"], row["attribute_value"], row["source_file"])
        current = best.get(key)
        if current is None or float(row["confidence"]) >= float(current["confidence"]):
            best[key] = dict(row)
    output = list(best.values())
    output.sort(key=lambda item: (-float(item["confidence"]), item["entity"], item["attribute_name"]))
    return output


def process_relation_file(csv_path: Path) -> Tuple[List[dict], List[dict]]:
    dataframe = pd.read_csv(csv_path, encoding="utf-8-sig")
    if dataframe.empty:
        return [], []

    relation_rows: List[dict] = []
    attribute_rows: List[dict] = []
    for _, row in dataframe.iterrows():
        attribute_candidate = infer_attribute_candidate(row)
        if attribute_candidate is not None:
            attribute_rows.append(attribute_candidate)
            continue
        normalized = normalize_relation_row(row)
        if normalized is not None:
            relation_rows.append(normalized)

    return deduplicate_relations(relation_rows), deduplicate_attributes(attribute_rows)
