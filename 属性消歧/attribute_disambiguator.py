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

from kg_common import clean_text, normalize_name


ATTRIBUTE_NAME_MAP = {
    "端口": "默认端口",
    "端口号": "默认端口",
    "默认端口号": "默认端口",
    "定义": "定义",
    "概念": "定义",
    "特点": "特性",
    "特征": "特性",
    "性质": "特性",
    "修饰符": "修饰符",
    "作用": "用途",
    "用途": "用途",
    "默认引擎": "默认存储引擎",
    "存储引擎": "默认存储引擎",
    "默认存储引擎": "默认存储引擎",
    "底层实现": "底层结构",
    "底层结构": "底层结构",
    "数据结构": "底层结构",
}


def discover_attribute_files(attribute_dir: Path) -> List[Path]:
    return sorted(path for path in attribute_dir.glob("*.csv") if path.name != "all_attributes.csv")


def canonical_attribute_name(name: str) -> str:
    normalized = normalize_name(name)
    return ATTRIBUTE_NAME_MAP.get(normalized, normalized)


def normalize_attribute_value(value: str) -> str:
    value = clean_text(value)
    value = value.strip("：:，,；;。.")
    if value.lower().startswith("用于"):
        value = value[2:]
    return normalize_name(value)


def process_attribute_file(csv_path: Path) -> List[dict]:
    dataframe = pd.read_csv(csv_path, encoding="utf-8-sig")
    if dataframe.empty:
        return []

    rows: List[dict] = []
    for _, row in dataframe.iterrows():
        entity = normalize_name(str(row.get("entity", "")))
        name = canonical_attribute_name(str(row.get("attribute_name", "")))
        value = normalize_attribute_value(str(row.get("attribute_value", "")))
        if not entity or not name or not value:
            continue
        rows.append(
            {
                "entity": entity,
                "attribute_name": name,
                "attribute_value": value,
                "section_title": normalize_name(str(row.get("section_title", ""))),
                "source_file": normalize_name(str(row.get("source_file", ""))),
                "evidence": clean_text(str(row.get("evidence", ""))),
                "confidence": round(float(row.get("confidence", 0.0)), 4),
                "method": "attribute_disambiguation_normalized",
            }
        )
    return deduplicate_rows(rows)


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


def save_rows(rows: Sequence[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(output_path, index=False, encoding="utf-8-sig")
