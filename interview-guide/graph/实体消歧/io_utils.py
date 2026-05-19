from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from models import EntityRecord
from pipeline import cluster_records
from text_utils import build_alias_forms, build_loose_key, normalize_name, read_text, split_mentions


def load_records(csv_path: Path) -> List[EntityRecord]:
    dataframe = pd.read_csv(csv_path, encoding="utf-8-sig")
    if dataframe.empty:
        return []

    records: List[EntityRecord] = []
    for row_id, row in dataframe.reset_index(drop=True).iterrows():
        main_entity = normalize_name(str(row.get("main_entity", "") or ""))
        entity_type = normalize_name(str(row.get("entity_type", "") or ""))
        mention_count = int(row.get("mention_count", 1) or 1)
        mentions = split_mentions(row.get("mentions"))
        source_file = normalize_name(str(row.get("source_file", "") or ""))
        record = EntityRecord(
            row_id=row_id,
            main_entity=main_entity,
            entity_type=entity_type,
            mention_count=mention_count,
            mentions=mentions,
            source_file=source_file,
        )
        record.normalized_main = normalize_name(main_entity)
        record.loose_key = build_loose_key(main_entity)
        record.alias_forms = build_alias_forms(main_entity, mentions)
        records.append(record)
    return records


def process_pair(raw_text_path: Path, csv_path: Path, output_path: Path) -> Tuple[int, int]:
    text = read_text(raw_text_path)
    records = load_records(csv_path)
    rows, merged_count = cluster_records(records, text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    pd.DataFrame(rows).to_csv(temp_path, index=False, encoding="utf-8-sig")
    if output_path.exists():
        output_path.unlink()
    temp_path.replace(output_path)
    return len(records), merged_count


def discover_pairs(raw_dir: Path, extract_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for raw_path in sorted(raw_dir.glob("*.txt")):
        csv_path = extract_dir / f"{raw_path.stem}.csv"
        if csv_path.exists():
            pairs.append((raw_path, csv_path))
    return pairs


def build_default_paths(script_path: Path) -> Tuple[Path, Path, Path]:
    project_root = script_path.parent.parent
    raw_dir = project_root / "原始文本"
    extract_dir = project_root / "实体抽取结果"
    output_dir = script_path.parent / "实体消歧结果"
    return raw_dir, extract_dir, output_dir
