from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from relation_extractor import extract_document_relations, load_entities, read_text


def build_default_paths(script_path: Path) -> Tuple[Path, Path, Path]:
    project_root = script_path.parent.parent
    raw_dir = project_root / "原始文本"
    entity_dir = project_root / "实体消歧" / "实体消歧结果"
    if not entity_dir.exists():
        entity_dir = project_root / "实体抽取结果"
    output_dir = project_root / "关系抽取结果"
    return raw_dir, entity_dir, output_dir


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_raw, default_entity, default_output = build_default_paths(script_path)
    parser = argparse.ArgumentParser(description="通用关系抽取：输入文本与实体表，输出知识图谱三元组。")
    parser.add_argument("--raw-dir", type=Path, default=default_raw, help="原始文本目录（*.txt）")
    parser.add_argument("--entity-dir", type=Path, default=default_entity, help="实体表目录（*.csv）")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="关系结果输出目录")
    return parser.parse_args()


def discover_pairs(raw_dir: Path, entity_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for raw_path in sorted(raw_dir.glob("*.txt")):
        csv_path = entity_dir / f"{raw_path.stem}.csv"
        if csv_path.exists():
            pairs.append((raw_path, csv_path))
    return pairs


def save_rows(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    pairs = discover_pairs(args.raw_dir, args.entity_dir)
    if not pairs:
        print("未找到可配对的 txt/csv 文件。请检查 --raw-dir 与 --entity-dir。")
        return

    all_rows: List[dict] = []
    total_entities = 0
    for raw_path, csv_path in pairs:
        text = read_text(raw_path)
        entities = load_entities(csv_path)
        total_entities += len(entities)

        rows = extract_document_relations(text=text, entities=entities, source_file=raw_path.name)
        all_rows.extend(rows)

        output_path = args.output_dir / f"{raw_path.stem}.csv"
        save_rows(rows, output_path)
        print(
            f"[done] {raw_path.stem}: entities={len(entities)}, relations={len(rows)}, output={output_path}"
        )

    merged_output = args.output_dir / "all_relations.csv"
    save_rows(all_rows, merged_output)
    print(
        f"处理完成：files={len(pairs)}, entities={total_entities}, "
        f"relations={len(all_rows)}, output={merged_output}"
    )


if __name__ == "__main__":
    main()
