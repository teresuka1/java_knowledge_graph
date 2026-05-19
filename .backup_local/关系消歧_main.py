from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from relation_disambiguator import (  # noqa: E402
    deduplicate_attributes,
    deduplicate_relations,
    discover_relation_files,
    process_relation_file,
    save_rows,
)


def build_default_paths(script_path: Path) -> tuple[Path, Path]:
    project_root = script_path.parent.parent
    relation_dir = project_root / "关系抽取结果"
    output_dir = script_path.parent / "关系消歧结果"
    return relation_dir, output_dir


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_relation, default_output = build_default_paths(script_path)
    parser = argparse.ArgumentParser(description="Normalize, deduplicate, and refine extracted relations.")
    parser.add_argument("--relation-dir", type=Path, default=default_relation, help="Directory of relation csv files.")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Directory of disambiguated relation csv files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_relation_files(args.relation_dir)
    if not files:
        print("No relation csv files were found.")
        return

    all_relations: List[dict] = []
    all_attribute_candidates: List[dict] = []
    for csv_path in files:
        relation_rows, attribute_rows = process_relation_file(csv_path)
        all_relations.extend(relation_rows)
        all_attribute_candidates.extend(attribute_rows)
        save_rows(relation_rows, args.output_dir / csv_path.name)
        print(
            f"[done] {csv_path.name}: normalized_relations={len(relation_rows)}, "
            f"reclassified_attributes={len(attribute_rows)}"
        )

    merged_relations = deduplicate_relations(all_relations)
    merged_attributes = deduplicate_attributes(all_attribute_candidates)
    save_rows(merged_relations, args.output_dir / "all_relations_disambiguated.csv")
    save_rows(merged_attributes, args.output_dir / "relation_attribute_candidates.csv")
    print(
        f"Processed {len(files)} relation files. "
        f"relations={len(merged_relations)}, attribute_candidates={len(merged_attributes)}, output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
