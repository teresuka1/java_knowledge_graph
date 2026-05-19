from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attribute_disambiguator import deduplicate_rows, discover_attribute_files, process_attribute_file, save_rows  # noqa: E402


def build_default_paths(script_path: Path) -> tuple[Path, Path]:
    project_root = script_path.parent.parent
    attribute_dir = project_root / "属性抽取" / "属性抽取结果"
    output_dir = script_path.parent / "属性消歧结果"
    return attribute_dir, output_dir


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_attribute, default_output = build_default_paths(script_path)
    parser = argparse.ArgumentParser(description="Normalize and deduplicate extracted attributes.")
    parser.add_argument("--attribute-dir", type=Path, default=default_attribute, help="Directory of attribute csv files.")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Directory of disambiguated attribute csv files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_attribute_files(args.attribute_dir)
    if not files:
        print("No attribute csv files were found.")
        return

    all_rows: List[dict] = []
    for csv_path in files:
        rows = process_attribute_file(csv_path)
        all_rows.extend(rows)
        save_rows(rows, args.output_dir / csv_path.name)
        print(f"[done] {csv_path.name}: attributes={len(rows)}")

    merged_rows = deduplicate_rows(all_rows)
    save_rows(merged_rows, args.output_dir / "all_attributes_disambiguated.csv")
    print(f"Processed {len(files)} attribute files. attributes={len(merged_rows)}, output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
