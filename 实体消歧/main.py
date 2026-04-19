from __future__ import annotations

import argparse
from pathlib import Path

from io_utils import build_default_paths, discover_pairs, process_pair


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_raw, default_extract, default_output = build_default_paths(script_path)

    parser = argparse.ArgumentParser(description="Entity disambiguation for paired text/csv files.")
    parser.add_argument("--raw-dir", type=Path, default=default_raw, help="Directory of raw txt files.")
    parser.add_argument("--extract-dir", type=Path, default=default_extract, help="Directory of extracted entity csv files.")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Directory of disambiguated csv files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = discover_pairs(args.raw_dir, args.extract_dir)
    if not pairs:
        print("No paired txt/csv files were found.")
        return

    processed = 0
    total_entities = 0
    total_merges = 0
    for raw_path, csv_path in pairs:
        output_path = args.output_dir / csv_path.name
        entity_count, merge_count = process_pair(raw_path, csv_path, output_path)
        processed += 1
        total_entities += entity_count
        total_merges += merge_count
        print(f"[done] {csv_path.name}: entities={entity_count}, merged_groups={merge_count}, output={output_path}")

    print(
        f"Processed {processed} file pairs. "
        f"Input entities={total_entities}, merged_links={total_merges}, output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
