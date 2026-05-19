from __future__ import annotations

import argparse
from pathlib import Path

from disambiguator import RelationDisambiguator
from io_utils import discover_csv_files, process_file, write_report
from taxonomy import RelationTaxonomy


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    default_input_dir = project_root / "关系抽取结果"
    default_output_dir = script_path.parent / "关系消歧结果"
    default_taxonomy = script_path.parent / "default_relation_taxonomy.json"

    parser = argparse.ArgumentParser(
        description="Generic relation disambiguation for knowledge graph triples in CSV files."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input_dir, help="Directory containing relation CSVs.")
    parser.add_argument("--input-file", type=Path, default=None, help="Optional single CSV file to process.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="Directory for result CSVs.")
    parser.add_argument("--taxonomy", type=Path, default=default_taxonomy, help="Relation taxonomy JSON.")
    parser.add_argument(
        "--tie-margin",
        type=float,
        default=0.015,
        help="Score margin below which a conflict decision is flagged as low-margin in the audit CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = RelationTaxonomy.from_json(args.taxonomy)
    disambiguator = RelationDisambiguator(taxonomy=taxonomy, tie_margin=args.tie_margin)

    if args.input_file:
        input_files = [args.input_file]
        input_dir_for_report = args.input_file.parent
    else:
        input_files = discover_csv_files(args.input_dir)
        input_dir_for_report = args.input_dir

    if not input_files:
        print(f"No CSV files found in {args.input_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    examples = []
    for input_path in input_files:
        result = process_file(input_path, args.output_dir, disambiguator)
        file_summary = {"file": input_path.name, **result.summary}
        summaries.append(file_summary)
        examples.extend(result.examples)
        print(
            f"[done] {input_path.name}: input={result.summary['input_rows']}, "
            f"output={result.summary['output_rows']}, "
            f"conflict_pairs={result.summary['relation_conflict_pairs']}, "
            f"inverse_rows={result.summary['inverse_relation_rows']}"
        )

    write_report(
        args.output_dir / "消歧报告.md",
        input_dir=input_dir_for_report,
        output_dir=args.output_dir,
        summaries=summaries,
        examples=examples,
    )
    print(f"Report written to {args.output_dir / '消歧报告.md'}")


if __name__ == "__main__":
    main()
