from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_store import build_graph_payload, save_csv, save_json, save_sqlite  # noqa: E402


def choose_first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_default_paths(script_path: Path) -> tuple[Path, Path, Path, Path]:
    project_root = script_path.parent.parent
    entity_dir = choose_first_existing(
        [
            project_root / "实体消歧" / "实体消歧结果",
            project_root / "实体抽取结果",
        ]
    )
    relation_path = choose_first_existing(
        [
            project_root / "关系消歧" / "关系消歧结果" / "all_relations_disambiguated.csv",
            project_root / "关系抽取结果" / "all_relations.csv",
        ]
    )
    attribute_path = choose_first_existing(
        [
            project_root / "属性消歧结果" / "all_properties.csv",
            project_root / "属性消歧" / "属性消歧结果" / "all_attributes_disambiguated.csv",
        ]
    )
    output_dir = script_path.parent / "graph_store"
    return entity_dir, relation_path, attribute_path, output_dir


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_entity, default_relation, default_attribute, default_output = build_default_paths(script_path)
    parser = argparse.ArgumentParser(description="Persist the latest knowledge graph results into CSV, JSON, and SQLite outputs.")
    parser.add_argument("--entity-dir", type=Path, default=default_entity, help="Directory of entity CSV files.")
    parser.add_argument("--relation-path", type=Path, default=default_relation, help="Path of relation CSV file.")
    parser.add_argument("--attribute-path", type=Path, default=default_attribute, help="Path of attribute/property CSV file.")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Directory of graph storage outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_graph_payload(args.entity_dir, args.relation_path, args.attribute_path)

    output_dir = args.output_dir
    save_csv(payload["nodes"], output_dir / "nodes.csv")
    save_csv(payload["edges"], output_dir / "edges.csv")
    save_csv(payload["attributes"], output_dir / "attributes.csv")
    save_json(payload, output_dir / "graph.json")
    save_sqlite(payload["nodes"], payload["edges"], payload["attributes"], output_dir / "knowledge_graph.db")

    stats = payload["stats"]
    print(
        f"Knowledge graph stored. nodes={stats['nodeCount']}, edges={stats['edgeCount']}, "
        f"attributes={stats['attributeCount']}, output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
