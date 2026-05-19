from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault("PANDAS_NO_USE_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_USE_BOTTLENECK", "1")
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd

from kg_common import domain_from_source, normalize_name


def first_present(row: pd.Series, candidates: Iterable[str]) -> str:
    for candidate in candidates:
        if candidate in row and pd.notna(row[candidate]):
            value = normalize_name(str(row[candidate]))
            if value:
                return value
    return ""


def first_numeric(row: pd.Series, candidates: Iterable[str], default: int = 0) -> int:
    for candidate in candidates:
        if candidate in row and pd.notna(row[candidate]):
            try:
                return int(float(row[candidate]))
            except (TypeError, ValueError):
                continue
    return default


def pick_first_source_file(value: str, fallback: str = "") -> str:
    raw = normalize_name(value)
    if not raw:
        return fallback
    if " | " in raw:
        return normalize_name(raw.split(" | ")[0])
    return raw


def load_entity_rows(entity_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for csv_path in sorted(entity_dir.glob("*.csv")):
        dataframe = pd.read_csv(csv_path, encoding="utf-8-sig")
        if dataframe.empty:
            continue
        for _, row in dataframe.iterrows():
            source_file = first_present(row, ["source_file"]) or f"{csv_path.stem}.txt"
            domain = domain_from_source(source_file, csv_path.stem)
            name = first_present(row, ["main_entity", "entity", "name"])
            if not name:
                continue
            mentions = first_present(row, ["mentions", "aliases", "merged_entities"]) or name
            rows.append(
                {
                    "id": f"{domain}:{name}",
                    "name": name,
                    "type": first_present(row, ["entity_type", "type"]) or None,
                    "domain": domain,
                    "mentionCount": first_numeric(row, ["mention_count", "mentionCount"], default=0),
                    "mentions": mentions,
                    "sourceFile": source_file or None,
                }
            )
    return rows


def load_relation_rows(relation_path: Path) -> List[dict]:
    if not relation_path.exists():
        return []
    dataframe = pd.read_csv(relation_path, encoding="utf-8-sig")
    if dataframe.empty:
        return []
    rows: List[dict] = []
    for index, row in dataframe.reset_index(drop=True).iterrows():
        source_file = pick_first_source_file(first_present(row, ["source_file", "source_files"]))
        domain = first_present(row, ["domain"]) or domain_from_source(source_file)
        head = first_present(row, ["head", "source", "sourceName"])
        tail = first_present(row, ["tail", "target", "targetName"])
        relation = first_present(row, ["relation"])
        if not head or not tail or not relation:
            continue
        evidence = first_present(row, ["evidence"])
        pattern_name = first_present(row, ["pattern_name", "pattern_names"])
        section_title = first_present(row, ["section_title", "section_titles"])
        method = first_present(row, ["method", "disambiguation_method"])
        confidence = None
        for candidate in ("confidence", "disambiguation_score"):
            if candidate in row and pd.notna(row[candidate]):
                try:
                    confidence = float(row[candidate])
                    break
                except (TypeError, ValueError):
                    continue
        rows.append(
            {
                "id": f"edge:{index}:{domain}:{head}:{relation}:{tail}",
                "source": f"{domain}:{head}",
                "target": f"{domain}:{tail}",
                "relation": relation,
                "sourceName": head,
                "targetName": tail,
                "sourceType": first_present(row, ["head_type", "source_type", "sourceType"]) or None,
                "targetType": first_present(row, ["tail_type", "target_type", "targetType"]) or None,
                "confidence": confidence,
                "evidence": evidence or None,
                "patternName": pattern_name or None,
                "sourceFile": source_file or None,
                "sectionTitle": section_title or None,
                "domain": domain,
                "method": method or None,
            }
        )
    return rows


def load_attribute_rows(attribute_path: Path) -> List[dict]:
    if not attribute_path.exists():
        return []
    dataframe = pd.read_csv(attribute_path, encoding="utf-8-sig")
    if dataframe.empty:
        return []
    rows: List[dict] = []
    for index, row in dataframe.reset_index(drop=True).iterrows():
        source_file = pick_first_source_file(first_present(row, ["source_file", "source_files"]))
        domain = first_present(row, ["domain"]) or domain_from_source(source_file)
        entity = first_present(row, ["entity", "main_entity"])
        attribute_name = first_present(row, ["attribute_name", "property_name"])
        attribute_value = first_present(row, ["attribute_value", "merged_values", "property_value"])
        if not entity or not attribute_name or not attribute_value:
            continue
        confidence = None
        if "confidence" in row and pd.notna(row["confidence"]):
            try:
                confidence = float(row["confidence"])
            except (TypeError, ValueError):
                confidence = None
        rows.append(
            {
                "id": f"attr:{index}:{domain}:{entity}:{attribute_name}",
                "nodeId": f"{domain}:{entity}",
                "entity": entity,
                "attributeName": attribute_name,
                "attributeValue": attribute_value,
                "valueCount": first_numeric(row, ["value_count"], default=1),
                "confidence": confidence,
                "evidence": first_present(row, ["evidence"]) or None,
                "sourceFile": source_file or None,
                "sectionTitle": first_present(row, ["section_title", "section_titles"]) or None,
                "domain": domain,
                "method": first_present(row, ["disambiguation_method", "method"]) or None,
            }
        )
    return rows


def ensure_relation_nodes(nodes: Dict[str, dict], edges: List[dict]) -> None:
    for edge in edges:
        for node_id, node_name, node_type in (
            (edge["source"], edge["sourceName"], edge["sourceType"]),
            (edge["target"], edge["targetName"], edge["targetType"]),
        ):
            if node_id in nodes:
                continue
            domain, _, _ = node_id.partition(":")
            nodes[node_id] = {
                "id": node_id,
                "name": node_name,
                "type": node_type,
                "domain": domain,
                "mentionCount": 0,
                "mentions": node_name,
                "sourceFile": edge["sourceFile"] or (f"{domain}.txt" if domain else None),
            }


def save_csv(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def save_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_sqlite(nodes: List[dict], edges: List[dict], attributes: List[dict], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT,
                domain TEXT NOT NULL,
                mention_count INTEGER NOT NULL,
                mentions TEXT,
                source_file TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE edges (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                relation TEXT NOT NULL,
                source_name TEXT NOT NULL,
                target_name TEXT NOT NULL,
                source_type TEXT,
                target_type TEXT,
                confidence REAL,
                evidence TEXT,
                pattern_name TEXT,
                source_file TEXT,
                section_title TEXT,
                domain TEXT NOT NULL,
                method TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE attributes (
                id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                entity TEXT NOT NULL,
                attribute_name TEXT NOT NULL,
                attribute_value TEXT NOT NULL,
                value_count INTEGER NOT NULL,
                confidence REAL,
                evidence TEXT,
                source_file TEXT,
                section_title TEXT,
                domain TEXT NOT NULL,
                method TEXT
            )
            """
        )
        cursor.executemany(
            "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["id"],
                    row["name"],
                    row["type"],
                    row["domain"],
                    row["mentionCount"],
                    row["mentions"],
                    row["sourceFile"],
                )
                for row in nodes
            ],
        )
        cursor.executemany(
            "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["id"],
                    row["source"],
                    row["target"],
                    row["relation"],
                    row["sourceName"],
                    row["targetName"],
                    row["sourceType"],
                    row["targetType"],
                    row["confidence"],
                    row["evidence"],
                    row["patternName"],
                    row["sourceFile"],
                    row["sectionTitle"],
                    row["domain"],
                    row["method"],
                )
                for row in edges
            ],
        )
        cursor.executemany(
            "INSERT INTO attributes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["id"],
                    row["nodeId"],
                    row["entity"],
                    row["attributeName"],
                    row["attributeValue"],
                    row["valueCount"],
                    row["confidence"],
                    row["evidence"],
                    row["sourceFile"],
                    row["sectionTitle"],
                    row["domain"],
                    row["method"],
                )
                for row in attributes
            ],
        )
        connection.commit()
    finally:
        connection.close()


def build_graph_payload(entity_dir: Path, relation_path: Path, attribute_path: Path) -> dict:
    entity_rows = load_entity_rows(entity_dir)
    nodes = {row["id"]: row for row in entity_rows}
    edges = load_relation_rows(relation_path)
    ensure_relation_nodes(nodes, edges)
    attributes = load_attribute_rows(attribute_path)

    attribute_count: Dict[str, int] = {}
    for item in attributes:
        attribute_count[item["nodeId"]] = attribute_count.get(item["nodeId"], 0) + 1

    node_rows = list(nodes.values())
    for row in node_rows:
        row["attributeCount"] = attribute_count.get(row["id"], 0)
    node_rows.sort(key=lambda item: (item["domain"], -item["mentionCount"], item["name"]))
    edges.sort(key=lambda item: (item["domain"], item["relation"], item["source"], item["target"]))
    attributes.sort(key=lambda item: (item["domain"], item["entity"], item["attributeName"]))

    return {
        "nodes": node_rows,
        "edges": edges,
        "attributes": attributes,
        "stats": {
            "nodeCount": len(node_rows),
            "edgeCount": len(edges),
            "attributeCount": len(attributes),
            "domainCount": len(sorted({row["domain"] for row in node_rows})),
            "domains": sorted({row["domain"] for row in node_rows}),
            "relationTypes": sorted({row["relation"] for row in edges}),
        },
    }
