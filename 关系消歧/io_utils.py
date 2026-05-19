from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from disambiguator import RelationDisambiguator
from models import DisambiguationResult, RelationRecord
from text_utils import normalize_entity, normalize_text, parse_float


FIELD_ALIASES = {
    "head": ("head", "subject", "source", "from", "实体1", "头实体"),
    "relation": ("relation", "predicate", "edge", "type", "关系"),
    "tail": ("tail", "object", "target", "to", "实体2", "尾实体"),
    "head_type": ("head_type", "subject_type", "source_type", "头实体类型"),
    "tail_type": ("tail_type", "object_type", "target_type", "尾实体类型"),
    "evidence": ("evidence", "context", "sentence", "text", "证据", "上下文"),
    "pattern_name": ("pattern_name", "pattern", "rule", "规则"),
    "section_title": ("section_title", "section", "title", "章节"),
    "source_file": ("source_file", "file", "document", "来源文件"),
    "confidence": ("confidence", "score", "probability", "置信度"),
    "method": ("method", "extractor", "strategy", "方法"),
}


OUTPUT_FIELDS = [
    "head",
    "relation",
    "tail",
    "head_type",
    "tail_type",
    "confidence",
    "evidence",
    "pattern_names",
    "section_titles",
    "source_files",
    "record_count",
    "source_row_ids",
    "source_relations",
    "conflicting_relations",
    "disambiguation_score",
    "disambiguation_method",
    "disambiguation_basis",
]


DECISION_FIELDS = [
    "input_file",
    "input_row_id",
    "original_head",
    "original_relation",
    "original_tail",
    "canonical_head",
    "canonical_relation",
    "canonical_tail",
    "kept_relation",
    "decision",
    "canonicalization_actions",
    "candidate_relations",
    "winner_score",
    "runner_up_score",
    "score_margin",
    "basis",
    "evidence",
]


def read_csv_records(path: Path) -> List[RelationRecord]:
    rows = _read_csv_dicts(path)
    records: List[RelationRecord] = []
    for row_id, row in enumerate(rows):
        head = normalize_entity(_pick(row, "head"))
        relation = normalize_text(_pick(row, "relation"))
        tail = normalize_entity(_pick(row, "tail"))
        if not head or not relation or not tail:
            continue
        records.append(
            RelationRecord(
                row_id=row_id,
                input_file=path.name,
                raw={str(key): normalize_text(value) for key, value in row.items()},
                head=head,
                relation=relation,
                tail=tail,
                head_type=normalize_text(_pick(row, "head_type")),
                tail_type=normalize_text(_pick(row, "tail_type")),
                evidence=normalize_text(_pick(row, "evidence")),
                pattern_name=normalize_text(_pick(row, "pattern_name")),
                section_title=normalize_text(_pick(row, "section_title")),
                source_file=normalize_text(_pick(row, "source_file")),
                confidence=parse_float(_pick(row, "confidence"), default=0.0),
                method=normalize_text(_pick(row, "method")),
            )
        )
    return records


def process_file(input_path: Path, output_dir: Path, disambiguator: RelationDisambiguator) -> DisambiguationResult:
    records = read_csv_records(input_path)
    result = disambiguator.disambiguate(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    write_csv(output_dir / f"{stem}_disambiguated.csv", result.output_rows, OUTPUT_FIELDS)
    write_csv(output_dir / f"{stem}_decisions.csv", result.decision_rows, DECISION_FIELDS)
    return result


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    if path.exists():
        path.unlink()
    temp_path.replace(path)


def discover_csv_files(input_dir: Path) -> List[Path]:
    return sorted(path for path in input_dir.glob("*.csv") if path.is_file())


def write_report(
    path: Path,
    input_dir: Path,
    output_dir: Path,
    summaries: Sequence[Dict[str, int | str]],
    examples: Sequence[Dict[str, str]],
) -> None:
    total_input = sum(int(item["input_rows"]) for item in summaries)
    total_output = sum(int(item["output_rows"]) for item in summaries)
    total_conflict_pairs = sum(int(item["relation_conflict_pairs"]) for item in summaries)
    total_inverse_rows = sum(int(item["inverse_relation_rows"]) for item in summaries)
    total_symmetric_rows = sum(int(item["symmetric_reordered_rows"]) for item in summaries)
    total_normalized_rows = sum(int(item["taxonomy_normalized_rows"]) for item in summaries)
    total_merged_rows = sum(int(item["merged_duplicate_rows"]) for item in summaries)

    lines = [
        "# 关系消歧运行报告",
        "",
        f"- 输入目录：`{input_dir}`",
        f"- 输出目录：`{output_dir}`",
        f"- 输入关系行数：{total_input}",
        f"- 输出规范关系行数：{total_output}",
        f"- 关系冲突实体对：{total_conflict_pairs}",
        f"- 互逆关系方向统一行数：{total_inverse_rows}",
        f"- 对称关系端点重排行数：{total_symmetric_rows}",
        f"- 关系标签规范化行数：{total_normalized_rows}",
        f"- 重复规范三元组合并行数：{total_merged_rows}",
        "",
        "## 分文件统计",
        "",
        "| 文件 | 输入行 | 输出行 | 冲突实体对 | 方向统一 | 标签规范化 | 重复合并 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| {item['file']} | {item['input_rows']} | {item['output_rows']} | "
            f"{item['relation_conflict_pairs']} | {item['inverse_relation_rows']} | "
            f"{item['taxonomy_normalized_rows']} | {item['merged_duplicate_rows']} |"
        )

    if examples:
        lines.extend(["", "## 冲突消歧示例", ""])
        for index, example in enumerate(examples[:12], start=1):
            lines.extend(
                [
                    f"### 示例 {index}",
                    "",
                    f"- 实体对：`{example['head']}` -> `{example['tail']}`",
                    f"- 保留关系：`{example['chosen_relation']}`",
                    f"- 候选关系：`{example['candidate_relations']}`",
                    f"- 依据：{example['basis']}",
                    f"- 证据：{example['evidence']}",
                    "",
                ]
            )

    path.write_text("\n".join(lines), encoding="utf-8")


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    raw = path.read_bytes()
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError as error:
            last_error = error
    else:
        if last_error:
            text = raw.decode("utf-8", errors="ignore")
        else:
            text = ""
    reader = csv.DictReader(text.splitlines())
    return [dict(row) for row in reader]


def _pick(row: Dict[str, str], logical_name: str) -> str:
    aliases = FIELD_ALIASES[logical_name]
    normalized_row = {str(key).strip().casefold(): value for key, value in row.items()}
    for alias in aliases:
        key = alias.casefold()
        if key in normalized_row:
            return normalized_row[key] or ""
    return ""
