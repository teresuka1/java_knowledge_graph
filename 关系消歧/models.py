from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class RelationRecord:
    row_id: int
    input_file: str
    raw: Dict[str, str]
    head: str
    relation: str
    tail: str
    head_type: str = ""
    tail_type: str = ""
    evidence: str = ""
    pattern_name: str = ""
    section_title: str = ""
    source_file: str = ""
    confidence: float = 0.0
    method: str = ""


@dataclass
class CanonicalRecord:
    record: RelationRecord
    head: str
    relation: str
    tail: str
    head_key: str
    tail_key: str
    head_type: str = ""
    tail_type: str = ""
    actions: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CandidateScore:
    relation: str
    score: float
    avg_confidence: float
    max_confidence: float
    cue_score: float
    support_score: float
    relation_priority: float
    support: int
    cue_hits: List[str]


@dataclass
class DisambiguationResult:
    output_rows: List[Dict[str, str]]
    decision_rows: List[Dict[str, str]]
    summary: Dict[str, int]
    examples: List[Dict[str, str]]
