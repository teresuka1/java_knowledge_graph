from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class RelationPattern:
    relation: str
    name: str
    cues: Sequence[str]
    direction: str
    base_score: float
    max_entity_gap: int = 48


RELATION_PATTERNS: List[RelationPattern] = [
    RelationPattern(
        relation="alias_of",
        name="别名关系",
        cues=(r"又称", r"也称", r"也叫", r"亦称", r"简称", r"缩写", r"即", r"俗称"),
        direction="both",
        base_score=0.92,
        max_entity_gap=20,
    ),
    RelationPattern(
        relation="is_a",
        name="上下位关系",
        cues=(r"是", r"属于", r"归类为", r"可视为", r"可看作"),
        direction="lr",
        base_score=0.83,
        max_entity_gap=36,
    ),
    RelationPattern(
        relation="is_a",
        name="上位包含关系",
        cues=(r"包括", r"包含", r"分为", r"由.+?组成"),
        direction="rl",
        base_score=0.78,
        max_entity_gap=48,
    ),
    RelationPattern(
        relation="part_of",
        name="部分整体关系",
        cues=(r"是.+?一部分", r"组成", r"构成", r"隶属于"),
        direction="lr",
        base_score=0.80,
        max_entity_gap=36,
    ),
    RelationPattern(
        relation="has_part",
        name="整体部分关系",
        cues=(r"包含", r"包括", r"由.+?构成", r"由.+?组成"),
        direction="lr",
        base_score=0.76,
        max_entity_gap=48,
    ),
    RelationPattern(
        relation="used_for",
        name="用途关系",
        cues=(r"用于", r"用来", r"用作", r"可用于", r"常用于", r"适用于"),
        direction="lr",
        base_score=0.81,
        max_entity_gap=48,
    ),
    RelationPattern(
        relation="depends_on",
        name="依赖关系",
        cues=(r"依赖", r"基于", r"建立在", r"取决于"),
        direction="lr",
        base_score=0.82,
        max_entity_gap=42,
    ),
    RelationPattern(
        relation="causes",
        name="因果关系",
        cues=(r"导致", r"引起", r"造成", r"使得", r"会使"),
        direction="lr",
        base_score=0.80,
        max_entity_gap=42,
    ),
    RelationPattern(
        relation="implemented_by",
        name="实现关系",
        cues=(r"实现", r"由.+?实现"),
        direction="rl",
        base_score=0.84,
        max_entity_gap=40,
    ),
    RelationPattern(
        relation="extends",
        name="继承关系",
        cues=(r"继承", r"扩展自", r"派生自"),
        direction="lr",
        base_score=0.85,
        max_entity_gap=36,
    ),
]
