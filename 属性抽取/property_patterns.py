from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

@dataclass(frozen=True)
class PropertyPattern:
    property_name: str
    name: str
    cues: Sequence[str]
    direction: str
    base_score: float
    max_value_len: int = 150

PROPERTY_PATTERNS: List[PropertyPattern] = [
    PropertyPattern(
        property_name="definition",
        name="概念定义",
        cues=(r"是指", r"可以被定义为", r"通常是指", r"是"),
        direction="lr",
        base_score=0.92,
        max_value_len=180,
    ),
    PropertyPattern(
        property_name="features",
        name="特性特点",
        cues=(r"特点(?:是|包括|有)", r"特性(?:是|包括|有)", r"特征(?:是|包括|有)", r"优点(?:是|包括|有)", r"缺点(?:是|包括|有)"),
        direction="lr",
        base_score=0.88,
        max_value_len=150,
    ),
    PropertyPattern(
        property_name="author",
        name="开发者",
        cues=(r"由.+?开发", r"由.+?提出", r"由.+?开源", r"由.+?创建"),
        direction="rl",
        base_score=0.95,
        max_value_len=40,
    ),
    PropertyPattern(
        property_name="usage",
        name="用途作用",
        cues=(r"主要用于", r"常用于", r"的作用是", r"可用于", r"用来"),
        direction="lr",
        base_score=0.85,
        max_value_len=120,
    ),
    PropertyPattern(
        property_name="version",
        name="版本信息",
        cues=(r"版本为", r"最新版本是", r"发布了.+?版本"),
        direction="lr",
        base_score=0.90,
        max_value_len=30,
    )
]
