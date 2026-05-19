from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from text_utils import dedupe_preserve_order, relation_key


@dataclass(frozen=True)
class RelationSpec:
    canonical: str
    aliases: tuple[str, ...]
    swap: bool = False
    symmetric: bool = False
    priority: float = 0.5
    cues: tuple[str, ...] = ()
    negative_cues: tuple[str, ...] = ()


class RelationTaxonomy:
    def __init__(self, specs: Iterable[RelationSpec], unknown_relation_priority: float = 0.45) -> None:
        self.specs = list(specs)
        self.unknown_relation_priority = unknown_relation_priority
        self._alias_index: Dict[str, RelationSpec] = {}
        self._profiles: Dict[str, RelationSpec] = {}
        self._build_indexes()

    @classmethod
    def from_json(cls, path: Path) -> "RelationTaxonomy":
        data = json.loads(path.read_text(encoding="utf-8"))
        specs: List[RelationSpec] = []
        for item in data.get("relations", []):
            aliases = tuple(str(alias) for alias in item.get("aliases", []))
            canonical = str(item["canonical"])
            if canonical not in aliases and not bool(item.get("swap", False)):
                aliases = (canonical, *aliases)
            specs.append(
                RelationSpec(
                    canonical=canonical,
                    aliases=aliases,
                    swap=bool(item.get("swap", False)),
                    symmetric=bool(item.get("symmetric", False)),
                    priority=float(item.get("priority", 0.5)),
                    cues=tuple(str(cue) for cue in item.get("cues", [])),
                    negative_cues=tuple(str(cue) for cue in item.get("negative_cues", [])),
                )
            )
        return cls(specs, float(data.get("unknown_relation_priority", 0.45)))

    def _build_indexes(self) -> None:
        grouped: Dict[str, List[RelationSpec]] = {}
        for spec in self.specs:
            for alias in spec.aliases:
                self._alias_index[relation_key(alias)] = spec
            grouped.setdefault(spec.canonical, []).append(spec)

        for canonical, specs in grouped.items():
            aliases = dedupe_preserve_order(alias for spec in specs for alias in spec.aliases)
            cues = dedupe_preserve_order(cue for spec in specs for cue in spec.cues)
            negative_cues = dedupe_preserve_order(cue for spec in specs for cue in spec.negative_cues)
            self._profiles[canonical] = RelationSpec(
                canonical=canonical,
                aliases=tuple(aliases),
                symmetric=any(spec.symmetric for spec in specs),
                priority=max(spec.priority for spec in specs),
                cues=tuple(cues),
                negative_cues=tuple(negative_cues),
            )

    def mapping_for(self, relation: str) -> RelationSpec:
        key = relation_key(relation)
        if key in self._alias_index:
            return self._alias_index[key]
        fallback = key or relation
        return RelationSpec(
            canonical=fallback,
            aliases=(fallback,),
            priority=self.unknown_relation_priority,
        )

    def profile_for(self, canonical_relation: str) -> RelationSpec:
        if canonical_relation in self._profiles:
            return self._profiles[canonical_relation]
        return RelationSpec(
            canonical=canonical_relation,
            aliases=(canonical_relation,),
            priority=self.unknown_relation_priority,
        )
