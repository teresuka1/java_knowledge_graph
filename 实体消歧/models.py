from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EntityRecord:
    row_id: int
    main_entity: str
    entity_type: str
    mention_count: int
    mentions: List[str]
    source_file: str
    normalized_main: str = ""
    loose_key: str = ""
    alias_forms: set[str] = field(default_factory=set)
    sections: List[str] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    recall_candidates: List[int] = field(default_factory=list)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return
        if self.rank[pa] < self.rank[pb]:
            pa, pb = pb, pa
        self.parent[pb] = pa
        if self.rank[pa] == self.rank[pb]:
            self.rank[pa] += 1
