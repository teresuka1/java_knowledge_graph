from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models import EntityRecord
from scoring import bounded_score, jaccard, token_overlap_score, variant_relation_score
from text_utils import build_loose_key


class LocalCrossEncoderReranker:
    def __init__(
        self,
        records: Sequence[EntityRecord],
        name_similarity: np.ndarray,
        context_similarity: np.ndarray,
    ) -> None:
        self.records = records
        self.name_similarity = name_similarity
        self.context_similarity = context_similarity
        self.backend_name = "LocalCrossEncoder"

    def score(self, left_id: int, right_id: int) -> Tuple[float, List[str]]:
        left = self.records[left_id]
        right = self.records[right_id]
        evidence: List[str] = []

        exact_norm = 1.0 if left.normalized_main == right.normalized_main else 0.0
        exact_loose = 1.0 if left.loose_key and left.loose_key == right.loose_key else 0.0
        alias_overlap = jaccard(left.alias_forms, right.alias_forms)
        section_overlap = jaccard(set(left.sections), set(right.sections))
        name_score = float(self.name_similarity[left_id, right_id])
        context_score = float(self.context_similarity[left_id, right_id])
        variant_score = variant_relation_score(left.main_entity, right.main_entity)

        score = (
            0.30 * max(exact_norm, exact_loose)
            + 0.17 * alias_overlap
            + 0.18 * name_score
            + 0.20 * context_score
            + 0.08 * section_overlap
            + 0.07 * variant_score
        )
        score = bounded_score(score)

        if exact_norm:
            evidence.append("标准化一致")
        if exact_loose and not exact_norm:
            evidence.append("去空格/符号后一致")
        if alias_overlap >= 0.34:
            evidence.append(f"别名重叠={alias_overlap:.2f}")
        if name_score >= 0.50:
            evidence.append(f"名称相似={name_score:.2f}")
        if context_score >= 0.42:
            evidence.append(f"上下文相似={context_score:.2f}")
        if section_overlap > 0:
            evidence.append(f"章节重叠={section_overlap:.2f}")
        if variant_score >= 0.70:
            evidence.append("存在可接受标准化变体")
        return score, evidence


def compute_similarity_matrix(texts: Sequence[str]) -> np.ndarray:
    clean_docs = [doc if doc and doc.strip() else "__empty__" for doc in texts]
    if len(clean_docs) <= 1:
        return np.eye(len(clean_docs), dtype=float)
    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        matrix = vectorizer.fit_transform(clean_docs)
        return cosine_similarity(matrix)
    except ValueError:
        return np.eye(len(clean_docs), dtype=float)


def build_candidate_pool(
    records: Sequence[EntityRecord],
    name_similarity: np.ndarray,
    context_similarity: np.ndarray,
) -> None:
    top_k = min(8, max(1, len(records) - 1))
    for record in records:
        candidates: set[int] = set()
        for other in records:
            if record.row_id == other.row_id:
                continue
            exact_match = record.normalized_main == other.normalized_main
            loose_match = record.loose_key and record.loose_key == other.loose_key
            alias_overlap = not record.alias_forms.isdisjoint(other.alias_forms)
            variant_score = variant_relation_score(record.main_entity, other.main_entity)
            if exact_match or loose_match or alias_overlap or variant_score >= 0.70:
                candidates.add(other.row_id)

        name_order = np.argsort(name_similarity[record.row_id])[::-1]
        ctx_order = np.argsort(context_similarity[record.row_id])[::-1]
        for idx in name_order:
            if idx == record.row_id:
                continue
            if name_similarity[record.row_id, idx] >= 0.22:
                candidates.add(int(idx))
            if len(candidates) >= top_k + 4:
                break
        for idx in ctx_order:
            if idx == record.row_id:
                continue
            if context_similarity[record.row_id, idx] >= 0.18:
                candidates.add(int(idx))
            if len(candidates) >= top_k + 6:
                break
        record.recall_candidates = sorted(candidates)


def type_section_adjustment(left: EntityRecord, right: EntityRecord) -> Tuple[float, List[str]]:
    evidence: List[str] = []
    adjustment = 0.0

    left_type = build_loose_key(left.entity_type)
    right_type = build_loose_key(right.entity_type)
    if left_type and right_type and left_type == right_type:
        adjustment += 0.10
        evidence.append("类型一致")
    else:
        overlap = token_overlap_score(left.entity_type, right.entity_type)
        if overlap >= 0.5:
            adjustment += 0.03
            evidence.append("类型部分重叠")
        else:
            adjustment -= 0.10
            evidence.append("类型冲突")

    section_overlap = jaccard(set(left.sections), set(right.sections))
    if section_overlap >= 0.5:
        adjustment += 0.07
    elif left.sections and right.sections and section_overlap == 0:
        adjustment -= 0.04
    return adjustment, evidence


def graph_consistency_score(
    left_id: int,
    right_id: int,
    graph: Dict[int, Counter],
    records: Sequence[EntityRecord],
) -> Tuple[float, List[str]]:
    left_neighbors = {
        records[idx].loose_key: weight
        for idx, weight in graph.get(left_id, {}).items()
        if idx != right_id and records[idx].loose_key
    }
    right_neighbors = {
        records[idx].loose_key: weight
        for idx, weight in graph.get(right_id, {}).items()
        if idx != left_id and records[idx].loose_key
    }
    if not left_neighbors or not right_neighbors:
        return 0.0, []

    shared_weight = 0.0
    total_weight = 0.0
    all_keys = set(left_neighbors) | set(right_neighbors)
    for key in all_keys:
        left_weight = float(left_neighbors.get(key, 0))
        right_weight = float(right_neighbors.get(key, 0))
        shared_weight += min(left_weight, right_weight)
        total_weight += max(left_weight, right_weight)

    if total_weight == 0:
        return 0.0, []

    score = shared_weight / total_weight
    evidence = [f"共现邻居重叠={score:.2f}"] if score > 0 else []
    return score, evidence


def should_merge(
    left: EntityRecord,
    right: EntityRecord,
    cross_score: float,
    type_adjustment: float,
    graph_score: float,
) -> bool:
    exact = left.normalized_main == right.normalized_main
    loose = left.loose_key and left.loose_key == right.loose_key
    alias_overlap = not left.alias_forms.isdisjoint(right.alias_forms)
    variant_score = variant_relation_score(left.main_entity, right.main_entity)
    compatible_type = type_adjustment >= -0.01
    final_score = bounded_score(cross_score + type_adjustment + 0.15 * graph_score)

    if exact or loose:
        return final_score >= 0.58
    if alias_overlap and compatible_type:
        return final_score >= 0.66
    if variant_score >= 0.85 and compatible_type:
        return final_score >= 0.76
    return False
