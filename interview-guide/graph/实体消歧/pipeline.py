from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from models import EntityRecord, UnionFind
from reranker import (
    LocalCrossEncoderReranker,
    build_candidate_pool,
    compute_similarity_matrix,
    graph_consistency_score,
    should_merge,
    type_section_adjustment,
)
from scoring import bounded_score, deduplicate_preserve_order
from text_utils import contains_alias, normalize_name, parse_sections, split_sentences


def collect_record_context(records: Sequence[EntityRecord], text: str) -> None:
    sections = parse_sections(text)
    for record in records:
        section_hits: List[str] = []
        contexts: List[str] = []
        aliases = sorted(record.alias_forms, key=len, reverse=True)
        for section_title, section_text in sections:
            matched_in_section = False
            for sentence in split_sentences(section_text):
                if any(contains_alias(sentence, alias) for alias in aliases):
                    matched_in_section = True
                    contexts.append(sentence)
            if matched_in_section:
                section_hits.append(section_title)
        if not contexts:
            fallback_sentences = split_sentences(text)
            contexts = fallback_sentences[:2] if fallback_sentences else [record.main_entity]
            section_hits = ["全文"]
        record.sections = deduplicate_preserve_order(section_hits)
        record.contexts = deduplicate_preserve_order(contexts[:8])


def build_cooccurrence_graph(records: Sequence[EntityRecord], text: str) -> Dict[int, Counter]:
    graph: Dict[int, Counter] = defaultdict(Counter)
    sentence_units: List[str] = []
    for _, section_text in parse_sections(text):
        sentence_units.extend(split_sentences(section_text))

    for sentence in sentence_units:
        hit_ids: List[int] = []
        for record in records:
            aliases = sorted(record.alias_forms, key=len, reverse=True)
            if any(contains_alias(sentence, alias) for alias in aliases):
                hit_ids.append(record.row_id)
        hit_ids = deduplicate_preserve_order(hit_ids)
        if len(hit_ids) < 2:
            continue
        for idx, left_id in enumerate(hit_ids):
            for right_id in hit_ids[idx + 1 :]:
                graph[left_id][right_id] += 1
                graph[right_id][left_id] += 1
    return graph


def choose_canonical(group_members: Sequence[EntityRecord]) -> str:
    def score(record: EntityRecord) -> Tuple[int, int, int, str]:
        has_spaces = 0 if " " in record.main_entity else 1
        return (record.mention_count, len(record.main_entity), has_spaces, record.main_entity)

    return max(group_members, key=score).main_entity


def choose_entity_type(group_members: Sequence[EntityRecord]) -> str:
    counter: Counter = Counter()
    for record in group_members:
        counter[record.entity_type] += max(record.mention_count, 1)
    return counter.most_common(1)[0][0]


def canonical_source_file(group_members: Sequence[EntityRecord]) -> str:
    sources = deduplicate_preserve_order(record.source_file for record in group_members if record.source_file)
    return sources[0] if sources else ""


def summarize_group(
    group_members: Sequence[EntityRecord],
    merge_pairs: Dict[Tuple[int, int], Tuple[float, List[str]]],
) -> Tuple[float, List[str]]:
    if len(group_members) == 1:
        record = group_members[0]
        basis = [
            f"保留单实体={record.main_entity}",
            f"章节={('|'.join(record.sections) if record.sections else '全文')}",
        ]
        return 1.0, basis

    pair_scores: List[float] = []
    evidence: List[str] = []
    member_ids = [member.row_id for member in group_members]
    for idx, left_id in enumerate(member_ids):
        for right_id in member_ids[idx + 1 :]:
            pair = (min(left_id, right_id), max(left_id, right_id))
            if pair in merge_pairs:
                score, details = merge_pairs[pair]
                pair_scores.append(score)
                evidence.extend(details[:4])

    confidence = float(sum(pair_scores) / len(pair_scores)) if pair_scores else 0.80
    evidence = deduplicate_preserve_order(evidence)
    if not evidence:
        evidence = ["候选重排后合并"]
    return confidence, evidence[:8]


def cluster_records(records: Sequence[EntityRecord], text: str) -> Tuple[List[dict], int]:
    if not records:
        return [], 0

    collect_record_context(records, text)
    graph = build_cooccurrence_graph(records, text)

    name_docs = [" ".join(sorted(record.alias_forms)) for record in records]
    context_docs = [
        " ".join([record.main_entity, record.entity_type, *record.sections, *record.contexts])
        for record in records
    ]
    name_similarity = compute_similarity_matrix(name_docs)
    context_similarity = compute_similarity_matrix(context_docs)
    build_candidate_pool(records, name_similarity, context_similarity)

    reranker = LocalCrossEncoderReranker(records, name_similarity, context_similarity)
    union_find = UnionFind(len(records))
    merge_pairs: Dict[Tuple[int, int], Tuple[float, List[str]]] = {}

    candidate_pairs: List[Tuple[float, int, int, List[str]]] = []
    for record in records:
        for candidate_id in record.recall_candidates:
            if candidate_id <= record.row_id:
                continue
            cross_score, cross_evidence = reranker.score(record.row_id, candidate_id)
            type_adjustment, type_evidence = type_section_adjustment(record, records[candidate_id])
            graph_score, graph_evidence = graph_consistency_score(record.row_id, candidate_id, graph, records)
            final_score = bounded_score(cross_score + type_adjustment + 0.15 * graph_score)
            evidence = cross_evidence + type_evidence + graph_evidence + [f"最终得分={final_score:.2f}"]
            candidate_pairs.append((final_score, record.row_id, candidate_id, evidence))

    candidate_pairs.sort(reverse=True, key=lambda item: item[0])
    merged_count = 0
    for final_score, left_id, right_id, evidence in candidate_pairs:
        left = records[left_id]
        right = records[right_id]
        cross_score, _ = reranker.score(left_id, right_id)
        type_adjustment, _ = type_section_adjustment(left, right)
        graph_score, _ = graph_consistency_score(left_id, right_id, graph, records)
        if should_merge(left, right, cross_score, type_adjustment, graph_score):
            if union_find.find(left_id) != union_find.find(right_id):
                merged_count += 1
            union_find.union(left_id, right_id)
            merge_pairs[(min(left_id, right_id), max(left_id, right_id))] = (final_score, evidence)

    groups: Dict[int, List[EntityRecord]] = defaultdict(list)
    for record in records:
        groups[union_find.find(record.row_id)].append(record)

    output_rows: List[dict] = []
    for group_members in groups.values():
        canonical = choose_canonical(group_members)
        entity_type = choose_entity_type(group_members)
        merged_entities = deduplicate_preserve_order([item.main_entity for item in group_members])
        merged_mentions = deduplicate_preserve_order(
            [alias for item in group_members for alias in [item.main_entity, *item.mentions]]
        )
        merged_sections = deduplicate_preserve_order([section for item in group_members for section in item.sections])

        confidence, evidence_items = summarize_group(group_members, merge_pairs)
        method_name = (
            "实体标准化 + 候选召回 + LocalCrossEncoder上下文重排 + 类型/章节约束 + 共现图一致性"
            if len(group_members) > 1
            else "实体标准化 + 候选召回（未发现需合并候选）"
        )
        output_rows.append(
            {
                "main_entity": canonical,
                "entity_type": entity_type,
                "mention_count": int(sum(item.mention_count for item in group_members)),
                "mentions": " | ".join(merged_mentions),
                "source_file": canonical_source_file(group_members),
                "normalized_entity": normalize_name(canonical),
                "aliases": " | ".join(merged_mentions),
                "section_titles": " | ".join(merged_sections),
                "merged_entities": " | ".join(merged_entities),
                "disambiguation_confidence": round(confidence, 4),
                "disambiguation_method": method_name,
                "disambiguation_basis": "；".join(evidence_items),
            }
        )

    output_rows.sort(key=lambda item: (-int(item["mention_count"]), item["main_entity"]))
    return output_rows, merged_count
