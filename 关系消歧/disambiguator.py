from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

from models import CanonicalRecord, CandidateScore, DisambiguationResult, RelationRecord
from taxonomy import RelationTaxonomy
from text_utils import (
    bounded,
    choose_display,
    compact_join,
    dedupe_preserve_order,
    entity_key,
    normalize_entity,
    normalize_text,
    shorten,
)


class RelationDisambiguator:
    def __init__(
        self,
        taxonomy: RelationTaxonomy,
        tie_margin: float = 0.015,
        max_evidence_items: int = 4,
    ) -> None:
        self.taxonomy = taxonomy
        self.tie_margin = tie_margin
        self.max_evidence_items = max_evidence_items

    def disambiguate(self, records: Sequence[RelationRecord]) -> DisambiguationResult:
        canonical_records = [self._canonicalize(record) for record in records]
        pair_groups: Dict[Tuple[str, str], List[CanonicalRecord]] = defaultdict(list)
        for record in canonical_records:
            pair_groups[(record.head_key, record.tail_key)].append(record)

        output_rows: List[Dict[str, str]] = []
        decision_rows: List[Dict[str, str]] = []
        examples: List[Dict[str, str]] = []
        summary = {
            "input_rows": len(records),
            "output_rows": 0,
            "pairs": len(pair_groups),
            "relation_conflict_pairs": 0,
            "relation_conflict_rows": 0,
            "inverse_relation_rows": 0,
            "symmetric_reordered_rows": 0,
            "taxonomy_normalized_rows": 0,
            "merged_duplicate_rows": 0,
        }

        for pair_records in pair_groups.values():
            relation_groups: Dict[str, List[CanonicalRecord]] = defaultdict(list)
            for record in pair_records:
                relation_groups[record.relation].append(record)
                if "inverse_relation_direction" in record.actions:
                    summary["inverse_relation_rows"] += 1
                if "symmetric_relation_order" in record.actions:
                    summary["symmetric_reordered_rows"] += 1
                if "relation_label_normalized" in record.actions:
                    summary["taxonomy_normalized_rows"] += 1

            candidate_scores = [
                self._score_candidate(relation, group) for relation, group in relation_groups.items()
            ]
            candidate_scores.sort(
                key=lambda item: (
                    item.score,
                    item.relation_priority,
                    item.max_confidence,
                    item.support,
                    item.relation,
                ),
                reverse=True,
            )
            winner = candidate_scores[0]
            runner_up = candidate_scores[1] if len(candidate_scores) > 1 else None
            has_conflict = len(candidate_scores) > 1
            if has_conflict:
                summary["relation_conflict_pairs"] += 1
                summary["relation_conflict_rows"] += sum(
                    len(group) for relation, group in relation_groups.items() if relation != winner.relation
                )

            chosen_records = relation_groups[winner.relation]
            all_relations = [score.relation for score in candidate_scores]
            basis = self._build_basis(candidate_scores, winner, runner_up)
            output_rows.append(
                self._build_output_row(
                    chosen_records=chosen_records,
                    all_pair_records=pair_records,
                    candidate_scores=candidate_scores,
                    winner=winner,
                    basis=basis,
                    has_conflict=has_conflict,
                )
            )

            if len(chosen_records) > 1:
                summary["merged_duplicate_rows"] += len(chosen_records) - 1

            for canonical_record in pair_records:
                kept = canonical_record.relation == winner.relation
                decision = "kept"
                if not kept:
                    decision = "relation_conflict_resolved"
                elif canonical_record.actions:
                    decision = "canonicalized"
                decision_rows.append(
                    self._build_decision_row(
                        canonical_record=canonical_record,
                        winner=winner,
                        runner_up=runner_up,
                        decision=decision,
                        candidate_relations=all_relations,
                        basis=basis,
                    )
                )

            if has_conflict and len(examples) < 12:
                examples.append(
                    {
                        "head": output_rows[-1]["head"],
                        "tail": output_rows[-1]["tail"],
                        "chosen_relation": winner.relation,
                        "candidate_relations": compact_join(all_relations, limit=8),
                        "basis": basis,
                        "evidence": output_rows[-1]["evidence"],
                    }
                )

        output_rows.sort(key=lambda item: (item["source_files"], item["head"], item["relation"], item["tail"]))
        decision_rows.sort(key=lambda item: (item["input_file"], int(item["input_row_id"])))
        summary["output_rows"] = len(output_rows)
        return DisambiguationResult(output_rows, decision_rows, summary, examples)

    def _canonicalize(self, record: RelationRecord) -> CanonicalRecord:
        relation_spec = self.taxonomy.mapping_for(record.relation)
        canonical_relation = relation_spec.canonical
        head = normalize_entity(record.head)
        tail = normalize_entity(record.tail)
        head_type = normalize_text(record.head_type)
        tail_type = normalize_text(record.tail_type)
        actions: List[str] = []

        if canonical_relation != record.relation:
            actions.append("relation_label_normalized")

        if relation_spec.swap:
            head, tail = tail, head
            head_type, tail_type = tail_type, head_type
            actions.append("inverse_relation_direction")

        profile = self.taxonomy.profile_for(canonical_relation)
        head_key = entity_key(head)
        tail_key = entity_key(tail)
        if profile.symmetric and tail_key < head_key:
            head, tail = tail, head
            head_type, tail_type = tail_type, head_type
            head_key, tail_key = tail_key, head_key
            actions.append("symmetric_relation_order")

        return CanonicalRecord(
            record=record,
            head=head,
            relation=canonical_relation,
            tail=tail,
            head_key=head_key,
            tail_key=tail_key,
            head_type=head_type,
            tail_type=tail_type,
            actions=dedupe_preserve_order(actions),
        )

    def _score_candidate(self, relation: str, records: Sequence[CanonicalRecord]) -> CandidateScore:
        confidences = [bounded(item.record.confidence) for item in records]
        avg_confidence = mean(confidences) if confidences else 0.0
        max_confidence = max(confidences) if confidences else 0.0
        confidence_score = 0.65 * max_confidence + 0.35 * avg_confidence
        cue_score, cue_hits = self._cue_score(relation, records)
        support_score = min(1.0, math.log1p(len(records)) / math.log(5))
        profile = self.taxonomy.profile_for(relation)
        score = bounded(
            0.62 * confidence_score
            + 0.23 * cue_score
            + 0.10 * bounded(profile.priority)
            + 0.05 * support_score
        )
        return CandidateScore(
            relation=relation,
            score=score,
            avg_confidence=avg_confidence,
            max_confidence=max_confidence,
            cue_score=cue_score,
            support_score=support_score,
            relation_priority=bounded(profile.priority),
            support=len(records),
            cue_hits=cue_hits,
        )

    def _cue_score(self, relation: str, records: Sequence[CanonicalRecord]) -> tuple[float, List[str]]:
        profile = self.taxonomy.profile_for(relation)
        if not profile.cues and not profile.negative_cues:
            return 0.50, []

        per_record_scores: List[float] = []
        all_hits: List[str] = []
        for item in records:
            text = " ".join(
                [
                    item.record.evidence,
                    item.record.pattern_name,
                    item.record.method,
                    item.record.section_title,
                ]
            )
            positive_hits = self._regex_hits(profile.cues, text)
            negative_hits = self._regex_hits(profile.negative_cues, text)
            all_hits.extend(positive_hits)
            all_hits.extend(f"!{hit}" for hit in negative_hits)
            score = 0.35
            if positive_hits:
                score = 0.72 + min(0.28, 0.07 * len(positive_hits))
            if negative_hits:
                score -= min(0.35, 0.12 * len(negative_hits))
            per_record_scores.append(bounded(score))

        if not per_record_scores:
            return 0.35, []
        return mean(per_record_scores), dedupe_preserve_order(all_hits)[:8]

    @staticmethod
    def _regex_hits(patterns: Iterable[str], text: str) -> List[str]:
        hits: List[str] = []
        for pattern in patterns:
            try:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    hits.append(pattern)
            except re.error:
                if pattern in text:
                    hits.append(pattern)
        return hits

    def _build_output_row(
        self,
        chosen_records: Sequence[CanonicalRecord],
        all_pair_records: Sequence[CanonicalRecord],
        candidate_scores: Sequence[CandidateScore],
        winner: CandidateScore,
        basis: str,
        has_conflict: bool,
    ) -> Dict[str, str]:
        confidences = [item.record.confidence for item in chosen_records]
        head = choose_display([item.head for item in chosen_records], confidences)
        tail = choose_display([item.tail for item in chosen_records], confidences)
        head_type = choose_display([item.head_type for item in chosen_records], confidences)
        tail_type = choose_display([item.tail_type for item in chosen_records], confidences)
        relation_names = [item.record.relation for item in all_pair_records]
        conflict_relations = [
            score.relation for score in candidate_scores if score.relation != winner.relation
        ]
        methods = ["relation_taxonomy_normalization", "direction_normalization"]
        if has_conflict:
            methods.append("evidence_weighted_relation_voting")
        else:
            methods.append("duplicate_merge")
        return {
            "head": head,
            "relation": winner.relation,
            "tail": tail,
            "head_type": head_type,
            "tail_type": tail_type,
            "confidence": f"{winner.avg_confidence:.4f}",
            "evidence": compact_join(
                [item.record.evidence for item in chosen_records if item.record.evidence],
                limit=self.max_evidence_items,
            ),
            "pattern_names": compact_join([item.record.pattern_name for item in chosen_records], limit=6),
            "section_titles": compact_join([item.record.section_title for item in chosen_records], limit=6),
            "source_files": compact_join([item.record.source_file or item.record.input_file for item in chosen_records], limit=8),
            "record_count": str(len(chosen_records)),
            "source_row_ids": compact_join([item.record.row_id for item in all_pair_records], limit=20),
            "source_relations": compact_join(relation_names, limit=12),
            "conflicting_relations": compact_join(conflict_relations, limit=12),
            "disambiguation_score": f"{winner.score:.4f}",
            "disambiguation_method": " + ".join(dedupe_preserve_order(methods)),
            "disambiguation_basis": basis,
        }

    def _build_decision_row(
        self,
        canonical_record: CanonicalRecord,
        winner: CandidateScore,
        runner_up: CandidateScore | None,
        decision: str,
        candidate_relations: Sequence[str],
        basis: str,
    ) -> Dict[str, str]:
        record = canonical_record.record
        return {
            "input_file": record.input_file,
            "input_row_id": str(record.row_id),
            "original_head": record.head,
            "original_relation": record.relation,
            "original_tail": record.tail,
            "canonical_head": canonical_record.head,
            "canonical_relation": canonical_record.relation,
            "canonical_tail": canonical_record.tail,
            "kept_relation": winner.relation,
            "decision": decision,
            "canonicalization_actions": compact_join(canonical_record.actions, limit=8, sep=","),
            "candidate_relations": compact_join(candidate_relations, limit=12),
            "winner_score": f"{winner.score:.4f}",
            "runner_up_score": f"{runner_up.score:.4f}" if runner_up else "",
            "score_margin": f"{(winner.score - runner_up.score):.4f}" if runner_up else "",
            "basis": basis,
            "evidence": shorten(record.evidence, 180),
        }

    def _build_basis(
        self,
        candidate_scores: Sequence[CandidateScore],
        winner: CandidateScore,
        runner_up: CandidateScore | None,
    ) -> str:
        pieces: List[str] = []
        for score in candidate_scores[:5]:
            hit_text = compact_join(score.cue_hits, limit=4, sep=",")
            if hit_text:
                hit_text = f", cues={hit_text}"
            pieces.append(
                f"{score.relation}: score={score.score:.3f}, conf={score.avg_confidence:.3f}, "
                f"support={score.support}{hit_text}"
            )
        if runner_up and winner.score - runner_up.score <= self.tie_margin:
            pieces.append(f"low_margin={winner.score - runner_up.score:.3f}")
        return "；".join(pieces)
