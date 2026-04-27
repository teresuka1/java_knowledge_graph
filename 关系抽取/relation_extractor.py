from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from relation_patterns import RELATION_PATTERNS, RelationPattern


ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-./]+")
SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;]\s*")
HEADING_RE = re.compile(
    r"^\s*(?:"
    r"第?[一二三四五六七八九十百千万零两\d]+(?:章|节|部分)?"
    r"|[一二三四五六七八九十百千万零两]+"
    r"|[0-9]+(?:\.[0-9]+)*"
    r"|[A-Za-z]"
    r")"
    r"(?:[、.．)）]|(?:\s+))+\s*(.+?)\s*$"
)
GENERIC_ENTITY_SUFFIXES = (
    "方式",
    "方法",
    "特性",
    "过程",
    "关系",
    "定义",
    "分类",
    "概述",
    "简介",
)


@dataclass
class EntityNode:
    canonical: str
    entity_type: str
    aliases: List[str]
    mention_count: int


@dataclass
class Mention:
    canonical: str
    surface: str
    start: int
    end: int
    entity_type: str


@dataclass
class SectionSentence:
    section_title: str
    sentence: str


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def normalize_text(text: str) -> str:
    text = (text or "").replace("\u3000", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    pieces: List[str] = []
    for block in normalize_text(text).split("\n"):
        block = block.strip()
        if not block:
            continue
        parts = SENTENCE_SPLIT_RE.split(block)
        if len(parts) == 1:
            parts = [block]
        for part in parts:
            sentence = part.strip()
            if sentence:
                pieces.append(sentence)
    return pieces


def normalize_name(text: str) -> str:
    text = normalize_text(text)
    text = text.strip("`'\"“”‘’[]{}【】<>《》")
    text = re.sub(r"\s*([()/._:+\-])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_loose_key(text: str) -> str:
    text = normalize_name(text).lower()
    return re.sub(r"[\s\-_/.:：,，;；()（）\[\]【】]+", "", text)


def split_aliases(raw_value: object) -> List[str]:
    if pd.isna(raw_value):
        return []
    parts = [normalize_name(part) for part in str(raw_value).split("|")]
    return [p for p in parts if p]


def split_type_tokens(text: str) -> List[str]:
    normalized = normalize_name(text)
    ascii_tokens = re.findall(r"[A-Za-z]+", normalized)
    chinese_tokens = re.findall(r"[\u4e00-\u9fff]{1,6}", normalized)
    return ascii_tokens + chinese_tokens


def token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(split_type_tokens(left))
    right_tokens = set(split_type_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _entity_score(canonical: str, mention_count: int) -> Tuple[int, int]:
    return (max(mention_count, 1), len(canonical))


def load_entities(csv_path: Path) -> List[EntityNode]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        return []

    grouped: Dict[str, Dict[str, object]] = {}
    for _, row in df.iterrows():
        canonical = normalize_name(
            str(row.get("main_entity", row.get("entity", "")) or "")
        )
        if not canonical:
            continue
        entity_type = normalize_name(str(row.get("entity_type", "") or ""))
        mention_count = int(row.get("mention_count", 1) or 1)
        aliases = [canonical]
        aliases.extend(split_aliases(row.get("aliases")))
        aliases.extend(split_aliases(row.get("mentions")))
        aliases = dedup_preserve_order(normalize_name(a) for a in aliases if a and len(normalize_name(a)) >= 2)

        key = build_loose_key(canonical) or canonical
        if key not in grouped:
            grouped[key] = {
                "canonical": canonical,
                "mention_count": max(mention_count, 1),
                "aliases": list(aliases),
                "type_counter": Counter(),
            }
        bucket = grouped[key]
        if _entity_score(canonical, mention_count) > _entity_score(
            str(bucket["canonical"]), int(bucket["mention_count"])
        ):
            bucket["canonical"] = canonical
        bucket["mention_count"] = int(bucket["mention_count"]) + max(mention_count, 1)
        bucket["aliases"] = dedup_preserve_order([*bucket["aliases"], *aliases])
        if entity_type:
            bucket["type_counter"][entity_type] += max(mention_count, 1)

    entities: List[EntityNode] = []
    for bucket in grouped.values():
        type_counter: Counter = bucket["type_counter"]
        entity_type = type_counter.most_common(1)[0][0] if type_counter else ""
        entities.append(
            EntityNode(
                canonical=str(bucket["canonical"]),
                entity_type=entity_type,
                aliases=dedup_preserve_order(bucket["aliases"]),
                mention_count=int(bucket["mention_count"]),
            )
        )
    entities.sort(key=lambda e: (-e.mention_count, -len(e.canonical), e.canonical))
    return entities


def dedup_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _match_ascii_alias(sentence: str, alias: str) -> List[Tuple[int, int]]:
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(sentence)]


def _match_non_ascii_alias(sentence: str, alias: str) -> List[Tuple[int, int]]:
    hits: List[Tuple[int, int]] = []
    start = 0
    while True:
        idx = sentence.find(alias, start)
        if idx < 0:
            break
        hits.append((idx, idx + len(alias)))
        start = idx + 1
    return hits


def find_mentions(sentence: str, entities: Sequence[EntityNode]) -> List[Mention]:
    sentence_norm = normalize_text(sentence)
    mentions: List[Mention] = []
    alias_items: List[Tuple[str, str, str, int]] = []
    for entity in entities:
        for alias in entity.aliases:
            alias_items.append((alias, entity.canonical, entity.entity_type, entity.mention_count))
    alias_items.sort(key=lambda x: (-len(x[0]), x[0]))

    for alias, canonical, entity_type, _ in alias_items:
        if len(alias) < 2:
            continue
        if ASCII_TOKEN_RE.fullmatch(alias):
            spans = _match_ascii_alias(sentence_norm, alias)
        else:
            spans = _match_non_ascii_alias(sentence_norm, alias)
        for start, end in spans:
            mentions.append(
                Mention(
                    canonical=canonical,
                    surface=sentence_norm[start:end],
                    start=start,
                    end=end,
                    entity_type=entity_type,
                )
            )
    mentions.sort(key=lambda m: (m.start, -(m.end - m.start)))
    return _resolve_overlaps(mentions)


def _resolve_overlaps(mentions: Sequence[Mention]) -> List[Mention]:
    selected: List[Mention] = []
    occupied: List[Tuple[int, int]] = []
    for mention in mentions:
        conflict = False
        for s, e in occupied:
            if not (mention.end <= s or mention.start >= e):
                conflict = True
                break
        if conflict:
            continue
        selected.append(mention)
        occupied.append((mention.start, mention.end))
    return selected


def _pair_gap(left: Mention, right: Mention) -> int:
    return max(0, right.start - left.end)


def _contains_cue(text: str, cues: Sequence[str]) -> str | None:
    for cue in cues:
        if re.search(cue, text):
            return cue
    return None


def _is_generic_entity_name(name: str) -> bool:
    s = normalize_name(name)
    if not s:
        return True
    if len(s) <= 2:
        return False
    return any(s.endswith(sfx) for sfx in GENERIC_ENTITY_SUFFIXES)


def _is_noisy_enumeration_between(text: str) -> bool:
    s = normalize_text(text)
    if not s:
        return False
    if re.fullmatch(r"[、,，/和及与或以及 ]+", s):
        return True
    if re.fullmatch(r"[：:、,， ]*(是|属于|继承|实现|包括|包含)[、,， ]*", s):
        return True
    return False


def _relation_specific_guard(
    pattern: RelationPattern,
    left: Mention,
    right: Mention,
    between: str,
    head: str,
    tail: str,
) -> bool:
    if _is_noisy_enumeration_between(between):
        return False

    if pattern.relation in {"extends", "implemented_by"} and _is_generic_entity_name(head):
        return False
    if pattern.relation in {"extends", "implemented_by", "is_a"} and _is_generic_entity_name(left.surface):
        return False
    if pattern.relation == "alias_of" and abs(len(left.surface) - len(right.surface)) > 10:
        return False
    if pattern.relation == "causes" and len(between) < 1:
        return False
    if pattern.relation in {"implemented_by", "extends"}:
        overlap = token_overlap_score(left.entity_type, right.entity_type)
        if overlap <= 0 and left.entity_type and right.entity_type and len(normalize_text(between)) > 10:
            return False
    if head == tail:
        return False
    return True


def _pattern_match(
    pattern: RelationPattern,
    left: Mention,
    right: Mention,
    sentence: str,
) -> Tuple[str, str, float] | None:
    if right.start <= left.end:
        return None
    gap = _pair_gap(left, right)
    if gap > pattern.max_entity_gap:
        return None

    between = sentence[left.end : right.start]
    cue = _contains_cue(between, pattern.cues)
    if cue is None:
        return None

    if pattern.direction == "lr":
        head, tail = left.canonical, right.canonical
    elif pattern.direction == "rl":
        head, tail = right.canonical, left.canonical
    else:
        if left.canonical == right.canonical:
            return None
        head, tail = sorted([left.canonical, right.canonical], key=len)

    if not _relation_specific_guard(pattern, left, right, between, head, tail):
        return None

    confidence = score_confidence(pattern.base_score, gap, left, right)
    return head, tail, confidence


def score_confidence(base_score: float, gap: int, left: Mention, right: Mention) -> float:
    gap_penalty = min(0.20, gap * 0.006)
    type_bonus = 0.03 if left.entity_type and right.entity_type else 0.0
    diversity_bonus = 0.02 if left.entity_type != right.entity_type else 0.0
    score = base_score - gap_penalty + type_bonus + diversity_bonus
    return max(0.0, min(0.99, score))


def _extract_enumeration_relations(
    sentence: str,
    mentions: Sequence[Mention],
    source_file: str,
    section_title: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    colon_pos = max(sentence.find("："), sentence.find(":"))
    if colon_pos < 0:
        return rows

    left_mentions = [m for m in mentions if m.end <= colon_pos]
    right_mentions = [m for m in mentions if m.start > colon_pos]
    if not left_mentions or len(right_mentions) < 2:
        return rows

    head_mention = max(left_mentions, key=lambda m: m.end)
    if _is_generic_entity_name(head_mention.surface):
        return rows

    for tail_mention in right_mentions:
        if head_mention.canonical == tail_mention.canonical:
            continue
        between = sentence[head_mention.end : tail_mention.start]
        if len(normalize_text(between)) > 24:
            continue
        if not re.search(r"[、,，/和及与或]|包括|包含|分为", sentence[colon_pos + 1 :]):
            continue
        confidence = score_confidence(0.76, _pair_gap(head_mention, tail_mention), head_mention, tail_mention)
        rows.append(
            {
                "head": head_mention.canonical,
                "relation": "has_part",
                "tail": tail_mention.canonical,
                "head_type": head_mention.entity_type,
                "tail_type": tail_mention.entity_type,
                "evidence": sentence,
                "pattern_name": "枚举结构关系",
                "section_title": section_title,
                "source_file": source_file,
                "confidence": round(confidence, 4),
                "method": "rule_pattern_sentence_level",
            }
        )
    return rows


def extract_sentence_relations(
    sentence: str,
    entities: Sequence[EntityNode],
    source_file: str,
    section_title: str = "全文",
) -> List[Dict[str, object]]:
    mentions = find_mentions(sentence, entities)
    if len(mentions) < 2:
        return []

    rows: List[Dict[str, object]] = _extract_enumeration_relations(
        sentence=sentence,
        mentions=mentions,
        source_file=source_file,
        section_title=section_title,
    )
    n = len(mentions)
    for i in range(n):
        for j in range(i + 1, n):
            left = mentions[i]
            right = mentions[j]
            if left.canonical == right.canonical:
                continue
            for pattern in RELATION_PATTERNS:
                hit = _pattern_match(pattern, left, right, sentence)
                if hit is None:
                    continue
                head, tail, confidence = hit
                if head == tail:
                    continue
                rows.append(
                    {
                        "head": head,
                        "relation": pattern.relation,
                        "tail": tail,
                        "head_type": left.entity_type if head == left.canonical else right.entity_type,
                        "tail_type": right.entity_type if tail == right.canonical else left.entity_type,
                        "evidence": sentence,
                        "pattern_name": pattern.name,
                        "section_title": section_title,
                        "source_file": source_file,
                        "confidence": round(confidence, 4),
                        "method": "rule_pattern_sentence_level",
                    }
                )
    return rows


def deduplicate_relations(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    best: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for row in rows:
        key = (
            str(row["head"]),
            str(row["relation"]),
            str(row["tail"]),
            str(row["source_file"]),
        )
        prev = best.get(key)
        if prev is None:
            best[key] = dict(row)
            continue
        if float(row.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
            best[key] = dict(row)
    output = list(best.values())
    output.sort(key=lambda r: (-float(r.get("confidence", 0.0)), str(r.get("head", "")), str(r.get("tail", ""))))
    return output


def extract_heading(line: str) -> str | None:
    m = HEADING_RE.match(line.strip())
    if not m:
        return None
    heading = normalize_name(m.group(1))
    return heading[:80] if heading else None


def parse_sections(text: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    current_title = "全文"
    buffer: List[str] = []

    def flush() -> None:
        content = "\n".join([line for line in buffer if line]).strip()
        if content:
            sections.append((current_title, content))
        buffer.clear()

    for raw_line in normalize_text(text).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        heading = extract_heading(line)
        if heading:
            flush()
            current_title = heading
        else:
            buffer.append(line)
    flush()
    if not sections:
        fallback = normalize_text(text)
        if fallback:
            sections.append(("全文", fallback))
    return sections


def iter_section_sentences(text: str) -> List[SectionSentence]:
    items: List[SectionSentence] = []
    for section_title, section_text in parse_sections(text):
        for sentence in split_sentences(section_text):
            items.append(SectionSentence(section_title=section_title, sentence=sentence))
    return items


def extract_document_relations(text: str, entities: Sequence[EntityNode], source_file: str) -> List[Dict[str, object]]:
    all_rows: List[Dict[str, object]] = []
    for item in iter_section_sentences(text):
        sentence_rows = extract_sentence_relations(
            item.sentence,
            entities,
            source_file=source_file,
            section_title=item.section_title,
        )
        all_rows.extend(sentence_rows)
    return deduplicate_relations(all_rows)
