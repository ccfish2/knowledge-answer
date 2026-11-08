"""Identity-based resolution of source-grounded numeric observations.

A crawl date says when a document was seen, not when its contents became true.
Only explicit effective dates or a change to the same source establish ordering
between different values. Incomparable competing sources remain unresolved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from itertools import combinations
from typing import Iterable

from ingestion import StructuredFact


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _qualifier_value(value: object) -> str:
    text = str(value).strip().lower().replace(",", "")
    text = re.sub(r"\s*[-–—]\s*", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text


@dataclass(frozen=True, order=True)
class FactIdentity:
    entity: str
    attribute: str
    unit: str
    qualifiers: tuple[tuple[str, str], ...]


class ConflictType(str, Enum):
    TEMPORAL = "temporal"
    TIER = "tier"
    ATTRIBUTE = "attribute"
    ENTITY = "entity"
    SOURCE = "source"
    GENUINE = "genuine"


def fact_identity(fact: StructuredFact) -> FactIdentity:
    """Identity includes every extracted qualifier; provenance is not identity."""
    return FactIdentity(
        _normalize(fact.entity), _normalize(fact.attribute), _normalize(fact.unit or ""),
        tuple(sorted((_normalize(str(k)), _qualifier_value(v))
                     for k, v in fact.qualifiers.items() if k != "page_title")),
    )


def normalized_value(fact: StructuredFact) -> str:
    raw = str(fact.value).strip().replace(",", "")
    match = re.fullmatch(r"\$?\s*([+-]?\d+(?:\.\d+)?)\s*(%?)", raw)
    if match:
        try:
            return f"{Decimal(match.group(1)).normalize()}{match.group(2)}"
        except InvalidOperation:
            pass
    # Normalize punctuation without discarding the distinction between a
    # range and a list (or losing one of its endpoints).
    return re.sub(r"\s+", " ", raw.lower().replace("–", "-").replace("—", "-")).strip()


def _instant(value: str, *, end_of_day: bool = False) -> datetime:
    if len(value) == 10:
        day = datetime.fromisoformat(value).date()
        return datetime.combine(day, time.max if end_of_day else time.min, tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


@dataclass(frozen=True)
class EvidenceRelationship:
    kind: str  # corroborates, supersedes, or conflicts
    left: StructuredFact
    right: StructuredFact
    reason: str
    conflict_type: ConflictType | None = None


@dataclass(frozen=True)
class FactResolution:
    identity: FactIdentity
    observations: tuple[StructuredFact, ...]
    active: tuple[StructuredFact, ...]
    relationships: tuple[EvidenceRelationship, ...]
    winner: StructuredFact | None
    reason: str


def classify_conflict(left: StructuredFact, right: StructuredFact) -> ConflictType:
    first, second = fact_identity(left), fact_identity(right)
    if first.entity != second.entity:
        return ConflictType.ENTITY
    if first.attribute != second.attribute or first.unit != second.unit:
        return ConflictType.ATTRIBUTE
    if first.qualifiers != second.qualifiers:
        return ConflictType.TIER
    if left.source.url == right.source.url and left.observed_on != right.observed_on:
        return ConflictType.TEMPORAL
    if left.effective_from and right.effective_from and left.effective_from != right.effective_from:
        older, newer = sorted((left, right), key=lambda fact: fact.effective_from or "")
        if not older.effective_until or older.effective_until < newer.effective_from:
            return ConflictType.TEMPORAL
    if left.source.url != right.source.url and (not left.effective_from or not right.effective_from):
        return ConflictType.SOURCE
    return ConflictType.GENUINE


def _supersedes(new: StructuredFact, old: StructuredFact, *, validity_only: bool = False) -> str | None:
    if validity_only:
        if (new.effective_from and old.effective_from and new.effective_from > old.effective_from
                and (not old.effective_until or old.effective_until < new.effective_from)):
            return "later explicit effective date"
        return None
    if classify_conflict(new, old) != ConflictType.TEMPORAL:
        return None
    # Explicit validity is stronger evidence than page observation time.
    if new.effective_from and old.effective_from:
        if new.effective_from > old.effective_from:
            return "later explicit effective date"
        if new.effective_from < old.effective_from:
            return None
    # A changed version of the same URL is an observed update, not proof that
    # a different, repeatedly crawled document was also updated.
    if (new.source.url == old.source.url and new.observed_on and old.observed_on
            and _instant(new.observed_on) > _instant(old.observed_on)):
        if old.effective_from and not new.effective_from:
            return None
        return "later version of the same source"
    return None


def resolve_facts(facts: Iterable[StructuredFact], as_of: str | None = None, *,
                  observation_only: bool = False, validity_only: bool = False) -> list[FactResolution]:
    """Group same-identity observations and conservatively resolve each group."""
    groups: dict[FactIdentity, list[StructuredFact]] = {}
    cutoff = _instant(str(as_of), end_of_day=True) if as_of is not None else None
    for fact in facts:
        if cutoff and fact.observed_on and _instant(fact.observed_on) > cutoff:
            continue
        groups.setdefault(fact_identity(fact), []).append(fact)
    results = []
    for identity, items in groups.items():
        relationships = []
        for left, right in combinations(items, 2):
            if normalized_value(left) == normalized_value(right):
                # Identical captures are not an independent corroborating source.
                if left.source.url != right.source.url:
                    relationships.append(EvidenceRelationship("corroborates", left, right, "same value"))
                continue
            forward = _supersedes(right, left, validity_only=validity_only)
            backward = _supersedes(left, right, validity_only=validity_only)
            conflict_type = classify_conflict(left, right)
            if forward:
                relationships.append(EvidenceRelationship("supersedes", right, left, forward, conflict_type))
            elif backward:
                relationships.append(EvidenceRelationship("supersedes", left, right, backward, conflict_type))
            else:
                relationships.append(EvidenceRelationship("conflicts", left, right, "no evidence establishes precedence", conflict_type))
        valid = [item for item in items if observation_only or not cutoff or
             ((not item.effective_from or _instant(item.effective_from) <= cutoff) and
              (not item.effective_until or _instant(item.effective_until, end_of_day=True) >= cutoff))]
        active = [item for item in valid if not any(
            normalized_value(other) != normalized_value(item)
            and (not observation_only or other.source.url == item.source.url)
            and _supersedes(other, item, validity_only=validity_only)
            for other in valid)]
        values = {normalized_value(item) for item in active}
        winner = None
        if len(values) == 1:
            winner = max(active, key=lambda item: (item.effective_from or "", item.observed_on or "",
                                                   item.source.url, item.source.section_id))
            reason = "all current, non-superseded observations agree"
            if len({normalized_value(item) for item in valid}) > 1:
                reason = "older evidence superseded by a later effective date or source revision"
        elif not active:
            reason = "no observation is valid at the requested date"
        else:
            reason = "conflicting current sources have no established precedence"
        results.append(FactResolution(identity, tuple(items), tuple(active), tuple(relationships), winner, reason))
    return results