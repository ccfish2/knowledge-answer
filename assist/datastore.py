

"""Versioned structured-fact index and lightweight deterministic query pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, TYPE_CHECKING

from ingestion import StructuredFact, extract_facts
from resolver import fact_identity, normalized_value, resolve_facts
from timeline import build_timeline

if TYPE_CHECKING:
    from contract import Snapshot


WORD = re.compile(r"[a-z0-9]+", re.I)
STOP = set("a an the i my me we our you your to of for with on in at by is are be do does can how "
           "what when where which who it its this that and or if from as about get".split())
SYNONYMS = {
    "rates": "rate", "interest": "rate", "dividends": "dividend", "dividend": "rate",
    "charges": "fee", "costs": "fee", "cost": "fee", "prices": "price",
    "months": "month", "years": "year", "branches": "branch", "locations": "branch",
    "cards": "card", "accounts": "account", "vehicles": "vehicle", "hours": "hour",
}
ATTRIBUTE_TERMS = {
    "apy": {"apy"}, "apr": {"apr"}, "rate": {"rate", "apy", "apr", "dividend"},
    "fee": {"fee", "cost"}, "hours": {"hour", "hours", "schedule"},
    "address": {"address", "location"}, "phone": {"phone", "telephone", "contact"},
    "count": {"count", "many", "number"}, "email": {"email"},
}


def _tokens(text: str) -> set[str]:
    tokens = set()
    for word in WORD.findall(text.lower()):
        if word in STOP:
            continue
        word = SYNONYMS.get(word, word)
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        tokens.add(word)
    return tokens


def _as_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=value.tzinfo or timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, 23, 59, 59, tzinfo=timezone.utc)
    raw = str(value).strip().replace("Z", "+00:00")
    try:
        result = datetime.fromisoformat(raw if "T" in raw else raw + "T23:59:59+00:00")
    except ValueError:
        return None
    return result.replace(tzinfo=result.tzinfo or timezone.utc)


def _date_rank(value: str | None) -> float:
    parsed = _as_datetime(value)
    return parsed.timestamp() if parsed else 0.0


@dataclass(frozen=True)
class QuerySpec:
    question: str
    as_of: Any = None
    tokens: frozenset[str] = frozenset()
    requested_attributes: frozenset[str] = frozenset()
    numbers: frozenset[str] = frozenset()
    wants_list: bool = False
    wants_history: bool = False


@dataclass(frozen=True)
class QueryPlan:
    spec: QuerySpec
    search_terms: frozenset[str]
    limit: int = 12


@dataclass(frozen=True)
class EvidenceMatch:
    fact: StructuredFact
    score: float


@dataclass
class ResolutionResult:
    evidence: list[EvidenceMatch] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    conflict_note: str | None = None


def _reserve_evidence_kinds(candidates: list[EvidenceMatch], limit: int) -> list[EvidenceMatch]:
    if limit <= 0:
        return []
    kinds = []
    for candidate in candidates:
        fact = candidate.fact
        table = bool(" | " in fact.text)
        structured = fact.attribute != "claim"
        kinds.append((table, structured))
    available = set(kinds)
    quota = max(1, limit // max(1, len(available)))
    selected = set()
    counts = {}
    for position, kind in enumerate(kinds):
        if counts.get(kind, 0) < quota and len(selected) < limit:
            selected.add(position)
            counts[kind] = counts.get(kind, 0) + 1
    for position in range(len(candidates)):
        if len(selected) >= limit:
            break
        selected.add(position)
    return [candidates[position] for position in sorted(selected)]


def analyze_query(question: str, as_of: Any = None) -> QuerySpec:
    """Convert natural-language request into searchable intent and qualifiers."""
    tokens = _tokens(question)
    attrs = {attribute for attribute, words in ATTRIBUTE_TERMS.items() if tokens & words}
    nums = frozenset(re.findall(r"\d+(?:\.\d+)?%?", question.replace(",", "")))
    list_request = bool(re.search(r"\b(all|list|each|every|what are|options|rates)\b", question, re.I))
    return QuerySpec(question, as_of, frozenset(tokens), frozenset(attrs), nums,
                     list_request,
                     bool(tokens & {"was", "were", "previously", "historical", "history"}) or as_of is not None)


def plan_query(spec: QuerySpec) -> QueryPlan:
    """Build a simple lexical plan; it intentionally needs no external service."""
    terms = set(spec.tokens)
    # Attribute words add recall when the source uses APY/APR terminology.
    for attribute in spec.requested_attributes:
        terms.update(ATTRIBUTE_TERMS.get(attribute, ()))
    return QueryPlan(spec, frozenset(terms), 16 if spec.wants_list else 8)


class StructuredFactIndex:
    """Stores all page versions and exposes only the as-of-visible page version."""

    def __init__(self) -> None:
        self.page_versions: dict[str, list[tuple[datetime, str, list[StructuredFact]]]] = {}
        self.facts: list[StructuredFact] = []
        self.structured_entities: dict[str, list[StructuredFact]] = {}
        self.feature_entities: dict[str, list[StructuredFact]] = {}
        self.count_facts: list[StructuredFact] = []
        self.list_facts: list[StructuredFact] = []
        self.entity_inventory: dict[str, list[StructuredFact]] = {}
        self.list_facts_by_snapshot: dict[tuple[str, str], list[StructuredFact]] = {}
        self.inventory_by_snapshot: dict[tuple[str, str], list[StructuredFact]] = {}
        self.model_year_facts: list[StructuredFact] = []
        self.latest_observed_on: str | None = None

    def add_snapshot(self, snapshot: Snapshot) -> None:
        observed = _as_datetime(snapshot.taken_on)
        if observed is not None and len(snapshot.taken_on) == 10:
            observed = observed.replace(hour=0, minute=0, second=0, microsecond=0)
        for page in snapshot.pages:
            if page.url and page.markdown.strip():
                self.add_page(url=page.url, title=page.title, markdown=page.markdown,
                              snapshot_id=snapshot.snapshot_id, observed_on=observed,
                              fetched_at=page.fetched_at)

    def timeline(self, *, entity: str | None = None, attribute: str | None = None,
                 as_of: str | None = None):
        facts = (fact for fact in self.facts if fact.attribute != "claim"
                 and (entity is None or fact.entity.strip().casefold() == entity.strip().casefold())
                 and (attribute is None or fact.attribute == attribute))
        return build_timeline(facts, as_of)

    def add_page(self, *, url: str, title: str, markdown: str, snapshot_id: str,
                 observed_on: Any, file: str | None = None, fetched_at: str | None = None) -> None:
        observed = _as_datetime(observed_on)
        if observed is None:
            return
        observed_date = observed.date().isoformat()
        parsed = extract_facts(markdown, snapshot_id=snapshot_id, observed_on=observed.isoformat(),
                               url=url, file=file, fetched_at=fetched_at)
        if title:
            # Title is retrieval context, not fabricated page evidence.
            parsed = [StructuredFact(f.entity, f.attribute, f.value, f.unit,
                                     {**f.qualifiers, "page_title": title}, f.effective_from,
                                     f.effective_until, f.observed_on, f.source, f.text)
                      for f in parsed]
        versions = self.page_versions.setdefault(url, [])
        versions.append((observed, snapshot_id, parsed))
        versions.sort(key=lambda item: (item[0], item[1]))
        self.facts.extend(parsed)
        for fact in parsed:
            if fact.attribute == "entity" and fact.qualifiers.get("entity_class"):
                self.entity_inventory.setdefault(fact.qualifiers["entity_class"], []).append(fact)
                self.inventory_by_snapshot.setdefault((fact.source.snapshot_id, fact.qualifiers["entity_class"]), []).append(fact)
            if fact.attribute == "feature":
                self.feature_entities.setdefault(fact.entity.lower().strip(), []).append(fact)
            if fact.attribute == "count" and fact.unit == "count":
                self.count_facts.append(fact)
            if fact.qualifiers.get("entity_class") in {"branch", "location"} and fact.attribute != "claim":
                self.list_facts.append(fact)
                self.list_facts_by_snapshot.setdefault((fact.source.snapshot_id, fact.qualifiers["entity_class"]), []).append(fact)
            if fact.attribute in {"apr", "apy"} and re.search(r"\bmodel\s+years?\b", fact.entity, re.I):
                self.model_year_facts.append(fact)
            if fact.attribute != "claim" and fact.unit is not None:
                self.structured_entities.setdefault(fact.entity.lower().strip(), []).append(fact)
        if self.latest_observed_on is None or observed_date > self.latest_observed_on:
            self.latest_observed_on = observed_date

    def visible_facts(self, as_of: Any = None) -> list[StructuredFact]:
        cutoff = _as_datetime(as_of)
        visible: list[StructuredFact] = []
        for versions in self.page_versions.values():
            eligible = [entry for entry in versions if cutoff is None or entry[0] <= cutoff]
            if not eligible:
                continue
            _, _, facts = eligible[-1]
            visible.extend(facts)
        reference = cutoff.date().isoformat() if cutoff else self.latest_observed_on
        if reference:
            visible = [fact for fact in visible
                       if (not fact.effective_from or fact.effective_from <= reference)
                       and (not fact.effective_until or fact.effective_until >= reference)]
        return visible


def retrieve_evidence(plan: QueryPlan, index: StructuredFactIndex) -> list[EvidenceMatch]:
    """Rank visible facts by token overlap with provenance and value context."""
    query = set(plan.search_terms)
    if not query:
        return []
    out: list[EvidenceMatch] = []
    for fact in index.visible_facts(plan.spec.as_of):
        evidence = _tokens(fact.search_text)
        overlap = query & evidence
        if not overlap:
            continue
        coverage = len(overlap) / max(1, len(query))
        precision = len(overlap) / max(1, len(evidence))
        score = 0.78 * coverage + 0.12 * precision
        attr = fact.attribute.lower()
        if plan.spec.requested_attributes and any(a in attr for a in plan.spec.requested_attributes):
            score += 0.16
        if plan.spec.requested_attributes & {"apr", "apy"}:
            if "apr" in attr and "apr" in plan.spec.requested_attributes:
                score += 0.12
            if "apy" in attr and "apy" in plan.spec.requested_attributes:
                score += 0.12
        if fact.attribute == "claim":
            score -= 0.04
        if plan.spec.numbers:
            fact_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", fact.search_text.replace(",", "")))
            if plan.spec.numbers & fact_numbers:
                score += 0.14
        out.append(EvidenceMatch(fact, min(score, 1.0)))
    out.sort(key=lambda item: (-round(item.score, 7),
                               -_date_rank(item.fact.observed_on),
                               item.fact.source.url, item.fact.source.section_id))
    return _reserve_evidence_kinds(out, max(plan.limit * 8, 40))


def match_facts(candidates: list[EvidenceMatch], spec: QuerySpec) -> list[EvidenceMatch]:
    """Apply explicit metric and numeric qualifiers without collapsing tiers."""
    matched = []
    for candidate in candidates:
        fact = candidate.fact
        attr = fact.attribute.lower()
        if "apy" in spec.requested_attributes and "apr" in attr and "apy" not in attr:
            continue
        if "apr" in spec.requested_attributes and "apy" in attr and "apr" not in attr:
            continue
        if spec.numbers:
            fact_text = fact.search_text.replace(",", "")
            # Keep relevant qualifiers such as 60 months / $100,000, but do not
            # force the requested percentage value to be identical to all numbers.
            qualifier_numbers = {n for n in spec.numbers if n.rstrip("%") in fact_text}
            if qualifier_numbers:
                candidate = EvidenceMatch(fact, min(1.0, candidate.score + 0.05 * len(qualifier_numbers)))
        matched.append(candidate)
    return matched


def resolve_temporal(evidence: list[EvidenceMatch], spec: QuerySpec) -> list[EvidenceMatch]:
    """De-duplicate extraction duplicates, not independent source observations."""
    best: dict[tuple[Any, ...], EvidenceMatch] = {}
    for item in evidence:
        fact = item.fact
        key = (fact.source.url, fact.source.snapshot_id, fact.source.section_id,
               fact.entity.lower(), fact.attribute.lower(), str(fact.value).lower(),
               tuple(sorted((str(k).lower(), str(v).lower()) for k, v in fact.qualifiers.items()
                            if k != "page_title")))
        prior = best.get(key)
        if prior is None or (fact.observed_on or "") >= (prior.fact.observed_on or ""):
            best[key] = item
    return sorted(best.values(), key=lambda item: (-round(item.score, 7),
                                                    -_date_rank(item.fact.observed_on),
                                                    item.fact.source.url))


def assess_conflicts(evidence: list[EvidenceMatch]) -> ResolutionResult:
    """Detect competing values only when entity, attribute, and qualifiers agree."""
    conflicts = []
    for group in resolve_facts(item.fact for item in evidence if item.fact.attribute != "claim"):
        values = {str(fact.value) for fact in group.observations}
        if len(values) > 1:
            conflicts.append({"key": group.identity, "values": sorted(values), "winner": group.winner,
                              "evidence": group.observations, "reason": group.reason})
    note = None
    if conflicts:
        note = "; ".join(dict.fromkeys(conflict["reason"] for conflict in conflicts))
    return ResolutionResult(evidence=evidence, conflicts=conflicts, conflict_note=note)


def compose_answer(result: ResolutionResult, spec: QuerySpec) -> tuple[str, list[StructuredFact], bool]:
    """Render the strongest source span(s), never inventing an unstated value."""
    evidence = result.evidence
    if any(conflict["winner"] is None for conflict in result.conflicts):
        return "I don't know.", [], False
    winners = {conflict["key"]: normalized_value(conflict["winner"])
               for conflict in result.conflicts}
    evidence = [item for item in evidence
                if fact_identity(item.fact) not in winners
                or normalized_value(item.fact) == winners[fact_identity(item.fact)]]
    if not evidence or evidence[0].score < 0.16:
        return "I don't know.", [], False
    top = evidence[0].score
    threshold = max(0.16, top - (0.18 if spec.wants_list else 0.08))
    selected: list[StructuredFact] = []
    seen = set()
    for item in evidence:
        fact = item.fact
        if item.score < threshold:
            continue
        key = (fact.source.url, fact.source.snapshot_id, fact.source.section_id, fact.text)
        if key in seen:
            continue
        seen.add(key)
        selected.append(fact)
        if len(selected) >= (5 if spec.wants_list else 1):
            break
    if not selected:
        return "I don't know.", [], False
    lines = [fact.text.strip() for fact in selected if fact.text.strip()]
    return "\n".join(dict.fromkeys(lines)), selected, True
    