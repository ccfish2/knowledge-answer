"""Conservative enumeration of explicitly recorded entity relationships."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from ingestion import StructuredFact
import diagnostics


def _relationship_name(value: str) -> str:
    words = re.findall(r"[a-z0-9]+", value.lower())
    while words and words[0] in {"a", "an", "the"}:
        words.pop(0)
    if len(words) > 1 and words[-1] in {"service", "services"}:
        words.pop()
    return " ".join(words)


@dataclass(frozen=True)
class ListQueryPlan:
    entity_class: str
    relationship: str | None = None
    subject: str | None = None


class ListQueryPlanner:
    def plan(self, question: str) -> ListQueryPlan | None:
        options = re.search(r"\b(?:list|show|what are)\s+(?:me\s+)?(?:all\s+)?"
                            r"(?:applicable\s+)?(?:features|options)\s+(?:for|of)\s+(.+?)[?.!]*$",
                            question, re.I)
        if options:
            return ListQueryPlan("option", subject=options.group(1).strip())
        tier = re.search(r"\b(?:list|show|what are)\s+(?:me\s+)?(?:all\s+)?"
                         r"(?:eligible\s+)?tiers\s+for\s+(.+?)[?.!]*$", question, re.I)
        if tier:
            return ListQueryPlan("tier", subject=tier.group(1).strip())
        relationship = re.search(
            r"\b(?:which|what|list|all)\s+(branches|locations)\s+"
            r"(?:offer|offering|provide|providing|have|support)\s+(.+?)[?.!]*$", question, re.I)
        if relationship:
            entity_class = "branch" if relationship.group(1).lower() == "branches" else "location"
            return ListQueryPlan(entity_class, relationship.group(2).strip())
        inventory = re.search(
            r"\b(?:list|show|which|what)\s+(?:me\s+)?(?:are\s+)?(?:all\s+)?"
            r"(?:eligible\s+|applicable\s+)?(branches|locations|products|accounts|certificates|services|tiers|options)"
            r"(?:\s+(?:are\s+there|are\s+available|available|do\s+you\s+offer))?[?.!]*$", question, re.I)
        if not inventory:
            return None
        noun = inventory.group(1).lower()
        entity_class = {"branches": "branch", "locations": "location", "products": "product",
                "accounts": "account", "certificates": "certificate", "services": "service",
                        "tiers": "tier", "options": "option"}[noun]
        return ListQueryPlan(entity_class)


@dataclass(frozen=True)
class ListCoverage:
    inventory: int
    verified: int
    unverified: int
    supported: int
    included: int
    retrieval_complete: bool | None
    entity_complete: bool | None
    answer_complete: bool | None


class CompletenessVerifier:
    def verify(self, inventory: set[str], verified: set[str], supported: set[str],
               included: set[str]) -> ListCoverage:
        return ListCoverage(len(inventory), len(verified), len(inventory - verified),
                            len(supported), len(included), None, None,
                            supported == included if verified else None)


@dataclass(frozen=True)
class ListResult:
    matches: tuple[StructuredFact, ...]
    uncertain: bool
    complete: bool = False
    coverage: ListCoverage | None = None


class ListCompletenessResolver:
    def resolve(self, facts: Iterable[StructuredFact], *, entity_class: str, relationship: str,
                snapshot_id: str, cutoff: datetime, historical: bool = False,
                observation_only: bool = False) -> ListResult:
        relation = _relationship_name(relationship)
        candidates: dict[str, list[tuple[StructuredFact, bool]]] = {}
        statuses: dict[str, set[str]] = {}
        inventory = set()
        uncertain = False
        visible = []
        for fact in facts:
            diagnostics.fact("considered_fact", fact)
            if fact.source.snapshot_id != snapshot_id or fact.qualifiers.get("entity_class") != entity_class:
                diagnostics.fact("excluded_fact", fact, "different snapshot or entity class")
                continue
            day = cutoff.date().isoformat()
            if not observation_only and ((fact.effective_from and fact.effective_from > day)
                                         or (fact.effective_until and fact.effective_until < day)):
                diagnostics.fact("excluded_fact", fact, "outside the requested effective interval")
                continue
            if historical and not fact.effective_from and (not fact.observed_on or fact.observed_on[:10] != day):
                uncertain = True
                diagnostics.fact("excluded_fact", fact, "historical applicability is not established")
                continue
            visible.append(fact)
            key = " ".join(fact.entity.lower().split())
            inventory.add(key)
            status = str(fact.qualifiers.get("status", "")).strip().lower()
            if status in {"closed", "permanently closed"}:
                statuses.setdefault(key, set()).add("closed")
            elif status in {"open", "active"}:
                statuses.setdefault(key, set()).add("open")
        for fact in visible:
            attribute = _relationship_name(fact.attribute)
            value = " ".join(re.findall(r"[a-z0-9]+", str(fact.value).lower()))
            if attribute in {"service", "services"}:
                entries = [_relationship_name(part) for part in re.split(r"[,;/]", str(fact.value))]
                if relation not in entries:
                    diagnostics.fact("excluded_fact", fact, "requested relationship not stated in this service list")
                    continue
                supported = True
            elif attribute == relation and value in {"yes", "available", "offered", "no", "unavailable", "not offered"}:
                supported = value in {"yes", "available", "offered"}
            else:
                diagnostics.fact("excluded_fact", fact, "not an explicit assertion about the requested relationship")
                continue
            diagnostics.fact("matched_fact", fact, "explicit relationship assertion")
            candidates.setdefault(" ".join(fact.entity.lower().split()), []).append((fact, supported))
        selected = []
        verified = set()
        supported = set()
        inventory = {key for key in inventory if "closed" not in statuses.get(key, set())
                     or "open" in statuses.get(key, set())}
        for key, items in candidates.items():
            if key not in inventory:
                continue
            if len(statuses.get(key, set())) > 1:
                uncertain = True
                for fact, _ in items:
                    diagnostics.fact("excluded_fact", fact, "entity status sources disagree")
                continue
            if "closed" in statuses.get(key, set()):
                for fact, _ in items:
                    diagnostics.fact("excluded_fact", fact, "entity is marked closed")
                continue
            states = {supported for _, supported in items}
            if len(states) != 1:
                uncertain = True
                for fact, _ in items:
                    diagnostics.fact("excluded_fact", fact, "relationship sources disagree")
                continue
            verified.add(key)
            if True in states:
                supported.add(key)
                chosen = min((fact for fact, _ in items), key=lambda fact: (fact.source.url, fact.source.section_id))
                selected.append(chosen)
                diagnostics.fact("selected_fact", chosen, "visible explicit positive relationship")
        included = {" ".join(fact.entity.lower().split()) for fact in selected}
        coverage = CompletenessVerifier().verify(inventory, verified, supported, included)
        diagnostics.record("list_coverage", coverage=coverage.__dict__, complete=False,
                   reason="No authoritative exhaustive inventory was established.")
        uncertain = uncertain or bool(coverage.unverified)
        return ListResult(tuple(sorted(selected, key=lambda fact: fact.entity.lower())), uncertain,
                          complete=False, coverage=coverage)