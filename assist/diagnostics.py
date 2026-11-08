"""Opt-in evidence traces; ordinary answer calls do not retain diagnostic data."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict

_ACTIVE: ContextVar[list[dict] | None] = ContextVar("evidence_trace", default=None)


def begin():
    events = []
    return events, _ACTIVE.set(events)


def end(token):
    _ACTIVE.reset(token)


def record(kind: str, **details):
    events = _ACTIVE.get()
    if events is not None:
        events.append({"kind": kind, **details})


def candidate_data(candidate):
    score, observed, url, snapshot_id, unit = candidate
    return {"url": url, "snapshot_id": snapshot_id, "observed_at": observed.isoformat(),
            "score": score, "question": unit.question, "heading": unit.headings, "passage": unit.answer,
            "effective_from": unit.effective_from.isoformat() if unit.effective_from else None,
            "effective_until": unit.effective_until.isoformat() if unit.effective_until else None}


def candidates(kind, values):
    if _ACTIVE.get() is not None:
        record(kind, candidates=[candidate_data(candidate) for candidate in values])


def candidate(kind, value, reason):
    if _ACTIVE.get() is not None:
        record(kind, candidate=candidate_data(value), reason=reason)


def ranking(values, score):
    if _ACTIVE.get() is not None:
        record("ranking", candidates=[{**candidate_data(candidate), "decision_score": score(candidate)}
                                      for candidate in values])


def selection_remainder(values, selected):
    if _ACTIVE.get() is not None:
        selected_keys = {(item[2], item[3], item[4].answer) for item in selected}
        record("selection_remainder", reason="Not selected for composition; the exact ranking or aggregation reason is not fully instrumented.",
               candidates=[candidate_data(item) for item in values
                           if (item[2], item[3], item[4].answer) not in selected_keys])


def fact(kind, value, reason=None):
    if _ACTIVE.get() is not None:
        record(kind, fact=asdict(value), reason=reason)


def resolution(group):
    if _ACTIVE.get() is not None:
        record("fact_resolution", identity=asdict(group.identity), reason=group.reason,
               observations=[asdict(item) for item in group.observations],
               active=[asdict(item) for item in group.active],
               winner=asdict(group.winner) if group.winner else None,
               relationships=[asdict(item) for item in group.relationships])


def filter_candidates(stage, before, after):
    if _ACTIVE.get() is not None:
        kept = {id(candidate) for candidate in after}
        record("filter", stage=stage,
               excluded=[candidate_data(candidate) for candidate in before if id(candidate) not in kept],
               retained=len(after))