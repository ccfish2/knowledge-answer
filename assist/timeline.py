"""Source-grounded observation and explicitly dated fact events."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from ingestion import StructuredFact
from resolver import FactIdentity, _instant, fact_identity, normalized_value


class EventType(str, Enum):
    OBSERVED = "observed"
    OBSERVED_CHANGE = "observed_change"
    EFFECTIVE = "effective"
    EXPIRED = "expired"


@dataclass(frozen=True)
class FactEvent:
    identity: FactIdentity
    kind: EventType
    fact: StructuredFact
    known_at: str
    event_at: str
    previous: StructuredFact | None = None


def build_timeline(facts: Iterable[StructuredFact], as_of: str | None = None) -> tuple[FactEvent, ...]:
    """Keep observations and dated claims distinct, without leaking future evidence."""
    cutoff = _instant(str(as_of), end_of_day=True) if as_of is not None else None
    ordered = sorted((fact for fact in facts if fact.observed_on and
                      (cutoff is None or _instant(fact.observed_on) <= cutoff)),
                     key=lambda fact: (_instant(fact.observed_on), fact.source.url,
                                       fact.source.snapshot_id, fact.source.section_id))
    events: list[FactEvent] = []
    previous: dict[tuple[FactIdentity, str], StructuredFact] = {}
    observation_keys = set()
    dated_keys = set()
    batches: dict[tuple[FactIdentity, str, object], list[StructuredFact]] = {}
    for fact in ordered:
        identity = fact_identity(fact)
        source_key = (identity, fact.source.url)
        value = normalized_value(fact)
        observation_key = (source_key, fact.source.snapshot_id, fact.source.section_id,
                           value, fact.effective_from, fact.effective_until)
        if observation_key in observation_keys:
            continue
        observation_keys.add(observation_key)
        batches.setdefault((identity, fact.source.url, _instant(fact.observed_on)), []).append(fact)
    for (identity, url, _), batch in batches.items():
        source_key = (identity, url)
        values = {normalized_value(fact) for fact in batch}
        prior = previous.get(source_key)
        current = batch[0]
        if len(values) == 1:
            if prior and normalized_value(prior) != normalized_value(current):
                events.append(FactEvent(identity, EventType.OBSERVED_CHANGE, current,
                                        current.observed_on, current.observed_on, prior))
            previous[source_key] = current
        else:
            previous.pop(source_key, None)
        for fact in batch:
            value = normalized_value(fact)
            known_at = fact.observed_on
            events.append(FactEvent(identity, EventType.OBSERVED, fact, known_at, known_at))
            for kind, day in ((EventType.EFFECTIVE, fact.effective_from), (EventType.EXPIRED, fact.effective_until)):
                key = (source_key, kind, value, day, fact.effective_from, fact.effective_until)
                if day and key not in dated_keys:
                    dated_keys.add(key)
                    events.append(FactEvent(identity, kind, fact, known_at, day))
    return tuple(events)