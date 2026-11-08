
"""Structure Markdown pages into sections and source-grounded observations.

Extraction is deliberately deterministic and dependency-free. Table headers,
heading paths, nearby FAQ questions, and the original source span are retained
so a later query stage can make a choice without losing provenance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import PurePosixPath
from typing import Any


WORD = re.compile(r"[a-z0-9]+", re.I)
NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?\s*%?", re.I)
DATE = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b", re.I)
STOP = set("a an the and or of to for in on at by with from is are was were be this that it "
           "what which when where how do does can you your we our their its".split())
QUALIFIER_WORDS = ("term", "month", "year", "balance", "minimum", "maximum", "loan", "model",
                   "tier", "location", "branch", "region", "type", "account", "vehicle")
ATTRIBUTE_WORDS = {
    "apy": "apy", "apr": "apr", "rate": "rate", "dividend": "dividend_rate",
    "fee": "fee", "cost": "fee", "price": "price", "phone": "phone", "telephone": "phone",
    "address": "address", "hours": "hours", "hour": "hours", "count": "count",
    "number": "number", "date": "date", "minimum": "minimum", "maximum": "maximum",
    "amount": "amount", "term": "term", "duration": "duration", "email": "email",
}


def _plain(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_`]+", "", text)
    return re.sub(r"\s+", " ", text).strip(" \t|>")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w.lower() for w in WORD.findall(text) if w.lower() not in STOP)


def _slug(text: str) -> str:
    words = _tokens(text)
    return "_".join(sorted(words)) or "section"


def _unit(value: str, header: str = "") -> str | None:
    low = (header + " " + value).lower()
    if "%" in value or "percent" in low or "apy" in low or "apr" in low or "rate" in low:
        return "percent" if ("%" in value or "percent" in low) else None
    if "$" in value or "dollar" in low or "balance" in low or "fee" in low or "amount" in low:
        return "currency" if ("$" in value or "dollar" in low) else None
    if re.search(r"\b(?:month|months|year|years|day|days|hour|hours)\b", low):
        return "duration"
    return None


def _parse_date(raw: str) -> str | None:
    raw = raw.strip().replace(",", "")
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class Section:
    section_id: str
    heading_path: tuple[str, ...]
    text_span: str
    faq_question: str = ""
    table_headers: tuple[str, ...] = ()
    table_rows: tuple[tuple[str, ...], ...] = ()
    effective_from: str | None = None
    effective_until: str | None = None
    table_label: str = ""


@dataclass(frozen=True)
class SourceReference:
    snapshot_id: str
    file: str
    url: str
    section_id: str
    quote: str
    published_on: str | None = None
    fetched_at: str | None = None
    context: str = ""


@dataclass(frozen=True)
class StructuredFact:
    entity: str
    attribute: str
    value: Any
    unit: str | None
    qualifiers: dict[str, Any]
    effective_from: str | None
    effective_until: str | None
    observed_on: str | None
    source: SourceReference
    text: str

    @property
    def search_text(self) -> str:
        return " ".join((self.entity, self.attribute, str(self.value), self.text,
                         " ".join(f"{k} {v}" for k, v in self.qualifiers.items())))


def parse_sections(markdown: str) -> list[Section]:
    """Split Markdown into heading-scoped prose, FAQ, and table-row sections."""
    sections: list[Section] = []
    headings: list[str] = []
    pending: list[str] = []
    faq = ""
    section_no = 0

    def flush() -> None:
        nonlocal pending, faq, section_no
        text = _plain(" ".join(pending))
        if text:
            section_no += 1
            sections.append(Section(f"sec_{section_no:03d}", tuple(headings), text, faq))
        pending = []

    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        if not raw:
            flush()
            i += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", raw)
        if heading:
            flush()
            level = len(heading.group(1))
            headings = headings[:level - 1] + [_plain(heading.group(2))]
            faq = ""
            i += 1
            continue
        if "|" in raw and i + 1 < len(lines) and re.search(r"\|\s*:?-{3,}", lines[i + 1]):
            previous_line = i - 1
            while previous_line >= 0 and not lines[previous_line].strip():
                previous_line -= 1
            label = _plain(lines[previous_line]) if previous_line >= 0 else ""
            if len(label) > 100 or re.match(r"^#{1,6}\s+", label):
                label = ""
            flush()
            raw_rows = []
            i += 2  # header and separator
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                raw_rows.append(lines[i].strip())
                i += 1
            split = lambda row: tuple(_plain(c) for c in row.strip().strip("|").split("|"))
            headers = split(raw)
            rows = tuple(split(row) for row in raw_rows)

            # Rate sites commonly put “Rates are effective as of …” after a
            # table. Carry that date onto each row observation rather than
            # treating it as a separate unrelated prose section.
            effective_from = effective_until = None
            lookahead = i
            inspected = 0
            while lookahead < len(lines) and inspected < 5:
                candidate = lines[lookahead].strip()
                lookahead += 1
                if not candidate:
                    continue
                if re.match(r"^#{1,6}\s+", candidate) or ("|" in candidate and
                        lookahead < len(lines) and re.search(r"\|\s*:?-{3,}", lines[lookahead])):
                    break
                inspected += 1
                if re.search(r"\b(effective|expires?|through|until)\b", candidate, re.I):
                    effective_from, effective_until = _dates(candidate)
                    if effective_from or effective_until:
                        break
            for row in rows:
                section_no += 1
                row_text = " | ".join(f"{h}: {v}" for h, v in zip(headers, row) if v)
                sections.append(Section(f"sec_{section_no:03d}", tuple(headings), row_text,
                                        faq, headers, (row,), effective_from, effective_until, label))
            faq = ""
            continue
        text = _plain(raw)
        if text.endswith("?") and len(text) < 220:
            flush()
            faq = text
        else:
            pending.append(text)
        i += 1
    flush()
    return sections


def _entity(section: Section) -> str:
    if section.faq_question:
        return section.faq_question.rstrip("?")
    if section.heading_path:
        return section.heading_path[-1]
    # A descriptive first clause is a better generic entity label than the
    # entire paragraph, while preserving the full paragraph in source evidence.
    return re.split(r"[.!?;:]", section.text_span, maxsplit=1)[0][:100].strip() or "Website fact"


def _dates(text: str) -> tuple[str | None, str | None]:
    matches = [(m.start(), _parse_date(m.group(0))) for m in DATE.finditer(text)]
    matches = [(pos, val) for pos, val in matches if val]
    effective_from = effective_until = None
    for pos, value in matches:
        prefix = text[max(0, pos - 35):pos].lower()
        if re.search(r"\b(through|until|expires?|ending|ends?)\b", prefix):
            effective_until = value
        elif re.search(r"\b(effective|starting|beginning|from|as of)\b", prefix):
            effective_from = value
    return effective_from, effective_until


def _table_facts(section: Section, source: SourceReference, observed_on: str | None) -> list[StructuredFact]:
    if not section.table_headers or not section.table_rows:
        return []
    headers = section.table_headers
    row = section.table_rows[0]
    header_lookup = [h.lower() for h in headers]
    # Product/name/first column supplies entity. Other descriptive columns
    # become qualifiers; numeric/rate columns become individual observations.
    entity_idx = next((i for i, h in enumerate(header_lookup)
                       if any(x in h for x in ("product", "account", "loan", "certificate", "name", "vehicle"))), None)
    branch_idx = next((i for i, h in enumerate(header_lookup)
                       if re.fullmatch(r"branch(?:\s+(?:name|location))?", h.strip())), None)
    location_idx = next((i for i, h in enumerate(header_lookup)
                         if re.fullmatch(r"location(?:\s+name)?", h.strip())), None)
    if branch_idx is not None or location_idx is not None:
        entity_idx = branch_idx if branch_idx is not None else location_idx
    if entity_idx is None and not any(word in header_lookup[0] for word in QUALIFIER_WORDS):
        entity_idx = 0
    entity = row[entity_idx] if entity_idx is not None and entity_idx < len(row) else _entity(section)
    if not entity:
        entity = _entity(section)
    qualifiers: dict[str, Any] = {}
    if branch_idx is not None or location_idx is not None:
        qualifiers["entity_class"] = "branch" if branch_idx is not None else "location"
        status_idx = next((i for i, h in enumerate(header_lookup) if h.strip() == "status"), None)
        if status_idx is not None and status_idx < len(row):
            qualifiers["status"] = row[status_idx]
    facts: list[StructuredFact] = []
    entity_class = None
    if entity_idx is not None:
        entity_header = header_lookup[entity_idx].strip()
        if re.fullmatch(r"branch(?:\s+(?:name|location))?", entity_header):
            entity_class = "branch"
        elif re.fullmatch(r"location(?:\s+name)?", entity_header):
            entity_class = "location"
        elif re.fullmatch(r"(?:product|account|certificate)(?:\s+(?:type|name))?", entity_header):
            entity_class = entity_header.split()[0]
    if entity_class:
        facts.append(StructuredFact(entity, "entity", entity, None,
                                    {**qualifiers, "entity_class": entity_class},
                                    section.effective_from, section.effective_until,
                                    observed_on, source, section.text_span))
    for i, (header, value) in enumerate(zip(headers, row)):
        if i == entity_idx or not value:
            continue
        key = _slug(header).replace("_", " ")
        header_text = header_lookup[i]
        if re.search(r"\b(?:apy|annual percentage yield)\b", header_text):
            normalized_attribute = "apy"
        elif re.search(r"\b(?:apr|annual percentage rate)\b", header_text):
            normalized_attribute = "apr"
        elif re.search(r"\bdividend\s+rate\b", header_text):
            normalized_attribute = "dividend_rate"
        elif re.search(r"\binterest\s+rate\b", header_text):
            normalized_attribute = "interest_rate"
        else:
            normalized_attribute = next((attr for word, attr in ATTRIBUTE_WORDS.items() if word in header_text), None)
        if normalized_attribute or NUMBER.search(value):
            numeric = NUMBER.search(value)
            if normalized_attribute is None:
                normalized_attribute = _slug(header)
            if any(word in header_lookup[i] for word in QUALIFIER_WORDS) and not normalized_attribute.endswith(("rate", "apy", "apr", "fee", "amount")):
                qualifiers[key] = value
            else:
                # Preserve both endpoints of an explicitly published range.
                # Trailing footnote markers and APR/APY labels are not part
                # of the numeric value.
                rate_range = re.search(
                    r"\d+(?:\.\d+)?\s*%\s*(?:-|–|—|to)\s*\d+(?:\.\d+)?\s*%",
                    value, re.I)
                fact_value = rate_range.group(0).strip() if rate_range else (numeric.group(0).strip() if numeric else value)
                facts.append(StructuredFact(entity, normalized_attribute,
                    fact_value, _unit(value, header), dict(qualifiers),
                    section.effective_from, section.effective_until, observed_on, source, section.text_span))
        elif any(word in header_lookup[i] for word in QUALIFIER_WORDS):
            qualifiers[key] = value
        else:
            # Keep descriptive columns available for retrieval and rendering.
            facts.append(StructuredFact(entity, _slug(header), value, None, dict(qualifiers),
                                        section.effective_from, section.effective_until,
                                        observed_on, source, section.text_span))
    # Re-attach all row qualifiers, including columns appearing after rate columns.
    if qualifiers:
        facts = [StructuredFact(f.entity, f.attribute, f.value, f.unit, dict(qualifiers),
                                f.effective_from, f.effective_until, f.observed_on, f.source, f.text)
                 for f in facts]
    return facts


def extract_facts(markdown: str, *, snapshot_id: str, observed_on: str | None,
                  url: str, file: str | None = None, fetched_at: str | None = None) -> list[StructuredFact]:
    """Extract generic claim and value facts, keeping exact source provenance."""
    source_file = file or PurePosixPath(url.split("?", 1)[0]).name or ""
    published_on = None
    # Publication metadata is accepted only from a stand-alone date or an
    # author/byline near the top, never from arbitrary later article text.
    for line in markdown.splitlines()[:12]:
        candidate = _plain(line)
        match = DATE.search(candidate)
        if match and (candidate == match.group(0) or re.match(r"^written by\b", candidate, re.I)):
            published_on = _parse_date(match.group(0))
            break
    facts: list[StructuredFact] = []
    for section in parse_sections(markdown):
        context = " > ".join(section.heading_path + ((section.table_label,) if section.table_label else ()))
        source = SourceReference(snapshot_id, source_file, url, section.section_id, section.text_span[:500], published_on, fetched_at, context)
        from_date, until_date = _dates(section.text_span)
        table = _table_facts(section, source, observed_on)
        facts.extend(table)
        entity = _entity(section)
        # A whole statement fact makes prose searchable even when it contains no
        # number or a domain-specific attribute vocabulary is unfamiliar.
        facts.append(StructuredFact(entity, "claim", section.text_span, None, {}, from_date, until_date,
                                    observed_on, source, section.text_span))
        # A nested section explicitly connects its feature heading and body to
        # its document's top-level entity. Preserve this relationship rather
        # than inferring it from a number or a keyword elsewhere on the page.
        if (not section.table_headers and len(section.heading_path) >= 2
            and 25 <= len(section.text_span) <= 500):
            facts.append(StructuredFact(section.heading_path[0], "feature", section.heading_path[-1], None,
                        {"heading_path": section.heading_path}, from_date, until_date,
                        observed_on, source, section.text_span))
        # A proximity scan cannot establish which product or metric owns a
        # number. Only extract prose values when a single explicit label
        # provides both the entity and relationship; retain all other prose as
        # searchable claim evidence without manufacturing numeric facts.
        if not section.table_headers:
            owner = re.search(r"\b([A-Z][a-z][\w&'-]*(?:\s+[A-Z][a-z][\w&'-]*){1,6})\s+is\s+(?:a|an)\b",
                              section.text_span)
            counts = re.finditer(r"\b(\d[\d,]*)\s+(branch(?: office)? locations?|branches)\b",
                                 section.text_span, re.I)
            if owner and published_on:
                for count in counts:
                    quote = section.text_span[max(0, count.start() - 70):min(len(section.text_span), count.end() + 30)]
                    count_source = SourceReference(snapshot_id, source_file, url, section.section_id, quote, published_on, fetched_at, context)
                    facts.append(StructuredFact(owner.group(1), "count", count.group(1), "count",
                                                {"measure": "branch office locations"}, None, None,
                                                observed_on, count_source, section.text_span))
            labeled = re.fullmatch(
                r"(.{1,120}?)\s+\b(annual percentage yield|annual percentage rate|"
                r"interest rate|dividend rate|apy|apr|rate|fee)\s*:\s*"
                r"(\$?\d[\d,]*(?:\.\d+)?\s*%?)\s*\.?", section.text_span, re.I)
            if labeled:
                label, metric, value = labeled.groups()
                metric = metric.lower()
                attr = {"annual percentage yield": "apy", "annual percentage rate": "apr",
                        "interest rate": "interest_rate", "dividend rate": "dividend_rate"}.get(metric, metric)
                facts.append(StructuredFact(label.strip(), attr, value.strip(), _unit(value, metric), {},
                                            from_date, until_date, observed_on, source, section.text_span))
    return facts
    
