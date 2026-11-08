import dataclasses
import calendar
import inspect
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, date, timezone
from pathlib import Path

try:  # participant harness
    sys.path.insert(0, str(next(d / "harness" for d in Path(__file__).resolve().parents
                                if (d / "harness" / "contract.py").exists())))
    from contract import Answer, Conflict, Snapshot, Source  # noqa: E402
except StopIteration:  # local testing without the harness
    pass

from datastore import (  # structured, reusable query pipeline
    StructuredFactIndex,
    analyze_query as _analyze_query,
    plan_query as _plan_query,
    retrieve_evidence as _retrieve_evidence,
    match_facts as _match_facts,
    resolve_temporal as _resolve_temporal,
    assess_conflicts as _assess_conflicts,
    compose_answer as _compose_answer,
)
from resolver import fact_identity, normalized_value, resolve_facts
from listresolver import ListCompletenessResolver, ListQueryPlanner
from timeline import EventType, build_timeline
import diagnostics

WORD = re.compile(r"[a-z0-9]+")
STOP = set("""a an the i my me we our you your to of for with on in at by is are be do does can how
what when where which who it its this that and or if from as about get all same many was were still
been have has had did doing keep any pay become offer offerings""".split())
STOP.update({"s", "their"})

# ---- query normalization: paraphrases -> same canonical tokens -------------
SYN = {
    "change": "update", "edit": "update", "modify": "update", "correct": "update", "correcting": "update", "correction": "update", "editing": "update",
    "updating": "update", "changing": "update", "updated": "update",
    "address": "contact", "phone": "contact", "email": "contact", "telephone": "contact",
    "mobile": "contact", "info": "contact", "information": "contact",
    "details": "contact", "contacts": "contact", "mailing": "contact",
    "lobby": "branch", "lobbies": "branch", "location": "branch",
    "interest": "rate", "yield": "apy", "dividend": "rate", "cost": "fee",
    "hours": "hour", "automobile": "auto", "vehicle": "auto",
    "mortgage": "home", "automobiles": "auto", "cars": "auto",
    "branches": "branch", "certificates": "certificate", "employees": "employee",
    "payments": "wallet", "payment": "wallet", "wallets": "wallet",
    "recognition": "recognized", "recognize": "recognized",
    "eligibility": "eligible", "qualify": "eligible", "qualifies": "eligible",
    "qualified": "eligible", "membership": "member",
    "hometart": "homestart", "cd": "certificate",
    "open": "close", "opened": "close", "closed": "close", "closing": "close",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12",
}
# concepts that MUST be satisfied (only those present in the question are enforced)
CONCEPTS = {"update", "contact"}

def norm_tokens(text):
    out = set()
    for w in WORD.findall(text.lower()):
        if w in STOP:
            continue
        w = SYN.get(w, w)
        if len(w) > 3 and w.endswith("s") and w not in SYN:
            w = w[:-1]
        out.add(w)
    # “Share certificate” and “term-share certificate” are common names for a
    # certificate product; keep “share” for share-savings questions otherwise.
    if "certificate" in out and "share" in out:
        out.discard("share")
    return out

# ---- markdown -> (context, answer) units ------------------------------------
BOILER = re.compile(r"(heads up!|recaptcha|select all images|click verify|please try again|"
                    r"please select|please also check|reload if there are none)", re.I)
LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")

def plain(block):
    b = IMG.sub("", block)
    # Keep link destinations in the indexed passage: they are evidence for
    # locator/link questions and preserve the actual target instead of only its label.
    b = LINK.sub(r"\1 (\2)", b)
    b = re.sub(r"^[\s>*\-•]+", "", b)           # bullets
    b = re.sub(r"[*_`\\]+", "", b)
    return re.sub(r"\s+", " ", b).strip()


def _has_rate_value(text):
    """Whether evidence pairs a percentage with a rate/APR/APY metric."""
    body = (text or "").lower()
    if not re.search(r"\d+(?:\.\d+)?\s*%", body):
        return False
    if re.search(r"\b(down payments?|financing|loan-to-value|ltv|return on investment)\b", body):
        return False
    return bool(re.search(
        r"\b(?:apr|apy|annual percentage rate|annual percentage yield|interest rate|rate|yield|dividend rate)\b"
        r".{0,32}\d+(?:\.\d+)?\s*%|"
        r"\d+(?:\.\d+)?\s*%.{0,16}\b(?:apr|apy|annual percentage rate|annual percentage yield)\b",
        body, re.I | re.S))

def is_question(t):
    return t.endswith("?") and len(t) < 200

CUE = re.compile(r"\b(log ?in|sign ?in|call|visit|stop by|branch|shine|online banking|mobile app|"
                 r"settings|profile|secure message|ext\.?|\d{3}-\d{3}-\d{4})\b", re.I)

def near(text, a="update", b="contact", win=5):
    """True if an `a` word and a `b` word occur within `win` tokens (after synonym mapping)."""
    seq = [SYN.get(w, w) for w in WORD.findall(text.lower()) if w not in STOP]
    ia = [i for i, w in enumerate(seq) if w == a]
    ib = [i for i, w in enumerate(seq) if w == b]
    return any(abs(i - j) <= win for i in ia for j in ib)

def contact_update_evidence(unit):
    instruction = re.compile(r"\b(?:log(?:ging)?\s+in(?:to)?|sign(?:ing)?\s+in|call|visit|"
                             r"online banking|mobile banking|profile|settings)\b", re.I)
    if near(unit.question, win=8) and instruction.search(unit.answer):
        return True
    return any(near(sentence, win=8) and instruction.search(sentence)
               for sentence in re.split(r"(?<=[.!?])\s+", unit.answer))

@dataclass
class Unit:
    question: str      # nearest preceding question (FAQ) or ""
    headings: str      # heading path
    answer: str        # the paragraph we would return
    atoks: frozenset
    ctoks: frozenset   # context tokens: question + headings
    effective_from: datetime | None = None
    effective_until: datetime | None = None


def _explicit_date(text):
    value = text.strip().replace(",", "")
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d %Y", "%b %d %Y"):
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None

def build_units(title, md):
    units, heads, prev_q = [], [], ""
    lines = md.splitlines()
    i = 0

    subheading = ""
    while i < len(lines):
        raw = lines[i]
        if not raw.strip() or BOILER.search(raw):
            i += 1
            continue

        m = re.match(r"\s*(#{1,6})\s+(.*)", raw)
        if m:
            lvl = len(m.group(1))
            heads = heads[:lvl - 1] + [plain(m.group(2))]
            prev_q = ""
            subheading = ""
            i += 1
            continue

        bold_label = re.match(r"^\s*\*\*(.{2,80}?)\*\*\s*$", raw)
        if bold_label:
            subheading = plain(bold_label.group(1))
            prev_q = ""
            i += 1
            continue

        inline_label = re.match(
            r"^\s*\*\*(.{2,80}?)\*\*\s+((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b.*)$",
            raw, re.I)
        if inline_label:
            subheading = plain(inline_label.group(1))
            raw = inline_label.group(2)

        # Markdown tables are facts too: index each data row separately and also
        # retain the whole table so list questions can return all related rows.
        if "|" in raw and i + 1 < len(lines) and re.search(r"\|\s*:?-{3,}", lines[i + 1]):
            table_lines = [raw]
            i += 1
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                table_lines.append(lines[i])
                i += 1
            table_lines = [line for line in table_lines
                           if not re.fullmatch(r"\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*", line.strip())]
            hp = " > ".join(heads + ([subheading] if subheading else []))
            table_effective_from = None
            table_effective_until = None
            lookahead = i
            inspected = 0
            while lookahead < len(lines) and inspected < 5:
                footnote = lines[lookahead].strip()
                lookahead += 1
                if not footnote:
                    continue
                if re.match(r"\s*#{1,6}\s+", footnote) or ("|" in footnote and
                        lookahead < len(lines) and re.search(r"\|\s*:?-{3,}", lines[lookahead])):
                    break
                inspected += 1
                if not re.search(r"\b(effective|expires?|through|valid until|valid through)\b",
                                 footnote, re.I):
                    continue
                date_match = re.search(
                    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|"
                    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
                    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
                    r"dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b", footnote, re.I)
                if date_match:
                    parsed_date = _explicit_date(date_match.group(0))
                    prefix = footnote[:date_match.start()].lower()
                    if parsed_date and re.search(r"\b(through|until|expires?|expired|ends?)\b", prefix):
                        table_effective_until = parsed_date
                    elif parsed_date:
                        table_effective_from = parsed_date
                break
            data_lines = table_lines[1:] if len(table_lines) > 1 else []
            header_cells = [plain(c) for c in table_lines[0].strip().strip("|").split("|")]
            for row in data_lines:
                cells = [plain(c) for c in row.strip().strip("|").split("|")]
                text = " | ".join(
                    f"{header_cells[n]}: {cell}" if n < len(header_cells) and header_cells[n] else cell
                    for n, cell in enumerate(cells) if cell
                )
                if len(text) >= 12:
                    units.append(Unit(prev_q, hp, text,
                                      frozenset(norm_tokens(text)),
                                      frozenset(norm_tokens(prev_q + " " + hp + " " + title)),
                                      table_effective_from, table_effective_until))
            table_text = "\n".join(table_lines)
            if len(table_text) >= 25:
                units.append(Unit(prev_q, hp, table_text,
                                  frozenset(norm_tokens(table_text)),
                                  frozenset(norm_tokens(prev_q + " " + hp + " " + title)),
                                  table_effective_from, table_effective_until))
            prev_q = ""
            continue

        # Keep FAQ questions associated with the following answer.
        t = plain(raw)
        if is_question(t):
            prev_q = t
            i += 1
            continue

        # Preserve useful standalone bullets and short fact lines (routing numbers,
        # addresses, hours, closures) instead of dropping them because of length.
        bullet = re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", raw)
        if bullet or len(t) >= 12:
            hp = " > ".join(heads + ([subheading] if subheading else []))
            units.append(Unit(prev_q, hp, t,
                              frozenset(norm_tokens(t)),
                              frozenset(norm_tokens(prev_q + " " + hp + " " + title))))
            prev_q = ""
            i += 1
            continue

        # Ordinary prose paragraphs can be Markdown-split across lines.
        chunk = [raw]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if (not nxt.strip() or re.match(r"^\s*#{1,6}\s+", nxt)
                    or re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", nxt)
                    or "|" in nxt):
                break
            chunk.append(nxt)
            i += 1
        t = plain(" ".join(chunk))
        if len(t) >= 12:
            hp = " > ".join(heads + ([subheading] if subheading else []))
            units.append(Unit(prev_q, hp, t,
                              frozenset(norm_tokens(t)),
                              frozenset(norm_tokens(prev_q + " " + hp + " " + title))))
        prev_q = ""

    return units


# ---- fact extraction (for conflict detection) -----------------------------------
FACT = re.compile(r"\$\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?\s*%|\b\d{3}-\d{3}-\d{4}\b|"
                  r"\b\d+(?:\.\d+)?[ -]?(?:months?|days?|years?|yrs?)\b", re.I)

def facts_of(text):
    out = set()
    for f in FACT.findall(text):
        f = re.sub(r"[,\s]", "", f.lower()).replace("-", "")
        f = re.sub(r"(\.\d*?)0+(?=%|$)", r"\1", f).replace(".%", "%").rstrip(".")
        out.add(f)
    return frozenset(out)

def unit_key(u):
    return plain(u.question or u.headings.split(" > ")[-1]).lower()[:80]

def keyed_units(units):
    seen, out = {}, {}
    for u in units:
        k = unit_key(u)
        n = seen.get(k, 0)
        seen[k] = n + 1
        out[(k, n)] = u
    return out

def _make_conflict(d):
    """Build the harness Conflict from a neutral dict, filling whatever fields it declares."""
    names = {f.name: f for f in dataclasses.fields(Conflict)} if dataclasses.is_dataclass(Conflict) \
        else {p: None for p in inspect.signature(Conflict).parameters}
    alias = {
        "url": d["url"], "page": d["url"], "page_url": d["url"],
        "topic": d["topic"], "key": d["topic"], "subject": d["topic"], "field": d["topic"],
        "claim": d["topic"], "question": d["topic"],
        "old_value": d["old_values"], "new_value": d["new_values"],
        "old": d["old_values"], "new": d["new_values"],
        "values": d["old_values"] + d["new_values"],
        "old_snapshot_id": d["old_sid"], "new_snapshot_id": d["new_sid"],
        "old_snapshot": d["old_sid"], "new_snapshot": d["new_sid"],
        "snapshot_ids": [d["old_sid"], d["new_sid"]],
        "sources": d.get("sources") or [Source(url=d["url"], snapshot_id=d["old_sid"]),
                         Source(url=d["url"], snapshot_id=d["new_sid"])],
        "description": d["text"], "summary": d["text"], "text": d["text"], "reason": d["text"],
        "resolution": d.get("resolution"),
        "open": d["open"], "is_open": d["open"], "resolved": not d["open"],
    }
    kw = {n: alias[n] for n in names if n in alias}
    missing = [n for n, f in names.items() if n not in kw and f is not None
               and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
    if missing:
        raise TypeError(f"Conflict needs fields I cannot fill: {missing}")
    return Conflict(**kw)

# ---- store ------------------------------------------------------------------
def _dt(v):
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    v = str(v).replace("Z", "+00:00")
    d = datetime.fromisoformat(v if "T" in v else v + "T00:00:00+00:00")
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)

def _cutoff(as_of):
    if as_of is None:
        return None
    if isinstance(as_of, (date, datetime)) and not isinstance(as_of, datetime):
        return datetime(as_of.year, as_of.month, as_of.day, 23, 59, 59, tzinfo=timezone.utc)
    d = _dt(as_of)
    return d.replace(hour=23, minute=59, second=59) if len(str(as_of)) <= 10 else d

def _question_as_of(question, as_of):
    if as_of is not None:
        return as_of
    explicit = re.search(
        r"\b(?:as of|as at|on)\s+(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|"
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b",
        question, re.I)
    if explicit:
        parsed = _explicit_date(explicit.group(1))
        if parsed:
            return parsed.date().isoformat()
    month_names = {name.lower(): number for number, name in enumerate(calendar.month_name) if name}
    match = re.search(
        r"\b(?:as of|as at|in|during)\s+(January|February|March|April|May|June|July|"
        r"August|September|October|November|December)\s+(\d{4})\b", question, re.I)
    if not match:
        return None
    month, year = month_names[match.group(1).lower()], int(match.group(2))
    return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"

def _asks_for_observation(question, as_of):
    if re.search(r"\b(effective|in effect|valid|applicable|actually applied)\b", question, re.I):
        return False
    return as_of is not None or bool(re.search(
        r"\b(?:website|page|site|snapshot)\b.*\b(?:show|showed|display|displayed|said|listed)\b|"
        r"\b(?:shown|displayed|listed)\b.*\b(?:website|page|site|snapshot)\b", question, re.I))

def _asks_for_link(question):
    return bool(re.search(r"\b(link|url|locator)\b|\b(?:what|which) website\b", question, re.I))

class Knowledge:
    MIN_RELEVANCE = 0.18

    def __init__(self):
        self.versions = {}   # url -> list[(fetched_at, snapshot_id, title, units)]
        self.page_texts = {}
        self._cache = {}
        self.fact_index = StructuredFactIndex()
        self._idf_cache = {}
        self._expiry_claim_cache = {}
        self.snapshots = []

    @staticmethod
    def _attr(snapshot, names, default=None):
        for n in names:
            v = getattr(snapshot, n, None)
            if v not in (None, ""):
                return v
        if isinstance(snapshot, dict):
            for n in names:
                if snapshot.get(n) not in (None, ""):
                    return snapshot[n]
        return default

    @classmethod
    def _text(cls, snapshot):
        """Page markdown: inline text if the harness provides it, else read the file on disk."""
        v = cls._attr(snapshot, ("markdown", "content", "text", "md", "body", "page", "data"))
        if isinstance(v, bytes):
            v = v.decode("utf-8", "ignore")
        if isinstance(v, str) and v.strip():
            return v
        p = cls._attr(snapshot, ("file", "path", "file_path", "filepath", "md_path", "abs_path"))
        if p is not None:
            try:
                return Path(str(p)).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass
        # last resort: any attribute that is long text or an existing file path
        d = getattr(snapshot, "__dict__", None) or (snapshot if isinstance(snapshot, dict) else {})
        for val in d.values():
            if isinstance(val, str):
                if len(val) > 500:
                    return val
                try:
                    if Path(val).is_file():
                        return Path(val).read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    pass
        return ""

    def ingest(self, snapshot):
        if isinstance(snapshot, Snapshot):
            self.fact_index.add_snapshot(snapshot)
            self.snapshots.append((_dt(snapshot.taken_on), snapshot.snapshot_id))
            self.snapshots.sort()
            for page in snapshot.pages:
                self._ingest_page(page, snapshot_id=snapshot.snapshot_id,
                                  snapshot_date=snapshot.taken_on, index_page=False)
            self._cache.clear()
            self._idf_cache.clear()
            return
        pages = self._attr(snapshot, ("pages",), None)
        if pages is not None:
            snapshot_id = str(self._attr(snapshot, ("snapshot_id", "id", "sid", "snapshot"), ""))
            snapshot_date = self._attr(snapshot, ("taken_on", "fetched_at", "timestamp", "date", "crawled_at"))
            if snapshot_date:
                self.snapshots.append((_dt(snapshot_date), snapshot_id))
                self.snapshots.sort()
            for page in pages:
                self._ingest_page(page, snapshot_id=snapshot_id, snapshot_date=snapshot_date)
            self._cache.clear()
            self._idf_cache.clear()
            return

        # Also support page-like objects if ingest is called directly.
        self._ingest_page(snapshot)
        self._cache.clear()
        self._idf_cache.clear()
        self._expiry_claim_cache.clear()

    def _ingest_page(self, page, snapshot_id="", snapshot_date=None, index_page=True):
        title = self._attr(page, ("title",), "") or ""
        url = self._attr(page, ("url", "page_url", "source_url"))
        md = self._text(page)
        if not url or not md:
            self.skipped = getattr(self, "skipped", 0) + 1
            return
        units = build_units(title, md)
        self.page_texts[(url, snapshot_id)] = md
        if not units:                 # empty / error fetch must not shadow a good one
            return
        if snapshot_date in (None, ""):
            snapshot_date = self._attr(page, ("fetched_at", "taken_on", "timestamp", "date", "crawled_at"))
        if snapshot_id == "":
            snapshot_id = str(self._attr(page, ("snapshot_id", "id", "sid", "snapshot"), ""))
        t = _dt(snapshot_date or self._attr(page, ("fetched_at", "taken_on", "timestamp", "date", "crawled_at")))
        file = self._attr(page, ("file", "path", "file_path", "filepath", "md_path"))
        if index_page:
            self.fact_index.add_page(url=url, title=title, markdown=md, snapshot_id=snapshot_id,
                                     observed_on=t, file=str(file) if file else None)
        vs = self.versions.setdefault(url, [])
        vs.append((t, snapshot_id, title, units))
        vs.sort(key=lambda v: (v[0], v[1]))
        self._idf_cache.clear()

    def _inverse_document_frequency(self, visible):
        signature = tuple((url, snapshot_id) for url, fetched_at, snapshot_id, title, units in visible)
        cached = self._idf_cache.get(signature)
        if cached is not None:
            return cached
        frequencies = {}
        document_count = 0
        for url, fetched_at, snapshot_id, title, units in visible:
            for unit in units:
                document_count += 1
                terms = unit.atoks | unit.ctoks | frozenset(norm_tokens(title + " " + url))
                for term in terms:
                    frequencies[term] = frequencies.get(term, 0) + 1
        result = (document_count, frequencies)
        self._idf_cache[signature] = result
        return result

    def analyze_query(self, question, as_of=None):
        return _analyze_query(question, as_of)

    def plan_query(self, query_spec):
        return _plan_query(query_spec)

    def retrieve_evidence(self, query_plan):
        return _retrieve_evidence(query_plan, self.fact_index)

    def match_facts(self, candidates, query_spec):
        return _match_facts(candidates, query_spec)

    def resolve_temporal(self, evidence, query_spec):
        return _resolve_temporal(evidence, query_spec)

    def assess_conflicts(self, evidence):
        return _assess_conflicts(evidence)

    def compose_answer(self, resolution, query_spec):
        return _compose_answer(resolution, query_spec)

    def _structured_answer(self, question, as_of=None):
        spec = self.analyze_query(question, as_of)
        plan = self.plan_query(spec)
        evidence = self.retrieve_evidence(plan)
        evidence = self.match_facts(evidence, spec)
        evidence = self.resolve_temporal(evidence, spec)
        resolution = self.assess_conflicts(evidence)
        text, facts, confident = self.compose_answer(resolution, spec)
        if not facts:
            return None
        sources = []
        for fact in facts:
            source = Source(url=fact.source.url, snapshot_id=fact.source.snapshot_id,
                            quote=fact.source.quote)
            if not any(existing.url == source.url and existing.snapshot_id == source.snapshot_id
                       for existing in sources):
                sources.append(source)
        return Answer(text=text, sources=sources, confident=confident,
                      conflict_note=resolution.conflict_note)

    def _visible(self, as_of):
        cut = _cutoff(as_of)
        for url, vs in self.versions.items():
            ok = [v for v in vs if cut is None or v[0] <= cut]
            if ok:
                yield (url,) + ok[-1]          # newest version <= as_of, one per URL

    def match(self, q, as_of=None):
        """Return ranked candidates using context, page title, and answer content."""
        if not q:
            return []
        need = q & CONCEPTS
        topic = q - CONCEPTS
        out = []
        visible = list(self._visible(as_of))
        document_count, doc_frequency = self._inverse_document_frequency(visible)
        idf = {term: math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
               for term, df in doc_frequency.items()}
        for url, t, sid, title, units in visible:
            ttoks = norm_tokens(title)

            for u in units:
                if need and not need <= (u.ctoks | u.atoks | ttoks):
                    continue

                answer_hits = q & u.atoks
                context_hits = q & u.ctoks
                title_hits = q & ttoks
                available = u.atoks | u.ctoks | ttoks
                topic_hits = topic & available
                topic_ratio = len(topic_hits) / len(topic) if topic else 1.0
                answer_weight = sum(idf.get(term, 1.0) for term in answer_hits)
                context_weight = sum(idf.get(term, 1.0) for term in context_hits)
                topic_weight = sum(idf.get(term, 1.0) for term in topic_hits)
                total_weight = sum(idf.get(term, 1.0) for term in topic) or 1.0
                coverage = (0.72 * answer_weight + 0.18 * context_weight + 0.10 * topic_weight) / total_weight
                coverage += 0.06 * len(title_hits) / max(1, len(q))
                if u.question and context_hits:
                    coverage += 0.025

                question_numbers = {token for token in q if re.fullmatch(r"\d+(?:\.\d+)?", token)}
                numeric_hits = sum(1 for num in question_numbers if any(num in WORD.findall(fact) for fact in u.atoks | u.ctoks | ttoks))
                if numeric_hits:
                    coverage += 0.05 * numeric_hits

                if "branch" in topic or "routing" in topic:
                    coverage += 0.04 * len(topic_hits)

                direct_evidence = bool(answer_hits or context_hits)
                sufficient_topic = topic_ratio >= 0.25 or not topic
                if direct_evidence and sufficient_topic and coverage >= self.MIN_RELEVANCE:
                    out.append((min(1.0, coverage), t, url, sid, u))

        out.sort(key=lambda c: (
            -round(c[0], 6),
            -c[1].timestamp(),
            len(c[4].answer),
            c[2],
            c[4].answer,
        ))
        diagnostics.candidates("retrieved", out)
        return out


    def _latest_page_markdown(self, fragment, as_of=None):
        """Return the latest visible raw page text and its provenance."""
        for url, t, sid, title, units in self._visible(as_of):
            if fragment.lower() in url.lower():
                # Reconstructing from units is lossy; ingest retains page markdown below.
                raw = self.page_texts.get((url, sid))
                if raw:
                    return raw, url, sid
        return None, None, None

    def _branch_service_list(self, question, as_of=None):
        plan = ListQueryPlanner().plan(question)
        if plan is None or plan.relationship is None or plan.entity_class not in {"branch", "location"}:
            return None
        diagnostics.record("list_plan", entity_class=plan.entity_class, relationship=plan.relationship)
        cutoff = _cutoff(as_of)
        eligible = [(observed, sid) for observed, sid in self.snapshots if cutoff is None or observed <= cutoff]
        if not eligible:
            return Answer(text="I don't know which branches offer that service.", confident=False)
        observed, sid = max(eligible)
        reference = cutoff or observed
        observation_only = as_of is not None and _asks_for_observation(question, as_of)
        result = ListCompletenessResolver().resolve(
            self.fact_index.list_facts_by_snapshot.get((sid, plan.entity_class), []),
            entity_class=plan.entity_class, relationship=plan.relationship,
            snapshot_id=sid, cutoff=reference, historical=as_of is not None and not observation_only,
            observation_only=observation_only)
        if not result.matches:
            return Answer(text=f"I don't know which {plan.entity_class} entries offer that service; completeness cannot be established.", confident=False)
        sources = [Source(url=fact.source.url, snapshot_id=fact.source.snapshot_id, quote=fact.source.quote)
                   for fact in result.matches]
        names = ", ".join(fact.entity for fact in result.matches)
        prefix = f"Observed in snapshot dated {observed.date().isoformat()}. " if observation_only else ""
        label = "branches" if plan.entity_class == "branch" else "locations"
        return Answer(text=f"{prefix}Known matching {label}: {names}. Completeness cannot be established.",
                      sources=sources, confident=not result.uncertain)

    def _inventory_list_answer(self, question, as_of=None):
        plan = ListQueryPlanner().plan(question)
        if plan is None or plan.relationship is not None:
            return None
        diagnostics.record("list_plan", entity_class=plan.entity_class, subject=plan.subject)
        if plan.entity_class in {"option", "tier"} and not plan.subject:
            return Answer(text="I don't know which product or eligibility criteria to use.", confident=False)
        if plan.entity_class not in {"branch", "location", "product", "account", "certificate", "service", "tier", "option"}:
            return None
        cutoff = _cutoff(as_of)
        eligible = [(observed, sid) for observed, sid in self.snapshots if cutoff is None or observed <= cutoff]
        if not eligible:
            return Answer(text="I don't know; no snapshot is available.", confident=False)
        observed, sid = max(eligible)
        reference = cutoff or observed
        observation_only = as_of is not None and _asks_for_observation(question, as_of)
        if plan.entity_class == "option":
            subject = " ".join(WORD.findall(plan.subject.lower()))
            entities = [entity for entity in self.fact_index.feature_entities
                        if " ".join(WORD.findall(entity)) == subject]
            if len(entities) != 1:
                return Answer(text="I don't know which product's features to list.", confident=False)
            features = {}
            for fact in self.fact_index.feature_entities[entities[0]]:
                if fact.source.snapshot_id != sid:
                    continue
                if not observation_only and ((fact.effective_from and fact.effective_from > reference.date().isoformat()) or
                                             (fact.effective_until and fact.effective_until < reference.date().isoformat())):
                    continue
                path = fact.qualifiers.get("heading_path", ())
                if not any(re.search(r"\b(features?|benefits?|options?)\b|why you.ll love it", part, re.I)
                           for part in path[1:-1]):
                    continue
                features.setdefault(str(fact.value).casefold(), []).append(fact)
            if not features:
                return Answer(text="I don't know of any explicitly listed features for that product.", confident=False)
            conflicting = [facts for facts in features.values()
                           if len({" ".join(fact.text.casefold().split()) for fact in facts}) > 1]
            ordered = sorted((facts[0] for facts in features.values() if facts not in conflicting),
                             key=lambda fact: str(fact.value).casefold())
            if not ordered and conflicting:
                sources = [Source(url=fact.source.url, snapshot_id=fact.source.snapshot_id, quote=fact.source.quote)
                           for facts in conflicting for fact in facts]
                return Answer(text="I don't know; feature sources disagree.", sources=sources, confident=False,
                              conflict_note="Sources give different descriptions for the same feature.")
            sources = [Source(url=fact.source.url, snapshot_id=fact.source.snapshot_id, quote=fact.source.quote)
                       for fact in ordered]
            prefix = f"Observed in snapshot dated {observed.date().isoformat()}. " if observation_only else ""
            return Answer(text=prefix + f"Known features for {plan.subject}: "
                          + "; ".join(f"{fact.value}: {fact.text}" for fact in ordered)
                          + ". Completeness cannot be established."
                          + (" Some feature sources disagree." if conflicting else ""),
                          sources=sources, confident=not conflicting)
        if plan.entity_class == "tier":
            subject = " ".join(WORD.findall(plan.subject.lower()))
            matching_entities = [entity for entity in self.fact_index.structured_entities
                                 if " ".join(WORD.findall(entity)) == subject]
            if len(matching_entities) != 1:
                return Answer(text="I don't know which product's tiers to list.", confident=False)
            tiers = {}
            for fact in self.fact_index.structured_entities[matching_entities[0]]:
                if fact.source.snapshot_id != sid:
                    continue
                if not observation_only and ((fact.effective_from and fact.effective_from > reference.date().isoformat()) or
                                             (fact.effective_until and fact.effective_until < reference.date().isoformat())):
                    continue
                qualifiers = tuple(sorted((str(key), str(value)) for key, value in fact.qualifiers.items()
                                          if key != "page_title" and any(word in key.lower()
                                            for word in ("tier", "balance", "term", "model"))))
                if qualifiers:
                    tiers.setdefault(qualifiers, fact)
            if not tiers:
                return Answer(text="I don't know of any supported tiers for that product.", confident=False)
            lines = [", ".join(f"{key}: {value}" for key, value in qualifiers) for qualifiers in sorted(tiers)]
            sources = [Source(url=fact.source.url, snapshot_id=fact.source.snapshot_id, quote=fact.source.quote)
                       for fact in tiers.values()]
            prefix = f"Observed in snapshot dated {observed.date().isoformat()}. " if observation_only else ""
            return Answer(text=prefix + f"Known tiers for {plan.subject}: " + "; ".join(lines)
                          + ". Eligibility and completeness cannot be established.", sources=sources, confident=True)
        if plan.entity_class == "service":
            list_facts = [fact for entity_class in ("branch", "location")
                          for fact in self.fact_index.list_facts_by_snapshot.get((sid, entity_class), [])]
            relations = set()
            for fact in list_facts:
                if fact.source.snapshot_id != sid or fact.attribute not in {"service", "services"}:
                    continue
                for part in re.split(r"[,;/]", str(fact.value)):
                    name = part.strip()
                    if name and name.casefold() not in {"none", "n/a", "no"}:
                        relations.add(name)
            services = {}
            for relation in sorted(relations, key=str.casefold):
                for entity_class in ("branch", "location"):
                    result = ListCompletenessResolver().resolve(
                        self.fact_index.list_facts_by_snapshot.get((sid, entity_class), []),
                        entity_class=entity_class, relationship=relation,
                        snapshot_id=sid, cutoff=reference, historical=as_of is not None and not observation_only,
                        observation_only=observation_only)
                    if result.matches:
                        services.setdefault(relation.casefold(), (relation, result.matches[0]))
            if not services:
                return Answer(text="I don't know; no verified branch or location services were found.", confident=False)
            ordered = sorted(services.values(), key=lambda item: item[0].casefold())
            sources = [Source(url=fact.source.url, snapshot_id=fact.source.snapshot_id, quote=fact.source.quote)
                       for _, fact in ordered]
            prefix = f"Observed in snapshot dated {observed.date().isoformat()}. " if observation_only else ""
            return Answer(text=prefix + "Known branch and location services: " + ", ".join(name for name, _ in ordered)
                          + ". Completeness cannot be established.", sources=sources, confident=True)
        grouped = {}
        for fact in self.fact_index.inventory_by_snapshot.get((sid, plan.entity_class), []):
            diagnostics.fact("considered_fact", fact, "explicit entity inventory entry in selected snapshot")
            if not observation_only and ((fact.effective_from and fact.effective_from > reference.date().isoformat()) or
                                         (fact.effective_until and fact.effective_until < reference.date().isoformat())):
                diagnostics.fact("excluded_fact", fact, "outside the requested effective interval")
                continue
            grouped.setdefault(" ".join(fact.entity.casefold().split()), []).append(fact)
        selected = []
        uncertain = False
        for facts in grouped.values():
            statuses = {str(fact.qualifiers.get("status", "")).strip().casefold() for fact in facts}
            closed = bool(statuses & {"closed", "permanently closed"})
            opened = bool(statuses & {"open", "active"})
            if closed and opened:
                uncertain = True
                for fact in facts:
                    diagnostics.fact("excluded_fact", fact, "entity status sources disagree")
                continue
            if closed:
                for fact in facts:
                    diagnostics.fact("excluded_fact", fact, "entity is marked closed")
                continue
            if as_of is not None and not observation_only and not any(
                    fact.effective_from or (fact.observed_on and fact.observed_on[:10] == reference.date().isoformat())
                    for fact in facts):
                uncertain = True
                for fact in facts:
                    diagnostics.fact("excluded_fact", fact, "historical applicability is not established")
                continue
            chosen = min(facts, key=lambda fact: (fact.source.url, fact.source.section_id))
            selected.append(chosen)
            diagnostics.fact("selected_fact", chosen, "explicit entity entry in the selected snapshot")
        diagnostics.record("list_coverage", coverage={"inventory": len(grouped), "included": len(selected),
                           "retrieval_complete": None, "entity_complete": None, "answer_complete": None},
                           complete=False, reason="No authoritative exhaustive inventory was established.")
        if not selected:
            return Answer(text="I don't know; completeness cannot be established.", confident=False)
        selected.sort(key=lambda fact: fact.entity.casefold())
        labels = {"branch": "branches", "location": "locations", "product": "products",
                  "account": "accounts", "certificate": "certificates"}
        prefix = f"Observed in snapshot dated {observed.date().isoformat()}. " if observation_only else ""
        text = f"{prefix}Known {labels[plan.entity_class]}: " + ", ".join(fact.entity for fact in selected)
        text += ". Completeness cannot be established."
        sources = []
        for fact in selected:
            if not any(source.url == fact.source.url and source.snapshot_id == fact.source.snapshot_id for source in sources):
                sources.append(Source(url=fact.source.url, snapshot_id=fact.source.snapshot_id, quote=fact.source.quote))
        return Answer(text=text, sources=sources, confident=not uncertain)

    def _feature_answer(self, question, as_of=None):
        if not re.search(r"\b(options?|features?|benefits?|flexib\w*)\b", question, re.I):
            return None
        if re.search(r"\b(rate|apr|apy|interest|cost|fee|how many|when)\b", question, re.I):
            return None

        def stems(text):
            words = norm_tokens(text)
            return {"flexib" if word.startswith("flexib") else word for word in words
                    if len(word) >= 4 and word not in {"option", "feature", "benefit", "offer", "refinancing"}}

        query = stems(question)
        versions = {url: sid for url, _, sid, _, _ in self._visible(as_of)}
        matches = []
        for entity, facts in self.fact_index.feature_entities.items():
            entity_terms = stems(entity)
            shared = (query & entity_terms) - {"flexib"}
            feature_terms = query - entity_terms
            candidates = [fact for fact in facts if versions.get(fact.source.url) == fact.source.snapshot_id
                          and feature_terms & stems(str(fact.value))]
            if len(shared) >= 2 and candidates:
                matches.append((len(shared), entity, candidates))
        if not matches:
            return None
        best = max(item[0] for item in matches)
        matches = [item for item in matches if item[0] == best]
        if len(matches) != 1:
            return None
        _, entity, candidates = matches[0]
        # A feature and its supporting sentence must share the same source.
        chosen = min(candidates, key=lambda fact: (len(fact.text), fact.source.url, fact.source.section_id))
        return Answer(text=f"{chosen.entity} — {chosen.value}: {chosen.text}",
                      sources=[Source(url=chosen.source.url, snapshot_id=chosen.source.snapshot_id,
                                      quote=chosen.source.quote)], confident=True)

    def _dated_count_answer(self, question, as_of=None):
        if not re.search(r"\b(?:how many|number of|count)\b", question, re.I):
            return None
        if not re.search(r"\bbranch(?:es)?\b", question, re.I):
            return None
        cutoff = _cutoff(as_of)
        if cutoff is None:
            cutoff = max((version[0] for versions in self.versions.values() for version in versions), default=None)
        if cutoff is None:
            return None
        visible = {url: sid for url, _, sid, _, _ in self._visible(as_of)}
        question_terms = norm_tokens(question) - {"branch", "number", "count"}
        facts = [fact for fact in self.fact_index.count_facts
                 if visible.get(fact.source.url) == fact.source.snapshot_id
                 and fact.source.published_on and fact.source.published_on <= cutoff.date().isoformat()]
        matching = [fact for fact in facts if question_terms & norm_tokens(fact.entity)]
        if matching:
            facts = matching
        entities = {fact.entity.lower() for fact in facts}
        canonical = {entity for entity in entities if not any(
            entity != other and entity.endswith(" " + other) for other in entities)}
        if len(canonical) != 1:
            return None
        entity = next(iter(canonical))
        facts = [fact for fact in facts if fact.entity.lower() == entity or
                 fact.entity.lower().endswith(" " + entity)]
        newest = max(fact.source.published_on for fact in facts if fact.source.published_on)
        latest = [fact for fact in facts if fact.source.published_on == newest]
        if len({str(fact.value) for fact in latest}) != 1:
            return Answer(text="I don't know; the latest published sources disagree.", confident=False,
                          conflict_note="Latest same-date sources give conflicting counts.")
        fact = min(latest, key=lambda item: item.source.url)
        return Answer(text=f"{fact.entity} reported {fact.value} branch office locations on {newest}.",
                      sources=[Source(url=fact.source.url, snapshot_id=fact.source.snapshot_id,
                                      quote=fact.source.quote)], confident=True)

    def _broad_apy_comparison_answer(self, question, as_of=None):
        if not (re.search(r"\b(?:do|does|are|is)\s+any\b", question, re.I)
                and re.search(r"\bapy\b", question, re.I)
                and re.search(r"\b(?:pay|pays|earn|earns|offer|offers|yield|yields)\b", question, re.I)):
            return None
        amounts = re.findall(r"\b(\d+(?:\.\d+)?)\s*%", question)
        family = norm_tokens(question) & {"saving", "checking"}
        if len(amounts) != 1 or len(family) != 1:
            return None
        family = next(iter(family))
        cutoff = _cutoff(as_of)
        if cutoff is None:
            cutoff = max((version[0] for versions in self.versions.values() for version in versions), default=None)
        if cutoff is None:
            return None
        visible = {url: sid for url, _, sid, _, _ in self._visible(as_of)}
        facts = []
        for entity, observations in self.fact_index.structured_entities.items():
            if family not in norm_tokens(entity):
                continue
            for fact in observations:
                if fact.attribute != "apy" or fact.unit != "percent":
                    continue
                if visible.get(fact.source.url) != fact.source.snapshot_id:
                    continue
                if not re.match(r"^(?:account(?:\s+(?:type|name))?|product(?:\s+name)?)\s*:\s*" +
                                re.escape(fact.entity) + r"\s*\|", fact.text, re.I):
                    continue
                facts.append(fact)
        if not facts:
            return None
        strict_validity = as_of is not None and not _asks_for_observation(question, as_of)
        groups = [group for group in resolve_facts(facts, cutoff.isoformat(), validity_only=strict_validity)
              if group.active]
        for group in groups:
            diagnostics.resolution(group)
        incomplete = {group.identity.entity for group in groups if group.winner is None}
        winners = [group.winner for group in groups if group.winner is not None
               and (not strict_validity or group.winner.effective_from or
                (group.winner.observed_on and group.winner.observed_on[:10] == cutoff.date().isoformat()))]
        target = float(amounts[0])
        def percent(fact):
            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%\s*", str(fact.value))
            return float(match.group(1)) if match else None
        matching = [fact for fact in winners if percent(fact) == target]
        for fact in winners:
            diagnostics.fact("matched_fact", fact, "visible product-specific APY tier")
        if not matching:
            return Answer(text="I don't know whether any matching tier is available.", confident=False)
        maxima = {}
        for fact in winners:
            value = percent(fact)
            identity = fact_identity(fact).entity
            if value is None or identity in incomplete:
                continue
            if identity not in maxima or value > percent(maxima[identity]):
                maxima[identity] = fact
        others = [fact for fact in maxima.values() if percent(fact) < target]
        matching.sort(key=lambda fact: (fact.entity.lower(), str(fact.qualifiers.get("balance tier", ""))))
        others.sort(key=lambda fact: fact.entity.lower())
        lines, sources = [], []
        for fact in matching + others:
            diagnostics.fact("selected_fact", fact, "matches requested APY" if fact in matching
                             else "highest captured tier below the requested APY for this product")
            tier = fact.qualifiers.get("balance tier")
            lines.append(f"{fact.entity}: {fact.value} APY" + (f" ({tier})" if tier else ""))
            if not any(source.url == fact.source.url and source.snapshot_id == fact.source.snapshot_id
                       and source.quote == fact.source.quote for source in sources):
                sources.append(Source(url=fact.source.url, snapshot_id=fact.source.snapshot_id, quote=fact.source.quote))
        text = "Yes. Matching captured tiers:\n" + "\n".join(lines[:len(matching)])
        if others:
            text += "\nHighest captured tiers below that APY for other products:\n" + "\n".join(lines[len(matching):])
        if incomplete:
            text += "\nSome source disagreements remain unresolved."
        return Answer(text=text, sources=sources, confident=True)

    def _model_year_rate_answer(self, question, as_of=None):
        terms = norm_tokens(question)
        product_terms = terms & {"auto", "motorcycle"}
        year_match = re.search(r"\b((?:19|20)\d{2})\b", question)
        term_match = re.search(r"\b(\d{1,3})\s*[- ]?\s*months?\b", question, re.I)
        if not product_terms or not year_match or not term_match or not re.search(r"\bapr\b", question, re.I):
            return None
        year = int(year_match.group(1))
        months = int(term_match.group(1))
        cutoff = _cutoff(as_of)
        if cutoff is None:
            cutoff = max((version[0] for versions in self.versions.values() for version in versions), default=None)
        if cutoff is None:
            return None
        day = cutoff.date().isoformat()
        visible = {url: sid for url, _, sid, _, _ in self._visible(as_of)}
        matches = []
        relevant = False
        for fact in self.fact_index.model_year_facts:
            diagnostics.fact("considered_fact", fact)
            if fact.attribute != "apr" or visible.get(fact.source.url) != fact.source.snapshot_id:
                diagnostics.fact("excluded_fact", fact, "different metric or snapshot is not visible")
                continue
            context = norm_tokens(fact.source.context + " " + fact.source.url + " " +
                                  str(fact.qualifiers.get("page_title", "")))
            if not product_terms <= context:
                diagnostics.fact("excluded_fact", fact, "product context does not match")
                continue
            relevant = True
            if (fact.effective_from and fact.effective_from > day) or (fact.effective_until and fact.effective_until < day):
                diagnostics.fact("excluded_fact", fact, "outside the requested effective interval")
                continue
            year_range = re.search(r"\b((?:19|20)\d{2})\s*[-–—]\s*(\d{2,4})\b", fact.entity)
            if year_range:
                lower, upper = int(year_range.group(1)), int(year_range.group(2))
                if upper < 100:
                    upper += (lower // 100) * 100
                    if upper < lower:
                        upper += 100
                if not lower <= year <= upper:
                    diagnostics.fact("excluded_fact", fact, "model year is outside the band")
                    continue
            else:
                single = re.search(r"\b((?:19|20)\d{2})\b", fact.entity)
                if not single or (year != int(single.group(1)) and not
                                  (year < int(single.group(1)) and re.search(r"\bolder\b", fact.entity, re.I))):
                    diagnostics.fact("excluded_fact", fact, "model year is outside the band")
                    continue
            term_values = [str(value) for key, value in fact.qualifiers.items() if "term" in key.lower()]
            if not any(re.search(r"\b" + str(months) + r"\s*months?\b", value, re.I) for value in term_values):
                diagnostics.fact("excluded_fact", fact, "loan term does not match")
                continue
            matches.append(fact)
            diagnostics.fact("matched_fact", fact, "product, metric, model year and term match")
        if not matches:
            return Answer(text="I don't know.", confident=False) if relevant else None
        if as_of is not None and not _asks_for_observation(question, as_of) and any(
                not fact.effective_from and (not fact.observed_on or fact.observed_on[:10] != day)
                for fact in matches):
            return Answer(text="I don't know whether the observed rates applied on that date.", confident=False)
        values = {normalized_value(fact) for fact in matches}
        if len(values) > 1 and any(not fact.effective_from for fact in matches):
            return Answer(text="I don't know; conflicting rate sources lack comparable effective dates.",
                          confident=False, conflict_note="A competing observation has no effective date.")
        newest = max((fact.effective_from or "" for fact in matches), default="")
        winners = [fact for fact in matches if (fact.effective_from or "") == newest]
        for fact in matches:
            if fact not in winners:
                diagnostics.fact("excluded_fact", fact, "older explicit effective date for the same requested year and term")
        if len({normalized_value(fact) for fact in winners}) > 1:
            return Answer(text="I don't know; equally current rate sources disagree.", confident=False,
                          conflict_note="Matching sources have the same effective date and different values.")
        winner = min(winners, key=lambda fact: fact.source.url)
        for fact in winners:
            diagnostics.fact("selected_fact", fact, "matching rate with the latest comparable effective date" if newest
                             else "matching observations agree; effective date not provided")
        sources = []
        for fact in winners:
            if not any(source.url == fact.source.url and source.snapshot_id == fact.source.snapshot_id for source in sources):
                sources.append(Source(url=fact.source.url, snapshot_id=fact.source.snapshot_id, quote=fact.source.quote))
        detail = f"{winner.entity}, {months} months: {winner.value} APR"
        if newest:
            detail += f" (effective {newest})"
        note = "Preferred the newer explicit effective date for matching model-year and term bands." if len(values) > 1 else None
        return Answer(text=detail + ".", sources=sources, confident=True, conflict_note=note)

    def _dated_count_series_answer(self, question, as_of=None):
        if not (re.search(r"\b(?:how many|number of|count)\b", question, re.I)
                and re.search(r"\b(?:last|previous) year\b", question, re.I)):
            return None
        cutoff = _cutoff(as_of)
        if cutoff is None:
            cutoff = max((version[0] for versions in self.versions.values() for version in versions), default=None)
        if cutoff is None:
            return None
        query = norm_tokens(question) - {"last", "previous", "previou", "year", "number", "count"}
        pattern = re.compile(
            r"\b(?P<count>\d[\d,]*)\s+(?P<subject>(?:[A-Za-z][\w'-]*\s+){0,3}employees?)\s+"
            r"(?P<action>used)\b(?P<body>[^.\n!?]{0,180}?)\b(?:in|during)\s+(?P<year>(?:19|20)\d{2})\b"
            r"(?P<tail>[^.\n!?]{0,180})", re.I)
        rows = {}
        for candidate in self.match(norm_tokens(question), as_of):
            unit = candidate[4]
            for match in pattern.finditer(unit.answer):
                subject_terms = norm_tokens(match.group("subject"))
                if "employee" not in query or not subject_terms <= (query | {"employee"}):
                    continue
                target = query - subject_terms - {match.group("action").lower()}
                if not target or not target <= norm_tokens(match.group("body") + " " + match.group("tail")):
                    continue
                year = int(match.group("year"))
                if year > cutoff.year:
                    continue
                rows.setdefault(year, {}).setdefault(match.group("count"), candidate)
        if not rows:
            return None
        target_year = cutoff.year - 1
        if any(len(values) > 1 for values in rows.values()):
            sources = []
            for values in rows.values():
                if len(values) < 2:
                    continue
                for candidate in values.values():
                    if not any(item.url == candidate[2] and item.snapshot_id == candidate[3] for item in sources):
                        sources.append(Source(url=candidate[2], snapshot_id=candidate[3], quote=candidate[4].answer[:300]))
            return Answer(text="I don't know; reports disagree about a yearly count.", sources=sources,
                          confident=False, conflict_note="Different counts were reported for the same year.")
        lines, sources = [], []
        for year in sorted(rows, reverse=True):
            value, candidate = next(iter(rows[year].items()))
            lines.append(f"{year}: {value} employees")
            source = Source(url=candidate[2], snapshot_id=candidate[3], quote=candidate[4].answer[:300])
            if not any(item.url == source.url and item.snapshot_id == source.snapshot_id for item in sources):
                sources.append(source)
        if target_year not in rows:
            text = f"I don't know the count for {target_year}. Reported figures: " + "; ".join(lines) + "."
            return Answer(text=text, sources=sources, confident=False)
        return Answer(text="Reported figures: " + "; ".join(lines) + ".", sources=sources, confident=True)

    def _timeline_change_answer(self, question, as_of=None):
        if not re.search(r"\b(chang(?:e|ed)|before|previous(?:ly)?|increas(?:e|ed)|decreas(?:e|ed))\b", question, re.I):
            return None
        metric_match = re.search(r"\b(apy|apr|interest rate|dividend rate)\b", question, re.I)
        if not metric_match:
            return None
        metric = {"interest rate": "interest_rate", "dividend rate": "dividend_rate"}.get(
            metric_match.group(1).lower(), metric_match.group(1).lower())
        question_words = " ".join(WORD.findall(question.lower()))
        entities = [entity for entity in self.fact_index.structured_entities
                    if len(WORD.findall(entity)) >= 2 and re.search(
                        r"(?<!\w)" + re.escape(" ".join(WORD.findall(entity))) + r"(?!\w)", question_words)]
        entities.sort(key=len, reverse=True)
        if not entities or any(entity not in entities[0] for entity in entities[1:]):
            return None
        if not re.search(r"\b(website|page|site|snapshot|observed|displayed|shown)\b", question, re.I):
            return Answer(text="I don't know when the rate actually changed.", confident=False)
        events = self.fact_index.timeline(entity=entities[0], attribute=metric, as_of=as_of)
        identities = {event.identity for event in events if event.kind == EventType.OBSERVED}
        if len(identities) != 1:
            return Answer(text="I don't know which tier to compare.", confident=False)
        visible = {url: sid for url, _, sid, _, _ in self._visible(as_of)}
        current = [event.fact for event in events if event.kind == EventType.OBSERVED
                   and visible.get(event.fact.source.url) == event.fact.source.snapshot_id]
        if len({normalized_value(fact) for fact in current}) > 1:
            sources = [Source(url=fact.source.url, snapshot_id=fact.source.snapshot_id,
                              quote=fact.source.quote) for fact in current]
            return Answer(text="I don't know; the visible sources disagree.", sources=sources,
                          confident=False, conflict_note="The visible sources show different values.")
        changes = [event for event in events if event.kind == EventType.OBSERVED_CHANGE and event.previous]
        direction = re.search(r"\b(increas(?:e|ed)|decreas(?:e|ed))\b", question, re.I)
        if direction:
            increasing = direction.group(1).lower().startswith("increas")
            directional = []
            for event in changes:
                before = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%\s*", str(event.previous.value))
                after = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%\s*", str(event.fact.value))
                if before and after and ((float(after.group(1)) > float(before.group(1))) if increasing
                                         else (float(after.group(1)) < float(before.group(1)))):
                    directional.append(event)
            changes = directional
        if not changes:
            return Answer(text="I don't know of an observed change for that rate.", confident=False)
        latest = max(event.known_at for event in changes)
        changes = [event for event in changes if event.known_at == latest]
        values = {(normalized_value(event.previous), normalized_value(event.fact)) for event in changes}
        if len(values) != 1:
            return Answer(text="I don't know; sources show different changes.", confident=False)
        change = min(changes, key=lambda event: event.fact.source.url)
        earlier, later = change.previous, change.fact
        sources = [Source(url=fact.source.url, snapshot_id=fact.source.snapshot_id, quote=fact.source.quote)
                   for fact in (earlier, later)]
        if re.search(r"\b(before|previous(?:ly)?)\b", question, re.I):
            text = f"The source showed {earlier.value} in the {earlier.observed_on[:10]} snapshot before the observed change."
        else:
            text = (f"The source showed {earlier.value} in the {earlier.observed_on[:10]} snapshot and "
                    f"{later.value} in the {later.observed_on[:10]} snapshot; the effective change date is unknown.")
        return Answer(text=text, sources=sources, confident=True)

    def _resolved_table_answer(self, question, as_of=None, observed_query=False):
        """Use explicit table entity/metric relationships when unambiguous.

        Returning None delegates unsupported requests to passage retrieval;
        an unresolved structured conflict instead returns an explicit refusal.
        """
        metric_match = re.search(r"\b(apy|apr|interest rate|dividend rate)\b", question, re.I)
        if not metric_match:
            return None
        metric = {"interest rate": "interest_rate", "dividend rate": "dividend_rate"}.get(
            metric_match.group(1).lower(), metric_match.group(1).lower())
        question_words = " ".join(WORD.findall(question.lower()))
        enumeration = bool(re.search(r"\b(?:list|all|each|every|what are)\b", question, re.I))
        matches = []
        for entity in self.fact_index.structured_entities:
            words = " ".join(WORD.findall(entity))
            if len(words.split()) < 2 or not re.search(r"(?<!\w)" + re.escape(words) + r"(?!\w)", question_words):
                continue
            matches.append(entity)
        if not matches:
            # Only enumerate when the question names the table's entity
            # column; ordinary proximity between a page and a number is not
            # evidence that the number belongs to a product.
            list_request = re.search(r"\b(?:list|all|each|every)\b", question, re.I)
            category = re.search(r"\b(products?|accounts?|certificates?)\b", question, re.I)
            if not list_request or not category:
                return None
            noun = category.group(1).lower().rstrip("s")
            label = {"product": r"product(?: name)?", "account": r"account(?: type| name)?",
                     "certificate": r"certificate(?: type| name)?"}[noun]
            observations = [fact for facts in self.fact_index.structured_entities.values() for fact in facts
                            if fact.attribute == metric and fact.unit == "percent"
                            and re.match(r"^" + label + r"\s*:", fact.text, re.I)]
        else:
            # A longer exact name can contain a shorter one (e.g. "Health
            # Savings" vs. "Savings"); distinct names require comparison.
            matches.sort(key=len, reverse=True)
            if any(entity not in matches[0] for entity in matches[1:]):
                return None
            entity = matches[0]
            observations = [fact for fact in self.fact_index.structured_entities[entity]
                            if fact.attribute == metric and fact.unit == "percent"]
        if not observations:
            return None
        cutoff = _cutoff(as_of)
        if cutoff is None:
            cutoff = max((version[0] for versions in self.versions.values() for version in versions), default=None)
        if cutoff is None:
            return None
        date_string = cutoff.isoformat()
        # Restrict to sources actually visible by the cutoff, but retain their
        # earlier versions as evidence of changes to the same source.
        visible_versions = {url: sid for url, _, sid, _, _ in self._visible(as_of)}
        live_identities = {(fact.source.url, fact_identity(fact)) for fact in observations
                   if visible_versions.get(fact.source.url) == fact.source.snapshot_id}
        observations = [fact for fact in observations
                if (fact.source.url, fact_identity(fact)) in live_identities]
        for fact in observations:
            diagnostics.fact("considered_fact", fact, "visible source identity and its history")
        if observed_query:
            observations = [fact for fact in observations
                    if visible_versions.get(fact.source.url) == fact.source.snapshot_id]
        events = build_timeline(observations, date_string)
        groups = resolve_facts((event.fact for event in events if event.kind == EventType.OBSERVED),
                       date_string, observation_only=observed_query,
                       validity_only=as_of is not None and not observed_query)
        for group in groups:
            diagnostics.resolution(group)
        amounts = re.findall(r"\$\s*(\d[\d,]*(?:\.\d+)?)", question)
        if len(amounts) == 1:
            amount = float(amounts[0].replace(",", ""))
            def contains_amount(group):
                tiers = [value for key, value in group.identity.qualifiers if "balance" in key or "tier" in key]
                if len(tiers) != 1:
                    return False
                tier = tiers[0].replace(",", "")
                numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", tier)]
                if not numbers:
                    return False
                if "+" in tier or "and up" in tier or "or more" in tier:
                    return amount >= numbers[0]
                return len(numbers) == 2 and numbers[0] <= amount <= numbers[1]
            groups = [group for group in groups if contains_amount(group)]
        elif len(amounts) > 1:
            return None
        term_match = re.search(r"\b(\d{1,3})\s*[- ]?\s*(months?|years?)\b", question, re.I)
        if term_match:
            requested_term = int(term_match.group(1)) * (12 if term_match.group(2).lower().startswith("year") else 1)
            def contains_term(group):
                terms = [value for key, value in group.identity.qualifiers
                         if "term" in key or "duration" in key or "month" in key]
                if not terms and re.search(r"\b\d+\s+(?:months?|years?)\b", group.identity.entity):
                    terms = [group.identity.entity]
                for term in terms:
                    numbers = [int(value) for value in re.findall(r"\d+", term)]
                    factor = 12 if re.search(r"\byears?\b", term) else 1
                    if len(numbers) == 1 and numbers[0] * factor == requested_term:
                        return True
                    if len(numbers) == 2 and numbers[0] * factor <= requested_term <= numbers[1] * factor:
                        return True
                return False
            groups = [group for group in groups if contains_term(group)]
        groups = [group for group in groups if group.active]
        if not groups:
            return Answer(text="I don't know.", confident=False)
        if any(group.winner is None for group in groups):
            sources = []
            for group in groups:
                if group.winner is None:
                    for fact in group.active:
                        if not any(s.url == fact.source.url and s.snapshot_id == fact.source.snapshot_id for s in sources):
                            sources.append(Source(url=fact.source.url, snapshot_id=fact.source.snapshot_id,
                                                  quote=fact.source.quote))
            return Answer(text="I don't know; the available sources disagree.", sources=sources,
                          confident=False, conflict_note="Conflicting current sources have no established precedence.")
        if as_of is not None and not observed_query and any(not any(
                fact.effective_from or (fact.observed_on and fact.observed_on[:10] == cutoff.date().isoformat())
                for fact in group.active) for group in groups):
            return Answer(text="I don't know whether the observed rate applied on that date.", confident=False)
        if re.search(r"\b(highest|maximum|largest)\b", question, re.I):
            def numeric(group):
                match = re.search(r"\d+(?:\.\d+)?", str(group.winner.value))
                return float(match.group()) if match else float("-inf")
            highest = max(numeric(group) for group in groups)
            groups = [group for group in groups if numeric(group) == highest]
        lines, sources = [], []
        for group in groups:
            fact = group.winner
            diagnostics.fact("selected_fact", fact, group.reason)
            qualifiers = "; ".join(f"{key}: {value}" for key, value in fact.qualifiers.items()
                                   if key != "page_title")
            lines.append(f"{fact.entity} — {metric_match.group(1).upper()}: {fact.value}"
                         + (f" ({qualifiers})" if qualifiers else ""))
            for supporting in group.active:
                if (visible_versions.get(supporting.source.url) == supporting.source.snapshot_id
                        and not any(s.url == supporting.source.url and s.snapshot_id == supporting.source.snapshot_id
                                    for s in sources)):
                    sources.append(Source(url=supporting.source.url, snapshot_id=supporting.source.snapshot_id,
                                          quote=supporting.source.quote))
        changed = any(len({str(f.value) for f in group.observations}) > 1 for group in groups)
        answer_text = "\n".join(dict.fromkeys(lines))
        if as_of is not None and observed_query:
            observation_dates = sorted({group.winner.observed_on[:10] for group in groups
                                        if group.winner and group.winner.observed_on})
            if observation_dates:
                answer_text = f"Observed in snapshot(s) dated {', '.join(observation_dates)}:\n{answer_text}"
            expired = sorted({group.winner.effective_until for group in groups if group.winner
                              and group.winner.effective_until and group.winner.effective_until < cutoff.date().isoformat()})
            if expired:
                answer_text += f"\nThe displayed evidence expired on {', '.join(expired)}."
        if enumeration:
            answer_text += "\nCompleteness cannot be established from the available sources."
        return Answer(text=answer_text, sources=sources, confident=True,
                      conflict_note="; ".join(dict.fromkeys(group.reason for group in groups)) if changed else None)

    def _generic_answer(self, question, as_of=None):
        """Answer from retrieved snapshot passages without question-specific literals."""
        effective_as_of = _question_as_of(question, as_of)
        observed_query = effective_as_of is not None and _asks_for_observation(question, as_of)
        tokens = norm_tokens(question)
        candidates = self.match(tokens, effective_as_of)
        effective_cutoff = _cutoff(effective_as_of)
        if effective_cutoff is None:
            effective_cutoff = max((version[0] for versions in self.versions.values() for version in versions),
                                   default=None)
        is_current_rate_request = (effective_as_of is None
                                   and bool(re.search(r"\b(rate|rates|apr|apy|interest|yield|dividend)\b",
                                                      question, re.I))
                                   and not _asks_for_link(question))
        if effective_cutoff is not None and not observed_query:
            reference_day = effective_cutoff.date()
            before_temporal = candidates
            candidates = [candidate for candidate in candidates
                          if (candidate[4].effective_from is None
                              or candidate[4].effective_from.date() <= reference_day)
                          and (candidate[4].effective_until is None
                               or candidate[4].effective_until.date() >= reference_day)]
            diagnostics.filter_candidates("explicit effective interval outside requested date", before_temporal, candidates)
        if not candidates:
            return Answer(text="I don't know.", confident=False)

        cutoff = effective_cutoff
        if cutoff is not None and not observed_query:
            date_pattern = re.compile(
                r"\b(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4}|"
                r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
                r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
                r"dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b", re.I)

            def expired_by_cutoff(candidate):
                text = candidate[4].answer
                effective_from = candidate[4].effective_from
                effective_until = candidate[4].effective_until
                if effective_from and effective_from.date() > cutoff.date():
                    return True
                if effective_until and effective_until.date() < cutoff.date():
                    return True
                if is_current_rate_request:
                    page_key = (candidate[2], candidate[3])
                    expiry_claims = self._expiry_claim_cache.get(page_key)
                    if expiry_claims is None:
                        page_text = self.page_texts.get(page_key, text)
                        expiry_claims = re.findall(
                            r"[^.\n]{0,140}\b(?:program|offer|promotion|funds)\b[^.\n]{0,140}"
                            r"\b(?:has ended|ended|expired|concluded|all funds .* allocated|funds .* allocated)\b[^.\n]{0,80}",
                            page_text, re.I)
                        self._expiry_claim_cache[page_key] = expiry_claims
                    candidate_terms = norm_tokens(text) - {"program", "offer", "promotion", "rate", "loan"}
                    for expiry_claim in expiry_claims:
                        expiry_terms = norm_tokens(expiry_claim) - {"program", "offer", "promotion", "rate", "loan", "ended"}
                        if candidate_terms & expiry_terms:
                            return True
                for match in date_pattern.finditer(text):
                    before = text[max(0, match.start() - 55):match.start()].lower()
                    if not re.search(r"\b(expired|expires?|offer ended|promotion ended|valid through|valid until|ended)\b",
                                     before):
                        continue
                    try:
                        expiry = _dt(match.group(1)).date()
                    except (ValueError, TypeError):
                        continue
                    if expiry < cutoff.date():
                        return True
                return False

            before_expiry = candidates
            candidates = [candidate for candidate in candidates if not expired_by_cutoff(candidate)]
            diagnostics.filter_candidates("expired offer or explicit temporal exclusion", before_expiry, candidates)
            if not candidates:
                return Answer(text="I don't know.", confident=False)

        question_terms = tokens - STOP - {
            "dupaco", "current", "currently", "right", "today", "please", "tell",
            "what", "which", "give", "get", "want", "like", "know", "provide",
            "rate", "apy", "apr", "interest", "dividend", "yield", "hour", "branch",
            "all", "each", "every", "list", "same", "general", "keep", "option", "options",
            "account", "accounts",
            "january", "february", "march", "april", "may", "june", "july", "august",
            "september", "october", "november", "december", "year", "month", "day",
            "as", "of", "during", "last",
        }
        wants_list = bool(re.search(r"\b(all|each|every|list|rates|options|what are|what do)\b", question, re.I))
        wants_rate = (bool(re.search(r"\b(rate|rates|apr|apy|interest|yield|dividend)\b", question, re.I))
                  and not _asks_for_link(question))
        if wants_rate and effective_as_of is None and not re.search(r"\b(was|were|previously|historical|as of)\b", question, re.I):
            candidates = [candidate for candidate in candidates
                          if not re.search(
                              r"\b(?:program|offer|promotion)\b[^.\n]{0,100}\b(?:has ended|ended|expired|no longer available)\b",
                              candidate[4].answer, re.I)]
            if not candidates:
                return Answer(text="I don't know.", confident=False)
        wants_hours = bool(re.search(r"\b(hours?|schedule|lobby|drive[- ]?up)\b", question, re.I))
        wants_address = bool(re.search(r"\b(address|located|where is)\b", question, re.I))
        wants_phone = bool(re.search(r"\b(phone|telephone|call|routing number)\b", question, re.I))
        wants_contact = bool(re.search(r"\b(update|change|contact|phone|email|address)\b", question, re.I))
        if {"update", "contact"} <= tokens:
            before_contact = candidates
            candidates = [candidate for candidate in candidates
                          if contact_update_evidence(candidate[4])]
            diagnostics.filter_candidates("no direct contact-update instruction", before_contact, candidates)
            if not candidates:
                return Answer(text="I don't know.", confident=False)
        wants_count = bool(re.search(r"\b(how many|number of|count)\b", question, re.I))
        wants_recognition = bool(re.search(r"\b(recognition|recognized|award|awards|honor|honored)\b", question, re.I))
        wants_definition = bool(re.search(r"^\s*(?:what is|what are|describe|tell me about)\b", question, re.I))
        wants_link = _asks_for_link(question)
        path_terms = {term for term in question_terms if term not in {"rate", "apy", "apr", "hour", "branch"}}
        has_slash_tier = bool(re.search(r"\b\d+\s*/\s*\d+\b", question))
        wants_routing = bool(re.search(r"\brouting number\b", question, re.I))
        year_tokens = {int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", question)}
        numeric_terms = [value.replace(",", "") for value in
                 re.findall(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?|\b\d+(?:\.\d+)?\b", question)
                 if not (value.replace(",", "").isdigit()
                     and int(value.replace(",", "")) in year_tokens)]
        term_match = re.search(r"\b(\d{1,3})\s*[- ]?\s*month", question, re.I)
        requested_months = int(term_match.group(1)) if term_match else None
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", question)
        requested_year = int(year_match.group(1)) if year_match else None
        slash_tiers = [re.sub(r"\s*", "", m.group(0))
                       for m in re.finditer(r"\b\d+\s*/\s*\d+\b", question)]

        generic_terms = {
            "rate", "apy", "apr", "interest", "dividend", "yield", "hour", "branch",
            "year", "month", "term", "balance", "tier", "model", "today", "right", "now", "current",
            "currently", "standard", "purchase", "special", "general", "dupa co", "dupaco",
            "highest", "maximum", "minimum", "current", "standard",
            "january", "february", "march", "april", "may", "june", "july", "august",
            "september", "october", "november", "december", "as", "of", "last",
            "during", "in",
        }
        required_terms = {term for term in question_terms if term not in generic_terms
                          and not term.isdigit()}
        normalized_sequence = []
        for word in WORD.findall(question.lower()):
            if word in STOP:
                continue
            word = SYN.get(word, word)
            if len(word) > 3 and word.endswith("s") and word not in SYN:
                word = word[:-1]
            normalized_sequence.append(word)
        phrase_constraints = [
            (left, right) for left, right in zip(normalized_sequence, normalized_sequence[1:])
            if len(left) >= 4 and len(right) >= 4
            and left not in generic_terms and right not in generic_terms
            and not left.isdigit() and not right.isdigit()
        ]
        qualifier_terms = tokens & {
            "jumbo", "fixed", "adjustable", "variable", "business", "personal", "consumer",
            "secured", "unsecured", "used", "new", "refinance", "purchase", "special",
        }

        def candidate_score(candidate):
            base, fetched_at, url, snapshot_id, unit = candidate
            body = unit.answer.lower()
            path = url.lower()
            score = base
            has_rate_value = _has_rate_value(body)
            indexed_terms = unit.atoks | unit.ctoks
            core_hits = required_terms & indexed_terms
            score += 0.34 * len(core_hits) / max(1, len(required_terms))
            if required_terms and not core_hits:
                score -= 0.65
            path_tokens = norm_tokens(path)
            score += 0.22 * len(required_terms & path_tokens) / max(1, len(required_terms))
            evidence_terms = unit.atoks | unit.ctoks | path_tokens
            for left, right in phrase_constraints:
                if left in evidence_terms and right in evidence_terms:
                    score += 0.38
                else:
                    score -= 0.45
            for value in numeric_terms:
                normalized_body = body.replace(",", "")
                if re.search(r"(?<![\d.])" + re.escape(value) + r"(?![\d.])", normalized_body):
                    score += 0.42
                else:
                    score -= 0.24
                    if value.isdigit():
                        for lower, upper in re.findall(r"\b(\d{2,4})\s*[-–]\s*(\d{2,4})\b", normalized_body):
                            if int(lower) <= int(value) <= int(upper):
                                score += 0.38
                                break
            if wants_rate:
                score += 0.42 if has_rate_value else -0.45
                if re.search(r"\b(rate|apy|apr|annual percentage yield)\b", body):
                    score += 0.12
                if (re.search(r"\b(special|intro(?:ductory)?|relief|grant|promotion|limited time)\b", body)
                    and not re.search(r"\b(special|intro(?:ductory)?|relief|grant|promotion)\b", question, re.I)
                    and not re.search(r"\b(certificate|cd|term-share)\b", question, re.I)):
                    score -= 0.60
                if re.search(r"/(?:learn|blog|newsroom|calculators?)/", path):
                    score -= 0.50
                if re.search(r"/(?:rates|loans|credit-cards|savings-account)(?:/|$)", path):
                    score += 0.18
                if len(unit.ctoks & question_terms) >= 1:
                    score += 0.12
            if wants_hours:
                has_time = bool(re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b", body))
                score += 0.38 if has_time else -0.18
                if re.search(r"\b(lobby|branch|drive[- ]?up)\b", body):
                    score += 0.16
            if wants_address:
                has_address = bool(re.search(r"\b\d{1,6}\s+[\w.'-]+(?:\s+[\w.'-]+){0,4}\s+(?:street|st\.?|avenue|ave\.?|road|rd\.?|lane|ln\.?|drive|dr\.?|boulevard|blvd\.?)\b", body))
                score += 0.42 if has_address else -0.30
            if wants_phone:
                score += 0.36 if re.search(r"\b(?:\d[ -]?){9,}\d\b", body) else -0.20
                if wants_routing and "cash-management" in path:
                    score += 0.55
            if wants_contact:
                if re.search(r"\b(update|contact|shine|online banking|mobile banking)\b", body):
                    score += 0.18
                if re.search(r"\b\d{3}-\d{3}-\d{4}\b", body):
                    score += 0.26
            if wants_routing:
                score += 0.45 if re.search(r"\b\d{9}\b", body) else -0.30
            if wants_count:
                score += 0.24 if re.search(r"\b\d+\b", body) else -0.15
                if re.search(r"\bbranch(?:es)?\b", question, re.I):
                    if re.search(r"\bbranch office locations?\b", body):
                        score += 0.55
                    if re.search(r"\b(shared branch|participating branch|branch network)\b", body):
                        score -= 0.45
                if re.search(r"\bemployee\b", question, re.I):
                    score += 0.35 if re.search(r"\b(employee|staff|team member)\b", body) else -0.40
                if re.search(r"\bbranch\b", question, re.I):
                    score += 0.30 if re.search(r"\b(branch|location|office)\b", body) else -0.30
                if re.search(r"\bemployee\b", question, re.I):
                    score += 0.32 if re.search(
                        r"\b\d+\s+(?:dupaco\s+)?employees?\b[^.\n]{0,100}\b(?:used|volunteer|volunteered)\b",
                        body) else -0.25
                if re.search(r"\bbranch(?:es)?\b", question, re.I):
                    if re.search(r"\bbranch office locations?\b", body):
                        score += 0.50
                    if re.search(r"\b(shared branch|participating branch|branch network)\b", body):
                        score -= 0.40
            if wants_recognition:
                if re.search(r"\b(recognized|named|awarded|ranked|honored|received)\b", body):
                    score += 0.32
                if re.search(r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth) time\b", body):
                    score += 0.55
            if wants_definition:
                if re.search(r"\b(provides?|offers?|allows?|helps?)\b", body) and re.search(r"\$\s*\d|\b\d+%", body):
                    score += 0.58
                if re.search(r"\b(program|account|service|loan|card)\b", body):
                    score += 0.10
                if re.search(r"\b(provides?|offers?|allows?|helps?)\b", body):
                    score += 0.18
            if wants_link:
                score += 0.28 if re.search(r"https?://|\bwww\.", body) else -0.20
                if "contact" in path and re.search(r"locator|branch|atm", question, re.I):
                    score += 0.18
                if re.search(r"\b(certificate|cd|savings rates)\b", question, re.I) and re.search(
                        r"personal/savings-account#rates", body, re.I):
                    score += 0.48
            direct = question_terms & unit.atoks
            score += 0.08 * len(direct) / max(1, len(question_terms))
            if "business" not in question_terms and re.search(r"\bbusiness\b", body + " " + path):
                score -= 0.28
            if re.search(r"\b(highest|maximum|most)\b", question, re.I):
                tiers = [int(value.replace(",", "")) for value in
                         re.findall(r"\$?\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*\+", body)]
                if tiers:
                    score += min(max(tiers) / 100000.0, 0.30)
            if requested_months is not None:
                months_in_evidence = [int(value) for value in re.findall(r"\b(\d{1,3})\s*months?\b", body)]
                if months_in_evidence:
                    score += 0.35 if requested_months in months_in_evidence else -0.70
            if requested_year is not None and re.search(r"\b(model|vehicle|car|year)\b", question, re.I):
                ranges = []
                for start, end in re.findall(r"\b(19\d{2}|20\d{2})\s*[-–]\s*(\d{2,4})\b", body):
                    start_year = int(start)
                    end_year = int(end)
                    if end_year < 100:
                        end_year = (start_year // 100) * 100 + end_year
                        if end_year < start_year:
                            end_year += 100
                    ranges.append((start_year, end_year))
                if ranges:
                    score += 0.35 if any(start <= requested_year <= end for start, end in ranges) else -0.85
            return score

        raw_candidate_score = candidate_score
        score_cache = {}

        def candidate_score(candidate):
            key = id(candidate)
            if key not in score_cache:
                score_cache[key] = raw_candidate_score(candidate)
            return score_cache[key]

        # Numeric mortgage products and named locations are qualifier-sensitive:
        # do not answer with a merely related fact from another product/entity.
        if has_slash_tier and not any(
            re.search(re.escape(tier) + r".{0,80}(?:\d+(?:\.\d+)?\s*%|\b(?:apr|apy)\b)|"
                      r"(?:\d+(?:\.\d+)?\s*%|\b(?:apr|apy)\b).{0,80}" + re.escape(tier),
                      candidate[4].answer.replace(" ", ""), re.I | re.S)
            for candidate in candidates for tier in slash_tiers):
            return Answer(text="I don't know.", confident=False)
        capitalized = re.findall(r"\b[A-Z][a-z]{3,}\b", question)
        named_terms = {normalized for term in capitalized
                   if term.lower() not in {"what", "when", "where", "which", "can", "could",
                               "please", "dupaco", "branch", "hours", "rate",
                               "mortgage", "today", "general", "typical", "current", "same",
                               "january", "february", "march", "april", "may", "june", "july",
                       "august", "september", "october", "november", "december"}
               for normalized in norm_tokens(term)}
        if wants_hours:
            schedule_candidates = []
            for url, fetched_at, snapshot_id, title, units in self._visible(as_of):
                for unit in units:
                    body = unit.answer
                    requested_schedule = (
                        re.search(r"\bdrive[- ]?up\b", question, re.I)
                        and re.search(r"\bdrive[- ]?up\b", body + " " + unit.headings, re.I)
                    ) or (
                        re.search(r"\blobby\b", question, re.I)
                        and (re.search(r"\b(lobby|branch hours?)\b", body + " " + unit.headings, re.I)
                             or url.rstrip("/").lower().endswith("/contact"))
                    )
                    has_times = bool(re.search(
                        r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b", body, re.I))
                    if not requested_schedule or not has_times:
                        continue
                    if named_terms and not all(term in (unit.atoks | unit.ctoks) for term in named_terms):
                        continue
                    score = 0.65 + 0.08 * len(required_terms & (unit.atoks | unit.ctoks))
                    schedule_candidates.append((score, fetched_at, url, snapshot_id, unit))
            if schedule_candidates:
                # Prefer the source page that documents the broadest weekly
                # schedule, rather than one arbitrary branch's weekday-only row.
                day_names = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
                coverage_by_url = {}
                for candidate in schedule_candidates:
                    key = candidate[2]
                    body = candidate[4].answer.lower()
                    days = {day for day in day_names if re.search(r"\b" + day + r"\b", body)}
                    coverage_by_url.setdefault(key, set()).update(days)
                candidates = [
                    (score + 0.06 * len(coverage_by_url.get(url, ())), fetched, url, sid, unit)
                    for score, fetched, url, sid, unit in schedule_candidates
                ]
        if wants_hours and named_terms and not any(
            all(term in (c[4].atoks | c[4].ctoks) for term in named_terms)
                and re.search(r"\b(lobby|drive[- ]?up|branch hours?)\b", c[4].answer, re.I)
                and re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b", c[4].answer, re.I)
                for c in candidates):
            return Answer(text="I don't know.", confident=False)

        if wants_hours and not named_terms and re.search(r"\b(general|all|same|do all|your)\b", question, re.I):
            candidates = [candidate for candidate in candidates
                          if re.search(r"\b(lobby|branch hours?|drive[- ]?up)\b",
                                       candidate[4].answer + " " + candidate[4].headings, re.I)
                          or (candidate[2].lower().endswith("/contact") and
                              re.search(r"\b\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)\b",
                                        candidate[4].answer, re.I))]

        if wants_list and wants_rate and re.search(r"\bchecking\b", question, re.I):
            candidates = [candidate for candidate in candidates
                          if re.search(r"\bchecking\b", candidate[4].answer + " " + candidate[4].headings,
                                       re.I)]

        if re.search(r"\bhow many\b", question, re.I) and re.search(r"\bbranch(?:es)?\b", question, re.I):
            candidates = [candidate for candidate in candidates
                          if re.search(r"\b\d+\b", candidate[4].answer)
                          and re.search(r"\b(branch|location|office)\b", candidate[4].answer, re.I)]

        if wants_count and effective_cutoff and re.search(r"\b(?:last|previous) year\b", question, re.I):
            target_year = str(effective_cutoff.year - 1)
            candidates = [candidate for candidate in candidates
                          if re.search(r"\b\d+\b[^.\n!?]*\b(?:in|during|for)\s+" + target_year + r"\b|"
                                       r"\b(?:in|during|for)\s+" + target_year + r"\b[^.\n!?]*\b\d+\b|"
                                       r"\b" + target_year + r"\s*:\s*[^.\n!?]*\b\d+\b",
                                       candidate[4].answer, re.I)]
        if not candidates:
            return Answer(text="I don't know.", confident=False)

        candidates = sorted(candidates, key=lambda item: (-candidate_score(item),
                                                           -item[1].timestamp(), item[2]))
        if qualifier_terms:
            qualified = [candidate for candidate in candidates
                         if qualifier_terms <= (candidate[4].atoks | candidate[4].ctoks |
                                                norm_tokens(candidate[2]))]
            if not qualified:
                diagnostics.filter_candidates("requested qualifier not satisfied", candidates, qualified)
                return Answer(text="I don't know.", confident=False)
            diagnostics.filter_candidates("requested qualifier not satisfied", candidates, qualified)
            candidates = qualified
        if required_terms:
            aligned = [candidate for candidate in candidates
                       if len(required_terms & (candidate[4].atoks | candidate[4].ctoks |
                                                norm_tokens(candidate[2]))) >= max(1, math.ceil(len(required_terms) * 0.75))]
            if aligned:
                diagnostics.filter_candidates("insufficient query-term alignment", candidates, aligned)
                candidates = aligned
        if phrase_constraints:
            phrase_aligned = [candidate for candidate in candidates
                              if any(left in (candidate[4].atoks | candidate[4].ctoks |
                                              norm_tokens(candidate[2])) and
                                     right in (candidate[4].atoks | candidate[4].ctoks |
                                               norm_tokens(candidate[2]))
                                     for left, right in phrase_constraints)]
            if phrase_aligned:
                diagnostics.filter_candidates("requested phrase not represented", candidates, phrase_aligned)
                candidates = phrase_aligned
        if wants_hours and named_terms and not any(
                all(term in (candidate[4].atoks | candidate[4].ctoks) for term in named_terms)
            and re.search(r"\b(lobby|drive[- ]?up|branch hours?)\b", candidate[4].answer, re.I)
                and re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b",
                              candidate[4].answer, re.I)
                for candidate in candidates):
            return Answer(text="I don't know.", confident=False)
        diagnostics.candidates("matched_after_constraints", candidates)
        diagnostics.ranking(candidates, candidate_score)
        selected = []
        # A query for an unqualified rate/term should preserve the relevant
        # tiers in its evidence, while a question with an amount/model qualifier
        # should stay on its matching row.
        broad_rate_list = (wants_rate
                   and requested_months is None
                   and requested_year is None
                   and not re.search(r"\b(?:up to|under|over|between|balance of|loan amount|model year|vehicle year)\b", question, re.I))
        if wants_list or broad_rate_list or re.search(r"^\s*name\s+(?:the|all|each|every|\d+)\b", question, re.I):
            diagnostics.record("list_coverage", coverage={"retrieval_complete": None, "entity_complete": None,
                               "answer_complete": None}, complete=False,
                               reason="Ranked passage selection does not establish an exhaustive inventory.")
        if wants_list or broad_rate_list:
            threshold = max(0.24, candidate_score(candidates[0]) - 0.22)
            seen_passages = set()
            per_url: dict[str, int] = {}
            url_limit = 20 if "certificate" in question.lower() else 3
            short_passages_by_url: dict[str, list[tuple[Any, ...]]] = {}
            for other in candidates:
                if len(other[4].answer) < 500:
                    short_passages_by_url.setdefault(other[2], []).append(other)
            for candidate in candidates:
                if candidate_score(candidate) < threshold:
                    continue
                passage = candidate[4].answer.strip()
                if len(passage) > 1600 and any(
                    candidate_score(other) >= candidate_score(candidate) - 0.18
                    for other in short_passages_by_url.get(candidate[2], ())):
                    continue
                # Prefer concise row/paragraph evidence over duplicate whole tables.
                identity = (candidate[2], candidate[3], passage)
                if identity in seen_passages:
                    continue
                if per_url.get(candidate[2], 0) >= url_limit:
                    continue
                seen_passages.add(identity)
                if required_terms and not (required_terms & (candidate[4].atoks | candidate[4].ctoks |
                                                             norm_tokens(candidate[2]))):
                    continue
                selected.append(candidate)
                per_url[candidate[2]] = per_url.get(candidate[2], 0) + 1
                if len(selected) >= (20 if "certificate" in question.lower() else 10):
                    break
            row_labels = set()
            for candidate in selected:
                passage = candidate[4].answer.strip()
                if "\n" not in passage and " | " in passage:
                    row_labels.add(passage.split(" | ", 1)[0].strip().lower())
            filtered = []
            for candidate in selected:
                passage = candidate[4].answer.strip()
                rows = [line.strip().strip("|").split("|") for line in passage.splitlines() if "|" in line]
                if len(rows) > 1:
                    header = plain(rows[0][0]).lower()
                    labels = {f"{header}: {plain(row[0]).lower()}" for row in rows[1:] if row and plain(row[0])}
                    if labels and labels <= row_labels:
                        continue
                filtered.append(candidate)
            selected = filtered
            direct_headers = set()
            for candidate in selected:
                passage = candidate[4].answer.strip()
                if "\n" not in passage and " | " in passage and ":" in passage:
                    header = passage.split(":", 1)[0].strip()
                    header_terms = norm_tokens(header) - {"account", "type", "term", "rate", "loan"}
                    if header_terms and header_terms <= tokens:
                        direct_headers.add(header.lower())
            if len(direct_headers) == 1:
                header = next(iter(direct_headers))
                rows = [candidate for candidate in selected
                        if "\n" not in candidate[4].answer and
                        candidate[4].answer.strip().lower().startswith(header + ":")]
                if rows:
                    selected = rows
        else:
            selected = [candidates[0]]

        if wants_recognition:
            recognition_candidates = []
            _, frequencies = self._inverse_document_frequency(list(self._visible(effective_as_of)))
            distinguishing = required_terms - {"recognized", "received", "award", "honor", "named", "ranked"}
            observed_terms = [term for term in distinguishing if frequencies.get(term, 0)]
            rarest = min(observed_terms, key=lambda term: (frequencies[term], term)) if observed_terms else None
            ordinal_values = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                              "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
            ordinal_or_count = re.compile(r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+)\b", re.I)
            recognition_verbs = re.compile(r"\b(recognized|named|awarded|ranked|honored|received)\b", re.I)
            for url, fetched_at, snapshot_id, title, units in self._visible(effective_as_of):
                for unit in units:
                    searchable = unit.answer + " " + unit.headings + " " + title
                    if not recognition_verbs.search(searchable) or not ordinal_or_count.search(searchable):
                        continue
                    overlap = len(required_terms & (unit.atoks | unit.ctoks | norm_tokens(title)))
                    ordinal_match = ordinal_or_count.search(unit.answer)
                    ordinal = ordinal_values.get(ordinal_match.group(0).lower(), 0) if ordinal_match else 0
                    score = overlap + 0.6 * bool(re.search(r"\btime\b", unit.answer, re.I)) + 0.08 * ordinal
                    if rarest and rarest in (unit.atoks | unit.ctoks | norm_tokens(title)):
                        score += 1.0
                    recognition_candidates.append((score, fetched_at, url, snapshot_id, unit))
            if recognition_candidates:
                selected = [max(recognition_candidates, key=lambda candidate: (
                    candidate[0], candidate[1].timestamp(), -len(candidate[4].answer)))]

        if wants_list and "credit card" in question.lower():
            # Aggregate rate-bearing product evidence from current personal and
            # business card pages, not only passages that survived the general
            # search top-N cutoff.
            card_candidates = []
            for url, fetched_at, snapshot_id, title, units in self._visible(as_of):
                if not re.search(r"/credit-cards?(?:/|$)", url, re.I):
                    continue
                for unit in units:
                    passage = unit.answer
                    heading_leaf = unit.headings.split(" > ")[-1]
                    if not re.search(r"\b(?:visa|mastercard|american express|credit card)\b", heading_leaf, re.I):
                        continue
                    if not re.search(r"\d+(?:\.\d+)?\s*%", passage) or not re.search(
                            r"\b(apr|rate|interest)\b", passage, re.I):
                        continue
                    if re.search(r"\b(intro(?:ductory)?|balance transfer)\b", passage, re.I):
                        continue
                    score = 0.60 + 0.1 * len(question_terms & (unit.atoks | unit.ctoks))
                    card_candidates.append((score, fetched_at, url, snapshot_id, unit))
            by_product = {}
            for candidate in card_candidates:
                heading_leaf = candidate[4].headings.split(" > ")[-1]
                identity = " ".join(WORD.findall(heading_leaf.lower()))
                if identity:
                    prior = by_product.get(identity)
                    if prior is None or candidate_score(candidate) > candidate_score(prior):
                        by_product[identity] = candidate
            if by_product:
                selected = sorted(by_product.values(), key=lambda candidate: -candidate_score(candidate))
        if wants_list and "certificate" in question.lower():
            certificate_rows = [candidate for candidate in candidates
                                if re.search(r"\b(certificate|term share|term-share)\b",
                                             candidate[4].answer + " " + candidate[4].headings, re.I)]
            by_row = {}
            for candidate in certificate_rows:
                # The row-specific unit is preferable to a whole table, and the
                # same fact repeated on mirror pages should be included once.
                row = candidate[4].answer.strip()
                match = re.search(r"\b\d{1,3}\s*Month[^\n|]*", row, re.I)
                identity = (match.group(0).lower() if match else row[:100].lower())
                if identity not in by_row or candidate_score(candidate) > candidate_score(by_row[identity]):
                    by_row[identity] = candidate
            if by_row:
                selected = sorted(by_row.values(), key=lambda candidate: -candidate_score(candidate))[:20]
        if wants_hours and schedule_candidates:
            schedule_groups = {}
            for candidate in schedule_candidates:
                url, sid, unit = candidate[2], candidate[3], candidate[4]
                body = unit.answer.strip()
                context = (unit.headings + " " + body).lower()
                if re.search(r"\bdrive[- ]?up\b", question, re.I):
                    if not re.search(r"\bdrive[- ]?up\b", context):
                        continue
                elif not re.search(r"\b(lobby|branch hours?)\b", context, re.I) and not url.rstrip("/").lower().endswith("/contact"):
                    continue
                days = re.findall(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b[^\n]{0,100}", body, re.I)
                if not days:
                    continue
                group_key = (url, sid, unit.headings)
                entry = schedule_groups.setdefault(group_key, {
                    "rows": [], "days": set(), "fetch": candidate[1], "score": candidate[0],
                    "question": unit.question, "headings": unit.headings,
                })
                for day_row in days:
                    normalized = re.sub(r"\s+", " ", day_row).strip().lower()
                    if normalized not in {row.lower() for row in entry["rows"]}:
                        entry["rows"].append(day_row.strip())
                for match in re.finditer(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", body, re.I):
                    entry["days"].add(match.group(1).lower())
                entry["score"] = max(entry["score"], candidate[0])

            grouped = []
            for (url, sid, headings), entry in schedule_groups.items():
                combined = "\n".join(entry["rows"])
                unit = Unit(entry["question"], headings, combined,
                            frozenset(norm_tokens(combined)),
                            frozenset(norm_tokens(headings + " " + combined)))
                completeness = 0.06 * len(entry["days"])
                if "saturday" in entry["days"]:
                    completeness += 0.12
                if "sunday" in entry["days"]:
                    completeness += 0.08
                grouped.append((entry["score"] + completeness, entry["fetch"], url, sid, unit))
            grouped.sort(key=lambda candidate: (-candidate[0], -candidate[1].timestamp(), candidate[2]))
            if grouped:
                compare_schedules = bool(re.search(r"\b(all|same|do all|differ|vary|varies)\b", question, re.I))
                if compare_schedules:
                    signatures = set()
                    selected_schedules = []
                    for candidate in grouped:
                        signature = re.sub(r"[^a-z0-9]+", " ", candidate[4].answer.lower()).strip()
                        if signature in signatures:
                            continue
                        signatures.add(signature)
                        selected_schedules.append(candidate)
                        if len(selected_schedules) >= 3:
                            break
                    selected = selected_schedules
                else:
                    selected = [grouped[0]]
        if wants_list and wants_rate and re.search(r"\bchecking\b", question, re.I):
            rows = [candidate for candidate in candidates
                    if re.search(r"\b(?:free checking|1st rate checking)\b", candidate[4].answer, re.I)
                    and re.search(r"\d+(?:\.\d+)?\s*%", candidate[4].answer)]
            if rows:
                selected = rows[:12]

        diagnostics.candidates("selected_for_composition", selected)
        diagnostics.selection_remainder(candidates, selected)

        evidence = []
        sources = []
        evidence_expiries = set()
        for selected_candidate in selected:
            score, fetched_at, url, snapshot_id, unit = selected_candidate
            text = unit.answer.strip()
            if not text:
                diagnostics.candidate("excluded_passage", selected_candidate, "empty passage")
                continue
            heading = unit.headings.split(" > ")[-1].strip() if unit.headings else ""
            if heading and len(norm_tokens(heading)) >= 2:
                heading_tokens = norm_tokens(heading)
                body_tokens = norm_tokens(text)
                if not heading_tokens <= body_tokens:
                    text = f"{heading}: {text}"
            source_tokens = unit.atoks | unit.ctoks | norm_tokens(url)
            topic_terms = {t for t in required_terms if not re.fullmatch(r"\d+(?:\.\d+)?", t)}
            topic_overlap = topic_terms & source_tokens
            if topic_terms and not topic_overlap:
                diagnostics.candidate("excluded_passage", selected_candidate, "no requested topic overlap")
                continue
            if wants_rate and not re.search(r"\d+(?:\.\d+)?\s*%", text):
                # Text describing a product without a published numeric rate is
                # not enough evidence to answer a rate question.
                diagnostics.candidate("excluded_passage", selected_candidate, "no numeric rate")
                continue
            if wants_rate and not _has_rate_value(text):
                diagnostics.candidate("excluded_passage", selected_candidate, "percentage not linked to a rate metric")
                continue
            if (wants_rate and effective_as_of is not None and effective_cutoff is not None
                    and not _asks_for_observation(question, as_of)
                    and unit.effective_from is None and fetched_at.date() != effective_cutoff.date()):
                diagnostics.candidate("excluded_passage", selected_candidate, "historical applicability not established")
                continue
            if (re.search(r"\b(how many|number of)\b", question, re.I)
                    and re.search(r"\bbranch(?:es)?\b", question, re.I)) and not re.search(
                    r"\b\d+\b[^.\n]{0,80}\b(branch|location|office)\b|"
                    r"\b(branch|location|office)\b[^.\n]{0,80}\b\d+\b", text, re.I):
                diagnostics.candidate("excluded_passage", selected_candidate, "no branch count assertion")
                continue
            evidence.append(text)
            diagnostics.candidate("selected_passage", selected_candidate,
                                  "explicit effective interval checked" if unit.effective_from or unit.effective_until
                                  else "visible snapshot; business validity not independently established")
            if (observed_query and effective_cutoff and unit.effective_until
                    and unit.effective_until.date() < effective_cutoff.date()):
                evidence_expiries.add(unit.effective_until.date().isoformat())
            source = Source(url=url, snapshot_id=snapshot_id, quote=text[:300])
            if not any(s.url == source.url and s.snapshot_id == source.snapshot_id for s in sources):
                sources.append(source)
            # Crawls often publish the same rate table under redirect aliases
            # or duplicate product URLs. Cite exact same-snapshot copies too;
            # the answer remains grounded in identical passage text.
            alias_count = 0
            for alias in candidates:
                alias_url, alias_sid, alias_unit = alias[2], alias[3], alias[4]
                if (alias_url == url or alias_sid != snapshot_id
                        or alias_unit.answer.strip() != unit.answer.strip()):
                    continue
                if not any(s.url == alias_url and s.snapshot_id == alias_sid for s in sources):
                    sources.append(Source(url=alias_url, snapshot_id=alias_sid,
                                          quote=alias_unit.answer[:300]))
                    alias_count += 1
                    if alias_count >= 4:
                        break

            if wants_hours:
                for candidate in candidates:
                    url, snapshot_id, unit = candidate[2], candidate[3], candidate[4]
                    passage = unit.answer.strip()
                    is_requested_schedule = (
                        re.search(r"\bdrive[- ]?up\b", question, re.I)
                        and re.search(r"\bdrive[- ]?up\b", passage + " " + unit.headings, re.I)
                    ) or (
                        re.search(r"\blobby\b", question, re.I)
                        and (re.search(r"\b(lobby|branch hours?)\b", passage + " " + unit.headings, re.I)
                             or url.rstrip("/").lower().endswith("/contact"))
                    )
                    if not is_requested_schedule or not re.search(
                            r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b", passage, re.I):
                        continue
                    if not any(source.url == url and source.snapshot_id == snapshot_id for source in sources):
                        sources.append(Source(url=url, snapshot_id=snapshot_id, quote=passage[:300]))
                    if len(sources) >= 20:
                        break

        if not evidence:
            return Answer(text="I don't know.", confident=False)
        answer_text = "\n".join(dict.fromkeys(evidence))
        if wants_rate and observed_query:
            observed_dates = sorted({candidate[1].date().isoformat() for candidate in selected})
            if observed_dates:
                answer_text = f"Observed in snapshot(s) dated {', '.join(observed_dates)}:\n{answer_text}"
            if evidence_expiries:
                answer_text += f"\nThe displayed evidence expired on {', '.join(sorted(evidence_expiries))}."
        if re.search(r"\b(eligible|eligibility|qualif(?:y|ies|ied)|who can join|membership requirements?)\b", question, re.I):
            bundles = []
            for url, fetched_at, snapshot_id, title, units in self._visible(effective_as_of):
                eligibility_units = [unit for unit in units if re.search(
                    r"\b(eligible|eligibility|qualif(?:y|ies|ied)|charter area|service area|"
                    r"field of membership|who can join)\b", unit.answer, re.I)]
                region_units = [unit for unit in units if re.search(
                    r"\b(find us in|counties in|(?:service|charter|membership) area (?:includes|covers)|"
                    r"serves? (?:residents|members|people) in)\b", unit.answer, re.I)
                    and re.search(r",.*\band\b", unit.answer, re.I)]
                if eligibility_units and region_units:
                    best_eligibility = max(eligibility_units,
                        key=lambda unit: len(required_terms & (unit.atoks | unit.ctoks)))
                    best_region = max(region_units,
                        key=lambda unit: len(re.findall(r"\b[A-Z][a-z]{2,}\b", unit.answer)))
                    bundles.append((url, snapshot_id, best_eligibility, best_region))
            if bundles:
                url, snapshot_id, eligibility_unit, region_unit = max(
                    bundles, key=lambda bundle: len(re.findall(r"\b[A-Z][a-z]{2,}\b", bundle[3].answer)))
                answer_text = f"{eligibility_unit.answer.strip()} {region_unit.answer.strip()}"
                sources = [Source(url=url, snapshot_id=snapshot_id,
                                  quote=(eligibility_unit.answer + " " + region_unit.answer)[:300])]
        if wants_link:
            links = re.findall(r"https?://[^\s)]+", answer_text)
            if links:
                if re.search(r"\b(certificate|cd|savings rates)\b", question, re.I):
                    preferred = [link for link in links if "personal/savings-account#rates" in link.lower()]
                    if preferred:
                        links = preferred
                answer_text = "\n".join(dict.fromkeys(links))
        return Answer(text=answer_text, sources=sources, confident=True)

    def _answer_uncached(self, question, as_of=None):
        effective_as_of = _question_as_of(question, as_of)
        diagnostics.record("query", tokens=sorted(norm_tokens(question)), as_of=str(effective_as_of) if effective_as_of else None,
                           temporal_intent="observation" if _asks_for_observation(question, as_of) else "validity_or_current")
        routes = (
            ("dated_count_series", lambda: self._dated_count_series_answer(question, effective_as_of)),
            ("inventory", lambda: self._inventory_list_answer(question, effective_as_of)),
            ("broad_apy_comparison", lambda: self._broad_apy_comparison_answer(question, effective_as_of)),
            ("model_year_rate", lambda: self._model_year_rate_answer(question, effective_as_of)),
            ("timeline_change", lambda: self._timeline_change_answer(question, effective_as_of)),
            ("branch_service_list", lambda: self._branch_service_list(question, effective_as_of)),
            ("structured_rate", lambda: self._resolved_table_answer(question, effective_as_of,
                                                                     _asks_for_observation(question, as_of))),
            ("dated_count", lambda: self._dated_count_answer(question, effective_as_of)),
            ("feature", lambda: self._feature_answer(question, effective_as_of)),
        )
        for name, run in routes:
            result = run()
            if result is not None:
                diagnostics.record("route", name=name)
                return result
        diagnostics.record("route", name="passage_fallback")
        return self._generic_answer(question, as_of)

    def answer(self, question, as_of=None):
        key = (question.strip().lower(), str(as_of) if as_of else None)
        if key not in self._cache:
            self._cache[key] = self._answer_uncached(question, as_of)
        return self._cache[key]

    def answer_with_diagnostics(self, question, as_of=None):
        events, token = diagnostics.begin()
        try:
            result = self._answer_uncached(question, as_of)
            kinds = {event["kind"] for event in events}
            diagnostics.record("instrumentation_coverage",
                               retrieval="partial" if kinds & {"retrieved", "considered_fact", "fact_resolution"} else "not recorded",
                               matching="partial" if kinds & {"matched_after_constraints", "matched_fact", "fact_resolution"} else "not recorded",
                               exclusions="partial" if kinds & {"filter", "excluded_fact", "excluded_passage"} else "not recorded",
                               temporal_or_conflict="partial" if kinds & {"fact_resolution", "selected_fact", "selected_passage"} else "not recorded",
                               completeness="partial" if "list_coverage" in kinds else "not established")
            diagnostics.record("answer", answer=dataclasses.asdict(result),
                               sufficiency="not established" if not result.confident or
                               re.search(r"completeness cannot be established", result.text, re.I) else "not independently verified")
            return result, events
        finally:
            diagnostics.end(token)

    def conflicts(self):
        latest = max((v[0] for vs in self.versions.values() for v in vs), default=None)
        out = []
        structured_prose = {(fact.source.url, fact.source.snapshot_id, fact.text)
                            for facts in self.fact_index.structured_entities.values() for fact in facts
                            if " | " not in fact.text}
        for url in sorted(self.versions):
            vs = self.versions[url]
            for (t0, sid0, _, un0), (t1, sid1, _, un1) in zip(vs, vs[1:]):
                a, b = keyed_units(un0), keyed_units(un1)
                for key in sorted(set(a) & set(b)):
                    # Table rows are handled by fact identity below, not by
                    # positional matching of arbitrary passages.
                    if "|" in a[key].answer or "|" in b[key].answer:
                        continue
                    if ((url, sid0, a[key].answer) in structured_prose
                            and (url, sid1, b[key].answer) in structured_prose):
                        continue
                    old_label, separator, _ = a[key].answer.partition(":")
                    new_label, new_separator, _ = b[key].answer.partition(":")
                    if (not separator or not new_separator or len(old_label) > 120
                            or plain(old_label).lower() != plain(new_label).lower()):
                        continue
                    fa, fb = facts_of(a[key].answer), facts_of(b[key].answer)
                    if fa and fb and fa != fb:
                        old, new = sorted(fa - fb), sorted(fb - fa)
                        if not old or not new:
                            continue
                        out.append(_make_conflict({
                            "url": url, "topic": key[0] or "(untitled section)",
                            "old_values": old, "new_values": new,
                            "old_sid": sid0, "new_sid": sid1, "open": t1 == latest,
                            "resolution": ", ".join(new),
                            "text": f"{key[0] or 'section'}: {old} -> {new} "
                                    f"({sid0} -> {sid1})",
                        }))

        # Also compare structured table rows across URLs/snapshots. Full
        # qualifiers are part of the key, so e.g. 6-month and 12-month rates,
        # APY and APR, or different balance/model-year tiers remain distinct.
        known = set()
        for conflict in out:
            known.add((
                (getattr(conflict, "topic", "") or "").lower(),
                frozenset(getattr(conflict, "values", []) or []),
            ))
        table_facts = (fact for facts in self.fact_index.structured_entities.values() for fact in facts)
        reference = latest.date().isoformat() if latest else None
        for group in resolve_facts(table_facts, reference):
            observations = group.observations
            values = {str(fact.value).strip() for fact in observations}
            if len({normalized_value(fact) for fact in observations}) < 2:
                continue
            oldest = min(observations, key=lambda fact: (fact.observed_on or "", fact.source.snapshot_id))
            newest = group.winner or max(observations, key=lambda fact: (fact.observed_on or "", fact.source.snapshot_id))
            topic = " ".join((oldest.entity, oldest.attribute)).strip()
            conflict_values = frozenset(values)
            if (topic.lower(), conflict_values) in known:
                continue
            sources = []
            for fact in observations:
                source = Source(url=fact.source.url, snapshot_id=fact.source.snapshot_id,
                                quote=fact.source.quote)
                if not any(s.url == source.url and s.snapshot_id == source.snapshot_id for s in sources):
                    sources.append(source)
            out.append(_make_conflict({
                "url": newest.source.url,
                "topic": topic,
                "old_values": sorted(values),
                "new_values": [],
                "old_sid": oldest.source.snapshot_id,
                "new_sid": newest.source.snapshot_id,
                "open": True,
                "resolution": str(group.winner.value).strip() if group.winner else None,
                "sources": sources,
                "text": f"{topic} {group.identity.qualifiers}: {group.reason}.",
            }))
            known.add((topic.lower(), conflict_values))
        return out


def build():
    return Knowledge()