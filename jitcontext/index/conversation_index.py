"""The hot index — a navigable table-of-contents over the conversation.

Each entry: turn id, role, a short summary, lightweight extracted metadata, and
an embedding of the summary. This is what stays in context always; full text is
fetched lazily from the cold store by turn id.

Summarisation is pluggable. `HeuristicSummariser` is offline (truncate + cheap
keyword/entity extraction) so the pipeline runs without an LLM. Swap
`LLMSummariser` for quality. Summarisation runs OFF the hot path (after the
response returns) in the proxy, so its latency never blocks the agent.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

from ..store.cold_store import Turn

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")
# crude "entity" heuristic: Capitalised tokens and code-ish identifiers
_CAP = re.compile(r"\b[A-Z][a-zA-Z0-9_]+\b")
_DECISION_HINTS = ("decided", "agreed", "chose", "will use", "let's go with",
                   "final", "conclusion", "must", "should", "don't")


@dataclass
class IndexEntry:
    turn_id: int
    role: str
    summary: str
    entities: list[str] = field(default_factory=list)
    has_decision: bool = False
    embedding: list[float] | None = None

    def toc_line(self) -> str:
        tag = " [decision]" if self.has_decision else ""
        ents = f" ({', '.join(self.entities[:5])})" if self.entities else ""
        return f"  #{self.turn_id} {self.role}{tag}: {self.summary}{ents}"


class HeuristicSummariser:
    """Offline, dependency-free. Truncates and extracts cheap metadata."""

    def __init__(self, max_chars: int = 160) -> None:
        self.max_chars = max_chars

    def summarise(self, turn: Turn) -> IndexEntry:
        text = " ".join(turn.content.split())
        summary = text[: self.max_chars] + ("…" if len(text) > self.max_chars else "")
        entities = list(dict.fromkeys(_CAP.findall(turn.content)))[:8]
        low = turn.content.lower()
        has_decision = any(h in low for h in _DECISION_HINTS)
        return IndexEntry(
            turn_id=turn.turn_id,
            role=turn.role,
            summary=summary,
            entities=entities,
            has_decision=has_decision,
        )


class LLMSummariser:
    """Placeholder for an LLM-backed summariser (better summaries + metadata)."""

    def __init__(self, client, model: str) -> None:
        self.client = client
        self.model = model

    def summarise(self, turn: Turn) -> IndexEntry:  # pragma: no cover
        raise NotImplementedError("Plug your LLM client here.")


class ConversationIndex:
    def __init__(self, summariser, embedder) -> None:
        self.summariser = summariser
        self.embedder = embedder
        self.entries: dict[int, IndexEntry] = {}

    def add(self, turn: Turn) -> IndexEntry:
        entry = self.summariser.summarise(turn)
        entry.embedding = self.embedder.embed([entry.summary])[0]
        self.entries[turn.turn_id] = entry
        return entry

    def all_entries(self) -> list[IndexEntry]:
        return [self.entries[k] for k in sorted(self.entries)]

    def toc(self, exclude_ids: set[int] | None = None,
            include_ids: set[int] | None = None) -> str:
        """Render a table-of-contents.

        If `include_ids` is given, only those entries are listed (scoped TOC) —
        this keeps the index from growing 1:1 with history. Decision-flagged
        turns are always kept so important commitments stay visible.
        """
        exclude_ids = exclude_ids or set()
        entries = self.all_entries()
        if include_ids is not None:
            entries = [e for e in entries
                       if e.turn_id in include_ids or e.has_decision]
        lines = [e.toc_line() for e in entries if e.turn_id not in exclude_ids]
        if not lines:
            return ""
        return ("Conversation index (relevant/important earlier turns; "
                "pull full detail by #id):\n" + "\n".join(lines))
