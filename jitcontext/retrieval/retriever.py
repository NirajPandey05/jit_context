"""Hybrid retrieval: semantic shortlist -> picker.

Stage 1 (semantic): embed the query, cosine-rank index entries, take top-K.
  Cheap, high recall, no LLM call.
Stage 2 (pick): from the K candidate *summaries* (never full text), choose which
  turn ids to actually load. `HeuristicPicker` is offline (keeps the strongest
  semantic matches, boosts decisions). `LLMPicker` is the real "model picks"
  step — one small/fast internal sub-call inside the proxy, invisible to the
  agent. Feed it summaries only and cap K to control cost/latency.
"""
from __future__ import annotations
from dataclasses import dataclass

from ..index.conversation_index import ConversationIndex, IndexEntry
from ..index.embeddings import cosine


@dataclass
class Candidate:
    entry: IndexEntry
    score: float


class HeuristicPicker:
    """Offline picker: take top-N by score, give decisions a small boost."""

    def pick(self, query: str, candidates: list[Candidate], max_select: int) -> list[int]:
        def key(c: Candidate) -> float:
            return c.score + (0.05 if c.entry.has_decision else 0.0)
        ranked = sorted(candidates, key=key, reverse=True)
        return [c.entry.turn_id for c in ranked[:max_select]]


class LLMPicker:
    """Real model-picks step. Plug a small fast model here.

    Prompt it with the query + numbered candidate summaries; parse back the ids
    it deems necessary. Implement `pick` against your client.
    """

    def __init__(self, client, model: str) -> None:
        self.client = client
        self.model = model

    def pick(self, query: str, candidates: list[Candidate], max_select: int) -> list[int]:  # pragma: no cover
        raise NotImplementedError("Plug your picker model here.")


class HybridRetriever:
    def __init__(self, index: ConversationIndex, embedder, picker) -> None:
        self.index = index
        self.embedder = embedder
        self.picker = picker

    def retrieve(self, query: str, shortlist_k: int, max_select: int,
                 exclude_ids: set[int]) -> tuple[list[int], list[int]]:
        """Returns (selected_ids, shortlist_ids)."""
        entries = [e for e in self.index.all_entries()
                   if e.turn_id not in exclude_ids and e.embedding is not None]
        if not entries:
            return [], []
        qvec = self.embedder.embed([query])[0]
        scored = [Candidate(e, cosine(qvec, e.embedding)) for e in entries]
        scored.sort(key=lambda c: c.score, reverse=True)
        shortlist = scored[:shortlist_k]
        shortlist_ids = [c.entry.turn_id for c in shortlist]
        selected = self.picker.pick(query, shortlist, max_select)
        return selected, shortlist_ids
