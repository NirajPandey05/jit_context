"""Hybrid retrieval: semantic shortlist -> picker.

Stage 1 (semantic): embed the query, cosine-rank index entries, take top-K.
  Cheap, high recall, no LLM call.
Stage 2 (pick): from the K candidate *summaries* (never full text), the picker
  chooses which turn ids to load. Pickers live in `picker.py` (heuristic or LLM).
"""
from __future__ import annotations

from ..index.conversation_index import ConversationIndex
from ..index.embeddings import cosine
from .candidate import Candidate


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
