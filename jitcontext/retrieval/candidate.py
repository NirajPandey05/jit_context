"""Shared Candidate dataclass (kept separate to avoid circular imports)."""
from __future__ import annotations
from dataclasses import dataclass

from ..index.conversation_index import IndexEntry


@dataclass
class Candidate:
    entry: IndexEntry
    score: float
