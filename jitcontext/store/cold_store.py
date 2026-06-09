"""Cold store: full turn text lives here, keyed by turn id.

The hot path keeps only the compact index; payloads are fetched lazily by id.
In-memory dict for testing; swap for sqlite/redis without touching callers.
"""
from dataclasses import dataclass


@dataclass
class Turn:
    turn_id: int
    role: str          # "user" | "assistant"
    content: str


class ColdStore:
    def __init__(self) -> None:
        self._turns: dict[int, Turn] = {}

    def put(self, turn: Turn) -> None:
        self._turns[turn.turn_id] = turn

    def get(self, turn_id: int) -> Turn | None:
        return self._turns.get(turn_id)

    def get_many(self, turn_ids: list[int]) -> list[Turn]:
        out = [self._turns[t] for t in turn_ids if t in self._turns]
        return sorted(out, key=lambda t: t.turn_id)

    def __len__(self) -> int:
        return len(self._turns)
