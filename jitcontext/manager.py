"""ContextManager — the core pipeline, provider-agnostic.

Operates on the standard chat-messages shape:
    [{"role": "system"|"user"|"assistant", "content": str}, ...]

It rewrites the message list the agent *intended* to send into the (possibly
smaller, denser) list actually sent upstream. The agent never knows.

Three modes (all the same pipeline, different config):
  full    -> return messages unchanged (baseline / control)
  recency -> keep system + only the most recent N turns (baseline)
  jit     -> index + hybrid retrieval + assembly (the system under test)

Returns both the rewritten messages and a `Report` for the eval harness.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .config import JITConfig
from .store.cold_store import ColdStore, Turn
from .index.conversation_index import ConversationIndex
from .retrieval.retriever import HybridRetriever


@dataclass
class Report:
    mode: str
    activated: bool
    n_input_turns: int
    n_sent_turns: int
    selected_turn_ids: list[int] = field(default_factory=list)
    recent_turn_ids: list[int] = field(default_factory=list)
    input_tokens_est: int = 0
    sent_tokens_est: int = 0

    @property
    def token_savings_pct(self) -> float:
        if self.input_tokens_est == 0:
            return 0.0
        return 100.0 * (1 - self.sent_tokens_est / self.input_tokens_est)


class ContextManager:
    def __init__(self, config: JITConfig, summariser, embedder, picker) -> None:
        self.cfg = config
        self.store = ColdStore()
        self.index = ConversationIndex(summariser, embedder)
        self.retriever = HybridRetriever(self.index, embedder, picker)
        self._indexed_ids: set[int] = set()

    # -- indexing (off the hot path in the proxy) ---------------------------
    def _ingest(self, turns: list[Turn]) -> None:
        for t in turns:
            if t.turn_id not in self._indexed_ids:
                self.store.put(t)
                self.index.add(t)
                self._indexed_ids.add(t.turn_id)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _split(messages: list[dict]) -> tuple[list[dict], list[Turn]]:
        system = [m for m in messages if m["role"] == "system"]
        convo = [m for m in messages if m["role"] != "system"]
        turns = [Turn(i, m["role"], m["content"]) for i, m in enumerate(convo)]
        return system, turns

    def _tok(self, messages: list[dict]) -> int:
        return sum(self.cfg.estimate_tokens(m["content"]) for m in messages)

    def _activated(self, turns: list[Turn]) -> bool:
        if len(turns) >= self.cfg.activation_turn_threshold:
            return True
        approx = sum(self.cfg.estimate_tokens(t.content) for t in turns)
        return approx >= self.cfg.activation_token_threshold

    # -- main entry ---------------------------------------------------------
    def process(self, messages: list[dict]) -> tuple[list[dict], Report]:
        system, turns = self._split(messages)
        cfg = self.cfg
        input_tokens = self._tok(messages)

        # Below threshold OR full mode -> pass through untouched.
        if cfg.mode == "full" or not self._activated(turns):
            rep = Report(cfg.mode, False, len(turns), len(turns),
                         input_tokens_est=input_tokens, sent_tokens_est=input_tokens)
            return messages, rep

        recent_n = cfg.recent_turns_verbatim
        recent = turns[-recent_n:] if recent_n else []
        recent_ids = [t.turn_id for t in recent]

        # --- recency baseline ---
        if cfg.mode == "recency":
            sent = system + [{"role": t.role, "content": t.content} for t in recent]
            st = self._tok(sent)
            return sent, Report("recency", True, len(turns), len(recent),
                                recent_turn_ids=recent_ids,
                                input_tokens_est=input_tokens, sent_tokens_est=st)

        # --- jit mode ---
        # Index everything except the very latest turn (the query) and recents.
        self._ingest(turns[:-1])
        query = turns[-1].content

        exclude = set(recent_ids) | {turns[-1].turn_id}
        selected_ids, shortlist_ids = self.retriever.retrieve(
            query=query,
            shortlist_k=cfg.shortlist_k,
            max_select=cfg.max_selected_turns,
            exclude_ids=exclude,
        )
        selected_turns = self.store.get_many(selected_ids)

        assembled: list[dict] = list(system)
        if cfg.always_inject_index:
            shown = set(selected_ids) | exclude
            # Scope the TOC to the shortlist (relevant) + decisions (important),
            # so the index doesn't grow 1:1 with history.
            toc = self.index.toc(exclude_ids=shown,
                                 include_ids=set(shortlist_ids))
            if toc:
                assembled.append({"role": "system", "content": toc})
        # Retrieved detail first (older), then recent verbatim, then the query.
        for t in selected_turns:
            assembled.append({"role": t.role,
                              "content": f"[retrieved #{t.turn_id}] {t.content}"})
        for t in recent:
            assembled.append({"role": t.role, "content": t.content})

        st = self._tok(assembled)
        n_sent = len(assembled) - len(system)
        return assembled, Report(
            "jit", True, len(turns), n_sent,
            selected_turn_ids=selected_ids, recent_turn_ids=recent_ids,
            input_tokens_est=input_tokens, sent_tokens_est=st,
        )
