"""Factory + proxy glue.

`build_manager` wires the offline-by-default stack (everything runs with no
network/keys). Swap any component for an API-backed one without touching the
manager.

`ProxyHandler` is the drop-in integration point: an agent's chat-completions
request comes in, we rewrite `messages`, and you forward to the real LLM. It's
transport-agnostic on purpose — wrap it in FastAPI/Flask, or call it directly
in tests. The agent changes nothing but its base URL.
"""
from __future__ import annotations
from typing import Callable

from .config import JITConfig
from .manager import ContextManager, Report
from .index.conversation_index import HeuristicSummariser
from .index.embeddings import LocalHashEmbedder
from .retrieval.retriever import HeuristicPicker


def build_manager(config: JITConfig | None = None,
                  summariser=None, embedder=None, picker=None) -> ContextManager:
    config = config or JITConfig()
    embedder = embedder or LocalHashEmbedder()
    summariser = summariser or HeuristicSummariser()
    picker = picker or HeuristicPicker()
    return ContextManager(config, summariser, embedder, picker)


class ProxyHandler:
    """Sits between the agent and the upstream LLM.

    `upstream` is any callable: (messages: list[dict]) -> response. In tests it
    can be a stub; in prod it's your real provider client call.
    """

    def __init__(self, manager: ContextManager,
                 upstream: Callable[[list[dict]], object]) -> None:
        self.manager = manager
        self.upstream = upstream
        self.last_report: Report | None = None

    def handle(self, request_messages: list[dict]):
        rewritten, report = self.manager.process(request_messages)
        self.last_report = report
        return self.upstream(rewritten)
