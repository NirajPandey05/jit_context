"""Factory + proxy glue.

`build_manager` wires the stack. By default it's fully offline (no keys). Pass
an `EmbedderSpec` / `PickerSpec` to select real, configurable backends without
touching the manager or the agent.

`ProxyHandler` is the drop-in integration point: an agent's chat-completions
request comes in, we rewrite `messages`, and forward to the real LLM. The agent
changes nothing but its base URL.
"""
from __future__ import annotations
from typing import Callable

from .config import JITConfig
from .manager import ContextManager, Report
from .index.conversation_index import HeuristicSummariser
from .index.embeddings import EmbedderSpec, make_embedder
from .retrieval.picker import PickerSpec, make_picker


def build_manager(config: JITConfig | None = None,
                  embedder_spec: EmbedderSpec | None = None,
                  picker_spec: PickerSpec | None = None,
                  summariser=None) -> ContextManager:
    config = config or JITConfig()
    embedder = make_embedder(embedder_spec or EmbedderSpec())   # local default
    picker = make_picker(picker_spec or PickerSpec())           # heuristic default
    summariser = summariser or HeuristicSummariser()
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
