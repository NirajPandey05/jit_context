"""Pickers — the 'model picks' step. Offline default + configurable LLM backend.

Given the query and a semantic shortlist of candidate *summaries* (never full
text), choose which turn ids to actually load. This runs as one internal
sub-call inside the proxy, invisible to the agent.

- `HeuristicPicker`: offline, deterministic (top-N by score, decision boost).
- `LLMPicker`: provider-agnostic. Supports Anthropic (messages) and
  OpenAI-compatible (chat.completions) clients via a small adapter. Feeds the
  model summaries only, asks for a strict JSON id list, parses defensively, and
  FALLS BACK to the heuristic on any error/parse failure — a picker failure
  must never crash the proxy or silently drop everything.

Select via `make_picker(spec)`.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field

from .candidate import Candidate  # shared dataclass


class HeuristicPicker:
    """Offline picker: take top-N by score, give decisions a small boost."""

    def pick(self, query: str, candidates: list[Candidate], max_select: int) -> list[int]:
        def key(c: Candidate) -> float:
            return c.score + (0.05 if c.entry.has_decision else 0.0)
        ranked = sorted(candidates, key=key, reverse=True)
        return [c.entry.turn_id for c in ranked[:max_select]]


_PROMPT = """You are selecting which earlier conversation turns are needed to \
answer the user's current request. You are given the request and a numbered list \
of candidate turns (id + short summary). Choose ONLY the turns whose full detail \
is actually needed. Prefer fewer. Select at most {max_select}.

Current request:
{query}

Candidates:
{candidates}

Respond with ONLY a JSON array of integer turn ids, e.g. [12, 4]. No prose."""


def _format_candidates(candidates: list[Candidate]) -> str:
    lines = []
    for c in candidates:
        tag = " [decision]" if c.entry.has_decision else ""
        lines.append(f"- id={c.entry.turn_id}{tag}: {c.entry.summary}")
    return "\n".join(lines)


def _parse_ids(text: str, valid: set[int], max_select: int) -> list[int]:
    """Defensive parse: try JSON array first, then fall back to int scraping."""
    text = text.strip()
    ids: list[int] = []
    try:
        m = re.search(r"\[.*?\]", text, re.DOTALL)
        if m:
            ids = [int(x) for x in json.loads(m.group(0))]
    except Exception:
        ids = []
    if not ids:
        ids = [int(x) for x in re.findall(r"-?\d+", text)]
    seen, out = set(), []
    for i in ids:
        if i in valid and i not in seen:
            seen.add(i)
            out.append(i)
        if len(out) >= max_select:
            break
    return out


class LLMPicker:
    """Real model-picks step. `api` is one of: 'anthropic', 'openai'.

    Anthropic: client.messages.create(model, max_tokens, messages=[...])
               -> resp.content[0].text
    OpenAI:    client.chat.completions.create(model, messages=[...])
               -> resp.choices[0].message.content

    On ANY failure, falls back to the provided heuristic so the proxy degrades
    gracefully instead of crashing or returning nothing.
    """

    def __init__(self, client, model: str, api: str = "anthropic",
                 fallback: HeuristicPicker | None = None,
                 max_tokens: int = 128) -> None:
        self.client = client
        self.model = model
        self.api = api.lower()
        self.fallback = fallback or HeuristicPicker()
        self.max_tokens = max_tokens

    def _call(self, prompt: str) -> str:
        if self.api == "anthropic":
            resp = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        if self.api in ("openai", "openai_compatible"):
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        raise ValueError(f"unknown picker api: {self.api}")

    def pick(self, query: str, candidates: list[Candidate], max_select: int) -> list[int]:
        if not candidates:
            return []
        valid = {c.entry.turn_id for c in candidates}
        prompt = _PROMPT.format(query=query, max_select=max_select,
                                candidates=_format_candidates(candidates))
        try:
            ids = _parse_ids(self._call(prompt), valid, max_select)
            if ids:
                return ids
        except Exception:
            pass
        # Empty parse or error -> heuristic fallback.
        return self.fallback.pick(query, candidates, max_select)


@dataclass
class PickerSpec:
    provider: str = "heuristic"        # heuristic | llm
    api: str = "anthropic"             # anthropic | openai  (when provider=llm)
    model: str = ""
    client: object | None = None
    extra: dict = field(default_factory=dict)


def make_picker(spec: PickerSpec):
    p = spec.provider.lower()
    if p == "heuristic":
        return HeuristicPicker()
    if p == "llm":
        if spec.client is None:
            raise ValueError("provider 'llm' requires spec.client")
        return LLMPicker(spec.client, spec.model, api=spec.api, **spec.extra)
    raise ValueError(f"unknown picker provider: {spec.provider}")
