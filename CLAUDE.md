# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**jit-context** is a just-in-time context management proxy for LLM agents. It intercepts messages between an agent and an upstream LLM, rewrites the conversation to include only the most relevant turns (via hybrid semantic retrieval), and returns token savings metrics. The core comparison is three modes: `full` (baseline), `recency` (recent-N turns), and `jit` (index + hybrid retrieval).

## Running Tests

There is no pytest setup. Tests are standalone scripts run directly:

```bash
python eval/harness.py          # Three-way comparison (full vs recency vs jit), no LLM needed
python eval/scaling.py          # Token crossover curve vs conversation length
python eval/test_backends.py    # Mock OpenAI/Anthropic client tests, no network needed
python eval/locomo/smoke.py     # LoCoMo benchmark offline smoke test
```

To run the LoCoMo benchmark with real LLM clients:
```bash
python eval/locomo/runner.py    # Full benchmark (configure real clients first)
```

## No Build Step / No Package Install

There is no `pyproject.toml`, `setup.py`, or `requirements.txt`. Run scripts directly from the repo root. Python 3.10+ required. Core library has no dependencies; optional backends require:

- `openai` — for OpenAI/Azure/OpenRouter embeddings or picker
- `cohere` — for Cohere embeddings
- `voyageai` — for Voyage embeddings
- `anthropic` — for Anthropic-backed LLMPicker

## Architecture

### Four-Layer Stack

**1. Proxy & Config** (`proxy.py`, `config.py`)
- `ProxyHandler`: Drop-in integration that intercepts messages and calls `ContextManager.process()`
- `JITConfig`: Controls activation thresholds, max tokens, operating mode, recent-N window
- `build_manager()`: Factory wiring embedder + picker + summariser into the manager

**2. Hot Index** (`index/`)
- `ConversationIndex`: Compact per-turn entries (summary, entities, decision flag, embedding)
- `HeuristicSummariser`: Offline, deterministic — truncates text + extracts capitalized entities + detects decision keywords. Default.
- `LLMSummariser`: Placeholder for LLM-backed summaries
- All index entries stored in `IndexEntry` with turn_id, role, summary, entities, decision flag, and L2-normalized embedding

**3. Cold Store** (`store/cold_store.py`)
- `ColdStore`: In-memory dict of `Turn` objects keyed by turn_id. Full verbatim text fetched lazily only for retrieved turns.

**4. Hybrid Retrieval** (`retrieval/`)
- Stage 1 (`HybridRetriever`): Embed query → cosine-rank all index entries → top-K candidates
- Stage 2 (`HeuristicPicker` or `LLMPicker`): Select which candidates to actually load into context
  - `LLMPicker` sends summaries only to a model, asks for JSON id list, **always falls back to heuristic on any error**

### Core Pipeline in `manager.py`

`ContextManager.process(messages)` → `(rewritten_messages, Report)`:

1. Split messages into system + conversation turns
2. Check activation thresholds (turn count and token estimate). If below threshold or `mode="full"`: pass through unchanged.
3. `mode="recency"`: return system + recent-N turns verbatim
4. `mode="jit"`: index all turns → retrieve (embed + rank + pick) → assemble scoped index block + retrieved detail + recent turns + query

Returns a `Report` with: `mode`, `activated`, `n_input_turns`, `n_sent_turns`, `selected_turn_ids`, `token_savings_pct`.

### Pluggable Backends Pattern

Provider SDKs are **never hard-imported** at module level. Clients are injected via spec dataclasses:

```python
EmbedderSpec(provider="openai", model="text-embedding-3-small", client=oai_client)
PickerSpec(provider="anthropic", model="claude-haiku-4-5-20251001", client=anth_client)
```

Factory functions (`make_embedder`, `make_picker`) select backend by string from a registry. This means importing any module never requires credentials or installed packages.

### Graceful Degradation Invariant

`LLMPicker` failure (exception, JSON parse error, empty response, timeout) **must never crash the proxy or silently drop context**. It always falls back to `HeuristicPicker`. This is the critical reliability contract throughout the retrieval layer.

### Local Embedder Caveat

`LocalHashEmbedder` (the offline default) is a bag-of-words hash — not semantically meaningful. It achieves high recall in the synthetic harness because the needle shares rare tokens with the query. For real quality, configure `OpenAICompatibleEmbedder`, `CohereEmbedder`, or `VoyageEmbedder`.

## Integration Example

```python
from jitcontext import JITConfig, build_manager, ProxyHandler

proxy = ProxyHandler(
    build_manager(JITConfig(mode="jit")),
    upstream=lambda msgs: my_llm_client.chat(messages=msgs)
)
response = proxy.handle(agent_messages)
```

For eval/research (no upstream needed):
```python
manager = build_manager(JITConfig(mode="jit"))
rewritten, report = manager.process(messages)
# report.token_savings_pct, report.selected_turn_ids, etc.
```

## Public API

`jitcontext/__init__.py` exports: `JITConfig`, `ContextManager`, `Report`, `build_manager`, `ProxyHandler`.
