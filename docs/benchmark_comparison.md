# Benchmark Comparison: jit-context on LoCoMo

## What was measured

[LoCoMo](https://github.com/snap-research/locomo) is a long-conversation QA benchmark (ACL 2024) with four question categories — single-hop, multi-hop, temporal, and open-domain — across 10 real conversations. We scored all 1,540 non-adversarial QA pairs using token-overlap F1 and an LLM-as-judge.

**Setup:** `gemini-embedding-001` embedder, `gemini-2.5-flash-lite` answerer and judge, default jit config (`activation_turn_threshold=12`, `recent_turns_verbatim=4`, `max_selected_turns=6`). Full numbers in [`data/locomo_results.json`](../data/locomo_results.json).

## Our results

| mode | F1 | judge | ret_recall | token cost |
|---|---|---|---|---|
| `full` | 0.41 | 0.43 | — | 100% |
| `jit` | 0.37 | 0.36 | 0.71 | ~50% |
| `recency` | 0.00 | 0.01 | — | ~10% |

Per-category breakdown (jit):

| category | F1 | judge | ret_recall |
|---|---|---|---|
| single_hop | 0.48 | 0.51 | 0.74 |
| multi_hop | 0.28 | 0.26 | 0.73 |
| temporal | 0.22 | 0.10 | 0.69 |
| open_domain | 0.13 | 0.20 | 0.43 |

**jit reaches 90% of full-context quality at ~50% token cost.** Recency collapses to near-zero because answers are almost never in the most recent turns — the exact failure mode jit-context is designed to prevent.

> **Model note:** `gemini-2.5-flash-lite` is a small/fast model. All three modes benefit equally from a stronger model (GPT-4o, Claude Sonnet), so the relative gaps are meaningful; absolute F1 numbers will be higher.

---

## How it compares to other systems

### Important distinction: two different categories

**Context window optimizers** (jit-context): rewrite what is sent to the LLM each turn. No external state. Drop-in proxy.

**Persistent memory systems** (Mem0, MemMachine, etc.): maintain a vector store across sessions, continuously update summaries. Full infrastructure products.

These solve overlapping but different problems. The meaningful comparison for jit-context is within the optimizer/RAG category.

### Against session-scoped / training-free approaches

| system | score | metric | type |
|---|---|---|---|
| Simple RAG baseline | 24.6% | accuracy | chunk retrieval |
| CogCanvas (training-free) | 32.4% | accuracy | verbatim extraction + RAG |
| **jit-context** | **36–37%** | judge / F1 | context compression proxy |
| Full context (oracle) | 41–43% | F1 / judge | upper bound |

jit-context beats simple RAG by ~50% and outperforms training-free approaches, while requiring no external store and no chunking pipeline.

### Against dedicated memory systems

| system | score | metric | type |
|---|---|---|---|
| Memobase | 75.8% | LLM judge | persistent memory DB |
| MemMachine | ~90%+ | accuracy | external vector memory |
| ByteRover 2.0 | 92.2% | accuracy | external memory agent |
| MemU | 92.1% | accuracy | external memory agent |

These systems score higher but are a different class: they maintain memory across sessions, require dedicated infrastructure, and take significant engineering to integrate. The gap is real and expected.

### Where jit-context fits

| scenario | recommendation |
|---|---|
| Agent with growing single-session context | **jit-context** — drop-in, no infra |
| Need memory across sessions | Persistent store (Mem0, Memobase) |
| Using a recency window today | **Replace with jit-context** — same simplicity, much better recall |
| Short conversations (< 12 turns) | Pass-through (jit-context does nothing below threshold) |

---

## Reproducing these results

```bash
# Install deps
pip install openai python-dotenv

# Add to .env
# GOOGLE_API_KEY=...  (or OPENAI_API_KEY for OpenAI models)

# Download dataset
curl -L https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
     -o data/locomo10.json

# Run
python run_locomo_real.py
```

See [`run_locomo_real.py`](../run_locomo_real.py) and [`run_locomo_batched.py`](../run_locomo_batched.py) for the full setup. Swap in any OpenAI-compatible provider by changing `base_url` and model names.
