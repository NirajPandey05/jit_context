# jit-context

Just-in-time context management for LLM agents. Keep a compact, navigable
**index** of the conversation hot; fetch full turn detail **lazily** only when a
turn is actually needed. The goal is twofold: lower token cost, and *higher
output quality* by keeping the working context small and dense (avoiding
lost-in-the-middle / distractor dilution on long histories).

## The idea

- **Main context** (the index) stays in cache: one compact entry per turn —
  summary + metadata (entities, decision flag) + embedding.
- **Needed context** is extracted from the cold store on demand via hybrid
  retrieval: a semantic shortlist, then a picker chooses which full turns to load.
- Activates only past a configurable length threshold — short conversations
  pass through untouched ("do no harm").

## Quickstart for agents

Run the proxy server, then point your agent at it — nothing else changes:

```bash
python server.py --port 8787 --mode jit
```

```bash
# Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8787

# Any OpenAI-compatible agent
client = OpenAI(base_url="http://localhost:8787/v1", api_key="...")
```

**→ [Full agent integration guide](docs/agent_integration.md)** — Claude Code, GitHub Copilot Extensions, OpenAI-compatible agents, embedding in application code, configuration reference.

Or call the library directly without the server:

```python
from jitcontext import JITConfig, build_manager, ProxyHandler

proxy = ProxyHandler(build_manager(JITConfig(mode="jit")), upstream=my_llm_call)
response = proxy.handle(agent_messages)   # context rewritten transparently
```

## Three modes (same pipeline, different config)

- `full`    — pass everything (control; complete but diluted, 0% savings)
- `recency` — keep only the most recent N turns (baseline)
- `jit`     — index + hybrid retrieval + scoped-index assembly (system under test)

## Results (LoCoMo benchmark, real conversations)

[LoCoMo](https://github.com/snap-research/locomo) is a long-conversation QA benchmark with four question categories. Results below use real embeddings (`gemini-embedding-001`) and a real answerer/judge (`gemini-2.5-flash-lite`) across all 10 conversations (1,540 non-adversarial QA pairs).

| mode | F1 | judge | ret_recall | token cost |
|---|---|---|---|---|
| `full` | 0.41 | 0.43 | — | 100% |
| `recency` | 0.00 | 0.01 | — | ~10% |
| `jit` | 0.37 | 0.36 | 0.71 | ~50% |

**jit reaches 90% of full-context F1** while retrieving only the relevant turns. Recency fails entirely because the answer is almost never in the most recent turns — exactly the regime jit-context is designed for.

Per-category breakdown (jit mode):

| category | F1 | judge | ret_recall |
|---|---|---|---|
| single_hop | 0.48 | 0.51 | 0.74 |
| multi_hop | 0.28 | 0.26 | 0.73 |
| temporal | 0.22 | 0.10 | 0.69 |
| open_domain | 0.13 | 0.20 | 0.43 |

Open_domain is the weakest — those questions require diffuse, inferential context that is harder to surface by similarity search. Full numbers in [`data/locomo_results.json`](data/locomo_results.json).

> **Model note:** answerer and judge are `gemini-2.5-flash-lite` (small/fast). All three modes benefit equally from a stronger model, so the *relative gaps* are meaningful; absolute F1 will be higher with GPT-4o or Claude Sonnet.

**→ [Full benchmark comparison](docs/benchmark_comparison.md)** — comparison against RAG baselines and persistent memory systems, reproduction instructions.

## Results (offline harness, synthetic needle-in-history)

Needle planted at 20% depth, asked at the end — a deliberately *low-locality*
task, the regime where the idea should win.

| n_turns | jit acc | jit recall | jit save% | recency acc |
|--------:|--------:|-----------:|----------:|------------:|
|      20 |    1.00 |       1.00 |       3.2 |        0.00 |
|      40 |    1.00 |       1.00 |      50.3 |        0.00 |
|      80 |    1.00 |       1.00 |      74.4 |        0.00 |
|     160 |    1.00 |       1.00 |      87.4 |        0.00 |
|     320 |    1.00 |       1.00 |      93.7 |        0.00 |

Token savings grow with conversation length. Recency saves tokens but loses
the old needle entirely — that gap is the whole thesis.

## Offline by default, configurable real backends

Everything runs with no network/keys via deterministic local components. Real
backends are **configurable by spec** — pass an `EmbedderSpec` / `PickerSpec`
to `build_manager`; the manager and agent code never change. Clients are
injected (no hard SDK imports), so importing the package never requires any
provider package.

| Component      | Offline default       | Configurable backends                          |
|----------------|-----------------------|------------------------------------------------|
| Embedder       | `local` (hash BoW)    | `openai` (+ any OpenAI-compatible base_url: OpenRouter/Azure/Together/Google), `cohere`, `voyage` |
| Picker (model) | `heuristic`           | `llm` with `api=anthropic` or `api=openai`     |
| Summariser     | `HeuristicSummariser` | `LLMSummariser` (plug a client)                |
| Upstream LLM   | stub / oracle         | your provider client                           |

```python
from jitcontext import JITConfig, build_manager
from jitcontext.index.embeddings import EmbedderSpec
from jitcontext.retrieval.picker import PickerSpec

manager = build_manager(
    JITConfig(mode="jit"),
    embedder_spec=EmbedderSpec(provider="openai",
                               model="text-embedding-3-small", client=openai_client),
    picker_spec=PickerSpec(provider="llm", api="anthropic",
                           model="claude-haiku-4-5-20251001", client=anthropic_client),
)
```

Vendors mix freely. Vectors are L2-normalised in every adapter, so cosine is a
plain dot product regardless of backend. The LLM picker falls back to the
heuristic on any error so a picker failure never crashes the proxy. See
`examples/configure_backends.py`.

## Run

```bash
# Offline tests (no keys needed)
python eval/harness.py              # three-way comparison, synthetic data
python eval/scaling.py              # token savings vs conversation length
python eval/test_backends.py        # mock client wiring tests
python eval/locomo/smoke.py         # LoCoMo schema smoke test

# Examples
python examples/run_proxy.py        # single proxy demo
python examples/configure_backends.py

# HTTP proxy server (for agent integration)
python server.py --port 8787 --mode jit

# LoCoMo benchmark with real backends
python run_locomo_real.py           # single run, all 10 conversations
python run_locomo_batched.py        # batched with per-pair checkpointing
```

To run the real benchmark, download the dataset and set your API key:

```bash
curl -L https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
     -o data/locomo10.json
echo "GOOGLE_API_KEY=..." > .env    # or OPENAI_API_KEY
python run_locomo_real.py
```

## Important caveats

- **The local default embedder is a bag-of-words hash, not real semantics.**
  Perfect recall in the offline harness is partly because the synthetic needle
  shares rare tokens with the query. Configure a real embedder for production.
- **Retrieval recall is the whole ballgame.** A missed fetch is a *silent*
  quality loss. The scoped index (always listing relevant + decision turns) is
  the guard against the worst case.
- **Indexing belongs off the hot path.** In a real proxy, summarise/embed each
  turn *after* the response returns so it never blocks the agent.
- **Provider prompt-cache** interactions with the injected index block are not
  yet handled — the index is injected as a new system message each turn.
- Validate on real low-locality logs before trusting the quality claim for
  your specific domain.
