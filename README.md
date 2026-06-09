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

## Integration: proxy / middleware

Chosen for zero-change drop-in. The agent points its base URL at the proxy;
the proxy owns context assembly and forwards to the real LLM. The "model picks"
step is an internal sub-call inside the proxy, invisible to the agent.

```python
from jitcontext import JITConfig, build_manager, ProxyHandler

proxy = ProxyHandler(build_manager(JITConfig(mode="jit")), upstream=my_llm_call)
response = proxy.handle(agent_messages)   # context rewritten transparently
```

## Three modes (same pipeline, different config)

- `full`    — pass everything (control; complete but diluted, 0% savings)
- `recency` — keep only the most recent N turns (baseline)
- `jit`     — index + hybrid retrieval + scoped-index assembly (system under test)

## Results (offline harness, synthetic needle-in-history)

Needle planted at 20% depth, asked at the end — a deliberately *low-locality*
task, the regime where the idea should win.

| n_turns | jit acc | jit recall | jit save% | recency acc |
|--------:|--------:|-----------:|----------:|------------:|
|   20    |  1.00   |    1.00    |    3.2    |    0.00     |
|   40    |  1.00   |    1.00    |   50.3    |    0.00     |
|   80    |  1.00   |    1.00    |   74.4    |    0.00     |
|  160    |  1.00   |    1.00    |   87.4    |    0.00     |
|  320    |  1.00   |    1.00    |   93.7    |    0.00     |

Reading: recency saves tokens but **loses the old needle entirely** (0%
accuracy). jit recovers full accuracy *and* saves tokens, with savings growing
as conversations lengthen. That gap is the whole thesis.

> Note: an earlier version injected the *full* index every turn and showed
> **negative** savings that worsened with length. Scoping the index to the
> shortlist + decision-flagged turns fixed it. The harness exists to catch
> exactly this.

## Offline by default, API-ready

Everything runs with no network/keys via deterministic local components, so the
pipeline is testable in isolation. Swap any one for an API-backed version
without touching the manager:

| Component        | Offline default      | Swap in                |
|------------------|----------------------|------------------------|
| Embedder         | `LocalHashEmbedder`  | `APIEmbedder`          |
| Summariser       | `HeuristicSummariser`| `LLMSummariser`        |
| Picker (model)   | `HeuristicPicker`    | `LLMPicker`            |
| Upstream LLM     | stub / oracle        | your provider client   |

## Run

```bash
python eval/harness.py     # three-way comparison
python eval/scaling.py     # token crossover vs length
python examples/run_proxy.py
```

## Important caveats (don't skip)

- **The local embedder is a bag-of-words hash, not real semantics.** Perfect
  recall above is partly because the synthetic needle shares rare tokens with
  the query. Real quality depends entirely on swapping in a real embedder +
  picker. Re-run the harness after swapping to get a trustworthy number.
- **Retrieval precision/recall is the whole ballgame.** A missed fetch is a
  *silent* quality loss. The scoped index (always listing relevant + decision
  turns) is the guard against the worst case.
- **Indexing belongs off the hot path.** In a real proxy, summarise/embed each
  turn *after* the response returns, so it never blocks the agent.
- **Streaming & provider prompt-cache** interactions are not handled in this
  prototype — see the design notes in code comments.
- Validate on *real* low-locality logs, not just the synthetic generator,
  before trusting the quality claim.
```
