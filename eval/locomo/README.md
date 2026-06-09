# LoCoMo benchmark harness

Runs the JIT-context system against [LoCoMo](https://github.com/snap-research/locomo)
(Maharana et al., ACL 2024) — the standard long-term conversational memory
benchmark that Mem0, Letta, and others report on — so results are comparable to
published work instead of only self-measured on the synthetic generator.

## What it measures

Per the literature, the **adversarial** category is excluded. The remaining four
are scored separately, which is the point: they stress the design differently.

- **single_hop** — direct fact retrieval. The JIT sweet spot.
- **multi_hop** — synthesis across disjoint turns; stresses `max_selected_turns`
  and the picker's ability to pull *several* related turns.
- **temporal** — event ordering / absolute timelines; stresses whether the
  index preserves timestamps (the loader prefixes each turn with its session
  date for this reason).
- **open_domain** — inference beyond stated text.

Two scorers are reported (both, for defensibility):
- **F1** — SQuAD-style token-overlap vs reference (no keys).
- **judge** — LLM-as-judge correctness (matches recent papers).

It also reports **ret_recall** in jit mode: did the selected turns include any
annotated `evidence` dia_id? This separates the two failure modes —
*didn't retrieve the evidence* vs *retrieved it but couldn't use it*.

## Run offline (no keys) — verifies the whole pipeline

```bash
python eval/locomo/smoke.py     # tiny synthetic file in the real schema
```

The offline run uses `StubAnswerer` (a perfect reader: answers iff the gold
answer's tokens are in the assembled context) and `StubJudge` (lenient match).
This isolates the variable that matters offline — *did retrieval surface the
evidence?* — but note the stub cannot do reasoning, so **temporal** questions
score low offline even when evidence recall is perfect. That gap is expected;
it resolves with a real answerer.

## Run for real (needs keys)

1. Get `data/locomo10.json` from github.com/snap-research/locomo.
2. Wire real backends + a real answerer/judge:

```python
from eval.locomo.runner import run_benchmark
from jitcontext.index.embeddings import EmbedderSpec
from jitcontext.retrieval.picker import PickerSpec
from eval.locomo.scoring import LLMAnswerer, LLMJudge
# from anthropic import Anthropic; ant = Anthropic()
# from openai import OpenAI; oai = OpenAI()

run_benchmark(
    "data/locomo10.json",
    embedder_spec=EmbedderSpec(provider="openai",
                               model="text-embedding-3-small"),  # client=oai
    picker_spec=PickerSpec(provider="llm", api="anthropic",
                           model="claude-haiku-4-5-20251001"),   # client=ant
    answerer=None,   # pass LLMAnswerer(ant, "claude-...", api="anthropic")
    judge=None,      # pass LLMJudge(ant, "claude-...", api="anthropic")
)
```

(`run_benchmark` forwards `embedder_spec`/`picker_spec`/`answerer`/`judge` to
each conversation run; leave any as None to use the offline default.)

## Reading the comparison

`full` is the complete-but-diluted ceiling; `recency` is the cheap baseline that
drops old evidence; `jit` should approach `full`'s accuracy at a fraction of the
tokens **if** retrieval recall is high. The per-category table tells you exactly
where it doesn't — almost always temporal first (timestamp handling) and
multi-hop second (selection cap).
