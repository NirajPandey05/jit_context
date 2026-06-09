"""Three-way harness: full vs recency vs jit, on needle tasks.

We don't need a real LLM to prove the central claim. The claim is about whether
the *needle survives into the context actually sent upstream*. So we use a
deterministic "oracle" upstream: it answers correctly iff the needle fact is
present in the messages it receives. That isolates the variable we care about —
context assembly — from LLM noise.

Outputs per mode:
  answer_accuracy : did the needle reach the model? (quality proxy)
  retrieval_recall: in jit, did we actually select the needle turn?
  mean_sent_tokens / token_savings_pct : cost (your #2)

Plug a real LLM in place of the oracle later to confirm end-to-end; the harness
shape doesn't change.
"""
from __future__ import annotations
import statistics
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jitcontext import JITConfig, build_manager
from eval.tasks import make_task, Task


def oracle_upstream(answer: str):
    """Returns a callable that 'answers correctly' iff answer is in context."""
    def _call(messages: list[dict]):
        blob = "\n".join(m["content"] for m in messages)
        return {"correct": answer in blob, "context_blob": blob}
    return _call


@dataclass
class ModeResult:
    mode: str
    accuracy: float
    retrieval_recall: float
    mean_sent_tokens: float
    mean_input_tokens: float

    @property
    def savings_pct(self) -> float:
        if self.mean_input_tokens == 0:
            return 0.0
        return 100 * (1 - self.mean_sent_tokens / self.mean_input_tokens)


def run_mode(mode: str, tasks: list[Task], cfg_kwargs: dict) -> ModeResult:
    correct = 0
    recall_hits = 0
    sent_toks: list[int] = []
    in_toks: list[int] = []

    for task in tasks:
        cfg = JITConfig(mode=mode, **cfg_kwargs)
        mgr = build_manager(cfg)
        upstream = oracle_upstream(task.answer)
        rewritten, report = mgr.process(task.messages)

        res = upstream(rewritten)
        correct += int(res["correct"])
        sent_toks.append(report.sent_tokens_est)
        in_toks.append(report.input_tokens_est)

        if mode == "jit":
            # needle index is among non-system turns; manager uses same indexing
            if task.needle_turn_index in report.selected_turn_ids:
                recall_hits += 1

    n = len(tasks)
    return ModeResult(
        mode=mode,
        accuracy=correct / n,
        retrieval_recall=(recall_hits / n) if mode == "jit" else float("nan"),
        mean_sent_tokens=statistics.mean(sent_toks),
        mean_input_tokens=statistics.mean(in_toks),
    )


def main() -> None:
    # Low-locality test set: needle early (20% in), asked at the end.
    tasks = [make_task(n_turns=40, needle_pos=0.2, seed=s) for s in range(50)]
    cfg_kwargs = dict(
        activation_turn_threshold=12,
        recent_turns_verbatim=4,
        shortlist_k=12,
        max_selected_turns=6,
    )

    print(f"{'mode':<10}{'accuracy':>10}{'recall':>10}"
          f"{'sent_tok':>12}{'in_tok':>10}{'savings%':>10}")
    print("-" * 62)
    for mode in ("full", "recency", "jit"):
        r = run_mode(mode, tasks, cfg_kwargs)
        recall = "n/a" if r.mode != "jit" else f"{r.retrieval_recall:.2f}"
        print(f"{r.mode:<10}{r.accuracy:>10.2f}{recall:>10}"
              f"{r.mean_sent_tokens:>12.0f}{r.mean_input_tokens:>10.0f}"
              f"{r.savings_pct:>10.1f}")

    print("\nReading: 'full' should be ~1.00 accuracy (everything present) but "
          "0% savings.\n'recency' should LOSE accuracy (needle is old, not "
          "recent) while saving tokens.\n'jit' should recover accuracy AND save "
          "tokens — IF retrieval recall is high. That gap is the whole thesis.")


if __name__ == "__main__":
    main()
