"""LoCoMo runner: write-then-read per conversation, scored by category.

Per conversation:
  1. WRITE phase: feed all turns through the manager once to populate index +
     cold store (jit mode). For baselines (full/recency) there's no store, but
     the same process() call is used so the comparison is apples-to-apples.
  2. READ phase: for each (non-adversarial) question, append it as the final
     turn, run process() to assemble context, hand context+question to the
     answerer, then score with F1 + judge. Also record whether any evidence
     dia_id was among the selected turns (retrieval recall).

Aggregates accuracy/F1 by the four scored categories — the breakdown is the
point, since single-hop (pure retrieval) and temporal (timestamp-dependent)
stress the design very differently.

Offline by default (StubAnswerer/StubJudge, local embedder, heuristic picker).
Pass real specs/clients to get comparable numbers.
"""
from __future__ import annotations
import statistics
from collections import defaultdict
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from jitcontext import JITConfig, build_manager
from jitcontext.index.embeddings import EmbedderSpec
from jitcontext.retrieval.picker import PickerSpec
from eval.locomo.loader import LoCoMoConversation, load_conversations
from eval.locomo.scoring import (StubAnswerer, StubJudge, f1_score, QAResult)

ADVERSARIAL = 5


@dataclass
class CategoryStats:
    n: int
    f1: float
    judge_acc: float
    retrieval_recall: float


def _evidence_recall(selected_ids, idx_to_dia, evidence) -> bool:
    if not evidence:
        return False
    sel_dia = {idx_to_dia.get(i) for i in selected_ids}
    return any(e in sel_dia for e in evidence)


def run_conversation(conv: LoCoMoConversation, mode: str, cfg_kwargs: dict,
                     embedder_spec=None, picker_spec=None,
                     answerer=None, judge=None) -> list[QAResult]:
    # Map non-system message position -> dia_id (manager ids turns by position).
    idx_to_dia = {i: m.get("dia_id", "") for i, m in enumerate(conv.messages)}

    gold_lookup = {q.question: q.answer for q in conv.qa}
    answerer = answerer or StubAnswerer(gold_lookup)
    judge = judge or StubJudge()

    cfg = JITConfig(mode=mode, **cfg_kwargs)
    mgr = build_manager(cfg, embedder_spec=embedder_spec or EmbedderSpec(),
                        picker_spec=picker_spec or PickerSpec())

    # WRITE: prime the store/index with the full conversation.
    mgr.process(list(conv.messages))

    results: list[QAResult] = []
    for q in conv.qa:
        if q.category == ADVERSARIAL:
            continue
        msgs = list(conv.messages) + [{"role": "user", "content": q.question}]
        assembled, report = mgr.process(msgs)
        pred = answerer.answer(assembled, q.question)
        results.append(QAResult(
            category_name=q.category_name,
            f1=f1_score(pred, q.answer),
            judged_correct=judge.correct(q.question, pred, q.answer),
            retrieved_evidence=_evidence_recall(report.selected_turn_ids,
                                                idx_to_dia, q.evidence),
        ))
    return results


def aggregate(results: list[QAResult]) -> dict[str, CategoryStats]:
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r.category_name].append(r)
    out = {}
    for cat, rs in by_cat.items():
        out[cat] = CategoryStats(
            n=len(rs),
            f1=statistics.mean(r.f1 for r in rs),
            judge_acc=statistics.mean(1.0 if r.judged_correct else 0.0 for r in rs),
            retrieval_recall=statistics.mean(1.0 if r.retrieved_evidence else 0.0 for r in rs),
        )
    # overall
    out["ALL"] = CategoryStats(
        n=len(results),
        f1=statistics.mean(r.f1 for r in results) if results else 0.0,
        judge_acc=statistics.mean(1.0 if r.judged_correct else 0.0 for r in results) if results else 0.0,
        retrieval_recall=statistics.mean(1.0 if r.retrieved_evidence else 0.0 for r in results) if results else 0.0,
    )
    return out


def run_benchmark(path: str, modes=("full", "recency", "jit"),
                  cfg_kwargs=None, limit=None, **backend):
    cfg_kwargs = cfg_kwargs or dict(activation_turn_threshold=12,
                                    recent_turns_verbatim=4, shortlist_k=12,
                                    max_selected_turns=6)
    convs = load_conversations(path)
    if limit:
        convs = convs[:limit]

    print(f"Loaded {len(convs)} conversations, "
          f"{sum(len(c.qa) for c in convs)} QA pairs "
          f"(adversarial excluded at scoring).\n")

    for mode in modes:
        all_results: list[QAResult] = []
        for conv in convs:
            all_results += run_conversation(conv, mode, cfg_kwargs, **backend)
        stats = aggregate(all_results)
        print(f"=== mode: {mode} ===")
        print(f"{'category':<14}{'n':>6}{'F1':>8}{'judge':>8}{'ret_recall':>12}")
        for cat in ("single_hop", "multi_hop", "temporal", "open_domain", "ALL"):
            if cat in stats:
                s = stats[cat]
                rr = "n/a" if mode != "jit" else f"{s.retrieval_recall:.2f}"
                print(f"{cat:<14}{s.n:>6}{s.f1:>8.2f}{s.judge_acc:>8.2f}{rr:>12}")
        print()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/locomo10.json",
                    help="path to locomo10.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not os.path.exists(args.data):
        print(f"Dataset not found at {args.data}.\n"
              "Download locomo10.json from github.com/snap-research/locomo "
              "(data/locomo10.json) and pass --data <path>.\n"
              "Running the synthetic smoke test instead is in eval/locomo/smoke.py")
    else:
        run_benchmark(args.data, limit=args.limit)
