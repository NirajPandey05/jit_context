"""Answerer, judge, and scorers for the LoCoMo harness.

All components are offline-runnable via stubs, then swappable for real clients
by passing an api/client. Two scorers are computed (per the choice to do both):
  - F1: token-overlap F1 vs reference (no keys needed)
  - Judge: LLM-as-judge correctness (matches recent papers); offline stub uses
    a lenient containment check so the pipeline runs end to end without keys.
"""
from __future__ import annotations
import re
import string
from collections import Counter
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Text normalisation + F1 (SQuAD-style)
# --------------------------------------------------------------------------- #
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalise(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = normalise(pred).split(), normalise(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------- #
# Answerer: turns assembled context + question into an answer string
# --------------------------------------------------------------------------- #
class StubAnswerer:
    """Offline answerer. Answers correctly iff the gold answer's tokens appear
    in the assembled context — i.e. it models a perfect reader, isolating the
    variable we care about (did retrieval surface the evidence?). This is the
    same logic as the synthetic oracle, generalised to token overlap.
    """
    def __init__(self, gold_lookup):
        self._gold = gold_lookup  # question -> gold answer

    def answer(self, messages, question: str) -> str:
        blob = normalise(" ".join(m["content"] for m in messages))
        gold = self._gold.get(question, "")
        gold_toks = normalise(gold).split()
        if gold_toks and all(t in blob for t in gold_toks):
            return gold              # perfect reader: evidence present
        return "i don't know"


class LLMAnswerer:
    """Real answerer. api in {'anthropic','openai'}; client injected."""
    def __init__(self, client, model: str, api: str = "anthropic",
                 max_tokens: int = 256):
        self.client, self.model, self.api, self.max_tokens = client, model, api, max_tokens

    def answer(self, messages, question: str) -> str:
        sys = ("Answer the question using ONLY the conversation context "
               "provided. Be concise. If the answer is not in the context, "
               "say 'I don't know'.")
        ctx = "\n".join(m["content"] for m in messages)
        prompt = f"{sys}\n\nContext:\n{ctx}\n\nQuestion: {question}\nAnswer:"
        if self.api == "anthropic":
            r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                             messages=[{"role": "user", "content": prompt}])
            return r.content[0].text.strip()
        r = self.client.chat.completions.create(model=self.model, max_tokens=self.max_tokens,
                                                messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()


# --------------------------------------------------------------------------- #
# Judge: decides correctness of pred vs gold
# --------------------------------------------------------------------------- #
class StubJudge:
    """Offline judge: lenient match (exact-normalised, containment, or high F1)."""
    def correct(self, question: str, pred: str, gold: str) -> bool:
        np_, ng = normalise(pred), normalise(gold)
        if not ng:
            return False
        if np_ == ng or ng in np_ or np_ in ng:
            return True
        return f1_score(pred, gold) >= 0.6


class LLMJudge:
    """Real LLM-as-judge. Returns True/False for semantic correctness."""
    def __init__(self, client, model: str, api: str = "anthropic"):
        self.client, self.model, self.api = client, model, api

    def correct(self, question: str, pred: str, gold: str) -> bool:
        prompt = (f"Question: {question}\nReference answer: {gold}\n"
                  f"Model answer: {pred}\n\nIs the model answer correct? "
                  "Reply with only 'yes' or 'no'.")
        if self.api == "anthropic":
            r = self.client.messages.create(model=self.model, max_tokens=8,
                                            messages=[{"role": "user", "content": prompt}])
            txt = r.content[0].text
        else:
            r = self.client.chat.completions.create(model=self.model, max_tokens=8,
                                                    messages=[{"role": "user", "content": prompt}])
            txt = r.choices[0].message.content
        return txt.strip().lower().startswith("y")


@dataclass
class QAResult:
    category_name: str
    f1: float
    judged_correct: bool
    retrieved_evidence: bool   # did selected turns include any evidence dia_id?
