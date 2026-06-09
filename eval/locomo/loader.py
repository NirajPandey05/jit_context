"""Load the real LoCoMo dataset (snap-research/locomo, data/locomo10.json).

Schema (per sample):
  sample_id
  conversation:
    speaker_a, speaker_b
    session_<n>          : list of turns; each turn = {speaker, dia_id, text, [img_url, blip_caption]}
    session_<n>_date_time: timestamp string for that session
  qa: list of {question, answer, category, evidence:[dia_id,...]}
      category ints (LoCoMo convention):
        1 = multi-hop, 2 = temporal, 3 = open-domain, 4 = single-hop, 5 = adversarial
      (Exact int->name mapping varies by release; we surface the raw category and
       a best-effort name, and ALWAYS let the runner exclude adversarial.)

We convert each conversation into the standard message-list shape the
ContextManager consumes, preserving chronological order and prefixing each turn
with its speaker and session timestamp (so temporal questions stand a chance).
Each message carries the originating dia_id so we can score retrieval against
the `evidence` annotations.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field


# Best-effort category id -> name. The benchmark's own scripts key on these;
# adversarial is the one that MUST be excluded for comparability.
CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


@dataclass
class QAItem:
    question: str
    answer: str
    category: int
    category_name: str
    evidence: list[str] = field(default_factory=list)  # dia_ids


@dataclass
class LoCoMoConversation:
    sample_id: str
    messages: list[dict]          # [{role, content, dia_id, session, ts}]
    qa: list[QAItem]
    speaker_a: str
    speaker_b: str


_SESSION_RE = re.compile(r"^session_(\d+)$")


def _session_indices(conv: dict) -> list[int]:
    idxs = []
    for k in conv:
        m = _SESSION_RE.match(k)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(idxs)


def _role_for(speaker: str, speaker_a: str) -> str:
    # Map the two speakers onto user/assistant deterministically. LoCoMo is
    # peer-to-peer; we treat speaker_a as "user" and speaker_b as "assistant"
    # purely so the message list is well-formed. Content carries real names.
    return "user" if speaker == speaker_a else "assistant"


def load_conversations(path: str) -> list[LoCoMoConversation]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: list[LoCoMoConversation] = []
    for sample in data:
        conv = sample["conversation"]
        speaker_a = conv.get("speaker_a", "A")
        speaker_b = conv.get("speaker_b", "B")

        messages: list[dict] = []
        for n in _session_indices(conv):
            ts = conv.get(f"session_{n}_date_time", "")
            for turn in conv[f"session_{n}"]:
                text = turn.get("text", "")
                cap = turn.get("blip_caption")
                if cap:
                    text = f"{text} [shared image: {cap}]"
                speaker = turn.get("speaker", speaker_a)
                content = f"[{ts}] {speaker}: {text}"
                messages.append({
                    "role": _role_for(speaker, speaker_a),
                    "content": content,
                    "dia_id": turn.get("dia_id", ""),
                    "session": n,
                    "ts": ts,
                })

        qa_items: list[QAItem] = []
        for q in sample.get("qa", []):
            cat = q.get("category", 0)
            qa_items.append(QAItem(
                question=q.get("question", ""),
                answer=str(q.get("answer", "")),
                category=cat,
                category_name=CATEGORY_NAMES.get(cat, f"cat_{cat}"),
                evidence=list(q.get("evidence", []) or []),
            ))

        out.append(LoCoMoConversation(
            sample_id=str(sample.get("sample_id", len(out))),
            messages=messages, qa=qa_items,
            speaker_a=speaker_a, speaker_b=speaker_b,
        ))
    return out
