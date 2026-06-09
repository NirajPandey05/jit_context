"""Synthetic eval tasks: needle planted mid-history, asked much later.

This is the make-or-break piece: quality improvement only shows up on
long-horizon, LOW-LOCALITY tasks — where the answer depends on something said
far back and NOT recent. If you test on tasks recency already solves, all three
configs tie and you wrongly conclude the idea fails.

Each task is a conversation plus a final question whose answer is a fact (the
"needle") planted at a known early turn, surrounded by plausible distractor
turns. We know the needle turn id, so we can score retrieval recall too.
"""
from __future__ import annotations
import random
from dataclasses import dataclass


@dataclass
class Task:
    messages: list[dict]      # full conversation incl. final question
    answer: str               # expected answer string
    needle_turn_index: int    # index (in non-system turns) holding the answer


_TOPICS = [
    "the database migration", "the auth refactor", "the pricing model",
    "the onboarding flow", "the caching layer", "the mobile rollout",
    "the API versioning", "the data retention policy",
]
_FACTS = [
    ("the deploy window", "Tuesday 02:00 UTC"),
    ("the chosen primary key", "a ULID"),
    ("the agreed retry limit", "5 attempts"),
    ("the rate limit", "120 requests per minute"),
    ("the fallback region", "eu-west-2"),
    ("the cache TTL", "900 seconds"),
    ("the feature flag name", "checkout_v3_enabled"),
    ("the SLA target", "99.95 percent"),
]


def make_task(n_turns: int = 40, needle_pos: float = 0.2,
              seed: int | None = None) -> Task:
    rng = random.Random(seed)
    fact_label, fact_value = rng.choice(_FACTS)
    needle_idx = max(1, int(n_turns * needle_pos))

    msgs: list[dict] = [{"role": "system",
                         "content": "You are a helpful engineering assistant."}]
    turns: list[dict] = []
    for i in range(n_turns):
        if i == needle_idx:
            turns.append({"role": "assistant",
                          "content": (f"Decision: {fact_label} for "
                                      f"{rng.choice(_TOPICS)} is {fact_value}. "
                                      "Let's lock that in.")})
        else:
            topic = rng.choice(_TOPICS)
            if i % 2 == 0:
                turns.append({"role": "user",
                              "content": f"Quick thought on {topic}? Some context and chatter."})
            else:
                turns.append({"role": "assistant",
                              "content": f"Sure — here are considerations about {topic}, with detail and noise."})

    question = {"role": "user",
                "content": f"Remind me, what did we settle on for {fact_label}?"}
    turns.append(question)
    return Task(messages=msgs + turns, answer=fact_value,
                needle_turn_index=needle_idx)
