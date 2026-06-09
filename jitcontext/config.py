"""Configuration for the JIT-context proxy.

Everything that decides *whether* and *how much* the system intervenes lives
here, so the three-way eval (full / recency / jit) is just three configs of the
same pipeline.
"""
from dataclasses import dataclass, field


@dataclass
class JITConfig:
    # --- Trigger -----------------------------------------------------------
    # Below this many *non-system* messages, pass through untouched. This is
    # the "do no harm" guarantee for short conversations.
    activation_turn_threshold: int = 12
    # Alternatively trigger on token estimate (whichever fires first).
    activation_token_threshold: int = 6000

    # --- Assembly ----------------------------------------------------------
    # Recent turns always kept verbatim (recency is free and usually relevant).
    recent_turns_verbatim: int = 4
    # Semantic shortlist size handed to the model-picker.
    shortlist_k: int = 12
    # Hard cap on how many full turns the model may pull in.
    max_selected_turns: int = 6
    # Always inject the index (table-of-contents) so the model knows what
    # exists even when detail isn't loaded. Guards against silent recall loss.
    always_inject_index: bool = True

    # --- Modes -------------------------------------------------------------
    # "jit"     -> full pipeline (index + hybrid retrieval + assembly)
    # "recency" -> baseline: keep only the most recent turns, drop the rest
    # "full"    -> baseline: pass everything through untouched
    mode: str = "jit"

    # --- Token accounting --------------------------------------------------
    # ~4 chars/token heuristic; only used for thresholds and reporting.
    chars_per_token: float = 4.0

    def estimate_tokens(self, text: str) -> int:
        return int(len(text) / self.chars_per_token)
