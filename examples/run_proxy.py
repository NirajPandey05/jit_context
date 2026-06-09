"""Drop-in proxy example.

Shows the integration story: an agent builds its messages as usual and calls
`proxy.handle(messages)` instead of the LLM directly. The proxy rewrites the
context and forwards to `upstream`. The agent code is otherwise unchanged — in
a real deployment you point the agent's base_url at an HTTP server wrapping
this handler.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jitcontext import JITConfig, build_manager, ProxyHandler
from eval.tasks import make_task


def fake_llm(messages):
    """Stand-in for a real provider call. Echoes what it actually received."""
    sent = sum(1 for m in messages if m["role"] != "system")
    return {"role": "assistant",
            "content": f"[upstream received {sent} turns + system/index]"}


def main():
    task = make_task(n_turns=60, needle_pos=0.15, seed=7)

    cfg = JITConfig(mode="jit", activation_turn_threshold=12,
                    recent_turns_verbatim=4, max_selected_turns=6)
    proxy = ProxyHandler(build_manager(cfg), upstream=fake_llm)

    response = proxy.handle(task.messages)
    r = proxy.last_report
    print("response:", response["content"])
    print(f"\nmode={r.mode} activated={r.activated}")
    print(f"input turns: {r.n_input_turns}  ->  sent turns: {r.n_sent_turns}")
    print(f"selected (retrieved) turn ids: {r.selected_turn_ids}")
    print(f"recent verbatim turn ids:      {r.recent_turn_ids}")
    print(f"needle was planted at turn:    {task.needle_turn_index}")
    print(f"tokens: {r.input_tokens_est} -> {r.sent_tokens_est} "
          f"({r.token_savings_pct:.1f}% saved)")


if __name__ == "__main__":
    main()
