"""Show the token crossover: jit only saves once conversations are long."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.harness import run_mode
from eval.tasks import make_task

cfg = dict(activation_turn_threshold=12, recent_turns_verbatim=4,
           shortlist_k=12, max_selected_turns=6)

print(f"{'n_turns':>8}{'jit_acc':>9}{'jit_recall':>11}{'jit_save%':>10}{'rec_acc':>9}")
print("-"*47)
for n in (20, 40, 80, 160, 320):
    tasks = [make_task(n_turns=n, needle_pos=0.2, seed=s) for s in range(40)]
    j = run_mode("jit", tasks, cfg)
    r = run_mode("recency", tasks, cfg)
    print(f"{n:>8}{j.accuracy:>9.2f}{j.retrieval_recall:>11.2f}"
          f"{j.savings_pct:>10.1f}{r.accuracy:>9.2f}")
