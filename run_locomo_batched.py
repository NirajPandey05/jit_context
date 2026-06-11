"""Run LoCoMo benchmark in batches of 2 conversations, saving results after each batch."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from jitcontext.index.embeddings import EmbedderSpec
from eval.locomo.runner import run_benchmark, run_conversation, aggregate, load_conversations
from eval.locomo.scoring import LLMAnswerer, LLMJudge

RESULTS_FILE = "data/locomo_results.json"
DATA_FILE    = "data/locomo10.json"
BATCH_SIZE   = 2

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GOOGLE_API_KEY"],
)

embedder_spec = EmbedderSpec(provider="openai", model="models/gemini-embedding-001", client=client)
answerer      = LLMAnswerer(client, model="models/gemini-2.5-flash-lite", api="openai")
judge         = LLMJudge(client,    model="models/gemini-2.5-flash-lite", api="openai")

cfg_kwargs = dict(activation_turn_threshold=12, recent_turns_verbatim=4,
                  shortlist_k=12, max_selected_turns=6)

# Load existing results so we can resume
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        all_saved = json.load(f)
else:
    all_saved = {}

convs = load_conversations(DATA_FILE)
total = len(convs)
print(f"Dataset: {total} conversations total.\n")

# Determine which batches are already done
done_batches = set(all_saved.keys())  # keys like "conv_3-4"

for start in range(0, total, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total)
    batch_key = f"conv_{start+1}-{end}"

    if batch_key in done_batches:
        print(f"[{batch_key}] already done, skipping.")
        continue

    batch = convs[start:end]
    print(f"[{batch_key}] Running {len(batch)} conversations ({sum(len(c.qa) for c in batch)} QA pairs)...")

    batch_results = {}
    for mode in ("full", "recency", "jit"):
        mode_results = []
        for conv in batch:
            mode_results += run_conversation(
                conv, mode, cfg_kwargs,
                embedder_spec=embedder_spec, answerer=answerer, judge=judge,
            )
        stats = aggregate(mode_results)
        batch_results[mode] = {
            cat: {"n": s.n, "f1": round(s.f1, 4),
                  "judge": round(s.judge_acc, 4),
                  "ret_recall": round(s.retrieval_recall, 4)}
            for cat, s in stats.items()
        }
        # Print inline
        print(f"  {mode}:")
        for cat in ("single_hop", "multi_hop", "temporal", "open_domain", "ALL"):
            if cat in stats:
                s = stats[cat]
                rr = f"{s.retrieval_recall:.2f}" if mode == "jit" else "n/a"
                print(f"    {cat:<14} n={s.n}  F1={s.f1:.2f}  judge={s.judge_acc:.2f}  recall={rr}")

    # Save after each successful batch
    all_saved[batch_key] = batch_results
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_saved, f, indent=2)
    print(f"  -> saved to {RESULTS_FILE}\n")

print("All batches complete.")
