"""Offline smoke test: builds a tiny file in the REAL LoCoMo schema and runs the
full harness on it. No network, no keys. Verifies loader + runner + both
scorers + category breakdown end to end.
"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.locomo.runner import run_benchmark


def build_fake_locomo(path):
    # Two short sessions; plant facts; ask one question per scored category.
    sample = {
        "sample_id": "smoke-1",
        "conversation": {
            "speaker_a": "Deborah", "speaker_b": "Jolene",
            "session_1_date_time": "1 January 2023",
            "session_1": [
                {"speaker": "Deborah", "dia_id": "D1:1",
                 "text": "I adopted a beagle named Max last week."},
                {"speaker": "Jolene", "dia_id": "D1:2",
                 "text": "That's lovely! I just started a pottery class."},
            ],
            "session_2_date_time": "15 March 2023",
            "session_2": [
                {"speaker": "Deborah", "dia_id": "D2:1",
                 "text": "Max is settling in. I also booked a trip to Lisbon for May."},
                {"speaker": "Jolene", "dia_id": "D2:2",
                 "text": "Nice. My pottery teacher says I have a knack for bowls."},
            ] + [
                # filler/noise turns so retrieval has to work
                {"speaker": "Jolene", "dia_id": f"DN:{i}",
                 "text": f"Random chatter number {i} about the weather and lunch."}
                for i in range(20)
            ],
        },
        "qa": [
            {"question": "What is the name of Deborah's dog?",
             "answer": "Max", "category": 4, "evidence": ["D1:1"]},          # single-hop
            {"question": "What pet did Deborah get and where is she travelling?",
             "answer": "a beagle named Max and Lisbon", "category": 1,
             "evidence": ["D1:1", "D2:1"]},                                   # multi-hop
            {"question": "Did Deborah adopt the dog before booking Lisbon?",
             "answer": "yes", "category": 2, "evidence": ["D1:1", "D2:1"]},   # temporal
            {"question": "What hobby might relax Jolene based on her classes?",
             "answer": "pottery", "category": 3, "evidence": ["D1:2"]},       # open-domain
            {"question": "Unanswerable adversarial?",
             "answer": "no answer", "category": 5, "evidence": []},           # excluded
        ],
    }
    with open(path, "w") as f:
        json.dump([sample], f)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "locomo_smoke.json")
        build_fake_locomo(p)
        run_benchmark(p, cfg_kwargs=dict(activation_turn_threshold=8,
                                         recent_turns_verbatim=2,
                                         shortlist_k=8, max_selected_turns=4))
