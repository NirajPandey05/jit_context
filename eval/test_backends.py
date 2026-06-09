"""Verify configurable real backends with mock SDK-shaped clients (no network).

Mocks mimic the exact response shapes the adapters expect, so this proves the
request construction, response parsing, id parsing, and fallback all work.
"""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jitcontext import JITConfig, build_manager
from jitcontext.index.embeddings import EmbedderSpec, make_embedder
from jitcontext.retrieval.picker import PickerSpec, make_picker, LLMPicker, _parse_ids
from jitcontext.retrieval.candidate import Candidate
from jitcontext.index.conversation_index import IndexEntry
from eval.tasks import make_task


# --- Mock OpenAI-shaped embeddings client -------------------------------------
class _Emb:        # resp.data[i].embedding
    def __init__(self, v): self.embedding = v
class _EmbResp:
    def __init__(self, vecs): self.data = [_Emb(v) for v in vecs]
def _toy_vec(text: str):
    # Toy "semantic" vector: presence of a few salient words. Enough for the
    # needle (which shares words like 'deploy'/'window'/'settle') to rank high
    # against the query — mimicking what a real embedder does.
    t = text.lower()
    keys = ["deploy", "window", "settle", "decision", "key", "retry",
            "rate", "limit", "region", "cache", "ttl", "flag", "sla", "percent",
            "ulid", "utc", "tuesday"]
    return [1.0 if k in t else 0.0 for k in keys] + [0.1]

class MockOpenAIEmbeddings:
    def create(self, model, input, **kw):
        return _EmbResp([_toy_vec(t) for t in input])
class MockOpenAIClient:
    def __init__(self): self.embeddings = MockOpenAIEmbeddings()


def test_openai_embedder():
    emb = make_embedder(EmbedderSpec(provider="openai", model="text-embedding-3-small",
                                     client=MockOpenAIClient()))
    out = emb.embed(["a", "bbb"])
    assert len(out) == 2 and len(out[0]) >= 3
    # normalised
    import math
    assert abs(math.sqrt(sum(x*x for x in out[0])) - 1.0) < 1e-6
    print("OK  openai-compatible embedder: shape + normalisation")


# --- Mock Anthropic-shaped picker client --------------------------------------
class _Block:
    def __init__(self, text): self.text = text
class _MsgResp:
    def __init__(self, text): self.content = [_Block(text)]
class MockAnthropicMessages:
    def __init__(self, reply): self._reply = reply
    def create(self, model, max_tokens, messages, **kw):
        return _MsgResp(self._reply)
class MockAnthropicClient:
    def __init__(self, reply): self.messages = MockAnthropicMessages(reply)


def _cands():
    e1 = IndexEntry(turn_id=9, role="assistant", summary="deploy window decision",
                    has_decision=True, embedding=[1,0,0])
    e2 = IndexEntry(turn_id=3, role="user", summary="chatter", embedding=[0,1,0])
    return [Candidate(e1, 0.9), Candidate(e2, 0.5)]


def test_anthropic_picker_parses_json():
    picker = make_picker(PickerSpec(provider="llm", api="anthropic",
                                    model="claude-haiku", client=MockAnthropicClient("[9]")))
    ids = picker.pick("what was the deploy window?", _cands(), max_select=6)
    assert ids == [9], ids
    print("OK  anthropic picker: parses JSON id list")


def test_picker_fallback_on_garbage():
    # Model returns nonsense -> falls back to heuristic (top by score+decision)
    picker = make_picker(PickerSpec(provider="llm", api="anthropic",
                                    model="x", client=MockAnthropicClient("no ids here")))
    ids = picker.pick("q", _cands(), max_select=1)
    assert ids == [9], ids   # heuristic picks highest score / decision
    print("OK  picker fallback: garbage reply -> heuristic")


def test_parse_ids_robustness():
    valid = {9, 3, 12}
    assert _parse_ids("[12, 9]", valid, 6) == [12, 9]
    assert _parse_ids("ids: 9 and 3", valid, 6) == [9, 3]
    assert _parse_ids("[99, 9]", valid, 6) == [9]      # drops invalid
    assert _parse_ids("[9,9,3]", valid, 6) == [9, 3]   # dedupes
    print("OK  id parser: json, scrape, invalid-drop, dedupe")


def test_end_to_end_with_mocks():
    # Full manager run using mock real backends end to end.
    task = make_task(n_turns=30, needle_pos=0.2, seed=1)
    cfg = JITConfig(mode="jit", activation_turn_threshold=12, max_selected_turns=6)
    mgr = build_manager(
        cfg,
        embedder_spec=EmbedderSpec(provider="openai", model="m", client=MockOpenAIClient()),
        picker_spec=PickerSpec(provider="llm", api="anthropic", model="m",
                               client=MockAnthropicClient(f"[{task.needle_turn_index}]")),
    )
    rewritten, report = mgr.process(task.messages)
    assert report.mode == "jit" and report.activated
    assert task.needle_turn_index in report.selected_turn_ids
    print(f"OK  end-to-end mocks: needle #{task.needle_turn_index} selected; "
          f"{report.input_tokens_est}->{report.sent_tokens_est} tok")


if __name__ == "__main__":
    test_openai_embedder()
    test_anthropic_picker_parses_json()
    test_picker_fallback_on_garbage()
    test_parse_ids_robustness()
    test_end_to_end_with_mocks()
    print("\nall backend wiring tests passed")
