"""How to plug in REAL providers (configurable embedder + picker).

This file does not run live calls — it shows the wiring. Uncomment and add keys
to use. Everything is selected by spec; the manager/agent code never changes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jitcontext import JITConfig, build_manager, ProxyHandler
from jitcontext.index.embeddings import EmbedderSpec
from jitcontext.retrieval.picker import PickerSpec


def example_openai_embedder_anthropic_picker():
    """Embeddings via OpenAI; picker via Claude. Mix vendors freely."""
    # from openai import OpenAI
    # from anthropic import Anthropic
    # oai = OpenAI()              # reads OPENAI_API_KEY
    # ant = Anthropic()           # reads ANTHROPIC_API_KEY

    embedder_spec = EmbedderSpec(
        provider="openai",
        model="text-embedding-3-small",
        # client=oai,
        dimensions=512,           # optional truncation (3-* models)
    )
    picker_spec = PickerSpec(
        provider="llm",
        api="anthropic",
        model="claude-haiku-4-5-20251001",   # small/fast model for the pick step
        # client=ant,
    )
    cfg = JITConfig(mode="jit", activation_turn_threshold=12, max_selected_turns=6)
    # manager = build_manager(cfg, embedder_spec=embedder_spec, picker_spec=picker_spec)
    print("openai embedder + anthropic picker: specs ready (add clients to run)")


def example_openrouter_everything():
    """Single OpenAI-compatible base_url for both, via OpenRouter."""
    # from openai import OpenAI
    # client = OpenAI(base_url="https://openrouter.ai/api/v1",
    #                 api_key=os.environ["OPENROUTER_API_KEY"])
    embedder_spec = EmbedderSpec(provider="openai",
                                 model="openai/text-embedding-3-small")  # client=client
    picker_spec = PickerSpec(provider="llm", api="openai",
                             model="anthropic/claude-haiku")             # client=client
    print("openrouter: one base_url drives both embedder and picker")


def example_cohere_voyage():
    """Cohere or Voyage embedder; heuristic picker (no LLM call)."""
    # import cohere; co = cohere.Client(os.environ["COHERE_API_KEY"])
    cohere_spec = EmbedderSpec(provider="cohere", model="embed-v4.0")     # client=co
    # import voyageai; vo = voyageai.Client()
    voyage_spec = EmbedderSpec(provider="voyage", model="voyage-3")       # client=vo
    picker_spec = PickerSpec(provider="heuristic")  # cheapest: no model call
    print("cohere/voyage embedder + heuristic picker: specs ready")


if __name__ == "__main__":
    example_openai_embedder_anthropic_picker()
    example_openrouter_everything()
    example_cohere_voyage()
    print("\nAll backends are config-selected. Offline default still needs no keys.")
