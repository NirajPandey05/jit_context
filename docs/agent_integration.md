# Using jit-context with AI Agents

jit-context sits between your agent and its upstream LLM as a transparent proxy. The agent sends its normal message list; the proxy compresses it, forwards to the real LLM, and returns the response. The agent changes nothing except its base URL.

## Quick start: run the proxy server

```bash
python server.py --port 8787 --mode jit
```

The server prints the configuration lines for each agent type on startup:

```
JIT-context proxy on http://localhost:8787
  mode=jit  activates at 12+ turns

  Claude Code:
    export ANTHROPIC_BASE_URL=http://localhost:8787

  OpenAI-compatible agents:
    base_url=http://localhost:8787/v1
```

jit-context only activates once a conversation reaches 12 turns (configurable). Below that threshold it passes everything through untouched — short conversations are never affected.

---

## Claude Code

Claude Code respects `ANTHROPIC_BASE_URL`. Set it before starting a session:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8787
export ANTHROPIC_API_KEY=sk-ant-...    # unchanged
claude
```

Every Claude Code session on that terminal will now go through jit-context. The proxy handles the Anthropic message format (including the separate `system` field) automatically. Streaming works.

To run only specific sessions through the proxy:

```bash
ANTHROPIC_BASE_URL=http://localhost:8787 claude
```

---

## Any OpenAI-compatible agent

Point `base_url` at the proxy:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8787/v1",
    api_key="your-key",               # forwarded to upstream unchanged
)

# Use exactly as before — no other changes needed
response = client.chat.completions.create(
    model="gpt-4o",
    messages=conversation_history,
)
```

This works for any agent or framework that uses the OpenAI SDK or OpenAI-compatible API: LangChain, LlamaIndex, AutoGen, CrewAI, custom scripts, etc.

---

## GitHub Copilot

**Regular Copilot Chat (VS Code / JetBrains / GitHub.com):** not supported. Copilot's internal API cannot be intercepted from outside.

**GitHub Copilot Extensions (GitHub Apps with `copilot` capability):** fully supported. Inside your extension handler you receive the messages array directly and can run jit-context before forwarding:

```python
from jitcontext import JITConfig, build_manager

manager = build_manager(JITConfig(mode="jit"))

def handle_copilot_request(messages: list[dict]) -> list[dict]:
    rewritten, report = manager.process(messages)
    return rewritten   # forward this to your upstream LLM
```

---

## Embedding in application code (no HTTP server)

If you control the agent code, you can call jit-context directly without the server:

```python
from jitcontext import JITConfig, build_manager, ProxyHandler

# Build once at startup
proxy = ProxyHandler(
    build_manager(JITConfig(mode="jit")),
    upstream=lambda msgs: your_llm_client.chat(messages=msgs),
)

# Call instead of your LLM directly
response = proxy.handle(conversation_messages)
report   = proxy.last_report   # token savings, retrieved turn ids, etc.
```

### Inspect what happened

```python
rewritten, report = manager.process(messages)

print(report.mode)               # "jit"
print(report.activated)          # False for short conversations
print(report.n_input_turns)      # turns the agent sent
print(report.n_sent_turns)       # turns actually forwarded
print(report.token_savings_pct)  # e.g. 52.3
print(report.selected_turn_ids)  # which turns were retrieved
```

---

## Using real embeddings (recommended for production)

The default embedder is a bag-of-words hash — fine for testing, not for real quality. Swap in a real embedder:

```python
from openai import OpenAI
from jitcontext import JITConfig, build_manager
from jitcontext.index.embeddings import EmbedderSpec
from jitcontext.retrieval.picker import PickerSpec

client = OpenAI()   # or any OpenAI-compatible provider

manager = build_manager(
    JITConfig(mode="jit"),
    embedder_spec=EmbedderSpec(
        provider="openai",
        model="text-embedding-3-small",
        client=client,
    ),
    picker_spec=PickerSpec(
        provider="llm",
        api="openai",
        model="gpt-4o-mini",   # small model for the pick step
        client=client,
    ),
)
```

Then pass this `manager` to `ProxyHandler` or use it directly. Any OpenAI-compatible provider works — change `base_url` on the client for Google, Cohere, Azure, OpenRouter, etc.

---

## Configuration reference

```python
JITConfig(
    mode="jit",                    # "jit" | "recency" | "full"
    activation_turn_threshold=12,  # don't activate below this many turns
    activation_token_threshold=6000,  # or this many tokens (whichever fires first)
    recent_turns_verbatim=4,       # always keep last N turns in full
    shortlist_k=12,                # semantic retrieval candidate pool size
    max_selected_turns=6,          # hard cap on retrieved turns
    always_inject_index=True,      # inject TOC so model knows what exists
)
```

**Tuning tips:**
- Raise `activation_turn_threshold` if you want to leave short conversations completely alone
- Raise `max_selected_turns` if recall is low (at the cost of higher token count)
- Lower `recent_turns_verbatim` to save more tokens on very long conversations
