"""JIT-context HTTP proxy server — drop-in base_url replacement.

Rewrites the messages list through jit-context before forwarding to the real
upstream. The calling agent changes nothing except its base URL.

Endpoints:
  POST /v1/messages            Anthropic format  (Claude Code, raw Anthropic SDK)
  POST /v1/chat/completions    OpenAI format     (Copilot Extensions, custom agents)
  GET  /v1/models              Pass-through      (agent health checks)

Usage:
  python server.py [--port 8787] [--mode jit|recency|full]

Agent setup:
  Claude Code:
    export ANTHROPIC_BASE_URL=http://localhost:8787
    (keep ANTHROPIC_API_KEY set as usual)

  OpenAI-compatible agents (Copilot Extensions, LangChain, etc.):
    client = OpenAI(base_url="http://localhost:8787/v1", api_key=os.environ["OPENAI_API_KEY"])

  GitHub Copilot (regular chat):
    Not supported — Copilot Chat's internal API cannot be intercepted.
    Use jit-context inside a Copilot Extension (GitHub App) instead.

  Pi.ai:
    No public API. Not directly applicable.

Required env vars (only those you actually use):
  ANTHROPIC_API_KEY
  OPENAI_API_KEY
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jitcontext import JITConfig, build_manager

_ANTHROPIC_UPSTREAM = "https://api.anthropic.com"
_OPENAI_UPSTREAM = "https://api.openai.com"

_PASSTHROUGH_RESPONSE_HEADERS = frozenset({
    "transfer-encoding", "connection", "keep-alive",
})


class _Handler(BaseHTTPRequestHandler):
    manager = None  # injected by main()

    def log_message(self, fmt, *args):
        pass  # suppress default access log; jit prints its own

    # ------------------------------------------------------------------
    def do_GET(self):
        # Health checks / model listing — forward to Anthropic as-is.
        self._passthrough_get(_ANTHROPIC_UPSTREAM + self.path,
                              {"x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                               "anthropic-version": self.headers.get("anthropic-version", "2023-06-01")})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        path = self.path.split("?")[0].rstrip("/")
        if path in ("/v1/messages", "/messages"):
            self._handle_anthropic(payload)
        elif path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_openai(payload)
        else:
            self._passthrough_post(raw, _ANTHROPIC_UPSTREAM + self.path,
                                   self._anthropic_headers())

    # ------------------------------------------------------------------
    def _handle_anthropic(self, payload):
        msgs = self._from_anthropic(payload)
        rewritten, report = self.manager.process(msgs)
        _log(report)
        new_payload = self._to_anthropic(payload, rewritten)
        body = json.dumps(new_payload).encode()
        self._forward(body, _ANTHROPIC_UPSTREAM + self.path,
                      self._anthropic_headers())

    def _handle_openai(self, payload):
        msgs = payload.get("messages", [])
        rewritten, report = self.manager.process(msgs)
        _log(report)
        new_payload = {**payload, "messages": rewritten}
        body = json.dumps(new_payload).encode()
        self._forward(body, _OPENAI_UPSTREAM + self.path,
                      self._openai_headers())

    # ------------------------------------------------------------------
    # Anthropic <-> internal format conversion
    # ------------------------------------------------------------------
    @staticmethod
    def _from_anthropic(payload):
        """Merge Anthropic's top-level `system` field into the messages list."""
        msgs = []
        sys_val = payload.get("system", "")
        if sys_val:
            if isinstance(sys_val, list):
                sys_text = "\n\n".join(
                    b.get("text", "") for b in sys_val if isinstance(b, dict)
                )
            else:
                sys_text = str(sys_val)
            if sys_text.strip():
                msgs.append({"role": "system", "content": sys_text})
        msgs.extend(payload.get("messages", []))
        return msgs

    @staticmethod
    def _to_anthropic(original_payload, rewritten):
        """Split system messages back out into the `system` field."""
        system_parts = [m["content"] for m in rewritten if m["role"] == "system"]
        non_system = [m for m in rewritten if m["role"] != "system"]
        new_payload = dict(original_payload)
        new_payload["messages"] = non_system
        if system_parts:
            new_payload["system"] = "\n\n".join(system_parts)
        else:
            new_payload.pop("system", None)
        return new_payload

    # ------------------------------------------------------------------
    def _anthropic_headers(self):
        headers = {
            "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "anthropic-version": self.headers.get("anthropic-version", "2023-06-01"),
            "Content-Type": "application/json",
        }
        beta = self.headers.get("anthropic-beta")
        if beta:
            headers["anthropic-beta"] = beta
        return headers

    def _openai_headers(self):
        return {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    def _forward(self, body_bytes, url, headers):
        req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in _PASSTHROUGH_RESPONSE_HEADERS:
                        self.send_header(k, v)
                self.end_headers()
                self._stream_body(resp)
        except urllib.error.HTTPError as e:
            err_body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(err_body)

    def _passthrough_post(self, raw_body, url, headers):
        req = urllib.request.Request(url, data=raw_body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in _PASSTHROUGH_RESPONSE_HEADERS:
                        self.send_header(k, v)
                self.end_headers()
                self._stream_body(resp)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(e.read())

    def _passthrough_get(self, url, headers):
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in _PASSTHROUGH_RESPONSE_HEADERS:
                        self.send_header(k, v)
                self.end_headers()
                self._stream_body(resp)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())

    def _stream_body(self, resp):
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()


def _log(report):
    if report.activated:
        print(
            f"[jit] mode={report.mode} "
            f"turns {report.n_input_turns}->{report.n_sent_turns}  "
            f"tokens {report.input_tokens_est}->{report.sent_tokens_est}  "
            f"({report.token_savings_pct:.0f}% saved)"
        )


def main():
    ap = argparse.ArgumentParser(description="JIT-context proxy server")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--mode", default="jit", choices=["jit", "recency", "full"])
    ap.add_argument("--activation-turns", type=int, default=12,
                    help="Min turns before JIT activates (default 12)")
    ap.add_argument("--max-selected", type=int, default=6,
                    help="Max retrieved turns injected (default 6)")
    ap.add_argument("--recent-verbatim", type=int, default=4,
                    help="Recent turns always kept verbatim (default 4)")
    args = ap.parse_args()

    cfg = JITConfig(
        mode=args.mode,
        activation_turn_threshold=args.activation_turns,
        max_selected_turns=args.max_selected,
        recent_turns_verbatim=args.recent_verbatim,
    )
    _Handler.manager = build_manager(cfg)

    server = HTTPServer(("0.0.0.0", args.port), _Handler)
    print(f"JIT-context proxy on http://localhost:{args.port}")
    print(f"  mode={args.mode}  activates at {args.activation_turns}+ turns")
    print()
    print("  Claude Code:")
    print(f"    export ANTHROPIC_BASE_URL=http://localhost:{args.port}")
    print()
    print("  OpenAI-compatible agents:")
    print(f"    base_url=http://localhost:{args.port}/v1")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
