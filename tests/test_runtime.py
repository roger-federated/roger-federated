"""Runtime wrapper: `--port` plan rewriting, the reverse proxy relaying byte-for-byte + streaming
live, and chat exchanges landing as reassembled JSON under messages/. Loopback only, no model."""
import json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from roger.runtime import capture, dialect, proxy

# ---------------------------------------------------------------------------
# plan(): argv rewriting
# ---------------------------------------------------------------------------

def test_plan_rewrites_port_and_keeps_public_side():
    p = dialect.plan(["llama-server", "-m", "x.gguf", "--port", "8080"], backend_port=45000)
    assert p.public_host == "127.0.0.1" and p.public_port == 8080 and p.backend_port == 45000
    assert p.child_argv == ["llama-server", "-m", "x.gguf", "--port", "45000"]


def test_plan_handles_equals_form_and_host():
    p = dialect.plan(["vllm", "serve", "m", "--host=0.0.0.0", "--port=8000"], backend_port=45000)
    assert p.public_host == "0.0.0.0" and p.public_port == 8000
    # The raw runtime is pinned to loopback even when the public side is LAN-exposed.
    assert p.child_argv == ["vllm", "serve", "m", "--host=127.0.0.1", "--port=45000"]


def test_plan_none_without_port():
    assert dialect.plan(["ollama", "serve"]) is None          # env-configured port: not supported
    assert dialect.plan(["llama-server", "--port", "abc"]) is None
    assert dialect.plan(["ollama", "pull", "llama3"]) is None


def test_plan_picks_a_free_backend_port():
    p = dialect.plan(["llama-server", "--port", "8080"])
    assert p.backend_port != 8080 and 1024 < p.backend_port < 65536


# ---------------------------------------------------------------------------
# SSE reassembly
# ---------------------------------------------------------------------------

def _sse(*objs):
    return "".join(f"data: {json.dumps(o) if not isinstance(o, str) else o}\n\n" for o in objs)


def _chunk(**delta_or_fields):
    choice = {"index": 0, "delta": delta_or_fields.pop("delta", {}), "finish_reason": delta_or_fields.pop("finish_reason", None)}
    return {"id": "c1", "object": "chat.completion.chunk", "model": "m", "choices": [choice], **delta_or_fields}


def test_assemble_sse_merges_content_reasoning_and_tool_calls():
    body = _sse(
        _chunk(delta={"role": "assistant", "content": ""}),
        _chunk(delta={"reasoning_content": "think"}),
        _chunk(delta={"content": "Hel"}), _chunk(delta={"content": "lo"}),
        _chunk(delta={"tool_calls": [{"index": 0, "id": "t1", "type": "function", "function": {"name": "f", "arguments": ""}}]}),
        _chunk(delta={"tool_calls": [{"index": 0, "function": {"arguments": '{"a":'}}]}),
        _chunk(delta={"tool_calls": [{"index": 1, "id": "t2", "function": {"name": "g", "arguments": "{}"}}]}),
        _chunk(delta={"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}),
        _chunk(finish_reason="tool_calls"),
        {"id": "c1", "object": "chat.completion.chunk", "model": "m", "choices": [], "usage": {"total_tokens": 9}},
        "[DONE]",
    )
    out = capture.assemble_sse(body)
    assert out["object"] == "chat.completion" and out["usage"] == {"total_tokens": 9}
    msg = out["choices"][0]["message"]
    assert msg["role"] == "assistant" and msg["content"] == "Hello" and msg["reasoning_content"] == "think"
    assert [t["function"] for t in msg["tool_calls"]] == [{"name": "f", "arguments": '{"a":1}'}, {"name": "g", "arguments": "{}"}]
    assert msg["tool_calls"][0]["id"] == "t1"
    assert out["choices"][0]["finish_reason"] == "tool_calls"


def test_assemble_sse_responses_api_and_garbage():
    body = "event: x\ndata: not json\n\n" + _sse(
        {"type": "response.created", "response": {"id": "r", "status": "in_progress"}},
        {"type": "response.output_text.delta", "delta": "hi"},
        {"type": "response.completed", "response": {"id": "r", "status": "completed", "output": ["hi"]}})
    assert capture.assemble_sse(body) == {"id": "r", "status": "completed", "output": ["hi"]}
    assert capture.assemble_sse(": keep-alive\n\n") is None


def test_assemble_detects_stream_from_the_wire():
    obj, streamed = capture.assemble("application/json", b'{"choices": []}')
    assert obj == {"choices": []} and streamed is False
    obj, streamed = capture.assemble("", _sse(_chunk(delta={"content": "x"})).encode())
    assert obj["choices"][0]["message"]["content"] == "x" and streamed is True
    assert capture.assemble("text/plain", b"nope") == (None, False)


def test_wants_matches_openai_chat_paths_only():
    assert capture.wants("POST", "/v1/chat/completions?x=1")
    assert capture.wants("POST", "/chat/completions") and capture.wants("POST", "/v1/completions")
    assert capture.wants("POST", "/v1/responses")
    assert not capture.wants("GET", "/v1/chat/completions")
    assert not capture.wants("POST", "/v1/models") and not capture.wants("POST", "/api/chat")


# ---------------------------------------------------------------------------
# Proxy end-to-end against a fake upstream
# ---------------------------------------------------------------------------

STREAM = [
    _chunk(delta={"role": "assistant", "content": "Hel"}),
    None,                                   # <- block here until the test releases the gate
    _chunk(delta={"content": "lo"}),
    _chunk(finish_reason="stop"),
    "[DONE]",
]


class _Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    gate = threading.Event()

    def log_message(self, *_): pass

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._json(200, {"path": self.path, "host": self.headers.get("Host")})

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", "5")
        self.end_headers()

    def do_POST(self):
        body = self._body()
        if self.path == "/echo":
            self._json(200, {"body": body.decode(), "enc": self.headers.get("Accept-Encoding"),
                             "host": self.headers.get("Host")})
            return
        req = json.loads(body)
        if req.get("model") == "bad":
            self._json(500, {"error": "boom"})
            return
        if not req.get("stream"):
            self._json(200, {"object": "chat.completion", "model": "m",
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}}]})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for ev in STREAM:
            if ev is None:
                self.gate.wait(5)
                continue
            piece = f"data: {ev if isinstance(ev, str) else json.dumps(ev)}\n\n".encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(piece), piece))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "state_dir", lambda: str(tmp_path))
    _Upstream.gate.clear()
    up = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    pub = dialect.free_port()
    srv = proxy.start("127.0.0.1", pub, f"http://127.0.0.1:{up.server_address[1]}", capture.make_sink("fake"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{pub}", up.server_address[1], tmp_path / "messages" / "fake"
    _Upstream.gate.set()
    proxy.stop(srv)
    up.shutdown(); up.server_close()


def _files(d):
    return sorted(d.glob("*.json")) if d.exists() else []


def test_passthrough_is_byte_exact_and_uncaptured(stack):
    base, backend_port, msgs = stack
    r = httpx.post(base + "/echo", content=b'{"raw": "bytes"}', headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200
    assert r.json() == {"body": '{"raw": "bytes"}', "enc": "gzip", "host": f"127.0.0.1:{backend_port}"}
    r = httpx.get(base + "/v1/models?x=1")
    assert r.json()["path"] == "/v1/models?x=1"
    assert _files(msgs) == []


def test_stream_relays_live_and_is_captured_assembled(stack):
    base, _, msgs = stack
    req = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    with httpx.Client(timeout=5) as c, c.stream("POST", base + "/v1/chat/completions", json=req) as r:
        assert r.status_code == 200 and r.headers["content-type"] == "text/event-stream"
        it = r.iter_raw()
        first = next(it)                         # arrives before the upstream gate opens ⇒ no buffering
        assert b"Hel" in first
        _Upstream.gate.set()
        rest = b"".join(it)
    assert b"[DONE]" in rest and b'"lo"' in rest
    for _ in range(50):                          # the sink runs after the last byte is relayed
        if _files(msgs):
            break
        time.sleep(0.05)
    (path,) = _files(msgs)
    rec = json.loads(path.read_text())
    assert rec["provider"] == "fake" and rec["endpoint"] == "/v1/chat/completions"
    assert rec["model"] == "m" and rec["stream"] is True and rec["request"] == req
    assert rec["response"]["choices"][0]["message"] == {"role": "assistant", "content": "Hello"}
    assert rec["response"]["choices"][0]["finish_reason"] == "stop"
    assert "raw_response" not in rec
    assert not list(msgs.glob("*.tmp"))


def test_non_stream_captured_verbatim_and_errors_skipped(stack):
    base, _, msgs = stack
    r = httpx.post(base + "/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 200
    time.sleep(0.2)
    (path,) = _files(msgs)
    rec = json.loads(path.read_text())
    assert rec["stream"] is False and rec["response"]["choices"][0]["message"]["content"] == "Hello"
    r = httpx.post(base + "/v1/chat/completions", json={"model": "bad", "messages": []})
    assert r.status_code == 500 and r.json() == {"error": "boom"}
    time.sleep(0.2)
    assert len(_files(msgs)) == 1


def test_head_then_get_on_one_keepalive_connection(stack):
    base, _, _ = stack
    with httpx.Client() as c:
        h = c.head(base + "/v1/models")
        assert h.status_code == 200 and h.content == b"" and h.headers["content-length"] == "5"
        g = c.get(base + "/v1/models")
        assert g.status_code == 200 and g.json()["path"] == "/v1/models"


def test_upgrade_refused_and_dead_backend_is_502(stack, monkeypatch):
    base, _, _ = stack
    r = httpx.get(base + "/", headers={"Upgrade": "websocket", "Connection": "Upgrade"})
    assert r.status_code == 501
    monkeypatch.setattr(proxy, "_CONNECT_RETRIES", 2)
    monkeypatch.setattr(proxy, "_CONNECT_BACKOFF", 0.01)
    dead = proxy.start("127.0.0.1", dialect.free_port(), f"http://127.0.0.1:{dialect.free_port()}", None)
    threading.Thread(target=dead.serve_forever, daemon=True).start()
    try:
        r = httpx.get(f"http://127.0.0.1:{dead.server_address[1]}/v1/models")
        assert r.status_code == 502 and b"not accepting connections" in r.content
    finally:
        proxy.stop(dead)


# ---------------------------------------------------------------------------
# Federation notice: what the runtime serves vs. what the federation accepts
# ---------------------------------------------------------------------------

from roger.federated import CLIENT_VERSION, transport
from roger.runtime import notice


def test_resolve_matches_runtime_names_to_accepted_ids():
    accepted = ["google/gemma-4-12B-it", "google/gemma-4-12B-it-assistant", "google/gemma-4-E2B-it"]
    # A gguf path with a quantization suffix, an exact HF id, and an alias all name the same base.
    assert notice.resolve(["/models/gemma-4-12B-it-Q4_K_M.gguf"], accepted) == "google/gemma-4-12B-it"
    assert notice.resolve(["google/gemma-4-12B-it"], accepted) == "google/gemma-4-12B-it"
    assert notice.resolve(["Gemma_4_12b_IT"], accepted) == "google/gemma-4-12B-it"
    # Longest name wins: the drafter must not resolve to the base it extends.
    assert notice.resolve(["gemma-4-12B-it-assistant.gguf"], accepted) == "google/gemma-4-12B-it-assistant"
    assert notice.resolve(["gemma-4-E2B-it-Q8_0.gguf"], accepted) == "google/gemma-4-E2B-it"
    assert notice.resolve(["meta-llama/Llama-3.1-8B-Instruct"], accepted) is None
    assert notice.resolve([], accepted) is None and notice.resolve(["x"], []) is None


class _Runtime(BaseHTTPRequestHandler):
    """What a runtime answers on /v1/models once its model is loaded."""
    def log_message(self, *_): pass

    def do_GET(self):
        data = json.dumps({"object": "list", "data": [{"id": "models/gemma-4-12B-it-Q4_K_M.gguf", "object": "model"}]}).encode()
        self.send_response(200 if self.path == "/v1/models" else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_served_models_waits_for_the_runtime(monkeypatch):
    monkeypatch.setattr(notice, "_POLL", 0.01)
    up = ThreadingHTTPServer(("127.0.0.1", 0), _Runtime)
    port = up.server_address[1]
    # Nothing listening yet: keeps polling (connection refused = still loading) until the server binds.
    threading.Timer(0.1, lambda: threading.Thread(target=up.serve_forever, daemon=True).start()).start()
    assert notice.served_models(f"http://127.0.0.1:{port}", lambda: True) == ["models/gemma-4-12B-it-Q4_K_M.gguf"]
    up.shutdown(); up.server_close()
    # Runtime gone before it ever answered: give up empty instead of polling forever.
    left = [3]
    def alive():
        left[0] -= 1
        return left[0] > 0
    assert notice.served_models(f"http://127.0.0.1:{dialect.free_port()}", alive) == []


def test_served_models_stops_on_missing_route(monkeypatch):
    monkeypatch.setattr(notice, "_POLL", 0.01)
    monkeypatch.setattr(notice.httpx, "get", lambda url, timeout: httpx.Response(404))
    calls = []
    assert notice.served_models("http://127.0.0.1:1", lambda: calls.append(1) or len(calls) < 50) == []
    assert len(calls) == 1                        # 404 = no such route, not "loading": one attempt


def _announce(served, statuses, cfg=None, monkeypatch=None):
    import io
    monkeypatch.setattr(transport, "federation_status", lambda url, mid: statuses[url])
    out = io.StringIO()
    notice.announce(served, {"federations": list(statuses), **(cfg or {})}, out=out)
    return out.getvalue()


def test_announce_verdicts(monkeypatch):
    accepted = ["google/gemma-4-12B-it", "google/gemma-4-E2B-it"]
    gguf = ["m/gemma-4-12B-it-Q4_K_M.gguf"]
    # Supported: resolved to the accepted id, encouraging.
    s = _announce(gguf, {"https://f": {"mode": "unsupported", "models": accepted}}, monkeypatch=monkeypatch)
    assert "✓ https://f trains google/gemma-4-12B-it" in s and "gemma-4-12B-it-Q4_K_M.gguf" in s
    # Unsupported: says so, lists what IS accepted so the user can switch.
    s = _announce(["llama-3.1-8b.gguf"], {"https://f": {"mode": "unsupported", "models": accepted}}, monkeypatch=monkeypatch)
    assert "⚠ https://f doesn't accept llama-3.1-8b.gguf" in s and "google/gemma-4-12B-it, google/gemma-4-E2B-it" in s
    # No allowlist (or a server predating `models`): the server's own verdict on the raw id decides.
    assert "✓ https://f accepts llama-3.1-8b.gguf" in _announce(["llama-3.1-8b.gguf"], {"https://f": {"mode": "bootstrap"}}, monkeypatch=monkeypatch)
    assert "⚠ https://f doesn't accept x" in _announce(["x"], {"https://f": {"mode": "unsupported"}}, monkeypatch=monkeypatch)
    # Runtime without /v1/models: still worth telling the user what the federation trains.
    s = _announce([], {"https://f": {"mode": "unsupported", "models": accepted}}, monkeypatch=monkeypatch)
    assert "https://f trains google/gemma-4-12B-it, google/gemma-4-E2B-it" in s and "⚠" not in s
    # Unreachable federation (fail-soft {}): silence, never a false warning.
    assert _announce(gguf, {"https://f": {}}, monkeypatch=monkeypatch) == ""
    # Client-version policy and leech mode ride along, like the legacy startup did.
    s = _announce(gguf, {"https://f": {"mode": "bootstrap", "models": accepted, "min_client": CLIENT_VERSION + 1}},
                  cfg={"contribute": False}, monkeypatch=monkeypatch)
    assert "out of date" in s and '"contribute" is off' in s
    s = _announce(gguf, {"https://f": {"mode": "bootstrap", "models": accepted, "latest_client": CLIENT_VERSION + 1}}, monkeypatch=monkeypatch)
    assert "newer roger client is available" in s and "out of date" not in s
    assert _announce(gguf, {}, monkeypatch=monkeypatch) == ""   # no federations configured: nothing to say


def test_transport_defaults_bare_host_to_https(monkeypatch):
    # The shipped default federation is a bare host; httpx refuses a scheme-less URL, which would have
    # made every probe fail-soft into silence.
    seen = []
    monkeypatch.setattr(transport.httpx, "get", lambda url, **kw: seen.append(url) or
                        httpx.Response(200, json={"mode": "busy"}, request=httpx.Request("GET", url)))
    assert transport.federation_status("server.rogerfederated.com", "m") == {"mode": "busy"}
    assert transport.federation_status("http://localhost:8000/", "m") == {"mode": "busy"}
    assert seen == ["https://server.rogerfederated.com/status", "http://localhost:8000/status"]
