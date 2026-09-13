"""Runtime wrapper: `--port` plan rewriting, the reverse proxy relaying byte-for-byte + streaming
live, and chat exchanges landing as reassembled JSON under messages/. Loopback only, no model."""
import json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from roger.runtime import capture, grader, proxy

# ---------------------------------------------------------------------------
# plan(): argv rewriting
# ---------------------------------------------------------------------------

def test_plan_rewrites_port_and_keeps_public_side():
    p = proxy.plan(["llama-server", "-m", "x.gguf", "--port", "8080"], backend_port=45000)
    assert p.public_host == "127.0.0.1" and p.public_port == 8080 and p.backend_port == 45000
    assert p.child_argv == ["llama-server", "-m", "x.gguf", "--port", "45000"]


def test_plan_handles_equals_form_and_host():
    p = proxy.plan(["vllm", "serve", "m", "--host=0.0.0.0", "--port=8000"], backend_port=45000)
    assert p.public_host == "0.0.0.0" and p.public_port == 8000
    # The raw runtime is pinned to loopback even when the public side is LAN-exposed.
    assert p.child_argv == ["vllm", "serve", "m", "--host=127.0.0.1", "--port=45000"]


def test_plan_none_without_port():
    assert proxy.plan(["ollama", "serve"]) is None          # env-configured port: not supported
    assert proxy.plan(["llama-server", "--port", "abc"]) is None
    assert proxy.plan(["ollama", "pull", "llama3"]) is None


def test_plan_picks_a_free_backend_port():
    p = proxy.plan(["llama-server", "--port", "8080"])
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
    evals: list = []                 # bodies of self-eval requests received
    eval_mode = "ok"                 # "ok" | "strict" (400 on consecutive assistant msgs) | "garbage"

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
        msgs = req.get("messages") or []
        if msgs and msgs[-1]["role"] == "assistant" and grader.SEED in (msgs[-1].get("content") or ""):
            self.evals.append(req)
            if self.eval_mode == "strict" and len(msgs) > 1 and msgs[-2]["role"] == "assistant":
                self._json(400, {"error": "roles must alternate"})
                return
            content = ("not json" if self.eval_mode == "garbage" else json.dumps(
                {"reasoning": "ok", "scores": {"efficiency": 0.7, "accuracy": 3, "completeness": -0.5}}))
            self._json(200, {"object": "chat.completion", "model": "m",
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]})
            return
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
    _Upstream.evals = []
    _Upstream.eval_mode = "ok"
    up = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    pub = proxy.free_port()
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
    dead = proxy.start("127.0.0.1", proxy.free_port(), f"http://127.0.0.1:{proxy.free_port()}", None)
    threading.Thread(target=dead.serve_forever, daemon=True).start()
    try:
        r = httpx.get(f"http://127.0.0.1:{dead.server_address[1]}/v1/models")
        assert r.status_code == 502 and b"not accepting connections" in r.content
    finally:
        proxy.stop(dead)


# ---------------------------------------------------------------------------
# Grader: conversation chaining + the self-eval call
# ---------------------------------------------------------------------------

def _rec(messages, reply="Hello", endpoint="/v1/chat/completions"):
    return {"endpoint": endpoint, "model": "m", "request": {"model": "m", "messages": messages},
            "response": {"choices": [{"index": 0, "message": {"role": "assistant", "content": reply}}]}}


def test_track_chains_by_prefix_and_branches():
    reg = []
    a = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    grader.track(reg, "A", _rec(a, "Hello"))
    # Next request re-sends the history (reply content stripped by the client) + a new user turn.
    b = a + [{"role": "assistant", "content": "Hello "}, {"role": "user", "content": "more"}]
    grader.track(reg, "B", _rec(b, "Sure"))
    assert len(reg) == 1 and reg[0]["path"] == "B" and reg[0]["graded"] is False
    assert reg[0]["transcript"][-1] == ("assistant", "Sure")
    # The user edited the second turn instead: not a prefix of A's transcript → its own conversation.
    c = a + [{"role": "assistant", "content": "Hello"}, {"role": "user", "content": "different"}]
    grader.track(reg, "C", _rec(c, "Ok"))
    assert len(reg) == 2 and reg[1]["path"] == "C"
    grader.track(reg, "D", _rec(a, endpoint="/v1/completions"))
    grader.track(reg, "E", {"endpoint": "/v1/chat/completions", "request": {}, "response": None})
    assert len(reg) == 2


def test_due_uses_idle_threshold(monkeypatch):
    monkeypatch.setattr(grader, "IDLE_S", 100)
    reg = [{"path": "x", "transcript": [], "endpoint": "/v1/chat/completions", "last": 1000.0, "graded": False},
           {"path": "y", "transcript": [], "endpoint": "/v1/chat/completions", "last": 1000.0, "graded": True}]
    assert grader.due(reg, 1050.0) == []
    assert [e["path"] for e in grader.due(reg, 1100.0)] == ["x"]


def _captured_entry(stack, msgs_dir):
    base, _, _ = stack
    req = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}]}
    assert httpx.post(base + "/v1/chat/completions", json=req).status_code == 200
    for _ in range(50):
        if _files(msgs_dir):
            break
        time.sleep(0.05)
    (path,) = _files(msgs_dir)
    reg = []
    grader.track(reg, str(path), json.loads(path.read_text()))
    return path, reg[0]


def test_grade_prefills_seed_forces_schema_and_persists_scores(stack):
    base, backend_port, msgs = stack
    path, entry = _captured_entry(stack, msgs)
    with httpx.Client() as c:
        res = grader.grade(c, f"http://127.0.0.1:{backend_port}", entry)
    (sent,) = _Upstream.evals
    assert "tools" not in sent and sent["stream"] is False and sent["max_tokens"] == grader.MAX_TOKENS
    assert list(sent["response_format"]["json_schema"]["schema"]["properties"]) == ["reasoning", "scores"]
    assert sent["messages"][-2:] == [{"role": "assistant", "content": "Hello"},
                                     {"role": "assistant", "content": grader.SEED}]
    rec = json.loads(path.read_text())
    assert rec["self_eval"] == res and entry["graded"] is True
    assert res["scores"] == {"efficiency": 0.7, "accuracy": 1.0, "completeness": -0.5}   # clamped
    assert res["reasoning"] == "ok" and res["seed_placement"] == "message" and "score" not in res
    assert res["graded_messages"] == 2
    assert len(_files(msgs)) == 1                       # the eval itself was never captured


def test_grade_falls_back_to_appended_seed_for_strict_templates(stack):
    base, backend_port, msgs = stack
    _Upstream.eval_mode = "strict"
    path, entry = _captured_entry(stack, msgs)
    with httpx.Client() as c:
        res = grader.grade(c, f"http://127.0.0.1:{backend_port}", entry)
    assert len(_Upstream.evals) == 2
    assert _Upstream.evals[1]["messages"][-1] == {"role": "assistant", "content": "Hello\n\n" + grader.SEED}
    assert res["seed_placement"] == "appended" and res["scores"]["efficiency"] == 0.7


def test_grade_records_error_and_never_raises(stack):
    base, backend_port, msgs = stack
    _Upstream.eval_mode = "garbage"
    path, entry = _captured_entry(stack, msgs)
    with httpx.Client() as c:
        res = grader.grade(c, f"http://127.0.0.1:{backend_port}", entry)
    assert "error" in res and "scores" not in res and entry["graded"] is True
    assert json.loads(path.read_text())["self_eval"]["error"] == res["error"]
    # Dead backend: also an error, not an exception.
    entry["graded"] = False
    with httpx.Client() as c:
        res = grader.grade(c, f"http://127.0.0.1:{proxy.free_port()}", entry)
    assert res["error"].startswith("ConnectError") and entry["graded"] is True


def test_grade_pending_only_grades_ungraded(stack):
    base, backend_port, msgs = stack
    path, entry = _captured_entry(stack, msgs)
    reg = [entry, {"path": "none", "transcript": [], "endpoint": "/v1/chat/completions", "last": 0.0, "graded": True}]
    assert grader.grade_pending(reg, f"http://127.0.0.1:{backend_port}") == 1
    assert len(_Upstream.evals) == 1 and "scores" in json.loads(path.read_text())["self_eval"]
    assert grader.grade_pending(reg, f"http://127.0.0.1:{backend_port}") == 0
