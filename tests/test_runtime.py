"""Runtime wrapper: `--port` plan rewriting, the reverse proxy relaying byte-for-byte + streaming
live, and chat exchanges landing as reassembled JSON under messages/. Loopback only, no model."""
import json, os, pathlib, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from roger.runtime import capture, dialect, grader, proxy, signals

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
        try:
            req = json.loads(body)
        except ValueError:
            self._json(400, {"body": body.decode()})
            return
        msgs = req.get("messages") or []
        if msgs and msgs[-1]["role"] == "assistant" and grader.SEED in (msgs[-1].get("content") or ""):
            self.evals.append(req)
            if self.eval_mode == "strict" and len(msgs) > 1 and msgs[-2]["role"] == "assistant":
                self._json(400, {"error": "roles must alternate"})
                return
            verdict = {"reasoning": "ok", "scores": {"efficiency": 0.7, "accuracy": 3, "completeness": -0.5}}
            asked = req["response_format"]["json_schema"]["schema"]["properties"].get("reactions")
            if asked:                                    # one per forced reply_N, alternating -2 / 0.5
                verdict["reactions"] = {k: (-2 if n % 2 == 0 else 0.5)
                                        for n, k in enumerate(asked["properties"])}
            content = "not json" if self.eval_mode == "garbage" else json.dumps(verdict)
            self._json(200, {"object": "chat.completion", "model": "m",
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]})
            return
        if req.get("model") == "bad":
            self._json(500, {"error": "boom"})
            return
        if not req.get("stream"):
            self._json(200, {"object": "chat.completion", "model": req.get("model", "m"),
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


def _stack(tmp_path, monkeypatch, request_model=None):
    monkeypatch.setattr(capture, "state_dir", lambda: str(tmp_path))
    _Upstream.gate.clear()
    _Upstream.evals = []
    _Upstream.eval_mode = "ok"
    up = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    pub = dialect.free_port()
    srv = proxy.start("127.0.0.1", pub, f"http://127.0.0.1:{up.server_address[1]}", capture.make_sink("fake"),
                      request_model)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{pub}", up.server_address[1], tmp_path / "messages" / "fake"
    _Upstream.gate.set()
    proxy.stop(srv)
    up.shutdown(); up.server_close()


@pytest.fixture
def stack(tmp_path, monkeypatch):
    yield from _stack(tmp_path, monkeypatch)


@pytest.fixture
def retargeting_stack(tmp_path, monkeypatch):
    yield from _stack(tmp_path, monkeypatch, request_model=dialect.LORA_NAME)


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


# ---------------------------------------------------------------------------
# signals: exit-code / error rewards from the tool results in the history
# ---------------------------------------------------------------------------

def test_signal_score_reads_the_common_exit_code_formats():
    assert signals.score("exit 0\nok") == 0.0 and signals.score("Exit code: 0\nall good") == 0.0
    assert signals.score("Wrote 42 bytes to foo.txt") == 0.0
    for failing in ("exit 1\nfailed", "Exit code: 2", "exit status 127", "Process exited with code 1",
                    "[Command finished with exit code 3]"):
        assert signals.score(failing) == pytest.approx(-signals.W_EXIT), failing
    assert signals.score("Error: file not found") < 0
    assert signals.score("Command rejected by user: rm -rf /") <= -signals.W_CMD_REJ
    assert signals.score("Command rejected by user: exit 1\nError: x not found") >= -1.0


def test_step_rewards_credit_results_to_the_turn_that_called_them():
    call = {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}, {"id": "b"}]}
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"},
            call, {"role": "tool", "tool_call_id": "a", "content": "exit 1\nboom"},
            {"role": "tool", "tool_call_id": "b", "content": [{"type": "text", "text": "Error: nope"}]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c"}]},
            {"role": "tool", "tool_call_id": "c", "content": "exit 0\nfine"},
            {"role": "assistant", "content": "done"}]
    assert signals.step_rewards(msgs) == {"2": pytest.approx(-signals.W_EXIT - signals.W_ERROR)}
    # responses API: a run of function_call items is one step, anchored at its first item
    items = [{"type": "message", "role": "user", "content": "go"},
             {"type": "function_call", "call_id": "a"}, {"type": "function_call", "call_id": "b"},
             {"type": "function_call_output", "call_id": "a", "output": "Exit code: 1"},
             {"type": "function_call_output", "call_id": "b", "output": "Permission denied"}]
    assert signals.step_rewards(items) == {"1": pytest.approx(-signals.W_EXIT - signals.W_ERROR)}
    assert signals.step_rewards("plain prompt") == {} and signals.step_rewards(None) == {}


def test_capture_records_tool_signals(stack):
    base, _, msgs = stack
    history = [{"role": "user", "content": "run it"},
               {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}]},
               {"role": "tool", "tool_call_id": "a", "content": "exit 2\nsegfault"}]
    httpx.post(base + "/v1/chat/completions", json={"model": "m", "messages": history})
    time.sleep(0.2)
    (path,) = _files(msgs)
    assert json.loads(path.read_text())["tool_signals"] == {"1": pytest.approx(-signals.W_EXIT)}


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
    # Client-version policy and leech mode ride along too.
    s = _announce(gguf, {"https://f": {"mode": "bootstrap", "models": accepted, "min_client": CLIENT_VERSION + 1}},
                  cfg={"contribute": False}, monkeypatch=monkeypatch)
    assert "out of date" in s and '"contribute" is off' in s
    s = _announce(gguf, {"https://f": {"mode": "bootstrap", "models": accepted, "latest_client": CLIENT_VERSION + 1}}, monkeypatch=monkeypatch)
    assert "newer roger client is available" in s and "out of date" not in s
    assert _announce(gguf, {}, monkeypatch=monkeypatch) == ""   # no federations configured: nothing to say


def test_privacy_notice_shows_once_per_machine(tmp_path, monkeypatch):
    # Transparency obligation: it must be printed before this client's first federation contact, and
    # the sentinel must survive across launches so it is not re-printed every time.
    import io
    monkeypatch.setattr(notice, "state_dir", lambda: str(tmp_path))
    out = io.StringIO()
    notice.privacy_notice({"federations": ["https://f"]}, out=out)
    assert "encrypted gradient update" in out.getvalue() and "PRIVACY.md" in out.getvalue()
    assert os.path.exists(os.path.join(str(tmp_path), "privacy_ack"))
    out2 = io.StringIO()
    notice.privacy_notice({"federations": ["https://f"]}, out=out2)
    assert out2.getvalue() == ""                          # shown once per machine, not per launch
    # No federation configured: nothing is ever sent, so there is nothing to disclose (and no sentinel,
    # so the notice still fires the first time the user does configure one).
    fresh = io.StringIO()
    monkeypatch.setattr(notice, "state_dir", lambda: str(tmp_path / "fresh"))
    notice.privacy_notice({"federations": []}, out=fresh)
    assert fresh.getvalue() == "" and not os.path.exists(str(tmp_path / "fresh"))


def test_transport_defaults_bare_host_to_https(monkeypatch):
    # The shipped default federation is a bare host; httpx refuses a scheme-less URL, which would have
    # made every probe fail-soft into silence.
    seen = []
    monkeypatch.setattr(transport.httpx, "get", lambda url, **kw: seen.append(url) or
                        httpx.Response(200, json={"mode": "busy"}, request=httpx.Request("GET", url)))
    assert transport.federation_status("server.rogerfederated.com", "m") == {"mode": "busy"}
    assert transport.federation_status("http://localhost:8000/", "m") == {"mode": "busy"}
    assert seen == ["https://server.rogerfederated.com/status", "http://localhost:8000/status"]


def test_retarget_rewrites_chat_model_but_captures_the_original(retargeting_stack):
    base, _, msgs = retargeting_stack
    # vllm serves the federation adapter under its own name: chat requests are pointed at it on the
    # wire (the upstream echoes the model it was asked for), while the saved exchange keeps what the
    # client actually sent. Non-chat paths and non-JSON bodies pass untouched.
    r = httpx.post(base + "/v1/chat/completions", json={"model": "google/gemma-4-12B-it", "messages": []})
    assert r.status_code == 200 and r.json()["model"] == dialect.LORA_NAME
    time.sleep(0.2)
    (path,) = _files(msgs)
    rec = json.loads(path.read_text())
    assert rec["request"]["model"] == "google/gemma-4-12B-it"
    r = httpx.post(base + "/echo", content=b'{"model": "x"}', headers={"Content-Type": "application/json"})
    assert r.json()["body"] == '{"model": "x"}'
    r = httpx.post(base + "/v1/chat/completions", content=b"not json", headers={"Content-Type": "text/plain"})
    assert r.status_code == 400 and r.json()["body"] == "not json"   # forwarded as-is for the runtime to reject


# ---------------------------------------------------------------------------
# Federation update as a runtime adapter (runtime/dialect.py writers, runtime/adapter.py orchestration)
# ---------------------------------------------------------------------------

import numpy as np
from safetensors.numpy import save as st_save_np

from roger.runtime import adapter


def test_served_model_and_attach_adapter():
    assert dialect.served_model(["llama-server", "-m", "x.gguf", "--port", "1"]) == "x.gguf"
    assert dialect.served_model(["/opt/bin/llama-server.exe", "--model", "y.gguf"]) == "y.gguf"
    assert dialect.served_model(["vllm", "serve", "org/name", "--port", "1"]) == "org/name"
    assert dialect.served_model(["vllm", "serve", "--model", "org/name"]) == "org/name"
    assert dialect.served_model(["vllm", "serve", "--port", "1"]) is None       # config-file / no model
    assert dialect.served_model(["llama-server", "-hf", "org/name", "--port", "1"]) is None
    assert dialect.served_model(["someserver", "--model", "z"]) is None          # unknown runtime
    p = dialect.plan(["llama-server", "-m", "x.gguf", "--port", "8080"], backend_port=9)
    q = dialect.attach_adapter(p, "llama-server", "/a/b.gguf", 16)
    assert q.child_argv == p.child_argv + ["--lora", "/a/b.gguf"] and q.request_model is None
    p = dialect.plan(["vllm", "serve", "org/name", "--port", "8000"], backend_port=9)
    q = dialect.attach_adapter(p, "/usr/bin/vllm", "/a/dir", 20)
    assert q.child_argv[-5:] == ["--enable-lora", "--lora-modules", f"{dialect.LORA_NAME}=/a/dir",
                                 "--max-lora-rank", "32"]                   # vllm only takes its own rank steps
    assert q.request_model == dialect.LORA_NAME
    assert dialect.attach_adapter(p, "someserver", "/a", 8) == p                 # unknown runtime: untouched
    assert dialect._vllm_rank(8) == 8 and dialect._vllm_rank(600) == 512


# Module paths as the trainer records them: PEFT prefix + a multimodal checkpoint's nested decoder.
_MODS = ["base_model.model.model.language_model.layers.0.self_attn.q_proj",
         "base_model.model.model.language_model.layers.0.self_attn.v_proj",
         "base_model.model.model.language_model.layers.1.self_attn.q_proj"]


def _factor_blob(mods=_MODS, r=2, out=8, in_=6, scaling="2.0", seed=0):
    rng = np.random.default_rng(seed)
    t = {}
    for m in mods:
        t[m + ".lora_A.weight"] = rng.standard_normal((r, in_)).astype(np.float32)
        t[m + ".lora_B.weight"] = rng.standard_normal((out, r)).astype(np.float32)
    return st_save_np(t, metadata={"model_id": "google/gemma-4-12B-it", "compat": "x", "scaling": scaling}), t


def _base_gguf(path, arch="gemma4", out=8, in_=6, heads=2):
    import gguf
    w = gguf.GGUFWriter(str(path), arch)
    w.add_block_count(2)
    w.add_head_count(heads)
    for name in ("blk.0.attn_q.weight", "blk.0.attn_v.weight", "blk.1.attn_q.weight", "blk.1.attn_v.weight"):
        w.add_tensor(name, np.zeros((out, in_), dtype=np.float32))
    w.add_tensor("blk.0.ffn_up.weight", np.zeros((3, in_), dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return str(path)


def test_load_factors_folds_scaling_and_merges_federations_at_1_over_n(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, "state_dir", lambda: str(tmp_path))
    b1, t1 = _factor_blob(seed=1, scaling="2.0")
    b2, t2 = _factor_blob(seed=2, scaling="0.5", mods=_MODS[:1])
    transport.save_global("http://a", b1, "m.gguf")
    transport.save_global("http://b", b2, "m.gguf")
    f = adapter.load_factors(["http://a", "http://b"], "m.gguf")
    assert set(f) == set(_MODS)
    A, B = f[_MODS[0]]
    assert A.shape == (4, 6) and B.shape == (8, 4)                          # ranks concatenated
    # two federations with a global → each merged at 1/2 on top of its own scaling
    want = 0.5 * (2.0 * t1[_MODS[0] + ".lora_B.weight"] @ t1[_MODS[0] + ".lora_A.weight"]
                  + 0.5 * t2[_MODS[0] + ".lora_B.weight"] @ t2[_MODS[0] + ".lora_A.weight"])
    assert np.allclose(B @ A, want, atol=1e-5)
    A, B = f[_MODS[1]]
    assert np.allclose(B @ A, 0.5 * 2.0 * t1[_MODS[1] + ".lora_B.weight"] @ t1[_MODS[1] + ".lora_A.weight"],
                       atol=1e-5)
    # a configured federation with no global yet doesn't dilute the one that has it: N counts globals
    A, B = adapter.load_factors(["http://a", "http://none"], "m.gguf")[_MODS[1]]
    assert np.allclose(B @ A, 2.0 * t1[_MODS[1] + ".lora_B.weight"] @ t1[_MODS[1] + ".lora_A.weight"], atol=1e-5)
    # A dense (pre-factor) global has no factor keys → nothing.
    transport.save_global("http://a", st_save_np({"m": np.zeros((2, 2), np.float32)}), "d.gguf")
    assert adapter.load_factors(["http://a"], "d.gguf") == {}


def test_build_gguf_adapter_matches_base_and_skips_mismatch(tmp_path, monkeypatch):
    import gguf
    base = _base_gguf(tmp_path / "base.gguf")
    _, t = _factor_blob()
    factors = {m: (t[m + ".lora_A.weight"], t[m + ".lora_B.weight"]) for m in _MODS}
    factors["base_model.model.model.language_model.layers.1.self_attn.v_proj"] = (np.zeros((2, 5), np.float32),
                                                                                    np.zeros((8, 2), np.float32))
    factors["base_model.model.model.language_model.layers.0.mlp.gate_proj"] = (np.zeros((2, 6), np.float32),
                                                                                 np.zeros((3, 2), np.float32))
    import io
    err = io.StringIO()
    path, rank = dialect.build_gguf(factors, base, str(tmp_path / "out"), out=err)
    assert rank == 2 and "v_proj" in err.getvalue() and "gate_proj" in err.getvalue()   # in≠6 / not in the base
    r = gguf.GGUFReader(path)
    assert r.fields["general.architecture"].contents() == "gemma4"
    assert r.fields["general.type"].contents() == "adapter"
    assert r.fields["adapter.type"].contents() == "lora"
    assert float(r.fields["adapter.lora.alpha"].contents()) == 0.0
    names = {x.name: tuple(int(d) for d in reversed(list(x.shape))) for x in r.tensors}
    assert names == {"blk.0.attn_q.weight.lora_a": (2, 6), "blk.0.attn_q.weight.lora_b": (8, 2),
                     "blk.0.attn_v.weight.lora_a": (2, 6), "blk.0.attn_v.weight.lora_b": (8, 2),
                     "blk.1.attn_q.weight.lora_a": (2, 6), "blk.1.attn_q.weight.lora_b": (8, 2)}
    b = next(x for x in r.tensors if x.name == "blk.0.attn_q.weight.lora_b")
    assert np.allclose(np.array(b.data, dtype=np.float32).reshape(8, 2), t[_MODS[0] + ".lora_B.weight"], atol=1e-2)
    assert dialect.build_gguf({"nope.q_proj": factors[_MODS[0]]}, base, str(tmp_path / "out"), out=err) is None
    assert dialect.build_gguf(factors, str(tmp_path / "missing.gguf"), str(tmp_path / "out"), out=err) is None
    assert "isn't a local file" in err.getvalue()


def test_build_gguf_permutes_q_rows_for_llama(tmp_path, monkeypatch):
    import gguf
    base = _base_gguf(tmp_path / "l.gguf", arch="llama", heads=2)
    A = np.ones((1, 6), np.float32)
    B = np.arange(8, dtype=np.float32).reshape(8, 1)
    mods = {"base_model.model.model.layers.0.self_attn.q_proj": (A, B),
            "base_model.model.model.layers.0.self_attn.v_proj": (A, B)}
    path, _ = dialect.build_gguf(mods, base, str(tmp_path / "out"))
    r = gguf.GGUFReader(path)
    rows = lambda n: np.array(next(x for x in r.tensors if x.name == n).data, dtype=np.float32).reshape(8)
    # convert_hf_to_gguf's llama permute on 8 rows / 2 heads: [0,2,1,3, 4,6,5,7]; v is left alone.
    assert rows("blk.0.attn_q.weight.lora_b").tolist() == [0, 2, 1, 3, 4, 6, 5, 7]
    assert rows("blk.0.attn_v.weight.lora_b").tolist() == list(range(8))


def test_build_peft_dir(tmp_path):
    _, t = _factor_blob()
    factors = {m: (t[m + ".lora_A.weight"], t[m + ".lora_B.weight"]) for m in _MODS}
    d, rank = dialect.build_peft(factors, "org/name", str(tmp_path / "out"), "org/name")
    cfg = json.loads((pathlib.Path(d) / "adapter_config.json").read_text())
    assert rank == 2 and cfg["r"] == cfg["lora_alpha"] == 2 and cfg["target_modules"] == ["q_proj", "v_proj"]
    assert cfg["base_model_name_or_path"] == "org/name"
    from safetensors.numpy import load_file
    saved = load_file(str(pathlib.Path(d) / "adapter_model.safetensors"))
    assert set(saved) == {m + s for m in _MODS for s in (".lora_A.weight", ".lora_B.weight")}


def _fed_env(tmp_path, monkeypatch, status, blob):
    monkeypatch.setattr(transport, "state_dir", lambda: str(tmp_path))
    monkeypatch.setattr(adapter, "state_dir", lambda: str(tmp_path))
    monkeypatch.setattr(notice, "state_dir", lambda: str(tmp_path))   # the privacy sentinel, not ~/.roger
    monkeypatch.setattr("roger.config.load", lambda: {"federations": ["http://f"]})
    pulls, statuses = [], []
    monkeypatch.setattr(transport, "federation_status", lambda url, mid: statuses.append(mid) or status)
    monkeypatch.setattr(transport, "pull", lambda url, cur, mid: pulls.append((cur, mid)) or (blob, "v7"))
    return pulls, statuses


def test_prepare_pulls_once_a_day_and_attaches(tmp_path, monkeypatch):
    import io
    base = _base_gguf(tmp_path / "gemma-4-12B-it-Q4_K_M.gguf")
    blob, _ = _factor_blob()
    pulls, statuses = _fed_env(tmp_path, monkeypatch,
                               {"mode": "bootstrap", "models": ["google/gemma-4-12B-it", "google/gemma-4-E2B-it"]}, blob)
    argv = ["llama-server", "-m", base, "--port", "8080"]
    err = io.StringIO()
    path, rank = adapter.prepare(argv, out=err)
    assert path.endswith(".gguf") and os.path.isfile(path) and rank == 2
    assert pulls == [(None, "google/gemma-4-12B-it")]               # resolved from the gguf path via the allowlist
    assert statuses == [base]                                      # /status is probed with what we know
    st = transport.load_state("http://f", base)
    assert st["cursor"] == "v7" and st["model_id"] == "google/gemma-4-12B-it" and st["last_sync"]
    assert "attaching" in err.getvalue()
    # Same day: no second pull, but the adapter is rebuilt from the persisted global all the same.
    os.remove(path)
    assert adapter.prepare(argv, out=err)[0] == path and os.path.isfile(path) and len(pulls) == 1
    # A federation that doesn't accept the model is never pulled from, and there's nothing to attach.
    pulls2, _ = _fed_env(tmp_path / "other", monkeypatch, {"mode": "unsupported", "models": ["x/y"]}, blob)
    assert adapter.prepare(argv, out=err) is None and pulls2 == []
    # No allowlist advertised (older server): pulled under the raw command-line name.
    pulls3, _ = _fed_env(tmp_path / "old", monkeypatch, {"mode": "busy"}, blob)
    adapter.prepare(argv, out=err)
    assert pulls3 == [(None, base)]
    # An unreachable federation: nothing pulled, day stamped, no adapter, runtime unaffected.
    pulls4, _ = _fed_env(tmp_path / "down", monkeypatch, {}, blob)
    assert adapter.prepare(argv, out=err) is None and pulls4 == []
    assert transport.load_state("http://f", base)["last_sync"]


def test_prepare_reports_dense_global_and_builds_peft_for_vllm(tmp_path, monkeypatch):
    import io
    dense = st_save_np({"base_model.model.model.layers.0.self_attn.q_proj": np.zeros((8, 6), np.float32)},
                       metadata={"model_id": "google/gemma-4-12B-it", "compat": "x"})
    _fed_env(tmp_path, monkeypatch, {"mode": "busy", "models": None}, dense)
    err = io.StringIO()
    assert adapter.prepare(["vllm", "serve", "google/gemma-4-12B-it", "--port", "8000"], out=err) is None
    assert "adapter (LoRA-factor) form" in err.getvalue()
    blob, _ = _factor_blob()
    _fed_env(tmp_path / "v", monkeypatch, {"mode": "busy", "models": ["google/gemma-4-12B-it"]}, blob)
    d, rank = adapter.prepare(["vllm", "serve", "google/gemma-4-12B-it", "--port", "8000"], out=err)
    assert os.path.isfile(os.path.join(d, "adapter_config.json")) and rank == 2
    assert json.loads(open(os.path.join(d, "adapter_config.json")).read())["base_model_name_or_path"] == "google/gemma-4-12B-it"
    # Unknown runtime / no model on the command line / no federations: nothing happens at all.
    assert adapter.prepare(["someserver", "--model", "x", "--port", "1"], out=err) is None
    assert adapter.prepare(["vllm", "serve", "--port", "8000"], out=err) is None
    monkeypatch.setattr("roger.config.load", lambda: {"federations": []})
    assert adapter.prepare(["vllm", "serve", "google/gemma-4-12B-it", "--port", "8000"], out=err) is None


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


def test_grade_reads_the_users_reactions_per_answered_reply(stack, tmp_path):
    base, backend_port, msgs = stack
    history = [{"role": "system", "content": "s"},
               {"role": "user", "content": "fix the bug"},
               {"role": "assistant", "content": "done", "tool_calls": [{"id": "1", "type": "function",
                                                                         "function": {"name": "sh", "arguments": "{}"}}]},
               {"role": "tool", "content": "ok"},
               {"role": "assistant", "content": "fixed it"},
               {"role": "user", "content": "no,\n  it still   crashes"},
               {"role": "assistant", "content": "try this"},
               {"role": "user", "content": "works, thanks!"}]
    path = tmp_path / "conv.json"
    path.write_text(json.dumps(_rec(history, "glad to help")))
    entry = {"path": str(path), "endpoint": "/v1/chat/completions"}
    with httpx.Client() as c:
        res = grader.grade(c, f"http://127.0.0.1:{backend_port}", entry)
    (sent,) = _Upstream.evals
    schema = sent["response_format"]["json_schema"]["schema"]
    assert list(schema["properties"]) == ["reasoning", "scores", "reactions"]
    assert schema["properties"]["reactions"]["required"] == ["reply_1", "reply_2"]
    seed = sent["messages"][-1]["content"]
    assert seed.startswith(grader.SEED) and grader.REACTION_SEED in seed
    assert 'reply_1: the user answered "no, it still crashes"' in seed           # whitespace collapsed
    assert 'reply_2: the user answered "works, thanks!"' in seed
    assert sent["max_tokens"] == grader.MAX_TOKENS + 32
    # credited to the assistant message the user answered (the run's last one, not the tool-calling one),
    # keyed by message index like tool_signals, clamped
    assert res["reactions"] == {"4": -1.0, "6": 0.5}
    assert json.loads(path.read_text())["self_eval"]["reactions"] == res["reactions"]


def test_grade_asks_the_adapter_the_relay_retargets_to(stack):
    # vllm + federation adapter: the capture holds the client's model name, but the relay served the
    # chat from the `roger` adapter — the grade must come from that same policy, not the bare base.
    base, backend_port, msgs = stack
    path, entry = _captured_entry(stack, msgs)
    with httpx.Client() as c:
        grader.grade(c, f"http://127.0.0.1:{backend_port}", entry, model=dialect.LORA_NAME)
    (sent,) = _Upstream.evals
    assert sent["model"] == dialect.LORA_NAME


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
        res = grader.grade(c, f"http://127.0.0.1:{dialect.free_port()}", entry)
    assert res["error"].startswith("ConnectError") and entry["graded"] is True


def test_grade_pending_only_grades_ungraded(stack):
    base, backend_port, msgs = stack
    path, entry = _captured_entry(stack, msgs)
    reg = [entry, {"path": "none", "transcript": [], "endpoint": "/v1/chat/completions", "last": 0.0, "graded": True}]
    assert grader.grade_pending(reg, f"http://127.0.0.1:{backend_port}") == 1
    assert len(_Upstream.evals) == 1 and "scores" in json.loads(path.read_text())["self_eval"]
    assert grader.grade_pending(reg, f"http://127.0.0.1:{backend_port}") == 0
