"""capture.py — turn a relayed OpenAI-style chat exchange into one JSON file under ~/.roger/messages/.

Each request already carries the full conversation (the chat API is stateless), so one file per
exchange is the complete record; earlier prefixes of the same conversation become obsolete and are
left for a later dedup pass. A streamed reply is reassembled into the object the non-streaming API
would have returned, so consumers see one shape regardless of how the client asked for it.
"""
import json, os, re, tempfile, uuid
from datetime import datetime, timezone
from typing import Callable

from roger.paths import state_dir
from roger.runtime import signals

# Suffix match, not exact: llama-server also serves `/chat/completions` without the `/v1`, and some
# proxies mount the API under a prefix. `/completions` covers the legacy text endpoint too.
_CHAT_SUFFIXES = ("/chat/completions", "/completions", "/responses")


def wants(method: str, path: str) -> bool:
    return method == "POST" and path.split("?", 1)[0].rstrip("/").endswith(_CHAT_SUFFIXES)


def messages_dir(provider: str) -> str:
    return os.path.join(state_dir(), "messages", provider)


# ---------------------------------------------------------------------------
# SSE reassembly
# ---------------------------------------------------------------------------

def _events(body: str):
    for block in re.split(r"\r?\n\r?\n", body):
        data = [line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")]
        if not data:
            continue   # comments, `event:` lines, keep-alives
        payload = "\n".join(data)
        if payload.strip() == "[DONE]":
            continue
        try:
            yield json.loads(payload)
        except ValueError:
            continue


def _merge_tool_calls(calls: list, deltas: list) -> None:
    for pos, d in enumerate(deltas):
        idx = d.get("index", pos)
        while len(calls) <= idx:
            calls.append({"index": len(calls), "function": {"name": "", "arguments": ""}})
        tc = calls[idx]
        for k, v in d.items():
            if v is None or k == "index":
                continue
            if k == "function":
                if v.get("name"):
                    tc["function"]["name"] = v["name"]          # sent once, in the first delta
                if v.get("arguments"):
                    tc["function"]["arguments"] += v["arguments"]
            else:
                tc[k] = v                                        # id, type


def _merge_delta(msg: dict, delta: dict) -> None:
    # Deltas are declared increments, so every string field concatenates — that covers `content`,
    # `reasoning_content`, `reasoning`, and whatever a vendor adds — without naming any of them.
    for k, v in delta.items():
        if v is None:
            continue
        if k == "tool_calls":
            _merge_tool_calls(msg.setdefault("tool_calls", []), v)
        elif k == "role":
            msg.setdefault("role", v)
        elif isinstance(v, str):
            msg[k] = msg.get(k, "") + v
        else:
            msg[k] = v


def assemble_sse(body: str) -> dict | None:
    """Fold a `text/event-stream` body into the equivalent non-streaming response object."""
    chunks = list(_events(body))
    if not chunks:
        return None
    # Responses API: lifecycle events; the terminal one (`response.completed` / `.incomplete` /
    # `.failed`) carries the entire final object, so nothing needs merging.
    finals = [c["response"] for c in chunks
              if isinstance(c.get("response"), dict) and str(c.get("type", "")).startswith("response.")
              and c["type"] not in ("response.created", "response.in_progress", "response.queued")]
    if finals:
        return finals[-1]
    if not any("choices" in c for c in chunks):
        return None
    out = {k: v for k, v in chunks[0].items() if k != "choices"}
    if out.get("object") == "chat.completion.chunk":
        out["object"] = "chat.completion"
    choices: dict[int, dict] = {}
    usage = None
    for c in chunks:
        if c.get("usage"):
            usage = c["usage"]                                   # only the last chunk carries it
        for pos, ch in enumerate(c.get("choices") or []):
            idx = ch.get("index", pos)
            acc = choices.setdefault(idx, {"index": idx})
            for k, v in ch.items():
                if v is None or k == "index":
                    continue
                if k == "delta":
                    _merge_delta(acc.setdefault("message", {}), v)
                elif k == "text":                                # legacy /v1/completions
                    acc["text"] = acc.get("text", "") + v
                elif k == "logprobs" and isinstance(v, dict):
                    acc.setdefault("logprobs", {"content": []})["content"].extend(v.get("content") or [])
                else:
                    acc[k] = v                                   # finish_reason: last non-null wins
    out["choices"] = [choices[i] for i in sorted(choices)]
    if usage is not None:
        out["usage"] = usage
    return out


def assemble(content_type: str, body: bytes) -> tuple[dict | None, bool]:
    """(response object or None, was_streamed). Streaming is detected from the wire, not from the
    request's `stream` flag, so a runtime that ignores the flag is still recorded correctly."""
    text = body.decode("utf-8", "replace")
    if "text/event-stream" in content_type.lower() or text.lstrip().startswith(("data:", ":")):
        return assemble_sse(text), True
    try:
        obj = json.loads(text)
    except ValueError:
        return None, False
    return (obj if isinstance(obj, dict) else None), False


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_exchange(provider: str, record: dict) -> str:
    """Atomic write (tmp + rename) so a crash mid-write never leaves a half file for the trainer."""
    d = messages_dir(provider)
    os.makedirs(d, exist_ok=True)
    # No colons in the name (Windows), sorts chronologically; the suffix disambiguates concurrent
    # connections finishing in the same microsecond.
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:6] + ".json"
    path = os.path.join(d, name)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


def update_record(path: str, patch: dict) -> None:
    """Merge top-level keys into an existing exchange file, atomically. A file that vanished
    meanwhile (a dedup pass, the user tidying up) is simply skipped."""
    try:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
    except FileNotFoundError:
        return
    record.update(patch)
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def make_sink(provider: str, on_record: Callable[[str, dict], None] | None = None,
              served: str | None = None) -> Callable[[str, bytes, str, int, bytes], None]:
    """proxy.Sink that records one file per chat exchange under messages/<provider>/, then hands
    (path, record) to `on_record` — the grader's conversation tracking. `served`: the model as the
    runtime's command line names it (dialect.served_model), recorded so the trainer knows which base
    produced a chat — the request's own `model` is whatever alias the client chose."""
    def sink(path: str, request: bytes, content_type: str, status: int, response: bytes) -> None:
        try:
            req = json.loads(request)
        except ValueError:
            return
        if not isinstance(req, dict):
            return                                               # nothing chat-shaped to keep
        resp, streamed = assemble(content_type, response)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "endpoint": path.split("?", 1)[0],
            "model": req.get("model") or (resp or {}).get("model"),
            "served_model": served,
            "stream": streamed,
            "request": req,
            "response": resp,
            # Verifiable per-step rewards from the tool results in the history (exit codes, errors).
            # Every request carries the full history, so the newest file of a conversation has them all.
            "tool_signals": signals.step_rewards(req.get("messages") or req.get("input")),
        }
        if resp is None:                                         # never lose data to a parser gap
            record["raw_response"] = response.decode("utf-8", "replace")
        written = write_exchange(provider, record)
        if on_record is not None:
            on_record(written, record)
    return sink
