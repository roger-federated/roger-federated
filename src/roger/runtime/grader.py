"""grader.py — the model's own end-of-session self-evaluation, injected over the wire.

The later optimisation step needs a reward per conversation, and the only judge available is the model
itself: once a conversation has gone idle, the runtime is asked — through its own chat API, never via the relay, so nothing is recorded as a chat —
to continue an assistant message that starts with a self-grading seed. A JSON schema whose properties
are ordered `reasoning` then `scores` is the wire form of "reason-then-force": the model reasons in
free text, then the grammar forces one number per metric. Only the per-metric scores are stored;
averaging them into a reward is the optimiser's job.

The same call also reads the user's reactions — the only human signal available, since roger can't
prompt the user inside a third-party client. Every assistant reply the user answered
gets one number, judged from the user's own words (a correction, a repeated request, a thank-you), not
from the model's opinion of its reply — so it stays a human signal. Each answered reply is named in the
seed with a snippet of the user's answer and forced as its own schema property, so the grammar can't
skip or misalign one; they're stored as `reactions` keyed by the reply's message index (the same keys
as `tool_signals`). The newest reply has no answer yet: it gets one if the chat continues and is graded
again at its new end.
"""
import json, sys, threading, time
from datetime import datetime, timezone

import httpx

from roger.runtime import capture

METRICS = {"efficiency": "how directly I reached the goal, with very few wasted, wrong or redundant steps",
           "accuracy": "how correct the result is",
           "completeness": "how fully the result covers what was asked"}
SEED = ("Let me honestly grade how well I completed the task in this conversation, one score per "
        "criterion — " + "; ".join(f"{m}: {d}" for m, d in METRICS.items()) + ". For each: 1 for a "
        "clean, fully-correct solve, around 0 for partial or clumsy, negative if I largely failed.")
REACTION_SEED = ("Separately, and judging only from the user's own words rather than from how good I think "
                 "my reply was, how did the user react to each reply of mine they answered: -1 if they said "
                 "it was wrong or didn't work, or had to repeat or rephrase their request; 1 if they confirmed "
                 "it worked or thanked me; 0 if neutral or they simply moved on. The replies they answered: ")
SNIPPET = 160       # chars of each user answer quoted in the seed, enough to tell the replies apart
IDLE_S = 300        # a conversation nobody has extended for this long is over
TICK_S = 15
MAX_TOKENS = 512
# Property order matters: the grammar emits `reasoning` before `scores`. No minimum/maximum — llama.cpp's
# schema→grammar converter only honours them for integers — so the scores are clamped client-side.
SCHEMA = {"type": "object",
          "properties": {"reasoning": {"type": "string"},
                         "scores": {"type": "object",
                                    "properties": {m: {"type": "number"} for m in METRICS},
                                    "required": list(METRICS), "additionalProperties": False}},
          "required": ["reasoning", "scores"], "additionalProperties": False}

_LOCK = threading.Lock()   # registry is touched by every relay thread (track) and the grader (due)


# ---------------------------------------------------------------------------
# Conversation registry: chaining exchanges into conversations
# ---------------------------------------------------------------------------

def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, sort_keys=True)          # multimodal part lists


def _key(messages: list) -> list[tuple[str, str]]:
    """Identity used for prefix comparison: role + content only, since clients re-send exactly that
    and may drop reasoning fields or tool-call ids on the way back."""
    return [(m.get("role", ""), _text(m.get("content"))) for m in messages]


def _reply(record: dict) -> dict | None:
    try:
        return record["response"]["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None


def track(registry: list, path: str, record: dict) -> None:
    """Fold a freshly captured exchange into the conversation it extends, or open a new one.
    Every request carries the full history, so the previous exchange's transcript (its request +
    reply) is a prefix of the next request in the same chat."""
    reply = _reply(record)
    if not str(record.get("endpoint", "")).endswith("/chat/completions") or reply is None:
        return                                            # /completions or responses: no message list to prefill
    req_key = _key(record["request"].get("messages") or [])
    transcript = req_key + [("assistant", _text(reply.get("content")))]
    with _LOCK:
        for e in registry:
            if req_key[:len(e["transcript"])] == e["transcript"]:
                # Same chat, extended — grade it (again) at its new boundary, on its newest file.
                e.update(path=path, transcript=transcript, endpoint=record["endpoint"],
                         last=time.time(), graded=False)
                return
        registry.append({"path": path, "transcript": transcript, "endpoint": record["endpoint"],
                         "last": time.time(), "graded": False})


def due(registry: list, now: float) -> list[dict]:
    with _LOCK:
        return [e for e in registry if not e["graded"] and now - e["last"] >= IDLE_S]


# ---------------------------------------------------------------------------
# The eval call
# ---------------------------------------------------------------------------

def _clamp(x) -> float:
    return max(-1.0, min(1.0, float(x)))


def _answered(msgs: list) -> list[tuple[int, str]]:
    """[(index of an assistant message, the user message that directly answered it)]. In an agentic run
    the user speaks after the run's last assistant message (tool results sit in between the others), so
    that message is the one the reaction is credited to."""
    return [(i - 1, _text(msgs[i].get("content"))) for i in range(1, len(msgs))
            if msgs[i].get("role") == "user" and msgs[i - 1].get("role") == "assistant"]


def _prompt(rec: dict) -> tuple[str, dict, list[int]]:
    """(seed, schema, answered-reply indices) for one conversation. Without answered replies it is
    exactly SEED + SCHEMA; otherwise the reaction request is appended to the seed and a `reactions`
    object with one forced `reply_N` property per answered reply is added after `scores`."""
    answered = _answered(rec["request"].get("messages") or [])
    if not answered:
        return SEED, SCHEMA, []
    quotes = "; ".join(f'reply_{n}: the user answered "{" ".join(text.split())[:SNIPPET]}"'
                       for n, (_, text) in enumerate(answered, 1))
    names = [f"reply_{n}" for n in range(1, len(answered) + 1)]
    schema = json.loads(json.dumps(SCHEMA))
    schema["properties"]["reactions"] = {"type": "object",
                                         "properties": {k: {"type": "number"} for k in names},
                                         "required": names, "additionalProperties": False}
    schema["required"].append("reactions")
    return SEED + " " + REACTION_SEED + quotes + ".", schema, [i for i, _ in answered]


def _eval_messages(rec: dict, appended: bool, seed: str = SEED) -> list[dict]:
    reply = _reply(rec)
    last = {k: reply[k] for k in ("role", "content", "tool_calls") if reply.get(k) is not None}
    last.setdefault("role", "assistant")
    msgs = list(rec["request"].get("messages") or [])
    if appended:
        # Alternation-strict chat templates (gemma family) refuse two assistant messages in a row, so
        # the seed rides at the end of the reply instead of opening its own message.
        last["content"] = (_text(last.get("content")) + "\n\n" + seed).strip()
        return msgs + [last]
    return msgs + [last, {"role": "assistant", "content": seed}]


def grade(client: httpx.Client, backend: str, entry: dict, model: str | None = None) -> dict:
    """Ask the runtime to grade one conversation and persist the verdict on its newest file.
    `model`: the name the relay retargets chat requests to (Plan.request_model, the vllm adapter). The
    capture keeps what the client sent, so without it this direct call would be graded by the bare base
    rather than the policy that actually produced the conversation.
    Never raises: a failure is recorded as `{"error": …}` so the conversation isn't retried forever."""
    ts = datetime.now(timezone.utc).isoformat()
    result: dict = {"ts": ts}
    try:
        with open(entry["path"], encoding="utf-8") as f:
            rec = json.load(f)
        resp = None
        seed, schema, answered = _prompt(rec)
        for appended in (False, True):
            msgs = _eval_messages(rec, appended, seed)
            body = {"model": model or rec.get("model"), "messages": msgs, "stream": False,
                    # room for the reactions after the free-text reasoning: ~16 tokens per forced property
                    "max_tokens": MAX_TOKENS + 16 * len(answered),
                    "response_format": {"type": "json_schema",
                                        "json_schema": {"name": "self_evaluation", "schema": schema}}}
            resp = client.post(backend.rstrip("/") + entry["endpoint"], json=body)
            if 200 <= resp.status_code < 300 or resp.status_code >= 500:
                break                                     # 4xx → try the merged-seed placement once
        if not 200 <= resp.status_code < 300:
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
        else:
            obj = json.loads(resp.json()["choices"][0]["message"]["content"])
            result.update(scores={m: _clamp(obj["scores"][m]) for m in METRICS},
                          reasoning=obj.get("reasoning", ""),
                          seed_placement="appended" if appended else "message",
                          graded_messages=len(msgs) - 1)
            if answered:
                result["reactions"] = {str(i): _clamp(obj["reactions"][f"reply_{n}"])
                                       for n, i in enumerate(answered, 1)}
    except Exception as e:                                # httpx errors, bad JSON, missing metric, …
        result["error"] = f"{e.__class__.__name__}: {str(e)[:200]}"
    capture.update_record(entry["path"], {"self_eval": result})
    with _LOCK:
        entry["graded"] = True
    return result


def _client() -> httpx.Client:
    # A long-context prefill on a small GPU can take minutes before the first byte: no read timeout.
    return httpx.Client(timeout=httpx.Timeout(connect=5.0, read=None, write=None, pool=None))


def run(registry: list, backend: str, stop: threading.Event, model: str | None = None) -> None:
    """Thread target: grade idle conversations until `stop` is set."""
    with _client() as client:
        while not stop.wait(TICK_S):
            for e in due(registry, time.time()):
                if stop.is_set():
                    return
                try:
                    grade(client, backend, e, model)
                except Exception as e2:               # belt and braces: grading never kills the relay
                    print(f"roger: self-grading failed: {e2!r}", file=sys.stderr)


def grade_pending(registry: list, backend: str, model: str | None = None) -> int:
    """Shutdown path: grade everything still ungraded while the runtime is up. A KeyboardInterrupt
    propagates to the caller, which treats it as "skip the rest"."""
    with _LOCK:
        pending = [e for e in registry if not e["graded"]]
    if not pending:
        return 0
    print(f"roger: grading {len(pending)} conversation(s) before shutdown (Ctrl-C again to skip)",
          file=sys.stderr)
    with _client() as client:
        for e in pending:
            grade(client, backend, e, model)
    return len(pending)
