"""grader.py — the model's own end-of-session self-evaluation, injected over the wire.

The later optimisation step needs a reward per conversation, and the only judge available is the model
itself: once a conversation has gone idle, the runtime is asked — through its own chat API, never via the relay, so nothing is recorded as a chat —
to grade it: the conversation and the model's last reply, then ONE user message asking for the grade. A
JSON schema whose properties are ordered `reasoning` then `scores` is the wire form of "reason-then-force":
the model reasons in free text, then the grammar forces one number per metric. Only the per-metric scores
are stored; averaging them into a reward is the optimiser's job.

It is the same request on every runtime, so members of one federation are graded on one scale. An
assistant prefill (the seed continued in the model's own voice) was dropped: llama-server replays a prefill
through the json_schema grammar and rejects any prose one, and a separate seed message means two assistant
turns in a row. Thinking is switched off (chat_template_kwargs), `reasoning` is capped so the JSON always
closes inside max_tokens, and an unparseable answer is asked for once more. On gemma-4-E2B (Q4_0,
llama-server) this parsed 42/42 on the first try.

The same call also reads the user's reactions — the only human signal available, since roger can't
prompt the user inside a third-party client. Every assistant reply the user answered
gets one label, judged from the user's own words (a correction, a repeated request, a thank-you), not
from the model's opinion of its reply — so it stays a human signal. Each answered reply is named in the
request with a snippet of the user's answer and forced as its own schema property, so the grammar can't
skip or misalign one; they're stored as `reactions` (-1/0/1) keyed by the reply's message index (the
same keys as `tool_signals`). The newest reply has no answer yet: it gets one if the chat continues and
is graded again at its new end.

The reactions are named labels, forced BEFORE `reasoning`: as a bare -1..1 number emitted after the
model had argued its own work was good, a small model scored pushback +1 (gemma-4-E2B did on every
pushback chat in an external test), handing the worst chats the maximum return.
"""
import json, sys, threading, time
from datetime import datetime, timezone

import httpx

from roger.runtime import capture

METRICS = {"efficiency": "how directly you reached the goal, with very few wasted, wrong or redundant steps",
           "accuracy": "how correct the result is",
           "completeness": "how fully the result covers what was asked"}
ASK = ("Grade honestly how well you completed the task in the conversation above, one score per criterion: "
       + "; ".join(f"{m}: {d}" for m, d in METRICS.items()) + ". For each: 1 for a clean, fully-correct "
       "solve, around 0 for partial or clumsy, negative if you largely failed.")
# "still fails" and frustration are named: without them, "Ugh, the third time, it still crashes" was read as
# moving on (gemma-4-E2B, every repeat).
REACTION_ASK = ("Before grading, judging only from the user's own words and not from how good you think your "
                "reply was, label how the user reacted to each of your replies they answered: pushed_back if "
                "they said it was wrong or didn't work, said it still fails or is the same problem again, "
                "corrected you, objected, complained, or had to repeat or rephrase their request; confirmed if "
                "they said it worked or thanked you; neutral if they simply moved on. The replies they answered: ")
REACTIONS = {"pushed_back": -1.0, "neutral": 0.0, "confirmed": 1.0}   # label → stored reaction
SNIPPET = 160       # chars of each user answer quoted in the request, enough to tell the replies apart
IDLE_S = 300        # a conversation nobody has extended for this long is over
TICK_S = 15
MAX_TOKENS = 512
REASONING_MAX = 800  # chars: the JSON closes inside MAX_TOKENS, and a repetition loop is cut short
ATTEMPTS = 2        # an unparseable answer is asked for once more
# Property order matters: the grammar emits `reasoning` before `scores`. No minimum/maximum — llama.cpp's
# schema→grammar converter only honours them for integers — so the scores are clamped client-side.
SCHEMA = {"type": "object",
          "properties": {"reasoning": {"type": "string", "maxLength": REASONING_MAX},
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
        return                                            # /completions or responses: no message list to grade
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
    """(ask, schema, answered-reply indices) for one conversation. Without answered replies it is
    exactly ASK + SCHEMA; otherwise the reaction request comes before ASK and a `reactions` object with
    one forced `reply_N` label per answered reply is put before `reasoning`."""
    answered = _answered(rec["request"].get("messages") or [])
    if not answered:
        return ASK, SCHEMA, []
    quotes = "; ".join(f'reply_{n}: the user answered "{" ".join(text.split())[:SNIPPET]}"'
                       for n, (_, text) in enumerate(answered, 1))
    names = [f"reply_{n}" for n in range(1, len(answered) + 1)]
    schema = json.loads(json.dumps(SCHEMA))
    reactions = {"type": "object",
                 "properties": {k: {"type": "string", "enum": list(REACTIONS)} for k in names},
                 "required": names, "additionalProperties": False}
    schema["properties"] = {"reactions": reactions, **schema["properties"]}
    schema["required"].insert(0, "reactions")
    return REACTION_ASK + quotes + ". " + ASK, schema, [i for i, _ in answered]


def _conversation(rec: dict) -> list[dict]:
    """The graded conversation: what the client sent, plus the reply the runtime gave."""
    reply = _reply(rec)
    last = {k: reply[k] for k in ("role", "content", "tool_calls") if reply.get(k) is not None}
    last.setdefault("role", "assistant")
    return list(rec["request"].get("messages") or []) + [last]


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
        ask, schema, answered = _prompt(rec)
        conv = _conversation(rec)
        body = {"model": model or rec.get("model"), "messages": conv + [{"role": "user", "content": ask}],
                "stream": False,
                # room for the reactions after the free-text reasoning: ~16 tokens per forced property
                "max_tokens": MAX_TOKENS + 16 * len(answered),
                # a thinking model would otherwise spend the budget before the JSON (ignored elsewhere)
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "self_evaluation", "schema": schema}}}
        for attempt in range(1, ATTEMPTS + 1):
            resp = client.post(backend.rstrip("/") + entry["endpoint"], json=body)
            if 400 <= resp.status_code < 500:             # the request itself is refused: asking again won't help
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            try:
                if resp.status_code >= 300:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                obj = json.loads(resp.json()["choices"][0]["message"]["content"])
                scores = {m: _clamp(obj["scores"][m]) for m in METRICS}
                # a runtime that ignored the enum: unknown labels count as neutral, not as a failed grade
                reactions = {str(i): REACTIONS.get(obj["reactions"][f"reply_{n}"], 0.0)
                             for n, i in enumerate(answered, 1)}
                break
            except Exception:                             # bad JSON, a missing metric, a 5xx: once more
                if attempt == ATTEMPTS:
                    raise
        result.update(scores=scores, reasoning=obj.get("reasoning", ""), attempts=attempt,
                      graded_messages=len(conv))
        if answered:
            result["reactions"] = reactions
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
