"""train.py — the automatic training round, run once the runtime has shut down.

While the runtime is up it owns the GPU, so the round waits for it to exit (Ctrl-C or on its own) and
then, if enough graded conversations for the model it served have piled up (`train_every`, default 8),
trains in the foreground: the chosen federation's global is refreshed, the model is loaded from the very
file/cache the runtime served (training/wire_trainer.py — the torch half, imported only when the gate
passes, so the wrapper stays light), and the epoch's factor update is uploaded to that one federation.
Conversations are deleted once the upload is accepted — each with the earlier capture files its newest one
supersedes — and kept otherwise; Ctrl-C skips the round and keeps them too.

A conversation is the chain of capture files one chat produced (every request re-sends the full history,
so the newest file of a chat contains all earlier ones); it is ready once its newest file carries the
model's self-evaluation scores (runtime/grader.py).
"""
import glob, json, os, sys

from roger.federated import CLIENT_VERSION, UPDATE_CMD, transport
from roger.runtime import adapter, capture, grader, notice


# ---------------------------------------------------------------------------
# Conversations on disk
# ---------------------------------------------------------------------------

def conversations(provider: str, served: str) -> list[dict]:
    """[{path, record, superseded: [earlier files of the same chat]}] for every chat captured while
    `served` was the served model, newest file only. `record["_path"]` is set for the trainer."""
    files = []
    for path in sorted(glob.glob(os.path.join(capture.messages_dir(provider), "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, ValueError):
            continue
        reply = grader._reply(rec)
        if (rec.get("served_model") != served or reply is None
                or not str(rec.get("endpoint", "")).endswith("/chat/completions")):
            continue
        req = grader._key(rec["request"].get("messages") or [])
        rec["_path"] = path
        files.append((path, rec, req, req + [("assistant", grader._text(reply.get("content")))]))
    out = []
    for path, rec, req, transcript in files:
        # Superseded: another file's request continues past this file's whole transcript.
        if any(len(r) > len(transcript) and r[: len(transcript)] == transcript for _, _, r, _ in files):
            continue
        prior = [p for p, _, _, t in files if p != path and req[: len(t)] == t]
        out.append({"path": path, "record": rec, "superseded": prior})
    return out


def ready(convs: list[dict]) -> list[dict]:
    """Conversations the model has graded (a failed grade stores an `error`, not scores)."""
    return [c for c in convs if (c["record"].get("self_eval") or {}).get("scores")]


def discard(convs: list[dict]) -> None:
    """Delete trained conversations, earlier prefix files included (consume-once, like the legacy runs)."""
    for c in convs:
        for p in [c["path"], *c["superseded"]]:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# The round
# ---------------------------------------------------------------------------

def _federation(feds: list[str], served: str) -> tuple[tuple[str, str, dict] | None, list[tuple[str, str]]]:
    """((url, model_id, /status) of the one federation to train for, or None; [(url, why) skipped before
    it]). The pick is the first, in `federations` config order (so the order is the user's priority),
    that trains this model and accepts this client. A member trains against — and contributes to —
    exactly one federation, since its Δ only means something next to that federation's frozen factor."""
    skipped = []
    for url in feds:
        probe = transport.federation_status(url, served)
        if not probe:
            skipped.append((url, "unreachable"))
            continue
        accepted = probe.get("models")
        model_id = notice.resolve([served], accepted) if accepted is not None else served
        st = transport.federation_status(url, model_id) if model_id is not None else {}
        if model_id is None or st.get("mode", "busy") == "unsupported":
            skipped.append((url, f"doesn't train {os.path.basename(served)}"))
        elif CLIENT_VERSION < int(st.get("min_client", 0) or 0):
            skipped.append((url, f"needs a newer roger — update: {UPDATE_CMD}"))
        elif not st:
            skipped.append((url, "unreachable"))
        else:
            return (url, model_id, st), skipped
    return None, skipped


def _refresh(url: str, served: str, model_id: str) -> bytes | None:
    """The federation's current global, pulled now whatever the day (the epoch may have moved since the
    morning's pull); falls back to the persisted copy when nothing new or unreachable."""
    st = transport.load_state(url, served)
    res = transport.pull(url, st.get("cursor"), model_id)
    if res is not None:
        blob, st["cursor"] = res
        transport.save_global(url, blob, served)
        transport.save_state(url, st, served)
    return transport.load_global(url, served)


def maybe_train(provider: str, served: str | None, cfg: dict, out=sys.stderr) -> None:
    feds = cfg.get("federations") or []
    if served is None or not feds or not cfg.get("contribute", True):
        return                                       # nothing to send a gradient to: never train
    convs = ready(conversations(provider, served))
    if len(convs) < int(cfg.get("train_every", 8)):
        return
    fed, skipped = _federation(feds, served)
    # The config order is the user's priority, so every federation passed over is named with its reason.
    for url, why in skipped:
        print(f"roger: not training for {url}: {why}.", file=out)
    if fed is None:
        print("roger: no configured federation can take this model's training; keeping its conversations.",
              file=out)
        return
    if skipped:
        print(f"roger: training for {fed[0]} instead (next in your federations list).", file=out)
    url, model_id, _ = fed
    blob = _refresh(url, served, model_id)
    status = transport.federation_status(url, model_id)   # the epoch/phase to train, read right before
    if not status:
        print(f"roger: {url} is unreachable; training later.", file=out)
        return
    if blob is not None:
        meta = adapter._metadata(blob)
        if int(meta.get("epoch", status.get("epoch", 0))) != int(status.get("epoch", 0)):
            print("roger: the federation moved to a new epoch mid-pull; training later.", file=out)
            return
    print(f"roger: training {model_id} on {len(convs)} conversation(s) for {url} "
          "(Ctrl-C to skip; they're kept)…", file=out)
    from roger.federated import client as fed_client
    from roger.training import wire_trainer
    try:
        res = wire_trainer.train([c["record"] for c in convs], served, model_id, blob, status)
        if not res["trained"]:
            print(f"roger: not training this time: {res['reason']}.", file=out)
            return
        ok = fed_client.contribute_factor(url, res["update"], res["base"], res["epoch"], model_id,
                                          status.get("mode", "busy"))
    except KeyboardInterrupt:
        print("roger: training skipped; the conversations are kept for next time.", file=out)
        return
    except Exception as e:
        print(f"roger: training failed ({e.__class__.__name__}: {e}); conversations kept.", file=out)
        return
    if ok:
        discard(convs)
        print(f"roger: contributed a {res['phase']}-factor update from {res['n_episodes']} conversation(s) "
              f"(mean return {res['mean_return']:+.2f}); they've been deleted.", file=out)
    else:
        print(f"roger: {url} didn't accept the update; the conversations are kept for next time.", file=out)

