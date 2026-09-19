"""notice.py — once the runtime is up, tell the user whether the model it serves is one the configured
federations train, and if not, which models they do accept.

The wrapper never sees a model id on the command line (`-m x.gguf`, `--hf-repo …`, `vllm serve org/name`
are all runtime-specific), so the runtime is asked instead, through the one standard surface every
OpenAI-compatible server has: `GET /v1/models`. Its ids are whatever the runtime chose to call the model
(vllm: the HF repo id; llama-server: the gguf path, or `--alias`), so they are matched against a
federation's advertised allowlist by normalised containment rather than equality: as far as the
federation's LoRA is concerned, `gemma-4-12B-it-Q4_K_M.gguf` *is* the `google/gemma-4-12B-it` base.

It also carries the one-time privacy notice (`privacy_notice`), which must be shown before this
client's first-ever federation contact.

Everything here only ever prints (stderr, like the wrapper's other notices) and is fail-soft: an
unreachable federation stays silent.
"""
import os, re, sys, time

import httpx

from roger.federated import CLIENT_VERSION, UPDATE_CMD, transport
from roger.paths import state_dir
from roger.runtime import dialect

_POLL = 2.0        # seconds between /v1/models attempts while the runtime loads

_PRIVACY_URL = "https://github.com/roger-federated/roger-federated/blob/main/PRIVACY.md"


def privacy_notice(cfg: dict, out=sys.stderr) -> None:
    """One-time notice, before this client's first-ever federation activity, of what gets sent to the
    configured federation server(s) and how to opt out. Sentinel in state_dir() so it shows once per
    machine, not once per launch. Not gated on a keypress: the processing basis is legitimate interest
    with a standing right to object (opt out), not consent, so transparency is what's required, not
    sign-off. `CLIENT_VERSION` 2 is the build that adopted it (see federated/__init__.py)."""
    if not cfg.get("federations"):
        return
    sentinel = os.path.join(state_dir(), "privacy_ack")
    if os.path.exists(sentinel):
        return
    print("roger: this client contributes an encrypted gradient update to your configured federation "
          f"server(s) by default (never raw data). Details: {_PRIVACY_URL}\n"
          'roger: opt out anytime by setting "federations": [] in ~/.roger/config.json.', file=out)
    os.makedirs(os.path.dirname(sentinel), exist_ok=True)
    open(sentinel, "w").close()


def served_models(backend_url: str, alive) -> list[str]:
    """Ids from the runtime's `GET /v1/models`, polled until it answers or the runtime is gone. A refused
    connection or 5xx means "still loading" (vllm binds only after the weights are in; llama-server 503s
    most routes meanwhile), a 404/405 means the runtime has no such route — nothing to match against."""
    url = backend_url.rstrip("/") + "/v1/models"
    while alive():
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code == 200:
                ids = [m.get("id") for m in (r.json().get("data") or []) if isinstance(m, dict)]
                # vllm lists the attached federation adapter as a model of its own; it isn't one.
                return [i for i in ids if isinstance(i, str) and i and i != dialect.LORA_NAME]
            if r.status_code in (404, 405):
                return []
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(_POLL)
    return []


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def resolve(served: list[str], accepted: list[str]) -> str | None:
    """The accepted model id that one of the runtime's served ids names, or None. A served id names an
    accepted `org/name` when `name` occurs in it once both are stripped to [a-z0-9] — that survives
    quantization suffixes, file paths, case and separator differences alike. The longest such name wins
    so `…-it-assistant` never resolves to `…-it` (both would match the former)."""
    hits = []
    for a in accepted:
        name = _norm(a.rsplit("/", 1)[-1])
        if name and any(name in _norm(s) for s in served):
            hits.append((len(name), a))
    return max(hits)[1] if hits else None


def announce(served: list[str], cfg: dict, out=sys.stderr) -> None:
    """One /status probe per federation → one line each: accepted (and as which id), or not, plus what
    it does accept so the user can switch, followed by the version verdicts (outdated client, update
    available) — all keyed off what the runtime says it serves."""
    feds = cfg.get("federations") or []
    if not feds:
        return
    if not cfg.get("contribute", True):
        print('roger: "contribute" is off in ~/.roger/config.json: chats are saved but nothing is ever '
              "shared back. Set it to true to pull your weight.", file=out)
    what = ", ".join(served)
    for url in feds:
        st = transport.federation_status(url, served[0] if served else "")
        if not st:
            continue                                # unreachable: fail-soft, nothing to say
        accepted = st.get("models")                 # None: no allowlist, or a server predating the field
        if not served:
            # The runtime never told us its model: can't judge, but the list is still the useful part.
            if accepted:
                print(f"roger: {url} trains {', '.join(accepted)}; couldn't tell which model this runtime "
                      "serves (no /v1/models), so make sure it is one of these to contribute.", file=out)
        elif accepted is None:
            # Old server, or one accepting any model: only its own verdict on the raw id is available.
            if st.get("mode", "busy") == "unsupported":
                print(f"roger: ⚠ {url} doesn't accept {what}: chats are still saved but won't train the "
                      "federation. Pick a model it accepts to contribute and receive its updates.", file=out)
            else:
                print(f"roger: ✓ {url} accepts {what}; the chats saved here will train it.", file=out)
        elif (hit := resolve(served, accepted)) is not None:
            print(f"roger: ✓ {url} trains {hit}, which this runtime serves as {what}; the chats saved "
                  "here will feed it.", file=out)
        else:
            print(f"roger: ⚠ {url} doesn't accept {what}: chats are still saved but won't train the "
                  f"federation. Models it accepts: {', '.join(accepted) or 'none right now'} — run one of "
                  "these to contribute and receive its updates.", file=out)
        if CLIENT_VERSION < int(st.get("min_client", 0) or 0):
            print(f"roger: ⚠ your roger client is out of date: {url} rejects this version's gradients. "
                  f"Update to keep contributing:\n  {UPDATE_CMD}", file=out)
        elif int(st.get("latest_client", 0) or 0) > CLIENT_VERSION:
            print(f"roger: a newer roger client is available. Update with: {UPDATE_CMD}", file=out)


def run(backend_url: str, alive) -> None:
    """Daemon-thread body for the wrapper: wait for the runtime, then print the verdicts. Swallows
    everything — a notice must never take the relay down or spray a traceback over the runtime's output."""
    try:
        from roger import config              # first run writes the default config = default federation
        cfg = config.load()
        if not cfg.get("federations"):
            return
        served = served_models(backend_url, alive)
        if alive():                                # a runtime that died while we waited has nothing to hear
            announce(served, cfg)
    except Exception:
        pass
