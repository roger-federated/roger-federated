"""transport.py — per-federation HTTP I/O + on-disk sync state.

Defines the client side of the wire protocol (the aggregation server is future work; see
federated_server_requirements). Everything fails soft: a federation that is unreachable or
misbehaving returns None / a status string rather than raising, so a sharing hiccup never takes
down the agent (same convention as the web_search/web_fetch tools).

Endpoints (all under a federation's base URL, served over HTTPS):
  POST {url}/round/register   {model_id, pubkey(hex)} -> {peers: [hex, ...]}   (server distributes
                              the round's peer X25519 public keys; keys are collected centrally)
  POST {url}/contribute       octet-stream = the masked, packed contribution -> 200
  GET  {url}/global?since=&model_id=  -> 200 octet-stream (re-factored global adapter) + X-Cursor
                              header, or 204 when nothing new since `since`.
"""
import hashlib, json, os

import httpx

from roger.agency.path_utils import state_dir

_TIMEOUT = 30.0


def _fed_path(url: str, ext: str) -> str:
    d = os.path.join(state_dir(), "federated")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, hashlib.sha1(url.encode()).hexdigest() + ext)


def _state_path(url: str) -> str:
    return _fed_path(url, ".json")


def load_state(url: str) -> dict:
    try:
        with open(_state_path(url)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(url: str, state: dict) -> None:
    with open(_state_path(url), "w") as f:
        json.dump(state, f, indent=2)


def save_global(url: str, blob: bytes) -> None:
    """Persist the federation's current cumulative global ΔW so it can be re-folded at every load
    without re-downloading; refreshed only when a new day's pull returns fresh bytes."""
    with open(_fed_path(url, ".global"), "wb") as f:
        f.write(blob)


def load_global(url: str) -> bytes | None:
    try:
        with open(_fed_path(url, ".global"), "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


def register_and_peers(url: str, my_pub: bytes, model_id: str) -> list[bytes] | None:
    """Announce our round public key and get back the round's peer keys. None on any failure (so the
    caller skips this federation rather than uploading an unmaskable contribution)."""
    try:
        r = httpx.post(f"{url.rstrip('/')}/round/register", timeout=_TIMEOUT,
                       json={"model_id": model_id, "pubkey": my_pub.hex()})
        r.raise_for_status()
        return [bytes.fromhex(h) for h in r.json().get("peers", [])]
    except Exception:
        return None


def contribute(url: str, blob: bytes) -> str:
    try:
        r = httpx.post(f"{url.rstrip('/')}/contribute", content=blob, timeout=_TIMEOUT,
                       headers={"Content-Type": "application/octet-stream"})
        r.raise_for_status()
        return "ok"
    except Exception as e:
        return f"failed: {e}"


def pull(url: str, cursor: str | None, model_id: str) -> tuple[bytes, str] | None:
    """Fetch the aggregated global since `cursor`. Returns (bytes, new_cursor) or None (nothing new /
    unreachable)."""
    try:
        r = httpx.get(f"{url.rstrip('/')}/global", timeout=_TIMEOUT,
                      params={"since": cursor or "", "model_id": model_id})
        if r.status_code == 204:
            return None
        r.raise_for_status()
        return r.content, r.headers.get("X-Cursor", cursor or "")
    except Exception:
        return None
