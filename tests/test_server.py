"""Tests for the federated aggregation server (roger.federated.server).

CPU-only, download-free. The core tests drive the synchronous Aggregator directly (deterministic, no
event loop); the HTTP test exercises the FastAPI wire layer + the register/seal barrier via concurrent
TestClient calls. Synthetic clients reuse the real client crypto (secure_agg.quantize/mask), so the
mask-cancellation the server relies on is genuinely tested end-to-end.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from safetensors.torch import save as st_save

from roger.federated import delta, secure_agg
from roger.federated.server.aggregate import Aggregator

KEY = "base_model.model.model.layers.0.self_attn.q_proj"


def _mask_cohort(payloads):
    """Mirror the client: quantize each dense ΔW, pairwise-mask against the whole cohort's pubkeys.
    Returns (uploads, pubs) where uploads = [(masked int64, compat, spec_json)]."""
    pairs = [secure_agg.gen_keypair() for _ in payloads]
    pubs = [pub for _, pub in pairs]
    uploads = []
    for (priv, _), p in zip(pairs, payloads):
        q, spec = secure_agg.quantize(p)
        masked = secure_agg.mask(q, priv, pubs)
        uploads.append((masked, delta.compat_hash(p), json.dumps([[k, list(s)] for k, s in spec])))
    return uploads, pubs


def _pack(masked, compat, spec_json, round_id, model_id="m"):
    return st_save({"masked": masked}, metadata={"model_id": model_id, "compat": compat,
                                                  "spec": spec_json, "round_id": round_id})


def _seal_with(agg, pubs, ip="1.2.3.4", model="m"):
    rnd = None
    for pub in pubs:
        rnd = agg.add_registrant(model, pub.hex(), ip, now=0.0)
    agg.try_seal(rnd)
    return rnd


# --- core: recovery + FedAvg accumulation ----------------------------------------------------

def test_aggregator_recovers_mean(tmp_path):
    torch.manual_seed(0)
    N = 4
    payloads = [{KEY: torch.randn(6, 4) * 0.05} for _ in range(N)]
    uploads, pubs = _mask_cohort(payloads)
    agg = Aggregator(str(tmp_path), k_min=2, k_target=N)
    rnd = _seal_with(agg, pubs)
    assert rnd.sealed and set(rnd.sealed_peers) == {p.hex() for p in pubs}
    for masked, compat, spec_json in uploads:
        assert agg.submit(rnd.round_id, masked, compat, spec_json, "1.2.3.4") == "ok"
    agg.finalize(rnd)
    blob, version = agg.serve_global("m", "")
    tensors, _ = delta.from_bytes(blob)
    expected = sum(p[KEY] for p in payloads) / N            # η=1, no clip ⇒ plain mean of ΔW
    assert torch.allclose(tensors[KEY], expected, atol=1e-2)
    assert version == 1
    print("PASS test_aggregator_recovers_mean")


def test_two_rounds_accumulate(tmp_path):
    torch.manual_seed(1)
    agg = Aggregator(str(tmp_path), k_min=2, k_target=2)
    totals = torch.zeros(6, 4)
    for _ in range(2):
        payloads = [{KEY: torch.randn(6, 4) * 0.02} for _ in range(2)]
        uploads, pubs = _mask_cohort(payloads)
        rnd = _seal_with(agg, pubs)
        for masked, compat, spec_json in uploads:
            agg.submit(rnd.round_id, masked, compat, spec_json, "1.2.3.4")
        agg.finalize(rnd)
        totals += sum(p[KEY] for p in payloads) / 2          # cumulative Σ mean(ΔW)
    blob, version = agg.serve_global("m", "")
    tensors, _ = delta.from_bytes(blob)
    assert version == 2 and torch.allclose(tensors[KEY], totals, atol=1e-2)
    print("PASS test_two_rounds_accumulate")


def test_dropout_voids_round(tmp_path):
    payloads = [{KEY: torch.randn(6, 4) * 0.05} for _ in range(3)]
    uploads, pubs = _mask_cohort(payloads)
    agg = Aggregator(str(tmp_path), k_min=2, k_target=3)
    rnd = _seal_with(agg, pubs)
    for masked, compat, spec_json in uploads[:-1]:          # one sealed member never uploads
        agg.submit(rnd.round_id, masked, compat, spec_json, "1.2.3.4")
    agg.finalize(rnd)
    assert agg.serve_global("m", "") is None                # global untouched (masks wouldn't cancel)
    print("PASS test_dropout_voids_round")


def test_subquorum_register_fails(tmp_path):
    # Only 1 registrant at the deadline with k_min=2 ⇒ the round FAILS rather than leaking a lone ΔW.
    agg = Aggregator(str(tmp_path), k_min=2, k_target=8)
    rnd = agg.add_registrant("m", "aa", "1.2.3.4", now=0.0)
    agg.try_seal(rnd, final=True)
    assert rnd.failed and not rnd.sealed
    print("PASS test_subquorum_register_fails")


def test_norm_bound_clips_aggregate(tmp_path):
    torch.manual_seed(2)
    N = 3
    payloads = [{KEY: torch.randn(6, 4) * 3.0} for _ in range(N)]   # huge ⇒ ΣΔW well over k·clip
    uploads, pubs = _mask_cohort(payloads)
    agg = Aggregator(str(tmp_path), k_min=2, k_target=N, clip_norm=1.0, eta=1.0)
    rnd = _seal_with(agg, pubs)
    for masked, compat, spec_json in uploads:
        agg.submit(rnd.round_id, masked, compat, spec_json, "1.2.3.4")
    agg.finalize(rnd)
    tensors, _ = delta.from_bytes(agg.serve_global("m", "")[0])
    assert float(torch.linalg.vector_norm(tensors[KEY])) <= 1.0 + 1e-3   # η·clip
    print("PASS test_norm_bound_clips_aggregate")


def test_rejects_bad_uploads(tmp_path):
    payloads = [{KEY: torch.randn(6, 4) * 0.05} for _ in range(2)]
    uploads, pubs = _mask_cohort(payloads)
    agg = Aggregator(str(tmp_path), k_min=2, k_target=2)
    rnd = _seal_with(agg, pubs)
    masked, compat, spec_json = uploads[0]
    assert agg.submit(rnd.round_id, masked.float(), compat, spec_json, "1.2.3.4") == "bad tensor"      # wrong dtype
    assert agg.submit(rnd.round_id, masked, "deadbeef", spec_json, "1.2.3.4") == "ok"                  # first fixes compat
    assert agg.submit(rnd.round_id, uploads[1][0], "different", spec_json, "1.2.3.4") == "compat mismatch"
    assert agg.submit(rnd.round_id, masked, compat, spec_json, "9.9.9.9") == "ip not in cohort"        # IP-binding
    print("PASS test_rejects_bad_uploads")


def test_serve_global_cursor(tmp_path):
    payloads = [{KEY: torch.randn(6, 4) * 0.05} for _ in range(2)]
    uploads, pubs = _mask_cohort(payloads)
    agg = Aggregator(str(tmp_path), k_min=2, k_target=2)
    rnd = _seal_with(agg, pubs)
    for masked, compat, spec_json in uploads:
        agg.submit(rnd.round_id, masked, compat, spec_json, "1.2.3.4")
    agg.finalize(rnd)
    assert agg.serve_global("m", "1") is None            # since == current version ⇒ nothing new
    assert agg.serve_global("m", "") is not None
    assert agg.serve_global("other", "") is None         # unknown model
    print("PASS test_serve_global_cursor")


def test_concurrent_cohorts_per_model(tmp_path):
    # While cohort A is sealed and collecting, new registrants must form a SEPARATE cohort B (not be
    # rejected), and the two collect + finalize independently, each routed by its own round_id.
    torch.manual_seed(4)
    agg = Aggregator(str(tmp_path), k_min=2, k_target=2)
    payloads_a = [{KEY: torch.randn(6, 4) * 0.05} for _ in range(2)]
    payloads_b = [{KEY: torch.randn(6, 4) * 0.05} for _ in range(2)]
    up_a, pubs_a = _mask_cohort(payloads_a)
    up_b, pubs_b = _mask_cohort(payloads_b)

    rnd_a = _seal_with(agg, pubs_a)                          # cohort A seals, now COLLECTING
    rnd_b = _seal_with(agg, pubs_b)                          # B forms + seals while A still collects
    assert rnd_a.round_id != rnd_b.round_id
    assert rnd_a.round_id in agg.collecting and rnd_b.round_id in agg.collecting  # both live at once

    # Interleave uploads; each routes by its round_id.
    agg.submit(rnd_b.round_id, up_b[0][0], up_b[0][1], up_b[0][2], "1.2.3.4")
    agg.submit(rnd_a.round_id, up_a[0][0], up_a[0][1], up_a[0][2], "1.2.3.4")
    agg.submit(rnd_b.round_id, up_b[1][0], up_b[1][1], up_b[1][2], "1.2.3.4")
    agg.submit(rnd_a.round_id, up_a[1][0], up_a[1][1], up_a[1][2], "1.2.3.4")
    agg.finalize(rnd_a); agg.finalize(rnd_b)

    blob, version = agg.serve_global("m", "")
    expected = (sum(p[KEY] for p in payloads_a) + sum(p[KEY] for p in payloads_b)) / 2  # two rounds of mean
    assert version == 2 and torch.allclose(delta.from_bytes(blob)[0][KEY], expected, atol=1e-2)
    print("PASS test_concurrent_cohorts_per_model")


def test_global_persists_across_restart(tmp_path):
    payloads = [{KEY: torch.randn(6, 4) * 0.05} for _ in range(2)]
    uploads, pubs = _mask_cohort(payloads)
    agg = Aggregator(str(tmp_path), k_min=2, k_target=2)
    rnd = _seal_with(agg, pubs)
    for masked, compat, spec_json in uploads:
        agg.submit(rnd.round_id, masked, compat, spec_json, "1.2.3.4")
    agg.finalize(rnd)
    reloaded = Aggregator(str(tmp_path))                  # fresh instance reads the persisted blob
    blob, version = reloaded.serve_global("m", "")
    assert version == 1 and KEY in delta.from_bytes(blob)[0]
    print("PASS test_global_persists_across_restart")


# --- HTTP wire layer + seal barrier ----------------------------------------------------------

def test_http_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from roger.federated.server.app import create_app

    monkeypatch.setenv("ROGER_AGG_W", "10")              # bound any hang if requests serialized
    torch.manual_seed(3)
    N = 3
    payloads = [{KEY: torch.randn(6, 4) * 0.05} for _ in range(N)]
    uploads, pubs = _mask_cohort(payloads)
    # ip_binding off: every TestClient request shares the "testclient" host, but be explicit.
    agg = Aggregator(str(tmp_path), k_min=2, k_target=N, ip_binding=False)

    with TestClient(create_app(agg)) as client:
        # Concurrent registration: the cohort seals once all N are in-flight, returning the same peers.
        with ThreadPoolExecutor(max_workers=N) as ex:
            regs = list(ex.map(
                lambda pub: client.post("/round/register", json={"model_id": "m", "pubkey": pub.hex()}),
                pubs))
        assert all(r.status_code == 200 for r in regs)
        assert set(regs[0].json()["peers"]) == {p.hex() for p in pubs}
        round_id = regs[0].json()["round_id"]                 # all members of one cohort share it
        assert all(r.json()["round_id"] == round_id for r in regs)

        for masked, compat, spec_json in uploads:
            r = client.post("/contribute", content=_pack(masked, compat, spec_json, round_id),
                            headers={"Content-Type": "application/octet-stream"})
            assert r.status_code == 200

        r = client.get("/global", params={"model_id": "m", "since": ""})
        assert r.status_code == 200
        tensors, _ = delta.from_bytes(r.content)
        assert torch.allclose(tensors[KEY], sum(p[KEY] for p in payloads) / N, atol=1e-2)
        cursor = r.headers["X-Cursor"]
        assert client.get("/global", params={"model_id": "m", "since": cursor}).status_code == 204
    print("PASS test_http_end_to_end")


if __name__ == "__main__":
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    test_aggregator_recovers_mean(d / "a"); test_two_rounds_accumulate(d / "b")
    test_dropout_voids_round(d / "c"); test_subquorum_register_fails(d / "d")
    test_norm_bound_clips_aggregate(d / "e"); test_rejects_bad_uploads(d / "f")
    test_serve_global_cursor(d / "g"); test_global_persists_across_restart(d / "h")
