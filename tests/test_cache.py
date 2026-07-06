"""Tests for RollbackSlidingWindowLayer — the sliding-window cache that retains recently-evicted
K/V so a bounded crop can roll back. Run with:  PYTHONPATH=src python -m pytest tests/test_cache.py

No model is loaded: the layer is pure tensor bookkeeping. Each token is filled with its absolute
position value, so we can read back exactly which positions a buffer holds."""
import torch
import pytest
from roger.loading.rollback_cache import RollbackSlidingWindowLayer, ROLLBACK_WINDOW

SW = 8            # small window; the layer keeps the last SW-1 = 7 tokens physically
B, H, D = 1, 2, 4


def _tok(pos):
    return torch.full((B, H, 1, D), float(pos))


def _feed(layer, start, n):
    for p in range(start, start + n):
        layer.update(_tok(p), _tok(p))


def _positions(t):
    return [int(t[0, 0, j, 0].item()) for j in range(t.shape[-2])]


def test_registry_wired():
    # Importing model_setup must repoint the cache registry at our layer (normal + assisted decode).
    import roger.loading.model_setup  # noqa: F401
    from transformers.cache_utils import LAYER_TYPE_CACHE_MAPPING
    assert LAYER_TYPE_CACHE_MAPPING["sliding_attention"] is RollbackSlidingWindowLayer
    assert LAYER_TYPE_CACHE_MAPPING["chunked_attention"] is RollbackSlidingWindowLayer


def test_full_window_rollback_restores_evicted():
    L = RollbackSlidingWindowLayer(sliding_window=SW)
    _feed(L, 0, 20)                                  # positions 0..19; window full
    assert L.cumulative_length == 20
    assert _positions(L.keys) == list(range(13, 20)) # physically holds last 7

    L.crop(15)                                       # roll back 5 tokens (into evicted territory)
    assert L.cumulative_length == 15
    # restored: physical again holds the last 7 tokens ending at 15 → positions 8..14
    assert _positions(L.keys) == list(range(8, 15))
    assert _positions(L.values) == list(range(8, 15))

    # Byte-for-byte identical to a cache that only ever saw 0..14 (restoration is exact).
    ref = RollbackSlidingWindowLayer(sliding_window=SW)
    _feed(ref, 0, 15)
    assert torch.equal(L.keys, ref.keys)
    assert torch.equal(L.values, ref.values)


def test_crop_preserves_mask_invariant():
    # The bug: after crop, physical < sliding_window-1 while is_full → get_mask_sizes disagreed
    # with the KV tensor and the next forward crashed. Assert they agree now.
    L = RollbackSlidingWindowLayer(sliding_window=SW)
    _feed(L, 0, 20)
    L.crop(15)
    assert L.keys.shape[-2] == min(L.cumulative_length, SW - 1)   # physical == keep when full
    kv_length, _ = L.get_mask_sizes(1)
    full_k, _ = L.update(_tok(15), _tok(15))          # one more decode step
    assert full_k.shape[-2] == kv_length              # mask size matches the actual KV length


def test_not_full_crop_is_plain_truncate():
    L = RollbackSlidingWindowLayer(sliding_window=SW)
    _feed(L, 0, 5)                                    # never filled the window; nothing evicted
    assert L._ring_len() == 0
    L.crop(3)
    assert L.cumulative_length == 3
    assert _positions(L.keys) == [0, 1, 2]


def test_crop_noop_when_not_shrinking():
    L = RollbackSlidingWindowLayer(sliding_window=SW)
    _feed(L, 0, 5)
    L.crop(5)                                         # equal → no change
    L.crop(9)                                         # beyond current → no change
    assert L.cumulative_length == 5
    assert _positions(L.keys) == [0, 1, 2, 3, 4]


def test_rollback_past_window_raises():
    L = RollbackSlidingWindowLayer(sliding_window=SW)
    _feed(L, 0, 20)
    with pytest.raises(ValueError):                   # k=15 > physical window (7) → recompute fallback
        L.crop(5)


def test_rollback_beyond_ring_capacity_raises():
    # Window larger than the ring, and enough tokens that a rollback can stay full yet need more
    # evicted states than we retained (restore_count == k > ROLLBACK_WINDOW while k <= physical).
    sw = ROLLBACK_WINDOW + 50                         # keep = 177
    L = RollbackSlidingWindowLayer(sliding_window=sw)
    _feed(L, 0, 340)
    with pytest.raises(ValueError):                   # k=150 > ROLLBACK_WINDOW=128 (still within the window)
        L.crop(190)
    L.crop(240)                                       # k=100 <= ring → succeeds
    assert L.cumulative_length == 240


def test_reset_clears_ring():
    L = RollbackSlidingWindowLayer(sliding_window=SW)
    _feed(L, 0, 20)
    assert L._ring_len() > 0
    L.reset()
    assert L._ring_len() == 0
