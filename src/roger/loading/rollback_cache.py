"""A sliding-window KV-cache layer that can be rolled back a bounded distance.

The stock `DynamicSlidingWindowLayer` evicts everything older than the last `sliding_window-1`
tokens on every `update`, so once the window is full it *refuses* to `crop()` (the states a
rollback would need are gone). Our agent needs exactly that rollback: `_run_grade_nudge`
(agency/rollout_utils.py) runs a silent self-grade onto the live cache and then crops it away;
speculative decoding's rejection step does the same for a few tokens. Both roll back only a
*small, bounded* distance.

So instead of throwing evicted states away, we keep the most-recently-evicted ones in a
fixed-size FIFO ring (`ROLLBACK_WINDOW` tokens). `crop` then splices them back in. Restoration
is exact: KV entries are causal and already carry RoPE at their original position, so putting a
saved state back at its slot is correct. A rollback deeper than the ring still raises
ValueError, so the caller's recompute-from-scratch fallback stays intact for pathological spans.

This replaces the old `_patch_sliding_crop` monkeypatch, which cropped a full window in place
and left `physical < sliding_window-1` while the layer still reported `is_full` — a mask/KV
size mismatch that crashed the next forward.
"""
import torch
from transformers.cache_utils import DynamicSlidingWindowLayer

# Max tokens we can roll back. Must exceed the largest tentative span: the grade-nudge seed
# (~35 tokens) + its <=32 generated tokens, and any speculative lookahead (a handful). 128 is a
# comfortable margin; the extra VRAM is ROLLBACK_WINDOW tokens of K/V per sliding layer (tiny).
ROLLBACK_WINDOW = 128


class RollbackSlidingWindowLayer(DynamicSlidingWindowLayer):
    is_sliding = True   # keep the base's sliding classification (offload/mask logic reads this)

    def lazy_initialization(self, key_states, value_states):
        super().lazy_initialization(key_states, value_states)
        # Ring of evicted K/V, newest at the end. Empty 1-D tensors until the first eviction;
        # torch.cat skips them, so they promote to [b, heads, n, dim] once fed a real slice
        # (same convention the base uses for self.keys/self.values).
        self._evicted_keys = key_states.new_empty(0)
        self._evicted_values = value_states.new_empty(0)

    def _ring_len(self) -> int:
        return self._evicted_keys.shape[-2] if self._evicted_keys.ndim == 4 else 0

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        keep = self.sliding_window - 1
        full_key_states = _cat(self.keys, key_states)
        full_value_states = _cat(self.values, value_states)
        # Whatever the base would drop (everything but the last `keep`) is what we retain.
        n_evict = full_key_states.shape[-2] - keep
        if n_evict > 0:
            self._evicted_keys = _cat(self._evicted_keys, full_key_states[:, :, :n_evict, :])[:, :, -ROLLBACK_WINDOW:, :]
            self._evicted_values = _cat(self._evicted_values, full_value_states[:, :, :n_evict, :])[:, :, -ROLLBACK_WINDOW:, :]
        self.cumulative_length += key_states.shape[-2]
        self.keys = full_key_states[:, :, -keep:, :]
        self.values = full_value_states[:, :, -keep:, :]
        return full_key_states, full_value_states   # the forward attends over the full states

    def crop(self, max_length: int) -> None:
        """Roll the cache back to absolute position `max_length`, restoring evicted states.

        Splices the last `restore_count` ring entries in front of the surviving physical columns
        so the buffer again holds the last `min(max_length, sliding_window-1)` tokens ending at
        `max_length`, with `cumulative_length == max_length`. Raises if the rollback needs more
        than the ring retained (deeper than one window, or > ROLLBACK_WINDOW back)."""
        if not self.is_initialized:
            return
        if max_length < 0:
            max_length = self.cumulative_length + max_length
        if max_length >= self.cumulative_length:
            return
        keep = self.sliding_window - 1
        physical = self.keys.shape[-2]
        k = self.cumulative_length - max_length          # tokens to roll back (>0)
        target_phys = min(max_length, keep)              # physical size the result must have
        keep_from_physical = physical - k                # leading physical cols that survive
        if keep_from_physical < 0:                        # rolling back further than one window
            raise ValueError(f"crop past the physical window ({k} > {physical}); recompute instead")
        restore_count = target_phys - keep_from_physical  # cols to pull back from the ring
        ring_len = self._ring_len()
        if restore_count > ring_len:
            raise ValueError(f"crop needs {restore_count} evicted tokens; only {ring_len} retained")
        kept_k = self.keys[:, :, :keep_from_physical, :]
        kept_v = self.values[:, :, :keep_from_physical, :]
        if restore_count > 0:
            restored_k = self._evicted_keys[:, :, ring_len - restore_count:, :]
            restored_v = self._evicted_values[:, :, ring_len - restore_count:, :]
            self.keys = _cat(restored_k, kept_k)
            self.values = _cat(restored_v, kept_v)
            # Those entries are back in the window; drop them from the ring (keep the older ones).
            self._evicted_keys = self._evicted_keys[:, :, :ring_len - restore_count, :]
            self._evicted_values = self._evicted_values[:, :, :ring_len - restore_count, :]
        else:
            self.keys, self.values = kept_k, kept_v
        self.cumulative_length = max_length

    def reset(self) -> None:
        super().reset()
        if self.is_initialized:
            self._evicted_keys = self._evicted_keys[:, :, :0, :] if self._evicted_keys.ndim == 4 else self._evicted_keys
            self._evicted_values = self._evicted_values[:, :, :0, :] if self._evicted_values.ndim == 4 else self._evicted_values

    # Beam/contrastive search reshuffle the batch dim; keep the ring in lockstep (unused by the
    # agent's greedy/sampling rollout, but the layer must stay internally consistent).
    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        if self._ring_len() > 0:
            self._evicted_keys = self._evicted_keys.repeat_interleave(repeats, dim=0)
            self._evicted_values = self._evicted_values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices) -> None:
        super().batch_select_indices(indices)
        if self._ring_len() > 0:
            self._evicted_keys = self._evicted_keys[indices, ...]
            self._evicted_values = self._evicted_values[indices, ...]

    def reorder_cache(self, beam_idx) -> None:
        super().reorder_cache(beam_idx)
        if self._ring_len() > 0:
            dev = self._evicted_keys.device
            self._evicted_keys = self._evicted_keys.index_select(0, beam_idx.to(dev))
            self._evicted_values = self._evicted_values.index_select(0, beam_idx.to(dev))


def _cat(a, b):
    """torch.cat along the sequence dim; a 1-D empty placeholder is skipped (base-cache convention)."""
    return torch.cat([a, b], dim=-2)
