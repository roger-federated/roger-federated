"""privacy_filter.py — anonymize PII in a recorded sequence before it reaches the gradient.

The LoRA REINFORCE++ update is trained over the recorded token sequence; if that sequence carries
real PII (emails, names, addresses, phone numbers, account numbers, secrets), the values get baked
into the weights — and, after aggregation, into the federated gradient. Masking the *loss* on PII
tokens is not enough: the kept generated tokens are still *conditioned on* the PII context, so the
association leaks anyway. So we rewrite the sequence itself, at train time only — inference is left
untouched, so the live agent still sees and acts on the real text (no capability loss).

Detection uses `openai/privacy-filter` (single-pass token classifier) via the shared VRAM-aware
loader. Substitution is done at the *token* level, 1:1 positional, so the count is preserved and the
trainer's token-aligned tensors (old_logp / masks / gen_start) stay valid with no rebuild.

`detect_pii_spans` is the seam tests stub out, so unit tests need no model download.
"""
import hashlib
import random

import torch

PII_MODEL_ID = "openai/privacy-filter"

# Per-type structural characters to retain so the surrogate keeps the original shape (e.g. an email
# still reads as `…@….com`). Secrets/passwords deliberately appear nowhere here: their punctuation
# is part of the value, so it must be scrambled too.
_RETAIN = {"email": set("@."), "phone": set("+-() ")}

_PIPE = None   # lazily-built HF token-classification pipeline; freed after data is anonymized


def _retained_chars(entity_type: str) -> set:
    t = (entity_type or "").lower()
    for key, chars in _RETAIN.items():
        if key in t:
            return chars
    return set()


def _get_pipe():
    global _PIPE
    if _PIPE is None:
        from transformers import AutoModelForTokenClassification, pipeline
        from roger.serving.model_setup import fetch_model
        # Reuse the policy loader so the filter quantizes/offloads against whatever VRAM is free
        # (the policy may already be resident on the Ctrl-D / reuse training path).
        model, tok = fetch_model(PII_MODEL_ID, model_cls=AutoModelForTokenClassification)
        # aggregation_strategy="simple" → coherent entity spans with char offsets + type, in one pass
        _PIPE = pipeline("token-classification", model=model, tokenizer=tok,
                         aggregation_strategy="simple")
    return _PIPE


def free_filter():
    """Drop the detector so its VRAM is reclaimed before the training epoch loop runs."""
    global _PIPE
    if _PIPE is not None:
        _PIPE = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def detect_pii_spans(text: str) -> list[tuple[int, int, str]]:
    """(char_start, char_end, entity_type) for every PII span. The test seam."""
    return [(d["start"], d["end"], d["entity_group"]) for d in _get_pipe()(text)]


def _token_char_spans(ids: list[int], tokenizer) -> tuple[list[tuple[int, int]], str]:
    """Per-token [char_start, char_end) via incremental decode-diff: token k's span is the growth of
    the cumulative decode. Model-agnostic (handles SentencePiece `▁` spacing) — no offset-mapping
    support assumed of the tokenizer."""
    spans, prev = [], ""
    for k in range(len(ids)):
        full = tokenizer.decode(ids[:k + 1])
        spans.append((len(prev), len(full)))
        prev = full
    return spans, prev


def _rand_token(rng: random.Random, vocab: int, special: set) -> int:
    while True:
        t = rng.randrange(vocab)
        if t not in special:   # never inject EOS/pad/other control tokens mid-sequence
            return t


def anonymize_sequence(seq_ids: torch.Tensor, tokenizer) -> tuple[torch.Tensor, set[int]]:
    """Return (rewritten sequence, absolute indices of PII tokens).

    PII tokens are replaced 1:1 with random non-special vocab tokens (so length is unchanged),
    except tokens that decode to solely their type's retained structural chars, which are kept.
    The RNG is seeded by a hash of the real value so the same value maps to the same surrogate
    within the episode (coreference stays coherent) while distinct values differ.
    The returned index set lets the trainer drop synthetic PII tokens from the loss."""
    ids = seq_ids.tolist()
    spans, text = _token_char_spans(ids, tokenizer)
    pii = detect_pii_spans(text)
    if not pii:
        return seq_ids, set()

    special = set(tokenizer.all_special_ids)
    vocab = tokenizer.vocab_size
    new_ids = list(ids)
    pii_positions: set[int] = set()
    for s, e, typ in pii:
        retain = _retained_chars(typ)
        # Seed by the value (stripped) so the same PII gets the same surrogate everywhere it appears
        # in this sequence. Holds when the value tokenizes the same way at each occurrence;
        # a divergent subword split would desync the draws, which strict count-preservation can't fix.
        rng = random.Random(_seed(text[s:e].strip()))
        for k, (cs, ce) in enumerate(spans):
            if cs < e and ce > s:                       # token overlaps the PII char span
                pii_positions.add(k)
                piece = tokenizer.decode([ids[k]]).strip()
                if piece and all(c in retain for c in piece):
                    continue                            # structural token → keep (shape preserved)
                new_ids[k] = _rand_token(rng, vocab, special)
    return torch.tensor(new_ids, dtype=seq_ids.dtype), pii_positions


def _seed(value: str) -> int:
    # blake2b (not Python's salted hash()) so the surrogate is reproducible across processes/runs.
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")
