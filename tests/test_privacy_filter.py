"""Tests for the train-time PII anonymizer.
Run with:  PYTHONPATH=src python -m pytest tests/test_privacy_filter.py

`detect_pii_spans` is stubbed so the tests need no 1.5B model download. A tiny stub tokenizer with
fully-controlled token<->char boundaries lets us assert exactly which tokens are kept vs replaced.
"""
import torch

import roger.training.privacy_filter as pf


class StubTok:
    """Minimal tokenizer: ids start at 10 (clear of the special range); decode is plain join."""
    def __init__(self, pieces, special_ids=(0, 1, 2), vocab_size=50000):
        self.id2piece = {i + 10: p for i, p in enumerate(pieces)}
        self.ids = list(self.id2piece)
        self.all_special_ids = list(special_ids)
        self.vocab_size = vocab_size

    def decode(self, ids):
        return "".join(self.id2piece[int(i)] for i in ids)


def _run(pieces, spans):
    """Anonymize `pieces` with `detect_pii_spans` stubbed to return `spans`. Returns (tok, in, out, pii)."""
    tok = StubTok(pieces)
    pf.detect_pii_spans = lambda text, _s=spans: list(_s)
    seq = torch.tensor(tok.ids)
    out, pii = pf.anonymize_sequence(seq, tok)
    return tok, seq, out, pii


def test_anonymize_email_preserves_length_and_structure():
    # "Email me at john@doe.com please" — the email occupies chars [12, 24).
    pieces = ["Email", " me", " at", " john", "@", "doe", ".", "com", " please"]
    tok, seq, out, pii = _run(pieces, [(12, 24, "EMAIL")])
    assert out.shape == seq.shape                               # 1:1 token swap → length preserved
    assert pii == {3, 4, 5, 6, 7}                               # every token overlapping the span
    # non-PII tokens untouched
    for k in (0, 1, 2, 8):
        assert int(out[k]) == int(seq[k])
    # email structural tokens (@ and .) retained; content tokens replaced with non-special ids
    assert int(out[4]) == int(seq[4]) and int(out[6]) == int(seq[6])
    for k in (3, 5, 7):
        assert int(out[k]) != int(seq[k])
        assert int(out[k]) not in tok.all_special_ids
    print("PASS test_anonymize_email_preserves_length_and_structure")


def test_anonymize_secret_scrambles_punctuation():
    # "pass=p@ss1" — the secret value "p@ss1" is chars [5, 10); '=' at [4,5) is outside it.
    pieces = ["pass", "=", "p", "@", "ss", "1"]
    tok, seq, out, pii = _run(pieces, [(5, 10, "SECRET")])
    assert pii == {2, 3, 4, 5}
    assert int(out[1]) == int(seq[1])                           # '=' outside the span, untouched
    # unlike email, the '@' here is part of the secret → replaced, not retained
    for k in (2, 3, 4, 5):
        assert int(out[k]) != int(seq[k])
    print("PASS test_anonymize_secret_scrambles_punctuation")


def test_anonymize_deterministic_and_no_special():
    pieces = ["Email", " me", " at", " john", "@", "doe", ".", "com", " please"]
    spans = [(12, 24, "EMAIL")]
    tok = StubTok(pieces)
    pf.detect_pii_spans = lambda text: list(spans)
    seq = torch.tensor(tok.ids)
    a, _ = pf.anonymize_sequence(seq, tok)
    b, _ = pf.anonymize_sequence(seq, tok)
    assert torch.equal(a, b)                                    # seeded by the real value → stable
    assert all(int(t) not in tok.all_special_ids for t in a)
    print("PASS test_anonymize_deterministic_and_no_special")


def test_anonymize_repeated_value_consistent():
    # Same email appears twice (think: context + echoed output). Both must map to the SAME surrogate
    # tokens so the model isn't trained to emit a different value than the one it copied from.
    pieces = ["to", " john", "@", "doe", ".", "com",
              " from", " john", "@", "doe", ".", "com"]
    # email 1 = chars [3,15); email 2 = chars [21,33)
    tok, seq, out, pii = _run(pieces, [(3, 15, "EMAIL"), (21, 33, "EMAIL")])
    # content tokens of occurrence 1 (1,3,5) match those of occurrence 2 (7,9,11) pairwise
    assert int(out[1]) == int(out[7])
    assert int(out[3]) == int(out[9])
    assert int(out[5]) == int(out[11])
    # and they were actually changed (not coincidentally equal because untouched)
    assert int(out[1]) != int(seq[1]) and int(out[3]) != int(seq[3])
    print("PASS test_anonymize_repeated_value_consistent")


def test_anonymize_no_pii_is_noop():
    tok = StubTok(["hello", " world"])
    pf.detect_pii_spans = lambda text: []
    seq = torch.tensor(tok.ids)
    out, pii = pf.anonymize_sequence(seq, tok)
    assert pii == set() and torch.equal(out, seq)
    print("PASS test_anonymize_no_pii_is_noop")


def test_keep_mask_index_select_no_nan():
    # The trainer index-selects kept tokens before the ratio. An extreme masked log-prob (which would
    # overflow the ratio and, under post-hoc `* 0` masking, give a nan grad) must stay finite here.
    new_lp = torch.tensor([0.0, 200.0, 0.0, 0.0], requires_grad=True)   # idx1 masked, would overflow
    old_lp = torch.zeros(4)
    adv, clip_eps = 0.7, 0.2
    keep = torch.tensor([True, False, True, True])
    nl, ol = new_lp[keep], old_lp[keep]
    ratio = torch.exp(nl - ol)
    loss = -torch.sum(torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)) / keep.sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(new_lp.grad).all()
    assert float(new_lp.grad[1]) == 0.0                      # masked token: zero grad, no nan
    assert torch.all(new_lp.grad[[0, 2, 3]] != 0)
    print("PASS test_keep_mask_index_select_no_nan")


if __name__ == "__main__":
    test_anonymize_email_preserves_length_and_structure()
    test_anonymize_secret_scrambles_punctuation()
    test_anonymize_deterministic_and_no_special()
    test_anonymize_repeated_value_consistent()
    test_anonymize_no_pii_is_noop()
    test_keep_mask_index_select_no_nan()
