"""_result_to_ids: text → no mm kwargs; image → mm kwargs passthrough (seq-aligned sliced, per-image whole)."""
import torch
from PIL import Image
from transformers import BatchEncoding

from roger.agency.rollout_utils import _result_to_ids

TOOL_RES = 99  # stand-in <|tool_response> boundary id


class _StubProcessor:
    # Mimics a VLM processor: pure text → input_ids only; image → + seq-aligned token_type_ids
    # and per-image pixel_values. Branching is on the tool content, not on what this returns.
    def apply_chat_template(self, msgs, **kw):
        has_image = any(c.get("type") == "image" for m in msgs for c in m.get("content", []))
        ids = torch.tensor([[1, 2, TOOL_RES, 7, 8] + ([9] if has_image else [])])
        data = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if has_image:
            data["token_type_ids"] = torch.tensor([[0, 0, 0, 1, 1, 0]])
            data["pixel_values"] = torch.zeros(1, 3, 4, 4)
        return BatchEncoding(data)


def test_text_result_no_mm():
    ids, extra = _result_to_ids([("t", "hello")], _StubProcessor(), TOOL_RES, "cpu")
    assert extra is None
    assert ids.tolist() == [[TOOL_RES, 7, 8]]  # sliced from the boundary


def test_image_result_passes_mm_and_slices():
    img = Image.new("RGB", (4, 4))
    ids, extra = _result_to_ids([("screenshot", img)], _StubProcessor(), TOOL_RES, "cpu")
    assert ids.shape == (1, 4)                     # token delta sliced from the boundary (index 2 → 4 tokens)
    assert extra["token_type_ids"].shape == (1, 4)  # seq-aligned tensor sliced to the delta
    assert extra["pixel_values"].shape == (1, 3, 4, 4)  # per-image tensor kept whole
    assert "input_ids" not in extra and "attention_mask" not in extra
