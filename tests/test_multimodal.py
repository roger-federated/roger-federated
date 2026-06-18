"""_result_to_ids: text → no mm kwargs; image → mm kwargs passthrough (seq-aligned sliced, per-image whole).
Also covers media extraction: _parse_result flattening and MCP block → media/placeholder mapping."""
import base64, io
from types import SimpleNamespace
import torch
from PIL import Image
from transformers import BatchEncoding

from roger.agency.rollout_utils import _result_to_ids, _parse_result
from roger.tools.mcp_utils import _mcp_block

TOOL_RES = 99  # stand-in <|tool_response> boundary id


def _png_b64() -> str:
    buf = io.BytesIO(); Image.new("RGB", (4, 4)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


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


def test_parse_result_flattens_mixed():
    blocks = _parse_result([Image.new("RGB", (2, 2)), "caption", Image.new("RGB", (2, 2))])
    assert [b["type"] for b in blocks] == ["image", "text", "image"]


def test_mcp_block_extracts_image_text_resource_and_placeholder():
    assert _mcp_block(SimpleNamespace(type="text", text="hi")) == "hi"
    assert isinstance(_mcp_block(SimpleNamespace(type="image", data=_png_b64(), mimeType="image/png")), Image.Image)
    res_img = SimpleNamespace(type="resource",
        resource=SimpleNamespace(blob=_png_b64(), mimeType="image/png", uri="screen://1"))
    assert isinstance(_mcp_block(res_img), Image.Image)
    # unknown/link block → compact placeholder, not a base64 dump
    assert _mcp_block(SimpleNamespace(type="resource_link", uri="http://x")).startswith("[resource_link")
