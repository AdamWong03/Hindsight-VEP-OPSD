"""Mechanism tests for Cross-Depth Latent Self-Distillation."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_opsd.cdlsd_data import CDLSDDataset
from vep_opsd.cdlsd_latent import answer_logits
from vep_opsd.cdlsd_objective import (
    forward_kl,
    response_cross_entropy,
    route_kd_through_states,
)


def test_forward_kl_is_zero_for_identical_distributions():
    torch.manual_seed(0)
    logits = torch.randn(5, 17)
    loss, per_token = forward_kl(logits, logits.clone())
    assert abs(float(loss)) < 1e-6
    assert per_token.shape == (5,)


def test_response_ce_supervises_only_supplied_tokens():
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    ids = torch.tensor([0, 1])
    assert float(response_cross_entropy(logits, ids)) < 1e-3


def test_routed_kd_updates_upstream_state_path_not_decoder_parameter():
    upstream = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    decoder = torch.nn.Parameter(torch.tensor([[0.3], [-0.7]]))
    states = (2.0 * upstream).view(1, 2)
    prediction = states @ decoder
    kd = prediction.square().mean()
    routed = route_kd_through_states(kd, states)
    assert torch.allclose(routed.routed.detach(), kd.detach())
    routed.routed.backward()
    assert upstream.grad is not None and float(upstream.grad.abs().sum()) > 0
    assert decoder.grad is None, "KD leaked through the answer reader"


def test_advantage_mask_can_zero_the_routed_gradient():
    upstream = torch.nn.Parameter(torch.randn(2))
    states = upstream.view(1, 2)
    kd = states.square().mean()
    routed = route_kd_through_states(kd * 0.0, states)
    routed.routed.backward()
    assert torch.equal(upstream.grad, torch.zeros_like(upstream))


def test_answer_logits_align_one_row_per_response_token():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            del add_special_tokens
            if text == "<answer>":
                return {"input_ids": [9]}
            return {"input_ids": [3, 4, 5]}

    config = Qwen3VLConfig(
        text_config=dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, intermediate_size=128, vocab_size=256,
                         max_position_embeddings=512, head_dim=16,
                         rope_scaling=dict(mrope_section=[3, 3, 2],
                                           mrope_interleaved=True,
                                           rope_type="default")),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=64, patch_size=16, spatial_merge_size=2,
                           temporal_patch_size=2, num_position_embeddings=64,
                           deepstack_visual_indexes=[0]))
    model = Qwen3VLForConditionalGeneration(config).eval()
    states = torch.randn(4, 64, requires_grad=True)
    response = torch.tensor([11, 12, 13, 14])
    logits = answer_logits(model, Tokenizer(), "question", states, response)
    assert logits.shape == (4, 256)
    loss = response_cross_entropy(logits, response)
    loss.backward()
    assert states.grad is not None and float(states.grad.abs().sum()) > 0


def test_dataset_never_requires_bridge_or_future_files():
    from PIL import Image

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "frames").mkdir()
        Image.new("RGB", (8, 8), "red").save(root / "frames" / "past.png")
        manifest = root / "train.jsonl"
        manifest.write_text(json.dumps({
            "sample_id": "one",
            "question": "What happens next?",
            "answer": "The person moves.",
            "past_files": ["frames/past.png"],
            "bridge_file": "does-not-exist.png",
            "future_files": ["also-does-not-exist.png"],
        }) + "\n", encoding="utf-8")
        item = CDLSDDataset(str(root), str(manifest))[0]
        assert len(item["observed"]) == 1


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            function()
            print(f"ok    {name}")
        except Exception as error:
            failures += 1
            print(f"FAIL  {name}: {type(error).__name__}: {error}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
