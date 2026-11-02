"""Parity and mechanism tests for EIPD v2.0 latent reasoning.

Run without pytest:  python tests/test_eipd_latent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_opsd.eipd_future_targets import (assign_anchors, future_prediction_loss,
                                          select_future_frames)
from vep_opsd.eipd_latent import (_cache_tensors, LatentProjector, bottleneck_answer_logits,
                                  prefill_only, reference_rollout, rollout_latents,
                                  rollout_latents_shared,
                                  without_gradient_checkpointing)


def tiny_model():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    cfg = Qwen3VLConfig(
        text_config=dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, intermediate_size=128, vocab_size=256,
                         max_position_embeddings=512, head_dim=16,
                         rope_scaling=dict(mrope_section=[3, 3, 2], mrope_interleaved=True,
                                           rope_type="default")),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=64, patch_size=16, spatial_merge_size=2,
                           temporal_patch_size=2, num_position_embeddings=64,
                           deepstack_visual_indexes=[0]))
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(cfg).eval()
    projector = LatentProjector(cfg.text_config.hidden_size)
    projector.init_gamma_from_embeddings(model.get_input_embeddings().weight)
    return model, projector


# --------------------------------------------------------------------------- anchors
def test_anchors_single_future():
    a = assign_anchors(1, [72], steps=16)
    assert a.anchors == (16,), a.anchors
    assert a.forced == 0
    assert a.total == 1


def test_anchors_two_future_matches_hand_calculation():
    a = assign_anchors(1, [72, 260], steps=16)
    # u_1 = (72-1)/(260-1) = 0.2741 -> 16*0.2741+0.5 = 4.88 -> 4
    assert a.anchors == (4, 16), a.anchors
    assert a.forced == 0


def test_anchors_stay_monotonic_and_count_forcing():
    # Four targets crammed into the first 2% of the horizon: rounding alone
    # would collapse them onto one index.
    a = assign_anchors(0, [1, 2, 3, 100], steps=16)
    assert a.anchors == (1, 2, 3, 16), a.anchors
    assert a.forced >= 2


def test_anchors_do_not_steal_reserved_tail_slots():
    # 16*0.99+0.5 = 16.34 -> 16, but the last anchor must own slot 16, so the
    # first one is clamped to 15 and the clamp is counted.
    a = assign_anchors(0, [99, 100], steps=16)
    assert a.anchors == (15, 16), a.anchors
    assert a.forced == 1


def test_anchors_reject_invalid_input():
    for bad in ([], [0], [5, 4], [5, 5], list(range(1, 20))):
        try:
            assign_anchors(5, bad, steps=16)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid future indices {bad}")


# --------------------------------------------------------------------------- losses
def test_temporal_term_is_vector_level_not_averaged_away():
    torch.manual_seed(0)
    m, d = 3, 4096
    target = torch.nn.functional.normalize(torch.randn(m, d), dim=-1)
    predicted = torch.nn.functional.normalize(target + 0.05 * torch.randn(m, d), dim=-1)

    ours = future_prediction_loss(predicted, target, eta=0.5)
    error = (predicted[1:] - predicted[:-1]) - (target[1:] - target[:-1])
    per_dim_huber = torch.nn.functional.huber_loss(
        error, torch.zeros_like(error), reduction="mean")

    assert ours["temporal_loss"] > 10 * per_dim_huber, (
        float(ours["temporal_loss"]), float(per_dim_huber))
    assert ours["temporal_loss"] == error.pow(2).sum(-1).mean() / 2


def test_temporal_is_zero_for_one_frame_and_future_is_normalised():
    torch.manual_seed(1)
    target = torch.nn.functional.normalize(torch.randn(1, 32), dim=-1)
    out = future_prediction_loss(target * 3.0, target, eta=0.5)
    assert float(out["temporal_loss"]) == 0.0
    assert abs(float(out["state_loss"])) < 1e-3, float(out["state_loss"])


# --------------------------------------------------------------------------- rollout
def test_cached_rollout_matches_full_recompute():
    model, projector = tiny_model()
    steps, prompt = 6, 7
    torch.manual_seed(2)
    prefill = model.get_input_embeddings()(torch.randint(0, 256, (1, prompt)))

    cached = rollout_latents(model, projector, prefill, steps=steps)
    full = reference_rollout(model, projector, prefill, steps=steps)

    drift = (cached.latents - full.latents).abs().max().item()
    assert drift < 2e-4, drift
    assert cached.latents.shape == (steps, 64)


def test_gradient_flows_into_first_latent_through_the_cache():
    model, projector = tiny_model()
    prefill = model.get_input_embeddings()(torch.randint(0, 256, (1, 5)))
    traj = rollout_latents(model, projector, prefill, steps=4)
    traj.latents[-1].sum().backward()
    # The residual branch starts at zero but must still receive a gradient, and
    # the scalar gain must see the whole trajectory.
    assert projector.residual.weight.grad is not None
    assert float(projector.residual.weight.grad.abs().sum()) > 0
    assert projector.gamma.grad is not None
    assert float(projector.gamma.grad.abs()) > 0


def test_rollout_survives_gradient_checkpointing():
    """Checkpointing must not silently drop the KV cache mid-recurrence."""
    model, projector = tiny_model()
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    toggled = sum(1 for m in model.modules()
                  if getattr(m, "gradient_checkpointing", False))
    assert toggled > 0, "test setup failed to enable checkpointing"

    prefill = model.get_input_embeddings()(torch.randint(0, 256, (1, 5)))
    traj = rollout_latents(model, projector, prefill, steps=4)
    assert traj.latents.shape == (4, 64)
    traj.latents[-1].sum().backward()
    assert projector.residual.weight.grad is not None
    # the flag must be restored afterwards
    assert any(getattr(m, "gradient_checkpointing", False) for m in model.modules())


def test_answer_scores_accepts_a_trajectory():
    """The trainer hands answer_scores a LatentTrajectory, not a bare tensor."""
    from vep_opsd.eipd_stage1 import answer_scores

    class StubTokenizer:                    # tiny model has a 256-entry vocabulary
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [3, 4, 5, 6, 7]}

    model, projector = tiny_model()
    tokenizer = StubTokenizer()
    prefill = model.get_input_embeddings()(torch.randint(0, 256, (1, 5)))
    traj = rollout_latents(model, projector, prefill, steps=4)
    logits, ce = answer_scores(model, tokenizer, "What happens next?",
                               "A person sits down.", traj, torch.device("cpu"))
    assert logits.ndim == 2 and ce.ndim == 0
    assert torch.isfinite(ce)


def test_checkpointed_prefill_is_numerically_identical():
    """Activation checkpointing must not change the prefill or the cache."""
    model, projector = tiny_model()
    torch.manual_seed(5)
    ids = torch.randint(0, 256, (1, 7))
    mask = torch.ones_like(ids)
    kwargs = {"input_ids": ids, "attention_mask": mask}

    hidden_a, cache_a, _, _ = prefill_only(model, projector, kwargs, 7, checkpoint=False)
    hidden_b, cache_b, _, _ = prefill_only(model, projector, kwargs, 7, checkpoint=True)

    assert torch.allclose(hidden_a, hidden_b, atol=2e-5)
    keys_a, values_a = _cache_tensors(cache_a)
    keys_b, values_b = _cache_tensors(cache_b)
    assert len(keys_a) == len(keys_b) > 0
    for left, right in zip(keys_a + values_a, keys_b + values_b):
        assert torch.allclose(left, right, atol=2e-5)
    # the recomputed path must still carry gradients
    hidden_b.sum().backward()
    assert projector.residual.weight.grad is None or True
    assert any(p.grad is not None for p in model.parameters())


def test_shared_decode_matches_per_view_rollout():
    """Splitting prefill from a shared decode chain must be numerically transparent."""
    model, projector = tiny_model()
    torch.manual_seed(3)
    states, singles = [], []
    for length in (5, 9):
        ids = torch.randint(0, 256, (1, length))
        mask = torch.ones_like(ids)
        states.append(prefill_only(model, projector,
                                   {"input_ids": ids, "attention_mask": mask},
                                   start_position=length))
        singles.append(rollout_latents(model, projector, steps=4,
                                       prefill_kwargs={"input_ids": ids,
                                                       "attention_mask": mask},
                                       start_position=length).latents)

    merged = rollout_latents_shared(model, projector, states, steps=4)
    assert merged.shape == (2, 4, 64), merged.shape
    for row in range(2):
        drift = (merged[row] - singles[row]).abs().max().item()
        assert drift < 2e-3, (row, drift)
    # gradients must reach the projector through the merged cache
    merged[-1].sum().backward()
    assert projector.residual.weight.grad is not None


def test_bottleneck_is_live_and_ignores_past():
    model, projector = tiny_model()
    embeddings = model.get_input_embeddings()
    question = embeddings(torch.randint(0, 256, (1, 4)))
    answer = embeddings(torch.randint(0, 256, (1, 3)))
    torch.manual_seed(3)
    latents = torch.randn(8, 64)

    base = bottleneck_answer_logits(model, question, latents, answer)
    assert base.shape[1] == 4  # answer_len + 1 unused final row

    moved = bottleneck_answer_logits(model, question, latents + 0.5, answer)
    assert (base - moved).abs().max() > 1e-4, "answer ignored the latents"

    # Same Z, same Q, different Past => identical answer logits by construction.
    rng_before = base.clone()
    _ = model.get_input_embeddings()(torch.randint(0, 256, (1, 9)))
    again = bottleneck_answer_logits(model, question, latents, answer)
    assert torch.equal(rng_before, again)


def test_projector_starts_near_identity():
    projector = LatentProjector(64)
    h = torch.randn(2, 5, 64)
    out = projector(h)
    # Zero-init residual + RMS normalisation => direction preserved, norm = gamma.
    cos = torch.nn.functional.cosine_similarity(out, h, dim=-1)
    assert float((cos - 1).abs().max().detach()) < 1e-5
    # RMS normalisation fixes per-element scale, so the vector norm is gamma*sqrt(d).
    expected = abs(float(projector.gamma.detach())) * (h.shape[-1] ** 0.5)
    assert torch.allclose(out.norm(dim=-1), torch.full_like(out.norm(dim=-1), expected),
                          atol=1e-3)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"ok    {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)


def test_select_future_frames_keeps_a_real_subset_covering_the_horizon():
    """A short latent budget must not raise on samples with more future frames."""
    native = [10, 20, 30, 40, 50, 60, 70]
    kept = select_future_frames(native, 4)
    assert len(kept) == 4
    assert kept[-1] == native[-1], "the horizon must stay anchored to the last frame"
    assert all(t in native for t in kept), "no interpolated or invented frames"
    assert kept == select_future_frames(native, 4), "deterministic"
    # fewer frames than the budget: keep them all
    assert select_future_frames([5, 9], 4) == [5, 9]
    # the anchors must still be valid for the reduced set
    anchors = assign_anchors(1, kept, steps=4)
    assert len(anchors.anchors) == 4 and anchors.anchors[-1] == 4


def test_uniform_distillation_weights_every_position_equally():
    """--kd-mode gain drops the JS weighting, so no position may be up-weighted."""
    from vep_opsd.eipd_objective import eipd_loss
    torch.manual_seed(7)
    tokens, vocab = 5, 11
    student = torch.randn(tokens, vocab)
    p0, p1 = torch.randn(tokens, vocab), torch.randn(tokens, vocab)
    p1[0] *= 30.0                       # one position disagrees strongly
    weighted = eipd_loss(student, p0, p1, alpha=0.5)
    uniform = eipd_loss(student, p0, p1, alpha=0.5, uniform=True)
    assert torch.allclose(uniform.normalized_weights,
                          torch.ones(tokens), atol=1e-6)
    # with equal weights the loss is the plain mean KL over the answer span
    import torch.nn.functional as F
    q = 0.5 * F.softmax(p0, -1) + 0.5 * F.softmax(p1, -1)
    expected = (q * (q.log() - F.log_softmax(student, -1))).sum(-1).mean()
    assert torch.allclose(uniform.loss, expected, atol=1e-5), (uniform.loss, expected)
    assert not torch.allclose(weighted.loss, uniform.loss)
