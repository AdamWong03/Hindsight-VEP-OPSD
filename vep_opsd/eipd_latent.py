"""Continuous latent reasoning + latent bottleneck for EIPD v2.0.

Two-stage forward, exactly as specified:

    Stage A  (X_p, Q) -> z_1 .. z_K        recurrent, KV-cached, differentiable
    Stage B  (Q, Z)   -> answer            fresh context, no Past images, no
                                           inherited cache, Z keeps its graph

Stage A never samples tokens: z_k is `P_psi(h_k)` where `h_k` is the last
hidden state of the same Qwen3-VL backbone. Stage B is a single fresh forward
over `[Q_embeds ; Z ; answer_tokens]`, which makes the answer distribution
conditionally independent of the Past image tokens given Z --- that is the
bottleneck, and it is what stops the model from reading the answer off the
observed frames.

`reference_rollout` recomputes the full prefix at every step (the old, correct
but O(K) implementation). It exists only so tests can prove that the cached
path is numerically equivalent; training must use `rollout_latents`.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn

EPS = 1e-6


class LatentProjector(nn.Module):
    """h_k -> z_k with an identity-initialised residual, then RMSNorm.

    z = gamma * (h + R(h)) / sqrt(mean((h + R(h))^2) + eps)

    * `residual` is zero-initialised so the map starts as the identity.
    * The normaliser has **no per-dimension affine**; only a scalar `gamma`,
    * initialised to the RMS of the real token embeddings so the injected
      vectors start in the same scale regime as text tokens.
    """

    def __init__(self, hidden_size: int, *, gamma_init: float = 1.0,
                 eps: float = EPS):
        super().__init__()
        self.residual = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.residual.weight)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        u = hidden + self.residual(hidden)
        rms = torch.sqrt(u.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (u / rms)

    @torch.no_grad()
    def init_gamma_from_embeddings(self, embedding_weight: torch.Tensor) -> float:
        rms = float(torch.sqrt(embedding_weight.float().pow(2).mean()).item())
        self.gamma.fill_(rms)
        return rms


@dataclass
class LatentTrajectory:
    latents: torch.Tensor          # (K, d), graph attached
    hidden_states: list[torch.Tensor]


@contextmanager
def without_gradient_checkpointing(model):
    """Temporarily disable activation checkpointing for the latent recurrence.

    `GradientCheckpointingLayer.__call__` forces `use_cache=False` and
    `past_key_values=None` whenever checkpointing is active in training mode.
    That is correct for a normal teacher-forced forward but fatal for a KV-cached
    recurrence: the cache would be dropped at every layer, so each latent step
    would restart from scratch. Checkpointing is restored on exit, so the
    bottleneck answer pass still benefits from it.
    """
    toggled = []
    for module in model.modules():
        if getattr(module, "gradient_checkpointing", False):
            module.gradient_checkpointing = False
            toggled.append(module)
    try:
        yield len(toggled)
    finally:
        for module in toggled:
            module.gradient_checkpointing = True


def _mrope_positions(start: int, length: int, device) -> torch.Tensor:
    """Text-only positions: all three mrope sections share the same index."""
    positions = torch.arange(start, start + length, device=device)
    return positions.view(1, 1, -1).expand(3, 1, -1)


def inner_model(model):
    """The decoder that owns the KV cache, without the language-model head.

    Latent steps only need the last hidden state, so calling the head-free model
    avoids materialising logits (151k-wide) and, with output_hidden_states, the
    37 per-layer tensors of every step.

    Wrappers are unwrapped first: under DeepSpeed the trainer hands us the engine
    (`.module`) and under PEFT a PeftModel (`get_base_model()`). Calling the bare
    decoder removes wrapper bookkeeping from each of the 32 single-token forwards.
    """
    inner = model
    for _ in range(4):
        if hasattr(inner, "module") and not hasattr(inner, "get_base_model"):
            inner = inner.module
        else:
            break
    backbone = inner.get_base_model() if hasattr(inner, "get_base_model") else inner
    return getattr(backbone, "model", backbone)


def _last_hidden(output) -> torch.Tensor:
    hidden = output.hidden_states[-1][:, -1:, :]
    if hidden.ndim != 3 or hidden.shape[1] != 1:
        raise RuntimeError(f"Unexpected last hidden state shape {tuple(hidden.shape)}")
    return hidden


def rollout_latents(model, projector: LatentProjector, prefill_embeds: torch.Tensor | None = None,
                    *, steps: int, cache=None, prefill_kwargs: dict | None = None,
                    start_position: int | None = None) -> LatentTrajectory:
    """Public entry: same recurrence, with activation checkpointing suspended."""
    with without_gradient_checkpointing(model):
        return _rollout_latents_cached(
            model, projector, prefill_embeds, steps=steps, cache=cache,
            prefill_kwargs=prefill_kwargs, start_position=start_position)


def _rollout_latents_cached(model, projector: LatentProjector,
                            prefill_embeds: torch.Tensor | None = None,
                            *, steps: int, cache=None, prefill_kwargs: dict | None = None,
                            start_position: int | None = None) -> LatentTrajectory:
    """Recurrent continuous latent generation with an incremental KV cache.

    Two prefill forms are supported:

    * ``prefill_embeds`` — embeddings only (used by the parity tests),
    * ``prefill_kwargs`` — the real multimodal call (``input_ids`` +
      ``pixel_values`` + ``image_grid_thw``). This is the correct path for
      Qwen3-VL: it lets the model merge image features *and* apply the DeepStack
      injections itself, which a hand-built embedding tensor cannot do.

    With images the continuation position is not the sequence length, so the
    caller passes ``start_position`` (``max(mrope positions) + 1``).

    The cache carries autograd edges, so step K back-propagates into step 1
    (verified in tests). ``cache`` is created here unless supplied.
    """
    from transformers.cache_utils import DynamicCache

    if (prefill_embeds is None) == (prefill_kwargs is None):
        raise ValueError("pass exactly one of prefill_embeds / prefill_kwargs")
    if steps <= 0:
        raise ValueError("steps must be positive")
    cache = cache if cache is not None else DynamicCache()
    if prefill_embeds is not None:
        if prefill_embeds.ndim != 3 or prefill_embeds.shape[0] != 1:
            raise ValueError("rollout_latents expects one prefill of shape (1, P, d)")
        length = prefill_embeds.shape[1]
        prompt_length = cache.get_seq_length() + length
        out = inner_model(model)(inputs_embeds=prefill_embeds, past_key_values=cache,
                                 cache_position=torch.arange(prompt_length - length,
                                                             prompt_length,
                                                             device=prefill_embeds.device),
                                 position_ids=_mrope_positions(prompt_length - length,
                                                               length, prefill_embeds.device),
                                 use_cache=True, return_dict=True)
        hidden = out.last_hidden_state[:, -1:, :]
        position = prompt_length
    else:
        kwargs = dict(prefill_kwargs)
        length = kwargs["input_ids"].shape[1]
        prompt_length = cache.get_seq_length() + length
        device = kwargs["input_ids"].device
        kwargs.setdefault("cache_position", torch.arange(prompt_length - length,
                                                         prompt_length, device=device))
        out = inner_model(model)(**kwargs, past_key_values=cache, use_cache=True,
                                 return_dict=True)
        hidden = out.last_hidden_state[:, -1:, :]
        position = prompt_length if start_position is None else int(start_position)
    device = hidden.device

    latents: list[torch.Tensor] = []
    hidden_states: list[torch.Tensor] = []
    for step in range(steps):
        hidden_states.append(hidden)
        step_position = position + step
        z = projector(hidden)                       # (1, 1, d) — keeps the graph
        latents.append(z[0, 0])
        out = inner_model(model)(inputs_embeds=z, past_key_values=cache,
                                 cache_position=torch.tensor([step_position], device=device),
                                 position_ids=_mrope_positions(step_position, 1, device),
                                 use_cache=True, return_dict=True)
        hidden = out.last_hidden_state[:, -1:, :]

    return LatentTrajectory(torch.stack(latents), hidden_states)


def reference_rollout(model, projector: LatentProjector, prefill_embeds: torch.Tensor,
                      *, steps: int) -> LatentTrajectory:
    """Same recurrence, full prefix recomputed at every step (no cache).

    Correct but O(K) full forwards; used only as the parity oracle in tests.
    """
    if prefill_embeds.ndim != 3 or prefill_embeds.shape[0] != 1:
        raise ValueError("reference_rollout expects one prefill of shape (1, P, d)")
    device = prefill_embeds.device
    prompt_length = prefill_embeds.shape[1]
    latents: list[torch.Tensor] = []
    hidden_states: list[torch.Tensor] = []
    produced: list[torch.Tensor] = []

    out = model(inputs_embeds=prefill_embeds,
                cache_position=torch.arange(prompt_length, device=device),
                position_ids=_mrope_positions(0, prompt_length, device),
                use_cache=False, output_hidden_states=True, logits_to_keep=1)
    hidden = _last_hidden(out)
    for step in range(steps):
        hidden_states.append(hidden)
        z = projector(hidden)
        latents.append(z[0, 0])
        produced.append(z)
        total = prompt_length + step + 1
        out = model(inputs_embeds=torch.cat([prefill_embeds, *produced], dim=1),
                    cache_position=torch.arange(total, device=device),
                    position_ids=_mrope_positions(0, total, device),
                    use_cache=False, output_hidden_states=True, logits_to_keep=1)
        hidden = _last_hidden(out)

    return LatentTrajectory(torch.stack(latents), hidden_states)


def bottleneck_answer_inputs(question_embeds: torch.Tensor, latents: torch.Tensor,
                             answer_embeds: torch.Tensor | None = None
                             ) -> torch.Tensor:
    """[Q ; Z ; A] in embedding space — no Past image tokens, no inherited cache."""
    if question_embeds.ndim != 3:
        raise ValueError("question_embeds must be (1, Lq, d)")
    if latents.ndim not in (2, 3):
        raise ValueError("latents must be (K, d) or (1, K, d)")
    z = latents.unsqueeze(0) if latents.ndim == 2 else latents
    if z.shape[0] != 1 or z.shape[2] != question_embeds.shape[2]:
        raise ValueError("latent/question embedding shapes disagree")
    parts = [question_embeds, z]
    if answer_embeds is not None:
        parts.append(answer_embeds)
    return torch.cat(parts, dim=1)


def bottleneck_answer_logits(model, question_embeds: torch.Tensor, latents: torch.Tensor,
                             answer_embeds: torch.Tensor, *, keep: int | None = None
                             ) -> torch.Tensor:
    """Fresh forward over [Q ; Z ; A]; returns logits predicting the answer span.

    `keep=len(answer)` returns exactly the rows that predict each answer token
    under the causal shift (row i predicts token i of the answer), so the caller
    never has to slice indices itself.
    """
    inputs = bottleneck_answer_inputs(question_embeds, latents, answer_embeds)
    length = inputs.shape[1]
    answer_len = answer_embeds.shape[1]
    keep = answer_len if keep is None else keep
    out = model(inputs_embeds=inputs,
                cache_position=torch.arange(length, device=inputs.device),
                position_ids=_mrope_positions(0, length, inputs.device),
                use_cache=False, logits_to_keep=keep + 1)
    return out.logits


def _cache_tensors(cache):
    """(keys, values) across layers, for both the old and the 4.5x cache APIs."""
    if hasattr(cache, "key_cache"):
        return list(cache.key_cache), list(cache.value_cache)
    return [layer.keys for layer in cache.layers], [layer.values for layer in cache.layers]


def _set_cache_tensors(cache, keys, values):
    if hasattr(cache, "key_cache"):
        for index, (key, value) in enumerate(zip(keys, values)):
            cache.key_cache[index] = key
            cache.value_cache[index] = value
        return
    for layer, key, value in zip(cache.layers, keys, values):
        layer.keys, layer.values = key, value


def _cache_from_tensors(keys, values):
    """Build a DynamicCache from tensors, going through update() so the cache's own
    sequence-length bookkeeping stays consistent (assigning .layers[i].keys directly
    leaves it at zero and the next forward then mis-builds its mask)."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for index, (key, value) in enumerate(zip(keys, values)):
        cache.update(key, value, index)
    return cache


def prefill_only(model, projector: "LatentProjector", prefill_kwargs: dict,
                 start_position: int, *, checkpoint: bool = False):
    """Multimodal prefill for one view; returns (trajectory seed, cache, start, length).

    With `checkpoint=True` the prefill runs under activation checkpointing: the
    per-layer activations of a ~2.5k-token multimodal prompt are the single largest
    block of memory in the step (~14 GiB each), and they are rebuilt once during the
    backward pass. The recomputation is exact, so nothing about the objective
    changes; it costs one extra prefill forward and buys back enough memory to put
    several rows through each weight read.

    The cached keys/values are returned as plain tensors (not a cache object) because
    `torch.utils.checkpoint` only propagates tensor outputs.
    """
    from transformers.cache_utils import DynamicCache

    core = inner_model(model)
    kwargs = dict(prefill_kwargs)
    length = int(kwargs["input_ids"].shape[1])
    device = kwargs["input_ids"].device
    positions = kwargs.pop("cache_position", torch.arange(length, device=device))

    def run(input_ids=None, attention_mask=None, pixel_values=None, image_grid_thw=None):
        # The outer checkpoint must see the exact same graph during its original
        # forward and backward recomputation.  HF's per-layer checkpoint flag is
        # normally restored before backward and would otherwise suppress the KV
        # cache only during recomputation, producing a different tensor count.
        with without_gradient_checkpointing(model):
            cache = DynamicCache()
            out = core(input_ids=input_ids, attention_mask=attention_mask,
                       pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                       cache_position=positions, past_key_values=cache,
                       use_cache=True, return_dict=True)
            keys, values = _cache_tensors(cache)
            return (out.last_hidden_state[:, -1:, :], *keys, *values)

    args = (kwargs["input_ids"], kwargs.get("attention_mask"),
            kwargs.get("pixel_values"), kwargs.get("image_grid_thw"))
    if checkpoint:
        from torch.utils.checkpoint import checkpoint as _checkpoint
        outputs = _checkpoint(run, *args, use_reentrant=False)
    else:
        outputs = run(*args)

    depth = (len(outputs) - 1) // 2
    cache = _cache_from_tensors(list(outputs[1:1 + depth]), list(outputs[1 + depth:]))
    return outputs[0], cache, int(start_position), length


def rollout_latents_shared(model, projector: "LatentProjector", states: list, *,
                           steps: int) -> torch.Tensor:
    """Run one batched latent loop over several prefilled views: (B, steps, d).

    Each view keeps its own cache and its own rope position; the shorter caches are
    left-padded with zeros only so that a single `cache_position` is valid for the
    whole batch. The zero slots are masked out by `attention_mask`, so every row
    computes exactly what it would compute on its own — the saving is that the 8B
    weights are streamed once per step instead of once per view.
    """
    hidden, caches, starts, lengths = zip(*states)
    batch = len(states)
    device = hidden[0].device
    width = max(lengths)

    merged_keys, merged_values = [], []
    for cache, length in zip(caches, lengths):
        keys, values = _cache_tensors(cache)
        pad = width - length
        if pad > 0:
            keys = [torch.cat([torch.zeros((*k.shape[:-2], pad, k.shape[-1]),
                                           dtype=k.dtype, device=k.device), k], dim=-2)
                    for k in keys]
            values = [torch.cat([torch.zeros((*v.shape[:-2], pad, v.shape[-1]),
                                             dtype=v.dtype, device=v.device), v], dim=-2)
                      for v in values]
        merged_keys.append(keys)
        merged_values.append(values)

    merged = caches[0]
    _set_cache_tensors(
        merged,
        [torch.cat([row[index] for row in merged_keys], dim=0)
         for index in range(len(merged_keys[0]))],
        [torch.cat([row[index] for row in merged_values], dim=0)
         for index in range(len(merged_values[0]))],
    )

    mask = torch.zeros((batch, width), dtype=torch.long, device=device)
    for row, length in enumerate(lengths):
        mask[row, width - length:] = 1
    hidden = torch.cat([h for h in hidden], dim=0)
    start = torch.tensor([int(s) for s in starts], device=device)

    latents: list[torch.Tensor] = []
    with without_gradient_checkpointing(model):
        for step in range(steps):
            z = projector(hidden)
            latents.append(z[:, 0])
            mask = torch.cat([mask, mask.new_ones((batch, 1))], dim=1)
            positions = (start + step).view(1, batch, 1).expand(3, batch, 1)
            out = inner_model(model)(
                inputs_embeds=z,
                attention_mask=mask,
                past_key_values=merged,
                cache_position=torch.tensor([width + step], dtype=torch.long, device=device),
                position_ids=positions,
                use_cache=True,
                return_dict=True,
            )
            hidden = out.last_hidden_state[:, -1:, :]
    return torch.stack(latents, dim=1)
