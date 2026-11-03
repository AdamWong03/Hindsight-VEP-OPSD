"""Shared latent rollout and answer reader for CDLSD-VEP."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .cdlsd_data import encode_multimodal_prefill
from .eipd_latent import (
    LatentProjector,
    _mrope_positions,
    inner_model,
    prefill_only,
    rollout_latents_shared,
    without_gradient_checkpointing,
)


@dataclass
class DepthStates:
    all_latents: torch.Tensor
    short: torch.Tensor
    long: torch.Tensor


def embedding_layer(model):
    module = model.module if hasattr(model, "module") else model
    return module.get_input_embeddings()


def latent_rollout(model, projector: LatentProjector, processor, images, question: str,
                   *, short_steps: int = 4, long_steps: int = 8,
                   checkpoint_prefill: bool = True) -> DepthStates:
    """One native multimodal prefill and one differentiable long latent chain."""
    if not 0 < short_steps < long_steps:
        raise ValueError("require 0 < short_steps < long_steps")
    if long_steps != 2 * short_steps:
        raise ValueError("v1 uses equal-capacity short/long windows (K_l = 2*K_s)")
    device = next(model.parameters()).device
    backbone = inner_model(model)
    prefill, start = encode_multimodal_prefill(
        backbone, processor, images, question, device)
    # Qwen's internal gradient-checkpoint wrappers suppress KV cache creation.
    # Disable only those wrappers; prefill_only supplies its own exact recompute.
    with without_gradient_checkpointing(model):
        seed = prefill_only(
            model, projector, prefill, start, checkpoint=checkpoint_prefill)
    latents = rollout_latents_shared(model, projector, [seed], steps=long_steps)[0]
    return DepthStates(
        all_latents=latents,
        short=latents[:short_steps],
        long=latents[long_steps - short_steps:long_steps],
    )


def _ids(tokenizer, text: str, device) -> torch.Tensor:
    values = tokenizer(text, add_special_tokens=False)["input_ids"]
    return torch.tensor([values], dtype=torch.long, device=device)


def answer_prefix(model, tokenizer, question: str, states: torch.Tensor):
    """Embedding prefix [Question ; four latent states ; <answer>]."""
    device = states.device
    embed = embedding_layer(model)
    question_ids = _ids(tokenizer, question, device)
    marker_ids = _ids(tokenizer, "<answer>", device)
    return torch.cat(
        [embed(question_ids), states.unsqueeze(0), embed(marker_ids)], dim=1)


def gold_token_ids(tokenizer, answer: str, device) -> torch.Tensor:
    """Gold response after the already-visible <answer> prefix."""
    return _ids(tokenizer, answer.strip() + "</answer>", device)[0]


def answer_logits(model, tokenizer, question: str, states: torch.Tensor,
                  response_ids: torch.Tensor) -> torch.Tensor:
    """Next-token logits on exactly the supplied response trajectory.

    Returns ``(L, vocab)``: row t predicts ``response_ids[t]`` from the common
    prefix and ``response_ids[:t]``.  Prompt, padding and latent positions are
    never included in the loss.
    """
    if response_ids.ndim != 1 or response_ids.numel() == 0:
        raise ValueError("response_ids must be a non-empty 1-D tensor")
    prefix = answer_prefix(model, tokenizer, question, states)
    response = embedding_layer(model)(response_ids.view(1, -1))
    inputs = torch.cat([prefix, response], dim=1)
    length = inputs.shape[1]
    out = model(
        inputs_embeds=inputs,
        cache_position=torch.arange(length, device=inputs.device),
        position_ids=_mrope_positions(0, length, inputs.device),
        use_cache=False,
        logits_to_keep=response_ids.numel() + 1,
        return_dict=True,
    )
    return out.logits[0, :response_ids.numel()]


def _sample_next(logits: torch.Tensor, *, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    scores = logits.float() / temperature
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        probabilities = torch.softmax(sorted_scores, dim=-1)
        remove = probabilities.cumsum(dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf"))
        scores.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)
    return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)


@torch.no_grad()
def sample_answer(model, tokenizer, question: str, states: torch.Tensor, *,
                  max_new_tokens: int = 48, temperature: float = 0.8,
                  top_p: float = 0.95) -> torch.Tensor:
    """Sample answer token IDs from the short path without retaining a graph."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    prefix = answer_prefix(model, tokenizer, question, states.detach())
    length = prefix.shape[1]
    out = model(
        inputs_embeds=prefix,
        cache_position=torch.arange(length, device=prefix.device),
        position_ids=_mrope_positions(0, length, prefix.device),
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )
    cache = out.past_key_values
    next_logits = out.logits[:, -1]
    produced: list[int] = []
    eos_id = tokenizer.eos_token_id
    for offset in range(max_new_tokens):
        token = _sample_next(next_logits, temperature=temperature, top_p=top_p)
        token_id = int(token.item())
        produced.append(token_id)
        decoded = tokenizer.decode(produced, skip_special_tokens=False)
        if token_id == eos_id or "</answer>" in decoded:
            break
        position = length + offset
        out = model(
            inputs_embeds=embedding_layer(model)(token),
            past_key_values=cache,
            cache_position=torch.tensor([position], device=prefix.device),
            position_ids=_mrope_positions(position, 1, prefix.device),
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        cache = out.past_key_values
        next_logits = out.logits[:, -1]
    return torch.tensor(produced, dtype=torch.long, device=prefix.device)


def decode_answer(tokenizer, response_ids: torch.Tensor) -> str:
    text = tokenizer.decode(response_ids.tolist(), skip_special_tokens=True)
    return text.split("</answer>", 1)[0].strip()
