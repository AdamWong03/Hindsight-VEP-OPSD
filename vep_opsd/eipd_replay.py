"""Differentiable Future-L1 latent replay using upstream Qwen3-VL lvr_* hooks.

Reference implementation: O(K) *full-prefix* forwards, memory-heavy. We
intentionally do NOT pretend that a single text-only teacher-forced forward
reconstructs autoregressive continuous latent recurrence. GPU performance
optimization is deferred until this version passes end-to-end parity tests.
"""
from __future__ import annotations
import torch

def _prefix_kwargs(encoded: dict, ids: torch.Tensor, prompt_length: int,
                   states: dict[int, torch.Tensor]) -> dict:
    result = dict(encoded)
    result["input_ids"] = ids
    result["attention_mask"] = torch.ones_like(ids, dtype=torch.long)
    result["logits_to_keep"] = 1
    result["use_cache"] = False
    result["return_dict"] = True
    if states:
        count = ids.shape[1] - prompt_length
        if count <= 0:
            raise RuntimeError("Latents cannot occur inside the immutable prompt")
        dim = next(iter(states.values())).shape[-1]
        zero = next(iter(states.values())).new_zeros(dim)
        mask = torch.tensor([i in states for i in range(prompt_length, ids.shape[1])],
                            dtype=torch.bool, device=ids.device).unsqueeze(0)
        state_list = [states.get(i, zero) for i in range(prompt_length, ids.shape[1])]
        result.update(lvr_mask=mask, lvr_states=torch.stack(state_list).unsqueeze(0),
                      prompt_length=prompt_length)
    return result

def replay_with_latents(model, encoded_prompt: dict, continuation_ids: torch.Tensor,
                        latent_id: int, *, require_latents: bool = True,
                        answer_tokens: int | None = None):
    """Recompute model-specific continuous states; gradients flow into trainable layers.

    Requires Future-L1 monkey patches, including lvr_mask, lvr_states,
    prompt_length and Qwen3-VL GenerationOutput.latent_hidden_state.
    No torch.no_grad. The prefix is fixed but each latent is freshly generated.
    """
    if continuation_ids.ndim != 2 or continuation_ids.shape[0] != 1:
        raise ValueError("Single-sample continuation IDs required")
    prompt_ids = encoded_prompt["input_ids"]
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("Single-sample prompt IDs required")
    prefix_len = prompt_ids.shape[1]
    ids = torch.cat([prompt_ids, continuation_ids], dim=1)
    latent_positions = [i for i in range(prefix_len, ids.shape[1])
                        if ids[0, i].item() == latent_id]
    if require_latents and not latent_positions:
        raise ValueError("No latent tokens in student's sampled reasoning")
    states: dict[int, torch.Tensor] = {}
    for position in latent_positions:
        # The previous token's hidden state predicts the latent input at position.
        prefix = ids[:, :position]
        output = model(**_prefix_kwargs(encoded_prompt, prefix, prefix_len, states))
        state = getattr(output, "latent_hidden_state", None)
        if state is None or state.ndim != 2 or state.shape[0] != 1:
            raise RuntimeError("Future-L1 projected latent_hidden_state missing: check upstream patches")
        if not state.requires_grad and model.training:
            raise RuntimeError("Latent recurrence unexpectedly detached from autograd")
        states[position] = state[0]
    final_kwargs = _prefix_kwargs(encoded_prompt, ids, prefix_len, states)
    # Only answer prediction rows + one unused final row are needed.
    final_kwargs["logits_to_keep"] = answer_tokens + 1 if answer_tokens is not None else 0
    outputs = model(**final_kwargs)
    logits = outputs.logits
    expected = answer_tokens + 1 if answer_tokens is not None else ids.shape[1]
    if logits.shape[1] != expected:
        raise RuntimeError(f"Model returned {logits.shape[1]} logits, expected {expected}")
    return logits, states

def answer_prediction_rows(logits: torch.Tensor, prompt_tokens: int, reasoning_tokens: int,
                           answer_tokens: int) -> torch.Tensor:
    """Return [T,V] where row j predicts answer token j under causal shift."""
    start = prompt_tokens + reasoning_tokens
    if logits.shape[1] == answer_tokens + 1:
        selected = logits[0, :answer_tokens, :]
    else:
        selected = logits[0, start - 1:start - 1 + answer_tokens, :]
    if selected.shape[0] != answer_tokens:
        raise ValueError("Answer logit alignment failed")
    return selected
