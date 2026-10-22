"""No 8B checkpoint needed: verify differentiable recurrent latent replay semantics."""
import unittest
from types import SimpleNamespace
import torch
from torch import nn

from vep_opsd.eipd_replay import answer_prediction_rows, replay_with_latents

class FakeFutureL1(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(60, 8)
        self.project = nn.Linear(8, 8)
        self.head = nn.Linear(8, 60)

    def forward(self, input_ids, attention_mask=None, lvr_mask=None,
                lvr_states=None, prompt_length=None, logits_to_keep=0, **kwargs):
        x = self.embed(input_ids)
        if lvr_states is not None:
            s = int(prompt_length)
            suffix = torch.where(lvr_mask[..., None], lvr_states, x[:, s:])
            x = torch.cat([x[:, :s], suffix], dim=1)
        # A simple causal recurrent approximation to mock non-detached states.
        h = x.cumsum(dim=1)
        logits = self.head(h[:, -logits_to_keep:] if logits_to_keep else h)
        return SimpleNamespace(latent_hidden_state=self.project(h[:, -1, :]), logits=logits)

class ReplayTest(unittest.TestCase):
    def test_two_latents_and_answer_loss_reach_generator(self):
        torch.manual_seed(123)
        mock = FakeFutureL1().train()
        prompt = {"input_ids": torch.tensor([[1, 2]])}
        # [latent_start, latent, text, latent, text, answer_0, answer_1]
        continuation = torch.tensor([[10, 42, 11, 42, 12, 13, 14]])
        logits, states = replay_with_latents(mock, prompt, continuation,
                                             latent_id=42, answer_tokens=2)
        self.assertEqual(len(states), 2)
        self.assertEqual(tuple(logits.shape), (1, 3, 60))
        rows = answer_prediction_rows(logits, prompt_tokens=2, reasoning_tokens=5, answer_tokens=2)
        self.assertEqual(tuple(rows.shape), (2, 60))
        loss = torch.nn.functional.cross_entropy(rows, torch.tensor([13, 14]))
        latent_grad = torch.autograd.grad(loss, tuple(states.values()), retain_graph=True)
        self.assertTrue(all(grad is not None for grad in latent_grad))
        loss.backward()
        self.assertIsNotNone(mock.project.weight.grad)
        self.assertGreater(float(mock.project.weight.grad.abs().sum()), 0.0)

    def test_latents_required(self):
        mock = FakeFutureL1().train()
        prompt = {"input_ids": torch.tensor([[1, 2]])}
        with self.assertRaisesRegex(ValueError, "No latent"):
            replay_with_latents(mock, prompt, torch.tensor([[3, 4, 5]]),
                                latent_id=42, answer_tokens=2)

if __name__ == "__main__":
    unittest.main()
