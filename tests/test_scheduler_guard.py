import unittest

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from vep_opsd.trainer import drop_empty_param_groups, sync_scheduler_lambdas


def make_optimizer(with_empty_group: bool):
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    groups = [{"params": list(model.parameters()), "weight_decay": 0.1}]
    if with_empty_group:
        groups.insert(0, {"params": [], "weight_decay": 0.1})
    return model, torch.optim.AdamW(groups, lr=1e-5)


class EmptyParamGroupTest(unittest.TestCase):
    """HF's decay/no-decay split leaves an empty group for a LoRA student, and
    DeepSpeed's ZeRO-2 wrapper then exposes fewer groups than the scheduler was
    built with; torch >= 2.9 raises in zip(..., strict=True)."""

    def test_empty_group_is_dropped_and_parameters_are_kept(self):
        model, optimizer = make_optimizer(with_empty_group=True)
        self.assertEqual(len(optimizer.param_groups), 2)
        before = {id(p) for p in optimizer.param_groups[1]["params"]}

        self.assertEqual(drop_empty_param_groups(optimizer), 1)

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual({id(p) for p in optimizer.param_groups[0]["params"]}, before)

    def test_no_empty_group_is_a_no_op(self):
        _, optimizer = make_optimizer(with_empty_group=False)
        self.assertEqual(drop_empty_param_groups(optimizer), 0)
        self.assertEqual(len(optimizer.param_groups), 1)


class SchedulerSyncTest(unittest.TestCase):
    def test_lambdas_follow_the_live_group_count(self):
        _, optimizer = make_optimizer(with_empty_group=False)
        scheduler = LambdaLR(optimizer, lambda epoch: 1.0)
        self.assertEqual(len(scheduler.lr_lambdas), 1)

        # Simulate DeepSpeed handing back an optimizer with fewer groups: the
        # scheduler still carries the lambdas it was built with.
        scheduler.lr_lambdas = scheduler.lr_lambdas * 2
        self.assertTrue(sync_scheduler_lambdas(scheduler, optimizer))
        self.assertEqual(len(scheduler.lr_lambdas), 1)

        optimizer.step()
        scheduler.step()          # torch would raise here without the alignment
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)

    def test_already_aligned_is_untouched(self):
        _, optimizer = make_optimizer(with_empty_group=False)
        scheduler = LambdaLR(optimizer, lambda epoch: 0.5)
        self.assertFalse(sync_scheduler_lambdas(scheduler, optimizer))

    def test_scheduler_without_lambdas_is_ignored(self):
        _, optimizer = make_optimizer(with_empty_group=False)
        self.assertFalse(sync_scheduler_lambdas(object(), optimizer))


if __name__ == "__main__":
    unittest.main()
