"""Protocol checks for source matching, causal scoring and cluster statistics."""
from __future__ import annotations

import copy
import unittest

from vep_opsd.visual_gain_data import matched_permutation
from vep_opsd.visual_gain_report import cluster_bootstrap, evaluate_groups


def fixture_rows(n=6):
    return [{'sample_id': str(i), 'source_id': f'source-{i}', 'category': 'Predictive',
             'question': f'What will the person do with the {"cup" if i % 2 else "ball"}?',
             'answer': 'This must not affect matching.', 'past_files': [f'{i}.png']}
            for i in range(n)]


class MatchingTests(unittest.TestCase):
    def test_permutation_is_deterministic_source_disjoint_and_gold_independent(self):
        rows = fixture_rows()
        donors, metadata = matched_permutation(rows)
        altered = copy.deepcopy(rows)
        for row in altered:
            row['answer'] = 'a completely different reference'
        self.assertEqual(donors, matched_permutation(altered)[0])
        self.assertEqual(sorted(donors), list(range(len(rows))))
        self.assertTrue(all(i != j for i, j in enumerate(donors)))
        self.assertFalse(metadata['gold_used_for_matching'])

    def test_odd_stratum_uses_no_self_donor(self):
        donors, _ = matched_permutation(fixture_rows(3))
        self.assertEqual(sorted(donors), [0, 1, 2])
        self.assertTrue(all(i != j for i, j in enumerate(donors)))

    def test_different_frame_count_fallback_is_recorded(self):
        rows = fixture_rows(4)
        for i, row in enumerate(rows):
            row['past_files'] *= i + 1
        _, metadata = matched_permutation(rows)
        self.assertTrue(any(p['frame_count'] != p['donor_frame_count'] for p in metadata['pairs']))

    def test_same_original_source_is_not_an_eligible_donor(self):
        rows = fixture_rows(4)
        for i, row in enumerate(rows):
            row['source_id'] = 'A' if i < 2 else 'B'
        donors, _ = matched_permutation(rows)
        self.assertTrue(all(rows[i]['source_id'] != rows[j]['source_id'] for i, j in enumerate(donors)))

    def test_singleton_category_cannot_silently_cross_categories(self):
        rows = fixture_rows(2)
        rows[0]['category'] = 'Camera'
        with self.assertRaises(ValueError):
            matched_permutation(rows)


class BootstrapTests(unittest.TestCase):
    def test_source_cluster_not_sample_is_the_resampling_unit(self):
        a = cluster_bootstrap([('A', [1.]), ('B', [3.])], repeats=200)
        b = cluster_bootstrap([('A', [1.]), ('A', [1.]), ('B', [3.])], repeats=200)
        for key in ('mean', 'ci95_low', 'ci95_high'):
            self.assertEqual(a[key], b[key])
        self.assertEqual(b['sources'], 2)
        self.assertEqual(b['samples'], 3)

    def test_condition_tokenization_mismatch_is_rejected(self):
        row = {'sample_id': 'a', 'source_id': 's', 'question': 'Q', 'gold': 'A',
               'category': 'Predictive', 'gold_token_ids': [1], 'gold_tokens': 1,
               'nll_mean': 1., 'nll_sum': 1., 'token_f1': 1., 'exact': 1.,
               'truncated': False, 'prediction': 'A'}
        outputs = {name: {'a': copy.deepcopy(row)} for name in ('q_only', 'vq', 'shuffled_vq', 'cdlsd_k4')}
        outputs['cdlsd_k4']['a']['gold_token_ids'] = [2]
        with self.assertRaises(ValueError):
            evaluate_groups(outputs, repeats=10)


class CausalScoringTests(unittest.TestCase):
    def test_fp32_text_and_native_multimodal_cache_match_full_gold_scoring(self):
        import torch
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        from vep_opsd.visual_gain_eval import verify_native_alignment, native_logits, score_logits

        torch.manual_seed(7)
        torch.set_num_threads(2)
        config = Qwen3VLConfig(
            image_token_id=250, video_token_id=249, vision_start_token_id=248,
            text_config=dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, intermediate_size=128, vocab_size=256,
                             max_position_embeddings=512, head_dim=16,
                             rope_scaling=dict(mrope_section=[3, 3, 2],
                                               mrope_interleaved=True, rope_type='default')),
            vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
                               out_hidden_size=64, patch_size=16, spatial_merge_size=2,
                               temporal_patch_size=2, num_position_embeddings=64,
                               deepstack_visual_indexes=[0]))
        model = Qwen3VLForConditionalGeneration(config).float().eval()
        gold = torch.tensor([11, 12, 13, 14])
        with torch.no_grad():
            for image in (False, True):
                ids = [1, 248, 250, 250, 250, 250, 251, 2, 3] if image else [1, 2, 3]
                batch = {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones((1, len(ids)), dtype=torch.long)}
                if image:
                    batch.update(pixel_values=torch.randn(16, 3 * 2 * 16 * 16), image_grid_thw=torch.tensor([[1, 4, 4]]))
                verify_native_alignment(model, batch, gold)
                scores = score_logits(native_logits(model, batch, gold), gold)
                self.assertEqual(scores['gold_tokens'], len(gold))
                self.assertAlmostEqual(scores['nll_sum'] / len(gold), scores['nll_mean'], places=5)


if __name__ == '__main__':
    unittest.main()
