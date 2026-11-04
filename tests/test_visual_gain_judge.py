"""Non-GPU validation of the blinded semantic review contract."""
import json
from pathlib import Path
import tempfile
import unittest

from vep_opsd.visual_gain_judge import messages, parse_decision, recover


class SemanticContractTests(unittest.TestCase):
    def test_prompt_does_not_include_condition_or_model_metadata(self):
        item = {'review_id': 'blind-id', 'question': 'Which way?', 'gold': 'Right.',
                'candidate': 'Left.', 'condition': 'cdlsd_k4', 'model': 'secret-model'}
        prompt = messages(item)
        content = json.dumps(prompt)
        self.assertNotIn('cdlsd_k4', content)
        self.assertNotIn('secret-model', content)
        self.assertEqual(set(json.loads(prompt[1]['content'])), {'question', 'reference', 'candidate'})

    def test_uncertain_is_not_implicitly_incorrect(self):
        row = parse_decision('{"verdict":"uncertain","error_types":[],"reason":"Plausible alternative future."}')
        self.assertEqual(row['verdict'], 'uncertain')
        self.assertEqual(row['error_types'], [])

    def test_unknown_labels_and_non_error_error_types_are_rejected(self):
        for text in ['{"verdict":"mostly correct","error_types":[],"reason":"x"}',
                     '{"verdict":"correct","error_types":["direction"],"reason":"x"}',
                     '{"verdict":"incorrect","error_types":["word_overlap"],"reason":"x"}']:
            with self.assertRaises(ValueError):
                parse_decision(text)

    def test_fenced_json_can_be_read_without_inventing_labels(self):
        row = parse_decision('```json\n{"verdict":"incorrect","error_types":["direction"],"reason":"Opposite direction."}\n```')
        self.assertEqual(row['error_types'], ['direction'])

    def test_partial_tail_is_recovered_but_mixed_protocol_is_not(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'labels.jsonl'
            complete = {'review_id': 'one', 'fingerprint': 'f', 'valid': True}
            path.write_text(json.dumps(complete)+'\n'+ '{"review_id":')
            self.assertEqual(set(recover(path, 'f')), {'one'})
            self.assertEqual(path.read_text(), json.dumps(complete)+'\n')
            with self.assertRaises(ValueError):
                recover(path, 'changed')


if __name__ == '__main__':
    unittest.main()
