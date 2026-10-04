"""Plan generation and acceptance dry run only: no models or optimization."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_pseudo_quality import fixture
from test_pseudo_quality import teacher_decision, mock_response
from unittest.mock import patch
import os
from utils import pseudo_quality as q
from prepare_experiments import prepare, CORE
from server_acceptance import balanced_subset
from utils.pseudo_quality import read_json
from training.options import resolve_variant


class PreparationTests(unittest.TestCase):
    def test_teacher_only_matrix_is_ready_and_discloses_no_human_review(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'TEST_TEACHER_KEY':'synthetic'}):
            root = Path(temp)
            rows = fixture(root / 'source')
            plan = q.make_plan(root / 'source', rows)
            log = root / 'responses.jsonl'
            q.generate(plan, log, 'https://example.org', execute=True,
                       api_key_env='TEST_TEACHER_KEY', transport=mock_response)
            q.create_audit(root / 'source', root / 'audit', responses=log)
            teacher_decision(root / 'audit', root / 'decision.json')
            q.build_release(root / 'source', root / 'audit', None, root / 'teacher_v2',
                            release_policy='teacher_only', decision_record=root / 'decision.json')
            config = dict(model=dict(comm_rounds=2, use_gate=True), data={}, runtime={}, experiment={},
                          hardware={}, training={'loss_weights':{'gate':.1, 'cons':.5}})
            result = prepare(config, root / 'teacher_v2', root / 'matrix',
                             model_path='${AEGIS_MODEL_ROOT}', model_manifest='${AEGIS_MODEL_MANIFEST}')
            self.assertTrue(result['data_ready_for_formal_training'])
            self.assertEqual(result['pseudo_release_policy'], 'teacher_only')
            self.assertFalse(result['human_review_completed'])
            self.assertEqual(result['source_manifest_sha256'], q.f.file_hash(root / 'teacher_v2/freeze_manifest.json'))

    def test_same_split_single_factor_matrix_and_dry_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / 'source')
            config = dict(model=dict(comm_rounds=2, use_gate=True), data={},
                          runtime={'save_dir':'stale', 'run_id':'stale'}, experiment={},
                          hardware={}, training={'loss_weights':{'gate':.1, 'cons':.5}})
            original = copy.deepcopy(config)
            result = prepare(config, root / 'source', root / 'matrix', model_path='${AEGIS_MODEL_ROOT}',
                             model_manifest='${AEGIS_MODEL_MANIFEST}', events=['germanwings-crash'], seeds=[42,43])
            self.assertEqual(original, config)
            self.assertEqual(len(result['runs']), len(CORE) * 2)
            self.assertFalse(result['data_ready_for_formal_training'])
            configs = [read_json(root / 'matrix' / run['config']) for run in result['runs']]
            self.assertEqual(len({cfg['data']['train_file'] for cfg in configs}), 1)
            self.assertTrue(all('save_dir' not in cfg['runtime'] for cfg in configs))
            no_comm = next(cfg for cfg in configs if cfg['experiment']['variant'] == 'no_comm')
            self.assertEqual(no_comm['model']['comm_rounds'], 2)
            self.assertEqual(resolve_variant(no_comm)['model']['comm_rounds'], 0)
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'server_acceptance.py'),
                       '--config', str(root / 'matrix' / result['runs'][0]['config']), '--output', str(root / 'accept')]
            subprocess.run(command, check=True, capture_output=True)
            status = read_json(root / 'accept' / 'acceptance.json')
            self.assertFalse(status['model_loaded'])
            self.assertFalse(status['training_started'])

    def test_subset_is_deterministic_balanced_and_fails_if_insufficient(self):
        rows = [dict(label=str(i % 2), sample_id=str(i)) for i in range(12)]
        self.assertEqual(balanced_subset(rows, 8), balanced_subset(list(reversed(rows)), 8))
        self.assertEqual(sum(r['label'] == '1' for r in balanced_subset(rows, 8)), 4)
        with self.assertRaises(ValueError):
            balanced_subset(rows[:2], 8)


if __name__ == '__main__':
    unittest.main()
