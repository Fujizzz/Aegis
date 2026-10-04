"""Offline scheduling, recovery and report contracts; synthetic data only."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_ready_baselines import baseline_fixture
from test_pseudo_quality import fixture
from prepare_experiments import prepare
from run_experiments import load_matrix, run_key, execute, require_gate, select
from utils.pseudo_quality import object_hash
from utils.run_archive import write_json, sha256
from summarize_experiments import summarize


class PipelineTests(unittest.TestCase):
    def test_matrix_all_baselines_three_neural_seeds_and_no_extra_api_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root/'source')
            cfg=dict(model={},data={},runtime={},experiment={},hardware={},training={})
            matrix=prepare(cfg,root/'source',root/'matrix',model_path='${AEGIS_MODEL_ROOT}',
                model_manifest='${AEGIS_MODEL_MANIFEST}',seeds=[42,43,44],include_baselines=True)
            self.assertEqual(len(matrix['runs']),145)
            self.assertEqual(sum(r['variant']=='single_agent' for r in matrix['runs']),15)
            self.assertEqual(sum(r['variant']=='teacher_direct' for r in matrix['runs']),5)
            self.assertEqual(sum(r['variant']=='tfidf_nb' for r in matrix['runs']),5)
            self.assertEqual(len(select(matrix,'all')),145)
            self.assertEqual(select(matrix,'all')[0]['variant'],'full')
            load_matrix(root/'matrix/matrix.json')
            path=root/'matrix'/matrix['runs'][0]['config']
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'hash'):load_matrix(root/'matrix/matrix.json')

    def test_default_clis_only_inspect_without_writing_or_credentials(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,path,_=baseline_fixture(root,'teacher_direct')
            source=Path(__file__).resolve().parents[1]
            for script,args in [('run_experiments.py',['--matrix',str(root/'plan/matrix.json')]),
                                ('run_baseline.py',['--config',str(path)]),
                                ('resume_acceptance.py',['--acceptance-dir',str(root/'nonexistent')])]:
                command=[sys.executable,str(source/script),*args,'--output',str(root/'should_not_exist')]
                child=subprocess.run(command,capture_output=True,text=True,timeout=60)
                self.assertEqual(child.returncode,0,child.stderr)
                self.assertIn('planned_not_executed',child.stdout)
                self.assertFalse((root/'should_not_exist').exists())

    def test_required_acceptance_cannot_be_dry_or_wrong_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);matrix,_,_,_=baseline_fixture(root)
            path=root/'gate';path.mkdir()
            for value in (dict(status='planned_not_executed'),dict(status='passed',data_freeze_sha256='wrong')):
                write_json(path/'acceptance.json',value)
                with self.assertRaises(ValueError):require_gate(path,matrix)
            with patch('run_experiments.invoke') as call:
                with self.assertRaisesRegex(ValueError,'before data release'):execute(root/'plan/matrix.json',root/'pipeline','all')
                call.assert_not_called()

    def test_order_skip_complete_and_stop_after_failed_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);matrix,_,_,_=baseline_fixture(root)
            # Synthetic release flag is restricted to this fixture, never real artifacts.
            matrix['data_ready_for_formal_training']=True
            matrix['matrix_sha256']=object_hash({k:v for k,v in matrix.items() if k!='matrix_sha256'})
            write_json(root/'plan/matrix.json',matrix)
            calls=[]
            def invoke(command,log):
                calls.append(Path(command[1]).name)
                if '--output' in command:target=Path(command[command.index('--output')+1])
                else:target=Path(command[command.index('--save_dir')+1])
                target.mkdir(parents=True)
                if calls[-1]=='server_acceptance.py':
                    config=Path(command[command.index('--config')+1])
                    write_json(target/'acceptance.json',dict(status='passed',data_freeze_sha256=matrix['source_manifest_sha256'],
                                                           source_config_sha256=sha256(config)))
                elif calls[-1]=='resume_acceptance.py':
                    accept=Path(command[command.index('--acceptance-dir')+1])
                    write_json(target/'acceptance.json',dict(status='passed',data_freeze_sha256=matrix['source_manifest_sha256'],
                                                           acceptance_sha256=sha256(accept/'acceptance.json')))
                else:write_json(target/'status.json',dict(status='completed'))
            with patch.dict(os.environ,{'AEGIS_DATA_ROOT':str(root/'source'),'AEGIS_RUNS_ROOT':str(root/'runs')}), \
                 patch('run_experiments.invoke',side_effect=invoke),patch('run_experiments.verify_run',return_value={}):
                execute(root/'plan/matrix.json',root/'pipeline','all')
                self.assertEqual(calls[:3],['server_acceptance.py','resume_acceptance.py','train.py'])
                self.assertEqual(len(calls),13)
                calls.clear()
                execute(root/'plan/matrix.json',root/'pipeline','all')
                self.assertFalse(calls)
                with patch('run_experiments.invoke',side_effect=RuntimeError('synthetic failure')) as failing:
                    with self.assertRaises(RuntimeError):execute(root/'plan/matrix.json',root/'failed','all')
                    self.assertEqual(failing.call_count,1)
                    ledger=json.loads((root/'failed/ledger.json').read_text())
                    self.assertFalse(ledger['runs'])

    def test_summary_preserves_missing_cells_and_single_seed_std_null(self):
        from run_baseline import execute as run_text
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);matrix,item,path,cfg=baseline_fixture(root)
            pipeline=root/'pipeline';pipeline.mkdir()
            ledger=dict(matrix_sha256=matrix['matrix_sha256'],runs={})
            write_json(pipeline/'ledger.json',ledger)
            result=summarize(root/'plan/matrix.json',pipeline,root/'empty_report')
            self.assertEqual(len(result['missing']),11)
            self.assertTrue(all(row['mean'] is None for row in result['dispersion']))
            with patch.dict(os.environ,{'AEGIS_DATA_ROOT':str(root/'source'),'AEGIS_RUNS_ROOT':str(root/'runs')}):
                run_text(cfg,path,root/'text_run')
                ledger['runs'][run_key(item)]=[str(root/'text_run')]
                write_json(pipeline/'ledger.json',ledger)
                result=summarize(root/'plan/matrix.json',pipeline,root/'partial_report')
                self.assertEqual(len(result['missing']),10)
                text=[r for r in result['dispersion'] if r['variant']=='tfidf_nb']
                self.assertTrue(all(r['complete'] and r['mean'] is not None and r['sample_std'] is None for r in text))


class ControlledRecoveryTests(unittest.TestCase):
    def test_full_and_single_controlled_stop_resume_match_continuous_cpu(self):
        from test_resume import ResumeTests
        from training.trainer import Trainer
        from training.options import resolve_variant
        from resume_acceptance import compare
        for variant in ('full','single_agent'):
            with self.subTest(variant=variant):
                helper=ResumeTests('test_full_state_restore_and_source_immutable')
                helper.setUp()
                try:
                    helper.cfg['experiment']['variant']=variant
                    helper.cfg=resolve_variant(helper.cfg)
                    full=helper.fresh('continuous');full.train_loop()
                    helper.cfg['runtime']['stop_after_epoch']=1
                    interrupted=helper.fresh('interrupted');interrupted.train_loop()
                    interrupted_root=Path(interrupted.save_dir)
                    self.assertEqual(json.loads((interrupted_root/'status.json').read_text())['status'],'interrupted')
                    self.assertFalse((interrupted_root/'test_predictions.csv').exists())
                    hashes=helper.hashes(interrupted_root)
                    cfg=helper.resume_config(interrupted,'resumed');cfg['runtime'].pop('stop_after_epoch')
                    cfg,run=helper.prepare(interrupted,cfg=cfg)
                    resumed=Trainer(cfg);resumed.train_loop()
                    result=compare(Path(full.save_dir),run)
                    self.assertEqual(result['prediction_max_error'],0)
                    self.assertEqual(helper.hashes(interrupted_root),hashes)
                finally:helper.doCleanups()


if __name__=='__main__':unittest.main()
