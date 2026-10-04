"""Synthetic CPU fixtures and mock HTTP only; no pretrained weights/API calls."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from test_training_core import TinyAgent, fixture_trainer, records
from test_pseudo_quality import fixture
from baselines import teacher_direct as teacher
from baselines.tfidf_nb import TfidfNB
from baselines.common import export
from utils import freeze_pheme as f
from utils.checkpoints import checkpoint_paths
from utils.run_archive import write_json
from prepare_experiments import prepare
from run_baseline import execute
from run_experiments import load_matrix, verify_run


def response(content='{"p_positive":0.65,"rationale":"Synthetic unverified claim."}'):
    return dict(id='synthetic',model='mock-classifier',choices=[dict(finish_reason='stop',message=dict(content=content))])


def baseline_fixture(root, variant='tfidf_nb'):
    fixture(root/'source')
    cfg=dict(model=dict(comm_rounds=2),data={'label_schema':'pheme_rumour_v1'},runtime={},
             experiment={'method_id':'aegis_neural_v1'},hardware={},training={})
    matrix=prepare(cfg,root/'source',root/'plan',model_path='unused',model_manifest='unused',
                   events=['germanwings-crash'],include_baselines=True)
    item=next(r for r in matrix['runs'] if r['variant']==variant)
    path=root/'plan'/item['config']
    return matrix,item,path,json.loads(path.read_text())


class TextTests(unittest.TestCase):
    def test_train_only_vocabulary_serialize_and_extreme_inputs(self):
        model=TfidfNB(min_df=1).fit(['blue official','red unverified','blue confirmed','red alleged'],[0,1,0,1])
        unseen='holdout_unique_token'
        before=copy.deepcopy(model.state())
        values=model.predict(['blue official','red alleged',unseen,''])
        self.assertLess(values[0],values[1])
        self.assertEqual(values[2:],[.5,.5])
        self.assertEqual(before,model.state())
        self.assertNotIn(unseen,model.idf)
        self.assertEqual(TfidfNB.restore(json.loads(json.dumps(before))).predict(['red alleged']),values[1:2])
        for texts,targets in (([],[]),(['only one'],[0]),(['',''],[0,1])):
            with self.assertRaises(ValueError):TfidfNB().fit(texts,targets)

    def test_text_archive_and_independent_provenance_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            matrix,item,path,cfg=baseline_fixture(root)
            with patch.dict(os.environ,{'AEGIS_DATA_ROOT':str(root/'source'),'AEGIS_RUNS_ROOT':str(root/'runs')}), \
                 patch('utils.run_archive.model_manifest',side_effect=AssertionError('Must not load model')):
                result=execute(cfg,path,root/'run')
                self.assertEqual(result['status'],'completed')
                report=verify_run(root/'run',item,matrix,root/'plan/matrix.json')
                self.assertEqual(report['test']['sample_pooled']['n'],2)
                with self.assertRaises(FileExistsError):execute(cfg,path,root/'run')
                source=root/'run/data/test.csv'
                source.write_text(source.read_text()+'\n')
                with self.assertRaisesRegex(ValueError,'integrity'):verify_run(root/'run',item,matrix,root/'plan/matrix.json')


class TeacherTests(unittest.TestCase):
    def test_strict_format_refuses_fallback_and_incomplete_output(self):
        self.assertEqual(teacher.strict_probability(response()),.65)
        contents=['{"p_positive":true,"rationale":"x"}','{"p_positive":NaN,"rationale":"x"}',
            '{"p_positive":1.1,"rationale":"x"}','{"p_positive":0.2,"rationale":""}',
            '{"p_positive":0.2,"p_positive":0.3,"rationale":"x"}',
            '{"p_positive":0.2,"rationale":"x","label":1}','```json\n{}\n```']
        for text in contents:
            with self.subTest(text=text),self.assertRaises(ValueError):teacher.strict_probability(response(text))
        truncated=response();truncated['choices'][0]['finish_reason']='length'
        with self.assertRaises(ValueError):teacher.strict_probability(truncated)

    def test_plan_has_no_gold_labels_and_resume_skips_successes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,path,cfg=baseline_fixture(root,'teacher_direct')
            for split in ('train','val','test'):
                cfg['data'][split+'_file']=str(root/'source/split_germanwings'/(split+'.csv'))
            plan=teacher.make_plan(cfg)
            self.assertEqual({r['split'] for r in plan['requests']},{'val','test'})
            original=copy.deepcopy(plan)
            for split in ('val','test'):
                rows=f.read_csv(cfg['data'][split+'_file'])
                for row in rows:row['label']=str(1-int(row['label']))
                f.write_csv(Path(cfg['data'][split+'_file']),rows,list(rows[0]))
            self.assertEqual(teacher.make_plan(cfg),original)  # Gold labels cannot influence API payload.
            log=root/'log.jsonl'
            transport=unittest.mock.Mock(return_value=response())
            self.assertEqual(teacher.generate(plan,log,transport=transport)['planned'],4)
            self.assertFalse(log.exists());transport.assert_not_called()
            with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'mock'}):
                first=teacher.generate(plan,log,execute=True,limit=1,transport=transport)
                self.assertEqual(first['remaining'],3)
                second=teacher.generate(plan,log,execute=True,transport=transport)
                self.assertEqual(second['status'],'complete')
                teacher.generate(plan,log,execute=True,transport=transport)
                self.assertEqual(transport.call_count,4)
            log.write_text(log.read_text().replace('mock-classifier','changed'))
            with self.assertRaises(ValueError):teacher.generate(plan,log)

    def test_error_stops_dispatch_and_full_run_recovers_exports_without_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);matrix,item,path,cfg=baseline_fixture(root,'teacher_direct')
            transport=unittest.mock.Mock(side_effect=[response(),OSError('do not persist secret text')])
            with patch.dict(os.environ,{'AEGIS_DATA_ROOT':str(root/'source'),'AEGIS_RUNS_ROOT':str(root/'runs'),
                                        'DEEPSEEK_API_KEY':'mock'}),patch('utils.pseudo_quality.http_transport',transport):
                self.assertEqual(execute(cfg,path,root/'run')['status'],'partial')
                self.assertEqual(transport.call_count,2)
                self.assertNotIn('do not persist', (root/'run/teacher_responses.jsonl').read_text())
                transport.side_effect=None;transport.return_value=response()
                self.assertEqual(execute(cfg,path,root/'run',resume=True)['status'],'completed')
                self.assertEqual(transport.call_count,5)
                before=f.file_hash(root/'run/test_predictions.csv')
                execute(cfg,path,root/'run',resume=True)
                self.assertEqual(transport.call_count,5)
                self.assertEqual(before,f.file_hash(root/'run/test_predictions.csv'))
                verify_run(root/'run',item,matrix,root/'plan/matrix.json')


class SingleAgentTests(unittest.TestCase):
    def test_non_neural_variants_fail_before_encoder_construction(self):
        from training.trainer import Trainer
        from inference.predictor import Predictor
        for variant in ('tfidf_nb','teacher_direct'):
            with self.subTest(variant=variant),patch('training.trainer.BaseAgent') as ctor:
                with self.assertRaisesRegex(ValueError,'non-neural'):Trainer({'experiment':{'variant':variant}})
                ctor.assert_not_called()
                with self.assertRaisesRegex(ValueError,'text/API'):Predictor({'experiment':{'variant':variant}}, {})

    def test_one_encoder_gradient_and_checkpoint_predictor_roundtrip(self):
        from inference.predictor import Predictor
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()), \
             patch('training.trainer.tqdm',lambda rows,**kw:rows):
            torch.set_num_threads(1)
            trainer=fixture_trainer(Path(temp)/'run',n=4,epochs=1,variant='single_agent')
            self.assertEqual(sum(isinstance(getattr(trainer,k),TinyAgent) for k in ('agent_e','agent_x','agent_b','agent_c')),1)
            before={k:v.detach().clone() for k,v in trainer.agent_e.encoder.state_dict().items()}
            trainer.train_loop()
            losses=records(trainer.metrics_path)[0]['train_losses']
            self.assertTrue(all(losses[k]==0 for k in ('dim','cons','pred','div','gate')))
            self.assertTrue(any(not torch.equal(v,trainer.agent_e.encoder.state_dict()[k]) for k,v in before.items()))
            self.assertFalse(any(p.requires_grad for p in trainer.agent_x.parameters()))
            batch=next(iter(trainer.val_loader))
            enc={'input_ids':batch['input_ids'][:1],'attention_mask':batch['attention_mask'][:1]}
            expected=trainer.final_head(trainer.agent_e(enc['input_ids'],enc['attention_mask'])[1]).sigmoid().item()
            tokenizer=type('Tokenizer',(),{'pad_token':'pad'})()
            with patch('inference.predictor.BaseAgent',side_effect=TinyAgent) as ctor, \
                 patch('inference.predictor.AutoTokenizer.from_pretrained',return_value=tokenizer), \
                 patch('inference.predictor.tokenize_texts',return_value=enc):
                predictor=Predictor(trainer.cfg,checkpoint_paths(Path(trainer.save_dir)/'best'))
                actual=predictor.predict('synthetic text')
                self.assertEqual(ctor.call_count,1)
                self.assertAlmostEqual(actual['p_positive'],expected)
                self.assertNotIn('post_emotion',actual)


if __name__=='__main__':unittest.main()
