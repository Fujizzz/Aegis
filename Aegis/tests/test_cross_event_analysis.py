"""Synthetic Cross-event/Teacher supervision/T25/T26 contracts; no teacher or pretrained model calls."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

from test_pseudo_quality import fixture
from test_training_core import fixture_trainer, TinyAgent, records
from utils import freeze_pheme as f
from utils.cross_event import build, verify, fold_rows
from utils.experiment_protocol import CROSS_EVENT
from utils.post_analysis import expertise, cases, correlation, ranks, stability_report
from utils.run_archive import sha256, write_json
from prepare_experiments import prepare
from run_experiments import verify_run, run_key
from run_baseline import execute
from summarize_experiments import summarize
from analyze_run import analyze


def neural_analysis_fixture(root):
    """Random tiny local Qwen2/LoRA through real archive/train/predict/analyze APIs."""
    from test_input_lora_checkpoint import local_model
    from utils.run_archive import prepare_run, model_manifest
    from training.trainer import Trainer
    from inference.predictor import Predictor
    from utils.checkpoints import checkpoint_config,checkpoint_paths
    from benchmark_checkpoint import execute as benchmark
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);local_model(root/'model')
    write_json(root/'model_manifest.json',model_manifest(root/'model'))
    source=root/'inputs';source.mkdir()
    for split in f.SPLITS:
        rows=[dict(sample_id=f'{split}-{i}',event_name='synthetic',text='body '+word,title='news',label=str(i%2),
                   **{d+'_label':str(.2+.1*i) for d in f.DIMS}) for i,word in enumerate(('one','two','three','four'))]
        f.write_csv(source/(split+'.csv'),rows,list(rows[0]))
    cfg=dict(experiment=dict(dataset='synthetic',event='synthetic',variant='full',method_id='synthetic_lora'),
        runtime=dict(seed=42,save_dir=str(root/'run'),deterministic=False,debug_sync=False),
        data=dict(label_schema='pheme_rumour_v1',max_length=24,**{s+'_file':str(source/(s+'.csv')) for s in f.SPLITS}),
        model=dict(base_model=str(root/'model'),manifest=str(root/'model_manifest.json'),use_lora=True,lora_r=2,
            lora_alpha=4,lora_dropout=0.,quantization='none',use_fp16=False,gradient_checkpointing=True,comm_heads=2,comm_rounds=2),
        hardware=dict(device_map={**{f'agent_{d}':'cpu' for d in f.DIMS},'communication':'cpu'}),
        training=dict(batch_size=2,grad_accum_steps=1,epochs=1,warmup_epochs=0,collab_epochs=1,learning_rate=.01,
            loss_weights=dict(final=1.,cons=.3,pred=.2,div=.01,gate=.1,**{d:.4 for d in f.DIMS})))
    write_json(root/'config.json',cfg)
    cfg,run=prepare_run(cfg,root/'config.json');Trainer(cfg).train_loop()
    pred=Predictor(checkpoint_config(run/'best'),checkpoint_paths(run/'best'))
    rows=f.read_csv(run/'data/test.csv');outputs=pred.predict_batch(rows)
    exported=f.read_csv(run/'test_predictions.csv');max_error=0.
    for row,one,batch in zip(exported,[pred.predict(r) for r in rows],outputs):
        for key in ('p_positive',*[s+'_'+d for s in ('pre','post') for d in f.DIMS]):
            max_error=max(max_error,abs(float(row[key])-batch[key]),abs(one[key]-batch[key]))
    assert max_error<1e-6,max_error
    cost=benchmark(run/'best',root/'benchmark',batch_size=2,warmup=1,repeats=2,limit=4)
    report=analyze(run,root/'analysis',benchmarks=[root/'benchmark/benchmark.json'])
    assert report['expertise_status']=='available'
    from summarize_costs import summarize as summarize_cost
    comparison=summarize_cost([root/'benchmark/benchmark.json'],root/'cost_table')
    assert len(comparison['conditions'])==1
    cross=root/'cross';cross.mkdir();fixture(cross/'parent');build(cross/'parent',cross/'frozen')
    cross_cfg=copy.deepcopy(cfg);event=f.EVENTS[0]
    cross_cfg['experiment'].update(dataset='synthetic_cross',event=event,variant='no_dim',
        evaluation_scope=CROSS_EVENT,held_out_event=event,training_events=list(f.EVENTS[1:]))
    cross_cfg['runtime'].pop('run_id',None);cross_cfg['runtime']['save_dir']=str(cross/'run')
    cross_cfg['data']['freeze_manifest']=str(cross/'frozen/freeze_manifest.json')
    for split in f.SPLITS:cross_cfg['data'][split+'_file']=str(cross/'frozen/folds'/event/(split+'.csv'))
    cross_cfg,cross_run=prepare_run(cross_cfg,root/'config.json');Trainer(cross_cfg).train_loop()
    cross_report=analyze(cross_run,cross/'analysis',reference_freeze=cross/'frozen')
    assert cross_report['evaluation_scope']==CROSS_EVENT
    cross_val=f.read_csv(cross_run/'val_predictions.csv')
    assert {r['event'] for r in cross_val}==set(f.EVENTS[1:])
    evidence=dict(status='passed',synthetic_only=True,batch_single_export_max_error=max_error,
                  expertise_status=report['expertise_status'],benchmark_repeats=cost['repeats'],
                  cross_validation_events=4,cross_test_rows=cross_report['metrics']['sample_pooled']['n'])
    write_json(root/'evidence.json',evidence)
    return evidence


def cross_fixture(root, events=None):
    fixture(root/'parent')
    build(root/'parent',root/'frozen')
    cfg=dict(model={},data={'label_schema':'pheme_rumour_v1'},runtime={},experiment={'method_id':'aegis_neural_v1'},
             hardware={},training={})
    matrix=prepare(cfg,root/'frozen',root/'plan',model_path='unused',model_manifest='unused',
                   events=events,include_baselines=True)
    return matrix


class CrossEventTests(unittest.TestCase):
    def test_membership_excludes_global_groups_and_never_uses_heldout_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=fixture(Path(tmp)/'parent');event=f.EVENTS[0]
            # One source row shares a duplicate group with the held-out event.
            foreign=next(r for r in rows if r['event_name']==f.EVENTS[1] and r['split']=='train')
            foreign['group_id']=rows[0]['group_id']
            parts,excluded=fold_rows(rows,event)
            self.assertEqual(len(parts['test']),6);self.assertEqual(len(excluded),1)
            self.assertNotIn(foreign['sample_id'],{r['sample_id'] for r in parts['train']})
            self.assertTrue(all('emotion_label' not in r and 'emotion_rationale' not in r for r in parts['val']+parts['test']))
            altered=copy.deepcopy(rows)
            for r in altered:
                if r['event_name']==event:r['label']=str(1-int(r['label']));r['emotion_label']='.99'
            other,_=fold_rows(altered,event)
            self.assertEqual(parts['train'],other['train']);self.assertEqual(parts['val'],other['val'])

    def test_freeze_rebuild_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root/'parent')
            before=sha256(root/'parent/freeze_manifest.json')
            build(root/'parent',root/'a');build(root/'parent',root/'b')
            self.assertEqual(sha256(root/'a/freeze_manifest.json'),sha256(root/'b/freeze_manifest.json'))
            self.assertEqual(before,sha256(root/'parent/freeze_manifest.json'))
            self.assertEqual(verify(root/'a')['folds'],5)
            with self.assertRaises(FileExistsError):build(root/'parent',root/'a')
            path=root/'a/folds'/f.EVENTS[0]/'train.csv';path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'hash'):verify(root/'a')

    def test_cross_text_pipeline_summary_analysis_and_event_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);matrix=cross_fixture(root)
            ledger=dict(matrix_sha256=matrix['matrix_sha256'],runs={})
            with patch.dict(os.environ,{'AEGIS_CROSS_DATA_ROOT':str(root/'frozen'),'AEGIS_RUNS_ROOT':str(root/'runs')}):
                for i,item in enumerate(r for r in matrix['runs'] if r['variant']=='tfidf_nb'):
                    path=root/'plan'/item['config'];cfg=json.loads(path.read_text());run=root/f'run{i}'
                    execute(cfg,path,run)
                    result=verify_run(run,item,matrix,root/'plan/matrix.json')
                    self.assertEqual(len(result['val']['per_event']),4)
                    self.assertEqual(result['test']['sample_pooled']['n'],6)
                    ledger['runs'][run_key(item)]=[str(run)]
                pipeline=root/'pipeline';pipeline.mkdir();write_json(pipeline/'ledger.json',ledger)
                result=summarize(root/'plan/matrix.json',pipeline,root/'summary')
                val=[r for r in result['dispersion'] if r['variant']=='tfidf_nb' and r['split']=='val']
                self.assertEqual({r['aggregation'] for r in val},{'fold_equal_mean'})
                test=next(r for r in result['by_seed'] if r['variant']=='tfidf_nb' and r['split']=='test')
                self.assertEqual(test['sample_pooled']['n'],30)
                out=analyze(root/'run0',root/'analysis',reference_freeze=root/'frozen')
                self.assertEqual(out['expertise_status'],'unavailable')
                self.assertFalse(out['human_validation_claim'])
                self.assertTrue((root/'analysis/cases.csv').exists())
                with self.assertRaises(FileExistsError):analyze(root/'run0',root/'analysis')
                path=root/'run0/test_predictions.csv';path.write_text(path.read_text()+'\n')
                with self.assertRaisesRegex(ValueError,'hash'):analyze(root/'run0',root/'tampered')

    def test_multi_event_prediction_rejects_holdout_in_validation_and_mixed_scopes(self):
        from utils.predictions import prediction_rows
        from utils.label_schema import get_label_schema
        from evaluate import evaluate_files
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);matrix=cross_fixture(root,events=[f.EVENTS[0]])
            cfg=json.loads((root/'plan'/matrix['runs'][0]['config']).read_text());cfg['runtime']['run_id']='test'
            identity=dict(sample_id='id',event=f.EVENTS[0],identity_source='fixture')
            with self.assertRaisesRegex(ValueError,'event'):
                prediction_rows([identity],[1],[.5],get_label_schema('pheme_rumour_v1'),cfg,'val','test')
            a=prediction_rows([identity],[1],[.5],get_label_schema('pheme_rumour_v1'),cfg,'test','test')
            b=copy.deepcopy(a);b[0].update(sample_id='other',evaluation_scope='in_event_v1')
            for name,rows in (('a',a),('b',b)):f.write_csv(root/(name+'.csv'),rows,list(rows[0]))
            with self.assertRaisesRegex(ValueError,'scope'):evaluate_files([root/'a.csv',root/'b.csv'])

    def test_acceptance_subset_covers_all_source_events(self):
        from server_acceptance import acceptance_subset
        with tempfile.TemporaryDirectory() as tmp:
            rows=fixture(Path(tmp)/'parent');parts,_=fold_rows(rows,f.EVENTS[0])
            for split in ('train','val'):
                selected=acceptance_subset(parts[split],4,cross=True)
                self.assertEqual(len(selected),8)
                self.assertEqual({r['event_name'] for r in selected},set(f.EVENTS[1:]))


def analysis_rows():
    refs={};preds=[]
    for i in range(8):
        sid=str(i);target=i%2;p=.9 if i%3 else .1
        refs[sid]=dict(sample_id=sid,text='short synthetic text',label=str(target),split='test',
                       **{d+'_label':str(i/10) for d in f.DIMS})
        preds.append(dict(sample_id=sid,event='synthetic',split='test',target=str(target),
            predicted_target=str(int(p>=.5)),p_positive=str(p),checkpoint_id='synthetic',label_schema='pheme_rumour_v1',
            **{stage+'_'+d:str(i/10 if d=='emotion' else .2) for stage in ('pre','post') for d in f.DIMS}))
    train=[dict(split='train',**{d+'_label':'.1' for d in f.DIMS})]
    return preds,refs,train


class AnalysisTests(unittest.TestCase):
    def test_half_dimension_heads_align_with_float_communication_and_team(self):
        from utils.diagnostics import dimension_outputs
        class HalfHead(torch.nn.Module):
            def __init__(self):
                super().__init__();self.classifier=torch.nn.Linear(4,1).half()
            def head_on_embed(self,values):return self.classifier(values.half())
        agents=[HalfHead() for _ in range(4)]
        states=torch.randn(2,4,4);logits=[torch.randn(2,1).half() for _ in range(4)]
        team=torch.nn.Linear(4,1)
        result=dimension_outputs(agents,logits,states,torch.nn.Identity(),team)
        self.assertEqual(len(result),2)
        self.assertTrue(all(0<=r['team_p_positive']<=1 for r in result))

    def test_real_local_lora_batch_export_analysis_and_cost_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=neural_analysis_fixture(Path(tmp)/'acceptance')
            self.assertLess(result['batch_single_export_max_error'],1e-6)

    def test_comparison_corrections_and_population_rejection(self):
        from utils.post_analysis import compare_predictions
        rows,_,_=analysis_rows();previous=copy.deepcopy(rows)
        previous[0]['predicted_target']='1'
        previous[1]['predicted_target']='0'
        current,result=compare_predictions(rows,previous)
        self.assertEqual(result['counts']['corrected'],2)
        self.assertEqual(current[0]['comparison_change'],'corrected')
        with self.assertRaisesRegex(ValueError,'population'):compare_predictions(rows,previous[:-1])

    def test_consistency_constant_null_correlations_and_train_only_baseline(self):
        preds,refs,train=analysis_rows()
        refs['0']['commonsense_label']=''
        result=expertise(preds,refs,train)
        emotion=result['dimensions']['post']['emotion'];bias=result['dimensions']['post']['bias']
        self.assertEqual(emotion['mae'],0);self.assertAlmostEqual(emotion['spearman'],1)
        self.assertEqual(emotion['train_mean_constant'],.1)
        self.assertTrue(bias['constant_output']);self.assertIsNone(bias['pearson'])
        self.assertEqual(result['dimensions']['post']['commonsense']['missing_teacher'],1)
        self.assertEqual(ranks([2,1,1]),[3,1.5,1.5]);self.assertIsNone(correlation([1],[2]))
        preds[0]['post_bias']='NaN'
        with self.assertRaises(ValueError):expertise(preds,refs,train)

    def test_partial_dimensions_fail_instead_of_silently_filling(self):
        preds,refs,train=analysis_rows();del preds[0]['pre_bias']
        with self.assertRaisesRegex(ValueError,'Partial'):expertise(preds,refs,train)
        for row in preds:
            for key in list(row):
                if key.startswith(('pre_','post_')):del row[key]
        self.assertEqual(expertise(preds,refs,train)['status'],'unavailable')

    def test_case_selection_balances_success_and_failure_and_is_order_invariant(self):
        preds,refs,_=analysis_rows()
        selected,policy=cases(preds,refs,per_stratum=2)
        other,_=cases(list(reversed(preds)),refs,per_stratum=2)
        self.assertEqual(selected,other)
        self.assertEqual({r['outcome'] for r in selected},{'success','failure'})
        self.assertTrue(all(r['review_sarcasm']=='' for r in selected))
        self.assertGreater(policy['eligible_counts']['all/failure'],0)

    def test_training_exports_dimensions_stability_and_gradients_without_changing_probabilities(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';trainer=fixture_trainer(root,n=5,epochs=1)
            trainer.train_loop()
            rows=f.read_csv(root/'test_predictions.csv')
            self.assertTrue(all('post_commonsense' in r and 'pre_emotion' in r for r in rows))
            micro=records(root/'stability.jsonl');updates=records(root/'optimizer_steps.jsonl')
            report=stability_report(micro,updates)
            self.assertEqual({r['batch_size'] for r in micro},{1,2})
            self.assertTrue(report['available'])
            self.assertGreater(json.loads((root/'resource_profile.json').read_text())['parameters']['trainable_parameter_elements'],0)
            single=fixture_trainer(Path(tmp)/'single',n=4,epochs=1,variant='single_agent');single.train_loop()
            self.assertNotIn('post_emotion',f.read_csv(Path(tmp)/'single/test_predictions.csv')[0])

    def test_benchmark_warmup_repetition_scope_and_default_no_load(self):
        from benchmark_checkpoint import measure
        from torch import nn
        class Fake:
            def __init__(self):
                self.ae=nn.Linear(1,1);self.ax=self.ab=self.ac=self.comm=self.final_head=self.team_layer=None
                self.calls=[]
            def predict_batch(self,rows,diagnostics):self.calls.append((len(rows),diagnostics))
        model=Fake();clock=iter(range(100))
        result=measure(model,[{}, {}, {}],batch_size=2,warmup=1,repeats=3,clock=lambda:next(clock))
        self.assertEqual(len(model.calls),8);self.assertTrue(all(not d for _,d in model.calls))
        self.assertEqual(result['pass_seconds'],[2,2,2]);self.assertEqual(result['cuda_peak_memory'],{})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            child=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'benchmark_checkpoint.py'),
                '--checkpoint-dir',str(root/'absent'),'--output',str(root/'out')],capture_output=True,text=True)
            self.assertEqual(child.returncode,0,child.stderr);self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
