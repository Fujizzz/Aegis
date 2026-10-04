"""Offline teacher-consistency, stability and rule-selected case analysis.

No score is a human gold label; PHEME rumour status is not factual truth.
"""
import math
import statistics
from collections import Counter
from utils.freeze_pheme import DIMS, digest


def number(value, *, probability=False):
    value=float(value)
    if not math.isfinite(value) or (probability and not 0<=value<=1):
        raise ValueError('Non-finite or out-of-range diagnostic value')
    return value


def describe(values):
    values=[number(v) for v in values]
    if not values:return dict(n=0,mean=None,std=None,minimum=None,maximum=None)
    return dict(n=len(values),mean=statistics.mean(values),std=statistics.pstdev(values),
                minimum=min(values),maximum=max(values))


def correlation(x,y):
    if len(x)!=len(y):raise ValueError('Unaligned correlation inputs')
    if len(x)<2 or max(x)==min(x) or max(y)==min(y):return None
    mx,my=statistics.mean(x),statistics.mean(y)
    dx,dy=[v-mx for v in x],[v-my for v in y]
    return sum(a*b for a,b in zip(dx,dy))/math.sqrt(sum(a*a for a in dx)*sum(b*b for b in dy))


def ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);result=[0.]*len(values);start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[end]]==values[order[start]]:end+=1
        for i in order[start:end]:result[i]=(start+end-1)/2+1
        start=end
    return result


def expertise(predictions, references, train_rows, *, near_constant_std=.01):
    expected={f'{stage}_{d}' for stage in ('pre','post') for d in DIMS}
    present=[expected & set(row) for row in predictions]
    if not any(present):return dict(status='unavailable',reason='No four-dimension outputs; single/text/API baseline or legacy export.')
    if any(keys!=expected for keys in present):raise ValueError('Partial dimension output columns')
    for row in predictions:
        for key in expected:number(row[key],probability=True)
    reports={};cross={};pairwise={}
    for stage in ('pre','post'):
        reports[stage]={};cross[stage]={}
        for dim in DIMS:
            values=[number(r[f'{stage}_{dim}'],probability=True) for r in predictions]
            pairs=[(values[i],number(references[r['sample_id']][dim+'_label'],probability=True))
                   for i,r in enumerate(predictions) if references[r['sample_id']].get(dim+'_label','')!='']
            x,y=([a for a,b in pairs],[b for a,b in pairs])
            source=[number(r[dim+'_label'],probability=True) for r in train_rows if r.get(dim+'_label','')!='']
            constant=statistics.mean(source) if source else None
            dist=describe(values)
            reports[stage][dim]=dict(output=dist,paired_rows=len(pairs),missing_teacher=len(predictions)-len(pairs),
                teacher=describe(y),mae=statistics.mean(abs(a-b) for a,b in pairs) if pairs else None,
                rmse=math.sqrt(statistics.mean((a-b)**2 for a,b in pairs)) if pairs else None,
                pearson=correlation(x,y),spearman=correlation(ranks(x),ranks(y)),
                constant_output=max(values)==min(values),near_constant=dist['std']<near_constant_std,
                train_mean_constant=constant,
                train_mean_baseline_mae=statistics.mean(abs(constant-v) for v in y) if y and constant is not None else None,
                teacher_zero_fraction=sum(v==0 for v in y)/len(y) if y else None)
            cross[stage][dim]={}
            for other in DIMS:
                pairs2=[(values[i],number(references[r['sample_id']][other+'_label'],probability=True))
                        for i,r in enumerate(predictions) if references[r['sample_id']].get(other+'_label','')!='']
                cross[stage][dim][other]=dict(n=len(pairs2),pearson=correlation([a for a,b in pairs2],[b for a,b in pairs2]))
        pairwise[stage]={}
        for i,a in enumerate(DIMS):
            for b in DIMS[i+1:]:
                x=[number(r[f'{stage}_{a}']) for r in predictions];y=[number(r[f'{stage}_{b}']) for r in predictions]
                pairwise[stage][a+'__'+b]=dict(pearson=correlation(x,y),
                    mean_absolute_difference=statistics.mean(abs(v-w) for v,w in zip(x,y)))
    return dict(status='available',dimensions=reports,prediction_teacher_correlation=cross,
        inter_head=pairwise,near_constant_std_threshold=near_constant_std,
        interpretation='Teacher agreement and collapse diagnostics only; no human gold, causal specialization or factual verification claim. '
                       'Pre heads are diagnostic projections; training supervises post-communication heads. '
                       'Undefined correlations (constant or fewer than two values) are null, not zero.')


def cases(predictions, references, *, per_stratum=5, seed=42):
    if type(per_stratum) is not int or per_stratum<1:raise ValueError('per_stratum must be positive')
    strata={};selected={}
    for row in predictions:
        ref=references[row['sample_id']]
        p=number(row['p_positive'],probability=True);target=int(row['target']);pred=int(row['predicted_target'])
        outcome='success' if target==pred else 'failure'
        words=len(ref['text'].split());tags=['all',outcome]
        if max(p,1-p)>=.9:tags.append('high_confidence')
        if abs(p-.5)<=.1:tags.append('near_threshold')
        if words<=8:tags.append('short_text_le8_words')
        dims=[number(row['post_'+d],probability=True) for d in DIMS] if 'post_emotion' in row else None
        if dims is not None:
            if max(dims)-min(dims)>=.5:tags.append('dimension_range_ge0.5')
            if dims[0]>=.6 and target==0:tags.append('high_emotion_non_rumour_candidate')
            if max(dims)<=.2 and target==1:tags.append('low_dimension_rumour_candidate')
        team=row.get('team_p_positive','')
        if team!='' and (number(team,probability=True)>=.5)!=bool(pred):tags.append('team_final_disagreement')
        if row.get('comparison_change'):tags.append(row['comparison_change'])
        # Balanced success/failure strata, selected by identity hash, not manual cherry-picking.
        for tag in tags:
            key=tag+'/'+outcome
            strata.setdefault(key,[]).append(row)
    for key,rows in sorted(strata.items()):
        for row in sorted(rows,key=lambda r:(digest(f'{seed}:{key}:{r["sample_id"]}'),r['sample_id']))[:per_stratum]:
            selected.setdefault(row['sample_id'],dict(row=row,strata=[]))['strata'].append(key)
    output=[]
    for sid,value in sorted(selected.items()):
        row=value['row'];ref=references[sid]
        result=dict(sample_id=sid,event=row['event'],split=row['split'],target=row['target'],
            predicted_target=row['predicted_target'],p_positive=row['p_positive'],
            outcome='success' if row['target']==row['predicted_target'] else 'failure',
            selected_strata=';'.join(value['strata']),text=ref['text'],text_sha256=digest(ref['text']),
            checkpoint_id=row['checkpoint_id'],label_schema=row['label_schema'])
        for d in DIMS:
            result['post_'+d]=row.get('post_'+d,'')
            result['teacher_'+d]=ref.get(d+'_label','')
        result.update(review_sarcasm='',review_insufficient_information='',review_category='',review_note='',reviewer_id='')
        result.update(comparison_change=row.get('comparison_change',''),
                      comparison_p_positive=row.get('comparison_p_positive',''))
        output.append(result)
    return output,dict(seed=seed,per_stratum=per_stratum,eligible_counts={k:len(v) for k,v in sorted(strata.items())},
        unique_selected=len(output),selection='First identity hashes per outcome/stratum; candidates can have multiple tags.',
        caveats=['PHEME rumour/non-rumour is not a true/false annotation.',
                 'Sarcasm and information sufficiency require review; blank review fields are intentional.',
                 'Dimension range is stylistic intensity disagreement, not four votes on the main class.',
                 'Team/final comparison is neural fusion, not API second-layer correction.',
                 'No generated rationale is treated as verified evidence.'])


def compare_predictions(current, previous):
    """Pair an explicit earlier stage/baseline with current, without guessing method identity."""
    old={r['sample_id']:r for r in previous}
    if len(old)!=len(previous) or set(old)!={r['sample_id'] for r in current}:
        raise ValueError('Comparison requires exactly the same unique population')
    counts=Counter();result=[]
    for row in current:
        prior=old[row['sample_id']]
        if any(row[k]!=prior[k] for k in ('event','target','label_schema','split')):
            raise ValueError('Comparison labels/events/schema/split differ')
        before=prior['target']==prior['predicted_target'];after=row['target']==row['predicted_target']
        change={ (False,True):'corrected',(True,False):'introduced_error',
                 (True,True):'both_correct',(False,False):'both_wrong'}[(before,after)]
        counts[change]+=1
        result.append(dict(row,comparison_change=change,comparison_p_positive=prior['p_positive']))
    return result,dict(counts={k:counts[k] for k in ('corrected','introduced_error','both_correct','both_wrong')},
        net_corrections=counts['corrected']-counts['introduced_error'],
        note='Current versus explicitly supplied earlier stage/baseline on identical labels; no causal or factual-verification claim.')


def stability_report(micro, updates):
    groups={}
    for row in micro:groups.setdefault((row['epoch'],row['batch_size']),[]).append(row)
    summaries=[]
    for (epoch,batch),rows in sorted(groups.items()):
        gates=[r['gate'] for r in rows if r.get('gate') is not None]
        count=sum(g['n'] for g in gates)
        summaries.append(dict(epoch=epoch,batch_size=batch,micro_batches=len(rows),
            diversity_raw=describe(r['diversity_raw'] for r in rows if r['diversity_raw'] is not None),
            diversity_disabled_batches=sum(r['diversity_raw'] is None for r in rows),
            gate_mean=sum(g['mean']*g['n'] for g in gates)/count if count else None,
            gate_histogram=[sum(g['histogram_0_to_1'][i] for g in gates) for i in range(10)] if count else None,
            gradient_before_clip=describe(r['grad_norm_before_clip'] for r in updates if r['epoch']==epoch)))
    return dict(available=bool(micro),by_epoch_micro_batch_size=summaries,
        note='Gradients are optimizer-window values per epoch; not independent measurements for each micro-batch group. '
             'Diversity is the detached unweighted loss when active; disabled/warmup batches stay null; one-sample batches are degenerate. '
             'Small gates do not skip computation or establish a speedup.')
