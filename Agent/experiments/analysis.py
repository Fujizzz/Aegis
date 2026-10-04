"""Offline recomputation, explicit coverage and paired net gain; no model imports."""
from collections import Counter
import csv
import json
from pathlib import Path

from .protocol import digest, write_json
from .transport import read_journal

METHODS = ('first_only', 'full_two_layer', 'direct_llm')


def divide(a, b):
    return a / b if b else None


def metrics(rows, method):
    valid = [(r['target'], r['stages'][method]['p_positive']) for r in rows
             if r['stages'][method]['status'] == 'success']
    tp = sum(y == 1 and p >= .5 for y, p in valid)
    fp = sum(y == 0 and p >= .5 for y, p in valid)
    tn = sum(y == 0 and p < .5 for y, p in valid)
    fn = sum(y == 1 and p < .5 for y, p in valid)
    positive_f1 = (divide(2*tp, 2*tp+fp+fn) or 0.) if valid else None
    negative_f1 = (divide(2*tn, 2*tn+fp+fn) or 0.) if valid else None
    # Mann-Whitney rank sum, tied probabilities get average ranks.
    ordered = sorted(valid, key=lambda pair: pair[1])
    rank_sum, i = 0., 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        rank_sum += ((i+1+j)/2) * sum(y for y, _ in ordered[i:j])
        i = j
    positives, negatives = tp+fn, tn+fp
    auc = divide(rank_sum - positives*(positives+1)/2, positives*negatives)
    return dict(total=len(rows), scored=len(valid), abstentions=len(rows)-len(valid),
        coverage=divide(len(valid), len(rows)), accuracy=divide(tp+tn, len(valid)),
        correct_over_all=divide(tp+tn, len(rows)), tp=tp, fp=fp, tn=tn, fn=fn,
        positive_precision=(divide(tp, tp+fp) or 0.) if valid else None,
        positive_recall=(divide(tp, tp+fn) or 0.) if valid else None, binary_f1=positive_f1,
        class_macro_f1=(positive_f1+negative_f1)/2 if valid else None,
        roc_auc=auc, auc_defined=auc is not None)


def gain(rows):
    corrected = corrupted = paired = before_correct = after_correct = lost = 0
    for row in rows:
        before, after = (row['stages'][m] for m in METHODS[:2])
        b = before['status'] == 'success' and before['predicted_target'] == row['target']
        a = after['status'] == 'success' and after['predicted_target'] == row['target']
        before_correct += b
        after_correct += a
        if before['status'] == after['status'] == 'success':
            paired += 1
            corrected += not b and a
            corrupted += b and not a
        if before['status'] == 'success' and after['status'] != 'success':
            lost += 1
    return dict(paired_scored=paired, corrected=corrected, corrupted=corrupted,
        net_gain_count=corrected-corrupted, net_gain_over_paired=divide(corrected-corrupted, paired),
        net_gain_over_all=divide(corrected-corrupted, len(rows)), lost_to_abstention=lost,
        correct_count_delta=after_correct-before_correct,
        correct_over_all_delta=divide(after_correct-before_correct, len(rows)))


def cost(records, stages):
    starts = [r for r in records if r['kind'] == 'start' and r['stage'] in stages]
    finishes = [r for r in records if r['kind'] == 'finish' and r['stage'] in stages]
    final = {r['request_id']: r for r in finishes}
    finished_ids = {r['attempt_id'] for r in finishes}
    by_attempt = {r['attempt_id']: r for r in finishes}
    unknown = sum(r['attempt_id'] not in finished_ids for r in starts)
    ids = {r['request_id'] for r in starts}
    token_keys = ('prompt_tokens', 'completion_tokens', 'total_tokens')
    observed, missing = dict.fromkeys(token_keys, 0), dict.fromkeys(token_keys, 0)
    def response_object(record):
        value = (record or {}).get('response')
        return value if isinstance(value, dict) else {}
    for start in starts:
        record = by_attempt.get(start['attempt_id'])
        usage = response_object(record).get('usage')
        usage = usage if isinstance(usage, dict) else {}
        for key in token_keys:
            value = usage.get(key)
            if type(value) is int and value >= 0:
                observed[key] += value
            else:
                missing[key] += 1
    failures = sum(r['status'] == 'error' for r in finishes)
    models = Counter()
    for record in finishes:
        if record.get('response') is not None:
            model = response_object(record).get('model')
            models[model if isinstance(model, str) and model else 'unreported'] += 1
    return dict(call_attempts=len(starts), logical_requests=len(ids), retry_attempts=len(starts)-len(ids),
        successful_attempts=sum(r['status'] == 'success' for r in finishes),
        failed_attempts=failures, unknown_attempts=unknown,
        failure_types=dict(Counter(r['error_type'] for r in finishes if r['status'] == 'error')),
        attempt_failure_rate=divide(failures, len(finishes)),
        terminal_failed_requests=sum(r['status'] == 'error' for r in final.values()),
        request_failure_rate=divide(sum(r['status'] == 'error' for r in final.values()), len(ids)),
        observed_tokens=observed, token_usage_missing_attempts=missing,
        token_totals={key: None if missing[key] else observed[key] for key in token_keys},
        observed_duration_seconds=sum(r['duration_seconds'] for r in finishes),
        mean_completed_attempt_seconds=divide(sum(r['duration_seconds'] for r in finishes), len(finishes)),
        returned_models=dict(models))


def analyze(output):
    output = Path(output)
    plan = json.loads((output/'plan.json').read_text(encoding='utf-8'))
    signature = plan.pop('plan_sha256')
    if digest(plan) != signature:
        raise ValueError('Plan hash mismatch')
    plan['plan_sha256'] = signature
    execution = json.loads((output/'execution.json').read_text(encoding='utf-8'))
    if execution['plan_sha256'] != signature:
        raise ValueError('Execution/plan mismatch')
    records = read_journal(output/'requests.jsonl')
    if any(r['plan_sha256'] != signature for r in records):
        raise ValueError('Foreign journal records')
    rows = [r['result'] for r in records if r['kind'] == 'sample']
    expected = {s['sample_id']: s for s in plan['samples']}
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate completed sample')
    for row in rows:
        sample = expected[row['sample_id']]
        if any(sample[k] != row[k] for k in ('target', 'event', 'input_sha256', 'title_missing')):
            raise ValueError('Prediction/sample binding mismatch')
    groups = dict(all=rows, triggered=[r for r in rows if r['triggered'] is True],
        not_triggered=[r for r in rows if r['triggered'] is False],
        routing_failed=[r for r in rows if r['triggered'] is None],
        title_missing=[r for r in rows if r['title_missing']],
        title_present=[r for r in rows if not r['title_missing']],
        common_success=[r for r in rows if all(r['stages'][m]['status'] == 'success' for m in METHODS)])
    comparisons = {group: dict(metrics={m: metrics(subset, m) for m in METHODS}, gain=gain(subset))
                   for group, subset in groups.items()}
    by_event = {event: {m: metrics([r for r in rows if r['event'] == event], m) for m in METHODS}
                for event in sorted({r['event'] for r in rows})}
    summary = dict(format='t24_analysis_v1', mode=execution['mode'],
        evidence='synthetic_only' if execution['mode'] == 'simulate' else 'live_api',
        plan_sha256=signature, journal_sha256=records[-1]['sha256'] if records else None,
        population=len(plan['samples']), completed=len(rows), pending=len(plan['samples'])-len(rows),
        status='partial' if len(rows) < len(plan['samples']) else 'complete_with_errors' if any(
            r['stages'][m]['status'] != 'success' for r in rows for m in METHODS) else 'complete',
        triggered_count=len(groups['triggered']), routing_failed_count=len(groups['routing_failed']),
        trigger_rate_over_completed=divide(len(groups['triggered']), len(rows)),
        trigger_rate_over_routed=divide(len(groups['triggered']), len(rows)-len(groups['routing_failed'])),
        trigger_reasons=dict(Counter(r['trigger_reason'] for r in rows)),
        comparisons=comparisons, per_event=by_event,
        costs={name: cost(records, stages) for name, stages in dict(
            first_only=['screening'], review_increment=['review'],
            full_two_layer=['screening', 'review'], direct_llm=['direct'],
            physical_total=['screening', 'review', 'direct']).items()},
        real_api_call_attempts=sum(r['kind'] == 'start' for r in records) if execution['mode'] == 'live' else 0,
        cost_notes='Method costs overlap: full_two_layer includes shared first_only. Do not sum method totals. '
            'Durations are observed request wall time, excluding retry waits. Unknown usage is not zero. '
            'Model aliases do not identify immutable weights. Simulated tokens/time are not real cost.')
    write_json(output/'summary.json', summary)
    write_json(output/'predictions.json', rows)
    fields = ['sample_id', 'event', 'target', 'title_missing', 'triggered', 'trigger_reason', 'routing_label',
              'method', 'status', 'error_type', 'p_positive', plan['label_schema']['probability_field'],
              'predicted_target', 'final_label', 'class_confidence', 'request_ids']
    with (output/'stage_predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for method, stage in row['stages'].items():
                flat = dict(row, **stage, method=method)
                flat['request_ids'] = json.dumps(stage['request_ids'])
                writer.writerow({field: flat.get(field) for field in fields})
    metric_rows = [dict(group=group, method=method, **value)
                  for group, comparison in comparisons.items() for method, value in comparison['metrics'].items()]
    gain_rows = [dict(group=group, **comparison['gain']) for group, comparison in comparisons.items()]
    cost_rows = []
    for method, values in summary['costs'].items():
        flat = dict(method=method)
        for key, value in values.items():
            if isinstance(value, dict):
                flat[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            else:
                flat[key] = value
        cost_rows.append(flat)
    for name, table in [('comparison_metrics', metric_rows), ('review_gain', gain_rows), ('api_costs', cost_rows)]:
        with (output/(name+'.csv')).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return summary
