"""Teacher supervision: immutable, text-bound teacher requests, audits and reviewed releases.

Standard library only. No network operation occurs without generate --execute.
Human review is supplied separately; automated checks never count as human labels.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import urllib.request

from utils import freeze_pheme as f
from utils.pseudo_policy import HUMAN_REVIEWED, TEACHER_ONLY

DIMS = f.DIMS
READY = 'reused_identical_normalized_text'
GENERATED = 'generated_strict_json_v1'
LEVELS = (0., .2, .4, .6, .8, 1.)
ASPECTS = {
    'emotion': 'EMOTIONAL INCITEMENT (emotive, inflammatory, polarizing tone)',
    'exaggeration': 'EXAGGERATION (hyperbole, absolutist claims, clickbait framing)',
    'bias': 'BIASED NARRATIVE (one-sided, selective evidence, smearing opponents)',
    'commonsense': 'COMMON-SENSE CONFLICT (contradiction with widely known facts)',
}
SYSTEM = 'You are a careful rater. Output JSON only, one line, no extra text.'
PROMPT = '''You are an expert news rater.
Given the news item below, rate the DEGREE of **{aspect}** on a 0–1 scale.

Rubric:
- 0.0: None.
- 0.2: Slight.
- 0.4: Mild.
- 0.6: Moderate.
- 0.8: Strong.
- 1.0: Extreme.

IMPORTANT:
- Output ONE line of JSON only. No markdown, no extra text.
- The score MUST be one of: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
- The JSON must contain exactly two fields: "score" and "rationale"

Return STRICT JSON with two fields only:
{{"score": <float in [0,1]>, "rationale": "<brief reason>"}}

News Body:
{content}
'''


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def object_hash(obj):
    return f.digest(canonical(obj))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def strict_score(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    obj = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(obj, dict) or set(obj) != {'score', 'rationale'}:
        raise ValueError('Expected exactly score and rationale')
    score = obj['score']
    if type(score) not in (float, int) or not math.isfinite(score) or score not in LEVELS:
        raise ValueError('Score must be a numeric rubric level, without clipping or rounding')
    if not isinstance(obj['rationale'], str) or not obj['rationale'].strip():
        raise ValueError('A nonempty rationale is required')
    return float(score), obj['rationale']


def load_source(source, root=None):
    source = Path(source)
    check = f.verify(source, Path(root) if root else None, recompute_near=False)
    rows = f.read_csv(source / 'canonical.csv')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate canonical sample IDs')
    for row in rows:
        if f.digest(row['text']) != row['text_sha256']:
            raise ValueError('Canonical text hash mismatch')
    queue = f.read_csv(source / 'pseudo_regeneration_queue.csv')
    expected = {r['sample_id']: r for r in rows if r['pseudo_status'] not in (READY, GENERATED)}
    if len(queue) != len(expected) or {r['sample_id'] for r in queue} != set(expected):
        raise ValueError('Queue membership differs from canonical pending rows')
    for row in queue:
        if any(expected[row['sample_id']][key] != value for key, value in row.items()):
            raise ValueError('Queue identity/text differs from canonical')
    return rows, check


def make_plan(source, rows, model='deepseek-chat', repeats=1, scope='train'):
    if type(repeats) is not int or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    plan = dict(schema='teacher_requests_v1', parent_sha256=f.file_hash(Path(source) / 'freeze_manifest.json'),
                model=model, model_identity_limit='Requested/returned aliases are not immutable weight revisions.',
                temperature=0., repeats=repeats, aggregation='arithmetic_mean_without_requantization', scope=scope,
                prompt_template=PROMPT, system_prompt=SYSTEM, requests=[])
    for row in rows:
        if row['pseudo_status'] in (READY, GENERATED) or (scope == 'train' and row['split'] != 'train'):
            continue
        for dim in DIMS:
            for repeat in range(repeats):
                payload = dict(model=model, temperature=0., messages=[{'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': PROMPT.format(aspect=ASPECTS[dim], content=row['text'])}])
                item = dict(sample_id=row['sample_id'], event=row['event_name'], split=row['split'],
                            text_sha256=row['text_sha256'], dimension=dim, repeat=repeat,
                            parent_sha256=plan['parent_sha256'], payload=payload)
                item['request_id'] = object_hash(item)
                plan['requests'].append(item)
    plan['plan_sha256'] = object_hash(plan)
    return plan


def validate_plan(plan):
    body = {k: v for k, v in plan.items() if k != 'plan_sha256'}
    if object_hash(body) != plan['plan_sha256']:
        raise ValueError('Request plan hash mismatch')
    seen = set()
    for item in plan['requests']:
        rid = item['request_id']
        if rid in seen or object_hash({k: v for k, v in item.items() if k != 'request_id'}) != rid:
            raise ValueError('Duplicate or changed request')
        seen.add(rid)


def response_score(record):
    envelope = record['response']
    choice = envelope['choices'][0]
    if choice.get('finish_reason') != 'stop' or not envelope.get('model') or not envelope.get('id'):
        raise ValueError('Incomplete completion or missing provider identity')
    return strict_score(choice['message']['content'])


def load_responses(path, plan):
    validate_plan(plan)
    if not path or not Path(path).exists():
        return {}, []
    by_id = {r['request_id']: r for r in plan['requests']}
    accepted, records = {}, []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        record = json.loads(line)  # Truncated/tampered logs fail; never silently skip them.
        rid = record['request_id']
        if record.get('plan_sha256') != plan['plan_sha256'] or rid not in by_id:
            raise ValueError('Response is bound to another request plan')
        if object_hash({k: v for k, v in record.items() if k != 'record_sha256'}) != record['record_sha256']:
            raise ValueError('Response record hash mismatch')
        if record['status'] == 'ok':
            score, rationale = response_score(record)
            if rid in accepted:
                raise ValueError('Duplicate successful response; cannot choose favorable outputs')
            accepted[rid] = (score, rationale, record)
        elif record['status'] not in ('api_error', 'invalid_response'):
            raise ValueError('Unknown response status')
        records.append(record)
    return accepted, records


def http_transport(payload, endpoint, api_key):
    if not endpoint.startswith('https://') or '@' in endpoint or '?' in endpoint or '#' in endpoint:
        raise ValueError('Use an HTTPS endpoint without credentials/query/fragment')
    request = urllib.request.Request(endpoint, data=canonical(payload).encode('utf-8'),
        headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'})
    # Refuse redirects: never forward credentials to an unexpected host.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=90) as response:
        return json.loads(response.read())


def generate(plan, output, endpoint, *, execute=False, api_key_env='DEEPSEEK_API_KEY', limit=None, transport=None,
             workers=1):
    validate_plan(plan)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('workers must be an integer between 1 and 8')
    accepted, _ = load_responses(output, plan)
    pending = [r for r in plan['requests'] if r['request_id'] not in accepted]
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        pending = pending[:limit]
    if not execute:
        return dict(status='dry_run', planned_calls=len(pending), completed_calls=len(accepted), network_called=False)
    key = os.environ.get(api_key_env, '')
    if not key:
        raise ValueError('Required API key environment variable is missing')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    call = transport or http_transport
    counts = Counter()
    def request_record(item):
        record = dict(request_id=item['request_id'], plan_sha256=plan['plan_sha256'], endpoint=endpoint,
                      timestamp_utc=datetime.now(timezone.utc).isoformat())
        try:
            record['response'] = call(item['payload'], endpoint, key)
            record['status'] = 'invalid_response'
            response_score(record)
            record['status'] = 'ok'
        except Exception as exc:
            record.setdefault('status', 'api_error')
            record['error_type'] = type(exc).__name__  # Never serialize exception text or credentials.
        record['record_sha256'] = object_hash(record)
        return record

    # One process owns the log. Bounded workers only perform HTTP calls; the
    # coordinator commits every result before submitting replacement work.
    # On API error, stop dispatching and persist the already in-flight calls.
    with Path(output).open('a', encoding='utf-8') as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        items = iter(pending)
        inflight = {pool.submit(request_record, item) for item in [next(items, None) for _ in range(workers)]
                    if item is not None}
        stopped = False
        while inflight:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            records = [future.result() for future in done]
            stopped = stopped or any(r['status'] == 'api_error' for r in records)
            for record in records:
                stream.write(canonical(record) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                counts[record['status']] += 1
            if not stopped:
                for _ in done:
                    item = next(items, None)
                    if item is not None:
                        inflight.add(pool.submit(request_record, item))
    completed = len(accepted) + counts['ok']
    return dict(status='complete' if completed == len(plan['requests']) else 'partial',
                completed_calls=completed, remaining_calls=len(plan['requests']) - completed,
                counts=dict(counts), network_called=True)


def apply_responses(rows, plan, accepted):
    result = copy.deepcopy(rows)
    requests = defaultdict(list)
    for request in plan['requests']:
        requests[request['sample_id']].append(request)
    for row in result:
        items = requests.get(row['sample_id'], [])
        if not items:
            continue
        if any(r['text_sha256'] != row['text_sha256'] or r['split'] != row['split'] or
               r['event'] != row['event_name'] for r in items):
            raise ValueError('Request identity does not match data')
        if not all(r['request_id'] in accepted for r in items):
            continue  # Partial four-vectors never enter a dataset.
        for dim in DIMS:
            values = [accepted[r['request_id']] for r in items if r['dimension'] == dim]
            if len(values) != plan['repeats']:
                raise ValueError('Incomplete dimension replicates')
            score = statistics.mean(v[0] for v in values)
            row[dim + '_label'] = repr(score)
            row[dim + '_score_raw'] = repr(score)
            row[dim + '_rationale'] = canonical([v[1] for v in values])
            row[dim + '_raw'] = canonical([v[2]['response']['choices'][0]['message']['content'] for v in values])
            row[dim + '_parse_ok'] = '1'
            row[dim + '_parse_method'] = 'strict_json_all_replicates'
        row['pseudo_source_records'] = ' '.join(accepted[r['request_id']][2]['record_sha256'] for r in items)
        row['pseudo_status'] = GENERATED
    return result


def audit_rows(rows):
    distributions, flags = [], []
    for event in sorted({r['event_name'] for r in rows}) + ['ALL']:
        for split in ('train', 'val', 'test', 'ALL'):
            part = [r for r in rows if (event == 'ALL' or r['event_name'] == event) and
                    (split == 'ALL' or r['split'] == split) and r['pseudo_status'] in (READY, GENERATED)]
            for dim in DIMS:
                vals = [float(r[dim + '_label']) for r in part]
                hist = Counter(vals)
                distributions.append(dict(event=event, split=split, dimension=dim, n=len(vals),
                    mean=statistics.mean(vals) if vals else None,
                    std=statistics.pstdev(vals) if vals else None,
                    zero_fraction=hist[0.] / len(vals) if vals else None,
                    constant=len(hist) == 1, histogram=dict(sorted((str(k), v) for k, v in hist.items()))))
    for row in rows:
        if row['pseudo_status'] != READY:
            continue
        for dim in DIMS:
            reasons = []
            try:
                score, rationale = strict_score(row[dim + '_raw'])
                if score != float(row[dim + '_label']):
                    reasons.append('last_response_differs_from_aggregate')
                if rationale != row[dim + '_rationale']:
                    reasons.append('last_rationale_differs_from_saved_rationale')
                if row[dim + '_parse_ok'] != '1' or row[dim + '_parse_method'] != 'strict_json':
                    reasons.append('legacy_parse_metadata_invalid')
            except (ValueError, KeyError, TypeError):
                reasons.append('raw_response_not_strict_rubric_json')
            if reasons:
                flags.append(dict(sample_id=row['sample_id'], dimension=dim, reasons=';'.join(reasons)))
    return distributions, flags


def redact(text):
    text = re.sub(r'https?://\S+', '[URL]', text)
    return re.sub(r'(?<!\w)@\w+', '[USER]', text)


def select_review(rows, flags, seed=20261003):
    ready = [r for r in rows if r['split'] == 'train' and r['pseudo_status'] in (READY, GENERATED)]
    chosen = defaultdict(set)
    def take(pool, n, reason):
        for row in sorted(pool, key=lambda r: f.digest(f'{seed}:{reason}:{r["sample_id"]}'))[:n]:
            chosen[row['sample_id']].add(reason)
    for event in sorted({r['event_name'] for r in ready}):
        for label in ('0', '1'):
            take([r for r in ready if r['event_name'] == event and r['label'] == label and
                  r['pseudo_status'] == READY], 10, 'representative_legacy')
            take([r for r in ready if r['event_name'] == event and r['label'] == label and
                  r['pseudo_status'] == GENERATED], 5, 'representative_generated')
    risky = {r['sample_id'] for r in flags if 'last_response_differs' in r['reasons']}
    take([r for r in ready if r['sample_id'] in risky], 20, 'aggregation_trace_risk')
    take([r for r in ready if float(r['commonsense_label']) == 0], 20, 'commonsense_zero_risk')
    for dim in DIMS:
        take([r for r in ready if float(r[dim + '_label']) >= .8], 10, dim + '_high_risk')
    by_id = {r['sample_id']: r for r in ready}
    key, blind = [], []
    for sid in sorted(chosen):
        row = by_id[sid]
        review_id = f.digest(f'Teacher supervision:{seed}:{sid}')[:24]
        for dim in DIMS:
            visible = redact(row['text'])
            item = dict(review_id=review_id, dimension=dim, sample_id=sid, text_sha256=row['text_sha256'],
                        display_sha256=f.digest(visible), teacher_score=row[dim + '_label'],
                        sampling_reasons=';'.join(sorted(chosen[sid])))
            key.append(item)
            for annotator in ('A', 'B'):
                blind.append(dict(review_id=review_id, dimension=dim, annotator_slot=annotator,
                                  text=visible, display_sha256=item['display_sha256'], score='',
                                  reviewer_id='', rationale='', critical_issue='', adjudicated_score='',
                                  adjudicator_id='', adjudication_rationale=''))
    return key, blind


def seal(directory, name='audit_manifest.json'):
    paths = sorted(p for p in Path(directory).rglob('*') if p.is_file() and p.name != name)
    write_json(Path(directory) / name, {'files': {p.relative_to(directory).as_posix(): f.file_hash(p) for p in paths}})


def verify_audit(directory):
    directory = Path(directory)
    for name, expected in read_json(directory / 'audit_manifest.json')['files'].items():
        if f.file_hash(directory / name) != expected:
            raise ValueError('Audit artifact changed: ' + name)


def create_audit(source, output, *, root=None, plan_path=None, responses=None, scope='train', repeats=1):
    if Path(source).resolve() == Path(output).resolve() or Path(source).resolve() in Path(output).resolve().parents:
        raise ValueError('Audit output cannot be inside its immutable source')
    rows, check = load_source(source, root)
    plan = read_json(plan_path) if plan_path else make_plan(source, rows, repeats=repeats, scope=scope)
    expected = make_plan(source, rows, model=plan['model'], repeats=plan['repeats'], scope=plan['scope'])
    if plan != expected:
        raise ValueError('Plan differs from canonical request definition')
    accepted, records = load_responses(responses, plan)
    merged = apply_responses(rows, plan, accepted)
    distributions, flags = audit_rows(merged)
    key, blind = select_review(merged, flags)
    mappings = f.read_csv(Path(source) / 'legacy_mapping.csv')
    candidates = defaultdict(list)
    for item in mappings:
        if item['source_kind'] == 'pseudo':
            for sid in item['candidate_ids'].split():
                candidates['pheme:' + sid].append(item)
    queue = []
    for row in rows:
        if row['pseudo_status'] in (READY, GENERATED):
            continue
        linked = candidates[row['sample_id']]
        methods = sorted({m['method'] for m in linked})
        reason = ('ambiguous_legacy_identity' if any(m['status'].startswith('ambiguous') for m in linked)
                  else 'whitespace_repaired_text' if 'whitespace_repaired' in methods
                  else 'no_unique_legacy_pseudo_match')
        queue.append({k: row[k] for k in ('sample_id', 'event_name', 'split', 'text', 'text_sha256',
                                          'source_member', 'source_json_sha256', 'group_id')} |
                     dict(reason=reason, candidate_legacy_records=' '.join(m['legacy_record_id'] for m in linked)))
    summary = dict(source_check=check, total_rows=len(rows), pending_total=len(queue),
        pending_train=sum(r['split'] == 'train' for r in queue),
        pending_by_event_split=dict(Counter(r['event_name'] + '/' + r['split'] for r in queue)),
        training_reasons=dict(Counter(r['reason'] for r in queue if r['split'] == 'train')),
        merged_training_pending=sum(r['split'] == 'train' and r['pseudo_status'] not in (READY, GENERATED) for r in merged),
        response_records=len(records), accepted_calls=len(accepted), plan_sha256=plan['plan_sha256'],
        response_log_sha256=f.file_hash(responses) if responses else None,
        review_samples=len(key) // 4, human_review_status='pending',
        flag_counts=dict(Counter(reason for item in flags for reason in item['reasons'].split(';'))),
        limits=['Format validity is not label quality.', 'Last historical response cannot reconstruct all replicate scores.',
                'Teacher aliases lack historical immutable model revisions.',
                'Commonsense labels are not independent fact verification.',
                'Dimension-head agreement/collapse requires future trained predictions.'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', summary)
    shutil.copyfile(Path(__file__), output / 'pseudo_quality_source.py')
    write_json(output / 'request_plan.json', plan)
    write_json(output / 'distributions.json', distributions)
    f.write_csv(output / 'pending_inventory.csv', queue, list(queue[0]) if queue else ['sample_id'])
    f.write_csv(output / 'legacy_flags.csv', flags, ['sample_id', 'dimension', 'reasons'])
    f.write_csv(output / 'review_key.csv', key, list(key[0]))
    f.write_csv(output / 'human_review_blind.csv', blind, list(blind[0]))
    shutil.copyfile(Path(source) / 'mixed_label_group_review.csv', output / 'mixed_label_groups.csv')
    if responses:
        shutil.copyfile(responses, output / 'responses.jsonl')
    seal(output)
    return summary


def review_metrics(audit, annotations):
    verify_audit(audit)
    key = f.read_csv(Path(audit) / 'review_key.csv')
    entries = f.read_csv(annotations)
    indexed = {}
    for entry in entries:
        ident = (entry['review_id'], entry['dimension'], entry['annotator_slot'])
        if ident in indexed:
            raise ValueError('Duplicate human review')
        indexed[ident] = entry
    expected = {(k['review_id'], k['dimension'], slot) for k in key for slot in ('A', 'B')}
    if set(indexed) != expected:
        raise ValueError('Human review coverage differs from the blind sample')
    scores = defaultdict(list)
    critical, disagreements = 0, 0
    for item in key:
        pair = [indexed[(item['review_id'], item['dimension'], slot)] for slot in ('A', 'B')]
        values = []
        for entry in pair:
            if (entry['display_sha256'] != item['display_sha256'] or f.digest(entry['text']) != item['display_sha256']
                    or not entry['reviewer_id'].strip() or not entry['rationale'].strip()
                    or entry['critical_issue'] not in ('0', '1')):
                raise ValueError('Incomplete or text-mismatched human annotation')
            score = float(entry['score'])
            if score not in LEVELS:
                raise ValueError('Human score must use the six-level rubric')
            values.append(score)
            critical += int(entry['critical_issue'])
        if pair[0]['reviewer_id'].strip() == pair[1]['reviewer_id'].strip():
            raise ValueError('Two distinct reviewers are required')
        if abs(values[0] - values[1]) > .20000001:
            disagreements += 1
            adjudications = {(e['adjudicated_score'], e['adjudicator_id'], e['adjudication_rationale']) for e in pair}
            if len(adjudications) != 1:
                raise ValueError('Disagreement needs one consistent adjudication on both rows')
            value, person, rationale = adjudications.pop()
            if not person.strip() or not rationale.strip() or float(value) not in LEVELS:
                raise ValueError('Missing adjudication for a difference greater than one level')
            target = float(value)
        else:
            target = statistics.mean(values)
        for population in ('all_reviewed', 'representative_legacy', 'representative_generated'):
            if population == 'all_reviewed' or population in item['sampling_reasons'].split(';'):
                scores[(population, item['dimension'])].append((float(item['teacher_score']), target, *values))
    metrics = {}
    for (population, dim), values in scores.items():
        left = [round(v[2] * 5) for v in values]
        right = [round(v[3] * 5) for v in values]
        observed = statistics.mean((a-b)**2 for a,b in zip(left,right))
        expected = sum((a-b)**2 for a in left for b in right) / len(values)**2
        metrics[population + '/' + dim] = dict(n=len(values),
            teacher_mae=statistics.mean(abs(a-b) for a, b, _, _ in values),
            teacher_within_one_level=statistics.mean(abs(a-b) <= .20000001 for a, b, _, _ in values),
            inter_reviewer_mae=statistics.mean(abs(a-b) for _, _, a, b in values),
            inter_reviewer_quadratic_kappa=1-observed/expected if expected else None,
            kappa_defined=bool(expected))
    # Operational screening thresholds, not a claim of validated population accuracy.
    relevant = [v for k, v in metrics.items() if k.startswith('representative_')]
    passed = bool(relevant) and critical == 0 and all(v['teacher_within_one_level'] >= .8 for v in relevant)
    return dict(status='passed' if passed else 'failed', reviewer_kind='human_supplied',
        audit_sha256=f.file_hash(Path(audit) / 'audit_manifest.json'), annotations_sha256=f.file_hash(annotations),
        critical_flags=critical, adjudicated_disagreements=disagreements, metrics=metrics,
        gate='Each representative source/dimension >=80% within 0.2; no critical flags; all risk rows reviewed.',
        limitation='Stratified screen, not an unbiased population estimate or trained-head evaluation.')


def build_release(source, audit, annotations, output, *, release_policy='human_reviewed', decision_record=None):
    if release_policy not in ('human_reviewed', 'teacher_only'):
        raise ValueError('Unknown pseudo-label release policy')
    rows, _ = load_source(source)
    verify_audit(audit)
    audit = Path(audit)
    plan = read_json(audit / 'request_plan.json')
    if plan != make_plan(source, rows, model=plan['model'], repeats=plan['repeats'], scope=plan['scope']):
        raise ValueError('Audit request plan does not match source')
    response_path = audit / 'responses.jsonl'
    summary = read_json(audit / 'summary.json')
    if not response_path.exists() or f.file_hash(response_path) != summary['response_log_sha256']:
        raise ValueError('A complete audited response log is required')
    accepted, _ = load_responses(response_path, plan)
    merged = apply_responses(rows, plan, accepted)
    if any(r['split'] == 'train' and r['pseudo_status'] not in (READY, GENERATED) for r in merged):
        raise ValueError('Training pseudo labels are incomplete; refusing a new release')
    for row in merged:
        if row['split'] == 'train':
            for dim in DIMS:
                score = float(row[dim + '_label'])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('Training dimension scores must be finite and in [0, 1]')
    key, _ = select_review(merged, audit_rows(merged)[1])
    expected_key = [{k: str(v) for k, v in item.items()} for item in key]
    if f.read_csv(audit / 'review_key.csv') != expected_key:
        raise ValueError('Audit review sample does not cover this generated data')
    if release_policy == 'human_reviewed':
        if annotations is None or decision_record is not None:
            raise ValueError('Human-reviewed release requires annotations and no teacher-only decision')
        quality = review_metrics(audit, annotations)
        if quality['status'] != 'passed':
            raise ValueError('Human quality gate failed; resolve findings before release')
        quality_status = HUMAN_REVIEWED
        policy_text = 'Legacy exact-text reuse plus strict recorded teacher replicates; human screening passed.'
    else:
        if annotations is not None or decision_record is None:
            raise ValueError('Teacher-only release requires a decision record and no human annotations')
        decision = read_json(decision_record)
        if (decision.get('schema') != 'teacher_only_release_decision_v1'
                or decision.get('release_policy') != 'teacher_only'
                or not isinstance(decision.get('user_instruction'), str)
                or not decision['user_instruction'].strip()
                or decision.get('audit_sha256') != f.file_hash(audit / 'audit_manifest.json')):
            raise ValueError('Teacher-only decision must record the user instruction and match this audit')
        quality_status = TEACHER_ONLY
        quality = dict(status='accepted_teacher_only_not_human_validated', release_policy=release_policy,
            reviewer_kind='none', human_review_completed=False, human_scores_imported=False,
            automated_format_and_integrity='passed', teacher_scores_modified=False,
            audit_sha256=f.file_hash(audit / 'audit_manifest.json'),
            response_log_sha256=f.file_hash(response_path), accepted_calls=len(accepted),
            decision_record_sha256=f.file_hash(decision_record),
            limitation='Original teacher scores are supervision targets, not human gold labels. '
                       'Previously identified rationale risks remain unresolved observations.')
        policy_text = 'Legacy exact-text reuse plus original recorded teacher scores; no human scoring or validation.'
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if Path(source).resolve() in output.resolve().parents:
        raise ValueError('Output cannot be inside its immutable parent')
    shutil.copytree(source, output)
    for row in merged:
        row['pseudo_quality_status'] = quality_status if row['split'] == 'train' else 'not_used_for_supervision'
    fields = list(merged[0])
    f.write_csv(output / 'canonical.csv', merged, fields)
    for event in f.EVENTS:
        for split in f.SPLITS:
            f.write_csv(output / ('split_' + f.STEMS[event]) / (split + '.csv'),
                        [r for r in merged if r['event_name'] == event and r['split'] == split], fields)
    manifest_fields = list(f.read_csv(Path(source) / 'split_manifest.csv')[0])
    f.write_csv(output / 'split_manifest.csv', [{k: r[k] for k in manifest_fields} for r in merged], manifest_fields)
    queue_fields = ['sample_id', 'event_name', 'split', 'text', 'pseudo_status']
    f.write_csv(output / 'pseudo_regeneration_queue.csv',
        [{k: r[k] for k in queue_fields} for r in merged if r['pseudo_status'] not in (READY, GENERATED)], queue_fields)
    evidence = output / 'pseudo_provenance'
    shutil.copytree(audit, evidence)
    if annotations is not None:
        shutil.copyfile(annotations, evidence / 'human_annotations.csv')
    else:
        shutil.copyfile(decision_record, evidence / 'release_decision.json')
    shutil.copyfile(Path(__file__), evidence / 'release_builder_source.py')
    shutil.copyfile(Path(__file__).with_name('pseudo_policy.py'), evidence / 'release_policy_source.py')
    shutil.copyfile(Path(source) / 'freeze_manifest.json', evidence / 'parent_freeze_manifest.json')
    write_json(evidence / 'quality_review.json', quality)
    protocol = read_json(output / 'protocol.json')
    protocol.update(version=output.name, parent_freeze_sha256=plan['parent_sha256'],
                    pseudo_policy=policy_text, pseudo_release_policy=release_policy,
                    human_review_completed=release_policy == 'human_reviewed')
    write_json(output / 'protocol.json', protocol)
    old_summary = read_json(output / 'audit_summary.json')
    old_summary.update(pseudo_statuses=dict(Counter(r['pseudo_status'] for r in merged)),
                       training_ready_with_full_dimension_supervision=True, pseudo_quality_review=quality,
                       parent_freeze_sha256=plan['parent_sha256'], pseudo_release_policy=release_policy,
                       human_review_completed=release_policy == 'human_reviewed')
    for event, parts in old_summary.get('events', {}).items():
        for split, stats in parts.items():
            stats['pseudo_ready'] = sum(r['event_name'] == event and r['split'] == split and
                                       r['pseudo_status'] in (READY, GENERATED) for r in merged)
    write_json(output / 'audit_summary.json', old_summary)
    (output / 'README.md').write_text(f'# {output.name}\n\nParent: {plan["parent_sha256"]}\n\n'
        'The official population, labels, text and split membership are unchanged. '
        f'Pseudo-label policy: {policy_text} Provenance is recorded in pseudo_provenance/. '
        'The copied audit is a historical generation-stage snapshot; its unfilled blind-review forms '
        'do not establish human review. The current release decision is quality_review.json. '
        'Held-out dimension labels may remain pending; they are not used for training. '
        'This release does not establish trained-head specialization or reproduce historical model metrics.\n', encoding='utf-8')
    hashes = {p.relative_to(output).as_posix(): f.file_hash(p) for p in sorted(output.rglob('*'))
              if p.is_file() and p.relative_to(output).as_posix() not in ('FREEZE.sha256', 'freeze_manifest.json')}
    write_json(output / 'freeze_manifest.json', dict(version=output.name, files=hashes))
    (output / 'FREEZE.sha256').write_text(f.file_hash(output / 'freeze_manifest.json') + '  freeze_manifest.json\n', encoding='ascii')
    return f.verify(output, recompute_near=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--source', required=True, type=Path)
    audit.add_argument('--output', required=True, type=Path)
    audit.add_argument('--root', type=Path)
    audit.add_argument('--plan', type=Path)
    audit.add_argument('--responses', type=Path)
    audit.add_argument('--scope', choices=['train', 'all'], default='train')
    audit.add_argument('--repeats', type=int, default=1)
    gen = sub.add_parser('generate')
    gen.add_argument('--plan', required=True, type=Path)
    gen.add_argument('--output', required=True, type=Path)
    gen.add_argument('--endpoint', default='https://api.deepseek.com/chat/completions')
    gen.add_argument('--api-key-env', default='DEEPSEEK_API_KEY')
    gen.add_argument('--limit', type=int)
    gen.add_argument('--workers', type=int, default=1, help='Bounded concurrent requests (1-8); one log writer')
    gen.add_argument('--execute', action='store_true')
    review = sub.add_parser('review')
    review.add_argument('--audit', required=True, type=Path)
    review.add_argument('--annotations', required=True, type=Path)
    review.add_argument('--output', required=True, type=Path)
    build = sub.add_parser('build')
    for flag in ('source', 'audit', 'output'):
        build.add_argument('--' + flag, required=True, type=Path)
    build.add_argument('--annotations', type=Path)
    build.add_argument('--release-policy', choices=('human_reviewed', 'teacher_only'), default='human_reviewed')
    build.add_argument('--decision-record', type=Path)
    args = parser.parse_args()
    if args.command == 'audit':
        result = create_audit(args.source, args.output, root=args.root, plan_path=args.plan,
                              responses=args.responses, scope=args.scope, repeats=args.repeats)
    elif args.command == 'generate':
        result = generate(read_json(args.plan), args.output, args.endpoint, execute=args.execute,
                          api_key_env=args.api_key_env, limit=args.limit, workers=args.workers)
    elif args.command == 'review':
        result = review_metrics(args.audit, args.annotations)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    else:
        result = build_release(args.source, args.audit, args.annotations, args.output,
                               release_policy=args.release_policy, decision_record=args.decision_record)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == 'generate' and any(result.get('counts', {}).get(k, 0) for k in ('api_error', 'invalid_response')):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
