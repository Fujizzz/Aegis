"""Direct P(rumour) baseline: text-only prompts, strict responses and resumable log."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from baselines.common import split_data, export
from utils import pseudo_quality as q
from utils.run_archive import sha256, write_json

PROMPT = '''Classify the following social-media news post as rumour or non-rumour.
For PHEME, rumour means information that was unverified at the time it circulated;
this is NOT simply a fake-versus-real factuality classification. Use only the visible
post; do not claim to have opened links or verified external sources. Give the
probability of the rumour class, and a brief reason that states uncertainty where needed.
Return exactly one JSON object with numeric "p_positive" in [0,1] and nonempty "rationale".
POST:
{text}'''


def strict_probability(response):
    if not response.get('id') or not response.get('model'):
        raise ValueError('Missing provider identity')
    choice = response['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Incomplete teacher completion')
    def pairs(values):
        result = {}
        for key,value in values:
            if key in result:
                raise ValueError('Duplicate response key')
            result[key] = value
        return result
    value = json.loads(choice['message']['content'], object_pairs_hook=pairs)
    if not isinstance(value, dict) or set(value) != {'p_positive', 'rationale'}:
        raise ValueError('Expected p_positive and rationale only')
    p = value['p_positive']
    if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Probability must be numeric, finite and in [0,1]')
    if not isinstance(value['rationale'], str) or not value['rationale'].strip():
        raise ValueError('Missing rationale')
    return float(p)


def make_plan(config):
    if config['data']['label_schema'] != 'pheme_rumour_v1':
        raise ValueError('This preregistered teacher prompt is PHEME-specific')
    setting = config.get('baseline', {}).get('teacher', {})
    requests = []
    for split in ('val', 'test'):
        if not config['data'].get(split + '_file'):
            continue
        rows, texts, _ = split_data(config, split)
        for row,text in zip(rows, texts):
            item = dict(sample_id=row['sample_id'], split=split, text_sha256=q.f.digest(text),
                payload=dict(model=setting.get('model', 'deepseek-chat'), temperature=0,
                    messages=[dict(role='system', content='Output strict JSON only.'),
                              dict(role='user', content=PROMPT.format(text=text))]))
            item['request_id'] = q.object_hash(item)
            requests.append(item)
    plan = dict(format='pheme_teacher_direct_v1', prompt=PROMPT, requests=requests,
                endpoint=setting.get('endpoint', 'https://api.deepseek.com/chat/completions'))
    plan['plan_sha256'] = q.object_hash(plan)
    return plan


def load_log(path, plan):
    requests = {r['request_id'] for r in plan['requests']}
    accepted, records = {}, []
    if not Path(path).exists():
        return accepted, records
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        item = json.loads(line)
        if (item['request_id'] not in requests or item['plan_sha256'] != plan['plan_sha256']
                or item['record_sha256'] != q.object_hash({k:v for k,v in item.items() if k != 'record_sha256'})):
            raise ValueError('Teacher baseline log binding/hash mismatch')
        if item['status'] == 'ok':
            if item['request_id'] in accepted:
                raise ValueError('Duplicate successful teacher response')
            accepted[item['request_id']] = strict_probability(item['response'])
        elif item['status'] not in ('api_error', 'invalid_response'):
            raise ValueError('Unknown teacher request status')
        records.append(item)
    return accepted, records


def generate(plan, log, *, execute=False, limit=None, transport=None):
    if q.object_hash({k:v for k,v in plan.items() if k != 'plan_sha256'}) != plan['plan_sha256']:
        raise ValueError('Changed teacher classification plan')
    ids = [r['request_id'] for r in plan['requests']]
    if len(set(ids)) != len(ids) or any(r['request_id'] != q.object_hash({k:v for k,v in r.items() if k != 'request_id'}) for r in plan['requests']):
        raise ValueError('Changed or duplicate request')
    accepted, _ = load_log(log, plan)
    pending = [r for r in plan['requests'] if r['request_id'] not in accepted]
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError('limit must be positive')
        pending = pending[:limit]
    if not execute:
        return dict(status='dry_run', completed=len(accepted), planned=len(pending), network_called=False)
    key = os.environ.get('DEEPSEEK_API_KEY', '')
    if pending and not key:
        raise ValueError('Missing DEEPSEEK_API_KEY environment variable')
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with Path(log).open('a', encoding='utf-8') as stream:
        for req in pending:
            item = dict(request_id=req['request_id'], plan_sha256=plan['plan_sha256'],
                        timestamp_utc=datetime.now(timezone.utc).isoformat(), endpoint=plan['endpoint'])
            try:
                item['response'] = (transport or q.http_transport)(req['payload'], plan['endpoint'], key)
                item['status'] = 'invalid_response'
                strict_probability(item['response'])
                item['status'] = 'ok'
            except Exception as error:
                item.setdefault('status', 'api_error')
                item['error_type'] = type(error).__name__
            item['record_sha256'] = q.object_hash(item)
            stream.write(q.canonical(item) + '\n')
            stream.flush(); os.fsync(stream.fileno())
            counts[item['status']] += 1
            if item['status'] != 'ok':
                break
    completed = len(accepted) + counts['ok']
    return dict(status='complete' if completed == len(ids) else 'partial', completed=completed,
                remaining=len(ids)-completed, counts=dict(counts), network_called=bool(pending))


def run(config, *, limit=None):
    root = Path(config['runtime']['save_dir'])
    plan = make_plan(config)
    saved = root / 'teacher_request_plan.json'
    if saved.exists() and q.read_json(saved) != plan:
        raise ValueError('Teacher prompt/input plan changed during resume')
    if not saved.exists():
        write_json(saved, plan)
    log = root / 'teacher_responses.jsonl'
    result = generate(plan, log, execute=True, limit=limit)
    if result['status'] != 'complete':
        return result
    accepted, records = load_log(log, plan)
    checkpoint = root / 'teacher_checkpoint.json'
    write_json(checkpoint, dict(format='teacher_direct_response_set_v1', plan_sha256=plan['plan_sha256'],
        responses_sha256=sha256(log), returned_models=dict(Counter(r['response']['model'] for r in records if r['status']=='ok')),
        alias_limit='Requested/returned model aliases are not immutable weights.'))
    for split in ('val', 'test'):
        requests = [r for r in plan['requests'] if r['split'] == split]
        if requests:
            export(config, split, [accepted[r['request_id']] for r in requests], checkpoint)
    write_json(root / 'baseline_cost.json', dict(successful_requests=len(accepted),
        token_usage={key:sum(r.get('response', {}).get('usage', {}).get(key,0) for r in records)
                     for key in ('prompt_tokens','completion_tokens','total_tokens')}))
    return result
