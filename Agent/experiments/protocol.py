"""Immutable inputs, explicit task semantics and experiment configuration."""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

from Aegis.utils.label_schema import get_label_schema
from probability import normalized_weights, thresholds

ASPECTS = ['Emotion/Sentiment', 'Exaggeration', 'Bias', 'Commonsense Conflict']
CHINESE = ['情绪煽动', '夸张程度', '偏见倾向', '常识冲突']
DEFAULTS = dict(model='deepseek-chat', base_url='https://api.deepseek.com',
    api_key_env='DEEPSEEK_API_KEY', real_threshold=.3, fake_threshold=.7,
    trigger_cap=.9, self_consistency_n=1, first_weights=[.30, .25, .30, .15],
    second_weights=[.30, .25, .30, .15], temperature_first=.1,
    temperature_true=.25, temperature_fake=1.3, temperature_compare=.2,
    temperature_direct=.1, max_tokens=512, reference_max_tokens=800,
    timeout=45., max_attempts=3, retry_delay=1., reference_context='metadata_only')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def configuration(path, input_override=None):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding='utf-8-sig'))
    allowed = set(DEFAULTS) | {'input_csv', 'label_schema'}
    if set(raw) - allowed:
        raise ValueError('Unknown configuration fields: ' + str(sorted(set(raw) - allowed)))
    config = dict(DEFAULTS, **raw)
    get_label_schema(config.get('label_schema'))  # Required, never inferred.
    source = Path(input_override or config['input_csv'])
    config['input_csv'] = str((source if source.is_absolute() else path.parent / source).resolve())
    thresholds(config['real_threshold'], config['fake_threshold'], config['trigger_cap'])
    for name in ('first_weights', 'second_weights'):
        if not isinstance(config[name], list) or len(config[name]) != 4:
            raise ValueError(name + ' must contain four weights in the documented aspect order')
        config[name] = list(normalized_weights(dict(enumerate(config[name])), range(4)).values())
    for name in ('self_consistency_n', 'max_tokens', 'reference_max_tokens', 'max_attempts'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(name + ' must be a positive integer')
    for name in ('timeout', 'retry_delay', 'temperature_first', 'temperature_true',
                 'temperature_fake', 'temperature_compare', 'temperature_direct'):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(name + ' must be finite and nonnegative')
    if config['timeout'] == 0:
        raise ValueError('timeout must be positive')
    if config['reference_context'] not in ('metadata_only', 'with_content'):
        raise ValueError('Unknown reference_context')
    for name in ('model', 'api_key_env', 'base_url'):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError('Missing ' + name)
    from urllib.parse import urlsplit
    endpoint = urlsplit(config['base_url'])
    if endpoint.scheme != 'https' or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError('base_url must be an HTTPS URL without credentials/query/fragment')
    return config


def prepare(config, output):
    schema = get_label_schema(config['label_schema'])
    source = Path(config['input_csv'])
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    samples, ids = [], set()
    with io.StringIO(source_bytes.decode('utf-8-sig'), newline='') as stream:
        for number, row in enumerate(csv.DictReader(stream), 1):
            sample_id = (row.get('sample_id') or '').strip() or f'csv:{source_hash}:{number}'
            if sample_id in ids:
                raise ValueError('Duplicate sample_id: ' + sample_id)
            ids.add(sample_id)
            visible = {k: (row.get(k) or '').strip() for k in ('title', 'subject', 'date')}
            visible['content'] = (row.get('text') or '').strip() or (row.get('content') or '').strip()
            if not visible['content']:
                raise ValueError(f'Empty body at row {number}')
            # Legacy numeric IDs and teacher labels/reasons are never model inputs.
            samples.append(dict(sample_id=sample_id, row_number=number, event=row.get('event', ''),
                target=schema.encode_source(row.get('label')), source_label=row.get('label'),
                input=visible, input_sha256=digest(visible), title_missing=not bool(visible['title'])))
    if not samples:
        raise ValueError('Empty dataset')
    root = Path(__file__).resolve().parents[1]
    files = [root/'probability.py', root/'run_detector.py', root/'run_review_experiment.py',
             root/'agent/deepseek_agent.py', root/'detector/first_layer_detector.py',
             root/'detector/second_layer_detector.py', root.parent/'Aegis/utils/label_schema.py']
    files += sorted((root/'experiments').glob('*.py'))
    code = {str(p.relative_to(root.parent)): file_hash(p) for p in files}
    adaptation = 'original_reference_fake_v1' if schema.positive == 'fake' else 'reference_rumour_adaptation_v1'
    plan = dict(format='t24_review_v1', config=config, label_schema=schema.metadata(),
        method=adaptation, reference_context=config['reference_context'], source_sha256=source_hash,
        samples=samples, code_sha256=code,
        protocol=dict(threshold=.5, uncertain_policy='binary_at_0.5_for_comparison; routing_label_retained',
            failed_review_policy='abstain_no_fallback', first_layer_shared=True,
            generated_references='synthetic_style_references_not_fact_checked_evidence'),
        planned_calls=dict(screening=len(samples)*4*config['self_consistency_n'], direct=len(samples),
            review_min=0, review_max=6*len(samples),
            attempt_upper_bound=len(samples)*(4*config['self_consistency_n']+7)*config['max_attempts']))
    plan['plan_sha256'] = digest(plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'plan.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Plan/data/config/code mismatch; use a new output directory')
    elif any(output.iterdir()):
        raise ValueError('Output directory is nonempty and has no plan')
    else:
        write_json(path, plan)
    return plan
