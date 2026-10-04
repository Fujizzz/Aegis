"""Durable per-attempt journal. SDK retries are disabled; every attempt is visible."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

from probability import parse_probability, normalize_comparison, probability
from .protocol import canonical, digest, write_json


class RequestFailed(RuntimeError):
    pass


class UnknownAttempt(RuntimeError):
    pass


class ResponseError(ValueError):
    pass


class HTTPFailure(RuntimeError):
    def __init__(self, response):
        super().__init__('HTTP request failed')
        self.response = response


def response_content(response, kind):
    try:
        choice = response['choices'][0]
        if choice.get('finish_reason') not in (None, 'stop'):
            raise ValueError('Incomplete or refused response')
        text = choice['message']['content'].strip()
        if not text:
            raise ValueError('Empty response')
        if kind == 'score':
            parse_probability(text)
        elif kind == 'compare':
            # Same code-block allowance as the original comparison parser.
            import re
            match = re.search(r'\{.*\}', text, re.S)
            normalize_comparison(json.loads(match.group(0) if match else text))
        elif kind == 'direct':
            value = json.loads(text)
            if type(value['p_positive']) not in (int, float):
                raise ValueError('p_positive must be a JSON number')
            probability(value['p_positive'])
            if not isinstance(value.get('reason'), str):
                raise ValueError('Direct response requires reason')
        return text
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        raise ResponseError('Invalid ' + kind + ' response') from error


def read_journal(path):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        raise ValueError('Truncated journal tail; preserve the file and repair explicitly before resuming')
    records, previous = [], None
    for line in data.splitlines():
        record = json.loads(line)
        signature = record.pop('sha256')
        if record.get('previous') != previous or digest(record) != signature:
            raise ValueError('Journal hash chain mismatch')
        record['sha256'] = signature
        records.append(record)
        previous = signature
    return records


class Journal:
    def __init__(self, output, plan, mode, transport, fixture_sha256=None, retry_unknown=False):
        self.output, self.plan = Path(output), plan
        self.path = self.output/'requests.jsonl'
        self.mode, self.transport, self.retry_unknown = mode, transport, retry_unknown
        execution = dict(plan_sha256=plan['plan_sha256'], mode=mode, fixture_sha256=fixture_sha256)
        path = self.output/'execution.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != execution:
            raise ValueError('Execution mode/fixture mismatch; simulated and live runs cannot mix')
        if not path.exists():
            write_json(path, execution)
        self.records = read_journal(self.path)
        if any(r.get('plan_sha256') != plan['plan_sha256'] for r in self.records):
            raise ValueError('Journal belongs to another plan')
        self.results = {r['sample_id']: r['result'] for r in self.records if r['kind'] == 'sample'}
        self.requests = {}
        self.finished = set()
        for record in self.records:
            if record['kind'] in ('start', 'finish'):
                self.requests.setdefault(record['request_id'], []).append(record)
            if record['kind'] == 'finish':
                self.finished.add(record['attempt_id'])
        unknown = [r for r in self.records if r['kind'] == 'start' and r['attempt_id'] not in self.finished]
        if unknown and not retry_unknown:
            raise UnknownAttempt('Uncommitted calls may have been billed; --retry-unknown explicitly permits another attempt')

    def append(self, record):
        record = dict(record, plan_sha256=self.plan['plan_sha256'],
            previous=self.records[-1]['sha256'] if self.records else None,
            recorded_at=datetime.now(timezone.utc).isoformat())
        record['sha256'] = digest(record)
        with self.path.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(canonical(record) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.records.append(record)
        if record['kind'] in ('start', 'finish'):
            self.requests.setdefault(record['request_id'], []).append(record)
        if record['kind'] == 'finish':
            self.finished.add(record['attempt_id'])
        return record

    def save_sample(self, result):
        self.append(dict(kind='sample', sample_id=result['sample_id'], result=result))
        self.results[result['sample_id']] = result

    def call(self, sample_id, stage, operation, ordinal, payload, response_kind):
        request_id = digest([sample_id, stage, operation, ordinal])
        payload_hash = digest(payload)
        previous = self.requests.get(request_id, [])
        if any(r['payload_sha256'] != payload_hash for r in previous):
            raise ValueError('Changed request payload during resume')
        for record in previous:
            if record['kind'] == 'finish' and record['status'] == 'success':
                return record['response']
        starts = [r for r in previous if r['kind'] == 'start']
        limit = self.plan['config']['max_attempts']
        for number in range(len(starts) + 1, limit + 1):
            context = dict(request_id=request_id, sample_id=sample_id, stage=stage, operation=operation,
                ordinal=ordinal, attempt=number, attempt_id=f'{request_id}:{number}', payload_sha256=payload_hash)
            self.append(dict(context, kind='start', payload=payload))
            before = time.perf_counter()
            response, error_type, status = None, None, 'success'
            try:
                response = self.transport(payload, context)
                response_content(response, response_kind)
            except Exception as error:
                # Never persist exception strings, headers, credentials or transport reprs.
                status, error_type = 'error', type(error).__name__
                response = getattr(error, 'response', response)
            self.append(dict(context, kind='finish', status=status, error_type=error_type,
                duration_seconds=time.perf_counter()-before, response=response))
            if status == 'success':
                return response
            if number < limit and self.mode == 'live':
                time.sleep(self.plan['config']['retry_delay'])
        raise RequestFailed('Request attempts exhausted: ' + request_id)

    def client(self, sample_id, stage, operation, response_kind):
        count = 0
        def create(**payload):
            nonlocal count
            ordinal = count
            count += 1
            response = self.call(sample_id, stage, operation, ordinal, payload, response_kind)
            text = response_content(response, response_kind)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class LiveTransport:
    """OpenAI-compatible HTTP request, one POST per journal attempt; no hidden SDK retries."""
    def __init__(self, config):
        self.config = config

    def __call__(self, payload, context):
        key = os.environ.get(self.config['api_key_env'], '').strip()
        if not key:
            raise RuntimeError('Missing API key environment variable')
        request = urllib.request.Request(self.config['base_url'].rstrip('/') + '/chat/completions',
            data=canonical(payload).encode('utf-8'), method='POST',
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        # Disable redirects: a POST may not silently create an uncounted request.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        try:
            with urllib.request.build_opener(NoRedirect).open(request, timeout=self.config['timeout']) as response:
                body = response.read().decode('utf-8')
        except urllib.error.HTTPError as error:
            body = error.read().decode('utf-8', errors='replace')
            raise HTTPFailure(dict(http_status=error.code, raw_body=body)) from None
        try:
            return json.loads(body, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        except ValueError:
            raise HTTPFailure(dict(http_status=200, raw_body=body, error='invalid_response_json')) from None


class MockTransport:
    """Deterministic fixtures keyed by sample_id. Ground-truth labels are never consulted."""
    def __init__(self, fixtures=None):
        self.fixtures = fixtures or {}

    def __call__(self, payload, context):
        row = self.fixtures.get(context['sample_id'], {})
        key = context['stage'] + '/' + context['operation'] + '/' + str(context['ordinal'])
        faults = row.get('faults', {}).get(key, [])
        fault = faults[context['attempt']-1] if context['attempt'] <= len(faults) else None
        if fault == 'error':
            raise TimeoutError('simulated')
        if fault == 'interrupt':
            raise KeyboardInterrupt('simulated interruption after durable start')
        stage, operation = context['stage'], context['operation']
        if stage == 'screening':
            value = row.get('first', .6)
            text = str(value[ASPECT_INDEX[operation]] if isinstance(value, list) else value)
        elif stage == 'direct':
            text = canonical(dict(p_positive=row.get('direct', .6), reason='simulated response'))
        elif operation == 'reference':
            text = 'Synthetic reference ' + str(context['ordinal'])
        else:
            p = row.get('review', .2)
            text = canonical(dict(closer='fake' if p > .5 else 'true' if p < .5 else 'tie',
                                  proximity_true=1-p, proximity_fake=p))
        if fault == 'malformed':
            text = 'invalid probability/JSON'
        return dict(id='mock-' + context['attempt_id'], model='mock-model-v1',
            choices=[dict(message=dict(content=text), finish_reason='stop')],
            usage=dict(prompt_tokens=10, completion_tokens=5, total_tokens=15), simulated=True)


ASPECT_INDEX = {'Emotion/Sentiment': 0, 'Exaggeration': 1, 'Bias': 2, 'Commonsense Conflict': 3}
