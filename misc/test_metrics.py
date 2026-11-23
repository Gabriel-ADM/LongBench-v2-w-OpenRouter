"""Offline checks for the usage/cost metrics pred.py records.

Stubs the OpenAI client and the OpenRouter generation lookup, so it runs without network
access or an API key. Run from the repo root: python misc/test_metrics.py
"""
import json
import os
import sys
import tempfile
import types

os.environ.setdefault('OPENROUTER_API_KEY', 'test-key')
sys.path.insert(0, os.getcwd())

import pred


class FakeCompletion:
    """Mimics the OpenAI SDK's pydantic completion, carrying OpenRouter's extra usage keys."""

    def __init__(self, content='The correct answer is (B)', gen_id='gen-abc', cost=0.0012,
                 prompt_tokens=50000, completion_tokens=300, reasoning_tokens=120):
        self._content = content
        self._payload = {
            'id': gen_id,
            'model': 'z-ai/glm-5.3-flash',
            'usage': {
                'prompt_tokens': prompt_tokens,
                'prompt_tokens_details': {'cached_tokens': 10},
                'completion_tokens': completion_tokens,
                'completion_tokens_details': {'reasoning_tokens': reasoning_tokens},
                'total_tokens': prompt_tokens + completion_tokens,
                'cost': cost,
                'cost_details': {'upstream_inference_cost': 0.0009},
            },
        }

    def model_dump(self):
        return self._payload

    @property
    def choices(self):
        message = types.SimpleNamespace(content=self._content)
        return [types.SimpleNamespace(message=message)]


class FakeTokenizer:
    """Stands in for tiktoken, whose encoding table is a network download."""

    def encode(self, text, disallowed_special=None):
        return list(text.encode('utf-8'))

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode('utf-8', 'ignore')


class FakeClient:
    """`errors` is a count of leading attempts that blow up, to exercise the retry path."""

    def __init__(self, errors=0, content='The correct answer is (B)'):
        self.errors = errors
        self.content = content
        self.calls = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.errors:
            raise RuntimeError('provider returned 502')
        return FakeCompletion(content=self.content, gen_id='gen-%d' % self.calls)


# The documented components.GenerationResponseData shape, wrapped the way
# components.GenerationResponse wraps it.
GENERATION_RECORD = {
    'data': {
        'id': 'gen-abc',
        'model': 'z-ai/glm-5.3-flash',
        'provider_name': 'DeepInfra',
        'latency': 980.0,
        'generation_time': 11200.0,
        'moderation_latency': None,
        'tokens_prompt': 50000,
        'tokens_completion': 300,
        'native_tokens_prompt': 50011,
        'native_tokens_completion': 301,
        'native_tokens_reasoning': 120,
        'native_tokens_cached': 10,
        'total_cost': 0.0012,
        'cache_discount': 0.0,
        'finish_reason': 'stop',
        'native_finish_reason': 'stop',
        'streamed': False,
    }
}


class FakeResponse:
    """Enough of a requests.Response for the GET /api/v1/generation path."""

    def __init__(self, status_code=200, payload=GENERATION_RECORD, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError('body is not JSON')
        return self._payload


class FakeSession:
    """Serves canned responses in order, repeating the last one."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.requests.append((url, params, timeout))
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        pass


class BrokenSDKClient:
    """An installed `openrouter` package that is not the SDK pred.py expects."""

    def __init__(self, error=None):
        self.error = error or AttributeError(
            "'OpenRouter' object has no attribute 'generations'")
        self.generations = types.SimpleNamespace(get_generation=self._raise)

    def _raise(self, id=None):
        raise self.error

    def close(self):
        pass


class FakeGenStats(pred.GenerationStats):
    """Skips the real client construction and serves the canned record.

    `misses` is how many 404s to return first, to exercise the not-indexed-yet retry.
    """

    def __init__(self, misses=0, record=GENERATION_RECORD):
        self.api_key = 'test-key'
        self.retries = 3
        self.timeout = 30
        self.client = None
        # Never used -- _fetch_once is overridden below -- but fetch() checks that some
        # backend exists before it tries anything.
        self.session = FakeSession([FakeResponse()])
        self.last_error = None
        self.warned = False
        self.misses = misses
        self.record = record
        self.fetches = 0

    def _fetch_once(self, generation_id):
        self.fetches += 1
        if self.fetches <= self.misses:
            return None, True
        raw = self.record
        return (raw.get('data') or raw), False


def check(label, got, want):
    assert got == want, '%s: got %r, want %r' % (label, got, want)
    print('ok   %s = %r' % (label, got))


def test_usage_from_completion():
    usage = pred.usage_from_completion(FakeCompletion())
    check('generation_id', usage['generation_id'], 'gen-abc')
    check('cost', usage['cost'], 0.0012)
    check('upstream_inference_cost', usage['upstream_inference_cost'], 0.0009)
    check('prompt_tokens', usage['prompt_tokens'], 50000)
    check('cached_tokens', usage['cached_tokens'], 10)
    check('completion_tokens', usage['completion_tokens'], 300)
    check('reasoning_tokens', usage['reasoning_tokens'], 120)
    check('total_tokens', usage['total_tokens'], 50300)
    # A provider that reports no usage at all must not take the run down.
    bare = pred.usage_from_completion(types.SimpleNamespace())
    check('no-usage fallback', bare, {})


def test_total_tokens_fallback():
    completion = FakeCompletion()
    del completion._payload['usage']['total_tokens']
    check('total_tokens derived', pred.usage_from_completion(completion)['total_tokens'], 50300)


def test_generation_fields():
    stats = FakeGenStats()
    record = stats.fetch('gen-abc')
    check('provider_name', record['provider_name'], 'DeepInfra')
    check('latency_ms', record['latency_ms'], 980.0)
    check('generation_time_ms', record['generation_time_ms'], 11200.0)
    check('native_tokens_reasoning', record['native_tokens_reasoning'], 120)
    check('tokens_prompt', record['tokens_prompt'], 50000)
    check('total_cost', record['total_cost'], 0.0012)
    check('finish_reason', record['finish_reason'], 'stop')
    check('null fields dropped', 'moderation_latency_ms' in record, False)


def test_generation_retry_then_give_up():
    stats = FakeGenStats(misses=2)
    check('record after 2 misses', stats.fetch('gen-abc')['provider_name'], 'DeepInfra')
    check('fetches used', stats.fetches, 3)
    never = FakeGenStats(misses=99)
    check('gives up empty', never.fetch('gen-abc'), {})
    check('attempts bounded', never.fetches, 4)
    check('no id, no fetch', FakeGenStats().fetch(None), {})


def test_sdk_failure_falls_back_to_http():
    """A broken `openrouter` package must not cost the run its latency numbers."""
    stats = pred.GenerationStats('test-key')
    stats.client = BrokenSDKClient()
    stats.session = FakeSession([FakeResponse()])
    record = stats.fetch('gen-abc')
    check('latency via the HTTP fallback', record['latency_ms'], 980.0)
    check('provider via the HTTP fallback', record['provider_name'], 'DeepInfra')
    check('broken SDK dropped for later calls', stats.client, None)
    url, params, timeout = stats.session.requests[0]
    check('GET url', url, pred.GENERATION_URL)
    check('GET params', params, {'id': 'gen-abc'})
    check('GET timeout', timeout, 30)
    stats.close()


def test_lookup_errors_are_reported_not_swallowed():
    """The reason an empty record is empty has to reach the operator."""
    stats = pred.GenerationStats('test-key')
    stats.client = None
    stats.session = FakeSession([FakeResponse(
        status_code=401, payload={'error': {'code': 401, 'message': 'No auth credentials'}})])
    check('401 -> empty record', stats.fetch('gen-abc'), {})
    assert '401' in stats.last_error, stats.last_error
    assert 'No auth credentials' in stats.last_error, stats.last_error
    print('ok   last_error = %r' % stats.last_error)
    check('warned', stats.warned, True)

    # A 200 whose body is not JSON, and a transport error, must also name themselves.
    stats.warned, stats.session = False, FakeSession([FakeResponse(payload=None, text='<html>')])
    check('bad body -> empty', stats.fetch('gen-abc'), {})
    assert 'not JSON' in stats.last_error, stats.last_error
    print('ok   last_error = %r' % stats.last_error)
    stats.session = FakeSession([OSError('connection reset')])
    check('transport error -> empty', stats.fetch('gen-abc'), {})
    assert 'connection reset' in stats.last_error, stats.last_error
    print('ok   last_error = %r' % stats.last_error)
    stats.close()


def test_http_404_is_retried_then_succeeds():
    """The record lands a moment after the completion, so a 404 means 'not yet'."""
    stats = pred.GenerationStats('test-key', retries=1)
    stats.client = None
    stats.session = FakeSession([FakeResponse(status_code=404, payload={}), FakeResponse()])
    check('record after one 404', stats.fetch('gen-abc')['latency_ms'], 980.0)
    check('GETs used', len(stats.session.requests), 2)
    check('error cleared on success', stats.last_error, None)

    stats.session = FakeSession([FakeResponse(status_code=404, payload={})])
    check('persistent 404 -> empty', stats.fetch('gen-abc'), {})
    assert 'not indexed yet' in stats.last_error, stats.last_error
    print('ok   last_error = %r' % stats.last_error)
    check('retries bounded', len(stats.session.requests), 2)
    stats.close()


def test_query_llm_records_each_attempt():
    tokenizer = FakeTokenizer()
    client = FakeClient(errors=2)
    content, calls = pred.query_llm('hi', 'z-ai/glm-5.3-flash', tokenizer, client,
                                    gen_stats=FakeGenStats())
    check('content', content, 'The correct answer is (B)')
    check('attempts recorded', len(calls), 3)
    check('failures recorded', [c['ok'] for c in calls], [False, False, True])
    check('error kept', calls[0]['error'], 'provider returned 502')
    assert calls[-1]['inference_time'] > 0, 'inference_time should be positive'
    print('ok   inference_time = %.6f' % calls[-1]['inference_time'])
    check('provider on success', calls[-1]['provider_name'], 'DeepInfra')
    check('no provider lookup on failure', 'provider_name' in calls[0], False)

    exhausted_content, exhausted_calls = pred.query_llm(
        'hi', 'z-ai/glm-5.3-flash', tokenizer, FakeClient(errors=99))
    check('exhausted content', exhausted_content, '')
    check('exhausted attempts', len(exhausted_calls), 5)

    # A provider answering with content=None used to crash on .strip() downstream.
    empty_content, _ = pred.query_llm('hi', 'z-ai/glm-5.3-flash', tokenizer,
                                      FakeClient(content=None))
    check('None content -> empty', empty_content, '')


def test_summarize_calls():
    tokenizer = FakeTokenizer()
    _, calls = pred.query_llm('hi', 'z-ai/glm-5.3-flash', tokenizer, FakeClient(errors=1),
                              gen_stats=FakeGenStats())
    metrics = pred.summarize_calls(calls)
    check('requests', metrics['requests'], 2)
    check('failed_requests', metrics['failed_requests'], 1)
    check('cost', round(metrics['cost'], 6), 0.0012)
    check('prompt_tokens', metrics['prompt_tokens'], 50000)
    check('reasoning_tokens', metrics['reasoning_tokens'], 120)
    check('completion_tokens', metrics['completion_tokens'], 300)
    check('total_tokens', metrics['total_tokens'], 50300)
    check('latency_ms', metrics['latency_ms'], 980.0)
    check('generation_time_ms', metrics['generation_time_ms'], 11200.0)
    check('providers', metrics['providers'], ['DeepInfra'])
    check('metrics json-serializable', isinstance(json.dumps(metrics), str), True)


ITEM = {
    '_id': 'id-1', 'domain': 'Single-Document QA', 'sub_domain': 'Event ordering',
    'difficulty': 'easy', 'length': 'short', 'question': 'q',
    'choice_A': 'a', 'choice_B': 'b', 'choice_C': 'c', 'choice_D': 'd',
    'answer': 'B', 'context': 'some context',
}


def run_pred_once(cot):
    args = pred.argparse.Namespace(model='z-ai/glm-5.3-flash', rag=0, no_context=False,
                                   cot=cot, skip_gen_stats=False)
    tokenizer = FakeTokenizer()
    with tempfile.NamedTemporaryFile('w+', suffix='.jsonl', delete=False) as fout:
        pred.run_pred([dict(ITEM)], args, fout, 'z-ai/glm-5.3-flash', tokenizer,
                      FakeClient(), FakeGenStats())
        path = fout.name
    rows = [json.loads(line) for line in open(path, encoding='utf-8')]
    os.unlink(path)
    return rows


def test_run_pred_attaches_metrics():
    rows = run_pred_once(cot=False)
    check('rows written', len(rows), 1)
    item = rows[0]
    check('judge still scored', item['judge'], True)
    check('pred still extracted', item['pred'], 'B')
    check('requests (no cot)', item['metrics']['requests'], 1)
    check('calls kept', len(item['metrics']['calls']), 1)
    check('cost on item', round(item['metrics']['cost'], 6), 0.0012)

    cot_item = run_pred_once(cot=True)[0]
    check('requests (cot)', cot_item['metrics']['requests'], 2)
    check('cost doubled (cot)', round(cot_item['metrics']['cost'], 6), 0.0024)
    check('tokens doubled (cot)', cot_item['metrics']['total_tokens'], 100600)


def test_truncation_still_works():
    """The context-window truncation path is upstream of the metrics changes; keep it covered."""
    model = 'z-ai/glm-5.3-flash'
    original = pred.maxlen_map[model]
    pred.maxlen_map[model] = 10
    try:
        client = FakeClient()
        sent = {}
        create = client._create

        def capture(**kwargs):
            sent.update(kwargs)
            return create(**kwargs)

        client.chat.completions.create = capture
        pred.query_llm('x' * 500, model, FakeTokenizer(), client)
        check('prompt truncated', len(sent['messages'][0]['content']), 10)
    finally:
        pred.maxlen_map[model] = original


def test_gen_stats_backend_selection():
    """Whichever backend is installed, a missing one must not break a run."""
    stats = pred.GenerationStats('test-key')
    try:
        check('a backend was picked', stats.client is not None or stats.session is not None, True)
    finally:
        stats.close()
    # No backend at all -> the lookup is a no-op, not an exception.
    stats.client, stats.session = None, None
    check('no backend -> empty', stats.fetch('gen-abc'), {})
    stats.close()


def test_skip_gen_stats():
    tokenizer = FakeTokenizer()
    _, calls = pred.query_llm('hi', 'z-ai/glm-5.3-flash', tokenizer, FakeClient(),
                              gen_stats=None)
    check('cost still recorded', calls[0]['cost'], 0.0012)
    check('no provider', 'provider_name' in calls[0], False)


def fake_run_rows(n=12):
    """Synthetic result rows whose totals are easy to check by hand."""
    rows = []
    for i in range(n):
        cost = round(0.001 + i * 0.0001, 6)
        provider = 'DeepInfra' if i % 3 else 'Chutes'
        calls = [{'attempt': 1, 'ok': True, 'error': None, 'inference_time': 10.0 + i,
                  'generation_id': 'gen-%d' % i, 'cost': cost,
                  'prompt_tokens': 50000 + i, 'cached_tokens': 0,
                  'completion_tokens': 300, 'reasoning_tokens': 120,
                  'total_tokens': 50300 + i, 'provider_name': provider,
                  'latency_ms': 900.0, 'generation_time_ms': 11000.0}]
        rows.append({
            '_id': 'id-%d' % i,
            'difficulty': 'easy' if i % 2 else 'hard',
            'length': ['short', 'medium', 'long'][i % 3],
            'answer': 'B', 'pred': 'B' if i % 4 else 'A', 'judge': bool(i % 4),
            'metrics': {
                'requests': 1, 'failed_requests': 0,
                'inference_time': 10.0 + i, 'latency_ms': 900.0,
                'generation_time_ms': 11000.0,
                'cost': cost, 'prompt_tokens': 50000 + i, 'cached_tokens': 0,
                'reasoning_tokens': 120, 'completion_tokens': 300,
                'total_tokens': 50300 + i, 'providers': [provider], 'calls': calls,
            },
        })
    return rows


def test_result_py_aggregates():
    """Runs result.py over a synthetic results dir and checks its totals.

    result.py reads ./results and writes ./result.txt, so it runs in a scratch cwd.
    """
    import shutil
    import subprocess

    repo = os.getcwd()
    scratch = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(scratch, 'results'))
        shutil.copy(os.path.join(repo, 'result.py'), scratch)
        with open(os.path.join(scratch, 'results', 'fake-model.jsonl'), 'w',
                  encoding='utf-8') as f:
            for row in fake_run_rows():
                f.write(json.dumps(row) + '\n')
        # A run from before usage accounting existed, to check it degrades to '-'.
        with open(os.path.join(scratch, 'results', 'old-model.jsonl'), 'w',
                  encoding='utf-8') as f:
            for row in fake_run_rows(4):
                row.pop('metrics')
                f.write(json.dumps(row) + '\n')
        # A run whose generation lookup found nothing: cost and tokens are real, the
        # provider-side columns must read '-' instead of a zero that looks like a number.
        with open(os.path.join(scratch, 'results', 'nolookup-model.jsonl'), 'w',
                  encoding='utf-8') as f:
            for row in fake_run_rows(4):
                row['metrics'].update(latency_ms=0, generation_time_ms=0, providers=[])
                f.write(json.dumps(row) + '\n')
        proc = subprocess.run([sys.executable, 'result.py'], cwd=scratch,
                              capture_output=True, text=True)
        check('result.py exit code', proc.returncode, 0)
        tsv = open(os.path.join(scratch, 'result.txt'), encoding='utf-8').read()
    finally:
        shutil.rmtree(scratch)

    rows = {line.split('\t')[0]: line.split('\t') for line in tsv.strip().split('\n')}
    header = rows['Model']
    fake = dict(zip(header, rows['fake-model']))
    # 9 of 12 judged correct; easy are the odd i (all correct), hard the even (3 of 6).
    check('Overall', fake['Overall'], '75.0')
    check('Easy', fake['Easy'], '100.0')
    check('Hard', fake['Hard'], '50.0')
    check('Provider', fake['Provider'], 'Chutes/DeepInfra')
    check('Requests', fake['Requests'], '12')
    check('Cost($)', fake['Cost($)'], '0.0186')      # 12*0.001 + 0.0001*66
    check('Infer(s)', fake['Infer(s)'], '186.0')     # 12*10 + 66
    check('Infer/q(s)', fake['Infer/q(s)'], '15.5')
    check('Lat/q(ms)', fake['Lat/q(ms)'], '900')      # 12*900 / 12
    check('Gen/q(s)', fake['Gen/q(s)'], '11.0')       # 12*11000ms / 12 / 1000
    check('PromptTok', fake['PromptTok'], '600066')
    check('ReasonTok', fake['ReasonTok'], '1440')
    check('CompTok', fake['CompTok'], '3600')
    check('TotalTok', fake['TotalTok'], '603666')
    check('$/1M tok', fake['$/1M tok'], '0.0308')    # 0.0186 / 603666 * 1e6

    old = dict(zip(header, rows['old-model']))
    check('metrics-less accuracy kept', old['Overall'], '75.0')
    check('metrics-less usage blank', [old[c] for c in USAGE_HEADER[1:]],
          ['-'] * (len(USAGE_HEADER) - 1))
    check('usage table printed', 'USAGE' in proc.stdout, True)

    nolookup = dict(zip(header, rows['nolookup-model']))
    check('cost survives a failed lookup', nolookup['Cost($)'], '0.0046')  # 4*0.001 + 0.0006
    check('wall-clock survives', nolookup['Infer(s)'], '46.0')            # 4*10 + 6
    check('no provider -> -', nolookup['Provider'], '-')
    check('no latency -> -', nolookup['Lat/q(ms)'], '-')
    check('no generation time -> -', nolookup['Gen/q(s)'], '-')


USAGE_HEADER = ["Model", "Provider", "Requests", "Cost($)", "Infer(s)", "Infer/q(s)",
                "Lat/q(ms)", "Gen/q(s)", "PromptTok", "ReasonTok", "CompTok", "TotalTok",
                "$/1M tok"]


BANNER = """\
NOTE: every API response below is STUBBED -- no network call is made and no API key is used.
Costs, token counts, providers and latencies printed here are literals from this file
(FakeCompletion, GENERATION_RECORD), not values OpenRouter returned. These checks prove the
parsing and aggregation logic, NOT that OpenRouter's live response still has these field
names. For that, run: python misc/check_live_metrics.py
"""


if __name__ == '__main__':
    print(BANNER)
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            print('\n--- %s' % name)
            test()
    print('\nAll metrics checks passed (against stubbed responses -- see the note above).')
