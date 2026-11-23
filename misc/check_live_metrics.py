"""One real OpenRouter call, to confirm the usage accounting against the live API.

Unlike a benchmark run this downloads no dataset and asks one trivial question, so it costs a
fraction of a cent. It exercises the same code pred.py uses -- query_llm, the inline usage
parsing and the generation-history lookup -- and prints the metrics block that would be
written to the jsonl.

Provider name, provider-side latency and the native token counts come from the generation
record rather than the completion, so when they are missing this prints the HTTP status and
body of a direct GET /api/v1/generation, and keeps polling for a while, to separate the
reasons apart: the record not being indexed yet, the key being refused, or the installed
`openrouter` package not being the SDK pred.py expects.

Usage (from the repo root, needs OPENROUTER_API_KEY in .env):
    python misc/check_live_metrics.py [model]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import pred
from openai import OpenAI

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'z-ai/glm-5.3-flash'
PROMPT = 'Reply with exactly this sentence and nothing else: The correct answer is (B)'
POLL_SECONDS = 30

if MODEL not in pred.maxlen_map:
    sys.exit("Add %r to config/model2maxlen.json first." % MODEL)


def describe_backends(gen_stats):
    """Which lookup backend is in play, and whether it has the method we call."""
    try:
        import openrouter
    except Exception as e:
        print('openrouter SDK: not importable (%s)' % e)
    else:
        print('openrouter SDK: %s (version %s)'
              % (getattr(openrouter, '__file__', '?'),
                 getattr(openrouter, '__version__', 'unknown')))
    if gen_stats.client is not None:
        generations = getattr(gen_stats.client, 'generations', None)
        getter = getattr(generations, 'get_generation', None)
        print('  client.generations.get_generation: %s'
              % (getter if getter is not None else
                 'MISSING -- the installed "openrouter" package is not the SDK pred.py '
                 'expects; the plain GET below is what will be used'))
    print('requests session: %s' % ('ready' if gen_stats.session is not None else 'unavailable'))
    if gen_stats.last_error:
        print('note: %s' % gen_stats.last_error)


def poll_raw_record(generation_id, seconds=POLL_SECONDS):
    """GET the record directly, retrying while it is merely not indexed yet.

    Goes straight to the documented endpoint (never the SDK) so the HTTP status and body are
    visible instead of being flattened into an exception.
    """
    probe = pred.GenerationStats(pred.API_KEY)
    probe.client = None
    if probe.session is None:
        print('requests is not installed, so the raw record cannot be fetched here.')
        return None, 'no requests session'
    deadline = time.time() + seconds
    try:
        while True:
            record, pending = probe._fetch_http(generation_id)
            if record:
                return record, None
            if not pending or time.time() >= deadline:
                return None, probe.last_error
            print('  %s -- retrying for up to %ds' % (probe.last_error, seconds))
            time.sleep(2)
    finally:
        probe.close()


client = OpenAI(base_url=pred.URL, api_key=pred.API_KEY)
tokenizer = pred.tiktoken.get_encoding('cl100k_base')
gen_stats = pred.GenerationStats(pred.API_KEY)
describe_backends(gen_stats)

try:
    content, calls = pred.query_llm(PROMPT, MODEL, tokenizer, client, temperature=0.1,
                                    max_new_tokens=64, gen_stats=gen_stats)
    lookup_error = gen_stats.last_error
finally:
    gen_stats.close()

print('\nresponse: %r' % content)
print('extracted answer: %r' % pred.extract_answer(content or ''))
print('\nmetrics block as written to the jsonl:')
print(json.dumps(pred.summarize_calls(calls), indent=2, ensure_ascii=False))

last = calls[-1] if calls else {}
print('\n--- provider-side timings (these come from the generation record) ---')
if last.get('latency_ms') is not None:
    print('provider_name        = %r' % last.get('provider_name'))
    print('latency_ms           = %r  (OpenRouter time to first token)' % last.get('latency_ms'))
    print('generation_time_ms   = %r' % last.get('generation_time_ms'))
    print('native_tokens_prompt = %r' % last.get('native_tokens_prompt'))
else:
    print('MISSING from the call record: %s' % (lookup_error or 'the lookup returned nothing'))

# The raw record too, so a field OpenRouter renamed is visible rather than silently missing.
generation_id = last.get('generation_id')
if generation_id:
    print('\nraw generation record for %s:' % generation_id)
    raw, error = poll_raw_record(generation_id)
    if not raw:
        print('could not be fetched: %s' % (error or 'no reason reported'))
        print('Cost and token counts above are unaffected -- they arrive inline with the '
              'completion. Only provider, latency and native token counts need this record.')
    else:
        print(json.dumps(raw, indent=2, default=str))
        missing = [name for name in pred.GENERATION_FIELDS if name not in raw]
        if missing:
            print('\nNOT in the record (GENERATION_FIELDS may need updating): %s'
                  % ', '.join(missing))
        else:
            print('\nEvery field in GENERATION_FIELDS is present.')
        if raw.get('latency') is None:
            print('The record carries no `latency`, so OpenRouter did not report one for '
                  'this generation.')

failed = [call for call in calls if not call.get('ok')]
if failed:
    print('\n%d of %d attempt(s) failed: %s'
          % (len(failed), len(calls), [call['error'] for call in failed]))
