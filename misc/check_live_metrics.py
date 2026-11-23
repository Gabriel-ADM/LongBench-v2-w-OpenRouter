"""One real OpenRouter call, to confirm the usage accounting against the live API.

Unlike a benchmark run this downloads no dataset and asks one trivial question, so it costs a
fraction of a cent. It exercises the same code pred.py uses -- query_llm, the inline usage
parsing and the generation-history lookup -- and prints the metrics block that would be
written to the jsonl, plus the raw generation record so unexpected field names show up.

Usage (from the repo root, needs OPENROUTER_API_KEY in .env):
    python misc/check_live_metrics.py [model]
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import pred
from openai import OpenAI

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'z-ai/glm-5.3-flash'
PROMPT = 'Reply with exactly this sentence and nothing else: The correct answer is (B)'

if MODEL not in pred.maxlen_map:
    sys.exit("Add %r to config/model2maxlen.json first." % MODEL)

client = OpenAI(base_url=pred.URL, api_key=pred.API_KEY)
tokenizer = pred.tiktoken.get_encoding('cl100k_base')
gen_stats = pred.GenerationStats(pred.API_KEY)
print('generation lookup backend: %s' % ('openrouter SDK' if gen_stats.client is not None
                                         else 'requests' if gen_stats.session is not None
                                         else 'none'))

try:
    content, calls = pred.query_llm(PROMPT, MODEL, tokenizer, client, temperature=0.1,
                                    max_new_tokens=64, gen_stats=gen_stats)
finally:
    gen_stats.close()

print('\nresponse: %r' % content)
print('extracted answer: %r' % pred.extract_answer(content or ''))
print('\nmetrics block as written to the jsonl:')
print(json.dumps(pred.summarize_calls(calls), indent=2, ensure_ascii=False))

# The raw record too, so a field OpenRouter renamed is visible rather than silently missing.
generation_id = calls[-1].get('generation_id') if calls else None
if generation_id:
    probe = pred.GenerationStats(pred.API_KEY)
    try:
        raw, _ = probe._fetch_once(generation_id)
    finally:
        probe.close()
    print('\nraw generation record for %s:' % generation_id)
    print(json.dumps(raw, indent=2, default=str))
    missing = [name for name in pred.GENERATION_FIELDS if name not in (raw or {})]
    if missing:
        print('\nNOT in the record (GENERATION_FIELDS may need updating): %s'
              % ', '.join(missing))
    else:
        print('\nEvery field in GENERATION_FIELDS is present.')

failed = [call for call in calls if not call.get('ok')]
if failed:
    print('\n%d of %d attempt(s) failed: %s'
          % (len(failed), len(calls), [call['error'] for call in failed]))
