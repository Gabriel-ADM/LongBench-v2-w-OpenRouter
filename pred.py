import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp

# avoid RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase.
# Python 3.14 uses forkserver as default method, which causes TypeError: cannot pickle 'TextIOWrapper' instances when using multiprocessing with file handles. So we set the start method to fork to avoid this issue.
mp.set_start_method("fork", force=True)

from dotenv import load_dotenv
load_dotenv(override=True)

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

URL = "https://openrouter.ai/api/v1"
GENERATION_URL = "https://openrouter.ai/api/v1/generation"
API_KEY = os.environ["OPENROUTER_API_KEY"]
template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def usage_from_completion(completion):
    """Pull OpenRouter's usage accounting off a chat completion.

    OpenRouter adds non-standard fields (`cost`, `cost_details`) to `usage`. The OpenAI SDK
    keeps them because its pydantic models allow extras, so dump the whole completion and
    read the resulting dict instead of touching attributes that may not exist.
    """
    try:
        data = completion.model_dump()
    except Exception:
        return {}
    usage = data.get('usage') or {}
    prompt_details = usage.get('prompt_tokens_details') or {}
    completion_details = usage.get('completion_tokens_details') or {}
    cost_details = usage.get('cost_details') or {}
    prompt_tokens = usage.get('prompt_tokens') or 0
    completion_tokens = usage.get('completion_tokens') or 0
    return {
        'generation_id': data.get('id'),
        'model': data.get('model'),
        'cost': usage.get('cost') or 0.0,
        'upstream_inference_cost': cost_details.get('upstream_inference_cost') or 0.0,
        'prompt_tokens': prompt_tokens,
        'cached_tokens': prompt_details.get('cached_tokens') or 0,
        'completion_tokens': completion_tokens,
        'reasoning_tokens': completion_details.get('reasoning_tokens') or 0,
        'total_tokens': usage.get('total_tokens') or (prompt_tokens + completion_tokens),
    }

# Field names of components.GenerationResponseData, mapped to the names we store. Times are
# documented as milliseconds, so they get a _ms suffix here.
GENERATION_FIELDS = {
    'provider_name': 'provider_name',
    'latency': 'latency_ms',
    'generation_time': 'generation_time_ms',
    'moderation_latency': 'moderation_latency_ms',
    'tokens_prompt': 'tokens_prompt',
    'tokens_completion': 'tokens_completion',
    'native_tokens_prompt': 'native_tokens_prompt',
    'native_tokens_completion': 'native_tokens_completion',
    'native_tokens_reasoning': 'native_tokens_reasoning',
    'native_tokens_cached': 'native_tokens_cached',
    'total_cost': 'total_cost',
    'cache_discount': 'cache_discount',
    'finish_reason': 'finish_reason',
    'native_finish_reason': 'native_finish_reason',
    'streamed': 'streamed',
}

class GenerationStats:
    """Reads the OpenRouter activity-log record for a generation.

    This is the same row the OpenRouter dashboard shows, and the only place that carries the
    provider that served the request, its provider-side latency and the native token counts.
    Uses the official `openrouter` SDK when it is installed and falls back to a plain GET on
    /api/v1/generation otherwise, so a run never hard-depends on the extra package.
    """

    def __init__(self, api_key, retries=3):
        self.api_key = api_key
        self.retries = retries
        self.client = None
        self.session = None
        try:
            from openrouter import OpenRouter
            self.client = OpenRouter(api_key=api_key)
        except Exception:
            try:
                import requests
                self.session = requests.Session()
            except Exception:
                # Neither backend available: the lookup turns into a no-op rather than
                # taking the run down over an optional metric.
                print("No OpenRouter SDK or requests available; skipping generation stats.")

    def close(self):
        for backend in (self.client, self.session):
            try:
                backend.close()
            except Exception:
                pass

    def fetch(self, generation_id):
        """Return the normalized record, or {} if it never showed up. Never raises."""
        if not generation_id:
            return {}
        delay = 0.5
        for attempt in range(self.retries + 1):
            record, pending = self._fetch_once(generation_id)
            if record:
                return {dest: record[src] for src, dest in GENERATION_FIELDS.items()
                        if record.get(src) is not None}
            # The record is written a moment after the completion returns, so a 404 right
            # after the call just means "not indexed yet" and is worth another look.
            if not pending or attempt == self.retries:
                return {}
            time.sleep(delay)
            delay *= 2
        return {}

    def _fetch_once(self, generation_id):
        """-> (record dict or None, whether it is worth retrying)"""
        if self.client is None and self.session is None:
            return None, False
        if self.client is not None:
            try:
                res = self.client.generations.get_generation(id=generation_id)
            except Exception as e:
                return None, 'notfound' in type(e).__name__.lower()
            try:
                raw = res.model_dump()
            except Exception:
                raw = res if isinstance(res, dict) else {}
        else:
            try:
                resp = self.session.get(
                    GENERATION_URL,
                    params={'id': generation_id},
                    headers={'Authorization': 'Bearer %s' % self.api_key},
                    timeout=30,
                )
            except Exception:
                return None, False
            if resp.status_code == 404:
                return None, True
            if resp.status_code >= 400:
                return None, False
            try:
                raw = resp.json()
            except Exception:
                return None, False
        if not isinstance(raw, dict):
            return None, False
        return (raw.get('data') or raw), False

def summarize_calls(calls):
    """Roll the per-call records of one question up into the item's usage block."""
    def total(key):
        return sum(call.get(key) or 0 for call in calls)
    providers = []
    for call in calls:
        provider = call.get('provider_name')
        if provider and provider not in providers:
            providers.append(provider)
    return {
        'requests': len(calls),
        'failed_requests': sum(1 for call in calls if not call.get('ok')),
        'inference_time': round(total('inference_time'), 3),
        'generation_time_ms': round(total('generation_time_ms'), 1),
        'cost': total('cost'),
        'prompt_tokens': total('prompt_tokens'),
        'cached_tokens': total('cached_tokens'),
        'reasoning_tokens': total('reasoning_tokens'),
        'completion_tokens': total('completion_tokens'),
        'total_tokens': total('total_tokens'),
        'providers': providers,
        'calls': calls,
    }

def query_llm(prompt, model, tokenizer, client=None, temperature=0.5, max_new_tokens=128, stop=None, gen_stats=None):
    # truncate
    max_len = maxlen_map[model]
    if model in model_map:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids)
    tries = 0
    calls = []  # one record per attempt, so retries are counted and their cost is not lost
    if model in model_map:
        model = model_map[model]
    while tries < 5:
        tries += 1
        completion = None
        started = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max(max_new_tokens, 2048),
            )
            # Inference time is the wall clock of the model call only, measured before any
            # usage bookkeeping below.
            inference_time = time.perf_counter() - started
            content = completion.choices[0].message.content or ''
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            call = {'attempt': tries, 'ok': False, 'error': str(e),
                    'inference_time': time.perf_counter() - started}
            call.update(usage_from_completion(completion) if completion is not None else {})
            calls.append(call)
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
            continue
        call = {'attempt': tries, 'ok': True, 'error': None, 'inference_time': inference_time}
        call.update(usage_from_completion(completion))
        if gen_stats is not None:
            call.update(gen_stats.fetch(call.get('generation_id')))
        calls.append(call)
        return content, calls
    else:
        print("Max tries. Failed.")
        return '', calls

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def get_pred(data, args, fout):
    model = args.model
    if "gpt" in model or "o1" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    client = OpenAI(
        base_url=URL,
        api_key=API_KEY
    )
    # One generation-stats client per worker process, so each owns its own connection pool.
    gen_stats = None if args.skip_gen_stats else GenerationStats(API_KEY)
    try:
        run_pred(data, args, fout, model, tokenizer, client, gen_stats)
    finally:
        if gen_stats is not None:
            gen_stats.close()

def run_pred(data, args, fout, model, tokenizer, client, gen_stats):
    for item in tqdm(data):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        calls = []  # every API call this question needed, CoT's second pass included
        if args.cot:
            output, item_calls = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=1024, gen_stats=gen_stats)
        else:
            output, item_calls = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128, gen_stats=gen_stats)
        calls.extend(item_calls)
        if output == '':
            print("No answer for %s after %d call(s), skipping (will be retried next run)." % (item['_id'], len(calls)))
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            output, item_calls = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128, gen_stats=gen_stats)
            calls.extend(item_calls)
            if output == '':
                print("No answer for %s after %d call(s), skipping (will be retried next run)." % (item['_id'], len(calls)))
                continue
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        item['metrics'] = summarize_calls(calls)
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_no_context.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ".jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train') # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument("--skip_gen_stats", action='store_true') # set to True to skip the OpenRouter generation lookup (drops provider and provider-side latency, keeps cost and tokens)
    args = parser.parse_args()
    main()