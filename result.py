import os, json

RESULTS_DIR = 'results'
OUT_FILE = 'result.txt'
compensated = False

ACCURACY_HEADER = ["Model", "Overall", "Easy", "Hard", "Short", "Medium", "Long"]
USAGE_HEADER = ["Model", "Provider", "Requests", "Cost($)", "Infer(s)", "Infer/q(s)",
                "Lat/q(ms)", "Gen/q(s)", "PromptTok", "ReasonTok", "CompTok", "TotalTok",
                "$/1M tok"]


def load_preds(filename):
    try:
        return json.load(open(filename, encoding='utf-8'))
    except Exception as e:
        return [json.loads(line) for line in open(filename, encoding='utf-8')]


def accuracy_row(name, pred_data):
    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    for pred in pred_data:
        acc = int(pred['judge'])
        if compensated and pred["pred"] == None:
            acc = 0.25
        if pred["difficulty"] == "easy":
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc

        if pred['length'] == "short":
            short += 1
            short_acc += acc
        elif pred['length'] == "medium":
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc

    def pct(hit, total):
        return '-' if total == 0 else str(round(100 * hit / total, 1))

    return [name,
            pct(easy_acc + hard_acc, len(pred_data)),
            pct(easy_acc, easy),
            pct(hard_acc, hard),
            pct(short_acc, short),
            pct(medium_acc, medium),
            pct(long_acc, long)]


def usage_row(name, pred_data):
    """Sum the per-item `metrics` blocks pred.py writes.

    Runs recorded before usage accounting existed carry no metrics at all, so their columns
    read '-' rather than a misleading zero.
    """
    metrics = [pred['metrics'] for pred in pred_data if isinstance(pred.get('metrics'), dict)]
    if not metrics:
        return [name] + ['-'] * (len(USAGE_HEADER) - 1)

    def total(key):
        return sum(m.get(key) or 0 for m in metrics)

    providers = []
    for m in metrics:
        for provider in m.get('providers') or []:
            if provider not in providers:
                providers.append(provider)

    cost = total('cost')
    infer = total('inference_time')
    total_tok = total('total_tokens')
    # Provider-side timings only exist when the generation lookup found the record, so they
    # read '-' when it did not -- a real zero would hide a failed lookup.
    latency = total('latency_ms')
    gen_time = total('generation_time_ms')
    return [name,
            '/'.join(providers) if providers else '-',
            str(total('requests')),
            '%.4f' % cost,
            '%.1f' % infer,
            '%.1f' % (infer / len(metrics)),
            '-' if latency == 0 else '%.0f' % (latency / len(metrics)),
            '-' if gen_time == 0 else '%.1f' % (gen_time / len(metrics) / 1000),
            str(total('prompt_tokens')),
            str(total('reasoning_tokens')),
            str(total('completion_tokens')),
            str(total_tok),
            '-' if total_tok == 0 else '%.4f' % (cost / total_tok * 1e6)]


def render_table(header, rows):
    widths = [max(len(str(cell)) for cell in column) for column in zip(header, *rows)]
    lines = []
    for row in [header, ['-' * w for w in widths]] + rows:
        lines.append('  '.join(str(cell).ljust(widths[i]) if i == 0 else str(cell).rjust(widths[i])
                               for i, cell in enumerate(row)))
    return '\n'.join(lines)


accuracy_rows, usage_rows = [], []
for file in sorted(os.listdir(RESULTS_DIR)):
    pred_data = load_preds(os.path.join(RESULTS_DIR, file))
    if not pred_data:
        continue
    name = '.'.join(file.split('.')[:-1])
    accuracy_rows.append(accuracy_row(name, pred_data))
    usage_rows.append(usage_row(name, pred_data))

if accuracy_rows:
    print('ACCURACY')
    print(render_table(ACCURACY_HEADER, accuracy_rows))
    print()
    print('USAGE')
    print(render_table(USAGE_HEADER, usage_rows))

# result.txt keeps one wide TSV row per model, accuracy columns first.
output = ['\t'.join(ACCURACY_HEADER + USAGE_HEADER[1:])]
for accuracy, usage in zip(accuracy_rows, usage_rows):
    output.append('\t'.join(accuracy + usage[1:]))
open(OUT_FILE, 'w', encoding='utf-8').write('\n'.join(output) + '\n')
