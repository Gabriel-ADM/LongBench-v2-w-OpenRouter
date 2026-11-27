# **Gemini Prompt-Compression Benchmark**

1. Proposed architecture
Input dataset

Google Sheets / JSON / Webhook / CSV


JavaScript Benchmark Runner

Node.js + Playwright or official Gemini API

Original prompt
Baseline

Compressed prompts
Independent variants



Gemini execution

One execution per prompt variant and test case


Measurement and evaluation

Latency, token count, output, errors, compression ratio, quality


Results storage

Google Sheets / JSON / Database / Webhook


2. Experiment structure
Each execution should generate an independent record.
Field	Description
experiment_id	Unique experiment identifier
test_case_id	Input question identifier
prompt_id	Original prompt identifier
variant_id	Compression variant
compression_method	Compression technique
original_prompt	Original prompt
modified_prompt	Compressed prompt
input_data	Context and question
started_at	Execution start timestamp
finished_at	Execution completion timestamp
latency_ms	Processing duration
response	Gemini response
error	Execution error
input_tokens	Input token count, if available
output_tokens	Output token count, if available
evaluation_score	Quality evaluation

The LongBench methodology you provided is a useful starting point: it separates input, context, answers, dataset and length, and uses automated prediction and evaluation stages.

The implementation will use **Node.js + Playwright + JSONL**, with an existing Google account authenticated in a persistent Chromium browser profile.

The benchmark will:

1. Load multiple test cases from a JSON dataset.
2. Generate independent prompt variations using different compression methods.
3. Execute the original prompt and every compressed variation in Gemini.
4. Measure execution time and record timestamps.
5. Capture Gemini responses and execution errors.
6. Save every execution independently in JSONL.
7. Support resuming interrupted batches without repeating completed executions.

**Important:** This implementation automates the Gemini website, not AI Studio. Your screenshot indicates that AI Studio access is disabled for your Workspace account. The implementation does not bypass Workspace restrictions, login controls, or usage limits. It requires an account that can access Gemini normally.

## **1. Project structure**

```
./
├── package.json
├── .env
├── .gitignore
├── data/
│   └── dataset.json
├── results/
│   └── executions.jsonl
├── browser-profile/
└── src/
    ├── index.js
    ├── browser.js
    ├── benchmark.js
    ├── compressors.js
    ├── dataset.js
    └── utils.js
```

## **2. Install dependencies**

`package.json`

```
{
  "name": "browser-prompt-benchmark",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "benchmark": "node src/index.js",
    "benchmark:headed": "HEADLESS=false node src/index.js"
  },
  "dependencies": {
    "dotenv": "^16.4.7",
    "playwright": "^1.51.0"
  }
}
```

Install:

```
npm install
npx playwright install chromium
```

`.env`

```
HEADLESS=false

GEMINI_URL=https://gemini.google.com/app

BROWSER_PROFILE=./browser-profile

DATASET_PATH=./data/dataset.json
RESULTS_PATH=./results/executions.jsonl

DELAY_BETWEEN_REQUESTS=3000
MAX_RETRIES=2

# Keep browser execution sequential.
CONCURRENCY=1
```

`.gitignore`

```
node_modules/
.env
browser-profile/
results/
```

## **3. Dataset format**

The dataset follows the separation between task input, context and expected answers used in LongBench, while adding an instruction field for the compression experiment.&#x20;

Pasted markdown.md



`data/dataset.json`

```
[
  {
    "id": "qa-001",
    "dataset": "custom-qa",
    "language": "en",
    "instruction": "You are an expert assistant. Read the provided context carefully and answer the question using only the information available in the context. Provide a concise and accurate answer.",
    "context": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed between 1887 and 1889 and was initially criticized by some artists and intellectuals.",
    "question": "When was the Eiffel Tower constructed?",
    "answers": [
      "Between 1887 and 1889"
    ]
  },
  {
    "id": "qa-002",
    "dataset": "custom-qa",
    "language": "en",
    "instruction": "You are an expert assistant. Analyze the context and answer the question accurately. Do not introduce information that is not supported by the context.",
    "context": "HTTP is an application-layer protocol used for communication between clients and servers. HTTPS adds encryption through TLS.",
    "question": "What does HTTPS add to HTTP?",
    "answers": [
      "Encryption through TLS"
    ]
  }
]
```

The `instruction` is compressed independently. The context and question remain unchanged across all variants. This prevents the experiment from unintentionally mixing prompt compression with context compression.

## **4. Compression methods**

Each compression method receives the original instruction independently. No method receives another method's output.

`src/compressors.js`

```

/**
 * Independent prompt compression strategies.
 *
 * These are deterministic transformations, not LLM-generated
 * paraphrases. Their actual semantic preservation must be evaluated.
 */

function normalizeWhitespace(text) {
  return text
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

// Method 1: Remove redundant expressions.
function removeRedundancy(text) {
  let result = text;

  const replacements = [
    [/\bcarefully\b/gi, ""],
    [/\bPlease\b/gi, ""],
    [/\bprovided by the user\b/gi, "provided"],
    [/\binformation available in the context\b/gi, "context"],
    [/\busing only the information\b/gi, "using"],
    [/\bDo not introduce information that is not supported by\b/gi, "Do not add unsupported information beyond"],
    [/\bProvide a concise and accurate answer\b/gi, "Answer accurately and concisely"]
  ];

  for (const [pattern, replacement] of replacements) {
    result = result.replace(pattern, replacement);
  }

  return normalizeWhitespace(result);
}

// Method 2: Concise instruction rewriting.
function conciseRewrite(text) {
  let result = text;

  const replacements = [
    [/\bYou are an expert assistant\./gi, "Act as an expert."],
    [/\bRead the provided context carefully and\b/gi, "Read the context and"],
    [/\banswer the question using only\b/gi, "answer using only"],
    [/\bthe information available in the context\b/gi, "context information"],
    [/\bAnalyze the context and\b/gi, "Analyze and"],
    [/\bProvide a concise and accurate answer\./gi, "Be concise and accurate."],
    [/\bDo not introduce information that is not supported by the context\./gi, "Do not add unsupported information."]
  ];

  for (const [pattern, replacement] of replacements) {
    result = result.replace(pattern, replacement);
  }

  return normalizeWhitespace(result);
}

// Method 3: Structured / imperative compression.
function imperativeCompression(text) {
  const lower = text.toLowerCase();

  const rules = [];

  if (lower.includes("expert assistant")) {
    rules.push("Role: Expert assistant.");
  }

  if (lower.includes("context")) {
    rules.push("Use the supplied context.");
  }

  if (
    lower.includes("only") ||
    lower.includes("supported")
  ) {
    rules.push("Do not infer unsupported facts.");
  }

  if (
    lower.includes("concise") ||
    lower.includes("accurate")
  ) {
    rules.push("Answer accurately and concisely.");
  } else {
    rules.push("Answer the question.");
  }

  return rules.join(" ");
}

// Method 4: Keyword-oriented compression.
function keywordCompression(text) {
  const lower = text.toLowerCase();
  const rules = [];

  if (lower.includes("expert")) {
    rules.push("Expert assistant.");
  }

  if (lower.includes("context")) {
    rules.push("Context-grounded.");
  }

  if (lower.includes("only")) {
    rules.push("Use context only.");
  }

  if (
    lower.includes("accurate") ||
    lower.includes("supported")
  ) {
    rules.push("No unsupported claims.");
  }

  if (lower.includes("concise")) {
    rules.push("Concise answer.");
  }

  return rules.join(" ");
}

export const compressionMethods = [
  {
    id: "baseline",
    name: "Original prompt",
    compress: (text) => text
  },
  {
    id: "redundancy",
    name: "Redundancy removal",
    compress: removeRedundancy
  },
  {
    id: "concise",
    name: "Concise rewriting",
    compress: conciseRewrite
  },
  {
    id: "imperative",
    name: "Imperative compression",
    compress: imperativeCompression
  },
  {
    id: "keyword",
    name: "Keyword compression",
    compress: keywordCompression
  }
];

export function generateVariants(instruction) {
  return compressionMethods.map(method => ({
    variant_id: method.id,
    compression_method: method.name,
    original_instruction: instruction,
    modified_instruction: method.compress(instruction)
  }));
}

```

This produces five independent experimental conditions:

| Variant    | Method                             |
| ---------- | ---------------------------------- |
| baseline   | Original prompt                    |
| redundancy | Redundant expression removal       |
| concise    | Concise rewriting                  |
| imperative | Structured imperative instructions |
| keyword    | Keyword-oriented compression       |

These methods are intentionally different. They are experimental transformations, not guaranteed semantics-preserving compressors.

## **5. Dataset loader**

`src/dataset.js`

```

import fs from "node:fs/promises";

export async function loadDataset(path) {
  const raw = await fs.readFile(path, "utf8");
  const dataset = JSON.parse(raw);

  if (!Array.isArray(dataset)) {
    throw new Error("Dataset must be a JSON array.");
  }

  for (const item of dataset) {
    if (
      !item.id ||
      !item.instruction ||
      typeof item.instruction !== "string" ||
      typeof item.question !== "string"
    ) {
      throw new Error(
        `Invalid dataset item: ${item.id ?? "unknown"}`
      );
    }

    if (typeof item.context !== "string") {
      throw new Error(
        `Missing context in item: ${item.id}`
      );
    }
  }

  return dataset;
}

export function buildPrompt(item, instruction) {
  return `${instruction}

CONTEXT:
${item.context}

QUESTION:
${item.question}

ANSWER:`;
}

```

## **6. Browser automation**

`src/browser.js`

This module uses a persistent browser profile. The first execution opens Gemini so you can log in manually.

It does not store Google credentials in the project.

```

import { chromium } from "playwright";

const sleep = ms =>
  new Promise(resolve => setTimeout(resolve, ms));

const GEMINI_URL =
  process.env.GEMINI_URL ||
  "https://gemini.google.com/app";

const PROFILE =
  process.env.BROWSER_PROFILE ||
  "./browser-profile";

export async function launchBrowser() {
  const context = await chromium.launchPersistentContext(
    PROFILE,
    {
      headless: process.env.HEADLESS === "true",
      viewport: {
        width: 1440,
        height: 1000
      },
      args: [
        "--disable-blink-features=AutomationControlled"
      ]
    }
  );

  const page = context.pages()[0] ??
    await context.newPage();

  await page.goto(GEMINI_URL, {
    waitUntil: "domcontentloaded"
  });

  console.log("Gemini opened.");

  console.log(
    "If necessary, log in manually in the browser."
  );

  await page.waitForURL(
    url => url.hostname === "gemini.google.com",
    { timeout: 0 }
  );

  return { context, page };
}

async function findEditor(page) {
  const selectors = [
    'div[contenteditable="true"][role="textbox"]',
    'rich-textarea [contenteditable="true"]',
    'div[contenteditable="true"]',
    'textarea'
  ];

  for (const selector of selectors) {
    const locator = page.locator(selector).last();

    if (await locator.count() > 0) {
      if (await locator.isVisible()) {
        return locator;
      }
    }
  }

  throw new Error(
    "Gemini input editor not found. Update browser.js selectors."
  );
}

async function findSendButton(page) {
  const selectors = [
    'button[aria-label*="Send"]',
    'button[aria-label*="send"]',
    'button[data-testid*="send"]'
  ];

  for (const selector of selectors) {
    const locator = page.locator(selector).last();

    if (
      await locator.count() > 0 &&
      await locator.isVisible() &&
      await locator.isEnabled()
    ) {
      return locator;
    }
  }

  return null;
}

function responseLocator(page) {
  return page.locator("model-response");
}

async function getLatestResponse(page) {
  const responses = responseLocator(page);
  const count = await responses.count();

  if (count === 0) {
    return "";
  }

  const latest = responses.last();

  return (await latest.innerText()).trim();
}

export async function sendPrompt(page, prompt) {
  const editor = await findEditor(page);

  const previousResponse = await getLatestResponse(page);

  await editor.click();
  await editor.fill(prompt);

  const sendButton = await findSendButton(page);

  if (sendButton) {
    await sendButton.click();
  } else {
    await editor.press("Enter");
  }

  const startedAt = new Date();
  const startTime = performance.now();

  let latestResponse = "";
  let previousText = "";
  let stableCount = 0;

  const timeoutMs = 180000;

  while (
    performance.now() - startTime < timeoutMs
  ) {
    await sleep(1000);

    const current = await getLatestResponse(page);

    if (
      current &&
      current !== previousResponse &&
      current === previousText
    ) {
      stableCount++;
    } else {
      stableCount = 0;
    }

    if (current) {
      latestResponse = current;
    }

    previousText = current;

    // Require a stable response before treating it as complete.
    if (stableCount >= 3) {
      break;
    }
  }

  if (!latestResponse) {
    throw new Error(
      "No Gemini response detected before timeout."
    );
  }

  const finishedAt = new Date();
  const latencyMs = Math.round(
    performance.now() - startTime
  );

  return {
    response: latestResponse,
    started_at: startedAt.toISOString(),
    finished_at: finishedAt.toISOString(),
    latency_ms: latencyMs
  };
}

```

**Browser selector note:** Gemini changes its frontend periodically. The editor, send-button and response selectors are isolated so they can be adjusted without modifying the benchmark engine. The response-stability check is a basic completion heuristic, not a guaranteed signal from Gemini's backend.

## **7. Benchmark execution engine**

`src/benchmark.js`

This module handles:

- Independent prompt variations.
- Sequential batch execution.
- Retry handling.
- Execution timestamps.
- Latency measurements.
- JSONL persistence.
- Resuming interrupted experiments.

```

import fs from "node:fs/promises";

import {
  generateVariants
} from "./compressors.js";

import {
  buildPrompt
} from "./dataset.js";

import {
  sendPrompt
} from "./browser.js";

import {
  createHash,
  randomUUID
} from "node:crypto";

import {
  sleep,
  countWords,
  sha256
} from "./utils.js";

const RESULTS_PATH =
  process.env.RESULTS_PATH ||
  "./results/executions.jsonl";

const DELAY = Number(
  process.env.DELAY_BETWEEN_REQUESTS || 3000
);

const MAX_RETRIES = Number(
  process.env.MAX_RETRIES || 2
);

async function appendResult(result) {
  await fs.appendFile(
    RESULTS_PATH,
    JSON.stringify(result) + "\n",
    "utf8"
  );
}

async function loadCompleted() {
  try {
    const content = await fs.readFile(
      RESULTS_PATH,
      "utf8"
    );

    return new Set(
      content
        .split("\n")
        .filter(Boolean)
        .flatMap(line => {
          try {
            const record = JSON.parse(line);

            return record.status === "success"
              ? [record.execution_key]
              : [];
          } catch {
            return [];
          }
        })
    );
  } catch (error) {
    if (error.code === "ENOENT") {
      return new Set();
    }

    throw error;
  }
}

function makeExecutionKey(
  experimentId,
  itemId,
  variantId,
  promptHash
) {
  return sha256(
    [
      experimentId,
      itemId,
      variantId,
      promptHash
    ].join(":")
  );
}

async function executeWithRetry(page, prompt) {
  let lastError;

  for (
    let attempt = 1;
    attempt <= MAX_RETRIES + 1;
    attempt++
  ) {
    try {
      const result = await sendPrompt(page, prompt);

      return {
        ...result,
        attempts: attempt
      };
    } catch (error) {
      lastError = error;

      if (attempt <= MAX_RETRIES) {
        await sleep(attempt * 5000);
      }
    }
  }

  throw lastError;
}

export async function runBenchmark({
  page,
  dataset,
  experimentId
}) {
  await fs.mkdir("./results", {
    recursive: true
  });

  const completed = await loadCompleted();

  let total = 0;
  let skipped = 0;
  let successful = 0;
  let failed = 0;

  for (const item of dataset) {
    const variants = generateVariants(
      item.instruction
    );

    for (const variant of variants) {
      total++;

      const prompt = buildPrompt(
        item,
        variant.modified_instruction
      );

      const promptHash = sha256(prompt);

      const executionKey = makeExecutionKey(
        experimentId,
        item.id,
        variant.variant_id,
        promptHash
      );

      if (completed.has(executionKey)) {
        skipped++;
        console.log(
          `SKIP ${item.id} / ${variant.variant_id}`
        );
        continue;
      }

      const executionId = randomUUID();

      console.log(
        `RUN ${item.id} / ${variant.variant_id}`
      );

      const experimentStart = new Date();

      try {
        const result = await executeWithRetry(
          page,
          prompt
        );

        const record = {
          schema_version: "1.0",
          execution_id: executionId,
          execution_key: executionKey,
          experiment_id: experimentId,

          test_case_id: item.id,
          dataset: item.dataset ?? null,
          language: item.language ?? null,

          variant_id: variant.variant_id,
          compression_method:
            variant.compression_method,

          original_instruction:
            variant.original_instruction,

          modified_instruction:
            variant.modified_instruction,

          original_instruction_chars:
            variant.original_instruction.length,

          modified_instruction_chars:
            variant.modified_instruction.length,

          original_instruction_words:
            countWords(variant.original_instruction),

          modified_instruction_words:
            countWords(variant.modified_instruction),

          prompt_chars: prompt.length,
          prompt_words: countWords(prompt),
          prompt_sha256: promptHash,

          context_chars: item.context.length,
          question: item.question,

          started_at: result.started_at,
          finished_at: result.finished_at,
          latency_ms: result.latency_ms,

          attempts: result.attempts,

          response: result.response,

          response_chars: result.response.length,
          response_words: countWords(result.response),

          answers: item.answers ?? [],

          status: "success",
          error: null
        };

        await appendResult(record);

        successful++;

        console.log(
          `SUCCESS ${item.id} | ${variant.variant_id} | ${result.latency_ms}ms`
        );

      } catch (error) {
        const record = {
          schema_version: "1.0",
          execution_id: executionId,
          execution_key: executionKey,
          experiment_id: experimentId,

          test_case_id: item.id,

          variant_id: variant.variant_id,
          compression_method:
            variant.compression_method,

          prompt_sha256: promptHash,

          started_at: experimentStart.toISOString(),
          finished_at: new Date().toISOString(),

          latency_ms: null,

          status: "error",

          error: {
            message: error.message,
            name: error.name
          }
        };

        await appendResult(record);

        failed++;

        console.error(
          `FAILED ${item.id} / ${variant.variant_id}`,
          error.message
        );

        // Stop if the browser session is no longer usable.
        if (page.isClosed()) {
          throw error;
        }
      }

      await sleep(DELAY);
    }
  }

  return {
    total,
    successful,
    failed,
    skipped
  };
}

```

## **8. Utility functions**

`src/utils.js`

```

import { createHash } from "node:crypto";

export function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export function countWords(text) {
  return text.trim()
    ? text.trim().split(/\s+/).length
    : 0;
}

export function sha256(text) {
  return createHash("sha256")
    .update(text)
    .digest("hex");
}

```

## **9. Main entry point**

`src/index.js`

```

import "dotenv/config";

import {
  launchBrowser
} from "./browser.js";

import {
  loadDataset
} from "./dataset.js";

import {
  runBenchmark
} from "./benchmark.js";

import {
  randomUUID
} from "node:crypto";

import fs from "node:fs/promises";

async function main() {
  const datasetPath =
    process.env.DATASET_PATH ||
    "./data/dataset.json";

  const dataset = await loadDataset(datasetPath);

  const { context, page } = await launchBrowser();

  const experimentId = randomUUID();

  console.log("\nBenchmark started");
  console.log("Experiment:", experimentId);
  console.log("Dataset size:", dataset.length);

  const startedAt = new Date();

  try {
    const summary = await runBenchmark({
      page,
      dataset,
      experimentId
    });

    const finishedAt = new Date();

    const report = {
      experiment_id: experimentId,

      started_at: startedAt.toISOString(),
      finished_at: finishedAt.toISOString(),

      duration_ms:
        finishedAt.getTime() - startedAt.getTime(),

      dataset_size: dataset.length,

      variants_per_case: 5,

      summary
    };

    await fs.writeFile(
      `./results/experiment-${experimentId}.json`,
      JSON.stringify(report, null, 2),
      "utf8"
    );

    console.log("\nBenchmark finished");
    console.table(summary);

  } finally {
    await context.close();
  }
}

main().catch(error => {
  console.error("Benchmark error:", error);
  process.exitCode = 1;
});

```

## **10. Execution**

Start the benchmark:

```
npm run benchmark
```

On the first execution:

1. Chromium opens.
2. Log in to Gemini manually.
3. Complete any required account verification yourself.
4. Open the Gemini application.
5. Leave the browser running.
6. The benchmark starts executing the dataset.

For subsequent executions, the persistent profile retains the browser session, subject to Google's authentication requirements.

Results:

```
results/
├── executions.jsonl
├── experiment-UUID-1.json
└── experiment-UUID-2.json
```

## **11. Experimental results format**

Each JSONL line represents one execution.

Example:

```
{
  "schema_version": "1.0",
  "execution_id": "f17b2b6e-...",
  "experiment_id": "exp-001",
  "test_case_id": "qa-001",
  "variant_id": "concise",
  "compression_method": "Concise rewriting",
  "original_instruction_chars": 205,
  "modified_instruction_chars": 128,
  "original_instruction_words": 34,
  "modified_instruction_words": 21,
  "prompt_chars": 340,
  "prompt_words": 53,
  "latency_ms": 4230,
  "response": "The Eiffel Tower was constructed between 1887 and 1889.",
  "response_chars": 60,
  "response_words": 10,
  "status": "success",
  "error": null
}
```

## **12. Important measurement improvements**

There are three adjustments I recommend making before running a larger benchmark.

**A. Correct response detection and latency measurement**

In `browser.js`, replace `sendPrompt` with this version. It records the start before submitting the prompt and waits for a newly created Gemini response rather than comparing only its text with the previous response.

```

export async function sendPrompt(page, prompt) {
  const editor = await findEditor(page);
  const responses = responseLocator(page);

  const previousCount = await responses.count();

  await editor.click();
  await editor.fill(prompt);

  const startedAt = new Date();
  const startTime = performance.now();

  const sendButton = await findSendButton(page);

  if (sendButton) {
    await sendButton.click();
  } else {
    await editor.press("Enter");
  }

  const timeoutMs = 180000;

  let latestResponse = "";
  let stableCount = 0;
  let previousText = "";
  let newResponseDetected = false;

  while (
    performance.now() - startTime < timeoutMs
  ) {
    await sleep(1000);

    const count = await responses.count();

    if (count > previousCount) {
      newResponseDetected = true;
    }

    if (newResponseDetected) {
      latestResponse = await getLatestResponse(page);

      if (
        latestResponse &&
        latestResponse === previousText
      ) {
        stableCount++;
      } else {
        stableCount = 0;
      }

      previousText = latestResponse;
    }

    if (stableCount >= 3) {
      break;
    }
  }

  if (!newResponseDetected || !latestResponse) {
    throw new Error(
      "Gemini response timeout or response element not detected."
    );
  }

  const finishedAt = new Date();

  return {
    response: latestResponse,
    started_at: startedAt.toISOString(),
    finished_at: finishedAt.toISOString(),
    latency_ms: Math.round(
      performance.now() - startTime
    )
  };
}

```

This measures browser-observed end-to-end latency, not Gemini's internal model inference time.

**B. Make experiment resumption reliable**

Currently, `index.js` creates a new experiment ID every time it starts. To resume the same experiment after interruption, replace:

```
const experimentId = randomUUID();
```

with:

```
const experimentId =
  process.env.EXPERIMENT_ID || randomUUID();
```

Then set a stable ID in `.env`:

```
EXPERIMENT_ID=prompt-compression-study-001
```

Use that same ID when restarting an interrupted batch. Use a new ID for a new experimental run.

**C. Distinguish latency from processing time**

The browser cannot directly expose Gemini's backend inference duration. Record the following separately when available:

- `latency_ms`: time from submission to observed response completion.
- `started_at`: client-side submission timestamp.
- `finished_at`: client-side response completion timestamp.
- `input_tokens`: unavailable through this browser-only implementation unless exposed by Gemini.
- `output_tokens`: likewise.
- `estimated_input_tokens`: optional tokenizer-based estimate, explicitly distinguished from actual usage.

## **13. Expected benchmark volume**

With five variants per test case:

**Execution calculator**

Number of test cases

**100**

Prompt variants

**5**

Repetitions per variant

**3**

Total Gemini submissions

# **1,500**

Assumes all executions are successful. This is a workload estimate, not an estimate of completion time or permitted account usage.

## **14. Experimental validity**

For a meaningful comparison, keep these conditions constant:

| Variable             | Control                    |
| -------------------- | -------------------------- |
| Gemini account       | Same account               |
| Gemini model         | Same selected model        |
| Context              | Identical                  |
| Question             | Identical                  |
| Original instruction | Identical                  |
| Compression          | Independent variants       |
| Temperature          | Record if configurable     |
| Repetitions          | Same number per variant    |
| Execution order      | Randomized or balanced     |
| Session              | Record session information |

One additional recommendation for the research: evaluate response quality separately from latency. A shorter prompt that produces a faster but incorrect answer should not automatically be considered a better compression method.

The current implementation measures execution and stores the original reference answers. It does not yet calculate LongBench task-specific evaluation scores or semantic equivalence.

**Implementation status:** The code above provides the browser runner, dataset handling, independent compression strategies, batch processing, retries and JSONL logging. Gemini frontend selectors and response-completion behavior should be validated with a small batch before scaling up.