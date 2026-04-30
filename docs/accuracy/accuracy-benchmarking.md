# Accuracy Benchmarking

Run accuracy evaluation alongside performance profiling using the `--accuracy-benchmark` flag.

## Quick Start

```bash
# MMLU benchmark with 5-shot prompting (chat endpoint, aligned with lighteval)
aiperf profile Qwen/Qwen2.5-1.5B-Instruct \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --accuracy-benchmark mmlu \
  --accuracy-n-shots 5 \
  --num-requests 15000 \
  --concurrency 10 \
  --extra-inputs '{"temperature": 0, "stop": ["\n"]}'
```

```bash
# AIME competition math (zero-shot, chain-of-thought, math grader)
aiperf profile Qwen/Qwen2.5-7B-Instruct \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --accuracy-benchmark aime \
  --accuracy-enable-cot \
  --num-requests 30 \
  --concurrency 10 \
  --extra-inputs '{"temperature": 0}'
```

## Available Benchmarks

| Benchmark | Default grader | Default n-shots | Source |
|---|---|---|---|
| `mmlu` | `multiple_choice` | 5 | `lighteval/mmlu` (57 subjects) |
| `aime` | `math` | 0 | `Maxwell-Jia/AIME_2024` |

## CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--accuracy-benchmark` | Benchmark name (`mmlu`, `aime`, `hellaswag`, ...) | — |
| `--accuracy-tasks` | Specific subtasks (e.g., MMLU subjects). Accepts comma-separated values (`abstract_algebra,anatomy`) or repeated flags. Omit for all. | all |
| `--accuracy-n-shots` | Few-shot example count (0–32). `None` uses the benchmark default (e.g. MMLU=5). | benchmark default |
| `--accuracy-enable-cot` | Enable chain-of-thought prompting | false |
| `--accuracy-grader` | Override default grader (`multiple_choice`, `exact_match`, ...) | auto |
| `--accuracy-system-prompt` | Custom system prompt | — |
| `--accuracy-verbose` | Show per-problem grading details | false |

## Endpoint Type: `completions` vs `chat`

Both endpoint types are supported. The choice affects prompt format and alignment with reference frameworks:

| Endpoint | Prompt format | Best for |
|----------|--------------|----------|
| `completions` | Single flat text to `/v1/completions` | Traditional MMLU evaluation |
| `chat` | Multi-turn user/assistant messages to `/v1/chat/completions` | Aligning with lighteval |

When `--endpoint-type chat` is used, MMLU few-shot examples are structured as separate user/assistant message turns (matching lighteval's `PromptManager._prepare_chat_template()`). The `completions` endpoint sends the entire prompt as a single text block.

**Temperature:** Must be explicitly set to `0` via `--extra-inputs '{"temperature": 0}'` for deterministic (greedy) decoding. Most LLM servers default to `temperature=1.0` when not specified, which introduces random sampling and causes run-to-run variance. lighteval defaults to `temperature=0` internally.

**Stop sequence:** Use `--extra-inputs '{"stop": ["\n"]}'` to match lighteval's MMLU behavior (stop at first newline). Can be combined with temperature: `--extra-inputs '{"temperature": 0, "stop": ["\n"]}'`.

**Concurrency:** Higher concurrency is faster. `--concurrency 10` or above is recommended. Minor run-to-run variance (~0.2% macro) is expected due to GPU floating-point non-determinism; this is independent of concurrency level.

**num-requests:** Set to at least the total number of benchmark problems (MMLU: 14,042 across 57 subjects).

## Examples

```bash
# Single subject, quick test
aiperf profile my-model --url http://localhost:8000 \
  --endpoint-type chat \
  --accuracy-benchmark mmlu \
  --accuracy-n-shots 5 \
  --accuracy-tasks abstract_algebra \
  --num-requests 100 \
  --concurrency 10 \
  --extra-inputs '{"temperature": 0, "stop": ["\n"]}'

# Full MMLU (57 subjects, 14042 problems)
aiperf profile my-model --url http://localhost:8000 \
  --endpoint-type chat \
  --accuracy-benchmark mmlu \
  --accuracy-n-shots 5 \
  --num-requests 15000 \
  --concurrency 50 \
  --extra-inputs '{"temperature": 0, "stop": ["\n"]}'

# Completions endpoint (traditional flat-text format)
aiperf profile my-model --url http://localhost:8000 \
  --endpoint-type completions \
  --accuracy-benchmark mmlu \
  --accuracy-n-shots 5 \
  --num-requests 15000 \
  --concurrency 50 \
  --extra-inputs '{"temperature": 0, "stop": ["\n"]}'

# AIME with explicit math grader and few-shot priming
aiperf profile my-model --url http://localhost:8000 \
  --endpoint-type chat \
  --accuracy-benchmark aime \
  --accuracy-grader math \
  --accuracy-n-shots 4 \
  --num-requests 30 \
  --concurrency 10 \
  --extra-inputs '{"temperature": 0}'
```

## Graders

| Grader | Selection rule | Coverage |
|---|---|---|
| `multiple_choice` | A/B/C/D match against gold letter (lighteval `ExactMatches`). | MMLU |
| `math` | Extract last `\boxed{...}`, fall back to "answer is X" / last number. Compare via `Fraction` parsing or normalized string equality. | AIME |
| `exact_match` | Stub. | (unused) |
| `code_execution` | Stub. | (unused) |

The `math` grader extracts the model's final answer using this priority order:

1. The contents of the **last** `\boxed{...}` in the response (the canonical MATH/AIME format).
2. The tail of an "the answer is X" / "answer: X" / "final answer X" phrase, recursively re-parsed for boxed/numeric content.
3. The last numeric literal in the response (int, decimal, or `a/b`).

Comparison is numeric when both sides parse as `fractions.Fraction` (so `\frac{1}{2}` ≡ `0.5` ≡ `1/2`), otherwise normalized string equality (whitespace-stripped, `\dfrac` / `\tfrac` / `\left` / `\right` / `\text{}` removed, `$...$` unwrapped). Symbolic equivalence (`\sqrt{2}` vs `2^{1/2}`) is **not** supported and will grade as incorrect.

When the response did not follow the `\boxed{}` instruction and a fallback extractor was used, the response is flagged `unparsed=True` in the per-record output. A correct unparsed response is still scored correct, mirroring `multiple_choice`'s convention.

## Output

Accuracy results are displayed in the console and exported to CSV:

```text
                  Accuracy Benchmark Results
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┓
┃ Task                    ┃ Correct ┃ Total ┃ Accuracy ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━┩
│ abstract_algebra        │      35 │   100 │   35.00% │
│ ...                     │     ... │   ... │      ... │
│ OVERALL                 │    8368 │ 14042 │   59.59% │
└─────────────────────────┴─────────┴───────┴──────────┘
```

CSV file: `<artifact_dir>/accuracy_results.csv`

## Architecture

```text
AccuracyDatasetLoader          → Conversation/Turn objects (dataset pipeline)
AccuracyRecordProcessor        → grades each response (record pipeline)
AccuracyResultsProcessor       → aggregates per-task accuracy (results pipeline)
AccuracyConsoleExporter         → Rich table output
AccuracyDataExporter            → CSV export
```

All components self-disable when `--accuracy-benchmark` is not set.
