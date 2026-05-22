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
# AIME competition math — defaults match the trt-llm benchmark recipe
# (8-shot, chain-of-thought on, sympy-backed math grader)
aiperf profile Qwen/Qwen2.5-7B-Instruct \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --accuracy-benchmark aime \
  --num-requests 30 \
  --concurrency 10 \
  --extra-inputs '{"temperature": 0}'
```

## trt-llm reference alignment

The `aime` benchmark is aligned with the trt-llm benchmark recipe's
DeepEval-backed AIME path
(`trt-llm-benchmark-recipe/src/accuracy/aime/`):

- **Dataset:** `Maxwell-Jia/AIME_2024`, train split.
- **Defaults:** `n_shots=8`, `enable_cot=True` (the recipe enforces
  `n_shots <= 8` and aiperf raises `ValueError` if you exceed it).
- **Prompt format:** byte-equal to `AIMETemplate.generate_output` —
  `**Problem**: ... **Solution**: ... **Answer**: ...` blocks for
  few-shots (Solution only when CoT is on), trailing
  `Let's think step-by-step.` after the final `**Answer**:`.
- **System prompt (auto-injected):**
  `"Please reason step by step, and put your final answer within \\boxed{}."`
  This default lives in `plugins.yaml` under the `aime` benchmark's
  `default_system_prompt` metadata. Override it with
  `--accuracy-system-prompt 'your prompt here'`. Pass `--accuracy-system-prompt ''`
  to disable injection.
- **Grader:** `MathGrader` with `_math_strip.strip_string` + sympy/
  latex2sympy2-extended `math_equal`. Requires the `[accuracy]` extra:
  `uv pip install 'aiperf[accuracy]'`. Without those packages installed
  the grader falls back to a stdlib normalize+Fraction comparison and
  emits a one-time warning; reference parity is only achieved with the
  full sympy stack.

### Per-benchmark default system prompts

| Benchmark | `default_system_prompt` |
|---|---|
| `aime` | `Please reason step by step, and put your final answer within \boxed{}.` |
| (others) | _none — pass via `--accuracy-system-prompt` if desired_ |

The CLI's `--accuracy-system-prompt` flag always wins; the per-benchmark
default is only consulted when the flag is unset. An empty-string default
in metadata is treated as no default (aiperf doesn't inject a zero-length
system message).

## Available Benchmarks

| Benchmark | Default grader | Default n-shots | Source |
|---|---|---|---|
| `mmlu` | `multiple_choice` | 5 | `lighteval/mmlu` (57 subjects) |
| `aime` | `math` | 8 | `Maxwell-Jia/AIME_2024` (trt-llm reference, 8-shot CoT) |
| `hellaswag` | `exact_match` | 10 | `Rowan/hellaswag` (trt-llm/DeepEval reference; one few-shot per unique activity_label) |
| `bigbench` | `exact_match` | 3 | `lukaemon/bbh` (trt-llm/DeepEval reference; 27 subtasks, canonical CoT/non-CoT prompt files) |

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
| `math` | Extract last `\boxed{...}`, fall back to "answer is X" / last number. Apply trt-llm `strip_string` normalization, then compare via `math_equal` (lowercase string → numeric `isclose` → symbolic equivalence via sympy + latex2sympy2-extended). | AIME |
| `exact_match` | Stub. | (unused) |
| `code_execution` | Stub. | (unused) |

The `math` grader pipeline (aligned with `trt-llm-benchmark-recipe/src/accuracy/aime/`):

1. **Extract** the model's final answer by priority:
    - The contents of the **last** `\boxed{...}` in the response (canonical MATH/AIME format).
    - The tail of an "the answer is X" / "answer: X" / "final answer X" phrase, recursively re-parsed for boxed/numeric content.
    - The last numeric literal in the response.
2. **Normalize** both prediction and gold via the recipe's `strip_string`: linebreaks/spacing/quote-style braces collapsed, `\dfrac`/`\tfrac` → `\frac`, `\left`/`\right` removed, `\text{...}` unwrapped, MathQA-derived unit tokens dropped, infinity/percent/months/dollar-sign normalization, trailing `.0` decimals trimmed, simple `a/b` rewritten as `\frac{a}{b}`.
3. **Compare** with `math_equal` (lowercase string equality → choice-prefix unwrap → numerical `isclose` (abs_tol=1e-4) with percentage variants → brace/paren strip + lowercase compare → equation-form rewrite (`f(x) = y` ↔ `y`) → symbolic equivalence via `sympy.parsing.sympy_parser.parse_expr` and `latex2sympy2_extended.latex2sympy`).

Symbolic equivalence (e.g. `\sqrt{2}` ↔ `2^{1/2}`, `\frac{1}{3}` ↔ `0.333333`, `1,2,3` ↔ `3,2,1`) requires the `[accuracy]` install:

```bash
uv pip install 'aiperf[accuracy]'
```

Without those optional dependencies (`sympy`, `latex2sympy2-extended`) the grader falls back to a stdlib normalize + `Fraction` comparison and emits a single warning the first time it runs. Reference parity with the trt-llm recipe requires the full sympy stack.

When extraction fell back past the `\boxed{}` step (i.e. the model didn't follow the boxed-answer instruction), the response is flagged `unparsed=True` in the per-record output. A correct unparsed response is still scored correct, mirroring `multiple_choice`'s convention.

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
