---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Measure Tool-Call Correctness with BFCL
---

# Measure Tool-Call Correctness with BFCL

## Context

AIPerf's other accuracy benchmarks all grade a natural-language answer channel.
That leaves a blind spot for anyone serving a tool-calling workload: a
deployment whose tool-call output is malformed can fail every call in
production and still score 100% on MMLU, AIME and the rest.

The `bfcl_ast` benchmark closes that gap. It runs the
[Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
single-turn suite and grades each response with bfcl-eval's deterministic AST
checker — no LLM judge, minimal run-to-run variance — while the server is under
whatever load the run applies.

It answers two separate questions at once:

- **Did the model call the right function with the right arguments?** — the
  accuracy rate.
- **Did the response contain an extractable call at all?** — the `Unparsed`
  rate, which is a format-adherence measurement.

This tutorial uses BFCL's **Prompt mode**: the tool schemas are injected into
the system prompt and the model answers in plain text with a Python-style call
list. Read [What these numbers mean](#what-these-numbers-mean) before comparing
the result to a model's advertised function-calling score — they are not the
same measurement.

## Setup

BFCL needs its own optional extra:

```bash
uv pip install 'aiperf[bfcl]'
```

> **`[bfcl]` cannot be installed alongside `[accuracy]`.** `bfcl-eval` pins
> `numpy==1.26.4` while lighteval (`[accuracy]`) requires `numpy>=2`. AIPerf
> declares this as a uv conflict so each extra resolves cleanly on its own. No
> benchmark needs graders from both, so install `[bfcl]` in the environment you
> run BFCL from.

The dataset ships inside the wheel — there is nothing to download.

### Setting up the server

```bash
# Start vLLM server
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000 &
```

```bash
# Wait for server to be ready
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```

## Tutorial

### Run the benchmark

```bash
aiperf profile \
  -m Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --accuracy-benchmark bfcl_ast \
  --accuracy-tasks simple_python,multiple,parallel,irrelevance \
  --concurrency 32 \
  --extra-inputs '{"temperature": 0}'
```

Run at `temperature=0`. BFCL AST is the most reproducible of the widely used
tool-calling benchmarks, but sampling variance would otherwise sit on top of
the signal you are trying to read.

Omitting `--accuracy-tasks` evaluates the full non-live set — 1,390 problems
across `simple_python`, `simple_java`, `simple_javascript`, `multiple`,
`parallel`, `parallel_multiple` and `irrelevance`. The `live_*` categories
(real-world user-contributed schemas) are opt-in by name.

### Read the console output

Because each record is labelled with its BFCL category, the existing per-task
table breaks the run down for free:

```text
Accuracy (Overall)                 71.2%
Accuracy (simple_python)           88.0%   Unparsed  2.5%
Accuracy (multiple)                74.5%   Unparsed  4.0%
Accuracy (parallel)                52.0%   Unparsed 11.5%
Accuracy (irrelevance)             68.0%   Unparsed  0.0%
```

Read the two columns separately:

- **Accuracy** — of the responses that were gradeable, how many gave the right
  answer.
- **Unparsed** — how many responses contained no extractable call list at all.

A decoded-but-wrong call counts against accuracy and is **not** unparsed. A
rising `Unparsed` rate as concurrency increases is a formatting/serving signal,
not a capability one, and it is the number to watch when triaging a tool-call
parser.

"The right answer" depends on the category. For the AST categories it is the
right call — correct function name, parameters and values. For the
hallucination categories it is the opposite: on `irrelevance` no offered
function can answer the question, so the correct behavior is to emit **no**
call, and a prose refusal scores *correct*. (`live_relevance` inverts it once
more: there a relevant tool exists, so refusing is the failure.)

Only an *empty* answer channel is counted as unparsed on those categories —
silence is not an abstention — so a truncated generation cannot inflate them.

### Triage the failures

Every graded record in `accuracy_export.jsonl` carries a normalized failure
bucket at the front of its `explanation`:

```bash
jq -r '.explanation | split(":")[0]' artifacts/*/accuracy_export.jsonl \
  | sort | uniq -c | sort -rn
```

```text
  312 correct
   84 param_value_error
   41 wrong_tool
   28 unparsed
   19 param_type_error
    6 should_not_have_called
```

| Bucket | Meaning | Usually points at |
|---|---|---|
| `wrong_tool` | Wrong function name, or wrong number of calls | Tool selection — schema descriptions, too many similar tools |
| `param_type_error` | Right tool, wrong argument type | Schema/type coercion; `"5"` where an integer was required |
| `param_value_error` | Right tool and types, wrong or missing argument value | The dominant failure mode at scale |
| `should_not_have_called` | Emitted a call on an `irrelevance` question | Over-eagerness / hallucinated tool use |
| `should_have_called` | Emitted no call on a `live_relevance` question | Over-refusal — the mirror image of the row above |
| `unparsed` | No call list could be extracted. On the hallucination categories only an *empty* answer channel counts, since a prose refusal is a valid answer there | Output format — truncation, prose wrapping, parser issues |
| `unclassified` | An `error_type` this `bfcl-eval` version added that AIPerf does not yet bucket | An upstream version bump — worth reporting |

(`correct` is the remaining value, for a passing verdict.)

To inspect specific failures with the model's actual output:

```bash
jq -r 'select(.passed == false) | [.task, .explanation, .model_output] | @tsv' \
  artifacts/*/accuracy_export.jsonl | head
```

### Pin the version

BFCL ships its dataset *and* its AST checker in the same wheel, so the package
version determines both which questions are asked and how answers are scored.
Runs on different versions are not comparable, so AIPerf hard-checks it:

```bash
export AIPERF_ACCURACY_BFCL_VERSION_PIN=2026.3.23   # default
export AIPERF_ACCURACY_BFCL_VERSION_PIN=2025.12.17  # strict BFCL v4 leaderboard parity
export AIPERF_ACCURACY_BFCL_VERSION_PIN=any         # skip the check (no version guarantee)
```

A mismatch fails in preflight — before any service starts — and names both
versions plus the `uv pip install` command that reconciles them.

## What these numbers mean

**Report per-category numbers, not a single "BFCL score".** The public
leaderboard averages the language subcategories *unweighted* despite very
different sample sizes (`simple_python` 400 vs `simple_javascript` 50), so any
single aggregate is not comparable to it and hides which category moved.
AIPerf reports per-category natively; the `Accuracy (Overall)` row is a
micro-average over whatever categories the run included.

**Prompt mode is not a model's native function-calling score.** Here the
schemas go in the system prompt and the model replies in BFCL's Python-style
call format. A model trained against a different tool-calling template can
score far below its real function-calling ability — MCPVerse reports
Claude-4-Sonnet dropping from 62.36 to 15.10 between the two modes. Treat
`bfcl_ast` as a model-selection and serving-reliability signal, not as a
capability ceiling and not as a production gate.

**Some things are deliberately out of scope.** `exec_*` (needs an execution
sandbox and external API keys), `multi_turn_*` (graded by comparing backend
state across turns, which a stateless per-record grader cannot do), and the v4
agentic categories (`web_search_*`, `memory_*`) are rejected by name with the
reason, rather than silently dropped.

**`--accuracy-system-prompt` is rejected for this benchmark.** BFCL builds the
system prompt per problem from that problem's tool schemas; a global override
would replace it, removing the tool definitions the model is being asked to
call.

## Further reading

- [Accuracy Benchmarking](../accuracy/accuracy-benchmarking.md) — the full
  benchmark, grader and CLI reference.
- [BFCL leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
