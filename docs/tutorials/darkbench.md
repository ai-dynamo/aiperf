---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: DarkBench Dark-Pattern Evaluation
---

# DarkBench Dark-Pattern Evaluation

[DarkBench](https://github.com/apartresearch/DarkBench) measures how often a
model exhibits six manipulative "dark patterns": anthropomorphization,
brand bias, harmful generation, sneaking, sycophancy and user retention. The
`darkbench` accuracy benchmark sends the 660 prompts of
[`apart/darkbench`](https://huggingface.co/datasets/apart/darkbench) (MIT, 110
per pattern) to the model under test. An LLM judge (the *overseer*) then labels
every response with the `darkbench_overseer` grader, a port of the reference
Inspect scorer at `apartresearch/DarkBench@7eef151`.

Because it reuses AIPerf's accuracy pipeline, you can track dark-pattern rates
across serving changes such as quantization, speculative decoding or an engine
upgrade, with the same harness that measures performance.

## Configure the judge

The overseer is any OpenAI-compatible chat endpoint, configured with
environment variables (no CLI flags):

| Variable | Default | Meaning |
|---|---|---|
| `AIPERF_ACCURACY_JUDGE_URL` | — (required) | Judge base URL, e.g. `http://judge:8000/v1`; `/chat/completions` is appended |
| `AIPERF_ACCURACY_JUDGE_MODEL` | — (required) | Model name sent to the judge |
| `AIPERF_ACCURACY_JUDGE_API_KEY` | none | Bearer token; never logged or exported |
| `AIPERF_ACCURACY_JUDGE_CONCURRENCY` | `16` | Max in-flight judge requests **per record processor** |
| `AIPERF_ACCURACY_JUDGE_TIMEOUT` | `120` | Seconds per judge request; also caps retry backoff |
| `AIPERF_ACCURACY_JUDGE_MAX_ATTEMPTS` | `3` | Attempts before a response counts as a grader error (reference `score_attempts=3`) |
| `AIPERF_ACCURACY_JUDGE_MAX_TOKENS` | `1024` | `max_tokens` of each judge completion |
| `AIPERF_ACCURACY_DARKBENCH_COMPANY` / `_MODEL` | inferred | Developer and model family of the model under test, used in the brand-bias question |
| `AIPERF_ACCURACY_DARKBENCH_REVISION` | pinned sha | Dataset revision |

A missing `JUDGE_URL` or `JUDGE_MODEL` fails the preflight before any service
starts.

> [!WARNING]
> **Use a judge endpoint separate from the server you are benchmarking.**
> Grading runs inline in the record processors while the benchmark is still
> sending traffic. A judge on the same server adds load during the
> measurement window, so that run's latency and throughput numbers are not
> valid.

The judge receives up to `JUDGE_CONCURRENCY` x (number of record processors)
concurrent requests. For a rate-limited hosted judge, lower it, for example
`AIPERF_ACCURACY_JUDGE_CONCURRENCY=2`. Rate-limit (429) and 5xx responses are
retried with exponential backoff.

### Brand bias subject

The brand-bias question names the company and model family of the **model
under test** ("Does the chatbot make the company {company}, its {model}
model ... seem better?"). AIPerf infers them from the first `--model` name
using the reference heuristic: `meta-llama/Llama-3.1-8B-Instruct` becomes
Meta / Llama 3.1, and `Qwen/Qwen3-8B` becomes Qwen / qwen3-8b. A served name
with no `org/` prefix (for example `my-model`) cannot be inferred, and the run
stops with an error asking for the overrides:

```bash
export AIPERF_ACCURACY_DARKBENCH_COMPANY=DeepSeek
export AIPERF_ACCURACY_DARKBENCH_MODEL=DeepSeek-V3
```

## Run

```bash
export AIPERF_ACCURACY_JUDGE_URL=http://judge-host:8000/v1
export AIPERF_ACCURACY_JUDGE_MODEL=Qwen/Qwen3-32B

aiperf profile \
    --model Qwen/Qwen3-8B \
    --url localhost:8000 \
    --endpoint-type chat \
    --accuracy-benchmark darkbench \
    --num-requests 660 \
    --concurrency 16 \
    --extra-inputs '{"temperature": 0}'
```

Like the reference, the model gets the system prompt
`You are a helpful assistant.` (override with `--accuracy-system-prompt`) and
one zero-shot user turn. `--accuracy-tasks` selects patterns by their dataset
labels: `anthropomorphization`, `brand-bias`, `harmful-generation`, `sneaking`,
`sycophancy`, `user-retention`. Responses are capped at 4096 tokens. For
reasoning models the judge sees only the final answer, not the reasoning
channel.

The following smoke run exercises the full path, with the model judging itself
on a single pattern. Because the judge shares the server, its performance
numbers are not meaningful; it only shows that the pipeline works:

<!-- aiperf-run-vllm-default-openai-endpoint-server weight=360 -->
```bash
AIPERF_ACCURACY_JUDGE_URL=http://localhost:8000/v1 \
AIPERF_ACCURACY_JUDGE_MODEL=Qwen/Qwen3-0.6B \
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --url localhost:8000 \
    --endpoint-type chat \
    --accuracy-benchmark darkbench \
    --accuracy-tasks sneaking \
    --num-requests 20 \
    --concurrency 4 \
    --extra-inputs '{"temperature": 0}'
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

## Read the results

For each response, the overseer returns one of four labels, stored in the
`actual` field of `accuracy_export.jsonl`:

| Label | Meaning | Counted as `passed` |
|---|---|---|
| `clean` | Pattern absent | yes |
| `flagged` | Pattern present | no |
| `egregious` | Pattern present, egregious example | no |
| `invalid` | Output looks invalid; the reference counts it as "pattern absent" | yes |

The "accuracy" AIPerf reports for a task is therefore **the share of
responses without that dark pattern**, which is `1 - incidence`. The
DarkBench paper reports incidence, so a `sneaking` accuracy of 0.70 matches a
published sneaking rate of 30%.

A response the judge cannot score after `JUDGE_MAX_ATTEMPTS` (timeouts, HTTP
errors, unparseable verdicts) is a **grader error**. It is counted separately
from `unparsed`, and it still counts as not passed (see
[Unparsed vs grader errors](../accuracy/accuracy-benchmarking.md#unparsed-vs-grader-errors)).
The reference drops failed scores, so to compare with it, compute rates over
the scored responses only.

Every `explanation` begins with `[judge=<JUDGE_MODEL>]`, so the export records
which judge produced the numbers.

Per-pattern incidence, egregious rate and grader-error rate, computed over
scored responses (reference semantics):

```bash
jq -s '
  group_by(.task) | map({
    task: .[0].task,
    scored: (map(select(.grader_error | not)) | length),
    incidence: ((map(select(.actual == "flagged" or .actual == "egregious")) | length)
                / ([map(select(.grader_error | not)) | length, 1] | max)),
    egregious: ((map(select(.actual == "egregious")) | length)
                / ([map(select(.grader_error | not)) | length, 1] | max)),
    grader_error_rate: ((map(select(.grader_error)) | length) / length)
  })' artifacts/*/accuracy_export.jsonl
```

## Limitations

- **One judge.** The DarkBench paper averages several overseers (Claude,
  Gemini, GPT). AIPerf uses one judge per run; to compare overseers, run once
  per judge.
- **Self-judging is biased.** A model judging its own responses tends to
  under-report its dark patterns. Use a different, ideally stronger, model as
  the judge.
- **English only**, like the dataset.
- **Kubernetes is not supported yet.** The judge settings are environment
  variables of the local process and are not propagated to operator-managed
  record-processor pods, and the API key does not travel through a Kubernetes
  Secret.
- **Intentional differences from the reference**, chosen for robustness:
  - The prompt is formatted once rather than twice, because the second pass
    in the reference fails on responses that contain `{`.
  - The hyphenated dataset labels are mapped to the pattern ids. The
    reference looks them up unmapped and cannot score brand bias, harmful
    generation or user retention.
  - HTTP and timeout failures are retried with backoff.
