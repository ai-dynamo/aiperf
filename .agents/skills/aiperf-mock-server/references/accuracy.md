<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Accuracy dataset mode (ground-truth-aware responses)

By default the mock returns arbitrary corpus text. Point it at an **accuracy dataset** and it
returns the *correct answer* — formatted for the real AIPerf grader — for a seeded fraction of
requests, the rest deliberately wrong. This drives the whole accuracy pipeline
deterministically and offline, and gives you an oracle for what the run *should* score. Source:
`rust/mock-server/src/accuracy.rs`, `RequestCtx::build` in `handlers.rs`.

Ground truth never crosses the wire in AIPerf (it lives only in the Python accuracy worker and
is stripped from everything the runner sees), so the mock loads the dataset itself and keys on
the request prompt.

```bash
# dataset.jsonl — one object per line: prompt + gold answer.
#   {"text": "Question 1: pick A, B, C, or D.", "ground_truth": "B", "task": "demo"}
./target/release/aiperf-mock-server --fast --no-tokenizer \
  --accuracy-dataset dataset.jsonl \
  --accuracy-format mmlu \
  --accuracy-correct-rate 0.5 \
  --random-seed 7
```

Drive it with the **same file** as an AIPerf `single_turn` input so the prompts line up — the
`text` field is sent verbatim and extra fields (`ground_truth`, `task`, …) are ignored by
AIPerf:

```bash
aiperf profile --url http://127.0.0.1:8000 --model gpt-4 \
  --endpoint-type chat --streaming \
  --input-file dataset.jsonl --custom-dataset-type single_turn \
  --request-count 12 --export-level raw --random-seed 7 --ui simple
```

`--fast` does NOT disable accuracy (only latency), so correctness works with instant timing.

## Flags

| Flag | Default | Effect |
|---|---|---|
| `--accuracy-dataset <path>` | — | JSONL ground-truth dataset |
| `--accuracy-format` | `passthrough` | Grader answer format (`mmlu`/`mmlu_pro`/`gsm8k`/`math`/`exact_match`/`passthrough`); per-row `format` overrides |
| `--accuracy-correct-rate` | 1.0 | Seeded fraction answered correctly (rest wrong) |
| `--accuracy-cot-rate` | 0.0 | Fraction rendered as chain-of-thought |
| `--accuracy-reasoning-field` | true | CoT in a separate `reasoning_content` field vs inline before the answer |
| `--accuracy-adversarial-rate` | 0.0 | Fraction rendered as a parser-choke shape |
| `--accuracy-match` | `substring` | Prompt-matching mode (`exact`/`exact_ci`/`substring`/`substring_ci`) |

## Dataset format (JSONL, one object per row)

- **prompt** (first present): `prompt` | `question` | `input` | `text`
- **gold** (first present): `ground_truth` | `answer` | `gold` | `target`
- optional: `task` (aliases `subject`/`category`, per-task rollup), `format` (aliases
  `benchmark`, per-row grader override), `choices` (MC option letters for wrong-answer
  selection), `match_key` (aliases `match`/`key`/`id`) — a stable fragment to match on.

## Answer formats (`--accuracy-format`; per-row `format` overrides)

Match the real graders in `src/aiperf/accuracy/graders/`:

| format | correct answer emitted |
|---|---|
| `mmlu` / `mmlu_pro` | `The answer is (B)` |
| `gsm8k` | `#### 42` |
| `math` (aliases `aime`) | `\boxed{42}` |
| `exact_match` (aliases `exact`/`hellaswag`/`bigbench`) | gold verbatim (strict, case-sensitive) |
| `passthrough` (default) | gold verbatim |

Wrong answers are grader-plausible-but-wrong: a different MC letter, `bump_number` for
gsm8k/math, `{gold}_wrong` for exact/passthrough.

## Correctness / rendering knobs

All seeded by `--random-seed` + the matched row's stable key, so a row's verdict is
deterministic and independent of arrival order and of how its prompt was wrapped on the wire.

## Matching (`--accuracy-match`)

All modes whitespace-normalize (collapse runs, trim).

- `exact` — request equals a row key exactly.
- `substring` (default) — exact, then the longest row key *contained in* the request (handles
  few-shot / system-prompt wrapping).
- `exact_ci` / `substring_ci` — case-insensitive variants.

A per-row `match_key` matches a stable fragment of a wrapped prompt — key on `q_id_4217` while
the wire prompt is a big formatted blob. The seeded verdict is derived from the matched key, so
a row's correct/wrong outcome is the same no matter how its prompt was wrapped.

## Adversarial shapes (`--accuracy-adversarial-rate`)

Reproduce real parser failures from GitHub issues, to stress-test that a run **survives** them
(a brittle parser crashes a worker): reasoning-only content (#1136), a streaming `object:null`
SSE frame before `[DONE]` (#1010), plus leading-whitespace, wrong-case, trailing-prose,
`\boxed{}` wrap, conflicting answers (take-LAST-match), and unicode. The run must exit 0 with
these on.

## Live tally — the oracle (`GET /accuracy` + `aiperf_mock_accuracy_*`)

The mock counts what it actually answered (not a pre-computed estimate):

```bash
curl -s http://127.0.0.1:8000/accuracy | python3 -m json.tool
# {"matched": 12, "correct": 6, "incorrect": 6, "accuracy": 0.5,
#  "unmatched": 0, "adversarial": 0, "cot": 0, "tasks": {"demo": {...}}}
```

The same tally is appended to the Prometheus scrape (`GET /metrics`) as `aiperf_mock_accuracy_*`:
`matched_total`, `correct_total`, `incorrect_total`, `unmatched_total`, `adversarial_total`,
`cot_total`, a `ratio` gauge, and per-task `aiperf_mock_accuracy_task_{matched_total,correct_total,ratio}{task="…"}`.
Compare either against what AIPerf's own grader reports for the run.

## e2e recipes (`test_accuracy_mock.rs`)

All use `--fast --no-tokenizer --random-seed 7 --accuracy-dataset <file> --accuracy-format mmlu`
on the mock and the profile shape:

```bash
aiperf profile --model gpt-4 --url http://127.0.0.1:8000 --endpoint-type chat --streaming \
  --input-file dataset.jsonl --custom-dataset-type single_turn \
  --request-count N --concurrency C --workers-max 1 \
  --random-seed 7 --export-level raw --ui simple
```

- **Correct ground truth** (`--accuracy-correct-rate 1.0`, N=6): every streamed `content ==
  "The answer is (B)"`.
- **Seeded split** (`--accuracy-correct-rate 0.5`, N=24): every content starts `"The answer is ("`;
  count of exact `"The answer is (B)"` lands in `5..=19`.
- **CoT separate field** (`--accuracy-cot-rate 1.0 --accuracy-reasoning-field true`): `content ==
  "The answer is (B)"`, `reasoning_content` contains it.
- **`match_key` fragment**: dataset rows key on `q_id_{i}` embedded in a bigger `text`; each
  `content == "The answer is (B)"`.
- **Live endpoint matches raw records** (`--accuracy-correct-rate 0.5`, N=24): `GET /accuracy`
  (`.no_proxy()`) yields `matched == records.len()` and `correct ==` the count of raw records
  whose content is `"The answer is (B)"`.
- **Adversarial survives** (`--accuracy-adversarial-rate 1.0`, and `fast=false ttft=0 itl=0` to
  hit the real streaming/null-object path): the run succeeds and produces all records.

### Verifying at the raw-record level (the definition of done)

With `--export-level raw`, reconstruct each response's content from `profile_export_raw.jsonl`
(`responses[].packets[].value` → SSE `choices[0].delta.content`, concatenated) and assert it
equals the formatted answer; `reasoning_content` deltas carry CoT. The `/accuracy`
`correct`/`matched` equal the raw-record counts exactly for a single-pass run with no warmup.
Worked example + assertions: `rust/e2e/tests/test_accuracy_mock.rs`.
