<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic Code Dataset Generator

This generator creates synthetic multi-turn coding-agent traces for replaying
long-context KV-cache workloads. It models shared prompt layers, session-specific
repo context, incremental conversation growth, inter-turn delays, resets, and
optional restart continuations. The output is Mooncake-trace compatible, so the
same `dataset.jsonl` can be generated once and replayed with `aiperf profile`.

## Prefix Layers

The generator divides each session's prompt into cache-reuse layers:

- **L1**: global tools and system prompt. These blocks are identical across all
  sessions and model globally reusable KV cache.
- **L1.5**: group-shared repository instructions and context. These blocks are
  shared by sessions in the same group, but differ across groups.
- **L2**: session-specific starting context, such as initially opened files.
  These blocks are unique to a session at turn 0.
- **L3**: conversation history added after turn 0. This layer grows as the session
  continues and is unique to that session.

Probabilistic resets and forced retires end a session; the next primary session
gets fresh L2 and L3 blocks while still reusing any shared L1 and L1.5 blocks.
Restart continuations are different: they split one logical run into Session A
and Session B, and Session B carries the accumulated context and hash IDs from
Session A so cache reuse is preserved across the split.

## Turn and Session Lifecycle

There are two turn-management modes:

- **Reset-driven mode** is the default. Turn 0 is `L1 + L1.5 + sampled L2`.
  Later turns sample a delay, sample `new_tokens_per_turn`, compute cumulative
  input as `previous_input + previous_output + new_tokens`, then sample output
  tokens and extend L3 hash IDs. Sessions end by forced retire
  (`max_prompt_tokens`), probabilistic reset, or optional restart split.
- **Explicit turn-count mode** is enabled by setting `turns`. The generator
  samples a target turn count and tries to build exactly that many turns. This
  mode cannot be combined with `reset` or `restart_initial_probability`. If the
  target cannot fit before `max_prompt_tokens`, `allow_truncation` controls
  whether the generator returns a partial forced-retire session or retries up to
  `max_session_attempts`.

In reset-driven mode, probabilistic reset uses:

```text
p = base_probability * (1 + (context_scaling - 1) * input_length / max_prompt_tokens)
```

Restart splits use `restart_initial_probability` and `restart_turn_range`.
Session A ends with `restart_split`; Session B gets a new `session_id`, keeps
the same `group_id`, carries Session A's accumulated context/hash IDs, and is
marked with `is_restart` on its first JSONL row.

Generate a dataset:

```bash
aiperf synthesize agentic-code --num-sessions 1000 --output .test/
```

Use a config JSON or a prior run manifest:

```bash
aiperf synthesize agentic-code --config my-config.json --num-sessions 500
```

Validate a generated dataset:

```bash
aiperf validate mooncake-trace --input dataset.jsonl
```

If `--config` is omitted, the generator uses [default.json](configs/default.json).
You can reference the bundled config by name with `--config default`.
[spec.json](configs/spec.json) is a generated JSON Schema reference for the config
fields; it documents the API shape but is not a runnable config. Regenerate it
after model changes with:

```bash
uv run python -m aiperf.dataset.agentic_code_gen.config
```

Each run writes a timestamped directory with `dataset.jsonl`, `manifest.json`, `quality.json`, `report.html`, `cache_explorer.html`, and `simulation.html`.

## Dataset Format

`dataset.jsonl` contains one JSON object per request turn. Example rows:

```jsonl
{"session_id":"sess-a1b2c3d4e5f6","input_length":1536,"output_length":320,"hash_ids":[0,1,2],"timestamp":0.0,"group_id":4}
{"session_id":"sess-a1b2c3d4e5f6","input_length":768,"output_length":180,"hash_ids":[1000,1001],"delay":2450.3}
{"session_id":"sess-f0e9d8c7b6a5","input_length":1024,"output_length":256,"hash_ids":[0,1],"timestamp":0.0,"group_id":4,"is_restart":true}
```

Fields:

- `session_id`: logical conversation/session identifier.
- `input_length`: new input tokens for this turn. Turn 0 includes the initial
  cached prefix; later turns contain only incremental tokens since the prior turn.
- `output_length`: generated output tokens for the turn.
- `hash_ids`: KV-cache block IDs for the new input tokens. Shared IDs model cache
  reuse across sessions or restart continuations.
- `timestamp`: absolute start time in milliseconds for turn 0.
- `delay`: delay in milliseconds before a later turn in the same session.
- `group_id`: shared-prefix group, emitted on turn 0.
- `is_restart`: present on turn 0 when the session continues from an earlier split.

## Run With `aiperf profile`

Generated `dataset.jsonl` files are Mooncake-trace compatible, so you can replay them directly with `aiperf profile`:

```bash
aiperf profile \
  --model YOUR_MODEL \
  --tokenizer YOUR_MODEL \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --input-file .test/default_1000s_seed42_YYYYMMDD-HHMMSS/dataset.jsonl \
  --custom-dataset-type mooncake_trace \
  --concurrency 50 \
  --workers-max 200 \
  --streaming \
  --ui dashboard
```

For larger trace-replay jobs, use the same dataset with the usual Mooncake trace flags:

```bash
aiperf profile \
  -m nvidia/Kimi-K2.5-NVFP4 \
  --tokenizer nvidia/Kimi-K2.5-NVFP4 \
  --tokenizer-trust-remote-code \
  --url http://__DGD_NAME__-frontend:8000 \
  --input-file /model-cache/traces/agentic-code-run/dataset.jsonl \
  --artifact-dir /model-cache/perf/${EPOCH}_${JOB_NAME}/synth_mt_qps10 \
  --custom-dataset-type mooncake_trace \
  --concurrency 50 \
  --concurrency-ramp-duration 300 \
  --benchmark-duration 2400 \
  --benchmark-grace-period 120 \
  --workers-max 200 \
  --request-timeout-seconds 1200 \
  --profile-export-level records \
  --streaming \
  --extra-inputs "ignore_eos:true" \
  --record-processors 8 \
  --goodput "time_to_first_token:8000 inter_token_latency:50"
```
