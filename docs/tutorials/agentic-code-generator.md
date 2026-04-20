<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic Code Dataset Generator

The Agentic Code dataset generator creates synthetic multi-turn coding-agent
traces for long-context and KV-cache benchmarking. It models shared prompt
layers, session-specific repository context, incremental conversation growth,
inter-turn delays, resets, and restart continuations.

The generator writes Mooncake trace JSONL, so the output can be replayed with
the existing `mooncake_trace` custom dataset loader.

## Generate a Dataset

Create a dataset with the built-in default configuration:

```bash
aiperf synthesize agentic-code --num-sessions 1000 --output .test/
```

Each run creates a timestamped directory:

```text
.test/default_1000s_seed42_YYYYMMDD-HHMMSS/
```

The directory contains:

- `dataset.jsonl`: Mooncake-compatible trace rows.
- `manifest.json`: seed, session count, config name, and generation parameters.
- `quality.json`: target-vs-observed distribution statistics.
- `report.html`: summary dashboard for generated sessions.
- `cache_explorer.html`: KV block reuse inspection view.
- `simulation.html`: browser-based KV cache pressure simulation.

`synthesize agentic-code` validates the generated `dataset.jsonl` before it
prints the run summary. You can also validate a saved or edited trace directly:

```bash
aiperf validate mooncake-trace --input .test/default_1000s_seed42_YYYYMMDD-HHMMSS/dataset.jsonl
```

## Replay With AIPerf

Use the generated `dataset.jsonl` as a Mooncake trace:

```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --tokenizer Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --input-file .test/default_1000s_seed42_YYYYMMDD-HHMMSS/dataset.jsonl \
  --custom-dataset-type mooncake_trace \
  --concurrency 50 \
  --workers-max 200 \
  --streaming \
  --ui dashboard
```

For longer runs, use the same generated trace with the usual Mooncake replay
controls:

```bash
aiperf profile \
  --model YOUR_MODEL \
  --tokenizer YOUR_MODEL \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --input-file .test/default_1000s_seed42_YYYYMMDD-HHMMSS/dataset.jsonl \
  --custom-dataset-type mooncake_trace \
  --concurrency 50 \
  --benchmark-duration 2400 \
  --workers-max 200 \
  --streaming
```

## Dataset Format

`dataset.jsonl` contains one JSON object per request turn:

```jsonl
{"session_id":"sess-a1b2c3d4e5f6","input_length":1536,"output_length":320,"hash_ids":[0,1,2],"timestamp":0.0,"group_id":4}
{"session_id":"sess-a1b2c3d4e5f6","input_length":768,"output_length":180,"hash_ids":[1000,1001],"delay":2450.3}
```

Important fields:

- `session_id`: logical conversation identifier.
- `input_length`: new input tokens for this turn. Turn 0 includes the initial
  cached prefix; later turns contain only incremental tokens.
- `output_length`: generated output tokens for the turn.
- `hash_ids`: KV-cache block IDs for the new input tokens.
- `timestamp`: absolute start time in milliseconds for turn 0.
- `delay`: delay in milliseconds before a later turn in the same session.
- `group_id`: shared-prefix group, emitted on turn 0.
- `is_restart`: present on turn 0 when the session continues from an earlier
  split.

## Configuration

Pass a bundled config name, a config JSON path, or a prior run manifest.
Currently, the only bundled runnable config is `default`.

The default config models long coding-agent sessions with:

- `max_prompt_tokens`: `167000`.
- `block_size`: `512` tokens.
- A `32000` token global L1 prefix shared by all sessions.
- No L1.5 group-shared prefix by default (`layer1_5_tokens: 0`,
  `num_groups: 1`).
- Session-specific initial context sampled around a `15000` token mean.
- New turn input sampled around a `6000` token mean, capped at `10000`.
- Output length sampled around a `1000` token mean, capped at `1500`.
- A small reset probability that grows with context utilization.

```bash
aiperf synthesize agentic-code \
  --config default \
  --num-sessions 1000 \
  --seed 42 \
  --output .test/

aiperf synthesize agentic-code \
  --config .test/default_1000s_seed42_YYYYMMDD-HHMMSS/manifest.json \
  --num-sessions 500 \
  --output .test/
```

Use `--max-isl` and `--max-osl` for quick sequence-length overrides:

```bash
aiperf synthesize agentic-code \
  --num-sessions 1000 \
  --max-isl 262144 \
  --max-osl 10000 \
  --output .test/
```

The config schema is generated at
`src/aiperf/dataset/agentic_code_gen/configs/spec.json`.

## Related Tutorials

- [Trace Benchmarking](../benchmark-modes/trace-replay.md) - deterministic trace replay.
- [Prefix Synthesis](prefix-synthesis.md) - KV cache testing with shared prefixes.
- [Fixed Schedule](fixed-schedule.md) - timestamp-based execution.
- [Multi-Turn Conversations](multi-turn.md) - session replay and conversation state.
