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

## Prefix Layers

Agentic Code traces divide each session's prompt into cache-reuse layers:

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

## Turns, Resets, and Restarts

The generator has two turn-management modes.

### Reset-Driven Mode

Reset-driven mode is the default. The generator does not choose a fixed turn
count up front. Instead, each session grows turn by turn until one of the end
conditions fires.

Turn construction works as follows:

1. Turn 0 samples the initial context: `L1 + L1.5 + sampled L2`.
2. Turn 0 has `delay_ms = 0` and `timestamp_ms = 0`.
3. Later turns sample an inter-turn delay from the agentic/human delay mixture.
4. Later turns sample `new_tokens_per_turn`.
5. The cumulative in-memory input length is:
   `previous_input + previous_output + new_tokens`.
6. Accepted turns sample `generation_length` for output tokens and extend the
   session's L3 hash IDs.

The JSONL output stores incremental turn input in `input_length`, even though
the in-memory `SynthesizedTurn.input_length` is cumulative. This is the Mooncake
trace format expected by AIPerf replay.

Reset-driven sessions can end in these ways:

- **Forced retire**: the next candidate turn would reach or exceed
  `max_prompt_tokens`. The overflowing turn is not added.
- **Probabilistic reset**: after the context-limit check, the generator applies:
  `p = base_probability * (1 + (context_scaling - 1) * input_length / max_prompt_tokens)`.
  If the draw succeeds, the session ends before adding that candidate turn.
- **Restart split**: if restart injection is enabled for that primary session,
  the session splits at a sampled turn index from `restart_turn_range`.

Restart splits are controlled by `restart_initial_probability` and
`restart_turn_range`. The restart probability decays linearly to zero over the
first 75% of primary sessions. When a split occurs:

- Session A ends with `restart_split`.
- Session B gets a new `session_id`, keeps the same `group_id`, and is marked
  with `is_restart` on its first JSONL row.
- Session B starts from Session A's accumulated input/output context and carries
  forward the same hash IDs.
- Session B is inserted later in the generated session order so it does not
  immediately overlap with Session A in the same concurrency window.

### Explicit Turn-Count Mode

If the config sets `turns`, the generator switches to explicit turn-count mode.
In this mode it samples a target number of turns from the `turns` distribution
and attempts to build a session with exactly that many turns.

Explicit turn-count mode cannot be combined with `reset` or
`restart_initial_probability`; config validation rejects that combination. A
session that reaches the sampled target ends with `target_turn_count`.

If the sampled session would hit `max_prompt_tokens` before reaching the target:

- With `allow_truncation: false`, the generator retries the whole session up to
  `max_session_attempts`, then raises an error if it still cannot fit.
- With `allow_truncation: true`, the generator returns the partial session and
  marks it as `forced_retire`.

## Inter-Turn Delays

`inter_turn_delay.context_scaling` defaults to `true` for compatibility with
existing workloads. After sampling the agentic/human mixture and applying
`inter_turn_delay.max`, the generator multiplies the delay by:

`max(0.2, 1 - 0.8 * previous_input / max_prompt_tokens)`

`previous_input` is the previous turn's cumulative input length. This is a
workload heuristic: fuller contexts produce shorter gaps. It is not an
inference-latency measurement. Set `"context_scaling": false` inside the
`inter_turn_delay` object to disable it. The setting is saved in the manifest
and generated config schema, and appears as 0/1 in the quality config summary.

Component `min`/`max` bounds apply during rejection sampling. After context
scaling, the selected component's `min` is enforced again. The global
`inter_turn_delay.max` is the final upper cap and takes precedence if it is
lower than that component's minimum; a global maximum of zero permits zero
delay. Turn 0 always has zero delay.

The random draws and cap-before-scaling order are unchanged. With scaling
enabled and no component minima, an existing config and seed produce the same
trace. Configs with minima may now produce longer gaps where scaling previously
violated the bound. Disabling scaling intentionally changes delays and timestamps.

Component means/medians describe the underlying lognormal distributions, not
targets for the final mixture. Rejection sampling, caps, and context scaling
change the output distribution. `quality.json` reports observed delay statistics
with null aggregate mean/median targets; the component values shown in reports
should not be interpreted as those aggregate targets.

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
