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

The timestamp means you cannot know the path in advance, so capture it rather
than typing it out:

```bash
DATASET=$(ls -d .test/*/dataset.jsonl | tail -1)
```

`synthesize agentic-code` validates the generated `dataset.jsonl` before it
prints the run summary. You can also validate a saved or edited trace directly:

```bash
aiperf validate mooncake-trace --input "$DATASET"
```

## Replay With AIPerf

The generated `dataset.jsonl` is a Mooncake trace, so it replays with
`--custom-dataset-type mooncake_trace`.

Every request must fit the server's context window. The bundled `default`
config targets a 200k-token context (`max_prompt_tokens: 200000`), so replaying
it against a small model fails on the very first turn. The config below scales
every layer down to fit a 40k-token model such as `Qwen/Qwen3-0.6B`, while
keeping the multi-turn growth the generator exists to model:

<!-- setup-file-vllm-default-openai-endpoint-server path=agentic-small.json -->
```json
{
  "max_prompt_tokens": 24000,
  "block_size": 512,
  "cache": {
    "layer1_tokens": 4096,
    "layer1_5_tokens": 2048,
    "layer2": {"mean": 2000, "median": 1500},
    "layer1_5_groups": {"num_groups": 4, "zipf_alpha": 1.2}
  },
  "new_tokens_per_turn": {"mean": 1500, "median": 1000},
  "generation_length": {"mean": 200, "median": 150},
  "inter_turn_delay": {
    "agentic_fraction": 0.7,
    "agentic_delay": {"mean": 200, "median": 150},
    "human_delay": {"mean": 1000, "median": 800}
  },
  "reset": {"base_probability": 0.02, "context_scaling": 2.0}
}
```
<!-- /setup-file-vllm-default-openai-endpoint-server -->

Synthesize and replay in one go. The run directory is timestamped, so the
dataset path is captured rather than typed:

<!-- aiperf-run-vllm-default-openai-endpoint-server weight=300 -->
```bash
aiperf synthesize agentic-code \
  --config agentic-small.json \
  --num-sessions 3 \
  --output .test/agentic-smoke/

DATASET=$(ls -d .test/agentic-smoke/*/dataset.jsonl | tail -1)

aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --tokenizer Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --input-file "$DATASET" \
  --custom-dataset-type mooncake_trace \
  --streaming
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

Three sessions expand to roughly 30 requests, because each session is a
multi-turn conversation. Scale the run with `--num-sessions`, not with
`--request-count`: turn 0 of every session carries a `timestamp`, which
auto-promotes the run to [fixed-schedule mode](../benchmark-modes/trace-replay.md#automatic-fixed-schedule-promotion),
and fixed-schedule mode takes its request count from the trace. `--concurrency`
and `--request-count` are overridden there, silently.

To ignore the recorded arrival times and replay the same trace as a plain
concurrency test, opt out of the promotion with `--no-fixed-schedule`:

```bash
aiperf profile \
  --model YOUR_MODEL \
  --tokenizer YOUR_MODEL \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --input-file "$DATASET" \
  --custom-dataset-type mooncake_trace \
  --no-fixed-schedule \
  --concurrency 8 \
  --benchmark-duration 300 \
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

- `max_prompt_tokens`: `200000`.
- `block_size`: `512` tokens.
- A `32000` token global L1 prefix shared by all sessions.
- A `20000` token L1.5 group-shared prefix spread over `50` Zipf-weighted
  groups.
- Session-specific initial context sampled around a `10000` token mean.
- New turn input sampled around a `3500` token mean, uncapped.
- Output length sampled around a `500` token mean, uncapped.
- A small reset probability that grows with context utilization.

A `max_prompt_tokens` of `200000` means the default config only replays
against a server with a 200k-token context. See
[Replay With AIPerf](#replay-with-aiperf) for a scaled-down config.

```bash
aiperf synthesize agentic-code \
  --config default \
  --num-sessions 1000 \
  --seed 42 \
  --output .test/

aiperf synthesize agentic-code \
  --config "$(ls -d .test/*/manifest.json | tail -1)" \
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

`--max-isl` overrides `max_prompt_tokens` only; it does not scale the prefix
layers. Setting it below `layer1_tokens + layer1_5_tokens + layer2`, roughly
`62000` for the default config, forces every session to retire at turn 0 and
produces a single-turn trace. Scale the layers in a config JSON instead.

The config schema is generated at
`src/aiperf/dataset/agentic_code_gen/configs/spec.json`.

## Related Tutorials

- [Trace Benchmarking](../benchmark-modes/trace-replay.md) - deterministic trace replay.
- [Prefix Synthesis](prefix-synthesis.md) - KV cache testing with shared prefixes.
- [Fixed Schedule](fixed-schedule.md) - timestamp-based execution.
- [Multi-Turn Conversations](multi-turn.md) - session replay and conversation state.
