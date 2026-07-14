---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Baseten Traces
---

# Baseten Trace Replay

Use `baseten_trace` to replay Parquet-exported completion traces against `/v1/completions` with fixed-schedule timing.

## Supported Input

The loader expects a Parquet file with these core columns:

- `timestamp_start_unix_ms`
- `prompt`
- `input_tokens`
- `output_tokens`

Common optional columns:

- `provided_session_id`
- `poor_man_session_id`
- `total_hashes`
- `block_size`
- `request_canceled`
- `duration_e2e_ms`
- `duration_ttft_ms`
- `output_text`

## Replay Semantics

- Requests are grouped into sessions by whichever of `provided_session_id` / `poor_man_session_id` shows the stronger repeated-session signal; ties prefer `provided_session_id`.
- If neither column repeats (or the chosen column is null for a row), the loader generates a fresh per-row session ID.
- All timestamps are normalized to `ms since first event in file`.
- Rows inside each session are sorted by normalized timestamp before replay.
- `prompt` is replayed as the literal completion prompt.
- `output_tokens` becomes both `max_tokens` and `min_tokens`. If `max_osl` caps the row, both values use the capped output length. Rows with `output_tokens=0` (e.g. canceled requests) are floored to 1, since `max_tokens` must be at least 1.
- `total_hashes` is forwarded as per-row request body metadata under `hash_ids`.
- `block_size` is forwarded per row when present.
- `request_canceled` is retained in trace metadata but is not filtered out.
- Per-row request metadata is forwarded through `Turn.extra_body` and merged after endpoint-level `extra` values, so replay metadata wins when keys collide.

`output_text` is preserved in the trace model for debugging and offline validation, but AIPerf still measures a fresh model response during the benchmark.

## Timing model

By default replay is **open-loop**: every request is scheduled at its absolute recorded timestamp, treating the trace as a fixed event log (but see the known limitation below).

- `--no-open-loop-replay` selects **closed-loop back-pressure**: each continuation turn fires a think-time delay after the prior turn completes. Think-time = recorded start-to-start gap minus the prior turn's recorded `duration_e2e_ms`, clamped by `--inter-turn-delay-cap-seconds`. Use closed-loop when replayed service times differ from recorded ones (e.g. A/A comparisons) and sessions must stay causally ordered.
- `--replay-speedup N` divides all timestamps and inter-turn delays by `N` (e.g. `10` replays a ~2h trace in ~12 minutes) without touching `hash_ids`, so KV-cache fidelity is preserved.
- `--max-idle-gap-cap-seconds S` collapses any global dead-air gap between consecutive requests (across all sessions) to at most `S` seconds. `S` is replay wall-clock seconds, applied after `--replay-speedup` compression — the replay never idles longer than `S` real seconds regardless of speedup.
- `--trace-session-sample-ratio R` keeps a fraction `R` of whole sessions, preserving multi-turn integrity. Sampling is only deterministic across runs with a fixed `--random-seed`.
- `--omit-kv-hints` stops forwarding `hash_ids`/`block_size` KV-cache hints in request bodies (some strict frontends reject unknown parameters).
- `--force-min-tokens` (default) pins `min_tokens` to the recorded output length so replayed generations match recorded lengths; `--no-force-min-tokens` lets EOS end generations naturally (some servers reject `min_tokens`).
- `--open-loop-strict` additionally treats every trace row as an independent single-turn session so ALL requests (not just each session's first turn) fire at their absolute recorded timestamps, even if earlier turns of the same recorded session have not completed. This trades away multi-turn session grouping and session metrics.

**Known limitation:** without `--open-loop-strict`, a session's continuation turns fire at max(scheduled time, prior turn completion) — an open-loop timestamp inside a session cannot preempt an in-flight prior turn.

## Command

```bash
aiperf profile \
  --model YOUR_MODEL \
  --url http://localhost:8000 \
  --endpoint-type completions \
  --input-file /path/to/trace.parquet \
  --custom-dataset-type baseten_trace \
  --fixed-schedule
```

The `completions` endpoint type already targets `/v1/completions`; if your gateway serves completions at a non-default path, add `--endpoint <path>`.

## Notes

- Session stickiness still works because rows are grouped into multi-turn conversations.
- For completion traces that already contain the fully expanded historical prompt, AIPerf replays that prompt verbatim rather than reconstructing history from prior turns.
