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

- Requests are grouped into sessions using `provided_session_id` when it actually forms multi-turn sessions.
- If `provided_session_id` is effectively unique per row, the loader falls back to `poor_man_session_id`.
- All timestamps are normalized to `ms since first event in file`.
- Rows inside each session are sorted by normalized timestamp before replay.
- `prompt` is replayed as the literal completion prompt.
- `output_tokens` becomes both `max_tokens` and `min_tokens`.
- `total_hashes` is forwarded as per-row request body metadata under `hash_ids`.
- `block_size` is forwarded per row when present.
- `request_canceled` is retained in trace metadata but is not filtered out.

`output_text` is preserved in the trace model for debugging and offline validation, but AIPerf still measures a fresh model response during the benchmark.

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

If your server expects OpenAI-style text completions explicitly:

```bash
aiperf profile \
  --model YOUR_MODEL \
  --url http://localhost:8000 \
  --endpoint /v1/completions \
  --endpoint-type completions \
  --input-file /path/to/trace.parquet \
  --custom-dataset-type baseten_trace \
  --fixed-schedule
```

## Notes

- This format is Parquet-only.
- Session stickiness still works because rows are grouped into multi-turn conversations.
- For completion traces that already contain the fully expanded historical prompt, AIPerf replays that prompt verbatim rather than reconstructing history from prior turns.
