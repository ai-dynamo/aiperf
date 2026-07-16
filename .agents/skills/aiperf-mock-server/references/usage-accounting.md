<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Extended usage accounting (`--usage-*`)

Inject deterministic, fixed values into the emitted `usage` object so AIPerf's `usage_*`
catalog metrics can be exercised end-to-end. Every knob defaults to `0` (`0.0` for the
seconds knob), meaning the corresponding sub-field is **omitted entirely** — a normal run's
usage payload is byte-identical. Source: `rust/mock-server/src/config.rs` and
`apply_usage_fields` in `rust/mock-server/src/handlers.rs`; the wire shape is in
`rust/mock-server/src/models.rs` (`Usage`, `PromptTokensDetails`, `CompletionTokensDetails`).

## Flags → exact emitted JSON key → AIPerf metric

| Flag | JSON key (location) | AIPerf metric |
|---|---|---|
| `--usage-cache-write-tokens` | top-level `cache_creation_input_tokens` (OpenAI) + Anthropic `messages` usage | `usage_prompt_cache_write_tokens` |
| `--usage-cache-miss-tokens` | top-level `prompt_cache_miss_tokens` | `usage_prompt_cache_miss_tokens` |
| `--usage-cache-read-tokens` | `cache_read_input_tokens` — **Anthropic `messages` usage only** (OpenAI reports cache reads via `prompt_tokens_details.cached_tokens`) | (Anthropic re-total) |
| `--usage-prompt-audio-tokens` | `prompt_tokens_details.audio_tokens` | `usage_prompt_audio_tokens` |
| `--usage-completion-audio-tokens` | `completion_tokens_details.audio_tokens` | `usage_completion_audio_tokens` |
| `--usage-prompt-audio-seconds` | top-level `prompt_audio_seconds` (f64) | `usage_prompt_audio_seconds` |
| `--usage-accepted-prediction-tokens` | `completion_tokens_details.accepted_prediction_tokens` | `usage_accepted_prediction_tokens` |
| `--usage-rejected-prediction-tokens` | `completion_tokens_details.rejected_prediction_tokens` | `usage_rejected_prediction_tokens` |
| `--usage-tool-use-prompt-tokens` | top-level `toolUsePromptTokenCount` (exact key AIPerf's `UsageView` reads) | `usage_tool_use_prompt_tokens` |

Nested details objects (`prompt_tokens_details`, `completion_tokens_details`) are created on
demand only when an extended field needs them. To see the usage on the wire in a streaming
run, the client must request it — pass `--use-server-token-count` on `aiperf profile`, which
sets `stream_options.include_usage` so the terminal usage chunk is emitted.

## e2e recipe (`test_usage_fields.rs`)

Mock config sets all nine knobs to distinct values:
`--usage-cache-write-tokens 11 --usage-cache-miss-tokens 22 --usage-cache-read-tokens 33
--usage-prompt-audio-tokens 44 --usage-completion-audio-tokens 55
--usage-prompt-audio-seconds 6.5 --usage-accepted-prediction-tokens 77
--usage-rejected-prediction-tokens 88 --usage-tool-use-prompt-tokens 99` (plus `--fast
--no-tokenizer`).

```bash
aiperf profile --model gpt-4 --url http://127.0.0.1:8000 --endpoint-type chat --streaming \
  --use-server-token-count \
  --input-file prompts.jsonl --custom-dataset-type single_turn \
  --request-count 4 --concurrency 2 --workers-max 1 \
  --export-level raw --ui simple
```

Raw-record assertions (the streamed usage frame per record): `cache_creation_input_tokens == 11`,
`prompt_cache_miss_tokens == 22`, `toolUsePromptTokenCount == 99`, `prompt_audio_seconds == 6.5`,
`prompt_tokens_details.audio_tokens == 44`, `completion_tokens_details.audio_tokens == 55`,
`completion_tokens_details.accepted_prediction_tokens == 77`,
`completion_tokens_details.rejected_prediction_tokens == 88`. (`cache_read_input_tokens == 33`
is Anthropic-only, so it appears only on the `/v1/messages` path.)

Summary assertions (`profile_export_aiperf.json`): for each metric, `avg == value` and
`total_<tag> == value × request_count` (e.g. `usage_prompt_cache_write_tokens.avg == 11`,
`total == 44` over 4 requests).
