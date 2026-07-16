<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Error injection

Drive AIPerf's error-handling and retry paths deterministically. Source of truth:
`rust/mock-server/src/config.rs` (flags) and `rust/mock-server/src/handlers.rs`
(`maybe_inject_error`, `chat_stream`). All draws come from the seeded `mock.errors` RNG
stream, so the injected sequence is reproducible under `--random-seed`.

## Flags

| Flag | Default | Effect |
|---|---|---|
| `--error-rate <0..1 or %>` | 0.0 | Fraction of requests failed before any bytes are sent |
| `--error-status-codes <csv>` | `500` | Menu of HTTP codes; one is picked per injected error from the seeded stream (e.g. `429,503,400,500`) |
| `--error-retry-after <secs>` | 1 | `Retry-After` header (whole seconds) emitted on injected `429`/`503` only |
| `--error-midstream-rate <0..1>` | 0.0 | Seeded probability a *streaming* chat request emits a few token frames then a terminal mid-stream `event: error` SSE frame |

Note: the e2e tests express `error_rate` as a percentage-style float (e.g. `45.0`, `100.0`)
in the `MockServerConfig` struct; the flag accepts the same value.

## Pre-stream error (`--error-rate` + `--error-status-codes`)

When the rate fires, the handler returns immediately with a chosen status and a
`{"detail": "Simulated error (status <code>)"}` JSON body. `429`/`503` also carry a
`Retry-After: <error-retry-after>` header (the backoff hint AIPerf's retry policy reads);
other codes carry none.

**e2e recipe — a 429 status shows in raw records** (`test_error_fidelity.rs`):

```bash
# mock: --fast --no-tokenizer --random-seed 7 --error-rate 45 --error-status-codes 429 --error-retry-after 3
aiperf profile --model gpt-4 --url http://127.0.0.1:8000 --endpoint-type chat \
  --request-count 40 --concurrency 4 --workers-max 1 \
  --random-seed 7 --export-level raw --ui simple
```

Raw-record assertions (`profile_export_raw.jsonl`): each errored record has
`error.code == 429`, `error.type == "HttpError"`, top-level `status == 429`; no record shows
`500`. The run still exits 0.

**Walking the status menu** (`--error-rate 100 --error-status-codes 429,503,400`): every
record's `error.code` is one of the menu codes, and multiple distinct codes appear across the
run.

## Mid-stream SSE error (`--error-midstream-rate`)

This is the only path that exercises the runner's **mid-stream** SSE error classification;
pre-stream injection fails at handler entry before any bytes are sent. When it fires on a
streaming chat request, the mock emits up to `MIDSTREAM_TOKENS_BEFORE_ERROR` (3) normal token
frames, then a terminal frame:

```
event: error
: <message>

```

The runner's SSE reader (`aiperf::transport_http::sse::reader`) classifies any frame whose
`event` field equals `error` as a transport error — pseudo-status **502**, type
`sse_error` / `SSEResponseError` — aborting the stream before `[DONE]` (no usage chunk, no
`[DONE]`). Runs even in fast mode, and never draws the adversarial null-object path.

**e2e recipe — mid-stream error truncates the record, run completes**
(`test_error_fidelity.rs`):

```bash
# mock: --fast --no-tokenizer --random-seed 7 --error-midstream-rate 0.5
aiperf profile --model gpt-4 --url http://127.0.0.1:8000 --endpoint-type chat --streaming \
  --output-tokens-mean 20 --output-tokens-stddev 0 \
  --request-count 24 --concurrency 4 --workers-max 1 \
  --random-seed 7 --export-level raw --ui simple
```

Raw-record assertions: some records errored, some OK. Errored records:
`error.type == "SSEResponseError"`, `error.code == 502`, and carry **1..=3** partial content
chunks (truncated, not zero). OK records carry exactly 20 content chunks. The summary
`profile_export_aiperf.json` is still written.
