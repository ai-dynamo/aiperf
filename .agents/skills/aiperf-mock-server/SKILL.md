<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
---
name: aiperf-mock-server
description: >-
  Build, launch, and use the in-repo Rust mock inference server
  (`aiperf-mock-server`, crate at `rust/mock-server`) as a local benchmark target
  for AIPerf. Use this whenever you need a fake/stand-in inference server to run
  `aiperf profile` against locally, to smoke-test the HTTP/SSE or gRPC transport,
  to reproduce an integration test outside `cargo test`, or when a task mentions
  "the mock server", "mock inference server", "aiperf-mock-server", "fastmock",
  or getting a `/v1/chat/completions` (or KServe/Riva gRPC, embeddings, rerank,
  responses, vllm generate) target up on localhost. It covers realistic-latency
  and saturation modes, error injection, extended usage fields, tool calls,
  accuracy-dataset ground-truth mode, GPU/vLLM/SGLang telemetry, and TLS/UDS/gRPC
  transports — each documented in a `references/*.md` file. Prefer this skill over
  guessing flags the server has many knobs and non-obvious startup gotchas
  (HF tokenizer download, IPv6-vs-IPv4 localhost) that this captures.
---

# Running the AIPerf Rust mock server

`aiperf-mock-server` is a high-throughput, OpenAI-compatible (plus KServe/Riva/TGI/rerank/
image/RAG) mock inference server. It is a **standalone** developer/test target — it is NOT part
of the `aiperf` dependency graph and Python never supervises it. You launch it yourself,
it exposes an ordinary HTTP (or gRPC/HTTPS/UDS) URL, and you point a benchmark (or the transport
tests) at that URL.

Source of truth: `rust/mock-server/src/` (binary `main.rs`, flags `config.rs`, routes `app.rs`).
Read those if a flag here looks stale — code is truth.

## The 30-second path (fast, offline, no HF download)

```bash
# From the repo root (the Cargo workspace lives under rust/). Build once:
cargo build --release --manifest-path rust/Cargo.toml -p aiperf-mock-server

# Run instant-latency + no tokenizer download, on 127.0.0.1:8000:
./rust/target/release/aiperf-mock-server --fast --no-tokenizer
```

`--fast` (`-f`) zeros every latency (TTFT/ITL, embeddings, ranking, images) and bypasses the
scheduler + prefix cache, so responses stream back instantly — the right mode for wiring/plumbing
smoke tests. `--no-tokenizer` skips loading the default HF tokenizer (`Qwen/Qwen3-0.6B`), which
otherwise triggers a **network download on first start** and hangs/fails offline. Always pass
`--no-tokenizer` unless you specifically need real token counting.

Confirm it's actually up before using it (a server that died on a bad flag otherwise masquerades
as "starting"):

```bash
curl -sf http://127.0.0.1:8000/health && echo " OK"
curl -s http://127.0.0.1:8000/v1/models | head -c 400
```

`/health` returns 200 when live; `/v1/models` returns the model list. If either fails, the
process is not serving — check its log for a startup error rather than assuming it's still booting.

### Use 127.0.0.1, never `localhost`

The mock binds `127.0.0.1` (IPv4) by default. `localhost` can resolve to `::1` (IPv6) first, and
the mock is not listening there — clients then fail with connection-refused / `FailedReply` that
looks like a server bug. Always use the literal `127.0.0.1` in curl commands, config URLs, and
`--url` flags. To serve IPv6 or all interfaces, pass `--host ::1` or `--host 0.0.0.0`.

## Running it in the background for a benchmark

```bash
./target/release/aiperf-mock-server --fast --no-tokenizer --port 8000 &
MOCK_PID=$!
until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 0.2; done
echo "mock up on 8000 (pid $MOCK_PID)"
# ... run your benchmark ...
kill $MOCK_PID
```

Pick a non-default port (e.g. `--port 18000`) if 8000 might be taken; the mock sets
`SO_REUSEPORT` on Linux but a foreign listener still wins the bind.

## Pointing an AIPerf run at it

The mock is just an OpenAI-compatible URL. Point the Python `aiperf` frontend at it:

```bash
aiperf profile \
  --url http://127.0.0.1:8000 \
  --model mock-model \
  --endpoint-type chat \
  --streaming \
  --request-count 20
```

Any model name works — the mock echoes whatever model you send and appends seen models to
`/v1/models`. Use `--models a,b,c` on the server to pre-advertise names. For real token-count
fidelity, drop `--no-tokenizer` and let the server load the matching tokenizer. Other endpoint
types (`vllm_generate`, `responses`, `embeddings`, `kserve_v2_infer`, `riva_asr`, …) and
transports (`grpc`, `https`) are covered in `references/endpoints.md` and
`references/grpc-and-transports.md`.

The Rust integration tests under `rust/aiperf/tests/` and the e2e harness under `rust/e2e/`
spawn this same binary. They locate it via `$AIPERF_MOCK_RS_BIN`, then next to the test
executable, then `target/{debug,release}/aiperf-mock-server`, then `PATH`. Export
`AIPERF_MOCK_RS_BIN=/abs/path` to force a specific build.

## Feature references

Each area has a focused reference file under `references/`. Read the one you need on demand.

| Reference | Covers |
|---|---|
| [`references/latency-and-load.md`](references/latency-and-load.md) | Analytic latency (`--ttft`/`--itl`/…), batched scheduler & saturation, prefix cache, `--processes` L4 balancer, timerfd precision |
| [`references/endpoints.md`](references/endpoints.md) | Full HTTP route/endpoint catalog + the `--endpoint-type` for each (chat/completions/embeddings/responses/messages, `vllm_generate`, `/openai/v1/*`, KServe v2 infer / v1 predict, rerank/TGI/image/RAG) |
| [`references/grpc-and-transports.md`](references/grpc-and-transports.md) | KServe gRPC (`--grpc-port`, `--grpc-behavior`, `--grpc-embedding-dim`), Riva ASR/TTS/NLP, UDS (`--uds`, driven via `endpoint.uds_path`), TLS/HTTPS + `grpcs` (`--tls-*`, driven via `endpoint.ssl_verify: false`) — all reachable through `aiperf profile` |
| [`references/telemetry.md`](references/telemetry.md) | `/metrics` + vLLM/SGLang/TRT-LLM/Dynamo dialects + DCGM `/dcgm*/metrics`; scraped via `--server-metrics` / `--gpu-telemetry` |
| [`references/error-injection.md`](references/error-injection.md) | `--error-rate`, `--error-status-codes`, `--error-retry-after`, `--error-midstream-rate` (mid-stream SSE error) |
| [`references/usage-accounting.md`](references/usage-accounting.md) | The `--usage-*` knobs and the exact extended usage JSON keys (cache write/miss/read, audio tokens+seconds, accepted/rejected prediction, tool-use prompt) |
| [`references/tool-calls.md`](references/tool-calls.md) | `--tool-call-rate`/`-name`/`-arguments`; non-stream + streamed `delta.tool_calls` shapes; `toolUsePromptTokenCount` |
| [`references/accuracy.md`](references/accuracy.md) | Ground-truth accuracy dataset mode: format, `--accuracy-*`, matching + `match_key`, adversarial shapes, `/accuracy` + `aiperf_mock_accuracy_*` |
| [`references/microbench-tools.md`](references/microbench-tools.md) | `fastmock` / `fastclient` / `fastmock-uring` loopback micro-benchmark helpers, plus the in-crate `--ludicrous-speed`/`--plaid` port of `fastmock` |

## Flags cheat-sheet (all flags, grouped)

Every flag has a `MOCK_SERVER_*` env twin; set the log level dynamically with `AIPERF_MOCK_LOG`
(a `tracing` env-filter). See `rust/mock-server/src/config.rs` for the authoritative list.

**Core / networking**

| Flag | Default | Purpose |
|---|---|---|
| `-p, --port` | 8000 | Listen port |
| `--host` | 127.0.0.1 | Bind address (`::1`, `0.0.0.0`, …) |
| `-f, --fast` | off | Zero all latency; bypass scheduler + cache |
| `--ludicrous-speed` / `--plaid` | off | **NOT a realistic mock server.** Skip the real server entirely; serve one hard-coded response via blocking `std::net` sockets (`src/fastmock.rs`) — raw-throughput extreme testing only, see `references/microbench-tools.md` |
| `--blocking` | off | **Alternative I/O engine.** Serve the real chat path on a blocking thread-per-connection engine (SO_REUSEPORT accept loops, no async runtime) instead of tokio/hyper. Same responses + metrics; implies `--fast`. **~40% faster** than tokio/hyper under saturating load (~1.5-1.6M vs ~1.1M rps metrics-on, 32-core loopback) by dropping the async-runtime per-request overhead. **Full endpoint coverage**: the hot paths (chat + text completions, streaming and non, and embeddings) use fast synchronous renderers; every other route (messages, responses, rerank, images, sagemaker, KServe, tgi, rag, all GET/metrics-dialect routes) falls back to the real axum `Router` via a per-thread current-thread runtime. gRPC is not served (HTTP engine). Error-injection / mid-stream-failure configs are not wired into the fast path |
| `--uring` | off | Alternative io_uring thread-per-core engine (monoio); same served routes as `--blocking`, implies `--fast`. Requires `--features uring`. In practice **slower than `--blocking`** for this real workload (its async overhead ~matches tokio) — prefer `--blocking` |
| `--no-tokenizer` | off | Skip HF tokenizer load (avoids network download) |
| `--no-metrics` | off | Disable all hot-path metric recording (`/metrics` still responds, reports zeros). Removes per-request histogram observes + shared-counter increments; ~+15-30% throughput under saturating load. Behavioral response content unaffected — for raw-throughput runs that don't scrape metrics |
| `--openmetrics` | off | Serve the `/metrics` family as OpenMetrics text (`application/openmetrics-text; version=1.0.0`, `# EOF`, suffix-less counter families) instead of classic Prometheus text, matching the vLLM Rust frontend. No effect with `--no-metrics`. See `references/telemetry.md` |
| `-w, --workers` | 0 (=nproc) | Tokio worker threads |
| `--processes` | 1 | Spawn N child servers behind an L4 round-robin balancer (0=nproc) |
| `--max-concurrent-streams` | 0 | h2 `SETTINGS_MAX_CONCURRENT_STREAMS` (0 = hyper default) |
| `--models a,b` | builtin list | Pre-advertise models on `/v1/models` |
| `--random-seed` | — | Deterministic latency/jitter/errors/accuracy verdicts |
| `-v, --verbose` / `--log-level` / `--access-logs` | off / INFO / off | DEBUG logging / level / per-request access logs |

**Transports** (see `references/grpc-and-transports.md`): `--grpc-port`, `--grpc-behavior`
(`auto`/`text`/`rankings`/`images`), `--grpc-embedding-dim`, `--uds`, `--tls-cert`, `--tls-key`,
`--tls-self-signed`.

**Latency / scheduler / cache** (see `references/latency-and-load.md`): `--ttft`, `--itl`,
`--ttft-per-isl-token-ms`, `--ttft-concurrency-quad-ms`, `--itl-per-osl-token-ms`,
`--itl-concurrency-lin-ms`, `--ttft-jitter-cv`, `--itl-jitter-cv`; `--scheduler-enabled` +
`--scheduler-*` (step-ms, max-batch-size, prefill chunk/work/throughput, goodput-collapse);
`--disable-prefix-cache` + `--prefix-cache-*`; `--embedding-*`/`--ranking-*`/`--image-retrieval-*`
per-endpoint latency.

**Error injection** (see `references/error-injection.md`): `--error-rate`,
`--error-status-codes`, `--error-retry-after`, `--error-midstream-rate`.

**Extended usage** (see `references/usage-accounting.md`): `--usage-cache-write-tokens`,
`--usage-cache-miss-tokens`, `--usage-cache-read-tokens`, `--usage-prompt-audio-tokens`,
`--usage-completion-audio-tokens`, `--usage-prompt-audio-seconds`,
`--usage-accepted-prediction-tokens`, `--usage-rejected-prediction-tokens`,
`--usage-tool-use-prompt-tokens`.

**Tool calls** (see `references/tool-calls.md`): `--tool-call-rate`, `--tool-call-name`,
`--tool-call-arguments`.

**Accuracy** (see `references/accuracy.md`): `--accuracy-dataset`, `--accuracy-format`,
`--accuracy-correct-rate`, `--accuracy-cot-rate`, `--accuracy-reasoning-field`,
`--accuracy-adversarial-rate`, `--accuracy-match`.

**Telemetry / DCGM** (see `references/telemetry.md`): `--dcgm-gpu-name`, `--dcgm-num-gpus`,
`--dcgm-min-throughput`, `--dcgm-window-sec`, `--dcgm-hostname`, `--dcgm-seed`, `--dcgm-auto-load`.

**Tokenizer**: `--tokenizer`, `--tokenizer-revision`, `--tokenizer-trust-remote-code`,
`--no-tokenizer`.

## Gotchas recap

- **Verify liveness before reporting "running."** Read the log / curl `/health`; a bad flag makes
  the process exit instantly while looking like it's booting.
- **`--no-tokenizer` unless you need token counts** — the default tokenizer downloads from Hugging
  Face on first start.
- **`127.0.0.1`, not `localhost`** — the default bind is IPv4-only.
- **Don't proxy loopback** — if `HTTP_PROXY`/`HTTPS_PROXY` are set, a client may route `127.0.0.1`
  through a proxy that 405s it; set `NO_PROXY=127.0.0.1,localhost` for the client if you hit that.
- **`--fast` and `--scheduler-enabled` are mutually exclusive** in effect — `--fast` turns the
  scheduler (and prefix cache) off.
- **Accuracy mode keys on the prompt, not an id.** If AIPerf re-templates the prompt beyond
  whitespace, give each row a `match_key` fragment guaranteed to survive in the wire prompt.
  `--fast` does NOT disable accuracy (only latency).
- **`--grpc-port` and `--uds` are skipped under `--processes N`** (the L4 balancer is TCP/HTTP-only).
- **Two runner-side transport limits:** `aiperf profile` has no UDS/`unix://` knob today, and its
  tonic `grpcs` client trusts system roots only (no accept-invalid), so a self-signed `grpcs` run
  isn't drivable via `aiperf profile`. Both are proven by direct clients — see
  `references/grpc-and-transports.md`.
