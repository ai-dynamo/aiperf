<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
---
name: aiperf-mock-server
description: >-
  Build, launch, and use the in-repo Rust mock inference server
  (`aiperf-mock-server`, crate at `rust/mock-server`) as a local OpenAI-compatible
  benchmark target. Use this whenever you need a fake/stand-in LLM server to run
  AIPerf against locally, to smoke-test the HTTP/SSE transport, to reproduce an
  integration test outside of `cargo test`, or when a task mentions "the mock
  server", "mock inference server", "aiperf-mock-server", "fastmock", pointing
  `aiperf profile` at a local endpoint, or getting a `/v1/chat/completions`
  target up on localhost. Also covers realistic-latency and saturation modes for
  latency/throughput experiments. Prefer this skill over guessing flags — the
  server has many knobs and non-obvious startup gotchas (HF tokenizer download,
  IPv6-vs-IPv4 localhost) that this captures.
---

# Running the AIPerf Rust mock server

`aiperf-mock-server` is a high-throughput, OpenAI-compatible mock inference server.
It is a **standalone** developer/test target — it is NOT part of the
`aiperf-runner` dependency graph and Python never supervises it. You launch it
yourself, it exposes an ordinary HTTP URL, and you point a benchmark (or the
transport tests) at that URL.

Source of truth: `rust/mock-server/src/` (binary `main.rs`, flags `config.rs`,
routes `app.rs`). Read those if a flag here looks stale — code is truth.

## The 30-second path (fast, offline, no HF download)

```bash
# From the workspace root (rust/... visible). Build once:
cargo build --release -p aiperf-mock-server

# Run instant-latency + no tokenizer download, on 127.0.0.1:8000:
./target/release/aiperf-mock-server --fast --no-tokenizer
```

`--fast` (`-f`) zeros every latency (TTFT/ITL, embeddings, ranking, images) and
bypasses the scheduler + prefix cache, so responses stream back instantly — the
right mode for wiring/plumbing smoke tests where you don't care about realistic
timing. `--no-tokenizer` skips loading the default HF tokenizer
(`Qwen/Qwen3-0.6B`), which otherwise triggers a **network download on first
start** and will hang/fail offline. Always pass `--no-tokenizer` unless you
specifically need real token counting.

Confirm it's actually up before using it (a server that died on a bad flag
otherwise masquerades as "starting"):

```bash
curl -sf http://127.0.0.1:8000/health && echo " OK"
curl -s http://127.0.0.1:8000/v1/models | head -c 400
```

`/health` returns 200 when live; `/v1/models` returns the OpenAI-style model
list. If either fails, the process is not serving — check its log output for a
startup error rather than assuming it's still booting.

### Use 127.0.0.1, never `localhost`

The mock binds `127.0.0.1` (IPv4) by default. `localhost` can resolve to `::1`
(IPv6) first, and the mock is not listening there — clients then fail with
connection-refused / `FailedReply` that looks like a server bug. Always use the
literal `127.0.0.1` in curl commands, config URLs, and `--url` flags. To
deliberately serve IPv6 or all interfaces, pass `--host ::1` or `--host 0.0.0.0`.

## Running it in the background for a benchmark

When you need the server alive while you drive traffic in the same session,
launch it in the background and wait for `/health` before proceeding:

```bash
./target/release/aiperf-mock-server --fast --no-tokenizer --port 8000 &
MOCK_PID=$!
until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 0.2; done
echo "mock up on 8000 (pid $MOCK_PID)"
# ... run your benchmark ...
kill $MOCK_PID
```

Pick a non-default port (e.g. `--port 18000`) if 8000 might be taken; the mock
sets `SO_REUSEPORT` on Linux but a foreign listener still wins the bind.

## Pointing an AIPerf run at it

The mock is just an OpenAI-compatible URL. Point the Python `aiperf` frontend at
it with the endpoint URL and a model the mock advertises:

```bash
aiperf profile \
  --url http://127.0.0.1:8000 \
  --model mock-model \
  --endpoint-type chat \
  --streaming \
  --request-count 20
```

Any model name works — the mock echoes whatever model you send and appends
seen models to `/v1/models`. Use `--models a,b,c` on the server to pre-advertise
specific names. For real token-count fidelity in the results, drop
`--no-tokenizer` on the server and let it load the matching tokenizer.

The Rust integration tests under `rust/aiperf/tests/` (e.g.
`scheduled_real_mock.rs`, `transport_http_*.rs`, `graph_*.rs`) spawn this
same binary automatically. They locate it in this order: `$AIPERF_MOCK_RS_BIN`,
then next to the test executable, then `target/{debug,release}/aiperf-mock-server`,
then `PATH`. To force a specific build, export `AIPERF_MOCK_RS_BIN=/abs/path`.

## Realistic latency instead of `--fast`

Drop `--fast` to get the default analytic latency model: **TTFT 20 ms, ITL
5 ms** per token, applied per request. Tune with flags (each also has a
`MOCK_SERVER_*` env var):

| Flag | Effect |
|---|---|
| `--ttft <ms>` / `--itl <ms>` | Base first-token / inter-token latency |
| `--ttft-per-isl-token-ms <ms>` | Prefill cost that scales with prompt length |
| `--ttft-jitter-cv` / `--itl-jitter-cv` | Lognormal jitter (stddev/mean) |
| `--error-rate <0..1>` | Fraction of requests failed, to test error handling |
| `--random-seed <n>` | Deterministic latency/jitter/errors |

Example — 40 ms TTFT, 8 ms ITL, 10% jitter, seeded:

```bash
./target/release/aiperf-mock-server --no-tokenizer \
  --ttft 40 --itl 8 --ttft-jitter-cv 0.1 --itl-jitter-cv 0.1 --random-seed 1
```

## Saturation / throughput-vs-concurrency curves

For a realistic throughput knee (tok/s that saturates and TTFT that grows with
load), enable the step-based batched scheduler instead of the closed-form model:

```bash
./target/release/aiperf-mock-server --no-tokenizer \
  --scheduler-enabled \
  --scheduler-max-batch-size 256 \
  --scheduler-max-prefill-chunks-per-step 8
```

The knee lands near concurrency ≈ `--scheduler-max-batch-size`. Prefill becomes
the binding constraint as you lower `--scheduler-max-prefill-chunks-per-step`.
There are many more scheduler knobs (goodput collapse, sublinear prefill
throughput, admit jitter) — read `rust/mock-server/src/config.rs` for the full
set with inline rationale. `--fast` disables the scheduler, so don't combine
them.

## Multi-process load balancer (`--processes N`)

A single server process shares one tokio runtime; at very high request rates the
runtime scheduler / allocator can become the ceiling. `--processes N` (N > 1)
turns the launched binary into a **lightweight L4 (TCP) round-robin load
balancer**: it binds the public `--host:--port`, spawns `N` child
`aiperf-mock-server` processes (the same binary, carrying the exact same config)
on internal loopback ports, and splices each accepted connection to the next
backend in rotation. The client sees the identical OpenAI-compatible frontend on
**one URL** — HTTP/1.1 keep-alive, HTTP/2, and SSE streaming all pass through
untouched, because the balancer never parses HTTP.

```bash
# 4 backend processes behind one round-robin front door on :8000.
./target/release/aiperf-mock-server --processes 4 --no-tokenizer --port 8000
# --processes 0 = auto = one process per CPU.
```

- **When to use it:** you have saturated a single mock process (it is CPU-bound
  on one runtime) and want more aggregate throughput on a many-core box.
- **Worker threads auto-divide:** with `--workers` on its default (auto), each
  child gets `max(1, nproc / processes)` tokio workers, so the total worker count
  stays bounded rather than `N × nproc`. An explicit `--workers` is honored
  per-child.
- **Round-robin is per connection, not per request** (the cheapest, HTTP-blind
  distribution). A benchmark driving concurrency `C` opens ~`C` keep-alive
  connections, which spread evenly across the `N` backends as long as `C >= N`
  (the intended regime); below that some backends idle.
- **Lifecycle:** the balancer health-gates every child before opening the public
  port, tears everything down on Ctrl-C, and fails fast if any child dies. On
  Linux children also get `PR_SET_PDEATHSIG`, so they are reaped even if the
  balancer is `SIGKILL`ed. `--processes 1` (the default) is the unchanged
  single-process path.

## Timer precision (timerfd)

Latency injection (TTFT/ITL pacing, scheduler step cadence) runs on the `aiperf`
`RealClock` backend: waits use a `CLOCK_MONOTONIC` **`timerfd`** awaited through
tokio's IO reactor (`aiperf::clock::sleep_ns`), giving nanosecond resolution
instead of `tokio::time`'s ~1 ms timer wheel (which would quantize a 5 ms ITL by
~20%). This matters whenever you set sub-10 ms `--itl` / `--ttft` or a small
`--scheduler-step-ms`; it is transparent otherwise. Non-Linux platforms fall back
to `tokio::time` (coarser).

## What the server exposes

All routes are registered in `rust/mock-server/src/app.rs`. Highlights:

- **LLM**: `POST /v1/chat/completions`, `POST /v1/completions` (real SSE when
  `stream: true`), `POST /v1/embeddings`
- **Model listing**: `GET /v1/models`, `GET /v1/models/{id}`
- **Rerank**: `POST /v1/ranking` (NIM), `POST /rerank` (HF TEI), `POST /v2/rerank` (Cohere)
- **TGI**: `POST /generate`, `POST /generate_stream`
- **Images / multimodal / RAG**: `POST /v1/images/generations`,
  `POST /v1/image/infer`, `POST /v1/custom-multimodal`, `POST /rag/api/prompt`
- **Telemetry** (for testing the metrics scrapers): `GET /metrics`,
  `/vllm/metrics`, `/sglang/metrics`, `/trtllm/metrics`, the `/dynamo_*/metrics`
  family, and synthetic DCGM at `/dcgm1/metrics`, `/dcgm2/metrics`

## Common flags cheat-sheet

| Flag | Default | Purpose |
|---|---|---|
| `-p, --port` | 8000 | Listen port |
| `--host` | 127.0.0.1 | Bind address (`::1`, `0.0.0.0`, …) |
| `-f, --fast` | off | Zero all latency; bypass scheduler + cache |
| `--no-tokenizer` | off | Skip HF tokenizer load (avoids network download) |
| `-w, --workers` | 0 (=nproc) | Tokio worker threads |
| `--processes` | 1 | Spawn N child servers behind an L4 round-robin balancer (0=nproc) |
| `-v, --verbose` | off | DEBUG logging (also `--log-level DEBUG`) |
| `--access-logs` | off | Per-request access logging |
| `--models a,b` | built-in list | Pre-advertise models on `/v1/models` |
| `--error-rate` | 0.0 | Inject a fraction of failed responses |
| `--random-seed` | — | Deterministic latency/jitter/errors |

Every flag has an env-var twin (`MOCK_SERVER_PORT`, `MOCK_SERVER_FAST`, …); set
the log level dynamically with `AIPERF_MOCK_LOG` (a `tracing` env-filter).

## `fastmock` — the ultra-minimal alternative

`rust/mock-server/tools/fastmock.rs` is a single-file, std-only TCP server that
returns one fixed streaming chat response. It is **not** a registered cargo bin —
it's a loose file you compile directly with `rustc`. Use it only when you need
the absolute lowest-overhead loopback target for a transport micro-benchmark and
don't need real routes/latency/token behavior:

```bash
rustc -O rust/mock-server/tools/fastmock.rs -o /tmp/fastmock
/tmp/fastmock 8131                       # 127.0.0.1:8131, one accept thread (baseline)
/tmp/fastmock 8131 --threads 4           # 4 accept threads share one listener, same process
/tmp/fastmock 8131 --procs 4             # 4 SO_REUSEPORT processes on :8131 (kernel-balanced)
/tmp/fastmock 8131 --procs 4 --threads 2 # compose: 4 processes × 2 accept threads
```

`fastmock` scales two ways without any proxy hop (both `0` = auto = CPU count):

- `--threads M` — M concurrent `accept()` threads over one shared listener, lifting
  the single-accept-loop ceiling in one process with zero added latency.
- `--procs N` — N independent server processes bound to the same port via
  `SO_REUSEPORT`; the kernel spreads new connections across them (true
  multi-process sharing, no L4 proxy in the data path — the right fit for a
  lowest-overhead target). The leader supervises the children and, on Linux,
  sets `PR_SET_PDEATHSIG` so they never orphan. Linux-only; elsewhere `--procs`
  degrades to a plain bind.

Prefer these over fronting `fastmock` with the `aiperf-mock-server --processes N`
balancer: that balancer re-execs `aiperf-mock-server` (not `fastmock`) and adds a
byte-splicing hop that undercuts `fastmock`'s whole reason to exist.

For anything realistic — multiple endpoints, latency modeling, telemetry,
token counting — use the full `aiperf-mock-server` server above (which has its own
`--processes N` round-robin balancer).

## Gotchas recap

- **Verify liveness before reporting "running."** Read the log / curl `/health`;
  a bad flag makes the process exit instantly while looking like it's booting.
- **`--no-tokenizer` unless you need token counts** — the default tokenizer
  downloads from Hugging Face on first start.
- **`127.0.0.1`, not `localhost`** — the default bind is IPv4-only.
- **Don't proxy loopback** — if `HTTP_PROXY`/`HTTPS_PROXY` are set in the env, a
  client may route `127.0.0.1` through a proxy that 405s it; set
  `NO_PROXY=127.0.0.1,localhost` for the client if you hit that.
- **`--fast` and `--scheduler-enabled` are mutually exclusive** in effect —
  `--fast` turns the scheduler off.
