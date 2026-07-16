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

## Accuracy dataset mode (ground-truth-aware responses)

By default the mock returns arbitrary corpus text. Point it at an **accuracy
dataset** and it returns the *correct answer* — formatted for the real AIPerf
grader — for a seeded fraction of requests, the rest deliberately wrong. This
lets you drive the whole accuracy pipeline deterministically and offline, and
gives you an oracle for what the run *should* score.

Ground truth never crosses the wire in AIPerf (it lives only in the Python
accuracy worker and is stripped from everything the runner sees), so the mock
loads the dataset itself and keys on the request prompt.

```bash
# dataset.jsonl — one object per line: prompt + gold answer.
#   {"text": "Question 1: pick A, B, C, or D.", "ground_truth": "B", "task": "demo"}
./target/release/aiperf-mock-server --fast --no-tokenizer \
  --accuracy-dataset dataset.jsonl \
  --accuracy-format mmlu \
  --accuracy-correct-rate 0.5 \
  --random-seed 7
```

Drive it with the **same file** as an AIPerf `single_turn` input so the prompts
line up — the `text` field is sent verbatim and extra fields (`ground_truth`,
`task`, …) are ignored by AIPerf:

```bash
aiperf profile --url http://127.0.0.1:8000 --model gpt-4 \
  --endpoint-type chat --streaming \
  --input-file dataset.jsonl --custom-dataset-type single_turn \
  --request-count 12 --export-level raw --random-seed 7
```

### Dataset format (JSONL, one object per row)

- **prompt** (first present): `prompt` | `question` | `input` | `text`
- **gold** (first present): `ground_truth` | `answer` | `gold` | `target`
- optional: `task` (per-task rollup), `format` (per-row grader override),
  `choices` (MC option letters for wrong-answer selection), `match_key`
  (aliases `match`/`key`/`id`) — a stable fragment to match on.

### Answer formats (`--accuracy-format`; per-row `format` overrides)

Match the real graders in `src/aiperf/accuracy/graders/`:

| format | correct answer emitted |
|---|---|
| `mmlu` / `mmlu_pro` | `The answer is (B)` |
| `gsm8k` | `#### 42` |
| `math` | `\boxed{42}` |
| `exact_match` | gold verbatim (strict, case-sensitive) |
| `passthrough` (default) | gold verbatim |

### Correctness / rendering knobs

All seeded by `--random-seed` + the matched row, so a row's verdict is
deterministic and independent of arrival order.

| Flag | Default | Effect |
|---|---|---|
| `--accuracy-correct-rate` | 1.0 | Seeded fraction answered correctly (rest wrong) |
| `--accuracy-cot-rate` | 0.0 | Fraction rendered as chain-of-thought |
| `--accuracy-reasoning-field` | true | CoT in a separate `reasoning_content` field vs inline before the answer |
| `--accuracy-adversarial-rate` | 0.0 | Fraction rendered as a parser-choke shape |
| `--accuracy-match` | substring | Prompt-matching mode (below) |

### Matching (`--accuracy-match`)

All modes whitespace-normalize (collapse runs, trim).

- `exact` — request equals a row key exactly.
- `substring` (default) — exact, then the longest row key *contained in* the
  request (handles few-shot / system-prompt wrapping).
- `exact_ci` / `substring_ci` — case-insensitive variants.

A per-row `match_key` matches a stable fragment of a wrapped prompt — key on
`q_id_4217` while the wire prompt is a big formatted blob. The seeded verdict is
derived from the matched key, so a row's correct/wrong outcome is the same no
matter how its prompt was wrapped.

### Adversarial shapes (`--accuracy-adversarial-rate`)

Reproduce real parser failures from GitHub issues, to stress-test that a run
**survives** them (a brittle parser crashes a worker): reasoning-only content
(#1136), a streaming `object:null` frame before `[DONE]` (#1010), plus
leading-whitespace, wrong-case, trailing-prose, `\boxed{}` wrap, conflicting
answers, and unicode. The run must exit 0 with these on.

### Live tally — the oracle

The mock counts what it actually answered (not a pre-computed estimate):

```bash
curl -s http://127.0.0.1:8000/accuracy | python3 -m json.tool
# {"enabled": true, "matched": 12, "correct": 6, "accuracy": 0.5,
#  "unmatched": 0, "adversarial": 0, "cot": 0, "tasks": {"demo": {...}}}
```

The same tally is on the Prometheus scrape (`GET /metrics`) as
`aiperf_mock_accuracy_*` — `matched_total`, `correct_total`, `incorrect_total`,
`unmatched_total`, `adversarial_total`, `cot_total`, a `ratio` gauge, and
per-task `aiperf_mock_accuracy_task_*{task="…"}`. Compare either against what
AIPerf's own grader reports for the run.

### Verifying at the raw-record level (the definition of done)

With `--export-level raw`, reconstruct each response's content from
`profile_export_raw.jsonl` (`responses[].packets[].value` → SSE
`choices[0].delta.content`, concatenated) and assert it equals the formatted
answer; `reasoning_content` deltas carry CoT. The `/accuracy` `correct`/`matched`
will equal the raw-record counts exactly for a single-pass run with no warmup.
Worked example + assertions: `rust/e2e/tests/test_accuracy_mock.rs`.

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
- **Accuracy tally**: `GET /accuracy` (live correct/matched for an
  `--accuracy-dataset` run; see the accuracy section above)

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
| `--random-seed` | — | Deterministic latency/jitter/errors/accuracy verdicts |
| `--accuracy-dataset <path>` | — | JSONL ground-truth dataset (see Accuracy section) |
| `--accuracy-format` | passthrough | Grader answer format (`mmlu`/`gsm8k`/`math`/…) |
| `--accuracy-correct-rate` | 1.0 | Seeded fraction answered correctly |
| `--accuracy-cot-rate` / `--accuracy-adversarial-rate` | 0.0 | Fraction rendered as CoT / parser-choke shapes |
| `--accuracy-match` | substring | Prompt-matching mode (`exact`/`substring`/`_ci`) |

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

### `fastclient` — a monster load generator for the fast targets

`rust/mock-server/tools/fastclient.rs` is a std-only (`rustc`-compiled) HTTP/1.1
load generator that blazes past the reqwest `examples/loadgen` (which caps ~650k
rps and is itself the bottleneck). Persistent keep-alive connections, response
framing by probed byte length (no per-request parsing) — >2M rps on loopback.

```bash
rustc -O rust/mock-server/tools/fastclient.rs -o /tmp/fastclient
/tmp/fastclient http://127.0.0.1:8131/v1/chat/completions --connections 512 --duration 6
```

`--pipeline` defaults to **1** (one in-flight per connection) — the honest setting,
since real HTTP/1.1 clients don't pipeline; express concurrency via `--connections`.
`--pipeline P>1` measures the server's raw *retirement ceiling* (batches syscalls in
a way real traffic won't) — label it as such, never quote it as client RPS.

### `fastmock-uring` — io_uring engine A/B

`rust/mock-server/tools/fastmock-uring/` is a monoio (io_uring) thread-per-core twin
of `fastmock` for A/B-ing the I/O engine. Result: at realistic `pipeline=1`, io_uring
beats blocking thread-per-connection **+27–54%**; under deep pipelining blocking wins
(unrepresentative). See its `README.md` and the durable finding at
`~/.claude/benchmark-findings/rust-io_uring-monoio-vs-blocking-threadperconn-http.md`.

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
- **Accuracy mode keys on the prompt, not an id.** The mock matches the wire
  prompt against the dataset (normalized exact, or `substring` for wrapped
  prompts). If AIPerf re-templates the prompt beyond whitespace, give each row a
  `match_key` fragment guaranteed to survive in the wire prompt. `--fast` does
  NOT disable accuracy (only latency), so correctness works with instant timing.
