<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Latency, saturation, and load knobs

By default (without `--fast`) the mock applies a **closed-form analytic** latency model:
**TTFT 20 ms, ITL 5 ms** per token, per request. `--fast` (`-f`) zeros every latency and
bypasses the scheduler + prefix cache. This reference covers the realistic-timing knobs,
the batched scheduler, the prefix cache, the `--processes` balancer, and timer precision.
Every flag has a `MOCK_SERVER_*` env twin. Source of truth: `rust/mock-server/src/config.rs`.

## Analytic latency model (default; scheduler OFF)

| Flag | Default | Effect |
|---|---|---|
| `--ttft <ms>` (`-t`) | 20.0 | Base first-token latency |
| `--itl <ms>` | 5.0 | Base inter-token latency |
| `--ttft-per-isl-token-ms <ms>` | 0.0 | Prefill cost scaling with prompt length: `TTFT += this × prompt_tokens` (0.05 ≈ 50 ms per 1k input tokens) |
| `--ttft-concurrency-quad-ms <ms>` | 0.0 | Super-linear prefill contention: `TTFT += this × active_inflight²` (analytic only) |
| `--itl-per-osl-token-ms <ms>` | 0.0 | `ITL += this × osl_tokens` (analytic only) |
| `--itl-concurrency-lin-ms <ms>` | 0.0 | `ITL += this × active_inflight` (analytic only) |
| `--ttft-jitter-cv` | 0.0 | Lognormal TTFT jitter (stddev/mean) |
| `--itl-jitter-cv` | 0.0 | Lognormal ITL jitter (stddev/mean) |
| `--random-seed <n>` | — | Deterministic latency/jitter/errors/accuracy verdicts |

For a **deterministic** e2e run, set fixed `--ttft`/`--itl` with both jitter CVs at `0`,
fixed synthetic ISL/OSL stddev `0`, scheduler off, and a pinned tokenizer — then each
record's TTFT ≈ `ttft`, ITL ≈ `itl`, and `request_latency ≈ ttft + (osl-1)·itl` within
transport overhead.

Example — 40 ms TTFT, 8 ms ITL, 10% jitter, seeded:

```bash
./target/release/aiperf-mock-server --no-tokenizer \
  --ttft 40 --itl 8 --ttft-jitter-cv 0.1 --itl-jitter-cv 0.1 --random-seed 1
```

## Batched scheduler (saturation / throughput-vs-concurrency curves)

Enable `--scheduler-enabled` to replace the closed-form model with a step-based batched
scheduler that produces a realistic throughput knee (tok/s that saturates, TTFT that grows
with load). `--fast` disables the scheduler, so don't combine them; the analytic
`*_per_*` / `*_concurrency_*` knobs are ignored while the scheduler is on.

```bash
./target/release/aiperf-mock-server --no-tokenizer \
  --scheduler-enabled \
  --scheduler-max-batch-size 256 \
  --scheduler-max-prefill-chunks-per-step 8
```

The knee lands near concurrency ≈ `--scheduler-max-batch-size`. Prefill becomes the binding
constraint as you lower `--scheduler-max-prefill-chunks-per-step`.

| Flag | Default | Effect |
|---|---|---|
| `--scheduler-enabled` | off | Turn on the step-based batched scheduler |
| `--scheduler-step-ms` | 5.0 | Virtual decode-step cadence (ms); each step admits up to max-batch decode tokens |
| `--scheduler-max-batch-size` | 256 | Concurrent decoders per step (effective batch); saturation knee lands near this |
| `--scheduler-max-prefill-chunks-per-step` | 8 | Prefill chunks admitted per step; lower ⇒ TTFT cliffs under concurrent arrivals |
| `--scheduler-prefill-chunk-tokens` | 512 | Tokens per prefill chunk; a P-token prompt needs `ceil(P/chunk)` chunks |
| `--scheduler-prefill-chunks-per-request` | 0 | Fixed prefill chunks per request (overrides ISL-derived); makes TTFT independent of prompt length. 0 = derive from ISL |
| `--scheduler-prefill-work-cv` | 0.0 | Per-request lognormal CV on prefill chunk count; spreads queue-wait/TTFT request-to-request (TTFT CV ≈ this) |
| `--scheduler-admit-jitter-cv` | 0.0 | Per-step lognormal CV on decode/prefill admit budgets; adds throughput burstiness |
| `--scheduler-prefill-throughput-exponent` | 0.0 | Sublinear prefill throughput; TTFT ~ C^(1-exponent). `exponent = 1 - log(ttft2/ttft1)/log(c2/c1)` |
| `--scheduler-prefill-throughput-ref` | 0 | Reference occupancy where the prefill budget equals base (0 = use max-batch-size); only with exponent > 0 |
| `--scheduler-goodput-collapse-enabled` | off | Model goodput collapse: past the knee, admit budget shrinks so aggregate tok/s drops (preemption/thrash) |
| `--scheduler-goodput-collapse-threshold` | 1.5 | Overload ratio (queue_len / max_batch) where collapse begins |
| `--scheduler-goodput-collapse-slope` | 0.5 | How fast the effective batch shrinks past threshold |
| `--scheduler-goodput-collapse-floor` | 0.3 | Floor fraction of max-batch-size under full collapse |

## Prefix (KV) cache

By default the mock models radix prefix caching like SGLang (ON by default): a prompt's
leading blocks that match a previously-seen prefix skip prefill and are reported as
`usage.prompt_tokens_details.cached_tokens`. Hits occur only on genuinely shared prefixes
(multi-turn history, shared system prompts). `--fast` disables it.

| Flag | Default | Effect |
|---|---|---|
| `--disable-prefix-cache` | off | Disable content-addressed KV-cache reuse (mirrors SGLang `--disable-radix-cache`) |
| `--prefix-cache-block-tokens` | 1 | Tokens per cache block (matching granularity); SGLang `page_size=1` default |
| `--prefix-cache-capacity-blocks` | 500000 | Cache capacity in tokens (LRU window); mirrors SGLang `max_total_num_tokens` |
| `--prefix-cache-hit-rate` | 0.0 | Force this fraction of every prompt served from cache, bypassing content addressing |
| `--prefix-cache-latency-aware` | off | Let hits reduce prefill/TTFT (OFF: reported in usage but does not move latency, matching saturated queue-bound regimes) |
| `--prefix-cache-eviction-policy` | lru | Eviction at capacity: `lru`/`lfu`/`fifo`/`mru`/`filo`/`priority`/`slru` (only observable under capacity pressure; `priority` needs a per-request `priority` field) |

## Per-endpoint (non-LLM) latency

| Flag | Default | Effect |
|---|---|---|
| `--embedding-base-latency` | 10.0 | Base ms for `/v1/embeddings` |
| `--embedding-per-input-latency` | 2.0 | ms per embedding input |
| `--ranking-base-latency` | 10.0 | Base ms for rerank/ranking |
| `--ranking-per-passage-latency` | 1.0 | ms per ranked passage |
| `--image-retrieval-base-latency` | 10.0 | Base ms for `/v1/image/infer` |
| `--image-retrieval-per-image-latency` | 5.0 | ms per retrieved image |

## Fetching content-server URLs (`--fetch-content-urls`)

By default the mock treats `image_url` / `video_url` values as **opaque strings** — it never
dials out, so an AIPerf content server (`AIPERF_CONTENT_SERVER_*`, which rewrites generated
media to `http://host:8090/content/...` URLs) is never actually hit, and its serving /
transfer-record path stays cold. `--fetch-content-urls` (env `MOCK_SERVER_FETCH_CONTENT_URLS`,
default **off**) makes the mock actually HTTP-GET those URLs so the content server is exercised
end to end.

| Flag | Default | Effect |
|---|---|---|
| `--fetch-content-urls` | `false` | GET `http(s)` `image_url`/`video_url` targets instead of ignoring them |
| `--content-fetch-timeout` | `30.0` | Per-request fetch timeout (seconds) |

Behavior when enabled:

- **`/v1/chat/completions`** — every `image_url`/`video_url` part (OpenAI string or `{url}` form)
  is fetched **concurrently** before latency simulation. `data:` URIs and non-`http(s)` schemes
  are skipped.
- **`/v1/image/infer`** (`image_retrieval`) — each `input[].url` is fetched and the **real
  downloaded byte count** feeds `usage.images_size_mb` (default-off keeps the old base64
  string-length proxy).
- Fetches are best-effort: any parse error, connect/transfer failure, or timeout is logged
  (`content fetch ...` at DEBUG/WARN) and counts as 0 bytes — a fetch **never** fails the mock
  response.
- Downloaded volume is exposed as Prometheus `aiperf_mock_content_bytes_fetched_total{endpoint}`.

**HTTP only.** The fetch client is hyper `client-legacy` over a plain `HttpConnector` (no TLS
stack, no proxy) — deliberate, to keep a second crypto provider out of the binary and because the
content server serves plain HTTP. `https://` targets will fail-and-log rather than download.
`--fast` is orthogonal (it zeros latencies, not network I/O); the flag defaults off so `--fast`
runs are unaffected unless you opt in.

Quick check:

```bash
MOCK_SERVER_FETCH_CONTENT_URLS=true ./target/release/aiperf-mock-server --no-tokenizer &
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":[
       {"type":"image_url","image_url":{"url":"http://HOST:8090/content/images/img_000001.png"}}]}]}' >/dev/null
curl -s localhost:8000/metrics | grep content_bytes_fetched  # > 0 after a fetch
```

### End-to-end with the content server and media-fetch metrics

This flag is what lets the mock stand in for a VLM server that fetches images, so AIPerf's
request-correlated media-fetch metrics (`time_to_media_fetch`, `media_serving_latency`,
`media_fetch_count`, ...) have something to measure. Full validated recipe:

```bash
CS=/tmp/aiperf-content; ART=/tmp/aiperf-e2e; mkdir -p "$CS" "$ART"
MOCK_SERVER_FETCH_CONTENT_URLS=true MOCK_SERVER_PORT=8300 \
  ./target/release/aiperf-mock-server --no-tokenizer &

AIPERF_CONTENT_SERVER_ENABLED=true AIPERF_CONTENT_SERVER_HOST=127.0.0.1 \
AIPERF_CONTENT_SERVER_PORT=8190 AIPERF_CONTENT_SERVER_CONTENT_DIR="$CS" \
./target/release/aiperf profile --model-names test --url http://127.0.0.1:8300 \
  --endpoint-type chat --image-width-mean 48 --image-height-mean 48 --image-batch-size 2 \
  --request-count 10 --concurrency 2 --artifact-dir "$ART" --tokenizer gpt2
```

Then: `$ART/media_records.jsonl` has one line per fetch (`rid`/`mi`/`td` + timings), and
`$ART/native-v2.json` `media_metrics` holds the six distributions. With `--image-batch-size 2`,
`media_fetch_count` avg is `2.0` and each request's records carry `mi` `0` and `1` — the
multi-media-per-turn disambiguation. `aiperf` logs `media-fetch metrics finalized total_fetches=N
unmatched=0`. Note `AIPERF_CONTENT_SERVER_PORT` (media URL origin) must differ from the mock's
port, and the content dir must exist before the run.

## Multi-process load balancer (`--processes N`)

A single server process shares one tokio runtime; at very high request rates the runtime
scheduler / allocator can become the ceiling. `--processes N` (N > 1) turns the launched
binary into a **lightweight L4 (TCP) round-robin load balancer**: it binds the public
`--host:--port`, spawns `N` child `aiperf-mock-server` processes (same binary, exact same
config) on internal loopback ports, and splices each accepted connection to the next backend
in rotation. The client sees the identical OpenAI-compatible frontend on **one URL** —
HTTP/1.1 keep-alive, HTTP/2, and SSE all pass through untouched (the balancer never parses
HTTP).

```bash
# 4 backend processes behind one round-robin entry point on :8000.
./target/release/aiperf-mock-server --processes 4 --no-tokenizer --port 8000
# --processes 0 = auto = one process per CPU.
```

- **When to use it:** you have saturated a single mock process (CPU-bound on one runtime)
  and want more aggregate throughput on a many-core box.
- **Worker threads auto-divide:** with `--workers` on its default (auto), each child gets
  `max(1, nproc / processes)` tokio workers, so the total worker count stays bounded rather
  than `N × nproc`. An explicit `--workers` is honored per-child.
- **Round-robin is per connection, not per request** (the cheapest, HTTP-blind distribution).
  A benchmark driving concurrency `C` opens ~`C` keep-alive connections, which spread evenly
  across the `N` backends as long as `C >= N`; below that some backends idle.
- **Lifecycle:** the balancer health-gates every child before opening the public port, tears
  everything down on Ctrl-C, and fails fast if any child dies. On Linux children also get
  `PR_SET_PDEATHSIG`. `--processes 1` (default) is the unchanged single-process path.
- **HTTP-only:** `--grpc-port` and `--uds` are warned-and-skipped under `--processes > 1`
  (the balancer is TCP/HTTP-only). See `references/grpc-and-transports.md`.

## Timer precision (timerfd)

Latency injection (TTFT/ITL pacing, scheduler step cadence) runs on the `aiperf` `RealClock`
backend: waits use a `CLOCK_MONOTONIC` **`timerfd`** awaited through tokio's IO reactor
(`aiperf::clock::sleep_ns`), giving nanosecond resolution instead of `tokio::time`'s ~1 ms
timer wheel (which would quantize a 5 ms ITL by ~20%). Matters whenever you set sub-10 ms
`--itl`/`--ttft` or a small `--scheduler-step-ms`; transparent otherwise. Non-Linux platforms
fall back to `tokio::time` (coarser).
