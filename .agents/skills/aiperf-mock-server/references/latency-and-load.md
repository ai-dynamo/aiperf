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
# 4 backend processes behind one round-robin front door on :8000.
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
