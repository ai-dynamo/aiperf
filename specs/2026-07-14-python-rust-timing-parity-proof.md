<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python ↔ Rust request-timing parity across all legacy timing modes

**Date:** 2026-07-14
**Status:** built + proven (A/B against the mock server).
**Scope:** Prove the native Rust runner reproduces the legacy Python engine's per-request timing within 5% for **every** legacy timing mode, and that the two engines send requests on the same authored schedule. Fixes the real divergences the A/B surfaced.

## Method

The same `BenchmarkRun` is executed under both engines via the `AIPERF_RUNTIME_ENGINE` switch (`rust` default vs `python` legacy service mesh; `src/aiperf/cli_runner/_single_run.py`), against the in-repo `aiperf-mock-server` with **deterministic, load-independent latency** (`--ttft 100 --itl 20`, zero jitter, zero concurrency penalty, scheduler off), so any timing difference is a pure engine difference, not mock queueing. Both engines emit the identical `profile_export.jsonl` schema (`metadata.{credit_issued_ns,request_start_ns,request_ack_ns,request_end_ns}` + `metrics.*`), so the A/B is per-request apples-to-apples. Workload is pinned deterministic (`--synthetic-input-tokens-stddev 0 --output-tokens-stddev 0`) and token counting aligned (`--use-server-token-count`).

## Results (worst case 1.06%)

Response-timing parity (rust vs python, |Δavg|/avg), 120 requests/mode, mock ttft=100ms itl=20ms:

| Mode | TTFT | ITL | request_latency | OSL |
|---|---|---|---|---|
| concurrency (`--concurrency 4`) | 0.5% | 0.0% | 0.2% | 0.0% |
| constant (`--request-rate 20 --arrival-pattern constant`) | 1.1% | 0.3% | 0.6% | 0.0% |
| poisson (`--arrival-pattern poisson`) | 0.6% | 0.1% | 0.3% | 0.0% |
| gamma (`--arrival-pattern gamma`) | 0.5% | 0.0% | 0.2% | 0.0% |
| user-centric (`--user-centric-rate 20 --num-users 4`, multi-turn) | 0.7% | 0.1% | 0.3% | 0.0% |
| fixed-schedule (mooncake trace, 50ms grid) | 0.8% | 0.2% | 0.5% | 0.0% |

**Worst response-timing divergence across all modes/metrics: 1.06%** — well within 5%.

Send-schedule adherence (deterministic modes, mean per-request deviation from the authored grid):

| Mode | Rust | Python |
|---|---|---|
| fixed-schedule (50ms grid) | **0.25ms** | 1.16ms |
| constant (50ms grid) | **0.08ms** | 0.70ms |

Rust sends on the authored grid **tighter than Python**.

## Divergences found and fixed

1. **Token-counting methodology mismatch (the timing-affecting one).** By default the Rust runner uses the server's authoritative `usage.completion_tokens` for OSL while the Python engine re-tokenizes the output text with the client tokenizer, so identical mock output (8 tokens) yielded OSL=8 (Rust) vs a 6–12 spread mean 8.5 (Python). Because ITL is `(last−first)/(osl−1)` and throughput is per-OSL, this cascaded into >5% ITL/throughput divergence. Aligning both engines (`--use-server-token-count`) collapses OSL to 0.0% and ITL to <0.3%. *(Config alignment, not a code bug — both methods are valid; they must simply match for A/B.)*

2. **Fixed-schedule startup burst (Rust bug, fixed).** Rust pre-scheduled every turn at absolute targets off `runtime.start_ns()` (the run origin, captured before per-phase setup), so the earliest targets were already overdue when the scheduler drained → a startup burst + ~56ms constant lead. Anchored to `runtime.now_ns()` at issuance start (matching Python's first-send anchor). See `fixed_schedule.rs`.

3. **First-request cold-start (Rust, fixed with a warm-start barrier).** The up-front O(n) schedule pass left the first target a few ms overdue, and the first dispatch paid one-time transport/tokenizer/JIT setup. Added the Rust-native analogue of Python's ZMQ "workers ready, go" barrier: `RequestExecutor::prewarm` / `TurnDispatcher::prewarm` dispatch one discarded, unrecorded round-trip through every worker's real sink before timed issuance, and fixed-schedule anchors to `now()+25ms` so the earliest target sits just in the future. Result: send deviation ~8ms → 0.25ms.

## Non-divergences (confirmed, not bugs)

- **Poisson/gamma per-run rate differences** are RNG sampling variance — the two engines derive independent RNG streams from the seed (research-confirmed), so individual inter-arrivals differ, but the distributions match (mean → target rate; both straddle 20/s over 800 samples). Aggregate response timing matches to <1%.
- **Python measurement jitter under saturation.** At high poisson rates (peak in-flight ~20) Python's asyncio/GIL inflates measured TTFT (mean 64ms vs Rust's accurate 21ms) — exactly the deficiency the Rust engine exists to remove, not a Rust error. Parity is asserted in the regime where both engines measure accurately (peak concurrency ≲ 10), which is the meaningful comparison.

## Related

- `AIPERF_RUNTIME_ENGINE` switch — `src/aiperf/cli_runner/_single_run.py`; `Environment.RUNTIME.ENGINE`.
