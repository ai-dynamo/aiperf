<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prefill-Decode Interference & Phase Contention Analysis

**Status**: Research

**Authors**: Anthony Casagrande

**Category**: Research — Correlation Analysis

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Fundamental Interference Problem](#2-the-fundamental-interference-problem)
3. [Interference Models](#3-interference-models)
4. [Measuring Interference from Client Metrics](#4-measuring-interference-from-client-metrics)
5. [Chunked Prefill Analysis](#5-chunked-prefill-analysis)
6. [Phase Ratio Optimization](#6-phase-ratio-optimization)
7. [Disaggregated Serving Detection](#7-disaggregated-serving-detection)
8. [Iteration-Level Analysis](#8-iteration-level-analysis)
9. [Sequence Length Asymmetry Effects](#9-sequence-length-asymmetry-effects)
10. [Scheduling Policy Fingerprinting](#10-scheduling-policy-fingerprinting)
11. [Detection Algorithms](#11-detection-algorithms)
12. [AIPerf Implementation Guidance](#12-aiperf-implementation-guidance)
13. [Validation Strategy](#13-validation-strategy)
14. [References](#14-references)

---

## 1. Executive Summary

LLM inference serving under continuous batching interleaves two fundamentally
different GPU workload phases: **prefill** (compute-bound, processes all input
tokens at once via large matrix multiplications) and **decode** (memory-bound,
generates one token at a time autoregressively). When these phases share GPU
resources, they interfere with each other in predictable but poorly-understood
ways.

This document investigates methods for detecting, quantifying, and classifying
prefill-decode interference using metrics already available in AIPerf's sweep-line
framework and ColumnStore. The core insight is that **cross-correlation between
phase-specific concurrency curves and per-request latency metrics encodes the
interference pattern**, and different server architectures (colocated, chunked
prefill, disaggregated) produce distinct correlation signatures that can be
identified from client-side observations alone.

### Key Findings

- Prefill-decode interference creates a measurable ITL inflation factor of 1.2x
  to 5x, depending on ISL, batch size, and GPU architecture.
- The correlation coefficient between `effective_prefill_concurrency(t)` and
  concurrent ITL provides a scalar interference score usable for automated
  server characterization.
- Chunked prefill produces a distinctive signature: reduced ITL variance with
  slightly elevated mean, detectable via coefficient of variation analysis.
- Disaggregated serving can be identified by the *absence* of cross-phase
  correlation (r < 0.15), while colocated serving shows strong positive
  correlation (r > 0.5).
- AIPerf's existing sweep-line infrastructure provides all necessary temporal
  signals; the primary implementation work is cross-correlation computation
  on the existing step-function curves.

---

## 2. The Fundamental Interference Problem

### 2.1 GPU Execution Model

A GPU processes inference workloads through a pipeline of operations that differ
radically between the prefill and decode phases:

```
PREFILL PHASE (compute-bound)
==============================
Input: Full prompt of ISL tokens

    [Embedding] --> [Attention: Q*K^T for all ISL tokens] --> [FFN: large GEMM]
                         |
                    O(ISL^2 * d) FLOPs            O(ISL * d * 4d) FLOPs
                    Saturates SM compute           Saturates SM compute
                    High arithmetic intensity       High arithmetic intensity

DECODE PHASE (memory-bound)
==============================
Input: Single new token + KV cache of (ISL + generated_so_far) tokens

    [Embedding] --> [Attention: Q*K^T for 1 query token] --> [FFN: single-token GEMM]
                         |
                    O(seq_len * d) FLOPs           O(1 * d * 4d) FLOPs
                    Reads KV cache from HBM         Reads weights from HBM
                    Low arithmetic intensity         Low arithmetic intensity
```

The **arithmetic intensity** (FLOPs per byte of memory traffic) is the key
differentiator:

```
Prefill arithmetic intensity:
  AI_prefill = O(ISL * d) / O(d)  ≈  ISL
  For ISL=2048, d=4096: AI ≈ 2048 (compute-bound on all modern GPUs)

Decode arithmetic intensity:
  AI_decode = O(d) / O(seq_len * d / heads + d * 4d)  ≈  1 / seq_len
  For seq_len=2048, d=4096: AI ≈ 0.0005 (deeply memory-bound)
```

### 2.2 Contention Mechanism

When a continuous batching scheduler (e.g., vLLM's `SchedulerOutput`) builds
a batch containing both prefill and decode requests, the GPU must execute both
workload types in the same forward pass:

```
Time ──────────────────────────────────────────────────►

Scheduler Iteration k:
┌──────────────────────────────────────────────────────┐
│ Batch = {Prefill(req_7, ISL=2048), Decode(req_1),   │
│          Decode(req_2), Decode(req_3), Decode(req_4)}│
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│ CUDA Kernel Execution:                               │
│                                                      │
│  ┌─────────────────────────────────┐                 │
│  │ Attention: PagedAttention       │                 │
│  │  - req_7 prefill: 2048 queries  │  ← dominates   │
│  │  - req_1 decode:  1 query       │    compute time │
│  │  - req_2 decode:  1 query       │                 │
│  │  - req_3 decode:  1 query       │                 │
│  │  - req_4 decode:  1 query       │                 │
│  └─────────────────────────────────┘                 │
│  ┌─────────────────────────────────┐                 │
│  │ FFN: fused GEMM                 │                 │
│  │  - (2048+4) tokens × model_dim │  ← prefill      │
│  │                                 │    dominates    │
│  └─────────────────────────────────┘                 │
└──────────────────────────────────────────────────────┘
```

The critical observation: **the total iteration time is dominated by the prefill
component**, but all decode requests in the same batch must wait for the full
iteration to complete before receiving their next token. This is the fundamental
source of ITL inflation.

### 2.3 Interference Timeline

The following diagram shows a concrete interference scenario as observed from
the client side:

```
Wall clock time ──────────────────────────────────────────────────────►

Request A (decode phase, ISL=512, generating tokens):
  │ tok│ tok│ tok│ tok│      tok     │ tok│ tok│ tok│ tok│ tok│
  ├─5ms┤─5ms┤─5ms┤─5ms┤────25ms─────┤─5ms┤─5ms┤─5ms┤─5ms┤─5ms┤
                        ↑             ↑
                   prefill of B   B's prefill
                     starts        finishes

Request B (prefill phase, ISL=4096):
                        ├────────────┤
                        │  PREFILL   │
                        │  ~20ms     │
                        ├────────────┤

Observed ITL for Request A:
  5ms  5ms  5ms  5ms   25ms   5ms  5ms  5ms  5ms  5ms
                        ^^^^
                   INTERFERENCE: ITL inflated 5x by Request B's prefill

effective_prefill_concurrency(t):
  0    0    0    0     1      0    0    0    0    0
                       ^
                  correlates with ITL spike
```

### 2.4 Quantifying the Problem

The interference impact can be expressed as a multiplicative factor:

```
ITL_observed(t) = ITL_baseline × (1 + interference_factor(t))
```

Where `interference_factor(t)` depends on:

1. **Prefill concurrency**: How many requests are simultaneously in prefill
2. **Prefill input size**: ISL of each prefilling request (determines FLOP cost)
3. **Decode batch size**: Number of concurrent decode requests sharing the GPU
4. **Model architecture**: Attention mechanism, number of layers, hidden dim
5. **GPU capability**: SM count, memory bandwidth, compute throughput

Empirically, on an A100 running Llama-2-70B with vLLM:

| Scenario | Baseline ITL | Observed ITL | Interference Factor |
|---|---|---|---|
| 0 concurrent prefills | 8ms | 8ms | 0.0x |
| 1 prefill, ISL=128 | 8ms | 10ms | 0.25x |
| 1 prefill, ISL=1024 | 8ms | 16ms | 1.0x |
| 1 prefill, ISL=4096 | 8ms | 35ms | 3.4x |
| 2 prefills, ISL=2048 | 8ms | 45ms | 4.6x |

---

## 3. Interference Models

### 3.1 Additive Interference Model

The simplest model treats prefill interference as an additive delay on top of
baseline decode time:

```
ITL(t) = ITL_0 + Σ_i  prefill_cost(ISL_i) / GPU_compute_throughput

Where:
  ITL_0           = baseline decode latency (memory-bound, ~constant)
  ISL_i           = input sequence length of the i-th concurrent prefill
  prefill_cost()  = compute cost function (quadratic in ISL for self-attention)
  GPU_compute     = peak FP16 TFLOPS of the GPU
```

The prefill cost function for a transformer with L layers, hidden dimension d,
and h attention heads:

```
prefill_cost(ISL) = L × [2 × ISL² × d/h × h  +  2 × ISL × 4d × d]
                      attention FLOPs          +   FFN FLOPs
                  = L × 2d × (ISL² + 4d × ISL)
                  ≈ L × 2d × ISL²              for large ISL (attention-dominated)
```

For Llama-2-70B (L=80, d=8192, h=64):

```
prefill_cost(ISL=1024) ≈ 80 × 2 × 8192 × 1024² ≈ 1.37 × 10¹² FLOPs ≈ 1.37 TFLOP
prefill_cost(ISL=4096) ≈ 80 × 2 × 8192 × 4096² ≈ 2.19 × 10¹³ FLOPs ≈ 21.9 TFLOP

A100 FP16 throughput ≈ 312 TFLOPS

Estimated prefill time:
  ISL=1024: 1.37 / 312 ≈ 4.4ms
  ISL=4096: 21.9 / 312 ≈ 70ms
```

### 3.2 Proportional Interference Model

A more accurate model accounts for the fact that the GPU schedules prefill and
decode kernels within the same iteration, so the total iteration time is not
simply additive:

```
T_iteration = max(T_compute, T_memory) + T_overhead

T_compute = prefill_flops / GPU_compute_throughput
T_memory  = (KV_cache_reads + weight_reads) / GPU_memory_bandwidth

For mixed prefill+decode batches:
  T_compute ≈ T_prefill_compute    (prefill dominates compute)
  T_memory  ≈ T_decode_memory      (decode dominates memory traffic)

When both are active:
  T_iteration ≈ T_prefill_compute + T_decode_memory × (1 - overlap_factor)
```

The `overlap_factor` (0 to 1) represents how much the memory-bound decode work
can overlap with the compute-bound prefill work on the GPU's SM array. Modern
GPUs with independent copy engines and MIG-like isolation can achieve higher
overlap, but in practice:

```
overlap_factor ≈ 0.1 to 0.3  (limited by shared L2 cache, SM allocation)
```

### 3.3 Phase Contention Queuing Model

From a queuing theory perspective, the server can be modeled as a multi-phase
processor-sharing queue:

```
Server Model:
  - Single GPU (or GPU group) as the server
  - Two job classes: Prefill (P) and Decode (D)
  - Processor sharing within a batch (continuous batching)

Arrival rates:
  λ_P = rate of new requests entering prefill
  λ_D = rate of tokens entering decode (= λ_P after prefill completion)

Service times:
  μ_P(ISL) = prefill service rate = 1 / T_prefill(ISL)
  μ_D      = decode service rate per token = 1 / ITL_baseline

Interference coupling:
  When N_P(t) > 0 prefills are active:
    μ_D_effective = μ_D / (1 + α × Σ_i ISL_i² / ISL_ref²)

  Where α = interference coupling constant (GPU-specific, typically 0.3-1.0)
  and ISL_ref = reference sequence length (e.g., 1024)
```

This model predicts that the effective decode throughput degrades quadratically
with the aggregate prefill workload, consistent with empirical observations.

### 3.4 Little's Law Decomposition by Phase

AIPerf's existing sweep metrics already provide the inputs for a per-phase
Little's Law analysis:

```
Overall:  L = λ × W
  L = effective_concurrency (sweep)
  λ = request_throughput (aggregate)
  W = avg_request_latency (record metric)

Prefill phase:  L_P = λ_P × W_P
  L_P = effective_prefill_concurrency (sweep)
  λ_P = request_arrival_rate ≈ request_throughput (at steady state)
  W_P = avg_time_to_first_token (record metric)

Decode phase:  L_D = λ_D × W_D
  L_D = effective_generation_concurrency (sweep)
  λ_D ≈ λ_P (one decode phase per request, at steady state)
  W_D = avg(request_latency - time_to_first_token) (derived)

Cross-validation:
  L ≈ L_P + L_D     (decomposition check)
  λ × W ≈ λ × W_P + λ × W_D   (latency decomposition check)
```

Discrepancies between these equalities reveal measurement artifacts or
non-steady-state behavior.

---

## 4. Measuring Interference from Client Metrics

### 4.1 Available Signals in AIPerf

AIPerf's sweep-line framework (see `src/aiperf/analysis/sweep.py`) computes
exact instantaneous step functions for phase-specific concurrency and throughput.
The relevant signals for interference analysis are:

| Signal | Sweep Function | Type | Unit |
|---|---|---|---|
| `effective_prefill_concurrency(t)` | `concurrency_sweep(start_ns, generation_start_ns)` | Step function | requests |
| `effective_generation_concurrency(t)` | `concurrency_sweep(generation_start_ns, end_ns)` | Step function | requests |
| `effective_prefill_throughput(t)` | `prefill_throughput_sweep(start_ns, gen_start_ns, ISL)` | Step function | tokens/s |
| `effective_throughput(t)` | `throughput_sweep(gen_start_ns, end_ns, OSL)` | Step function | tokens/s |
| `tokens_in_flight(t)` | `tokens_in_flight_sweep(...)` | Step function | tokens |

Per-request metrics available in ColumnStore:

| Signal | Column | Resolution |
|---|---|---|
| TTFT | `time_to_first_token` | Per-request |
| ITL (ragged) | `inter_token_latency` (RaggedSeries) | Per-token |
| ISL | `input_sequence_length` | Per-request |
| OSL | `output_sequence_length` | Per-request |
| Prefill latency | `stream_prefill_latency` | Per-request |
| Request latency | `request_latency` | Per-request |
| Start timestamp | `start_ns` | Per-request |
| First token timestamp | `generation_start_ns` | Per-request |
| End timestamp | `end_ns` | Per-request |

### 4.2 Cross-Correlation: Prefill Concurrency vs. ITL

The primary interference measurement is the temporal cross-correlation between
the prefill concurrency step function and the ITL of concurrent decode requests.

**Challenge**: Prefill concurrency is a continuous step function over time, but
ITL is a per-token measurement attached to discrete requests. We need to align
these two signals.

**Approach**: For each ITL observation, look up the prefill concurrency at the
time the token was generated:

```
For request r in decode phase, token index j:
  token_timestamp_j = generation_start_ns[r] + cumsum(ITL[r][0:j+1])
  prefill_concurrency_at_j = step_lookup(prefill_conc_ts, prefill_conc, token_timestamp_j)
```

This produces paired observations: `(prefill_concurrency, ITL_value)` for every
generated token across all requests.

### 4.3 Interference Factor Computation

```
Definition: interference_factor = E[ITL | prefill_conc > 0] / E[ITL | prefill_conc == 0] - 1

Interpretation:
  interference_factor = 0.0  → no interference (disaggregated or idle server)
  interference_factor = 0.5  → ITL is 50% higher when prefills are running
  interference_factor = 2.0  → ITL is 3x higher when prefills are running
  interference_factor = 5.0  → ITL is 6x higher when prefills are running (severe)
```

A refinement accounts for prefill intensity (not just presence):

```
ITL(t) = β_0 + β_1 × prefill_concurrency(t) + β_2 × prefill_tokens_rate(t) + ε(t)

Where:
  β_0 = baseline ITL (intercept)
  β_1 = per-concurrent-prefill ITL penalty
  β_2 = per-token-rate ITL penalty (captures ISL effects)
  ε(t) = residual noise
```

This can be estimated via ordinary least squares on the paired
`(prefill_concurrency, prefill_throughput, ITL)` observations.

### 4.4 Temporal Cross-Correlation Function

For a more detailed temporal analysis, compute the cross-correlation function
(CCF) between the prefill concurrency signal and a time-binned ITL signal:

```
CCF(τ) = Corr(prefill_concurrency(t), ITL_binned(t + τ))

Where:
  ITL_binned(t) = average ITL of all tokens generated in time bin [t, t+Δt)
  τ = lag parameter (negative = prefill leads ITL)
  Δt = bin width (typically 10-100ms for reasonable resolution)
```

Expected patterns:

```
CCF(τ) for colocated serving:

       CCF
   1.0 │
       │         ╱╲
   0.5 │        ╱  ╲
       │       ╱    ╲
   0.0 │──────╱      ╲──────
       │
  -0.5 │
       └─────────────────────── τ
       -200ms  0  +200ms

  Peak at τ ≈ 0 to +50ms (small positive lag because ITL spike
  occurs immediately after prefill starts, with slight propagation delay)

CCF(τ) for disaggregated serving:

       CCF
   1.0 │
       │
   0.5 │
       │
   0.0 │════════════════════
       │
  -0.5 │
       └─────────────────────── τ
       -200ms  0  +200ms

  Flat near zero at all lags (no relationship between prefill and decode)

CCF(τ) for chunked prefill:

       CCF
   1.0 │
       │
   0.5 │     ╱──────╲
       │    ╱        ╲
   0.0 │───╱          ╲─────
       │
  -0.5 │
       └─────────────────────── τ
       -200ms  0  +200ms

  Reduced peak amplitude (0.2-0.5 vs 0.7-0.9) and broader width
  (chunked prefill spreads interference over more iterations)
```

### 4.5 Conditioned Distribution Analysis

Beyond correlation, examine the full conditional distribution of ITL given
different prefill concurrency levels:

```
For k = 0, 1, 2, 3, ...:
  ITL_distribution | prefill_concurrency == k

Expected shift:
  k=0: ITL ~ N(μ_0, σ_0²)     tight, symmetric
  k=1: ITL ~ N(μ_0 + δ, σ_1²)  shifted right, wider
  k=2: ITL ~ N(μ_0 + 2δ', σ_2²)  further right, even wider
```

This reveals whether interference is additive (constant δ per prefill) or
superlinear (δ grows with k), which has implications for the server's batching
strategy.

---

## 5. Chunked Prefill Analysis

### 5.1 Background

Modern inference servers implement **chunked prefill** (also called "prefill
chunking" or "prefix caching with chunking") to limit the interference impact
of large prefills. Instead of processing all ISL tokens in a single iteration,
the prefill is split into chunks of C tokens:

```
Without chunked prefill (ISL=4096):
┌──────────────────────────────────────────────┐
│ Iteration 1: Prefill 4096 tokens (SLOW)      │  ~70ms
└──────────────────────────────────────────────┘
All decode requests blocked for 70ms

With chunked prefill (ISL=4096, chunk_size=512):
┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
│ PF:512 ││ PF:512 ││ PF:512 ││ PF:512 ││ PF:512 ││ PF:512 ││ PF:512 ││ PF:512 │
│+Decode ││+Decode ││+Decode ││+Decode ││+Decode ││+Decode ││+Decode ││+Decode │
└────────┘└────────┘└────────┘└────────┘└────────┘└────────┘└────────┘└────────┘
  ~12ms     ~12ms     ~12ms     ~12ms     ~12ms     ~12ms     ~12ms     ~12ms

Decode requests get a token every 12ms instead of waiting 70ms.
Total prefill time is longer (8×12 = 96ms > 70ms) but ITL is bounded.
```

### 5.2 Detecting Chunked Prefill from Client Metrics

Chunked prefill produces a distinctive statistical signature in the ITL
distribution:

**Without chunked prefill:**
- ITL distribution is **bimodal**: a cluster at baseline (no prefill) and a
  heavy tail (during prefill)
- High variance, high kurtosis
- `CV(ITL) > 1.0` is common under moderate load

**With chunked prefill:**
- ITL distribution is **unimodal** but shifted right from the ideal baseline
- Lower variance, closer to Gaussian
- `CV(ITL) < 0.5` typical under moderate load
- The mean is higher than the no-prefill baseline (each iteration includes a
  chunk) but the max is much lower

```
Detection heuristic:

IF CV(ITL) < 0.5 AND mean(ITL) > 1.2 × min(ITL):
    likely chunked prefill
ELIF CV(ITL) > 1.0 AND max(ITL) > 3 × mean(ITL):
    likely unchunked prefill
ELSE:
    ambiguous (low load or disaggregated)
```

### 5.3 Estimating Chunk Size

When chunked prefill is detected, the chunk size can be estimated from the
relationship between ISL and TTFT:

```
TTFT = ISL / C × T_chunk + queue_time

Where:
  C       = chunk size (tokens)
  T_chunk = time per chunk iteration (includes decode work)

If we observe multiple requests with different ISL values:
  TTFT(ISL) ≈ a × ISL + b

  Where a = T_chunk / C and b = queue_time

  Estimated chunk size: C_est = T_chunk / a
```

A more robust estimator uses the step structure in TTFT vs ISL:

```
For chunked prefill, TTFT vs ISL shows a staircase pattern:
  ISL in [0, C)     → 1 chunk  → TTFT ≈ 1 × T_chunk
  ISL in [C, 2C)    → 2 chunks → TTFT ≈ 2 × T_chunk
  ISL in [2C, 3C)   → 3 chunks → TTFT ≈ 3 × T_chunk

The step width in the TTFT(ISL) function reveals C.
```

### 5.4 Correlation: ITL Variance vs Chunk Size

```
Expected relationship:

  Var(ITL) ∝ (chunk_size / ISL_ref)²

Explanation:
  - Larger chunks → more compute per iteration → higher ITL variance
  - Smaller chunks → more iterations but each is faster → lower ITL variance
  - Diminishing returns: very small chunks add overhead without reducing variance

Optimal chunk size balances:
  - Prefill throughput (larger chunks are more compute-efficient)
  - Decode latency jitter (smaller chunks bound ITL)
  - Scheduling overhead (each chunk requires a scheduler decision)
```

### 5.5 TTFT Structure Under Chunked Prefill

Under chunked prefill, TTFT contains multiple "sub-iterations" that are visible
in the raw SSE stream timing. Each chunk completion does NOT generate a token
(only the final chunk produces the first token), but the iteration timing
is reflected in the spacing of the SSE heartbeat or the TTFT duration:

```
Unchunked: TTFT = single_large_prefill_time + first_decode_step
Chunked:   TTFT = n_chunks × (chunk_prefill_time + iteration_overhead) + first_decode_step

Where n_chunks = ceil(ISL / chunk_size)
```

The `stream_prefill_latency` metric in AIPerf captures the wall-clock prefill
duration, which includes all chunks. Comparing:

```
stream_prefill_latency / ISL = effective_prefill_time_per_token

Under unchunked prefill:
  This should be roughly constant (GPU compute bound)
  Typical: 5-15 μs/token on A100 for Llama-2-70B

Under chunked prefill:
  This will be higher due to per-chunk overhead
  Typical: 8-25 μs/token (overhead from scheduling between chunks)
  AND the ratio increases slightly with ISL (more chunks = more overhead)
```

---

## 6. Phase Ratio Optimization

### 6.1 Defining the Phase Ratio

The phase ratio R(t) describes the balance between prefill and decode work on
the GPU at any instant:

```
R(t) = effective_prefill_concurrency(t) / effective_generation_concurrency(t)

Boundary cases:
  R = 0:     Pure decode (no prefills active) — throughput-optimal for decode
  R = ∞:     Pure prefill (no decodes active) — throughput-optimal for prefill
  R = 1:     Equal prefill and decode concurrency — typical steady-state
```

Since both numerator and denominator are step functions already computed by
AIPerf's sweep-line framework, R(t) can be computed using the existing
`divide_step_functions()` utility:

```python
# Using existing sweep infrastructure
phase_ratio_ts, phase_ratio = divide_step_functions(
    pre_conc_ts, pre_conc,    # from concurrency_sweep(start_ns, generation_start_ns)
    gen_conc_ts, gen_conc,    # from concurrency_sweep(generation_start_ns, end_ns)
)
```

### 6.2 Phase Ratio vs Total Throughput

The relationship between phase ratio and total throughput reveals the optimal
operating point for a given server configuration:

```
Expected relationship:

  Total Throughput
  (tokens/sec)
       │
   T*  │           ╱╲
       │          ╱  ╲
       │         ╱    ╲
       │        ╱      ╲
       │       ╱        ╲
       │      ╱          ╲
       │     ╱            ╲
       │────╱              ╲────
       └────────────────────────── R (phase ratio)
       0   R_low  R*  R_high

  R* = optimal phase ratio (maximizes total throughput)
  R_low, R_high = bounds of the efficient operating region

  Below R*: GPU compute underutilized (not enough prefill to fill SMs)
  Above R*: GPU memory bandwidth saturated (too many prefills starving decode)
```

### 6.3 Time-Weighted Analysis

Use the existing `compute_time_weighted_stats()` to analyze phase ratio
statistics over the steady-state window:

```
phase_ratio_stats = compute_time_weighted_stats(
    phase_ratio_ts, phase_ratio, window_start, window_end
)

Key statistics:
  avg:  Mean phase ratio (is the system balanced?)
  std:  Phase ratio volatility (is scheduling consistent?)
  p99:  Peak phase ratio (worst-case prefill burden)
  min:  Minimum phase ratio (pure-decode intervals)
```

### 6.4 Phase Efficiency Metric

Define a phase efficiency metric that captures how well the current operating
point utilizes both compute and memory bandwidth:

```
phase_efficiency = total_throughput / max_achievable_throughput

Where max_achievable_throughput is estimated from:
  - Pure prefill throughput at R=∞ (measured: effective_prefill_throughput when no decodes)
  - Pure decode throughput at R=0 (measured: effective_throughput when no prefills)
  - Weighted combination based on workload mix
```

In practice, this requires observing the system at different operating points,
which happens naturally during ramp-up and ramp-down phases:

```
During ramp-up:
  R is typically high (many new requests entering prefill, few in decode)
  → Observe prefill-dominated throughput

During steady state:
  R stabilizes at the server's natural balance point
  → Observe mixed throughput

During ramp-down:
  R is typically low (no new prefills, many requests finishing decode)
  → Observe decode-dominated throughput
```

### 6.5 Capacity Planning from Phase Ratio

The phase ratio at saturation predicts the server's throughput limit:

```
At saturation (request_throughput stops increasing with load):
  R_sat = N_prefill_concurrent / N_decode_concurrent

  Prefill capacity: λ_max = N_prefill_concurrent × μ_P(avg_ISL)
  Decode capacity:  T_max = N_decode_concurrent × μ_D

  Bottleneck identification:
    IF increasing R_sat → higher total throughput:
      System is decode-bottlenecked (add more prefill capacity)
    IF decreasing R_sat → higher total throughput:
      System is prefill-bottlenecked (reduce concurrent prefills)
```

---

## 7. Disaggregated Serving Detection

### 7.1 Architecture Overview

Disaggregated serving (DistServe, Splitwise, TetriInfer) physically separates
prefill and decode onto different GPU pools:

```
Colocated Architecture:
┌─────────────────────────────────────┐
│         GPU 0 (shared)              │
│  ┌──────────┐  ┌──────────────────┐ │
│  │ Prefill  │  │     Decode       │ │
│  │ Engine   │──│     Engine       │ │
│  │          │  │                  │ │
│  └──────────┘  └──────────────────┘ │
│         SHARED SM + HBM             │
└─────────────────────────────────────┘
  → Prefill and decode contend for same resources
  → Strong cross-phase interference

Disaggregated Architecture:
┌──────────────────┐    ┌───────────────────┐
│   GPU 0 (prefill)│    │  GPU 1 (decode)   │
│  ┌──────────────┐│    │ ┌───────────────┐ │
│  │   Prefill    ││    │ │    Decode      │ │
│  │   Engine     ││───►│ │    Engine      │ │
│  │              ││ KV │ │               │ │
│  └──────────────┘│xfer│ └───────────────┘ │
│   Compute-bound  │    │  Memory-bound     │
└──────────────────┘    └───────────────────┘
  → Each GPU runs only its matching workload phase
  → No cross-phase interference
  → KV cache transfer cost between GPUs
```

### 7.2 Detection via Cross-Correlation

The key insight is that colocated and disaggregated architectures produce
fundamentally different correlation structures:

```
Detection Algorithm:

1. Compute prefill_concurrency(t) via concurrency_sweep(start_ns, generation_start_ns)
2. For each decode token, look up prefill_concurrency at token generation time
3. Compute Pearson correlation: r = Corr(prefill_concurrency, ITL)

Classification:
  r > 0.5:   Colocated serving (strong interference)
  r ∈ [0.15, 0.5]:  Colocated with chunked prefill (moderate interference)
  r < 0.15:  Disaggregated serving (no interference) OR low-load regime

Confidence guard:
  If effective_prefill_concurrency.p99 < 1.0:
    → insufficient prefill overlap to distinguish architectures
    → report "indeterminate" instead of "disaggregated"
```

### 7.3 Additional Disaggregation Signatures

Beyond the primary cross-correlation test, disaggregated serving produces
several secondary signatures:

**Signature 1: TTFT/ITL independence**

```
Colocated:
  High ISL requests → high TTFT AND high ITL for concurrent requests
  Correlation(TTFT_of_request_i, ITL_of_concurrent_request_j) > 0

Disaggregated:
  High ISL requests → high TTFT but NO effect on ITL of concurrent requests
  Correlation(TTFT_of_request_i, ITL_of_concurrent_request_j) ≈ 0
```

**Signature 2: KV cache transfer delay**

```
Disaggregated serving adds a KV transfer step between prefill and decode:
  TTFT_disagg = T_prefill + T_kv_transfer + T_first_decode_step

This manifests as:
  TTFT - stream_prefill_latency > expected_first_decode_step

The excess is the KV transfer time, typically 1-10ms depending on:
  - ISL (more tokens = more KV cache to transfer)
  - Interconnect bandwidth (NVLink vs PCIe vs network)
```

**Signature 3: Decode latency stability**

```
Colocated:
  Std(ITL) / Mean(ITL) ≈ 0.3 - 1.5 (high CV)
  Max(ITL) / Mean(ITL) > 3.0

Disaggregated:
  Std(ITL) / Mean(ITL) ≈ 0.05 - 0.2 (low CV)
  Max(ITL) / Mean(ITL) < 2.0

The decode GPU runs a homogeneous workload → very stable ITL
```

### 7.4 Mixed Architecture Detection

Some deployments use partial disaggregation (e.g., disaggregated prefill but
shared decode, or MIG-partitioned GPUs). These produce intermediate signatures:

```
Classification matrix:

                    │ r(PFC, ITL) high │ r(PFC, ITL) low  │
────────────────────┼───────────────────┼──────────────────┤
CV(ITL) high        │ Colocated,        │ Unlikely          │
                    │ no chunking       │ (check load)      │
────────────────────┼───────────────────┼──────────────────┤
CV(ITL) moderate    │ Colocated,        │ Partial disagg    │
                    │ with chunking     │ or MIG partition   │
────────────────────┼───────────────────┼──────────────────┤
CV(ITL) low         │ Unlikely          │ Full disagg        │
                    │ (check load)      │ or very low load   │
────────────────────┴───────────────────┴──────────────────┘
```

---

## 8. Iteration-Level Analysis

### 8.1 Server-Side Iteration Metrics

When vLLM's Prometheus endpoint is available, the
`vllm:iteration_tokens_total` counter provides direct visibility into scheduler
iteration behavior:

```
vllm:iteration_tokens_total = total tokens processed per scheduler step

Each increment represents one forward pass. The delta between consecutive
scrapes divided by the number of iterations gives the average batch size.
```

However, because AIPerf scrapes these at a configurable interval (typically
1-10s), individual iteration-level data is not directly available. Instead,
we work with aggregated iteration statistics.

### 8.2 Inferring Iteration Composition from Client Metrics

Even without direct iteration-level data, the client-side sweep signals encode
iteration composition implicitly:

```
At time t, the scheduler iteration likely contains:
  - prefill_concurrency(t) requests in prefill phase
  - generation_concurrency(t) requests in decode phase
  - Total tokens ≈ Σ ISL_prefill(t) + generation_concurrency(t) × 1

iteration_token_load(t) = tokens_in_flight(t) approximation
  (tokens_in_flight sweep counts total KV cache load, which correlates
   with but is not identical to per-iteration token count)
```

### 8.3 Batch Composition Variability

High variability in batch composition indicates scheduling instability:

```
batch_variability = Std(prefill_concurrency) / Mean(prefill_concurrency)

Interpretation:
  variability < 0.3:  Stable scheduling (consistent batching policy)
  variability ∈ [0.3, 1.0]:  Moderate variability (typical continuous batching)
  variability > 1.0:  Unstable scheduling (bursty arrivals or queue effects)
```

Cross-reference with ITL:

```
High batch variability + high ITL variance → scheduling-induced interference
High batch variability + low ITL variance  → chunked prefill masking the issue
Low batch variability + high ITL variance  → uniform but overloaded batches
Low batch variability + low ITL variance   → well-tuned server
```

### 8.4 Iteration Tokens and Throughput Correlation

The `vllm:iteration_tokens_total` delta rate correlates with the aggregate
throughput sweep:

```
iteration_token_rate(t) = d/dt [vllm:iteration_tokens_total]

Expected relationship:
  effective_total_throughput(t) ≈ iteration_token_rate(t)

Discrepancy indicates:
  - Measurement timing misalignment (client vs server clock)
  - Token counting differences (server counts prompt tokens differently)
  - Speculative decoding (server processes more tokens than emitted)
```

### 8.5 Server Queue Depth Analysis

The `vllm:num_requests_waiting` metric reveals the server-side queue:

```
When num_requests_waiting > 0:
  - Server is at capacity
  - New requests queue → TTFT increases
  - Prefill scheduling decisions determine interference pattern

Correlation:
  num_requests_waiting(t) → TTFT(t + lag)
  The lag is the expected queue drain time

At saturation (num_requests_waiting ≫ 0):
  Phase ratio is determined entirely by the scheduler policy
  → Scheduling fingerprinting becomes possible (Section 10)
```

---

## 9. Sequence Length Asymmetry Effects

### 9.1 ISL-Dependent Interference

The interference magnitude depends quadratically on the ISL of the prefilling
request because attention FLOPs scale as O(ISL^2):

```
interference_factor(ISL) = α × (ISL / ISL_ref)²

Where:
  α = base interference factor at ISL_ref (hardware-dependent)
  ISL_ref = reference sequence length (e.g., 1024 tokens)

Example (Llama-2-70B on A100):
  ISL=256:   interference ≈ 0.06 × α    (6% of reference)
  ISL=512:   interference ≈ 0.25 × α    (25% of reference)
  ISL=1024:  interference ≈ 1.0  × α    (100% = reference)
  ISL=2048:  interference ≈ 4.0  × α    (4x reference)
  ISL=4096:  interference ≈ 16.0 × α    (16x reference — severe)
```

### 9.2 Measuring ISL-Dependent Interference

For each decode token, pair the observed ITL with the ISL of any concurrent
prefill request:

```
For token j of request r (in decode phase):
  t_j = generation_start_ns[r] + cumsum(ITL[r][0:j+1])

  concurrent_prefills = {
    req q : start_ns[q] <= t_j < generation_start_ns[q]
  }

  max_prefill_ISL_j = max(ISL[q] for q in concurrent_prefills) or 0
  total_prefill_ISL_j = sum(ISL[q] for q in concurrent_prefills) or 0
```

Then bin ITL by `max_prefill_ISL` and compute statistics per bin:

```
ISL Bin        | Count  | Mean ITL | P99 ITL  | Interference Factor
[0, 0]         | 12,341 | 5.2 ms   | 7.1 ms   | 1.00 (baseline)
(0, 256]       |  1,892 | 5.8 ms   | 8.2 ms   | 1.12
(256, 512]     |  2,104 | 6.5 ms   | 11.3 ms  | 1.25
(512, 1024]    |  1,567 | 8.1 ms   | 18.7 ms  | 1.56
(1024, 2048]   |    892 | 12.3 ms  | 32.1 ms  | 2.37
(2048, 4096]   |    341 | 22.7 ms  | 58.4 ms  | 4.37
```

### 9.3 Workload-Weighted Interference Score

Since different workloads have different ISL distributions, a single
interference factor is insufficient. Define a workload-weighted score:

```
WIS = Σ_b  P(ISL_bin = b) × interference_factor(b)

Where P(ISL_bin = b) is the fraction of prefill tokens in bin b.

This gives a single number that captures the expected interference
for the specific workload being benchmarked.
```

### 9.4 OSL Effects on Interference Duration

While ISL determines interference *magnitude*, OSL determines interference
*duration*: a decode request with high OSL spends more time in the decode
phase and thus has more opportunities to be affected by concurrent prefills.

```
interference_exposure(r) = OSL[r] × P(prefill_concurrent | in_decode_phase)

Expected: requests with high OSL will have:
  - More ITL observations during prefill interference
  - Higher total decode-phase latency variance
  - More ITL spikes in their per-token latency trace
```

### 9.5 Cross-Request ISL-ITL Correlation

The ISL of *arriving* requests (not the request being measured) drives
interference. This cross-request correlation is the key signal:

```
For request r in decode phase:
  Define: arriving_ISL(t) = Σ ISL[q] for all q where start_ns[q] ∈ [t, t+dt)

  Cross-correlation:
    r_cross = Corr(arriving_ISL(t), ITL_of_request_r(t))

  Expected:
    r_cross > 0.3 → significant ISL-driven interference
    r_cross < 0.1 → interference not ISL-dependent (chunked or disaggregated)
```

---

## 10. Scheduling Policy Fingerprinting

### 10.1 Observable Scheduling Behaviors

Different scheduling policies create distinct patterns in the observable metrics.
The key policies used in modern inference servers are:

```
1. FCFS (First-Come-First-Served):
   Requests processed in arrival order. Simple, but can cause convoy effects
   where a long prefill blocks many short requests.

2. SJF (Shortest-Job-First / Shortest-Prefill-First):
   Shorter prefills scheduled first. Reduces average TTFT but may starve
   long-prefix requests.

3. Prefill-Priority:
   New prefill requests are prioritized over decode iterations. Minimizes
   TTFT at the cost of higher ITL (decode requests wait more).

4. Decode-Priority:
   Decode iterations are prioritized over new prefills. Minimizes ITL at
   the cost of higher TTFT (new requests wait in queue).

5. Token-Budget (vLLM default):
   Each iteration has a maximum token budget. Prefills and decodes are
   scheduled to fit within the budget. Provides natural chunking and
   balance.
```

### 10.2 Fingerprint Metrics

Each policy produces a characteristic "fingerprint" in the observable metrics:

```
Policy Fingerprint Matrix:

Metric                     │ FCFS    │ SJF     │ PF-Pri  │ DC-Pri  │ Token-Budget │
───────────────────────────┼─────────┼─────────┼─────────┼─────────┼──────────────┤
Corr(arrival_order, TTFT)  │ HIGH    │ LOW     │ LOW     │ MEDIUM  │ MEDIUM       │
Corr(ISL, TTFT)            │ MEDIUM  │ HIGH    │ MEDIUM  │ MEDIUM  │ HIGH         │
Mean(ITL) / Baseline(ITL)  │ 1.5-3x  │ 1.5-3x  │ 2-5x    │ 1.0-1.5x│ 1.2-2x      │
Std(ITL) / Mean(ITL)       │ HIGH    │ HIGH    │ V.HIGH  │ LOW     │ MEDIUM       │
Mean(TTFT) at saturation   │ MEDIUM  │ LOW     │ V.LOW   │ HIGH    │ MEDIUM       │
Prefill concurrency.p99    │ 1-3     │ 1-3     │ 3-8     │ 1       │ 1-2          │
Phase ratio stability      │ LOW     │ LOW     │ LOW     │ HIGH    │ HIGH         │
```

### 10.3 FCFS Signature

```
FCFS produces a convoy effect: a large prefill (ISL=4096) blocks the scheduler,
causing:

Timeline under FCFS with one large prefill arriving:

  Scheduler iteration: │ large_PF │ dec │ dec │ dec │ med_PF │ dec │ dec │
  ITL pattern:         │   HIGH   │ low │ low │ low │  med   │ low │ low │
  TTFT for arrivals:   │          │ wait│ wait│ wait│        │     │     │

Distinguishing features:
  - TTFT strongly correlated with arrival order (r > 0.7)
  - Bursty interference: long gaps of low ITL punctuated by single high spikes
  - Correlation between arrival ISL and TTFT of *subsequent* arrivals (convoy)
```

### 10.4 Prefill-Priority Signature

```
Prefill-priority scheduling aggressively starts new prefills, even if decode
requests are waiting:

Timeline under prefill-priority:

  Scheduler: │PF│PF│PF│ d │PF│PF│ d │ d │PF│PF│PF│ d │ d │ d │
  ITL:       │  │  │  │low│  │  │low│low│  │  │  │low│low│low│
                              ↑
                         decode only runs when no prefills waiting

Distinguishing features:
  - Very low TTFT (prefills start immediately)
  - Very high ITL variance (decode frequently preempted)
  - prefill_concurrency.p99 much higher than other policies
  - phase_ratio often > 1.0 (more prefills than decodes active)
  - Bimodal ITL: either very fast (no concurrent prefill) or very slow (many)
```

### 10.5 Decode-Priority Signature

```
Decode-priority scheduling ensures in-flight decode requests complete before
new prefills begin:

Timeline under decode-priority:

  Scheduler: │ d │ d │ d │ d │PF│ d │ d │ d │ d │PF│ d │ d │
  ITL:       │low│low│low│low│hi│low│low│low│low│hi│low│low│
                              ↑                   ↑
                    prefill only when decode batch is small

Distinguishing features:
  - Very stable ITL (low CV)
  - Higher TTFT (prefills delayed until decode batch shrinks)
  - prefill_concurrency.p99 ≈ 1 (at most one concurrent prefill)
  - phase_ratio.avg < 0.5 (decode-dominated)
  - TTFT shows queuing behavior: increases with num_requests_waiting
```

### 10.6 Classification Algorithm

```
Given a benchmark run, classify the server's scheduling policy:

INPUT:
  prefill_conc_stats   = compute_time_weighted_stats(prefill concurrency sweep)
  gen_conc_stats       = compute_time_weighted_stats(generation concurrency sweep)
  itl_stats            = metric_result for inter_token_latency
  ttft_stats           = metric_result for time_to_first_token
  cross_corr           = Corr(prefill_concurrency, ITL)
  arrival_ttft_corr    = Corr(arrival_order, TTFT)

RULES (decision tree):

  IF cross_corr < 0.15:
    → "disaggregated" (not a scheduling policy — an architecture)
  ELIF prefill_conc_stats.p99 >= 3.0 AND itl_stats.std / itl_stats.avg > 1.0:
    → "prefill_priority"
  ELIF prefill_conc_stats.p99 <= 1.5 AND itl_stats.std / itl_stats.avg < 0.3:
    → "decode_priority"
  ELIF arrival_ttft_corr > 0.7:
    → "fcfs"
  ELIF itl_stats.std / itl_stats.avg < 0.5:
    → "token_budget" (chunked prefill with balanced scheduling)
  ELSE:
    → "unknown" (possibly SJF or custom policy)
```

### 10.7 Limitations

Scheduling policy fingerprinting from client-side metrics has inherent
limitations:

1. **Load dependence**: At low load, all policies look similar (no contention).
   Classification is only meaningful when the server is near capacity
   (`num_requests_running / max_batch_size > 0.5`).

2. **Multiple policies**: Some servers dynamically switch policies based on
   load level. The fingerprint may change during a single run.

3. **Confounders**: Network jitter, client-side processing time, and OS
   scheduling can obscure the true scheduling pattern.

4. **Model size**: Smaller models have faster prefills, reducing the
   interference signal and making classification harder.

---

## 11. Detection Algorithms

### 11.1 Algorithm 1: Interference Factor from Sweep Signals

This is the primary algorithm for quantifying prefill-decode interference using
AIPerf's existing infrastructure.

```
Algorithm: compute_interference_factor

Input:
  store: ColumnStore (from MetricsAccumulator)
  window_start, window_end: float (steady-state boundaries)

Output:
  InterferenceResult with scalar factor, confidence, and per-ISL breakdown

Steps:

1. Extract sweep signals:
   pre_conc_ts, pre_conc = concurrency_sweep(start_ns, generation_start_ns)
   gen_conc_ts, gen_conc = concurrency_sweep(generation_start_ns, end_ns)

2. Extract per-token ITL observations with timestamps:
   itl_ragged = store.ragged("inter_token_latency")
   FOR each request r with valid generation_start_ns:
     gen_start = generation_start_ns[r]
     itl_values = itl_ragged.get(r)  # list of ITL durations (ns)
     token_timestamps = gen_start + np.cumsum(itl_values)
     # Each (timestamp, itl_value) is one observation

3. Look up prefill concurrency at each token timestamp:
   FOR each (t_token, itl_value):
     idx = searchsorted(pre_conc_ts, t_token, side='right') - 1
     pfc_at_token = pre_conc[clip(idx, 0, len-1)] if idx >= 0 else 0.0

4. Partition observations:
   group_0 = {itl : pfc_at_token == 0}  (no concurrent prefill)
   group_1 = {itl : pfc_at_token > 0}   (concurrent prefill active)

5. Compute interference factor:
   IF len(group_0) < 100 OR len(group_1) < 100:
     RETURN InsufficientData
   baseline_itl = mean(group_0)
   interfered_itl = mean(group_1)
   interference_factor = interfered_itl / baseline_itl - 1.0

6. Compute correlation:
   all_pfc = array of pfc_at_token for all observations
   all_itl = array of itl_value for all observations
   pearson_r = corrcoef(all_pfc, all_itl)[0,1]

7. Per-ISL breakdown:
   FOR each ISL bin b:
     observations_b = {itl : ISL of concurrent prefill in bin b}
     factor_b = mean(observations_b) / baseline_itl - 1.0

RETURN InterferenceResult(
   factor=interference_factor,
   correlation=pearson_r,
   baseline_itl_ns=baseline_itl,
   interfered_itl_ns=interfered_itl,
   n_baseline=len(group_0),
   n_interfered=len(group_1),
   per_isl_factors=per_isl_dict,
)
```

### 11.2 Algorithm 2: Chunked Prefill Detection

```
Algorithm: detect_chunked_prefill

Input:
  store: ColumnStore
  interference_result: InterferenceResult (from Algorithm 1)

Output:
  ChunkedPrefillResult with detection flag, estimated chunk size

Steps:

1. Compute ITL coefficient of variation:
   itl = all ITL observations within steady-state window
   cv_itl = std(itl) / mean(itl)

2. Compute interference correlation:
   r = interference_result.correlation

3. Primary classification:
   IF cv_itl < 0.5 AND interference_result.factor > 0.1:
     chunked = True  # Low variance + some interference = chunked
   ELIF cv_itl > 1.0 AND interference_result.factor > 0.5:
     chunked = False  # High variance + strong interference = unchunked
   ELSE:
     chunked = None  # Indeterminate

4. Estimate chunk size (if chunked):
   IF chunked:
     # Group requests by ISL, fit TTFT = a * ISL + b
     isl = store.numeric("input_sequence_length")
     ttft = store.numeric("time_to_first_token")
     valid = ~np.isnan(isl) & ~np.isnan(ttft)

     # Detect staircase in TTFT vs ISL
     sorted_idx = np.argsort(isl[valid])
     sorted_isl = isl[valid][sorted_idx]
     sorted_ttft = ttft[valid][sorted_idx]

     # Compute first derivative (TTFT per token) and look for jumps
     # Chunk boundaries appear as discrete jumps in TTFT
     dtdl = np.diff(sorted_ttft) / np.maximum(np.diff(sorted_isl), 1.0)

     # Find dominant periodicity in ISL spacing of jumps
     # This estimates chunk_size
     jump_mask = dtdl > 2 * np.median(dtdl)
     jump_isl = sorted_isl[1:][jump_mask]
     if len(jump_isl) >= 3:
       chunk_size_est = float(np.median(np.diff(jump_isl)))
     else:
       # Fallback: assume linear TTFT and estimate from slope
       slope = np.polyfit(sorted_isl.astype(float), sorted_ttft.astype(float), 1)[0]
       iteration_time = float(np.median(itl))  # approximate
       chunk_size_est = iteration_time / slope if slope > 0 else None

RETURN ChunkedPrefillResult(
   is_chunked=chunked,
   estimated_chunk_size=chunk_size_est,
   cv_itl=cv_itl,
   interference_correlation=r,
)
```

### 11.3 Algorithm 3: Disaggregated Serving Detection

```
Algorithm: detect_disaggregated_serving

Input:
  store: ColumnStore
  interference_result: InterferenceResult (from Algorithm 1)

Output:
  DisaggDetectionResult with classification and confidence

Steps:

1. Primary signal: cross-phase correlation
   r = interference_result.correlation

2. Secondary signal: ITL stability under varying prefill load
   itl_group_0 = ITL observations with prefill_concurrency == 0
   itl_group_1 = ITL observations with prefill_concurrency > 0
   cv_0 = std(itl_group_0) / mean(itl_group_0)
   cv_1 = std(itl_group_1) / mean(itl_group_1)

3. Tertiary signal: TTFT decomposition
   ttft = store.numeric("time_to_first_token")
   prefill_lat = store.numeric("stream_prefill_latency")
   valid = ~np.isnan(ttft) & ~np.isnan(prefill_lat)
   kv_transfer_est = ttft[valid] - prefill_lat[valid]
   # In disaggregated serving, this gap includes KV cache transfer time

4. Sufficiency check:
   prefill_conc_stats = time_weighted_stats of prefill concurrency
   IF prefill_conc_stats.p99 < 1.0:
     RETURN DisaggDetectionResult(
       classification="indeterminate",
       reason="insufficient prefill overlap",
       confidence=0.0,
     )

5. Classification:
   score = 0.0
   IF r < 0.15: score += 0.4
   IF cv_1 - cv_0 < 0.1: score += 0.3  # ITL stability unaffected by prefill
   IF mean(kv_transfer_est) > 2_000_000:  # > 2ms KV transfer gap
     score += 0.3

   IF score >= 0.7:
     classification = "disaggregated"
   ELIF score >= 0.4:
     classification = "partially_disaggregated"
   ELSE:
     classification = "colocated"

RETURN DisaggDetectionResult(
   classification=classification,
   confidence=score,
   cross_phase_correlation=r,
   cv_difference=cv_1 - cv_0,
   estimated_kv_transfer_ns=mean(kv_transfer_est),
)
```

### 11.4 Algorithm 4: Scheduling Policy Classification

```
Algorithm: classify_scheduling_policy

Input:
  store: ColumnStore
  sweep_curves: SweepCurves (from MetricsAccumulator)
  window_start, window_end: float
  interference_result: InterferenceResult

Output:
  SchedulingClassification with policy label and confidence

Steps:

1. Compute sweep statistics:
   pre_conc = compute_time_weighted_stats(prefill concurrency, window_start, window_end)
   gen_conc = compute_time_weighted_stats(generation concurrency, window_start, window_end)

2. Compute per-request statistics:
   itl_all = flatten all ITL observations in window
   cv_itl = std(itl_all) / mean(itl_all)

   ttft = store.numeric("time_to_first_token")[mask]
   valid_ttft = ttft[~np.isnan(ttft)]

3. Compute arrival-order correlation:
   # Sort by start_ns, compute rank correlation with TTFT
   start_order = np.argsort(store.start_ns[mask])
   ttft_in_order = ttft[mask][start_order]
   arrival_ttft_corr = spearman_rank_correlation(
     np.arange(len(ttft_in_order), dtype=float),
     ttft_in_order[~np.isnan(ttft_in_order)]
   )[0]

4. Compute feature vector:
   features = {
     "pfc_p99": pre_conc.p99,
     "cv_itl": cv_itl,
     "interference_r": interference_result.correlation,
     "arrival_ttft_r": arrival_ttft_corr,
     "phase_ratio_avg": pre_conc.avg / max(gen_conc.avg, 0.001),
     "phase_ratio_std": pre_conc.std / max(gen_conc.avg, 0.001),
   }

5. Decision tree classification:
   IF features["interference_r"] < 0.15:
     policy = "disaggregated"
     confidence = 1.0 - features["interference_r"] / 0.15
   ELIF features["pfc_p99"] >= 3.0 AND features["cv_itl"] > 1.0:
     policy = "prefill_priority"
     confidence = min(features["pfc_p99"] / 5.0, 1.0)
   ELIF features["pfc_p99"] <= 1.5 AND features["cv_itl"] < 0.3:
     policy = "decode_priority"
     confidence = 1.0 - features["cv_itl"] / 0.3
   ELIF features["arrival_ttft_r"] > 0.7:
     policy = "fcfs"
     confidence = features["arrival_ttft_r"]
   ELIF features["cv_itl"] < 0.5:
     policy = "token_budget"
     confidence = 1.0 - features["cv_itl"] / 0.5
   ELSE:
     policy = "unknown"
     confidence = 0.0

RETURN SchedulingClassification(
   policy=policy,
   confidence=confidence,
   features=features,
)
```

### 11.5 Algorithm 5: Phase Ratio Optimization Curve

```
Algorithm: compute_phase_ratio_throughput_curve

Input:
  sweep_curves: SweepCurves
  window_start, window_end: float
  n_bins: int = 20

Output:
  PhaseRatioOptimizationCurve with R vs throughput data points

Steps:

1. Compute phase ratio step function:
   ratio_ts, ratio_vals = divide_step_functions(
     pre_conc_ts, pre_conc, gen_conc_ts, gen_conc
   )

2. Compute total throughput step function (already in SweepCurves):
   total_ts = sweep_curves.total_throughput_ts
   total_vals = sweep_curves.total_throughput

3. Merge to common time grid:
   merged_ts = np.unique(np.concatenate([ratio_ts, total_ts]))
   merged_ts = merged_ts[(merged_ts >= window_start) & (merged_ts <= window_end)]

   ratio_at_merged = step_lookup(ratio_ts, ratio_vals, merged_ts)
   tput_at_merged = step_lookup(total_ts, total_vals, merged_ts)
   durations = np.diff(merged_ts, append=window_end) - merged_ts
   durations = np.maximum(durations, 0)

4. Bin by phase ratio:
   ratio_bins = np.linspace(0, np.percentile(ratio_at_merged, 99), n_bins + 1)
   bin_indices = np.digitize(ratio_at_merged, ratio_bins)

   FOR b in range(1, n_bins + 1):
     mask = bin_indices == b
     IF mask.any():
       weighted_tput = np.sum(tput_at_merged[mask] * durations[mask]) / np.sum(durations[mask])
       bin_center = (ratio_bins[b-1] + ratio_bins[b]) / 2
       curve_points.append((bin_center, weighted_tput))

5. Find optimal:
   R_optimal = bin_center at max(weighted_tput)

RETURN PhaseRatioOptimizationCurve(
   bin_centers=...,
   throughput_per_bin=...,
   optimal_ratio=R_optimal,
   optimal_throughput=max_tput,
)
```

---

## 12. AIPerf Implementation Guidance

### 12.1 Architecture Overview

The interference analysis should be implemented as an optional **AnalyzerProtocol**
plugin, following the same pattern as `SteadyStateAnalyzer`:

```
src/aiperf/
├── analysis/
│   ├── sweep.py                   # Existing sweep-line algorithms
│   ├── ramp_detection.py          # Existing CUSUM + MSER-5
│   ├── stationarity.py            # Existing trend testing
│   ├── interference.py            # NEW: interference computation functions
│   └── phase_analysis.py          # NEW: phase ratio + scheduling fingerprint
├── post_processors/
│   ├── steady_state_analyzer.py   # Existing analyzer
│   └── interference_analyzer.py   # NEW: AnalyzerProtocol implementation
├── exporters/
│   ├── interference_json_exporter.py   # NEW: JSON export
│   └── interference_console_exporter.py # NEW: console summary
└── common/
    └── config/
        └── interference_config.py  # NEW: configuration model
```

### 12.2 Data Types

```python
@dataclass(frozen=True, slots=True)
class InterferenceResult:
    """Quantified prefill-decode interference."""
    factor: float                           # ITL inflation ratio - 1
    correlation: float                      # Pearson r(prefill_concurrency, ITL)
    baseline_itl_ns: float                  # Mean ITL with no concurrent prefills
    interfered_itl_ns: float                # Mean ITL with concurrent prefills
    n_baseline: int                         # Token count in baseline group
    n_interfered: int                       # Token count in interfered group
    per_isl_factors: dict[str, float]       # ISL bin label -> factor


@dataclass(frozen=True, slots=True)
class ChunkedPrefillResult:
    """Chunked prefill detection result."""
    is_chunked: bool | None                 # True/False/None (indeterminate)
    estimated_chunk_size: float | None      # Estimated chunk size in tokens
    cv_itl: float                           # Coefficient of variation of ITL
    interference_correlation: float         # Cross-phase correlation


@dataclass(frozen=True, slots=True)
class DisaggDetectionResult:
    """Disaggregated serving detection result."""
    classification: str                     # "colocated", "disaggregated", "partial", "indeterminate"
    confidence: float                       # 0.0 to 1.0
    cross_phase_correlation: float
    cv_difference: float                    # CV(ITL|prefill) - CV(ITL|no_prefill)
    estimated_kv_transfer_ns: float


@dataclass(frozen=True, slots=True)
class SchedulingClassification:
    """Inferred scheduling policy."""
    policy: str                             # "fcfs", "prefill_priority", etc.
    confidence: float                       # 0.0 to 1.0
    features: dict[str, float]              # Feature vector used for classification


@dataclass(frozen=True, slots=True)
class PhaseRatioOptimizationCurve:
    """Phase ratio vs throughput curve."""
    bin_centers: list[float]
    throughput_per_bin: list[float]
    optimal_ratio: float
    optimal_throughput: float


class InterferenceSummary(AIPerfBaseModel):
    """Top-level summary from InterferenceAnalyzer."""
    interference: InterferenceResult
    chunked_prefill: ChunkedPrefillResult
    disaggregation: DisaggDetectionResult
    scheduling: SchedulingClassification
    phase_optimization: PhaseRatioOptimizationCurve
```

### 12.3 Analyzer Implementation Sketch

```python
class InterferenceAnalyzer:
    """Prefill-decode interference analysis.

    Implements AnalyzerProtocol. Reads sweep curves and ColumnStore
    from MetricsAccumulator at summarize time.
    """

    required_accumulators: ClassVar[set[AccumulatorType]] = {"metric_results"}
    summary_dependencies: ClassVar[list[AccumulatorType]] = ["metric_results"]

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        config = user_config.output.interference
        if not config.enabled:
            raise PluginDisabled("Interference analysis is disabled")
        self._isl_bins = config.isl_bins or [0, 256, 512, 1024, 2048, 4096, 8192]

    async def summarize(self, ctx: SummaryContext) -> InterferenceSummary:
        metrics_acc = ctx.get_accumulator(AccumulatorType.METRIC_RESULTS)
        store = metrics_acc.column_store

        # Use sweep curves (already computed by MetricsAccumulator)
        sweep_curves = ctx.accumulator_outputs.get("metric_results_sweep_curves")
        window_start = ctx.start_ns
        window_end = ctx.end_ns

        # 1. Compute interference factor
        interference = compute_interference_factor(
            store, window_start, window_end, self._isl_bins
        )

        # 2. Detect chunked prefill
        chunked = detect_chunked_prefill(store, interference)

        # 3. Detect disaggregated serving
        disagg = detect_disaggregated_serving(store, interference, sweep_curves)

        # 4. Classify scheduling policy
        scheduling = classify_scheduling_policy(
            store, sweep_curves, window_start, window_end, interference
        )

        # 5. Compute phase ratio optimization curve
        phase_opt = compute_phase_ratio_throughput_curve(
            sweep_curves, window_start, window_end
        )

        return InterferenceSummary(
            interference=interference,
            chunked_prefill=chunked,
            disaggregation=disagg,
            scheduling=scheduling,
            phase_optimization=phase_opt,
        )
```

### 12.4 Integration with Existing Sweep Infrastructure

The interference analysis reuses AIPerf's existing sweep-line framework
extensively. The key integration points are:

**ColumnStore access**: All per-request data is accessed through the ColumnStore
API. The relevant columns are:

```python
store.start_ns[:n]              # Request start timestamps
store.end_ns[:n]                # Request end timestamps
store.generation_start_ns[:n]   # First-token timestamps
store.numeric("input_sequence_length")
store.numeric("output_sequence_length")
store.numeric("time_to_first_token")
store.numeric("request_latency")
store.numeric("stream_prefill_latency")
store.ragged("inter_token_latency")  # RaggedSeries for per-token ITL
```

**Sweep curves**: The `SweepCurves` dataclass (from `sweep.py`) already contains
all necessary step functions. The interference analyzer consumes these rather
than recomputing them:

```python
# Already computed by MetricsAccumulator:
sweep_curves.prefill_concurrency_ts    # prefill concurrency step function
sweep_curves.prefill_concurrency
sweep_curves.generation_concurrency_ts  # decode concurrency step function
sweep_curves.generation_concurrency
sweep_curves.total_throughput_ts        # total throughput step function
sweep_curves.total_throughput
sweep_curves.tokens_in_flight_ts        # token load step function
sweep_curves.tokens_in_flight
```

**Step function utilities**: The existing `_step_lookup()`,
`add_step_functions()`, and `divide_step_functions()` functions provide the
mathematical operations needed for cross-signal analysis.

### 12.5 Configuration Model

```python
class InterferenceConfig(BaseConfig, frozen=True):
    """Configuration for prefill-decode interference analysis."""

    enabled: bool = Field(
        default=False,
        description="Enable prefill-decode interference analysis",
    )
    isl_bins: list[int] | None = Field(
        default=None,
        description="ISL bin boundaries for per-ISL interference breakdown. "
                    "Default: [0, 256, 512, 1024, 2048, 4096, 8192]",
    )
    min_observations: int = Field(
        default=100,
        description="Minimum token observations per group (baseline/interfered) "
                    "for reliable interference factor estimation",
    )
    ccf_bin_width_ms: float = Field(
        default=50.0,
        description="Time bin width (ms) for cross-correlation function computation",
    )
    ccf_max_lag_ms: float = Field(
        default=500.0,
        description="Maximum lag (ms) for cross-correlation function",
    )
```

### 12.6 CLI Integration

```
aiperf benchmark ... \
  --interference                        # Enable interference analysis
  --interference-isl-bins 256,512,1024  # Custom ISL bins

Environment variables:
  AIPERF_INTERFERENCE_ENABLED=true
  AIPERF_INTERFERENCE_ISL_BINS=256,512,1024,2048,4096
  AIPERF_INTERFERENCE_MIN_OBSERVATIONS=100
```

### 12.7 Console Output

```
Prefill-Decode Interference Analysis
======================================
Interference Factor:  1.87x  (ITL inflated 87% during concurrent prefills)
Correlation (r):      0.73   (strong positive — colocated serving)
Baseline ITL:         5.2 ms (no concurrent prefills, n=12,341)
Interfered ITL:       9.7 ms (with concurrent prefills, n=4,892)

Architecture:         Colocated (confidence: 0.91)
Chunked Prefill:      Not detected (CV(ITL) = 1.23)
Scheduling Policy:    token_budget (confidence: 0.74)

Interference by ISL:
  ISL (0, 256]:       +12% (n=1,892)
  ISL (256, 512]:     +25% (n=2,104)
  ISL (512, 1024]:    +56% (n=1,567)
  ISL (1024, 2048]:   +137% (n=892)
  ISL (2048, 4096]:   +337% (n=341)

Phase Ratio:
  Optimal R:          0.35 (prefill_conc / decode_conc)
  Current R (avg):    0.42 (7% above optimal)
```

### 12.8 JSON Export Structure

```json
{
  "interference_analysis": {
    "interference": {
      "factor": 1.87,
      "correlation": 0.73,
      "baseline_itl_ns": 5200000,
      "interfered_itl_ns": 9724000,
      "n_baseline": 12341,
      "n_interfered": 4892,
      "per_isl_factors": {
        "(0, 256]": 0.12,
        "(256, 512]": 0.25,
        "(512, 1024]": 0.56,
        "(1024, 2048]": 1.37,
        "(2048, 4096]": 3.37
      }
    },
    "architecture": {
      "classification": "colocated",
      "confidence": 0.91,
      "cross_phase_correlation": 0.73,
      "cv_difference": 0.42,
      "estimated_kv_transfer_ns": 0
    },
    "chunked_prefill": {
      "is_chunked": false,
      "estimated_chunk_size": null,
      "cv_itl": 1.23,
      "interference_correlation": 0.73
    },
    "scheduling": {
      "policy": "token_budget",
      "confidence": 0.74,
      "features": {
        "pfc_p99": 2.0,
        "cv_itl": 0.45,
        "interference_r": 0.73,
        "arrival_ttft_r": 0.31,
        "phase_ratio_avg": 0.42,
        "phase_ratio_std": 0.18
      }
    },
    "phase_optimization": {
      "optimal_ratio": 0.35,
      "optimal_throughput_tokens_per_sec": 4521.3,
      "current_ratio_avg": 0.42,
      "efficiency_pct": 93.2
    }
  }
}
```

### 12.9 Plugin Registration

```yaml
# In plugins.yaml, under the analyzer category:
interference_analyzer:
  class: aiperf.post_processors.interference_analyzer.InterferenceAnalyzer
  description: Prefill-decode interference and phase contention analysis
  metadata:
    required_accumulators:
      - metric_results
    summary_dependencies:
      - metric_results
```

### 12.10 Performance Considerations

The primary computational costs are:

1. **Token-level pairing** (Algorithm 1, Step 2-3): Iterating over all ITL
   observations and performing `searchsorted` lookups. For a run with N
   requests and average OSL of K tokens, this is O(N*K * log(S)) where S
   is the number of sweep events. With N=10,000 and K=200, this is ~2M
   lookups — fast with numpy vectorization.

2. **Phase ratio curve** (Algorithm 5): Merging two step functions and
   binning, O(S * log(S)) where S is the total sweep events.

3. **Cross-correlation function** (Section 4.4): Time-binning and correlation,
   O(N*K + B) where B is the number of time bins.

All operations are vectorizable with numpy and should complete in <100ms for
typical benchmark runs (10K-100K requests).

---

## 13. Validation Strategy

### 13.1 Synthetic Validation Profiles

Following the pattern established in AIPerf's existing synthetic validation
suite (see `proposal-statistical-foundations.md`), define interference-specific
synthetic profiles:

```
Profile 1: zero_interference
  Prefill and decode never overlap temporally.
  Expected: interference_factor ≈ 0, correlation ≈ 0
  Validates: baseline detection, no false positives

Profile 2: constant_interference
  One prefill always running during all decode tokens.
  ITL = baseline × constant_multiplier.
  Expected: interference_factor = multiplier - 1, correlation > 0.9
  Validates: factor computation accuracy

Profile 3: isl_proportional_interference
  Interference magnitude proportional to ISL^2.
  Expected: per-ISL factors follow quadratic scaling
  Validates: ISL-dependent analysis

Profile 4: chunked_prefill_signature
  Low ITL variance, slightly elevated mean, no spikes.
  Expected: chunked_prefill detected, CV < 0.5
  Validates: chunked prefill detection

Profile 5: disaggregated_signature
  Zero correlation between prefill concurrency and ITL.
  Expected: disaggregated classification, confidence > 0.8
  Validates: disaggregated detection

Profile 6: fcfs_scheduling
  TTFT strongly correlated with arrival order.
  Expected: policy = "fcfs", arrival_ttft_r > 0.7
  Validates: FCFS fingerprinting

Profile 7: decode_priority_scheduling
  Very stable ITL, max 1 concurrent prefill.
  Expected: policy = "decode_priority", cv_itl < 0.3
  Validates: decode-priority fingerprinting

Profile 8: high_load_saturation
  Server at capacity with queue buildup.
  Expected: all analysis modes produce meaningful results
  Validates: behavior under saturation

Profile 9: mixed_isl_workload
  Bimodal ISL distribution (short prompts + long prompts).
  Expected: per-ISL breakdown shows clear separation
  Validates: ISL bin analysis with realistic distributions

Profile 10: ramp_with_interference
  Interference changes during ramp-up/steady-state/ramp-down.
  Expected: steady-state window shows different interference than ramp phases
  Validates: integration with steady-state detection
```

### 13.2 Real-World Validation

Test against known server configurations:

```
Test Matrix:
┌─────────────────────────┬──────────────────────────────────────────┐
│ Server Configuration    │ Expected Detection                       │
├─────────────────────────┼──────────────────────────────────────────┤
│ vLLM default            │ colocated, token_budget, chunked=varies  │
│ vLLM --no-chunked-pref  │ colocated, strong interference           │
│ vLLM chunked_prefill=512│ colocated, chunked, chunk_est≈512       │
│ TensorRT-LLM            │ colocated, decode_priority (typical)     │
│ DistServe-like setup    │ disaggregated, low correlation           │
│ SGLang                  │ colocated, token_budget variant           │
└─────────────────────────┴──────────────────────────────────────────┘
```

### 13.3 Accuracy Metrics

For each detection algorithm, define accuracy criteria:

```
Interference factor:
  - Synthetic: |measured - true| / true < 0.15 (within 15%)
  - Real: directional accuracy (higher ISL → higher factor)

Chunked prefill detection:
  - Precision: > 0.90 (few false positives)
  - Recall: > 0.80 (may miss edge cases)

Disaggregated detection:
  - Precision: > 0.95 (critical to not misclassify)
  - Recall: > 0.70 (conservative is acceptable)

Scheduling policy:
  - Top-1 accuracy on synthetic: > 0.85
  - "Unknown" is acceptable (better than wrong)
```

### 13.4 Edge Cases

```
Edge Case 1: Very short requests (ISL < 32, OSL < 16)
  Prefill is so fast that interference is negligible.
  Expected behavior: interference_factor ≈ 0, report "low_load"

Edge Case 2: All requests identical (constant ISL, constant OSL)
  No ISL variation → per-ISL breakdown is a single bin.
  Expected behavior: single-bin report, chunk size estimation may fail

Edge Case 3: Streaming disabled (no SSE, no ITL data)
  ITL RaggedSeries is empty.
  Expected behavior: PluginDisabled("ITL data required for interference analysis")

Edge Case 4: Single concurrent request (concurrency=1)
  No overlapping prefill and decode phases.
  Expected behavior: InsufficientData or "no_overlap" classification

Edge Case 5: Server with speculative decoding
  Multiple tokens generated per iteration, ITL patterns differ.
  Expected behavior: may confuse chunked prefill detection, document limitation

Edge Case 6: Multi-model serving (different models on same GPU)
  Interference from other model's requests.
  Expected behavior: interference detected but ISL correlation may be weak
```

---

## 14. References

### Academic Papers

1. **Yu, G.-I. et al. (2022).** "Orca: A Distributed Serving System for
   Transformer-Based Generative Models." *OSDI 2022*. — Introduced continuous
   batching (iteration-level scheduling) for LLM serving.

2. **Zhong, Y. et al. (2024).** "DistServe: Disaggregating Prefill and Decoding
   for Goodput-optimized Large Language Model Serving." *OSDI 2024*. — Formalized
   the prefill-decode interference problem and proposed disaggregation.

3. **Agrawal, A. et al. (2024).** "Taming Throughput-Latency Tradeoff in LLM
   Inference with Sarathi-Serve." *OSDI 2024*. — Analyzed chunked prefill as a
   mitigation strategy for interference.

4. **Patel, P. et al. (2024).** "Splitwise: Efficient Generative LLM Inference
   Using Phase Splitting." *ISCA 2024*. — Mixed-resource disaggregation approach
   where prefill runs on compute-optimized and decode on memory-optimized hardware.

5. **Holmes, C. et al. (2024).** "DeepSpeed-FastGen: High-throughput Text
   Generation for LLMs via MII and DeepSpeed-Inference." *arXiv:2401.08671*. —
   Splitfuse scheduling: decompose prefills into micro-batches interleaved with
   decode, related to chunked prefill analysis.

6. **Wu, B. et al. (2024).** "Loongserve: Efficiently Serving Long-Context Large
   Language Models with Elastic Sequence Parallelism." *SOSP 2024*. — Elastic
   parallelism for long sequences, relevant to ISL-dependent interference scaling.

### Industry References

7. **vLLM Documentation.** "Chunked Prefill." — Implementation details of vLLM's
   chunked prefill feature and its effect on decode latency.

8. **Kwon, W. et al. (2023).** "Efficient Memory Management for Large Language
   Model Serving with PagedAttention." *SOSP 2023*. — PagedAttention and the
   vLLM scheduling framework.

9. **NVIDIA TensorRT-LLM.** "In-Flight Batching." — TensorRT-LLM's continuous
   batching implementation and scheduling policies.

### Related AIPerf Documents

10. **gap-analysis-research-grade-algorithms.md** — Identifies latency
    decomposition and coordinated omission as important gaps; interference
    analysis extends these to cross-phase effects.

11. **proposal-coordinated-omission-and-latency-decomposition.md** — Defines
    `service_latency` and `queue_wait_time` metrics; interference analysis adds
    the per-phase decomposition of latency causes.

12. **proposal-advanced-analysis.md** — Defines the AnalyzerProtocol plugin
    pattern that interference analysis should follow.

---

## Appendix A: Mathematical Notation Summary

| Symbol | Definition | Source |
|---|---|---|
| `PFC(t)` | effective_prefill_concurrency at time t | `concurrency_sweep(start_ns, generation_start_ns)` |
| `GC(t)` | effective_generation_concurrency at time t | `concurrency_sweep(generation_start_ns, end_ns)` |
| `PT(t)` | effective_prefill_throughput at time t | `prefill_throughput_sweep(start_ns, gen_start, ISL)` |
| `GT(t)` | effective_throughput (generation) at time t | `throughput_sweep(gen_start, end_ns, OSL)` |
| `TIF(t)` | tokens_in_flight at time t | `tokens_in_flight_sweep(...)` |
| `R(t)` | phase ratio = PFC(t) / GC(t) | `divide_step_functions(PFC, GC)` |
| `ITL(r,j)` | inter-token latency of request r, token j | ColumnStore ragged `inter_token_latency` |
| `IF` | interference factor | `E[ITL|PFC>0] / E[ITL|PFC=0] - 1` |
| `CV(X)` | coefficient of variation = Std(X) / Mean(X) | Standard definition |
| `CCF(tau)` | cross-correlation function at lag tau | `Corr(PFC(t), ITL_binned(t + tau))` |

## Appendix B: Relationship to Existing AIPerf Sweep Metrics

The interference analysis builds on top of — but does not modify — the existing
9 sweep metrics defined in `SWEEP_METRIC_SPECS`:

```
Existing (consumed as inputs):
  1. effective_concurrency            — total concurrency (baseline)
  2. effective_throughput              — generation throughput
  3. effective_prefill_throughput      — prefill throughput
  4. effective_generation_concurrency  — decode-phase concurrency
  5. effective_prefill_concurrency     — prefill-phase concurrency
  6. effective_total_throughput        — combined throughput
  7. effective_throughput_per_user     — per-user generation throughput
  8. effective_prefill_throughput_per_user — per-user prefill throughput
  9. tokens_in_flight                  — KV cache token load

New (computed by interference analysis):
  10. phase_ratio                      — PFC / GC (derived sweep)
  11. interference_factor              — scalar summary statistic
  12. cross_phase_correlation          — Corr(PFC, ITL) scalar
  13. interference_by_isl              — per-ISL-bin factor breakdown
```

The new metrics are NOT added to `SWEEP_METRIC_SPECS` because they are analysis
outputs, not time-weighted sweep statistics. They are reported in the
`InterferenceSummary` returned by the `InterferenceAnalyzer`.

## Appendix C: Computational Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Typical Runtime |
|---|---|---|---|
| Interference Factor (Alg 1) | O(N*K * log S) | O(N*K) | ~50ms for 10K reqs |
| Chunked Prefill Detection (Alg 2) | O(N * log N) | O(N) | ~10ms |
| Disaggregated Detection (Alg 3) | O(N*K * log S) | O(N*K) | ~50ms |
| Scheduling Classification (Alg 4) | O(N * log N) | O(N) | ~10ms |
| Phase Ratio Curve (Alg 5) | O(S * log S) | O(S) | ~5ms |

Where:
- N = number of requests (~10K typical)
- K = average output sequence length (~200 typical)
- S = number of sweep events (~2N typical)

Total: ~125ms for a 10K-request run. Well within the existing summarize() time
budget (which already runs sweep-line algorithms, steady-state detection,
bootstrap CIs, and stationarity tests).
