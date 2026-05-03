<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV Cache Pressure & Latency Degradation Correlation

**Status**: Research

**Authors**: Anthony Casagrande

**Category**: Research — Correlation Analysis

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background: KV Cache Architecture in LLM Inference](#2-background-kv-cache-architecture-in-llm-inference)
3. [KV Cache as Performance Predictor](#3-kv-cache-as-performance-predictor)
4. [Preemption Event Correlation](#4-preemption-event-correlation)
5. [Tokens-in-Flight as Client-Side KV Proxy](#5-tokens-in-flight-as-client-side-kv-proxy)
6. [Sequence Length Impact on KV Demand](#6-sequence-length-impact-on-kv-demand)
7. [Prefix Cache Efficiency](#7-prefix-cache-efficiency)
8. [Memory Pressure Cascade](#8-memory-pressure-cascade)
9. [Implementation: Correlation Analysis Engine](#9-implementation-correlation-analysis-engine)
10. [Academic Context & Literature](#10-academic-context--literature)
11. [Appendix: Metric Reference](#appendix-a-metric-reference)
12. [Appendix: Derivations](#appendix-b-derivations)

---

## 1. Executive Summary

KV cache utilization is the single most important predictor of latency
degradation in autoregressive LLM inference. Unlike traditional web services
where latency degrades linearly with load, LLM serving exhibits a
**phase-transition** behavior: latency remains stable until KV cache utilization
crosses a critical threshold (~80-95%, server-dependent), at which point
preemption events trigger re-computation, queue buildup, and cascading latency
spikes.

This document researches the correlation between KV cache pressure and client-
observed latency degradation, with the goal of enabling AIPerf to:

1. **Detect** the KV cache pressure regime from combined client-side and
   server-side metrics
2. **Quantify** the correlation between cache utilization and latency percentiles
3. **Predict** latency degradation onset from tokens-in-flight (client-side
   proxy) before server-side metrics confirm it
4. **Alert** users when benchmarks cross the KV cache saturation boundary

The analysis leverages AIPerf's existing metric infrastructure: 9 sweep-line
metrics (including `tokens_in_flight`), Prometheus server metrics
(`vllm:kv_cache_usage_perc`, `vllm:num_preemptions`), GPU telemetry, and per-
request latency decomposition (TTFT, ITL, request_latency).

---

## 2. Background: KV Cache Architecture in LLM Inference

### 2.1 What the KV Cache Is

In transformer-based autoregressive generation, each attention layer computes
key (K) and value (V) projections for every token in the sequence. During
generation, previously computed K/V pairs are cached to avoid recomputation:

```
Without KV cache (naive):
  Token 1: compute K,V for [token_1]                         → 1 token
  Token 2: compute K,V for [token_1, token_2]                → 2 tokens
  Token 3: compute K,V for [token_1, token_2, token_3]       → 3 tokens
  ...
  Token n: compute K,V for [token_1, ..., token_n]           → n tokens
  Total computation: O(n^2) key-value pairs

With KV cache:
  Token 1: compute & cache K,V for [token_1]                 → 1 token
  Token 2: compute & cache K,V for [token_2], reuse cached   → 1 token (new)
  Token 3: compute & cache K,V for [token_3], reuse cached   → 1 token (new)
  ...
  Token n: compute & cache K,V for [token_n], reuse cached   → 1 token (new)
  Total computation: O(n) key-value pairs
```

The KV cache transforms O(n^2) attention computation into O(n), but at the cost
of GPU memory proportional to `num_layers * 2 * seq_len * hidden_dim *
dtype_bytes` per request.

### 2.2 Memory Budget per Token

For a model with parameters:

```
kv_bytes_per_token = num_layers * 2 * num_kv_heads * head_dim * dtype_bytes
```

Example calculations:

| Model         | Layers | KV Heads | Head Dim | Dtype   | Bytes/Token |
|---------------|--------|----------|----------|---------|-------------|
| Llama 3 8B    | 32     | 8        | 128      | float16 | 131 KB      |
| Llama 3 70B   | 80     | 8        | 128      | float16 | 328 KB      |
| Mixtral 8x7B  | 32     | 8        | 128      | float16 | 131 KB      |
| GPT-4 class   | 120    | 16       | 128      | float16 | 983 KB      |

With an 80 GB GPU (A100/H100), and ~50% of memory available for KV cache after
model weights and activations:

```
Available KV memory ≈ 40 GB

Llama 3 8B:  40 GB / 131 KB ≈ 305,000 tokens
Llama 3 70B: 40 GB / 328 KB ≈ 122,000 tokens (per-GPU in tensor parallel)
```

### 2.3 PagedAttention and Block Management

vLLM's PagedAttention (Kwon et al., 2023) manages KV cache as a virtual memory
system:

```
Physical GPU Memory
┌────────────────────────────────────────────────┐
│  Model Weights (fixed)                         │
├────────────────────────────────────────────────┤
│  Activation Memory (per-batch, variable)       │
├────────────────────────────────────────────────┤
│  KV Cache Block Pool                           │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐         │
│  │Blk 0 │ │Blk 1 │ │Blk 2 │ │Blk 3 │  ...   │
│  │Req A │ │Req A │ │Req B │ │Free  │         │
│  │tok 0 │ │tok 16│ │tok 0 │ │      │         │
│  │..15  │ │..31  │ │..15  │ │      │         │
│  └──────┘ └──────┘ └──────┘ └──────┘         │
│                                                │
│  Block Table (per request):                    │
│  Req A: [Blk 0, Blk 1]  (32 tokens cached)    │
│  Req B: [Blk 2]          (16 tokens cached)    │
│  Free list: [Blk 3, ...]                       │
└────────────────────────────────────────────────┘
```

Key properties:
- **Non-contiguous allocation**: Blocks need not be adjacent in physical memory
- **Copy-on-write**: Shared prefixes (e.g., system prompts) share physical blocks
- **Block size**: Typically 16 tokens per block (configurable)
- **Fragmentation**: Unlike contiguous allocation, PagedAttention eliminates
  internal fragmentation within the block pool

### 2.4 The Preemption Mechanism

When the KV cache is full and a new request arrives (or an existing request
needs more blocks), vLLM must make a scheduling decision:

```
KV Cache Full Decision Tree
────────────────────────────
                    ┌─────────────────────┐
                    │ New block needed     │
                    │ (prefill or decode)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Free blocks > 0?    │
                    └──────────┬──────────┘
                       Yes │        │ No
                           │        │
                ┌──────────▼──┐  ┌──▼──────────────┐
                │ Allocate    │  │ Preemption       │
                │ from free   │  │ required         │
                │ list        │  └──────┬───────────┘
                └─────────────┘         │
                               ┌────────▼────────┐
                               │ Swap to CPU?     │
                               │ (if swap space   │
                               │  available)      │
                               └────────┬────────┘
                                  Yes │      │ No
                                      │      │
                           ┌──────────▼──┐  ┌▼─────────────┐
                           │ Swap lowest │  │ Recompute     │
                           │ priority    │  │ (abort +      │
                           │ request to  │  │  re-queue     │
                           │ CPU memory  │  │  request)     │
                           └─────────────┘  └──────────────┘
```

**Preemption costs:**
- **Swap**: Copies KV blocks to CPU memory. Latency = transfer time (PCIe
  bandwidth limited). On swap-in, blocks are copied back.
- **Recompute**: Aborts the preempted request entirely. It re-enters the
  waiting queue and must re-run prefill from scratch. This is the expensive
  case — all progress is lost.

The `vllm:num_preemptions` counter tracks recompute-style preemptions. Each
increment represents one request that lost all cached KV state and must restart.

---

## 3. KV Cache as Performance Predictor

### 3.1 The Non-Linear Relationship

KV cache utilization and latency do not have a linear relationship. The
relationship follows a characteristic "hockey stick" curve:

```
Latency
(ms)     │
         │
    500  │                                              ╱
         │                                            ╱
    400  │                                          ╱
         │                                        ╱
    300  │                                     ╱╱
         │                                   ╱
    200  │                               ╱╱
         │                            ╱╱
    100  │  ●━━━━━━━━━━━━━━━━━━━━━╱╱
         │                      ╱
     50  │  ●━━━━━━━━━━━━━━━━━●
         │
         └────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬──
              10   20   30   40   50   60   70   80   90  100
                                                     ↑
                          KV Cache Usage (%)      Critical
                                                  Threshold
```

**Three distinct regimes:**

| Regime | KV Cache % | Latency Behavior | Mechanism |
|--------|-----------|-------------------|-----------|
| **Stable** | 0 — 70% | Flat, low variance | Ample free blocks, no contention |
| **Pressure** | 70 — 90% | Gradual increase, variance rises | Scheduling becomes conservative, batch sizes shrink |
| **Saturation** | 90 — 100% | Exponential increase, preemptions | Block starvation, swap/recompute, queue buildup |

The exact thresholds depend on:
- Model size (larger models = fewer total blocks)
- Sequence length distribution (long sequences consume more blocks)
- vLLM scheduler configuration (max_num_batched_tokens, preemption_mode)
- Whether prefix caching is enabled (shared blocks reduce effective usage)

### 3.2 Theoretical Model

We can model the latency as a function of KV cache utilization using a
queueing-theoretic framework. Let:

- `U` = KV cache utilization (0 to 1)
- `U_c` = critical utilization threshold
- `L_base` = baseline latency (at low utilization)
- `P(U)` = preemption probability as a function of utilization
- `C_preempt` = cost of one preemption (re-prefill time)

**Latency model:**

```
E[L(U)] = L_base                               if U < U_c
         = L_base × 1/(1 - U/U_max)            if U >= U_c (M/M/1 approximation)
```

The M/M/1 queue waiting time formula `W = 1/(1-ρ)` provides intuition: as
utilization ρ approaches 1, waiting time approaches infinity. While LLM serving
is not a classical Markov queue, the qualitative behavior matches.

**With preemptions:**

```
E[L(U)] = L_base + P(U) × C_preempt

where P(U) ≈ 0                                  if U < U_c
             ≈ ((U - U_c) / (1 - U_c))^α        if U >= U_c
```

The exponent α captures how sharply preemptions onset. Empirically, α ≈ 2-4 for
vLLM's default scheduler (preemptions are rare until very close to capacity,
then increase rapidly).

### 3.3 Detecting the Critical Threshold

The critical threshold `U_c` is not fixed — it depends on the workload. Two
methods to detect it from benchmark data:

**Method 1: Piecewise Linear Regression**

Fit a two-segment piecewise linear model to (kv_cache_usage, latency_p99)
pairs across time windows:

```
L(U) = { a₁ × U + b₁   if U <= U_c
        { a₂ × U + b₂   if U > U_c

Minimize: Σ (L_observed - L_predicted)² over (a₁, b₁, a₂, b₂, U_c)
```

The breakpoint `U_c` is the estimated critical threshold. This can be solved via
grid search over candidate U_c values (1% granularity is sufficient) with OLS
on each segment.

**Method 2: CUSUM on Latency Residuals**

Apply CUSUM (already implemented in AIPerf's ramp detection) to the residuals
of latency after detrending by load:

```python
residuals = latency_p99 - linear_fit(concurrency)
cusum_detect(residuals)  # Finds changepoint where residuals jump
```

The timestamp of the changepoint maps back to a KV cache utilization level,
giving the empirical critical threshold.

### 3.4 Empirical Validation Approach

To validate the model, a benchmark run should:

1. Ramp load gradually (`--concurrency-ramp 1,100,5` — 1 to 100 in steps of 5)
2. Collect server metrics at high frequency (`--server-metrics-interval 1s`)
3. After the run, correlate `kv_cache_usage_perc` timeslice averages with
   `request_latency` percentiles in matching time windows

Expected observations:
- p50 latency barely changes until `U_c` (prefill is unaffected when blocks
  are available)
- p95/p99 latency increases first (tail latency is the canary)
- Preemption rate jumps sharply near 100% utilization
- TTFT is more sensitive than ITL (preempted requests restart prefill)

---

## 4. Preemption Event Correlation

### 4.1 Preemption as a Latency Amplifier

Each preemption event in vLLM has a concrete latency cost:

```
Cost of one recompute-style preemption:
  Lost progress:  All KV cache state for the preempted request
  Re-queue delay: Time waiting in the scheduler queue (again)
  Re-prefill:     Full prefill computation for ISL tokens (again)

Total cost ≈ queue_wait + prefill_time(ISL)

For ISL = 2048 tokens on Llama 3 70B:
  Prefill time ≈ 50-200ms (depending on batch size)
  Queue wait   ≈ 0-500ms (depending on queue depth)

  Total additional latency per preemption: 50-700ms
```

A single preemption can double or triple a request's total latency. If the same
request is preempted multiple times (pathological case), latency can increase by
an order of magnitude.

### 4.2 Temporal Analysis: Preemption Bursts

Preemptions tend to occur in bursts, not uniformly:

```
Preemption Count
(per second)
     │
   8 │           ██                           ██
     │           ██                           ██
   6 │          ███                          ███
     │          ███                          ███
   4 │         ████                         ████
     │         ████          █              ████
   2 │        █████         ███            █████
     │    █  ██████    █   █████      █   ██████
   0 │████████████████████████████████████████████
     └──────────┬──────────┬──────────┬──────────┬──
               t₁         t₂         t₃         t₄

     ↑ Burst 1: Many long      ↑ Burst 2: New batch of
       requests complete          long requests arrives
       simultaneously →           → sudden KV demand spike
       KV cache freed →
       new requests flood in
```

**Burst correlation algorithm:**

```python
def correlate_preemption_bursts(
    preemption_deltas: NDArray[np.float64],  # per-timeslice preemption count
    preemption_ts: NDArray[np.float64],       # timeslice timestamps
    latency_values: NDArray[np.float64],      # per-request latency
    latency_ts: NDArray[np.float64],          # per-request end timestamps
    window_ns: float,                         # correlation window
) -> list[PreemptionBurst]:
    """Identify preemption bursts and correlate with latency outliers.

    A burst is a contiguous window where preemption_deltas > 0.
    For each burst, collect latency observations that started during
    or just before the burst window (requests affected by preemption).

    Returns:
        List of PreemptionBurst with burst_start, burst_end,
        total_preemptions, affected_request_count,
        latency_p50_during, latency_p99_during,
        latency_p50_baseline, latency_p99_baseline.
    """
```

### 4.3 Mapping Server Counters to Client Observations

The `vllm:num_preemptions` counter is cumulative. To get per-interval rates:

```
preemption_rate[t] = (counter[t] - counter[t-1]) / (ts[t] - ts[t-1])
```

AIPerf's `CounterTimeslice` already computes this as `rate` (per-second) and
`total` (absolute delta) for each timeslice window.

**Cross-correlation with client latency:**

Given server-side preemption timeslices and client-side per-request latencies,
the correlation must account for **temporal offset**: a preemption at server
time `t_s` affects requests that were in-flight at time `t_s`, but those
requests complete at some later time `t_s + remaining_latency`. The client
observes the latency spike at completion time, not at preemption time.

```
Server timeline:  ──────[preemption burst]───────────────────
                         t_s

Client timeline:  ─────────────────────[latency spike]───────
                                        t_c = t_s + Δ

Where Δ ≈ remaining decode time for preempted request
        + re-queue wait
        + re-prefill time
```

This temporal offset `Δ` is itself a function of load. Under light load,
`Δ ≈ prefill_time(ISL)`. Under heavy load, `Δ ≈ queue_wait + prefill_time`.

**Implementation: Lagged Cross-Correlation**

```python
def lagged_cross_correlation(
    server_ts: NDArray[np.float64],    # preemption rate time series
    server_values: NDArray[np.float64],
    client_ts: NDArray[np.float64],    # latency percentile time series
    client_values: NDArray[np.float64],
    max_lag_ns: float,                 # maximum lag to search
    lag_step_ns: float,                # lag search granularity
) -> tuple[float, float]:
    """Find the lag that maximizes cross-correlation.

    Resamples both time series to a common grid, then computes
    Pearson correlation at each lag offset.

    Returns:
        (optimal_lag_ns, max_correlation)
    """
    # Resample to common grid (linear interpolation)
    common_ts = np.arange(
        max(server_ts[0], client_ts[0]),
        min(server_ts[-1], client_ts[-1]),
        lag_step_ns,
    )
    server_resampled = np.interp(common_ts, server_ts, server_values)
    client_resampled = np.interp(common_ts, client_ts, client_values)

    # Compute cross-correlation at each lag
    n_lags = int(max_lag_ns / lag_step_ns)
    correlations = np.zeros(2 * n_lags + 1)
    lags = np.arange(-n_lags, n_lags + 1) * lag_step_ns

    for i, lag_samples in enumerate(range(-n_lags, n_lags + 1)):
        if lag_samples >= 0:
            s = server_resampled[:len(server_resampled) - lag_samples]
            c = client_resampled[lag_samples:]
        else:
            s = server_resampled[-lag_samples:]
            c = client_resampled[:len(client_resampled) + lag_samples]

        if len(s) > 2:
            correlations[i] = np.corrcoef(s, c)[0, 1]

    best_idx = np.argmax(np.abs(correlations))
    return lags[best_idx], correlations[best_idx]
```

### 4.4 Expected Results

For a well-behaved vLLM deployment under increasing load:

```
Cross-correlation: preemption_rate vs. latency_p99

Correlation
     │
 1.0 │                        ●
     │                      ●   ●
 0.8 │                    ●       ●
     │                  ●           ●
 0.6 │                ●               ●
     │              ●                   ●
 0.4 │            ●                       ●
     │          ●                           ●
 0.2 │        ●                               ●
     │      ●                                   ●
 0.0 │━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●━━━
     │
     └────┬────┬────┬────┬────┬────┬────┬────┬────┬──
         -2s  -1s   0   +1s  +2s  +3s  +4s  +5s  +6s
                              ↑
                         Optimal lag ≈ 1-3 seconds
                         (re-queue + re-prefill time)
```

The optimal lag tells you the average re-computation penalty. If the lag is
consistently large (>5s), it suggests severe queue congestion during preemption
episodes.

---

## 5. Tokens-in-Flight as Client-Side KV Proxy

### 5.1 What tokens_in_flight Measures

The `tokens_in_flight` sweep metric (implemented in `src/aiperf/analysis/
sweep.py`) computes the instantaneous total tokens held in KV cache across all
active requests, as observed from the client side:

```python
# From tokens_in_flight_sweep():
# Events per request (up to 3):
#   +input_tokens            at start_ns         (prefill loads KV cache)
#   +output_tokens           at generation_start  (output tokens join KV)
#   -(input_tokens + output)  at end_ns           (KV cache freed)
```

This produces a step-function time series:

```
Tokens
  │
6K│              ┌──────────────┐
  │         ┌────┘              │         ┌────────┐
4K│    ┌────┘                   │    ┌────┘        │
  │    │                        └────┘             │
2K│────┘                                           └────
  │
  └──────────────────────────────────────────────────────
  t₀                                                  t_n

  Each step represents a request starting (adding ISL + OSL tokens)
  or completing (removing all tokens from KV cache)
```

The ICL-aware variant (`tokens_in_flight_sweep_icl`) refines this by modeling
gradual output token accumulation at each SSE chunk boundary, rather than adding
all output tokens at generation start.

### 5.2 Client Proxy vs. Server Ground Truth

`tokens_in_flight` is a **client-side approximation** of server-side KV cache
usage. The relationship between them:

```
tokens_in_flight (client view):
  = Σ (ISL_i + partial_OSL_i)  for all requests currently in-flight

kv_cache_usage_perc (server view):
  = allocated_blocks / total_blocks × 100
  = Σ ceil((ISL_i + generated_tokens_i) / block_size) / total_blocks × 100
```

**Sources of divergence:**

| Factor | Effect on Divergence | Direction |
|--------|---------------------|-----------|
| **Block granularity** | Server rounds up to block_size (typically 16 tokens). Client counts exact tokens. | Server > Client (slightly) |
| **Prefix caching** | Server shares physical blocks for common prefixes. Client counts full ISL per request. | Client >> Server |
| **Swap space** | Server may have swapped some blocks to CPU (still "allocated" but not in GPU cache). | Depends on accounting |
| **Scheduling delay** | Client considers a request "started" at HTTP send time; server hasn't allocated blocks until prefill begins. | Client slightly ahead |
| **Network latency** | Client observes request completion after network delay; server freed blocks earlier. | Client slightly behind |

### 5.3 Correlation Model

Define the normalized metrics:

```
TIF_norm = tokens_in_flight / max_kv_tokens
KV_norm  = kv_cache_usage_perc / 100
```

Where `max_kv_tokens` is the theoretical maximum tokens the KV cache can hold
(from server configuration or model properties).

Under ideal conditions (no prefix caching, no swap, negligible network delay):

```
TIF_norm ≈ KV_norm × (block_size / avg_occupancy_per_block)
```

Since PagedAttention has near-perfect block utilization (waste is at most
`block_size - 1` tokens per request), the ratio `block_size / avg_occupancy`
approaches 1.0 for long sequences.

**Correlation analysis:**

```python
def correlate_tif_with_kv_cache(
    tif_ts: NDArray[np.float64],
    tif_values: NDArray[np.float64],
    kv_ts: NDArray[np.float64],
    kv_values: NDArray[np.float64],
    window_ns: float = 1_000_000_000,  # 1-second windows
) -> CorrelationResult:
    """Compute windowed correlation between tokens_in_flight and kv_cache_usage.

    Resamples both series into aligned windows, computes Pearson r per window,
    and identifies divergence episodes.

    Returns:
        CorrelationResult with:
        - overall_r: Pearson correlation coefficient
        - windowed_r: per-window correlation time series
        - divergence_episodes: windows where |TIF_norm - KV_norm| > threshold
        - scaling_factor: best-fit linear coefficient (TIF = factor * KV)
    """
```

### 5.4 When They Diverge

Divergence between `tokens_in_flight` and `kv_cache_usage_perc` is informative:

**Case 1: TIF >> KV (client sees more pressure than server)**

```
tokens_in_flight
     │  ╱╱╱╱╱╱╱╱╱╱╱╱╱╱   ← Client view (no prefix sharing)
     │╱╱
     │╱
     │      ────────────   ← Server view (prefix cache active)
     │  ╱╱╱╱
     │╱╱
     └──────────────────
```

This indicates **prefix caching is effective**. The server is sharing KV blocks
across requests with common prefixes (system prompts, few-shot examples). The
client counts each request's full ISL independently, overcounting shared state.

**Actionable insight:** Prefix caching is working. The benchmark can sustain
higher concurrency than `tokens_in_flight` alone would suggest.

**Case 2: TIF << KV (server sees more pressure than client)**

This is unusual and suggests:
- Server-side speculative decoding (generating tokens the client hasn't
  requested)
- CPU swap space being counted as "used" in the KV cache metric
- Multiple tenants sharing the server (other clients consuming KV cache)

**Actionable insight:** The benchmark is not the only consumer of KV cache
resources. Results may not be reproducible in isolation.

**Case 3: TIF tracks KV closely, then diverges suddenly**

```
     │
KV%  │         ╱──────╲          ← Server: KV spikes then drops
     │       ╱╱        ╲╲           (preemption freed blocks)
     │     ╱╱            ╲╲
TIF  │   ╱╱╱╱╱╱╱╱╱╱╱╱╱╱   ╲    ← Client: TIF continues rising
     │ ╱╱                    ╲     (preempted requests still "in-flight"
     │╱                       ╲    from client's perspective)
     └────────────────────────────
                 ↑
            Preemption event:
            Server freed blocks,
            but client doesn't know yet
```

This divergence pattern is a **preemption signature** observable from combined
client+server data. The client still sees the request as in-flight (it hasn't
received a response), but the server has aborted it internally and will re-queue
it.

### 5.5 Scaling Factor Estimation

When prefix caching is active, we can estimate the effective prefix cache
sharing ratio:

```
sharing_ratio = 1 - (KV_norm / TIF_norm)
```

A `sharing_ratio` of 0.4 means 40% of token storage is shared via prefix
caching. This is a useful efficiency metric for evaluating prefix cache
effectiveness across different workloads.

---

## 6. Sequence Length Impact on KV Demand

### 6.1 KV Token Demand Model

The total KV cache demand at any instant is:

```
KV_demand(t) = Σ_i [ISL_i + generated_i(t)]    for all active requests i at time t
```

Where `generated_i(t)` is the number of output tokens generated so far for
request `i`. This is exactly what `tokens_in_flight` computes.

Decomposing by request phase:

```
KV_demand(t) = KV_prefill(t) + KV_decode(t)

KV_prefill(t) = Σ_j ISL_j                  for all requests j in prefill at time t
KV_decode(t)  = Σ_k (ISL_k + gen_k(t))     for all requests k in decode at time t
```

### 6.2 Sequence Length as an Amplifier

Two workloads with identical concurrency can have drastically different KV
demands:

```
Workload A: 10 concurrent requests, ISL=128, OSL=128
  KV_demand = 10 × (128 + ~64 avg generated) ≈ 1,920 tokens

Workload B: 10 concurrent requests, ISL=4096, OSL=512
  KV_demand = 10 × (4096 + ~256 avg generated) ≈ 43,520 tokens

Ratio: 22.7× more KV demand at same concurrency
```

This is why `effective_concurrency` alone is insufficient for predicting server
resource pressure — `tokens_in_flight` captures the actual memory impact.

### 6.3 ISL × Concurrency Interaction Surface

The relationship between ISL, concurrency, and KV cache pressure forms a 2D
surface:

```
KV Cache Usage %
100 │██████████████████████████████████████████████
    │██████████████████████████████████████████████
 90 │████████████████████████████████████████████░░
    │██████████████████████████████████████████░░░░
 80 │████████████████████████████████████████░░░░░░
    │██████████████████████████████████████░░░░░░░░
 70 │██████████████████████████████████░░░░░░░░░░░░
    │████████████████████████████████░░░░░░░░░░░░░░
 60 │██████████████████████████████░░░░░░░░░░░░░░░░
    │██████████████████████████░░░░░░░░░░░░░░░░░░░░
 50 │████████████████████████░░░░░░░░░░░░░░░░░░░░░░
    │██████████████████████░░░░░░░░░░░░░░░░░░░░░░░░
 40 │████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░
    │████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 30 │██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    │████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 20 │██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    │██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 10 │████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0 └──────────────────────────────────────────────
     128   256   512  1024  2048  4096  8192  16384
                    ISL (tokens)

    Legend: ░ = Stable region, █ = Saturation region

    Concurrency: 64 requests
    Model: Llama 3 70B (8×H100, ~122K max tokens in KV)

    At ISL=128:  KV demand = 64 × 128 = 8,192 tokens (7% usage)
    At ISL=4096: KV demand = 64 × 4096 = 262,144 tokens (overflow!)
```

### 6.4 Computing Maximum Safe Concurrency

Given known KV cache capacity, we can compute the maximum concurrency before
saturation:

```
max_concurrency = floor(max_kv_tokens / (avg_ISL + avg_OSL/2))
```

The `avg_OSL/2` term approximates the average generated tokens during the
request lifecycle (assuming uniform generation progress across concurrent
requests).

For bench runs that gradually increase concurrency, this formula predicts where
KV cache saturation will occur. The prediction can be validated against the
observed `kv_cache_usage_perc` timeline.

### 6.5 Per-Request KV Contribution Analysis

For each completed request, we can compute its peak KV contribution:

```python
peak_kv_contribution = ISL + OSL  # tokens
kv_duration = request_end_ns - request_start_ns  # nanoseconds
kv_token_seconds = peak_kv_contribution * kv_duration / 1e9  # token-seconds

# Aggregate: total KV token-seconds consumed per request
# Higher values = more KV cache "cost" to the system
```

This "KV token-seconds" metric captures both the magnitude and duration of a
request's KV cache impact. A request with ISL=4096, OSL=1000, latency=5s
consumes 5096 * 5 = 25,480 token-seconds, while a request with ISL=128,
OSL=100, latency=1s consumes 228 * 1 = 228 token-seconds — a 112x difference.

---

## 7. Prefix Cache Efficiency

### 7.1 Prefix Caching Mechanism

vLLM's prefix caching (automatic prefix caching / APC) stores computed KV
blocks for token sequences that are reused across requests. The typical case
is a system prompt shared by all requests in a batch:

```
Request 1: [system_prompt(2048 tokens) | user_query_1(128 tokens)]
Request 2: [system_prompt(2048 tokens) | user_query_2(256 tokens)]
Request 3: [system_prompt(2048 tokens) | user_query_3(64 tokens)]

Without prefix caching:
  KV blocks allocated = 3 × ceil(2048/16) + ceil(128/16) + ceil(256/16) + ceil(64/16)
                       = 3 × 128 + 8 + 16 + 4
                       = 412 blocks

With prefix caching:
  Shared prefix blocks = 1 × ceil(2048/16) = 128 blocks (computed once, shared by all 3)
  Unique suffix blocks = 8 + 16 + 4 = 28 blocks
  Total = 128 + 28 = 156 blocks

  Savings = 1 - 156/412 = 62% reduction in KV cache usage
```

### 7.2 Measuring Prefix Cache Effectiveness

vLLM exposes two counters:

```
vllm:prefix_cache_hits    — number of prefix blocks reused
vllm:prefix_cache_queries — number of prefix blocks looked up
```

**Hit rate computation:**

```
hit_rate = Δ(prefix_cache_hits) / Δ(prefix_cache_queries)
```

Where Δ denotes the delta over a time window (both are cumulative counters).

### 7.3 Hit Rate Impact on TTFT

Prefix cache hits directly reduce TTFT by eliminating redundant prefill
computation:

```
TTFT_expected = network_latency + prefill_time(unique_tokens)

where unique_tokens = ISL - prefix_cache_tokens
      prefix_cache_tokens = ISL × hit_rate × avg_prefix_fraction
```

When hit_rate drops, `unique_tokens` increases, and TTFT increases proportionally
to the additional prefill computation.

**Cross-correlation model:**

```
TTFT ∝ 1 / hit_rate    (approximately, when prefix is large relative to ISL)
```

More precisely:

```
TTFT = T_network + T_prefill_per_token × ISL × (1 - hit_rate × prefix_fraction)
       + T_decode_first_token
```

### 7.4 Detecting Prefix Cache Degradation

Prefix cache effectiveness can degrade for several reasons:

1. **Working set exceeds cache capacity**: Too many unique prefixes evict
   cached blocks before they can be reused
2. **Sequence length diversity**: Varied ISLs prevent prefix matching
3. **Low reuse ratio**: Each request has a unique prefix (no sharing possible)
4. **Hash collisions**: Internal prefix trie becomes inefficient (rare)

**Detection algorithm:**

```python
def detect_prefix_cache_degradation(
    hit_rate_ts: NDArray[np.float64],   # windowed hit rate
    ttft_p50_ts: NDArray[np.float64],   # windowed TTFT median
    timestamps: NDArray[np.float64],
    window_ns: float,
) -> list[DegradationEpisode]:
    """Detect episodes where prefix cache hit rate drops and TTFT rises.

    Uses CUSUM on -hit_rate (inverted — looking for decreases) to find
    changepoints, then validates by checking TTFT increase in same window.

    Returns:
        List of DegradationEpisode with start_ns, end_ns,
        hit_rate_before, hit_rate_during, ttft_increase_pct.
    """
```

### 7.5 Diagnostic Flowchart

```
TTFT is high. Why?
─────────────────
        │
        ▼
┌─────────────────────┐     No    ┌──────────────────────┐
│ prefix_cache_queries │────────▶ │ Prefix caching is    │
│ > 0?                 │          │ disabled. TTFT =      │
└──────────┬──────────┘          │ full prefill + network │
           │ Yes                  └──────────────────────┘
           ▼
┌─────────────────────┐     < 0.5  ┌──────────────────────┐
│ hit_rate =           │──────────▶│ Low reuse: workload   │
│ hits / queries?      │           │ has diverse prefixes. │
└──────────┬──────────┘           │ Expected TTFT ≈ full  │
           │ >= 0.5                │ prefill time.         │
           ▼                       └──────────────────────┘
┌─────────────────────┐     Yes   ┌──────────────────────┐
│ hit_rate decreased   │─────────▶│ Cache thrashing:      │
│ during the run?      │          │ working set exceeds   │
│ (CUSUM on -hit_rate) │          │ cache capacity.       │
└──────────┬──────────┘          │ Reduce concurrency    │
           │ No                   │ or increase GPU count.│
           ▼                      └──────────────────────┘
┌─────────────────────┐
│ TTFT correlated with │
│ kv_cache_usage_perc? │
│ (Pearson r > 0.6)    │
└──────────┬──────────┘
    Yes │        │ No
        ▼        ▼
┌──────────┐  ┌──────────────────┐
│ KV cache │  │ TTFT dominated   │
│ pressure │  │ by network or    │
│ causing  │  │ model compute    │
│ schedule │  │ (not memory      │
│ delays   │  │ pressure)        │
└──────────┘  └──────────────────┘
```

---

## 8. Memory Pressure Cascade

### 8.1 The Causal Chain

KV cache saturation triggers a cascading failure mode that propagates through
multiple observable metrics. Understanding the full causal chain is essential for
root-cause analysis.

```
Stage 1: GPU Memory Fills
──────────────────────────
  gpu_memory_used ↑ → KV block pool exhausted

      ↓ triggers

Stage 2: KV Cache Saturation
─────────────────────────────
  kv_cache_usage_perc → 95-100%
  Free blocks → 0

      ↓ triggers

Stage 3: Preemption Events
──────────────────────────
  num_preemptions ↑ (counter jumps)
  Victim request loses all KV state
  Victim re-enters waiting queue

      ↓ triggers

Stage 4: Queue Buildup
──────────────────────
  num_requests_waiting ↑
  num_requests_running may temporarily ↓ (freed slots)
  request_queue_time_seconds ↑

      ↓ triggers

Stage 5: TTFT Spike
────────────────────
  Preempted requests: TTFT = queue_wait + full_re_prefill
  New requests: TTFT = queue_wait (longer queue)

      ↓ triggers

Stage 6: ITL Variance Increase
──────────────────────────────
  Running batch changes composition (preempted reqs removed)
  Batch size oscillates → decode throughput oscillates
  ITL variance increases even for non-preempted requests

      ↓ triggers

Stage 7: End-to-End Latency Degradation
────────────────────────────────────────
  request_latency ↑ (TTFT + increased ITL)
  effective_throughput ↓ (requests take longer, fewer complete per second)
  tokens_in_flight remains high (requests stuck in system longer)

      ↓ creates feedback loop

Stage 8: Feedback Amplification
───────────────────────────────
  Slower completions → tokens stay in KV cache longer
  → KV cache usage stays high → more preemptions → ...
```

### 8.2 Temporal Signature

Each stage has a characteristic temporal delay from the root cause:

```
Time since KV cache saturation onset (t₀):

Stage  Metric                    Delay     Duration
───────────────────────────────────────────────────────
  1    gpu_memory_used           t₀        Continuous
  2    kv_cache_usage_perc       t₀        Continuous
  3    num_preemptions           t₀+0.1s   Burst (0.5-2s)
  4    num_requests_waiting      t₀+0.2s   Sustained
  4    request_queue_time        t₀+0.5s   Growing
  5    TTFT (client-observed)    t₀+1-5s   Sustained
  6    ITL variance              t₀+2-10s  Sustained
  7    request_latency p99       t₀+5-30s  Sustained
  7    effective_throughput      t₀+5-30s  Decreasing
```

The delays compound: each stage must propagate through the system before the
next becomes observable. This makes real-time detection challenging — by the
time `request_latency` shows degradation, the root cause occurred 5-30 seconds
earlier.

### 8.3 Detection from Available Metrics

We can detect the cascade at different stages using metrics already available
in AIPerf:

**Stage 2 Detection (earliest, requires server metrics):**

```python
def detect_kv_saturation_onset(
    kv_usage_ts: NDArray[np.float64],
    kv_usage_values: NDArray[np.float64],
    threshold: float = 0.90,
    sustained_ns: float = 2_000_000_000,  # 2 seconds
) -> float | None:
    """Find the first timestamp where KV cache stays above threshold
    for at least sustained_ns duration.

    Returns:
        Onset timestamp in nanoseconds, or None if never saturated.
    """
    above = kv_usage_values >= threshold
    if not above.any():
        return None

    # Find runs of consecutive above-threshold observations
    # Return start of first run that exceeds sustained_ns
    transitions = np.diff(above.astype(np.int8))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1

    # Handle edge cases (starts above, ends above)
    if above[0]:
        starts = np.concatenate([[0], starts])
    if above[-1]:
        ends = np.concatenate([ends, [len(above)]])

    for s, e in zip(starts, ends):
        duration = kv_usage_ts[min(e, len(kv_usage_ts) - 1)] - kv_usage_ts[s]
        if duration >= sustained_ns:
            return float(kv_usage_ts[s])

    return None
```

**Stage 5 Detection (client-side only, no server metrics needed):**

```python
def detect_ttft_regime_change(
    ttft_values: NDArray[np.float64],
    ttft_timestamps: NDArray[np.float64],
) -> float | None:
    """Detect TTFT regime change using CUSUM.

    A sudden sustained increase in TTFT indicates KV cache pressure
    is causing scheduling delays. This is detectable without server
    metrics — it's a client-side signal of server-side memory pressure.

    Returns:
        Changepoint timestamp, or None if no regime change detected.
    """
    # Reuse AIPerf's existing CUSUM implementation
    from aiperf.analysis.ramp_detection import cusum_steady_state_window

    # CUSUM on TTFT values (looking for upward shift)
    # The ramp-up boundary is where TTFT transitions from stable to elevated
    window_start, _ = cusum_steady_state_window(
        ttft_timestamps, ttft_values
    )
    return window_start
```

**Stage 7 Detection (compound signal):**

```python
def detect_throughput_collapse(
    throughput_ts: NDArray[np.float64],
    throughput_values: NDArray[np.float64],
    tif_ts: NDArray[np.float64],
    tif_values: NDArray[np.float64],
) -> bool:
    """Detect throughput collapse while tokens_in_flight remains high.

    This pattern is the "smoking gun" for KV cache pressure:
    throughput drops (fewer completions per second) while tokens_in_flight
    stays elevated (requests are stuck in the system, consuming KV cache).

    In a healthy system, throughput and tokens_in_flight move together.
    In a saturated system, they diverge.
    """
    # Compute rolling correlation in the last 30% of the run
    # If correlation is negative (throughput down, TIF up) → cascade detected
    ...
```

### 8.4 Cascade Severity Classification

Based on the detection results, classify the cascade severity:

```python
@dataclass(frozen=True)
class CascadeSeverity:
    """KV cache pressure cascade severity assessment."""

    level: str              # "none", "mild", "moderate", "severe"
    kv_peak_usage: float    # Peak KV cache usage observed
    preemption_count: int   # Total preemptions during run
    ttft_inflation: float   # Ratio: TTFT_under_pressure / TTFT_baseline
    throughput_loss: float  # Fraction: (peak_throughput - min_throughput) / peak_throughput
    onset_timestamp: float | None  # When cascade began (nanoseconds)
```

Classification thresholds:

| Level | KV Peak | Preemptions | TTFT Inflation | Throughput Loss |
|-------|---------|-------------|----------------|-----------------|
| None | < 80% | 0 | < 1.1x | < 5% |
| Mild | 80-90% | 1-10 | 1.1-1.5x | 5-15% |
| Moderate | 90-95% | 10-100 | 1.5-3.0x | 15-30% |
| Severe | > 95% | > 100 | > 3.0x | > 30% |

---

## 9. Implementation: Correlation Analysis Engine

### 9.1 Architecture Overview

The KV cache correlation analysis fits naturally into AIPerf's analyzer plugin
architecture:

```
┌─────────────────────────────────────────────────────────┐
│ RecordsManager                                          │
│                                                         │
│  ┌──────────────────┐  ┌──────────────────────────┐    │
│  │MetricsAccumulator│  │ServerMetricsAccumulator   │    │
│  │                  │  │                           │    │
│  │ ColumnStore      │  │ ServerMetricsTimeSeries   │    │
│  │ ├─ start_ns      │  │ ├─ kv_cache_usage_perc   │    │
│  │ ├─ end_ns        │  │ ├─ num_preemptions       │    │
│  │ ├─ latency       │  │ ├─ prefix_cache_hits     │    │
│  │ ├─ ttft          │  │ ├─ prefix_cache_queries  │    │
│  │ ├─ itl           │  │ ├─ num_requests_running  │    │
│  │ └─ isl/osl       │  │ └─ request_queue_time    │    │
│  └────────┬─────────┘  └────────────┬─────────────┘    │
│           │                          │                   │
│           ▼                          ▼                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ SweepCurves (from MetricsAccumulator)            │   │
│  │ ├─ effective_concurrency                         │   │
│  │ ├─ effective_throughput                          │   │
│  │ └─ tokens_in_flight                             │   │
│  └───────────────────────┬──────────────────────────┘   │
│                          │                               │
│                          ▼                               │
│  ┌──────────────────────────────────────────────────┐   │
│  │ KVCacheCorrelationAnalyzer (AnalyzerProtocol)    │   │
│  │                                                   │   │
│  │ Inputs:                                           │   │
│  │   - SweepCurves.tokens_in_flight                  │   │
│  │   - SweepCurves.effective_concurrency             │   │
│  │   - ColumnStore (latency, TTFT, ISL, OSL)         │   │
│  │   - ServerMetricsTimeSeries (KV %, preemptions)   │   │
│  │                                                   │   │
│  │ Outputs:                                          │   │
│  │   - KVCacheCorrelationSummary                     │   │
│  │     ├─ CascadeSeverity                            │   │
│  │     ├─ CorrelationCoefficients                    │   │
│  │     ├─ PreemptionBurstAnalysis                    │   │
│  │     ├─ PrefixCacheEfficiency                      │   │
│  │     └─ KVDemandModel                              │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                               │
│                          ▼                               │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Exporters                                         │   │
│  │ ├─ ConsoleKVCacheExporter (warnings + summary)    │   │
│  │ ├─ JsonKVCacheExporter (full correlation data)    │   │
│  │ └─ CsvKVCacheExporter (windowed time series)      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 9.2 Plugin Registration

```yaml
# plugins.yaml addition
analyzer:
  kv_cache_correlation:
    class: aiperf.analysis.kv_cache_analyzer:KVCacheCorrelationAnalyzer
    description: >
      Correlates KV cache pressure metrics with latency degradation.
      Requires server metrics (--server-metrics-url) for full analysis;
      provides client-side-only analysis from tokens_in_flight when
      server metrics are unavailable.
    metadata:
      requires_server_metrics: false  # Degraded mode without them
      requires_gpu_telemetry: false   # Optional enrichment
```

### 9.3 Analyzer Protocol Implementation

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aiperf.common.accumulator_protocols import AnalyzerProtocol


@dataclass(frozen=True, slots=True)
class KVCorrelationConfig:
    """Configuration for KV cache correlation analysis."""

    # Windowing
    window_duration_ns: float = 1_000_000_000  # 1-second correlation windows

    # Thresholds
    kv_saturation_threshold: float = 0.90
    preemption_burst_min_count: int = 3
    correlation_significance_threshold: float = 0.6

    # Cross-correlation
    max_lag_ns: float = 10_000_000_000   # 10-second max lag search
    lag_step_ns: float = 100_000_000     # 100ms granularity

    # Prefix cache
    low_hit_rate_threshold: float = 0.5


@dataclass(frozen=True, slots=True)
class CorrelationCoefficients:
    """Pearson correlation coefficients between metric pairs."""

    # KV cache vs. latency
    kv_usage_vs_latency_p50: float | None
    kv_usage_vs_latency_p99: float | None
    kv_usage_vs_ttft_p50: float | None
    kv_usage_vs_ttft_p99: float | None

    # Tokens-in-flight vs. server metrics
    tif_vs_kv_usage: float | None
    tif_vs_latency_p99: float | None

    # Preemptions vs. latency
    preemption_rate_vs_latency_p99: float | None
    preemption_rate_vs_ttft_p99: float | None
    preemption_optimal_lag_ns: float | None

    # Concurrency vs. KV
    concurrency_vs_kv_usage: float | None


@dataclass(frozen=True, slots=True)
class PreemptionBurst:
    """A detected burst of preemption events."""

    start_ns: float
    end_ns: float
    total_preemptions: int
    affected_requests_estimate: int
    latency_p99_during: float
    latency_p99_baseline: float
    latency_inflation: float  # p99_during / p99_baseline


@dataclass(frozen=True, slots=True)
class PrefixCacheAnalysis:
    """Prefix cache efficiency analysis."""

    overall_hit_rate: float | None
    hit_rate_trend: str | None  # "stable", "declining", "improving"
    ttft_correlation_with_hit_rate: float | None
    estimated_kv_savings_pct: float | None
    tif_kv_divergence_ratio: float | None  # tokens_in_flight / kv_cache_usage


@dataclass(frozen=True, slots=True)
class KVDemandModel:
    """Modeled KV cache demand from client-side data."""

    peak_tokens_in_flight: float
    avg_tokens_in_flight: float
    estimated_max_kv_tokens: float | None  # From server config, if available
    estimated_peak_utilization: float | None
    max_safe_concurrency: float | None

    # ISL impact analysis
    avg_isl: float
    avg_osl: float
    kv_tokens_per_request: float  # avg ISL + avg OSL
    isl_contribution_pct: float   # avg ISL / (avg ISL + avg OSL)


@dataclass(frozen=True, slots=True)
class KVCacheCorrelationSummary:
    """Complete KV cache correlation analysis results."""

    cascade_severity: str  # "none", "mild", "moderate", "severe"
    cascade_onset_ns: float | None

    correlations: CorrelationCoefficients
    preemption_bursts: list[PreemptionBurst]
    prefix_cache: PrefixCacheAnalysis
    kv_demand: KVDemandModel

    # Per-window time series for export
    window_timestamps_ns: list[float]
    window_kv_usage: list[float]
    window_latency_p99: list[float]
    window_tif: list[float]
    window_preemption_rate: list[float]

    has_server_metrics: bool
    analysis_mode: str  # "full" or "client_only"

    def to_json(self) -> dict:
        """Serialize to JSON-compatible dict for export."""
        ...

    def to_csv_rows(self) -> list[dict]:
        """Serialize windowed time series to CSV rows."""
        ...
```

### 9.4 Sliding Window Correlation

The core analysis operates on aligned time windows. Given two time series with
different sampling rates (server metrics at ~1-5s intervals, client requests at
variable rates), we align them to common windows:

```python
def compute_windowed_correlation(
    series_a_ts: NDArray[np.float64],
    series_a_values: NDArray[np.float64],
    series_b_ts: NDArray[np.float64],
    series_b_values: NDArray[np.float64],
    window_ns: float,
    start_ns: float,
    end_ns: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Compute windowed aggregation and overall correlation.

    For each window [t, t + window_ns):
      - series_a: mean of all observations in window
      - series_b: mean of all observations in window

    Windows with fewer than 2 observations in either series are excluded.

    Returns:
        (window_means_a, window_means_b, pearson_r)
    """
    n_windows = int((end_ns - start_ns) / window_ns)
    means_a = np.full(n_windows, np.nan)
    means_b = np.full(n_windows, np.nan)

    for i in range(n_windows):
        w_start = start_ns + i * window_ns
        w_end = w_start + window_ns

        mask_a = (series_a_ts >= w_start) & (series_a_ts < w_end)
        mask_b = (series_b_ts >= w_start) & (series_b_ts < w_end)

        if mask_a.sum() >= 1:
            means_a[i] = np.mean(series_a_values[mask_a])
        if mask_b.sum() >= 1:
            means_b[i] = np.mean(series_b_values[mask_b])

    # Correlation on windows where both have data
    valid = ~np.isnan(means_a) & ~np.isnan(means_b)
    if valid.sum() < 3:
        return means_a, means_b, np.nan

    r = np.corrcoef(means_a[valid], means_b[valid])[0, 1]
    return means_a, means_b, float(r)
```

### 9.5 Change-Point Detection for KV Cache Regime Changes

Extend the existing CUSUM infrastructure to detect KV cache regime transitions:

```python
def detect_kv_regime_changes(
    kv_ts: NDArray[np.float64],
    kv_values: NDArray[np.float64],
    min_segment_ns: float = 5_000_000_000,  # 5 seconds minimum segment
) -> list[KVRegime]:
    """Detect KV cache utilization regime changes.

    Uses a simplified PELT-like approach: grid search for the single
    best changepoint, then recursively search each segment.

    Regimes:
      - "low":       KV < 50%   (ample headroom)
      - "moderate":  50-80%     (approaching pressure)
      - "high":      80-95%     (pressure zone, batching affected)
      - "saturated": > 95%      (preemption zone)

    Returns:
        List of KVRegime(start_ns, end_ns, avg_kv_usage, label)
    """
    regimes = []

    def classify(avg_kv: float) -> str:
        if avg_kv < 0.50:
            return "low"
        if avg_kv < 0.80:
            return "moderate"
        if avg_kv < 0.95:
            return "high"
        return "saturated"

    def find_best_split(
        ts: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> int | None:
        """Find the index that minimizes total within-segment variance."""
        n = len(values)
        if n < 4:
            return None

        min_idx = max(2, int(min_segment_ns / np.median(np.diff(ts))))
        if min_idx >= n - 2:
            return None

        best_cost = np.inf
        best_split = None

        # O(n) using cumulative sums
        cumsum = np.cumsum(values)
        cumsum_sq = np.cumsum(values ** 2)

        for k in range(min_idx, n - min_idx):
            # Segment 1: [0, k)
            n1 = k
            mean1 = cumsum[k - 1] / n1
            var1 = cumsum_sq[k - 1] / n1 - mean1 ** 2

            # Segment 2: [k, n)
            n2 = n - k
            mean2 = (cumsum[-1] - cumsum[k - 1]) / n2
            var2 = (cumsum_sq[-1] - cumsum_sq[k - 1]) / n2 - mean2 ** 2

            cost = n1 * max(0, var1) + n2 * max(0, var2)
            if cost < best_cost:
                best_cost = cost
                best_split = k

        # BIC penalty: only split if reduction is significant
        total_var = np.var(values)
        if best_split is not None:
            reduction = 1.0 - best_cost / (n * total_var) if total_var > 0 else 0
            if reduction < 0.15:  # Less than 15% variance reduction — not worth splitting
                return None

        return best_split

    # Recursive splitting
    def segment(ts, values, depth=0):
        if depth > 4 or len(ts) < 4:
            avg = float(np.mean(values))
            regimes.append(KVRegime(
                start_ns=float(ts[0]),
                end_ns=float(ts[-1]),
                avg_kv_usage=avg,
                label=classify(avg),
            ))
            return

        split = find_best_split(ts, values)
        if split is None:
            avg = float(np.mean(values))
            regimes.append(KVRegime(
                start_ns=float(ts[0]),
                end_ns=float(ts[-1]),
                avg_kv_usage=avg,
                label=classify(avg),
            ))
            return

        segment(ts[:split], values[:split], depth + 1)
        segment(ts[split:], values[split:], depth + 1)

    segment(kv_ts, kv_values)
    return regimes
```

### 9.6 Automated Alert Thresholds

The analyzer should emit warnings/alerts when correlation patterns indicate
problems:

```python
def generate_alerts(
    summary: KVCacheCorrelationSummary,
) -> list[KVCacheAlert]:
    """Generate human-readable alerts from correlation analysis.

    Alert levels:
      - INFO:    Informational observation (e.g., "prefix caching is effective")
      - WARNING: Potential issue detected (e.g., "KV cache reached 85%")
      - ERROR:   Significant degradation (e.g., "47 preemptions detected")
    """
    alerts = []

    # KV saturation alert
    if summary.kv_demand.estimated_peak_utilization is not None:
        peak = summary.kv_demand.estimated_peak_utilization
        if peak > 0.95:
            alerts.append(KVCacheAlert(
                level="ERROR",
                message=f"KV cache saturated at {peak:.0%}. "
                        f"{len(summary.preemption_bursts)} preemption bursts detected. "
                        "Latency results are degraded by memory pressure.",
            ))
        elif peak > 0.80:
            alerts.append(KVCacheAlert(
                level="WARNING",
                message=f"KV cache reached {peak:.0%}. "
                        "Consider reducing concurrency or sequence length "
                        "to avoid latency degradation.",
            ))

    # TIF-only alert (no server metrics)
    if not summary.has_server_metrics:
        tif_peak = summary.kv_demand.peak_tokens_in_flight
        alerts.append(KVCacheAlert(
            level="INFO",
            message=f"Peak tokens in flight: {tif_peak:,.0f}. "
                    "Enable --server-metrics-url for KV cache utilization tracking.",
        ))

    # Preemption correlation
    r = summary.correlations.preemption_rate_vs_latency_p99
    if r is not None and abs(r) > 0.6:
        lag = summary.correlations.preemption_optimal_lag_ns
        lag_s = lag / 1e9 if lag is not None else 0
        alerts.append(KVCacheAlert(
            level="WARNING",
            message=f"Preemption-latency correlation: r={r:.2f} "
                    f"(lag={lag_s:.1f}s). "
                    "Preemption events are driving tail latency spikes.",
        ))

    # Prefix cache degradation
    pc = summary.prefix_cache
    if pc.hit_rate_trend == "declining":
        alerts.append(KVCacheAlert(
            level="WARNING",
            message="Prefix cache hit rate is declining during the run. "
                    "Working set may exceed cache capacity.",
        ))

    return alerts
```

### 9.7 Console Output Format

When the analyzer runs, it should produce a compact console summary:

```
KV Cache Correlation Analysis
├── Mode: Full (server metrics available)
├── Cascade Severity: Moderate
│   └── Onset: +45.2s into profiling
├── KV Cache Peak: 93.2%
├── Preemption Bursts: 3 (47 total preemptions)
│   ├── Burst 1 (+45.2s): 12 preemptions, latency 2.3× baseline
│   ├── Burst 2 (+52.8s): 23 preemptions, latency 3.1× baseline
│   └── Burst 3 (+61.0s): 12 preemptions, latency 1.8× baseline
├── Prefix Cache: hit_rate=0.72 (stable)
├── Correlations:
│   ├── KV usage vs. latency p99:  r=0.87 ****
│   ├── TIF vs. KV usage:          r=0.94 ****
│   └── Preemptions vs. TTFT p99:  r=0.76 *** (lag=1.2s)
└── KV Demand Model:
    ├── Peak tokens in flight: 98,432
    ├── Avg ISL contribution: 78.3%
    └── Max safe concurrency (estimated): ~42 requests

⚠ WARNING: KV cache reached 93.2%. Preemption events are driving
  tail latency spikes (r=0.76, 47 preemptions).
```

### 9.8 JSON Export Structure

```json
{
  "kv_cache_correlation": {
    "analysis_mode": "full",
    "cascade": {
      "severity": "moderate",
      "onset_ns": 1706745645200000000,
      "onset_relative_s": 45.2
    },
    "correlations": {
      "kv_usage_vs_latency_p50": 0.42,
      "kv_usage_vs_latency_p99": 0.87,
      "kv_usage_vs_ttft_p50": 0.65,
      "kv_usage_vs_ttft_p99": 0.81,
      "tif_vs_kv_usage": 0.94,
      "tif_vs_latency_p99": 0.83,
      "preemption_rate_vs_latency_p99": 0.76,
      "preemption_rate_vs_ttft_p99": 0.71,
      "preemption_optimal_lag_ns": 1200000000,
      "concurrency_vs_kv_usage": 0.91
    },
    "preemption_bursts": [
      {
        "start_ns": 1706745645200000000,
        "end_ns": 1706745647500000000,
        "total_preemptions": 12,
        "affected_requests_estimate": 8,
        "latency_p99_during_ms": 342.1,
        "latency_p99_baseline_ms": 148.7,
        "latency_inflation": 2.3
      }
    ],
    "prefix_cache": {
      "overall_hit_rate": 0.72,
      "hit_rate_trend": "stable",
      "ttft_correlation_with_hit_rate": -0.58,
      "estimated_kv_savings_pct": 31.2,
      "tif_kv_divergence_ratio": 1.45
    },
    "kv_demand": {
      "peak_tokens_in_flight": 98432,
      "avg_tokens_in_flight": 52100,
      "estimated_max_kv_tokens": 122000,
      "estimated_peak_utilization": 0.807,
      "max_safe_concurrency": 42,
      "avg_isl": 2048,
      "avg_osl": 512,
      "kv_tokens_per_request": 2560,
      "isl_contribution_pct": 80.0
    },
    "time_series": {
      "window_duration_ns": 1000000000,
      "windows": [
        {
          "timestamp_ns": 1706745600000000000,
          "kv_usage_pct": 45.2,
          "latency_p99_ms": 142.3,
          "tokens_in_flight": 35200,
          "preemption_rate": 0.0
        }
      ]
    }
  }
}
```

### 9.9 Client-Only Mode

When server metrics are unavailable (`--server-metrics-url` not set), the
analyzer operates in degraded "client-only" mode:

**Available in client-only mode:**
- `tokens_in_flight` (full sweep analysis)
- `effective_concurrency` and all sweep metrics
- Per-request latency, TTFT, ITL, ISL, OSL
- KV demand model (theoretical, based on token counts)
- TTFT regime change detection (CUSUM on client latency)

**NOT available without server metrics:**
- Actual KV cache utilization percentage
- Preemption counts and burst analysis
- Prefix cache hit rate
- Server-side queue depth and queue time
- KV usage vs. latency correlation (replaced by TIF vs. latency)

**Client-only heuristics:**

When `tokens_in_flight` exceeds a user-configurable threshold (or an estimated
max from model configuration), the analyzer infers likely KV pressure:

```python
def estimate_kv_pressure_from_tif(
    tif_ts: NDArray[np.float64],
    tif_values: NDArray[np.float64],
    model_max_tokens: float | None,  # From --model-max-kv-tokens or model config
) -> str:
    """Estimate KV cache pressure level from client-side tokens_in_flight.

    Without server metrics, we use TIF as a proxy. If the user provides
    model_max_tokens (or it's inferred from model config), we can estimate
    utilization. Otherwise, we use TIF trends + latency correlation.
    """
    if model_max_tokens is not None:
        peak_utilization = np.max(tif_values) / model_max_tokens
        if peak_utilization > 0.95:
            return "saturated"
        if peak_utilization > 0.80:
            return "high"
        if peak_utilization > 0.50:
            return "moderate"
        return "low"

    # Without max_tokens, use trend analysis
    # If TIF is rising and TTFT is rising → likely pressure
    # This is less precise but better than nothing
    return "unknown"
```

---

## 10. Academic Context & Literature

### 10.1 PagedAttention and vLLM Scheduling

**Kwon et al. (2023). "Efficient Memory Management for Large Language Model
Serving with PagedAttention."** *SOSP 2023.*

The foundational paper for modern KV cache management. Key contributions:
- Virtual memory abstraction for KV cache blocks
- Near-zero waste allocation (eliminates fragmentation from contiguous KV slots)
- Copy-on-write for shared prefixes (enables prefix caching)
- Preemption via swap or recompute when physical memory is exhausted

The paper reports that PagedAttention increases throughput by 2-4x over naive
contiguous allocation by eliminating memory fragmentation and enabling larger
batch sizes. However, it does not analyze the latency impact when the cache is
full — that is the focus of this research document.

**Relevance to AIPerf:** vLLM's `kv_cache_usage_perc` metric directly reports
the PagedAttention block pool utilization. The `num_preemptions` counter tracks
the recompute-style preemptions described in Section 5.3 of the paper.

### 10.2 Scheduling and Preemption Policies

**Yu et al. (2022). "Orca: A Distributed Serving System for
Transformer-Based Generative Models."** *OSDI 2022.*

Introduced iteration-level scheduling for LLM serving, where the scheduler
makes decisions at each iteration (token generation step) rather than at
request arrival. Key insight: continuous batching allows mixing prefill and
decode operations, but batch composition affects per-iteration latency.

**Relevance:** When KV cache is full and preemption occurs, the scheduler
must evict a running request. The eviction policy (typically lowest-priority
or largest-KV-footprint) determines which requests experience the latency
penalty. AIPerf can detect the *effects* of these policies via the correlation
between sequence length and preemption-induced latency spikes.

**Agrawal et al. (2024). "Taming Throughput-Latency Tradeoff in LLM Inference
with Sarathi-Serve."** *OSDI 2024.*

Introduces chunked prefill to eliminate prefill-decode interference. Under
chunked prefill, large prefills are split into chunks that interleave with
decode steps, preventing the long stalls that occur when a large prefill
monopolizes the GPU.

**Relevance:** With chunked prefill, TTFT increases (prefill is spread over
multiple iterations) but ITL variance decreases (no more full-prefill stalls
during decode). This changes the correlation signature: under chunked prefill,
KV cache pressure manifests more uniformly across tokens rather than
concentrating at the first token.

### 10.3 Memory-Efficient Attention Variants

**Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact
Attention with IO-Awareness."** *NeurIPS 2022.*

FlashAttention reduces the memory footprint of the attention computation itself
(activations), but does NOT reduce KV cache size. The KV cache stores past
key-value pairs for reuse; FlashAttention optimizes how those stored pairs are
accessed during attention computation.

**Relevance:** FlashAttention improves prefill speed (faster attention
computation on input tokens) but does not change the KV cache pressure dynamics.
A workload that saturates the KV cache will still saturate it with
FlashAttention enabled — the tokens just get processed faster.

**Sheng et al. (2023). "FlexGen: High-Throughput Generative Inference of Large
Language Models with a Single GPU."** *ICML 2023.*

Explores offloading KV cache to CPU memory and disk for throughput-oriented
workloads. Demonstrates that KV cache is the dominant memory bottleneck for
long-context generation.

**Relevance:** FlexGen's approach (trading latency for throughput via
offloading) is the extreme case of vLLM's swap-based preemption. The latency
cost of CPU↔GPU KV transfer is a direct contributor to the preemption penalty
measured in Section 4.

### 10.4 Prefix Caching and KV Reuse

**Zheng et al. (2023). "SGLang: Efficient Execution of Structured Language
Model Programs."** (RadixAttention)

Introduces RadixAttention, which uses a radix tree to identify and reuse
shared KV cache prefixes across requests. More flexible than vLLM's
block-level prefix matching (handles variable-length prefixes efficiently).

**Relevance:** RadixAttention-style prefix caching is the mechanism behind
vLLM's `prefix_cache_hits` / `prefix_cache_queries` counters. The hit rate
directly measures how much KV cache reuse is occurring, which determines the
gap between `tokens_in_flight` (client-side, no reuse assumed) and
`kv_cache_usage_perc` (server-side, with reuse).

### 10.5 Queuing Theory for LLM Serving

**Standard M/G/c queue model** provides the theoretical foundation for
understanding latency under KV cache pressure:

- **Arrival rate (lambda):** Request arrival rate (from client benchmark load)
- **Service time (S):** Per-request processing time (prefill + decode)
- **Servers (c):** Effective parallelism (limited by KV cache capacity, not
  just GPU compute)

The key insight is that in LLM serving, the number of "servers" is not fixed —
it depends on KV cache availability. As KV cache fills, the effective
parallelism `c` decreases (fewer concurrent requests can be served without
preemption), which increases the effective utilization `rho = lambda * E[S] / c`.

This explains the non-linear latency behavior: the system transitions from an
`M/G/c` queue (many servers, low utilization, low latency) to an `M/G/1` queue
(effectively single-server due to serial preemption handling, high utilization,
high latency).

**Pollaczek-Khinchine formula** for mean waiting time in `M/G/1`:

```
E[W] = (rho / (1 - rho)) * (1 + C_s^2) / 2 * E[S]
```

Where `C_s = sigma_S / E[S]` is the coefficient of variation of service time.
As `rho → 1` (approaching KV saturation), `E[W] → infinity`.

### 10.6 Correlation Methods

**Pearson vs. Spearman correlation for this problem:**

Pearson correlation captures linear relationships and is appropriate when we
expect a roughly proportional relationship (more KV usage → proportionally more
latency). However, the relationship is non-linear (Section 3.1).

Spearman rank correlation is more robust to non-linearity and is appropriate
for detecting monotonic relationships regardless of functional form. For the
KV cache problem, Spearman is preferred when:
- The relationship has a "hockey stick" shape (flat, then steep)
- Outliers are present (preemption bursts create extreme latency values)
- We care about "does more KV usage reliably predict higher latency?" rather
  than "by how much?"

**Recommendation:** Report both Pearson r (for linear strength) and Spearman
rho (for monotonic strength). When Spearman >> Pearson, it confirms non-linear
behavior (the hockey stick curve).

### 10.7 Time Series Cross-Correlation

**Box, Jenkins, Reinsel (2008). "Time Series Analysis: Forecasting and
Control."** *Wiley.*

Standard reference for cross-correlation function (CCF) analysis. The CCF
between two time series X_t and Y_t at lag k is:

```
CCF(k) = Cov(X_t, Y_{t+k}) / (Std(X_t) * Std(Y_t))
```

For the preemption-latency correlation (Section 4.3), we compute the CCF
between the preemption rate series and the latency percentile series. The
lag `k*` that maximizes |CCF(k)| estimates the average delay between a
preemption event and its observable latency impact.

**Granger causality** could formally test whether KV cache usage "causes"
latency increases (versus the reverse or common cause). However, Granger
causality requires stationarity and many observations, making it impractical
for most benchmark runs. The simpler lagged cross-correlation approach in
Section 4.3 is more robust for our use case.

---

## Appendix A: Metric Reference

### A.1 Server-Side Metrics (Prometheus)

| Metric | Type | Description | Relevance |
|--------|------|-------------|-----------|
| `vllm:kv_cache_usage_perc` | Gauge | KV cache block utilization (0-100%) | Primary pressure indicator |
| `vllm:num_preemptions` | Counter | Cumulative preemption events | Latency spike trigger |
| `vllm:prefix_cache_hits` | Counter | Prefix blocks reused from cache | Cache efficiency numerator |
| `vllm:prefix_cache_queries` | Counter | Prefix blocks looked up | Cache efficiency denominator |
| `vllm:num_requests_running` | Gauge | Currently executing requests | Batch size / parallelism |
| `vllm:num_requests_waiting` | Gauge | Queued requests | Backpressure indicator |
| `vllm:request_queue_time_seconds` | Histogram | Server-side queue wait time | Queuing latency component |

### A.2 Client-Side Metrics (AIPerf)

| Metric | Type | Description | Relevance |
|--------|------|-------------|-----------|
| `tokens_in_flight` | Sweep | Instantaneous KV token load (client proxy) | Client-side KV approximation |
| `effective_concurrency` | Sweep | Instantaneous concurrent requests | Load level |
| `effective_throughput` | Sweep | Instantaneous output token rate | Performance indicator |
| `request_latency` | Record | End-to-end request time | Primary latency metric |
| `time_to_first_token` | Record | Time to first generated token | Prefill + network latency |
| `inter_token_latency` | Record (ragged) | Per-token generation latency | Decode performance |
| `input_sequence_length` | Record | Prompt token count | KV demand factor |
| `output_sequence_length` | Record | Generated token count | KV demand factor |

### A.3 GPU Telemetry

| Metric | Type | Description | Relevance |
|--------|------|-------------|-----------|
| `gpu_memory_used` | Gauge (GB) | GPU memory consumption | Includes KV cache + weights + activations |
| `mem_utilization` | Gauge (%) | GPU memory bandwidth utilization | Memory subsystem pressure |

---

## Appendix B: Derivations

### B.1 KV Cache Bytes per Token

For a transformer model with:
- `L` = number of layers
- `H_kv` = number of KV attention heads (may differ from Q heads in GQA/MQA)
- `d` = head dimension
- `b` = bytes per element (2 for float16, 1 for int8)

```
bytes_per_token = L × 2 × H_kv × d × b
                  ↑   ↑   ↑      ↑   ↑
                  |   |   |      |   └─ dtype size
                  |   |   |      └─── head dimension
                  |   |   └────────── KV head count
                  |   └────────────── K and V (two matrices)
                  └────────────────── per layer
```

For Grouped Query Attention (GQA) where `H_kv < H_q`:

```
GQA ratio = H_kv / H_q

Example: Llama 3 70B has H_q=64, H_kv=8 → GQA ratio = 1/8
KV cache is 8× smaller than it would be with full MHA
```

### B.2 Effective Utilization with Prefix Caching

Let:
- `N` = number of concurrent requests
- `P` = shared prefix length (tokens)
- `U_i` = unique suffix length for request i (tokens)
- `G_i` = generated tokens so far for request i
- `B` = block size (tokens per block)
- `T` = total blocks in KV cache pool

Without prefix caching:

```
blocks_used = Σ_i ceil((P + U_i + G_i) / B)
utilization = blocks_used / T
```

With prefix caching:

```
shared_blocks = ceil(P / B)
unique_blocks = Σ_i ceil((U_i + G_i) / B)
blocks_used = shared_blocks + unique_blocks
utilization = blocks_used / T

savings = 1 - (shared_blocks + unique_blocks) / Σ_i ceil((P + U_i + G_i) / B)
```

When `P >> U_i` (long shared prefix, short unique suffixes):

```
savings ≈ 1 - 1/N    (approaches 100% savings as N → ∞)
```

### B.3 Maximum Safe Concurrency

Given:
- `T` = total KV cache capacity (tokens)
- `ISL` = average input sequence length
- `OSL` = average output sequence length
- `α` = safety margin (typically 0.85 — leave 15% headroom to avoid preemption)

```
max_concurrency = floor(α × T / (ISL + OSL/2))
```

The `OSL/2` term assumes requests are on average halfway through generation. For
more precise estimation, use the expected value of the output token count for
an active request, which depends on the generation progress distribution:

```
E[generated_tokens | active] = OSL × E[progress | active]
```

For exponentially distributed request lifetimes (Poisson process), `E[progress
| active] = 0.5`. For deterministic OSL, it depends on the arrival pattern and
whether the system is in steady state.

### B.4 Preemption Probability Model

Under a simplified M/G/c/K queue model (finite capacity K = total blocks):

```
P(preemption) ≈ P(queue_full) × P(new_request_arrives | full)

P(queue_full) = rho^K / Σ_{k=0}^{K} rho^k    (Erlang B formula)
              ≈ 1 - e^{-(rho-1)×K}             for large K near rho=1
```

This gives the characteristic "cliff" behavior: P(preemption) is near zero
for `rho < 0.9` and rises steeply toward 1 as `rho → 1`. The cliff position
depends on K (cache capacity) — larger caches have a steeper cliff closer to
`rho = 1`.

### B.5 Lagged Cross-Correlation Significance

For two independent white-noise time series of length n, the cross-correlation
at any lag follows approximately:

```
CCF(k) ~ N(0, 1/n)    under H₀: no correlation
```

A CCF value is significant at the 95% level if:

```
|CCF(k)| > 1.96 / sqrt(n)
```

For n = 100 windowed observations: threshold = 0.196
For n = 30 windowed observations: threshold = 0.358

When the observed CCF exceeds these thresholds, we have statistical evidence
of correlation between the two time series at the given lag.

---

## Appendix C: Implementation Roadmap

### Phase 1: Core Correlation Engine

**Effort:** Medium

1. Implement `KVCacheCorrelationAnalyzer` as AnalyzerProtocol plugin
2. Windowed correlation between sweep metrics and server metrics
3. Preemption burst detection and basic severity classification
4. Console warnings for detected KV pressure
5. JSON export of correlation results

**Dependencies:** Existing MetricsAccumulator, ServerMetricsAccumulator, SweepCurves.

### Phase 2: Advanced Correlation

**Effort:** Medium-High

1. Lagged cross-correlation with optimal lag detection
2. KV regime change detection (PELT-like segmentation)
3. Prefix cache efficiency analysis
4. Client-only mode (TIF-based heuristics)

**Dependencies:** Phase 1.

### Phase 3: Predictive Model

**Effort:** High

1. KV demand model calibration (fit `max_kv_tokens` from observed data)
2. Maximum safe concurrency estimation
3. Per-request KV cost attribution
4. ISL x concurrency interaction surface visualization

**Dependencies:** Phase 2 + model configuration metadata.

### Phase 4: Real-Time Integration

**Effort:** High

1. Streaming correlation computation during benchmark run
2. Real-time KV pressure alerts via ZMQ messages
3. Textual UI widget showing live KV cache status
4. Adaptive load control: reduce concurrency when KV pressure detected

**Dependencies:** Phase 3 + significant architecture changes (streaming analyzers).

---

## References

1. Kwon, W. et al. (2023). "Efficient Memory Management for Large Language
   Model Serving with PagedAttention." *SOSP 2023*.

2. Yu, G. et al. (2022). "Orca: A Distributed Serving System for
   Transformer-Based Generative Models." *OSDI 2022*.

3. Agrawal, A. et al. (2024). "Taming Throughput-Latency Tradeoff in LLM
   Inference with Sarathi-Serve." *OSDI 2024*.

4. Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact
   Attention with IO-Awareness." *NeurIPS 2022*.

5. Sheng, Y. et al. (2023). "FlexGen: High-Throughput Generative Inference of
   Large Language Models with a Single GPU." *ICML 2023*.

6. Zheng, L. et al. (2023). "SGLang: Efficient Execution of Structured
   Language Model Programs." *arXiv:2312.07104*.

7. Zhong, Y. et al. (2024). "DistServe: Disaggregating Prefill and Decoding
   for Goodput-optimized Large Language Model Serving." *OSDI 2024*.

8. Dean, J. & Barroso, L.A. (2013). "The Tail at Scale." *Communications of
   the ACM*.

9. Tene, G. (2013). "How NOT to Measure Latency." *Strange Loop Conference*.

10. Box, G.E.P., Jenkins, G.M., & Reinsel, G.C. (2008). *Time Series
    Analysis: Forecasting and Control*. 4th ed. Wiley.

11. Killick, R. et al. (2012). "Optimal Detection of Changepoints with a
    Linear Computational Cost." *JASA*.

12. Law, A.M. & Kelton, W.D. (2000). *Simulation Modeling and Analysis*.
    McGraw-Hill.
