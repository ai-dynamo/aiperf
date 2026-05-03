<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Queue Depth, Scheduling Dynamics & Client Latency Correlation

**Status**: Research

**Authors**: Anthony Casagrande

**Category**: Research — Correlation Analysis

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Metric Inventory and Data Sources](#2-metric-inventory-and-data-sources)
3. [Little's Law Validation](#3-littles-law-validation)
4. [Client vs Server Concurrency Gap Analysis](#4-client-vs-server-concurrency-gap-analysis)
5. [Queue Buildup Dynamics](#5-queue-buildup-dynamics)
6. [Scheduling Fairness Analysis](#6-scheduling-fairness-analysis)
7. [Phase Interference: Prefill vs Decode Competition](#7-phase-interference-prefill-vs-decode-competition)
8. [Admission Control Detection](#8-admission-control-detection)
9. [Arrival Pattern Impact on Queue Dynamics](#9-arrival-pattern-impact-on-queue-dynamics)
10. [Implementation Architecture](#10-implementation-architecture)
11. [Sampling, Interpolation & Alignment](#11-sampling-interpolation--alignment)
12. [Real-Time Correlation Dashboard](#12-real-time-correlation-dashboard)
13. [Validation Strategy](#13-validation-strategy)
14. [References](#14-references)

---

## 1. Introduction

### 1.1 Motivation

LLM inference benchmarking produces two classes of metrics that are rarely
analyzed together: **client-side** measurements (latency, throughput,
concurrency as observed by the load generator) and **server-side** measurements
(queue depth, scheduling delays, batch sizes as reported by Prometheus
endpoints). Each class tells an incomplete story. Client metrics capture the
end-user experience but cannot explain *why* latency spiked. Server metrics
reveal internal scheduling decisions but miss network-in-flight time and
client-side queuing.

Correlating these two views unlocks diagnostic power that neither provides
alone:

- **Causal attribution**: When TTFT spikes, is the cause server-side queuing
  (num_requests_waiting grew), batch scheduling (prefill contention), or
  network congestion (gap between client and server views)?
- **Benchmark validity**: Little's Law provides a closed-form relationship
  between concurrency, throughput, and latency. Violations indicate
  measurement error, non-stationarity, or system instability.
- **Capacity planning**: Queue buildup rate predicts saturation before it
  manifests in tail latency. Admission control detection reveals hard
  server limits invisible to the client.

### 1.2 Scope

This document researches the mathematical foundations, detection algorithms,
and implementation architecture for cross-correlating queue depth and
scheduling dynamics with client-observed latency in AIPerf. It covers:

1. Queue-theoretic models adapted to LLM inference (continuous batching,
   chunked prefill, decode iteration scheduling)
2. Concrete algorithms for correlation, anomaly detection, and causal
   analysis
3. AIPerf-specific integration points: sweep-line curves, ColumnStore,
   ServerMetricsAccumulator, steady-state detection
4. Sampling requirements and timestamp alignment strategies

### 1.3 LLM Inference Architecture Context

Understanding the correlation between queue metrics and latency requires
understanding the scheduling pipeline inside a typical LLM inference server
(e.g., vLLM):

```
                   AIPerf Client
                   ┌─────────────────────────────────────────────┐
                   │  TimingManager → Credit → Worker → HTTP     │
                   │       │                      │               │
                   │  credit_issued_ns    request_start_ns        │
                   │                      request_ack_ns          │
                   │                      request_end_ns          │
                   └──────────────────────┬──────────────────────┘
                                          │ HTTP/gRPC
                                          ▼
                   ┌─────────────────────────────────────────────┐
                   │  LLM Inference Server (vLLM)                │
                   │                                             │
                   │  ┌──────────┐     ┌──────────────────┐      │
                   │  │ Waiting  │────▶│ Running           │      │
                   │  │ Queue    │     │ (GPU Execution)   │      │
                   │  │          │     │                   │      │
                   │  │ N_wait   │     │ N_run             │      │
                   │  └──────────┘     │ ┌───────────────┐ │      │
                   │                   │ │ Prefill batch  │ │      │
                   │                   │ │ Decode batch   │ │      │
                   │                   │ └───────────────┘ │      │
                   │                   └──────────────────┘      │
                   │                                             │
                   │  Prometheus: num_requests_waiting            │
                   │              num_requests_running            │
                   │              request_queue_time_seconds      │
                   │              e2e_request_latency_seconds     │
                   └─────────────────────────────────────────────┘
```

The critical insight: **there are three queue segments between a credit being
issued and a response arriving**, each contributing to end-to-end latency:

```
Timeline for a single request:

credit_issued_ns ──┬── Client queue wait ──┬── request_start_ns
                   │  (credit_drop_latency │
                   │   + dataset fetch)    │
                   │                       │
                   ├── Network transit ────┤── Server receives request
                   │                       │
                   ├── Server queue wait ──┤── Server schedules prefill
                   │  (request_queue_time) │   (num_requests_waiting → running)
                   │                       │
                   ├── Prefill execution ──┤── First token generated
                   │                       │   (TTFT includes queue + prefill)
                   │                       │
                   ├── Decode iterations ──┤── Last token generated
                   │  (competes with new   │
                   │   prefill requests)   │
                   │                       │
                   └── Network transit ────┘── request_end_ns
```

---

## 2. Metric Inventory and Data Sources

### 2.1 Server-Side Metrics (via Prometheus / ServerMetricsAccumulator)

AIPerf collects server metrics through `ServerMetricsDataCollector`, which
scrapes Prometheus endpoints at a configurable interval. Metrics are stored
in `ServerMetricsHierarchy` with per-endpoint, per-metric time series.

| Metric | Type | Semantics |
|--------|------|-----------|
| `vllm:num_requests_running` | Gauge | Currently executing requests (prefill + decode). This is the **server-side concurrency**: requests that have been scheduled and are consuming GPU resources. |
| `vllm:num_requests_waiting` | Gauge | Requests received by the server but not yet scheduled. The **server-side queue depth**. Grows when arrival rate exceeds scheduling capacity. |
| `vllm:request_queue_time_seconds` | Histogram | Distribution of time spent in the waiting queue before scheduling. Bucketed — requires polynomial interpolation for accurate percentile estimation. |
| `vllm:request_success` | Counter | Cumulative count of completed requests. Delta rate = server-observed throughput. |
| `vllm:e2e_request_latency_seconds` | Histogram | Server-measured end-to-end latency. Includes queue time + prefill + decode but excludes network transit. |

**Collection characteristics:**
- Scraped at `--server-metrics-interval` (default: 1s in vLLM)
- Gauge values are instantaneous snapshots (not time-averaged)
- Histograms accumulate since server start (requires delta computation)
- Counter monotonicity can break on server restart (handled by delta logic)

### 2.2 Client-Side Metrics (via ColumnStore / Sweep-Line Algorithms)

| Metric | Source | Semantics |
|--------|--------|-----------|
| `effective_concurrency` | `concurrency_sweep()` | Time-weighted concurrent requests from the client's perspective. Counts requests between `request_start_ns` and `request_end_ns`. |
| `effective_prefill_concurrency` | Sweep-line | Requests between `request_start_ns` and `request_ack_ns` (first token). Approximates the server's prefill load. |
| `effective_generation_concurrency` | Sweep-line | Requests between `request_ack_ns` and `request_end_ns`. Approximates the server's decode load. |
| `request_throughput` | Derived | Requests per second (count / duration). |
| `TTFT` | Record | Time to first token. Includes network + server queue + prefill. |
| `request_latency` | Record | End-to-end latency as observed by the client. |
| `credit_drop_latency` | Record | AIPerf internal: `request_start_ns - credit_drop_received_ns`. Measures overhead from credit receipt through dataset fetch to HTTP send. |

**Collection characteristics:**
- Per-request granularity (nanosecond timestamps)
- Sweep-line curves are step functions: exact at event boundaries
- No sampling artifacts — every request contributes a start/end event

### 2.3 Derived Cross-Domain Metrics

These do not exist yet. This research proposes computing them from the
intersection of client and server data:

| Proposed Metric | Formula | Purpose |
|----------------|---------|---------|
| `littles_law_residual` | `L - lambda * W` | Benchmark validity check |
| `client_server_concurrency_gap` | `eff_concurrency - (N_run + N_wait)` | Network-in-flight estimation |
| `queue_growth_rate` | `d(N_wait)/dt` | Overload leading indicator |
| `scheduling_fairness_cv` | `std(queue_time) / mean(queue_time)` | FIFO violation detection |
| `prefill_decode_interference` | `corr(prefill_conc, ITL)` | Phase contention quantification |
| `admission_control_threshold` | Detected plateau in `N_run` | Server limit identification |

---

## 3. Little's Law Validation

### 3.1 Mathematical Foundation

Little's Law (Little, 1961) is one of the most general results in queueing
theory. In its simplest form:

```
L = lambda * W
```

Where:
- **L** = average number of items in the system (queue + service)
- **lambda** = average arrival rate (items per unit time)
- **W** = average time an item spends in the system (wait + service)

The law holds for **any** queueing discipline (FIFO, LIFO, priority,
random), **any** arrival process (Poisson, deterministic, general), and
**any** service time distribution, provided the system is in **steady state**
(stationary, ergodic, and stable: lambda < mu where mu is service rate).

This generality makes it an ideal health check: if the measured quantities
do not satisfy L = lambda * W, at least one of the following is true:

1. The system was not in steady state during measurement
2. The measurements are incorrect or misaligned
3. There is an unaccounted-for queue segment (e.g., network-in-flight)

### 3.2 Application to LLM Inference

For a complete LLM inference system, we can apply Little's Law at multiple
levels:

**Level 1: Entire system (client perspective)**

```
L_client = lambda_client * W_client

Where:
  L_client   = effective_concurrency (time-weighted avg from sweep-line)
  lambda_client = request_throughput (completed requests / duration)
  W_client   = avg(request_latency)
```

**Level 2: Server queue only**

```
L_queue = lambda_server * W_queue

Where:
  L_queue    = avg(num_requests_waiting)  (time-weighted gauge average)
  lambda_server = delta(request_success) / duration
  W_queue    = avg(request_queue_time_seconds)
```

**Level 3: Server execution only**

```
L_exec = lambda_server * W_exec

Where:
  L_exec     = avg(num_requests_running)  (time-weighted gauge average)
  lambda_server = same as above
  W_exec     = avg(e2e_request_latency) - avg(request_queue_time)
```

**Level 4: Phase-specific (prefill)**

```
L_prefill = lambda_prefill * W_prefill

Where:
  L_prefill  = effective_prefill_concurrency (sweep-line)
  lambda_prefill = request_throughput (same arrival, different service)
  W_prefill  = avg(TTFT)  (approximate: TTFT = network + queue + prefill)
```

### 3.3 Residual Analysis

Define the **Little's Law residual** at each level:

```
R = L - lambda * W
```

And the **normalized residual** (percentage deviation):

```
R_norm = (L - lambda * W) / L * 100%
```

Interpretation of R_norm:

| R_norm | Interpretation |
|--------|---------------|
| |R_norm| < 5% | Excellent agreement. System in steady state, measurements consistent. |
| 5% < |R_norm| < 15% | Acceptable. Minor transients or measurement granularity effects. |
| |R_norm| > 15% | Significant discrepancy. Investigate non-stationarity, measurement error, or missing queue segment. |
| R_norm > 0 (L too high) | More items in system than lambda * W predicts. Possible: requests counted as in-flight but not yet contributing to W (measurement timing), or queue segment not captured in W. |
| R_norm < 0 (L too low) | Fewer items than predicted. Possible: W inflated by outliers (heavy tail), or L is under-sampled (gauge snapshots miss peaks). |

### 3.4 Time-Varying Little's Law

The classic formulation assumes time-averaged quantities over a steady-state
interval. For richer analysis, we can compute windowed residuals:

```
For each time window [t, t + delta]:
    L(t)      = time-weighted concurrency in [t, t + delta]
    lambda(t) = requests completed in [t, t + delta] / delta
    W(t)      = mean latency of requests completed in [t, t + delta]
    R(t)      = L(t) - lambda(t) * W(t)
```

This produces a **residual time series** that reveals when the system
departs from equilibrium:

```
                         Steady State
                    ┌────────────────────┐
                    │                    │
R(t)   +20%  ─ ─ ─┬─ ─ ─ ─ ─ ─ ─ ─ ─ ─┬─ ─ ─ ─ ─ ─ ─ ─
               ╱   │                    │  ╱╲
              ╱    │    ≈ 0%            │ ╱  ╲
         0%  ──────┼────────────────────┼─────────────
              ╲    │                    │       ╲
               ╲   │                    │        ╲
       -20%  ─ ─ ─┴─ ─ ─ ─ ─ ─ ─ ─ ─ ─┴─ ─ ─ ─ ─╲─ ─ ─
              │                                      │
          Ramp-up                               Ramp-down
```

**Window size selection**: The window must be large enough to contain several
request completions. A minimum of `10 / lambda` seconds (10 completions per
window) ensures the per-window throughput estimate is not dominated by
quantization noise.

### 3.5 Pseudocode

```python
def littles_law_residual(
    concurrency_ts: NDArray[np.float64],      # from concurrency_sweep
    concurrency: NDArray[np.float64],
    request_end_ns: NDArray[np.float64],       # from ColumnStore
    request_latency_ns: NDArray[np.float64],   # from ColumnStore
    window_ns: float,                          # analysis window size
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute time-varying Little's Law residual.

    Returns:
        (window_centers, residuals) — residual time series.
    """
    t_min = float(concurrency_ts[0])
    t_max = float(concurrency_ts[-1])
    n_windows = int((t_max - t_min) / window_ns)

    centers = np.empty(n_windows, dtype=np.float64)
    residuals = np.empty(n_windows, dtype=np.float64)

    for i in range(n_windows):
        w_start = t_min + i * window_ns
        w_end = w_start + window_ns
        centers[i] = w_start + window_ns / 2

        # L: time-weighted concurrency in window
        L = compute_time_weighted_stats(
            concurrency_ts, concurrency, w_start, w_end
        ).avg

        # lambda: request completions in window / window duration
        mask = (request_end_ns >= w_start) & (request_end_ns < w_end)
        n_completed = np.sum(mask)
        lam = n_completed / (window_ns / NANOS_PER_SECOND)

        # W: mean latency of completed requests (in seconds)
        if n_completed > 0:
            W = np.mean(request_latency_ns[mask]) / NANOS_PER_SECOND
        else:
            W = 0.0

        residuals[i] = L - lam * W

    return centers, residuals
```

### 3.6 Cross-Level Consistency Check

When both client and server Little's Law residuals are available, the
**difference in residuals** isolates the network-in-flight component:

```
R_client - R_server ≈ lambda * T_network_round_trip
```

Where `T_network_round_trip` is the average network transit time (both
directions combined). This provides a measurement of effective network
overhead without requiring network-level instrumentation.

---

## 4. Client vs Server Concurrency Gap Analysis

### 4.1 The Concurrency Gap

At any instant, the client and server have different views of how many
requests are "in flight":

```
Client view (effective_concurrency):
  Counts request as in-flight from request_start_ns to request_end_ns

Server view (num_requests_running + num_requests_waiting):
  Counts request as in-flight from server-receive to server-complete
```

The **gap** between these two views represents requests that are:

1. **In network transit (client → server)**: Already sent by the client
   (`request_start_ns` has passed) but not yet received by the server.
   This contributes to client concurrency but not server concurrency.

2. **In the load balancer queue**: If a reverse proxy (nginx, envoy) sits
   between AIPerf and the inference server, requests may queue there.
   The server's Prometheus metrics do not see these requests.

3. **In network transit (server → client)**: The server has completed the
   response, but the client has not yet received the final bytes. The
   server has decremented its counters, but the client still counts the
   request as in-flight.

4. **Timing misalignment**: Prometheus gauge snapshots are taken at scrape
   time, not at the exact moment of client events. This creates artificial
   gaps even with no actual network delay.

```
                    Client View          Network          Server View
                  ┌─────────────┐    ┌───────────┐    ┌──────────────┐
                  │             │    │           │    │              │
 Request A:       │ ████████████│████│███████████│████│██████████████│
                  │ ^start      │    │ ^arrive   │    │    ^complete │
                  │             │    │           │    │              │
 Client counts:   │      1      │  1 │     0     │  0 │      0       │
 Server counts:   │      0      │  0 │     1     │  1 │      0       │
                  │             │    │           │    │              │
                  └─────────────┘    └───────────┘    └──────────────┘

 Concurrency gap =  client - server  =  network-in-flight requests
```

### 4.2 Gap Estimation

To estimate the gap, we must align the two time series. The client sweep-line
gives us an exact step function. The server gauge gives us periodic samples.

**Step 1: Interpolate server concurrency to client event times.**

The server gauge is a piecewise-constant function sampled at scrape times
`t_s[0], t_s[1], ..., t_s[m]`. Between scrapes, we use the last-known value
(zero-order hold):

```python
def interpolate_gauge(
    scrape_ts: NDArray[np.float64],
    gauge_values: NDArray[np.float64],
    query_ts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate gauge to query timestamps via zero-order hold."""
    idx = np.searchsorted(scrape_ts, query_ts, side="right") - 1
    idx = np.clip(idx, 0, len(gauge_values) - 1)
    return gauge_values[idx]
```

**Step 2: Compute the gap at each client event.**

```python
# Server total concurrency = running + waiting
server_total = interpolate_gauge(
    scrape_ts, running_values + waiting_values, client_event_ts
)

# Client concurrency at event times (already exact)
client_conc = concurrency_at_events  # from sweep-line

gap = client_conc - server_total
```

**Step 3: Compute time-weighted gap statistics.**

The gap is itself a step function. Apply `compute_time_weighted_stats()`
to get avg, p50, p99, etc.

### 4.3 Decomposing the Gap

The total gap can be decomposed if additional timing data is available:

```
gap_total = gap_outbound + gap_lb + gap_inbound + gap_timing

Where:
  gap_outbound  = requests sent but not server-received
                  (not directly measurable without server-side timestamps)
  gap_lb        = requests in load balancer queue
                  (measurable if LB exposes Prometheus metrics)
  gap_inbound   = responses server-completed but client-not-received
                  (approximated by tail of response streaming time)
  gap_timing    = artifact of gauge sampling vs sweep-line resolution
                  (reducible by increasing scrape frequency)
```

For vLLM deployments without a load balancer, the gap simplifies to:

```
gap ≈ lambda * RTT
```

Where RTT is the average network round-trip time. This provides an indirect
measurement of network overhead.

### 4.4 Gap as a Health Indicator

| Observed Gap Pattern | Diagnosis |
|---------------------|-----------|
| gap ≈ constant, small (< 2) | Healthy. Network adds minimal overhead. |
| gap ≈ constant, large (> 5) | High network latency or load balancer queuing. |
| gap grows over time | Network congestion or LB saturation. Client is flooding faster than network can deliver. |
| gap oscillates | Batch scheduling effects. Server processes bursts, creating periodic dips in server concurrency. |
| gap is negative | Server sees more requests than client thinks are in-flight. Possible: duplicate delivery, or stale client state after cancellation. |

---

## 5. Queue Buildup Dynamics

### 5.1 Queue Depth as a Leading Indicator

Tail latency is a **lagging** indicator: by the time p99 spikes, the queue
has already been building for some time. The rate of change of queue depth
is a **leading** indicator that predicts latency degradation before it
manifests in completed-request metrics.

The fundamental relationship between arrival rate, service rate, and queue
dynamics:

```
d(N_wait)/dt = lambda_arrival - mu_scheduling

Where:
  lambda_arrival  = rate at which new requests arrive at the server
  mu_scheduling   = rate at which requests move from waiting to running
```

When `lambda_arrival > mu_scheduling`, the queue grows. When this persists,
the system is **overloaded** and latency grows without bound.

### 5.2 Derivative Estimation from Gauge Samples

Server gauge values are sampled at discrete times. We estimate the derivative
using finite differences:

```python
def queue_growth_rate(
    scrape_ts: NDArray[np.float64],
    n_waiting: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate d(N_wait)/dt from gauge snapshots.

    Uses central differences (forward/backward at boundaries).

    Returns:
        (midpoint_timestamps, growth_rates) in requests/second.
    """
    dt = np.diff(scrape_ts) / NANOS_PER_SECOND  # seconds
    dq = np.diff(n_waiting)

    # Avoid division by zero for duplicate timestamps
    valid = dt > 0
    rates = np.zeros_like(dq)
    rates[valid] = dq[valid] / dt[valid]

    midpoints = (scrape_ts[:-1] + scrape_ts[1:]) / 2
    return midpoints, rates
```

### 5.3 Queue Stability Classification

Using the growth rate time series, classify the system's queue regime at
each point in time:

```
Queue Regime Classification:

  d(N_wait)/dt ≫ 0                    → OVERLOADING
    Queue growing fast. Arrival rate significantly exceeds scheduling rate.
    Latency will increase rapidly. This is an unstable regime.

  d(N_wait)/dt ≈ 0, N_wait > 0       → SATURATED_STABLE
    Queue depth is constant but nonzero. System at capacity.
    Arrival rate ≈ scheduling rate. Latency is elevated but stable.

  d(N_wait)/dt ≈ 0, N_wait ≈ 0       → HEALTHY
    No queuing delay. Requests are scheduled immediately.
    Arrival rate < scheduling rate.

  d(N_wait)/dt ≪ 0                    → DRAINING
    Queue shrinking. Load has decreased or scheduling capacity increased.
    Latency will decrease. Typical during ramp-down.
```

The classification thresholds should be derived from the mean arrival rate:

```
overload_threshold   = 0.1 * lambda   (queue growing at 10% of arrival rate)
drain_threshold      = -0.1 * lambda  (queue shrinking at 10% of arrival rate)
zero_threshold       = 0.5            (less than 0.5 requests/sec change)
```

### 5.4 Queue Buildup → Latency Lag Correlation

There is a predictable lag between queue buildup onset and latency increase.
Requests that arrive during buildup experience the growing queue. Their
latency is observed when they *complete*, which is approximately
`W_queue + W_service` seconds later.

The **cross-correlation** between queue growth rate and latency reveals
this lag:

```
                   Queue growth rate        Request latency
                        │                        │
              ┌─────────▼─────────┐    ┌─────────▼─────────┐
 t=0          │    ▲              │    │                    │
              │   ╱ ╲             │    │                    │
 t=T_lag      │  ╱   ╲           │    │    ▲               │
              │ ╱     ╲          │    │   ╱ ╲              │
              │╱       ╲         │    │  ╱   ╲             │
              │         ╲        │    │ ╱     ╲            │
              │          ╲       │    │╱       ╲           │
              └───────────────────┘    └────────────────────┘
                                   T_lag ≈ avg(W_queue) + avg(W_service)
```

**Estimating the lag:**

```python
def estimate_queue_latency_lag(
    growth_rate_ts: NDArray[np.float64],
    growth_rates: NDArray[np.float64],
    latency_ts: NDArray[np.float64],     # request_end_ns for completed reqs
    latencies: NDArray[np.float64],      # request_latency_ns
    max_lag_ns: float,
) -> tuple[float, float]:
    """Estimate the lag between queue buildup and latency increase.

    Uses cross-correlation on uniformly resampled signals.

    Returns:
        (lag_ns, correlation_at_lag)
    """
    # Resample both to uniform grid
    dt = min(np.median(np.diff(growth_rate_ts)),
             np.median(np.diff(latency_ts)))
    t_start = max(growth_rate_ts[0], latency_ts[0])
    t_end = min(growth_rate_ts[-1], latency_ts[-1])

    grid = np.arange(t_start, t_end, dt)
    g_resampled = np.interp(grid, growth_rate_ts, growth_rates)
    l_resampled = np.interp(grid, latency_ts, latencies)

    # Normalize
    g_resampled -= np.mean(g_resampled)
    l_resampled -= np.mean(l_resampled)

    # Cross-correlation via FFT (O(n log n))
    n = len(grid)
    G = np.fft.rfft(g_resampled, n=2*n)
    L = np.fft.rfft(l_resampled, n=2*n)
    xcorr = np.fft.irfft(G.conj() * L)[:n]

    # Normalize by product of standard deviations
    norm = np.sqrt(np.sum(g_resampled**2) * np.sum(l_resampled**2))
    if norm > 0:
        xcorr /= norm

    # Find peak in valid lag range
    max_lag_samples = int(max_lag_ns / dt)
    valid = xcorr[:max_lag_samples]
    peak_idx = np.argmax(valid)

    return peak_idx * dt, float(valid[peak_idx])
```

### 5.5 Overload Prediction

Using the queue growth rate and current queue depth, we can predict time
to critical latency:

```
If d(N_wait)/dt = r (constant growth rate):

  Time until queue depth reaches N_critical:
    T_critical = (N_critical - N_current) / r

  Expected latency at time T:
    W(T) ≈ N_wait(T) / mu_scheduling
         = (N_current + r * T) / mu_scheduling
```

This provides an **early warning**: "At current queue growth rate, p99
latency will exceed the SLA threshold in approximately T seconds."

---

## 6. Scheduling Fairness Analysis

### 6.1 FIFO Expectation vs Reality

In a pure FIFO queue, the waiting time for request `i` depends only on the
number of requests ahead of it and the service rate. The queue time
distribution would have moderate variance: all requests experience similar
delays that depend primarily on queue depth at arrival time.

LLM inference servers deviate from FIFO in several ways:

1. **Continuous batching**: The scheduler runs in iteration cycles. A request
   arriving just after a scheduling decision waits until the next iteration,
   while one arriving just before is scheduled immediately.

2. **Prefill prioritization**: Some schedulers prioritize shorter prefill
   requests to maximize batch utilization (more requests fit in GPU memory).

3. **Chunked prefill**: Long prompts may be split across multiple iterations.
   A request with a 32K token prompt may be partially prefilled, then
   paused while shorter requests complete prefill in a single iteration.

4. **Memory-based scheduling**: When GPU KV cache is nearly full, the
   scheduler may delay new requests (even if the batch has capacity) to
   avoid OOM. This creates priority inversion: older requests with large
   KV allocations effectively block newer, shorter requests.

### 6.2 Fairness Metrics from Queue Time Distribution

The `vllm:request_queue_time_seconds` histogram provides the distribution
of server-side waiting times. From this distribution, we can compute
fairness metrics.

**Coefficient of Variation (CV):**

```
CV = std(queue_time) / mean(queue_time)
```

| CV | Interpretation |
|----|---------------|
| CV < 1 | Sub-exponential variance. Consistent scheduling. Typical for well-behaved FIFO at moderate load. |
| CV ≈ 1 | Exponential variance. Consistent with M/M/1 (Poisson arrivals, exponential service). |
| CV > 1 | Super-exponential variance. Indicates priority scheduling, head-of-line blocking, or bimodal behavior. |
| CV > 3 | Extreme variance. Strong evidence of scheduling unfairness or multi-modal behavior. |

**Tail ratio:**

```
tail_ratio = p99(queue_time) / p50(queue_time)
```

A pure FIFO M/M/1 queue has a theoretical tail ratio of approximately:

```
For M/M/1:  p99/p50 ≈ -ln(0.01) / -ln(0.50) ≈ 4.6 / 0.69 ≈ 6.6
```

Ratios significantly above this suggest scheduling unfairness.

### 6.3 Correlation with Input Sequence Length (ISL)

To test whether longer prompts experience disproportionate scheduling delay,
correlate queue time with ISL:

```python
def scheduling_fairness_by_isl(
    queue_times: NDArray[np.float64],     # per-request queue times
    input_lengths: NDArray[np.float64],   # per-request ISL
    n_bins: int = 10,
) -> dict[str, Any]:
    """Analyze scheduling fairness by input sequence length.

    Returns:
        Dictionary with per-bin queue time statistics and
        Spearman rank correlation between ISL and queue time.
    """
    # Bin requests by ISL
    bin_edges = np.percentile(input_lengths, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(input_lengths, bin_edges[1:-1])

    bin_stats = []
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            qt = queue_times[mask]
            bin_stats.append({
                "isl_range": (float(bin_edges[b]), float(bin_edges[b + 1])),
                "count": int(np.sum(mask)),
                "mean_queue_time": float(np.mean(qt)),
                "p50_queue_time": float(np.median(qt)),
                "p99_queue_time": float(np.percentile(qt, 99)),
            })

    # Overall correlation
    # Spearman rank correlation (non-parametric, robust to outliers)
    n = len(queue_times)
    ranks_qt = np.argsort(np.argsort(queue_times)).astype(np.float64)
    ranks_isl = np.argsort(np.argsort(input_lengths)).astype(np.float64)
    d = ranks_qt - ranks_isl
    rho = 1.0 - 6.0 * np.sum(d**2) / (n * (n**2 - 1))

    return {
        "bin_stats": bin_stats,
        "spearman_rho": float(rho),
        "interpretation": (
            "positive_bias" if rho > 0.1
            else "negative_bias" if rho < -0.1
            else "no_significant_bias"
        ),
    }
```

**Interpretation:**

- `rho > 0.1` (positive correlation): Longer prompts wait longer. Consistent
  with schedulers that prioritize shorter requests for batch efficiency.
- `rho < -0.1` (negative correlation): Shorter prompts wait longer. Unusual
  but could indicate LIFO-like behavior or memory reservation for large
  requests.
- `|rho| < 0.1`: No significant correlation. Scheduling is approximately
  fair with respect to prompt length.

### 6.4 Head-of-Line Blocking Detection

Head-of-line blocking occurs when a single long-running request at the front
of the queue delays all subsequent requests. In LLM inference, this manifests
when:

- A very long prompt monopolizes the prefill stage
- A request with a large KV cache allocation blocks memory for new requests

Detection heuristic:

```
For each time window:
  1. Identify the request with max(queue_time) — the "head"
  2. Count requests that arrived after the head but were scheduled before it
  3. If count > 0, head-of-line blocking occurred

HOL blocking rate = windows_with_blocking / total_windows
```

This requires per-request server-side arrival and scheduling timestamps,
which are not directly available from Prometheus metrics. However, we can
approximate using the queue time distribution: a bimodal distribution
(cluster of short waits + cluster of long waits) strongly suggests
head-of-line blocking.

---

## 7. Phase Interference: Prefill vs Decode Competition

### 7.1 The Fundamental Contention

In transformer-based LLM inference, prefill and decode operations compete
for the same GPU resources:

- **Prefill** (prompt processing): Compute-bound, processes all input tokens
  in parallel. High arithmetic intensity. One large matrix multiplication
  per layer.
- **Decode** (token generation): Memory-bandwidth-bound, generates one token
  at a time per request. Low arithmetic intensity but high KV cache access.

In continuous batching, the scheduler interleaves prefill and decode work
within the same iteration:

```
Iteration Timeline (simplified):

  ┌─────────────────────────────────────────────────┐
  │ Iteration k                                     │
  │                                                 │
  │  ┌──────────────┐  ┌─────────────────────────┐  │
  │  │ Prefill      │  │ Decode                  │  │
  │  │ Req A (new)  │  │ Req B, C, D (ongoing)   │  │
  │  │ 512 tokens   │  │ 1 token each            │  │
  │  │ ~4ms         │  │ ~2ms (batched)           │  │
  │  └──────────────┘  └─────────────────────────┘  │
  │                                                 │
  │  Total iteration time: 6ms                      │
  │  Without prefill:      2ms                      │
  │  Prefill overhead:     4ms added to B,C,D ITL   │
  └─────────────────────────────────────────────────┘
```

Every prefill operation in a mixed iteration **adds latency to all in-flight
decode requests**. This is the fundamental source of phase interference.

### 7.2 Measuring Phase Interference

**Signal 1: Prefill concurrency vs ITL correlation**

```
Cross-correlate:
  X(t) = effective_prefill_concurrency at time t
  Y(t) = inter_token_latency for tokens generated at time t

Expected: positive correlation with near-zero lag.
When prefill concurrency increases, ITL for in-flight decode requests
increases simultaneously (same iteration).
```

**Signal 2: Prefill concurrency vs decode throughput anti-correlation**

```
Cross-correlate:
  X(t) = effective_prefill_concurrency
  Y(t) = effective throughput of decode tokens (output tokens per second)

Expected: negative correlation.
More prefill work → less GPU time for decode → lower decode throughput.
```

### 7.3 Quantifying the Interference Factor

Define the **prefill interference factor** (PIF):

```
PIF = ITL_high_prefill / ITL_low_prefill

Where:
  ITL_high_prefill = avg ITL when prefill_concurrency > p75(prefill_concurrency)
  ITL_low_prefill  = avg ITL when prefill_concurrency < p25(prefill_concurrency)
```

| PIF | Interpretation |
|-----|---------------|
| PIF ≈ 1.0 | Minimal interference. Server effectively isolates prefill from decode (e.g., chunked prefill with small chunks). |
| 1.0 < PIF < 1.5 | Moderate interference. Typical for continuous batching with reasonable prefill sizes. |
| PIF > 1.5 | Significant interference. Consider chunked prefill, prefill/decode separation, or reducing max batch prefill tokens. |
| PIF > 2.0 | Severe interference. Prefill operations are dominating iteration time. |

### 7.4 Implementation Sketch

```python
def prefill_interference_analysis(
    prefill_conc_ts: NDArray[np.float64],
    prefill_conc: NDArray[np.float64],
    token_timestamps_ns: NDArray[np.float64],   # per-token generation times
    itl_values_ns: NDArray[np.float64],          # per-token ITL
) -> dict[str, float]:
    """Quantify prefill-decode interference.

    Requires per-token ITL data (available from streaming responses).
    """
    # Look up prefill concurrency at each token generation time
    pc_at_token = np.interp(token_timestamps_ns, prefill_conc_ts, prefill_conc)

    # Quartile split
    p25 = np.percentile(pc_at_token, 25)
    p75 = np.percentile(pc_at_token, 75)

    low_mask = pc_at_token <= p25
    high_mask = pc_at_token >= p75

    itl_low = np.mean(itl_values_ns[low_mask]) if np.any(low_mask) else np.nan
    itl_high = np.mean(itl_values_ns[high_mask]) if np.any(high_mask) else np.nan

    pif = itl_high / itl_low if itl_low > 0 else np.nan

    # Cross-correlation (simplified: Pearson on concurrent values)
    if len(pc_at_token) > 10:
        corr = np.corrcoef(pc_at_token, itl_values_ns.astype(np.float64))[0, 1]
    else:
        corr = np.nan

    return {
        "prefill_interference_factor": float(pif),
        "prefill_itl_correlation": float(corr),
        "itl_at_low_prefill_ns": float(itl_low),
        "itl_at_high_prefill_ns": float(itl_high),
    }
```

### 7.5 Chunked Prefill Implications

Servers using chunked prefill (splitting large prompts across iterations)
create a different interference pattern:

```
Without chunked prefill:
  Prefill concurrency:  ╭───╮       ╭───╮
                        │   │       │   │
                    ────╯   ╰───────╯   ╰────
  ITL impact:       ─────╱╲───────────╱╲────── (sharp spikes)

With chunked prefill:
  Prefill concurrency:  ╭─╮╭─╮╭─╮ ╭─╮╭─╮╭─╮
                        │ ││ ││ │ │ ││ ││ │
                    ────╯ ╰╯ ╰╯ ╰─╯ ╰╯ ╰╯ ╰──
  ITL impact:       ─────/‾\/‾\/‾\──/‾\/‾\/‾\── (smaller, more frequent)
```

The correlation structure changes: without chunked prefill, we see
high-amplitude spikes with low frequency. With chunked prefill, we see
low-amplitude oscillations with high frequency. The PIF decreases, but the
autocorrelation in ITL increases (more regular oscillation pattern).

---

## 8. Admission Control Detection

### 8.1 Server-Side Admission Control

Many LLM inference servers enforce limits on concurrent execution:

- **vLLM**: `--max-num-seqs` (max concurrent sequences, default 256)
- **TensorRT-LLM**: `max_num_sequences` in model config
- **Triton**: `max_batch_size` and `instance_count` limit concurrency

When the admission control limit is reached:
- `num_requests_running` plateaus at the limit
- `num_requests_waiting` grows linearly with excess arrival rate
- Client observes sudden TTFT increase (dominated by queue time)

### 8.2 Detection Algorithm

```
Admission Control Detection:

  Step 1: Identify running-count plateau
    - Compute rolling max of num_requests_running over trailing window
    - If rolling_max is constant for > T_min seconds, candidate plateau

  Step 2: Confirm with waiting-count growth
    - During the plateau period, check if num_requests_waiting is increasing
    - Growth rate > 0 confirms requests are being queued

  Step 3: Estimate the limit
    - The admission control limit ≈ mode(num_requests_running) during plateau

  Step 4: Correlate with client TTFT
    - Partition requests into "before plateau" and "during plateau"
    - Compare TTFT distributions
```

### 8.3 Pseudocode

```python
@dataclass(frozen=True)
class AdmissionControlResult:
    """Result of admission control detection."""
    detected: bool
    estimated_limit: int
    plateau_start_ns: float
    plateau_end_ns: float
    ttft_before_plateau_ms: float   # median TTFT before admission limit hit
    ttft_during_plateau_ms: float   # median TTFT while admission-limited
    queue_growth_rate: float        # requests/sec growth in waiting queue


def detect_admission_control(
    scrape_ts: NDArray[np.float64],
    n_running: NDArray[np.float64],
    n_waiting: NDArray[np.float64],
    min_plateau_seconds: float = 10.0,
    tolerance: float = 1.0,
) -> AdmissionControlResult | None:
    """Detect server-side admission control from gauge time series.

    Args:
        scrape_ts: Prometheus scrape timestamps (nanoseconds).
        n_running: num_requests_running gauge values.
        n_waiting: num_requests_waiting gauge values.
        min_plateau_seconds: Minimum duration to consider a plateau.
        tolerance: Max variation in running count during plateau.

    Returns:
        AdmissionControlResult if detected, None otherwise.
    """
    if len(scrape_ts) < 3:
        return None

    # Step 1: Find sustained high-running periods
    max_running = np.max(n_running)
    near_max = n_running >= (max_running - tolerance)

    # Find contiguous runs of near-max running count
    transitions = np.diff(near_max.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1

    # Handle edge cases (starts at max, ends at max)
    if near_max[0]:
        starts = np.concatenate([[0], starts])
    if near_max[-1]:
        ends = np.concatenate([ends, [len(near_max)]])

    if len(starts) == 0:
        return None

    # Step 2: Find the longest plateau that meets the minimum duration
    best_start, best_end = 0, 0
    best_duration = 0.0

    for s, e in zip(starts, ends):
        duration_ns = scrape_ts[min(e, len(scrape_ts) - 1)] - scrape_ts[s]
        duration_s = duration_ns / NANOS_PER_SECOND
        if duration_s > best_duration:
            best_duration = duration_s
            best_start, best_end = s, min(e, len(scrape_ts) - 1)

    if best_duration < min_plateau_seconds:
        return None

    # Step 3: Confirm with queue growth during plateau
    plateau_waiting = n_waiting[best_start:best_end + 1]
    plateau_ts = scrape_ts[best_start:best_end + 1]

    if len(plateau_waiting) < 2:
        return None

    # Linear regression for growth rate
    dt = (plateau_ts - plateau_ts[0]) / NANOS_PER_SECOND
    if dt[-1] > 0:
        # Simple slope via least-squares
        n = len(dt)
        slope = (n * np.sum(dt * plateau_waiting) -
                 np.sum(dt) * np.sum(plateau_waiting)) / \
                (n * np.sum(dt**2) - np.sum(dt)**2)
    else:
        slope = 0.0

    # Only report if queue is actually growing
    if slope <= 0:
        return None

    estimated_limit = int(np.round(np.median(n_running[best_start:best_end + 1])))

    return AdmissionControlResult(
        detected=True,
        estimated_limit=estimated_limit,
        plateau_start_ns=float(scrape_ts[best_start]),
        plateau_end_ns=float(scrape_ts[best_end]),
        ttft_before_plateau_ms=0.0,    # filled by caller from ColumnStore
        ttft_during_plateau_ms=0.0,    # filled by caller from ColumnStore
        queue_growth_rate=float(slope),
    )
```

### 8.4 Client-Side Implications

When admission control is active, the client can observe several
characteristic patterns:

```
                       Admission Control Active
                    ┌─────────────────────────────┐
                    │                             │
 N_running:    ─────┤ ████████████████████████████ ├─────
                    │  (plateau at limit)          │
                    │                             │
 N_waiting:    ─────┤   ╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱  ├─────
                    │  (linear growth)            │
                    │                             │
 TTFT:         ─────┤    ╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱  ├─────
                    │  (dominated by queue wait)  │
                    │                             │
 ITL:          ─────┤ ────────────────────────────├─────
                    │  (unchanged — decode is     │
                    │   unaffected by queue depth) │
                    └─────────────────────────────┘
```

Key diagnostic: TTFT increases while ITL remains stable. This distinguishes
admission control from compute saturation (where both increase) and network
congestion (where both increase uniformly).

---

## 9. Arrival Pattern Impact on Queue Dynamics

### 9.1 AIPerf Timing Modes

AIPerf supports multiple arrival patterns via the TimingManager and
IntervalGenerator protocol:

| Mode | Distribution | Queue Model |
|------|-------------|-------------|
| Constant rate (`--request-rate N`) | Deterministic inter-arrivals (1/N seconds) | D/G/1 |
| Poisson (`--request-rate N --arrival-pattern poisson`) | Exponential inter-arrivals (mean 1/N) | M/G/1 |
| Gamma (`--request-rate N --arrival-pattern gamma`) | Gamma-distributed inter-arrivals | G/G/1 |
| Concurrency-limited (`--concurrency N`) | Closed-loop: new request on completion | Closed system |
| Rate ramp (`--ramp-start-rate ... --ramp-end-rate`) | Time-varying deterministic | Non-stationary |

Each pattern creates fundamentally different queue dynamics.

### 9.2 M/G/1 Queue Model (Poisson Arrivals)

The Poisson arrival pattern is the most analytically tractable and the most
realistic model for production traffic (superposition of many independent
users). The M/G/1 queue (Markov arrivals, General service times, 1 server)
provides closed-form performance predictions.

**Pollaczek-Khinchine formula** for mean waiting time in M/G/1 queue:

```
                  rho * (1 + C_s^2)
    W_q = E[S] * ─────────────────
                  2 * (1 - rho)

Where:
  E[S]  = mean service time
  rho   = lambda * E[S]  (server utilization, must be < 1 for stability)
  C_s   = std(S) / E[S]  (coefficient of variation of service time)
  W_q   = mean waiting time in queue
```

**Total sojourn time** (wait + service):

```
    W = W_q + E[S]

                  rho * (1 + C_s^2)
      = E[S] * ( ───────────────── + 1 )
                  2 * (1 - rho)

              E[S] * (2 - 2*rho + rho * (1 + C_s^2))
      = ──────────────────────────────────────────────
                      2 * (1 - rho)

              E[S] * (2 - rho + rho * C_s^2)
      = ──────────────────────────────────────
                  2 * (1 - rho)
```

**Average number in system** (via Little's Law):

```
    L = lambda * W
```

**Average number in queue:**

```
                       rho^2 * (1 + C_s^2)
    L_q = lambda * W_q = ─────────────────
                           2 * (1 - rho)
```

### 9.3 Service Time Distribution for LLM Inference

The "service time" in LLM inference has a highly variable distribution
because it depends on both input and output token counts:

```
    S = T_prefill(ISL) + T_decode(OSL)

Where:
  T_prefill(ISL) ≈ a * ISL + b        (approximately linear in input length)
  T_decode(OSL)  ≈ c * OSL             (approximately linear in output length)
  ISL, OSL are random variables from the dataset distribution
```

The coefficient of variation C_s is typically large (1.0-3.0) because:
- ISL can range from tens to thousands of tokens
- OSL depends on the model's generation behavior
- Continuous batching adds iteration-level variation

**Impact on queue dynamics**: The Pollaczek-Khinchine formula shows that
W_q grows with C_s^2. High service time variability dramatically increases
queuing delay even at moderate utilization:

```
Example: E[S] = 1s, C_s = 2.0

  rho = 0.5:  W_q = 1.0 * 0.5 * (1 + 4) / (2 * 0.5) = 2.5s
  rho = 0.8:  W_q = 1.0 * 0.8 * (1 + 4) / (2 * 0.2) = 10.0s
  rho = 0.9:  W_q = 1.0 * 0.9 * (1 + 4) / (2 * 0.1) = 22.5s

Compare with C_s = 0 (deterministic service, D/G/1 lower bound):
  rho = 0.5:  W_q = 1.0 * 0.5 * 1 / (2 * 0.5) = 0.5s
  rho = 0.8:  W_q = 1.0 * 0.8 * 1 / (2 * 0.2) = 2.0s
  rho = 0.9:  W_q = 1.0 * 0.9 * 1 / (2 * 0.1) = 4.5s
```

At rho = 0.9, high service time variability increases queuing delay by
a factor of 5. This is why LLM inference systems hit queuing problems
much earlier than traditional web services with more uniform request
processing times.

### 9.4 D/G/1 Queue Model (Constant Rate Arrivals)

Deterministic arrivals (constant rate) produce less queuing than Poisson
at the same utilization. The **Kingman bound** (G/G/1 approximation)
gives:

```
                  C_a^2 + C_s^2     rho
    W_q ≈ E[S] * ───────────── * ────────
                       2          1 - rho

Where:
  C_a = coefficient of variation of inter-arrival times
      = 0 for deterministic arrivals
      = 1 for Poisson arrivals
```

For deterministic arrivals (C_a = 0):

```
                  C_s^2     rho
    W_q ≈ E[S] * ───── * ────────
                    2     1 - rho
```

This is exactly half the Pollaczek-Khinchine result when C_a = 1, confirming
the intuition that removing arrival burstiness halves the queuing delay
contribution from arrival variability.

### 9.5 Closed-Loop System (Concurrency-Limited)

When AIPerf operates in concurrency mode (`--concurrency N`), the system is
**closed-loop**: a new request is not sent until a previous one completes
(and a concurrency slot opens). This is fundamentally different from
open-loop (rate-based) systems.

For a closed system with N clients:

```
    Throughput: X = N / (E[S] + E[Z])

Where:
  N    = number of concurrent clients (--concurrency)
  E[S] = mean service time (request_latency)
  E[Z] = mean think time (time between response and next request)
         For AIPerf: E[Z] ≈ credit_drop_latency (negligible if credits
         are pre-issued, significant if waiting for credit)
```

The closed-loop system is **self-regulating**: when the server slows down,
the effective arrival rate decreases proportionally. This means:

- Queue depth is bounded by N (cannot grow without bound)
- The system cannot enter the unstable regime (rho >= 1)
- But it suffers from coordinated omission (slow responses reduce
  the measurement of tail latency)

**Comparison of queue dynamics by arrival pattern:**

```
Queue Depth vs Time:

Open-loop (Poisson, rho > 1):
  N_wait ──╱╱╱╱╱╱╱╱╱╱╱╱──  (grows without bound)

Open-loop (Poisson, rho = 0.9):
                ╱╲
  N_wait ──╱╲──╱  ╲──╱╲──  (bursty, occasionally large)
             ╲╱    ╲╱  ╲╱

Open-loop (Constant, rho = 0.9):
             ╱╲    ╱╲
  N_wait ───╱  ╲──╱  ╲───  (smaller oscillations)
                ╲╱

Closed-loop (N = 10):
  N_wait ──────────────────  (≈ 0, self-regulated)
```

### 9.6 Rate Ramp Arrival Patterns

Rate ramping (`--ramp-start-rate`, `--ramp-end-rate`) creates a
non-stationary arrival process. The effective utilization rho(t) changes
over time:

```
    rho(t) = lambda(t) * E[S]

Where:
    lambda(t) = lambda_start + (lambda_end - lambda_start) * t / T_ramp
```

Queue dynamics during a ramp:

```
Phase 1 (rho(t) < 1):  Queue stays near zero. Latency stable.
                        Little's Law holds locally.

Phase 2 (rho(t) → 1):  Queue builds slowly. Latency begins to increase.
                        The system is "finding its limit."

Phase 3 (rho(t) > 1):  Queue grows rapidly. Latency diverges.
                        Little's Law breaks (system not in steady state).
```

The transition point rho(t) = 1 is where the server reaches saturation.
Finding this point precisely is valuable for capacity planning:

```
    lambda_max = 1 / E[S]
    t_saturation = T_ramp * (lambda_max - lambda_start) / (lambda_end - lambda_start)
```

Correlating the observed queue buildup onset with this theoretical
prediction validates both the measurement and the model.

### 9.7 Gamma Arrivals and Burstiness Control

AIPerf's gamma arrival pattern provides tunable burstiness via the
`arrival_smoothness` parameter (shape parameter k of the Gamma distribution):

```
  k → infinity:  Gamma approaches deterministic (D/G/1)
  k = 1:         Gamma equals exponential (M/G/1, Poisson process)
  k < 1:         More bursty than Poisson (super-Poisson)
```

The coefficient of variation of inter-arrival times:

```
  C_a = 1 / sqrt(k)
```

Plugging into the Kingman bound:

```
                  1/k + C_s^2     rho
    W_q ≈ E[S] * ─────────── * ────────
                      2          1 - rho
```

This allows researchers to explore the sensitivity of queue dynamics to
arrival burstiness in a controlled manner, isolating the effect from
service time variability.

---

## 10. Implementation Architecture

### 10.1 Integration with Existing Infrastructure

The correlation analysis builds on three existing AIPerf subsystems:

```
┌─────────────────────────────────────────────────────────┐
│                   QueueCorrelationAnalyzer               │
│                   (new AnalyzerProtocol)                 │
│                                                         │
│  Reads from:                                            │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │ MetricsAccumulator│ │ ServerMetricsAccumulator     │  │
│  │   .column_store  │  │   .hierarchy                 │  │
│  │   .sweep_curves  │  │   (gauge/histogram series)   │  │
│  └─────────────────┘  └──────────────────────────────┘  │
│                                                         │
│  Produces:                                              │
│  ┌─────────────────────────────────────────────────┐    │
│  │ QueueCorrelationSummary                          │    │
│  │   .littles_law_residual: LittlesLawResult        │    │
│  │   .concurrency_gap: ConcurrencyGapResult         │    │
│  │   .queue_dynamics: QueueDynamicsResult            │    │
│  │   .scheduling_fairness: FairnessResult           │    │
│  │   .phase_interference: InterferenceResult        │    │
│  │   .admission_control: AdmissionControlResult     │    │
│  │   .queueing_model: QueueModelFit                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  Exports to:                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Console  │  │ JSON     │  │ CSV      │              │
│  │ warnings │  │ full     │  │ summary  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

### 10.2 AnalyzerProtocol Integration

Following the pattern established by `SteadyStateAnalyzer`, the queue
correlation analyzer would be an AnalyzerProtocol plugin:

```python
class QueueCorrelationAnalyzer:
    """Correlate server queue metrics with client latency observations.

    Reads from both MetricsAccumulator (client-side sweep curves, ColumnStore)
    and ServerMetricsAccumulator (server-side gauge/histogram time series).

    Requires:
        - Server metrics collection enabled (--server-metrics-url)
        - vLLM-compatible metrics (num_requests_running, num_requests_waiting)
    """

    record_type: ClassVar[str] = "metric_records"

    def __init__(self, user_config: UserConfig) -> None:
        self._config = user_config
        self._enabled = bool(user_config.server_metrics.urls)

    def analyze(self, ctx: SummaryContext) -> QueueCorrelationSummary | None:
        """Run all correlation analyses.

        Returns None if server metrics are not available.
        """
        if not self._enabled:
            return None

        metrics_acc = self._get_metrics_accumulator(ctx)
        server_acc = self._get_server_metrics_accumulator(ctx)

        if metrics_acc is None or server_acc is None:
            return None

        # Extract time series
        column_store = metrics_acc.column_store
        sweep_curves = metrics_acc.sweep_curves
        server_ts = server_acc.hierarchy

        # Run analyses (each is independent, could parallelize)
        return QueueCorrelationSummary(
            littles_law=self._littles_law_check(
                sweep_curves, column_store, ctx.start_ns, ctx.end_ns
            ),
            concurrency_gap=self._concurrency_gap(
                sweep_curves, server_ts, ctx.start_ns, ctx.end_ns
            ),
            queue_dynamics=self._queue_dynamics(
                server_ts, ctx.start_ns, ctx.end_ns
            ),
            scheduling_fairness=self._scheduling_fairness(
                server_ts, column_store
            ),
            phase_interference=self._phase_interference(
                sweep_curves, column_store
            ),
            admission_control=self._admission_control(
                server_ts, column_store, ctx.start_ns, ctx.end_ns
            ),
            queueing_model=self._fit_queueing_model(
                column_store, ctx.start_ns, ctx.end_ns
            ),
        )
```

### 10.3 Data Model

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LittlesLawResult:
    """Results of Little's Law validation at multiple levels."""

    client_level: LittlesLawCheck
    server_queue_level: LittlesLawCheck | None    # None if no queue metrics
    server_exec_level: LittlesLawCheck | None
    prefill_level: LittlesLawCheck | None


@dataclass(frozen=True, slots=True)
class LittlesLawCheck:
    """Single-level Little's Law validation."""

    L: float                     # Average number in system
    lam: float                   # Arrival rate (req/sec)
    W: float                     # Average sojourn time (seconds)
    lam_times_W: float           # lambda * W (should equal L)
    residual: float              # L - lambda * W
    residual_pct: float          # (L - lambda * W) / L * 100
    valid: bool                  # |residual_pct| < threshold


@dataclass(frozen=True, slots=True)
class ConcurrencyGapResult:
    """Client vs server concurrency gap analysis."""

    avg_gap: float               # Time-weighted average gap
    p50_gap: float
    p99_gap: float
    estimated_rtt_ms: float      # Estimated network RTT from gap / lambda
    gap_trend: str               # "stable", "growing", "shrinking"


@dataclass(frozen=True, slots=True)
class QueueDynamicsResult:
    """Queue buildup analysis."""

    regime: str                  # "healthy", "saturated_stable", "overloading"
    avg_growth_rate: float       # requests/sec
    max_growth_rate: float
    overload_fraction: float     # Fraction of time in overloading regime
    latency_lag_ns: float        # Estimated lag from queue growth to latency
    latency_lag_correlation: float


@dataclass(frozen=True, slots=True)
class FairnessResult:
    """Scheduling fairness analysis."""

    queue_time_cv: float                 # Coefficient of variation
    queue_time_tail_ratio: float         # p99/p50
    isl_correlation: float | None        # Spearman rho (ISL vs queue time)
    isl_bias: str                        # "positive_bias", "negative_bias", "none"
    hol_blocking_suspected: bool


@dataclass(frozen=True, slots=True)
class InterferenceResult:
    """Prefill-decode phase interference."""

    prefill_interference_factor: float
    prefill_itl_correlation: float
    itl_at_low_prefill_ms: float
    itl_at_high_prefill_ms: float
    interference_severity: str           # "minimal", "moderate", "significant", "severe"


@dataclass(frozen=True, slots=True)
class QueueModelFit:
    """M/G/1 or closed-loop queue model fit to observed data."""

    model_type: str              # "M/G/1", "D/G/1", "closed"
    utilization: float           # rho = lambda * E[S]
    predicted_mean_wait: float   # Pollaczek-Khinchine prediction (seconds)
    observed_mean_wait: float    # From server metrics (seconds)
    prediction_error_pct: float  # |predicted - observed| / observed * 100
    service_time_cv: float       # C_s from observed latencies
    arrival_cv: float            # C_a from inter-arrival times


@dataclass(frozen=True, slots=True)
class QueueCorrelationSummary:
    """Complete queue correlation analysis results."""

    littles_law: LittlesLawResult
    concurrency_gap: ConcurrencyGapResult | None
    queue_dynamics: QueueDynamicsResult | None
    scheduling_fairness: FairnessResult | None
    phase_interference: InterferenceResult | None
    admission_control: AdmissionControlResult | None
    queueing_model: QueueModelFit | None

    def to_json(self) -> dict[str, Any]:
        """Serialize to JSON for export."""
        # Each sub-result has its own to_json or dataclasses.asdict
        ...

    def to_csv(self) -> list[dict[str, Any]]:
        """Flatten to CSV rows for tabular export."""
        ...
```

### 10.4 Plugin Registration

```yaml
# In plugins.yaml, under analyzers
queue_correlation:
  class: aiperf.analysis.queue_correlation.QueueCorrelationAnalyzer
  description: >
    Cross-correlates server queue metrics with client latency observations.
    Validates Little's Law, detects admission control, measures phase
    interference, and fits queueing models.
  metadata:
    requires_server_metrics: true
    record_types:
      - metric_records
      - server_metrics
```

### 10.5 CLI Integration

```
--queue-analysis                Enable queue correlation analysis (default: off)
--queue-analysis-window SEC     Window size for time-varying analysis (default: 5.0)
```

Environment variables:

```
AIPERF_QUEUE_ANALYSIS=true
AIPERF_QUEUE_ANALYSIS_WINDOW=5.0
```

The analysis is disabled by default because it requires server metrics
collection (`--server-metrics-url`) and adds computational overhead.
When enabled without server metrics, it falls back to client-only analysis
(Little's Law at the client level, phase interference from sweep curves).

---

## 11. Sampling, Interpolation & Alignment

### 11.1 The Alignment Problem

Client and server metrics exist on fundamentally different time grids:

```
Client events:     .|..|...|....|.....|......|
                   (per-request, nanosecond resolution)

Server scrapes:    |         |         |         |
                   (periodic, 1-5 second intervals)

Analysis needs:    |    |    |    |    |    |    |
                   (uniform grid for correlation)
```

Three challenges arise:

1. **Resolution mismatch**: Client events are nanosecond-resolution. Server
   scrapes are 1-5 seconds apart. Aligning these requires interpolation
   of the lower-resolution series.

2. **Clock alignment**: Client and server may have different clock sources.
   While both use wall-clock time, clock skew or NTP jitter can introduce
   offsets of 1-100ms.

3. **Semantic mismatch**: Client concurrency is a step function (exact at
   event boundaries). Server gauge is a point sample (unknown between
   scrapes). These are different mathematical objects.

### 11.2 Interpolation Strategies

**For gauge metrics (num_requests_running, num_requests_waiting):**

| Strategy | Assumption | Error Profile |
|----------|-----------|---------------|
| Zero-order hold (ZOH) | Value constant between scrapes | Misses transients between scrapes. Introduces step artifacts. But preserves actual measured values. |
| Linear interpolation | Value changes linearly between scrapes | Smoother, but assumes linearity that may not hold. Averages out spikes. |
| No interpolation | Only analyze at scrape times | Safest. Limits analysis to scrape-time coincidences. May miss important dynamics. |

**Recommendation: Zero-order hold** for gauge metrics. This matches the
semantics of a gauge (instantaneous snapshot) and does not introduce
artificial smoothing. The trade-off (missing transients) is inherent in
the scrape resolution, not the interpolation.

**For histogram metrics (request_queue_time_seconds):**

Histograms are cumulative counters. Between scrapes, we know the total
count of observations in each bucket increased by some unknown amount.
The rate of change (deltas) can be computed between consecutive scrapes,
but sub-scrape resolution is not available.

**Recommendation: No interpolation.** Analyze at scrape boundaries only.
Compute delta rates for each interval between scrapes.

### 11.3 Minimum Scrape Rate Requirements

The Nyquist theorem (adapted for monitoring) suggests the scrape interval
must be at most half the period of the fastest dynamics we want to observe.

| Phenomenon | Typical Period | Required Scrape Interval |
|-----------|---------------|--------------------------|
| Batch scheduling cycle | 10-50ms | Not observable (too fast for Prometheus) |
| Queue buildup onset | 1-10s | 0.5-5s |
| Admission control plateau | 10-60s | 5-30s |
| Ramp-up / ramp-down | 30-300s | 15-150s |
| Steady-state statistics | Entire run | Single measurement |

**Practical guidance:**

- 1s scrape interval: Captures queue buildup, admission control, phase
  interference (time-averaged). Adequate for most analyses.
- 0.5s scrape interval: Better temporal resolution for queue dynamics.
  May increase server-side overhead from Prometheus scrapes.
- 5s scrape interval (default in many deployments): Too slow for queue
  buildup detection. Can still detect admission control and compute
  aggregate statistics.

### 11.4 Clock Alignment

If client and server clocks differ by offset `delta_clock`:

```
    t_server_actual = t_server_reported + delta_clock
```

This offset affects cross-correlation but not within-domain analysis.

**Estimation approach:** Use the HTTP request/response round-trip to
bound the clock offset. For each request:

```
    t_client_send < t_server_receive < t_server_send < t_client_receive
    delta_clock ∈ [t_client_send - t_server_receive,
                   t_client_receive - t_server_send]
```

AIPerf already records `request_start_ns` (client send) and
`request_ack_ns` (server response). While the server-side timestamps are
not directly available in vLLM metrics, the `e2e_request_latency_seconds`
histogram provides the server's view of processing time. The difference
between client `request_latency` and server `e2e_request_latency` gives
an estimate of total network overhead, which bounds the clock alignment
error.

### 11.5 Resampling to Uniform Grid

For cross-correlation analysis (FFT-based), both signals must be on the
same uniform time grid.

```python
def resample_to_uniform_grid(
    ts: NDArray[np.float64],
    values: NDArray[np.float64],
    grid_start: float,
    grid_end: float,
    grid_step: float,
    method: str = "zoh",  # "zoh" or "linear"
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Resample a time series to a uniform grid.

    Args:
        ts: Original timestamps.
        values: Original values.
        grid_start: Start of output grid.
        grid_end: End of output grid.
        grid_step: Step size of output grid.
        method: "zoh" for zero-order hold, "linear" for linear interpolation.

    Returns:
        (grid_timestamps, resampled_values)
    """
    grid = np.arange(grid_start, grid_end, grid_step)

    if method == "zoh":
        idx = np.searchsorted(ts, grid, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return grid, values[idx]
    elif method == "linear":
        return grid, np.interp(grid, ts, values)
    else:
        raise ValueError(f"Unknown method: {method}")
```

**Grid step selection:** The grid step should be the coarser of the two
input resolutions. For client sweep + server scrape correlation, use the
server scrape interval (typically 1s). Using a finer grid creates no new
information but increases computation.

---

## 12. Real-Time Correlation Dashboard

### 12.1 Live Metrics

The following derived metrics could be computed incrementally during the
benchmark run and published via realtime ZMQ messages:

| Metric | Update Frequency | Computation |
|--------|-----------------|-------------|
| `littles_law_residual_pct` | Per server scrape | R_norm from trailing window |
| `client_server_gap` | Per server scrape | Sweep concurrency - (N_run + N_wait) |
| `queue_regime` | Per server scrape | Classify from growth rate |
| `prefill_interference_factor` | Every 10s | PIF from trailing window |
| `admission_control_detected` | Per server scrape | Boolean from plateau detection |

### 12.2 Alert Thresholds

Real-time alerting during a benchmark run:

```
WARNING: Little's Law residual > 20% — system not in steady state
WARNING: Queue growth rate > 5 req/s — system overloading
WARNING: Admission control detected — server limit at N=128 requests
WARNING: Prefill interference factor > 2.0 — severe phase contention
INFO:    Queue regime: SATURATED_STABLE — server at capacity
```

### 12.3 Integration with Textual UI

AIPerf's Textual-based UI could display a live correlation panel:

```
┌─ Queue Correlation ──────────────────────────────────┐
│                                                      │
│  Little's Law  ✓  R = -2.3%   (healthy)              │
│  Queue Regime      HEALTHY    N_wait = 0.2 avg       │
│  Concurrency Gap   1.3 req    (RTT ≈ 12ms)          │
│  Prefill IF        1.12       (minimal interference) │
│  Admission Ctrl    not detected                      │
│                                                      │
│  Server Queue: ▁▁▂▁▁▁▁▂▃▂▁▁▁▁▁▂▁▁▁ (sparkline)    │
│  Client Conc:  ▅▅▆▅▅▅▅▆▇▆▅▅▅▅▅▆▅▅▅ (sparkline)    │
└──────────────────────────────────────────────────────┘
```

### 12.4 Incremental Computation

For real-time display, the analyses must be incremental (O(1) per update):

**Little's Law**: Maintain running sums for L (from sweep-line running
average), lambda (request counter / elapsed time), W (running mean of
latency).

**Queue growth rate**: Store last two scrape values. Growth rate =
delta / dt.

**Concurrency gap**: Read latest sweep-line value and latest scrape value.
Gap = difference.

**Prefill interference**: More expensive. Requires maintaining a sliding
window of (prefill_concurrency, ITL) pairs. Could use the existing
exponential moving average pattern from realtime metrics.

---

## 13. Validation Strategy

### 13.1 Synthetic Test Scenarios

The validation suite should include synthetic scenarios with known
ground truth:

| Scenario | Setup | Expected |
|----------|-------|----------|
| `steady_state_low_load` | Constant rate at rho=0.3, D/M/1 | L approx lambda * W, gap approx 0, queue approx 0 |
| `steady_state_high_load` | Constant rate at rho=0.9 | L approx lambda * W, queue > 0 but stable |
| `overload_ramp` | Linear ramp from rho=0.5 to rho=1.5 | Queue growth onset at rho=1.0, Little's Law breaks |
| `admission_control` | Rate > capacity, server limit at N=10 | Plateau detected, TTFT spike correlated |
| `poisson_mg1` | Poisson arrivals, heavy-tail service | W_q matches Pollaczek-Khinchine prediction |
| `chunked_prefill` | Alternating short/long prompts | Low PIF, high ITL autocorrelation |
| `no_chunked_prefill` | Same workload, no chunking | High PIF, sharp ITL spikes |
| `clock_skew` | Artificial 50ms offset | Clock alignment algorithm detects and corrects |
| `network_congestion` | Add 100ms latency | Concurrency gap increases proportionally |
| `bursty_arrivals` | Gamma(k=0.5) arrivals | Queue variance matches G/G/1 prediction |

### 13.2 Accuracy Metrics

For each analysis component, define quantitative accuracy targets:

| Component | Metric | Target |
|-----------|--------|--------|
| Little's Law | |R_norm| in steady state | < 5% |
| Concurrency gap | Gap vs actual network delay | Within 1 request |
| Queue growth rate | Slope error vs known ramp | < 10% |
| Admission control | Estimated limit vs actual | Exact (integer) |
| PIF | Factor vs controlled experiment | Within 20% |
| M/G/1 fit | Predicted vs observed W_q | Within 25% |

### 13.3 Integration Testing

Test with real vLLM instances under controlled conditions:

1. **Known server config**: Set `--max-num-seqs` to a known value. Verify
   admission control detection finds it.
2. **Known arrival pattern**: Use constant rate at known rho. Verify
   Little's Law residual < 5%.
3. **Known network delay**: Add `tc netem delay` to introduce controlled
   latency. Verify concurrency gap estimates the delay.
4. **A/B chunked prefill**: Run the same workload with and without
   `--enable-chunked-prefill`. Verify PIF differs significantly.

### 13.4 Test Infrastructure

```python
@dataclass
class SyntheticServerMetrics:
    """Generate synthetic server metrics for testing."""

    scrape_interval_ns: float = 1_000_000_000  # 1s

    def generate_gauge_series(
        self,
        start_ns: float,
        end_ns: float,
        base_value: float,
        noise_std: float = 0.0,
        trend: float = 0.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Generate a synthetic gauge time series."""
        ts = np.arange(start_ns, end_ns, self.scrape_interval_ns)
        values = base_value + trend * (ts - start_ns) / NANOS_PER_SECOND
        if noise_std > 0:
            values += np.random.default_rng(42).normal(0, noise_std, len(ts))
        return ts, np.maximum(values, 0)  # gauges are non-negative

    def generate_mg1_queue(
        self,
        arrival_rate: float,
        service_rate: float,
        service_cv: float,
        duration_s: float,
        rng_seed: int = 42,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Simulate an M/G/1 queue and return (timestamps, n_running, n_waiting, queue_times).

        Used for ground-truth validation of queue correlation analyses.
        """
        rng = np.random.default_rng(rng_seed)
        n_arrivals = int(arrival_rate * duration_s * 1.5)  # oversample

        # Generate arrivals (Poisson process)
        inter_arrivals = rng.exponential(1.0 / arrival_rate, n_arrivals)
        arrival_times = np.cumsum(inter_arrivals)
        arrival_times = arrival_times[arrival_times < duration_s]

        # Generate service times (Gamma distribution with given CV)
        mean_service = 1.0 / service_rate
        if service_cv > 0:
            shape = 1.0 / (service_cv ** 2)
            scale = mean_service / shape
            service_times = rng.gamma(shape, scale, len(arrival_times))
        else:
            service_times = np.full(len(arrival_times), mean_service)

        # Simulate FIFO single-server queue
        n = len(arrival_times)
        departure_times = np.zeros(n)
        queue_times_out = np.zeros(n)

        server_free_at = 0.0
        for i in range(n):
            if arrival_times[i] >= server_free_at:
                # No wait
                queue_times_out[i] = 0.0
                departure_times[i] = arrival_times[i] + service_times[i]
            else:
                # Wait for server
                queue_times_out[i] = server_free_at - arrival_times[i]
                departure_times[i] = server_free_at + service_times[i]
            server_free_at = departure_times[i]

        return arrival_times, departure_times, queue_times_out, service_times
```

---

## 14. References

### Queue Theory Foundations

1. **Little, J.D.C.** (1961). "A Proof for the Queuing Formula: L = lambda W."
   *Operations Research*, 9(3), 383-387.
   - The original proof of Little's Law. Remarkably general: holds for any
     stationary queueing system regardless of arrival distribution, service
     distribution, or scheduling discipline.

2. **Pollaczek, F.** (1930) and **Khinchine, A.Y.** (1932).
   Pollaczek-Khinchine formula for M/G/1 queue mean waiting time.
   - The workhorse formula for predicting queuing delay with general
     service times. Key insight: delay grows with the *square* of service
     time variability.

3. **Kingman, J.F.C.** (1961). "The single server queue in heavy traffic."
   *Mathematical Proceedings of the Cambridge Philosophical Society*, 57(4).
   - The heavy-traffic approximation for G/G/1 queues. Provides the bound
     used in Section 9.4.

4. **Harchol-Balter, M.** (2013). *Performance Modeling and Design of
   Computer Systems: Queueing Theory in Action.* Cambridge University Press.
   - Comprehensive modern treatment of queueing theory applied to computer
     systems. Covers M/G/1, closed systems, and the impact of heavy tails.

### LLM Inference Scheduling

5. **Kwon, W., Li, Z., Zhuang, S., et al.** (2023). "Efficient Memory
   Management for Large Language Model Serving with PagedAttention." *SOSP*.
   - Introduces vLLM's PagedAttention and continuous batching. The KV cache
     management directly impacts queue dynamics (memory-based admission
     control).

6. **Agrawal, A., Panwar, A., Mohan, J., et al.** (2024). "Sarathi-Serve:
   The case for chunked-prefills in LLM serving." arXiv:2308.16369v4.
   - Analysis of prefill-decode interference and the chunked prefill
     solution. Directly relevant to Section 7.

7. **Zhong, Y., Liu, S., Chen, J., et al.** (2024). "DistServe: Disaggregating
   Prefill and Decoding for Goodput-optimized Large Language Model Serving."
   *OSDI*.
   - Proposes physical separation of prefill and decode to eliminate
     interference. Defines the goodput metric used in AIPerf.

### Performance Measurement

8. **Tene, G.** (2013). "How NOT to Measure Latency."
   - Foundational talk on coordinated omission. Directly relevant to the
     interaction between queue depth measurement and latency reporting.

9. **Dean, J. & Barroso, L.A.** (2013). "The Tail at Scale." *Communications
   of the ACM*, 56(2), 74-80.
   - Established that tail latency (p99, p99.9) is what matters at scale.
     Queue depth variability is a primary driver of tail latency.

10. **Law, A.M.** (2015). *Simulation Modeling and Analysis*, 5th ed. McGraw-Hill.
    - Standard reference for simulation output analysis: stationarity testing,
      variance estimation with autocorrelation, batch means, and Little's Law
      validation for discrete event simulations.

### Signal Processing & Correlation

11. **Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M.** (2015).
    *Time Series Analysis: Forecasting and Control*, 5th ed. Wiley.
    - Covers cross-correlation, ARIMA modeling, and transfer functions.
      Relevant to the lag estimation in Section 5.4.

12. **Stoica, P. & Moses, R.** (2005). *Spectral Analysis of Signals.*
    Prentice Hall.
    - FFT-based cross-correlation methods used in the lag estimation
      algorithm.

---

## Appendix A: Summary of Formulas

```
Little's Law:
    L = lambda * W

Pollaczek-Khinchine (M/G/1 mean wait):
    W_q = E[S] * rho * (1 + C_s^2) / (2 * (1 - rho))

Kingman Bound (G/G/1 mean wait):
    W_q ≈ E[S] * (C_a^2 + C_s^2) / 2 * rho / (1 - rho)

Server utilization:
    rho = lambda * E[S]

Closed system throughput:
    X = N / (E[S] + E[Z])

Little's Law residual:
    R = L - lambda * W
    R_norm = R / L * 100%

Queue growth rate:
    dN_wait/dt = lambda_arrival - mu_scheduling

Prefill interference factor:
    PIF = ITL_high_prefill / ITL_low_prefill

Scheduling fairness (CV):
    CV = std(queue_time) / mean(queue_time)

Network RTT from gap:
    RTT ≈ gap / lambda

Effective sample size:
    n_eff = n * (1 - rho_lag1) / (1 + rho_lag1)

Gamma inter-arrival CV:
    C_a = 1 / sqrt(k)

Saturation time (linear ramp):
    t_sat = T_ramp * (1/E[S] - lambda_start) / (lambda_end - lambda_start)
```

## Appendix B: Metric Name Mapping

Mapping between vLLM Prometheus metric names and the internal names used
in this document and in AIPerf's ServerMetricsAccumulator:

| Prometheus Metric | Type | This Document | AIPerf Internal |
|------------------|------|---------------|-----------------|
| `vllm:num_requests_running` | Gauge | N_run, L_exec | Stored in ServerMetricsHierarchy as gauge time series |
| `vllm:num_requests_waiting` | Gauge | N_wait, L_queue | Stored in ServerMetricsHierarchy as gauge time series |
| `vllm:request_queue_time_seconds` | Histogram | W_queue | Polynomial percentile estimation via `histogram_percentiles.py` |
| `vllm:request_success` | Counter | lambda_server (via delta rate) | Delta rate computed in `export_stats.py` |
| `vllm:e2e_request_latency_seconds` | Histogram | W_server | Polynomial percentile estimation |

Client-side metrics mapping:

| AIPerf Metric | Sweep/ColumnStore | This Document |
|--------------|-------------------|---------------|
| `effective_concurrency` | `concurrency_sweep()` → SweepCurves | L_client |
| `effective_prefill_concurrency` | Sweep-line | L_prefill |
| `effective_generation_concurrency` | Sweep-line | L_decode |
| `request_throughput` | Derived (count / duration) | lambda_client |
| `request_latency` | ColumnStore record field | W_client |
| `TTFT` | ColumnStore record field | W_prefill (approx) |
| `credit_drop_latency` | ColumnStore record field | Client-side queuing |

## Appendix C: Decision Matrix

| Analysis | Requires Server Metrics? | Real-Time Capable? | Computational Cost | Priority |
|----------|-------------------------|--------------------|--------------------|----------|
| Little's Law (client) | No | Yes (incremental) | O(1) per window | P0 (foundational) |
| Little's Law (server) | Yes | Yes | O(1) per scrape | P1 |
| Concurrency gap | Yes | Yes | O(1) per scrape | P1 |
| Queue growth rate | Yes | Yes | O(1) per scrape | P1 |
| Queue regime classification | Yes | Yes | O(1) per scrape | P1 |
| Overload prediction | Yes | Yes | O(1) per scrape | P2 |
| Queue-latency lag | Yes | No (FFT-based) | O(n log n) post-run | P2 |
| Scheduling fairness | Yes | No (histogram) | O(bins) post-run | P2 |
| ISL correlation | Partial (need ISL + queue time) | No | O(n log n) post-run | P3 |
| Phase interference | No (sweep curves) | Partial (windowed) | O(n) post-run | P2 |
| Admission control | Yes | Yes | O(n) streaming | P1 |
| M/G/1 model fit | No (client metrics sufficient) | No | O(n) post-run | P3 |
| Clock alignment | Yes | No | O(n) post-run | P3 |

---

*This document provides the mathematical foundations and algorithmic designs
for cross-domain correlation analysis. Implementation should proceed in
priority order (P0 through P3), with each phase validated against synthetic
and real-world test scenarios before advancing.*
