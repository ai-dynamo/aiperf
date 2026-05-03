# Client-Server Latency Decomposition & Correlation Analysis

**Status**: Research

**Authors**: Anthony Casagrande

**Category**: Research — Correlation Analysis

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Metric Inventory and Measurement Points](#2-metric-inventory-and-measurement-points)
3. [Latency Discrepancy Analysis](#3-latency-discrepancy-analysis)
4. [TTFT Decomposition](#4-ttft-decomposition)
5. [ITL Consistency Analysis](#5-itl-consistency-analysis)
6. [Temporal Correlation](#6-temporal-correlation)
7. [Causal Inference and Root Cause Attribution](#7-causal-inference-and-root-cause-attribution)
8. [Practical Implementation in AIPerf](#8-practical-implementation-in-aiperf)
9. [Visualization Approaches](#9-visualization-approaches)
10. [Academic References and Theoretical Foundations](#10-academic-references-and-theoretical-foundations)
11. [Appendix: Mathematical Notation](#appendix-mathematical-notation)

---

## 1. Introduction

### 1.1 Problem Statement

When an LLM inference benchmark reports "p99 TTFT = 142ms," users face an
immediate diagnostic question: *where does that time go?* Is the server slow
to prefill? Is the network adding 30ms of overhead? Is the request sitting in
a queue for 80ms before computation starts? Without decomposing client-measured
latency into its constituent phases and correlating those phases with server-side
telemetry, the benchmark result is a black box.

AIPerf already collects rich data from both sides of the client-server boundary:

- **Client side**: Per-request nanosecond timestamps covering the full HTTP
  lifecycle (DNS, TCP, TLS, send, wait, receive), credit scheduling, and
  token-level streaming events.
- **Server side**: Prometheus histograms from vLLM/TRT-LLM covering end-to-end
  latency, queue time, TTFT, ITL, and instantaneous concurrency gauges.
- **GPU side**: DCGM telemetry covering utilization, power, temperature, and
  memory bandwidth.

The challenge is *aligning* these heterogeneous data sources temporally and
*correlating* them statistically to produce actionable decompositions.

### 1.2 Goals of This Research

1. Define formal decomposition models for client-measured latencies.
2. Develop methods to quantify client-server discrepancies and attribute them
   to specific causes (network, queueing, compute, scheduling).
3. Design statistical correlation techniques suitable for the temporal structure
   of benchmark data (autocorrelation, non-stationarity, lag).
4. Provide concrete implementation guidance within AIPerf's existing architecture
   (ColumnStore, ServerMetricsHierarchy, sweep-line algorithms).

### 1.3 Scope and Constraints

This analysis operates under several practical constraints:

- **No request-level join key**: Client requests and server histograms cannot
  be linked at the individual request level. Server metrics are aggregated
  (histograms, gauges) scraped at configurable intervals (default 1-5s).
- **Clock domains differ**: Client uses `time.perf_counter_ns()` (monotonic),
  server Prometheus timestamps use wall-clock `time.time_ns()`. Correlation
  must account for clock skew.
- **Histogram resolution is lossy**: Server-side histograms use bucket
  boundaries (e.g., 0.001, 0.005, 0.01, 0.025, ..., 10.0 seconds). Individual
  request latencies are lost; only cumulative counts per bucket are available.
- **Scrape frequency limits temporal resolution**: Server metrics update at
  scrape intervals (typically 1-15s), while client metrics have per-request
  granularity (sub-millisecond).

---

## 2. Metric Inventory and Measurement Points

### 2.1 Client-Side Measurement Points

AIPerf's client-side metrics form a precise timeline for each request. The
measurement points, in chronological order:

```
                                 CLIENT TIMELINE
    ════════════════════════════════════════════════════════════════════

    credit_issued_ns                                          (TimingManager)
         │
         ├── [queue_wait_time]
         │
    request_start_ns ─── credit_drop_latency ──┐             (Worker)
         │                                      │
         ├── dns_lookup_start ─┐                │
         │   dns_lookup_end ───┘ http_req_dns   │             (aiohttp trace)
         │                                      │
         ├── tcp_connect_start ┐                │
         │   tcp_connect_end ──┘ http_req_conn  │             (aiohttp trace)
         │                                      │
         ├── request_send_start ┐               │
         │   request_send_end ──┘ http_req_send │             (aiohttp trace)
         │                                      │
         ├── [server processing + network]      │
         │                                      │
         ├── recv_start (HTTP 200 OK) ──────────┘             (stream_setup_latency)
         │
         ├── first_content_response ────────────── TTFT        (content_responses[0])
         │
         ├── second_content_response ───────────── TTST        (content_responses[1])
         │
         ├── ... [token-by-token streaming] ────── ITL         (inter-token gaps)
         │
         └── last_content_response ─────────────── request_latency end
                                                               (content_responses[-1])
```

The key metrics computed from these timestamps:

| Metric | Formula | Unit | Source |
|--------|---------|------|--------|
| `credit_drop_latency` | `request_start_ns - credit_drop_received_ns` | ns | Worker internal |
| `stream_setup_latency` | `recv_start_perf_ns - start_perf_ns` | ns | HTTP response header arrival |
| `stream_prefill_latency` | `TTFT - stream_setup_latency` | ns | Derived |
| `time_to_first_token` | `first_content_response.perf_ns - start_perf_ns` | ns | First token arrival |
| `time_to_second_token` | `second_content_response.perf_ns - start_perf_ns` | ns | Second token arrival |
| `inter_token_latency` | `(request_latency - TTFT) / (OSL - 1)` | ns | Average decode rate |
| `request_latency` | `last_content_response.perf_ns - start_perf_ns` | ns | End-to-end client |
| `http_req_blocked` | pool wait end - pool wait start | ns | Connection pool |
| `http_req_dns_lookup` | DNS end - DNS start | ns | DNS resolution |
| `http_req_connecting` | TCP end - TCP start (incl. TLS) | ns | Connection setup |
| `http_req_sending` | send end - send start | ns | Request transmission |
| `http_req_waiting` | first response chunk - send end | ns | TTFB (server processing) |
| `http_req_receiving` | last chunk - first chunk | ns | Response transfer |
| `http_req_duration` | sending + waiting + receiving | ns | HTTP exchange only |
| `http_req_connection_overhead` | blocked + DNS + connecting | ns | Pre-request overhead |
| `http_req_total` | all 6 phases summed | ns | Complete HTTP lifecycle |

### 2.2 Server-Side Measurement Points (vLLM Example)

vLLM exposes Prometheus metrics at `/metrics`. The relevant latency and load
metrics:

| Metric | Prometheus Type | Description |
|--------|----------------|-------------|
| `vllm:e2e_request_latency_seconds` | histogram | Server-side end-to-end per request |
| `vllm:time_to_first_token_seconds` | histogram | Server-side TTFT (queue + prefill) |
| `vllm:inter_token_latency_seconds` | histogram | Server-side per-token decode latency |
| `vllm:request_queue_time_seconds` | histogram | Time request spent in scheduler queue |
| `vllm:num_requests_running` | gauge | Instantaneous active request count |
| `vllm:num_requests_waiting` | gauge | Requests in queue awaiting scheduling |
| `vllm:num_preemptions_total` | counter | Preemption events (KV cache pressure) |
| `vllm:gpu_cache_usage_perc` | gauge | KV cache utilization (0-1) |
| `vllm:prompt_tokens_total` | counter | Cumulative prompt tokens processed |
| `vllm:generation_tokens_total` | counter | Cumulative output tokens generated |

**Histogram bucket structure** (vLLM defaults for latency metrics):

```
le="0.001", le="0.005", le="0.01", le="0.02", le="0.04", le="0.06",
le="0.08", le="0.1", le="0.25", le="0.5", le="0.75", le="1.0",
le="2.5", le="5.0", le="7.5", le="10.0", le="+Inf"
```

These bucket boundaries create resolution bands. Between 100ms and 250ms, for
example, all observations are grouped into a single bucket. AIPerf's polynomial
histogram algorithm (see `histogram_percentiles.py`) mitigates this by learning
per-bucket mean positions from scrape-to-scrape deltas, improving percentile
accuracy by approximately 2.5x over standard Prometheus linear interpolation.

### 2.3 GPU Telemetry (DCGM)

| Metric | Range | Update Rate | Relevance to Latency |
|--------|-------|-------------|---------------------|
| `gpu_utilization` | 0-100% | ~1s | Decode throughput ceiling |
| `sm_utilization` | 0-100% | ~1s | Streaming multiprocessor saturation |
| `mem_utilization` | 0-100% | ~1s | Memory bandwidth bottleneck |
| `gpu_power_usage` | 0-TDP (W) | ~1s | Thermal throttling indicator |
| `gpu_temperature` | 0-100+ (C) | ~1s | Approaching thermal limit |

GPU telemetry provides *context* for latency changes but not direct
decomposition. For example, a drop in SM utilization concurrent with rising
queue depth suggests the server is memory-bandwidth bound, not compute bound.

---

## 3. Latency Discrepancy Analysis

### 3.1 The Fundamental Decomposition

The gap between client-measured and server-measured end-to-end latency reveals
the total overhead introduced by the network and middleware stack:

```
network_overhead = client_e2e - server_e2e
```

More precisely, expanding both sides:

```
client_request_latency = (network_out + server_e2e + network_in) + client_overhead

where:
    network_out    = time for request to travel client → server
    server_e2e     = server-side processing (queue + prefill + decode)
    network_in     = time for response to travel server → client
    client_overhead = local processing (parsing, credit scheduling, etc.)
```

Since AIPerf uses streaming responses (SSE), `network_in` is not a single
value but is distributed across the token stream:

```
network_in_total = sum over tokens of (network_transit_per_token)
```

The first token's `network_in` contributes to TTFT discrepancy; subsequent
tokens' `network_in` contributes to ITL inflation.

### 3.2 Using HTTP Trace Metrics for Fine-Grained Decomposition

AIPerf's HTTP trace metrics (k6-compatible) provide sub-phase resolution.
For each request, the total time decomposes as:

```
http_req_total = http_req_blocked        (connection pool wait)
              + http_req_dns_lookup       (DNS resolution)
              + http_req_connecting       (TCP + TLS handshake)
              + http_req_sending          (request body transmission)
              + http_req_waiting          (TTFB — server processing time)
              + http_req_receiving        (response body transfer)
```

The **connection overhead** components (blocked + DNS + connecting) are pure
client/network overhead that the server never sees:

```
connection_overhead = http_req_blocked + http_req_dns_lookup + http_req_connecting
```

With connection reuse (HTTP keep-alive, typical in benchmarking), connection
overhead is zero for most requests after the first. The `http_req_connection_reused`
metric (0 or 1) tracks this.

The **transport overhead** is the time spent in request/response transmission
that is not server processing:

```
transport_overhead = http_req_sending + (http_req_receiving - server_decode_time)
```

Note: `http_req_receiving` includes both network transfer time AND server-side
decode time for streaming responses. Separating them requires the server-side
ITL histogram.

### 3.3 Quantifying the Network Overhead Budget

For a steady-state benchmark, we can compute aggregate statistics:

```python
def compute_network_overhead_budget(
    client_metrics: ColumnStore,
    server_histogram_percentiles: EstimatedPercentiles,
) -> NetworkOverheadBudget:
    """Compute the network overhead budget from matched client/server metrics.

    The budget decomposes the client-server latency gap into attributable
    components. Since we cannot join individual requests, we operate on
    distribution statistics (medians and percentiles).

    Returns:
        NetworkOverheadBudget with per-component overhead estimates.
    """
    # Client-side aggregates (from ColumnStore)
    client_p50_e2e = np.nanmedian(client_metrics.numeric("request_latency"))
    client_p50_ttft = np.nanmedian(client_metrics.numeric("time_to_first_token"))

    # HTTP trace aggregates
    p50_conn_overhead = np.nanmedian(
        client_metrics.numeric("http_req_connection_overhead")
    )
    p50_sending = np.nanmedian(client_metrics.numeric("http_req_sending"))
    p50_waiting = np.nanmedian(client_metrics.numeric("http_req_waiting"))
    p50_receiving = np.nanmedian(client_metrics.numeric("http_req_receiving"))

    # Server-side aggregates (from polynomial histogram estimation)
    server_p50_e2e = server_histogram_percentiles.p50_estimate  # seconds
    server_p50_e2e_ns = server_p50_e2e * 1e9  # convert to nanoseconds

    # Compute overhead components
    total_overhead = client_p50_e2e - server_p50_e2e_ns
    measured_overhead = p50_conn_overhead + p50_sending
    unexplained = total_overhead - measured_overhead

    return NetworkOverheadBudget(
        total_overhead_ns=total_overhead,
        connection_overhead_ns=p50_conn_overhead,
        sending_overhead_ns=p50_sending,
        receiving_includes_decode_ns=p50_receiving,
        unexplained_ns=unexplained,
        overhead_fraction=total_overhead / client_p50_e2e,
    )
```

### 3.4 Discrepancy Classification

The magnitude of the client-server discrepancy reveals the deployment topology:

| Overhead (% of client e2e) | Classification | Typical Cause |
|---------------------------|----------------|---------------|
| < 2% | Negligible | Localhost / same rack |
| 2-10% | Normal | Same datacenter, direct connection |
| 10-25% | Elevated | Cross-zone / load balancer in path |
| 25-50% | High | Proxy chain / WAN / TLS re-encryption |
| > 50% | Anomalous | Misconfiguration, routing issues, severe congestion |

The classification should be reported alongside latency results to help users
contextualize their measurements.

### 3.5 Formal Model: Request Latency Budget

Define the **latency budget** for a single request `i`:

```
L_client(i) = T_queue(i) + T_conn(i) + T_send(i) + T_server(i) + T_recv(i) + T_parse(i)

where:
    T_queue(i)  = credit scheduling + worker dispatch overhead
    T_conn(i)   = DNS + TCP + TLS (0 if connection reused)
    T_send(i)   = request body transmission to server
    T_server(i) = server-side end-to-end (queue + prefill + decode)
    T_recv(i)   = response byte transfer (includes decode wait for streaming)
    T_parse(i)  = SSE parsing + token extraction overhead (typically < 1ms)
```

The **measurable components** from AIPerf data:

| Component | Measurable? | AIPerf Metric |
|-----------|-------------|---------------|
| T_queue | Yes | `credit_drop_latency` or proposed `queue_wait_time` |
| T_conn | Yes | `http_req_connection_overhead` |
| T_send | Yes | `http_req_sending` |
| T_server | Estimated | Server histogram p50 (aggregate, not per-request) |
| T_recv | Partial | `http_req_receiving` minus server decode time |
| T_parse | No | Negligible; bundled into T_recv |

The gap `T_recv - server_decode_time` approximates the network transfer time
for the response stream, but this requires careful extraction since
`http_req_receiving` measures total time from first to last response chunk,
which includes both network transit and server-side inter-token gaps.

---

## 4. TTFT Decomposition

### 4.1 What TTFT Measures on Each Side

**Client-side TTFT** (from `TTFTMetric`):
```
TTFT_client = first_content_response.perf_ns - request.start_perf_ns
```

This includes everything from the HTTP request being sent to the first
meaningful token arriving at the client.

**Server-side TTFT** (from `vllm:time_to_first_token_seconds`):
```
TTFT_server = time_first_token_generated - time_request_received
```

This includes queue wait + prefill computation, but excludes network transit.

**The gap**:
```
TTFT_gap = TTFT_client - TTFT_server
         = network_out + network_in_first_token + client_overhead
```

### 4.2 Decomposing Client TTFT into Components

Using available AIPerf metrics, client TTFT decomposes as:

```
TTFT_client = stream_setup_latency + stream_prefill_latency

where:
    stream_setup_latency   = recv_start_perf_ns - start_perf_ns
                           = connection + send + HTTP response header arrival
    stream_prefill_latency = TTFT - stream_setup_latency
                           = time from HTTP 200 OK to first token content
```

Further decomposing `stream_setup_latency` using HTTP trace metrics:

```
stream_setup_latency ≈ http_req_connection_overhead + http_req_sending + server_queue_time
                      + server_stream_init_time + network_round_trip
```

The issue is that `stream_setup_latency` lumps together several unobservable
components. We can partially disentangle them:

```
stream_setup_latency = http_req_connection_overhead    (measurable, client-side)
                     + http_req_sending                (measurable, client-side)
                     + (server_queue + server_init)    (not directly measurable)
                     + network_rtt / 2                 (estimated)
```

### 4.3 Estimating Server Queue Time Contribution to TTFT

The server exposes `vllm:request_queue_time_seconds` as a histogram. Over
a benchmark run, the *change* in this histogram's cumulative counts represents
the distribution of queue times for requests processed during that period.

**Algorithm: Isolate queue contribution to TTFT**

```python
def estimate_queue_contribution_to_ttft(
    server_queue_time_percentiles: EstimatedPercentiles,
    server_ttft_percentiles: EstimatedPercentiles,
) -> QueueContribution:
    """Estimate what fraction of server-side TTFT is queueing vs compute.

    Uses the relationship:
        TTFT_server = queue_time + prefill_compute_time
        prefill_compute_time = TTFT_server - queue_time

    Since both are histogram distributions, we operate on percentiles.
    The subtraction of percentiles is approximate (not exact for non-independent
    random variables), but provides a useful first-order estimate.

    Returns:
        QueueContribution with per-percentile breakdown.
    """
    result = {}
    for pct_name in ["p50", "p75", "p90", "p95", "p99"]:
        ttft_val = getattr(server_ttft_percentiles, f"{pct_name}_estimate")
        queue_val = getattr(server_queue_time_percentiles, f"{pct_name}_estimate")

        if ttft_val is not None and queue_val is not None and ttft_val > 0:
            prefill_time = ttft_val - queue_val
            queue_fraction = queue_val / ttft_val
            result[pct_name] = {
                "ttft_server_s": ttft_val,
                "queue_time_s": queue_val,
                "prefill_compute_s": max(0, prefill_time),
                "queue_fraction": min(1.0, max(0.0, queue_fraction)),
            }

    return QueueContribution(percentile_breakdown=result)
```

**Important caveat**: Subtracting percentiles of two distributions does not
yield the percentile of their difference unless the random variables are
comonotonic (perfectly positively dependent). In practice, queue time and
prefill time are positively correlated (long prompts take longer to queue
AND longer to prefill), so the approximation is reasonable for central
percentiles but less reliable for tails (p99).

### 4.4 Full TTFT Decomposition Model

Combining client and server data:

```
TTFT_client = T_conn + T_send + T_network_out + T_queue_server + T_prefill + T_network_in_first

where:
    T_conn             = http_req_connection_overhead     [measured]
    T_send             = http_req_sending                 [measured]
    T_network_out      ≈ (http_req_waiting - TTFT_server) / 2  [estimated, assumes symmetric RTT]
    T_queue_server     = server queue_time histogram      [from server histogram]
    T_prefill          = TTFT_server - queue_time         [derived]
    T_network_in_first ≈ T_network_out                   [estimated, assumes symmetric RTT]
```

**Estimating one-way network latency**:

If the server does NOT have a queue (queue_time ~ 0) and the client HTTP trace
shows `http_req_waiting` as the time from send complete to first byte received,
then:

```
http_req_waiting ≈ T_network_out + T_server_processing + T_network_in_first_byte

If server TTFT is known (from histogram):
    T_network_rtt ≈ http_req_waiting - TTFT_server
    T_network_one_way ≈ T_network_rtt / 2
```

This estimate is only valid when `http_req_waiting > TTFT_server`. If the
inequality fails, it suggests clock skew, histogram aggregation effects, or
the HTTP 200 being sent before the first token.

### 4.5 Worked Example

Consider a benchmark run with:
- Client TTFT p50 = 85ms
- `stream_setup_latency` p50 = 35ms
- `stream_prefill_latency` p50 = 50ms
- `http_req_connection_overhead` p50 = 0.5ms (connection reuse active)
- `http_req_sending` p50 = 0.3ms
- `http_req_waiting` (TTFB) p50 = 34.2ms
- Server `vllm:time_to_first_token_seconds` p50 = 72ms
- Server `vllm:request_queue_time_seconds` p50 = 5ms

Decomposition:

```
TTFT breakdown (p50):
├── Connection overhead:     0.5ms  (0.6%)  [measured: http_req_connection_overhead]
├── Request send:            0.3ms  (0.4%)  [measured: http_req_sending]
├── Network outbound:       ~6.1ms  (7.2%)  [estimated: (http_req_waiting - TTFT_server) / 2 ... see note]
├── Server queue:            5.0ms  (5.9%)  [from server histogram]
├── Server prefill:         67.0ms  (78.8%) [TTFT_server - queue_time]
├── Network inbound (1st):  ~6.1ms  (7.2%)  [estimated: symmetric with outbound]
└── Total:                  85.0ms  (100%)
```

Note: The "network outbound" estimate here uses a different calculation. Since
`http_req_waiting` = 34.2ms measures from send-complete to first response byte,
and the server took 72ms for TTFT, the values seem inconsistent. This happens
because `http_req_waiting` measures to the *HTTP 200 OK header*, not the first
token. The actual network RTT is better estimated from:

```
network_rtt ≈ stream_setup_latency - http_req_connection_overhead - http_req_sending
            ≈ 35 - 0.5 - 0.3 = 34.2ms (this is http_req_waiting)

If server sends 200 OK before prefill completes:
    network_rtt ≈ stream_setup_latency - server_stream_init_time
```

The decomposition highlights that **server prefill dominates** (79% of TTFT),
making it the primary optimization target.

---

## 5. ITL Consistency Analysis

### 5.1 Client vs Server ITL Distributions

**Client-side ITL** is computed as an average:
```
ITL_client_avg = (request_latency - TTFT) / (output_sequence_length - 1)
```

This gives one ITL value per request, representing the average decode rate.
The distribution of `ITL_client_avg` across requests captures inter-request
variation (different prompt lengths, batching effects) but NOT intra-request
jitter.

**Server-side ITL** (`vllm:inter_token_latency_seconds`) is a histogram of
*individual* token-to-token intervals. This captures both inter-request and
intra-request variation.

These distributions are fundamentally different:

| Property | Client ITL | Server ITL |
|----------|-----------|------------|
| Granularity | Per-request average | Per-token individual |
| Includes network jitter | Yes | No |
| Includes batching effects | Yes (averaged) | Yes (per-token) |
| Sample size per request | 1 value | OSL - 1 values |
| Variance | Lower (averaging) | Higher (raw) |

### 5.2 Statistical Comparison of Distributions

To quantify how much the network inflates ITL, we compare the client and
server distributions using distribution distance measures:

#### Kolmogorov-Smirnov (KS) Test

The KS statistic measures the maximum absolute difference between two
empirical CDFs:

```
D_KS = sup_x |F_client(x) - F_server(x)|
```

For our use case, `F_client` is built from per-request average ITL values
stored in the ColumnStore, and `F_server` is reconstructed from the server
histogram using polynomial histogram percentile estimation.

```python
def ks_distance_client_server_itl(
    client_itl_ns: NDArray[np.float64],
    server_itl_percentiles: EstimatedPercentiles,
    n_synthetic: int = 10000,
) -> KSResult:
    """Compare client and server ITL distributions using KS test.

    Since server ITL is available only as histogram percentiles (not raw
    observations), we reconstruct a synthetic sample from the estimated
    percentiles using linear interpolation between quantile points.

    Args:
        client_itl_ns: Client-side ITL values in nanoseconds (one per request).
        server_itl_percentiles: Server-side ITL percentile estimates from
            polynomial histogram algorithm.
        n_synthetic: Number of synthetic observations to generate from
            server percentiles for KS comparison.

    Returns:
        KSResult with D statistic, p-value, and interpretation.
    """
    # Convert client ITL to seconds for comparison with server
    client_itl_s = client_itl_ns[~np.isnan(client_itl_ns)] / 1e9

    # Reconstruct server distribution from percentile estimates
    quantile_points = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    quantile_values = [
        server_itl_percentiles.p1_estimate,
        server_itl_percentiles.p5_estimate,
        server_itl_percentiles.p10_estimate,
        server_itl_percentiles.p25_estimate,
        server_itl_percentiles.p50_estimate,
        server_itl_percentiles.p75_estimate,
        server_itl_percentiles.p90_estimate,
        server_itl_percentiles.p95_estimate,
        server_itl_percentiles.p99_estimate,
    ]

    # Filter out None values
    valid = [(q, v) for q, v in zip(quantile_points, quantile_values) if v is not None]
    if len(valid) < 3:
        return KSResult(d_stat=float("nan"), p_value=float("nan"), interpretation="insufficient_data")

    qs, vs = zip(*valid)

    # Generate synthetic sample via inverse CDF interpolation
    uniform_samples = np.linspace(0.001, 0.999, n_synthetic)
    server_synthetic = np.interp(uniform_samples, qs, vs)

    # Two-sample KS test
    from scipy.stats import ks_2samp
    d_stat, p_value = ks_2samp(client_itl_s, server_synthetic)

    # Interpretation thresholds
    if d_stat < 0.05:
        interp = "distributions_match"  # Network adds negligible jitter
    elif d_stat < 0.15:
        interp = "minor_discrepancy"    # Moderate network effect
    elif d_stat < 0.30:
        interp = "significant_shift"    # Network or batching effects substantial
    else:
        interp = "major_divergence"     # Something beyond simple network overhead

    return KSResult(d_stat=d_stat, p_value=p_value, interpretation=interp)
```

#### Wasserstein Distance (Earth Mover's Distance)

The Wasserstein-1 distance measures the "work" needed to transform one
distribution into another:

```
W_1(F_client, F_server) = integral_0^inf |F_client(x) - F_server(x)| dx
```

Unlike KS, Wasserstein is sensitive to the *magnitude* of the shift, not just
the maximum difference. A uniform 2ms shift in ITL due to network overhead
produces a Wasserstein distance of 2ms regardless of distribution shape.

```python
def wasserstein_itl_overhead(
    client_itl_ns: NDArray[np.float64],
    server_itl_percentiles: EstimatedPercentiles,
) -> float:
    """Compute Wasserstein-1 distance between client and server ITL.

    The Wasserstein distance in seconds directly estimates the average
    per-token network overhead if the only difference between client
    and server ITL is an additive network delay.

    Returns:
        Wasserstein-1 distance in seconds. Positive means client ITL
        is shifted right (higher) than server ITL, as expected.
    """
    from scipy.stats import wasserstein_distance

    client_s = client_itl_ns[~np.isnan(client_itl_ns)] / 1e9
    server_synthetic = _reconstruct_server_distribution(server_itl_percentiles)

    return wasserstein_distance(client_s, server_synthetic)
```

**Interpretation**: If `W_1 = 0.003` (3ms), the average per-token network
overhead is approximately 3ms. For a 100-token response, this adds ~300ms
to total request latency beyond what the server reports.

### 5.3 ITL Jitter Analysis

Beyond the mean shift, network jitter introduces *variance* into the client-side
ITL distribution that is absent from the server side. Define:

```
Var(ITL_client) = Var(ITL_server) + Var(ITL_network) + 2*Cov(ITL_server, ITL_network)

Assuming independence (reasonable for network jitter):
    Var(ITL_network) ≈ Var(ITL_client) - Var(ITL_server)
```

The coefficient of variation (CV) of the network jitter component indicates
whether the network is a stable or unstable contributor:

```
CV_network = sqrt(Var(ITL_network)) / mean(ITL_network)
```

| CV_network | Interpretation |
|-----------|----------------|
| < 0.1 | Stable network, consistent overhead per token |
| 0.1-0.5 | Moderate jitter, typical for datacenter |
| 0.5-1.0 | High jitter, cross-zone or congested network |
| > 1.0 | Extreme jitter, likely pathological (buffering, retransmit) |

### 5.4 Per-Token vs Average ITL: The Averaging Problem

A critical subtlety: client ITL is an *average* over all tokens in a request,
while server ITL is a histogram of *individual* token intervals. For a request
with 100 tokens where 99 tokens decode in 10ms but the last token takes 100ms
due to a preemption:

- Client ITL = (99 * 10 + 100) / 99 = 11.01ms (average hides the spike)
- Server histogram includes the 100ms observation directly

AIPerf does have access to per-token timing via the `response_chunks` in
`TraceData`, which records `(timestamp_ns, size_bytes)` for each SSE chunk.
The RaggedSeries in ColumnStore stores variable-length per-request data.
Future work could extract per-chunk inter-arrival times to produce a
client-side ITL distribution with the same granularity as the server histogram.

```python
def extract_per_chunk_itl(
    trace_data: BaseTraceData,
) -> NDArray[np.float64]:
    """Extract inter-chunk arrival times from HTTP trace data.

    Each response_chunk is a (timestamp_ns, size_bytes) tuple.
    Inter-chunk intervals approximate per-token ITL (assuming one
    token per SSE chunk, which is typical for LLM streaming).

    Returns:
        Array of inter-chunk intervals in nanoseconds.
        Length = len(response_chunks) - 1.
    """
    if len(trace_data.response_chunks) < 2:
        return np.array([], dtype=np.float64)

    timestamps = np.array([ts for ts, _ in trace_data.response_chunks], dtype=np.float64)
    return np.diff(timestamps)
```

---

## 6. Temporal Correlation

### 6.1 Time Alignment Challenges

Client and server metrics exist on different time scales and sampling rates:

| Data Source | Time Reference | Sampling Rate | Granularity |
|-------------|---------------|---------------|-------------|
| Client metrics | `time.perf_counter_ns()` (monotonic) | Per-request | Sub-ms |
| Server histograms | Wall-clock (Prometheus scrape) | 1-15s intervals | Aggregated |
| GPU telemetry | Wall-clock (DCGM) | ~1s | Point-in-time |
| Sweep metrics | Derived from client | Continuous (sweep-line) | Sub-ms events |

**Clock domain mapping**: The ColumnStore uses `perf_counter_ns` for precise
interval measurement but also stores `generation_start_ns` (wall clock) for
alignment with external sources. The mapping is:

```
wall_clock_ns(event) = generation_start_ns + (event.perf_ns - first_request.perf_ns)
```

This assumes monotonic clock and wall clock advance at the same rate over the
benchmark duration, which is valid for typical benchmark durations (minutes to
hours) on modern hardware.

**Server timestamp alignment**: `ServerMetricsRecord.timestamp_ns` uses
`first_byte_ns` from the HTTP trace of the scrape request itself. This is
the server's response timestamp as seen by the client, already in the client's
wall-clock domain. No additional alignment is needed.

### 6.2 Nearest-Timestamp Matching

The simplest alignment strategy matches each client event to the nearest
server metric scrape in time:

```python
def nearest_timestamp_match(
    client_timestamps_ns: NDArray[np.float64],
    server_timestamps_ns: NDArray[np.int64],
    max_gap_ns: int = 5_000_000_000,  # 5 seconds
) -> NDArray[np.int64]:
    """Match each client timestamp to the nearest server scrape timestamp.

    Uses numpy searchsorted for O(n log m) matching where n = client events
    and m = server scrapes.

    Args:
        client_timestamps_ns: Client event wall-clock timestamps.
        server_timestamps_ns: Server scrape timestamps (sorted ascending).
        max_gap_ns: Maximum allowed gap. Events farther than this from any
            scrape get index -1 (unmatched).

    Returns:
        Array of server scrape indices. -1 for unmatched events.
    """
    # Find insertion points (index where client_ts would be inserted)
    idx = np.searchsorted(server_timestamps_ns, client_timestamps_ns)

    # For each client timestamp, compare distance to left and right neighbors
    n_server = len(server_timestamps_ns)
    result = np.full(len(client_timestamps_ns), -1, dtype=np.int64)

    for i, (client_ts, insert_idx) in enumerate(zip(client_timestamps_ns, idx)):
        candidates = []
        if insert_idx > 0:
            candidates.append((insert_idx - 1, abs(client_ts - server_timestamps_ns[insert_idx - 1])))
        if insert_idx < n_server:
            candidates.append((insert_idx, abs(client_ts - server_timestamps_ns[insert_idx])))

        if candidates:
            best_idx, best_gap = min(candidates, key=lambda x: x[1])
            if best_gap <= max_gap_ns:
                result[i] = best_idx

    return result
```

**Limitation**: This assigns each client event to a single server scrape,
which may not reflect the actual state when the request was processed. A request
starting at t=10.3s and ending at t=10.8s spans a single 1-second scrape
window, but its latency might be affected by conditions that changed *during*
the request.

### 6.3 Interpolation-Based Alignment

For gauge metrics (num_requests_running, gpu_utilization), linear interpolation
between scrape points provides a continuous time series:

```python
def interpolate_gauge_at_timestamps(
    gauge_timestamps_ns: NDArray[np.int64],
    gauge_values: NDArray[np.float64],
    query_timestamps_ns: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate gauge values at arbitrary query timestamps.

    Uses linear interpolation between consecutive gauge observations.
    Extrapolates using nearest value for timestamps outside the gauge range.

    This is appropriate for gauge metrics that represent instantaneous
    measurements (e.g., num_requests_running, gpu_utilization). It is NOT
    appropriate for counters or histograms.

    Args:
        gauge_timestamps_ns: Sorted gauge observation timestamps.
        gauge_values: Gauge values at each timestamp.
        query_timestamps_ns: Timestamps at which to estimate the gauge value.

    Returns:
        Interpolated gauge values at each query timestamp.
    """
    return np.interp(
        query_timestamps_ns,
        gauge_timestamps_ns.astype(np.float64),
        gauge_values,
    )
```

This allows computing, for example, the server queue depth at the moment each
client request started — a key input for understanding TTFT variation.

### 6.4 Cross-Correlation Analysis

Cross-correlation measures the similarity between two time series as a function
of temporal lag:

```
R_xy(tau) = E[(X(t) - mu_X)(Y(t + tau) - mu_Y)] / (sigma_X * sigma_Y)
```

For client-server analysis, we expect:
- **Client TTFT vs server queue depth**: Positive correlation at lag ~0 (higher
  queue depth when the request was sent correlates with higher TTFT).
- **Client throughput vs server num_requests_running**: Positive correlation at
  lag ~RTT (client throughput drives server concurrency with network delay).
- **Client p99 latency spikes vs GPU utilization drops**: Negative correlation
  (latency rises when GPU is less utilized, suggesting preemption or scheduling
  stalls).

```python
def windowed_cross_correlation(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    timestamps_x: NDArray[np.float64],
    timestamps_y: NDArray[np.float64],
    window_size_ns: int = 1_000_000_000,  # 1 second
    max_lag_windows: int = 10,
) -> CrossCorrelationResult:
    """Compute cross-correlation between two irregularly-sampled time series.

    Bins both series into fixed-width windows, then computes standard
    cross-correlation on the binned averages.

    Steps:
    1. Determine common time range.
    2. Bin both series into windows of window_size_ns.
    3. Compute per-window averages (NaN for empty windows).
    4. Apply np.correlate on the valid-only windows.

    Args:
        x: First time series values (e.g., client TTFT in ns).
        y: Second time series values (e.g., server queue depth).
        timestamps_x: Timestamps for x values.
        timestamps_y: Timestamps for y values.
        window_size_ns: Width of each time bin in nanoseconds.
        max_lag_windows: Maximum lag to compute (in window units).

    Returns:
        CrossCorrelationResult with lag values and correlation coefficients.
    """
    # Determine common time range
    t_start = max(np.min(timestamps_x), np.min(timestamps_y))
    t_end = min(np.max(timestamps_x), np.max(timestamps_y))

    n_windows = int((t_end - t_start) / window_size_ns) + 1
    if n_windows < 2 * max_lag_windows + 1:
        return CrossCorrelationResult(
            lags=np.array([]),
            correlations=np.array([]),
            peak_lag=0,
            peak_correlation=float("nan"),
        )

    # Bin x
    x_binned = np.full(n_windows, np.nan)
    x_bin_indices = ((timestamps_x - t_start) / window_size_ns).astype(int)
    x_bin_indices = np.clip(x_bin_indices, 0, n_windows - 1)
    for w in range(n_windows):
        mask = x_bin_indices == w
        if np.any(mask):
            x_binned[w] = np.mean(x[mask])

    # Bin y
    y_binned = np.full(n_windows, np.nan)
    y_bin_indices = ((timestamps_y - t_start) / window_size_ns).astype(int)
    y_bin_indices = np.clip(y_bin_indices, 0, n_windows - 1)
    for w in range(n_windows):
        mask = y_bin_indices == w
        if np.any(mask):
            y_binned[w] = np.mean(y[mask])

    # Interpolate NaN gaps (forward-fill then back-fill)
    x_filled = _fill_nan(x_binned)
    y_filled = _fill_nan(y_binned)

    # Normalize
    x_norm = (x_filled - np.mean(x_filled)) / (np.std(x_filled) + 1e-12)
    y_norm = (y_filled - np.mean(y_filled)) / (np.std(y_filled) + 1e-12)

    # Cross-correlation via numpy
    lags = np.arange(-max_lag_windows, max_lag_windows + 1)
    correlations = np.array([
        np.mean(x_norm[max(0, lag):min(n_windows, n_windows + lag)]
                * y_norm[max(0, -lag):min(n_windows, n_windows - lag)])
        for lag in lags
    ])

    peak_idx = np.argmax(np.abs(correlations))
    return CrossCorrelationResult(
        lags=lags * window_size_ns,  # Convert to nanoseconds
        correlations=correlations,
        peak_lag=int(lags[peak_idx] * window_size_ns),
        peak_correlation=float(correlations[peak_idx]),
    )
```

### 6.5 Lag Analysis

The **peak lag** in the cross-correlation function has physical meaning:

```
peak_lag(client_metric, server_metric) ≈ network_RTT + processing_pipeline_delay
```

For example, if cross-correlation between `effective_concurrency` (client-side
sweep metric) and `vllm:num_requests_running` (server gauge) peaks at lag =
+3ms, it means the server's running count reflects the client's concurrency
with a ~3ms delay — which is approximately one network round-trip.

**Expected lag relationships**:

| Client Metric | Server Metric | Expected Lag | Meaning |
|--------------|---------------|-------------|---------|
| effective_concurrency | num_requests_running | +RTT | Server concurrency trails client |
| request throughput | generation_tokens_total rate | +RTT + prefill_time | Server output trails client input |
| p99 latency spike | num_requests_waiting spike | -RTT | Queue builds before latency rises |
| throughput drop | gpu_utilization drop | ~0 | Both respond to same event |

### 6.6 Granger Causality Testing

Beyond correlation, Granger causality tests whether one time series *predicts*
another. If server `num_requests_waiting` Granger-causes client TTFT, it means
past queue depth values improve the prediction of future TTFT values.

```python
def granger_causality_test(
    x_binned: NDArray[np.float64],  # "effect" series (e.g., client TTFT)
    y_binned: NDArray[np.float64],  # "cause" series (e.g., server queue depth)
    max_lag: int = 5,
    significance: float = 0.05,
) -> GrangerResult:
    """Test if y Granger-causes x using F-test on nested OLS models.

    Compares:
    - Restricted model: x(t) = a0 + a1*x(t-1) + ... + ap*x(t-p) + e(t)
    - Unrestricted model: x(t) = a0 + a1*x(t-1) + ... + b1*y(t-1) + ... + e(t)

    If the unrestricted model has significantly lower residual sum of squares,
    y Granger-causes x (past y values help predict x beyond what past x alone
    provides).

    This is useful for answering: "Does server queue depth predict client
    latency spikes?" or "Does GPU utilization predict throughput drops?"

    Args:
        x_binned: Effect time series (regularly spaced, NaN-free).
        y_binned: Cause time series (same length and spacing as x).
        max_lag: Maximum lag order to test.
        significance: P-value threshold for significance.

    Returns:
        GrangerResult with F-statistic, p-value, and best lag order.
    """
    n = len(x_binned)
    best_p_value = 1.0
    best_lag = 0
    best_f_stat = 0.0

    for p in range(1, max_lag + 1):
        if n <= 2 * p + 1:
            continue

        # Build lagged matrices
        X_restricted = np.column_stack([x_binned[p - i - 1 : n - i - 1] for i in range(p)])
        X_unrestricted = np.column_stack([
            *[x_binned[p - i - 1 : n - i - 1] for i in range(p)],
            *[y_binned[p - i - 1 : n - i - 1] for i in range(p)],
        ])

        y_target = x_binned[p:]

        # Add constant
        ones = np.ones((len(y_target), 1))
        X_r = np.hstack([ones, X_restricted[:len(y_target)]])
        X_u = np.hstack([ones, X_unrestricted[:len(y_target)]])

        # OLS fits
        rss_r = _ols_rss(X_r, y_target)
        rss_u = _ols_rss(X_u, y_target)

        # F-test: F = ((RSS_r - RSS_u) / p) / (RSS_u / (n - 2p - 1))
        df_num = p
        df_den = len(y_target) - 2 * p - 1
        if df_den <= 0 or rss_u <= 0:
            continue

        f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
        from scipy.stats import f as f_dist
        p_value = 1.0 - f_dist.cdf(f_stat, df_num, df_den)

        if p_value < best_p_value:
            best_p_value = p_value
            best_lag = p
            best_f_stat = f_stat

    return GrangerResult(
        f_statistic=best_f_stat,
        p_value=best_p_value,
        best_lag=best_lag,
        significant=best_p_value < significance,
    )
```

---

## 7. Causal Inference and Root Cause Attribution

### 7.1 Latency Spike Root Cause Framework

When a latency spike is detected (e.g., client request_latency > 3x median),
the following diagnostic tree identifies the most likely root cause:

```
Latency spike detected (request_latency > 3 * p50)
│
├── Is TTFT elevated? (TTFT > 3 * TTFT_p50)
│   ├── YES → Prefill or queue problem
│   │   ├── Is server queue depth elevated at request start time?
│   │   │   ├── YES → Server queue saturation
│   │   │   │   └── Action: Reduce concurrency or scale server
│   │   │   └── NO → Prefill computation bottleneck
│   │   │       ├── Is input_sequence_length above p90?
│   │   │       │   ├── YES → Long prompt causing prefill latency
│   │   │       │   └── NO → KV cache pressure or preemption
│   │   │       │       └── Check vllm:num_preemptions_total delta
│   │   └── Is http_req_connection_overhead elevated?
│   │       ├── YES → Connection pool exhaustion
│   │       └── NO → Network congestion on outbound path
│   │
│   └── NO → Decode or network problem
│       ├── Is ITL elevated? (per-request ITL > 3 * ITL_p50)
│       │   ├── YES → Decode bottleneck
│       │   │   ├── Is gpu_utilization < 80%?
│       │   │   │   ├── YES → Memory bandwidth bound or scheduling stall
│       │   │   │   └── NO → Compute bound with high batch size
│       │   │   └── Is output_sequence_length above p90?
│       │   │       ├── YES → Long generation amplifying ITL overhead
│       │   │       └── NO → Batch contention from concurrent requests
│       │   │
│       │   └── NO → Network receiving bottleneck
│       │       └── Is http_req_receiving elevated?
│       │           ├── YES → Network congestion on return path
│       │           └── NO → Unexplained — check proxy/load balancer
│       │
│       └── (Only possible if request_latency is elevated but neither TTFT nor ITL)
│           └── Check for HTTP error + retry overhead
```

### 7.2 Automated Attribution Score

For each latency spike, compute an attribution score for each possible cause:

```python
@dataclass
class SpikeAttribution:
    """Attribution scores for a latency spike root cause."""
    queue_score: float       # 0-1, how much queueing explains the spike
    prefill_score: float     # 0-1, how much prefill explains the spike
    decode_score: float      # 0-1, how much decode explains the spike
    network_score: float     # 0-1, how much network explains the spike
    scheduling_score: float  # 0-1, how much client scheduling explains the spike
    unexplained: float       # 0-1, residual


def attribute_latency_spike(
    spike_request_idx: int,
    column_store: ColumnStore,
    server_queue_depth: NDArray[np.float64],
    server_queue_timestamps: NDArray[np.int64],
) -> SpikeAttribution:
    """Attribute a latency spike to its constituent causes.

    Uses the additive decomposition:
        request_latency = queue_wait + TTFT + decode_time
        TTFT = connection_overhead + network + server_queue + server_prefill
        decode_time = (OSL - 1) * ITL

    For each component, the attribution score is the fraction of the
    *excess* latency (above median) attributable to that component's
    excess above its own median.

    Args:
        spike_request_idx: Index of the spike request in ColumnStore.
        column_store: ColumnStore with all client metrics.
        server_queue_depth: Interpolated server queue depth at request times.
        server_queue_timestamps: Timestamps for server queue depth.

    Returns:
        SpikeAttribution with normalized scores summing to ~1.0.
    """
    # Get the spike request's metrics
    latency = column_store.numeric("request_latency")[spike_request_idx]
    ttft = column_store.numeric("time_to_first_token")[spike_request_idx]
    itl = column_store.numeric("inter_token_latency")[spike_request_idx]
    osl = column_store.numeric("output_sequence_length")[spike_request_idx]
    conn_overhead = column_store.numeric("http_req_connection_overhead")[spike_request_idx]

    # Median baselines
    med_latency = np.nanmedian(column_store.numeric("request_latency"))
    med_ttft = np.nanmedian(column_store.numeric("time_to_first_token"))
    med_itl = np.nanmedian(column_store.numeric("inter_token_latency"))
    med_conn = np.nanmedian(column_store.numeric("http_req_connection_overhead"))

    # Excess above baseline
    excess_total = max(0, latency - med_latency)
    if excess_total <= 0:
        return SpikeAttribution(0, 0, 0, 0, 0, 0)

    excess_ttft = max(0, ttft - med_ttft)
    excess_itl = max(0, itl - med_itl) * max(1, osl - 1)
    excess_conn = max(0, conn_overhead - med_conn)

    # Normalize to attribution scores
    total_attributable = excess_ttft + excess_itl + excess_conn
    if total_attributable <= 0:
        return SpikeAttribution(0, 0, 0, 0, 0, 1.0)

    scale = min(1.0, excess_total / total_attributable)

    return SpikeAttribution(
        queue_score=0.0,  # Requires queue_wait_time metric (proposed)
        prefill_score=(excess_ttft / total_attributable) * scale,
        decode_score=(excess_itl / total_attributable) * scale,
        network_score=(excess_conn / total_attributable) * scale,
        scheduling_score=0.0,  # credit_drop_latency excess
        unexplained=1.0 - scale,
    )
```

### 7.3 Little's Law Cross-Validation

As described in the existing proposal
(`proposal-coordinated-omission-and-latency-decomposition.md`), Little's Law
provides a consistency check between client-measured concurrency, throughput,
and latency:

```
L = lambda * W

where:
    L = effective_concurrency.avg       (from sweep-line)
    lambda = request_throughput          (from sweep-line)
    W = request_latency.avg / 1e9       (mean latency in seconds)
```

This can be extended to validate client-server consistency:

```
L_client = lambda_client * W_client     (all from client metrics)
L_server = lambda_server * W_server     (all from server metrics)

Discrepancy = |L_client - L_server| / L_client
```

If `L_client >> L_server`, requests are spending significant time in the
network (the client sees higher "in-flight" count than the server). If
`L_server >> L_client`, the server is processing requests from other clients
(shared server scenario).

### 7.4 Information-Theoretic Correlation: Mutual Information

For non-linear relationships between metrics (e.g., latency jumping
discontinuously when queue depth exceeds a threshold), mutual information
captures dependencies that Pearson/Spearman correlation miss:

```
I(X; Y) = sum_x sum_y p(x, y) * log(p(x, y) / (p(x) * p(y)))
```

```python
def mutual_information_binned(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    n_bins: int = 20,
) -> float:
    """Estimate mutual information between two continuous variables.

    Uses histogram-based binning to estimate the joint and marginal
    probability distributions. The bin count is a trade-off between
    resolution and estimation noise (bias-variance).

    For LLM benchmarking, typical use cases:
    - MI(server_queue_depth, client_TTFT): Should be high
    - MI(gpu_utilization, client_ITL): Non-trivial if compute-bound
    - MI(KV_cache_usage, client_latency): Threshold effects

    Args:
        x: First variable (e.g., server queue depth, binned by time window).
        y: Second variable (e.g., client TTFT, binned by same time window).
        n_bins: Number of histogram bins per dimension.

    Returns:
        Mutual information in nats (natural log). Higher = stronger
        dependence (0 = independent).
    """
    # Remove NaN pairs
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    if len(x) < n_bins * 2:
        return 0.0

    # 2D histogram for joint distribution
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
    hist_2d = hist_2d / hist_2d.sum()  # Normalize to probabilities

    # Marginals
    p_x = hist_2d.sum(axis=1)
    p_y = hist_2d.sum(axis=0)

    # MI = sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if hist_2d[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (p_x[i] * p_y[j]))

    return mi
```

---

## 8. Practical Implementation in AIPerf

### 8.1 Architecture Overview

The correlation analysis fits into AIPerf's existing accumulator architecture:

```
RecordProcessor (per-request)          ServerMetricsManager (periodic)
       │                                        │
       ▼                                        ▼
  MetricRecordsData                     ServerMetricsRecord
  (per-request metrics)                 (per-scrape histograms)
       │                                        │
       ▼                                        ▼
  MetricsAccumulator                    ServerMetricsAccumulator
  (ColumnStore)                         (ServerMetricsHierarchy)
       │                                        │
       └────────────┬───────────────────────────┘
                    │
                    ▼
          CorrelationAnalyzer (NEW)
          (reads from both accumulators at export_results time)
                    │
                    ▼
          CorrelationResults
          (JSON export + console warnings)
```

### 8.2 Data Access Patterns

The `CorrelationAnalyzer` needs read access to both accumulators. In AIPerf's
architecture, accumulators are plugins managed by `RecordsManager`. The analyzer
would need access during the `export_results` phase.

**Option A: SummaryContext passthrough**

The existing `SummaryContext` dataclass already bundles accumulator references:

```python
@dataclass
class SummaryContext:
    accumulators: dict[str, Any]
    accumulator_outputs: dict[str, Any]
    start_ns: int
    end_ns: int
    cancelled: bool
```

The `CorrelationAnalyzer` could be implemented as a post-export step that
receives the `SummaryContext` with both `MetricsAccumulator` and
`ServerMetricsAccumulator` outputs.

**Option B: Standalone Analyzer Plugin**

Register a new `AnalyzerProtocol` (similar to `SteadyStateAnalyzer`) that
receives both accumulator outputs and produces `CorrelationResults`.

### 8.3 Core Data Structure

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyDecomposition:
    """Per-percentile breakdown of client TTFT into components."""

    percentile: str  # e.g., "p50", "p95", "p99"
    client_ttft_ms: float
    connection_overhead_ms: float
    request_send_ms: float
    estimated_network_rtt_ms: float
    server_queue_time_ms: float | None  # None if server metrics unavailable
    server_prefill_time_ms: float | None
    unexplained_ms: float


@dataclass
class DistributionComparison:
    """Statistical comparison of a client vs server metric distribution."""

    metric_name: str  # e.g., "itl", "ttft", "e2e_latency"
    ks_d_statistic: float
    ks_p_value: float
    wasserstein_distance_ms: float
    mean_shift_ms: float  # client_mean - server_mean
    variance_ratio: float  # Var(client) / Var(server)
    interpretation: str


@dataclass
class TemporalCorrelation:
    """Cross-correlation result between a client and server time series."""

    client_metric: str
    server_metric: str
    peak_lag_ms: float  # Positive = server lags client
    peak_correlation: float  # -1 to 1
    granger_p_value: float | None  # None if insufficient data
    granger_significant: bool


@dataclass
class CorrelationResults:
    """Complete client-server correlation analysis results."""

    # Overall discrepancy
    network_overhead_pct: float  # (client_e2e - server_e2e) / client_e2e * 100
    network_overhead_p50_ms: float
    network_overhead_p99_ms: float

    # TTFT decomposition
    ttft_decomposition: list[LatencyDecomposition]

    # Distribution comparisons
    distribution_comparisons: list[DistributionComparison]

    # Temporal correlations
    temporal_correlations: list[TemporalCorrelation]

    # Little's Law validation
    littles_law_L: float
    littles_law_lambda_W: float
    littles_law_discrepancy_pct: float

    # Diagnostics
    data_quality_warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Export as structured JSON for the results file."""
        return {
            "network_overhead": {
                "overhead_pct": round(self.network_overhead_pct, 2),
                "p50_ms": round(self.network_overhead_p50_ms, 3),
                "p99_ms": round(self.network_overhead_p99_ms, 3),
            },
            "ttft_decomposition": [
                {
                    "percentile": d.percentile,
                    "client_ttft_ms": round(d.client_ttft_ms, 3),
                    "connection_overhead_ms": round(d.connection_overhead_ms, 3),
                    "request_send_ms": round(d.request_send_ms, 3),
                    "estimated_network_rtt_ms": round(d.estimated_network_rtt_ms, 3),
                    "server_queue_time_ms": (
                        round(d.server_queue_time_ms, 3) if d.server_queue_time_ms is not None else None
                    ),
                    "server_prefill_time_ms": (
                        round(d.server_prefill_time_ms, 3) if d.server_prefill_time_ms is not None else None
                    ),
                    "unexplained_ms": round(d.unexplained_ms, 3),
                }
                for d in self.ttft_decomposition
            ],
            "distribution_comparisons": [
                {
                    "metric": dc.metric_name,
                    "ks_d": round(dc.ks_d_statistic, 4),
                    "wasserstein_ms": round(dc.wasserstein_distance_ms, 3),
                    "mean_shift_ms": round(dc.mean_shift_ms, 3),
                    "interpretation": dc.interpretation,
                }
                for dc in self.distribution_comparisons
            ],
            "temporal_correlations": [
                {
                    "client": tc.client_metric,
                    "server": tc.server_metric,
                    "peak_lag_ms": round(tc.peak_lag_ms, 3),
                    "peak_correlation": round(tc.peak_correlation, 4),
                    "granger_significant": tc.granger_significant,
                }
                for tc in self.temporal_correlations
            ],
            "littles_law": {
                "L": round(self.littles_law_L, 2),
                "lambda_W": round(self.littles_law_lambda_W, 2),
                "discrepancy_pct": round(self.littles_law_discrepancy_pct, 2),
                "status": "pass" if self.littles_law_discrepancy_pct < 10.0 else "warn",
            },
            "warnings": self.data_quality_warnings,
        }
```

### 8.4 Implementation Algorithm

```python
async def run_correlation_analysis(
    metrics_accumulator: MetricsAccumulator,
    server_metrics_accumulator: ServerMetricsAccumulator | None,
    gpu_telemetry_accumulator: Any | None,  # GPUTelemetryAccumulator
    ctx: ExportContext,
) -> CorrelationResults | None:
    """Run the full client-server correlation analysis pipeline.

    This is the main entry point. It orchestrates:
    1. Network overhead budget computation
    2. TTFT decomposition across percentiles
    3. Distribution comparisons (ITL, TTFT, e2e)
    4. Temporal cross-correlation analysis
    5. Little's Law validation

    Args:
        metrics_accumulator: Client-side metrics in ColumnStore.
        server_metrics_accumulator: Server-side Prometheus metrics.
            If None, only client-side analysis is performed.
        gpu_telemetry_accumulator: GPU telemetry data.
            If None, GPU correlation is skipped.
        ctx: Export context with time range and error summary.

    Returns:
        CorrelationResults if analysis succeeds, None if insufficient data.
    """
    cs = metrics_accumulator.column_store
    warnings: list[str] = []

    # Minimum data threshold
    if cs.count < 30:
        return None

    # === Phase 1: Client-only analysis ===

    # Little's Law from sweep metrics (always available)
    littles_law = _compute_littles_law(metrics_accumulator, ctx)

    # === Phase 2: Client-server correlation (if server metrics available) ===

    ttft_decomposition: list[LatencyDecomposition] = []
    dist_comparisons: list[DistributionComparison] = []
    temporal_corrs: list[TemporalCorrelation] = []

    network_overhead_pct = 0.0
    network_overhead_p50 = 0.0
    network_overhead_p99 = 0.0

    if server_metrics_accumulator is not None:
        hierarchy = server_metrics_accumulator.get_hierarchy_for_export()

        # Extract server histogram percentiles for key metrics
        server_e2e_pcts = _extract_histogram_percentiles(
            hierarchy, "vllm:e2e_request_latency_seconds", ctx
        )
        server_ttft_pcts = _extract_histogram_percentiles(
            hierarchy, "vllm:time_to_first_token_seconds", ctx
        )
        server_itl_pcts = _extract_histogram_percentiles(
            hierarchy, "vllm:inter_token_latency_seconds", ctx
        )
        server_queue_pcts = _extract_histogram_percentiles(
            hierarchy, "vllm:request_queue_time_seconds", ctx
        )

        # Network overhead
        if server_e2e_pcts is not None:
            network_overhead_p50, network_overhead_p99, network_overhead_pct = (
                _compute_network_overhead(cs, server_e2e_pcts)
            )

        # TTFT decomposition
        if server_ttft_pcts is not None:
            ttft_decomposition = _decompose_ttft(cs, server_ttft_pcts, server_queue_pcts)

        # Distribution comparisons
        if server_itl_pcts is not None:
            itl_comparison = _compare_itl_distributions(cs, server_itl_pcts)
            dist_comparisons.append(itl_comparison)

        if server_ttft_pcts is not None:
            ttft_comparison = _compare_ttft_distributions(cs, server_ttft_pcts)
            dist_comparisons.append(ttft_comparison)

        # Temporal correlations
        temporal_corrs = _compute_temporal_correlations(cs, hierarchy, ctx)

    else:
        warnings.append(
            "Server metrics not available. Correlation analysis limited to "
            "client-side decomposition. Enable --server-metrics for full analysis."
        )

    return CorrelationResults(
        network_overhead_pct=network_overhead_pct,
        network_overhead_p50_ms=network_overhead_p50,
        network_overhead_p99_ms=network_overhead_p99,
        ttft_decomposition=ttft_decomposition,
        distribution_comparisons=dist_comparisons,
        temporal_correlations=temporal_corrs,
        littles_law_L=littles_law.L,
        littles_law_lambda_W=littles_law.lambda_W,
        littles_law_discrepancy_pct=littles_law.discrepancy_pct,
        data_quality_warnings=warnings,
    )
```

### 8.5 Server Histogram Percentile Extraction

Extracting percentiles from the server histogram requires computing deltas
over the benchmark period and applying the polynomial histogram algorithm:

```python
def _extract_histogram_percentiles(
    hierarchy: ServerMetricsHierarchy,
    metric_name: str,
    ctx: ExportContext,
) -> EstimatedPercentiles | None:
    """Extract percentile estimates from a server-side Prometheus histogram.

    Uses the polynomial histogram algorithm (histogram_percentiles.py) for
    improved accuracy over standard Prometheus linear interpolation.

    Steps:
    1. Find the metric in the hierarchy (across all endpoints).
    2. Get cumulative bucket counts at start and end of profiling period.
    3. Compute deltas (observations during profiling only).
    4. Apply learned bucket statistics for polynomial estimation.

    Args:
        hierarchy: Server metrics hierarchical storage.
        metric_name: Prometheus metric name (e.g., "vllm:e2e_request_latency_seconds").
        ctx: Export context with profiling time range.

    Returns:
        EstimatedPercentiles if the metric is found and has sufficient data.
        None if the metric is not available or has no observations in range.
    """
    from aiperf.server_metrics.histogram_percentiles import (
        accumulate_bucket_statistics,
        compute_estimated_percentiles,
    )

    for endpoint_url, time_series in hierarchy.endpoints.items():
        for metric_key, metric_entry in time_series.metrics.items():
            if metric_key.name != metric_name:
                continue
            if metric_entry.metric_type != PrometheusMetricType.HISTOGRAM:
                continue

            data = metric_entry.data
            # HistogramTimeSeries has: timestamps_ns, sums, counts, bucket_les, bucket_counts
            if len(data.timestamps_ns) < 2:
                return None

            # Find indices for profiling period
            start_ns = ctx.start_ns or 0
            end_ns = ctx.end_ns or data.timestamps_ns[-1]

            start_idx = np.searchsorted(data.timestamps_ns, start_ns)
            end_idx = np.searchsorted(data.timestamps_ns, end_ns, side="right") - 1

            if end_idx <= start_idx:
                return None

            # Compute deltas over profiling period
            bucket_deltas = {}
            for b, le in enumerate(data.bucket_les):
                delta = data.bucket_counts[end_idx, b] - data.bucket_counts[start_idx, b]
                bucket_deltas[le] = max(0, delta)

            total_sum = data.sums[end_idx] - data.sums[start_idx]
            total_count = int(data.counts[end_idx] - data.counts[start_idx])

            if total_count <= 0:
                return None

            # Learn bucket statistics from the profiling period
            bucket_stats = accumulate_bucket_statistics(
                data.sums, data.counts, data.bucket_les,
                data.bucket_counts, start_idx=start_idx,
            )

            return compute_estimated_percentiles(
                bucket_deltas, bucket_stats, total_sum, total_count,
            )

    return None
```

### 8.6 Configuration

The correlation analysis should be opt-in, with sensible defaults:

```python
@dataclass
class CorrelationConfig:
    """Configuration for client-server correlation analysis."""

    enabled: bool = False  # Opt-in via --correlation-analysis or AIPERF_CORRELATION_ANALYSIS
    temporal_window_ns: int = 1_000_000_000  # 1s binning for temporal analysis
    max_temporal_lag_windows: int = 10  # Cross-correlation up to 10 windows
    granger_max_lag: int = 5  # Granger causality max lag order
    granger_significance: float = 0.05  # P-value threshold
    min_requests: int = 30  # Minimum requests for meaningful analysis
    distribution_n_synthetic: int = 10000  # Synthetic samples for KS test
```

CLI integration:

```
--correlation-analysis              Enable client-server latency correlation
--correlation-temporal-window 2.0   Temporal binning window in seconds
```

Environment variable:

```
AIPERF_CORRELATION_ANALYSIS=true
AIPERF_CORRELATION_TEMPORAL_WINDOW=2.0
```

### 8.7 Dependencies

The correlation analysis requires `scipy` for:
- `scipy.stats.ks_2samp` — Two-sample Kolmogorov-Smirnov test
- `scipy.stats.wasserstein_distance` — Earth mover's distance
- `scipy.stats.f` — F-distribution for Granger causality

Currently, `scipy` is NOT a required dependency of AIPerf. The implementation
should handle its absence gracefully:

```python
def _check_scipy_available() -> bool:
    """Check if scipy is installed for advanced correlation analysis."""
    try:
        import scipy.stats  # noqa: F401
        return True
    except ImportError:
        return False
```

If scipy is not available, the analysis should fall back to numpy-only methods:
- Skip KS and Wasserstein tests (report as unavailable)
- Skip Granger causality (report as unavailable)
- Still perform: network overhead budget, TTFT decomposition (arithmetic),
  Pearson/Spearman correlation (numpy-only), Little's Law validation

### 8.8 Performance Considerations

The correlation analysis runs once at the end of the benchmark (during
`export_results`), not on the hot path. However, some operations can be
expensive:

| Operation | Complexity | Typical Time (10K requests) |
|-----------|-----------|---------------------------|
| Network overhead (median) | O(n log n) | ~1ms |
| TTFT decomposition | O(n log n) | ~2ms |
| KS test | O(n log n) | ~5ms |
| Wasserstein distance | O(n log n) | ~5ms |
| Cross-correlation (10 pairs) | O(n * max_lag) | ~10ms |
| Granger causality (10 pairs) | O(n * lag^2) | ~50ms |
| Mutual information | O(n * bins^2) | ~5ms |

Total: approximately 80ms for 10,000 requests — negligible compared to the
benchmark duration.

---

## 9. Visualization Approaches

### 9.1 Latency Waterfall Chart

A stacked bar chart showing the latency budget decomposition at each percentile:

```
 p99 │████████████████████████████████████████████████│ 320ms
     │▓▓▓▓│░░│████████████████████│████████│░░░░│████│
     │conn │net│    server queue   │prefill │net │unexpl│

 p95 │████████████████████████████████████│ 180ms
     │▓│░│████████████│████████████│░│████│

 p50 │██████████████████████│ 85ms
     │░│█████│████████████│░│██│

       0    20    40    60    80   100  120  140  160  180  200  220  240  260  280  300  320
                                    Time (ms)

     Legend: ▓ = connection overhead, ░ = network, █ = server queue, █ = prefill, █ = decode
```

AIPerf already has a plotting subsystem (`src/aiperf/plot/`). The waterfall
chart would be a new `PlotType.LATENCY_WATERFALL` using the existing
`PlotConfig` infrastructure.

### 9.2 Dual-Axis Time Series

Plot client latency (left Y-axis) and server queue depth (right Y-axis) on
the same time axis to visually identify correlated events:

```
TTFT (ms)                                                    Queue Depth
  300 │                                                        │ 15
      │                    ╱╲                                   │
  250 │                   ╱  ╲        ╱╲                        │ 12
      │                  ╱    ╲      ╱  ╲                       │
  200 │                 ╱      ╲    ╱    ╲                      │ 10
      │   client TTFT──╱        ╲──╱      ╲                    │
  150 │╱╲─────────────╱                    ╲───────────────     │ 8
      │   ╲          ╱                      ╲                   │
  100 │    ╲────────╱                        ╲─────────────     │ 5
      │                                                        │
   50 │    server queue ────────────╱╲──────╱╲──────────       │ 2
      │────────────────────────────╱  ╲────╱  ╲───────────     │
    0 │                                                        │ 0
      └────────────────────────────────────────────────────────┘
        0    10    20    30    40    50    60    70    80    90  100
                               Time (seconds)
```

### 9.3 Distribution Overlay

Overlay client ITL distribution (histogram) with server ITL distribution
(estimated from histogram percentiles) to visualize the shift:

```
Frequency
  │
  │       ╱╲ server ITL
  │      ╱  ╲           ╱╲ client ITL
  │     ╱    ╲         ╱  ╲
  │    ╱      ╲       ╱    ╲
  │   ╱        ╲     ╱      ╲
  │  ╱          ╲   ╱        ╲
  │ ╱            ╲─╱          ╲
  │╱              ╳            ╲
  │             ╱  ╲            ╲
  └─────────────────────────────────────────
    0    5    10   15   20   25   30   35   40
                  ITL (ms)

    ← Wasserstein distance = 3.2ms →
```

### 9.4 Correlation Heatmap

A heatmap showing Pearson correlation between all pairs of client and server
metrics:

```
                    server:queue  server:running  server:waiting  gpu:util  gpu:power
client:TTFT            0.82          0.65           0.78          -0.12      0.31
client:ITL             0.35          0.71           0.42          -0.45      0.52
client:latency         0.74          0.78           0.69          -0.28      0.44
client:throughput     -0.41         -0.23          -0.55           0.67     -0.18
client:concurrency     0.12          0.91           0.22           0.34      0.28
```

Color scale: deep blue (-1) through white (0) to deep red (+1).

---

## 10. Academic References and Theoretical Foundations

### 10.1 Latency Decomposition in Distributed Systems

**Sambasivan et al., "So, you want to trace your distributed system? Key
design insights from years of practical experience" (2014)**. HotOS workshop.
Establishes the taxonomy of distributed tracing approaches: black-box
(statistical inference from external metrics) vs white-box (instrumented
tracing with causal spans). AIPerf's correlation analysis is fundamentally
black-box since we cannot instrument the server's internal execution pipeline.

**Sigelman et al., "Dapper, a Large-Scale Distributed Systems Tracing
Infrastructure" (2010)**. Google Technical Report. Introduces the span-based
tracing model that would be the gold standard for request-level decomposition.
Without server-side trace integration, our decomposition relies on statistical
matching of aggregate distributions.

**Key insight**: In the absence of per-request join keys between client and
server, *distributional decomposition* (comparing CDFs and computing
Wasserstein distances) is the principled alternative to per-request tracing.

### 10.2 Queueing Theory Foundations

**Little's Law**: `L = lambda * W` (Little, 1961). Holds for ANY stable
system regardless of arrival distribution, service distribution, or number of
servers. The only requirement is stationarity (steady state). This is why
AIPerf's steady-state detection is a prerequisite for valid Little's Law
validation.

**M/M/c Queue Model**: For a system with Poisson arrivals (rate lambda),
exponential service times (rate mu), and c servers:

```
rho = lambda / (c * mu)                    (utilization)
P(queue) = C(c, rho) * rho / (1 - rho)     (probability of queueing)
W_q = C(c, rho) / (c * mu * (1 - rho))     (expected queue wait time)
```

Where `C(c, rho)` is the Erlang C formula. While LLM inference does not follow
exponential service times (prefill time depends on prompt length), the M/M/c
model provides:
1. A baseline for expected queue wait time given observed utilization.
2. A sensitivity estimate: how much does queue time increase per 10% increase
   in utilization?

**Kingman's Approximation**: For a G/G/1 queue (general arrivals, general
service, single server):

```
W_q ≈ (rho / (1 - rho)) * ((C_a^2 + C_s^2) / 2) * E[S]
```

Where `C_a` and `C_s` are the coefficients of variation of inter-arrival and
service times, and `E[S]` is the mean service time. This is more applicable to
LLM inference where service time distribution is far from exponential (bimodal:
short for cached KV + long for cold prefill).

### 10.3 LLM-Specific Inference Latency Models

**Agrawal et al., "Sarathi: Efficient LLM Inference by Piggybacking Decodes
with Chunked Prefills" (2023)**. Introduces the prefill-decode interference
model where chunked prefill operations interrupt decode iterations, causing ITL
spikes. This is directly observable as bimodality in the server-side ITL
histogram.

**Yu et al., "ORCA: A Distributed Serving System for Transformer-Based
Generative Models" (2022)**. Describes iteration-level scheduling where batch
composition changes every decode step. The number of concurrent sequences
(`num_requests_running`) directly determines per-token latency through shared
GPU memory bandwidth.

**Kwon et al., "Efficient Memory Management for Large Language Model Serving
with PagedAttention" (2023)**. vLLM's PagedAttention paper. Key insight: KV
cache pressure causes preemptions, which show up as:
- Server: `vllm:num_preemptions_total` counter increments
- Client: Latency spike for the preempted request
- GPU: Temporary utilization drop during recomputation

The temporal correlation between `num_preemptions_total` rate and client p99
latency is a directly testable hypothesis.

### 10.4 Time Series Correlation Methods

**Granger, C.W.J., "Investigating Causal Relations by Econometric Models and
Cross-spectral Methods" (1969)**. Econometrica. The foundational paper on
Granger causality. The key assumptions (stationarity, linearity) are
approximately satisfied within steady-state benchmark windows.

**Box, G.E.P., Jenkins, G.M., and Reinsel, G.C., "Time Series Analysis:
Forecasting and Control" (4th ed., 2008)**. The standard reference for
cross-correlation, ARIMA modeling, and transfer function analysis. Chapter 11
on cross-correlation functions is directly applicable to our lag analysis.

**Shannon, C.E., "A Mathematical Theory of Communication" (1948)**. Bell
System Technical Journal. Foundation for mutual information, which captures
non-linear dependencies between metrics that Pearson correlation misses.

### 10.5 Distribution Comparison Methods

**Kolmogorov, A.N., "Sulla determinazione empirica di una legge di
distribuzione" (1933)**. The original KS test paper. The two-sample variant
(Smirnov, 1939) is appropriate for comparing client and server metric
distributions of different sample sizes.

**Vaserstein, L.N., "Markov processes over denumerable products of spaces
describing large systems of automata" (1969)**. The mathematical foundation
for the Wasserstein distance. The 1-Wasserstein (Earth Mover's) distance has
the appealing property of being directly interpretable as the average shift
between distributions.

**Note on the Wasserstein-percentile relationship**: If the only difference
between client and server ITL distributions is an additive constant `delta`
(i.e., every token incurs identical network delay), then
`W_1(F_client, F_server) = delta`. In practice, the network delay has variance,
so `W_1` estimates the *mean* additive component.

### 10.6 Tail Latency in Distributed Systems

**Dean, J., and Barroso, L.A., "The Tail at Scale" (2013)**. Communications
of the ACM. Establishes why p99 and p99.9 matter for user-facing systems
under fan-out. While LLM inference is typically single-server, the principles
apply when multiple requests share GPU resources: a single slow decode
iteration (due to preemption) becomes a p99 tail event.

**Harchol-Balter, M., "Performance Modeling and Design of Computer Systems:
Queueing Theory in Action" (2013)**. Cambridge University Press. Chapters 24-26
cover heavy-tailed service times, which are relevant to LLM inference where
output sequence length (and thus request duration) follows a heavy-tailed
distribution.

---

## Appendix: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| `L_client(i)` | Client-measured latency for request i |
| `T_server(i)` | Server-side processing time for request i |
| `T_queue(i)` | Server-side queue wait time for request i |
| `T_prefill(i)` | Server-side prefill computation time for request i |
| `T_decode(i)` | Server-side total decode time for request i |
| `T_conn(i)` | Client-side connection overhead for request i |
| `T_send(i)` | Client-side request sending time for request i |
| `T_recv(i)` | Client-side response receiving time for request i |
| `T_network_out` | One-way network latency, client to server |
| `T_network_in` | One-way network latency, server to client |
| `RTT` | Round-trip time = T_network_out + T_network_in |
| `F_X(x)` | Cumulative distribution function of random variable X |
| `D_KS` | Kolmogorov-Smirnov statistic (max CDF difference) |
| `W_1` | Wasserstein-1 (Earth Mover's) distance |
| `R_xy(tau)` | Cross-correlation function at lag tau |
| `I(X; Y)` | Mutual information between X and Y |
| `rho` | Server utilization (arrivals / capacity) |
| `lambda` | Request arrival rate (requests per second) |
| `mu` | Service rate (requests per second per server) |
| `L` | Mean number of requests in system (Little's Law) |
| `W` | Mean time in system (Little's Law) |
| `C_a` | Coefficient of variation of inter-arrival times |
| `C_s` | Coefficient of variation of service times |
| `E[S]` | Expected (mean) service time |
| `OSL` | Output sequence length (tokens) |
| `ISL` | Input sequence length (tokens) |
| `p50, p95, p99` | 50th, 95th, 99th percentile |

### Key Relationships

**Latency identity** (per request, ideal):
```
request_latency = TTFT + (OSL - 1) * ITL_avg
```

**Network overhead identity** (aggregate):
```
median(client_e2e) - median(server_e2e) ≈ RTT + client_processing_overhead
```

**TTFT decomposition identity**:
```
TTFT_client = T_conn + T_send + T_network_out + T_queue_server + T_prefill + T_network_in_first_token
```

**Variance decomposition** (assuming independence):
```
Var(ITL_client) ≈ Var(ITL_server) + Var(ITL_network)
```

**Little's Law** (steady state):
```
effective_concurrency_avg = request_throughput * mean_request_latency_seconds
```

**Queue wait time** (Kingman's approximation):
```
W_q ≈ (rho / (1 - rho)) * ((C_a^2 + C_s^2) / 2) * E[S]
```

---

## Summary and Recommended Implementation Order

| Phase | Component | Effort | Dependencies |
|-------|-----------|--------|-------------|
| **P0** | Network overhead budget (arithmetic) | Low | HTTP trace metrics (exist) |
| **P0** | TTFT decomposition (per-percentile) | Low | `stream_setup_latency`, `stream_prefill_latency` (exist) |
| **P0** | Little's Law validation | Low | Sweep metrics (exist) |
| **P1** | Server histogram percentile extraction | Medium | `ServerMetricsHierarchy` access (exists) |
| **P1** | Distribution comparison (KS, Wasserstein) | Medium | scipy (optional dep) |
| **P1** | `CorrelationResults` JSON export | Medium | Export pipeline |
| **P2** | Temporal cross-correlation | Medium | Time alignment utilities |
| **P2** | Granger causality testing | Medium | scipy |
| **P2** | Mutual information | Low | numpy-only |
| **P3** | Latency waterfall visualization | Medium | Plot subsystem |
| **P3** | Dual-axis time series plots | Medium | Plot subsystem |
| **P3** | Automated spike attribution | High | All prior phases |

P0 is immediately implementable with existing data and no new dependencies.
P1 requires accessing server histogram data from within the analysis pipeline.
P2 adds temporal analysis requiring time-alignment utilities. P3 adds
visualization and automated diagnostics.

The total implementation is estimated at 2000-3000 lines of code, primarily in
a new `src/aiperf/analysis/correlation.py` module with supporting utilities in
`src/aiperf/analysis/time_alignment.py`.
