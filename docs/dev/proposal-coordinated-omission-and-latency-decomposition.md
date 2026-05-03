<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Proposal: Coordinated Omission Awareness & Latency Decomposition

**Phase:** P1 (Coordinated Omission), P2 (Latency Decomposition)
**Depends on:** P0 (tail percentiles)

## Part 1: Coordinated Omission

### Background

Coordinated omission (CO) is a measurement bias identified by Gil Tene (2013)
where a benchmark *under-reports* latency because it slows down sending new
requests when the server is slow.

**Classic CO (open-loop benchmarks):** A benchmark intends to send a request
every 10ms. When request #100 takes 500ms, the benchmark is blocked and skips
~50 intended send times. The 50 users who *would have* experienced 500ms+
latency are simply never measured. The reported p99 understates reality.

**AIPerf's natural mitigation:** The credit-based system doesn't suffer from
classic CO in the same way. Credits are issued at the intended rate by the
TimingManager regardless of response latency. Workers wait for credits, then
send immediately. The latency measured is `request_end - request_start`, which
captures the true response time for each request that was sent.

**The subtle form that still applies:** Under concurrency-limited operation
(e.g., `--concurrency 10`), when all 10 slots are occupied by slow requests,
no new requests can start. The users who *would have* submitted requests during
this period experience "omitted" latency — they're waiting in the virtual queue.
The credit_issued_ns captures when the credit was granted (≈ when the user
wanted to send), but the request doesn't actually start until a concurrency
slot opens.

### What We Already Have

The metadata already tracks the critical timestamp:

```python
# In ColumnStore metadata (ingested from MetricRecordsData.metadata):
"credit_issued_ns": meta.credit_issued_ns   # When TimingManager issued the credit
"request_start_ns": meta.request_start_ns   # When the HTTP request was actually sent
"request_ack_ns": meta.request_ack_ns       # When the server sent HTTP 200 OK (streaming only)
```

The gap `request_start_ns - credit_issued_ns` IS the client-side queuing delay —
exactly what CO correction adds to the latency measurement. Note that
`request_ack_ns` is the *server's* acknowledgement (HTTP 200 OK for streaming
requests), not the point when the worker picked up the credit.

### Proposed Metrics

**New RECORD metric: `service_latency`**

```
service_latency = request_end_ns - credit_issued_ns
```

This is the "user-perceived" latency: from the moment the system decided to
send a request (credit issuance) to the moment the response completed. It
includes:
- Client-side queue wait (credit → worker picks up)
- Network transit to server
- Server processing (prefill + decode)
- Network transit back
- Response parsing

Compare with the existing `request_latency`:
```
request_latency = request_end_ns - request_start_ns
```

This is the "server-focused" latency: from when the HTTP request was sent to
when the response completed. It excludes client-side queuing.

**New RECORD metric: `queue_wait_time`**

```
queue_wait_time = request_start_ns - credit_issued_ns
```

The time the request spent waiting in the client-side queue (from credit
issuance to HTTP send). Under low load this is ~0. Under high load or
concurrency limits, this is where CO manifests.

**Console output:**
- `service_latency` and `queue_wait_time` are `MetricFlags.NO_CONSOLE` by
  default (opt-in via `--metrics` or always in exports)
- When `queue_wait_time.p99 > 0.1 × request_latency.p99`, print a warning:
  "Significant queuing delay detected — service_latency includes client-side
  wait. See JSON export for details."

### Why Two Latency Views Matter

| Scenario | request_latency | service_latency | queue_wait_time |
|---|---|---|---|
| Low load (concurrency < server capacity) | ~same | ~same | ~0 |
| High load, fast server | ~same | ~same | ~0 |
| High load, server near saturation | 200ms | 350ms | 150ms |
| Overloaded server | 500ms | 2000ms | 1500ms |

The divergence between request_latency and service_latency is the smoking gun
for server saturation. Users comparing configurations need both views.

### Implementation

These are straightforward DERIVED metrics. The data already exists in the
ColumnStore metadata columns:

```python
class ServiceLatencyMetric(BaseDerivedMetric[float]):
    """User-perceived latency including client-side queuing (CO-corrected)."""
    tag = "service_latency"
    # derive from: request_end_ns - credit_issued_ns
    # Uses ColumnStore metadata_numeric("credit_issued_ns") and end_ns

class QueueWaitTimeMetric(BaseDerivedMetric[float]):
    """Client-side queuing delay between credit issuance and request send."""
    tag = "queue_wait_time"
    # derive from: request_start_ns - credit_issued_ns
```

**Challenge:** These can't be simple DERIVED metrics computed from scalar sums —
they need per-record computation (each record's credit_issued_ns minus its
request_start_ns). Two options:

**Option A:** Compute them in the RecordProcessor (alongside ITL, TTFT) and
include in MetricRecordsData.metrics. This means they flow through the existing
RECORD metric pipeline and get full statistical treatment (percentiles, etc.).

**Option B:** Compute them in MetricsAccumulator from ColumnStore metadata
columns at summarize time. This avoids changing the record processor but
requires special handling in `_compute_results()`.

**Recommendation:** Option A — compute in RecordProcessor. It's a clean
per-record metric like any other. The subtracted timestamps are both available
on the ParsedResponseRecord metadata.

---

## Part 2: Latency Decomposition

### Motivation

When request_latency is high, users need to know *where* the time is spent.
Currently we provide TTFT and ITL, but a complete decomposition would show:

```
Total service_latency breakdown (conceptual):
├── Queue wait:       credit_issued_ns → request_start_ns    (client-side, measurable)
├── TTFT:             request_start_ns → first_token_ts      (network + prefill, measurable)
│   ├── Network send:     (included in TTFT, not separable)
│   └── Prefill compute:  (included in TTFT, not separable)
├── Decode:           first_token_ts → request_end_ns        (GPU compute, measurable)
│   └── (includes per-token network receive time)
└── Total:            credit_issued_ns → request_end_ns
```

Note: we cannot separate network transit from server-side compute without
server-side instrumentation. TTFT includes both network send + server queue +
prefill. Decode time includes per-token network receive.

Some of these sub-phases we can measure, some we can only approximate:

| Phase | Can We Measure? | How |
|---|---|---|
| Queue wait | Yes | `request_start_ns - credit_issued_ns` |
| Prefill (TTFT) | Yes | `time_to_first_token` (already exists) |
| Decode | Yes | `request_latency - time_to_first_token` |
| Network overhead | Partial | HTTP trace metrics (when enabled) |
| Server-side queue | No | Would need server-side instrumentation |

### Proposed Metrics

**`decode_time`** — RECORD metric
```
decode_time = request_latency - time_to_first_token
```

**`decode_fraction`** — DERIVED metric
```
decode_fraction = total_decode_time / total_request_latency
```
Tells users what fraction of their latency is decode vs. prefill.

**`prefill_fraction`** — DERIVED metric
```
prefill_fraction = total_time_to_first_token / total_request_latency
```

These are simple to compute and high in diagnostic value. A user seeing
`prefill_fraction = 0.80` knows they're prefill-bound and should look at
input sequence lengths or prefill optimization.

---

## Part 3: Little's Law Cross-Validation

### Motivation

Little's Law states: **L = λ × W**

Where:
- L = average number of requests in the system (concurrency)
- λ = average throughput (requests/sec)
- W = average time a request spends in the system (latency)

We already compute all three quantities from sweep-line algorithms:
- L = `effective_concurrency.avg`
- λ = `request_throughput`
- W = `request_latency.avg`

If L ≠ λ × W (within some tolerance), it suggests a measurement error:
dropped records, timestamp skew, or incorrect throughput calculation.

### Proposed Validation

**New validation in SteadyStateAnalyzer (or as a standalone analyzer):**

```python
L = effective_concurrency_avg
lambda_W = request_throughput * (mean_request_latency / 1e9)  # convert ns to sec
discrepancy_pct = abs(L - lambda_W) / L * 100

if discrepancy_pct > 10.0:
    warning("Little's Law discrepancy: L={L:.1f}, λW={lambda_W:.1f} ({discrepancy_pct:.1f}%)")
```

Include in JSON export metadata:

```json
{
  "validation": {
    "littles_law": {
      "L": 42.3,
      "lambda_W": 43.1,
      "discrepancy_pct": 1.9,
      "status": "pass"
    }
  }
}
```

### Why This Matters

Little's Law holds for *any* stable system, regardless of distribution. A
violation indicates the system was not in steady state or the measurements
are inconsistent. This is a free sanity check with zero additional data
collection.

---

## Summary

| Change | Phase | Effort | Data Exists? |
|---|---|---|---|
| service_latency metric | P1 | Low | Yes (credit_issued_ns) |
| queue_wait_time metric | P1 | Low | Yes (credit_issued_ns, request_start_ns) |
| decode_time metric | P2 | Low | Yes (request_latency, TTFT) |
| prefill/decode fractions | P2 | Low | Yes (derived) |
| Little's Law validation | P2 | Low | Yes (all sweep quantities) |

All of these are low-effort because the underlying data is already captured.
The value is in surfacing it as named, documented metrics with proper
statistical treatment.
