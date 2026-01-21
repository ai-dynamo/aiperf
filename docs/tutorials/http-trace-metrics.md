<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HTTP Trace Metrics Guide

This guide explains the HTTP request lifecycle tracing metrics available in AIPerf, which provide granular timing information at the transport layer for performance analysis and debugging.

## Overview

AIPerf captures detailed timing information throughout the HTTP request lifecycle using the aiohttp tracing system. These metrics follow industry-standard conventions from **k6 load testing** and the **HAR (HTTP Archive) specification**, making them familiar and compatible with existing performance analysis tools.

**Key characteristics:**

- Trace metrics are captured using `time.perf_counter_ns()` for **nanosecond precision** during measurement
- When exported, timestamps are converted to **wall-clock time** (`time.time_ns()`) for correlation with logs and external systems
- The naming convention uses the `http_req_` prefix to match [k6's metric naming](https://grafana.com/docs/k6/latest/using-k6/metrics/reference/)

**Enabling trace timing output:**

To display HTTP trace timing metrics in the console output, use the `--show-trace-timing` flag:

```bash
aiperf profile ... --show-trace-timing
```

This displays a separate table with the HTTP trace timing breakdown after the main metrics table.

## Request Lifecycle

The HTTP request lifecycle breaks down into distinct phases, each measured independently:

```
Request Lifecycle ──────────────────────────────────────────────────────────────────────────────►
    │              │              │                │                    │                       │
    │◄─ dns_ns ───►│◄ connect_ns ►│◄─ sending_ns ─►│◄──── waiting_ns ──►│◄──── receiving_ns ───►│
    │              │              │                │                    │                       │
dns resolution   TCP+TLS      request send     request_send_end     first body chunk      last body chunk
(cache miss)    handshake     (last chunk)     (ready for server)   (response starts)     (response complete)
    │              │                                                   │
    └─ dns_cache_hit (skip lookup)                                     └── response_headers_received
                    │
                    └─ connection_reused (skip TCP/TLS)
```

## Metric Reference

### Connection Phase Metrics

These metrics capture the time spent establishing a connection before the HTTP request can be sent. They are specific to the aiohttp HTTP client.

| Metric | k6 Equivalent | HAR Equivalent | Description |
|--------|---------------|----------------|-------------|
| `http_req_blocked` | `http_req_blocked` | `blocked` | Time spent waiting for a free connection slot from the connection pool. High values indicate pool saturation. |
| `http_req_dns_lookup` | `http_req_looking_up` | `dns` | Time spent on DNS resolution. Returns `0` if DNS was cached or connection was reused. |
| `http_req_connecting` | `http_req_connecting` | `connect` | Time to establish TCP connection. **For HTTPS, this includes TLS handshake time.** Returns `0` if connection was reused. |
| `http_req_connection_reused` | — | — | Binary indicator (`0` or `1`) showing whether an existing connection was reused from the pool. |
| `http_req_connection_overhead` | — | — | Combined overhead: `blocked + dns_lookup + connecting`. Total pre-request setup cost. |

### Request/Response Phase Metrics

These core timing metrics measure the actual HTTP request and response transfer. They are available for any HTTP client that populates the base trace data model.

| Metric | k6 Equivalent | HAR Equivalent | Description |
|--------|---------------|----------------|-------------|
| `http_req_sending` | `http_req_sending` | `send` | Time to transmit the complete HTTP request (headers + body) to the server. |
| `http_req_waiting` | `http_req_waiting` | `wait` | **Time to First Byte (TTFB)** — time from request completion to first response body byte. Represents server processing time + network latency. Note: This measures time to the first *body* chunk, not the first header byte. |
| `http_req_receiving` | `http_req_receiving` | `receive` | Time to download the complete response body. Returns `0` for single-chunk responses. |
| `http_req_duration` | `http_req_duration` | `time` | **Request/response exchange time** (excluding connection overhead): `sending + waiting + receiving` |
| `http_req_total` | — | — | **Full end-to-end time**: sum of all 6 timing phases. See [HTTP Total Time vs Request Latency](#http-total-time-vs-request-latency). |

### Data Size Metrics

| Metric | Description |
|--------|-------------|
| `http_req_data_sent` | Total bytes transmitted in the request (transport layer). |
| `http_req_data_received` | Total bytes received in the response (transport layer). |
| `http_req_chunks_sent` | Number of transport-level write operations during the request. |
| `http_req_chunks_received` | Number of transport-level read operations during the response. |

## Key Relationships

### Duration Metric

The `http_req_duration` metric is **measured directly** from timestamps for maximum accuracy:

```
http_req_duration = response_receive_end_perf_ns - request_send_start_perf_ns
```

This measures from when the request started being sent to when the response was fully received/finalized. Conceptually this covers `sending + waiting + receiving`, but the direct measurement is more accurate than summing components.

### Total Connection Overhead

Connection overhead combines all pre-request setup time:

```
http_req_connection_overhead = http_req_blocked + http_req_dns_lookup + http_req_connecting
```

### Total Time Formula

The `http_req_total` metric sums all 6 timing phases for a **reconcilable breakdown**:

```
http_req_total = http_req_blocked + http_req_dns_lookup + http_req_connecting
               + http_req_sending + http_req_waiting + http_req_receiving
```

> **Note:** `http_req_total` and `http_req_duration` may differ slightly because:
> - `http_req_duration` is measured end-to-end (includes response finalization time)
> - `http_req_total` is computed from components (ends at last chunk, before finalization)
>
> Use `http_req_total` when you need the breakdown to add up exactly. Use `http_req_duration` when you want the most accurate single measurement of request/response exchange time.

### Important Distinctions

- **TTFB vs TTFT**: `http_req_waiting` measures Time to First **Byte** (specifically, the first *body* byte after headers), not Time to First **Token**. The server sends HTTP headers first, then body content. For LLM APIs, the first body byte may contain protocol overhead before actual tokens appear. Use the `time_to_first_token` metric for LLM-specific timing that measures when the first actual token content is received.

- **Connection reuse**: When `http_req_connection_reused = 1`, both `http_req_dns_lookup` and `http_req_connecting` will be `0` since no new connection was established.

### HTTP Total Time vs Request Latency

You may notice that `http_req_total` can be **larger** than `request_latency`. This is expected behavior — the two metrics measure different things:

| Metric | Start | End | What it measures |
|--------|-------|-----|------------------|
| `request_latency` | Before HTTP call | Last **content** response | Time to receive all meaningful tokens |
| `http_req_total` | Sum of phases starting at pool wait | Last **network** chunk | Sum of all HTTP timing phases |
| `http_req_duration` | Request send start | Response **finalized** | Measured request/response exchange |

**Why `http_req_total` > `request_latency`:**

For streaming LLM responses (SSE), the HTTP stream typically ends with:

```
[content chunk 1]  ─► included in both metrics
[content chunk 2]  ─► included in both metrics
[content chunk N]  ─► request_latency ends here (last actual token)
[usage info]       ─► http_req_total includes this
[DONE]             ─► http_req_total ends here (last network chunk)
                   ─► http_req_duration ends here (response finalized)
```

The `request_latency` metric excludes trailing metadata (`[DONE]` markers, usage statistics) because those don't represent meaningful content delivery. The HTTP trace metrics include all network traffic.

**Which metric should I use?**

| Use Case | Recommended Metric |
|----------|-------------------|
| User-perceived latency ("when did I get the last useful token?") | `request_latency` |
| Transport-level timing with reconcilable breakdown | `http_req_total` |
| Most accurate single request/response measurement | `http_req_duration` |
| Debugging: gap between content and stream end | `http_req_total - request_latency` |

## Accessing Trace Data

### Enabling HTTP Trace Export

By default, raw HTTP trace data is **not included** in `profile_export.jsonl` to keep file sizes small. The computed metrics (`http_req_duration`, `http_req_waiting`, etc.) are always available regardless of this setting.

To include the full trace data (timestamps, chunks, headers, socket info), use the `--export-http-trace` flag:

```bash
aiperf profile ... --export-http-trace
```

### Export Levels

The `--export-http-trace` flag works with `records` or `raw` export levels:

| Export Level | Trace Data (with `--export-http-trace`) | Use Case |
|--------------|----------------------------------------|----------|
| `summary` | Not available | Quick benchmark summaries |
| `records` | ✓ Included | Per-request analysis with timing details |
| `raw` | ✓ Included | Full debugging with complete request/response data |

Example with both flags:

```bash
aiperf profile ... --export-level records --export-http-trace
```

### Output Format

When exported to `profile_export.jsonl`, trace data uses **wall-clock timestamps** (nanoseconds since epoch) for cross-system correlation. The trace data is included in each record:

```json
{
  "metadata": { "x_request_id": "9568f1d7-10e9-4d42-bb69-b06c87caae9f", "..." : "..." },
  "metrics": {
    "http_req_duration": {"value": 37421.26, "unit": "ms"},
    "http_req_waiting": {"value": 4473.26, "unit": "ms"},
    "http_req_blocked": {"value": 0.0, "unit": "ms"},
    "..." : "..."
  },
  "trace_data": {
    "trace_type": "aiohttp",
    "request_send_start_ns": 1768309400341882300,
    "request_headers": {"Host": "localhost:8000", "Content-Type": "application/json", "...": "..."},
    "request_headers_sent_ns": 1768309400342515706,
    "request_chunks": [[1768309400342549481, 91586]],
    "response_status_code": 200,
    "response_reason": "OK",
    "response_receive_start_ns": 1768309404815807889,
    "response_headers": {"Content-Type": "text/event-stream; charset=utf-8", "...": "..."},
    "response_headers_received_ns": 1768309400369701553,
    "response_chunks": [[1768309404815807889, 565], [1768309405191294711, 268], "..."],
    "response_receive_end_ns": 1768309437763141384,
    "request_send_end_ns": 1768309400342549481,
    "sending_ns": 667181,
    "waiting_ns": 4473258408,
    "receiving_ns": 32947333495,
    "duration_ns": 37421259084,
    "tcp_connect_start_ns": 1768309400341999707,
    "tcp_connect_end_ns": 1768309400342480993,
    "connecting_ns": 481286,
    "dns_cache_hit_ns": 1768309400342023726,
    "local_ip": "127.0.0.1",
    "local_port": 48362,
    "remote_ip": "127.0.0.1",
    "remote_port": 8000
  },
  "error": null
}
```

> **Note:** Computed duration fields (`blocked_ns`, `dns_lookup_ns`, `connection_reused_ns`) are **omitted** from `trace_data` when the underlying event did not occur. The corresponding metrics (e.g., `http_req_blocked`) will report `0` for aggregation purposes, but the trace field itself is absent.

### Trace Data Fields

The `trace_data` object contains both raw timestamps and computed durations:

**Raw Timestamps** (wall-clock nanoseconds):

| Field | Description |
|-------|-------------|
| `request_send_start_ns` | When the HTTP request started being sent |
| `request_headers_sent_ns` | When the request headers finished being sent |
| `request_send_end_ns` | When the request body finished being sent (computed from last chunk) |
| `response_headers_received_ns` | When response headers were received |
| `response_receive_start_ns` | When the first response body chunk was received |
| `response_receive_end_ns` | When the response finished being received |
| `error_timestamp_ns` | When an error occurred during the request (if any) |
| `connection_pool_wait_start_ns` | When waiting for a connection started (aiohttp only) |
| `connection_pool_wait_end_ns` | When a connection was obtained (aiohttp only) |
| `tcp_connect_start_ns` | When TCP connection establishment started (aiohttp only) |
| `tcp_connect_end_ns` | When TCP connection completed (aiohttp only) |
| `connection_reused_ns` | When an existing connection was reused (aiohttp only) |
| `dns_lookup_start_ns` | When DNS resolution started (aiohttp only) |
| `dns_lookup_end_ns` | When DNS resolution completed (aiohttp only) |
| `dns_cache_hit_ns` | When a DNS cache hit occurred (aiohttp only) |
| `dns_cache_miss_ns` | When a DNS cache miss occurred (aiohttp only) |

**Chunk Data** (transport-layer granularity):

| Field | Description |
|-------|-------------|
| `request_chunks` | Array of `[timestamp_ns, size_bytes]` tuples for each request chunk sent |
| `response_chunks` | Array of `[timestamp_ns, size_bytes]` tuples for each response chunk received |

**Computed Durations** (nanoseconds):

| Field | Description |
|-------|-------------|
| `sending_ns` | Request send time |
| `waiting_ns` | Time to first byte (TTFB) |
| `receiving_ns` | Response transfer time |
| `duration_ns` | Total request duration |
| `blocked_ns` | Connection pool wait time |
| `dns_lookup_ns` | DNS resolution time |
| `connecting_ns` | TCP/TLS connection time |

**Request/Response Metadata**:

| Field | Description |
|-------|-------------|
| `request_headers` | Dictionary of request headers sent |
| `response_status_code` | HTTP status code of the response |
| `response_reason` | HTTP status reason phrase (e.g., "OK", "Not Found") |
| `response_headers` | Dictionary of response headers received |

**Connection Info** (aiohttp only):

| Field | Description |
|-------|-------------|
| `local_ip` | Local IP address used for the connection |
| `local_port` | Local (ephemeral) port used |
| `remote_ip` | Remote server IP address |
| `remote_port` | Remote server port |

## Common Use Cases

### Identifying Connection Pool Saturation

If `http_req_blocked` is consistently high, your connection pool is exhausted. Consider:

- Increasing the connection pool size
- Reducing the number of concurrent requests
- Investigating slow responses that hold connections

### Detecting DNS Issues

If `http_req_dns_lookup` is high:

- DNS resolution is slow
- Consider using DNS caching or a faster resolver
- Check if DNS TTLs are appropriate

### Measuring Server Processing Time

`http_req_waiting` (TTFB) isolates server-side latency:

- Low `sending` + High `waiting` = Server is the bottleneck
- High `receiving` = Large response or slow network throughput

### Analyzing Connection Efficiency

Track `http_req_connection_reused` aggregated values:

- Values close to `1.0` (100% reuse) indicate efficient keep-alive usage
- Low reuse rates suggest connection churn, adding overhead via DNS lookups and TCP/TLS handshakes

### Chunk-Level Analysis

The `request_chunks` and `response_chunks` arrays provide transport-layer granularity useful for:

- Analyzing streaming response patterns from LLM APIs
- Debugging chunked transfer encoding issues
- Understanding network-level timing variations

## Standards Compliance

AIPerf trace metrics align with industry standards for compatibility with existing tools:

| Standard | Reference |
|----------|-----------|
| **k6** | [Grafana k6 Built-in Metrics](https://grafana.com/docs/k6/latest/using-k6/metrics/reference/) |
| **HAR 1.2** | [W3C HTTP Archive Specification](https://w3c.github.io/web-performance/specs/HAR/Overview.html) |

### HAR Specification Notes

Per the [HAR 1.2 specification](https://w3c.github.io/web-performance/specs/HAR/Overview.html):

- `blocked`, `dns`, `connect` use `-1` when not applicable (AIPerf uses `0` or `null`)
- `send`, `wait`, `receive` are required non-negative values
- `time` (duration) equals the sum of all applicable timing phases
- `ssl` timing is included within `connect` for backwards compatibility

### k6 Metric Differences

The k6 `http_req_tls_handshaking` metric is **not separated** in AIPerf. TLS time is combined with TCP connection time in `http_req_connecting` because aiohttp's tracing API provides a combined measurement via `on_connection_create_start/end` events.

## Quick Reference

| What You Want to Know | Metric to Use |
|----------------------|---------------|
| Full end-to-end HTTP time | `http_req_total` |
| Request/response time (excl. connection) | `http_req_duration` |
| Server processing time | `http_req_waiting` (TTFB) |
| Network transfer efficiency | `http_req_receiving` / `http_req_data_received` |
| Connection pool health | `http_req_blocked` |
| Connection reuse rate | `http_req_connection_reused` |
| DNS performance | `http_req_dns_lookup` |
| Pre-request overhead | `http_req_connection_overhead` |
| User-perceived latency (LLM) | `request_latency` (not an HTTP trace metric) |

## Related Documentation

- [Working with Profile Export Files](./working-with-profile-exports.md) - How to parse and analyze AIPerf output files
- [Source: trace_models.py](../../src/aiperf/common/models/trace_models.py) - Trace data model definitions
- [Source: http_trace_metrics.py](../../src/aiperf/metrics/types/http_trace_metrics.py) - HTTP trace metric implementations
