<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HTTP transport

## Purpose

The `aiperf_runtime::transport::http` module is the Clock-injected hyper HTTP
stack. It sends inference requests, streams SSE, and records precise
per-request/per-token/per-chunk timing into a record. It owns wire I/O and timing
recording only; scheduling, admission, and observation vocabulary live above it
(see [execution-model.md](execution-model.md) and [scheduling.md](scheduling.md)).

## Built

### Wire and protocols

HTTP/1.1, HTTP/2 (h2 over TLS via ALPN and h2c cleartext prior-knowledge), UDS,
and TLS are supported. All time access — timestamps, deadlines, cancellation
timers — routes through `Clock`, never `Instant::now`, `SystemTime::now`, or raw
`tokio::time`, so `SimClock` tests are deterministic.

### Streaming and measurement

SSE is preserved as bytes until complete lines are available (a UTF-8 code point
may span network chunks). The transport captures TTFT as the first token
observation, per-token and per-chunk timing, and authoritative server `usage`
token counts, reconciling them when present and keeping absent usage fields
absent. It feeds the shared reduction and measurement seams
(`transport::reduce` / `transport::measure`).

### Connection tracing

`transport::core::trace::TraceData` is the capture surface, populated across the
client stack: `client/pool.rs` (pool wait, socket 4-tuple), `client/resolver.rs`
(DNS cache hit/miss, lookup span), `client/connection.rs` (TCP and TLS spans),
and `transport/polling.rs` (`merge_trace` folds poll cycles into one trace). It
holds absolute `Clock::now_ns()` timestamps for pool wait, DNS lookup, TCP
connect, TLS connect, connection reuse, request send/headers-sent/send-end,
response receive-start/headers-received/receive-end, and error; the socket
`local`/`remote` ip and port; response status code and reason; and opt-in
per-chunk `(timestamp_ns, size)` vectors with byte and chunk counters for both
directions. Splitting TLS from TCP is a distinct span, not a combined connect.

Derived durations are k6/HAR math: `sending`, `waiting`, `time_to_first_header`,
`receiving`, `duration`, `blocked`, `dns_lookup`, `tcp_connect`, `tls_handshake`,
and `connecting`. `to_export` converts to wall clock from a caller-supplied
`TraceReference { clock_ns, wall_ns }` pair; the module reads no wall clock
itself.

The metrics plane consumes a narrower projection. `http_trace` in
`sink/endpoint_dispatch.rs` collapses `TraceData` into
`metrics_core::ingest::RequestTrace`: eight derived durations, four byte/chunk
counters, a `connection_reused` bool, and `stream_setup_ns` computed from the
record's `recv_start_ns - start_ns`. Absolute timestamps, per-chunk vectors, the
socket 4-tuple, and status/reason stay on `TraceData` and do not reach metric
ingestion.

### Endpoint binding

The object-safe `HttpEndpointBinding` seam owns URL construction (including
`{model_name}` expansion and `/v1` de-duplication), header composition, body
encoding, multipart JSON/binary encoding, request-local inline-media fetch with
deduplication, Clock-paced video submit→poll→download, SSE framing, and decoding
back into the canonical `ServerResponse` shape. The same endpoint implementation
binds to HTTP or another transport without transport-specific subclasses (see
[endpoints.md](endpoints.md) and [endpoint-body-construction.md](endpoint-body-construction.md)).

### Cancellation and reuse

Post-send request cancellation (`cancel_after_ns`) is anchored to body-send
completion, across the entire poll lifecycle for polled endpoints. Three
connection-reuse strategies are supported: `Pooled`, `Never`, and
`StickyUserSessions`.

## Future requirements

- Full HTTP/2 connection-reuse and multiplexing semantics are narrower than the
  general design.

## Source anchors

- `rust/runtime/src/transport/http/` (`client`, `config`, `models`, `sse`,
  `transport`, `sink`).
- `rust/runtime/src/transport/core/trace.rs` (`TraceData`, derived k6/HAR
  durations, `TraceReference`/`TraceExport`).
- `rust/runtime/src/transport/http/sink/endpoint_dispatch.rs` (`http_trace`
  projection onto `metrics_core::ingest::RequestTrace`).
- `rust/runtime/src/transport/{reduce.rs,measure.rs}` (shared reduction/measure).
- `rust/cli/tests/*_stdio.rs` and tier-2 online endpoint tests.
