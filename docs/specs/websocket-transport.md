<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WebSocket transport

## Purpose

Benchmark WebSocket inference over persistent WS/WSS connections. The native
transport composes the runtime's worker-local transport, content-lowering, and
measurement seams, and defines WebSocket application-event lag metrics that
the HTTP trace and existing audio endpoint support do not provide.

## Built

The runtime provides the WebSocket transport and the seams it composes over:

- `Clock`, worker-local `WorkerSink`, `ExecutionSinkBuilder`, and `LocalSet`
  execution define transport timing and placement.
- The segment store and `BodyPlan` preserve content as handles and
  pre-serialized bytes while dataset materialization still owns the store.
- The HTTP transport provides a byte-preserving SSE decoder and streaming
  response path used by the explicitly declared Responses fallback.
- The gRPC transport provides worker-local bidirectional message encoding and
  response decoding through endpoint bindings. It is not a persistent,
  cross-request multiplexing or round-trip-measurement precedent.
- `RequestObserver` supplies token, classified-token, usage, endpoint-metric,
  and terminal observations shared by transports. TTFT is derived from the first
  token observation; there is no separate first-token callback.
- The extension registry registers the `websocket` transport plus `responses`
  and `realtime` endpoint bindings, with duplicate rejection and a frozen
  application inventory.
- `WsRequestMaterializer` lowers handle-bearing body plans while the segment
  store remains available. The worker receives ordered immutable
  `PreparedWsOperation` application messages, never a handle-bearing plan.
- The worker-local split driver keeps read, write, control, deadline, and
  Clock-driven keepalive progress live behind bounded application queues. It
  records post-flush input and complete decoded application-event timing for
  eligible turn-serialized operations.

<!--
Provenance: external backend issue #706 and PR #713, squash
482e464482b63a9bfb87f91c95a59738a7b40b9e (2026-06-22); metric issue #832 and
PR #850, squash 7ac046f681ba5bb7da228bebd64d59fcc997aaa5 (2026-08-12). Formulas,
sampling boundaries, output behavior, and test coverage were verified against
the merged code rather than inferred from PR prose.
-->

## Current contract

A Clock-injected `transport::ws` implementation is registered alongside the
existing native transports. It reuses the segment content IR and `BodyPlan` lowering
contract (see [dataset.md](dataset.md) and
[endpoint-body-construction.md](endpoint-body-construction.md)), but it cannot
defer a handle-bearing `BodyPlan` to `WorkerSink`: the current dispatch boundary
rejects such a plan because the segment store no longer accompanies it. While
the store is still in scope, a dialect-owned `WsRequestMaterializer` therefore
produces a handle-free `PreparedWsOperation`: ordered immutable
`PreparedWsMessage` buffers with their logical roles, plus an independently
prepared HTTP/SSE request only when the dialect authorizes fallback. The worker
sink consumes this operation directly; it never reconstructs WebSocket messages
from `RequestBody::Wire`. Store-free HTTP plans continue to cross the existing
`RequestBody::Plan` seam; every text/token/media conversion and JSON escaping
happens before the segment store leaves scope. Realtime audio is first validated
as base64 wire data rather than assumed to be encoded merely because it is a
`Payload::Media`.

A dialect may splice bytes already serialized as valid JSON values into an event
envelope; arbitrary text must be escaped during lowering or emission. The hot
path does not rebuild complete requests through per-request `Value` trees. The
single-`Full<Bytes>` / `SendCompletion` rule stays HTTP-local and does not apply.

### Dialects — separate protocols, one transport

The transport is dialect-neutral; each protocol is its own binding behind the
registry and must not be conflated:

- **Responses** — the reference turn-serialized dialect at
  `wss://api.openai.com/v1/responses`. One affinity-bound connection carries
  exactly one in-flight `response.create` operation. Its request fields mirror
  the Responses create body; continuation uses `previous_response_id`; and its
  streamed event ordering matches the existing Responses SSE decoder. A socket
  is rotated before sixty minutes and parallel operations use a bounded
  connection pool.
- **Realtime** — a distinct dialect: OpenAI's bidirectional voice/text WS with its
  own client/server event schema, response/item correlation, and base64 audio
  inside JSON events (see the
  [Realtime conversations guide](https://developers.openai.com/api/docs/guides/realtime-conversations)).

They share transport mechanics and measurement plumbing — not an event schema,
connection-affinity rule, recovery contract, or logical-operation identity.
Responses may select its equivalent HTTP/SSE fallback before any application
message is flushed. Realtime has no HTTP/SSE fallback in this contract.

### Standards constraints

- All elapsed measurements use one injected, monotonic `Clock` domain. Client
  and server wall-clock timestamps are never combined without an explicit
  synchronization contract. The requirement follows the monotonic-duration
  guidance in [High Resolution Time](https://www.w3.org/TR/hr-time-3/).
- Measurement operates on complete WebSocket application messages after
  fragmentation is reassembled and the dialect decoder classifies the event.
  RFC 6455 permits fragmented messages and interjected control frames, so wire
  frame boundaries cannot define metric samples
  ([RFC 6455 sections 5.4–5.6](https://www.rfc-editor.org/rfc/rfc6455.html#section-5.4)).
- Ping/Pong measures control-path responsiveness, not inference latency. It is
  excluded from the application metrics. A future Ping/Pong metric must use a
  separate identity, correlate unique payloads, and follow the protocol's
  allowance for unsolicited or coalesced Pong responses
  ([RFC 6455 sections 5.5.2–5.5.3](https://www.rfc-editor.org/rfc/rfc6455.html#section-5.5.2)).
- WebSocket itself supplies no request multiplexing identity. A dialect must
  correlate every measured event through its logical operation, response, item,
  or round identifier. Socket order and “the next receive” are insufficient.

### Realization onto the seams

- Transport: `tokio-tungstenite` is a direct rustls/WebPKI-root dependency and
  runs in a worker-local connection driver on the worker's `LocalSet`. The
  driver continuously progresses reads, writes, control frames, cancellation,
  and deadlines. It uses split read/write halves with bounded byte-accounted
  queues, or one owner poll state machine that explicitly advances both
  `Stream::poll_next` and `Sink::{poll_ready,start_send,poll_flush}` in each
  poll cycle. It must not await `SinkExt::send` in a way that prevents receives.
  A dispatch drop guard marks its route cancelled and wakes the driver through a
  nonblocking path independent of the data queue. Shutdown closes admission,
  fails unsent work, terminates routed operations once, and runs a Clock-bounded
  close handshake. A binding chooses connection count and session/thread
  affinity; one socket per worker is not a universal invariant.
- Security: WSS and any permitted HTTPS fallback share the resolved HTTP TLS
  verification, custom trust, mTLS, proxy, authentication, and redaction policy.
- Recovery: first-class and dialect-scoped. Connect/upgrade failure uses the
  shared bounded retry policy. A cached-socket send failure, close before a
  terminal event, idle timeout, or lost continuation state invalidates that
  socket. A turn-serialized dialect may retry once on a fresh socket with a
  self-contained full-history request only when it has emitted no user-visible
  output for that operation, or after a dialect-classified continuation-state
  rejection. It drops any stale stream before reconnecting and omits the
  unusable continuation identity. Once any output-bearing backend event has been
  attributed, automatic replay is forbidden unless the dialect supplies an
  explicit idempotency contract. A replay starts a new measurement epoch; send
  timestamps from the abandoned attempt do not enter the successful record.
- Alternative transport: HTTP/SSE is not a universal WebSocket fallback. One
  Clock-bounded operation deadline covers retry and fallback. Only allowlisted
  pre-application network and unsupported-upgrade failures may select a
  dialect-declared equivalent HTTP/SSE operation. Certificate/hostname errors,
  authentication/authorization failures, malformed handshakes, required
  subprotocol mismatches, and application protocol failures fail closed. A
  continuation unavailable on a fresh socket is rematerialized as full history
  or fails; incremental input never carries an unusable response identity. Each
  record reports its actual route and stable fallback reason.
- Content: `message`/`text`/`token-ids` segments splice into event JSON; Realtime
  audio is a `media` segment base64-encoded once at lowering and spliced by
  reference. Dialects declare whether application messages are text or binary;
  an unexpected opcode is a protocol violation.
- Completion is gated on the dialect terminal event, not socket FIN. Connection
  loss terminates every routed in-flight operation exactly once.

### Public configuration and capability checks

Config v2 adds this strict transport variant. Every duration and size is finite
and positive; `fallback` defaults to `disabled`:

```yaml
transport:
  type: websocket
  fallback: disabled        # disabled | http_sse
  ping_interval_seconds: 30
  stream_idle_timeout_seconds: 900
  max_queued_commands: 64
  max_queued_bytes: 1048576
  max_frame_bytes: 1048576
  max_message_bytes: 8388608
  max_response_bytes: 67108864
```

The existing `endpoint.type` selects a WebSocket-capable dialect. Endpoint URL
schemes must be `ws` or `wss`. Existing `timeout_seconds`, `connection_limit`,
`keepalive_timeout`, `ssl_verify`, and `proxy`/`proxy_from_env` policies retain
their current units and precedence; `keepalive_timeout` bounds retention of an
idle cached connection and is distinct from the Ping interval and active-stream
idle bound above. Protocol v2 carries the same fields plus its existing
`max_connect_retries` connect policy. UDS is accepted only when the WebSocket
connector implements it. `endpoint.streaming` does not select WebSocket; the
transport does. The runtime and CLI expose an explicit `websocket` feature.
Configuration deserializes `transport.type: websocket` unconditionally;
feature-off builds fail closed when execution registry selection finds no
registered WebSocket transport, and never fall back to HTTP. The first
supported workloads are scheduled single/multi-turn execution and cellular
scheduled execution. Graph execution and unsupported artifacts fail closed
until their own WebSocket contracts are implemented.

Each registered dialect advertises its transport support, connection model
(`turn_serialized` or `duplex`), supported application opcodes, continuation and
replay capabilities, and any semantically equivalent alternative transport.
Bootstrap rejects an HTTP-only endpoint under `transport.type: websocket`, an
unsupported fallback request, or a dialect whose required connection model the
driver cannot provide. Config expansion emits the effective choices so a run is
reproducible.

### WebSocket round-trip measurement

The transport exposes two optional per-request distribution metrics. Their
public identifiers are retained for compatibility, but both are
application-event lag estimators rather than network or paired-message RTTs.
AIPerf metric identifiers omit unit suffixes, following `time_to_first_token`;
the catalog supplies `ms` as the display unit:

- `time_to_last_round_trip`: the last user-visible content-event timestamp minus
  the last successfully flushed, request-scoped application-message timestamp.
  Its precise meaning is last-content-after-final-send latency.
- `avg_round_trip_time`: the mean of user-visible content receive timestamps
  minus the mean of successfully flushed, request-scoped application-message
  timestamps. It is the difference of two unpaired event-population means, not
  the arithmetic mean of correlated round trips. Send and receive counts may
  differ.

The dialect defines which request-scoped application messages belong to the
measured input epoch. Connection/session setup messages shared by multiple
operations do not count. A content receive is observed once per decoded event
carrying non-empty user-visible content; token multiplicity inside one event does
not multiply timestamp samples. Reasoning-only deltas, usage, terminal,
Ping/Pong, and other control messages do not count. If a terminal envelope is
the first carrier of non-empty user-visible content, the dialect emits one
content observation before emitting the terminal fact; the terminal status
itself is never a sample.

The send timestamp is sampled immediately after the application-message
`SinkExt::send(...).await` succeeds and flushes the WebSocket sink into the
underlying async stream. It establishes neither peer receipt nor exact wire
transmission and excludes serialization, command-queue wait, and socket
backpressure before successful completion. The reported measures are therefore
post-flush response lags. Connection/upgrade latency remains a separate
measurement. No first-round-trip metric is specified; TTFT is not treated as an
equivalent because it may include connection establishment and the input upload
epoch.

The WebSocket worker sink owns one constant-size `RoundTripTimingState` per
in-flight logical operation:

- `last_send_ns: Option<i64>`, `send_timestamp_sum_ns: i128`, and
  `send_count: u64`;
- `last_content_receive_ns: Option<i64>`,
  `content_receive_timestamp_sum_ns: i128`, and
  `content_receive_count: u64`; and
- an invalid flag for checked sum/count overflow, invalid ordering, or broken
  dialect attribution.

The wider checked sums preserve integer-nanosecond inputs for long streams. The
mean difference remains `f64` nanoseconds until catalog display conversion;
integer division must not discard fractional nanoseconds. State is keyed by
logical operation identity, never by persistent connection alone, so adjacent
or concurrently interleaved operations cannot contaminate each other.

Before a proven completed terminal, the sink emits a transport-neutral
`ObservedRoundTripMetrics` fact with accurately named internal fields
`last_send_to_last_content_ns: Option<i64>` and
`mean_timestamp_lag_ns: Option<f64>` through
`RequestObserver::on_round_trip_metrics`; `ObserverTee` forwards it.
`NativeMetricsObserver` retains the small terminal fact in `PendingRequest` and
maps finite, nonnegative values into `RecordIngest::metric_overrides` for the two
catalog tags. This uses the existing explicit metric-injection seam and avoids
changing the positional `RecordIngest` wire layout.

New `MetricTag` variants append after the existing final discriminant so current
dense column/sketch identities do not shift. Their catalog rows are record
distributions with native nanoseconds and millisecond display units. They are
not HTTP trace fields and do not enter `RequestTrace`. No per-message vectors or
`Arc<Mutex<_>>` state is added to the hot path. Exact records, cellular record
shards, column-store folds, and sketch distributions then use their existing
paths.

Both values remain absent when their operands are missing, the operation did not
complete successfully, arithmetic overflows, either result is non-finite or
negative, or the dialect cannot prove a valid input-to-output timing window.
They are also absent when an equivalent HTTP/SSE operation is selected: sharing
an event decoder does not make an HTTP request a WebSocket message exchange.
Values are never substituted with zero, wrapped, saturated, or clamped. Invalid
measurement and failed inference remain distinct conditions even though neither
contributes a successful-request distribution sample.

The formulas are eligible only for a dialect whose measured input epoch
completes before its measured output epoch. A binding with truly interleaved
bidirectional input and output must define and correlate logical round
boundaries before enabling them. Until it does, the metrics stay absent;
request-wide subtraction on an uncorrelated duplex stream is misleading. This
rule is dialect policy, not a transport-order heuristic.

### Metric outputs

The native report exposes both optional metrics with unit `ms` and normal
request-weighted distribution statistics. It does not imply a round-weighted
population. Summary JSON/CSV and console omit the metric row when no eligible
record supplies a value. Per-record JSONL omits an absent key. Wide records CSV
and Parquet use the metric catalog effective for that AIPerf version and encode
ineligible records as empty/null cells, including HTTP, gRPC,
alternative-transport, and ineligible WebSocket runs; they never encode absence
as zero. Adding the two catalog entries intentionally adds two nullable columns
to newly produced wide artifacts; it does not promise byte-identical schemas
across AIPerf versions.

The first implementation does not project these values into OTLP. The current
OTLP accumulator recognizes a fixed set of GenAI duration metrics, and no
standard semantic-convention identity precisely describes these unpaired
application-event lag estimators. OTLP support requires a separately specified
instrument name, unit, buckets, attributes, and per-record accumulation path.

### Verification

- Clock-driven unit tests fix successful-flush and decoded-content instants and
  assert both formulas exactly, including unequal populations and fractional
  mean nanoseconds. Separate cases cover pre-send queue time and post-flush
  measurement.
- A large event-count test proves state remains scalar and exercises checked
  count/sum overflow behavior without brittle `size_of` assertions.
- Fragmented application messages with interjected Ping/Pong prove that wire
  framing and control traffic do not alter content samples.
- Missing send, missing content, failed flush, non-completed terminal,
  alternative-transport selection, invalid ordering, and
  interleaved-without-round-policy cases assert absence
  rather than zero, wrapping, saturation, clamping, or a negative metric.
- Multiple correlated logical operations complete out of order on one reused
  connection without cross-attribution. Connection death terminates all routed
  operations once, and bounded command-channel backpressure does not fabricate
  a successful send timestamp.
- Turn-serialized coverage proves that concurrent turns sharing one affinity key
  are serialized, the connection is reused only after terminal handling, a
  terminal acknowledgement does not enter the measured send population, and
  Ping/Pong continues during an otherwise idle receive loop. Duplex coverage
  proves outbound audio/control and cancellation progress while reads are
  pending and inbound events progress under outbound backpressure.
- Recovery coverage injects stale cached sockets, send failure, pre-terminal
  close, idle timeout, and missing continuation state. It proves recovery drops
  the old stream first, replays a self-contained request only before the first
  output-bearing backend event or after a classified continuation rejection,
  resets the metric epoch, omits stale continuation identity, and never
  duplicates partial output. WS-only dialects fail closed rather than silently
  changing protocols.
- A deterministic `aiperf-mock-server` WebSocket scenario fixes the input-message
  sequence and content-event delays. An end-to-end profile inspects raw
  per-record JSONL, wide CSV nulls, console, summary CSV, and native JSON for
  both metrics with a narrow real-transport overhead tolerance. Parquet nulls
  are asserted when that feature is enabled.
- Realtime-specific coverage asserts audio input/output facts, response content,
  terminal status, and errors. Other dialect tests assert their own content
  contract rather than inheriting unrelated audio requirements.
- Config validation and expansion cover `transport.type: websocket`, unsupported
  endpoint/transport pairs, `ws`/`wss` scheme enforcement, TLS verification,
  explicit/ambient proxy precedence with loopback exclusion, connection limits,
  and unsupported UDS/fallback combinations.
- Cold upgrade and warm reused-connection cases prove the two lag metrics exclude
  connection setup. WSS coverage exercises peer verification and
  `ssl_verify = false` against the deterministic TLS fixture.
- Multi-worker/cellular coverage proves distributions merge without mixing
  operation identities. Sketch coverage proves distributions remain available
  while per-record output is rejected by the existing sketch-mode contract.

## Source anchors

- WebSocket implementation: `rust/runtime/src/engine/ws_execution.rs` and
  `rust/runtime/src/transport/ws/{connector,dialect,driver}.rs`.
- Reusable prerequisites: `rust/runtime/src/transport/grpc/binding.rs`
  (bidirectional message binding), `rust/runtime/src/transport/http/sse/` (SSE
  decoder), `rust/runtime/src/body_plan.rs` (`BodyEmitter` materializer contract),
  `rust/runtime/src/dataset/segment.rs` (content handles), and
  `rust/runtime/src/dispatch/sink.rs` (`RequestObserver`).
- Placement and registration seams:
  `rust/runtime/src/engine/turn_execution.rs` (`WorkerSink`,
  `ExecutionSinkBuilder`) and `rust/runtime/src/engine/registry.rs`.
- Metric integration: `rust/runtime/src/metrics.rs` (`PendingRequest`,
  `NativeMetricsObserver`, `ObserverTee`),
  `rust/runtime/src/metrics_core/ingest.rs` (`metric_overrides`),
  `rust/runtime/src/metrics_core/{catalog,store,accumulator,report}.rs`, and
  `rust/runtime/src/export/`.
- Product verification: `rust/mock-server/src/`,
  `rust/e2e-tests/tests/common/`, and a dedicated
  `rust/e2e-tests/tests/test_websocket.rs`.
