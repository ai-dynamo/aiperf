<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WebSocket mock-server support

## Purpose

Provide a deterministic local target for developing and verifying AIPerf's
WebSocket transport and application-event lag metrics. The mock target models
protocol events, connection reuse, duplex progress, control traffic, and
recoverable failures. It is a test and benchmark target, not a general-purpose
WebSocket scripting server or a production protocol emulator.

This record is subordinate to [websocket-transport.md](websocket-transport.md):
the transport record defines client and metric semantics; this record defines
the server behavior needed to prove them.

## Built

`aiperf-mock-server` already provides the reusable substrate:

- The default Axum router is served through Hyper connections with upgrade
  support over cleartext TCP, rustls TLS, and Unix-domain sockets.
- `MockServerConfig` is the single Clap/serde configuration shared with
  child-process launch.
- `AppState` owns one `RealClockAnchor`; analytic waits use
  `aiperf_runtime::clock::sleep_ns` rather than Tokio's millisecond timer wheel.
- `LatencySimulator` supplies deterministic zero-jitter TTFT/ITL schedules and
  optional seeded jitter or scheduler contention.
- `MetricRecorder` tracks endpoint requests, in-flight work, usage, and
  latency. Existing HTTP and gRPC handlers share it.
- TLS uses the same router and advertises HTTP/1.1, so a WSS upgrade follows the
  existing certificate and `--tls-self-signed` paths.

The Axum router now conditionally exposes the turn and Realtime upgrade routes,
with a typed route codec, connection-local scenario state, absolute clock
scheduling, configured inbound message bounds, and a bounded metadata-only
capture store. The `--ludicrous-speed`, `--blocking`, and `--uring` engines do
not drive Hyper upgrades and cannot host this feature.

## Scope

The first implementation supports two test-only routes:

| Route | Connection model | Purpose |
| --- | --- | --- |
| `/mock/websocket/turns` | One in-flight turn per reusable connection | Persistent turn streaming, continuation, acknowledgement, and replay recovery |
| `/mock/websocket/realtime` | Duplex | Upload-then-receive transcription plus deliberately interleaved input/output |

Both routes use complete JSON application messages. The route handlers share a
typed scenario engine, timing scheduler, capture vocabulary, and terminal/error
policy, but each owns its event codec and connection state. The mock does not
pretend their schemas are interchangeable.

HTTP/SSE behavior remains on existing HTTP routes. The mock does not silently
convert a failed WebSocket operation to HTTP; an end-to-end fallback test starts
or selects the explicit HTTP target itself.

## Public configuration

WebSocket routes are disabled by default. These CLI fields also serialize into
the child-process configuration:

```text
--websocket-mode disabled|turn_serialized|realtime|both
--websocket-scenario normal|done_only|close_before_terminal|dirty_close_after_terminal|stale_reuse|reject_continuation|interleaved_realtime
--websocket-content-events <u32>                 default 1
--websocket-first-content-delay-ms <f64>         default 20
--websocket-content-interval-ms <f64>            default 5
--websocket-fragment-bytes <usize>               default 0
--websocket-control-before-content <none|ping|pong> default none
--websocket-capture-capacity <usize>             default 1024
--websocket-max-message-bytes <usize>            default 8388608
```

All durations must be finite and nonnegative. `content_events` must be positive
except under `done_only`, which synthesizes one content-bearing terminal event.
`fragment_bytes = 0` sends an unfragmented message; a positive value must be at
least four and fragments text messages into chunks no larger than that byte
count without splitting a UTF-8 code point. Capture capacity must be positive
when WebSocket mode is enabled. Maximum message bytes must be positive, and
fragment bytes cannot exceed it.

Scenarios are intentionally closed and validated:

- `stale_reuse` and `reject_continuation` require the turn-serialized route.
- `interleaved_realtime` requires the Realtime route.
- `both` accepts only scenarios shared by both routes; route-specific scenarios
  require their single-route mode.
- `close_before_terminal` emits no terminal event.
- `dirty_close_after_terminal` emits a complete terminal event, then drops the
  transport without a WebSocket close handshake.
- `--fast` zeros the two WebSocket delay fields along with existing latency
  fields, without changing the chosen scenario.
- WebSocket mode with `--ludicrous-speed`, `--blocking`, or `--uring` is rejected
  before binding. The first implementation also rejects `--processes > 1` so
  scenario state and capture assertions remain deterministic rather than
  child-local behind the L4 balancer.

## Scenario engine

`WebSocketScenarioConfig` validates authored configuration and lowers it into an
immutable `PreparedWebSocketScenario`. Each accepted connection owns a
`ConnectionScenario` with:

- a monotonic connection id allocated from `AppState`;
- the selected route and scenario;
- a turn counter and last completed response identity;
- whether the preceding connection is intentionally stale;
- a connection-local `Vec<WebSocketCaptureEvent>`; and
- no global lock on its application-message loop.

The scenario engine consumes typed facts from a route codec rather than raw JSON
field lookups:

```rust
enum MockClientEvent {
    StartTurn { request_id: String, continuation: Option<String> },
    ConfigureSession,
    AppendAudio { bytes: usize },
    CommitInput,
    RequestResponse { response_id: String },
    TerminalAck { response_id: String },
}
```

It produces typed server actions:

```rust
enum MockServerAction {
    SendText(Bytes),
    SendFragmentedText { payload: Bytes, max_fragment_bytes: usize },
    SendPing(Bytes),
    SendPong(Bytes),
    Close(Option<CloseFrame>),
    DropTransport,
}
```

Unknown JSON, an event illegal in the current state, an unexpected opcode, a
second in-flight turn on the serialized route, or an acknowledgement for the
wrong response receives one protocol error event and a policy close. The mock
does not panic or silently ignore invalid client behavior.

## Timing

Each logical operation samples a start time only after its final required input
message is decoded:

- the request event for the turn-serialized route; and
- input commit for upload-then-receive Realtime.

Content event `i` is scheduled at:

```text
input_complete + first_content_delay + i * content_interval
```

The handler computes absolute nanosecond targets from `AppState.clock_anchor`
and waits with `sleep_ns`. This prevents processing time or a late wake from
accumulating into later event spacing. Control frames are emitted immediately
before the first content event and do not shift its target.

The normal turn route emits a created event, `content_events` nonempty content
deltas, and one terminal event with deterministic usage. `done_only` emits no
delta and carries the generated content in the terminal envelope. Realtime
normal mode waits for input commit before output; `interleaved_realtime` emits
one output event after the first audio append and before commit so the client
must mark request-wide lag metrics ineligible.

## Connection and failure behavior

The turn-serialized handler accepts sequential turns on one connection and
rejects overlapping turns. A completed turn records its response identity. A
continuation is accepted only when it matches that identity.

Failure scenarios have exact boundaries:

- `stale_reuse`: turn one completes normally; the next request event fails at
  the transport boundary before any server application event for turn two.
- `reject_continuation`: turn one completes; turn two receives a typed
  continuation-state rejection before any output-bearing event, after which a
  full-history request without continuation succeeds.
- `close_before_terminal`: content events may be emitted, then the connection
  closes without terminal completion. This is not replay-safe after the first
  content event.
- `dirty_close_after_terminal`: success remains success because terminal
  completion precedes the unclean transport end.

Ping receives a matching Pong through the protocol stack. Unsolicited Pong and
configured control injection are captured but never treated as inference
content. A client close ends connection state without recording a failed
operation after a completed terminal.

## Fragmentation

Metric tests must prove that WebSocket wire frames are not application events.
Axum's high-level `WebSocket` API exposes reassembled messages and cannot author
fragment boundaries, so positive `websocket_fragment_bytes` selects a focused
raw-upgrade path. That path performs the validated WebSocket handshake over
Hyper's upgraded stream, then uses Tungstenite frame messages to write one text
message as initial/continuation fragments. Ping or Pong may be inserted between
fragments as RFC 6455 permits.

The ordinary unfragmented path stays on Axum extraction. Both paths call the
same route codec, scenario state machine, timing scheduler, capture finalizer,
and metric recorder; only wire emission differs.

## Captures and metrics

`WebSocketCaptureStore` is a bounded `VecDeque` under one `parking_lot::Mutex`.
Connections collect locally and append one completed `WebSocketCapture` at
terminal/close, so there is no shared lock per message. Oldest captures are
evicted at capacity.

A capture contains connection id, route, scenario, turn number, opcode, event
type, payload length, BLAKE3 payload digest, relative receive/send nanoseconds,
terminal classification, and close classification. It never stores full audio,
authorization headers, cookies, or query credentials. Unit tests access the
store directly. In a single-process run, `GET /mock/websocket/captures` returns
the sanitized snapshots for black-box tests; it returns 404 when WebSocket mode
is disabled.

`MetricRecorder` counts each logical operation, not each frame. It increments
in-flight on the operation's start event and releases it exactly once on
terminal, protocol failure, close, or task cancellation. Successful terminal
operations record existing request/usage/latency metrics under distinct
`mock_websocket_turns` and `mock_websocket_realtime` endpoint labels. Fragment,
Ping, Pong, acknowledgement, and capture events do not increment request count.

## Concurrency and resource bounds

Every connection runs in one Axum/Hyper task. Per-connection state is local.
The only shared mutations are existing recorder atomics, connection-id
allocation, and one capture append at connection end. Output content is
generated once per logical operation and reused by its delta/terminal encoder.

The server enforces:

- one in-flight operation on the turn-serialized route;
- a bounded number of queued outbound Realtime actions derived from the closed
  scenario, never an unbounded channel;
- the existing HTTP body limit policy for the upgrade request;
- the authored `websocket_max_message_bytes` limit on every decoded application
  message; and
- capture memory bounded by capacity and metadata-only payload summaries.

## Verification

- Config tests cover defaults, serde/Clap parity, `--fast`, every invalid
  mode/scenario combination, engine rejection, message-size limits, and
  multi-process rejection.
- Codec tests cover valid events, malformed JSON, wrong opcodes, illegal state
  transitions, overlapping serialized turns, mismatched acknowledgements, and
  continuation identity.
- Clocked scenario tests assert absolute content targets, event counts,
  done-only content, interleaving, control injection, and no cumulative drift.
- Wire tests use an independent client to prove UTF-8-safe fragmentation,
  interjected Ping/Pong, clean close, pre-terminal close, and dirty
  post-terminal transport drop over WS and WSS.
- Recovery tests prove the exact stale-reuse and continuation-rejection
  boundaries and distinguish replay-safe pre-output failure from partial output.
- Capture tests prove redaction, digest/length recording, bounded eviction,
  connection-local ordering, and single append at connection end.
- Recorder tests prove one request/in-flight lifecycle per logical operation and
  no accounting from frames or control traffic.
- Product e2e launches the standalone binary, drives both routes through native
  `aiperf profile`, and inspects per-record lag metrics, terminal status,
  response content, errors, CSV/JSON/console output, TLS verification, reuse,
  recovery, and ineligible interleaved Realtime behavior.

## Source anchors

- Router and state: `rust/mock-server/src/{app,state}.rs`.
- Listener and TLS upgrade support: `rust/mock-server/src/{listener,tls}.rs`.
- Deterministic timing: `rust/mock-server/src/latency.rs` and
  `rust/runtime/src/clock/`.
- Existing request accounting: `rust/mock-server/src/metrics.rs`.
- Configuration and process modes: `rust/mock-server/src/{config,main,balancer}.rs`.
- Client contract and product verification:
  `docs/specs/websocket-transport.md` and
  `rust/e2e-tests/tests/test_websocket.rs`.
