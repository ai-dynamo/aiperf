# aiperf_runtime::transport_http — Rust-native AIPerf HTTP client + timing recording

**Date:** 2026-07-10
**Author:** Anthony Casagrande (with Claude)
**Status:** Built as `aiperf_runtime::transport_http` (Clock-injected hyper stack). Two areas remain
narrower than this design: full h2 connection-reuse/multiplexing semantics and the complete
aiohttp-style trace field set. Post-send cancellation is built.

## 1. Summary

The Clock-injected hyper HTTP transport described here is realized as the
`aiperf_runtime::transport_http` module (formerly the standalone `aiperf-transport-http`
crate, now a module of the `aiperf-runtime` crate; code lives under
`rust/runtime/src/transport_http*`). It ports AIPerf's
Python `aiohttp`-based transport layer and its timing-*recording* machinery to
idiomatic, high-performance Rust. It reproduces the measurement behavior of the
Python `src/aiperf/transports/` package — streaming SSE inference over HTTP,
first-token (TTFT) capture, authoritative `usage` token counts, fine-grained
connection trace timing, request cancellation, and connection-reuse strategies
— without any of AIPerf's scheduling/credit subsystem or its ZMQ service mesh.
Transport-neutral scheduling and observation are reserved for `loadgen-core`;
`aiperf_runtime::transport_http` is the single Clock-injected hyper stack used by both
the CLI online path (`rust/runtime/src/http.rs`) and the graph benchmark.

"Timing logic" here means **timing recording** (per-request/per-token/per-chunk
measurement into a record), **not** load scheduling (interval generators,
phases, credit issuance, ramping) — those are explicitly out of scope.

Fidelity target is **behavioral parity**: same distributions, semantics, and
measured metrics as the Python transport; the RNG and byte layout need not
match Python bit-for-bit. Data models are **idiomatic Rust**, not a 1:1
transcription of the Pydantic classes.

**Hard constraint:** all time access goes through the `aiperf_runtime::clock` `Clock`
abstraction — never `Instant::now()`, `SystemTime::now()`, or raw
`tokio::time`. See §4.

## 2. Goals / Non-Goals

### Goals
- Port the transport leaf: send inference requests, stream SSE, and record
  precise timing into a `RequestRecord`.
- Full fine-grained connection trace parity (DNS / TCP / TLS / pool-wait /
  request-chunk-sent / response-chunk-received / headers timing).
- HTTP/1.1 **and** HTTP/2 (h2 over TLS via ALPN; h2c cleartext via
  prior-knowledge preface).
- Request cancellation (`cancel_after_ns`): cancel N ns after the request body
  is fully sent.
- Connection-reuse strategies: `Pooled` / `Never` / `StickyUserSessions`.
- Idiomatic Rust data models (enums, duration accessors, typed errors, builders).
- **Every timestamp and timer sourced from `Clock`**, enabling deterministic
  `SimClock` tests.
- Validation against the workspace-owned standalone `aiperf-mock-server` binary.

### Non-Goals (YAGNI)
- The `timing/` scheduling subsystem: interval generators (Poisson/Gamma/
  Constant/ConcurrencyBurst), phase orchestration, credit issuance, ramping,
  adaptive/SLA logic.
- Binary responses (`video/*`, `image/*`, `audio/*`, octet-stream).
- Multipart/form-data request bodies; the video submit→poll→download flow.
- The ZMQ message bus, worker services, and record-processor pipeline.
- Byte-exact RNG parity with Python.

## 3. Reference (Python source being ported)

Repo-relative to `/home/anthony/nvidia/projects/aiperf/ajc/rust`:
- `src/aiperf/transports/aiohttp_client.py` — `AioHttpClient`: `_request`,
  `post_request`, `_request_with_cancellation`, `create_tcp_connector`.
- `src/aiperf/transports/aiohttp_transport.py` — `AioHttpTransport`: `get_url`,
  `_dedup_path_overlap`, `send_request`, `ConnectionLeaseManager`. (Video and
  form-data paths intentionally not ported.)
- `src/aiperf/transports/aiohttp_trace.py` — the `TraceConfig` factory mapping
  aiohttp lifecycle events → `AioHttpTraceData` timestamps.
- `src/aiperf/transports/sse_utils.py` — `AsyncSSEStreamReader` incremental
  parser.
- `src/aiperf/transports/http_defaults.py` — `SocketDefaults`, `AioHttpDefaults`.
- `src/aiperf/transports/base_transports.py` — `BaseTransport`, `build_url`,
  `build_headers`, `FirstTokenCallback`.
- Models in `src/aiperf/common/models/`: `record_models.py`
  (`RequestRecord`, `RequestInfo`, `SSEMessage`, `SSEField`, `TextResponse`),
  `trace_models.py` (`BaseTraceData`, `AioHttpTraceData`, `*Export`),
  `error_models.py` (`ErrorDetails`).

Local Rust reference: `rust/runtime/src/clock` (the `Clock` contract) and
`rust/runtime` (existing `dynamo-aiperf` — HttpSink/SSE prior art, and the
`graph::runtime` `drive_real`/`drive_sim` execution model we mirror).

## 4. Clock abstraction (mandatory foundation)

Everything time-related is sourced from `aiperf_runtime::clock`:

```rust
pub trait Clock {
    fn now_ns(&self) -> i64;                                       // monotonic (real) / virtual (sim)
    fn sleep(self: Rc<Self>, duration_ns: i64) -> Pin<Box<dyn Future<Output=()>>>;
    fn next_event_time(&self) -> Option<i64> { None }
    fn advance_to(&self, _ns: i64) {}
    fn is_virtual(&self) -> bool { false }
}
```
- `RealClock` — monotonic `now_ns`, ns-precision `timerfd` sleeps (Linux) / tokio
  fallback. The live backend.
- `SimClock` — virtual discrete-event time, advanced explicitly; deterministic,
  no wall-clock waits.

**Rules the crate obeys:**
- No `Instant::now()`, `SystemTime::now()`, `Instant`, or raw `tokio::time` in
  crate code. Timestamps = `clock.now_ns()` (returns `i64` ns). Timers
  (`cancel_after`, send-timeout safety net) = `clock.clone().sleep(ns)`.
- The `Clock` is passed as `Rc<dyn Clock>` into the client/transport at
  construction and threaded to every component that stamps or waits.
- Because `Clock::sleep` takes `Rc<Self>` and yields a `!Send` future, the
  transport runs on a **current-thread tokio runtime under a `LocalSet`**; the
  hyper `Connection` driver future is spawned with `tokio::task::spawn_local`.
  This mirrors `aiperf_runtime::graph::runtime::{drive_real, drive_sim}`.
- **Execution modes:** live network I/O (against `aiperf-mock-server`) runs on
  `RealClock`. `SimClock` drives pure-logic tests where no real socket is
  involved (SSE parsing with clock-stamped arrivals, cancellation-timer logic,
  trace-duration math) — deterministic and wall-clock-free.

Wall-clock export (§5, `to_export`) is monotonic-only by default; if a caller
needs absolute wall timestamps they pass an explicit `(clock_ns, wall_ns)`
reference pair in — the crate never reads a wall clock itself.

## 5. HTTP stack

Built directly on **hyper 1.10.x core's low-level client** —
`hyper::client::conn::http1` and `hyper::client::conn::http2` (features
`client`, `http1`, `http2`; the latter pulls `h2 0.4.x`). We deliberately do
**not** use `hyper_util::client::legacy::Client`: its generic pool doesn't model
our sticky-per-session strategy, its opacity fights precise trace attribution,
and owning the connection layer is core to this crate. Same newest-lineage stack
as `aiperf-mock-server` (server: `hyper-util` `auto::Builder`) and reqwest 0.12
(which itself wraps this conn layer).

- **Connection manager (ours):** a keyed store of live connections, each
  established by us end-to-end so every phase is timestamped *directly* via
  `clock.now_ns()`:
  1. DNS resolve (`tokio::net::lookup_host` or swappable resolver), bracketed →
     `dns_lookup_start/end_ns`.
  2. TCP connect (`tokio::net::TcpStream`, socket opts applied) →
     `tcp_connect_start_ns`.
  3. rustls handshake for `https`, reading the negotiated ALPN protocol →
     `tcp_connect_end_ns` after handshake (TLS folded into the connect span, as
     aiohttp does).
  4. `http1::handshake` or `http2::handshake` → `(SendRequest, Connection)`; the
     `Connection` future is driven via `spawn_local`, the `SendRequest` handle
     retained/pooled per the reuse strategy.
  New connection → create timestamps recorded; reused handle →
  `connection_reused_ns`. Because we hold the handles, attribution is explicit.
- **TLS:** rustls via `tokio-rustls`, ALPN advertising `h2`, `http/1.1`. Roots
  from `webpki-roots` (or `rustls-native-certs`); SSL-verify toggle mirrors
  `AioHttpDefaults.SSL_VERIFY`.
- **Runtime:** tokio current-thread + `LocalSet` (per §4).

### 5.1 HTTP/2

- `HttpVersion { Auto, Http1Only, Http2PriorKnowledge }` selects which
  `conn::httpN::handshake` we call:
  - `Auto` — on `https://`, dispatch on the rustls-negotiated ALPN (`h2` →
    `http2::handshake`, else `http1`); on cleartext `http://`, HTTP/1.1.
  - `Http1Only` — always `http1::handshake`.
  - `Http2PriorKnowledge` — always `http2::handshake`, including cleartext (h2c
    preface), matching `curl --http2-prior-knowledge` /
    reqwest `.http2_prior_knowledge()`. Proven against `aiperf-mock-server` (accepts
    h2c; 800k-request h2c soak).
- Under h2, streams multiplex over one connection: connection-level trace fires
  once at creation; subsequent streams record `connection_reused_ns`. Reuse
  strategies operate at the connection layer (pool disabled / pool-of-1 / shared),
  decoupled from per-request concurrency as h2 intends. **Narrower than design:**
  h2c prior-knowledge dispatch is built and soak-proven, but the full h2
  connection-reuse and multiplexing semantics described here remain broader than
  what the current `HttpTransport` facade exposes.

## 6. Idiomatic model layer (`src/models/`)

All timing fields are **`i64` Clock-nanoseconds** (`clock.now_ns()` readings),
optional ones `Option<i64>`. Durations are computed on demand as `Option<i64>`
ns (or `Option<Duration>`) accessor methods, not materialized fields. Errors are
plain enums with hand-written `Display` (no `thiserror`), per the workspace
error-handling convention.

| Python | Rust (idiomatic) |
|---|---|
| `list[SSEMessage \| TextResponse \| BinaryResponse]` union | `enum Response { Sse(SseMessage), Text(TextResponse) }`; `responses: Vec<Response>` (Binary out of scope) |
| `SSEField{name: str, value: str\|None}` | `struct SseField { name: SseFieldName, value: Option<String> }`; `enum SseFieldName { Data, Event, Id, Retry, Comment, Other(String) }` |
| `SSEMessage{perf_ns, packets}` + `parse()` | `struct SseMessage { perf_ns: i64, packets: Vec<SseField> }`; `SseMessage::parse(raw: &str, perf_ns: i64)` |
| `TextResponse{perf_ns, text, content_type}` | `struct TextResponse { perf_ns: i64, text: String, content_type: Option<String> }` |
| `AioHttpTraceData` (~30 `Optional[int]` + `computed_field` props) | `struct TraceData` with `Option<i64>` capture points; methods `sending()/waiting()/receiving()/duration()/blocked()/dns_lookup()/connecting() -> Option<i64>`; `to_export(reference)` → wall-clock `TraceExport` |
| `ErrorDetails{type: str, code, message}` | `struct ErrorDetails { kind: ErrorKind, code: Option<u16>, message: String }`; `enum ErrorKind { Http, Sse, Cancelled, Connect, Timeout, Other }` |
| `RequestRecord` (Pydantic) | `struct RequestRecord { start_ns: i64, end_ns: Option<i64>, recv_start_ns: Option<i64>, status: Option<u16>, responses: Vec<Response>, error: Option<ErrorDetails>, trace: Option<TraceData>, request_headers: Option<HeaderMap>, cancellation_ns: Option<i64> }`; methods `was_cancelled()`, `has_error()`, `is_valid()` |
| `RequestInfo` + kwargs dicts | typed `RequestConfig`/builder: url(s), headers, params, `cancel_after_ns`, `url_index`, `is_final_turn`, `correlation_id`, `request_id`, `reuse: ConnectionReuseStrategy` |
| `ConnectionReuseStrategy` str-enum | `enum ConnectionReuseStrategy { Pooled, Never, StickyUserSessions }` |

`TraceData` retains the aiohttp field set (all `Option<i64>` Clock-ns): pool
wait, TCP connect start/end, connection reused, DNS lookup start/end + cache
hit/miss, request send start / headers-sent / send-end, request chunk
count+bytes (+ optional `Vec<(i64, usize)>`), response status/reason/headers,
response receive start/end, response chunk count+bytes (+ optional per-chunk
vec), error timestamp, socket info (local/remote ip+port). `to_export(reference)`
produces wall-clock instants and the k6/HAR-compatible durations (`sending`,
`waiting`, `receiving`, `duration`, `blocked`, `dns_lookup`, `connecting`).
**Narrower than design:** this is the full aiohttp-style field set as intended;
the current trace model and writers populate a subset of it, so complete
aiohttp-style trace field parity remains a target.

## 7. Client / transport layers

### 7.1 `SseStreamReader` (`src/sse/reader.rs`)
Incremental parser mirroring `AsyncSSEStreamReader`. Consumes a
`Stream<Item = Result<Bytes, _>>`; buffers in a `Vec<u8>`; on each inbound chunk
captures the arrival time via `clock.now_ns()` (drives TTFT/ICL); scans `\n\n`
first then `\r\n\r\n` with a 3-byte back-scan for split delimiters; compacts once
per chunk (offset tracking, not per-message memmove); yields
`SseMessage::parse(msg, arrival_ns)`; handles JSON-continuation lines; flushes a
trailing delimiter-less message at stream end. `inspect_message_for_error` raises
`ErrorKind::Sse` on an `event: error` message.

### 7.2 `HttpClient` (`src/client/http_client.rs`) ≈ `AioHttpClient`
- Holds `Rc<dyn Clock>` and the connection manager (§5). Acquires a `SendRequest`
  handle per the reuse strategy (new connection → create timestamps; reused →
  `connection_reused_ns`), writing all connection-phase timings straight into the
  record's `TraceData`.
- `request(method, url, headers, body, opts) -> RequestRecord`:
  - Sets `start_ns = clock.now_ns()`; fills `TraceData` connection timings.
  - Sends via the `SendRequest` handle; on response headers received, records
    status/reason/headers + `response_headers_received_ns` + socket info.
  - Streaming (`content-type: text/event-stream`, POST): wraps the response body
    to timestamp each transport chunk (`response_receive_start/end_ns`, counts,
    bytes), feeds `SseStreamReader`, appends `Response::Sse`, fires the
    first-token callback with `ttft_ns = first_message.perf_ns - start_ns`.
  - Non-streaming: reads the full body → `Response::Text`.
  - Maps HTTP ≥ 300 → `ErrorKind::Http`; transport failure → `Connect`/`Timeout`/
    `Other`; cancellation → `Cancelled` (499).

### 7.3 Cancellation (`src/client/cancellation.rs`) ≈ `_request_with_cancellation`
Post-send cancellation is built. HTTP cancellation is armed from the captured
request-body `SendCompletion` signal, so the configured cancellation delay starts
only after the outbound body has been fully sent — not from request submission.
An outbound-body wrapper (`SendCompletion`) fires a "request fully sent" signal
once the body is fully written; the in-flight request races that signal. A request
that finishes or fails before send-completion is returned normally. Once the
send-completion signal fires, the request races `clock.sleep(cancel_after_ns)`
anchored to the captured send time. On timer win: abort the
request, record `cancellation_ns` + `ErrorKind::Cancelled` (499). Both the h1 and
h2 paths key off that body-wrapper completion signal rather than racing the entire
dispatch future from submission time; the "body fully sent" point is well-defined
on both. Timer starts at send-complete, not request-start — matching Python. Under
`SimClock` this is fully deterministic.

### 7.4 `HttpTransport` (`src/transport/…`) ≈ `AioHttpTransport`
- `build_url` (`transport/url.rs`): scheme-fill (case-insensitive `://` check),
  path join with `_dedup_path_overlap` (empty sub-path; full-suffix already
  present; `/v1` + `v1/…` collapse), then query-param merge (endpoint params
  override).
- `build_headers` (`transport/headers.rs`): base (`User-Agent`) + correlation/
  request-id (session-header override) + endpoint headers + per-turn extra
  headers + transport headers (`Accept`, `Content-Type`).
- `ConnectionLeaseManager` (`src/client/pool.rs`): `StickyUserSessions` →
  `HashMap<correlation_id, Connection>` (pool-of-1), released on final turn /
  cancellation / error; `Never` → pool-disabled per request; `Pooled` → shared.
- `send_request(config, payload, first_token_cb) -> RequestRecord`.

### 7.5 Endpoint binding (`transport/endpoint_binding.rs`)
The module depends on `aiperf_runtime::endpoints` only at the translation boundary:
`transport/endpoint_binding.rs` defines the object-safe `HttpEndpointBinding` and
its metadata-driven implementation. The binding lowers canonical endpoint JSON to
HTTP URL/body/lifecycle policy and decodes HTTP/SSE responses back into
`ServerResponse`; `aiperf` retains endpoint parsing, observer emission, usage
aggregation, and scheduled outcomes. Future gRPC and WebSocket transports are
peer modules with their own bindings, not endpoint forks.

### 7.6 Config / defaults (`src/config/defaults.rs`)
Port `SocketDefaults` (TCP_NODELAY, keepalive idle/intvl/cnt, SO_RCVBUF/SNDBUF
with ENOBUFS halving fallback, TCP_QUICKACK, TCP_USER_TIMEOUT — Linux-gated via
`cfg`) applied to the socket before connect, and `AioHttpDefaults` (connection
limit, DNS cache TTL, keepalive timeout, family/IP version, SSL verify) as typed
config with matching default values.

## 8. Module layout

The transport lives as the `aiperf_runtime::transport_http` module under
`rust/runtime/src/transport_http/`:

```
rust/runtime/src/transport_http/
  mod.rs
  models/{mod,record,response,sse,trace,error,request}.rs
  client/{mod,http_client,connection,resolver,pool,cancellation}.rs
  transport/{mod,http_transport,url,headers,endpoint_binding,body,polling,inline_media}.rs
  sse/reader.rs
  config/defaults.rs
```

Dependencies: `aiperf_runtime::clock` (the Clock), `tokio` (current-thread + macros +
net + io-util), `hyper` (features `client`, `http1`, `http2`), `hyper-util`
**only** for `rt::{TokioIo, TokioExecutor}`, `http`, `http-body-util`,
`tokio-rustls`, `rustls`, `webpki-roots`, `bytes`, `futures`, `serde`,
`serde_json`, `tracing`, `url`, `socket2`.

## 9. Validation — integrate against `aiperf-mock-server`

`aiperf-mock-server` (at `rust/mock-server`) is an
OpenAI-compatible mock server: `/v1/chat/completions` (streaming SSE),
`/v1/completions`, `/v1/models`, `/health`, with `--fast`, `--ttft`/`--itl`,
`--error-rate`, `--host`/`--port`, and h1+h2 (h2c prior-knowledge) support.

- **Fixture** (`tests/mock_fixture.rs`): spawn the binary as a child on a
  test-chosen free `127.0.0.1` port, poll `GET /health` until ready, kill on
  drop. Binary via `AIPERF_MOCK_RS_BIN` env (fallback: the current workspace
  target directory, then `PATH`); absent → skip with a clear message. These
  live tests run on **`RealClock`**.
- **Scenarios (RealClock):**
  - `--fast` streaming happy path: `responses` populated, `usage` counts
    captured, TTFT = first-SSE `perf_ns` − `start_ns`, token count == content-
    delta count.
  - `--ttft 50 --itl 10`: TTFT ≈ configured; inter-token gaps ≈ ITL, monotonic;
    trace ordering `send_start ≤ headers_sent ≤ send_end ≤ recv_start ≤ recv_end`.
  - `--error-rate 1.0`: non-2xx → `ErrorKind::Http`.
  - `/v1/completions` non-streaming → `Response::Text` + JSON parse.
  - Reuse strategies via captured `TraceData.local_port` (HTTP/1.1): `Never` →
    distinct ephemeral port per request; `Sticky`/`Pooled` → shared port.
  - h2c multiplexing (`Http2PriorKnowledge`): N concurrent streams share one
    `local_port`; connection-level trace captured once.
  - `cancel_after_ns`: mid-stream cancel → `was_cancelled()` +
    `ErrorKind::Cancelled` (499), request confirmed sent first.
- **Pure unit tests (SimClock / no server):** SSE parser edge cases (split
  delimiters, CRLF, continuations, comments, `[DONE]`, error events, trailing
  message) with clock-stamped synthetic arrivals; cancellation-timer logic under
  `SimClock` (deterministic, no wall wait); URL dedup/query-merge; header
  composition; trace-duration math; wall-clock export conversion from an explicit
  reference.

## 10. Success criteria
- `cargo build`, `cargo test`, `cargo clippy` clean.
- Zero direct `Instant::now()` / `SystemTime::now()` / `tokio::time` in crate
  code (all time via `Clock`) — enforced by a grep check in CI/tests.
- A populated `RequestRecord` (responses + trace + timings) produced against
  `aiperf-mock-server` on `RealClock`.
- SSE parser passes the Python parser's edge cases under `SimClock`.
- Reuse strategies observably distinct via captured trace (`local_port`).
- h2c prior-knowledge streaming completes against the mock.

## 11. Open questions / risks
- **DNS timing:** we own connect, so resolution is a bracketed
  `tokio::net::lookup_host` (swappable resolver) → `dns_lookup_start/end_ns`
  directly. `127.0.0.1` resolution is trivial; real-host DNS captured the same way.
- **TLS handshake span:** aiohttp folds TLS into `tcp_connect_*`; we do the same
  (record `tcp_connect_end_ns` after the rustls handshake) rather than a separate
  TLS span.
- **SimClock vs real sockets:** `SimClock` can't advance real network/tokio-io
  timers, so integration against the live mock uses `RealClock`; `SimClock` is
  reserved for pure-logic time-dependent tests. hyper's own internal timers
  (h2 keepalive/pings) are left at defaults and are irrelevant on localhost.
- **h2 request-sent signal for cancellation:** the "body fully sent" point
  differs on h2; the cancellation timer keys off the body-wrapper completion in
  both h1 and h2, so it stays well-defined.
