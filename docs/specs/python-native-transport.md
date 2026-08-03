<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python native transport

## Purpose

The Python `aiperf` package dispatches every inference request through aiohttp.
That path cannot measure sub-millisecond inter-token latency accurately: the
timestamp is taken after the bytes have traversed the kernel, aiohttp's
`StreamReader`, an `iter_any()` future resolution, an event-loop schedule, and two
async-generator resumes. At 1 ms ITL under worker concurrency, event-loop
scheduling jitter is a material fraction of the quantity being measured.

This record states the seam that lets the native hyper stack
([http-transport.md](http-transport.md)) serve one Python request per call, the
widened transport contract that keeps aiohttp and the native client
interchangeable, and the packaging consequence of shipping a CPython extension in
an artifact whose defining property is that it links no CPython ABI
([wheel-packaging.md](wheel-packaging.md)).

The unit of crossing is one **request**, not one token. Per-token Python is what
destroys the measurement; one GIL reacquisition per request is not.

The implementation is a **vendored lightweight crate in the Python product**, not
a dependency on `aiperf-runtime`. This repository is the full Rust rewrite and
holds the reference implementations; the Python product carries small, standalone
Rust implementations of the pieces its own seams need. The two are related by
provenance, not by a build edge — no Cargo path or git dependency crosses between
them. This record states which reference code is portable verbatim, which is
coupled to rewrite-only abstractions and must be reimplemented, and what the
lightweight crate is permitted to leave out.

## Built

### The Python transport seam is already replaceable

`src/aiperf/transports/base_transports.py` defines `TransportProtocol` (a
`@runtime_checkable` `Protocol`) and `BaseTransport` (an ABC). Selection is by
plugin id: `src/aiperf/workers/inference_client.py` resolves
`plugins.get_class(PluginType.TRANSPORT, str(model_endpoint.transport))`, and
`detect_transport_from_url` matches a URL scheme against each registered entry's
`url_schemes` metadata. `src/aiperf/plugin/plugins.yaml` registers one entry —
`http` → `AioHttpTransport`, `{transport_type: http, url_schemes: [http, https]}`.

A second implementation is therefore a registry entry plus a class. No new
selection mechanism is required.

`BaseTransport` supplies concrete `build_headers` and `build_url`. `build_headers`
holds the session-affinity rules — the `--session-header` rename, and the additive
`X-Session-ID`, `X-SMG-Routing-Key`, `X-Dynamo-Session-ID` /
`X-Dynamo-Parent-Session-ID` headers, each stripped case-insensitively before
being reapplied as authoritative. Only `get_url` and `send_request` are abstract.

### What the caller does around the transport

`InferenceClient._send_request_to_transport` canonicalizes the payload to bytes
before dispatch (`request_info.payload_bytes = orjson.dumps(...)`), so the
transport receives wire bytes rather than a dict. Multipart endpoints
(`RequestContentType.MULTIPART_FORM_DATA`) are the exception and receive the
structured dict.

After dispatch, `Worker._populate_response_metrics` calls
`endpoint.extract_response_data(record)`, which JSON-parses every response chunk,
then derives four values onto the `CreditReturn`: `content_perf_ns`,
`request_latency_ns`, `output_sequence_length` (from `usage.completion_tokens`),
and `inter_token_latency_ns`.

`Worker._phase_needs_first_token_callback` gates the mid-request callback: it is
true only when `phase.prefill_concurrency is not None`, or when adaptive-scale SLA
filters require first-token observation. Concurrency and request-rate phases
without prefill concurrency never fire it.

### What the reference implementation provides, and what of it ports

The hyper stack owns wire I/O and timing recording, with all time access routed
through `Clock`. `transport::reduce::reduce_parsed_response` absorbs usage, data,
and endpoint metrics and emits first-token, output-token, usage, and terminal
observer events; `transport::measure::{WorkerMeasurement, measure_dispatch}` is
the shared measurement loop. Together these produce the same four values the
Python worker derives.

Coupling to rewrite-only abstractions is not spread through the transport; it is
concentrated in the layer that binds a transport to the worker pool. Measured per
subdirectory rather than per module:

| Reference component | Lines | Foreign imports | Disposition |
|---|---|---|---|
| `clock/` | 603 | `graph::runtime::RunOutcome` | vendor verbatim |
| `transport/core/` minus `dispatch.rs` | 1,072 | none | vendor verbatim |
| `transport/http/{client,config,models,sse}` | 3,777 | none | vendor verbatim |
| `transport/http/transport/` | 2,612 | `endpoints` (1 site) | vendor verbatim |
| `endpoints/` | 10,465 | `body_plan` (1) | vendor verbatim |
| `body_plan.rs` | 627 | `dataset::{segment,materialize,error}` | vendor verbatim |
| `dataset/{segment,materialize,error}.rs` | 1,401 | `dataset::model::MediaKind` | vendor verbatim |
| `transport/core/dispatch.rs` | 369 | `dispatch`, `metrics*`, `multiturn`, `scheduled` | replaced |
| `transport/http/sink.rs` + `sink/` | 2,090 | same | replaced |

Roughly 20,500 lines vendor unchanged; about 2,450 are replaced by the PyO3
layer. Every dependency chain terminates: `clock`'s single foreign symbol is a
one-field struct (`RunOutcome { deadlocked: bool }`), `endpoints`' is `BodyPlan`,
and the dataset substrate under it bottoms out at one `MediaKind` enum.

The two replaced files are precisely the worker-sink binding — the role Python's
own worker plays. `transport/core/dispatch.rs` and `transport/http/sink*` are
where `dispatch::sink`, `metrics_core`, `multiturn`, and `scheduled` enter; the
wire path beneath them imports none of it.

`TraceData` is also a superset of Python's `BaseTraceData` + `AioHttpTraceData`,
adding a TLS span split out from TCP connect and the response status code and
reason.

The reference sink is `!Send` by construction — `WorkerSink` is
`#[async_trait(?Send)]`, `Clock` is `Rc<dyn Clock>`, and `RequestObserver` carries
no `Send`/`Sync` supertrait — and `ExecutionSinkBuilder` (`Send + Sync + 'static`)
exists to construct it inside a target thread's reactor
([execution-model.md](execution-model.md)). The vendored crate inherits the
constraint that a reactor-bound client stays on its thread. It does not vendor the
builder, because it has one caller and no pool to place sinks into.

### A drift in the Python seam

`TransportProtocol.send_request` declares `(self, request_info, payload)`.
`BaseTransport.send_request` declares
`(self, request_info, payload, *, first_token_callback=None)`. Because
`@runtime_checkable` `isinstance` checks verify method presence and not
signatures, the divergence is not detected at runtime.

## Future requirements

### Crate shape

The crate mirrors the reference tree so vendored files keep their paths and stay
diffable against it: `clock/`, `transport/core/`, `transport/http/`, `endpoints/`,
`body_plan.rs`, and the three `dataset/` files. Vendored files are copies, not
adaptations — a file that needs editing to compile is a file to reconsider
vendoring.

"Lightweight" describes the **new-code surface**, not the line count. The crate is
large because copying is cheaper than rewriting an 8,500-line hyper stack, and
because a verbatim copy can be re-diffed against the reference when the rewrite
moves. What is actually written is the ~2,450-line replacement for the worker-sink
binding: a `lib.rs` holding the `#[pyclass]`, the job and result types, and the
thread shim.

The `Clock` seam is vendored intact, `SimClock` included. The Python product has
no immediate use for virtual time, but stripping the seam would edit
`Clock::now_ns()` call sites throughout the wire path and permanently fork every
file it touches from its reference. Keeping it costs 603 lines and preserves the
diff.

`body_plan.rs` and the `dataset/` substrate under it exist only to satisfy
`format_payload` on the `PreparedEndpoint` trait, which this path never calls —
`InferenceClient` canonicalizes the payload to bytes before dispatch. They are
vendored rather than stubbed for the same reason: a stub would fork
`endpoints/`'s 10,465 lines from the reference to save 2,028.

### One abi3 extension, one long-lived client object

A `#[pyclass]` constructed once per Python worker process holds the connection
pool for that process's lifetime. A per-call client would force a TCP and TLS
handshake per request.

The client is reactor-bound and does not cross threads. The pyclass holds an mpsc
`Sender<Job>`, which is `Send`; construction spawns one OS thread that builds the
client inside its own `current_thread` runtime and services jobs.

One client per process is the whole concurrency story: Python's `workers > 1` is
already N OS processes with N event loops, so each process owns exactly one
client, one thread, one runtime. The crate never needs to shard, and inherits no
`workers == 1` assertion because it has no multi-worker path to guard.

The GIL is released for the whole request.

### Completions are batched, never resolved on the transport thread

A `Job` carries an integer id, not `Py<PyAny>` handles. The transport thread parks
finished requests as plain Rust data and schedules **one** drain when none is
pending; the loop thread — which already holds the GIL — builds every response in
a single pass and resolves the futures Python owns.

Resolving a future on the transport thread instead is the obvious design and it
does not work. Building a Python object requires the GIL, so taking it per
request makes the transport thread and the event loop trade the interpreter at
its switch interval (5 ms by default). Batching removes that: a burst of
completions costs one acquisition rather than one each.

### Frames cross as tuples, not as a serialized blob

`NativeResponse` hands back `(perf_ns, [(name, value), ...])` per frame. PyO3
converts that shape directly, so Python builds its `SSEMessage` objects in one
pass with nothing to parse.

An earlier design serialized frames to JSON in Rust and parsed them back with
`orjson` in Python, on the theory that it avoided constructing per-token objects.
It did not — the objects still had to exist downstream — and it added two full
passes over every frame on top of the SSE decode that had just produced them.
Measurement settled it: materializing 260,000 frames costs nothing (479 req/s
untouched against 491 req/s fully materialized), while removing the JSON round
trip was worth 502 → 665 req/s on its own.

### Measured behavior

Against `aiperf-mock-server --fast --processes 8`, 40,000 requests, 16 workers,
ISL and OSL 128, zero errors on every run:

| Concurrency | Transport | req/s | tok/s | TTFT (ms) |
|---|---|---|---|---|
| 256 | `native_http` | 8,077 | 982,305 | 1.07 |
| 256 | `http` | 5,916 | 719,549 | 18.87 |
| 512 | `native_http` | 10,973 | 1,334,868 | 2.51 |
| 512 | `http` | 7,612 | 926,132 | 26.88 |
| 1024 | `native_http` | 11,153 | 1,357,472 | 2.14 |
| 1024 | `http` | 7,802 | 949,539 | 52.39 |

The throughput margin is 1.37–1.43×. The TTFT margin is 17.6× at 256 and 24.5× at
1024, and it widens with load because against a server that answers instantly
TTFT is measuring client-side scheduling delay — the quantity moving the timestamp
off the event loop removes, and the one that grows as the loop gets busier.

Accuracy is what the crate exists for, and the performance work did not disturb
it. Against the analytic server (`ttft=100ms`, `itl=10ms`, both jitter
coefficients zero), through `aiperf profile`: TTFT 100.28 ms, time-to-second-token
9.956 ms, request latency 290.21 ms against an analytic 290 ms. Output sequence
length agrees with the aiohttp transport to 0.01 tokens over 20,000 requests.

Benchmark the optimized profile, not a debug build. Debug `serde_json` alone
accounts for a 3× throughput difference in the per-frame reduction, which is
enough to invert the comparison against optimized C and read as a regression that
is not there.

### The widened transport contract

The seam must stay symmetric: aiohttp and the native client must both satisfy it,
and neither may be privileged. Returning a `RequestRecord` whose `responses` is a
`list[SSEMessage]` would defeat the purpose — constructing N `SSEMessage` +
N `SSEField` objects through PyO3 is slower than constructing them in Python, so
the allocation cost would move across the boundary rather than disappear.

`RequestRecord` gains one optional, self-describing field — a `ReducedOutcome`.
The transport returns two things:

- a reduced-outcome struct carrying the four derived values plus TTFT, status,
  and error;
- the response frames as `(perf_ns, [(name, value), ...])` tuples, which PyO3
  converts directly.

`Worker._populate_response_metrics` branches once per request on the presence of
the reduced outcome, not per token. Absent it, the existing
`extract_response_data` path runs unchanged, so `AioHttpTransport` requires no
modification and remains a conforming implementation. A transport declares the
capability in its `TransportMetadata` for fail-closed validation; the worker
branches on the record, so a mixed or replayed record set stays correct.

`TransportProtocol.send_request` is corrected to match `BaseTransport`.

### Division of labor

Python composes headers and the URL; the native client performs the wire
exchange. `NativeHttpTransport` subclasses `BaseTransport` and overrides only
`get_url` and `send_request`, inheriting `build_headers` unchanged. Reimplementing
the session-affinity header rules in Rust would create two divergent copies of
logic whose correctness is not locally checkable.

### Trace surface

The boundary carries `TraceData`'s clock-ns fields plus the
`TraceReference { clock_ns, wall_ns }` pair. Python's existing
`BaseTraceData.to_wall_clock()` performs the conversion — it is driven by exactly
that pair (`reference_perf_ns` / `reference_time_ns`), so the Python trace path is
untouched. `TraceData::to_export` is not used on this path: `TraceExport` drops
fields (`tcp_connect_start`/`end`, both TLS spans, pool-wait bounds, DNS lookup
bounds, `request_headers_sent`, `request_send_end`, `response_headers_received`,
`error_timestamp`, `local_ip`/`remote_ip`) that the Python model carries.

### Fail-closed envelope

Vendoring `endpoints/` whole means the envelope is the reference registry's, not a
reduced subset: the dialects `HttpEndpointBinding` binds are the dialects
available. Refusal is at selection time, not per request, and a refused run
proceeds on `AioHttpTransport`.

The Python product consequently holds two endpoint implementations —
`src/aiperf/endpoints/` and the vendored `endpoints/`. On this path they split by
direction rather than overlap: Python owns request construction (it formats and
canonicalizes the payload to bytes before dispatch), and the vendored side is
reached only for response decoding. The duplication is real and is the price of
keeping the wire path verbatim; the alternative is a hand-written reducer that
diverges from the reference silently.

Connection tracing is not a refusal condition — the vendored trace is the richer
of the two.

### Packaging

The wheel's stated property is `py3-none-<platform>`: platform-specific for the
ELF, interpreter-agnostic because nothing links a CPython ABI, one wheel across
`requires-python = ">=3.11,<3.14"`. A CPython extension invalidates that tag.

Building with `pyo3/abi3-py311` preserves the one-wheel property as
`cp311-abi3-<platform>`. `platform_tag_for()` in `tools/wheel_repack.py` composes
the tag string and is the single site that changes; `glibc_versions()` and
`manylinux_tag()`, which read the ELF's `.gnu.version_r` table to derive the
platform floor, are unaffected. Declining abi3 multiplies the `nightly.yml` matrix
by the supported interpreter count.

## Source anchors

- `src/aiperf/transports/base_transports.py` (`TransportProtocol`,
  `BaseTransport`, `FirstTokenCallback`, `build_headers`, `build_url`).
- `src/aiperf/transports/{aiohttp_transport.py,aiohttp_client.py,sse_utils.py}`
  (the aiohttp implementation and its SSE read loop).
- `src/aiperf/workers/{inference_client.py,worker.py}` (transport selection,
  payload canonicalization, `_populate_response_metrics`,
  `_phase_needs_first_token_callback`).
- `src/aiperf/plugin/plugins.yaml` (`transport:` registry).
- `src/aiperf/common/models/trace_models.py` (`BaseTraceData`,
  `AioHttpTraceData`, `to_wall_clock`).
- `tools/wheel_repack.py` (`platform_tag_for`, `glibc_versions`,
  `manylinux_tag`, `rewrite_wheel_tag`).

Reference implementations in this repository. The vendored crate copies these; no
build edge connects the two trees.

- `rust/runtime/src/clock/` (the seam, vendored intact including `SimClock`).
- `rust/runtime/src/transport/core/` except `dispatch.rs`.
- `rust/runtime/src/transport/http/` except `sink.rs` and `sink/`.
- `rust/runtime/src/endpoints/`; `rust/runtime/src/body_plan.rs`;
  `rust/runtime/src/dataset/{segment.rs,materialize.rs,error.rs}`.
- `rust/runtime/src/transport/core/dispatch.rs` and
  `rust/runtime/src/transport/http/sink*` — the worker-sink binding the PyO3
  layer replaces, and where every foreign import in the wire path originates.
- `rust/runtime/src/engine/turn_execution.rs` (`WorkerSink`,
  `ExecutionSinkBuilder`, `build_native`) — the threading constraint the crate
  inherits, without the pool abstractions.
