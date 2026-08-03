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

Portability of that reference code to a standalone crate divides sharply along
whether a file depends on rewrite-only abstractions:

| Reference file | Lines | Non-`std` dependencies | Disposition |
|---|---|---|---|
| `transport/core/trace.rs` | 243 | none | vendor verbatim |
| `transport/core/sse.rs` | 178 | `smallvec` | vendor verbatim |
| `transport/reduce.rs` | 276 | `dispatch::sink`, `endpoints`, `scheduled` | reimplement |
| `transport/measure.rs` | 138 | `clock`, `dispatch`, `metrics`, `metrics_core` | reimplement |
| `transport/http/` | 8,497 | `Clock`, endpoint registry, dispatch seam | reimplement |

`TraceData` and the SSE framer are the whole of what transfers unchanged: 421
lines carrying one external crate between them. `TraceData` is a struct plus
`diff` arithmetic and derived k6/HAR duration methods with no imports at all.

`TraceData` is also a superset of Python's `BaseTraceData` + `AioHttpTraceData`,
adding a TLS span split out from TCP connect and the response status code and
reason.

The reference sink is `!Send` by construction — `WorkerSink` is
`#[async_trait(?Send)]`, `Clock` is `Rc<dyn Clock>`, and `RequestObserver` carries
no `Send`/`Sync` supertrait — and `ExecutionSinkBuilder` (`Send + Sync + 'static`)
exists to construct it inside a target thread's reactor
([execution-model.md](execution-model.md)). The lightweight crate inherits the
constraint that a reactor-bound client stays on its thread, but not the
abstractions: it has one transport, one clock, and no registry to dispatch
through.

### A drift in the Python seam

`TransportProtocol.send_request` declares `(self, request_info, payload)`.
`BaseTransport.send_request` declares
`(self, request_info, payload, *, first_token_callback=None)`. Because
`@runtime_checkable` `isinstance` checks verify method presence and not
signatures, the divergence is not detected at runtime.

## Future requirements

### Crate shape

Five modules, no dependency on `aiperf-runtime`:

- `trace.rs` — vendored verbatim.
- `sse.rs` — vendored verbatim.
- `client.rs` — a hyper client that timestamps at read-return and populates
  `TraceData`.
- `reduce.rs` — a minimal reducer over the SSE dialects the crate claims, not the
  registry-coupled reference.
- `lib.rs` — the `#[pyclass]`, the job and result types, and the thread shim.

The crate omits what the Python product has no use for. There is no `Clock` seam:
the product has no `SimClock` and no deterministic-replay requirement, so time
reads go to a monotonic clock directly. There is no endpoint registry, no
`ExecutionSinkBuilder`/`WorkerSink` pair, no `RequestObserver`, and no dispatch
seam — one transport with one caller needs none of them. Reproducing them would
import the rewrite's extensibility cost without its extensibility requirement.

### One abi3 extension, one long-lived client object

A `#[pyclass]` constructed once per Python worker process holds the connection
pool for that process's lifetime. A per-call client would force a TCP and TLS
handshake per request.

The client is reactor-bound and does not cross threads. The pyclass holds an mpsc
`Sender<RequestJob>`, which is `Send`; construction spawns one OS thread that
builds the client inside its own `current_thread` runtime and services jobs.
Results return over a oneshot as plain data, and the Python worker's asyncio
future is resolved through `call_soon_threadsafe`.

One client per process is the whole concurrency story: Python's `workers > 1` is
already N OS processes with N event loops, so each process owns exactly one
client, one thread, one runtime. The crate never needs to shard, and inherits no
`workers == 1` assertion because it has no multi-worker path to guard.

The GIL is released for the whole request.

### The widened transport contract

The seam must stay symmetric: aiohttp and the native client must both satisfy it,
and neither may be privileged. Returning a `RequestRecord` whose `responses` is a
`list[SSEMessage]` would defeat the purpose — constructing N `SSEMessage` +
N `SSEField` objects through PyO3 is slower than constructing them in Python, so
the allocation cost would move across the boundary rather than disappear.

`RequestRecord` gains two optional, self-describing fields:

- a reduced-outcome struct carrying the four derived values plus TTFT, status,
  and error;
- a pre-serialized responses blob (`bytes`) for the hop to the record processor,
  spliced into the outgoing message with `orjson.Fragment` so it is never
  re-encoded.

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

The envelope is set by the lightweight reducer, not by the reference
`HttpEndpointBinding`. The crate claims a named set of SSE dialects — the
OpenAI-shaped chat and completions families first — and refuses selection for any
endpoint outside it, for multipart endpoints, and for polled submit→poll→download
endpoints. Refusal is at selection time, not per request, and the run proceeds on
`AioHttpTransport`.

Widening the envelope is adding a dialect to the reducer. Deliberately, it is not
porting the endpoint registry: the registry earns its cost in a runtime that must
dispatch arbitrary registered endpoints, and the Python product already has that
dispatch in Python.

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

Reference implementations in this repository. The lightweight crate copies the
first two and reimplements against the rest; no build edge connects them.

- `rust/runtime/src/transport/core/{trace.rs,sse.rs}` (vendored verbatim).
- `rust/runtime/src/transport/{reduce.rs,measure.rs}` (reimplemented).
- `rust/runtime/src/transport/http/` (reimplemented).
- `rust/runtime/src/engine/turn_execution.rs` (`WorkerSink`,
  `ExecutionSinkBuilder`, `build_native`) — the threading constraint the crate
  inherits, without the abstractions.
