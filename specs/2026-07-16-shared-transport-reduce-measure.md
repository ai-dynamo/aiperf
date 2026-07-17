<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared transport reduce + measure: a transport contributes only a sink builder

**Date:** 2026-07-16
**Status:** BUILT. Both the native HTTP (hyper) and native gRPC (tonic) sinks decode their
own wire record into a sequence of `ServerResponse`, then feed that sequence through ONE
shared reduction (`transport::reduce::reduce_parsed_response`) and wrap dispatch in ONE
shared worker-local measurement envelope (`transport::measure`). Each sink supplies only its
wire decode and its error-enum→terminal mapping.

Ground every claim below against the cited `rust/runtime/src/…:line`. Specs are intent; code
is truth.

---

## 1. Summary

The reduction of a decoded endpoint response into observer facts — absorb usage, absorb
response data, absorb endpoint metrics, emit the first-token callback, emit output/classified
tokens, reconstruct the assistant turn — is **identical regardless of how the bytes arrived**.
It is written once in `transport::reduce` (`rust/runtime/src/transport/reduce.rs`). Likewise,
the worker-local measurement — owning the `NativeMetricsObserver`, deriving the optional live
record, and wrapping a dispatch future in the register→arrival→record-response/failed-terminal
envelope — is written once in `transport::measure` (`rust/runtime/src/transport/measure.rs`).

This is what makes the architectural claim "a transport contributes only a sink builder"
literally true. Previously the gRPC sink re-implemented the whole per-response reduction loop
plus five verbatim helper functions (usage absorption, response-data absorption, endpoint-
metric absorption, token classification, assistant-message reconstruction). Now:

- The **HTTP** endpoint path feeds `reduce_parsed_response`
  (`rust/runtime/src/transport/http/sink/endpoint_dispatch.rs:352`) and measures through
  `measure_dispatch` (`rust/runtime/src/transport/http/sink.rs:981`).
- The **gRPC** path feeds the same `reduce_parsed_response`
  (`rust/runtime/src/transport/grpc/sink.rs:412`) and measures through the same
  `measure_dispatch` (`rust/runtime/src/transport/grpc/sink.rs:230`).

Both import the shared reduction seam identically:
`use crate::transport::reduce::{EndpointReduceAccumulators, TokenEmitter, assistant_message,
reduce_parsed_response};` (`endpoint_dispatch.rs:34`, `grpc/sink.rs:33`).

---

## 2. The reduction seam (`transport::reduce`)

The module doc states the contract directly: "Every native transport (`http`, `grpc`, and any
future wire) decodes its own record into a sequence of `ServerResponse` items and then reduces
that sequence identically … That reduction is the same regardless of how the bytes arrived, so
it lives here once instead of being copied per transport. A transport contributes only the
wire-decode that produces the `ServerResponse` iterator and the mapping from its own error enum
to a `ReplayTerminalStatus`." (`reduce.rs:4-14`).

### 2.1 The three shared parameters

`reduce_parsed_response(parsed, emit, acc)` (`reduce.rs:62`) takes:

| Param | Type | Role | Cite |
|---|---|---|---|
| `parsed` | `&ParsedResponse` | one endpoint-decoded response | `reduce.rs:63` |
| `emit` | `&TokenEmitter` | transport-supplied token-emission context | `reduce.rs:41-56` |
| `acc` | `EndpointReduceAccumulators` | the aggregated state the reduction folds into | `reduce.rs:27-37` |

`EndpointReduceAccumulators` bundles `&mut` borrows of the four accumulated outputs
(`reduce.rs:28-37`): `response_text` (concatenated text), `model_response`
(`ModelResponseMetadata` — content/reasoning/token-ids/errors), `endpoint_metrics`
(`ObservedEndpointMetrics` — video timing/memory), and `observed_usage` (reconciled terminal
`ObservedUsage`).

`TokenEmitter` (`reduce.rs:41-56`) is the transport-supplied token context: `uuid`,
`produces_tokens`, the run-origin `start_ns`, the `&dyn RequestObserver` (`obs`), a
`to_ms: &dyn Fn(i64) -> f64` perf-ns→run-relative-ms map, a shared `first_token_released:
&Cell<bool>` once-only latch, and `on_first_token: &dyn Fn(i64)`. Both sinks build this
identically — see `endpoint_dispatch.rs:312-320` and `grpc/sink.rs:387-395`; the ONLY
difference is each supplies its own `to_ms` closure (`endpoint_dispatch.rs:311`,
`grpc/sink.rs:386`) and `start_ns` (from its own transport record).

### 2.2 What the reduction does per response

`reduce_parsed_response` (`reduce.rs:62-93`):

1. `absorb_usage(parsed, acc.observed_usage)` — always, even when there is no content payload
   (`reduce.rs:67`). Reconciles all extended usage fields (prompt/completion/total, reasoning,
   cache read/write/miss, audio tokens+seconds, accepted/rejected prediction, tool-use-prompt),
   preferring the latest reported value (`reduce.rs:166-221`).
2. If `parsed.data` is `None`, return `false` (no content) after absorbing usage
   (`reduce.rs:68-70`).
3. `absorb_endpoint_metrics(data, acc.endpoint_metrics)` — currently video
   inference-seconds / peak-memory (`reduce.rs:71`, `:150-162`).
4. `absorb_response_data(data, acc.model_response)` — folds text/reasoning/tool-call/token-id
   payloads into the model response and returns the plain text it contributed
   (`reduce.rs:72`, `:107-143`); the text is appended to `acc.response_text` (`reduce.rs:73`).
5. If `emit.produces_tokens` (`reduce.rs:74`): for a non-empty `TokenIds` payload, release the
   first-token latch on the first call and emit `on_output_tokens` with one perf-ns-mapped
   timestamp per token (`reduce.rs:76-83`); otherwise for non-empty text, release the latch and
   emit `on_classified_token` with `token_kind(data)` (Reasoning vs Output) (`reduce.rs:84-90`).
6. Return `true` (this response carried content) (`reduce.rs:92`).

`token_kind` (`reduce.rs:96-103`) classifies a chunk as `Reasoning` when it is a non-empty
`ResponseData::Reasoning`, else `Output`. `assistant_message` (`reduce.rs:225-245`)
reconstructs the assistant message JSON from a rebuilt `Turn` — preferring the raw wire message
else synthesizing `{role, content}` — and is called by both sinks when the endpoint
`captures_assistant_turn()` (`endpoint_dispatch.rs:377-387`, `grpc/sink.rs:424-428`).

---

## 3. The measurement seam (`transport::measure`)

The module doc: "Each thread-per-core worker owns exactly one `NativeMetricsObserver` that
accumulates the complete record (arrival → admit → tokens → usage → terminal → response) so
the end-of-run drain yields one authoritative `RecordIngest` per request. Owning that observer,
deriving the live-record snapshot, and wrapping a dispatch future in the register/arrival/
record-response/failed-terminal envelope are identical across transports, so they live here
once. A transport contributes only the dispatch future." (`measure.rs:4-12`).

### 3.1 `WorkerMeasurement` (`measure.rs:29-57`)

A per-worker `RefCell<Option<Rc<NativeMetricsObserver>>>`. `configure` installs a fresh
observer (`measure.rs:36-39`); `observer()` accesses it, erroring if measured execution runs
before configure (`measure.rs:43-48`); `drain(end_ns)` finalizes it into the drained
`Vec<(Uuid, RecordIngest)>` at run end (`measure.rs:51-56`). Both sinks embed a
`WorkerMeasurement` field: HTTP at `http/sink.rs:183`, gRPC at `grpc/sink.rs:122`, and both
delegate `configure_measurement`/`drain_records` to it (`http/sink.rs:836-843`, `:894-896`;
`grpc/sink.rs:520-524`, `:544-546`).

### 3.2 `measure_dispatch` — the envelope (`measure.rs:91-137`)

```text
measure_dispatch(observer, clock, uuid, context, dispatch):
    observer.register_metadata(uuid, context.metadata)         measure.rs:101
    observer.on_arrival(uuid, arrival_ms, input_len, req_len)  measure.rs:102
    result = dispatch.await                                     measure.rs:108   ← the ONLY per-transport part
    match result:
      Ok(collected) → observer.record_response(uuid, {start,end,prompt,completion,http})  measure.rs:111-121
      Err(_)        → observer.on_terminal(uuid, Failed)                                    measure.rs:125
                      observer.record_response(uuid, {now,now,default})                     measure.rs:126-132
    return result
```

The `dispatch` future is the sole transport-specific input; everything wrapping it (metadata
registration, arrival, terminal-facts recording, the failed-terminal fallback) is identical.
`live_record` (`measure.rs:64-79`) derives the optional live/consumed record clone from
`MeasuredContext` (moved out in sketch mode, cloned otherwise). Both sinks call it identically
after `dispatch_measured` (`http/sink.rs:856`, `grpc/sink.rs:537`).

Each sink's `dispatch_measured` is a two-line wrapper around `measure_dispatch` passing its own
collect future: HTTP passes `dispatch_collect_streaming(...)` (`http/sink.rs:981-988`), gRPC
passes `dispatch_collect(...)` (`grpc/sink.rs:230-237`).

---

## 4. What each transport still owns (the thin per-transport shell)

| Responsibility | HTTP | gRPC |
|---|---|---|
| Wire client + request lowering | hyper via `HttpTransport` (`http/sink.rs:39`, `:253`) | tonic via `GrpcTransport` (`grpc/sink.rs:31`, `:158`) |
| Wire record → `ServerResponse` decode | `binding.decode_response(...)` per response (`endpoint_dispatch.rs:322`) | build `ServerResponse` from each `record.responses[i].json` (`grpc/sink.rs:397-401`) |
| First-token meaningfulness filter | streaming SSE filter (`endpoint_dispatch.rs:280-294`) | `meaningful_response` closure (`grpc/sink.rs:362-374`, `:558-571`) |
| Error enum → `ReplayTerminalStatus` | `ErrorKind::{Cancelled,…}` match (`endpoint_dispatch.rs:408-420`) | `GrpcErrorKind::{RequestCancellation,…}` match (`grpc/sink.rs:436-448`) |
| Transport-timing → `RequestTrace` | `http_trace(&record)` (`endpoint_dispatch.rs:535-558`) | `grpc_metrics_trace(&record)` (`grpc/sink.rs:592-613`) |
| The shared reduction + measurement | **imports** `reduce`/`measure` | **imports** `reduce`/`measure` |

Both sinks converge their per-response loops on the *same* five lines
(`endpoint_dispatch.rs:352-362`, `grpc/sink.rs:412-422`):

```rust
let carried_content = reduce_parsed_response(
    &parsed,
    &emitter,
    EndpointReduceAccumulators {
        response_text: &mut response_text,
        model_response: &mut model_response,
        endpoint_metrics: &mut endpoint_metrics,
        observed_usage: &mut usage,      // gRPC names it `usage`; HTTP `observed_usage`
    },
);
parsed_content |= carried_content;
```

---

## 5. End-to-end flow

```text
                       WIRE BYTES (hyper h1/h2c/UDS/TLS)         WIRE BYTES (tonic h2c/grpcs)
                                 │                                          │
                     ┌───────────▼──────────────┐            ┌─────────────▼─────────────┐
                     │ HTTP sink  (http/sink.rs) │            │ gRPC sink (grpc/sink.rs)  │
                     │  dispatch_collect_streaming│            │  dispatch_endpoint        │
                     └───────────┬──────────────┘            └─────────────┬─────────────┘
                                 │ PER-TRANSPORT DECODE                     │ PER-TRANSPORT DECODE
                                 │ binding.decode_response(r)               │ ServerResponse::from
                                 │   → ServerResponse   ed.rs:322           │   record.responses[i].json
                                 │                                          │   → ServerResponse  g.rs:397
                                 ▼                                          ▼
                     ┌────────────────────────────┐          ┌────────────────────────────┐
                     │ endpoint.parse_response(sr) │          │ endpoint.parse_response(sr) │
                     │   → Option<ParsedResponse>  │          │   → Option<ParsedResponse>  │
                     │      ed.rs:329               │          │      g.rs:403               │
                     └──────────────┬─────────────┘          └──────────────┬─────────────┘
                                    │  Some(parsed)                          │  Some(parsed)
                                    └───────────────┬────────────────────────┘
                                                    ▼
                        ╔══════════════════════════════════════════════════════════╗
                        ║  SHARED  reduce_parsed_response(parsed, emit, acc)         ║
                        ║                                    reduce.rs:62            ║
                        ║   1. absorb_usage           → observed_usage  r.rs:67      ║
                        ║   2. absorb_endpoint_metrics→ endpoint_metrics r.rs:71     ║
                        ║   3. absorb_response_data   → model_response   r.rs:72     ║
                        ║          + push text        → response_text    r.rs:73     ║
                        ║   4. if produces_tokens:                                   ║
                        ║        TokenIds  → first-token latch + on_output_tokens    ║
                        ║                                   r.rs:79-83               ║
                        ║        text      → first-token latch + on_classified_token ║
                        ║                     (token_kind: Output|Reasoning) r.rs:84 ║
                        ║   returns carried_content (bool)                           ║
                        ╚═══════════════════════════════┬══════════════════════════╝
                                                        │  events out
                                                        ▼
                        ┌──────────────────────────────────────────────────────────┐
                        │  RequestObserver  (worker-local NativeMetricsObserver)     │
                        │   on_output_tokens / on_classified_token / (first-token)   │
                        │   … then per-request terminal (each sink):                 │
                        │     obs.on_usage(uuid, observed_usage)   ed.rs:454/g.rs:450│
                        │     obs.on_endpoint_metrics(uuid, ...)   ed.rs:455/g.rs:451│
                        │     obs.on_terminal(uuid, terminal)      ed.rs:456/g.rs:452│
                        └──────────────────────────────────────────────────────────┘

  ── the whole per-request dispatch above is wrapped by the SHARED measure envelope ──

   ╔══════════════════════════════════════════════════════════════════════════════════╗
   ║ SHARED  measure_dispatch(observer, clock, uuid, context, DISPATCH)  measure.rs:91  ║
   ║                                                                                    ║
   ║   register_metadata(uuid, context.metadata)                     measure.rs:101     ║
   ║   on_arrival(uuid, arrival_ms, input_length, requested_output)  measure.rs:102     ║
   ║        │                                                                            ║
   ║        ▼   DISPATCH  =  the ONLY per-transport future                              ║
   ║   ┌──────────────────────────────────────────────────────────────────────────┐    ║
   ║   │ HTTP: dispatch_collect_streaming(...)   http/sink.rs:986                   │    ║
   ║   │ gRPC: dispatch_collect(...)             grpc/sink.rs:235                   │    ║
   ║   │   (each runs the decode → reduce_parsed_response loop shown above)         │    ║
   ║   └──────────────────────────────────────────────────────────────────────────┘    ║
   ║        │                                                                            ║
   ║        ▼                                                                            ║
   ║   Ok  → record_response(uuid, {start,end,prompt,completion,http})  measure.rs:111  ║
   ║   Err → on_terminal(Failed); record_response(now,now,default)      measure.rs:123  ║
   ╚══════════════════════════════════════════════════════════════════════════════════╝
        │  observer accumulates the complete record (arrival→…→response)
        ▼
   WorkerMeasurement::drain(end_ns) → Vec<(Uuid, RecordIngest)>   measure.rs:51 / sink drain_records
```

---

## 6. Verification index

| Claim | Cite |
|---|---|
| `reduce_parsed_response` — single shared reduction | `reduce.rs:62-93` |
| `EndpointReduceAccumulators` — the four `&mut` outputs | `reduce.rs:27-37` |
| `TokenEmitter` — transport-supplied token context | `reduce.rs:41-56` |
| `absorb_usage` — reconciles all extended usage fields | `reduce.rs:166-221` |
| `absorb_response_data` — folds text/reasoning/tool/tokenids | `reduce.rs:107-143` |
| `absorb_endpoint_metrics` — video timing/memory | `reduce.rs:150-162` |
| `token_kind` — Reasoning vs Output classification | `reduce.rs:96-103` |
| `assistant_message` — assistant turn reconstruction | `reduce.rs:225-245` |
| Module doc: "a transport contributes only the wire-decode … + error mapping" | `reduce.rs:4-14` |
| `WorkerMeasurement` (configure/observer/drain) | `measure.rs:29-57` |
| `measure_dispatch` envelope | `measure.rs:91-137` |
| `live_record` derivation | `measure.rs:64-79` |
| Module doc: "A transport contributes only the dispatch future." | `measure.rs:4-12` |
| HTTP imports the shared reduce seam | `endpoint_dispatch.rs:34` |
| HTTP builds `TokenEmitter`, calls `reduce_parsed_response` | `endpoint_dispatch.rs:312-362` |
| HTTP `on_usage`/`on_endpoint_metrics`/`on_terminal` | `endpoint_dispatch.rs:454-456` |
| HTTP `dispatch_measured` → `measure::measure_dispatch` | `http/sink.rs:972-989` |
| HTTP `WorkerMeasurement` field + configure/drain | `http/sink.rs:183`, `:836-843`, `:894-896` |
| gRPC imports the SAME shared reduce seam | `grpc/sink.rs:33` |
| gRPC builds `TokenEmitter`, calls `reduce_parsed_response` | `grpc/sink.rs:387-422` |
| gRPC `on_usage`/`on_endpoint_metrics`/`on_terminal` | `grpc/sink.rs:450-452` |
| gRPC `dispatch_measured` → `measure::measure_dispatch` | `grpc/sink.rs:222-238` |
| gRPC `WorkerMeasurement` field + configure/drain | `grpc/sink.rs:122`, `:520-524`, `:544-546` |
| gRPC error enum → terminal mapping (per-transport) | `grpc/sink.rs:436-448` |
| HTTP error enum → terminal mapping (per-transport) | `endpoint_dispatch.rs:408-420` |
| `Request`/`DispatchResult`/`MeasuredContext`/`MeasuredOutcome`/`PreparedTurn` | `transport/core/dispatch.rs:36`, `:116`, `:137`, `:167`, `:187` |

---

## 7. The direct (non-endpoint-aware) HTTP path — a documented exception

The HTTP sink's endpoint-aware path (`dispatch_prepared_endpoint_collect_record_with_hooks`,
`endpoint_dispatch.rs:168`) is the shared-reduce path. The HTTP sink ALSO retains a legacy
direct OpenAI-chat loop, `dispatch_collect_record_with_hooks`
(`http/sink.rs:374-549`), which inline-parses `ChatChunk`/non-streaming responses and emits
`on_classified_token`/`on_usage` itself. It is reached only when `endpoint_aware` is false
(`http/sink.rs:1049-1055`) — i.e. no prepared endpoint binding. All product v2 dispatch is
endpoint-aware (the runner materializes prepared endpoints), so the direct loop is a
convenience/back-compat path, not the shared seam. gRPC has no such fallback: it *requires*
endpoint-aware materialization (`grpc/sink.rs:246-249`) and therefore only ever runs through
`reduce_parsed_response`.

---

## 8. Extension notes (design-ahead)

- Adding a transport is: implement a wire client + a `ServerResponse` decode + an error→terminal
  map + a `RequestTrace` builder, embed a `WorkerMeasurement`, and call `measure_dispatch` and
  `reduce_parsed_response`. No new reduction loop, no re-implemented usage/data/metric absorption,
  no re-derived first-token latch. The test `transport_and_grpc_sinks_are_dispatchers`
  (`http/sink.rs:1104-1110`) asserts both sinks satisfy the `Dispatcher` seam.
- The reduction operates on `ParsedResponse`/`ResponseData` (`endpoints` module), so any new
  endpoint dialect is absorbed by the SAME reduction the moment it produces a `ParsedResponse` —
  the transport never learns the new shape.
- `EndpointReduceAccumulators` is a struct of `&mut` borrows (not positional args), so adding a
  fifth accumulated output is a named-field change, not a re-threaded call signature at every
  sink.
