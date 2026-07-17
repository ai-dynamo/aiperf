<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WebSocket inference transport (future)

**Date:** 2026-07-13
**Status:** design (proposed; **not built** — future third transport)
**Scope:** A Clock-injected WebSocket **inference** transport (`transport_ws`) modeled on
**Codex's** WebSocket protocol (the "based on codex" reference — a persistent WSS turn
stream, **WS-primary with an HTTPS/SSE fallback**), with the OpenAI **Realtime** API as a
second, **distinct** WS dialect. It reuses the segment content IR and the `BodyPlan`
contract, adds a third **materializer** (`BodyPlan → WS frames`), a **streaming/session
dispatch** shape, and a first-class **HTTPS/SSE fallback** that reuses the existing
`transport_http` streaming path when the WS upgrade is unavailable.

> **Codex is its own protocol.** It is **not** the generic OpenAI Responses API and must
> not be modeled from it, nor folded together with it or with Realtime — each is a
> separate dialect behind the binding registry. (Also not the aiperf dashboard/API
> websocket, `src/aiperf/api/routers/websocket.py`, which streams results to a UI.)

## 1. Why

aiperf benchmarks HTTP/SSE and gRPC today; WS inference is unbenchmarkable. The seam
already generalizes here (endpoint-body-construction §10): segments are
transport-agnostic, `BodyPlan` carries no framing assumption, and the
single-`Full<Bytes>`/`SendCompletion` rule is HTTP-local (verified absent from
`endpoints/`/`dataset/`). gRPC **bidi** is the in-tree precedent for framed messages over
a persistent connection (`transport_grpc`; `encode_bidi_requests` emits *"ordered
config-first messages"*, `binding.rs:62`).

The load pattern that matters: a newer inference frontend opens a **persistent WS turn
stream** and **falls back to HTTPS/SSE** when the upgrade is unavailable. Benchmarking it
means measuring both the WS path and its fallback — under one transport.

## 2. Dialects — separate protocols, one transport

The transport is dialect-neutral; each protocol is its own binding (as KServe/Riva are
for gRPC). **Do not conflate them.**

- **Codex (the reference dialect).** A persistent WSS **turn stream**, WS-primary with an
  **HTTPS/SSE fallback** carrying the identical event stream. A turn is: one
  turn-request frame → streamed **incremental output deltas** (and separate
  **reasoning** deltas) → a **terminal completion event** (with a non-terminal /
  failed variant). Turns chain via **server-side thread continuation** — a conversation
  is a thread on one socket, not a per-turn history resend. Codex-specific framing,
  headers, and event names are the dialect's own; this spec treats its *shape*, not the
  generic Responses API.
- **Realtime (a distinct dialect).** OpenAI's bidirectional voice/text WS. Client events
  `session.update`, `conversation.item.create`, `response.create`,
  `input_audio_buffer.append`/`commit`/`clear`, `response.cancel`; server events
  `session.created/updated`, `response.created`, streamed
  `response.output_text.delta`/`response.output_audio.delta`, terminal `response.done`.
  Base64 audio rides inside JSON events.
  ([Realtime WS guide](https://developers.openai.com/api/docs/guides/realtime-websocket))

They share the transport, framing mechanics, measurement, and fallback — **not** an event
schema. The generic **Responses API** (HTTP+SSE) is a *separate* surface again; if ever
added it is its own dialect, never derived from Codex.

## 3. Mapping onto the aiperf seams

| Concern | WS realization |
|---|---|
| **Transport** | `transport_ws`: Clock-injected, TLS via `connect_async_tls_with_config` over the shared `Connector` (`http-connector-seam`); one persistent socket per worker/thread driven by a **single `!Send` task** (unsplit `WebSocketStream` + `stream::unfold` owning both send and read — `.split()` full-duplex suits a *relay*, not a per-turn client) on the worker's `LocalSet`. |
| **HTTPS/SSE fallback** | **First-class.** On WS-upgrade/connect failure (or capability unavailability), the transport **latches WS off for the rest of the run** (a one-shot atomic latch) and degrades to the dialect's **same event stream over HTTP SSE**, reusing the existing `transport_http` streaming decoder — **not** a mid-stream resume. One shared event decoder feeds both WS and SSE; only framing differs (WS frame vs SSE `data:` line). |
| **Body construction** | Endpoint declares a `BodyPlan`; the **WS materializer** emits **text** frames (these dialects are text-JSON — a binary frame is a protocol violation, §5) — a turn-request frame (+ `session.update`/input frames for Realtime) — by splicing pre-serialized segment bytes into each event envelope. A **sequence of frames**, not one body; §6's one-`Full<Bytes>` rule is HTTP-local and does not apply. |
| **Content / segments** | Reused unchanged. `message`/`text`/`token-ids` segments splice into event JSON; **audio** (Realtime) is a `media` segment **base64-encoded once at lowering** and spliced by reference. |
| **Stateful thread** | A WS connection is a **thread**; turns chain by server-side continuation. Maps to a `Trace`/multi-turn `Session` whose node dispatches share one socket — no per-turn history resend. |
| **Dispatch** | A WS request is a **turn on a session**. Streaming dispatch (bidi precedent): one turn-request = one `Request`; the session = a `Trace`. Deltas push into the observer. |
| **Measurement** | The `RequestObserver` maps directly: **TTFT** = first output delta; per-delta = `on_token` (classify `output` vs `reasoning`); **terminal** = the dialect's completion event → `on_terminal`; **usage** = terminal-event usage → `on_usage`. No new measurement seam. |
| **Keepalive** | Periodic WS **Ping** while awaiting backend data; a missed pong / stalled read is an **idle-timeout**, surfaced as a structured nonfatal (§5). |
| **Cancellation** | Not HTTP-499/close — send the dialect's cancel event; server replies terminal `cancelled`. `CancellationPolicy` is already transport-abstract. |

## 4. Library and the "not slow" bar

- **Library — keep `tokio-tungstenite` (upstream, ≥0.26); a 3-way microbench refuted the
  case for switching.** The reputational premise was that `tungstenite` allocates an owned
  `Message` per frame, fighting aiperf's no-alloc discipline. That was true **pre-0.26**
  (`String`-backed `Text`) but **0.26+ is `bytes`-backed → zero-copy reads**, same as
  `fastwebsockets` (borrowed `Frame`) and `tokio-websockets` (`bytes`). The bench
  (`aiperf-ws-lib-bench`; `benchmark-findings/rust-websocket-client-libs-read-path.md`)
  confirms it: **0–9 total allocations across 150k–300k received frames for all three** at
  256 B / 2 KB / 32 KB — `alloc/frame ≈ 0` universally — and throughput ranks flip
  run-to-run (loopback noise dwarfs any delta, i.e. the WS lib is not the bottleneck).
  `tungstenite` also showed the **tightest small-frame p99**, which suits a low-jitter
  measurement client. RFC-6455 is why the alloc concern was small anyway: server→client
  frames are **unmasked**, so the received-delta flood costs parse, not un/masking (and
  `tokio-websockets`' SIMD-*un*mask headline barely helps a client). Net: strict,
  battle-tested, zero-alloc, competitive — **no speed reason to switch.** Revisit only if a
  *quiet-machine* bench shows a material, repeatable throughput/jitter delta.
- **Frames are spliced, not serialized.** The `BodyPlan → WS-frame` materializer (the
  third, after JSON-splice and proto-encode) concatenates pre-serialized segment bytes
  into each event envelope; base64 audio is pre-encoded at lowering. **No per-frame
  serde, no per-request `Value`** — the trap the gRPC audit flagged.
- **Persistent connection, no per-turn handshake.** One upgrade per worker; turns reuse
  the socket and chain server-side.
- **Clock injection** on frame read/write timestamps (like the HTTP `timerfd` path), so a
  `SimClock` drives offline WS replay — preserving the three-modes property.
- **`permessage-deflate` is a measured knob, not an assumption.** Real deployments
  negotiate it; it trades wire bytes for CPU. Whether it helps our envelope sizes is part
  of the not-slow microbench, not a default we guess at.

## 5. WS lifecycle & stream correctness (the earned-in-blood part)

- **Completeness is gated on the terminal event, not the socket FIN.** A turn is complete
  iff a terminal completion event arrived. A close *before* that is a **truncated
  stream** (error); a close *after* it is a **clean EOF**.
- **Benign-close taxonomy.** Backends routinely end a stream by closing the socket right
  after the terminal frame, **without** a WS close handshake or TLS `close_notify`. Treat
  `ConnectionClosed` / `AlreadyClosed` / `ResetWithoutClosingHandshake` / `UnexpectedEof`
  as clean EOF **when a terminal event was already seen** — otherwise every successful
  turn mislabels as a transport error.
- **Retry-before-first-event.** A WS attempt that fails **before any backend event**
  (connect/upgrade error, send failure, closed-before-terminal, eof-before-terminal,
  idle-timeout, connection-limit) is retryable — reconnect and retry the turn.
- **Upgrade timeout.** Bound the time to establish the WS upgrade; on expiry, retry or
  fall back to HTTPS/SSE (§3).
- **Idle timeout.** Ping-keepalive detects a silent stall; classify as a distinct
  nonfatal, not a generic failure.
- **Text frames only.** These dialects are text-JSON; a **binary** (or unexpected
  continuation) frame is a protocol violation — surface it as an error, never feed it to
  the decoder.
- **Reuse is fragile — reconnect-and-retry.** A pooled socket the peer dirty-closes while
  idle makes the *next* turn's send fail; an outer layer must transparently
  reconnect-and-retry so the first post-idle turn doesn't surface a spurious error. And a
  keepalive Ping must **not** assume the next frame is its Pong — read-and-classify, or it
  swallows a real backend event.

These are transport-owned; nothing leaks up to the endpoint/observer seams.

## 6. Constraints & design-ahead (do not leak up)

- **No single `Full<Bytes>` body** — a WS turn is a frame stream; keep the one-body rule
  in `transport_http` only (endpoint-body §10 guard 2).
- **`BodyPlan` stays framing-agnostic** — the WS materializer owns frame boundaries.
- **Per-transport framing/completion/cancellation** — WS owns its "request-sent" edge
  (the turn-request frame written), its cancel event, and its close taxonomy; it borrows
  neither HTTP's `SendCompletion` nor gRPC's status.
- **Shared event decoder across WS and SSE** — the fallback (§3) reuses one decoder per
  dialect; only framing differs. This is the seam that makes fallback cheap.
- The `Trace`/`Session` model (greenfield vocab) absorbs the persistent-session
  lifecycle: a WS thread is a `Trace`.

## 7. Open questions

1. **Fallback trigger scope — resolved (§3).** A connect/upgrade failure **latches WS off
   for the run** (one-shot) and continues on HTTPS/SSE; a mid-stream truncation retries on
   WS first (§5) — it does **not** resume mid-turn on SSE.
2. **Session ↔ Request granularity:** one turn-request = one `Request`, whole thread = a
   `Trace` chained server-side. (Lean: yes.)
3. **Streaming `Dispatcher` signature:** push into the existing `RequestObserver`
   (`on_token`/`on_terminal`) vs. return `impl Stream<Outcome>`. (Lean: observer push.)
4. **Realtime input-audio cadence:** `input_audio_buffer.append` chunk size / send pacing,
   Server-VAD vs manual `commit` as a workload knob.
5. **Send backpressure** on a shared socket across concurrent in-flight turns.
6. **Thread-chain on retry:** after a mid-thread failure, does the retried turn keep the
   server-side chain (server may have lost it) or restart the thread? Affects trace
   fidelity.

## 8. Non-goals

- Not built — the future third transport; no runner wiring yet.
- Not WebRTC (OpenAI's browser path); server-to-server benchmarking uses WS.
- Not the dashboard/API result-streaming websocket.
- Not the generic Responses API (a separate HTTP/SSE surface) — never modeled from Codex.

## 9. Related

- `2026-07-13-endpoint-body-construction-design.md` §10 — the transport-parametric materializer contract this extends with a WS branch.
- `2026-07-13-segment-unification-design.md` — the transport-agnostic content IR WS reuses (audio = `media` segment, base64 at lowering).
- `2026-07-10-aiperf-transport-rust-port-design.md` — the Clock-injected hyper transport WS mirrors, and whose SSE decoder the HTTPS **fallback** reuses (and whose `Full<Bytes>`/`SendCompletion` rule stays HTTP-local).
- `2026-07-12-aiperf-native-grpc-kserve-v2-design.md` — `transport_grpc` bidi, the persistent-stream precedent.
