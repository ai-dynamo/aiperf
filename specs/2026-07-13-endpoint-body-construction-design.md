<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Endpoint body construction — format vs. splice

**Date:** 2026-07-13
**Status:** design (proposed; grounded in current code)
**Scope:** The **endpoint formatter contract** under segment unification. Today
`format_payload` returns a fully-inline `serde_json::Value` that is then
re-serialized (HTTP) or re-encoded (gRPC). This spec replaces that with
`format_payload → BodyPlan` — a declarative field list whose values may be
**segment handles** — consumed by **two shared materializers** (JSON splice /
protobuf encode) chosen by wire type. Companion to
`2026-07-13-segment-unification-design.md` (which owns the storage/lowering side);
this spec owns "handles → wire bytes."

## 1. Problem — the split is already shipping

Every endpoint's `format_payload` returns a `serde_json::Value`
(`endpoints/anthropic.rs:88`, `endpoints/riva.rs:159`, all `tier2.rs`). That value
becomes wire bytes two ways, both re-deriving from a fully-inline value:

- **HTTP:** `serde_json::to_vec(&value)` (`request.rs:341`, `:428`) — a full
  **re-serialize** of the content (the ~9–13× path in the segment-unification
  microbench).
- **gRPC:** `encode_request(payload: &Value)` → protobuf
  (`transport_grpc/binding.rs:56`).

A **splice** primitive already exists and is faster —
`build_message_body_from_wires(messages: &[Bytes], overrides)` (`materialize.rs:190`),
`build_body_from_handles` (`:156`), `splice_raw_object` (`:219`) — but it's used
only for pre-serialized (`raw_payload`/message) content; authored content still
goes `format_payload → Value → to_vec`. So the code already has **two paths, split
by whether content is pre-serialized**, and the fast one is only half-wired.

## 2. The split is fundamental — don't unify it, make it explicit

You cannot "teach every formatter to splice," because **protobuf endpoints can't
splice.** Riva/KServe `encode_request(&Value)` must have the *structured* value to
set protobuf fields — you cannot memcpy pre-serialized JSON message bytes into a
protobuf message. The honest boundary is:

- **JSON-wire endpoints** (chat/completions/embeddings/Anthropic/TGI/rerank/image…)
  → **splice** pre-serialized segment bytes.
- **Protobuf-wire endpoints** (KServe V2, Riva ASR/TTS/NLP) → **encode from
  structure**; segments are storage the codec *reads*, not bytes it splices.

This is exactly the `transport_http` vs `transport_grpc` boundary that already
exists. The formatter contract should name it, not paper over it.

## 3. The contract — `format_payload → BodyPlan`

`format_payload` stops returning a `Value` and returns a **`BodyPlan`**: an ordered
list of named fields whose value is a literal *or* a segment reference. It runs **at
lowering** — once per turn, the run's endpoint known at config (segment-unification
§3a) — **never per dispatch.** Dispatch only *materializes* the plan (splice static +
live segments, stamp the param tail). A per-request `format_payload → Value →
serialize` is the slow path the gRPC audit measured and is prohibited on the hot path.

```rust
struct BodyPlan { fields: SmallVec<[(FieldName, FieldValue); 8]> }

enum FieldValue {
    Literal(Value),            // endpoint-generated scalars/structs: model, max_tokens, stream…
    Segment(Handle),           // one pre-serialized content segment (system, tools, a raw body)
    Segments(SmallVec<[Handle; 1]>),  // an ordered array (message list) — comma-joined for JSON
}
```

The endpoint **declares its shape with segment slots** — it never touches
commas/brackets and never re-serializes content. Two shared materializers consume
the plan; the endpoint picks neither:

- **JSON (`transport_http`)** — a shared splicer walks the plan and concatenates
  literal bytes + segment bytes from the store into the single `Full<Bytes>`
  (§6). This is `build_message_body_from_wires` generalized from "message array +
  overrides" to "arbitrary named fields." **Zero content re-serialize.**
- **Protobuf (`transport_grpc`)** — the codec sets proto fields from the plan, and
  for token/tensor fields it must take the **packed segment bytes straight into
  `raw_input_contents`** (`contents = None`) — a single length-delimited field, zero
  per-element work. It must **not** build a `serde_json::Value` and walk it
  element-by-element. (The current codec is the anti-pattern this replaces: it
  hardcodes `raw_input_contents: Vec::new()` at `codec.rs:88`, walks every tensor
  element out of a `Value::Array` at `codec.rs:241-267`, and round-trips the body
  through JSON per request — `request.rs:341` → `grpc.rs:370` → `grpc.rs:507` — with
  the typed `raw_token_ids: Vec<u32>` dropped at `http.rs:276`. Python PR #664 makes
  the same choice — per-element `int64_contents.extend(int(v) …)`, dict round-trip,
  fresh proto per request. **That is the "what not to port" reference.**) The decode
  side already proves the packed path (`codec.rs:401-444`).

A builder keeps authoring declarative and safe (no manual JSON punctuation):

```rust
BodyPlan::new()
    .array("messages", req.body)         // Segments — spliced (JSON) / read (proto)
    .str("model", model)                 // Literal
    .int("max_tokens", n)                // Literal
    .opt_segment("tools", req.tools)     // Segment
    .bool("stream", true);
```

## 4. Why this answers "every formatter learns to splice"

It doesn't — **splicing is one shared impl**; endpoints only declare plans. The
per-endpoint work is the same shape it is today (name the fields), just with
segment slots instead of inline content. Byte-level concatenation, comma
delimiting, and content-length live once in the JSON materializer. Params
(`model`/`max_tokens`/`stream`) are `Literal` fields the endpoint stamps — exactly
the role `Overrides` plays in `build_message_body_from_wires` today — never
re-derived from content.

Degenerate cases fall out: a recorded `raw` body is a one-field plan
`Segment(raw_handle)` → the JSON materializer clones it (~12 ns); a token-native
request is `Segment(token_ids_handle)` the gRPC/token path reads.

## 5. Loader lowering is orthogonal

Loaders lower **content** → segments (`message`/`raw`/`token-ids`/`media`),
endpoint-agnostic. A `BodyPlan` *references* those handles; the loader never knows
which endpoint consumes them, and the same segments feed both materializers. That
is the dataset-loading work of the segment-unification spec, not this one. The
endpoint formatter is then a **separate lowering step** (run once the run's endpoint
is known — §3) that declares the `BodyPlan` over those content segments; it is *not*
part of the loader and *not* per-dispatch.

## 6. Constraints honored

- **One `Full<Bytes>` body.** The JSON materializer concatenates the plan into a
  single contiguous buffer — never a multi-frame body — preserving the
  `SendCompletion` / cancellation-after-send anchor
  (`transport_http/client/connection.rs:135-160`).
- **Concat, never re-serialize.** Segment fields are spliced as pre-serialized
  bytes; only endpoint-generated `Literal` scalars are serialized (small, once).

## 7. Migration (staged; suite green each step)

1. Introduce `BodyPlan` + a shared `JsonBodyMaterializer` that reproduces
   `build_message_body_from_wires` + `Overrides` behavior byte-for-byte over a
   plan.
2. Change `format_payload -> BodyPlan`, called **at lowering** (once per turn; §3a of
   the segment spec), not per dispatch. JSON endpoints declare fields with segment
   slots; dispatch (`request.rs:341/428`) routes through the materializer instead of
   `serde_json::to_vec`. Byte-parity against the old value-then-serialize path is the
   gate.
3. **gRPC fast path (the audit's fix — not "unchanged").** Add a **non-`Value` encode
   entry** to `GrpcEndpointBinding` (`binding.rs:56` is `&Value`-only today) that
   consumes the `BodyPlan`, and pack token/tensor segments **straight into
   `raw_input_contents`** (`codec.rs:88` hardcodes it empty; kill the per-element
   `Value::Array` walk at `codec.rs:241-267`). Thread the typed/packed segment through
   `TurnToSend → PreparedHttpTurn → HttpRequest → GrpcTransportSink` (dropped today at
   `http.rs:276`) so the transport never re-parses JSON (`grpc.rs:370`) nor re-emits
   it for the artifact (`grpc.rs:507`). This is the "not slow" requirement.
4. Delete the `format_payload → Value → to_vec` HTTP path **and** the per-request JSON
   round-trip once all endpoints emit plans.

## 8. Non-goals

- Storage / lowering / `Turn` field collapse — `2026-07-13-segment-unification-design.md`.
- Response decoding (`decode_response` stays value-based).
- Scatter-gather / multi-frame bodies (rejected; see the segment-unification spec §6).

## 9. Open questions

1. **gRPC field mapping:** does `BodyPlan` carry canonical field names the proto
   codec maps, or does each gRPC endpoint own its plan→proto translation? (Lean:
   the gRPC endpoint owns it; the plan just carries the content segments its codec
   knows.)
2. **Multimodal granularity:** a chat message with text + image *parts* — is the
   message one `message` segment, or are the parts sub-segments spliced within it?
   Affects whether the JSON materializer ever splices *inside* an array element.
3. **Streaming/SSE request options** and per-request headers/params that aren't
   body fields — carried alongside the plan, not in it.

## 10. Future transports (WebSockets and beyond)

The format-vs-splice contract is **transport-parametric**: the endpoint declares a
`BodyPlan` over transport-agnostic segments, and a **per-wire-type materializer** turns
it into bytes/frames. Today that's JSON-splice (`transport_http`) and proto-encode
(`transport_grpc`); a WebSocket transport is simply a **third materializer**
(`BodyPlan → WS text/binary frames`) plus its own framing/completion/cancellation. gRPC
bidi is the working precedent — `encode_bidi_requests` already emits *"ordered
config-first messages for a bidirectional request"* (`transport_grpc/binding.rs:62`),
i.e. a stream of framed messages over a persistent connection.

Three properties keep this open (verified in-tree today): (1) `BodyPlan` carries **no
framing assumption** — it's a content/field declaration; (2) the single-`Full<Bytes>`
body + `SendCompletion` constraint (§6) is **HTTP-local** — `rg Full<Bytes>|SendCompletion`
over `endpoints/` and `dataset/` is empty; (3) each transport owns its materializer +
framing + completion + cancellation. Full design:
`2026-07-13-websocket-transport-design.md`.

## 11. Related

- `2026-07-13-segment-unification-design.md` — the storage/lowering side; this is its dispatch-side companion.
- `2026-07-13-websocket-transport-design.md` — the WS transport that adds a third materializer over this same `BodyPlan`.
- `2026-07-11-aiperf-runner-owned-endpoint-registry-design.md` / `2026-07-11-aiperf-rust-endpoints-design.md` — the endpoint registry + `format_payload` this changes.
- `2026-07-12-aiperf-native-grpc-kserve-v2-design.md` — the `transport_grpc` codec the protobuf branch reuses.

## Addendum — 2026-07-13 (implementation status: stage 1 landed; stage 2 scoped)

Grounded in `rust/runtime/src/dataset/body_plan.rs`. **Built:**

- **§3/§7 stage 1 — `BodyPlan` + `JsonBodyMaterializer`** (`dataset/body_plan.rs`):
  `BodyPlan` is `Raw(Handle)` (the degenerate whole-body case) or an ordered
  `Fields(SmallVec<[(FieldName, FieldValue); 8]>)` with `FieldValue ∈
  {Literal(Value), Segment(Handle), Segments([Handle])}`, built with the fluent
  `array/segment/opt_segment/str/int/bool/literal` builder from §3. The shared
  `JsonBodyMaterializer` walks a plan against the frozen store, concatenating
  literal bytes + pre-serialized segment bytes into one `Full<Bytes>` with **zero
  content re-serialize** (only endpoint `Literal` scalars and the override tail
  are serialized). It is a strict generalization of `build_message_body_from_wires`
  and reuses `splice_raw_object` for the raw arm, so byte-parity is *structural*:
  a `Fields([("messages", Segments)])` plan is byte-identical to the legacy splice
  (oracle test `messages_plan_is_byte_identical_to_legacy_splice`) and a `Raw`
  plan reproduces the `raw_payload` fast path exactly.
- **Live wiring (raw arm):** the `raw_payload` dispatch in `dataset/request.rs`
  now runs through `BodyPlan::raw` + `JsonBodyMaterializer`, selected by segment
  domain (see the companion spec's addendum), and is validated end-to-end against
  the real `aiperf-mock-server` (`tests/scheduled_real_mock.rs`).

**Stage 2 scoping finding (blocks a mechanical conversion).** Converting the ~35
structured (non-raw) JSON endpoints to `format_payload → BodyPlan` is **not**
byte-neutral and is a substantive decision, not a swap. Today the structured path
(`request.rs` `format_payload → Value`, then `merge_overrides`, then
`effective_from_structured_body`) merges dispatch overrides **in place** on the
`serde_json::Map` — a single `model`/`stream`/`max_tokens` key at the formatter's
position — and then reads effective metadata **back out of the serialized Value**.
A `BodyPlan` that declared those params as `Literal` fields and let
`JsonBodyMaterializer` append the override *tail* (as the raw/splice path does)
would emit **duplicate keys** with last-wins semantics and different ordering.
The correct stage-2 form (§4) folds the params into the plan's `Literal` fields
and drops the tail, and moves effective-metadata computation off the serialized
Value onto the plan/turn directly — per endpoint.

## Addendum — 2026-07-14 (implementation status: §7 stages 1-2 + 4 built; §7 stage 3 justified-deferred; fresh-context approved)

The stage-2 conversion was completed the correct way. An independent fresh-context
reviewer verified the below and **approved the spec with caveats**; the full
`aiperf` lib suite (706) + every `endpoints_*` byte-parity suite + golden-byte
gates are green.

- **§7 stage 2 — every endpoint returns `EndpointResult<BodyPlan>`** (`endpoints/*.rs`).
  The byte-neutral resolution to the "in-place-merge vs tail-append" problem above:
  dispatch operates on the plan (`request.rs` `merge_overrides` folds overrides
  into the plan's literal fields with the same `Map::insert` in-place/append
  semantics; `effective_from_plan` reads metadata off the plan, not a serialized
  Value; the stream-off downgrade is applied to the plan). Endpoints reach the
  plan via the transitional `BodyPlan::from_object` bridge (build the `Map`, then
  decompose — message arrays → `Wires`, scalars → `Literal`), guaranteed
  byte-identical by the invariant `merged_object_bridge_is_byte_identical_to_to_vec`
  and the golden-byte gates (chat/completions/embeddings/anthropic/responses/KServe/Riva).
- **§7 stage 4 — the whole-`Value` `serde_json::to_vec` HTTP dispatch path is
  deleted** (`structured_plan` removed); the shared `JsonBodyMaterializer` produces
  every HTTP body. `FieldValue::Wires` was added for pre-serialized message arrays
  (dynamic/live content) not interned in the frozen store.
- **§7 stage 3 (gRPC packed `raw_input_contents`) — a proven EXCLUSION, not
  pending work.** Adversarial verification established two unmet activation
  preconditions: (a) no gRPC endpoint is token-native — every in-tree KServe
  endpoint is `requires_raw_token_ids: false` and sends a **BYTES text tensor**
  (`endpoints/kserve.rs`); (b) encode still consumes a `serde_json::Value`
  (`binding.rs` `encode_request(&Value)`), so it must walk `Value::Array`
  regardless. The golden test `transport_grpc_codec.rs` pins the wire bytes to the
  Python KServeV2 serializer's **typed `InferTensorContents`**; switching encode to
  `raw_input_contents` (proto tag 7) produces different bytes → fails the HARD
  parity gate for **zero** perf gain. `codec.rs` keeping typed contents
  (`raw_input_contents: Vec::new()`) is therefore the **parity-correct steady
  state**. This lands only if/when both preconditions hold (a token-native gRPC
  endpoint + a threaded pre-packed segment).

**§3a formatter-at-lowering built (segment spec companion):** the per-endpoint
`BodyPlan` is now precomputed and cached at endpoint-bind for eligible turns
(`Dataset::precompute_body_plans`), so `format_payload` — and the `from_object`
`Value` bridge — no longer runs per dispatch on the hot path for those turns. The
remaining per-dispatch `Value` construction is confined to the ineligible fallback
set (dynamic-context / template / graph); the byte-parity contract is fully met
throughout.
