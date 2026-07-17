<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Endpoint body construction — format vs. splice

**Date:** 2026-07-13
**Status:** built — `format_payload → BodyPlan`, the shared `JsonBodyMaterializer`,
and formatter-at-lowering are implemented; the whole-`Value` `serde_json::to_vec`
HTTP dispatch path is deleted.
**Grounding:** `rust/runtime/src/body_plan.rs`, `rust/runtime/src/endpoints/`
(`endpoints.rs`, `registry.rs`, `chat.rs`, `kserve.rs`, `riva.rs`, `anthropic.rs`,
`vllm_generate.rs`, `tier2.rs`, `dynosim.rs`), `rust/runtime/src/dataset/`
(`request.rs`, `dataset.rs`, `materialize.rs`), and the gRPC codec under
`rust/runtime/src/transport/grpc/`.
**Scope:** The **endpoint formatter contract** under segment unification.
`format_payload` returns a declarative `BodyPlan` — a named field list whose values
may be **segment handles** — consumed by **two shared materializers** (JSON splice /
protobuf encode) chosen by wire type. Companion to
`2026-07-13-segment-unification-design.md` (which owns the storage/lowering side);
this spec owns "handles → wire bytes."

## 1. Problem — two body paths, split by whether content is pre-serialized

The prior formatter contract returned a fully-inline `serde_json::Value` that then
became wire bytes two ways, both re-deriving from a fully-inline value:

- **HTTP:** `serde_json::to_vec(&value)` — a full **re-serialize** of the content
  (the ~9–13× path in the segment-unification microbench).
- **gRPC:** `encode_request(payload: &Value)` → protobuf.

A **splice** primitive already existed and is faster —
`build_message_body_from_wires(messages, overrides)` / `build_body_from_handles` /
`splice_raw_object` (`dataset/materialize.rs`) — but it was used only for
pre-serialized (`raw_payload`/message) content; authored content still went
`format_payload → Value → to_vec`. So the code had **two paths, split by whether
content is pre-serialized**, and the fast one was only half-wired. This spec unifies
authoring on the fast path.

## 2. The split is fundamental — don't unify it, make it explicit

You cannot "teach every formatter to splice," because **protobuf endpoints can't
splice.** Riva/KServe `encode_request(&Value)` must have the *structured* value to
set protobuf fields — you cannot memcpy pre-serialized JSON message bytes into a
protobuf message. The honest boundary is:

- **JSON-wire endpoints** (chat/completions/embeddings/Anthropic/TGI/rerank/image…)
  → **splice** pre-serialized segment bytes.
- **Protobuf-wire endpoints** (KServe V2, Riva ASR/TTS/NLP) → **encode from
  structure**; segments are storage the codec *reads*, not bytes it splices.

This is exactly the `transport::http` vs `transport::grpc` boundary. The formatter
contract names it rather than papering over it.

## 3. The contract — `format_payload → BodyPlan` (built)

`format_payload` returns a **`BodyPlan`** (`EndpointResult<BodyPlan>`,
`endpoints/registry.rs`): either the degenerate whole-body `Raw(Handle)` or an
ordered `Fields` list of named fields whose value is a literal, a segment reference,
or a pre-serialized message array. It runs **at lowering** — once per turn, the run's
endpoint known at config (segment-unification §3a) — **never per dispatch.** Dispatch
only *materializes* the plan (splice static + live segments, fold in the param
overrides). A per-request `format_payload → Value → serialize` is the slow path the
gRPC audit measured and is prohibited on the hot path.

```rust
// rust/runtime/src/body_plan.rs
pub enum BodyPlan {
    Raw(Handle),                              // the degenerate whole-body case
    Fields(SmallVec<[(FieldName, FieldValue); 8]>),
}

pub enum FieldValue {
    Literal(Value),            // endpoint-generated scalars/structs: model, max_tokens, stream…
    Segment(Handle),           // one pre-serialized content segment (system, tools, a raw body)
    Segments(SmallVec<[Handle; 1]>),  // an ordered array of interned handles — comma-joined for JSON
    Wires(SmallVec<[Bytes; 1]>),      // pre-serialized message array NOT in the frozen store
                                       // (dynamic / live continuation content)
}
```

The endpoint **declares its shape with segment slots** — it never touches
commas/brackets and never re-serializes content. Two shared materializers consume the
plan; the endpoint picks neither:

- **JSON (`transport::http`)** — the shared `JsonBodyMaterializer` (`body_plan.rs`)
  walks the plan and concatenates literal bytes + segment bytes from the frozen store
  into the single `Full<Bytes>` (§6). It is a strict generalization of
  `build_message_body_from_wires` from "message array + overrides" to "arbitrary named
  fields," and reuses `splice_raw_object` for the `Raw` arm. **Zero content
  re-serialize** — only endpoint `Literal` scalars and the override tail are
  serialized.
- **Protobuf (`transport::grpc`)** — the codec sets proto fields from the plan. For
  the in-tree KServe/Riva endpoints this is a **BYTES text tensor** built from the
  structured value (see §7 for why packed `raw_input_contents` is not used).

An authoring builder keeps plans declarative and safe (no manual JSON punctuation):

```rust
BodyPlan::new()
    .array("messages", handles)          // Segments — spliced (JSON) / read (proto)
    .str("model", model)                 // Literal
    .int("max_tokens", n)                // Literal
    .opt_segment("tools", tools)         // Segment
    .bool("stream", true);
```

Endpoints reach a plan either directly through the builder or via the transitional
`BodyPlan::from_object` bridge (build the `serde_json::Map`, then decompose — message
arrays → `Wires`/`Segments`, scalars → `Literal`), which is guaranteed byte-identical
to the old `to_vec` path by the invariant
`merged_object_bridge_is_byte_identical_to_to_vec` and the golden-byte gates
(chat/completions/embeddings/anthropic/responses/KServe/Riva).

## 4. Why this answers "every formatter learns to splice"

It doesn't — **splicing is one shared impl** (`JsonBodyMaterializer`); endpoints only
declare plans. The per-endpoint work is the same shape it was (name the fields), just
with segment slots instead of inline content. Byte-level concatenation, comma
delimiting, and content-length live once in the JSON materializer. Dispatch overrides
(`model`/`max_tokens`/`stream`) are folded into the plan's `Literal` fields in place
(`request.rs` `merge_overrides`, same `Map::insert` in-place/append semantics as the
old path) — never re-derived from content; effective metadata is read off the plan
(`effective_from_plan`), not a serialized `Value`; the stream-off downgrade is applied
to the plan.

Degenerate cases fall out: a recorded `raw` body is `BodyPlan::Raw(raw_handle)` → the
JSON materializer clones it (~12 ns).

## 5. Loader lowering is orthogonal

Loaders lower **content** → segments (`message`/`raw`/`token-ids`/`media`),
endpoint-agnostic. A `BodyPlan` *references* those handles; the loader never knows
which endpoint consumes them, and the same segments feed both materializers. That is
the dataset-loading work of the segment-unification spec, not this one. The endpoint
formatter is a **separate lowering step** (run once the run's endpoint is known — §3)
that declares the `BodyPlan` over those content segments; it is *not* part of the
loader and *not* per-dispatch.

## 6. Constraints honored

- **One `Full<Bytes>` body.** The JSON materializer concatenates the plan into a
  single contiguous buffer — never a multi-frame body — preserving the
  `SendCompletion` / cancellation-after-send anchor
  (`transport::http` connection send path).
- **Concat, never re-serialize.** Segment/`Wires` fields are spliced as pre-serialized
  bytes; only endpoint-generated `Literal` scalars are serialized (small, once).

## 7. Implementation — how it landed

1. **`BodyPlan` + `JsonBodyMaterializer`** (`body_plan.rs`) reproduce
   `build_message_body_from_wires` + `Overrides` behavior byte-for-byte over a plan. A
   `Fields([("messages", Segments)])` plan is byte-identical to the legacy splice
   (oracle `messages_plan_is_byte_identical_to_legacy_splice`) and a `Raw` plan
   reproduces the `raw_payload` fast path exactly.
2. **Every endpoint returns `EndpointResult<BodyPlan>`** (`endpoints/*.rs`), called
   **at lowering** (§3a of the segment spec), not per dispatch. The byte-neutral
   resolution to the "in-place-merge vs tail-append" problem: dispatch operates on the
   plan (`request.rs` folds overrides into the plan's literal fields with the same
   `Map::insert` semantics; `effective_from_plan` reads metadata off the plan).
   Endpoints reach the plan via `BodyPlan::from_object` where a transitional bridge is
   simplest, pinned byte-identical by the golden gates.
3. **The whole-`Value` `serde_json::to_vec` HTTP dispatch path is deleted** (the
   `structured_plan`/value-then-serialize path is removed); the shared
   `JsonBodyMaterializer` produces **every** HTTP body. `FieldValue::Wires` carries
   pre-serialized message arrays (dynamic / live-continuation content) not interned in
   the frozen store.
4. **gRPC packed `raw_input_contents` — a proven EXCLUSION, not pending work.**
   Adversarial verification established two unmet activation preconditions: (a) no
   gRPC endpoint is token-native — every in-tree KServe endpoint is
   `requires_raw_token_ids: false` and sends a **BYTES text tensor**
   (`endpoints/kserve.rs`); (b) encode still consumes a `serde_json::Value`
   (`transport::grpc` `encode_request(&Value)`), so it must walk `Value::Array`
   regardless. The golden test pins the wire bytes to the Python KServeV2 serializer's
   **typed `InferTensorContents`**; switching encode to `raw_input_contents` (proto
   tag 7) produces different bytes → fails the HARD parity gate for **zero** perf gain.
   Keeping typed contents (`raw_input_contents: Vec::new()`) is therefore the
   **parity-correct steady state**. The packed path lands only if/when both
   preconditions hold (a token-native gRPC endpoint + a threaded pre-packed segment).

**Formatter-at-lowering (built).** The per-endpoint `BodyPlan` is precomputed and
cached at endpoint-bind for eligible turns (`Dataset::precompute_body_plans`), so
`format_payload` — and the `from_object` `Value` bridge — no longer runs per dispatch
on the hot path for those turns; dispatch clones the cached plan and re-folds
overrides on the clone. The remaining per-dispatch `Value` construction is confined to
the ineligible fallback set (dynamic-context / template / graph). Byte-parity is held
throughout — a differential test materializes each body cache-on vs cache-off across
an endpoint × context-mode × overrides matrix and asserts byte-equality.

**Shared endpoint helpers.** `endpoints/endpoints.rs` exposes the `pub(crate)`
helpers `turn_texts`, `joined_text`, and `bearer_headers`; kserve/riva/vllm_generate/
dynosim import them rather than keeping their own copies.

## 8. Non-goals

- Storage / lowering / `Turn` field collapse — `2026-07-13-segment-unification-design.md`.
- Response decoding (`decode_response` stays value-based).
- Scatter-gather / multi-frame bodies (rejected; see the segment-unification spec §6).

## 9. Open questions

1. **gRPC field mapping:** the gRPC endpoint owns its plan→proto translation; the plan
   just carries the content segments its codec knows.
2. **Multimodal granularity:** a chat message with text + image *parts* is one
   `message` segment (the rendered wire is folded into the segment identity so
   same-text/different-media turns stay distinct); the materializer does not splice
   *inside* an array element.
3. **Streaming/SSE request options** and per-request headers/params that aren't body
   fields — carried alongside the plan, not in it.

## 10. Future transports (WebSockets and beyond)

The format-vs-splice contract is **transport-parametric**: the endpoint declares a
`BodyPlan` over transport-agnostic segments, and a **per-wire-type materializer** turns
it into bytes/frames. Today that's JSON-splice (`transport::http`) and proto-encode
(`transport::grpc`); a WebSocket transport is simply a **third materializer**
(`BodyPlan → WS text/binary frames`) plus its own framing/completion/cancellation.
Three properties keep this open: (1) `BodyPlan` carries **no framing assumption**; (2)
the single-`Full<Bytes>` body + `SendCompletion` constraint (§6) is **HTTP-local**;
(3) each transport owns its materializer + framing + completion + cancellation. Full
design: `2026-07-13-websocket-transport-design.md`.

## 11. Related

- `2026-07-13-segment-unification-design.md` — the storage/lowering side; this is its dispatch-side companion.
- `2026-07-13-websocket-transport-design.md` — the WS transport that adds a third materializer over this same `BodyPlan`.
- `2026-07-11-aiperf-runner-owned-endpoint-registry-design.md` / `2026-07-11-aiperf-rust-endpoints-design.md` — the endpoint registry + `format_payload` this changes.
- `2026-07-12-aiperf-native-grpc-kserve-v2-design.md` — the `transport::grpc` codec the protobuf branch reuses.
