<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Endpoint body construction

## Purpose

Define how an endpoint declares the *shape* of a request body and how that shape
becomes wire bytes, as a two-stage split:

1. **Declare** — an endpoint's `format_payload` returns a declarative **body plan**:
   an ordered set of named fields whose values are either endpoint-generated
   scalars or *handles* into a frozen content store. This happens once per turn,
   at lowering, with no serialization of content.
2. **Materialize** — at dispatch, a wire-typed **materializer** turns the plan
   plus a small per-dispatch **override set** into the final body (JSON bytes or a
   protobuf message). Content is spliced from its already-serialized form; it is
   never re-serialized here.

This record owns the **plan vocabulary** and the **handles → wire bytes** contract.
Segment interning, content addressing, and lowering belong to
[dataset.md](dataset.md). The design is language-neutral: it currently exists as a
Rust realization, but every rule below is stated so a future Python (or other)
implementation can reproduce it byte-for-byte.

## Motivation

A naive endpoint builds a request body per dispatch by constructing an in-memory
object (a dict / `Value` / struct) and serializing the whole thing. On a load
generator's hot path that pays three costs on every request: rebuilding the
object graph, re-serializing content that never changed between requests, and
allocating scatter buffers. When the same conversation prefix is reused across
thousands of requests, re-serializing its messages every time dominates.

The body plan removes all three:

- **Content serializes exactly once**, at composition time, into a frozen store.
- **Dispatch only splices bytes** — concatenate pre-serialized segments plus a
  tiny override tail into one contiguous buffer.
- **The plan is built once per turn**, not once per dispatch; the run's endpoint
  is fixed at config time, so nothing about the plan's structure varies per
  request. Only the override set (model, token cap, stream flag, seed, …) varies,
  and it is small.

A per-request `format_payload → object → serialize` on the hot path is therefore
**prohibited**. `format_payload` returns a plan; dispatch materializes it.

## The content store (summary)

Body construction sits on top of a **frozen segment store** (full design in
[dataset.md](dataset.md); referenced here only as far as body construction
depends on it):

- Content is **interned once** into immutable, content-addressed **segments**,
  each named by a **handle**. Identical content collapses to one segment.
- A segment carries a **domain** describing what kind of content it is and how it
  may appear in a body:
  - **Message** — a complete JSON message object (`{"role":…,"content":…}`),
    destined to be one element of a message array.
  - **Raw** — a complete, opaque request body (verbatim replay / prebuilt body).
  - **Text-only** — text whose bytes are not retained (only a token count
    survives); not directly spliceable as a body field.
  - **Token IDs** — an integer token array, for endpoints that send token IDs.
  - **Media** — an image/audio/video reference.
  - **Trace-hash-IDs** — authored source-trace block identities; carried for
    prefix accounting and never spliceable into a body.
- The store is **frozen** before dispatch: no new content is added on the hot
  path, so segment lookup is a pure read.

Body construction consumes handles; it never mints content.

## The body plan

A body plan is one of three shapes. An endpoint's `format_payload` returns one of
the first two; the third is a derived collapse the runtime applies afterwards.

- **Raw plan** — a single handle to a Raw segment: the degenerate whole-body case
  (recorded-payload replay, or a complete prebuilt body). Materialization emits
  that segment's bytes verbatim, with only the override tail folded in.
- **Fields plan** — an **ordered** list of `(field-name, field-value)` pairs
  describing a JSON object. Order is significant and preserved into the wire
  bytes (endpoints and downstream diff tooling depend on stable key order).
- **Prebuilt plan** — a complete body already serialized into inline bytes.
  Semantically identical to a Raw plan (materialization emits the bytes, folding
  in any override tail) and differs only in provenance: it holds its bytes
  inline rather than by handle, so it can be produced *after* the segment store
  is frozen. No endpoint constructs one; it is produced only by the static
  collapse described under [Precompute and caching](#precompute-and-caching),
  where a Fields plan with no per-dispatch field is materialized once at lowering
  so dispatch degrades to a refcount clone of those bytes. Because the collapse
  runs the ordinary materializer, a Prebuilt plan is byte-identical to the Fields
  plan it replaced.

Each field value is one of four kinds:

| Kind | Meaning | Serialized where |
|------|---------|------------------|
| **Literal** | An endpoint-generated scalar or small struct (`model`, `max_tokens`, `stream`, `stream_options`, a nested options object, a string array). | Once, into the buffer at materialize time. Small. |
| **Segment** | One stored content segment that is itself a complete JSON value (a Message object, or a Raw sub-body such as `tools`). | Never here — spliced from the store. |
| **Segments** | An ordered array of stored Message handles, joined with commas inside `[ ]`. | Never here — each element spliced from the store. |
| **Wires** | An ordered array of already-serialized JSON values **not** interned in the frozen store — dynamic or live-continuation content (e.g. a multi-turn assistant reply produced mid-run). Spliced identically to Segments; needs no store lookup. | Once, by the producer that created the wire — never here. |

The **Segment/Segments vs. Wires** distinction is the only subtlety: both splice
pre-serialized bytes and are byte-identical at the field level. The difference is
*provenance* — Segments reference the frozen store by handle; Wires carry their
bytes inline because the content was produced after the store was frozen (live
continuation) or otherwise never interned. A materializer resolves Segments via a
store read and Wires directly.

**The endpoint declares shape only.** It chooses field names, field order, which
slot is a Literal vs. a content handle, and which handles fill message arrays. It
**never** emits commas or brackets, and **never** serializes content. This keeps
every endpoint dialect free of wire-assembly logic and guarantees all dialects
share one splicer.

### Plan-building surface

Endpoints build a Fields plan through a small ordered builder vocabulary
(names are illustrative; the semantics are the contract):

- add a **literal** field (with scalar convenience forms: string / int / bool);
- add a **content segment** field, or an **optional** one that is omitted when the
  handle is absent (so absent fields do not appear as `null`);
- add a **segment array** field (a message array from stored handles);
- add a **wire array** field (a message array from inline pre-serialized wires);
- **replace** a message-array field's contents with pre-serialized wires while
  preserving the field's position (live-continuation splice);
- **set** a literal field: replace in place if the name already exists (position
  preserved) else append — matching insert-order map semantics.

A separate bridge builds a plan from an already-materialized JSON object (for a
legacy formatter that still emits a whole object): every top-level non-empty
array-of-objects becomes a **Wires** field (so it splices), and every other value
becomes a **Literal**. This lets an object-emitting formatter cross into the plan
path without special-casing, and is the natural adoption seam for a Python port
whose existing formatters return dicts.

## The override set

Per-dispatch variation is carried in a small **override set** — an ordered map of
field name → value — applied at materialize time. Typical members:

- `model` (the effective model for this request),
- the token-cap field (`max_tokens` **or** `max_completion_tokens`, per dialect),
- `stream`,
- `stream_options.include_usage` (forced on when streaming under server token
  counting),
- `seed` and arbitrary user `extra_body` keys.

The override set folds into the plan with **insert-order map semantics**: a key
that already exists as a literal is replaced **in place** (its position
unchanged); a new key is **appended** after the plan's own fields. This is exactly
what "take the object, `map.insert(k, v)` for each override, serialize" would
produce — and the materialized bytes must equal that, key-for-key.

Two equivalent foldings are permitted and must agree byte-for-byte:

- **Merge into the plan** — rewrite the plan's literal fields before
  materialization (used when downstream logic needs to read the effective
  model / token cap / stream flag back off the merged plan).
- **Override tail** — append the serialized override members after the plan's
  fields at materialize time.

Both express the same insert-order merge; the choice is an implementation detail,
not a wire difference.

## The two materializers

The plan is wire-agnostic. Materialization is **not**, and the split is
fundamental — a protobuf endpoint cannot splice pre-serialized JSON, and a JSON
endpoint cannot pack a tensor. So there are two materializers, chosen by the
endpoint's wire type, and **the endpoint picks neither** (the transport does):

- **JSON materializer.** Walks the Fields plan in order and concatenates into a
  single contiguous byte buffer: `{`, then for each field its quoted name, `:`,
  and its value — Literals serialized straight into the buffer, Segment/Segments/
  Wires spliced as raw bytes (arrays framed with `[ ]` and comma joins), then the
  override tail, then `}`. Message-object wires are validated as JSON objects as
  they are spliced (a malformed segment is a construction error, not a silent bad
  body). The result is **one** buffer — no scatter-gather — honoring transports
  that require a single complete body. A Raw plan takes a shortcut: emit the Raw
  segment's bytes and splice the override tail into its top-level object. A
  Prebuilt plan takes that same shortcut against its inline bytes, skipping the
  store lookup entirely — with an empty override set (the scheduled path's normal
  case) this reduces to handing back the already-built buffer.
- **Protobuf materializer.** Packs Token-IDs / tensor segments directly into the
  wire message's tensor-contents fields from the plan's structure — no
  intermediate JSON `Value`, no per-element walk. Here segments are *storage the
  codec reads*, not bytes it splices.

Live-continuation content (the **Wires** kind) materializes on the JSON path
exactly like stored Segments; the only difference upstream is that its bytes are
carried inline rather than fetched from the store.

## Precompute and caching

Because a Fields plan for a static conversation turn does not vary between
profiling-phase dispatches, plans are **precomputed and cached** per
`(conversation, turn)` when eligible, and dispatch clones the cached plan and
folds only the override set. Eligibility gates (all required):

- the endpoint declares its body **precomputable** — that is, static bind-time
  inputs fully determine it. This is the *default*; a dialect opts out only when
  its body genuinely cannot be known at bind: a template dialect (its Jinja
  template may reference per-dispatch identity such as `x_request_id`), raw
  passthrough (it splices the dispatching turn's own authored payload), and
  token-native composition (it sends exact per-turn raw token IDs). **Every other
  dialect qualifies, including the non-message-array input-array shapes** —
  embeddings, rankings, image retrieval. Those are in fact the biggest
  beneficiaries: a 32-image batch inlined as data URLs pays its whole
  `format_payload` serialization cost once at bind instead of once per timed
  request. Restricting the cache to message-array dialects would be leaving the
  largest win on the table;
- the conversation is not a graph/DAG conversation (those dispatch through a
  separate execution path), and either uses a static context mode — where no
  assembled turn depends on a live reply, so *every* turn caches — or a
  without-responses mode, where only turn 0 is response-independent and caches
  while continuation turns take the live path;
- the turn carries no per-turn endpoint override, no complete raw body, and no
  token-native raw token IDs;
- it is the **profiling** phase against the run's **default** endpoint.

Formatter failure at precompute is non-fatal: the slot simply stays empty and the
identical error resurfaces on the live dispatch path. The precompute pass is
idempotent — it rebuilds the whole cache from the current conversations on each
call.

### The static collapse

An eligible cached plan is then offered to a second, stronger optimization: if
its materialization carries **no per-dispatch field at all**, it collapses to a
**Prebuilt** plan — serialized once, cloned wholesale at dispatch. Two conditions
gate the collapse, and both exist to keep it byte-exact:

- the plan carries none of the per-dispatch literals (`model`, `stream`,
  `max_tokens`, `max_completion_tokens`, `max_output_tokens`) — these are exactly
  the fields the effective-field pass may rewrite per dispatch;
- the endpoint does not support streaming, since a streaming-capable endpoint's
  `stream` flag can be toggled per dispatch even when absent from the plan.

The scheduled path that consumes precomputed plans always dispatches with an
empty override set, so a collapsed body needs no per-dispatch mutation. If the
collapse's trial materialization fails, the uncollapsed plan is kept and the live
path surfaces the identical error.

The warmup phase bypasses the cache entirely (it folds the system prompt into the
first message inside the formatter, so its plans may differ); the profiling phase
reuses the cached plan. A Python port may skip both caching and the collapse
initially — they are performance optimizations, not correctness requirements —
but the *plan itself must be reusable*: materializing the same plan with the same
override set twice must produce identical bytes, and cloning a plan must not
alter the original.

## Invariants (the acceptance contract)

A conforming implementation, in any language, must satisfy all of:

1. **Content serializes exactly once.** No content bytes are produced at dispatch;
   only endpoint Literals and the override tail serialize per request, and both
   are small. Re-serializing message content on the hot path is a conformance
   failure even if the bytes happen to match.
2. **Byte-identity to the object-merge baseline.** For any plan, the materialized
   body must be **byte-for-byte identical** to constructing the equivalent JSON
   object (messages array + literals + overrides, in the plan's field order) and
   serializing it once with overrides applied via insert-order map semantics.
   This is the primary test oracle: build the object, `insert` the overrides,
   serialize, and assert equality against the materialized plan.
3. **Field order is preserved** from the plan into the wire, including where an
   in-place override replaces an existing field (position unchanged) versus a new
   override key (appended).
4. **Optional fields are omitted, not nulled** — an absent content handle
   produces no field.
5. **Whole-body plans replay verbatim** — a Raw segment's bytes, and a Prebuilt
   plan's inline bytes, reach the wire unchanged except for the override tail
   folded into the top-level object.
6. **The static collapse is byte-neutral** — collapsing an eligible Fields plan
   to a Prebuilt plan must not change a single byte on the wire. An
   implementation that collapses must produce, for every collapsed turn, the
   bytes the uncollapsed plan would have produced; the gates above (no
   per-dispatch literal, no streaming support) exist precisely to make that
   guarantee mechanical rather than probabilistic.
7. **Domain safety** — only Message and Raw segments are field-spliceable on the
   JSON path; splicing a Text-only / Token-IDs / Media segment as a JSON field is
   a construction error. Token/tensor segments reach the wire only through the
   protobuf materializer.
8. **Single contiguous body** on the JSON path — no scatter-gather — for
   transports that send one complete body.
9. **Numeric boundary discipline** — values are finite or explicitly absent at
   the serialization boundary (shared with the rest of the runtime).

## Testing

The defining tests assert **byte-identity**, not structural equality:

- A messages-array plan materializes byte-identically to the legacy message-splice
  path and to a hand-written expected byte string.
- A Wires plan and a Segments plan of the same content materialize identically.
- The object-bridge plan materializes byte-identically to serializing the source
  object once — covering messages array, scalars, nested objects, string arrays,
  and arbitrary user keys.
- Override folding matches "clone the object, `insert` each override, serialize"
  for both in-place (existing key) and append (new key) cases.
- A Raw plan reproduces the verbatim payload, with and without an override tail.
- A mixed literal/segment/array plan concatenates in declared order and parses
  back to the expected values.
- Splicing a non-spliceable segment domain as a JSON field is rejected.
- The static collapse produces a Prebuilt plan whose bytes equal the source
  Fields plan's, and **declines** to collapse both a plan carrying a per-dispatch
  literal and a plan from a streaming-capable endpoint — the negative case
  matters more than the positive one, since an over-eager collapse would freeze a
  field that must vary per request.

New endpoint dialects add a test asserting their exact wire bytes against a
deterministic mock server, per the repository's end-to-end verification
requirement.

## Source anchors (current Rust realization)

- `rust/runtime/src/body_plan.rs` — the plan vocabulary (`BodyPlan::{Raw, Fields,
  Prebuilt}`, `FieldValue`) and the JSON materializer (`JsonBodyMaterializer`),
  plus the object bridge (`BodyPlan::from_object`), override folding
  (`set_literal` / `merge_overrides`), and the static collapse
  (`prebuilt_if_static`, gated by `PER_DISPATCH_LITERALS`).
- `rust/runtime/src/dataset/materialize.rs` — the override set (`Overrides`) and
  the shared message-splice primitives the materializer reuses.
- `rust/runtime/src/dataset/segment.rs` — segment domains, handles, and the frozen
  store (`SegmentStore`, `Payload`).
- `rust/runtime/src/dataset/dataset.rs` / `dataset/request.rs` — profiling-phase
  plan precompute/cache (`precompute_body_plans`, `cached_body_plan`) and the
  dispatch-time materialize path.
- `rust/runtime/src/endpoints/registry.rs` — the `format_payload → BodyPlan`
  contract and the `precomputable_body` gate.
- `rust/runtime/src/transport/grpc/codec.rs` — protobuf encode-from-structure (the
  second materializer).

The lineage of this design (Python precursor: a raw-payload bytes fast path plus a
memory-mapped segment store) predates the plan abstraction; a future Python
implementation should target the contract above rather than that precursor.
