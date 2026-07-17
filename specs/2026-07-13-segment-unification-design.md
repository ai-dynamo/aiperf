<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Segment unification — one content IR, handles-only dispatch units

**Date:** 2026-07-13
**Status:** built — the core architecture (unified `body` handles, `content→segment`
lowering, domain-driven materialize/splice, formatter-at-lowering, live-continuation
segments) is implemented; two fields (`content`, `raw_messages`) are retained by
design (§9).
**Grounding:** `rust/runtime/src/dataset/` (`segment.rs`, `model.rs`, `dataset.rs`,
`request.rs`, `materialize.rs`, `compose.rs`, `multiturn.rs`, `loader/`) +
`rust/runtime/src/body_plan.rs`.
**Scope:** Collapse the overlapping body representations on `Turn` into **segment
handles in the one segment store**, make **lowering** the single compiler into it, and
make **dispatch materialize exactly one `Full<Bytes>` body by concatenation, never
re-serialization**. Live continuation writes runtime segments. The `Turn`→`Request`
*renaming* at the executor layer belongs to the greenfield/P1 specs; **this spec is
the content/storage unification**, which is what makes that rename clean.

## 1. Problem — five ways to say "the body"

`Turn` (`dataset/model.rs`) originally carried the request body in **five overlapping
optional fields**, resolved by implicit precedence:

- `content: SmallVec<[ContentGroup; 1]>` — inline authored groups (the one field that
  is *not* a segment handle).
- `messages: SmallVec<[Handle; 1]>` — pre-serialized message handles.
- `raw_messages: Option<Handle>` — a complete preformatted messages array.
- `raw_payload: Option<Handle>` — a complete prebuilt request body.
- `raw_token_ids: Option<Handle>` — token-native input.

Precedence: `raw_payload` won outright; else `raw_token_ids` when the endpoint
required it; else `content`/`messages`/`raw_messages` were **merged** into one array.
Plus additive side-fields (`tools`, `raw_system`, `extra_body`, `extra_headers`,
`request_parameters`).

The result was unclear (which field wins? which merge?) and it was *why* `Turn` felt
overloaded: it tried to be a single message, a message array, a raw body, and a token
array at once — none of which fits an agentic node.

But note: **four of those five were already `Handle`s into the segment store.** The
convolution was that they were *named by representation* instead of *typed by segment
domain*. The pool already has the domains for all of them.

## 2. The model — the segment store is the one content IR

`SegmentStore` (`dataset/segment.rs`) content-addresses every content shape across
**six disjoint BLAKE3 domains** (`segment.rs`): `message`, `text-only`, `raw`,
`token-ids`, `media`, `trace-hash-ids` (`SegmentDomain::kind_name`). Hashing is
prefix-dependent (a child folds its parent's content hash), so shared prefixes dedup
and identical content under different prefixes stays distinct. The public address is a
dense `Handle`.

Every body representation lowers into it, and the **domain is the discriminant**. A
`SegmentStore::domain(handle) -> SegmentDomain` accessor (returning the stable
`Payload::kind_name`) is the disjoint-domain discriminant that replaces the five-field
precedence. `Turn` collapses to **metadata + a unified `body: SmallVec<[Handle; 1]>` +
side handles**:

```rust
struct Turn {                       // (== Request at the executor; see naming specs)
    // metadata (unchanged): role, model, endpoint, max_tokens, streaming,
    // input_tokens, tool_tokens, timestamp_ms, delay_ms, trace_hash_ids, …
    body: SmallVec<[Handle; 1]>,        // message handles | one raw handle | one token-ids handle
    content: SmallVec<[ContentGroup; 1]>, // retained: input to endpoint-specific lowering (§9)
    raw_messages: Option<Handle>,       // retained: preformatted array; its own field (§9)
    tools: Option<Handle>,
    system: Option<Handle>,
    extra_body: Option<Handle>,
    extra_headers: Option<Handle>,
    request_parameters: Option<Handle>,
}
```

The **segment domain of `body`** replaces the old precedence:

- all `message` → a message array to format;
- one `raw` → a complete body (endpoint bypass);
- one `token-ids` → the token-native path.

The `if raw_payload … else …` branching is gone — it is a domain lookup. `body` is
populated once centrally in `Dataset::new` (the single freeze choke point — no churn
across the loader construction sites) by `Turn::populate_body`, mirroring the old
precedence: `raw_payload` handle wins, else `raw_token_ids`, else the ordered
`messages` handles. A `raw_payload`+`raw_token_ids` turn keeps both handles in `body`
so token-count validation is preserved.

## 3. Lowering — the single compiler into the store

Every input format lowers to segments **once, at load**, producing one frozen
`SegmentStore`:

| Input | Lowers to |
|---|---|
| authored `ContentGroup`s / synthetic prompts | `message` segments |
| recorded message array (`raw_messages`) | one `message` segment |
| recorded whole body (`raw_payload`) | one `raw` segment |
| authored token array (`raw_token_ids`) | one `token-ids` segment |
| media | `media` segments |
| tools / system / extras | their own segments |

`dag_jsonl` / `weka_trace` / `dynamo_trace` already enter one compiler → one store;
this extends the same discipline to the linear/authored path so there is **one
lowering surface**. Static chat/anthropic/responses content turns are rendered and
interned to `Message` segments at load-for-endpoint
(`Dataset::lower_messages_for_endpoint`, driven at endpoint-bind by
`lower_static_messages` in `multiturn.rs`); dispatch then **splices** the stored wire
(`resolve_turn` → `EndpointTurn.lowered` → `format_chat_messages`), zero content
re-serialize. A multimodal hash-key correctness fix folds the rendered wire into the
`Message` identity so same-text/different-media turns don't mis-dedup.

### 3a. The endpoint formatter runs at **lowering**, not dispatch (perf-critical)

A run targets **one endpoint** (known at config time), so the endpoint formatter —
`format_payload → BodyPlan` (see `2026-07-13-endpoint-body-construction-design.md`) —
runs **at lowering, once per turn**, producing the endpoint-shaped plan.
`Dataset::precompute_body_plans` (invoked at the bind seam alongside lowering) builds
each eligible turn's `BodyPlan` once and caches it (`Dataset.body_plans`, dense
`[conv][turn]`); dispatch clones the cached plan (`materialize_prepared`) instead of
calling `format_payload` per request. **Dispatch never calls `format_payload`** for
eligible turns — it only materializes (splice static + live segments, fold in the
small param overrides). Eligibility is gated (`precomputable_body()` trait method:
message-array shape; static context modes; profiling phase; default endpoint;
non-raw/non-token-native; graph excluded); the ineligible fallback set (dynamic-context
/ template / graph, and multi-endpoint graph nodes) lowers per node — still at load,
never per dispatch.

This is the whole "not slow" result. The gRPC audit found the *dispatch-time*
formatter was the slow path: `format_payload → Value → serde_json::to_vec` → JSON
bytes → `from_slice` back to `Value` → per-element tensor walk → `encode_to_vec` →
re-serialize to JSON, **per request**. Building the plan once at lowering eliminates
the per-request `Value`, the JSON round-trip, and the per-element walk.

## 4. Dispatch — materialize one `Full<Bytes>`, never re-serialize

Constrained by the transport (§6): the body must be **one contiguous `Full<Bytes>`**.
Materialization branches on the body domain and **concatenates pre-serialized segment
bytes** (the `materialize = concat, never re-serialize` rule). Both
`EndpointRequestMaterializer` paths select the raw body via `raw_body_handle()` (a
`store.domain(body[0]) == "raw"` lookup over the unified `body`, not a `raw_payload`
Option) and materialize it through `BodyPlan::raw` + `JsonBodyMaterializer`:

- **`raw`** → `store.bytes(handle).clone()` — a `Bytes` refcount bump (**~12 ns**,
  size-independent). This *is* the old `raw_payload` fast path; it needs no special
  field, just a `raw`-domain segment.
- **`message`** → concat the message-segment bytes into the pre-built envelope (baked
  at lowering, §3a; lowered messages spliced as `FieldValue::Wires`). One memcpy of the
  body (**~40–500 ns** by size); only the small param overrides are folded in.
- **`token-ids` / tensor** → the token-native body; for gRPC the in-tree endpoints
  build a BYTES text tensor from structure (see the endpoint spec §7 for why packed
  `raw_input_contents` is a proven exclusion, not pending work).

**No `serde_json::Value` on the dispatch content path** (that path is 9–13× slower;
§7), and — per §3a — **`format_payload` does not run here** for eligible turns.

## 5. Live continuation — runtime segments (built, lean form)

The one thing that cannot pre-lower: turn *N+1*'s body includes turn *N*'s **live**
reply. Each captured live reply is lowered **once at capture** under the default shape
(`multiturn.rs` `build_next_turn` → `ShapeLowerer::lower_turn`) and spliced via
`Turn.lowered` on subsequent dispatches, instead of re-serializing it every turn.
Byte-identical for shape-homogeneous conversations (the regime static lowering
supports); guarded by a two-dispatch idempotence test. The accumulated prefix dedups
via prefix-dependent hashing.

Because the body must stay one `Full` buffer (§6), a growing conversation **re-concats
the whole body each turn → O(depth²)** total. The microbench (§7) shows this is cheap
in practice (sub-µs/turn, ~60 µs total at 64 turns); it only bites at pathological
depth. Scatter-gather/multi-frame sending (which would make it O(depth)) is
**rejected** — see §6.

## 6. Constraints honored (do not relitigate)

- **One `Full<Bytes>` body, sent whole — no partial/multi-frame.** The request body is
  `TimedBody { inner: Full::new(bytes), … }` (`transport::http` connection send path).
  `SendCompletion` stamps the exact instant the *complete* body reaches end-of-stream,
  because hyper's `send_request().await` resolves at response headers, not body-sent;
  **cancellation-after-send / HTTP-499 arms its deadline off that instant.** A segment
  frame-list + `writev` would break that anchor and is out of scope. Materialization
  therefore always yields one contiguous `Bytes`.
- **Concat, never re-serialize.** Segments are pre-serialized bytes; materialize
  clones/concats them.

## 7. Evidence (microbench, release + LTO)

| body | `raw` (clone) | splice (concat) | serde re-serialize |
|---|---|---|---|
| ~1 KB | 12 ns | 40 ns | 350 ns |
| ~8.5 KB | 12 ns | 102 ns | 2137 ns |
| ~32 KB | 12 ns | 498 ns | 6398 ns |

Growing conversation, per-turn body assembly: full re-concat **O(depth²)** (241 → 929
ns/turn at 8 → 64 turns) vs. a hypothetical non-materializing frame list **O(depth)**
(~150–210 ns/turn) — the latter is unavailable under §6, and the absolute cost of the
former is acceptable. Takeaways: (a) a `raw` body is a flat 12 ns clone; (b) concat
beats serialize **~10×**, so the store pays off via **dedup + concat-not-reserialize**,
not via avoiding the memcpy.

## 8. Turn vs Segment (settled)

A **`Turn`** is the *structure* — one exchange's metadata + body/side handles. A
**`Segment`** is the *storage* — content-addressed bytes a handle points into. A Turn
*references* segments; it is not one. `Turn` stays the right word in the
**dataset/conversation** layer; at the **executor** layer the dispatched unit is a
`Request` (same handles-only shape) — which is exactly why a scheduled turn and a graph
node become the same thing.

## 9. What was built vs retained by design

Built:

- **`SegmentStore::domain(handle)`** and the six-domain discriminant (`segment.rs`).
- **`Turn.body`** populated once in `Dataset::new` via `Turn::populate_body`
  (`model.rs`).
- **`content → message` lowering** at load-for-endpoint
  (`lower_messages_for_endpoint` / `lower_static_messages`), with the multimodal
  hash-key fix; byte-parity proven by real (non-circular) oracles
  (`lowered_wire_matches_rendered_dispatch_wire_{text_only,responses,multimodal}`,
  `lowered_dispatch_body_is_byte_identical_to_pre_lowering`,
  `same_text_different_media_lowers_to_distinct_wires`).
- **Domain-driven dispatch materialize/splice** for raw (`raw_body_handle`) and message
  (spliced `Wires`) bodies; validated end-to-end against the real `aiperf-mock-server`
  (`tests/scheduled_real_mock.rs`).
- **Formatter-at-lowering** (`precompute_body_plans`, §3a); byte-identical by
  construction (differential cache-on/cache-off test across endpoint × context-mode ×
  overrides).
- **Live-continuation runtime segments** (§5).
- **`messages` / `raw_payload` / `raw_token_ids` deleted** from `dataset::model::Turn`
  (subsumed by `body` + domain).

Retained by design (a proven WONTFIX, not a shortcut):

- **`content` and `raw_messages` stay their own fields.** Adversarial verification
  confirmed both are load-bearing at dispatch. `content` is the required *input* to
  endpoint-specific lowering (which cannot run at load for carve-out turns, since the
  bound endpoint shape may be unknown then) and has three dispatch consumers with no
  `body` equivalent (completions/embeddings/rerank `turn.texts`, per-turn-override
  turns, warmup first-turn re-render). `raw_messages` is a preformatted message *array*
  whose exclusion from `body` is what keeps a Raw-domain `body[0]` unambiguously "the
  complete body"; it **coexists with `content`** on the accuracy dual-view turn
  (`loader/public.rs`), and folding either into `body` flips `resolve_turn`'s
  `lowered_content` discriminator and breaks warmup byte-parity. The field split *is*
  the discriminator. Pinned by a dispatch-level coexistence parity test.
- **`Turn`/`EndpointTurn` reconciliation** (the executor-layer naming, greenfield/P1
  specs) — naming only, no wire effect; not done here.

## 10. Non-goals

- Renaming `Turn`→`Request` (greenfield/P1 specs).
- Scatter-gather / multi-frame request bodies (rejected in §6).
- Redesigning `SendCompletion` / cancellation-after-send.
- Response-side segmenting (this is request-body content).

## 11. Open questions

1. Can a `body` ever legitimately mix domains (a `message` array *plus* a `raw`
   suffix)? Current precedence says no — the invariant is "one domain per `body`"
   (`raw_payload`+`raw_token_ids` coexistence is the deliberate exception for
   token-count validation).
2. ~~Where does per-endpoint formatting live?~~ **Resolved (§3a): at lowering.**
3. Runtime-segment lifetime: the live-reply store stays writable per run without
   contending the frozen store (the graph channel-store mechanism).

## 12. Related

- `2026-07-13-endpoint-body-construction-design.md` — the dispatch-side companion (handles → wire bytes).
- `2026-07-10-aiperf-rust-dataset-segment-seam-design.md` — the segment store this completes.
- `2026-07-13-p1-generic-execution-substrate-names.md` — the `Turn`→`Request` naming this enables.
- `2026-07-10-aiperf-transport-rust-port-design.md` — the `Full<Bytes>` + `SendCompletion` send path §6 protects.
- `2026-07-13-websocket-transport-design.md` — segments are transport-agnostic; a WS materializer consumes the same store (§6's one-body rule is HTTP-local, so WS is additive).
