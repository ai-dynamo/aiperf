<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Segment unification — one content IR, handles-only dispatch units

**Date:** 2026-07-13
**Status:** design (proposed; grounded in current code + microbench evidence)
**Scope:** Collapse the five overlapping body representations on `Turn` into **segment
handles in the one `SegmentPool`**, make **lowering** the single compiler into it,
and make **dispatch materialize exactly one `Full<Bytes>` body by concatenation,
never re-serialization**. Live continuation writes runtime segments. The
`Turn`→`Request` *renaming* at the executor layer belongs to the greenfield/P1
specs; **this spec is the content/storage unification**, which is what makes that
rename clean.

## 1. Problem — five ways to say "the body"

`Turn` (`rust/runtime/src/dataset/model.rs:211`) carries the request body in **five
overlapping optional fields**, resolved by implicit precedence:

- `content: SmallVec<[ContentGroup; 1]>` — inline authored groups (the one field
  that is *not* a segment handle).
- `messages: SmallVec<[Handle; 1]>` — pre-serialized message handles.
- `raw_messages: Option<Handle>` — a complete preformatted messages array.
- `raw_payload: Option<Handle>` — a complete prebuilt request body.
- `raw_token_ids: Option<Handle>` — token-native input.

Precedence today (`request.rs`): `raw_payload` wins outright (`:297`, `:389`,
`:944`); else `raw_token_ids` when the endpoint requires it (`:554`, `:944`); else
`content`/`messages`/`raw_messages` are **merged** into one array (`:1028`,
`raw_messages.extend(...)`). Plus additive side-fields (`tools`, `raw_system`,
`extra_body`, `extra_headers`, `request_parameters`).

The result is unclear (which field wins? which merge?) and it's *why* `Turn` feels
overloaded: it is trying to be a single message, a message array, a raw body, and
a token array at once — none of which fits an agentic node.

But note: **four of those five are already `Handle`s into the segment store.** The
convolution is that they are *named by representation* instead of *typed by segment
domain*. The pool already has the domains for all of them.

## 2. The model — `SegmentPool` is the one content IR

`SegmentPool` (`rust/runtime/src/dataset/segment.rs:232`) already content-addresses
every content shape across **six BLAKE3 domains** (`segment.rs:5-11`): `message`,
`text-only`, `raw`, `token-ids`, `media`, `trace-hash-ids`. Hashing is
prefix-dependent (a child folds its parent's content hash), so shared prefixes
dedup and identical content under different prefixes stays distinct. The public
address is a dense `Handle`.

**Lower every body representation into it, and let the domain be the discriminant:**

- `content` (inline groups) → lowered to a `message` segment at load.
- `messages` / `raw_messages` → `message` segments.
- `raw_payload` → a `raw` segment.
- `raw_token_ids` → a `token-ids` segment.

Then `Turn`/`Request` collapses to **metadata + body handles + side handles**:

```rust
struct Turn {                       // (== Request at the executor; see naming specs)
    // metadata (unchanged): role, model, endpoint, max_tokens, streaming,
    // input_tokens, tool_tokens, timestamp_ms, delay_ms, trace_hash_ids,
    // tool_walk_start, prerequisites, branch_ids, audio_duration_seconds
    body: SmallVec<[Handle; 1]>,        // message handles | one raw handle | one token-ids handle
    tools: Option<Handle>,
    system: Option<Handle>,             // was raw_system
    extra_body: Option<Handle>,
    extra_headers: Option<Handle>,
    request_parameters: Option<Handle>,
}
```

Gone: `content`, `messages`, `raw_payload`, `raw_messages`, `raw_token_ids`. The
**segment domain of `body`** replaces the five-field precedence:

- all `message` → a message array to format;
- one `raw` → a complete body (endpoint bypass);
- one `token-ids` → the token-native path.

The `if raw_payload … else …` branching disappears — it's a domain lookup.

## 3. Lowering — the single compiler into the pool

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

`content` — the last inline field — dies here: content groups lower to a
`message` segment at load, exactly like everything else. `dag_jsonl` / `weka_trace`
/ `dynamo_trace` already enter one compiler → one store; this extends the same
discipline to the linear/authored path so there is **one lowering surface**.

### 3a. The endpoint formatter runs at **lowering**, not dispatch (perf-critical)

A run targets **one endpoint** (known at config time), so the endpoint formatter —
`format_payload → BodyPlan` (see `2026-07-13-endpoint-body-construction-design.md`)
— runs **at lowering, once per turn**, producing the endpoint-shaped plan: static
message/tensor segments + `Splice` slots for live continuation + the param-tail
shape. **Dispatch never calls `format_payload`** — it only materializes (splice
static + live segments, stamp the small param tail).

This is not a style choice; it is the whole "not slow" result. The gRPC audit
(2026-07-13) found the *dispatch-time* formatter is the slow path: today
`format_payload → Value → serde_json::to_vec` (`request.rs:341`) → JSON bytes →
`from_slice` back to `Value` (`grpc.rs:370`) → per-element tensor walk
(`codec.rs:241-267`) → `encode_to_vec` → re-serialize to JSON (`grpc.rs:507`),
**per request**. Building the plan once at lowering eliminates the per-request
`Value`, the JSON round-trip, and the per-element walk. **Multi-endpoint** (a graph
node selecting an endpoint at runtime) is the only exception — it lowers **per
node/per endpoint**, still at load, never per dispatch.

## 4. Dispatch — materialize one `Full<Bytes>`, never re-serialize

This is constrained by the transport (§6): the body must be **one contiguous
`Full<Bytes>`**. Materialization branches on the body domain and **concatenates
pre-serialized segment bytes** (the `materialize = concat, never re-serialize`
rule):

- **`raw`** → `store.bytes(handle).clone()` — a `Bytes` refcount bump (**~12 ns**,
  size-independent). This *is* the old `raw_payload` fast path; it needs no special
  field, just a `raw`-domain segment.
- **`message`** → concat the message-segment bytes into the **pre-built envelope**
  (baked at lowering, §3a), close. One memcpy of the body (**~40–500 ns** by size);
  only the small param tail (`model`/`max_tokens`/`stream`) is **stamped**.
- **`token-ids` / tensor** → the token-native body; for gRPC the **packed segment
  bytes go straight into `raw_input_contents`** (`contents=None`), never a
  `Value::Array` walk — see `2026-07-13-endpoint-body-construction-design.md`.

**No `serde_json::Value` on the dispatch content path** (that path is **9–13× slower**;
§7), and — per §3a — **`format_payload` does not run here.** Dispatch splices static +
live segments and stamps the param tail. A per-request `format_payload → Value →
serialize` is exactly the slow path the gRPC audit found; it is prohibited on the
hot path.

## 5. Live continuation — runtime segments

The one thing that cannot pre-lower: turn *N+1*'s body includes turn *N*'s **live**
reply. Handle it the way the graph already does — the live reply is written as a
**runtime `message` segment** (the pool stays writable during the run, à la the
graph channel store), and turn *N+1*'s `body` references it. Materialization then
concats as in §4. The accumulated prefix dedups via prefix-dependent hashing.

Because the body must stay one `Full` buffer (§6), a growing conversation
**re-concats the whole body each turn → O(depth²)** total. The microbench (§7) shows
this is cheap in practice (sub-µs/turn, ~60 µs total at 64 turns); it only bites at
pathological depth. Scatter-gather/multi-frame sending (which would make it O(depth))
is **rejected** — see §6.

## 6. Constraints honored (do not relitigate)

- **One `Full<Bytes>` body, sent whole — no partial/multi-frame.** The request body
  is `TimedBody { inner: Full::new(bytes), … }` (`transport_http/client/connection.rs:135-160`).
  `SendCompletion` (`:56`, `:112`) stamps the exact instant the *complete* body
  reaches end-of-stream, because hyper's `send_request().await` resolves at response
  headers, not body-sent (`:131-134`); **cancellation-after-send / HTTP-499 arms its
  deadline off that instant.** A segment frame-list + `writev` would break that anchor
  and is out of scope. Materialization therefore always yields one contiguous `Bytes`.
- **Concat, never re-serialize.** Segments are pre-serialized bytes; materialize
  clones/concats them.

## 7. Evidence (microbench, release + LTO)

| body | `raw_payload` (clone) | splice (concat) | serde re-serialize |
|---|---|---|---|
| ~1 KB | 12 ns | 40 ns | 350 ns |
| ~8.5 KB | 12 ns | 102 ns | 2137 ns |
| ~32 KB | 12 ns | 498 ns | 6398 ns |

Growing conversation, per-turn body assembly: full re-concat **O(depth²)** (241 →
929 ns/turn at 8 → 64 turns) vs. a hypothetical non-materializing frame list
**O(depth)** (~150–210 ns/turn) — the latter is unavailable under §6, and the
absolute cost of the former is acceptable. Takeaways: (a) a `raw` body is a flat
12 ns clone; (b) concat beats serialize **~10×**, so the pool pays off via **dedup +
concat-not-reserialize**, not via avoiding the memcpy.

## 8. Turn vs Segment (settled)

A **`Turn`** is the *structure* — one exchange's metadata + body/side handles. A
**`Segment`** is the *storage* — content-addressed bytes a handle points into. A
Turn *references* segments; it is not one. `Turn` stays the right word in the
**dataset/conversation** layer (a conversation genuinely has turns); at the
**executor** layer the dispatched unit is a `Request` (same handles-only shape) —
which is exactly why a scheduled turn and a graph node become the same thing.

## 9. Migration (staged; suite green each step)

1. **Add `body: SmallVec<[Handle; 1]>` + a `SegmentStore::domain(handle)` accessor.**
   Populate `body` from today's fields at load; keep the old fields as deprecated
   shims that lowering fills. No behavior change.
2. **Lower `content` to a `message` segment at load** (kill the one inline field);
   route dispatch through `body` + domain instead of the five-field precedence in
   `request.rs`.
3. **Delete `content`/`messages`/`raw_payload`/`raw_messages`/`raw_token_ids`** and
   the `raw_payload`-wins branching once all producers/consumers use `body`.
4. Reconcile the second `Turn` (`endpoints/models.rs:80`) and `EndpointTurn`
   (`request.rs` resolved form) onto the same handles model.

Parity: `raw_payload`/token/message dispatch must produce byte-identical wire bodies
through `body`+domain as through the old fields; existing dataset/dispatch parity
tests are the guard.

## 10. Non-goals

- Renaming `Turn`→`Request` (greenfield/P1 specs).
- Scatter-gather / multi-frame request bodies (rejected in §6).
- Redesigning `SendCompletion` / cancellation-after-send.
- Response-side segmenting (this is request-body content).

## 11. Open questions

1. Can a `body` ever legitimately mix domains (e.g. a `message` array *plus* a `raw`
   suffix)? Current precedence says no — enforce "one domain per `body`" as an
   invariant, or model a `Mixed` case explicitly.
2. ~~Where does per-endpoint formatting live?~~ **Resolved (§3a): at lowering.** The
   formatter runs once per turn at load (endpoint known at run config), producing the
   endpoint-shaped plan; dispatch never formats. Dispatch-time formatting is the slow
   path the gRPC audit measured. Multi-endpoint (graph per-node endpoint selection)
   lowers per node — still at load.
3. Runtime-segment lifetime: the live-reply pool must stay writable per run without
   contending the frozen store — reuse the graph channel-store mechanism.

## 12. Related

- `2026-07-10-aiperf-rust-dataset-segment-seam-design.md` — the segment store this completes.
- `2026-07-13-greenfield-execution-vocabulary.md` / `2026-07-13-p1-generic-execution-substrate-names.md` — the `Turn`→`Request` naming this enables.
- `2026-07-10-aiperf-transport-rust-port-design.md` — the `Full<Bytes>` + `SendCompletion` send path §6 protects.
- `2026-07-13-websocket-transport-design.md` — segments are transport-agnostic; a WS materializer consumes the same pool (§6's one-body rule is HTTP-local, so WS is additive).

## Addendum — 2026-07-13 (implementation status: stage 1 + raw stage 2 landed)

Grounded in `rust/runtime/src/`; the four migration stages in §9 are landing
incrementally with the suite green each step. **Built so far:**

- **§2/§9 stage 1 — `SegmentStore::domain(handle)`** (`dataset/segment.rs`): the
  disjoint-domain discriminant, returning the stable `Payload::kind_name`
  (`message`/`raw`/`token-ids`/…). Unit-tested.
- **§2/§9 stage 1 — `Turn.body: SmallVec<[Handle; 1]>`** (`dataset/model.rs`):
  the unified body handles, populated once centrally in `Dataset::new` (the
  single freeze choke point — no churn across the fifteen loader construction
  sites) by `Turn::populate_body`, mirroring today's precedence (`raw_payload`
  wins, else `raw_token_ids`, else the ordered `messages` handles;
  content-only / `raw_messages`-only turns stay formatter-driven with an empty
  `body`). The legacy fields remain authoritative; `serde(default,
  skip_serializing_if)` keeps existing dataset round-trips byte-stable.
- **§4 — raw dispatch on the new materializer, domain-driven** (`dataset/request.rs`):
  both `EndpointRequestMaterializer` paths select the raw body via
  `raw_body_handle()` (a `store.domain(body[0]) == "raw"` lookup over the unified
  `body`, not the `raw_payload` Option) and materialize it through
  `BodyPlan::raw` + `JsonBodyMaterializer` (see the companion endpoint spec).
  Byte-identical to the prior `store.build_body` path — guarded by the existing
  `raw_payload_is_byte_exact…` and `token_ids_inside_an_ordinary_raw_body…`
  tests — and validated end-to-end against the real `aiperf-mock-server` binary
  (`tests/scheduled_real_mock.rs`).

## Addendum — 2026-07-14 (implementation status: core architecture built; fresh-context approved)

The staged migration completed through a workflow-orchestrated build (stages 0/A/B/C).
An independent fresh-context reviewer verified the below against the code and
**approved both specs with documented caveats**; the full `aiperf` lib suite
(706 tests) + every `endpoints_*` byte-parity suite + the lowering oracles are green.

**Built (this supersedes the "not yet built" list above):**

- **§3 content→segment lowering** (`Dataset::lower_messages_for_endpoint`,
  `dataset/dataset.rs`; driven at endpoint-bind by `lower_static_messages` in
  `multiturn.rs`): static chat/anthropic/responses content turns are rendered and
  interned to `Message` segments at load-for-endpoint; dispatch **splices** the
  stored wire (`resolve_turn` → `EndpointTurn.lowered` → `format_chat_messages`),
  zero content re-serialize. Byte-parity proven by real (non-circular) oracles:
  `lowered_wire_matches_rendered_dispatch_wire_{text_only,responses,multimodal}`
  and `lowered_dispatch_body_is_byte_identical_to_pre_lowering`. Includes a
  **multimodal hash-key correctness fix** (fold the rendered wire into the
  `Message` identity so same-text/different-media turns don't mis-dedup:
  `same_text_different_media_lowers_to_distinct_wires`).
- **§4 dispatch materialize/splice, domain-driven** for both raw and message
  bodies (raw via `raw_body_handle`; lowered messages spliced as `Wires`).
- **§9 stage 3 (partial, as designed): `messages`/`raw_payload`/`raw_token_ids`
  deleted** from `dataset::model::Turn`. `content` and `raw_messages` are
  **retained by design** — `content` is the required *input* to endpoint-specific
  lowering (which cannot run at load, since the bound endpoint shape is unknown
  then, and carve-out turns render it at dispatch); `raw_messages` is a
  preformatted message *array* whose exclusion from `body` is what keeps a
  Raw-domain `body[0]` unambiguously "the complete body". The
  `raw_payload`+`raw_token_ids` coexistence keeps both handles in `body` so the
  token-count validation is preserved.

- **§3a formatter-at-lowering — built** (`Dataset::precompute_body_plans`,
  invoked at the bind seam alongside lowering): each eligible turn's `BodyPlan`
  is built once at endpoint-bind and cached (`Dataset.body_plans`, dense
  `[conv][turn]`); dispatch clones the cached plan (`materialize_prepared`)
  instead of calling `format_payload` per request, so no `format_payload`/`Value`
  construction on the hot path for eligible turns. Eligibility is gated
  (`precomputable_body()` trait method; message-array shape; static context modes
  `MessageArrayWithResponses`/`DeltasWithResponses`; profiling phase; default
  endpoint; non-raw/non-token-native; graph excluded). **Byte-identical by
  construction** — the cache only relocates the identical `format_payload` call
  in time and re-folds overrides on a clone — guarded by a differential test that
  materializes each body cache-on vs cache-off and asserts byte-equality across an
  endpoint × context-mode × overrides matrix.
- **§5 live continuation — built (lean form)** (`multiturn.rs` `build_next_turn`):
  each captured live reply is lowered **once at capture** under the default shape
  (`ShapeLowerer::lower_turn`) and spliced via `Turn.lowered` on subsequent
  dispatches, instead of re-serializing it every turn. Byte-identical for
  shape-homogeneous conversations (the regime static lowering already supports);
  guarded by a two-dispatch idempotence test.

**Deliberately not built (documented exclusions):**
- **§9 full field deletion — `content` + `raw_messages` retention is a proven
  WONTFIX**, not a shortcut. Adversarial verification confirmed both are
  load-bearing at dispatch: `content` has three dispatch consumers with no `body`
  equivalent (completions/embeddings/rerank `turn.texts`, per-turn-override turns,
  warmup first-turn re-render); `raw_messages` **coexists with `content`** on the
  accuracy dual-view turn (`loader/public.rs`), and folding either into `body`
  flips `resolve_turn`'s `lowered_content` discriminator and breaks warmup
  byte-parity. The field split *is* the discriminator. Pinned by a dispatch-level
  coexistence parity test.
- `Turn`/`EndpointTurn` reconciliation (§9 stage 4) — naming, no wire effect.
