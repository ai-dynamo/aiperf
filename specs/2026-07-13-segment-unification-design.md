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

`Turn` (`rust/aiperf/src/dataset/model.rs:211`) carries the request body in **five
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

`SegmentPool` (`rust/aiperf/src/dataset/segment.rs:232`) already content-addresses
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

`content` — the last inline field — dies here: endpoint-agnostic groups lower to a
`message` segment at load, exactly like everything else. `dag_jsonl` / `weka_trace`
/ `dynamo_trace` already enter one compiler → one store; this extends the same
discipline to the linear/authored path so there is **one lowering surface**.

## 4. Dispatch — materialize one `Full<Bytes>`, never re-serialize

This is constrained by the transport (§6): the body must be **one contiguous
`Full<Bytes>`**. Materialization branches on the body domain and **concatenates
pre-serialized segment bytes** (the `materialize = concat, never re-serialize`
rule):

- **`raw`** → `store.bytes(handle).clone()` — a `Bytes` refcount bump (**~12 ns**,
  size-independent). This *is* the old `raw_payload` fast path; it needs no special
  field, just a `raw`-domain segment.
- **`message`** → write the JSON envelope + params, concat the message-segment
  bytes into the array, close. One memcpy of the body (**~40–500 ns** by size);
  params (`model`/`max_tokens`/`stream`) are **stamped**, not re-derived.
- **`token-ids`** → the token-native body from the token segment.

No `serde_json::Value` round-trip on the content path (that path is **9–13× slower**;
§7). `format_payload` wraps pre-serialized bytes and stamps params.

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
2. Where does per-endpoint formatting live once `content` is pre-lowered — a single
   canonical `message` segment formatted at dispatch, or per-endpoint lowering when a
   trace targets multiple endpoints? (Lean: canonical message segment + dispatch-time
   envelope/param stamping.)
3. Runtime-segment lifetime: the live-reply pool must stay writable per run without
   contending the frozen store — reuse the graph channel-store mechanism.

## 12. Related

- `2026-07-10-aiperf-rust-dataset-segment-seam-design.md` — the segment store this completes.
- `2026-07-13-greenfield-execution-vocabulary.md` / `2026-07-13-p1-generic-execution-substrate-names.md` — the `Turn`→`Request` naming this enables.
- `2026-07-10-aiperf-transport-rust-port-design.md` — the `Full<Bytes>` + `SendCompletion` send path §6 protects.
