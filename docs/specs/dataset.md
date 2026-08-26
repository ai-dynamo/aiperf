<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dataset and segment store

## Purpose

`aiperf_runtime::dataset` is the input-resolution plane: one content-addressed
segment store with dense integer handles, plus the loader → compose →
store → sampler → materializer pipeline shared by the runner, evaluator-authored
static accuracy, the graph runtime, and the offline adapter. Conversations and
turns carry handles, not bytes. This record owns the content IR (storage and
lowering); how handles become wire bytes belongs to
[endpoint-body-construction.md](endpoint-body-construction.md).

## Built

### Segment store — the one content IR

`SegmentStore` content-addresses every content shape across six disjoint BLAKE3
domains: `message`, `text-only`, `raw`, `token-ids`, `media`, and
`trace-hash-ids`. Hashing is prefix-dependent — a child folds its parent's content
hash — so shared prefixes dedup and identical content under different prefixes
stays distinct, and KV-cache prefix-reuse reasoning falls out for free. The public
address is a dense `Handle`; bytes live exactly once, in the store.

`Turn` is metadata plus a unified `body: SmallVec<[Handle; 1]>` and side handles
(`tools`, `system`, `extra_body`, `extra_headers`, `request_parameters`,
`raw_messages`, and `content` inputs). The segment domain of `body` is the
dispatch discriminant: all `message` handles → a message array to format; one
`raw` → a complete body (endpoint bypass); one `token-ids` → the token-native
path. `body` is populated once centrally at dataset freeze.
Lowering is the single compiler into the store; validated content-addressed
raw-token handles (`Payload::TokenIds` / `Turn::raw_token_ids`) support token-in
input.

A `Segment` pairs a `Payload` with its `SegmentId` — the 32-byte BLAKE3 content
address. `Role` is framed into that hash, so the same text under a different role
is a distinct segment. Interning is domain-specific (`intern_message`,
`intern_text`, `intern_raw`, `intern_token_ids`, `intern_media`,
`intern_trace_hash_ids`), each taking an optional prefix parent whose id folds
into the child's.

### Write side: pool, freeze, thaw

The store has two states, and the transition between them is load-bearing.

```mermaid
stateDiagram-v2
    [*] --> Pool: SegmentPool::new()
    Pool --> Pool: intern_*(parent, payload)<br/>dedup via SegmentId → Handle map
    Pool --> Frozen: freeze()<br/>arena into_boxed_slice,<br/>write-side map discarded
    Frozen --> Pool: thaw(&dyn SegmentStore)<br/>rebuild arena 0..len through the trait,<br/>reconstruct ids from stored SegmentIds
    Frozen --> [*]: shared across worker threads

    note right of Frozen
        Dispatch reads only this state.
        Lookup is an arena index — no hashing,
        no allocation, no synchronization.
    end note
    note left of Pool
        Handle indices are append-only.
        A thaw→intern→freeze cycle never
        renumbers an existing handle.
    end note
```

`SegmentPool` is the mutable write side: a `Vec<Segment>` arena plus a
`SegmentId → Handle` map for dedup. `freeze` converts the arena to a boxed slice
and drops the map — the frozen `InMemorySegmentStore` is a pure dense array,
shareable across worker threads with no lock.

`thaw` reopens a frozen store as a pool, **preserving every existing handle index
and stored `SegmentId` exactly**. It rebuilds the arena by walking `0..len`
through the `SegmentStore` trait rather than downcasting, and reconstructs the
dedup map from stored ids rather than re-hashing — so content keeps the identity
it was interned under even if the hashing scheme later changes.

That stability is what makes endpoint lowering possible:
`lower_messages_for_endpoint` runs *after* the dataset is composed and frozen, and
appends freshly-rendered `Message` wires for the selected endpoint. Because new
segments append after the existing arena, every handle minted during composition
stays valid across the cycle.

### Pipeline

- **Loaders** parse each real format. Each is paired one-to-one with a
  format-specific composer in `LoaderRegistry::register_builtin_formats` and
  resolved either by explicit format name or by ordered structural
  auto-detection over a probe row. The registered set is `synthetic`,
  `synthetic_rankings`, `single_turn`, `multi_turn`, `random_pool`,
  `raw_payload`, `inputs_json`, `exgentic`, `exgentic_v2`, `sharegpt`,
  `mt_bench`, `mmvu`, `spec_bench`, `speed_bench`, `accuracy`,
  `sagemaker_data_capture`, the trace formats `mooncake_trace`,
  `bailian_trace`, `burst_gpt_trace`, and `baseten_trace`, and the Hugging Face
  formats `hf_asr`, `hf_instruction_response`, `hf_conversation`, and the
  field-inferring `hf` auto loader. `dag_jsonl` is deliberately *not* in this
  registry — it takes the graph path below.
- **Composition** does turn finalization, ISL/OSL sequence-distribution sampling,
  context injection, model selection, and `max_tokens`. `--system-prompt` and
  `--system-prompt-file` resolve once at startup to one exact owned string for
  synthetic, file, and public datasets. After format-specific leading-system
  hoisting, composition installs that string or prepends it to authored system
  text with exactly two newlines, then rebases every prefix-dependent descendant
  handle. The generated synthetic user ISL remains unchanged and additive to
  the system tokens; equal resolved text has equal segment identity regardless
  of whether it came from a flag or file path.
- **Sampling strategies** (random, sequential, shuffle) operate on ids and are
  deterministic under a seed.
- **Tokenization** runs at load when semantic text must become token-keyed
  segments. Verbatim `raw_payload` and `inputs_json` bodies remain opaque,
  preserve exact wire bytes, and leave composed `Turn::input_tokens` unset
  (`None`). Only an endpoint declaring `requires_raw_token_ids` (such as
  `vllm_generate`) causes those loaders to validate and intern authored
  `token_ids` and record `Some(length)`; text is never BPE-encoded to
  synthesize IDs for that path.
- **Materialization** splices a static segment plus dynamic
  predecessor-reply/live continuation content into the request body.

The pipeline adds endpoint-required load validation, payload-byte release, and
synthetic no-decode generation. Synthetic media generation lives here and selects
inline representation or persisted URLs through an injected publication trait (see
[content-server.md](content-server.md)). Direct `dag_jsonl` bypasses linear
`Dataset`/`Conversation` composition and produces graph plans plus the same frozen
store (see [graph-runtime.md](graph-runtime.md)).

## Source anchors

- `rust/runtime/src/dataset/segment.rs` — `Handle`, `SegmentId`, `Role`,
  `Payload`, `SegmentDomain`, `Segment`, the `SegmentStore` trait, the mutable
  `SegmentPool` (`intern_*`, `thaw`, `freeze`), and the frozen
  `InMemorySegmentStore`.
- `rust/runtime/src/dataset/` (`model.rs` `Turn`, `dataset.rs`
  `lower_messages_for_endpoint`, `compose.rs`, `sampler.rs`, `materialize.rs`,
  `loader/`, `generator/`, `tokenizer.rs`, `synthesis.rs`, `media.rs`).
- `rust/runtime/src/body_plan.rs` (splice into wire bytes).
