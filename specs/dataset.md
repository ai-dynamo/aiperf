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

### Pipeline

- **Loaders** parse each real format: synthetic, single/multi-turn JSONL,
  random-pool, mooncake/bailian/burst_gpt traces, `dag_jsonl`, raw-payload,
  sharegpt, Hugging Face public datasets, and accuracy fixtures.
- **Composition** does turn finalization, ISL/OSL sequence-distribution sampling,
  context injection, model selection, and `max_tokens`.
- **Sampling strategies** (random, sequential, shuffle) operate on ids and are
  deterministic under a seed.
- **Tokenization** runs at load so segment ids can be token-keyed.
- **Materialization** splices a static segment plus dynamic
  predecessor-reply/live continuation content into the request body.

The pipeline adds endpoint-required load validation, payload-byte release, and
synthetic no-decode generation. Synthetic media generation lives here and selects
inline representation or persisted URLs through an injected publication trait (see
[content-server.md](content-server.md)). Direct `dag_jsonl` bypasses linear
`Dataset`/`Conversation` composition and produces graph plans plus the same frozen
store (see [graph-runtime.md](graph-runtime.md)).

## Source anchors

- `rust/runtime/src/dataset/` (`segment.rs` `SegmentStore`/domains, `model.rs`
  `Turn`, `dataset.rs`, `compose.rs`, `sampler.rs`, `materialize.rs`, `loader/`,
  `generator/`, `tokenizer.rs`, `synthesis.rs`, `media.rs`).
- `rust/runtime/src/body_plan.rs` (splice into wire bytes).
