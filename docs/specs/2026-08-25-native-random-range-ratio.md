<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native random range ratio and reference random corpus

## Status

Implemented and independently review-approved port design for origin-main tracker 56,
upstream `94fee7338b`. Closure evidence is recorded in
`docs/origin-main-findings/commit-056-94fee7338b.md`.

## Public contract

Synthetic prompt configuration accepts `random_range_ratio` and
`random_corpus_style` (`vllm`, default, or `sglang`). The CLI exposes the equivalent
`--random-range-ratio` and `--random-corpus-style` flags. A scalar ratio applies to both
ISL and OSL; vLLM additionally accepts an object/string object with `input` and `output`.

Ratio mode requires explicitly authored fixed ISL and OSL means. It rejects sequence
mixtures, non-zero ISL/OSL standard deviation, non-finite ratios, and incompatible style
shapes. Validation happens while projecting the dataset, before tokenization or traffic.

## Sampling contract

The vLLM style accepts each ratio in `[0, 1)`. After subtracting the tokenizer's known
automatic special-token count from the authored input mean, its inclusive bounds are
`floor(mean*(1-r))..=ceil(mean*(1+r))`; output is floored at one, while a zero adjusted
input body is representable only when an additive prefix makes the final prompt nonempty.

The SGLang style accepts one ratio in `[0, 1]`, uses
`max(1, floor(mean*r))..=mean`, and subtracts known special tokens from each sampled ISL,
floored at one.

For `entries = n`, one style-owned generator draws all `n` ISLs, then all `n` OSLs, then
all `n` prompt offsets. vLLM uses the existing byte-exact NumPy PCG64 adapter. SGLang uses
a private NumPy-RandomState-compatible MT19937 adapter seeded by the XOR fold of 64-bit
words. Cached pairs and offsets are consumed in authored conversation/turn order; a
multi-turn workload that exceeds `n` falls back to a deterministic style-owned stream and
emits one warning because reference parity no longer exists past the cache boundary.

## Prompt generation

In vLLM mode the base pool is the tokenizer's allowed random-token IDs, excluding special
tokens. In SGLang mode it is dense `0..vocab_size`. Request `i` with offset `o` receives
`pool[(o+i+j) % pool.len()]` for token `j`. Offset bounds use `vocab_size`, as the
reference does, even when the vLLM allowed pool is smaller. Existing native decode/encode
repair remains authoritative for text endpoints with the upstream ten-attempt budget,
and the exact-ID path remains authoritative for raw-token endpoints. Independently
sampled random prompts outside ratio mode subtract the tokenizer's automatic special
tokens after the ISL draw, floored at one, so the authored ISL remains the server-side
budget there too.

Prefix IDs are additive and assembled before one decode. Building a prefix must not
advance the body request ordinal. A body length of zero is valid only when a nonempty
prefix produces a nonempty final prompt.

## Ownership and scope

The immutable checked policy and preseed vectors live in the dataset layer. The synthetic
composer owns their consumption; transports and schedulers never sample them. Existing
recorded/hash-backed random corpus behavior stays unchanged unless it is configured
through this synthetic ratio contract.

## Errors and observability

Library boundaries use `DatasetError::Validation` with the authored field/style and
accepted interval. Cache exhaustion is one structured `warn!` event containing style,
preseed size, and consumed ordinal. No per-token logging is added.

## Acceptance

The port is complete only when parsing, checked bounds, pinned PCG64 and MT19937 vectors,
offset arithmetic, special-token pool policy, exact token length, prefix/zero guards, raw
token composition, and mock-server request evidence pass under the shared seed contract.

In addition to Rust-owned fixtures, a heavy A/B parity gate launches the actual Python
profile and the actual native `aiperf profile` against the same deterministic
request-capturing Rust mock. Equivalent authored Config-v2 files cover multiple seeds,
scalar/split and boundary ratios, both corpus styles, zero/two server-added special
tokens, and all-ISL→all-OSL→offset order. The server's ordered captures must match for
method, route, content type, exact outbound UTF-8 body bytes, and re-tokenized prompt IDs.
The native half must traverse config parsing, projection, dataset construction, prompt
generation, endpoint body planning, and production HTTP transport; test-local request
serialization is not evidence.

Before review, three separate semantic audits must enumerate the complete upstream delta
and native equivalent with executable evidence: RNG/reference streams and boundaries;
config/dataset/formatter/prefix semantics; and Python-to-native production protocol and
output behavior. Each audit must report no unresolved divergence after the final rebuilt
13-test/48-capture gate.
