<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native vLLM speculative-decode wire contract

## Status

Design for the native Rust port of origin/main commit
`810fd8bdd40a1c35b64d487b3b8487f0a71a0f6b`. This spec supersedes only the
vLLM wire-location and histogram-shape sections of
`2026-08-25-native-spec-decode-acceptance-metrics.md`; that earlier spec's
engine-neutral record, metrics, folding, artifacts, and console contracts remain
authoritative.

## Purpose

Consume vLLM's reviewed per-request speculative-decoding metrics without an
author flag and without coupling their availability to server-side token
counting. Chat and completions must accept the same wire object in streaming
and non-streaming responses, normalize it to the existing native acceptance
record, and preserve all existing metrics and artifact behavior.

## Wire contract

The only accepted vLLM location is:

```text
response.metrics.speculative_decoding
```

The value must be a JSON object. A missing `metrics` object, a missing or null
`speculative_decoding` member, or a non-object member means no stats. The client
does not inspect or constrain `choices` when deciding whether root metrics are
attributable: current vLLM leaves the member null for `n > 1` and multi-prompt
completions.

For a non-streaming response the object shares the response root with content
and usage. For a stream it appears on the final usage frame, which has an empty
`choices` array. That frame is still reduced for usage and acceptance but
contributes no token observation or content. Across decoded response frames the
last non-empty accepted object is authoritative.

The obsolete `choices[0].speculative_decoding_stats` location is not accepted.
The streamed-chat typed codec therefore carries no spec-decode field and does
not allocate a generic JSON value on ordinary token or finish chunks. The
metrics-bearing usage chunk already falls through to the generic JSON path,
because it carries usage, and is captured there before terminal observation.

## Dense histogram normalization

`acceptance_histogram` is a dense JSON integer array. Array index `j` means
exactly `j` accepted draft tokens and the element is the number of verification
steps in that bucket. Deserialization accepts non-negative JSON integers only;
negative numbers, floats, booleans, strings, nulls, objects, or nested arrays
are invalid. Zero-count entries are omitted when constructing the canonical
`BTreeMap<u64, u64>`.

When `num_spec_tokens` is present, the dense array length must equal
`num_spec_tokens + 1`. Length arithmetic and index conversion are checked. When
the field is absent, no width assumption is made. Existing canonical checks
remain mandatory after conversion:

- bucket counts sum to `num_spec_steps`;
- the index-weighted sum equals `num_accepted_draft_tokens`;
- accepted drafts do not exceed proposed drafts;
- optional detailed arrays have `num_spec_steps` entries and reconcile to their
  aggregates; and
- each detailed accepted count is no greater than its drafted count.

A malformed payload logs one structured warning in the request dispatch and
degrades to absent acceptance stats. It does not fail an otherwise successful
request.

## Request negotiation

Chat and completions construct the normal payload, then merge endpoint extras,
then the latest turn's `extra_body`. Only after those merges do they inspect the
effective `stream` member.

If effective `stream` is exactly `true`:

- absent or null `stream_options` becomes
  `{"include_usage": true}`;
- an object is copied and receives `include_usage: true` only when that member
  is absent;
- explicit `include_usage: false` remains false;
- every other object member is preserved; and
- a non-object, non-null authored value is left untouched, allowing the server
  to apply its ordinary request validation.

If effective `stream` is false, absent, or not a boolean true, the formatter
does not synthesize `stream_options`. Existing authored values remain authored;
the formatter does not delete them. The rule is independent of
`use_server_token_count`: that option continues to choose the visible token
count source, while the trailing usage frame is transport metadata required by
vLLM's metrics wire.

Responses API request shaping is unchanged. The upstream behavior and the vLLM
wire apply to the OpenAI Chat Completions and legacy Completions endpoints.

## Mock and integration contract

The opt-in Rust mock fixture emits one reviewed canonical object with dense
histogram `[1, 1, 2, 3, 1]`, eight verification steps, 18 accepted drafts, 32
proposed drafts, fixed draft width four, and detailed arrays. Non-streaming chat
and completions place it under root `metrics.speculative_decoding`. Streaming
chat and completions place it on the empty-choice usage frame and emit no stats
when `include_usage` is false.

Focused tests cover exact request bodies, root extraction, dense normalization,
malformed degradation, and mock wire layout. A transport integration sends a
real request through Hyper/SSE and proves the empty-choice usage frame reaches
the terminal metrics record. Product E2E launches the real `aiperf profile`
binary against the in-process mock for both chat and completions with streaming
enabled but without authoring `stream_options` or enabling server token counts.
It asserts the existing scalar metrics, pooled histogram, canonical JSONL
record, and clean absence when the mock fixture is disabled.

## Cache boundary

Native request bodies have no persistent preformatted mmap cache. A resolved
run builds and owns its `BodyPlan` values from that run's endpoint policy, so
the changed automatic field is reflected immediately and cannot reuse a stale
cross-run body. The Python `MANIFEST_VERSION` increment has no native analog.

## Source anchors

- `rust/runtime/src/endpoints/implementation.rs`
- `rust/runtime/src/endpoints/spec_decode.rs`
- `rust/runtime/src/transport/http/sink/endpoint_dispatch.rs`
- `rust/mock-server/src/handlers.rs`
- `rust/e2e-tests/tests/test_spec_decode_acceptance.rs`
