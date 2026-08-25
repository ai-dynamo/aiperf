<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Commit 008 — `5566aae1e1`

Upstream subject: `perf(dataset): batch-encode ShareGPT to fix 300s configuration timeout (#1206)`.

## Scope decision

**Applicable native dataset-composition performance port.** Upstream removes
one tokenizer call per ShareGPT prompt and completion from the Python loader's
sequential validation loop. It flattens all adjacent human/assistant pairs,
tokenizes their text through bounded batch calls, and maps the resulting token
lengths back to their original rows before applying the existing admission
rules. This is a behavior-preserving performance fix for the approximately
90,000-row public ShareGPT corpus, whose old configuration path exceeded the
300-second default timeout.

Native Rust does not execute that Python loader. It has its own
`ShareGptComposer`, so merging the Python change alone cannot affect native
profile configuration. The native composer parallelizes rows with Rayon, but
still enters `TextTokenizer::encode` once for every prompt and once for every
completion. `TextTokenizer` exposes no ordered batch contract even though the
native Hugging Face backend already wraps Dynamo's `Encoder::encode_batch`.

## Code evidence

- Upstream `5566aae1e1` adds
  `Tokenizer::encode_lengths_batch(texts, chunk_size=4096)`, using native
  tiktoken or Hugging Face batch calls, and changes `ShareGPTLoader` to make
  bounded batch calls outside the per-entry validation loop.
- `rust/runtime/src/dataset/loader/public.rs::ShareGptComposer::compose`
  currently runs a Rayon `par_iter` over rows but calls
  `tokenizer.encode(&prompt)` and `tokenizer.encode(&completion)` for every
  adjacent pair.
- `rust/runtime/src/dataset/tokenizer.rs::TextTokenizer` has only scalar
  `encode`; no caller can express one ordered batch operation.
- `HuggingFaceTokenizer` wraps `DynamoHuggingFaceTokenizer`, whose imported
  `Encoder` trait provides `encode_batch(&[&str])` and preserves the scalar
  tokenizer's special-token options.
- ShareGPT composition needs complete prompt token IDs for segment interning,
  not only lengths. The native port therefore must batch full encodings and
  derive completion lengths from them, rather than copying the Python
  length-only API.

## Port decision

Add an ordered `TextTokenizer::encode_batch(&[&str])` seam whose default
implementation preserves compatibility for every existing tokenizer by
parallelizing scalar `encode` calls and collecting results in input order.
Override it for `HuggingFaceTokenizer` so one call reaches Dynamo's native
batch API. Keep the existing no-special-token contract and return full token
vectors because ShareGPT prompt tokens remain authoritative segment identity.

Change `ShareGptComposer` to extract all row-local prompt/completion pairs,
flatten their text in stable prompt-then-completion order, submit chunks of at
most 4096 texts to `encode_batch`, and reconstruct prepared rows in the same
order. Preserve every existing filter, session-id order, segment-parent chain,
token count, output-length override, and error boundary. Treat a batch backend
that returns a different number of encodings than inputs as a tokenizer error
instead of indexing or silently misaligning rows.

## Verification requirements

- A tokenizer unit test must prove the Hugging Face batch result is byte-for-byte
  identical to scalar encoding and retains input order.
- A ShareGPT composer regression must use more than 4096 texts and a tokenizer
  that refuses scalar encoding or oversized batches. Successful conversation
  output then proves the composer uses the batch seam and honors the bound
  without asserting only on a mock call counter.
- Existing ShareGPT pair/admission tests must remain green and continue proving
  all-or-nothing row rejection and multi-turn order.
- Focused runtime tests, formatting, Clippy for the runtime library, and a
  release CLI build must use `sccache` with `CARGO_TARGET_DIR` under `/mnt/4tb`.
- The final implementation must pass a full Graham code review before the
  campaign tracker can mark commit 008 complete.
