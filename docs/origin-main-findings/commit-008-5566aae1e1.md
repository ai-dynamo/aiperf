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

## Port outcome

**Complete.** The isolated port branch merged upstream commit
`5566aae1e129f63c2d761d4c3fa5ee18de0ba9be` exactly through merge commit
`6521729344cad4b8791fe94c85a876a2ee52b8e0`, whose first parent is the commit
008 design record on top of integrated commit 007 and whose second parent is
the upstream commit itself.

The native implementation is split across two independently reviewed commits:

- `4e39be3aeec509bc43853aeeb75f29452d1d2c33` adds the ordered tokenizer batch
  seam and Hugging Face native override.
- `67e6f3988d7664900d4410910748cf5ff6352e20` adds bounded ShareGPT batch
  preparation, checked reconstruction, and behavior regressions.

Task 1's RED compile failed because `encode_batch` did not exist. Its GREEN
tokenizer module run passed 13 tests with one ignored. Task 2's RED run failed
both batch-only regressions with `scalar encoding is forbidden`. Its GREEN
public-loader module run passed all 24 tests, including the exact `[4096, 4]`
batch shape over 4,100 texts, alignment after an invalid two-pair row, semantic
and segment equality, and cardinality refusal before interning.

The plan's proposed `Some(9)` output-distribution assertion was corrected to
`Some(1)` against the binding design: ShareGPT retains its authored completion
length, while the configured distribution bypasses only minimum completion
admission. This preserves the pre-port native finalizer contract.

## Closure evidence

- `cargo fmt --all --check`: passed.
- `cargo clippy -p aiperf-runtime --all-targets --features engine`: passed with
  pre-existing warnings outside the port files.
- `cargo build -p aiperf-cli --release`: passed in the isolated target.
- Focused merged Python tokenizer and ShareGPT tests: 70 passed; warnings were
  limited to stale Docker-owned pytest temporary-directory cleanup.
- `git diff --check 6521729344..67e6f3988d`: passed, and the implementation
  range changes only `rust/runtime/src/dataset/tokenizer.rs` and
  `rust/runtime/src/dataset/loader/public.rs`.
- All Rust commands used `/usr/bin/sccache` and
  `/mnt/4tb/aiperf-origin-port-008-target`. Final cache evidence reported the
  local cache at `/mnt/4tb/.cache/sccache` with 17,454 Rust cache hits.

The full engine-enabled runtime suite compiled and ran 2,296 tests: 2,284
passed, 7 were ignored, and 5 unrelated baseline/churn failures remained. Two
failures reference missing recorded-agent fixture files; two are existing
engine transport/registry expectation mismatches; one expects report version
`0.0.0` while the package now reports `0.12.0`. None involves the two port
files, and focused port coverage remained green.

Both task reviews approved their ranges with no Critical or Important finding.
The final Graham review explicitly reports `Graham approval: APPROVED` with no
Critical or Important finding. Its two non-blocking Minors are a possible peak
configuration-memory reduction by consuming source rows during extraction and
the absence of a direct empty/all-malformed ShareGPT zero-batch regression;
the implementation's empty chunk iterator makes no backend call, and both
tokenizer batch seams already cover empty input.

## 2026-08-26 independent revalidation and approval

The campaign tracker remained pending after the original implementation landed,
so an isolated revalidation began from current shared HEAD rather than
duplicating the upstream merge or Rust implementation. The exact upstream merge
`6521729344cad4b8791fe94c85a876a2ee52b8e0` and both native commits are
ancestors of shared HEAD. Its second parent is the exact upstream commit
`5566aae1e129f63c2d761d4c3fa5ee18de0ba9be`.

Fresh verification used `/usr/bin/sccache`, clang/lld, and
`/mnt/4tb/aiperf-target-port008-rev2`:

- `cargo test -p aiperf-runtime sharegpt_ --lib`: 5 passed, including the
  4096/4 batch boundary, invalid-row alignment, scalar-forbidden seam, and
  cardinality refusal before interning.
- `cargo test -p aiperf-runtime
  text_tokenizer_default_batch_is_ordered_and_empty_safe --lib`: 1 passed.
- `cargo test -p aiperf-runtime
  hugging_face_batch_encoding_matches_scalar_order --lib`: 1 passed.
- `cargo fmt --all --check` and `git diff --check
  4e39be3aee^..67e6f3988d`: passed.

Independent Graham review covered
`4e39be3aeec509bc43853aeeb75f29452d1d2c33^..67e6f3988d7664900d4410910748cf5ff6352e20`
in two focused passes. It approved the range with no blocking, important, or
style findings. Current shared changes to these files are limited to unrelated
Hugging Face tokenizer metadata and a loader comment correction. Commit 008 is
therefore complete.
