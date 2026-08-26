<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sol Plan: Origin/Main 008 ShareGPT Batch Encoding Revalidation

## Scope

Revalidate the native Rust port of upstream `5566aae1e129f63c2d761d4c3fa5ee18de0ba9be`.
The exact upstream merge and its Rust implementation are already ancestors of
this branch. This flow confirms their behavior against current shared HEAD and
adds no replacement implementation unless the review identifies a concrete
contract gap.

## Contract

1. `TextTokenizer::encode_batch` returns one full token vector per input in
   input order, using the same no-special-token policy as scalar `encode`.
2. `HuggingFaceTokenizer` delegates a nonempty batch to Dynamo's native
   `Encoder::encode_batch`; the compatibility default remains ordered scalar
   encoding for every other tokenizer.
3. `ShareGptComposer` extracts adjacent human/assistant pairs in stable row and
   turn order, then submits prompt/completion text in bounded batches of at
   most 4096 entries.
4. Reconstruction consumes both encoded texts for every extracted pair even
   after that row becomes invalid. It preserves row rejection, session-id
   allocation, segment ancestry, prompt token identity, completion lengths,
   output-length override, and pre-interning failure on batch-cardinality
   mismatch.

## Verification sequence

1. Inspect `5566aae1e1` and compare its Python flatten/batch/reconstruct flow
   with `rust/runtime/src/dataset/tokenizer.rs` and
   `rust/runtime/src/dataset/loader/public.rs`.
2. Run focused tokenizer batch-fidelity and ShareGPT composer tests with
   `/usr/bin/sccache`, clang/lld, and this lane's target below `/mnt/4tb`.
3. Run the runtime library suite with `engine`, formatting, runtime Clippy, and
   a release CLI build. Classify any unrelated baseline failures with their
   exact test names rather than weakening the port gate.
4. Perform two Graham passes over the immutable native implementation range:
   one for correctness/error handling and one for allocations, concurrency,
   comments, naming, and test quality.
5. Send the exact range, upstream-merge ancestry, review result, and command
   evidence to the parent reviewer. Do not alter tracker closure until that
   independent review approves.
