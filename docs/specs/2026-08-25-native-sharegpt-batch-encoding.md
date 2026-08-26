<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native ShareGPT Batch Encoding

## Status

Approved and independently revalidated native Rust port of origin/main commit
`5566aae1e129f63c2d761d4c3fa5ee18de0ba9be`.

## Problem

The public ShareGPT corpus contains roughly 90,000 rows. Its upstream Python
loader previously called the tokenizer independently for every prompt and
completion while validating rows. That configuration work exceeded AIPerf's
300-second default timeout. Upstream fixes the bottleneck by flattening the
conversation pairs and using bounded tokenizer batch calls.

Native profiles do not run the Python ShareGPT loader. The native
`ShareGptComposer` performs the same scalar prompt/completion tokenization in
Rust. Rayon spreads rows across workers, but it cannot use the tokenizer
backend's batch interface because `TextTokenizer` exposes only scalar
`encode`. Merging the upstream Python change therefore leaves the native path
without the intended batching contract.

## Goals

1. Give native composers an ordered batch-encoding seam without breaking any
   existing tokenizer implementation.
2. Make ShareGPT composition enter that seam with no more than 4096 texts per
   call.
3. Use the Hugging Face backend's native batch API rather than expanding the
   batch back into scalar calls.
4. Preserve all existing ShareGPT conversations, token IDs, admission rules,
   ordering, segment ancestry, session IDs, and error behavior.
5. Fail safely if a tokenizer implementation violates the batch result-count
   contract.

## Non-goals

- This port does not change the Python batching implementation brought in by
  the upstream merge.
- It does not add a user-facing batch-size flag. The upstream value 4096 is a
  private composition bound.
- It does not replace Rayon globally or redesign tokenization for other
  composers.
- It does not discard prompt token vectors after counting them; native segment
  identity depends on those exact vectors.
- It does not alter tokenizer special-token policy, server-tokenizer wire
  formats, or public dataset schemas.

## Architecture

### Ordered tokenizer batch seam

Extend `TextTokenizer` with:

```rust
fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>>;
```

The contract is strict:

- Result element `i` is the token vector for input `texts[i]`.
- Success returns exactly `texts.len()` vectors.
- Each element is semantically identical to `encode(texts[i])`, including the
  existing no-automatic-special-token policy.
- An empty input returns an empty vector without backend work.

The default implementation maps scalar `encode` over an indexed Rayon
parallel iterator and collects in input order. This keeps all existing local,
server, and test tokenizers source-compatible while retaining the native
composer's current parallelism.

`HuggingFaceTokenizer` overrides the default and delegates one call to
`DynamoHuggingFaceTokenizer::encode_batch`. It converts each returned
`Encoding::token_ids()` slice into the owned `Vec<u32>` required by the
composition layer. Any backend error becomes `DatasetError::Tokenizer`, as in
scalar `encode`.

### ShareGPT preparation

`ShareGptComposer::compose` remains a two-stage operation:

1. Prepare and validate row-local data without mutating the segment pool or
   session-id generator.
2. Sequentially intern accepted prompts and allocate session IDs.

The preparation stage changes as follows:

1. Extract every row's adjacent human/assistant pairs into
   `Vec<Option<Vec<(String, String)>>>`, preserving row and pair order.
2. Flatten borrowed prompt/completion strings in
   `[prompt0, completion0, prompt1, completion1, ...]` order.
3. Call `encode_batch` for consecutive chunks of at most
   `SHAREGPT_TOKENIZER_BATCH_SIZE = 4096` texts and append each ordered result.
4. Verify the total number of returned token vectors equals the number of
   submitted texts. A mismatch returns a `DatasetError::Tokenizer` carrying
   expected and actual counts.
5. Reconstruct each row by consuming exactly two token vectors per pair. Apply
   `TokenBudget::pair_ok` exactly as before. Reject the whole row when any pair
   fails, while continuing to consume its already-produced encodings so later
   rows remain aligned.
6. Store each accepted prompt with its exact tokens and each completion's
   checked `u32` token length in `PreparedShareGptPair`.

The sequential interning stage is unchanged. In particular, rejected and
malformed rows do not consume session IDs, every accepted turn keeps the prior
prompt handle as its segment parent, and output-length distributions continue
to bypass only the minimum completion length.

## Error handling

- Scalar and native batch tokenizer failures retain the
  `DatasetError::Tokenizer` boundary.
- A batch cardinality mismatch is reported before any token vector is assigned
  to a ShareGPT row. The message identifies both expected and actual counts.
- Existing invalid-row behavior remains filtering, not a fatal error.
- Existing completion lengths that cannot fit `u32` remain fatal validation
  errors with the current message.
- There is no indexing into unchecked batch output and no partial mutation of
  `SegmentPool` before all tokenization succeeds.

## Performance and memory

The maximum backend batch input is 4096 text references. Output token vectors
remain resident until the existing sequential segment-interning pass, matching
the current native prepared-row ownership requirement. The port removes
per-text Hugging Face API crossings and uses its internal batch parallelism,
while the default batch method retains parallel scalar behavior for tokenizers
without a native batch primitive.

No batch work runs on the request or token dispatch hot path; dataset
composition happens before execution.

## Test design

### Batch seam fidelity

Construct the existing in-repo WordLevel Hugging Face tokenizer fixture. Encode
a mixed ordered input with scalar calls and `encode_batch`, then assert exact
vector equality. This fails if the backend reverses output, adds special
tokens, drops an item, or does not implement the native batch override
correctly.

### Bounded ShareGPT batching

Add a test-only `BatchOnlyTokenizer` that:

- returns an error from scalar `encode`;
- returns deterministic tokens from `encode_batch`;
- returns an error when a batch exceeds 4096 texts.

Compose 2,049 valid single-pair rows, producing 4,098 flattened texts and
therefore requiring at least two bounded calls. Assert 2,049 output
conversations and representative prompt/output token counts. The observable
conversation result is the assertion; the tokenizer double's refusals ensure
the old scalar path and an unbounded one-shot batch cannot satisfy it.

### Existing semantic coverage

Retain and run existing tests for adjacent pair collection, minimum/maximum
admission, total-token limits, output-length overrides, row rejection, and
turn ordering. No test may depend only on implementation call counters.

## Verification gate

Before closure:

1. Run focused tokenizer and public-loader unit tests.
2. Run all `aiperf-runtime` library tests with the `engine` feature.
3. Run `cargo fmt --check` and runtime Clippy.
4. Build `aiperf-cli --release` with the default feature set.
5. Use `/usr/bin/sccache` and a commit-specific `CARGO_TARGET_DIR` below
   `/mnt/4tb` for every Rust build/test invocation.
6. Complete independent task review and full Graham review, with no unresolved
   Critical or Important findings.

## Independent closure evidence

On 2026-08-26, the current-shared-HEAD revalidation confirmed that actual merge
`6521729344cad4b8791fe94c85a876a2ee52b8e0` has the exact upstream commit as
its second parent and that native range
`4e39be3aeec509bc43853aeeb75f29452d1d2c33^..67e6f3988d7664900d4410910748cf5ff6352e20`
remains behaviorally intact. Fresh focused ShareGPT composition coverage (5
tests) and tokenizer batch-fidelity coverage (2 tests) passed with sccache and
clang/lld using an isolated `/mnt/4tb` target. Independent Graham review
approved the full native range with no findings.
