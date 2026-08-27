# Streaming Dynamo Content Reconstruction Course Correction

Date: 2026-08-26

Design amendment: `3fea6f2fe06cf901b070f068a96fa845d54f5cde`

This record translates the amended shared hash-to-content reconstruction
invariants into implementation-plan changes. It is intentionally narrower than
the normative design and exists to keep later task branches from implementing
the pre-amendment plan.

## Current-code findings

- `graph::recorded::content::RecordedContentSynthesizer` is the correct shared
  algorithm seam, but the current implementation combines pure synthesis with
  an unbounded `(scope, hash, block_size)` memoization map.
- `CorpusShared::new` reads `AIPERF_WEKA_FAST_CONTENT` during construction and
  records only a boolean. The selected algorithm is not yet a versioned frozen
  semantic profile.
- `TextTokenizer::name()` is diagnostic, not sufficient semantic identity.
  Prepared reconstruction must bind immutable tokenizer artifact/revision,
  vocabulary/decode behavior, and chat-template inputs.
- Finite Dynamo already owns strict arbitrary-precision hash parsing,
  capture-wide block-size validation, alignment, partial-block removal, and the
  legacy complete-trace fallback allocator. Those behaviors must be extracted,
  not independently reimplemented.
- Finite recorded-message reconstruction uses future-descendant lookahead. A
  streaming path cannot promise message-role parity while releasing an ancestor
  before sufficient tree/session closure.
- Existing streaming item/byte/action leases are reusable, but a small hash
  descriptor does not fund potentially large token and decoded-text buffers.

## Binding rulings

1. Add adapter prerequisite **A5P** to freeze
   `ContentSynthesisProfileV1`/`BoundContentSynthesisProfileV1`, tokenizer
   semantic receipts, explicit algorithm versions, and a cache-free pure
   synthesis seam shared by finite and streaming Dynamo.
2. Task **A5** emits typed deferred replay descriptors. It validates and
   checkpoints `SynthesisAuthority::{Unbound, Bound}`, requires replay metadata
   for executable generation-1 records, and never uses the finite virtual
   fallback allocator.
3. Add session task **P1C** after A5 and P1B. It waits for the closure evidence
   required by finite message reconstruction, checkpoints producer root/tail
   scope, reserves content capacity, and emits ordinary canonical actions.
4. P3 reserves transient token/text/action capacity before allocation. Optional
   memoization is worker/cell-local, byte-bounded, evicting, non-waiting, and
   excluded from checkpoints; capacity zero is a valid generation-1 mode.
5. P4 consumes only reconstructed canonical actions. It never interprets Dynamo
   hashes or owns another synthesis algorithm/cache.
6. Existing stable checkpoint participant IDs remain unchanged. The format
   participant stores cursor plus synthesis authority; the session participant
   stores proven root/tail scope and bound program digest. Cache entries are not
   durable state.
7. The frozen execution/checkpoint identity binds authored and bound synthesis
   semantics. Resume refuses participant inventory, execution plan, result plan,
   tokenizer, corpus, algorithm, or bound block-size mismatch before state use.
8. Cellular generation 1 reconstructs on the controller. Cells authenticate a
   run-scoped synthesis-profile binding and refuse mismatched prepare before any
   allocation, release, or endpoint issue.
9. Stable decode codes distinguish missing replay metadata, invalid geometry,
   synthesis authority mismatch, and unavailable/unsupported immutable profile.

## Required evidence

- Finite/streaming equality for token IDs, message roles, decoded text, and
  prefix relationships with valid replay metadata.
- Repeated/shared hashes; zero, tiny, full-block, and full-plus-partial inputs.
- Missing replay, empty-hash/nonzero-input, block-size drift, tokenizer/profile
  mismatch, and future-descendant lookahead refusal/closure behavior.
- Checkpoint before binding, after binding, and after producer-root proof;
  cache-free rebuild must be byte-identical.
- Environment changes after profile preparation have no effect.
- Cellular bind ordering/mismatch with zero issued requests.
- A high-cardinality unique-hash resource slope with cache disabled and forced
  eviction, including transient descriptor-to-decoded-content amplification.

## Non-changes

- Do not add a Dynamo-specific source; S3/local/HF remain independent sources.
- Do not add new dynamic checkpoint participant IDs.
- Do not change decoder record identity when synthesis authority binds.
- Do not construct a complete `GraphInputBundle` in the streaming path.
- Do not change finite Dynamo fallback behavior.
