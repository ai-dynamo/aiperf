# Task 1 scoped independent re-review

## Verdict

APPROVED.

Review target: Task 1 commit `7efbc6b4cb` plus review-fix commit
`579b2e7ce1`, against parent `945ed781b5`. Task 2 was not reviewed or started.

## Rejected-finding verification

### Important — `draft_acceptance_rate` range

Resolved. `parse_vllm_spec_decode_stats` first rejects non-finite values and
then applies the inclusive `0.0..=1.0` range. The regression covers values
below zero and above one. The implementation preserves valid boundary values
and does not re-derive floating relationships from integer counts.

### Minor — real finish-only typed dispatch path

Resolved. `finish_only_spec_decode_chunk_reaches_the_terminal_record` launches
a real local SSE endpoint and dispatches through
`TransportSink::dispatch_prepared_endpoint_collect_record_with_hooks`. Its
response sequence contains:

1. one ordinary content chunk;
2. one finish-reason chunk with stats but no content or usage;
3. one later usage-only chunk; and
4. `[DONE]`.

Because the request is a final streamed `chat` turn, the stats-bearing chunk
necessarily traverses the typed fast path. The test proves a completed terminal
dispatch and a finalized `RecordIngest` with eight steps, 18 accepted drafts,
32 proposed drafts, and the later reconciled completion-token count of two.

## Remaining Task 1 scan

No blockers found. The combined diff:

- captures only built-in `chat` and `completions` responses and suppresses any
  choice cardinality other than one;
- keeps the last non-empty stats object and avoids an unconditional generic
  JSON parse on ordinary streamed chat chunks;
- validates finite floats, the fractional rate range, checked histogram sums
  and weighted sums, accepted-versus-drafted counts, and optional per-step
  lengths/sums/pairs;
- normalizes after usage reduction, emits one owned observer event before
  terminal status, forwards it in deterministic tee order, and moves it into
  the tail-appended serde-compatible `RecordIngest` field;
- adds no mutex, channel, task, dependency, clock call, or shared cross-thread
  state to the request path; and
- uses structured `tracing::warn!` degradation for captured malformed objects.

The existing SGLang telemetry path and unrelated runtime behavior are
unchanged. Production additions introduce no `unwrap()` or `expect()`.

## Fresh verification

Run from `rust/` with the shared environment, `sccache`, and the isolated Task
13 target:

```text
RUSTC_WRAPPER=sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port-013 \
  cargo test -p aiperf-runtime --features engine --lib spec_decode -- --nocapture
```

Result: 8 passed, 0 failed, 2300 filtered out. This includes
`vllm_draft_acceptance_rate_must_be_a_fraction` and
`finish_only_spec_decode_chunk_reaches_the_terminal_record`. Four pre-existing
unrelated warnings remained. `cargo fmt --all --check` and `git diff --check`
also completed without findings.
