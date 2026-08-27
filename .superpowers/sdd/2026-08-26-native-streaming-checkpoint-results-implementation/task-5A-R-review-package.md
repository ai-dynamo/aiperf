# Task 5A-R review package

## Review focus

This patch supplies the logical-run binding required between landed Tasks 5A/1C and Task 5B. Review only the five owned Rust files listed in `task-5A-R-report.md`.

## Required API checks

1. `StreamRunIdentity` can be constructed only from a validated `LogicalReplayRunId`; it exposes no incarnation-based constructor.
2. `ParticipantStateDescriptor` contains no run field and retains its existing digest and strict DTO shape.
3. Prepared-state extraction cannot erase the run: `into_parts` returns the run with descriptor and budgeted payload.
4. Barrier, prepared/committed wrappers, candidate, generation, and receipt all carry the same logical run through private or authoritative construction.
5. Candidate verification accepts an explicit expected run and checks it before participant plans or digests; promotion consumes a proof only under that expected run.
6. Canonical v3 hashing frames the raw logical replay run as a separate field and is pinned by a golden test.
7. Counting and blocking participants reject foreign-run barriers and greater-epoch receipts before any state mutation.

## Security/correctness regressions

- Same checkpoint content under distinct logical runs has distinct generation digests.
- A serialized candidate whose run is changed cannot deserialize as valid.
- A valid proof for a foreign run cannot be promoted against the local expected run.
- An otherwise valid greater-epoch foreign receipt cannot advance participant state or idempotency.
- A foreign barrier cannot fence a blocking owner or consume checkpoint budget.

## Compatibility boundaries

- No new crates or features.
- No `RunIncarnationId` in streaming checkpoint authority.
- No run field in `ParticipantStateDescriptor`.
- No conversion between result and record horizons.
- No change to `!Send` worker-local participant futures.
- No Task 5B storage/backend implementation.

## Verification summary

- Focused integrations: 27/27 passed.
- Private run-binding unit tests: 7/7 passed.
- Budget and identity compatibility: 25/25 passed.
- Doctests: 10/10 passed.
- Targeted Clippy, exact-file rustfmt, and `git diff --check`: exit 0.
- Broad streaming library: 2,514 passed, 8 unrelated/out-of-scope failures, 7 ignored; see the implementation report for exact categories and attribution limits.
