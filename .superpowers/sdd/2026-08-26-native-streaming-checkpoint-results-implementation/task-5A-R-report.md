# Task 5A-R implementation report

## Provenance

- Branch: `ajc/streaming-task-5a-r`
- Worktree: `/home/anthony/nvidia/projects/aiperf/ajc/rust/.worktrees/streaming/task-5a-r`
- Base commit: `9f2cd1f4bdddc3eb9be044c1a052991f9b2c7ea1`
- Approved plan head: `f4bd60e95b`
- Final commit: this report is committed atomically with the implementation; the exact immutable commit ID is recorded in the parent-agent handoff because a Git commit cannot contain its own hash.

## Owned files

- `rust/runtime/src/streaming/checkpoint.rs`
- `rust/runtime/src/streaming/blocking.rs`
- `rust/runtime/tests/support/streaming_checkpoint.rs`
- `rust/runtime/tests/streaming_checkpoint_participants.rs`
- `rust/runtime/tests/streaming_blocking.rs`
- This report and `task-5A-R-review-package.md` (ignored SDD package, force-added)

No dependency, manifest, Task 1C, Task 5B, Config-v2, protocol, or unrelated documentation files were changed.

## RED evidence

Tests were authored before production changes.

- Focused integration RED command:
  `cargo test -p aiperf-runtime --features streaming --test streaming_checkpoint_participants --test streaming_blocking`
  exited 101. Compilation failed on the intentionally absent `StreamRunIdentity`, `CheckpointBarrier::run`, run-bound state/candidate/verification APIs, and run-aware participant fixture constructors.
- Full streaming library RED command:
  `cargo test -p aiperf-runtime --features streaming --lib`
  exited 101 with 21 compile errors for the intentionally absent private proof, promotion, and receipt run-binding APIs used by the new crate-private regression tests.

All Cargo commands used:

```text
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target
TMPDIR=/mnt/4tb/aiperf-streaming-tmp
CARGO_PROFILE_DEV_DEBUG=0
CARGO_INCREMENTAL=0
RUSTC_WRAPPER=''
```

## Implementation and invariants

- Added checked `StreamRunIdentity`, which wraps only `LogicalReplayRunId`. The private field and compile-fail coverage prevent construction from `RunIncarnationId`.
- Kept `ParticipantStateDescriptor` deliberately run-free, preserving its strict public DTO literal and digest semantics.
- Bound the logical run privately into prepared/committed state wrappers, candidates, committed generations, and participant receipts.
- Preserved run identity across staging: `PreparedParticipantState::into_parts` returns `(StreamRunIdentity, ParticipantStateDescriptor, BudgetedCheckpointBytes)`.
- Added the run to barriers. Counting and blocking participants reject a foreign-run barrier before fencing, shutdown, budget, or checkpoint-state mutation.
- `CheckpointGenerationCandidate::verify_against` checks the expected run first; `promote` also requires the expected run and cannot promote a proof across logical runs.
- Bumped canonical generation hashing to `aiperf.streaming.committed-checkpoint-generation.v3` and length-framed raw `LogicalReplayRunId::as_bytes()` as a distinct field immediately after the domain.
- Added a fixed v3 golden digest: `519bf192518f43e9d4accd6bd8ed38e885a1dce06d8d35579bf5f99b794d10f1`.
- Candidate deserialization includes the private run and re-verifies canonical content, so serialized run tampering is rejected.
- Receipts derive their run from the authoritative committed generation. Receipt equality and participant idempotency include the run; a greater-epoch foreign receipt is rejected before mutation.
- Blocking executor initialization and restore are run-bound. Its worker-local `!Send` model, budget ownership, stable participant IDs, exact payload verification, and represented-horizon domain separation remain intact.

## GREEN evidence

Observed command results on the final source tree:

- Focused integrations: `streaming_blocking` 12/12 passed; `streaming_checkpoint_participants` 15/15 passed.
- Private checkpoint unit filter: 5/5 passed.
- Private blocking unit filter: 2/2 passed.
- Compatibility: `streaming_budget` 15/15 passed; `streaming_identity` 10/10 passed.
- Rust doctests: 10/10 passed, including the `RunIncarnationId` compile-fail and existing privacy compile-fails.
- Targeted Clippy:
  `cargo clippy -p aiperf-runtime --features streaming --lib --test streaming_checkpoint_participants --test streaming_blocking`
  exited 0. It emitted existing runtime warnings but no diagnostic in the owned diff.
- Exact-file formatting:
  `rustfmt --edition 2024 --check` over the five owned Rust files exited 0.
- `git diff --check` exited 0.

The broad streaming library command executed 2,529 tests: 2,514 passed, 8 failed, and 7 were ignored. Every new checkpoint/blocking test passed. The eight failures were outside the owned modules: duplicate cellular shard collision, cellular multi-phase ordinal range, two missing recorded-agent fixture cases, missing HTTP transport registration, unknown `acme_remote` workload, and the metrics report version expectation. One duplicated cellular failure is reported twice by the harness. These failures were not reproduced on the base commit in this task, so this report records them as out-of-scope failures rather than claiming they are pre-existing.

Broad workspace `cargo fmt --check` also reports formatting differences in unrelated e2e recorded-agent, graph execution/executor/runtime, phase-runtime, and endpoint-binding files. Broad all-test Clippy exits nonzero after unrelated test warnings; the exact owned target is green.

## Residual risks and deferred authority

- Task 5A-R does not implement Task 5B persistence or backend transaction logic.
- Fresh/resume logical-run discovery, Config-v2/protocol wiring, `RunIncarnationId`, and durable writer-lease allocation remain assigned to later planned tasks.
- The whole streaming library suite is not globally green for the unrelated failures itemized above. Targeted tests, compatibility tests, doctests, formatting, and targeted Clippy are green.
- No cross-domain horizon conversion was introduced; run identity is orthogonal to represented-cut validation.
