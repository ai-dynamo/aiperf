<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native streaming implementation progress

- Integration branch: `ajc/native-rust-runtime-plugins`
- Normative specification: base `505efc06b0`, content-reconstruction amendment `3fea6f2fe0`
- Approved executable plan set: base `e16aa2c71f802a9ad17a241464374e4d7b5ba19b`, content amendment final `3621ec56e5`, checkpoint run-authority amendment `f4bd60e95b`
- Implementation start: `cd4a600e6c`
- Status: paused at `handoff-2026-08-27.md`; Task 1D-R ledger requires one review fix wave before checkpoint integration

## Durable milestones

| Milestone | Status | Evidence |
|---|---|---|
| Architecture specification | Complete | `spec-review-record.md` |
| Executable implementation plan | Complete | `implementation-plan-review-record.md` |
| Pre-0 feature-off compile prerequisite | Complete | Task `1320e84a18`; merge `7d54838a7b`; two reviews approved; focused test 1/1 passed |
| Pre-0b engine-only feature topology | Complete | Task `ceeccf9969`; merge `e8ffaf0cac`; two feature checks and 11 focused tests passed; two reviews approved |
| Task 0 — feature/dependency freeze | Complete | Task `8e3d5b57b3`; merge `dc81d2987a`; lightweight and S3 inventory tests 1/1 each; two reviews approved |
| Task 1A — identity and unit vocabulary | Complete | Task `283b6c872c`; merge `f8096d3041`; identity tests 10/10 and visibility doctest 1/1; two exact-head reviews approved |
| Task 1B — item/byte resource budgets | Complete | Task `f27c352251`; merge `2aa38c3acc`; budget tests 15/15 and identity compatibility 10/10; two exact-head reviews approved |
| Streaming Dynamo reconstruction amendment | Complete | Design `3fea6f2fe0`; executable plan final `3621ec56e5`; two reviews approved; durable ruling and review record committed |
| Task 5A — typed checkpoint cuts and authority | Complete | Task head `407116c2a2`; merge `20cf021e93`; checkpoint 13/13, authority 2/2, API doctests 8/8, budget/identity 25/25; two exact-head reviews approved |
| Task 1C — bounded blocking execution owner | Complete | Task head `f7af2069e8`; merge `b82ff70942`; blocking 11/11, budget 15/15, participants 13/13, authority 2/2, doctests 9/9; two exact-head reviews approved |
| Checkpoint logical-run authority plan correction | Complete | Plan head `f4bd60e95b`; merge `11f41a3b2a`; two exact-head reviews approved; mandatory Task 5A-R inserted before 5B |
| Task 5A-R — logical-run checkpoint authority retrofit | Complete | Task head `d020f3c616`; merge `f44863b7df`; checkpoint units 5/5, blocking units 2/2, targeted integrations 52/52, doctests 10/10; two exact-head reviews approved |
| Task 5B backend budget/atomicity contract correction | Complete | Plan head `329fc592b2`; merge `aaeb93c612`; two exact-head reviews approved; implementation may start |
| Reliability-first continuation amendment | Complete | Final head `68984da352`; merge `cf1ba627db`; spec and architecture review approved; inserts Task 1D-R after Task 1D and before every executable streaming path |
| Task 5B — atomic checkpoint backend and results contract | Complete | Task head `0be64a1386`; merge `f761d6cd82`; backend 27/27, compatibility 27/27, memory units, doctests, clippy, and scoped rustfmt passed in the root final batch; spec and Rust-quality review approved |
| Task 1D-R implementation-readiness correction | Complete | Final head `c3146fb476`; merge `5c06fd34d7`; spec and architecture review approved; closes versioned-head, pre-CAS receipt-root, move-only handoff, bounded legacy fixture, and stale-writer plan gaps |
| Task 1D — object-safe streaming contracts | Complete | Task head `84ad2da32e` rebased as `90a88e04fe` plus final EOF fix; merge `1264653069`; root batch passed contracts 22/22, streaming units 16/16, doctests 16/16, clippy, and scoped rustfmt; spec and Rust-quality review approved |
| Task 1D-R budget primitives | Complete | Task head `e2e4d77703`; merge `979f7d4ff0`; root batch passed 74 integration tests, 16 streaming units, 16 doctests, clippy, and scoped rustfmt; spec and Rust-quality review approved |
| Task 1D-R reliability policy | Complete | Task heads `66d4274946` and `472848142d`; merge `2757ca7e3e`; root batch passed contracts 22/22, policy 9/9, reliability units 8/8, streaming doctests 22/22, clippy, scoped rustfmt, and diff checks; spec and Rust-quality review approved |
| Task 1E — streaming adapter conformance factories | Complete | Branch `ajc/streaming-task-1e` at `ea24a94519`; spec APPROVE + quality APPROVE (rustfmt fix); 4 conformance + 91 lib + 22 contracts green; merged |
| Benchmark security-scope correction | Complete | Normative overlay `benchmark-security-scope-course-correction.md` approved; generation one retains BLAKE3 identity/integrity and existing provider/cellular boundaries, while encrypted resumable closed-loop state and direct base-streaming crypto dependencies are deferred |
| Base-streaming direct crypto removal | Awaiting final gate | Clean branch `ajc/streaming-security-scope` at `39555dad68`; focused RED/GREEN and feature inventories passed; targeted Clippy status and rustfmt must be completed before review/merge |
| Task 1D-R reliability ledger | Complete | Branch `ajc/streaming-task-1dr-ledger` at `6afd10587b`; two fix waves addressed all ten findings (B1-B5 code-quality + SC1-SC4 spec-compliance); dual APPROVE verdicts; merge `6afd10587b` into integration; 87 lib + 15 reliability + 22 contracts + 29 checkpoint_backend tests green |
| Task 1D-R versioned checkpoint authority | Complete | Branch `ajc/streaming-task-1dr-checkpoint` at `9d2006242c`; rebased onto post-ledger integration; dual APPROVE; 91 lib + 39 checkpoint_backend + 22 contracts + 15 participants green; merged |
| Task 7A — boundary UTC/monotonic clock anchor | Complete | Branch `ajc/streaming-task-7a` at `2d68918e43`; merge `16031101a1`; adds `UtcMonotonicAnchor`, `ClockAnchorError`, and `Clock::capture_utc_anchor` to `aiperf-core` with the `RealClock` bracketed implementation; `cargo test -p aiperf-core` green at 49 + 6 + 8 lib/integration tests plus 1 doctest; agent-file and docs sync tools exit 0 |
| Task 5C — crash-durable local checkpoint backend | Complete | Branch `ajc/native-rust-runtime-plugins`; `LocalCheckpointFilesystem` seam + `BlockingLocalFilesystem` production impl; `LocalCheckpointBackend`; 9 local checkpoint tests + 39 checkpoint_backend + 15 participants + 12 blocking + 22 contracts green |
| Task A0 — AWS S3 streaming client foundation | Complete | Committed directly on `ajc/native-rust-runtime-plugins` as `f1389a5cda` (`rust/runtime/src/streaming/aws.rs`, `streaming-s3` feature gate), `6f18d03d29` (rustfmt), and `80da509ec8` (resolved-endpoint observation through retained settings); `cargo test -p aiperf-runtime --features streaming,streaming-s3 --test streaming_aws_client` green 3/3 plus 3 in-module unit tests; no separate merge commit because no task branch was cut |
| Task A3 — HuggingFace credential streaming source | Complete | Branch `ajc/streaming-task-a3` at `b3586769c8`; merge `0ca4e4726a`; adds `streaming::hf_credentials` with `HfCredentialProvider`, refresh authority, and the pinned-host HTTP client factory; `cargo test -p aiperf-runtime --features streaming --test streaming_hf_credentials` green at 15/15, covering source-id stability across rotation, typed refresh exhaustion that never downgrades to anonymous, host-pinned bearer stamping, loopback never proxied, and clock-driven capped exponential backoff |
| Task A2 — HuggingFace rows streaming source | Complete | Branch `ajc/streaming-task-a2` at `54ad950f83`; merge `60635f8a20`; adds `streaming::sources::hf_rows` with the injected `HfPageTransport` page seam, revision pinned exactly once to an immutable 40-hex commit, arithmetic page inventory with infallible synchronous partition identity, bounded authorization through the shared credential authority, clock-driven capped exponential read backoff, partial-split refusal in finite mode with follow-mode parking, and checkpoint restore that resumes after the committed page and refuses drift; widens `StreamingSourcePrepareContext` with `run`, `stream_semantic_digest`, and `clock: Rc<dyn Clock>`; follow-up `72ba05f169` declares `streaming::sources` because the task branch shipped `hf_rows.rs` without a module declaration, so the file and its tests had never been compiled; `cargo build -p aiperf-runtime --features streaming` clean and `cargo test -p aiperf-runtime --features streaming --lib streaming::` green at 139/139 including all 15 hf_rows behaviour tests |
| Task A4 — Baseten Parquet streaming format | Complete | Branch `ajc/streaming-task-a4` at `f44aec5905`; merge `fce1d05383`; adds `streaming::formats::baseten` (Parquet decoder with `RetainedWindow`, canonical cursor, row projection, and the canonical replay-parameter document) gated on `streaming` + `parquet`, and extends `StreamingFormatPrepareContext` with `run`, `fragment_budget`, and `acquisition_budget`; the merge also performed the deferred formats-module wiring pass that `67e642c2ea` had unwired, so follow-up `d07fe380cb` completes the newly compiled `synthesis` test prepare context against the widened context; `cargo test -p aiperf-runtime --features streaming,parquet --lib` green at 2639 passed with all 136 `streaming::` tests including every baseten behaviour test; the 4 remaining failures are pre-existing `engine::*` cases (velo isolated-RSS child, transport-binding differential, registry workload resources, `global_push` pacing) untouched by this task's seven streaming-only files |
| Task A4S — synthesis streaming format | Complete | Branch `ajc/streaming-task-a4s` at `67e642c2ea`; merge `ea93339ae7`; adds `streaming::formats::synthesis`, a format adapter that wraps the existing corpus generators behind bounded shard manifests, exact-ordinal cursors, and checkpoint participation, and extends `rng::namespace` with its derivation namespace; the merge resolved the add/add conflict on `rust/runtime/src/streaming/formats.rs` so `baseten` (parquet-gated) and `synthesis` are both declared; `cargo test -p aiperf-runtime --features streaming --lib streaming::formats::synthesis` green at 12/12, covering canonical cursor round-trip, exact input-token length under prefix reuse, batch-shape independence of generated content, plan-drift resume as authority mismatch, and foreign-partition/short-cursor rejection |

Baseline note: the full feature-off runtime suite reached execution after the
repair and reported 1907 passing tests plus one pre-existing version-fixture
failure (`0.0.0` expected versus package version `0.12.0`). The streaming work
does not alter or mask that fixture.

Task 0's default test-discovery check correctly skipped its
`required-features = ["streaming"]` target, then encountered the unrelated
pre-existing `runtime/examples/rps_bench.rs` header-type compile error. The
S3-enabled focused test passed after clearing only the generated shared Cargo
cache and disabling dev debug/incremental artifacts to fit available storage.

Task 1A review hardened the durable boundary before merge: the transitional
zero-charge fragment lease is crate-private, logical-ID conflicts retain both
complete provenance receipts, and fixed golden vectors pin all six identity
domains plus length framing. Stable IDs remain independent of worker, cell,
route, discovery order, and global sequence.

Task 1B establishes cancellation-safe two-dimensional RAII capacity. Session
fragments always retain exactly one item permit, cross-chunk actions compose
distinct leases without minting, and the final continuation/capture/receipt
owner releases capacity. Checked construction charges retained `Vec`/`String`
capacity and spilled predecessor/lease-set allocations, preventing small-length
large-capacity payloads from bypassing the fixed-memory invariant.

Task 5A separates canonical checkpoint candidates from authoritative committed
generations. Candidates bind the exact stable participant inventory plus frozen
execution/result-plan digests, while only an opaque backend publication or
leased-read proof can promote one and mint participant release receipts.
Checkpoint bytes are compact-owned under inseparable move-only leases; prepared
and restored state cannot bypass digest, length, or budget verification.

Task 1C owns every accepted blocking job through a cancellation-safe reaper,
retains the complete caller-declared output reservation with arbitrary typed
outputs, and closes/cancels all owned work on final-owner drop. Checkpoint views
are quiescent single-flight operations; shutdown is terminal and restored or
committed decode horizons cannot move backward. A subsequent run-identity
authority amendment is an explicit prerequisite to Task 5B; it is not folded
silently into the completed 1C contract.

Task 5A-R makes logical replay-run identity part of every durable checkpoint
authority boundary without conflating it with process incarnation. Barriers,
prepared and committed participant state, generation candidates, publication
promotion, committed generations, and release receipts all reject cross-run
substitution before mutation. The canonical generation digest uses the frozen
v3 domain and binds the raw logical-run bytes; participant descriptors remain
content-addressed and deliberately run-neutral.

## Implementation rulings

- Generation one is a reliability-first benchmarking system, not a general
  encrypted state platform. Recorded-input replay remains restartable; encrypted
  target-closed-loop persistence, key resolution, XChaCha envelopes, and
  zeroization are deferred. BLAKE3 remains for deterministic identity,
  corruption/conflict detection, CAS lineage, and truthful restart. Cost if
  wrong: target-derived closed-loop state cannot resume across process loss
  until a separately justified non-default capability is designed.

- Parallel worktrees may edit concurrently, but Cargo invocations against the
  shared `/mnt/4tb/aiperf-streaming-target` are serialized and run only after
  the tested worktree is rebased onto the current integration head. Cargo's
  relative dep-info allowed a same-package, same-feature artifact from another
  worktree to be treated as fresh; the unchanged policy branch reproduced the
  false missing-symbol failure until only `aiperf-runtime` build artifacts were
  cleared and rebuilt. Cost if wrong: less compiler parallelism, while source
  implementation remains fully parallel; the alternative can produce invalid
  green or red evidence.

- Task 0 AWS compatibility: use exact `aws-config 1.8.14` and
  `aws-sdk-s3 1.123.0` with the approved feature lists. The planned 1.11.0 /
  1.144.0 pair requires Tokio 1.49 through Smithy 1.3, while pinned Dynamo and
  Velo require exact Tokio 1.48 for their unstable runtime-metrics contract.
  The selected pair is already present in the lock graph. Cost if wrong: the
  later S3 adapter may require a reviewed API adaptation or coordinated
  Dynamo/Velo/AWS dependency upgrade.
- Content reconstruction amendment: finite and streaming Dynamo share one
  frozen-profile, cache-free synthesis algorithm. Streaming retains typed hash
  descriptors until session/root closure, reserves decoded-content capacity,
  and uses only bounded local non-durable memoization. The controller performs
  generation-1 reconstruction; cells authenticate the bound profile digest
  before prepare. See `content-reconstruction-course-correction.md`.
- Checkpoint logical-run authority: durable generations, barriers, prepared and
  committed participant wrappers, publication proofs, receipts, and result
  reachability bind one `StreamRunIdentity(LogicalReplayRunId)`. Process
  incarnation remains separate. Task 5A-R retrofits this authority before 5B;
  V1 owns explicit fresh/resume resolution and bootstrap-before-issue product
  ordering. See `checkpoint-run-identity-course-correction.md`.
- Checkpoint backend atomicity and budgets: backend-owned transaction,
  prepared-index, storage, result-summary, and read budgets have stable typed
  errors. Result descriptors and payloads retain inseparable move-only leases;
  staging is cancellation-safe; every backend shares exact predecessor/next-
  epoch validation before effects; publication has no fallible post-fence path;
  repeated barriers return the already authoritative generation without
  restaging or consuming newly borrowed inputs. See
  `checkpoint-backend-budget-contract-correction.md`.

This file is updated and force-added after every reviewed task merge. Detailed
ephemeral RED/GREEN output, task briefs, review packages, and rulings live in
the plan-owned SDD workspace; exact task commits and merge commits are retained
here so progress survives context compaction and ignored-file cleanup.
