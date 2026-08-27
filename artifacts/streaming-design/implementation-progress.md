<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native streaming implementation progress

- Integration branch: `ajc/native-rust-runtime-plugins`
- Normative specification: base `505efc06b0`, content-reconstruction amendment `3fea6f2fe0`
- Approved executable plan set: base `e16aa2c71f802a9ad17a241464374e4d7b5ba19b`, content amendment final `3621ec56e5`, checkpoint run-authority amendment `f4bd60e95b`
- Implementation start: `cd4a600e6c`
- Status: Task 5B checkpoint backend integrated; Task 1D contracts in progress

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
