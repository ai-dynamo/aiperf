<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native streaming implementation progress

- Integration branch: `ajc/native-rust-runtime-plugins`
- Normative specification: `505efc06b0`
- Approved executable plan set: `e16aa2c71f802a9ad17a241464374e4d7b5ba19b`
- Implementation start: `cd4a600e6c`
- Status: foundation contract wave started

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

## Implementation rulings

- Task 0 AWS compatibility: use exact `aws-config 1.8.14` and
  `aws-sdk-s3 1.123.0` with the approved feature lists. The planned 1.11.0 /
  1.144.0 pair requires Tokio 1.49 through Smithy 1.3, while pinned Dynamo and
  Velo require exact Tokio 1.48 for their unstable runtime-metrics contract.
  The selected pair is already present in the lock graph. Cost if wrong: the
  later S3 adapter may require a reviewed API adaptation or coordinated
  Dynamo/Velo/AWS dependency upgrade.

This file is updated and force-added after every reviewed task merge. Detailed
ephemeral RED/GREEN output, task briefs, review packages, and rulings live in
the plan-owned SDD workspace; exact task commits and merge commits are retained
here so progress survives context compaction and ignored-file cleanup.
