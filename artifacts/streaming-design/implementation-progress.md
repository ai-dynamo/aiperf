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
| Task 0 — feature/dependency freeze | In progress | Pending task branch and reviews |

Baseline note: the full feature-off runtime suite reached execution after the
repair and reported 1907 passing tests plus one pre-existing version-fixture
failure (`0.0.0` expected versus package version `0.12.0`). The streaming work
does not alter or mask that fixture.

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
