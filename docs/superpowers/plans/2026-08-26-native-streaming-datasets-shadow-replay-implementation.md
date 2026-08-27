<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Streaming Datasets and Shadow Replay Implementation Program

This file is the dependency and worktree index for the implementation plan set. The executable TDD steps, exact Rust signatures, representative RED tests, commands, and commit boundaries live in the linked subsystem plans; workers execute those documents, not this index.

**Goal:** Implement the complete pure-Rust streaming-dataset and shadow-replay architecture approved at base spec commit `505efc06b0` and amended by content-reconstruction commit `3fea6f2fe0`.

**Spec:** `artifacts/streaming-design/streaming-dataset-shadow-replay-design.md`; immutable approval record: `artifacts/streaming-design/spec-review-record.md`.

## Executable Plan Set

1. [Foundation and runtime seams](2026-08-26-native-streaming-foundation-runtime-implementation.md)
   - Cargo features and exact dependencies
   - stable identity, budgets, blocking owner, traits, conformance harnesses
   - frozen registries and strict Protocol/Config-v2 resources
   - bounded terminal lane, phase/capture seam, UTC/event-time authority

2. [Checkpoint and results](2026-08-26-native-streaming-checkpoint-results-implementation.md)
   - typed cuts and stable participants
   - atomic memory/local backends, leases, GC, coordinator
   - logical membership, bounded indexes, epochs, provisional holes
   - partial/final/aborted compaction and delivery-mode crash semantics

3. [Sources and formats](2026-08-26-native-streaming-adapters-implementation.md)
   - local finite/follow and reference JSONL
   - pinned HF disk catalog and Baseten Parquet
   - frozen recorded-content synthesis profile and shared cache-free algorithm
   - strict deferred-content `streaming_dynamo_trace`
   - native S3 reconciliation and lossless/lossy policy

4. [Pipeline, sessions, and workload](2026-08-26-native-streaming-pipeline-sessions-implementation.md)
   - cross-chunk conversations and graph sessions
   - multiplexed action host, local placement, bounded pipeline
   - scheduled-request sink and executable `shadow_replay`
   - recorded-input and encrypted target-closed-loop policies

5. [Cellular execution](2026-08-26-native-streaming-cellular-implementation.md)
   - authenticated bounded protocol and transfer
   - prepare/release no-early-issue placement
   - ownership fencing and crash-safe migration
   - cell result partitions and controller-last global commit

6. [Product and verification](2026-08-26-native-streaming-product-verification-implementation.md)
   - public CLI/config/capabilities/partial results/docs
   - delivery and checkpoint fault conformance
   - real-binary dry-run, HTTP/gRPC, and cellular E2E
   - 8-GiB and accelerated 24-hour bounded-resource soaks
   - invariant-by-invariant completion ledger

## Dependency DAG

```text
foundation 0 -> 1A -> 1B -> checkpoint 5A -> 1C -> checkpoint 5B -> 1D -> 1E
                                                               |       |
                                                               |       +-> registry 2 -> protocol/config 3
                                                               +----------> terminal 4A -> capture 4B

checkpoint 5B -> local 5C -> leases 5D ------------------.
             |-> coordinator 5E --------------------------+-> result 6B -> compaction 6C1 -> delivery 6C2 -> report order 6D
             `-> result index 6A -------------------------'
result index 6A -> leases 5D

foundation 1D/1E + checkpoint 5A/5B
    |-> local A1     |-> JSONL A2
    |-> HF A3        |-> Baseten A4
    |-> synthesis A5P -> Dynamo A5
    `-> S3 A6
foundation 0 -> AWS construction A0 -> S3 A6

checkpoint 5A + foundation clock -> event time 7A
session P1 -> closure P1B -> action P2 -> pipeline P3
A5P + A5 + closure P1B -> deferred reconstruction P1C
local A1 + JSONL A2 + P3 + capture 4B + results 6D -> workload P4
P1C + A5 -> Dynamo-capable workload P4
P1/P2 -> graph P5
P1/P4 + crypto foundation -> sensitive state P6
P3/P4 + results 6B + event time 7A -> observability P7

registry 2 + local 5C -> built-in backends 5F1
AWS construction A0 + coordinator 5E + 5F1 -> object CAS 5F2
leases 5D + sensitive P6 + 5F2 -> object retention 5F3
workload P4 + A5P/A5/P1C + checkpoint/results 6D -> cellular C1 -> C2 -> C3 -> C4 -> C5 -> C6
all implementation plans -> product V1 -> V2 -> V3 -> V4 -> V5 -> V6
```

The individual plans contain the exact dependency clauses. If this overview and a subsystem plan ever disagree, stop and correct the overview before branching.

## Three-Worktree Execution Rules

- The current user branch is the integration branch. It remains in the original worktree.
- Task worktrees live at `/mnt/4tb/aiperf-streaming-worktrees/<task>` and all builds share `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target`.
- Each branch is cut from the current integrated `HEAD`, contains one task's focused commits and review fixes, and merges with `git merge --no-ff`. Never cherry-pick.
- Keep three worktrees active whenever the DAG exposes three file-disjoint tasks. If fewer are ready, unused slots run Graham/behavior reviews and gates; they do not branch from stale state.
- Every task branch owns the minimal nearest-parent module declaration needed for its GREEN build. The integration owner resolves declaration conflicts plus Cargo.lock, registry/protocol hotspots, cellular controller/cell hotspots, and `artifacts/streaming-design/implementation-progress.md` during `--no-ff` merges.
- A downstream worktree is created only after every declared prerequisite merge is present in the integration `HEAD`.

Initial worktree waves after the serialized 0/1A-1B → 5A → 1C → 5B → 1D-1E contract foundation:

| Wave | Worktree A | Worktree B | Worktree C |
|---|---|---|---|
| 1 | Registry 2 → Config 3 | Terminal 4A → Capture 4B | Local store 5C |
| 2 | Coordinator 5E | Result index 6A | Event time 7A |
| 3 | Leases/GC 5D | Local A1 | AWS construction A0 |
| 4 | Result epochs 6B → compaction 6C1 → delivery 6C2 → report order 6D | Conversation P1 → closure P1B → Action P2 | HF A3 |
| 5 | Pipeline P3 | Baseten A4 | JSONL A2 |
| 6 | Shared synthesis A5P → Dynamo A5 | Graph P5 | review/fix |
| 7 | Deferred reconstruction P1C → Workload P4 | Sensitive state P6 | S3 A6 |
| 8 | Observability P7 | Local/none backends 5F1 | review/fix |
| 9 | Object CAS 5F2 | Cellular C1 | review/fix |
| 10 | Object retention 5F3 | Cellular C2 | review/fix |
| 11 | Cellular C3 | review/fix | review/fix |

Cellular C2-C6 are serialized according to the cellular plan because they share controller/cell/protocol hotspots. Product V1, V2, V3, V4A, V4B, V5A, V5B, and V6 follow after Task 5F3 and all other implementation plans, with review work occupying spare slots.

## Merge and Review Gate

Before every `--no-ff` merge:

1. The task's exact RED test was observed failing for the intended missing behavior.
2. The task's exact GREEN command passes from its worktree using the shared target.
3. Graham review and independent behavior review approve the exact branch commit with zero blockers.
4. The root agent inspects the diff and reruns the named gate.
5. The branch merges into the original starting branch; the root updates and commits the progress ledger.

The implementation is complete only when Product Task V6 records fresh evidence for spec invariants 1-38, every public config/capability, all subsystem gates, E2E binary digest, soak observations, exact merge commits, and both final reviews on the same `HEAD`.
