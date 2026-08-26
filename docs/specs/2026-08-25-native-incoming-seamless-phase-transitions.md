<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native incoming seamless phase transitions

## Purpose

Define the direction and lifecycle contract of the public per-phase `seamless`
flag. The flag is incoming: a phase carrying `seamless: true` may begin as soon
as its predecessor finishes issuance, without waiting for that predecessor's
in-flight requests to return.

## Built

Config v2 retains `seamless` on each authored `PhaseSpec`. The native timing
runtime owns handoff on the phase being left, so the engine performs one
directional lowering at the adapter boundary:

```text
authored phase i + 1: seamless = true
                    |
                    v
runtime phase i: outbound handoff = true
```

`phase_seamless_to_next(phases, i)` reads only `phases[i + 1]`. It returns
false for the final phase. The resulting outbound value is stored in the
internal `PhaseConfig` used by the shared scheduled/graph phase orchestrator.
This lowering is shared by unsharded scheduled execution, worker sharding,
offline Dynamo execution, and Graph-IR execution.

An outbound seamless handoff changes only the return barrier. The predecessor
still completes setup, issuance, sending-complete publication, ramp stop, and
all normal error checks. Its return wait continues on the current-thread
`LocalSet`; the orchestrator retains it as active while the successor starts.
The run-wide final barrier waits for every predecessor to drain and returns
phase statistics in authored order.

Detached return failures are terminal. The first background failure is retained
by `SeamlessFailureSignal`; an active successor is cancelled before another
phase can advance, and the run reports the predecessor's phase id and error.
Low-rate sidecars remain phase-owned. Their `finish` hooks execute from phase
finalization after return drain, so a profiling server is not stopped merely
because issuance hands off.

The incoming flag never grants a final phase an outbound handoff. Likewise, a
phase carrying `seamless: true` does not by itself alter the transition to a
non-seamless successor. These rules preserve the public authored meaning while
letting the timing runtime keep its explicit current-to-next ownership model.

### Verification contract

Authored-phase lowering tests prove positive, inverse, middle-of-workflow, and
final-phase direction. Runtime tests separately prove real-HTTP overlap,
non-seamless drain ordering, shared-capacity debt drainage, final barriers, and
detached-failure cancellation. Together those tests cover both sides of the
adapter rather than relying on direct internal `PhaseConfig` construction as
evidence for public behavior.

## Source anchors

- `rust/runtime/src/engine/execute/dataset_build.rs`
- `rust/runtime/src/engine/execute/compose_sidecars.rs`
- `rust/runtime/src/engine/execute/sharding.rs`
- `rust/runtime/src/engine/offline_execution.rs`
- `rust/runtime/src/engine/graph_phase_runtime.rs`
- `rust/runtime/src/timing/phase/orchestrator.rs`
- `rust/runtime/src/timing/phase/runner.rs`
- `rust/runtime/src/phase_runtime.rs`
- `rust/runtime/tests/timing_phase_orchestrator.rs`
- `rust/runtime/tests/phase_runtime_online.rs`
