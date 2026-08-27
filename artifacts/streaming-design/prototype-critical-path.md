<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Streaming prototype critical path

## Outcome

The first real native vertical should be a **new streaming local-follow JSONL
conversation-fragment workload**. It is not finite `agent_recording` or
`dynamo_trace` replay repackaged as streaming. The narrow product path is:

```text
local publish-by-rename JSONL chunks
  -> cross-chunk conversation coordinator
  -> event-time anchor plus recorded offset pacing
  -> existing native HTTP request execution (Dynamo affinity headers)
  -> local checkpoint generation and result epoch
  -> process restart and deterministic final compaction
```

The existing native transport seam provides an executor factory and request
materializer without making the workload transport-specific
(`rust/runtime/src/engine/registry.rs:235-339`). HTTP already emits derived
`X-Dynamo-Session-ID` and parent-session headers from correlation identity
(`rust/runtime/src/transport/http/transport/http_transport.rs:127-157`). The
prototype therefore uses a Dynamo-compatible frontend target without requiring
the separate streaming Dynamo trace decoder.

This document records the shortest implementable slice, not a revision to the
approved plans. If the final reliability-continuation amendment changes a
dependency, owner, or acceptance gate, that amendment takes precedence.

## Required slice

After Task 5B and the reliability contract gate are integrated, the prototype
requires the following approved task outcomes:

- Contract/configuration: 1D, 1D-R, 1E, 2, and 3.
- Existing-runtime construction: 4A, 4B, and 7A.
- Local durable checkpoint/result path: 5C, 5D, 5E, 5F1, 6A, 6B, 6C1, 6C2,
  and 6D.
- Source and format: A1 local finite/follow source and A2 bounded reference
  JSONL format.
- Causality/execution: P1, P1B, P2, P3, and P4.
- A focused real-binary local-follow/restart proof based on V4A's scenario,
  rather than the full production V4A matrix.

These are required because the current implementation has only typed cuts and
run-bound candidate authority. Backend publication remains a 5B responsibility
(`rust/runtime/src/streaming/checkpoint.rs:451-475`, `:711-715`), and current
metrics still retain a run-wide accumulator/record collection
(`rust/runtime/src/metrics.rs:219-248`). The result tasks establish epoch
rotation, committed partial roots, final compaction, and report-persistence
ordering. Existing report persistence calls a prepared commit only after the
authoritative report write (`rust/runtime/src/engine/coordinator.rs:483-539`).

Task 4A is also a real prerequisite: `ScheduledRuntime` currently owns
per-session maps and a vector of per-record processor tasks
(`rust/runtime/src/scheduled.rs:487-517`). Task 7A provides the UTC/monotonic
event-time bridge; deterministic virtual pacing rests on the `Clock` sleep
contract (`rust/runtime/src/clock/runtime_clock.rs:11-44`) and `SimClock`
implementation (`rust/runtime/src/clock/sim_clock.rs:153-183`).

## Deferred work

The first vertical intentionally defers:

- AWS/S3 and object-store work: A0, A6, 5F2, and 5F3.
- HF and Baseten acquisition/decoding: A3 and A4.
- Streaming `dynamo.request.trace.v1` ingestion and deferred recorded-content
  reconstruction: A5P, A5, and P1C. Existing finite Dynamo loading compiles a
  complete capture into Graph-IR (`rust/runtime/src/graph/recorded/dynamo/mod.rs:41-100`);
  it cannot substitute for a streaming decoder.
- Graph/agent/closed-loop/sensitive policies P5-P7, cellular C1-C6, and the
  full public product/conformance/soak sequence V1-V6.

If the prototype must ingest Dynamo trace objects, rather than merely issue to
a Dynamo-compatible frontend, A5P -> A5 and P1C become an additional serial
branch before the Dynamo composition is enabled in P4.

## Dependency and three-agent waves

The reliability amendment establishes the contract order:

```text
0 -> 1A -> 1B -> 5A -> 1C -> 5A-R -> 5B -> 1D -> 1D-R -> 1E
```

For the proposed post-5B/reliability gate, use this ownership-aware schedule.
Each downstream branch starts only from the integrated prerequisite head.

| Wave | Agent A: configuration/execution | Agent B: durable results | Agent C: coordinator/session |
|---|---|---|---|
| 0 | 1E -> 2 -> 3 | 5C | 5E |
| 1 | 4A -> 4B | 6A -> 5D -> 6B | 7A -> P1 -> P1B |
| 2 | A1 -> A2, then 5F1 after 2 and 5C | 6C1 -> 6C2 -> 6D | P2 -> P3 |
| 3 | integration/review fixes | result/restart review fixes | P4 |
| 4 | focused real-binary E2E | review | review |

Keep 5C/5D/6A/6B/6C*/6D with the durability owner once their files converge;
keep 4A/4B with one owner because both modify scheduled/phase construction;
and keep P1/P1B/P2/P3 with one owner because they share causal state. The
integration owner resolves `streaming.rs` declaration conflicts and registry/
protocol hotspots.

## Prototype acceptance proof

Run a real `aiperf` binary against an in-process Rust mock HTTP target whose
request capture records headers. The fixture should:

1. Publish `000.jsonl` by rename with turn 0 for session S and a UTC event time
   plus recorded offset.
2. Wait for committed generation 1 and assert a non-final partial root contains
   only turn 0 and its epoch metrics.
3. Terminate the process, publish `001.jsonl` by rename with turn 1 for the
   same session, and resume from the exact local checkpoint root.
4. Assert the restored source cursor does not duplicate `000`, the restored
   conversation request includes both turns, and the captured requests share
   `X-Dynamo-Session-ID` (plus the parent header when authored).
5. Assert target times equal the frozen UTC anchor plus the two record offsets
   under `SimClock`, then repeat against the HTTP target with a bounded real
   clock tolerance.
6. Seal and assert the second generation is final; its compacted logical-record
   multiset and metric store equal a one-shot sealed reference.

The crash point is deliberately after a committed generation. A target-accepted
but not committed request remains at-least-once unless target idempotency is
explicitly available; the prototype must not claim exactly-once delivery.

## Risks and gates

- 1D-R changes generation/cut authority. Do not revive branches created before
  its integrated head.
- P1B remains necessary even in this small slice: partition EOF is not proof of
  session closure, and missing predecessors need an authored bounded policy.
- 5C alone is insufficient for partial/final result semantics; reader leases,
  membership, epoch rotation, final compaction, delivery policy, and report
  ordering live in 5D/6A-6D.
- Prove time arithmetic with `SimClock` before toleranced wall-clock HTTP E2E.
- CLI/operator pagination of partial generations is V1 scope. This prototype
  may inspect the local checkpoint reader/manifest from its test harness, but
  must not represent that as a completed public results surface.

## Estimate

The completed serialized foundation milestones are the calibration, not an
optimistic LOC extrapolation: Task 0 (179 changed lines), 1A (910), 1B (1,039),
5A (741 net), 1C, and 5A-R (728 net) ran from the Task-0 implementation commit
at 18:00 to the 5A-R merge at 23:49 on 2026-08-26: 5 hours 49 minutes, including
review/integration gates. That is roughly 58 minutes per task-sized serialized
milestone.

The required prototype has about 24 task leaves and a 12-14-unit dependency
spine. With three agents and the waves above, the evidence supports **12-16
active elapsed hours**, planned as **two working days** after the Task-5B/
reliability gate. Requiring streaming Dynamo trace input adds the A5P -> A5 ->
P1C serial chain and should add roughly 3-5 more elapsed hours.
