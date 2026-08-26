<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Scope and principles

## Problem statement

AIPerf needs a native Rust data plane that can consume a dataset whose contents
continue to arrive while a run is active. The first concrete use case is
time-shifted shadow replay: ingest request/response trace batches produced by
NVCF, normalize them through a format adapter, and replay requests against a
target frontend at their event-time spacing plus a configured lag.

This is not an S3 downloader attached to the existing finite Dynamo trace
compiler. It is architectural streaming-dataset support with source, format,
session program, action sink, ordering, replay, placement, checkpoint, and
durability responsibilities behind explicit seams.

## Architectural rules

1. Prefer traits and registered factories over source-, format-, transport-, or
   workload-specific branching.
2. Keep acquisition distinct from decoding, decoding distinct from event-time
   ordering, and ordering distinct from workload execution.
3. Let finite and streaming datasets share canonical records and consumers
   without forcing one lifecycle onto the other.
4. Preserve bounded memory and explicit backpressure at every asynchronous
   boundary. A perpetual source must not rely on an unbounded channel or an
   ever-growing replay history.
5. Route all replay scheduling through `Clock`; translate external wall-clock
   event time to a run-owned monotonic timeline exactly once.
6. Acquire external objects once into immutable owned snapshots before decoding.
   Provenance binds the consumed bytes, not a mutable origin name.
7. Separate event-time progress (watermarks), source progress (cursors), and
   execution progress (acknowledgements/checkpoints).
8. Keep credentials and source authority at the controller boundary. Workers and
   cells receive canonical, credential-free data-plane messages.
9. Make delivery, late-event, overload, gap, restart, and shutdown semantics
   authored and observable rather than implicit.
10. Preserve current finite dataset and Graph-IR behavior unless a streaming
    consumer explicitly selects the new lifecycle.
11. Treat source chunks as acquisition boundaries only. A stable logical
    session may span objects, shards, row groups, decoder batches, checkpoints,
    phases, and cellular transfer chunks without reset. A run-scoped session
    coordinator owns multi-turn, agentic, and graph continuity.
12. Publish results incrementally at checkpoint barriers through immutable,
    idempotent result segments and a bounded content-addressed index rooted by
    one atomic generation. Input progress, typed participant state, metric
    state, record results, and result publication share that committed epoch.

## First-use-case constraints

- Trace objects arrive periodically, approximately every five minutes.
- Replay preserves source event timing with a fixed delay from real time.
- Preparation time consumes part of the available lookahead; the configured lag
  must exceed publication delay plus acquisition/decoding allowance.
- The target may be a Dynamo frontend, but target selection remains an endpoint
  and transport concern.
- The source format may already be Dynamo-compatible, but streaming architecture
  cannot depend on that assumption.

## Out of scope for the first implementation slice

- Mutating an executing `GraphRecord` in place.
- Giving object-store credentials to cells or workers.
- Claiming exactly-once effects across a process crash without target-supported
  idempotency.
- Treating periodic absence of data as end-of-stream.
- Coupling streaming support to one cloud object-store SDK.
