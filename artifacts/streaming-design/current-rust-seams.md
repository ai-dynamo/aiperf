<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Current Rust seams and required evolution

## Purpose

This inventory grounds the streaming-dataset design in executable Rust code.
It classifies each relevant seam as:

- **Reuse**: the contract already has the correct ownership and lifecycle.
- **Adapt**: keep the contract and add a narrow adapter around it.
- **Finite-only**: preserve it for resident datasets; do not overload it with
  streaming state.
- **New seam**: add a registered or injected trait because no current contract
  owns that responsibility.

The central finding is that AIPerf already has strong execution, transport,
clock, phase, and extension boundaries. Its missing abstraction is the run-time
dataset plane between immutable source discovery and `TurnToSend`. Current
dataset and graph adapters are preparation-time compilers that return complete
resident values.

## Seam inventory

| Current seam | Current contract and lifetime | Disposition | Streaming role |
|---|---|---|---|
| `AIPerfExtension` / `AIPerfRegistry` (`rust/runtime/src/extensions/mod.rs`) | Extensions transactionally register named factories before the application registry is frozen. Directly injected clocks, observers, stores, and materializers are deliberately not named registry entries. | **Reuse and extend** | Add statically composed registries for streaming source, format, session-program, action-sink, and checkpoint-backend factories. They freeze with workloads and transports. Dynamic-library exposure is a separate plugin-API generation decision. |
| `WorkloadFactory` (`rust/runtime/src/engine/registry.rs`) | Strictly validates workload-owned config, declares resource requirements, performs effect-free run validation, and prepares one `PreparedRunnerOperation`. | **Reuse** | Register `shadow_replay` as a workload. It composes a selected streaming source, format, policies, transport execution, and phase lifecycle. No coordinator switch on S3, HF, Dynamo, or Baseten is added. |
| `PreparedRunnerOperation` (`rust/runtime/src/engine/registry.rs`) | Owns a fully prepared, intentionally `!Send` operation over current-thread Tokio and worker-local `Rc`/`RefCell` state. | **Reuse** | Own the run-scoped streaming pipeline and execute it without extending `NativeDatasetPlan`. |
| `NativeTransportExecution` (`rust/runtime/src/engine/registry.rs`) | Produces request executors, request materialization, endpoint-local bindings, readiness policy, and graph dispatch. | **Reuse** | Shadow replay resolves the existing request executor and `RequestMaterializer`; the source and format never build HTTP/gRPC requests or know that the target is Dynamo. |
| `DatasetInputAdapter` / `DatasetInputAdapterResolver` (`rust/runtime/src/engine/dataset_input.rs`) | Loads one complete authored dataset into `PreparedDatasetInput` during preparation. | **Finite-only** | Retain byte-for-byte behavior for resident datasets. A finite-to-stream adapter can expose it through the new plane, but a live source cannot implement this completion contract honestly. |
| `GraphInputAdapter` / `GraphInputAdapterResolver` (`rust/runtime/src/engine/graph_input.rs`) | Lowers a complete source into a `GraphInputBundle` with all graph programs and a frozen segment store. | **Finite-only** | Preserve Graph-IR compilation. Streaming agent/graph sessions use registered fragments/programs and an incremental graph action sink; they do not mutate this bundle. |
| `DatasetLoader` (`rust/runtime/src/dataset/loader/mod.rs`) | `load` returns `Vec<RawRow>`. Format registration pairs a loader with a composer. | **Finite-only, decoder reuse** | Keep the public resident contract. Extract reusable row/column decoders behind the new streaming format implementation where profitable; do not change `Vec<RawRow>` into a pseudo-stream whose caller still collects it. |
| `DatasetFetcher` (`rust/runtime/src/dataset/fetch.rs`) | Fetches one URL to a complete `Bytes` value and uses a persistent exact-byte cache. The production implementation may separately opt into HF Hub local-file streaming. | **Finite-only** | Keep for small artifacts/tokenizers and compatibility. It cannot express shard catalogs, range/seek access, resumable acquisition, follow mode, source cursors, or bounded disk leases. |
| HF public loader (`rust/runtime/src/dataset/loader/public.rs`) | Resolves a revision and can use `hf-hub` to cache/resume a large shard, then reads only needed rows from a local file. The outer contract still collects every retained row. | **Adapt** | Reuse revision resolution, authentication, resumable shard acquisition, and bounded local row decoding. Move shard iteration and row limits behind `StreamingDatasetSource`/`StreamingDatasetFormat` so total memory is independent of retained row count. |
| Baseten loader/composer (`rust/runtime/src/dataset/loader/baseten.rs`) | Projects Parquet/Arrow columns and decodes bounded batches, but then retains O(rows + session IDs), globally groups/sorts sessions, and returns a resident `Dataset`. | **Adapt** | Reuse projected columnar decoding and replay semantics. Emit session-addressed fragments; the run-scoped coordinator preserves sessions across shards and uses watermark-bounded state or finite external sort. |
| `Dataset` / `SegmentStore` (`rust/runtime/src/dataset/runtime_dataset.rs`, `dataset/segment.rs`) | `Dataset` owns a complete conversation population over a segment arena; sampling and body-plan precomputation assume stable enumeration. | **Finite-only plus new lease** | Do not mutate a resident `Dataset`. Streaming fragments get batch-scoped/ref-counted leases transferred into session state and released only after incorporation, spill, or terminal dispatch. |
| `ConversationSource` (`rust/runtime/src/multiturn.rs`) | Synchronously enumerates stable metadata and draws sessions from a pre-existing population. `next` cannot distinguish pending input from terminal EOF. | **Finite-only** | Preserve sampling semantics. Add an asynchronous fragment/action stream whose pending future represents no data yet and whose terminal event represents a real seal/EOF. |
| `FixedScheduleSource` / `FixedScheduleWorkload` (`rust/runtime/src/fixed_schedule.rs`) | Builds, validates, and sorts the complete schedule before execution; schedules every first turn up front. | **Finite-only, policy reuse** | Reuse timestamp validation/conversion rules, not its eager plan. Streaming replay incrementally admits only events behind the event-time watermark and inside the bounded scheduling horizon. |
| `scheduled::Workload` (`rust/runtime/src/scheduled.rs`) | An async, `!Send` schedule generator that can remain pending, issue turns incrementally, and return when sending ends. | **Reuse** | The request-action sink pulls causally ready actions, waits through `Clock`, issues through `ScheduledRuntime`, and observes stop without prebuilding the run. |
| `ScheduledRuntime` (`rust/runtime/src/scheduled.rs`) | Owns the injected clock, stop checker, scheduling, issuance gates, request observation, counters, and transport-neutral `TurnDispatcher`. | **Reuse with bounded-horizon discipline** | It is the final single-process dispatch seam. The streaming workload must not enqueue the unbounded future into `ClockTaskScheduler`; it waits/pulls incrementally and keeps only an authored horizon resident. |
| `ClockTaskScheduler` / `LocalTaskScheduler` (`rust/runtime/src/scheduler.rs`) | Tracks local delayed/running tasks and supports pending/all cancellation and drain. It has no capacity limit and creates one task per scheduled item. | **Adapt** | Continue tracking admitted dispatch. The upstream reorder/schedule buffer is explicitly bounded, and only near-horizon requests become scheduler tasks. |
| `Clock` / `RealClockAnchor` (`rust/runtime/src/clock`) | All runtime waits and measurement use an injected monotonic/virtual nanosecond timeline. | **Reuse** | Convert source UTC event time to the run's monotonic timeline exactly once through an immutable event-time anchor. No source, format, or replay policy calls wall-clock or Tokio time directly. |
| `TurnDispatcher`, `RequestMaterializer`, `RequestObserver` (`rust/runtime/src/scheduled.rs`, `dataset/request.rs`, `dispatch/sink.rs`) | Transport-neutral request lowering, dispatch, response observation, and terminal completion. | **Reuse** | Causally ready conversation actions become ordinary `TurnToSend` immediately before admission. Responses use the existing observer/metrics path. |
| `NativeMetricsObserver`, `MetricsAccumulator`, mergeable stores (`rust/runtime/src/metrics.rs`, `metrics_core`) | Accumulates worker-local terminal facts and can merge exact/sketch partitions, but normal finalization consumes the run-wide observer at the end. | **Adapt** | Rotate worker-local epoch accumulators at a contiguous terminal checkpoint horizon and persist versioned mergeable partitions. Final report reduction uses a fixed epoch/cell/worker order. |
| `RecordArtifactLane` (`rust/runtime/src/engine/record_lane.rs`) | Streams completed rows into held-open monolithic JSONL/raw/CSV/outputs/Parquet writers and flushes/finalizes at run end. | **Adapt, not durability authority** | Reuse row projection/writer logic for immutable checkpoint segments or final compaction. A flush is not a checkpoint: add epoch/range/digest identity and atomic manifest publication. |
| `Exporter` / `ExporterRegistry` (`rust/runtime/src/export/mod.rs`) | Runs after the final `NativeReport` exists; exporters are presentation/file/upload sinks. | **Reuse** | Keep exporters out of checkpoint coordination. They consume the final compacted report/projections; optional live dashboards read committed checkpoint generations through a separate result-reader seam. |
| `PhaseExecution`, `PhaseExecutionFactory`, `PhaseRunner` (`rust/runtime/src/timing/phase/runner.rs`) | Defines setup, start, execute, stop issuance, cancel pending/in-flight, drain, ramp stop, and finalize ordering. | **Reuse** | Streaming ingestion and replay must be represented as phase-owned execution/resources, so cancellation, grace, drain, partial reports, and signals retain one lifecycle authority. |
| `ScheduledPhaseSidecar` / `ScheduledRuntimeExtension` (`rust/runtime/src/phase_runtime.rs`) | Adds low-rate phase-synchronized services or observer/gate/controller policy without replacing normal measurement. | **Adapt** | Source health and checkpoint writers may use sidecars. The dataset stream itself is not a sidecar because its failure and completion determine workload execution. |
| cellular controller/cell protocols (`rust/runtime/src/cellular`) | Partition a fixed run, push fixed inputs before execution, and merge terminal records/stores. | **New incremental protocol** | Add bounded sequence/ack streaming placement. The controller remains ordering authority for live shadow replay; immutable finite shards may use partition-local placement after exact snapshot identity is bound. |

## Existing strengths to preserve

### Workload/transport independence

The registry already treats workload and transport as independent named
components. `shadow_replay` therefore belongs beside `scheduled` and `graph` as
a workload, not as a transport, endpoint, Dynamo mode, or hidden branch in the
native driver. Its output must reach any `NativeTransportExecution` that can
provide the ordinary request-executor seam.

### Phase lifecycle authority

`PhaseRunner` already defines the only credible shutdown order:

```text
configure -> setup -> start ramps -> execute issuance
          -> stop issuing -> cancel pending -> grace/drain
          -> cancel in flight -> release stuck slots
          -> stop ramps -> finalize
```

Streaming source polling, acquisition, decoding, scheduling, and checkpoint
flush must be attached to these hooks. A separate background runtime with its
own signal handling would create two owners for stop and drain.

### Clock and dispatch neutrality

`ScheduledRuntime::wait_until_or_stop` and `issue_turn` already provide the
right final scheduling and dispatch primitives. `TurnDispatcher` and
`RequestMaterializer` already prevent endpoint wire details from leaking into a
dataset loader. The new data plane ends at these seams.

### Frozen component universe

Streaming sources, formats, session programs, and checkpoint backends are
selected and validated before runtime effects and then frozen. Live *data* and
checkpoint generations change during the run; the implementation universe does
not. This follows the native-runtime-plugin design's distinction between a
frozen capability universe and run-time values produced by those capabilities.

## Finite assumptions that must not leak into the new plane

1. `Vec<RawRow>` means successful load is equivalent to reaching EOF.
2. `Dataset::conversations()` means the complete population is enumerable.
3. `ConversationSource::next()` means a missing item is an error or exhaustion,
   never temporary absence.
4. `FixedSchedule` means the global timestamp sort has already completed.
5. `GraphInputBundle` means topology and segment identity are frozen.
6. Current cellular startup means the complete source snapshot is known before
   cell execution.
7. Current schedulers can create one future per planned request because the plan
   is finite; a perpetual feed cannot inherit that behavior.

These contracts remain valid in their finite domains. The design adds a new
lifecycle instead of weakening their meanings.

## New seam summary

The detailed design introduces these narrow boundaries:

1. `StreamingDatasetSourceFactory` and `StreamingDatasetSource` for immutable
   partition discovery/acquisition and source progress.
2. `StreamingDatasetFormatFactory` and `StreamingDatasetFormat` for bounded
   decode into canonical session-addressed fragments.
3. `StreamingSessionProgramFactory` and `StreamingSessionCoordinator` for
   registered conversation/agent-graph semantics, cross-partition causal state,
   explicit session closure, checkpoint restore, and seamless continuation.
4. `DatasetActionStream` for pull-based action/watermark/barrier/end delivery.
5. `EventTimePolicy`, `LateRecordPolicy`, and `ReplayAdmissionPolicy` for
   ordering and scheduling decisions independent of source/format.
6. `StreamingCheckpointParticipant` for typed source/decoder/session/reorder/
   action/placement state at one barrier.
7. `StreamingCheckpointBackendFactory`, `StreamingCheckpointBackend`, and one
   generation transaction for atomic resume/result publication, epoch-aligned
   metric/record segments, and restart-safe final report assembly.
8. `StreamingPlacement` for single-process, centrally ordered cellular, and
   immutable-partition cellular execution.

Source, format, session program, action sink, and checkpoint backend are named
registry categories. Policies are validated host-owned values or narrow injected traits
unless/until there is a concrete need for third-party registration. This avoids
turning every strategy knob into a global plugin surface.
