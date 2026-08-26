<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native streaming datasets and shadow replay

## Status

Working architectural specification. This document is intentionally broader
than the first NVCF integration. It defines a native Rust streaming dataset
plane and selects shadow replay as its first run-time consumer.

## Purpose

AIPerf will consume datasets whose complete row population is not resident in
memory and may not exist when the run starts. The same plane will support:

- multi-gigabyte finite Hugging Face datasets, including Baseten Parquet trace
  shards, with memory independent of total retained rows;
- immutable local or remote shard collections;
- object-store prefixes that publish new immutable objects throughout a run;
- time-delayed shadow replay that preserves source event ordering and target
  timing subject to explicit late/overload policy; and
- future streaming consumers without coupling source acquisition to replay.

The first live integration is an NVCF object-store feed published in roughly
five-minute chunks. AIPerf will discover and acquire those objects, decode the
selected trace format, establish an event-time frontier, and replay requests
against an ordinary configured frontend at `source_timestamp + delay`.

S3 is one source adapter. Hugging Face Hub is another. Dynamo trace and Baseten
trace are format adapters. Shadow replay is a workload. Dynamo as the replay
target is an endpoint/transport selection and does not appear in the streaming
dataset contracts.

## Goals

1. Add a general streaming dataset lifecycle without weakening the meanings of
   existing finite dataset and Graph-IR contracts.
2. Select source, format, session program, workload, endpoint, and transport
   through independent traits and frozen registries.
3. Bound memory, local disk, queued partitions, decoded fragments/actions, session state,
   reorder state, scheduled work, and cellular transfer independently of total
   dataset size or stream lifetime.
4. Support sealed finite streams and indefinitely following streams through one
   data-plane contract with distinct terminal semantics.
5. Preserve exact immutable source identity, provenance, and resumable cursors
   without retaining credentials or raw source data in ordinary artifacts.
6. Make event-time completeness, late records, overload, gaps, duplicates,
   restarts, source loss, cancellation, drain, and checkpoint behavior authored
   and observable.
7. Reuse `Clock`, `Workload`, `ScheduledRuntime`, `TurnDispatcher`,
   `RequestMaterializer`, `RequestObserver`, `PhaseExecution`, and worker-local
   execution rather than constructing a second benchmark engine.
8. Keep the implementation pure Rust. No Python loader, replay process, or
   Python compatibility path participates in validation or execution.
9. Permit single-process and cellular execution without giving every cell
   uncontrolled source authority or destroying global event order.
10. Provide a migration path for current finite loaders to reuse bounded
    decoders without requiring them to become streaming APIs.
11. Preserve one logical session seamlessly across source partitions and
    checkpoints for multi-turn conversations, recorded agent/tool trajectories,
    and graph sessions, including causal predecessors that arrive in later
    chunks.

## Non-goals

- Mutating an executing `GraphRecord` or `GraphInputBundle` in generation 1.
- Claiming exactly-once effects at a target that offers no idempotency contract.
- Treating a quiet follow source as EOF.
- Silently replaying a partially published HF conversion as the complete
  dataset.
- Passing S3/HF credentials through dataset records, checkpoints, artifacts,
  request bodies, or ordinary cellular DTOs.
- Making source or format factories dynamically loadable under generation 1 of
  the native shared-library plugin design. They are statically composed Rust
  registry categories in this specification.
- Using unbounded channels, unbounded in-memory deduplication, one task per
  future record, or a complete in-memory timestamp sort.
- Defining a general distributed log service inside AIPerf.

## Normative language and invariants

`MUST`, `MUST NOT`, `REQUIRED`, `SHALL`, `SHALL NOT`, `SHOULD`, `SHOULD NOT`,
and `MAY` are normative. A component that violates a `MUST` or `MUST NOT` is
not a conforming implementation of this design.

The following invariants apply together:

1. **Independent composition.** Source, format, session program, replay policy,
   checkpoint backend, placement, endpoint, and transport are independently
   selected or injected. No source adapter constructs endpoint wire requests,
   no format adapter lists S3 or resolves HF revisions, and no result sink owns
   input progress.
2. **Frozen implementation universe.** Every source, format, session-program,
   action-sink, and checkpoint-backend factory is registered, selected, and strictly
   validated before source acquisition, network clients, worker runtimes,
   cells, or benchmark traffic begin. Live data and committed generations
   change; the factory universe does not.
3. **One lifecycle authority.** Phase execution owns start, stop issuance,
   pending cancellation, in-flight cancellation, drain, and finalization.
   Streaming tasks have no independent signal handler or detached shutdown
   path.
4. **Bounded resources.** Every queue and state owner has finite item and byte
   limits. Total memory is bounded by the configured pipeline and in-flight
   request budgets, not source size or run duration. Disk spill/cache is
   separately finite and measured.
5. **Backpressure before loss.** A full downstream boundary suspends upstream
   progress. Dropping, sampling, coalescing, or skipping data occurs only under
   an explicit authored policy and produces counts plus stable reason codes.
6. **Pull distinguishes pending from terminal.** Temporary absence is a pending
   future. `End` is emitted only for a sealed finite source, an authored row/time
   limit, or an explicit terminal source event. Poll timeout and quiet period do
   not imply EOF.
7. **Immutable acquired partitions.** A partition is decoded only from one
   acquired immutable object/version. Pathname hash-then-open is forbidden.
   Replacement during acquisition or decoding fails the partition/run under the
   authored source-error policy.
8. **Exact snapshot identity.** Finite sources resolve mutable names such as HF
   `main` exactly once into an immutable generation before row execution.
   Follow sources identify every object generation independently and never
   reinterpret an already admitted key as new bytes.
9. **Separate progress domains.** Discovery cursor, acquired-byte cursor,
   decoded-record cursor, event-time watermark, scheduled/admitted horizon, and
   terminal-execution acknowledgement are distinct typed values. One MUST NOT
   be substituted for another.
10. **Watermark honesty.** A hard event-time watermark asserts that no later
    admitted source record can have event time at or below it. A heuristic
    frontier is labeled estimated and is usable only with an explicit late-data
    policy. Silence alone never advances a hard watermark.
11. **Clock authority.** All replay waits and measurements use the injected
    `Clock`. UTC source time is mapped to the monotonic run timeline through one
    immutable anchor. Source/format code does not call `SystemTime::now`,
    `Instant::now`, or Tokio timers directly.
12. **No unbounded future scheduling.** Only records within the authored
    scheduling horizon may become `ClockTaskScheduler` tasks. Later records
    remain in a bounded reorder store or backpressure the source.
13. **Canonical execution boundary.** A streaming format emits canonical AIPerf
    session fragments. A selected session program produces causally ready
    actions. Endpoint-specific materialization occurs just before admission
    through the existing `RequestMaterializer`; dispatch occurs through the
    existing `TurnDispatcher` and observer plane.
14. **Lifetime-safe segments.** Bytes and segments referenced by a fragment,
    active session, or admitted request remain alive through incorporation or
    terminal dispatch. They are reclaimed only after the final owning session,
    request/continuation, and checkpoint receipt release them; a perpetual
    source does not retain a perpetual segment arena.
15. **Deterministic ordering.** Equal event-time records are ordered by a stable
    source-derived key and global sequence. Filesystem listing order, request
    completion order, worker wake order, and hash-map iteration never decide
    replay order.
16. **Explicit session closure.** A session program may close a multi-turn,
    agentic, or graph session only from an explicit end marker, a proven
    key/time watermark, sealed finite validation, or an authored
    bounded-inactivity rule. Otherwise it MUST keep bounded/spilled active state
    or fail validation; it cannot equate partition EOF with closure.
17. **Restart is authored.** Checkpoint policy declares whether restart resumes
    from terminal acknowledgement, admission, decode, or source acquisition.
    The report states the resulting at-least-once/at-most-once window. Exactly
    once is claimed only when target idempotency is configured and verified.
18. **Controller ordering authority.** Centrally ordered cellular replay assigns
    global sequence and ownership at the controller. Cells cannot independently
    poll a live prefix and infer a common order from incomparable clocks.
19. **Credential confinement.** Source credentials remain inside the source
    adapter and approved acquisition role. Cross-host cells receive immutable
    data/partition authority only through an explicit scoped mechanism; they do
    not receive controller credentials by default.
20. **Finite compatibility.** Existing finite `DatasetLoader`,
    `DatasetInputAdapter`, `ConversationSource`, `FixedSchedule`, and
    `GraphInputAdapter` behavior remains unchanged unless a run explicitly
    selects a streaming workload/resource.
21. **No silent partial HF dataset.** An HF source binds a commit and a complete
    selected split shard inventory. Dataset Viewer partial conversion is not a
    complete snapshot. Partial inventory is accepted only when explicitly
    authored as a row/byte-limited source and reported as such.
22. **Observable fidelity.** Source publication delay, acquisition time, decode
    time, watermark age, queue occupancy, event lateness, schedule slip,
    admission delay, drops, duplicates, gaps, and checkpoint horizons are
    observable independently of endpoint latency.
23. **Typed failures.** Source, acquisition, decode, ordering, state-budget,
    checkpoint, placement, and dispatch failures retain distinct stable codes.
    A retryable acquisition failure is not reported as malformed data, and a
    late record is not reported as transport latency.
24. **No raw-source artifact by default.** Provenance records identities,
    digests, cursors, schemas, and counts. Raw source bytes are emitted only
    through an explicit artifact policy subject to existing secret/raw-payload
    controls.
25. **Chunk-transparent sessions.** Source objects, HF shards, row groups,
    decoder batches, checkpoint barriers, and cellular chunks are acquisition or
    transfer boundaries only. They MUST NOT implicitly open, close, reset, or
    fork a logical session. One session may span any number of these boundaries.
26. **One session owner.** Every active multi-turn, agentic, or graph session is
    assigned exactly one logical owner at a time. All fragments for its stable
    key reach that owner in causal order. Cellular ownership changes only after
    an authenticated state-transfer transaction commits.
27. **Atomic session progress.** A checkpoint MUST NOT advance source progress
    past a cross-chunk fragment unless the session state or terminal execution
    acknowledgement needed to resume it is committed in the same generation.
    Restart cannot fabricate a new session at the next chunk.
28. **Missing cross-chunk dependencies stay missing.** An absent parent,
    predecessor turn, tool result, or graph edge at partition EOF is incomplete
    session state, not a new root or terminal. Only explicit close, an applicable
    hard watermark, or sealed-source validation can resolve that absence.
29. **Checkpointed results.** Every durable checkpoint epoch binds input
    horizons, active-session state, metric accumulator state, record-result
    segments, and results inventory. Immutable segments are made durable first;
    one canonical generation record then atomically commits both checkpoint and
    result visibility. There is no separately authoritative checkpoint head and
    results head that can diverge.
30. **Idempotent result publication.** Result segment identity derives from run
    identity, checkpoint epoch, cell/worker partition, projection/schema, and
    payload digest. Retrying publication cannot duplicate a record or metric
    contribution. Final results are a deterministic compaction of committed
    segments, never an independent best-effort reconstruction.
31. **Typed atomic participants.** Durable progress advances only after every
    participant required by the frozen plan has produced a versioned,
    digest-addressed view at the barrier and the one backend transaction commits
    those views with the result-index root. Opaque state-store writes are not a
    checkpoint receipt.
32. **Bounded durable roots.** A generation record has bounded size and reaches
    cumulative result/state data through immutable content-addressed indexes.
    Publication uses compare-and-swap/single-writer fencing and crash-durable
    pointer installation; rename without expected-generation validation is not
    a conforming commit.
33. **Logical action uniqueness.** Logical action identity and physical attempt
    identity are distinct. Exactly one terminal logical result contributes to
    authoritative metrics; duplicate submission attempts are either proven
    idempotent or retained as separately classified, non-contributing telemetry.
34. **Causality must terminate or be bounded.** A cross-chunk session declares a
    hard closure/completeness proof or an explicit lossy timeout/drop/fail
    policy plus finite state/spill bounds. Backpressure alone is not a solution
    to a predecessor that may appear arbitrarily far in the future.
35. **Cross-host time is not shared by assumption.** A monotonic deadline from
    one host is never interpreted on another host. Cellular issue authority uses
    controller release/fencing or a separately validated synchronization
    protocol, and early dispatch is forbidden.
36. **Blocking work is isolated and bounded.** Network file acquisition,
    Parquet/Arrow decoding, compression, external sort, fsync, index building,
    and final compaction MUST NOT block a worker `LocalSet` or clock/scheduling
    loop. Submission, queued work, and result bytes are bounded; shutdown
    cancels cooperatively and joins every blocking owner.
37. **Session-update authority is authored.** Recorded-input replay and
    target-derived closed-loop execution are distinct modes. Target responses
    never silently rewrite later recorded requests. Restartable target-derived
    state is committed only through a backend capability that encrypts sensitive
    session material under external secret authority.
38. **Logical identity is content-bound and topology-stable.** Record, session,
    action, and attempt identities have host-owned derivations. Restart, worker/
    cell placement, discovery order, and global sequence assignment cannot
    change logical IDs. The same ID with different canonical content is a hard
    conflict, never a duplicate.

## Decision traceability

| Decision | Normative resolution | Detailed section |
|---|---|---|
| Product center | General streaming dataset plane; shadow replay is the first consumer | Purpose; architecture |
| First source adapters | HF Hub finite snapshots and S3-compatible finite/follow catalogs | Source contracts |
| First format adapters | Baseten trace and Dynamo/NVCF request trace; ordinary row formats follow | Format contracts |
| Execution model | Registered `shadow_replay` workload over ordinary native transports | Registry and composition |
| Finite compatibility | Existing resident dataset and Graph-IR APIs remain unchanged | Invariant 20; migration |
| Canonical boundary | Versioned session-addressed fragments plus registered session programs that emit causally ready actions | Canonical session fragments |
| Pending versus EOF | Async pending is absence; typed seal/limit is terminal | Invariant 6 |
| Memory model | Bounded queues, batch arenas, spill/cache budgets, near-horizon tasks | Resource model |
| Event-time order | Watermark-gated stable ordering with explicit hard/estimated quality | Event time and replay |
| Wall-clock replay | `event_utc + replay_delay` mapped once onto monotonic `Clock` | Event time and replay |
| Overload | Backpressure by default; drop/abort only when authored | Backpressure and late data |
| Checkpoint | Separate monotonic stage horizons; default restart from terminal ack | Checkpoint and delivery |
| Cellular live replay | Controller owns discovery, event order, sequence, and placement | Cellular placement |
| Cellular finite replay | Immutable shard assignment is allowed when snapshot/digest is bound | Cellular placement |
| HF completeness | Pin commit, enumerate full split inventory, reject silent Viewer partials | HF source |
| Baseten boundedness | Reuse projected batch decode; remove O(total rows) outer collection | Baseten format |
| Cross-chunk sessions | Run-scoped keyed session coordinator; chunks never imply close; cellular ownership is sticky | Session continuity |
| Results durability | Checkpoint-aligned immutable result segments, atomic bounded generation roots, deterministic final compaction | Checkpoint-based results |
| Plugin relationship | Static registry category now; dynamic plugin category requires new API generation | Registry and composition |

## Invariant enforcement map

| Invariants | Enforcement | Required evidence |
|---|---|---|
| 1, 2, 20 | Independent registry IDs, strict factory config, prepared composition, production search for source/format/session/action switches | Cross-product source×format×session×action-sink×transport tests; unknown/duplicate ID tests |
| 3 | Streaming operation implemented through `PhaseExecution`/`PhaseRunner` hooks | Cancellation at every lifecycle boundary; no leaked-task checks |
| 4, 5, 12 | Count+byte bounded channels/stores, resource permits, high-water metrics, near-horizon scheduler gate | Multi-GB logical fixtures under fixed RSS/disk ceilings; saturation tests |
| 6, 10 | Typed `SourceEvent`, explicit seal, watermark quality enum | Long quiet follow tests; late-after-estimated-frontier tests |
| 7, 8, 21 | No-follow/local leases, version/commit binding, conditional object reads, complete catalog receipts | Mutation races; HF revision drift; partial conversion fixtures |
| 9, 17 | Newtype cursors and monotonic checkpoint record | Compile-time/API tests; crash/restart matrix by stage |
| 11 | `Clock` in policy contexts; denied direct wall/timer calls in streaming modules | Static searches/lints plus SimClock deterministic tests |
| 13, 14 | Host-owned fragments/actions, existing conversation materializer/dispatcher, ref-counted leases | HTTP/gRPC/dry-run parity; terminal-drop lifetime tests |
| 15, 16, 25-28 | Stable total-order key; declared session closure capability; run-scoped keyed coordinator; atomic session-state checkpoint | Shuffled listing, equal timestamp, interleaved cross-chunk multi-turn/agentic/graph sessions, restart and owner-migration tests |
| 18, 19 | Controller-assigned global sequence; scoped partition/data transfer; no secret fields | Multi-cell skew tests; protocol schema/secret scans |
| 22, 23, 24 | Host-owned metrics/failure/provenance vocabulary | Golden report/artifact tests; redaction fixtures |
| 29-33 | Typed checkpoint participants, one epoch transaction, bounded content-addressed result index, compare-and-commit root, logical-action membership/dedup, deterministic compactor | Crash at every publish/restore/GC boundary; retry/dedup; live partial-read; non-growing root; final equivalence tests |
| 34 | Declared session closure and predecessor-lateness capability plus bounded spill/failure policy | Missing-parent beyond each chunk; indefinite follow quietness; state-budget exhaustion |
| 35 | Controller release protocol and ownership-epoch receipts | Host clock skew, transfer delay, early-issue refusal, stale-owner receipt tests |
| 36 | Host-owned bounded blocking executor with byte/item permits and joined cancellation | Saturated decode/fsync/compaction while SimClock dispatch remains responsive; shutdown leak tests |
| 37 | Strict session-update policy plus encrypted-sensitive-state capability agreement | Recorded replay with divergent target text; closed-loop restart/key-loss/redaction tests |
| 38 | BLAKE3 domain-separated ID constructors and canonical-content conflict ledger | Duplicate notification/overlapping chunk/collision fixtures across restart and topology |

## Terminology

- **stream resource**: a validated `{source, format, session program}`
  composition referenced by a workload.
- **source**: discovers immutable partitions and supplies acquired byte access;
  it does not interpret rows.
- **partition**: one immutable source object or bounded page with a stable
  identity and ordering coordinate.
- **format**: decodes partition bytes into canonical session-addressed
  fragments; partition exhaustion has no session meaning.
- **session fragment**: one conversation turn, agent/tool event, graph node,
  graph edge, or explicit close addressed by stable logical session key.
- **dataset action**: one causally ready executable action emitted by the
  session coordinator. Strict row replay commonly yields one request action per
  source record.
- **source frontier**: source-specific progress in discovery/order space.
- **event-time watermark**: a claim about completeness in decoded event time.
- **replay target**: the monotonic clock time at which a unit should be issued.
- **stage horizon**: the greatest monotonically contiguous position completed
  at one pipeline stage.
- **seal**: authoritative terminal declaration for a finite snapshot.

## Architecture

```text
Frozen AIPerfRegistry
  |-- StreamingDatasetSourceFactory["hf_hub", "s3", "local", ...]
  |-- StreamingDatasetFormatFactory["baseten_trace", "dynamo_trace", ...]
  |-- StreamingSessionProgramFactory["conversation", "agent_graph", ...]
  |-- StreamingActionSinkFactory["scheduled_request", "streaming_graph", ...]
  |-- StreamingCheckpointBackendFactory["local", "object_store", "none"]
  |-- WorkloadFactory["shadow_replay"]
  `-- existing endpoint / transport / exporter / actuator factories

Validated stream resource
  source config + format config + capability agreement + resource budgets
                              |
                              v
Run-scoped streaming dataset runtime
  discover -> acquire -> decode fragments -> session coordinator
     |           |              |                |-- keyed causal state
  cursor      digest/cache   format state        `-- explicit close/watermarks
                                                   |
                                                   v
                                     global watermark/reorder -> action stream
Shadow replay phase execution
  time mapping -> near-horizon admission -> RequestMaterializer
               -> ScheduledRuntime -> TurnDispatcher -> RequestObserver
                              |
                              v
               terminal acknowledgement
                         |
                         v
       checkpoint epoch -> immutable result/index objects -> atomic generation root
```

The pipeline MAY fuse adjacent stages on one current-thread `LocalSet` when no
parallelism is useful. Trait boundaries define ownership and testability; they
do not require a channel or task hop. When stages are concurrent, the host owns
bounded channels with both item and byte permits.

## Registry and composition

### New registry categories

`AIPerfRegistry` gains five transactional, frozen categories:

```rust
stream_sources: TransactionalRegistry<Arc<dyn StreamingDatasetSourceFactory>>,
stream_formats: TransactionalRegistry<Arc<dyn StreamingDatasetFormatFactory>>,
stream_session_programs:
    TransactionalRegistry<Arc<dyn StreamingSessionProgramFactory>>,
stream_action_sinks:
    TransactionalRegistry<Arc<dyn StreamingActionSinkFactory>>,
stream_checkpoint_backends:
    TransactionalRegistry<Arc<dyn StreamingCheckpointBackendFactory>>,
```

The categories follow existing duplicate rejection, descriptor validation, and
freeze behavior. `AIPerfExtension` gains registration methods. Built-ins use the
same registration path as statically linked extensions.

`StreamingSessionProgramFactory` is the semantic bridge between a format's
fragment schema and a workload's accepted action schema. It owns causal
readiness and session-state behavior for conversation or agent/graph programs.
`StreamingActionSinkFactory` binds one action schema to the selected
transport/endpoint and shared execution services. This prevents the workload
from acquiring a closed source/session/action `match` while keeping timing,
phase, and checkpoint policy host-owned.
`shadow_replay` is a normal `WorkloadFactory`. Its validation resolves stream
resource identity and performs capability agreement without network or file
effects. Its preparation returns its own `PreparedRunnerOperation`; it does not
add a `Streaming` variant to `NativeDatasetPlan` merely to pass through the old
finite driver.

### Protocol-v2 resource

Protocol v2 adds a named stream collection so a workload references a resource
rather than embedding a source implementation:

```yaml
run:
  resources:
    dataset_streams:
      items:
        - id: shadow_input
          source:
            id: s3
            config:
              bucket: nvcf-traces
              prefix: production/chat/
              mode: follow
              poll_interval: 15s
          format:
            id: dynamo_trace
            config:
              schema: dynamo.request.trace.v1
          session_program:
            id: agent_graph
            config:
              missing_predecessor: wait
              update_policy: recorded_inputs
          limits:
            acquired_partitions: 4
            decoded_fragments: 10000
            decoded_bytes: 512MiB
            state_memory: 512MiB
            state_disk: 100GiB

  workload:
    id: shadow_replay
    config:
      stream: shadow_input
      actions:
        request:
          sink: scheduled_request
          config: {}
        graph_node:
          sink: streaming_graph
          config: {}
        session_terminal:
          sink: session_state
          config: {}
      time:
        mode: wall_clock_delay
        delay: 6m
      ordering:
        max_out_of_order: 30s
        watermark: source_manifest_or_bounded
        late: fail
      overload:
        mode: backpressure
      checkpoint:
        mode: terminal
        interval: 10s
        backend:
          id: local
          config: {}
        results:
          publish_partial: true
          retention:
            resume_roots: 2
            partial_history: 12
            final_compaction: until_exported
            source_cache: through_resume_root
            orphan_ttl: 24h
```

Names and exact field spelling remain subject to implementation schema review;
the ownership is normative. Source config belongs to its factory, format config
belongs to its factory, session/causal config belongs to its session-program
factory, checkpoint storage config belongs to its checkpoint-backend factory,
and replay/result policy belongs to the workload. Mixed or unknown fields fail
strict validation.

The workload requires exactly one action-sink binding for every schema the
selected session program can emit and forbids two bindings for the same schema.
Resolution is a frozen registry lookup by sink ID followed by descriptor
agreement; there is no first-match or registration-order selection.

`RunResourceV2` and `ResourceRequirementsV2` gain dataset-stream presence.
Workloads that do not use streams neither validate nor open them.

### Capability agreement

Source, format, session-program, workload, action-sink, and checkpoint-backend
descriptors are
side-effect-free. Validation intersects:

- source mode: `finite`, `follow`, or both;
- byte access: sequential chunks, immutable local seekable lease, or range
  reads;
- source ordering: none, partition order, or event-time-related guarantee;
- resumability granularity: partition, byte, row group, or record;
- format media/schema identifiers;
- format access requirement and projection support;
- canonical fragment output schema (`aiperf.session_fragment.v1` initially);
- session-program accepted fragment and emitted action schemas;
- workload accepted action schemas;
- checkpoint backend atomic-generation, segment, durability, and reader
  capabilities;
- event-time and stable-record-ID availability;
- session-closure requirements;
- report/export retention requirements, including whether a component requires
  an O(total-records) resident collection;
- placement support; and
- virtual-clock compatibility.

An incompatible composition fails before source effects and names every
participating descriptor plus the missing capability. There is no coordinator
switch on a source/format/session-program combination.

All five categories are feature-gated and reported through the same frozen
application capability inventory as workloads and transports. A lean build
omits both an unavailable descriptor and any workload whose required built-ins
cannot be composed. Duplicate registration, unknown IDs, and descriptor/build
feature mismatches fail in the existing transactional bootstrap path.

Generation 1 streaming validation rejects accuracy/evaluator mode because
`NativeReport::accuracy_records` is a resident `Vec` with no external projection
contract. It also rejects any exporter or sidecar descriptor that requires the
complete exact record population in memory. Summary-only exporters remain
compatible, while requested JSONL/CSV/Parquet/raw record artifacts are produced
by bounded segment compaction. Streaming accuracy requires a later explicit
externalized accuracy-report schema and exporter API; it is not simulated by
loading checkpoint segments back into one vector.

## Source contracts

### Factory and run-time source

The following sketches fix responsibility, not final spelling:

```rust
pub trait StreamingDatasetSourceFactory: Debug + Send + Sync {
    fn descriptor(&self) -> &'static StreamingSourceDescriptor;
    fn validate(
        &self,
        authored: &RawValue,
    ) -> Result<Box<dyn ValidatedStreamingSourceConfig>>;
    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingSourceConfig>,
        context: &StreamingSourcePrepareContext,
    ) -> Result<Box<dyn PreparedStreamingDatasetSource>>;
}

#[async_trait(?Send)]
pub trait PreparedStreamingDatasetSource {
    async fn open(
        self: Box<Self>,
        stop: StreamingStopReceiver,
    ) -> Result<OpenedStreamingDatasetSource, StreamSourceError>;
}

#[async_trait(?Send)]
pub trait StreamingDatasetSource: StreamingCheckpointParticipant {
    fn snapshot(&self) -> &SourceSnapshotReceipt;
    async fn next_event(&mut self) -> Result<SourceEvent, StreamSourceError>;
}

pub struct OpenedStreamingDatasetSource {
    pub source: Box<dyn StreamingDatasetSource>,
    pub control: StreamingSourceControl,
}

pub enum SourceEvent {
    Partition(SourcePartition),
    Frontier(SourceFrontier),
    Seal(SourceSeal),
}
```

`next_event` remains pending when a follow source has no new object. It does not
return an `Idle` event that invites polling loops. The cloneable, host-owned
stop receiver/control handle wakes a pending call without borrowing the source;
phase shutdown therefore works while a `next_event(&mut self)` future is alive.
Stop is not a fabricated source seal.

### Partition access

A source partition contains metadata and an opaque acquired-content handle. The
format requests one access shape declared during capability agreement:

```rust
pub enum PartitionAccessRequest {
    Sequential { resume_offset: u64 },
    SeekableLocal,
    RangeReadable,
}

#[async_trait(?Send)]
pub trait SourcePartitionContent {
    fn identity(&self) -> &ImmutableObjectIdentity;
    fn size_bytes(&self) -> Option<u64>;
    async fn acquire(
        &self,
        request: PartitionAccessRequest,
        budget: &AcquisitionBudget,
    ) -> Result<AcquiredPartition, StreamSourceError>;
}
```

- `Sequential` yields bounded `Bytes` chunks and a rolling digest.
- `SeekableLocal` yields an owned no-follow file/snapshot lease suitable for the
  current Arrow/Parquet readers. The lease, not a mutable pathname, is
  authority. Download size consumes disk budget, not memory budget.
- `RangeReadable` supports bounded immutable ranges for a future native Parquet
  range reader. It is not required for generation 1.

The acquired value retains the cache/staging lease until decoding finishes. A
decoder never reopens the caller's original mutable path.

### S3-compatible source

The pure Rust S3 adapter owns SDK client construction, authentication, region,
endpoint, listing/reconciliation, conditional acquisition, retry, and object
identity. It supports:

- sealed finite inventory from an explicit manifest or versioned prefix
  snapshot;
- lossless follow mode only when the producer supplies sealed manifests/time
  buckets or an immutable monotonic publication key with a hard no-backfill
  guarantee;
- periodic full reconciliation of every unsealed producer interval, backed by a
  durable seen-generation index rather than a listing continuation token;
- optional notification hints that accelerate discovery but do not replace
  reconciliation authority;
- stable lexicographic or manifest-authored partition order;
- VersionId-qualified GET when available, otherwise a provider-supported
  conditional GET bound to the listed ETag/version token, followed by size and
  exact acquired-byte BLAKE3 verification; and
- continuation-token handling without treating a page boundary as a stream
  frontier.

An object is published to the decoder only after its immutable generation is
bound. A later object with the same key and a different version is either a new
partition under an explicit versioned policy or a source mutation failure. It
is never silently substituted.

Precondition failure or identity mutation aborts that acquisition. An
unversioned endpoint without a conditional-read primitive cannot provide
lossless/restartable follow mode; it is accepted only under an explicitly
lossy source policy. Multipart ETags are tokens, never content digests.

A lexicographic cursor cannot prove that a later-created key/version will not
appear behind it. After a producer interval is hard-sealed, its exact manifest
or reconciled generation set becomes the durable source checkpoint and may be
removed from the active seen index. Without a seal/no-backfill contract, the
adapter must retain and rescan an explicitly bounded time/key window; late
generations outside it follow an authored lossy/fail policy and the source
cannot advertise a hard watermark or lossless restart. Perpetually rescanning
or remembering an arbitrary unsealed prefix is rejected as unbounded state.

Retries use bounded exponential backoff with jitter driven by a source-control
clock/service, not benchmark request timing. Retry exhaustion, authorization,
not-found-after-list, checksum mismatch, throttling, and source mutation are
distinct errors.

### Hugging Face source

The HF source is a finite source unless a future Hub event contract is designed.
It:

1. resolves repository/revision once to an exact commit SHA;
2. resolves one subset and split;
3. accepts only a fully expanded static data-file mapping or complete Hub/API
   inventory whose entries and metadata are bound to that commit;
4. optionally selects a complete Parquet inventory only when the API reports it
   complete;
5. sorts shard identities deterministically;
6. acquires shards through the native `hf-hub` cache/resume path into immutable
   local leases; and
7. seals only after the pinned inventory is exhausted or an authored limit is
   reached.

Hugging Face documents shard-based iterable streaming and resumable loader
state, and its Dataset Viewer documents Parquet shard inventories and partial
conversion for datasets above its conversion bound. AIPerf mirrors the useful
properties—pinned shards, bounded iteration, and explicit cursor state—but does
not depend on Python `datasets`. A Viewer response marked partial cannot
silently stand for a complete split.

Repository-card filename inference, glob guesses, executable dataset scripts,
generated builders, and ambiguous subset/split heuristics cannot prove a
complete snapshot and are rejected by the streaming HF source. Supporting one
of those dataset classes requires a separately versioned inventory resolver
whose completeness receipt is part of the source descriptor and conformance
suite.

Inventory construction is itself streaming and budgeted. Entries are validated
into bounded sorted runs on disk and merged into an immutable content-addressed
catalog; memory does not retain every shard name. The source checkpoint and
provenance record the catalog root plus cumulative counts/bytes, not an inline
ever-growing entry list. Catalog files consume source-cache/state-disk budget
and remain leased through every retained resume root.

The resumable cursor is at least:

```text
repo + commit_sha + subset + split + shard_identity
+ row_group_or_byte_offset + row_ordinal + decoder_schema_digest
```

Auth tokens are accepted through existing secret/config authority but are
redacted from debug, checkpoint, and provenance values.

### Local source

The local adapter acquires explicit files/directories once using no-follow
descriptors, deterministic entry ordering, and exact metadata/digests. Follow
mode uses a dedicated watched-directory contract with immutable publish/rename
rules; it does not reinterpret arbitrary in-place file appends as committed
records unless a format explicitly supports an append log and owns torn-record
recovery.

## Format contracts

### Factory and decoder

A streaming format owns byte decoding and format-private state. The host does
not force every optimized columnar row through `serde_json::Value`.

```rust
pub trait StreamingDatasetFormatFactory: Debug + Send + Sync {
    fn descriptor(&self) -> &'static StreamingFormatDescriptor;
    fn validate(
        &self,
        authored: &RawValue,
        source: &StreamingSourceDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingFormatConfig>>;
    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingFormatConfig>,
        context: &StreamingFormatPrepareContext,
    ) -> Result<Box<dyn StreamingDatasetFormat>>;
}

#[async_trait(?Send)]
pub trait StreamingDatasetFormat: StreamingCheckpointParticipant {
    async fn begin_partition(
        &mut self,
        partition: AcquiredPartition,
        resume: Option<DecoderCheckpoint>,
    ) -> Result<Box<dyn StreamingPartitionDecoder>, StreamFormatError>;

    async fn advance_source_frontier(
        &mut self,
        frontier: SourceFrontier,
        output: &mut dyn FormatEventSink,
    ) -> Result<(), StreamFormatError>;

    async fn seal(
        &mut self,
        seal: SourceSeal,
        output: &mut dyn FormatEventSink,
    ) -> Result<FormatSealReceipt, StreamFormatError>;
}

#[async_trait(?Send)]
pub trait StreamingPartitionDecoder {
    async fn next_batch(
        &mut self,
        budget: DecodeBatchBudget,
    ) -> Result<DecodeStep, StreamFormatError>;
    fn resume_state(&self) -> Result<DecoderResumeState, StreamFormatError>;
}

pub enum DecodeStep {
    Batch(DecodedFragmentBatch),
    End(DecodeReceipt),
}

#[async_trait(?Send)]
pub trait FormatEventSink {
    async fn send(&mut self, event: FormatEvent) -> Result<(), StreamFormatError>;
}

pub enum FormatEvent {
    Fragment(StreamingSessionFragment),
    SessionFrontier(SessionWatermark),
}
```

The decoder is pulled one bounded batch at a time. A blocked downstream stage
simply stops pulling it. Each `DecodedFragmentBatch` carries a typed
`resume_after` cursor for in-process accounting. The stable host-owned
`StreamingDecodeStage` owns whichever dynamic partition decoder is active and
implements `StreamingCheckpointParticipant`; at a barrier it binds that cursor
plus `StreamingPartitionDecoder::resume_state` to immutable partition identity and the
format semantic digest. Dynamic partition decoders are never added to the
frozen participant inventory. No format event independently claims checkpoint
publication. A resumability granularity is valid only when the stage participant
round-trips after process restart. `Pending` is represented by the future, and
only `DecodeStep::End` exhausts the partition.

Format-private typed rows stay behind the implementation. The public output is
a host-owned canonical session fragment or session-frontier contribution. No
`Any` downcast is needed between independently selected source and format
implementations.

### Canonical session fragments

The decoder boundary is session-addressed rather than complete-conversation
addressed. Generation 1 defines this versioned host-owned vocabulary:

```rust
pub struct StreamingSessionFragment {
    pub record_id: StableRecordId,
    pub session_key: StableSessionKey,
    pub source_position: SourcePosition,
    pub source_partition: ImmutableObjectIdentity,
    pub event_time: Option<EventTimeUtc>,
    pub stable_tie_break: StableOrderKey,
    pub predecessors: SmallVec<[StableRecordId; 2]>,
    pub mutation: SessionMutationV1,
    pub provenance: UnitProvenance,
    pub lease: SessionFragmentLease,
}

pub enum SessionMutationV1 {
    ConversationTurn(ConversationTurnFragment),
    AgentEvent(AgentEventFragment),
    GraphNode(GraphNodeFragment),
    GraphEdge(GraphEdgeFragment),
    SessionClose(SessionCloseFragment),
}
```

The vocabulary is host-owned so selected formats and consumers do not exchange
`Any` values or format-private downcasts. New mutation families require an
explicit schema version. Conversation fragments carry endpoint-neutral turn
data; agent and graph fragments carry stable causal identities but no
executable tools or endpoint requests.

The fragment deliberately does not include a transport request. A
`SessionFragmentLease` retains batch-local bytes until the session coordinator has
incorporated or durably spilled the mutation. The coordinator later emits an
`ExecutableDatasetAction` only when its predecessors and timing gates are
ready. Conversation actions use a new bounded `StreamingConversationState` and
the existing `RequestMaterializer`; the immutable-dataset-backed
`ConversationSession` remains finite-only. Agentic and graph actions use their
corresponding streaming-capable workload binding.

### Stable identity derivation

All constructors use domain-separated BLAKE3 over canonical length-delimited
bytes; they are host-owned rather than format-local string concatenation:

```text
physical_record_id = H("aiperf.stream.physical.v1",
  stream_identity, immutable_partition_generation, decoder_record_coordinate,
  format_semantic_digest)

stable_record_id = H("aiperf.stream.logical-record.v1",
  stream_semantic_namespace, validated_producer_record_key)
  OR physical_record_id when no producer key exists

stable_session_key = H("aiperf.stream.session.v1",
  stream_semantic_namespace, validated_producer_session_key)
  OR H("aiperf.stream.one-turn-session.v1", stable_record_id)

stable_action_id = H("aiperf.stream.action.v1",
  session_program_semantic_digest, stable_session_key,
  causative_stable_record_ids, action_kind, causal_action_ordinal)

attempt_id = H("aiperf.stream.attempt.v1",
  stable_action_id, run_incarnation_id, incarnation_local_attempt_ordinal)
```

A producer key is used only when the format descriptor defines its canonical
type and uniqueness scope. A cross-chunk session program requires a validated
producer session key; the one-turn fallback cannot join records. Formats pass
the producer key to the host constructor rather than hashing ad hoc. Repeated discovery of the same immutable partition
generation yields the same physical IDs. Overlapping objects deduplicate a
logical row only when they carry the same validated producer key and canonical
mutation digest; without that key they are distinct source records. The same
logical ID plus identical canonical digest is idempotent. The same ID plus
different content, session key, predecessors, or timing identity fails as
`logical_identity_conflict` with both provenance receipts.

Global sequence, arrival/discovery order, worker/cell ID, route ownership epoch,
and target attempt never participate in logical record/action identity. Their
own typed values remain in ordering, placement, and result provenance.

The checkpoint backend allocates a unique `run_incarnation_id` while acquiring
the fenced writer lease, before that process may issue actions. Resume acquires
a new incarnation, so a crash-before-checkpoint redelivery cannot reuse the
physical attempt ID even though its logical action ID is stable. Ephemeral mode
uses a process-unique random incarnation. Target idempotency always uses the
logical action ID, never the attempt ID.

Separately, a fresh replay invocation allocates one `logical_replay_run_id` and
commits it before first issue. It remains stable across process restarts but is
different for an independent replay of the same stream. Endpoint idempotency
keys derive from `(logical_replay_run_id, stable_action_id)`; an explicitly
authored idempotency namespace may intentionally join runs, and provenance
records that choice. Process incarnation remains attempt telemetry only.

## Session continuity

### Run-scoped session coordinator

One selected program factory constructs the coordinator that survives every
partition and decoder call:

```rust
pub trait StreamingSessionProgramFactory: Debug + Send + Sync {
    fn descriptor(&self) -> &'static StreamingSessionProgramDescriptor;
    fn validate(
        &self,
        authored: &RawValue,
        format: &StreamingFormatDescriptor,
        workload: &WorkloadDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingSessionProgramConfig>>;
    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingSessionProgramConfig>,
        context: &StreamingSessionPrepareContext,
    ) -> Result<Box<dyn StreamingSessionCoordinator>>;
}

#[async_trait(?Send)]
pub trait StreamingSessionCoordinator: StreamingCheckpointParticipant {
    async fn ingest(
        &mut self,
        fragment: StreamingSessionFragment,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError>;

    async fn advance_watermark(
        &mut self,
        watermark: SessionWatermark,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError>;

    async fn observe_execution(
        &mut self,
        event: ActionExecutionEvent,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError>;

    async fn seal(
        &mut self,
        seal: SourceSeal,
        output: &mut dyn DatasetActionSink,
    ) -> Result<SessionSealReceipt, SessionCoordinatorError>;

}

#[async_trait(?Send)]
pub trait DatasetActionSink {
    async fn send_action(
        &mut self,
        action: ExecutableDatasetAction,
    ) -> Result<(), SessionCoordinatorError>;
    async fn advance_causal_frontier(
        &mut self,
        frontier: SessionCausalFrontier,
    ) -> Result<(), SessionCoordinatorError>;
}
```

The stock `conversation` program extends canonical conversation state and emits
request/continuation actions. The stock `agent_graph` program retains recorded
agent/tool/graph causal state and emits graph actions without executing source
tools. A workload must declare the action schemas it accepts. This keeps graph
session support out of source/format switches and permits a dedicated streaming
graph executor without mutating the existing frozen `GraphInputBundle` path.

State is keyed by `(stream_identity, stable_session_key)`, never by source
partition. `DecodeStep::End` means only that those bytes are
exhausted. It MUST NOT close the session, flush incomplete state, create roots,
or discard request/tool/graph history. The next partition may immediately
extend any active session.

There are two order domains: per-session causal order and global replay event
order. A session action becomes globally reorderable only after its declared
predecessors are satisfied. A global watermark does not close a session unless
the format contract proves that it is also a hard session frontier.

The coordinator also emits a monotonic `SessionCausalFrontier`: a proof that no
currently buffered or future source fragment covered by the applicable hard
session/source frontier can reveal another action with event time at or below
that value. Its receipt binds the source/session frontiers and the exact set of
unresolved lower-time dependencies. The host advances `EventTimePolicy` only to
`min(source_event_watermark, session_causal_frontier)`. If an unresolved child
at time 100 depends on a future parent, a ready action at time 200 remains held
until the coordinator either releases the child, proves it impossible through
closure, or applies the authored lossy/fail policy. A source watermark alone
never lets hidden causal work be overtaken.

A session closes only through an explicit close mutation, an expected monotonic
per-session sequence proven complete, a session-scoped hard watermark, or
sealed-source validation. A session program that can span chunks MUST declare
at least one applicable closure proof during capability agreement. Otherwise it
must use an authored inactivity close or unresolved-state policy explicitly
labeled lossy; arbitrary follow-mode retention is refused when no finite state
bound can be proved. At seal, unresolved predecessors fail with stable
identities unless incomplete-session drop is explicitly selected.

The generation-1 `agent_graph` program is an append-only per-session state
machine. Node and edge identities are immutable; an identical duplicate is
idempotent and a conflicting duplicate fails. A node declares its complete
predecessor set before it can become ready. Edge addition after either endpoint
has executed is forbidden. Incremental cycle detection rejects an edge that
would reach its own source. Tool-result references are inert recorded data and
must bind a declared predecessor. Retry attempts are children of one stable
logical action, not new graph roots. A session-close proof freezes the graph;
remaining missing predecessors then fail or follow the authored incomplete
policy. The existing batch `dynamo_trace` compiler remains bit-compatible;
the separately versioned `streaming_dynamo_trace` format owns these stricter
duplicate, block-size, missing-parent, closure, and watermark semantics.

### Session-update policy

The session-program config selects one strict policy:

- `recorded_inputs` is the shadow-replay default. Source fragments contain the
  complete recorded request snapshot or deterministic recorded delta needed to
  materialize later actions. First-token/terminal events satisfy declared
  timing gates, but target response content cannot mutate canonical request or
  graph state. `EndpointSessionUpdate` content is rejected or retained only in
  an explicitly selected comparison projection.
- `target_closed_loop` applies normalized target response/tool output to later
  actions. It is a different workload fidelity mode and provenance says so.
  Restartable execution requires a checkpoint backend advertising
  `encrypted_sensitive_session_state`; normalized update bytes are stored in a
  dedicated encrypted participant object, while the generation record contains
  only ciphertext identity, schema, and digest. The envelope key comes from
  existing external secret authority and never enters config, logs, provenance,
  or checkpoint bytes. Without that capability, this policy is accepted only
  with checkpoint mode `none` and produces no resume claim.

Both policies checkpoint causal/timing receipts. Recorded mode reconstructs
content from immutable source identities plus its recorded-state cursor;
closed-loop mode restores and authenticates the encrypted participant before
source polling. Missing/wrong keys fail closed and never fall back to recorded
inputs.

### Checkpoint and cellular handoff

A checkpoint contains either the complete canonical/spilled state and causal
frontier for every active session through its decoded horizon, or a source
horizon before the first unrepresented fragment. Restore installs session state
and ownership before source polling resumes.

Cellular placement hashes the stable session key, not the source partition. A
session's execution route remains sticky to one cell, while generation 1 keeps
canonical session/graph state at the controller. Route migration is a fenced
transaction:

```text
freeze session sequence N
-> stop preparing actions for the old route epoch
-> drain/record or explicitly cancel old-cell actions <= N
-> commit controller session state + terminal receipts + old fence
-> new cell installs immutable request/content leases
-> commit new route epoch and owner at the controller
-> release prepared actions > N
```

Until acknowledgement commits, new fragments remain bounded at the controller;
concurrent dual ownership is forbidden.

Every session route carries a monotonically increasing ownership epoch/lease,
and every action, terminal update, and receipt echoes it. Migration fences the
old epoch, accounts for pending/unadmitted work, and drains or explicitly
cancels in-flight work before route handoff. No canonical session state moves
cell-to-cell. The durable global generation binds the old fence, controller
state version, new route owner, next session sequence, and idempotency/attempt
identities before the new route releases. Late old-cell receipts are rejected
by epoch. If either side or the controller fails mid-handoff, restore selects
the route named by the last committed global generation; new-cell prepared data
without that commit is unreachable staging, not ownership authority.

### Executable actions and sinks

The program output is host-owned and versioned:

```rust
pub struct ExecutableDatasetAction {
    pub action_id: StableActionId,
    pub session_key: StableSessionKey,
    pub predecessors: SmallVec<[StableActionId; 2]>,
    pub event_time: Option<EventTimeUtc>,
    pub stable_order: StableOrderKey,
    pub source_position: SourcePosition,
    pub provenance: UnitProvenance,
    pub payload: DatasetActionV1,
}

pub enum DatasetActionV1 {
    Request(SessionRequestAction),
    GraphNode(SessionGraphAction),
    SessionTerminal(SessionTerminalAction),
}

#[async_trait(?Send)]
pub trait StreamingActionSubmitter {
    fn accepted_schema(&self) -> DatasetActionSchema;
    async fn submit(
        &mut self,
        action: OrderedDatasetAction,
    ) -> Result<SubmittedAction, ActionExecutionError>;
}

#[async_trait(?Send)]
pub trait StreamingActionDriver: StreamingCheckpointParticipant {
    async fn next_event(&mut self) -> Result<ActionExecutionEvent, ActionExecutionError>;
    async fn drain(&mut self) -> Result<ActionDrainReceipt, ActionExecutionError>;
}

#[async_trait(?Send)]
pub trait StreamingActionDriverControl {
    fn stop_issuing(&self);
    fn cancel_pending(&self);
    async fn cancel_inflight(&self) -> Result<ActionCancelReceipt, ActionExecutionError>;
}

pub struct SubmittedAction {
    pub handle_id: ActionHandleId,
    pub control: ActionExecutionControl,
}

pub struct PreparedStreamingActionBinding {
    pub submitter: Box<dyn StreamingActionSubmitter>,
    pub driver: Box<dyn StreamingActionDriver>,
    pub control: Box<dyn StreamingActionDriverControl>,
}

pub enum ActionExecutionEvent {
    Admitted(ActionAdmissionReceipt),
    FirstToken(ActionFirstTokenReceipt),
    SessionUpdate(EndpointSessionUpdate),
    Terminal(ActionTerminalReceipt),
}

pub trait StreamingActionSinkFactory: Debug + Send + Sync {
    fn descriptor(&self) -> &'static StreamingActionSinkDescriptor;
    fn validate_binding(
        &self,
        authored: &RawValue,
        action: &DatasetActionSchema,
        transport: &TransportDescriptor,
        endpoint: &EndpointDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingActionSinkConfig>>;
    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingActionSinkConfig>,
        context: &StreamingActionSinkPrepareContext,
    ) -> Result<PreparedStreamingActionBinding>;
}
```

The shadow-replay workload prepares the sink set admitted by the selected
session program:

- `ScheduledRequestActionSink` lowers a controller-owned session request through
  `RequestMaterializer` and issues through `ScheduledRuntime`/`TurnDispatcher`.
- `StreamingGraphActionSink` owns bounded per-session incremental graph
  execution, reusing `Clock`, `GraphSink`/dispatcher, observation, placement,
  and failure policy. It does not append to or reinterpret an existing
  `GraphInputBundle`; the selected session program has already validated causal
  readiness for each graph action.
- `SessionStateActionSink` is a host built-in for `SessionTerminal`. It performs
  no endpoint dispatch: the submitter allocates one bounded slab entry and its
  driver emits deterministic admitted then terminal state-only events after the
  coordinator proves all declared session actions closed. It contributes only
  session-result membership and participates in the same lifecycle/barrier
  rules.
- Recorded tool/agent events update session state and release dependent graph
  actions. Source tools are never executed merely because their records arrive.

Each prepared binding has one type-erased submitter and one multiplexed driver,
not one boxed trait object per request. The binding owns a bounded worker-local
slab of active executions. `submit` returns only a compact handle ID and control;
the host registers those values in `ActiveExecutionSet`, while the driver emits
events for every slab entry through `next_event`. This avoids a required
per-action heap/vtable allocation and lets submission proceed while the separate
driver has a pending event future. It feeds first-token, terminal, and
endpoint-derived session updates back into the sole session coordinator.
Only that coordinator mutates canonical conversation/graph state or releases
dependent actions. `Terminal` is the final event and exhausts the handle; every
submitted action produces exactly one terminal/drop/cancel receipt. At a
checkpoint barrier, `ActiveExecutionSet` fences `H`, joins/exhausts all handles
at or below it, and proves that no later execution event for those identities
can enter session state or results. The driver participant captures slab and
admission/backend state required by nonterminal delivery modes; the host set
captures logical membership, controls, event ordinals, and terminal coverage.

`ActionExecutionControl` is separately cloneable and wakes a pending event
future, so phase cancellation never requires borrowing the event stream it is
canceling. Phase hooks synchronously fence issuance/cancel pending work, then
use the async in-flight hook in the existing order. `ActiveExecutionSet` drives
every action control, while the separately borrowable binding control wakes a
pending driver event future. The driver joins every slab entry before `drain`
returns. The
scheduled driver delegates those operations to `ScheduledRuntime`; the graph driver delegates
to its bounded incremental executor. The prepared workload never downcasts a
sink or relies on its destructor to cancel active endpoint work.

Each execution event binds `(action_id, attempt_id, ownership_epoch,
event_ordinal)`. Ordinals are strictly increasing per attempt; admitted precedes
first-token/session updates, and terminal is unique and final. An identical
duplicate is idempotent, a conflicting duplicate or post-terminal event fails,
and a stale ownership epoch is rejected. `observe_execution` records the event
and emits newly causally ready actions through the same bounded output seam.

The workload descriptor declares accepted action schemas, and capability
agreement fails before source effects if a session program can emit an action
for which no prepared sink exists. This makes multi-turn/agentic/graph support a
real execution boundary rather than an opaque payload the first implementation
cannot consume.

## Initial format implementations

### Baseten format

The streaming Baseten implementation reuses the native projected
Parquet/Arrow batch decoder and request/timing semantics. It changes the outer
lifetime:

- decoded batches flow directly into the run-scoped session coordinator;
- no complete `Vec<RawRow>` or complete resident `Dataset` is constructed;
- prompt bytes and segments are allocated only for retained units;
- strict open-loop mode emits each row as a one-turn session fragment
  immediately behind ordering policy;
- session-preserving mode requires provable closure or uses a bounded external
  spill/sort plan; and
- global idle-gap reflow and minimum-time normalization are explicit policies
  whose need for a preliminary scan or bounded lookahead is declared before
  execution.

For a sealed finite HF snapshot, exact compatibility mode MAY perform two
bounded passes over locally cached shards: a metadata pass that builds a
disk-backed index/sort and a replay pass that emits units. This can delay first
request but keeps memory independent of rows and preserves current grouping and
global timing. A one-pass low-latency mode MUST declare the ordering/session
guarantees it relies on and fail when the source does not provide them.

Arrow IPC is enabled only when the reader can inspect message/buffer lengths and
reject a record batch whose projected allocation would exceed the remaining
decode-byte budget before allocating it. If the selected Arrow dependency
cannot provide that pre-allocation guard, generation 1 refuses Arrow IPC and
ships Parquet-only Baseten streaming. “Bounded by the largest authored batch”
does not satisfy the configured memory invariant.

### Dynamo/NVCF request-trace format

Generation 1 accepts exactly `dynamo.request.trace.v1`, using a separately
registered `streaming_dynamo_trace` format ID and strict unknown-field/version
rejection. The current native field validation and request reconstruction are
extracted and reused only where their semantics are independent of complete
capture loading. The streaming format does not compile a complete
`GraphInputBundle` merely because the bytes use a Dynamo wire schema. Existing
finite `dynamo_trace` behavior is unchanged. NVCF production use requires its
published objects to conform to this exact schema; any different NVCF schema is
a new versioned format contract, not implementation-time inference.
Request-level shadow replay emits session-addressed turn fragments as soon as
they decode. The coordinator retains request history and causal state across
objects and emits executable actions when ordering allows. Missing parents
before an applicable hard session watermark are incomplete data, not roots.
Under `recorded_inputs`, source responses are retained as recorded-outcome/
reference metadata only when a defined evaluator or comparison consumer
requests them; they are not sent as input to the target. `target_closed_loop`
uses target-derived updates under its separate authored and encrypted-state
contract, never by implicitly substituting source response bytes.

## Stream events and ordering

There is one event path: `SourceEvent` enters the host processor and bounded
`FormatEvent` values leave decoding. There is no second aggregate stream enum
and no `Idle`. Source seal, authored row/time limit, cancellation, and policy
termination remain distinct typed terminal reasons propagated by the host.
Cancellation is not a source seal and cannot advance a resumable source cursor
past unprocessed data. Checkpoint barriers originate only at the host
coordinator and visit participants; formats do not emit them.

### Host-owned event processor

`StreamEventProcessor` is composition, not another plugin. It owns the typed
stage sequence:

```text
SourceEvent
  -> incremental decoder / FormatEvent
  -> session coordinator / causally-ready ExecutableDatasetAction
  -> event-time policy / OrderedDatasetAction
  -> near-horizon action sink / terminal receipt
```

Source frontiers first update the decoder. Decoder-produced session frontiers
then update the session coordinator. Only causally ready actions enter the
global event-time reorder policy. A barrier propagates in that same order and
collects one typed checkpoint participant from the source cursor, active
decoder, session coordinator, reorder policy, action sinks, placement, and
results plane. `End` seals each stage in order; it never bypasses unresolved
session state. This host owner is also the sole allocator of global sequence and
the sole caller of the atomic checkpoint backend, preventing either function
from hiding inside an adapter.

### Stable order

Replay order is:

```text
(event_time, stable_tie_break, source_partition_identity, source_record_ordinal)
```

The host assigns a dense `global_sequence` only after an action is causally
ready and this key is safe behind the watermark. An unresolved fragment
therefore occupies session/decoder checkpoint state, not a hole in the terminal
action range. Each dispatching action maps to exactly one terminal receipt;
`SessionTerminal` maps to one state-only terminal receipt after all declared
session actions close. The sequence is the placement and checkpoint order but
does not replace source identity. Authored predecessor-lateness/state budgets
must eventually spill, fail, or explicitly drop unresolved state; backpressure
alone cannot resolve a parent that exists only in a later partition.

### Watermark policies

The event-time policy is an injected host-owned trait:

```rust
pub trait EventTimePolicy: StreamingCheckpointParticipant {
    fn observe(&mut self, action: &ExecutableDatasetAction)
        -> Result<Vec<OrderedDatasetAction>, EventTimeError>;
    fn advance(&mut self, frontier: EventTimeWatermark)
        -> Result<Vec<OrderedDatasetAction>, EventTimeError>;
    fn seal(&mut self) -> Result<Vec<OrderedDatasetAction>, EventTimeError>;
}
```

Supported policy shapes are:

- **hard manifest watermark**: producer manifests declare complete event-time
  intervals;
- **bounded disorder**: `max_seen_event_time - max_out_of_order`, explicitly
  estimated and paired with late policy;
- **sealed external sort**: finite input is externally sorted before emission;
- **source order**: event time is unused and units retain deterministic source
  order; and
- **as ready**: allowed for non-fidelity ingestion, never described as
  event-time preserving.

Object publication cadence is not itself a watermark. For NVCF, the strongest
design is a producer-authored manifest per interval containing object identities,
min/max event time, record count, and a sealed-through timestamp. If unavailable,
AIPerf must use bounded-disorder policy and surface that replay ordering is
estimated.

## Event time and replay

### Time mappings

`ReplayTimeMapping` is independent of source and format:

- `wall_clock_delay`: target UTC = source event UTC + authored delay;
- `relative`: subtract the stream's selected origin and start at phase/run
  monotonic zero;
- `as_ready`: issue after decode/order without an authored timestamp target.

For wall-clock delay, startup captures one immutable mapping:

```text
anchor_utc
anchor_monotonic_ns
target_monotonic_ns = anchor_monotonic_ns
                    + (event_utc + replay_delay - anchor_utc)
```

The real-clock implementation obtains the pair from one clock-authority method
that brackets one UTC read with monotonic reads and rejects excessive sampling
uncertainty; tests inject the pair directly. The pair and uncertainty are
persisted in run provenance and never resampled after a wall-clock adjustment.
Virtual clocks accept only an explicitly authored UTC epoch.

All arithmetic is checked signed integer nanoseconds. Non-finite, out-of-range,
or missing required timestamps fail before the unit is scheduled.

The configured delay SHOULD exceed the high percentile of publication cadence,
object visibility delay, acquisition, decode, and ordering lookahead. AIPerf
reports observed headroom; it does not pretend that `5m` is safe merely because
objects are usually published every five minutes.

### Near-horizon scheduling

Ordered actions remain in a bounded replay buffer. The workload admits an action to
`ScheduledRuntime` only when:

1. its per-session predecessors are satisfied and it is safe behind the
   event-time watermark;
2. its target is within `schedule_horizon` of `Clock::now_ns()`;
3. admission and byte budgets are available; and
4. stop policy allows a new session.

The workload may call `wait_until_or_stop` itself and issue immediately at the
target, avoiding one delayed task per buffered record. Alternatively it may
hand near-horizon work to `ClockTaskScheduler`. Either shape preserves the
bounded-task invariant.

### Late data

Late policy is authored from:

- `fail`: terminate with the first late-record receipt;
- `issue_immediately`: preserve the record but report source and scheduling
  lateness;
- `drop`: omit it with a stable reason and count;
- `bounded_catch_up`: issue without additional wait while accumulated lateness
  remains below a finite bound, then fail or drop as separately configured.

No default silently rewrites timestamps or compresses gaps. Generation 1 shadow
replay defaults to `fail` for hard-watermark violations and
`issue_immediately` for records whose target passed because acquisition/target
execution was slow, with both policies authorable independently.

## Backpressure and resource model

### Blocking execution owner

Preparation injects one cheaply cloneable, host-owned
`StreamingBlockingExecutor` handle into sources, formats, state/checkpoint
backends, and the final compactor. It is a concrete service, not a named plugin,
an object-safe erased callback, or an adapter-owned thread pool:

```rust
impl StreamingBlockingExecutor {
    pub async fn run<T, F>(
        &self,
        class: BlockingWorkClass,
        budget: BlockingWorkBudget,
        work: F,
    ) -> Result<BudgetedBlockingOutput<T>, BlockingWorkError>
    where
        F: FnOnce(BlockingCancellation) -> Result<T, BlockingWorkError>
            + Send
            + 'static,
        T: Send + 'static;

    pub async fn cancel_and_join(&self) -> Result<(), BlockingWorkError>;
}
```

The implementation acquires item and input/output-byte permits before enqueue,
then uses a fixed host-owned blocking pool (or bounded permits in front of
`spawn_blocking`) so Tokio's blocking queue is never the capacity authority.
Long decode/sort/compaction work checks a cooperative cancellation token at
bounded intervals. `BudgetedBlockingOutput<T>` owns the output-byte permit until
the typed value is consumed or dropped; there is no `Any` downcast or unaccounted
return buffer. Results return through bounded owned values. Phase shutdown
stops submission, signals cancellation, joins all accepted work, and only then
drops source/state/checkpoint leases. No adapter starts its own untracked pool.
The checkpoint participant set includes the blocking owner whenever prepared
work contains durable derived state needed for resume.

### Budget hierarchy

One host-owned `StreamingResourceBudget` grants permits for:

- discovered but unacquired partitions and bytes;
- concurrent acquisitions and local cached bytes;
- decoded batch items and bytes;
- format/session state memory and spill bytes;
- reorder items and bytes;
- ready/scheduled items and bytes;
- active action-execution handles and terminal-processing items/bytes;
- in-flight requests/conversations; and
- completed-but-provisional facts above the terminal cut;
- prepared checkpoint transaction, participant, result-segment, and index bytes;
- final-compaction input/output bytes; and
- cellular unacknowledged chunks and bytes.

Each object that owns bytes retains the relevant permit. Moving an object moves
the permit; cloning payload storage does not mint capacity. Diagnostics expose
current and high-water usage per category.

### Required scheduled-runtime adaptation

The streaming request sink cannot use the current unbounded terminal-processor
path unchanged: `ScheduledRuntime` retains one `JoinHandle` per detached record
processor and one `session_numbers` entry per unique session for the run. Before
the sink is enabled, shared runtime code gains a `BoundedTerminalProcessorLane`
with item/byte permits and a fixed worker-local drain task (or inline
credit-dispatch processing), plus terminal cleanup of session-number state.
Streaming session numbering is derived from the stable session identity rather
than the lifetime size of a map. Finite behavior remains covered by existing
tests, but the follow-mode soak gate rejects monotonic task-handle, processor
queue, session-number, provisional-result, or prepared-index growth. Merely
calling `wait_record_processors` at run end does not satisfy boundedness.

Likewise, a perpetual run does not feed one lifetime-retaining
`NativeMetricsObserver`. A worker-local `EpochMetricsObserver` retains only
bounded in-flight slots, removes each slot at terminal, and folds the completed
fact into the current mergeable `MetricsAccumulator`. At a barrier it rotates
that accumulator plus exact-record projections into an immutable epoch
partition and starts an empty one. Completed facts above a terminal hole rotate
into bounded provisional partitions with membership roots and prepare leases;
they remain unreachable from committed results until the hole closes. Exhausting
the provisional/prepare budget fences new admission or applies the authored
overload policy. Processor errors are surfaced promptly to phase execution,
not accumulated as an unbounded vector of strings until shutdown.

### Overload policies

- `backpressure`: stop pulling upstream; replay delay grows and is measured.
- `fail`: terminate when a configured occupancy or lateness threshold is
  exceeded.
- `drop_newest`, `drop_oldest`, or `sample`: permitted only with explicit
  configuration and deterministic identity-based selection.

Backpressure does not mean unlimited disk. When spill/cache is full and no
downstream progress can free it, the source stalls or the authored overload
policy applies.

### Stateful assembly and spill

Session programs/coordinators and formats receive a process-owned
`StreamingStateStore` capability rather than a path:

```rust
pub trait StreamingStateStore {
    fn namespace(&self, owner: &StreamStateOwner) -> Result<Box<dyn StreamStateNamespace>>;
}
```

Namespaces provide checked put/get/remove, ordered iteration where required,
and byte accounting. The initial implementation may use bounded sorted run
files plus k-way merge; selecting an embedded database is an implementation
decision only after dependency and performance review. Implementations cannot
bypass the store to create unaccounted scratch trees.

Memory-only session assembly is allowed when a hard watermark or explicit end
marker bounds active keys. Arbitrary finite external sort consumes disk budget
and produces an immutable derived-run digest. Exhaustion is a typed
`state_budget_exceeded` failure, not OOM.

### Segment lifetime

Streaming formats create batch-scoped `SegmentStore` arenas or a reclaimable
streaming segment store. A fragment holds an opaque lease and transfers it into
session state or spill. Admission clones the resulting lease into every
continuation. Terminal completion releases it only after the last session/action
no longer needs materialization or raw capture. Deduplication may share segments
across live batches through weak/content-addressed entries, but unreferenced
entries are reclaimable.

## Shadow replay workload

`shadow_replay` consumes each causally ready conversation/request action once,
maps its event time, materializes it through the transport-selected
`RequestMaterializer`, and issues through `ScheduledRuntime`. A run-scoped
session table retains conversation history and affinity across every source
chunk. Agentic/graph actions route to the prepared incremental graph action
sink. Validation fails only when the selected workload distribution lacks a sink
for an action schema; it never flattens graph actions into independent requests.

It owns:

- event-time policy and stable sequence assignment;
- replay time mapping;
- admission/late/overload policy;
- session affinity and continuation release;
- action-to-terminal acknowledgement mapping;
- source-versus-target fidelity metrics; and
- phase participation.

It does not own:

- S3/HF clients or credentials;
- Parquet/JSONL/Dynamo/Baseten parsing;
- HTTP/gRPC/Dynamo frontend wire formatting;
- endpoint response parsing;
- ordinary request measurement;
- process signal handling; or
- report/export implementation.

### Continuations

Strict row replay uses one one-turn session per source record. Session-preserving
replay uses the existing continuation callbacks and retains source session
affinity. Open-loop continuation target is the maximum of its recorded target
and causal availability from the prior terminal/first-token policy, matching the
selected existing replay semantics. Closed-loop think time is measured through
`Clock` after prior completion.

Fixed-delay fidelity is therefore best effort subject to declared causal gates,
not a promise that an impossible child deadline is met. Every action records
`recorded_target`, `causal_release`, `actual_issue`, prerequisite kind
(`first_token`, `terminal`, graph/tool result), and one classified miss reason.
Causal waiting is distinct from source publication lag, reorder lateness,
scheduler slip, and admission delay.

### Phase behavior

The stream runtime is run-scoped; phase executions acquire explicit views:

- `continue`: consume the next unit after the prior phase's committed horizon;
- `rewind`: finite, seekable streams only, to a validated checkpoint;
- `separate`: use another named stream resource.

Generation 1 live shadow replay accepts one profiling phase. Warmup must use a
separate finite stream or endpoint readiness path; it cannot silently consume
and discard live production records before profiling. The run-scoped seam keeps
multi-phase support possible without reopening a follow prefix per phase.

### Phase lifecycle mapping

| Phase hook | Streaming behavior |
|---|---|
| `configure` | Apply phase admission/byte limits; validate phase stream-view policy |
| `setup` | Open source from validated checkpoint, acquire state/cache leases, start bounded pipeline, establish initial snapshot/frontier |
| `start_ramps` | Existing behavior; no stream-specific ramp owner |
| `execute` | Pull ordered near-horizon actions and issue through `ScheduledRuntime` until seal, stop, or policy termination |
| `stop_issuing` | Atomically close admission and wake source/format/reorder waits |
| `cancel_pending` | Cancel unadmitted near-horizon tasks; preserve typed dropped/pending receipts |
| `cancel_inflight` | Use existing transport cancellation; no source-specific request cancellation |
| `release_stuck_slots` | Release only host-accounted permits whose owning task cannot return |
| `stop_ramps` | Existing behavior |
| `finalize` | Drain bounded stage tasks, flush allowed checkpoint horizon, seal metrics/provenance, remove scratch by RAII |

Source acquisition failure propagates through phase execution. It is not a
best-effort sidecar warning.

Shutdown first fences new admission, then joins source/decode/reorder owners
before tearing down state leases. If grace expires before a new
terminal-contiguous generation can commit, the prior committed generation
remains the resume point and the report declares the resulting at-least-once
window. Cancelled actions contribute terminal cancellation receipts only when
their input/session participant state is included in the same generation;
shutdown never advances a cursor merely to make finalization succeed.

## Checkpoint and delivery semantics

### Checkpoint participants

Checkpointability is a host contract, not an implied property of a state-store
namespace. Every stateful stage implements a narrow participant seam:

```rust
#[async_trait(?Send)]
pub trait StreamingCheckpointParticipant {
    fn participant_id(&self) -> CheckpointParticipantId;
    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError>;
    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError>;
}
```

Prepared participant state is immutable, digest-addressed, schema-versioned,
and names the greatest horizon it represents. The frozen plan names stable
stage owners: source, streaming format, decode stage, session coordinator,
event-time/reorder policy, action drivers, active-execution set, placement
policy, placement driver, active-placement set, terminal/result stage, and
blocking owner when it has resumable derived work.
Those owners aggregate dynamic partition decoders, action handles, worker
accumulators, and prepared objects beneath stable participant IDs. The
coordinator verifies the exact required set before commit and restores all of
it before polling resumes. `initialize(None)` establishes fresh state;
`initialize(Some(...))` restores the exact committed state and may be called
only once. A prepared source `open` constructs handles/control but does not poll
or resolve a mutable snapshot before this initialization. `StreamingStateStore` is storage beneath these
participants, never proof that a typed snapshot exists.

### Typed horizons

One checkpoint record contains separate monotonically contiguous horizons:

```text
discovered -> acquired -> decoded -> ordered -> admitted -> terminal
```

The host-owned cut never collapses those cursor types:

```rust
pub struct CheckpointCut {
    pub discovered: DiscoveryHorizon,
    pub acquired: AcquisitionHorizon,
    pub decoded: DecodeHorizon,
    pub ordered: OrderedActionHorizon,
    pub admitted: AdmissionHorizon,
    pub terminal: TerminalActionHorizon,
    pub event_watermark: EventTimeWatermark,
    pub causal_frontier: SessionCausalFrontier,
}
```

Each horizon binds stream-resource digest, source snapshot/generation,
partition/record cursor, format semantic digest, policy digest, and global
sequence where assigned. Holes are retained as bounded exception sets until
they close; a later item cannot advance a contiguous horizon over a missing
earlier item.

The event-time watermark is recorded beside these horizons but never used as a
source cursor.

Every resumable generation also declares how its exact immutable input can be
recovered. It either retains acquired objects/derived decoder state through the
persisted horizon under accounted source-cache leases, or records identities
that the source can conditionally reacquire and verify. A transient cache lease
alone cannot justify advancing a durable cursor. If a pinned HF shard or S3
version is later unavailable, restore fails as `source_unavailable_on_resume`;
it never substitutes current bytes or fabricates decoder/session state.

### Checkpoint backend

```rust
#[async_trait(?Send)]
pub trait StreamingCheckpointBackend {
    async fn open_latest(&self, run: &StreamRunIdentity)
        -> Result<Option<Box<dyn LeasedGenerationReader>>, CheckpointError>;
    async fn begin_generation(
        &self,
        expected: Option<CheckpointGeneration>,
    ) -> Result<Box<dyn StreamingGenerationTransaction>, CheckpointError>;
}

#[async_trait(?Send)]
pub trait LeasedGenerationReader {
    fn generation(&self) -> &CommittedCheckpointGeneration;
    async fn scan_result_index(
        &self,
        after: Option<ResultIndexCursor>,
        budget: ResultIndexReadBudget,
    ) -> Result<ResultIndexPage, CheckpointError>;
    async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError>;
    async fn read_participant(
        &self,
        descriptor: &ParticipantStateDescriptor,
    ) -> Result<CommittedParticipantState, CheckpointError>;
}

#[async_trait(?Send)]
pub trait StreamingGenerationTransaction {
    async fn stage_participant(
        &mut self,
        state: PreparedParticipantState,
    ) -> Result<(), CheckpointError>;
    async fn stage_results(
        &mut self,
        partitions: Vec<ResultPartition>,
    ) -> Result<PreparedResultEpoch, CheckpointError>;
    async fn commit(
        self: Box<Self>,
        metadata: CheckpointCommitMetadata,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError>;
}
```

One transaction stages typed participant state and result segments, then
constructs and publishes the generation from host-validated commit metadata;
the caller cannot supply a mismatched result root, and types do not expose an
independent result commit.
The committed generation is the single authority for resume state and result
inventory. A stale or divergent writer fails; it does not merge horizons by
max. The selected backend must prove conditional pointer update and durable
immutable-object semantics during capability agreement. A backend without that
primitive is restricted to checkpoint mode `none`.

`open_latest` atomically resolves the current root and acquires its renewable
generation lease before returning metadata. Every participant/segment/index
read occurs through that leased reader. Dropping it releases the lease; lease
renewal failure makes further reads fail before GC can reclaim an object in use.
Restore and compaction never perform a load-then-unleased-read sequence.
`scan_result_index` verifies every traversed index-block digest and returns a
bounded item/byte page in canonical merge order plus an opaque next cursor;
partial readers and the compactor never materialize the cumulative descriptor
inventory.

## Checkpoint-based results

### Purpose and ownership

A perpetual or multi-hour replay cannot retain its complete authoritative
report and exact record set until shutdown. The host therefore adds a
checkpoint result plane below post-run exporters. It periodically rotates
completed request facts and mergeable metric state into immutable result
segments, then atomically commits those segments beside the input/session
checkpoint.

`Exporter` remains a post-report presentation/upload seam. Exporters do not own
checkpoint barriers, worker coordination, deduplication, or durable progress.
At finalization they receive the deterministic compacted `NativeReport` and any
host-owned projections requested by the validated capture plan. This preserves
one authoritative metrics path and prevents an OTLP/MLflow/W&B/file sink from
defining restart semantics.

Checkpoint state and result bytes are staged through one selected backend
transaction. A backend cannot combine one checkpoint head with a different
result namespace and still claim an atomic generation:

```rust
pub trait StreamingCheckpointBackendFactory: Debug + Send + Sync {
    fn descriptor(&self) -> &'static StreamingCheckpointBackendDescriptor;
    fn validate(
        &self,
        authored: &RawValue,
        requirements: &CheckpointBackendRequirements,
    ) -> Result<Box<dyn ValidatedCheckpointBackendConfig>>;
    fn prepare(
        &self,
        config: Box<dyn ValidatedCheckpointBackendConfig>,
        context: &CheckpointBackendPrepareContext,
    ) -> Result<Box<dyn StreamingCheckpointBackend>>;
}
```

The descriptor and conformance suite prove one commit namespace and atomic
generation mechanism. The `none` backend is valid only with checkpoint mode
`none` and disables durable partial results; it cannot satisfy a restartable
run by silently keeping state in memory.

The current `RecordArtifactLane` is useful evidence for per-terminal streaming,
but its held-open monolithic JSON/CSV/outputs/Parquet files are not themselves a
checkpoint contract: flush does not provide epoch identity, deduplication,
atomic cross-file publication, or restart truncation. Generation 1 writes
checkpoint-native internal segments and produces legacy monolithic artifacts by
deterministic final compaction.

### Result capture plan

Run validation derives one host-owned `CheckpointResultPlan` from metrics mode,
artifacts, and exporter requirements:

```rust
pub struct CheckpointResultPlan {
    pub metrics: MetricsCheckpointProjection,
    pub exact_records: Option<ExactRecordProjection>,
    pub raw_records: Option<RawRecordProjection>,
    pub session_results: SessionResultProjection,
    pub provenance: StreamingProvenanceProjection,
    pub interval: CheckpointInterval,
    pub durability: CheckpointDurability,
}
```

- A mergeable metrics partition is always present.
- Exact/native/raw rows are present only when existing artifact/exporter policy
  requires them; a sketch run does not silently begin retaining exact records.
- Session results include stable session key, causal/close status, action range,
  and terminal outcome needed to explain partial active sessions.
- Provenance segments contain immutable partition/source receipts, acquired-byte
  digests, decoder/schema identities, cursor transitions, and policy events.
  They use the same bounded content-addressed index and never accumulate in a
  participant or report vector.
- Every projection has a versioned schema and canonical ordering.

The result plane consumes terminal `RecordIngest`/captured-record facts through
the existing worker-local observation/record-processor boundary. It does not add
a callback to the response-token path. Epoch accumulators are worker local and
merge only at the checkpoint boundary.

Existing `RecordIngest` does not carry streaming logical identity. The action
sink therefore installs one host-owned correlation envelope in the per-request
dispatch context:

```rust
pub struct StreamingRecordCorrelation {
    pub logical_action_id: StableActionId,
    pub attempt_id: ActionAttemptId,
    pub global_sequence: GlobalSequence,
    pub ownership_epoch: SessionOwnershipEpoch,
    pub membership: ResultMembership,
}
```

The terminal record processor joins this envelope with `RecordIngest` and emits
one `CorrelatedRecordIngest`; it does not infer identity from completion order or
`global_dispatch_index`. Request and endpoint-dispatching graph actions belong
to request-metric and optional exact/raw projections. A state-only
`SessionTerminal` belongs only to the session-result projection and terminal
coverage ledger. Every projection declares its accepted membership kinds, and
metric/record membership roots are built from logical action IDs, never source
record IDs or attempt IDs.

### Barrier and epoch selection

The checkpoint coordinator periodically requests a barrier by time, terminal
record count, byte count, or explicit control request. It selects the greatest
contiguous terminal global sequence `H` for which:

1. every action at or below `H` has one terminal/drop/cancel receipt;
2. every session mutation required to interpret those actions has a versioned
   checkpoint view at `H`;
3. no worker or cell can later emit another result for an identity at or below
   `H`; and
4. all result partitions through `H` can be deterministically closed.

`H` is the terminal action cut, not a substitute for source, acquisition,
decoder, ordering, or admission position. The same barrier carries the complete
`CheckpointCut`. If decoded progress is ahead of `H` in its own domain, the
session participant contains every unresolved fragment and emitted/pending
action through that decode cursor; otherwise decoding is rolled back before the
first unrepresented mutation. The action-sink fence proves that every handle at or below `H` is
terminal and exhausted, so no later first-token/session/terminal event can
change the committed cut.

Ingestion and decoding MAY continue into a bounded uncommitted overlay while
the barrier is prepared. Each stateful participant exposes a versioned view of
the full typed `CheckpointBarrier { cut: CheckpointCut, ... }` backed by the state store;
state after that component's cut remains in the live overlay and is not mistaken
for committed resume state. If any participant cannot provide that view, the
processor briefly backpressures source/decode while rotating the epoch.

A long-running earlier request can hold the authoritative horizon. Completed
results above the hole may feed explicitly provisional dashboards but MUST NOT
enter a committed result epoch. Cancellation/failure policy eventually closes
the hole. This favors restart correctness over falsely durable partial totals.

### Transaction contract

`StreamingGenerationTransaction::stage_results` writes immutable
content-addressed segments and returns their verified descriptors without
changing visible progress. The checkpoint coordinator stages every required
participant on the same transaction and calls `commit`, which atomically
publishes one canonical record containing:

- prior generation digest and next epoch number;
- all typed input/stage horizons and watermark;
- active-session checkpoint/state-store snapshot digest;
- pending placement/action state permitted by delivery mode;
- one bounded content-addressed result-index root;
- cumulative projection counts and byte totals;
- metrics/format/policy/placement semantic digests; and
- terminal/final presence plus reason.

The result-index root addresses immutable Merkle/LSM-style index blocks. Each
epoch adds bounded new blocks and structurally shares old ones, so a generation
record stays bounded while its root reaches every segment required for partial
and final compaction. It never copies the cumulative descriptor inventory.

The local backend takes a single-writer generation lease, compares the expected
`CURRENT` digest and epoch under that lease, writes immutable segments/index/
participant objects and `generation-N`, fsyncs each required file and parent
directory according to durability policy, atomically replaces the `CURRENT`
pointer, then fsyncs its parent. An object-store backend writes immutable
objects first and conditionally replaces one generation pointer using the prior
provider token. Readers trust only objects reachable from the committed root.

### Segment identity and ordering

One segment descriptor binds:

```text
run_identity + epoch + cell_id + worker_id + projection_id + schema_version
+ first_global_sequence + last_global_sequence + item_count + byte_length
+ canonical_membership_root + canonical_payload_digest
```

Segments contain facts sorted by `(global_sequence, logical_action_id)` and
bind a canonical membership representation: a disjoint interval set where the
placement plan proves disjoint intervals, otherwise sorted action IDs in
content-addressed membership blocks. Metric and record partitions bind the same
membership root. Min/max/count are diagnostics, not proof of coverage. A
retry producing the same bytes resolves to the same segment identity. The same
logical membership staged twice for one generation, or already reachable from
a committed root, with different bytes is a checkpoint conflict and fails. An
unreachable orphan from a crash-before-commit may coexist with a later retry's
different payload digest; it contributes nothing and is eventually collected.

`logical_action_id` is distinct from `attempt_id`. The logical result index has
one authoritative terminal result per action. With verified target idempotency,
the receipt is reused. Otherwise terminal-mode restart uses the first terminal
receipt made durable in a committed generation; later redelivery attempts are
optional attempt-telemetry records excluded from logical metrics. If no receipt
was committed before the crash, the first terminal retry becomes authoritative.
The report records this response-selection policy and its target-side duplicate
window.

Merge and compaction order is fixed by
`(epoch, cell_id, worker_id, projection_id, first_global_sequence, digest)`.
Floating-point accumulator merges use this order so restart/topology iteration
does not introduce arbitrary reduction order. Membership equality,
disjointness, and expected action coverage are checked before a partition
contributes; overlapping logical membership is rejected.

### Restart and crash behavior

1. A crash before all segments are durable leaves unreachable temporary/orphan
   data; the committed generation is unchanged.
2. A crash after segment preparation but before generation commit leaves
   content-addressed orphans that may be reused after digest verification or
   garbage-collected later.
3. A crash after generation commit resumes from that exact checkpoint and opens
   epoch `N+1`; epoch `N` is not emitted again.
4. A crash during final compaction does not change committed epochs. Compaction
   restarts from their immutable inventory.
5. Private transaction temporaries are not GC candidates. Prepared immutable
   objects carry a bounded renewable prepare lease; readers, restorers, and
   compactors lease generation roots. GC marks all committed roots and valid
   transaction/generation leases, waits an authored grace period, then removes
   only still-unreachable objects. Lease-renewal failure aborts publication or
   reading rather than racing deletion.

Generation 1 pins worker count, cell topology, placement digest, projection
plan, and action membership scheme across restart. A mismatch is refused.
Topology-independent logical repartitioning requires a later fenced migration
generation; physical `cell_id`/`worker_id` segment identity is never silently
reinterpreted.

Retention is four separate validated policies: minimum resumable generation
roots, partial-result history, final-compaction/index roots and produced final
artifacts, and source snapshot/cache material. Orphan/prepare TTL is separate.
Disk/object-store quotas account for index blocks, state spill, source leases,
prepared objects, committed segments, and compaction outputs. Deleting partial
history cannot delete the last restart root or final-compaction reachability.

### Partial and final results

`aiperf` and the operator results API may expose the latest committed generation
as a partial result with:

- `is_final: false`;
- checkpoint timestamp and generation/epoch;
- committed source/event/terminal horizons;
- cumulative merged metrics through `H`;
- active/incomplete session counts; and
- lag/late/source health at the barrier.

The partial view does not include provisional completions above `H` unless they
are clearly separated and excluded from authoritative totals.

Normal seal commits one `is_final: true`, terminal reason `completed` result
generation before compaction; here `is_final` means the checkpoint/result ledger
is sealed, not that a presentation artifact already exists. A deterministic
compactor reads that leased final root and reconstructs the ordinary
`NativeReport`, exact records, JSONL/CSV/Parquet/outputs, and exporter capture
projections. The prepared operation returns that report plus the existing
synchronous `PreparedReportCommit`, whose only streaming responsibility is to
release the compaction/report-retention lease after the process coordinator has
durably written and atomically renamed the report. It does not perform an async
checkpoint write.

If compaction fails after the final generation commit, execution returns
`PreparedRunFailure` with the final root as diagnostic evidence; no
`NativeReport` is fabricated, and compaction can be retried from the immutable
root. If coordinator report persistence fails, the sealed generation likewise
remains authoritative and reconstructable while the terminal envelope reports a
reporting failure. This matches the existing `PreparedRunnerOperation`/
`PreparedRunOutcome` ownership without changing its success/error sum type.

An execution `Err` still returns `PreparedRunFailure` and no `NativeReport`. If
phase cleanup can form a consistent typed cut, it may commit an `aborted`
terminal generation and attach its root receipt as a content-addressed
diagnostic artifact. That generation exposes partial metrics and is never
compacted into an ordinary final report. If the triggering failure prevents a
safe commit, the last partial generation remains authoritative and the failure
names it; no synthetic final generation is fabricated. User cancellation follows
the run's existing success/failure policy but the same report-commit ordering.
Byte identity with today's completion-order artifacts is not promised where
canonical global order is required, but record multiset and documented canonical
ordering are.

### Cellular results

Workers rotate local epoch partitions; cells merge or package them with cell ID,
sequence range, count, length, and digest. The controller accepts only the exact
partition set required by the barrier, uploads/stages immutable segments through
the bounded existing artifact route, and commits the global generation last.
Missing cells, gaps, duplicate ranges, mismatched plan digests, or conflicting
payloads block checkpoint publication.

A cell restart retransmits the same content-addressed segments. A controller
restart reconstructs state only from the last committed global generation; a
cell-local flush that was never included cannot affect authoritative results.

### Relationship to checkpoint delivery modes

Result epochs include only terminally classified actions even when durable
source/decode state is ahead of the terminal action cut. Every generation
records all typed horizons plus pending-action/session participant state. A
delivery mode controls which action state is authoritative on restart; it does
not erase the source/decoder distinction or require their cursor types to equal
`H`.

### Delivery modes

- `terminal` (default for replay): result authority advances only through
  contiguous terminal acknowledgements. Source/decode may advance in the same
  generation only when their complete session/pending-action state is captured;
  nonterminal actions are reissued after restart. Crash after target acceptance
  but before commit can replay a request, so semantics are at least once unless
  target idempotency is enabled.
- `admitted`: commit after successful local admission. Crash may lose in-flight
  effects, so semantics are at most once from AIPerf's perspective.
- `decoded` or `acquired`: useful for ingestion diagnostics, not accepted as
  faithful shadow-replay completion unless explicitly authored.
- `none`: ephemeral benchmark, no restart claim.

When target idempotency is configured, request keys derive from logical replay
run ID plus stable logical action ID and are injected through an endpoint-supported
field or header. The endpoint descriptor must declare support. AIPerf then reports
`idempotent_at_least_once_submission`; it still does not claim exactly once
without verified target semantics.

Checkpoint roots and default result artifacts contain no credentials, source
bearer tokens, raw prompts, or response bodies. Existing explicit raw-artifact
policy may authorize raw request/response projections, subject to its redaction
and retention controls. The other exception is an explicit
`target_closed_loop` sensitive-session participant, whose content is encrypted
and referenced as defined by the session-update policy; raw material never
appears in the generation record or an unauthorized projection.

## Cellular placement

Placement is a prepared host-owned policy plus multiplexed transfer/event driver
selected from validated topology and stream capabilities:

```rust
pub trait StreamingPlacementPolicy: StreamingCheckpointParticipant {
    fn place(&mut self, action: &OrderedDatasetAction)
        -> Result<PlacementDecision, PlacementError>;
}

#[async_trait(?Send)]
pub trait StreamingPlacementSubmitter {
    async fn prepare(
        &mut self,
        decision: PlacementDecision,
        action: OrderedDatasetAction,
    ) -> Result<PlacementHandle, PlacementError>;
    async fn release(&mut self, handle: PlacementHandleId)
        -> Result<(), PlacementError>;
}

#[async_trait(?Send)]
pub trait StreamingPlacementDriver: StreamingCheckpointParticipant {
    async fn next_event(&mut self) -> Result<PlacementEvent, PlacementError>;
    async fn drain(&mut self) -> Result<(), PlacementError>;
}

#[async_trait(?Send)]
pub trait StreamingPlacementControl {
    fn stop_preparing(&self);
    fn cancel_pending(&self);
    async fn cancel_inflight(&self) -> Result<(), PlacementError>;
}

pub enum PlacementEvent {
    Prepared(PlacementPreparedReceipt),
    Released(PlacementReleasedReceipt),
    Action(ActionExecutionEvent),
    Failed(PlacementFailureReceipt),
}
```

The local implementation is a bounded pass-through binding. The cellular
implementation owns bounded authenticated transfer/control frames and emits all
first-token, session-update, and terminal action events received from cells.
The host feeds `PlacementEvent::Action` through `ActiveExecutionSet` into the
controller session coordinator exactly as it does for local action drivers.
Policy, driver, active-placement set, and result receipt stage are stable
checkpoint participants; dynamic handles are aggregated beneath them. The
separate control handle wakes a pending driver future during phase shutdown.

### Centrally ordered live placement

For live shadow replay:

1. controller opens source and owns credentials;
2. controller decodes fragments and advances run-scoped session state;
3. controller establishes watermark order and assigns dense global sequence;
4. stable-session-affinity placement maps each action to its sole owning cell;
5. controller sends bounded, versioned chunks containing canonical action data,
   sequence, controller release policy/lookahead, lease/content references,
   count, byte length, and digest;
6. cell acknowledges receipt/admission/terminal and returns versioned
   endpoint-derived session updates through ordered receipts; and
7. controller advances checkpoint horizons only across contiguous verified
   receipts.

Every route has finite unacknowledged chunk/byte windows. Gaps, duplicates,
wrong cells, digest/length/count mismatch, and plan/policy digest mismatch fail
before checkpoint advance. Controller and cell use the same host-owned DTO;
wall-clock estimates are not compared between hosts to establish order.

Generation 1 makes the controller the canonical session-state owner. A cell is
an execution owner only: its terminal/session-update receipt carries action ID,
ownership epoch, prior session-state version, deterministic update, and terminal
facts. The controller linearizes that update before releasing dependent
actions. Cells cannot independently mutate durable causal state.

A controller monotonic deadline is not meaningful on a cell host. Generation 1
therefore uses two messages. `PrepareAction` transfers and validates the action
inside an authored lookahead but leaves it fenced. At the controller's target
instant, `ReleaseAction { global_sequence, ownership_epoch }` authorizes the
cell to issue immediately; the cell never interprets a controller timestamp.
Release-network delay is measured schedule slip, and an absent/stale release
cannot issue early. A later synchronized-clock protocol would be a new
capability. Source polling, backoff, checkpoints, cell release, and shutdown
waits all use their owning process's injected `Clock`; UTC appears only in the
persisted source-to-run anchor.

This protocol may reuse Velo/authenticated route machinery and global-push
credit concepts, but current fixed startup dataset pushes are not sufficient.

### Partition-local finite placement

For a pinned finite HF/local snapshot, the controller may assign immutable
shards to cells before execution only when session ownership remains correct.
Assignment binds snapshot digest, shard
identity, byte size/digest, decoder/policy digest, and deterministic global
ordering rule. A cell acquires only through explicitly provisioned scoped
authority or receives the exact shard through bounded artifact transfer.

Partition-local mode is rejected when one session can span shards assigned to
different cells unless a preliminary session index routes every fragment to the
same owner. It is also rejected when global session grouping, external sort, or
event-time order cannot be reproduced by a deterministic merge. A topology
must not change the selected record multiset. Report provenance records whether
order is globally exact or topology-deterministic.

## Provenance, metrics, and artifacts

### Provenance

The run records:

- source, format, session-program, action-sink, and checkpoint-backend factory
  IDs/descriptors/semantic config digests;
- finite snapshot identity or follow-source namespace identity;
- every admitted immutable partition identity and acquired-byte digest;
- HF commit/subset/split/shard inventory or S3 bucket/prefix/version identities
  with sensitive names redacted according to artifact policy;
- source/format/policy/checkpoint schema versions;
- watermark quality and late/overload/delivery policies;
- starting and terminal stage horizons;
- derived external-sort/spill digest when used; and
- placement topology and global-order guarantee.

The default artifact contains manifests and receipts, not raw dataset bytes.

For a perpetual source, `NativeReport` retains only the provenance index root,
cumulative receipt counts/bytes, semantic digests, and summarized flags. The
full ordered receipts remain a checkpoint-native indexed projection readable in
bounded pages and optionally compacted into `stream_provenance.jsonl` or
Parquet. Final compaction streams that projection; it never reconstructs a
resident vector of every partition receipt.

### Streaming-plane metrics

At minimum:

- discovery polls/events/reconciliations and publication-to-discovery lag;
- acquisition attempts, retries, bytes, throughput, cache hit/miss, and latency;
- decoded rows/units/bytes, malformed and filtered counts, and decode latency;
- active session keys, spill bytes, state evictions, and external-sort progress;
- watermark event time, quality, wall age, and advancement stalls;
- reorder/ready queue current/high-water items and bytes;
- target lead time, source lateness, schedule slip, admission delay, and catch-up;
- duplicate/gap/late/drop counts by stable reason;
- admitted, terminal, failed, and cancelled stage horizons;
- checkpoint latency/generation/failures; and
- cellular unacknowledged bytes/chunks and per-cell lag.

These metrics remain distinct from endpoint TTFT, latency, token, and error
metrics already emitted by `RequestObserver`.

## Failure policy

| Condition | Default | Authorable alternatives |
|---|---|---|
| Follow source temporarily empty | Remain pending | None that imply EOF |
| Finite source exact seal reached | Drain and finish | Repeat requires an explicit repeat policy |
| Source auth failure | Fail before replay | Bounded retry only when classified retryable |
| Listed object disappears/mutates | Fail source consistency | Skip only under explicit lossy policy |
| Checksum/digest mismatch | Fail | No silent retry from a different generation |
| Malformed record | Fail partition/run | Deterministic drop with count and origin |
| Hard watermark violation | Fail | None advertised as hard correctness |
| Record target already passed | Issue immediately and measure | Fail, drop, bounded catch-up |
| Queue/state memory full | Backpressure | Fail or explicit deterministic loss |
| Disk/cache/spill full | Backpressure, then fail if no progress is possible | Explicit deterministic loss where valid |
| Checkpoint CAS conflict | Fail checkpoint/run | None that merge by maximum |
| Target dispatch failure | Existing benchmark failure policy | Existing threshold/cancellation policies |
| Cell receipt gap/digest mismatch | Fail before horizon advance | Retransmit exact chunk within bounded window |
| Cancellation | Stop admission, cancel/drain through phase policy, checkpoint only allowed horizon | No fabricated source seal |

## Security and authority

- Source adapters receive narrow secret resolution and cache/state services,
  not the complete `RunContext` or raw artifact directory.
- Debug implementations and typed errors redact tokens, signed URLs, headers,
  and credential-provider internals.
- Local snapshots and spill/checkpoint trees are private, no-follow, and
  removed by RAII unless an explicit retention artifact is selected.
- S3 endpoint overrides and HF URLs are validated under existing proxy/TLS
  policy. Benchmark loopback proxy exclusions remain unchanged.
- Source data is untrusted. Strict schema/range/size limits apply before
  allocation proportional to authored values.
- Compressed inputs have compressed and expanded byte limits and reject bombs.
- Cellular data transfer binds plan, stream, partition, sequence, digest, and
  destination identity through the existing authenticated application layer.
- A stream factory cannot start detached tasks during validation/registration.
  All run-time tasks are owned and joined by the prepared operation.

## Implementation map

Suggested native modules:

```text
rust/runtime/src/streaming/
|-- mod.rs                 # public core vocabulary
|-- source.rs              # source factory/runtime/access traits
|-- format.rs              # format factory/decoder traits
|-- session.rs             # program factories, keyed causal state, handoff
|-- action.rs              # request/graph action vocabulary and sinks
|-- unit.rs                # fragments, actions, leases, stream events
|-- budget.rs              # count/byte permits and high-water facts
|-- state.rs               # bounded state/spill capability
|-- event_time.rs          # watermarks, stable reorder, time mapping
|-- checkpoint.rs          # stage horizons and stores
|-- results.rs             # epoch barriers, result/index segments, readers
|-- pipeline.rs            # owned bounded stage composition
`-- placement.rs           # local and cellular placement seam

rust/runtime/src/streaming/sources/
|-- local.rs
|-- hf_hub.rs
`-- s3.rs

rust/runtime/src/streaming/formats/
|-- baseten.rs
`-- dynamo.rs

rust/runtime/src/engine/
`-- streaming_execution.rs # shadow_replay factory/prepared operation
```

Required integration changes:

- `extensions/mod.rs`: source/format/session-program/action-sink/checkpoint-
  backend registries and registration methods;
- `engine/registry.rs`: dataset-stream resource requirement and frozen lookup;
- `engine/protocol_v2.rs`: strict named stream resources;
- CLI/config projection: user-facing source/format/replay options without
  source-format combination switches;
- `phase_runtime.rs`/timing: streaming phase execution adapter, not lifecycle
  duplication; extract the reusable `PhaseExecutionFactory`/`PhaseExecution`
  orchestration, `NativeMetricsObserver` construction, terminal record capture,
  and transport executor setup now embedded in scheduled/graph entrypoints;
- cellular protocol/controller/cell: incremental placement chunks and receipts;
- report/export: generic streaming provenance and metrics;
- record/metrics/result plane: epoch rotation, immutable segment schemas,
  atomic checkpoint-generation publication, partial readers, and final
  compaction;
- Baseten/HF modules: extract projected decoders/acquisition helpers while
  retaining current finite adapters.

`streaming_execution.rs` is a separate `PreparedRunnerOperation` assembled from
those lower-level services. It does not add a `NativeDatasetPlan` variant, call
the synchronous `GraphTraceSource`, or copy the closed `NativeRunSpec` dispatch
stack. Stage 0 first lands the reusable phase/capture construction seams with
finite-path conformance tests; only then does the streaming operation consume
them.

No Python module is invoked. The S3 adapter uses a native Rust SDK/client behind
the source trait. Dependency selection and feature ownership require a separate
implementation review, but do not change this architecture.

## Migration and delivery plan

### Stage 0: contracts and executable proof

- Add core stream vocabulary, source/format/session-program/action-sink/
  checkpoint-backend registries,
  in-memory fake source, conversation and agent-graph program fixtures, budget
  accounting, request/graph/session-state action sinks, and no-network dry-run
  consumers.
- Add an in-memory checkpoint backend and prove barrier/membership/dedup/final
  compaction equivalence before filesystem durability.
- Prove pending-versus-seal, backpressure, cancellation, and lease lifetime with
  `SimClock` and current-thread `LocalSet`.
- Prove one multi-turn/agentic/graph session across several partitions and a
  checkpoint restore before any real source adapter is added.
- Establish RSS, disk, and task-count instrumentation before real adapters.

### Stage 1: large finite HF/Baseten

- Implement pinned HF shard catalog and immutable local leases using native
  `hf-hub` acquisition.
- Extract Baseten projected batch decode into a streaming format.
- Implement strict-row one-pass mode and exact finite two-pass/spill mode.
- Differentially compare emitted requests, timing, KV hints, filtering, session
  behavior, and recorded outcomes with the current finite Baseten path.
- Demonstrate bounded RSS on a multi-GB logical/physical dataset.

This stage proves that “streaming” changes the outer dataset lifetime rather
than only the download mechanism.

### Stage 2: local/object follow and shadow replay

- Register `shadow_replay` workload.
- Add wall-clock-delay mapping, watermark/reorder policies, late/overload
  behavior, streaming metrics, local checkpoint backend, checkpoint result
  segments/indexes, atomic generation roots, and live partial-result reads.
- Implement a local immutable-object follow source and run against HTTP/gRPC
  mock targets.
- Exercise both request actions and incremental agent-graph actions through the
  shadow workload without mutating finite `GraphInputBundle` state.
- Validate five-minute publication cadence with accelerated deterministic tests
  and wall-clock soak tests.

### Stage 3: S3/NVCF adapter

- Implement native S3 finite/follow source, reconciliation, conditional object
  acquisition, retries, version/digest identity, credential confinement, and
  checkpoint cursor.
- Implement strict `streaming_dynamo_trace` for
  `dynamo.request.trace.v1`; validate production NVCF samples against that
  frozen schema and reject nonconforming versions.
- Prefer producer manifests with sealed event-time intervals; otherwise ship an
  explicitly estimated bounded-disorder mode.
- Run a shadow replay soak at expected and overload rates.

### Stage 4: cellular streaming placement

- Add bounded controller-to-cell chunks and ordered receipts.
- Preserve session affinity and controller global sequence.
- Add immutable finite shard assignment where capability validation permits it.
- Extend checkpoint, provenance, and merged reports across cells.
- Rotate and publish cellular result partitions through the same global epoch
  barrier and generation commit.

### Stage 5: broader streaming dataset adoption

- Add JSONL/CSV/ordinary HF formats and a non-timestamped streaming scheduled
  consumer if product demand requires it.
- Evaluate representing finite datasets internally through a sealed stream plus
  explicit collector while retaining public compatibility.
- Add further session programs/action sinks through the registered schemas;
  cross-chunk graph execution remains part of the conformance suite rather than
  an optional reinterpretation of finite Graph-IR.

### Indicative effort

This is not a one-loader patch. A production generation including large HF,
live S3 shadow replay, checkpointed results, multi-turn/agentic/graph continuity,
observability, and single-process execution is approximately 4-6 engineers for
16-24 weeks after schema/access availability. Cellular streaming placement is a
further 3-4 engineers for 8-12 weeks and should not block a single-process
launch. The critical technical risks are format closure/watermark guarantees,
bounded causal/session fidelity, incremental graph action execution, native S3
packaging/auth, and crash/result semantics—not basic object polling.

The stages are independently demonstrable, but no stage may claim general
streaming support while its outer execution path still collects the complete
row population.

## Verification

### Contract suites

Every source implementation passes one reusable suite covering:

- strict validation before effects;
- deterministic inventory and immutable identity;
- long pending periods without EOF;
- seal/limit/cancellation distinctions;
- retry classification and mutation races;
- resumable cursor round trips;
- count/byte backpressure; and
- credential/debug redaction.

Every format implementation passes one reusable suite covering:

- declared access capability;
- bounded output under a blocked sink;
- stable record IDs and total order tie-breaks;
- malformed/oversized input failures;
- watermark and session closure correctness;
- checkpoint cursor stability; and
- batch/segment lease destruction after terminal acknowledgement.

### Required behavior fixtures

- empty finite source, empty follow source, one partition, multiple partitions,
  delayed partition, duplicate notification, overwritten key, missing listed
  object, and explicit seal;
- equal timestamps across shards and randomized list/page order;
- hidden causal action at time 100 behind a future predecessor while an
  unrelated time-200 action is ready; causal frontier prevents overtaking;
- hard and estimated watermarks, late records on both sides of the bound, and
  source silence;
- backpressure at every boundary with no item loss under lossless policy;
- cancellation during discovery, acquisition, decode, spill, reorder wait,
  scheduled wait, dispatch, and checkpoint commit;
- blocking decode/sort/fsync/compaction saturation while clock-driven issuance
  remains responsive, followed by cooperative cancel-and-join;
- crash/restart after every stage horizon, with expected duplicate/loss window;
- checkpoint-result crash injection before/during/after segment write, fsync,
  generation CAS/rename, final compaction, and garbage collection;
- result retry with identical payload, conflicting same-range payload, missing
  range/cell/worker, schema mismatch, and deterministic merge order;
- partial results through each committed epoch and final compacted metrics/
  record equivalence to a one-shot reference run;
- final compaction failure, coordinator report-persistence failure, and aborted
  run with/without a safe terminal generation;
- target idempotency supported/unsupported cases;
- duplicate partition notification, overlapping logical row with identical
  producer key/content, conflicting content, and restart-distinct attempt IDs;
- `recorded_inputs` with divergent target output and encrypted
  `target_closed_loop` restore, missing key, and checkpoint-none behavior;
- multi-turn session closure by marker, watermark, inactivity, external sort,
  and unbounded-session refusal;
- Baseten one-pass strict and two-pass exact modes over Parquet, plus Arrow IPC
  pre-allocation guard or feature refusal;
- HF pinned revision change, gated token, resumable shard, partial Viewer
  inventory, unsupported script/generated dataset, disk-backed very-large shard
  catalog, multi-shard split, and row-limit terminal reason;
- S3 pagination, versioned/unversioned object identity, multipart ETag,
  reconciliation after missed notification, throttling, and checksum failure;
- wall-clock delay with source timestamp before/after target and monotonic anchor
  stability under system-clock adjustment;
- HTTP, gRPC, dry-run, worker-count, and endpoint-profile parity;
- cellular cell skew, session affinity, chunk retransmission, gap/digest/plan
  mismatch, and checkpoint horizon merge; and
- report/provenance artifacts with no source secret or raw bytes by default.

### Resource and performance gates

Before production launch:

1. A multi-GB HF/Baseten run MUST demonstrate peak RSS bounded by the authored
   memory budgets plus a measured fixed implementation allowance, independent
   of total row count.
2. Follow-mode soak duration MUST not produce monotonic growth in RSS, task
   count, open descriptors, segment entries, dedup state, mutable result state,
   or checkpoint generation-root size. Immutable committed segment growth must match
   authored retention and be garbage-collectable by generation reachability.
3. Every queue/store high-water mark MUST remain within its configured item and
   byte limit under a target slower than the source.
4. Streaming Baseten request/timing output MUST match the finite compatibility
   mode for the same pinned dataset and policy.
5. At expected NVCF rate, publication-to-target schedule slip and CPU overhead
   MUST be measured separately from target inference latency and meet a frozen
   acceptance inventory.
6. No new lock, synchronization, allocation, serialization, or task hop is
   added to the existing per-token response path. Streaming work occurs before
   request admission or at terminal acknowledgement boundaries.
7. `cargo fmt --check`, runtime tests with `engine` and relevant `parquet`/S3/
   cellular features, Clippy, dry-run E2E, and mock-server HTTP/gRPC E2E pass.

## Documentation deliverables

- Config-v2 schema and examples for finite HF, finite S3, and live shadow replay.
- Source/format descriptor and capability reference.
- Watermark, late-data, overload, checkpoint, and delivery-semantics guide.
- Checkpoint result epochs, partial-result visibility, retention/compaction, and
  restart troubleshooting guide.
- HF completeness and revision-pinning guide, including partial Parquet
  behavior.
- NVCF producer manifest contract if hard watermarks are adopted.
- Operational dashboard/metrics reference and capacity-sizing guide.
- Cellular authority and ordering model.
- Troubleshooting for source auth, acquisition retry, malformed data, late
  replay, budget exhaustion, and checkpoint divergence.

## External behavior references

- Hugging Face's official dataset documentation describes iterable streaming,
  shard distribution, and checkpoint/resume behavior:
  <https://huggingface.co/docs/datasets/use_with_pytorch>.
- The official Dataset Viewer documentation describes Parquet shard inventory,
  row-group-oriented access, and partial conversion behavior for large
  datasets: <https://huggingface.co/docs/dataset-viewer/parquet>.

These references inform the HF adapter behavior. They do not introduce a Python
runtime dependency or override the normative AIPerf contracts above.
