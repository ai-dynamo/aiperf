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
2. Select source, format, workload, endpoint, and transport through independent
   traits and frozen registries.
3. Bound memory, local disk, queued partitions, decoded units, session state,
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

1. **Independent composition.** Source, format, replay policy, placement,
   endpoint, and transport are independently selected or injected. No source
   adapter constructs endpoint wire requests, and no format adapter lists S3 or
   resolves HF revisions.
2. **Frozen implementation universe.** Every source and format factory is
   registered, selected, and strictly validated before source acquisition,
   network clients, worker runtimes, cells, or benchmark traffic begin. Live
   data changes; the factory universe does not.
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
    dataset units. Endpoint-specific materialization occurs just before
    admission through the existing `RequestMaterializer`; dispatch occurs
    through the existing `TurnDispatcher` and observer plane.
14. **Lifetime-safe segments.** Bytes and segments referenced by an admitted
    request remain alive through terminal dispatch. They are reclaimed after
    the final owning request/continuation and checkpoint receipt release them;
    a perpetual source does not retain a perpetual segment arena.
15. **Deterministic ordering.** Equal event-time records are ordered by a stable
    source-derived key and global sequence. Filesystem listing order, request
    completion order, worker wake order, and hash-map iteration never decide
    replay order.
16. **Explicit session closure.** A format may emit a complete multi-turn unit
    only from an explicit end marker, a proven key/time watermark, a sealed
    finite external sort, or an authored bounded-inactivity rule. Otherwise it
    MUST use strict row replay or fail validation; it cannot retain unbounded
    sessions while claiming bounded memory.
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

## Decision traceability

| Decision | Normative resolution | Detailed section |
|---|---|---|
| Product center | General streaming dataset plane; shadow replay is the first consumer | Purpose; architecture |
| First source adapters | HF Hub finite snapshots and S3-compatible finite/follow catalogs | Source contracts |
| First format adapters | Baseten trace and Dynamo/NVCF request trace; ordinary row formats follow | Format contracts |
| Execution model | Registered `shadow_replay` workload over ordinary native transports | Registry and composition |
| Finite compatibility | Existing resident dataset and Graph-IR APIs remain unchanged | Invariant 20; migration |
| Canonical unit | A stream envelope containing a canonical `Conversation` plus source/event metadata | Canonical dataset units |
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
| Results durability | Checkpoint-aligned immutable result segments, atomic manifests, deterministic final compaction | Checkpoint-based results |
| Plugin relationship | Static registry category now; dynamic plugin category requires new API generation | Registry and composition |

## Invariant enforcement map

| Invariants | Enforcement | Required evidence |
|---|---|---|
| 1, 2, 20 | Independent registry IDs, strict factory config, prepared composition, production search for source/format switches | Cross-product source×format×transport tests; unknown/duplicate ID tests |
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
| 29, 30 | Epoch barrier, content-addressed result segments, compare-and-commit manifest, deterministic compactor | Crash at every publish boundary; retry/dedup; live partial-read; final equivalence tests |

## Terminology

- **stream resource**: a validated `{source, format}` composition referenced by
  a workload.
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
       checkpoint epoch -> immutable result segments -> atomic manifest
```

The pipeline MAY fuse adjacent stages on one current-thread `LocalSet` when no
parallelism is useful. Trait boundaries define ownership and testability; they
do not require a channel or task hop. When stages are concurrent, the host owns
bounded channels with both item and byte permits.

## Registry and composition

### New registry categories

`AIPerfRegistry` gains two transactional, frozen categories:

```rust
stream_sources: TransactionalRegistry<Arc<dyn StreamingDatasetSourceFactory>>,
stream_formats: TransactionalRegistry<Arc<dyn StreamingDatasetFormatFactory>>,
```

The categories follow existing duplicate rejection, descriptor validation, and
freeze behavior. `AIPerfExtension` gains registration methods. Built-ins use the
same registration path as statically linked extensions.

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
          limits:
            acquired_partitions: 4
            decoded_units: 10000
            decoded_bytes: 512MiB
            state_memory: 512MiB
            state_disk: 100GiB

  workload:
    id: shadow_replay
    config:
      stream: shadow_input
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
```

Names and exact field spelling remain subject to implementation schema review;
the ownership is normative. Source config belongs to its factory, format config
belongs to its factory, and replay config belongs to the workload. Mixed or
unknown fields fail strict validation.

`RunResourceV2` and `ResourceRequirementsV2` gain dataset-stream presence.
Workloads that do not use streams neither validate nor open them.

### Capability agreement

Source and format descriptors are side-effect-free. Validation intersects:

- source mode: `finite`, `follow`, or both;
- byte access: sequential chunks, immutable local seekable lease, or range
  reads;
- source ordering: none, partition order, or event-time-related guarantee;
- resumability granularity: partition, byte, row group, or record;
- format media/schema identifiers;
- format access requirement and projection support;
- canonical output schema (`aiperf.session_fragment.v1` initially);
- event-time and stable-record-ID availability;
- session-closure requirements;
- placement support; and
- virtual-clock compatibility.

An incompatible pair fails before source effects and names both descriptors and
the missing capability. There is no coordinator switch on a source/format pair.

## Source contracts

### Factory and run-time source

The following sketches fix responsibility, not final spelling:

```rust
#[async_trait(?Send)]
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
        resume: Option<SourceCheckpoint>,
    ) -> Result<Box<dyn StreamingDatasetSource>, StreamSourceError>;
}

#[async_trait(?Send)]
pub trait StreamingDatasetSource {
    fn snapshot(&self) -> &SourceSnapshotReceipt;
    async fn next_event(&mut self) -> Result<SourceEvent, StreamSourceError>;
    fn request_stop(&self);
}

pub enum SourceEvent {
    Partition(SourcePartition),
    Frontier(SourceFrontier),
    Seal(SourceSeal),
}
```

`next_event` remains pending when a follow source has no new object. It does not
return an `Idle` event that invites polling loops. `request_stop` wakes a pending
call and causes phase-owned shutdown, not a fabricated source seal.

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
- follow mode with periodic full reconciliation of the not-yet-checkpointed key
  range;
- optional notification hints that accelerate discovery but do not replace
  reconciliation authority;
- stable lexicographic or manifest-authored partition order;
- object VersionId when available, otherwise conditional ETag/size plus exact
  acquired-byte BLAKE3 digest; and
- continuation-token handling without treating a page boundary as a stream
  frontier.

An object is published to the decoder only after its immutable generation is
bound. A later object with the same key and a different version is either a new
partition under an explicit versioned policy or a source mutation failure. It
is never silently substituted.

Retries use bounded exponential backoff with jitter driven by a source-control
clock/service, not benchmark request timing. Retry exhaustion, authorization,
not-found-after-list, checksum mismatch, throttling, and source mutation are
distinct errors.

### Hugging Face source

The HF source is a finite source unless a future Hub event contract is designed.
It:

1. resolves repository/revision once to an exact commit SHA;
2. resolves one subset and split;
3. enumerates the complete selected original shard inventory from repository
   metadata/card mappings;
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
pub trait StreamingDatasetFormat {
    async fn decode_partition(
        &mut self,
        partition: AcquiredPartition,
        output: &mut dyn DatasetEventSink,
    ) -> Result<DecodeReceipt, StreamFormatError>;

    async fn advance_source_frontier(
        &mut self,
        frontier: SourceFrontier,
        output: &mut dyn DatasetEventSink,
    ) -> Result<(), StreamFormatError>;

    async fn seal(
        &mut self,
        seal: SourceSeal,
        output: &mut dyn DatasetEventSink,
    ) -> Result<FormatSealReceipt, StreamFormatError>;
}
```

`DatasetEventSink::send` is asynchronous and budgeted, so a format cannot
outpace downstream state. Implementations may expose an equivalent pull stream;
the semantic requirement is bounded backpressure, not callback style.

Format-private typed rows stay behind the implementation. The public output is
a host-owned canonical session fragment, watermark contribution, or checkpoint
barrier. No `Any` downcast is needed between independently selected source and
format implementations.

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
ready. Conversation actions use the existing
`ConversationSession`/`RequestMaterializer` path; agentic and graph actions use
their corresponding streaming-capable workload binding.

### Run-scoped session coordinator

One coordinator survives every partition and decoder call:

```rust
#[async_trait(?Send)]
pub trait StreamingSessionCoordinator {
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

    async fn seal(
        &mut self,
        seal: SourceSeal,
        output: &mut dyn DatasetActionSink,
    ) -> Result<SessionSealReceipt, SessionCoordinatorError>;
}
```

State is keyed by `(stream_identity, stable_session_key)`, never by source
partition. `decode_partition` returning means only that those bytes are
exhausted. It MUST NOT close the session, flush incomplete state, create roots,
or discard request/tool/graph history. The next partition may immediately
extend any active session.

There are two order domains: per-session causal order and global replay event
order. A session action becomes globally reorderable only after its declared
predecessors are satisfied. A global watermark does not close a session unless
the format contract proves that it is also a hard session frontier.

A session closes only through an explicit close mutation, a format-defined
terminal outcome, a session-scoped hard watermark, or sealed-source validation.
An authored inactivity close is estimated/lossy and reported as such. At seal,
unresolved predecessors fail with stable identities unless incomplete-session
drop is explicitly selected.

### Checkpoint and cellular handoff

A checkpoint contains either the complete canonical/spilled state and causal
frontier for every active session through its decoded horizon, or a source
horizon before the first unrepresented fragment. Restore installs session state
and ownership before source polling resumes.

Cellular placement hashes the stable session key, not the source partition. A
session remains sticky to one cell. Migration is a transaction:

```text
freeze session sequence N
-> drain/record actions <= N at old owner
-> transfer authenticated state + digest + next sequence
-> new owner validates and durably installs
-> controller atomically changes route
-> release fragments > N
```

Until acknowledgement commits, new fragments remain bounded at the controller;
concurrent dual ownership is forbidden.

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

For Arrow IPC, memory remains bounded by the largest authored projected record
batch unless the implementation adds a lower-level streaming reader. The
design does not claim a 128-row decode bound that the dependency cannot enforce.

### Dynamo/NVCF request-trace format

The initial live format validates the exact NVCF schema during implementation.
If it is `dynamo.request.trace.v1`, the current native field validation and
request reconstruction may be extracted and reused. The streaming format does
not compile a complete `GraphInputBundle` merely because the bytes use a Dynamo
Request-level shadow replay emits session-addressed turn fragments as soon as
they decode. The coordinator retains request history and causal state across
objects and emits executable actions when ordering allows. Missing parents
before an applicable hard session watermark are incomplete data, not roots.
Responses in
the source are retained as recorded-outcome/reference metadata only when a
defined evaluator or comparison consumer requests them; they are not sent as
input to the target.

## Stream events and ordering

The format-to-consumer plane uses typed events:

```rust
pub enum DatasetStreamEvent {
    Fragment(StreamingSessionFragment),
    Watermark(EventTimeWatermark),
    CheckpointBarrier(StreamBarrier),
    End(StreamEnd),
}
```

There is no `Idle`. `End` states whether it resulted from source seal, authored
row/time limit, cancellation, or policy termination. Cancellation end is not a
source seal and cannot advance a resumable source cursor past unprocessed data.

### Stable order

Replay order is:

```text
(event_time, stable_tie_break, source_partition_identity, source_record_ordinal)
```

The host assigns a dense `global_sequence` only after this key is safe behind
the watermark. The sequence is the placement and checkpoint order but does not
replace source identity.

### Watermark policies

The event-time policy is an injected host-owned trait:

```rust
pub trait EventTimePolicy {
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

### Budget hierarchy

One host-owned `StreamingResourceBudget` grants permits for:

- discovered but unacquired partitions and bytes;
- concurrent acquisitions and local cached bytes;
- decoded batch items and bytes;
- format/session state memory and spill bytes;
- reorder items and bytes;
- ready/scheduled items and bytes;
- in-flight requests/conversations; and
- cellular unacknowledged chunks and bytes.

Each object that owns bytes retains the relevant permit. Moving an object moves
the permit; cloning payload storage does not mint capacity. Diagnostics expose
current and high-water usage per category.

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

Formats receive a process-owned `StreamingStateStore` capability rather than a
path:

```rust
pub trait StreamingStateStore {
    fn namespace(&self, owner: &StreamStateOwner) -> Result<Box<dyn StreamStateNamespace>>;
}
```

Namespaces provide checked put/get/remove, ordered iteration where required,
and byte accounting. The initial implementation may use bounded sorted run
files plus k-way merge; selecting an embedded database is an implementation
decision only after dependency and performance review. Formats cannot bypass
the store to create unaccounted scratch trees.

Memory-only session assembly is allowed when a hard watermark or explicit end
marker bounds active keys. Arbitrary finite external sort consumes disk budget
and produces an immutable derived-run digest. Exhaustion is a typed
`state_budget_exceeded` failure, not OOM.

### Segment lifetime

Streaming formats create batch-scoped `SegmentStore` arenas or a reclaimable
streaming segment store. An envelope holds an opaque lease. Admission clones
that lease into every continuation. Terminal completion releases it only after
the last request no longer needs materialization or raw capture. Deduplication
may share segments across live batches through weak/content-addressed entries,
but unreferenced entries are reclaimable.

## Shadow replay workload

`shadow_replay` consumes each causally ready conversation/request action once,
maps its event time, materializes it through the transport-selected
`RequestMaterializer`, and issues through `ScheduledRuntime`. A run-scoped
session table retains conversation history and affinity across every source
chunk. Agentic/graph actions select a corresponding streaming-capable consumer;
`shadow_replay` fails validation rather than flattening them.

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

## Checkpoint and delivery semantics

### Typed horizons

One checkpoint record contains separate monotonically contiguous horizons:

```text
discovered -> acquired -> decoded -> ordered -> admitted -> terminal
```

Each horizon binds stream-resource digest, source snapshot/generation,
partition/record cursor, format semantic digest, policy digest, and global
sequence where assigned. Holes are retained as bounded exception sets until
they close; a later item cannot advance a contiguous horizon over a missing
earlier item.

The event-time watermark is recorded beside these horizons but never used as a
source cursor.

### Checkpoint store

```rust
#[async_trait(?Send)]
pub trait StreamingCheckpointStore {
    async fn load_generation(&self, run: &StreamRunIdentity)
        -> Result<Option<CommittedCheckpointGeneration>, CheckpointError>;
    async fn commit_generation(
        &self,
        expected: Option<CheckpointGeneration>,
        next: CheckpointGenerationRecord,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError>;
}
```

The committed generation is the single atomic authority for both resume state
and result inventory. The default local implementation writes canonical bytes
to a private sibling temporary file, fsyncs according to authored durability,
and atomically renames. Object-store checkpointing may be added through
conditional generation writes. A stale or divergent writer fails; it does not
merge horizons by max.

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
    pub interval: CheckpointInterval,
    pub durability: CheckpointDurability,
}
```

- A mergeable metrics partition is always present.
- Exact/native/raw rows are present only when existing artifact/exporter policy
  requires them; a sketch run does not silently begin retaining exact records.
- Session results include stable session key, causal/close status, action range,
  and terminal outcome needed to explain partial active sessions.
- Every projection has a versioned schema and canonical ordering.

The result plane consumes terminal `RecordIngest`/captured-record facts through
the existing worker-local observation/record-processor boundary. It does not add
a callback to the response-token path. Epoch accumulators are worker local and
merge only at the checkpoint boundary.

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

Ingestion and decoding MAY continue into a bounded uncommitted overlay while
the barrier is prepared. `StreamingStateStore` exposes a versioned
`checkpoint_view(H)`; state after `H` remains in the live overlay and is not
mistaken for committed resume state. If the implementation cannot provide that
view, it briefly backpressures source/decode while rotating the epoch.

A long-running earlier request can hold the authoritative horizon. Completed
results above the hole may feed explicitly provisional dashboards but MUST NOT
enter a committed result epoch. Cancellation/failure policy eventually closes
the hole. This favors restart correctness over falsely durable partial totals.

### Sink contract

```rust
#[async_trait(?Send)]
pub trait CheckpointResultSink {
    async fn prepare_epoch(
        &self,
        epoch: ResultEpoch,
        partitions: Vec<ResultPartition>,
    ) -> Result<PreparedResultEpoch, ResultCheckpointError>;

    async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, ResultCheckpointError>;
}
```

`prepare_epoch` writes immutable content-addressed segments and returns their
verified descriptors without changing visible progress. The checkpoint
coordinator passes those descriptors to
`StreamingCheckpointStore::commit_generation`, which atomically publishes one
canonical record containing:

- prior generation digest and next epoch number;
- all typed input/stage horizons and watermark;
- active-session checkpoint/state-store snapshot digest;
- pending placement/action state permitted by delivery mode;
- deterministic result segment descriptors;
- cumulative projection counts and byte totals;
- metrics/format/policy/placement semantic digests; and
- terminal/final presence plus reason.

The local implementation writes segment files under a private generation store,
fsyncs them according to durability policy, writes the complete generation to a
private sibling temporary file, and atomically renames one `CURRENT` generation
record. An object-store implementation writes segments first and conditionally
creates/replaces one generation object using the prior generation token. Readers
trust only segments reachable from that committed record.

### Segment identity and ordering

One segment descriptor binds:

```text
run_identity + epoch + cell_id + worker_id + projection_id + schema_version
+ first_global_sequence + last_global_sequence + item_count + byte_length
+ canonical_payload_digest
```

Segments contain records sorted by `(global_sequence, stable_record_id)`.
Metric partitions declare the exact record/action range they summarize. A
retry producing the same bytes resolves to the same segment identity. The same
logical range with different bytes is a checkpoint conflict and fails; it is
never accepted as a second contribution.

Merge and compaction order is fixed by
`(epoch, cell_id, worker_id, projection_id, first_global_sequence, digest)`.
Floating-point accumulator merges use this order so restart/topology iteration
does not introduce arbitrary reduction order. Counts and range coverage are
checked before a partition contributes.

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
5. Garbage collection removes only segments unreachable from every retained
   committed generation and uses generation leases so a live reader is safe.

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

At normal seal or cancelled/failed terminal finalization, the coordinator closes
one final epoch and commits `is_final: true` with terminal reason. A deterministic
compactor then reconstructs the ordinary final `NativeReport`, exact records,
JSONL/CSV/Parquet/outputs, and exporter capture projections from committed
segments. The report binds the final generation digest. Byte identity with
today's completion-order artifacts is not promised where canonical global order
is required, but record multiset and documented canonical ordering are.

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

Result epochs include only terminally classified actions even when source resume
uses `admitted` or `decoded` delivery. If those modes advance input beyond the
terminal result horizon, the generation records both values and the possible
result-loss window explicitly. `terminal` mode keeps the authoritative input and
result horizons equal and is the default for restartable shadow replay.

### Delivery modes

- `terminal` (default for replay): commit only through contiguous terminal
  acknowledgements. Crash after target acceptance but before commit can replay a
  request, so semantics are at least once unless target idempotency is enabled.
- `admitted`: commit after successful local admission. Crash may lose in-flight
  effects, so semantics are at most once from AIPerf's perspective.
- `decoded` or `acquired`: useful for ingestion diagnostics, not accepted as
  faithful shadow-replay completion unless explicitly authored.
- `none`: ephemeral benchmark, no restart claim.

When target idempotency is configured, request keys derive from stream identity
plus stable record ID and are injected through an endpoint-supported field or
header. The endpoint descriptor must declare support. AIPerf then reports
`idempotent_at_least_once_submission`; it still does not claim exactly once
without verified target semantics.

Checkpoint artifacts contain no credentials, source bearer tokens, raw prompts,
or response bodies.

## Cellular placement

`StreamingPlacement` is an injected trait selected from validated topology and
stream capabilities:

```rust
pub trait StreamingPlacement {
    fn place(&self, action: OrderedDatasetAction)
        -> Result<PlacementDecision, PlacementError>;
    fn acknowledge(&self, receipt: PlacementReceipt)
        -> Result<Vec<TerminalStreamAck>, PlacementError>;
}
```

### Centrally ordered live placement

For live shadow replay:

1. controller opens source and owns credentials;
2. controller decodes fragments and advances run-scoped session state;
3. controller establishes watermark order and assigns dense global sequence;
4. stable-session-affinity placement maps each action to its sole owning cell;
5. controller sends bounded, versioned chunks containing canonical action data,
   sequence, target timing, lease/content references, count, byte length, and
   digest;
6. cell acknowledges receipt/admission/terminal through ordered receipts; and
7. controller advances checkpoint horizons only across contiguous verified
   receipts.

Every route has finite unacknowledged chunk/byte windows. Gaps, duplicates,
wrong cells, digest/length/count mismatch, and plan/policy digest mismatch fail
before checkpoint advance. Controller and cell use the same host-owned DTO;
wall-clock estimates are not compared between hosts to establish order.

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

- source and format factory IDs/descriptors/semantic config digests;
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
|-- unit.rs                # envelopes, leases, stream events
|-- budget.rs              # count/byte permits and high-water facts
|-- state.rs               # bounded state/spill capability
|-- event_time.rs          # watermarks, stable reorder, time mapping
|-- checkpoint.rs          # stage horizons and stores
|-- results.rs             # epoch barriers, result segments, manifests/reader
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

- `extensions/mod.rs`: source/format registries and registration methods;
- `engine/registry.rs`: dataset-stream resource requirement and frozen lookup;
- `engine/protocol_v2.rs`: strict named stream resources;
- CLI/config projection: user-facing source/format/replay options without
  source-format combination switches;
- `phase_runtime.rs`/timing: streaming phase execution adapter, not lifecycle
  duplication;
- cellular protocol/controller/cell: incremental placement chunks and receipts;
- report/export: generic streaming provenance and metrics; and
- record/metrics/result plane: epoch rotation, immutable segment schemas,
  atomic checkpoint-generation publication, partial readers, and final
  compaction; and
- Baseten/HF modules: extract projected decoders/acquisition helpers while
  retaining current finite adapters.

No Python module is invoked. The S3 adapter uses a native Rust SDK/client behind
the source trait. Dependency selection and feature ownership require a separate
implementation review, but do not change this architecture.

## Migration and delivery plan

### Stage 0: contracts and executable proof

- Add core stream vocabulary, source/format registries, in-memory fake source,
  canonical conversation format, budget accounting, and a no-network dry-run
  consumer.
- Add an in-memory checkpoint result sink and prove barrier/range/dedup/final
  compaction equivalence before filesystem durability.
- Prove pending-versus-seal, backpressure, cancellation, and lease lifetime with
  `SimClock` and current-thread `LocalSet`.
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
  behavior, streaming metrics, local checkpoint store, checkpoint result
  segments, atomic generation manifests, and live partial-result reads.
- Implement a local immutable-object follow source and run against HTTP/gRPC
  mock targets.
- Validate five-minute publication cadence with accelerated deterministic tests
  and wall-clock soak tests.

### Stage 3: S3/NVCF adapter

- Implement native S3 finite/follow source, reconciliation, conditional object
  acquisition, retries, version/digest identity, credential confinement, and
  checkpoint cursor.
- Implement the exact NVCF format adapter after schema samples are acquired.
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
- Design streaming Graph-IR fragments separately if required.

### Indicative effort

This is not a one-loader patch. A production generation including large HF,
live S3 shadow replay, checkpointing, observability, and single-process
execution is approximately 3-4 engineers for 12-16 weeks after schema/access
availability. Cellular streaming placement is a further 2-3 engineers for
6-10 weeks and should not block a single-process launch. The critical technical
risks are format closure/watermark guarantees, bounded session fidelity, native
S3 packaging/auth, and crash semantics—not basic object polling.

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
- hard and estimated watermarks, late records on both sides of the bound, and
  source silence;
- backpressure at every boundary with no item loss under lossless policy;
- cancellation during discovery, acquisition, decode, spill, reorder wait,
  scheduled wait, dispatch, and checkpoint commit;
- crash/restart after every stage horizon, with expected duplicate/loss window;
- checkpoint-result crash injection before/during/after segment write, fsync,
  generation CAS/rename, final compaction, and garbage collection;
- result retry with identical payload, conflicting same-range payload, missing
  range/cell/worker, schema mismatch, and deterministic merge order;
- partial results through each committed epoch and final compacted metrics/
  record equivalence to a one-shot reference run;
- target idempotency supported/unsupported cases;
- multi-turn session closure by marker, watermark, inactivity, external sort,
  and unbounded-session refusal;
- Baseten one-pass strict and two-pass exact modes over Parquet and Arrow IPC;
- HF pinned revision change, gated token, resumable shard, partial Viewer
  inventory, multi-shard split, and row-limit terminal reason;
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
   or checkpoint manifest size. Immutable committed segment growth must match
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
