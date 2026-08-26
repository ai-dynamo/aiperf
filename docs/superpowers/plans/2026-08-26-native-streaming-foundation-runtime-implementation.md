<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Streaming Foundation and Runtime Seams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the feature gates, typed streaming vocabulary, resource ownership, object-safe extension contracts, frozen registries, strict Protocol-v2 configuration, bounded scheduled-runtime terminal processing, reusable phase/capture construction, and UTC/event-time authority required by every later native streaming dataset and shadow-replay subsystem.

**Architecture:** This plan builds only the foundation and existing-runtime adaptations from master Tasks 0, 1A–1E, 2, 3, 4A–4B, and 7A. The lightweight `streaming` feature owns all host contracts and local execution prerequisites; `streaming-s3` adds only AWS dependencies and advertises no S3 factory until its later executable adapter lands. Existing finite execution remains the reference path and is migrated onto reusable seams without adding a `NativeDatasetPlan::Streaming` variant or a non-executable `shadow_replay` factory.

**Tech Stack:** Rust 2024, Tokio current-thread runtimes and `LocalSet`, `async_trait(?Send)`, BLAKE3, strict Serde DTOs, `chacha20poly1305` and `zeroize` behind `streaming`, optional `aws-config` and `aws-sdk-s3` behind `streaming-s3`, existing `Clock`, `ScheduledRuntime`, `RunCapture`, `TransactionalRegistry`, Protocol v2, and Config v2.

**Spec:** `artifacts/streaming-design/streaming-dataset-shadow-replay-design.md` at approved commit `505efc06b0`; master plan: `docs/superpowers/plans/2026-08-26-native-streaming-datasets-shadow-replay-implementation.md`.

## Global Constraints

- Pure Rust only; do not invoke or extend Python dataset, replay, or live-streaming code.
- Freeze `streaming = ["engine", "dep:chacha20poly1305", "dep:zeroize"]` and `streaming-s3 = ["streaming", "dep:aws-config", "dep:aws-sdk-s3"]` exactly.
- Forward both features through `aiperf-cli`, include `streaming` in CLI `default`, and include `streaming-s3` in CLI `full`.
- A feature-off source or workload is absent from the registry and catalog. Never register a rejecting placeholder factory.
- Task 3 may use a test-local fake workload to prove resource requirements, but the stock registry must not advertise `shadow_replay` until the later executable vertical slice.
- Preserve finite `DatasetLoader`, `Dataset`, `ConversationSource`, `FixedSchedule`, `GraphInputAdapter`, `NativeDatasetPlan`, scheduled report, and graph report behavior.
- Route replay waits and wall/monotonic mapping through `Clock`. Only `clock/` may read `SystemTime`; streaming modules receive an immutable anchor.
- Keep worker state local. No `Arc<Mutex<_>>`, unbounded channel, per-record processor task, or lifetime-growing session map on streaming request paths.
- Every queue owns item and byte permits. Moving a value moves its permits; cloning payload storage does not mint capacity.
- Runtime async traits use `#[async_trait(?Send)]`; factory traits are `Debug + Send + Sync`.
- Library APIs use explicit error enums with `Display` and `Error`; `anyhow` stays at engine/application boundaries.
- Every public item has `///` documentation. Every new Rust file has exactly the two NVIDIA SPDX lines and `//!` module documentation.
- Each task has one focused test-suite invocation, two reviews, one focused commit, and no unrelated changes.
- Run commands from the nested `rust/` workspace after activating `/home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate`.
- Use `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target` for every Cargo command.

---

## File Structure and Ownership

```text
rust/runtime/src/streaming.rs                 feature inventory and module root
rust/runtime/src/streaming/identity.rs        canonical typed identities
rust/runtime/src/streaming/unit.rs            canonical fragments/actions/positions
rust/runtime/src/streaming/budget.rs          item+byte capacity and owned leases
rust/runtime/src/streaming/blocking.rs        bounded blocking-work owner
rust/runtime/src/streaming/source.rs          source factories/runtime contract
rust/runtime/src/streaming/format.rs          format/decoder contract
rust/runtime/src/streaming/session.rs         session-program contract
rust/runtime/src/streaming/action.rs          action binding/driver contract
rust/runtime/src/streaming/checkpoint.rs      participant/backend/result I/O contract
rust/runtime/src/streaming/terminal_lane.rs   bounded scheduled terminal processing
rust/runtime/src/streaming/event_time.rs      UTC mapping, watermark, late policy, horizon
rust/runtime/src/config/model/dataset_stream.rs strict public Config-v2 types
rust/runtime/src/engine/execute/capture_service.rs reusable finite/streaming construction
```

Task ordering is strict through Task 1E: `0 → 1A → 1B → 1C → 1D → 1E`. After Task 1E merges, Tasks 2 and 4A may run in parallel. Task 3 depends on Task 2. Task 4B depends on Task 4A. Task 7A depends on Tasks 1D and the checkpoint participant vocabulary from master Task 5A; do not start 7A before that merge.

### Task 0: Freeze Native Streaming Features and Dependencies

**Files:**
- Modify: `rust/Cargo.toml:64-86`
- Modify: `rust/runtime/Cargo.toml:11-70`
- Modify: `rust/runtime/Cargo.toml:75-125`
- Modify: `rust/cli/Cargo.toml:18-82`
- Modify: `rust/Cargo.lock`
- Create: `rust/runtime/src/streaming.rs`
- Modify: `rust/runtime/src/lib.rs:40-42`
- Test: `rust/runtime/tests/streaming_feature_inventory.rs`

**Interfaces:**
- Consumes: existing `engine`, `parquet`, `cellular`, and CLI forwarding conventions.
- Produces: Cargo features `streaming` and `streaming-s3`; `aiperf_runtime::streaming::{STREAMING_RUNTIME_COMPILED, STREAMING_S3_COMPILED}`; a feature-gated module root that later tasks extend.

- [ ] **Step 1: Add the RED feature inventory test**

Create `rust/runtime/tests/streaming_feature_inventory.rs`:

```rust
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::streaming::{STREAMING_RUNTIME_COMPILED, STREAMING_S3_COMPILED};

#[test]
fn lightweight_streaming_inventory_excludes_s3() {
    assert!(STREAMING_RUNTIME_COMPILED);
    assert!(!STREAMING_S3_COMPILED);
}
```

This is compilable test code. RED is Cargo-level: `--features streaming` does not exist yet.

- [ ] **Step 2: Run the one task suite and verify RED**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd rust
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --no-default-features --features streaming --test streaming_feature_inventory
```

Expected: Cargo reports that package `aiperf-runtime` does not contain feature `streaming`.

- [ ] **Step 3: Add the exact dependency and feature graph**

Add these exact workspace dependencies to `rust/Cargo.toml`:

```toml
aws-config = { version = "=1.11.0", default-features = false, features = ["behavior-version-latest", "rt-tokio", "rustls"] }
aws-sdk-s3 = { version = "=1.144.0", default-features = false, features = ["behavior-version-latest", "default-https-client", "http-1x", "rt-tokio", "rustls"] }
chacha20poly1305 = { version = "=0.11.0", default-features = false, features = ["alloc", "getrandom", "zeroize"] }
zeroize = { version = "=1.9.0", features = ["derive"] }
```

Add to `rust/runtime/Cargo.toml`:

```toml
[features]
streaming = ["engine", "dep:chacha20poly1305", "dep:zeroize"]
streaming-s3 = ["streaming", "dep:aws-config", "dep:aws-sdk-s3"]

[dependencies]
aws-config = { workspace = true, optional = true }
aws-sdk-s3 = { workspace = true, optional = true }
chacha20poly1305 = { workspace = true, optional = true }
zeroize = { workspace = true, optional = true }
```

Add to `rust/cli/Cargo.toml`:

```toml
default = ["streaming", "grpc", "cellular", "parquet", "websocket"]
streaming = ["aiperf-runtime/streaming"]
streaming-s3 = ["streaming", "aiperf-runtime/streaming-s3"]
full = ["streaming-s3", "dynosim", "parquet", "cellular", "grpc", "websocket"]
```

- [ ] **Step 4: Add only the conditional module inventory**

Add to `rust/runtime/src/lib.rs` immediately above the existing `engine` declaration:

```rust
#[cfg(feature = "streaming")]
pub mod streaming;
```

Create `rust/runtime/src/streaming.rs`:

```rust
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native streaming dataset contracts and execution support.

/// Whether the lightweight native streaming runtime is compiled.
pub const STREAMING_RUNTIME_COMPILED: bool = true;

/// Whether S3 source and object-store support are compiled.
pub const STREAMING_S3_COMPILED: bool = cfg!(feature = "streaming-s3");
```

Do not declare source IDs, backend IDs, factories, or catalog entries in this task.

- [ ] **Step 5: Regenerate the lockfile and verify GREEN**

Run the Step 2 command. Expected: one passing test, and the build log contains no `aws-sdk-s3` or `aws-config` compilation because `streaming-s3` is off.

- [ ] **Step 6: Review and commit**

```bash
git add rust/Cargo.toml rust/runtime/Cargo.toml rust/cli/Cargo.toml rust/Cargo.lock rust/runtime/src/lib.rs rust/runtime/src/streaming.rs rust/runtime/tests/streaming_feature_inventory.rs
git commit -m "build: add native streaming feature gates"
```

### Task 1A: Streaming Vocabulary and Stable Identity

**Files:**
- Modify: `rust/runtime/src/streaming.rs:4-end`
- Create: `rust/runtime/src/streaming/identity.rs`
- Create: `rust/runtime/src/streaming/unit.rs`
- Test: `rust/runtime/tests/streaming_identity.rs`

**Interfaces:**
- Consumes: BLAKE3 and Serde.
- Produces: checked `EventTimeUtc`, `SourcePosition`, `ImmutableObjectIdentity`, `StableRecordId`, `StableSessionKey`, `StableActionId`, `AttemptId`, `LogicalReplayRunId`, `RunIncarnationId`, `StableOrderKey`, `UnitProvenance`, `StreamingSessionFragment`, and `ExecutableDatasetAction`.

The public identity constructors are:

```rust
pub fn physical_record_id(
    stream_identity: &[u8],
    partition_generation: &ImmutableObjectIdentity,
    decoder_coordinate: &[u8],
    format_semantic_digest: &[u8; 32],
) -> PhysicalRecordId;

pub fn stable_record_id_from_key(namespace: &[u8], producer_key: &[u8]) -> StableRecordId;
pub fn stable_session_key(namespace: &[u8], producer_key: &[u8]) -> StableSessionKey;
pub fn one_turn_session_key(record_id: StableRecordId) -> StableSessionKey;
pub fn stable_action_id(
    program_digest: &[u8; 32],
    session: StableSessionKey,
    causes: &[StableRecordId],
    kind: DatasetActionKind,
    causal_ordinal: u64,
) -> StableActionId;
pub fn attempt_id(
    action: StableActionId,
    incarnation: RunIncarnationId,
    attempt_ordinal: u64,
) -> AttemptId;
pub fn classify_logical_duplicate(
    existing: &LogicalRecordReceipt,
    candidate: &LogicalRecordReceipt,
) -> Result<DuplicateDisposition, IdentityError>;
```

- [ ] **Step 1: Add representative RED identity tests**

Create `rust/runtime/tests/streaming_identity.rs` with these compilable tests:

```rust
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::streaming::identity::{
    RunIncarnationId, attempt_id, stable_record_id_from_key, stable_session_key,
};

#[test]
fn stable_record_id_is_discovery_order_independent() {
    let first = stable_record_id_from_key(b"tenant/model", b"producer-record-7");
    let second = stable_record_id_from_key(b"tenant/model", b"producer-record-7");
    assert_eq!(first, second);
}

#[test]
fn stable_session_key_joins_partitions() {
    let from_partition_a = stable_session_key(b"trace-v1", b"session-42");
    let from_partition_b = stable_session_key(b"trace-v1", b"session-42");
    assert_eq!(from_partition_a, from_partition_b);
}

#[test]
fn attempt_identity_changes_with_incarnation() {
    let action = aiperf_runtime::streaming::identity::StableActionId::from_bytes([9; 32]);
    let first = attempt_id(action, RunIncarnationId::from_bytes([1; 32]), 0);
    let second = attempt_id(action, RunIncarnationId::from_bytes([2; 32]), 0);
    assert_ne!(first, second);
}
```

Add table-driven cases in the same file for topology-independent action IDs, conflicting logical content, negative event time if disallowed by the selected constructor, and `u64` coordinate overflow during checked addition.

- [ ] **Step 2: Run the task suite and verify RED**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd rust
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_identity
```

Expected: unresolved imports for `streaming::identity`.

- [ ] **Step 3: Implement canonical identity encoding**

Extend `streaming.rs`:

```rust
pub mod identity;
pub mod unit;
```

Use private length-delimited hashing in `identity.rs`:

```rust
fn domain_hash(domain: &'static [u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    for field in fields {
        hasher.update(&(field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    *hasher.finalize().as_bytes()
}

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Deserialize, serde::Serialize)]
        pub struct $name([u8; 32]);

        impl $name {
            /// Construct from canonical digest bytes.
            #[must_use]
            pub const fn from_bytes(bytes: [u8; 32]) -> Self { Self(bytes) }

            /// Borrow canonical digest bytes.
            #[must_use]
            pub const fn as_bytes(&self) -> &[u8; 32] { &self.0 }
        }
    };
}
```

Use the exact domains from the approved spec: `aiperf.stream.physical.v1`, `aiperf.stream.logical-record.v1`, `aiperf.stream.session.v1`, `aiperf.stream.one-turn-session.v1`, `aiperf.stream.action.v1`, and `aiperf.stream.attempt.v1`. No worker, cell, route, discovery order, or global sequence argument may appear in these functions.

- [ ] **Step 4: Implement canonical host units**

In `unit.rs`, define the initial closed host vocabulary:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetActionKind { Request, GraphNode, SessionTerminal }

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionMutationV1 {
    ConversationTurn(ConversationTurnFragment),
    AgentEvent(AgentEventFragment),
    GraphNode(GraphNodeFragment),
    GraphEdge(GraphEdgeFragment),
    SessionClose(SessionCloseFragment),
}

#[derive(Clone, Debug)]
pub struct StreamingSessionFragment {
    pub record_id: StableRecordId,
    pub session_key: StableSessionKey,
    pub source_position: SourcePosition,
    pub source_partition: ImmutableObjectIdentity,
    pub event_time: Option<EventTimeUtc>,
    pub stable_tie_break: StableOrderKey,
    pub predecessors: smallvec::SmallVec<[StableRecordId; 2]>,
    pub mutation: SessionMutationV1,
    pub provenance: UnitProvenance,
    pub lease: SessionFragmentLease,
}
```

`SessionFragmentLease` is a non-cloneable placeholder-free ownership token with zero charged bytes in this task; Task 1B gives it a budget lease constructor. Payload structs contain endpoint-neutral authored data only and no executable tools or transport requests.

- [ ] **Step 5: Run the suite and verify GREEN**

Run the Step 2 command. Expected: all identity and checked-constructor tests pass.

- [ ] **Step 6: Review and commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/identity.rs rust/runtime/src/streaming/unit.rs rust/runtime/tests/streaming_identity.rs
git commit -m "feat(runtime): define streaming identity vocabulary"
```

### Task 1B: Item/Byte Resource Budgets

**Files:**
- Create: `rust/runtime/src/streaming/budget.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Modify: `rust/runtime/src/streaming/unit.rs`
- Test: `rust/runtime/tests/streaming_budget.rs`

**Interfaces:**
- Consumes: Tokio semaphore and Task 1A lease-bearing units.
- Produces: `BudgetLimits`, `StreamingResourceBudget`, `BudgetLease`, `BudgetSnapshot`, and `BudgetError`.

```rust
pub struct BudgetLimits { pub max_items: usize, pub max_bytes: usize }
pub struct StreamingResourceBudget;
impl StreamingResourceBudget {
    pub fn new(limits: BudgetLimits) -> Result<Self, BudgetError>;
    pub async fn acquire(&self, items: usize, bytes: usize) -> Result<BudgetLease, BudgetError>;
    pub fn close(&self);
    pub fn snapshot(&self) -> BudgetSnapshot;
}
```

- [ ] **Step 1: Add the RED ownership tests**

```rust
// rust/runtime/tests/streaming_budget.rs
use aiperf_runtime::streaming::budget::{BudgetLimits, StreamingResourceBudget};

#[tokio::test(flavor = "current_thread")]
async fn dropping_owned_lease_returns_item_and_bytes() {
    let budget = StreamingResourceBudget::new(BudgetLimits { max_items: 1, max_bytes: 64 })
        .expect("valid limits");
    let lease = budget.acquire(1, 64).await.expect("first lease");
    assert_eq!(budget.snapshot().used_bytes, 64);
    drop(lease);
    assert_eq!(budget.snapshot().used_bytes, 0);
    assert!(budget.acquire(1, 64).await.is_ok());
}

#[tokio::test(flavor = "current_thread")]
async fn cancellation_wakes_blocked_acquire() {
    let budget = StreamingResourceBudget::new(BudgetLimits { max_items: 1, max_bytes: 1 })
        .expect("valid limits");
    let _held = budget.acquire(1, 1).await.expect("held lease");
    budget.close();
    assert!(budget.acquire(1, 1).await.is_err());
}
```

Add move-without-minting, zero capacity, request-larger-than-capacity, overflow, and high-water tests.

- [ ] **Step 2: Run the task suite and verify RED**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd rust
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_budget
```

- [ ] **Step 3: Implement two-dimensional RAII capacity**

Use two owned Tokio semaphore permits acquired in a fixed item-then-byte order. Store exact charged counts in `BudgetLease`; implement no `Clone`. `close` closes both semaphores and wakes waiters. Snapshot counters and high-water marks use atomics outside the per-token path. Convert `SessionFragmentLease` into a newtype over `BudgetLease`.

Representative constructor:

```rust
impl StreamingResourceBudget {
    pub fn new(limits: BudgetLimits) -> Result<Self, BudgetError> {
        if limits.max_items == 0 || limits.max_bytes == 0 {
            return Err(BudgetError::ZeroCapacity);
        }
        Ok(Self::from_validated_limits(limits))
    }
}
```

- [ ] **Step 4: Run the suite and verify GREEN**

Run the Step 2 command.

- [ ] **Step 5: Review and commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/budget.rs rust/runtime/src/streaming/unit.rs rust/runtime/tests/streaming_budget.rs
git commit -m "feat(runtime): add streaming resource budgets"
```

### Task 1C: Bounded Blocking Execution Owner

**Files:**
- Create: `rust/runtime/src/streaming/blocking.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Test: `rust/runtime/tests/streaming_blocking.rs`

**Interfaces:**
- Consumes: `StreamingResourceBudget` and Tokio `spawn_blocking`.
- Produces the approved API:

```rust
impl StreamingBlockingExecutor {
    pub async fn run<T, F>(
        &self,
        class: BlockingWorkClass,
        budget: BlockingWorkBudget,
        work: F,
    ) -> Result<BudgetedBlockingOutput<T>, BlockingWorkError>
    where
        F: FnOnce(BlockingCancellation) -> Result<T, BlockingWorkError> + Send + 'static,
        T: Send + 'static;

    pub async fn cancel_and_join(&self) -> Result<(), BlockingWorkError>;
}
```

- [ ] **Step 1: Add RED saturation and join tests**

```rust
use aiperf_runtime::streaming::blocking::{
    BlockingWorkBudget, BlockingWorkClass, StreamingBlockingExecutor,
};

#[tokio::test(flavor = "current_thread")]
async fn output_permit_lives_until_output_drop() {
    let executor = StreamingBlockingExecutor::for_test(1, 8, 8).expect("executor");
    let output = executor.run(
        BlockingWorkClass::Decode,
        BlockingWorkBudget { input_bytes: 1, output_bytes: 8 },
        |_cancel| Ok(vec![0_u8; 8]),
    ).await.expect("first output");
    assert_eq!(executor.snapshot().output_bytes, 8);
    drop(output);
    assert_eq!(executor.snapshot().output_bytes, 0);
    executor.cancel_and_join().await.expect("clean shutdown");
}
```

Add a barrier-controlled accepted job proving `cancel_and_join` waits until cooperative cancellation is observed, plus saturation and `SimClock` responsiveness tests.

- [ ] **Step 2: Run the task suite and verify RED**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd rust
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_blocking
```

- [ ] **Step 3: Implement the bounded owner**

Use one fixed count of accepted `spawn_blocking` jobs guarded before enqueue. Retain accepted join handles in a slab whose entries are removed when joined; never use Tokio's global blocking queue as capacity authority. `BlockingCancellation` wraps an atomic flag; long work calls `is_cancelled()` between bounded chunks. `BudgetedBlockingOutput<T>` owns its output-byte lease and dereferences to `T` without exposing the permit.

- [ ] **Step 4: Run the suite and verify GREEN**

Run the Step 2 command.

- [ ] **Step 5: Review and commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/blocking.rs rust/runtime/tests/streaming_blocking.rs
git commit -m "feat(runtime): add bounded streaming blocking owner"
```

### Task 1D: Object-Safe Streaming Contracts

**Files:**
- Create: `rust/runtime/src/streaming/source.rs`
- Create: `rust/runtime/src/streaming/format.rs`
- Create: `rust/runtime/src/streaming/session.rs`
- Create: `rust/runtime/src/streaming/action.rs`
- Create: `rust/runtime/src/streaming/checkpoint.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Test: `rust/runtime/tests/streaming_contracts.rs`

**Interfaces:**
- Consumes: Tasks 1A–1C types.
- Produces: the five extension factory contracts, runtime source/decoder/session/action contracts, checkpoint participant/backend contracts, and backend-facing result vocabulary. Exact method signatures follow.

```rust
pub trait StreamingDatasetSourceFactory: std::fmt::Debug + Send + Sync {
    fn descriptor(&self) -> &'static StreamingSourceDescriptor;
    fn validate(&self, authored: &serde_json::value::RawValue)
        -> Result<Box<dyn ValidatedStreamingSourceConfig>, StreamSourceError>;
    fn prepare(&self, config: Box<dyn ValidatedStreamingSourceConfig>, context: &StreamingSourcePrepareContext)
        -> Result<Box<dyn PreparedStreamingDatasetSource>, StreamSourceError>;
}

#[async_trait::async_trait(?Send)]
pub trait PreparedStreamingDatasetSource {
    async fn open(self: Box<Self>, stop: StreamingStopReceiver)
        -> Result<OpenedStreamingDatasetSource, StreamSourceError>;
}

#[async_trait::async_trait(?Send)]
pub trait StreamingDatasetSource: StreamingCheckpointParticipant {
    fn snapshot(&self) -> &SourceSnapshotReceipt;
    async fn next_event(&mut self) -> Result<SourceEvent, StreamSourceError>;
}

#[async_trait::async_trait(?Send)]
pub trait SourcePartitionContent {
    fn identity(&self) -> &ImmutableObjectIdentity;
    fn size_bytes(&self) -> Option<u64>;
    async fn acquire(
        &self,
        request: PartitionAccessRequest,
        budget: &AcquisitionBudget,
    ) -> Result<AcquiredPartition, StreamSourceError>;
}

pub trait StreamingDatasetFormatFactory: std::fmt::Debug + Send + Sync {
    fn descriptor(&self) -> &'static StreamingFormatDescriptor;
    fn validate(&self, authored: &serde_json::value::RawValue, source: &StreamingSourceDescriptor)
        -> Result<Box<dyn ValidatedStreamingFormatConfig>, StreamFormatError>;
    fn prepare(&self, config: Box<dyn ValidatedStreamingFormatConfig>, context: &StreamingFormatPrepareContext)
        -> Result<Box<dyn StreamingDatasetFormat>, StreamFormatError>;
}

pub trait StreamingSessionProgramFactory: std::fmt::Debug + Send + Sync {
    fn descriptor(&self) -> &'static StreamingSessionProgramDescriptor;
    fn validate(
        &self,
        authored: &serde_json::value::RawValue,
        format: &StreamingFormatDescriptor,
        workload: &crate::engine::registry::WorkloadDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingSessionProgramConfig>, SessionCoordinatorError>;
    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingSessionProgramConfig>,
        context: &StreamingSessionPrepareContext,
    ) -> Result<Box<dyn StreamingSessionCoordinator>, SessionCoordinatorError>;
}

#[async_trait::async_trait(?Send)]
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

#[async_trait::async_trait(?Send)]
pub trait StreamingPartitionDecoder {
    async fn next_batch(&mut self, budget: DecodeBatchBudget) -> Result<DecodeStep, StreamFormatError>;
    fn resume_state(&self) -> Result<DecoderResumeState, StreamFormatError>;
}

#[async_trait::async_trait(?Send)]
pub trait StreamingSessionCoordinator: StreamingCheckpointParticipant {
    async fn ingest(&mut self, fragment: StreamingSessionFragment, output: &mut dyn DatasetActionSink)
        -> Result<(), SessionCoordinatorError>;
    async fn advance_watermark(&mut self, watermark: SessionWatermark, output: &mut dyn DatasetActionSink)
        -> Result<(), SessionCoordinatorError>;
    async fn observe_execution(&mut self, event: ActionExecutionEvent, output: &mut dyn DatasetActionSink)
        -> Result<(), SessionCoordinatorError>;
    async fn seal(&mut self, seal: SourceSeal, output: &mut dyn DatasetActionSink)
        -> Result<SessionSealReceipt, SessionCoordinatorError>;
}

#[async_trait::async_trait(?Send)]
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

pub struct PreparedStreamingActionBinding {
    pub submitter: Box<dyn StreamingActionSubmitter>,
    pub driver: Box<dyn StreamingActionDriver>,
    pub control: Box<dyn StreamingActionDriverControl>,
}

pub trait StreamingActionSinkFactory: std::fmt::Debug + Send + Sync {
    fn descriptor(&self) -> &'static StreamingActionSinkDescriptor;
    fn validate_binding(
        &self,
        authored: &serde_json::value::RawValue,
        action: &DatasetActionSchema,
        transport: &crate::engine::registry::TransportDescriptor,
        endpoint: &crate::endpoints::EndpointDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingActionSinkConfig>, ActionExecutionError>;
    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingActionSinkConfig>,
        context: &StreamingActionSinkPrepareContext,
    ) -> Result<PreparedStreamingActionBinding, ActionExecutionError>;
}

#[async_trait::async_trait(?Send)]
pub trait StreamingActionSubmitter {
    fn accepted_schema(&self) -> DatasetActionSchema;
    async fn submit(&mut self, action: OrderedDatasetAction)
        -> Result<SubmittedAction, ActionExecutionError>;
}

#[async_trait::async_trait(?Send)]
pub trait StreamingActionDriver: StreamingCheckpointParticipant {
    async fn next_event(&mut self) -> Result<ActionExecutionEvent, ActionExecutionError>;
    async fn drain(&mut self) -> Result<ActionDrainReceipt, ActionExecutionError>;
}

#[async_trait::async_trait(?Send)]
pub trait StreamingActionDriverControl {
    fn stop_issuing(&self);
    fn cancel_pending(&self);
    async fn cancel_inflight(&self) -> Result<ActionCancelReceipt, ActionExecutionError>;
}

#[async_trait::async_trait(?Send)]
pub trait StreamingCheckpointParticipant {
    fn participant_id(&self) -> CheckpointParticipantId;
    async fn checkpoint_view(&mut self, barrier: &CheckpointBarrier)
        -> Result<PreparedParticipantState, CheckpointError>;
    async fn initialize(&mut self, state: Option<CommittedParticipantState>)
        -> Result<(), CheckpointError>;
    async fn checkpoint_committed(&mut self, receipt: &CommittedParticipantReceipt)
        -> Result<(), CheckpointError>;
}

pub trait StreamingCheckpointBackendFactory: std::fmt::Debug + Send + Sync {
    fn descriptor(&self) -> &'static StreamingCheckpointBackendDescriptor;
    fn validate(
        &self,
        authored: &serde_json::value::RawValue,
        requirements: &CheckpointBackendRequirements,
    ) -> Result<Box<dyn ValidatedCheckpointBackendConfig>, CheckpointError>;
    fn prepare(
        &self,
        config: Box<dyn ValidatedCheckpointBackendConfig>,
        context: &CheckpointBackendPrepareContext,
    ) -> Result<Box<dyn StreamingCheckpointBackend>, CheckpointError>;
}

#[async_trait::async_trait(?Send)]
pub trait StreamingCheckpointBackend {
    async fn open_latest(&self, run: &StreamRunIdentity)
        -> Result<Option<Box<dyn LeasedGenerationReader>>, CheckpointError>;
    async fn begin_generation(
        &self,
        expected: Option<CheckpointGeneration>,
    ) -> Result<Box<dyn StreamingGenerationTransaction>, CheckpointError>;
}

#[async_trait::async_trait(?Send)]
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

#[async_trait::async_trait(?Send)]
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

Also define `StreamingCheckpointBackend`, `LeasedGenerationReader`, and `StreamingGenerationTransaction` exactly as the spec, including `open_latest`, `begin_generation`, paged result index reads, participant state reads, immutable result-segment reads, staging participant/result data, and CAS commit. Put `ResultPartition`, `PreparedResultEpoch`, `ResultIndexCursor`, `ResultIndexReadBudget`, `ResultIndexPage`, `ResultSegmentDescriptor`, and `ResultSegmentReader` in `checkpoint.rs` now so later results code does not create a cycle.

- [ ] **Step 1: Add compile-time RED contract checks**

```rust
use aiperf_runtime::streaming::{action::*, checkpoint::*, format::*, session::*, source::*};

fn assert_factory<T: std::fmt::Debug + Send + Sync + ?Sized>() {}
fn assert_object_safe(_: Box<dyn StreamingCheckpointParticipant>) {}

#[test]
fn factories_have_host_safe_bounds() {
    assert_factory::<dyn StreamingDatasetSourceFactory>();
    assert_factory::<dyn StreamingDatasetFormatFactory>();
    assert_factory::<dyn StreamingSessionProgramFactory>();
    assert_factory::<dyn StreamingActionSinkFactory>();
    assert_factory::<dyn StreamingCheckpointBackendFactory>();
}

#[allow(dead_code)]
fn action_binding_is_split(binding: PreparedStreamingActionBinding) {
    let _: Box<dyn StreamingActionSubmitter> = binding.submitter;
    let _: Box<dyn StreamingActionDriver> = binding.driver;
    let _: Box<dyn StreamingActionDriverControl> = binding.control;
}
```

- [ ] **Step 2: Run the task suite and verify RED**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd rust
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_contracts
```

- [ ] **Step 3: Define the contracts and explicit errors**

Add module declarations to `streaming.rs`. Every factory-owned validated configuration exposes only `as_any`/`into_any`, mirroring engine validated configs. Every runtime trait is `?Send`; prepared factories remain `Send + Sync`. `StreamingStopReceiver` and each driver control are separately cloneable/borrowable so a pending `&mut self` future can be woken without aliasing the stage owner.

- [ ] **Step 4: Run the suite and verify GREEN**

Run the Step 2 command.

- [ ] **Step 5: Review and commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/source.rs rust/runtime/src/streaming/format.rs rust/runtime/src/streaming/session.rs rust/runtime/src/streaming/action.rs rust/runtime/src/streaming/checkpoint.rs rust/runtime/tests/streaming_contracts.rs
git commit -m "feat(runtime): define streaming extension contracts"
```

### Task 1E: Reusable Source and Format Conformance Harnesses

**Files:**
- Create: `rust/runtime/tests/support/streaming_source_conformance.rs`
- Create: `rust/runtime/tests/support/streaming_format_conformance.rs`
- Create: `rust/runtime/tests/streaming_contract_conformance.rs`

**Interfaces:**
- Consumes: Task 1D public contracts.
- Produces:

```rust
pub async fn assert_source_conformance(factory: &dyn StreamingDatasetSourceFactory, cases: SourceConformanceCases);
pub async fn assert_format_conformance(factory: &dyn StreamingDatasetFormatFactory, cases: FormatConformanceCases);
```

- [ ] **Step 1: Add RED fake adapters and harness calls**

At the top of `streaming_contract_conformance.rs`, load support explicitly:

```rust
#[path = "support/streaming_source_conformance.rs"]
mod streaming_source_conformance;
#[path = "support/streaming_format_conformance.rs"]
mod streaming_format_conformance;
```

Create a fake source whose scripted events are `Pending`, one immutable partition, duplicate rediscovery, frontier, and seal; its control handle must wake the pending `next_event`. Create a fake decoder that emits one leased batch, blocks while its output lease is held, resumes from the exact cursor, then returns `DecodeStep::End`.

Representative harness assertion:

```rust
pub async fn assert_pending_is_not_seal(mut opened: OpenedStreamingDatasetSource) {
    let pending = opened.source.next_event();
    tokio::pin!(pending);
    assert!(futures::poll!(&mut pending).is_pending());
    opened.control.stop();
    let error = pending.await.expect_err("stop wakes pending source");
    assert!(matches!(error, StreamSourceError::Stopped));
}
```

- [ ] **Step 2: Run the task suite and verify RED**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd rust
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_contract_conformance
```

Expected: fake adapters demonstrate missing reusable harness functions.

- [ ] **Step 3: Implement the reusable harness functions**

Cover pending-versus-seal, stop wakeup, immutable identity, mutation refusal, backpressure, lease lifetime, exact cursor restore, duplicate replay, and idempotent post-commit notification. Harnesses accept factories and case data; they do not inspect concrete adapter types.

- [ ] **Step 4: Run the suite and verify GREEN**

Run the Step 2 command.

- [ ] **Step 5: Review and commit**

```bash
git add rust/runtime/tests/support/streaming_source_conformance.rs rust/runtime/tests/support/streaming_format_conformance.rs rust/runtime/tests/streaming_contract_conformance.rs
git commit -m "test(runtime): define streaming adapter conformance"
```

### Task 2: Frozen Streaming Registries and Capability Agreement

**Files:**
- Modify: `rust/runtime/src/extensions/mod.rs:348-430`
- Modify: `rust/runtime/src/engine/registry.rs:48-153`
- Modify: `rust/runtime/src/engine/registry.rs:559-735`
- Modify: `rust/runtime/src/engine/protocol.rs:63-135`
- Test: `rust/runtime/tests/streaming_registry.rs`
- Test: `rust/runtime/tests/extensions_compile_time_extension.rs`

**Interfaces:**
- Consumes: five Task 1D factory traits and `TransactionalRegistry<T>`.
- Produces registry methods:

```rust
pub fn register_stream_source(&mut self, factory: Arc<dyn StreamingDatasetSourceFactory>) -> Result<()>;
pub fn register_stream_format(&mut self, factory: Arc<dyn StreamingDatasetFormatFactory>) -> Result<()>;
pub fn register_stream_session_program(&mut self, factory: Arc<dyn StreamingSessionProgramFactory>) -> Result<()>;
pub fn register_stream_action_sink(&mut self, factory: Arc<dyn StreamingActionSinkFactory>) -> Result<()>;
pub fn register_stream_checkpoint_backend(&mut self, factory: Arc<dyn StreamingCheckpointBackendFactory>) -> Result<()>;
pub fn stream_source_factory(&self, id: &str) -> Option<Arc<dyn StreamingDatasetSourceFactory>>;
pub fn stream_format_factory(&self, id: &str) -> Option<Arc<dyn StreamingDatasetFormatFactory>>;
pub fn stream_session_program_factory(&self, id: &str) -> Option<Arc<dyn StreamingSessionProgramFactory>>;
pub fn stream_action_sink_factory(&self, id: &str) -> Option<Arc<dyn StreamingActionSinkFactory>>;
pub fn stream_checkpoint_backend_factory(&self, id: &str) -> Option<Arc<dyn StreamingCheckpointBackendFactory>>;
pub fn stream_source_descriptors(&self) -> Vec<&'static StreamingSourceDescriptor>;
pub fn stream_format_descriptors(&self) -> Vec<&'static StreamingFormatDescriptor>;
pub fn stream_session_program_descriptors(&self) -> Vec<&'static StreamingSessionProgramDescriptor>;
pub fn stream_action_sink_descriptors(&self) -> Vec<&'static StreamingActionSinkDescriptor>;
pub fn stream_checkpoint_backend_descriptors(&self) -> Vec<&'static StreamingCheckpointBackendDescriptor>;
```

`StreamingCapabilityAgreement::validate(...)` accepts descriptors only and performs no preparation or I/O.

- [ ] **Step 1: Add RED registry tests**

Add a test-local `FakeSourceFactory` implementing the Task 1D trait, then:

```rust
#[test]
fn duplicate_stream_source_registration_is_atomic() {
    let mut registry = AIPerfRegistry::empty_or_base();
    registry.register_stream_source(Arc::new(FakeSourceFactory::new("fake")))
        .expect("first registration");
    let error = registry
        .register_stream_source(Arc::new(FakeSourceFactory::new("FAKE")))
        .expect_err("normalized duplicate must fail");
    assert!(error.to_string().contains("duplicate streaming source ID"));
    assert_eq!(registry.stream_source_descriptors().len(), 1);
}
```

Add ordered-inventory, unknown lookup, transactional extension rollback, cross-product mismatch, and catalog serialization tests. Extend the compile-time extension test with one custom factory in every category.

- [ ] **Step 2: Run the single task suite and verify RED**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd rust
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_registry --test extensions_compile_time_extension
```

- [ ] **Step 3: Add the five transactional fields**

In `AIPerfRegistry`, add five `TransactionalRegistry<Arc<dyn ...>>` fields under `#[cfg(feature = "streaming")]`, initialize them in `empty_or_base`, clone them with the registry, and register only factories supplied by real built-in or external extensions. Do not add source IDs here.

- [ ] **Step 4: Add feature-accurate catalog fields**

Extend `Catalog` with `stream_source`, `stream_format`, `stream_session_program`, `stream_action_sink`, and `stream_checkpoint_backend` maps under `#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]`. Populate them solely from frozen registry descriptors. Preserve existing serialized bytes when all maps are empty.

- [ ] **Step 5: Implement capability agreement**

Validate source mode/access/order/resume, format media/access/projection/output schema, session input/emitted actions/closure, action accepted schema and transport/endpoint, backend durability/readers/sensitive state, and report retention. Return an error containing every selected descriptor ID and the first incompatible capability.

- [ ] **Step 6: Run the suite and verify GREEN**

Run the Step 2 command.

- [ ] **Step 7: Review and commit**

```bash
git add rust/runtime/src/extensions/mod.rs rust/runtime/src/engine/registry.rs rust/runtime/src/engine/protocol.rs rust/runtime/tests/streaming_registry.rs rust/runtime/tests/extensions_compile_time_extension.rs
git commit -m "feat(runtime): register streaming capabilities"
```

### Task 3: Protocol-v2 Dataset Streams and Shadow-Replay Configuration

**Files:**
- Create: `rust/runtime/src/config/model/dataset_stream.rs`
- Modify: `rust/runtime/src/config/model/config.rs:63-125`
- Modify: `rust/runtime/src/config/model/mod.rs:10-30`
- Modify: `rust/runtime/src/config/model/workload_kind.rs:49-115`
- Modify: `rust/runtime/src/config/resolve.rs`
- Modify: `rust/runtime/src/config/validate.rs`
- Modify: `rust/runtime/src/engine/protocol_v2.rs:301-530`
- Modify: `rust/runtime/src/engine/protocol_v2.rs:635-830`
- Modify: `rust/runtime/src/engine/registry.rs:90-153`
- Modify: `rust/cli/src/load.rs`
- Modify: `rust/cli/src/yaml.rs`
- Test: `rust/runtime/tests/streaming_protocol_v2.rs`

**Interfaces:**
- Consumes: Task 2 lookups and `StreamingCapabilityAgreement`.
- Produces strict Config-v2 types `DatasetStreams`, `DatasetStream`, `StreamingComponent`, `StreamLimits`, and `ShadowReplay`; Protocol-v2 `DatasetStreamsSpecV2`; `RunResourceV2::DatasetStreams`; `ResourceRequirementsV2::shadow_replay()`.
- Does not register a stock `shadow_replay` workload or placeholder factory.

Representative Config-v2 types:

```rust
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingComponent {
    pub id: String,
    #[serde(default)]
    pub config: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetStream {
    pub id: String,
    pub source: StreamingComponent,
    pub format: StreamingComponent,
    pub session_program: StreamingComponent,
    pub limits: StreamLimits,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowReplay {
    pub stream: String,
    pub actions: std::collections::BTreeMap<DatasetActionSchema, StreamingComponent>,
    pub time: ReplayTimeConfig,
    pub ordering: OrderingConfig,
    pub overload: OverloadConfig,
    pub checkpoint: CheckpointConfig,
}
```

- [ ] **Step 1: Add a RED normative projection test**

```rust
#[test]
fn dataset_stream_resource_projects_without_opening_factories() {
    let yaml = r#"
datasets: null
dataset_streams:
  items:
    - id: shadow_input
      source: { id: local, config: { mode: follow, path: /traces } }
      format: { id: jsonl, config: { schema: aiperf.session_fragment.v1 } }
      session_program: { id: conversation, config: {} }
      limits:
        acquired_partitions: 2
        decoded_fragments: 32
        decoded_bytes: 4096
        state_memory: 4096
        state_disk: 8192
shadow_replay:
  stream: shadow_input
  actions:
    request: { id: scheduled_request, config: {} }
    session_terminal: { id: session_state, config: {} }
  time: { mode: relative }
  ordering: { watermark: source_order, late: fail }
  overload: { mode: backpressure }
  checkpoint: { mode: none }
"#;
    let cfg: aiperf_runtime::config::model::BenchmarkConfig =
        serde_yaml::from_str(yaml).expect("strict config");
    assert_eq!(cfg.dataset_streams.as_ref().expect("streams").items.len(), 1);
    assert_eq!(aiperf_runtime::config::model::workload_kind(&cfg).workload_id(), "shadow_replay");
}
```

Add unknown-field, duplicate stream/action, mixed finite+streaming, missing stream, invalid limits/durations, accuracy refusal, resident exporter refusal, feature-unavailable lookup, and effect-counter tests.

- [ ] **Step 2: Run the task suite and verify RED**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd rust
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_protocol_v2
```

- [ ] **Step 3: Add strict Config-v2 projection**

Add `dataset_streams: Option<DatasetStreams>` and `shadow_replay: Option<ShadowReplay>` to `BenchmarkConfig`. Add `WorkloadKind::ShadowReplay`; mixed `datasets` and `dataset_streams` is a validation error, not precedence. CLI YAML loading must preserve the typed objects without source-specific flags.

- [ ] **Step 4: Add Protocol-v2 resource presence**

Extend `AuthoredRunResourcesV2`, `AuthoredRunSpecV2`, `ResourcePresenceV2`, `RunResourceV2`, `field_name`, `validate_outer`, and `resource_is_present`. Extend `ResourceRequirementsV2::entries` from five to six entries. Keep `inference()` stream-forbidden and add:

```rust
pub const fn shadow_replay() -> Self {
    Self {
        models: ResourceRequirementV2::Required,
        endpoints: ResourceRequirementV2::Required,
        metrics: ResourceRequirementV2::Optional,
        artifacts: ResourceRequirementV2::Optional,
        sidecars: ResourceRequirementV2::Optional,
        dataset_streams: ResourceRequirementV2::Required,
    }
}
```

- [ ] **Step 5: Validate with a test-local workload only**

The integration test defines `FakeShadowReplayWorkloadFactory` and returns `ResourceRequirementsV2::shadow_replay()`. Production projects workload ID/config but does not add it to `register_online_workloads`; stock catalog absence is asserted. Validation resolves each named stream component once, compares descriptors without I/O, and freezes semantic digests.

- [ ] **Step 6: Run the suite and verify GREEN**

Run the Step 2 command.

- [ ] **Step 7: Review and commit**

```bash
git add rust/runtime/src/config/model/dataset_stream.rs rust/runtime/src/config/model/config.rs rust/runtime/src/config/model/mod.rs rust/runtime/src/config/model/workload_kind.rs rust/runtime/src/config/resolve.rs rust/runtime/src/config/validate.rs rust/runtime/src/engine/protocol_v2.rs rust/runtime/src/engine/registry.rs rust/cli/src/load.rs rust/cli/src/yaml.rs rust/runtime/tests/streaming_protocol_v2.rs
git commit -m "feat(engine): add dataset stream resources"
```

### Task 4A: Bounded Scheduled-Runtime Terminal Lane and Session Identity

**Files:**
- Create: `rust/runtime/src/streaming/terminal_lane.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Modify: `rust/runtime/src/scheduled.rs:487-635`
- Modify: `rust/runtime/src/scheduled.rs:950-1027`
- Modify: `rust/runtime/src/scheduled.rs:1099-1249`
- Modify: `rust/runtime/src/scheduled.rs:875-915`
- Modify: `rust/runtime/src/phase_runtime.rs:898-915`
- Test: `rust/runtime/tests/streaming_terminal_lane.rs`

**Interfaces:**
- Consumes: `BudgetLease`, existing `TurnRecordProcessor`, `IssuedCredit`, and `TurnDispatchOutcome`.
- Produces `TerminalLaneLimits`, `TerminalLanePermit`, `BoundedTerminalProcessorLane`, `TerminalLaneControl`, `TerminalLaneSnapshot`, `ScheduledSessionIdentity`, and a new opt-in issue method. Existing issue methods retain their signatures.

```rust
pub struct ScheduledSessionIdentity { pub stable_ordinal: u64 }

impl ScheduledRuntime {
    pub async fn reserve_terminal_processing(
        &self,
        estimated_bytes: usize,
    ) -> Result<TerminalLanePermit>;

    #[allow(clippy::too_many_arguments)]
    pub fn issue_turn_with_streaming_identity(
        self: &Rc<Self>,
        turn: TurnToSend,
        scheduled_ns: i64,
        user_id: Option<u64>,
        session: ScheduledSessionIdentity,
        terminal_permit: TerminalLanePermit,
        on_first_token: FirstTokenHandler,
        on_complete: CompletionHandler,
        cancellation: Option<Rc<dyn DispatchCancellation>>,
    ) -> bool;
}
```

- [ ] **Step 1: Add RED boundedness tests**

```rust
#[tokio::test(flavor = "current_thread")]
async fn record_count_does_not_increase_drain_task_count() {
    let lane = BoundedTerminalProcessorLane::new_for_test(TerminalLaneLimits {
        max_items: 4,
        max_bytes: 4096,
    }).expect("lane");
    let control = lane.control();
    lane.start_local_drain().expect("one drain owner");
    for index in 0..100_000_u64 {
        lane.submit_test_terminal(index, 1).await.expect("bounded submit");
    }
    control.close();
    control.drain().await.expect("drain");
    let snapshot = control.snapshot();
    assert_eq!(snapshot.drain_tasks_started, 1);
    assert_eq!(snapshot.queued_items, 0);
    assert!(snapshot.high_water_items <= 4);
}
```

Add full-lane backpressure, first-error latch/wakeup, cancellation permit return, 100,000 one-turn sessions with zero active map entries, stable external ordinal, and finite-default compatibility tests.

- [ ] **Step 2: Run the task suite and verify RED**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd rust
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_terminal_lane
```

- [ ] **Step 3: Implement reservation-before-issue**

The streaming caller asynchronously reserves item+estimated bytes before the synchronous issue call. Settlement moves that permit into `TerminalWork`; `submit` cannot block or fail for capacity after reservation. One `spawn_local` drain owner invokes the existing processors in order. It retains the first typed failure, counts later failures, wakes the phase owner, and never stores a per-record `JoinHandle` or error string.

- [ ] **Step 4: Bound active session numbering without changing finite IDs**

Replace `sessions.len()` allocation with `next_session_number: Cell<u64>` plus an active map. Remove both `session_numbers` and `session_url_indices` on `credit.is_final_turn()`. Existing issue methods allocate monotonic finite ordinals; the streaming method uses `ScheduledSessionIdentity::stable_ordinal` directly and does not insert it into the lifetime map.

- [ ] **Step 5: Join the lane through phase finalization**

Replace `wait_record_processors` with a mode-aware drain: legacy finite mode reaps its existing tasks continuously; streaming mode closes and drains `TerminalLaneControl`. Surface the first lane error before report construction.

- [ ] **Step 6: Run the suite and verify GREEN**

Run the Step 2 command. During review also run existing `scheduled_sim` and `phase_runtime_sim`; do not add those as a second task-suite command.

- [ ] **Step 7: Review and commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/terminal_lane.rs rust/runtime/src/scheduled.rs rust/runtime/src/phase_runtime.rs rust/runtime/tests/streaming_terminal_lane.rs
git commit -m "refactor(runtime): bound streaming terminal processing"
```

### Task 4B: Reusable Phase/Capture Construction Service

**Files:**
- Create: `rust/runtime/src/engine/execute/capture_service.rs`
- Modify: `rust/runtime/src/engine/execute/mod.rs`
- Modify: `rust/runtime/src/engine/execute/capture.rs:127-240`
- Modify: `rust/runtime/src/engine/execute/compose_sidecars.rs:350-420`
- Modify: `rust/runtime/src/engine/execute/sharding.rs:400-500`
- Modify: `rust/runtime/src/engine/execute/entrypoints.rs`
- Modify: `rust/runtime/src/engine/sharded_scheduled.rs`
- Modify: `rust/runtime/src/phase_runtime.rs:898-1050`
- Unit test: `rust/runtime/src/engine/execute/capture_service.rs`
- Test: `rust/runtime/tests/streaming_phase_runtime.rs`

**Interfaces:**
- Consumes: Task 4A and current `RunCapture`, `ConfiguredDispatcher`, observer tee, `ScheduledPhaseExecutionFactory`.
- Produces crate-private typed construction:

```rust
pub(crate) struct RunCapturePolicy {
    pub is_raw_enabled: bool,
    pub needs_live_record: bool,
    pub needs_adaptive_record: bool,
    pub is_exact_fold: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CaptureMode { Retained, ExactFold, Sketch }

impl RunCapturePolicy {
    pub(crate) fn capture_mode(
        self,
        storage: &crate::metrics_core::MetricsStorageMode,
    ) -> CaptureMode;
}

pub(crate) struct PreparedCaptureService {
    pub capture: Rc<RunCapture>,
    pub dispatcher: Rc<dyn TurnDispatcher>,
}

pub(crate) fn prepare_capture_service(
    request: CaptureServiceRequest,
) -> Result<PreparedCaptureService>;
```

`CaptureServiceRequest` owns the exact existing dispatcher inputs: clock, origin, metrics config, issuance authority, phase ordinal bases, executor factory, prepared endpoints, transport policy, artifact lane, OTEL accumulator, output capture, and worker label. It has no source/format/session fields.

- [ ] **Step 1: Add characterization tests through public behavior**

In `streaming_phase_runtime.rs`, characterize retained, exact-fold, and sketch summary bytes; exactly-once first-token/terminal observations; phase labels and ordinal bases; and sidecar setup/finalize order. Use the existing public scheduled report outputs, not access to private fields. These assertions must pass before extraction and remain byte-identical afterward.

- [ ] **Step 2: Add the structural RED unit test beside the new service**

Declare `pub(crate) mod capture_service;` in `execute/mod.rs`, create `capture_service.rs`, and add this unit test before defining the production types:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_maps_to_the_existing_capture_modes() {
        let policy = RunCapturePolicy {
            is_raw_enabled: false,
            needs_live_record: false,
            needs_adaptive_record: false,
            is_exact_fold: true,
        };
        assert_eq!(
            policy.capture_mode(&crate::metrics_core::MetricsStorageMode::Exact),
            CaptureMode::ExactFold,
        );
    }
}
```

This is a crate unit test, so it may exercise crate-private construction without widening the product API. RED is an unresolved `RunCapturePolicy`/`CaptureMode`.

- [ ] **Step 3: Run the single task suite and verify RED**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd rust
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --lib --test streaming_phase_runtime
```

- [ ] **Step 4: Replace boolean constructor arguments with policy**

Implement this delegation first and update every caller:

```rust
impl RunCapture {
    pub(crate) fn from_policy(
        clock: Rc<dyn Clock>,
        origin_ns: i64,
        config: MetricsConfig,
        policy: RunCapturePolicy,
    ) -> Self {
        Self::new(
            clock,
            origin_ns,
            config,
            policy.is_raw_enabled,
            policy.needs_live_record,
            policy.needs_adaptive_record,
            policy.is_exact_fold,
        )
    }
}
```

Once all callers use `RunCapturePolicy`, make the old positional constructors private to `capture.rs`.

- [ ] **Step 5: Move shared construction into `capture_service.rs`**

Move, without semantic edits, the duplicated `RunCapture` plus `ConfiguredDispatcher` assembly from `compose_sidecars.rs` and `execute/sharding.rs` into `prepare_capture_service`. Both finite call sites build `CaptureServiceRequest` and consume the returned pair. No streaming workload is added.

- [ ] **Step 6: Extract phase runtime construction**

Introduce a crate-private `ScheduledPhaseRuntimeBuilder` trait whose default implementation contains the current collector/native-metrics/observer/runtime construction from `ScheduledPhaseExecutionFactory::create`. Its result owns `Rc<ScheduledRuntime>` and finalization controls. Preserve finalization order: scheduler idle, credit-return drain, sidecars, terminal lane/processors, report.

- [ ] **Step 7: Run the suite and verify GREEN**

Run the Step 3 command. Existing phase-runtime online/simulated tests are mandatory review evidence.

- [ ] **Step 8: Review and commit**

```bash
git add rust/runtime/src/engine/execute/capture_service.rs rust/runtime/src/engine/execute/mod.rs rust/runtime/src/engine/execute/capture.rs rust/runtime/src/engine/execute/compose_sidecars.rs rust/runtime/src/engine/execute/sharding.rs rust/runtime/src/engine/execute/entrypoints.rs rust/runtime/src/engine/sharded_scheduled.rs rust/runtime/src/phase_runtime.rs rust/runtime/tests/streaming_phase_runtime.rs
git commit -m "refactor(engine): expose reusable phase capture service"
```

### Task 7A: UTC/Monotonic Authority, Event Ordering, and Near-Horizon Scheduling

**Dependencies:** Tasks 1A–1D and master Task 5A must be merged. This task implements `StreamingCheckpointParticipant` for event-time state; it must not invent a temporary checkpoint shape.

**Files:**
- Modify: `rust/runtime/src/clock/runtime_clock.rs:11-45`
- Modify: `rust/runtime/src/clock/real_clock.rs:20-80`
- Modify: `rust/runtime/src/clock/sim_clock.rs:145-180`
- Modify: `rust/runtime/src/clock/mod.rs:8-15`
- Create: `rust/runtime/src/streaming/event_time.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Test: `rust/runtime/tests/streaming_event_time.rs`

**Interfaces:**
- Consumes: `Clock`, stable IDs/order, budgets, checkpoint participant and barrier.
- Produces:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct UtcMonotonicAnchor {
    pub utc_ns: i64,
    pub monotonic_ns: i64,
    pub uncertainty_ns: i64,
}

pub trait Clock {
    fn capture_utc_anchor(
        &self,
        authored_utc_epoch_ns: Option<i64>,
        max_uncertainty_ns: i64,
    ) -> Result<UtcMonotonicAnchor, ClockAnchorError>;
}

pub struct ReplayTimeMapping;
impl ReplayTimeMapping {
    pub fn new(anchor: UtcMonotonicAnchor, replay_delay_ns: i64) -> Result<Self, EventTimeError>;
    pub fn target_ns(&self, event_time: EventTimeUtc) -> Result<i64, EventTimeError>;
}

pub trait EventTimePolicy: StreamingCheckpointParticipant {
    fn observe(&mut self, action: &ExecutableDatasetAction)
        -> Result<Vec<OrderedDatasetAction>, EventTimeError>;
    fn advance(&mut self, frontier: EventTimeWatermark)
        -> Result<Vec<OrderedDatasetAction>, EventTimeError>;
    fn seal(&mut self) -> Result<Vec<OrderedDatasetAction>, EventTimeError>;
}
```

- [ ] **Step 1: Add RED anchor and mapping tests**

```rust
use aiperf_runtime::clock::{Clock, ClockAnchorError, UtcMonotonicAnchor};
use aiperf_runtime::streaming::event_time::ReplayTimeMapping;
use aiperf_runtime::streaming::unit::EventTimeUtc;

#[test]
fn one_anchor_is_reused_for_the_run() {
    let anchor = UtcMonotonicAnchor {
        utc_ns: 1_000_000_000,
        monotonic_ns: 50,
        uncertainty_ns: 2,
    };
    let mapping = ReplayTimeMapping::new(anchor, 300).expect("mapping");
    let event = EventTimeUtc::new(1_000_000_100).expect("event time");
    assert_eq!(mapping.target_ns(event).expect("target"), 450);
    assert_eq!(mapping.target_ns(event).expect("same target"), 450);
}

#[test]
fn virtual_clock_requires_authored_utc_epoch() {
    let clock = aiperf_runtime::clock::SimClock::new();
    let error = clock.capture_utc_anchor(None, 10).expect_err("missing epoch");
    assert!(matches!(error, ClockAnchorError::AuthoredEpochRequired));
}
```

Add uncertainty rejection, overflow, negative checked arithmetic, immutability after a fake system-clock adjustment, and late-target classification tests.

- [ ] **Step 2: Run the task suite and verify RED**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd rust
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_event_time
```

- [ ] **Step 3: Add clock-owned anchor capture**

Add a default `Clock::capture_utc_anchor` that accepts an authored epoch only for virtual clocks and otherwise returns `ClockAnchorError::Unsupported`. Override it in `RealClock`: read `now_ns`, then `SystemTime::now`, then `now_ns`; use the checked monotonic midpoint and half-bracket width as uncertainty; reject a negative maximum or excessive uncertainty. No streaming file imports `SystemTime`.

Representative virtual default:

```rust
fn capture_utc_anchor(
    &self,
    authored_utc_epoch_ns: Option<i64>,
    max_uncertainty_ns: i64,
) -> Result<UtcMonotonicAnchor, ClockAnchorError> {
    if max_uncertainty_ns < 0 {
        return Err(ClockAnchorError::InvalidUncertainty(max_uncertainty_ns));
    }
    if !self.is_virtual() {
        return Err(ClockAnchorError::Unsupported);
    }
    let utc_ns = authored_utc_epoch_ns.ok_or(ClockAnchorError::AuthoredEpochRequired)?;
    Ok(UtcMonotonicAnchor { utc_ns, monotonic_ns: self.now_ns(), uncertainty_ns: 0 })
}
```

- [ ] **Step 4: Implement event ordering and late policy**

Use a budgeted min-order heap keyed by `(EventTimeUtc, StableOrderKey)`. Hard frontiers never regress. Bounded disorder computes `max_seen - max_out_of_order` with checked arithmetic. Implement distinct typed outcomes for `Fail`, `IssueImmediately`, `Drop`, and bounded catch-up. Persist policy kind, max seen, watermark, sequence frontier, and heap membership through the Task 5A participant contract.

- [ ] **Step 5: Implement bounded near-horizon admission**

`NearHorizonBuffer::pop_admissible(now_ns)` releases only actions whose target is at most `now_ns + schedule_horizon_ns`; later actions remain in the single budgeted heap. The caller waits with `ScheduledRuntime::wait_until_or_stop`; this task creates no task per buffered action. Record publication lag, decode lag, watermark wait, causal wait, scheduling lateness, and endpoint latency as distinct typed fields.

- [ ] **Step 6: Run the suite and verify GREEN**

Run the Step 2 command.

- [ ] **Step 7: Review and commit**

```bash
git add rust/runtime/src/clock/runtime_clock.rs rust/runtime/src/clock/real_clock.rs rust/runtime/src/clock/sim_clock.rs rust/runtime/src/clock/mod.rs rust/runtime/src/streaming.rs rust/runtime/src/streaming/event_time.rs rust/runtime/tests/streaming_event_time.rs
git commit -m "feat(runtime): map streaming event time through clock authority"
```

## Completion Gate for This Subsystem Plan

After Task 7A is merged, run one consolidated foundation gate from `rust/`:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_feature_inventory --test streaming_identity --test streaming_budget --test streaming_blocking --test streaming_contracts --test streaming_contract_conformance --test streaming_registry --test extensions_compile_time_extension --test streaming_protocol_v2 --test streaming_terminal_lane --test streaming_phase_runtime --test streaming_event_time
```

Then run `cargo fmt --check` and `cargo clippy -p aiperf-runtime --all-targets --features streaming -- -D warnings` as review gates. Confirm the stock catalog still lacks `shadow_replay`, `s3`, and `object_store` factories; `streaming-s3` only enables dependencies and later adapter compilation authority. Confirm no file outside the paths named in this plan changed.

## Self-Review Checklist

- Spec invariants covered here: stable topology-independent identity; typed source/format/session/action/checkpoint seams; bounded memory and blocking ownership; feature-accurate absence; strict resource ownership; cross-format capability agreement; bounded terminal processing; reusable phase/capture construction; one immutable UTC anchor; deterministic event-time ordering; no task per far-future action.
- Deferred intentionally to later subsystem plans: checkpoint storage implementation, result segments/compaction, session state machines, pipeline execution, concrete local/HF/Baseten/Dynamo/S3 adapters, executable shadow workload, graph action sink, sensitive state, and cellular execution.
- Type consistency: Task 1D owns all factory and checkpoint I/O vocabulary; Task 2 registers those exact traits; Task 3 references their descriptor lookups; Task 4A consumes Task 1B permits; Task 7A consumes Task 1A IDs, Task 1B budget, and Task 5A checkpoint cuts.
- Placeholder scan: production registration is deliberately absent until executable implementations exist; no task asks for a rejecting placeholder, temporary workload, source-format switch, or `NativeDatasetPlan` variant.
