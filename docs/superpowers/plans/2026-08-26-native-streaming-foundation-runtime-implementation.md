<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Streaming Foundation and Runtime Seams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the feature gates, typed streaming vocabulary, resource ownership, object-safe extension contracts, host-owned reliability continuation authority, frozen registries, strict Protocol-v2 configuration, bounded scheduled-runtime terminal processing, reusable phase/capture construction, and UTC/event-time authority required by every later native streaming dataset and shadow-replay subsystem.

**Architecture:** This plan builds only the foundation and existing-runtime adaptations from master Tasks 0, 1A–1E, 1D-R, 2, 3, 4A–4B, and 7A. The lightweight `streaming` feature owns all host contracts and local execution prerequisites, including deterministic scoped issue classification and disposition; `streaming-s3` adds only AWS dependencies and advertises no S3 factory until its later executable adapter lands. Existing finite execution remains the reference path and is migrated onto reusable seams without adding a `NativeDatasetPlan::Streaming` variant or a non-executable `shadow_replay` factory.

**Tech Stack:** Rust 2024, Tokio current-thread runtimes and `LocalSet`, `async_trait(?Send)`, BLAKE3, strict Serde DTOs, `chacha20poly1305` and `zeroize` behind `streaming`, optional `aws-config` and `aws-sdk-s3` behind `streaming-s3`, existing `Clock`, `ScheduledRuntime`, `RunCapture`, `TransactionalRegistry`, Protocol v2, and Config v2.

**Spec:** `artifacts/streaming-design/streaming-dataset-shadow-replay-design.md` at base approval `505efc06b0`, amended by `3fea6f2fe0` and `artifacts/streaming-design/reliability-continuation-course-correction.md`; master plan: `docs/superpowers/plans/2026-08-26-native-streaming-datasets-shadow-replay-implementation.md`.

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
- Ordinary partition, record, session, action, export, and checkpoint-attempt faults are host-classified scoped issues. Only the authority/invariant boundary in the reliability course correction may select `FailRun`.
- Every public item has `///` documentation. Every new Rust file has exactly the two NVIDIA SPDX lines and `//!` module documentation.
- Each task has one focused test-suite invocation, two reviews, one focused commit, and no unrelated changes.
- Cargo commands run from the nested `rust/` workspace after activating `/home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate`; git commands run from the repository root.
- Each task includes the nearest parent module declaration required for its own GREEN build; declaration conflicts are resolved during integration.
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
rust/runtime/src/streaming/reliability.rs     scoped issue policy, receipts, thresholds, checkpoint ledger
rust/runtime/src/streaming/terminal_lane.rs   bounded scheduled terminal processing
rust/runtime/src/streaming/event_time.rs      UTC mapping, watermark, late policy, horizon
rust/runtime/src/config/model/dataset_stream.rs strict public Config-v2 types
rust/runtime/src/engine/execute/capture_service.rs reusable finite/streaming construction
```

The cross-plan foundation order is `0 → 1A → 1B → checkpoint 5A → 1C → checkpoint 5A-R → checkpoint 5B → 1D → 1D-R → 1E`. Task 5A lands the participant vocabulary used by the blocking owner; Task 5A-R then binds the landed checkpoint and blocking-participant APIs to one logical run; Task 5B lands the run-scoped backend vocabulary consumed by the remaining contracts. Task 1D-R binds typed failures to host-owned reliability dispositions without adding adapter-specific behavior. After Task 1E merges, Tasks 2 and 4A may run in parallel. Task 3 depends on Tasks 2 and 1D-R. Task 4B depends on Task 4A. Task 7A depends on Tasks 1D-R and 5A-R.

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
- Produces: checked `EventTimeUtc`, `SourcePosition`, `GlobalSequence`, `ImmutableObjectIdentity`, `StableRecordId`, `StableSessionKey`, `StableActionId`, `ActionAttemptId`, `LogicalReplayRunId`, `RunIncarnationId`, `StableOrderKey`, `SessionCausalFrontier`, `SessionOwnershipEpoch`, `StateBudgetFailureCode`, `UnitProvenance`, `StreamingSessionFragment`, and `ExecutableDatasetAction`.

The public identity constructors are:

```rust
pub fn physical_record_id(
    stream_identity: &[u8],
    partition_generation: &ImmutableObjectIdentity,
    decoder_coordinate: &[u8],
    format_semantic_digest: &[u8; 32],
) -> StableRecordId;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GlobalSequence(u64);

impl GlobalSequence {
    pub const fn new(value: u64) -> Self { Self(value) }
    pub const fn get(self) -> u64 { self.0 }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContentDigest([u8; 32]);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionCausalFrontier {
    pub through_sequence: GlobalSequence,
    pub event_time: Option<EventTimeUtc>,
    pub digest: ContentDigest,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionOwnershipEpoch(u64);

impl SessionOwnershipEpoch {
    pub const fn new(value: u64) -> Self { Self(value) }
    pub const fn get(self) -> u64 { self.0 }
}

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
) -> ActionAttemptId;
pub fn classify_logical_duplicate(
    existing: &LogicalRecordReceipt,
    candidate: &LogicalRecordReceipt,
) -> Result<DuplicateDisposition, IdentityError>;

pub struct LogicalRecordReceipt {
    pub record_id: StableRecordId,
    pub content_digest: ContentDigest,
}

pub enum DuplicateDisposition { Identical, New }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateBudgetFailureCode {
    ItemCapacity,
    ByteCapacity,
    SpillCapacity,
    ProvisionalCapacity,
}
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

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionMutationV1 {
    ConversationTurn(ConversationTurnFragment),
    AgentEvent(AgentEventFragment),
    GraphNode(GraphNodeFragment),
    GraphEdge(GraphEdgeFragment),
    SessionClose(SessionCloseFragment),
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct ConversationTurnFragment { pub role: String, pub content: Vec<u8>, pub turn_ordinal: u64 }
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct AgentEventFragment { pub event_kind: String, pub payload: Vec<u8>, pub event_ordinal: u64 }
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct GraphNodeFragment { pub node_key: String, pub request: Vec<u8> }
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct GraphEdgeFragment { pub from: String, pub to: String }
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct SessionCloseFragment { pub reason: String }

#[derive(Debug)]
pub struct SessionFragmentLease { _private: () }

#[derive(Debug)]
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
#[derive(Clone)]
pub struct StreamingResourceBudget;
impl StreamingResourceBudget {
    pub fn new(limits: BudgetLimits) -> Result<Self, BudgetError>;
    pub async fn acquire(&self, items: usize, bytes: usize) -> Result<BudgetLease, BudgetError>;
    pub fn close(&self);
    pub fn snapshot(&self) -> BudgetSnapshot;
}

impl BudgetLease {
    pub fn charged_items(&self) -> usize;
    pub fn charged_bytes(&self) -> usize;
    pub fn shrink_to(&mut self, items: usize, bytes: usize) -> Result<(), BudgetError>;
}

pub struct RetainedContentLease(Rc<RetainedContentLeaseInner>);
struct RetainedContentLeaseInner { lease: BudgetLease }

impl Clone for RetainedContentLease {
    fn clone(&self) -> Self { Self(Rc::clone(&self.0)) }
}

pub struct ActionContentLeaseSet {
    leases: smallvec::SmallVec<[RetainedContentLease; 2]>,
}

impl ActionContentLeaseSet {
    pub fn retain_for_continuation(&self) -> Self {
        Self { leases: self.leases.iter().cloned().collect() }
    }
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

#[test]
fn content_permit_releases_only_after_every_terminal_continuation_and_receipt_owner() {
    let fixture = content_lifetime_fixture(4096);
    let action = fixture.action_from_fragment();
    let continuation = action.content_leases().retain_for_continuation();
    let raw_capture = action.content_leases().retain_for_continuation();
    let checkpoint_receipt = action.content_leases().retain_for_continuation();
    drop(action);
    drop(continuation);
    drop(raw_capture);
    assert_eq!(fixture.budget().snapshot().used_bytes, 4096);
    drop(checkpoint_receipt);
    assert_eq!(fixture.budget().snapshot().used_bytes, 0);
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

Use two owned Tokio semaphore permits acquired in a fixed item-then-byte order. Store exact charged counts in `BudgetLease`; implement no `Clone`. `close` closes both semaphores and wakes waiters. Snapshot counters and high-water marks use atomics outside the per-token path. Convert `SessionFragmentLease` into a newtype over `BudgetLease`; incorporation consumes it into one `RetainedContentLease`. `ExecutableDatasetAction` owns an `ActionContentLeaseSet`. Admission, graph continuations, raw capture, session state, and prepared checkpoint receipts explicitly retain handles; cloning a handle only shares the original permit and never acquires or mints capacity. The final owner drop releases capacity.

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
- Consumes: `StreamingResourceBudget`, checkpoint Task 5A's `StreamingCheckpointParticipant`, and Tokio `spawn_blocking`.
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

#[async_trait::async_trait(?Send)]
impl StreamingCheckpointParticipant for StreamingBlockingExecutor {
    fn participant_id(&self) -> CheckpointParticipantId { self.participant_id.clone() }
    async fn checkpoint_view(&mut self, barrier: &CheckpointBarrier)
        -> Result<PreparedParticipantState, CheckpointError> {
        self.prepare_quiescent_view_or_refuse(barrier).await
    }
    async fn initialize(&mut self, state: Option<CommittedParticipantState>)
        -> Result<(), CheckpointError> { self.restore_completed_horizon_only(state).await }
    async fn checkpoint_committed(&mut self, receipt: &CommittedParticipantReceipt)
        -> Result<(), CheckpointError> { self.advance_committed(receipt) }
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

#[tokio::test(flavor = "current_thread")]
async fn checkpoint_never_serializes_or_skips_an_accepted_blocking_closure() {
    let fixture = BlockingCheckpointFixture::held_job();
    let (mut owner, barrier, job, probe) = fixture.into_parts();
    assert!(matches!(owner.checkpoint_view(&barrier).await,
        Err(CheckpointError::CutBlockedByInflight { .. })));
    assert_eq!(probe.backend_commit_count(), 0);
    job.complete();
    let state = owner.checkpoint_view(&barrier).await.unwrap();
    assert_eq!(state.inflight_job_count(), 0);
    assert_eq!(state.completed_horizon(), &barrier.cut.decoded);
}
```

Add a barrier-controlled accepted job proving `cancel_and_join` waits until cooperative cancellation is observed, plus saturation, cut rollback, restore rejection for any in-flight-job claim, and `SimClock` responsiveness tests.

- [ ] **Step 2: Run the task suite and verify RED**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd rust
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_blocking
```

- [ ] **Step 3: Implement the bounded owner**

Use one fixed count of accepted `spawn_blocking` jobs guarded before enqueue. Retain accepted join handles in a slab whose entries are removed when joined; never use Tokio's global blocking queue as capacity authority. `BlockingCancellation` wraps an atomic flag; long work calls `is_cancelled()` between bounded chunks. `BudgetedBlockingOutput<T>` owns its output-byte lease and dereferences to `T` without exposing the permit.

Arbitrary `FnOnce` closures are never serialized or replayed. A checkpoint barrier first fences new blocking acceptance. If any accepted job can affect or precede the requested cut, the participant returns `CutBlockedByInflight` and the coordinator retains the prior generation; after cooperative drain it stores only the completed typed horizon and an empty in-flight count. Restore rejects any state claiming an in-flight closure.

```rust
pub struct BudgetedBlockingOutput<T> {
    value: T,
    lease: BudgetLease,
}

impl<T> std::ops::Deref for BudgetedBlockingOutput<T> {
    type Target = T;
    fn deref(&self) -> &T { &self.value }
}
```

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
- Create: `rust/runtime/src/streaming/failure.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Test: `rust/runtime/tests/streaming_contracts.rs`

**Interfaces:**
- Consumes: Tasks 1A–1C plus checkpoint Tasks 5A, 5A-R, and 5B, which already own the run-bound participant/backend traits and backend-facing result vocabulary. Task 1D must not start from landed 5A/1C alone.
- Produces: source, format, session-program, action-sink, and checkpoint-backend factory contracts plus runtime source/decoder/session/action contracts. Exact new method signatures follow.

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingFailureStage { Source, Acquisition, Decode, Ordering, StateBudget, Session, Placement, Dispatch, Checkpoint, Result }
pub enum SourceFailureCode { Discovery, Snapshot, MutatedObject, SourceUnavailable }
pub enum AcquisitionFailureCode { Open, Read, IdentityMismatch, ObjectLimitExceeded }
pub enum DecodeFailureCode {
    Syntax,
    Schema,
    OversizedRecord,
    InvalidCursor,
    MissingReplayMetadata,
    InvalidReplayGeometry,
    SynthesisAuthorityMismatch,
    SynthesisProfileUnavailable,
}
pub enum OrderingFailureCode { LateData, WatermarkViolation, CoordinateOverflow }
// `StateBudgetFailureCode` is the neutral vocabulary owned by Task 1A so
// checkpoint Task 5A can preserve it without a dependency cycle.
pub enum SessionFailureCode { MissingPredecessor, ConflictingMutation, UnboundedCausalityState }
pub enum PlacementFailureCode { RouteUnavailable, StaleOwnershipEpoch, DigestMismatch, TargetOverflow, Cancelled }
pub enum ActionFailureCode { MissingBinding, Dispatch, Endpoint, Cancelled }

pub trait StableStreamingFailure: std::error::Error {
    fn stage(&self) -> StreamingFailureStage;
    fn code(&self) -> &'static str;
}

pub trait StreamingDatasetSourceFactory: std::fmt::Debug + Send + Sync {
    fn descriptor(&self) -> &'static StreamingSourceDescriptor;
    fn validate(&self, authored: &serde_json::value::RawValue)
        -> Result<Box<dyn ValidatedStreamingSourceConfig>, StreamSourceError>;
    fn prepare(&self, config: Box<dyn ValidatedStreamingSourceConfig>, context: &StreamingSourcePrepareContext)
        -> Result<Box<dyn PreparedStreamingDatasetSource>, StreamSourceError>;
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DatasetActionSchema(String);

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

```

Task 1D consumes, and must not redefine, the exact run-bound
`StreamingCheckpointParticipant`, `StreamingCheckpointBackend`,
`LeasedGenerationReader`, `StreamingGenerationTransaction`, and backend-facing
result DTOs landed by checkpoint Tasks 5A, 5A-R, and 5B. The checkpoint factory
above is the only new checkpoint-facing interface in this task.

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

#[test]
fn failure_stages_and_codes_do_not_collapse() {
    let acquisition = test_acquisition_failure(AcquisitionFailureCode::Read);
    let decode = test_decode_failure(DecodeFailureCode::Syntax);
    let late = test_ordering_failure(OrderingFailureCode::LateData);
    let budget = test_budget_failure(StateBudgetFailureCode::ByteCapacity);
    let placement = test_placement_failure(PlacementFailureCode::RouteUnavailable);
    assert_eq!((acquisition.stage(), acquisition.code()), (StreamingFailureStage::Acquisition, "read"));
    assert_eq!((decode.stage(), decode.code()), (StreamingFailureStage::Decode, "syntax"));
    assert_eq!((late.stage(), late.code()), (StreamingFailureStage::Ordering, "late_data"));
    assert_eq!((budget.stage(), budget.code()), (StreamingFailureStage::StateBudget, "byte_capacity"));
    assert_eq!((placement.stage(), placement.code()), (StreamingFailureStage::Placement, "route_unavailable"));
}
```

The Dynamo-specific codes above are stable format failures, not free-form
`Schema` aliases. Missing `request.replay`, impossible hash/input geometry,
block-size drift, and immutable tokenizer/profile preparation refusal retain
their exact code through `StableStreamingFailure`. Cellular disagreement uses
the existing `PlacementFailureCode::DigestMismatch` and must occur before
prepare or endpoint issue.

- [ ] **Step 2: Run the task suite and verify RED**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd rust
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_contracts
```

- [ ] **Step 3: Define the contracts and explicit errors**

Add module declarations to `streaming.rs`. Every factory-owned validated configuration exposes only `as_any`/`into_any`, mirroring engine validated configs. Every runtime trait is `?Send`; prepared factories remain `Send + Sync`. `StreamingStopReceiver` and each driver control are separately cloneable/borrowable so a pending `&mut self` future can be woken without aliasing the stage owner.

Each seam gets a distinct explicit error enum implementing `StableStreamingFailure`, `Display`, and `Error`. Its stable lowercase code maps one-to-one to the exact code enum above. Conversions may attach context but cannot change stages: acquisition never becomes decode, late-data/order never becomes endpoint latency, and checkpoint/result failures retain their own categories.

- [ ] **Step 4: Run the suite and verify GREEN**

Run the Step 2 command.

- [ ] **Step 5: Review and commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/source.rs rust/runtime/src/streaming/format.rs rust/runtime/src/streaming/session.rs rust/runtime/src/streaming/action.rs rust/runtime/src/streaming/failure.rs rust/runtime/tests/streaming_contracts.rs
git commit -m "feat(runtime): define streaming extension contracts"
```

### Task 1D-R: Host-Owned Scoped Issue and Disposition Authority

**Depends on:** Task 1D plus checkpoint Tasks 5A-R and 5B.

**Files:**
- Create: `rust/runtime/src/streaming/reliability.rs`
- Modify: `rust/runtime/src/streaming/budget.rs`
- Modify: `rust/runtime/src/streaming/blocking.rs`
- Modify: `rust/runtime/src/streaming/failure.rs`
- Modify: `rust/runtime/src/streaming/action.rs`
- Modify: `rust/runtime/src/streaming/session.rs`
- Modify: `rust/runtime/src/streaming/checkpoint.rs`
- Modify: `rust/runtime/src/streaming/checkpoint_backend.rs`
- Modify: `rust/runtime/src/streaming/checkpoints/memory.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Unit tests: `#[cfg(test)]` in `rust/runtime/src/streaming/reliability.rs`
- Unit tests: `#[cfg(test)]` in `rust/runtime/src/streaming/checkpoint_backend.rs`
- Extend: `rust/runtime/tests/support/streaming_checkpoint.rs`
- Extend: `rust/runtime/tests/streaming_budget.rs`
- Extend: `rust/runtime/tests/streaming_blocking.rs`
- Extend: `rust/runtime/tests/streaming_checkpoint_participants.rs`
- Extend: `rust/runtime/tests/streaming_checkpoint_backend.rs`
- Create: `rust/runtime/tests/streaming_reliability.rs`

**Interfaces:**
- Consumes: concrete typed failure enums, `StreamRunIdentity`, `StreamingCheckpointParticipant`, `CheckpointGeneration`, stable record/session/action identities, typed source positions, `ContentDigest`, and `StreamingResourceBudget`.
- Produces: the exact live/persisted split, prepared policy, ordered sequencer, budget-owned receipt/view, handled-cut, action/tombstone/export preparation methods, and reporter vocabulary in the reliability course correction. `CheckpointCut` gains `HandledIssueCut`; canonical generation hashing moves to v4. Backend open returns sealed versioned leased authority, and generation begin accepts only `CurrentV4CheckpointGeneration` as predecessor; legacy participant reads return only non-convertible `LegacyParticipantState`.
- Contract: public owners construct only ordinary live facts. A module-private exhaustive classifier is the sole `FailRun` authority; serde cannot restore live authority. The central sequencer, not worker arrival, owns threshold order.

**Implementation-readiness correction:**
`artifacts/streaming-design/task-1dr-implementation-readiness-correction.md`
is normative for this task. Implement its exact synchronous and paired budget
seams, public checked constructor/accessor surface, clone-safe handled cut,
result-index-root receipt authority, versioned leased generation API, bounded
legacy-v3 fixture, and non-fallback wire discriminator. Preserve the landed
Task 5B validation, acquisition, fault-point, head-comparison, and publication
order verbatim.

- [ ] **Step 1: Add the RED policy, receipt, and restore matrix**

Create `rust/runtime/tests/streaming_reliability.rs` with current-thread Tokio tests and pure golden checks:

```rust
#[test]
fn policy_matching_is_order_invariant_exact_before_wildcard_and_unambiguous() {
    let left = PreparedStreamingIssuePolicy::new(reversed_rules()).unwrap();
    let right = PreparedStreamingIssuePolicy::new(forward_rules()).unwrap();
    assert_eq!(left.digest(), right.digest());
    assert_eq!(left.rule_for(&coded_record_fact()).unwrap().rule_id(), exact_rule_id());
    assert!(PreparedStreamingIssuePolicy::new(duplicate_exact_rules()).is_err());
    assert!(PreparedStreamingIssuePolicy::new(duplicate_wildcard_rules()).is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn reverse_skew_and_worker_count_produce_identical_receipt_order() {
    let one = run_ordered_issue_script(1, reverse_and_skewed_updates()).await;
    let eight = run_ordered_issue_script(8, forward_updates()).await;
    assert_eq!(one.receipt_ids, eight.receipt_ids);
    assert_eq!(one.decisions, eight.decisions);
    assert_eq!(one.receipt_root, eight.receipt_root);
}

#[tokio::test(flavor = "current_thread")]
async fn late_domain_and_action_arrival_do_not_change_thresholds_across_restart() {
    let forward = run_domain_and_action_script(
        1,
        forward_domains_and_terminal_actions(),
        RestartAt::AfterFirstFailure,
    ).await;
    let reversed = run_domain_and_action_script(
        8,
        late_second_domain_and_reverse_terminal_actions(),
        RestartAt::BeforeNoMoreActions,
    ).await;
    assert_eq!(forward.domain_local_counters, reversed.domain_local_counters);
    assert_eq!(forward.action_counters, reversed.action_counters);
    assert_eq!(forward.decisions, reversed.decisions);
    assert_eq!(forward.receipt_root, reversed.receipt_root);
}

#[tokio::test(flavor = "current_thread")]
async fn raw_action_issue_cannot_bypass_verified_failure_membership() {
    assert!(reporter().report(IssueSequenceUpdate::Issue(raw_action_issue())).await.is_err());
}

// The following checked-view cases live in reliability.rs's #[cfg(test)]
// module, where the foundation may implement its private test seals.
#[tokio::test(flavor = "current_thread")]
async fn quarantine_is_not_handled_before_same_generation_tombstone_ack() {
    let mut ledger = ledger_with_session_quarantine_issue().await;
    assert!(ledger.receipt_partition_view(&barrier(1)).await.is_err());
    let tombstones = test_quarantine_view_at(frontier(4));
    let prepared = ledger.prepare_session_quarantine_install(
        &tombstones,
        quarantined_issue_id(),
        &barrier(1),
        &tombstone_ack_budget(),
    ).await.unwrap();
    assert_eq!(prepared.payload_charge_bytes(), expected_compact_ack_bytes());
    assert_eq!(prepared.view_charge_bytes(), expected_view_metadata_bytes());
    ledger.report(IssueSequenceUpdate::PreparedSessionQuarantineInstall(prepared)).await.unwrap();
    let view = ledger.receipt_partition_view(&barrier(1)).await.unwrap();
    assert_eq!(view.handled_cut().quarantine_tombstone_root(), ledger.tombstone_root());
    assert!(tombstones.was_borrowed_not_consumed());
}

#[tokio::test(flavor = "current_thread")]
async fn stale_tombstone_root_refuses_and_reprepare_charges_exactly() {
    let mut ledger = ledger_with_session_quarantine_issue().await;
    let tombstones = test_quarantine_view_at(frontier(4));
    let stale = ledger.prepare_session_quarantine_install(
        &tombstones, quarantined_issue_id(), &barrier(1), &tombstone_ack_budget(),
    ).await.unwrap();
    tombstones.checked_extend(frontier(5)).unwrap();
    assert!(ledger.verify_session_quarantine_install(&stale, &tombstones, &barrier(1)).is_err());
    let fresh = ledger.prepare_session_quarantine_install(
        &tombstones, quarantined_issue_id(), &barrier(1), &tombstone_ack_budget(),
    ).await.unwrap();
    ledger.verify_session_quarantine_install(&fresh, &tombstones, &barrier(1)).unwrap();
    assert_eq!(fresh.payload_charge_bytes(), expected_reencoded_ack_bytes());
    assert_eq!(fresh.view_charge_bytes(), expected_view_metadata_bytes());
}

#[tokio::test(flavor = "current_thread")]
async fn reporter_prepares_bound_exactly_charged_export_attempt_failure() {
    let mut reporter = reporter_with_retained_export_issue(run(1), generation(3), sink("jsonl"), 4).await;
    let prepared = reporter.prepare_export_attempt_failure(
        &run(1),
        &generation(3),
        &sink("jsonl"),
        4,
        ResultSinkAttemptOutcome::Failed(ordinary_export_issue()),
        &export_receipt_budget(),
    ).await.unwrap();
    assert_eq!(prepared.receipt().encoded_charge_bytes(), expected_export_encoded_bytes());
    assert_eq!(prepared.receipt().parsed_charge_bytes(), expected_export_parsed_bytes());
    assert!(reporter.prepare_export_attempt_failure(
        &run(2), &generation(3), &sink("jsonl"), 4,
        ResultSinkAttemptOutcome::Failed(ordinary_export_issue()), &export_receipt_budget(),
    ).await.is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn pre_cas_failure_retains_receipt_view_and_resume_commits_once() {
    let mut ledger = ledger_with_one_ordered_issue().await;
    let first = ledger.receipt_partition_view(&barrier(1)).await.unwrap();
    fail_transaction_before_cas(first).await;
    let retry = ledger.receipt_partition_view(&barrier(1)).await.unwrap();
    let committed = commit_view_and_ledger(retry, &mut ledger).await;
    ledger.checkpoint_committed(committed.receipt()).await.unwrap();
    let restored = restore_ledger(committed).await;
    assert_eq!(restored.committed_receipt_count(), 1);
    assert_eq!(restored.pending_receipt_count(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn high_fault_receipts_never_escape_budget_or_clone_heap_state() {
    let report = run_many_issues_with_tiny_receipt_budget(100_000).await;
    assert!(report.high_water_items <= report.authored_items);
    assert!(report.high_water_bytes <= report.authored_bytes);
    assert_eq!(report.committed_receipt_count, report.observed_issue_count);
}
```

In the reliability module unit tests add `ordinary_and_deserialized_receipts_cannot_reach_fail_run`,
`exhaustive_classifier_maps_only_terminal_boundary`,
`tampered_terminal_marker_cannot_restore_live_authority`, and
`continue_requires_private_no_membership_loss_proof`,
`failed_action_preparation_retains_issue_without_advancing_frontier`,
`dropped_failed_action_preparation_retries_same_id_without_double_count`,
`dropped_queued_action_failure_reenqueues_same_decision_without_double_count`,
`action_retry_and_backpressure_cannot_construct_terminal_receipt`,
`terminal_action_receipt_is_the_only_disposition_with_failure_identity`,
`dense_action_classification_enqueue_and_poll_never_borrows_across_await`,
`conflicting_failure_content_for_same_terminal_evidence_is_rejected`,
`checked_action_fact_rejects_foreign_membership_action_or_sequence`, and
`checked_gap_rejects_incomplete_frozen_inventory`, and
`tampered_or_foreign_export_receipt_cannot_restore_live_authority`. These in-crate tests use the
crate-private checked view fixtures; public integration tests use only the
host fixture and public behavior. Rustdoc compile-fail tests
prove callers cannot name/implement the private classifier or proof, construct
private receipt fields, clone `BudgetOwnedStreamingIssueReceipt`, or deserialize
a live fact/decision. They also prove callers cannot implement the action/
session sealed view traits or literal-construct checked action facts, gap
proofs, tombstone acknowledgements, export decisions, or export receipts. Pin v2 issue ID
`92e68da0eae7dc5acf38db5f66eeb0f2214cbe358fdbfc43c4c0dcdd59892db6`
and add `v4_generation_digest_binds_handled_issue_roots_and_refuses_v3_bytes`.
Add `record_and_session_hashes_change_with_stream_or_source_identity`,
`frontier_cannot_advance_without_contiguous_or_no_more_before_evidence`, and the
complete scope-by-disposition table test. Add
`valid_v3_generation_restores_read_only_under_v3_hash_domain`,
`v3_cannot_be_previous_of_v4`, `mixed_v3_v4_run_is_refused`, and
`malformed_or_oversized_current_bytes_never_infer_legacy_version`. In the
backend integration suite add `open_latest_returns_explicit_current_v4_or_legacy_v3`
and the compile-fail cases `legacy_v3_public_handle_cannot_be_passed_to_begin`
and `legacy_participant_state_cannot_initialize_checkpoint_participant`. The
public behavior fixture reads a legacy participant for export, observes zero
participant-initialize calls, and proves its descriptor/payload remain readable
only through `LegacyParticipantState` borrow accessors. Its compile-fail case
also copies those bytes into a newly budgeted payload and proves the retired public
`CommittedParticipantState::new` cannot bypass current-v4 context promotion. Add
`begin_with_none_over_legacy_v3_returns_legacy_read_only_head_without_mutation`
so omission cannot replace the legacy head.
In `checkpoint_backend.rs`'s crate unit module add
`current_v4_projection_exists_only_for_verified_v4` and
`legacy_v3_has_no_current_predecessor_projection`; memory/backend behavior adds
`begin_with_none_over_legacy_v3_returns_legacy_read_only_head_without_mutation`.
Those unit tests alone may
invoke crate-private `CurrentV4PredecessorProjection`.

Also add the readiness REDs
`synchronous_action_enqueue_refuses_immediately_without_advancing_state`,
`combined_pair_acquisition_cannot_hold_one_sublease_while_waiting_for_other`,
`cancelled_combined_pair_acquisition_leaves_zero_charge`,
`handled_issue_cut_is_clone_compatible_with_checkpoint_cut`,
`committed_receipt_binds_exact_result_index_root`,
`mismatched_result_index_root_retains_detailed_receipts`,
`current_participant_restore_uses_verified_reader_not_public_constructor`,
`checked_legacy_fixture_is_bounded_read_only_and_cannot_overwrite_head`,
`unknown_or_malformed_explicit_v4_never_falls_back_to_v3`,
`v4_shape_without_explicit_v4_discriminator_is_refused`,
`action_disposition_variants_expose_only_their_approved_type_state`, and
`first_and_later_exhaustion_compare_status_owned_ordinal_and_counter` in the
owned unit/integration files named above.

- [ ] **Step 2: Run the suite and verify RED**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd rust
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_budget --test streaming_blocking --test streaming_checkpoint_participants --test streaming_reliability --test streaming_checkpoint_backend
```

Expected: compilation fails because the synchronous and paired budget seams,
sealed classifier/action authority, v4 handled cut, versioned backend
authority, non-destructive tombstone install, ordered sequencer, and
budget-owned receipt view do not exist.

- [ ] **Step 3: Implement the minimal host authority**

Implement the exact public vocabulary frozen in
`artifacts/streaming-design/reliability-continuation-course-correction.md`.
Apply the exact readiness seams frozen in
`artifacts/streaming-design/task-1dr-implementation-readiness-correction.md`.
Keep the host failure enum,
verified terminal/no-membership-loss proofs, classifier, persisted-wire
verifier, and decision constructor module-private. The public reporter accepts
only `IssueSequenceUpdate` and returns a fixed-size outcome. Detailed encoded
receipts remain non-Clone and budget-owned inside the ledger until the exact
generation callback proves their partition reachable.

The central sequencer owns per-input-domain pending maps, domain-local rule
counters, and checkpointed contiguous/no-more-before frontiers. It separately
owns the bounded action pending map and terminal/no-more-actions-before global
frontier, receiving only reporter-minted checked terminal facts and gap
proofs. Failed terminals use the two-stage reporter API: sealed P2 terminal
evidence plus an ordinary action issue produces a move-only retained identity
without frontier advancement; P2 consumes that identity into its terminal
receipt, after which the finalized membership view can produce the fact. The reporter revalidates exact action/sequence membership, failure issue
scope, and contiguous gap closure; public raw success/failure/gap updates do not
exist. Exact-code rules precede one wildcard;
ambiguous rules refuse preparation. Record/session scope and order include the
same stream/source identity. Extend `CheckpointCut` with `HandledIssueCut`,
update canonical generation hashing to v4, and require cut/participant/receipt
root equality. No hole/quarantine frontier crosses missing data without its
same-generation receipt or the separately budgeted move-only tombstone install
acknowledgement prepared non-destructively from P1B's retained map. Dropping a
pre-CAS acknowledgement preserves P1B for identical retry; a checked late
fragment extension invalidates its root and requires re-acknowledgement.
Implement the bounded strict v3/v4 decoder and version-selected hash verifier;
extend `checkpoint_backend.rs` and memory support so open returns explicit
`CurrentV4` versus `LegacyV3ReadOnly` leased authority. Only sealed
`CurrentV4CheckpointGeneration` is accepted by `begin_generation`; v3 cannot be
erased into a raw predecessor or participate in v4 succession. The versioned
common reader exposes no participant initializer state; its legacy branch
returns only opaque `LegacyParticipantState`, which has no conversion to
`CommittedParticipantState` and cannot enter `StreamingCheckpointParticipant::initialize`.
Committed-state storage promotion becomes crate-private and requires the
private current-v4 reader context; test support must obtain current state
through the verified current reader rather than a public constructor.
`LegacyParticipantState` lives in `checkpoint.rs`; its doctests import it from
`aiperf_runtime::streaming::checkpoint`. The memory integration fixture may
install only a strictly verified, completely budgeted legacy-v3 read-only head
into an empty run and cannot overwrite or mint current authority.

Define the public borrowed view traits in `action.rs` and `session.rs`, each
with a private sealed supertrait in that same parent module. Later P2/P4 child
action modules and the P1B child session module can implement those seals for
their host-owned private state, while unrelated siblings and external callers
cannot. Reliability owns only the checked constructors that consume the views;
it never asks P2, P4, or P1B to construct a reliability-private token.

- [ ] **Step 4: Run the suite and verify GREEN**

Run Step 2, then:

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --lib
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --doc
```

Expected: privacy,
serde refusal, policy precedence, the v2 golden and v4 handled-cut domain proof, order independence, non-
destructive receipt/tombstone view retry, sealed action/gap refusal, restore
once-only, handled-cut/tombstone acknowledgement and invalidation, explicit
v3 read-only backend refusal, doctest privacy, and budget high-water
cases pass.

- [ ] **Step 5: Review and commit**

Review that no adapter-specific code or Config-v2 surface entered this task,
then commit:

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/budget.rs rust/runtime/src/streaming/blocking.rs rust/runtime/src/streaming/failure.rs rust/runtime/src/streaming/action.rs rust/runtime/src/streaming/session.rs rust/runtime/src/streaming/checkpoint.rs rust/runtime/src/streaming/checkpoint_backend.rs rust/runtime/src/streaming/checkpoints/memory.rs rust/runtime/src/streaming/reliability.rs rust/runtime/tests/support/streaming_checkpoint.rs rust/runtime/tests/streaming_budget.rs rust/runtime/tests/streaming_blocking.rs rust/runtime/tests/streaming_checkpoint_participants.rs rust/runtime/tests/streaming_checkpoint_backend.rs rust/runtime/tests/streaming_reliability.rs
git commit -m "feat(runtime): classify streaming issues for continuation"
```

### Task 1E: Reusable Source and Format Conformance Harnesses

**Files:**
- Create: `rust/runtime/tests/support/streaming_source_conformance.rs`
- Create: `rust/runtime/tests/support/streaming_format_conformance.rs`
- Create: `rust/runtime/tests/streaming_contract_conformance.rs`

**Interfaces:**
- Consumes: Tasks 1D and 1D-R public contracts.
- Produces:

```rust
pub async fn assert_source_conformance(
    factory: &dyn StreamingDatasetSourceFactory,
    reporter: Box<dyn StreamingIssueReporter>,
    cases: SourceConformanceCases,
);
pub async fn assert_format_conformance(
    factory: &dyn StreamingDatasetFormatFactory,
    reporter: Box<dyn StreamingIssueReporter>,
    cases: FormatConformanceCases,
);
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
pub async fn assert_pending_is_not_seal(
    mut opened: OpenedStreamingDatasetSource,
    reporter: Box<dyn StreamingIssueReporter>,
) {
    let pending = opened.source.next_event();
    tokio::pin!(pending);
    assert!(futures::poll!(&mut pending).is_pending());
    opened.control.stop();
    let error = pending.await.expect_err("stop wakes pending source");
    assert!(matches!(error, StreamSourceError::Stopped));
    assert_eq!(reporter.summary().unwrap().total, 0);
}
```

The harness takes ownership of the separately constructed reporter; no adapter
owns it. Borrow it only after a source/format future returns and release the
borrow before every later source, format, checkpoint, or control await.
Ordinary scripted faults are reported and the next valid unit is then observed.
`Stopped` belongs only to host stop control: it follows an
explicit `opened.control.stop()`, creates no issue receipt, and is neither a
source seal nor an adapter-selected disposition. Add
`conformance_reporter_is_released_before_each_await` and
`host_stop_wakes_pending_source_without_issue_or_seal`.

- [ ] **Step 2: Run the task suite and verify RED**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd rust
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_contract_conformance
```

Expected: fake adapters demonstrate missing reusable harness functions.

- [ ] **Step 3: Implement the reusable harness functions**

Cover pending-versus-seal, stop wakeup without an issue receipt, immutable identity, mutation refusal, ordinary-fault reporter continuation, backpressure, lease lifetime, exact cursor restore, duplicate replay, and idempotent post-commit notification. Harnesses accept factories, the separately owned reporter, and case data; they do not inspect concrete adapter types.

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

#[test]
fn supported_capability_cross_product_composes_without_concrete_type_switches() {
    let registry = fake_cross_product_registry(
        ["finite", "follow"],
        ["jsonl", "columnar"],
        ["conversation", "agent_graph"],
        ["scheduled_request", "session_state"],
        ["dry_run", "http"],
    );
    for selection in registry.declared_supported_cross_product() {
        let plan = StreamingCapabilityAgreement::validate(selection.descriptors()).unwrap();
        assert_eq!(plan.selected_ids(), selection.ids());
        assert_eq!(plan.preparation_count_per_factory(), 1);
    }
}
```

Add ordered-inventory, unknown lookup, transactional extension rollback, cross-product mismatch, and catalog serialization tests. The positive matrix varies source × format × session program × action sink × transport and asserts preparation remains descriptor-driven with no concrete source-format branch. Extend the compile-time extension test with one custom factory in every category.

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
- Consumes: Task 2 lookups, `StreamingCapabilityAgreement`, and Task 1D-R's checked prepared reliability policy/digest.
- Produces strict Config-v2 types `DatasetStreams`, `DatasetStream`, `StreamingComponent`, `StreamLimits`, and `ShadowReplay`; Protocol-v2 `DatasetStreamsSpecV2`; `RunResourceV2::DatasetStreams`; `ResourceRequirementsV2::shadow_replay()`; an internal/test-injected reliability-policy digest carried through capability agreement. Product Task V1 later owns the strict public reliability-policy Config-v2 fields and exact projection into this seam.
- Does not register a stock `shadow_replay` workload or placeholder factory.
- Rejects a reliability-policy digest mismatch before construction, polling, or issue. Add `protocol_rejects_reliability_digest_mismatch_before_effects` to the existing RED/GREEN suite; no adapter-specific default may replace the prepared policy.

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
- Consumes: `BudgetLease`, existing `TurnRecordProcessor`, `IssuedCredit`, `TurnDispatchOutcome`, and Task 1D-R `StreamingIssueReporter`.
- Produces `TerminalLaneLimits`, `TerminalLanePermit`, `BoundedTerminalProcessorLane`, `TerminalLaneControl`, `TerminalLaneSnapshot`, `ScheduledSessionIdentity`, and a new opt-in issue method. Existing issue methods retain their signatures.

```rust
pub struct ScheduledSessionIdentity { pub stable_ordinal: u64 }
pub struct TerminalRecordSizeBound(NonZeroUsize);

impl ScheduledRuntime {
    pub async fn reserve_terminal_processing(
        &self,
        bound: TerminalRecordSizeBound,
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

Add full-lane backpressure, checked-invariant latch/wakeup, cancellation permit return, 100,000 one-turn sessions with zero active map entries, stable external ordinal, and finite-default compatibility tests. Add `ordinary_terminal_processor_error_reports_export_issue_and_drain_continues` and `terminal_lane_accounting_corruption_wakes_failed_run`; Task 4A consumes the Task 1D-R reporter and cannot promote an ordinary processor/export fault to the invariant latch.

- [ ] **Step 2: Run the task suite and verify RED**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd rust
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_terminal_lane
```

- [ ] **Step 3: Implement reservation-before-issue**

Capability agreement constructs a conservative `TerminalRecordSizeBound` from endpoint response limits, maximum output tokens/bytes, usage/metric envelopes, and the configured raw-capture policy. If no finite bound can be proven, streaming execution is refused before dispatch. The streaming caller asynchronously reserves one item plus that exact conservative maximum before issue; settlement computes the actual terminal size, verifies it is within the validated bound, shrinks the owned byte lease to actual size, and moves it into `TerminalWork`. Add RED cases `unbounded_terminal_payload_is_refused_before_dispatch` and `actual_terminal_bytes_never_exceed_reserved_bound`. Settlement cannot block or fail for capacity after reservation. One `spawn_local` drain owner invokes existing processors in order, reports ordinary export/processor failures through the scoped issue reporter, retains only the first checked invariant plus bounded counters, wakes the phase owner only for that invariant, and never stores a per-record `JoinHandle` or error string.

- [ ] **Step 4: Bound active session numbering without changing finite IDs**

Replace `sessions.len()` allocation with `next_session_number: Cell<u64>` plus an active map. Remove both `session_numbers` and `session_url_indices` on `credit.is_final_turn()`. Existing issue methods allocate monotonic finite ordinals; the streaming method uses `ScheduledSessionIdentity::stable_ordinal` directly and does not insert it into the lifetime map.

- [ ] **Step 5: Join the lane through phase finalization**

Replace `wait_record_processors` with a mode-aware drain: legacy finite mode reaps its existing tasks continuously; streaming mode closes and drains `TerminalLaneControl`. Surface a checked authority/accounting invariant before report construction; otherwise return the issue and derived-sink status needed for degraded or export-incomplete reporting.

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

**Dependencies:** Tasks 1A–1D and checkpoint Task 5A-R must be merged. This task implements `StreamingCheckpointParticipant` for event-time state; it must not invent a temporary or run-free checkpoint shape.

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

#[test]
fn quiet_follow_cannot_advance_hard_watermark() {
    let mut policy = event_policy_with_hard_frontier(event_ns(100));
    policy.observe_quiet_poll();
    policy.observe_quiet_poll();
    assert_eq!(policy.hard_watermark(), event_ns(100));
}

#[test]
fn estimated_frontier_requires_authored_late_policy() {
    assert!(matches!(EventTimePolicyConfig::estimated_without_late_policy().validate(),
        Err(EventTimeError::EstimatedFrontierRequiresLatePolicy)));
}

#[test]
fn shuffled_and_equal_time_inputs_have_one_stable_order() {
    let expected = stable_order_fixture().ordered_ids();
    for permutation in shuffled_listing_completion_and_worker_orders() {
        assert_eq!(stable_order_fixture().with_permutation(permutation).ordered_ids(), expected);
    }
}
```

Add uncertainty rejection, overflow, negative checked arithmetic, immutability after a fake system-clock adjustment, late-target classification, equal-time tie-break, hash-map insertion, worker-count, and completion-order independence cases.

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

Use a budgeted min-order heap keyed by `(EventTimeUtc, StableOrderKey)`. Hard frontiers never regress. Bounded disorder computes `max_seen - max_out_of_order` with checked arithmetic. Implement distinct typed outcomes for `Fail`, `IssueImmediately`, `Drop`, and bounded catch-up. Persist policy kind, max seen, watermark, sequence frontier, and heap membership through the run-bound Task 5A-R participant contract.

- [ ] **Step 5: Implement bounded near-horizon admission**

`NearHorizonBuffer::pop_admissible(now_ns)` releases only actions whose target is at most `now_ns + schedule_horizon_ns`; later actions remain in the single budgeted heap. The caller waits with `ScheduledRuntime::wait_until_or_stop`; this task creates no task per buffered action. Record publication lag, acquisition duration, decode duration, watermark wait, causal wait, scheduling lateness, and endpoint latency as distinct typed fields.

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
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_feature_inventory --test streaming_identity --test streaming_budget --test streaming_blocking --test streaming_contracts --test streaming_reliability --test streaming_checkpoint_backend --test streaming_contract_conformance --test streaming_registry --test extensions_compile_time_extension --test streaming_protocol_v2 --test streaming_terminal_lane --test streaming_phase_runtime --test streaming_event_time
```

Then run `cargo fmt --check` and `cargo clippy -p aiperf-runtime --all-targets --features streaming -- -D warnings` as review gates. Confirm the stock catalog still lacks `shadow_replay`, `s3`, and `object_store` factories; `streaming-s3` only enables dependencies and later adapter compilation authority. Confirm no file outside the paths named in this plan changed.

## Self-Review Checklist

- Spec invariants covered here: stable topology-independent identity; typed source/format/session/action/checkpoint seams; host-owned scoped issue/disposition authority; deterministic checkpointed issue receipts and thresholds; bounded memory and blocking ownership; feature-accurate absence; strict resource ownership; cross-format capability agreement; bounded terminal processing; reusable phase/capture construction; one immutable UTC anchor; deterministic event-time ordering; no task per far-future action.
- Deferred intentionally to later subsystem plans: checkpoint storage implementation, result segments/compaction, session state machines, pipeline execution, concrete local/HF/Baseten/Dynamo/S3 adapters, executable shadow workload, graph action sink, sensitive state, and cellular execution.
- Type consistency: checkpoint Tasks 5A, 5A-R, and 5B own run-bound participant/backend I/O vocabulary; Task 1D owns the five factory and source/format/session/action contracts; Task 1D-R owns neutral issue/disposition authority; Task 2 registers those exact traits; Task 3 references their descriptor lookups and freezes the reliability-policy digest; Task 4A consumes Task 1B permits; Task 7A consumes Task 1A IDs, Task 1B budget, Task 1D-R issue authority, and Task 5A-R run-bound checkpoint cuts.
- Placeholder scan: production registration is deliberately absent until executable implementations exist; no task asks for a rejecting placeholder, temporary workload, source-format switch, or `NativeDatasetPlan` variant.
