<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Streaming Checkpoint and Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the atomic, bounded checkpoint and result plane required by native streaming datasets: typed stage cuts, crash-durable local generations, leased bounded indexes, checkpoint-aligned metric/record/session/provenance epochs, delivery-mode restart semantics, and deterministic partial/final/aborted results.

**Architecture:** Every stateful stage exposes one stable `StreamingCheckpointParticipant`. The coordinator collects non-destructive views at one typed cut, stages participant and result objects in one backend transaction, atomically publishes one generation, and only then sends idempotent commit receipts. Results are immutable content-addressed projections reached through a bounded persistent index; final presentation artifacts are streamed from a leased final generation and never become checkpoint authority.

**Tech Stack:** Rust 2024, Tokio current-thread runtimes, `async_trait(?Send)`, BLAKE3, strict Serde DTOs, existing `Clock`, `MetricsAccumulator`, `RecordIngest`, `NativeReport`, `PreparedRunOutcome`, `PreparedReportCommit`, and the Task-1 `StreamingBlockingExecutor`/resource budgets.

**Spec:** `artifacts/streaming-design/streaming-dataset-shadow-replay-design.md` at approved commit `505efc06b0`, especially “Checkpoint and delivery semantics” and “Checkpoint-based results”.

## Global Constraints

- Task 5A prerequisites are foundation Tasks 0 and 1A-1C. Task 5B follows 5A; foundation Task 1D then consumes these exact checkpoint contracts. Later checkpoint/result tasks declare their additional terminal/capture/registry dependencies explicitly.
- Cargo commands run from the nested `rust/` workspace; git commands run from the repository root. Every targeted test-suite invocation uses `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target`.
- Each task includes the nearest parent module declaration required for its own GREEN build. The integration owner resolves overlapping declaration edits during the required `--no-ff` merge.
- Checkpoint and result library APIs use explicit `CheckpointError`/`ResultPlaneError`, never `anyhow`.
- `checkpoint_view` is non-destructive. No participant releases live state until the backend has returned a committed generation and `checkpoint_committed` is delivered.
- The generation record is the only authority. A result head, report file, flush, or participant-local cursor cannot advance independently.
- All queues, index pages, prepared objects, provisional facts, compaction buffers, and filesystem jobs hold item and byte permits.
- Filesystem writes, hashing of large objects, fsync, index compaction, and final artifact compaction run through `StreamingBlockingExecutor`.
- No lock is held across `.await`; no `Arc<Mutex<_>>` enters request/token paths; no unbounded channel or cumulative descriptor `Vec` is permitted.
- Test-only fault injection is injected through private traits/enums and cannot be selected in production config.
- Each task ends with one focused commit and independent behavior plus Graham review. Downstream work starts only from an integrated commit containing every declared dependency.

## Exact File Map

```text
rust/runtime/src/streaming/checkpoint.rs                 # cuts, participants, generation DTOs
rust/runtime/src/streaming/checkpoint_backend.rs         # backend/transaction/leased-reader traits
rust/runtime/src/streaming/checkpoints/memory.rs         # executable reference backend
rust/runtime/src/streaming/checkpoints/local.rs          # durable local object store and CURRENT CAS
rust/runtime/src/streaming/checkpoints/lease_gc.rs       # reader/prepare leases and mark-grace-sweep
rust/runtime/src/streaming/checkpoints/none.rs           # explicit no-resume backend capability
rust/runtime/src/streaming/checkpoints/object_store.rs   # conditional object-store generation backend
rust/runtime/src/streaming/checkpoint_coordinator.rs     # barrier and post-CAS notification owner
rust/runtime/src/streaming/results.rs                    # capture plan, correlation, public result DTOs
rust/runtime/src/streaming/results/index.rs              # bounded persistent content-addressed index
rust/runtime/src/streaming/results/epoch.rs              # worker rotation, holes, provisional state
rust/runtime/src/streaming/results/compactor.rs          # partial/final/aborted assembly
rust/runtime/tests/streaming_checkpoint_participants.rs
rust/runtime/tests/streaming_checkpoint_backend.rs
rust/runtime/tests/streaming_local_checkpoint.rs
rust/runtime/tests/streaming_checkpoint_gc.rs
rust/runtime/tests/streaming_checkpoint_coordinator.rs
rust/runtime/tests/streaming_result_index.rs
rust/runtime/tests/streaming_result_epochs.rs
rust/runtime/tests/streaming_result_finalization.rs
rust/runtime/tests/support/streaming_checkpoint.rs
rust/runtime/tests/support/streaming_checkpoint_coordinator.rs
```

Existing integration anchors:

- Extend `rust/runtime/src/metrics.rs:220` (`NativeMetricsObserver`) with epoch rotation in Task 6B; do not create a second metrics vocabulary.
- Consume `rust/runtime/src/metrics_core/ingest.rs:136` (`RecordIngest`) and `rust/runtime/src/metrics_core/accumulator.rs:456` (`MetricsAccumulator`).
- Extend bounded summary fields at `rust/runtime/src/metrics_core/report.rs:1082` (`NativeReport`).
- Join captured terminal facts at `rust/runtime/src/engine/records.rs:51` (`CapturedRecord`); do not write checkpoint authority through `record_lane.rs:228` (`RecordArtifactLane`).
- Preserve `rust/runtime/src/engine/registry.rs:392` (`PreparedReportCommit`) as a synchronous post-report lease release and `registry.rs:506` (`PreparedRunOutcome`) as the success shape.
- Use `rust/runtime/src/engine/protocol_v2.rs:1233` (`FailureStageV2`) and `protocol_v2.rs:1342` (`RunDiagnosticArtifactV2`) for failed-run evidence.

Every integration-test file in this plan starts with the following exact test-support import and adds only the named helpers required by its task to that support module:

```rust
#[path = "support/streaming_checkpoint.rs"]
mod support;
```

## Dependency and Parallelization Graph

```text
5A typed cuts
 `-> 5B backend + memory
      |-> 5C local durability -----> 5D leases/GC --.
      |-> 6A result index --------------------------+-> 6B epochs/holes/partial
      `-> 5E coordinator/post-CAS ------------------'          |
                                                               `-> 6C final/aborted/delivery matrix -> 6D report order

2 + 5C -> 5F1 local/none factories
5B + 5E + 5F1 + A6 -> 5F2 object CAS
5D + 5F2 + P6 -> 5F3 object leases/GC/encryption
```

After 5B merges, two worktrees may run concurrently: one owns 5C; the other owns 6A. A third worktree may run 5E because it owns only `checkpoint_coordinator.rs` and its dedicated support/test files. Merge 5C and 6A before starting 5D. Merge 5D, 5E, and 6A before cutting 6B. Tasks 6C and 6D serialize after 6B. Each worktree lands the minimal parent module declaration needed to compile; the integration owner resolves declaration conflicts. Tasks 5F1-5F3 follow their explicit cross-plan prerequisites.

---

### Task 5A: Typed Cuts and Stable Checkpoint Participants

**Depends on:** foundation Tasks 0 and 1A-1C.

**Files:**
- Create: `rust/runtime/src/streaming/checkpoint.rs`; Task 5A owns the participant declaration consumed by foundation Task 1D.
- Modify: `rust/runtime/src/streaming.rs`
- Create: `rust/runtime/tests/support/streaming_checkpoint.rs`
- Create: `rust/runtime/tests/streaming_checkpoint_participants.rs`

**Produces these exact interfaces:**

```rust
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointBarrier {
    pub epoch: CheckpointEpoch,
    pub cut: CheckpointCut,
    pub plan_digest: ContentDigest,
}

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
    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError>;
}
```

`PreparedParticipantState` contains participant ID, schema ID/version, represented cut, immutable bytes, BLAKE3 digest, item count, and byte length. `CommittedParticipantState` contains the verified descriptor plus owned bytes. `CommittedParticipantReceipt` binds generation, participant ID, committed descriptor digest, and represented cut. `CheckpointParticipantPlan::new` sorts by stable ID and rejects duplicates.

The generation identity used by every later task is defined here:

```rust
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CheckpointEpoch(u64);

impl CheckpointEpoch {
    pub const fn new(value: u64) -> Self { Self(value) }
    pub const fn get(self) -> u64 { self.0 }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointGeneration {
    pub epoch: CheckpointEpoch,
    pub digest: ContentDigest,
}

impl CheckpointGeneration {
    pub fn new(epoch: CheckpointEpoch, digest: ContentDigest) -> Self { Self { epoch, digest } }
    pub const fn epoch(&self) -> CheckpointEpoch { self.epoch }
    pub const fn digest(&self) -> &ContentDigest { &self.digest }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommittedCheckpointGeneration {
    pub generation: CheckpointGeneration,
    pub previous: Option<ContentDigest>,
    pub cut: CheckpointCut,
    pub participant_descriptors: Vec<ParticipantStateDescriptor>,
    pub result_index_root: ContentDigest,
    pub is_final: bool,
    pub terminal_reason: Option<CheckpointTerminalReason>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointTerminalReason {
    Completed,
    Aborted,
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParticipantStateDescriptor {
    pub participant_id: CheckpointParticipantId,
    pub schema_id: String,
    pub schema_version: u32,
    pub represented_cut: CheckpointCut,
    pub content_digest: ContentDigest,
    pub item_count: u64,
    pub byte_length: u64,
}

pub struct PreparedParticipantState {
    pub descriptor: ParticipantStateDescriptor,
    pub bytes: Bytes,
}

pub struct CommittedParticipantState {
    pub descriptor: ParticipantStateDescriptor,
    pub bytes: Bytes,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommittedParticipantReceipt {
    pub generation: CheckpointGeneration,
    pub participant_id: CheckpointParticipantId,
    pub descriptor_digest: ContentDigest,
    pub represented_cut: CheckpointCut,
}

impl CommittedCheckpointGeneration {
    pub fn generation(&self) -> CheckpointGeneration { self.generation.clone() }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CheckpointError {
    AlreadyInitialized,
    GenerationConflict {
        expected: Option<CheckpointGeneration>,
        actual: Option<CheckpointGeneration>,
    },
    ParticipantSetMismatch,
    ObjectVerification,
    LeaseLost { generation: CheckpointGeneration },
    PostCommitNotification { participant: CheckpointParticipantId },
    SourceUnavailableOnResume,
    Storage { message: String },
}
```

Implement `Display` and `std::error::Error` directly, following existing runtime library error enums; do not add `thiserror` or another dependency.

- [ ] **Step 1: Write representative RED tests**

```rust
#[test]
fn horizon_domains_cannot_be_substituted_and_round_trip() {
    let cut = support::cut_at(7);
    let encoded = serde_json::to_vec(&cut).unwrap();
    let restored: CheckpointCut = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(restored, cut);
    assert_eq!(restored.decoded.global_sequence(), GlobalSequence::new(7));
    assert_eq!(restored.terminal.global_sequence(), GlobalSequence::new(7));
}

#[tokio::test(flavor = "current_thread")]
async fn participant_view_is_non_destructive_until_commit_receipt() {
    let mut participant = support::CountingParticipant::new("session", 4);
    participant.initialize(None).await.unwrap();
    let prepared = participant.checkpoint_view(&support::barrier_at(4)).await.unwrap();
    assert_eq!(participant.released_items(), 0);
    participant
        .checkpoint_committed(&support::receipt_for(&prepared))
        .await
        .unwrap();
    assert_eq!(participant.released_items(), 4);
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_checkpoint_participants
```

Expected: compilation fails on missing typed cut/state/receipt APIs.

- [ ] **Step 3: Implement the minimal typed state machine**

```rust
pub struct ParticipantInitialization {
    is_initialized: bool,
}

impl ParticipantInitialization {
    pub fn initialize_once(&mut self) -> Result<(), CheckpointError> {
        if self.is_initialized {
            return Err(CheckpointError::AlreadyInitialized);
        }
        self.is_initialized = true;
        Ok(())
    }
}
```

Add checked constructors for every horizon; do not implement cross-domain `From` conversions. Validate finite lengths and exact digest before a committed state reaches a participant.

- [ ] **Step 4: Verify GREEN**

Run the Step-2 command. Expected: all typed-domain, duplicate-ID, one-shot initialization, non-destructive-view, and idempotent-receipt tests pass.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/checkpoint.rs rust/runtime/tests/support/streaming_checkpoint.rs rust/runtime/tests/streaming_checkpoint_participants.rs
git commit -m "feat(runtime): define streaming checkpoint cuts"
```

### Task 5B: Atomic Backend Contract and In-Memory Reference

**Depends on:** Task 5A.

**Files:**
- Create: `rust/runtime/src/streaming/checkpoint_backend.rs`
- Create: `rust/runtime/src/streaming/checkpoints.rs`
- Create: `rust/runtime/src/streaming/checkpoints/memory.rs`
- Create: `rust/runtime/src/streaming/results.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Create: `rust/runtime/tests/streaming_checkpoint_backend.rs`

**Produces these exact interfaces:**

```rust
#[async_trait(?Send)]
pub trait StreamingCheckpointBackend {
    async fn open_latest(
        &self,
        run: &StreamRunIdentity,
    ) -> Result<Option<Box<dyn LeasedGenerationReader>>, CheckpointError>;
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

Task 5B also defines the content-neutral DTOs needed by those object-safe signatures: `ResultPartition`, `PreparedResultEpoch`, `ResultSegmentDescriptor`, `ResultSegmentReader`, `ResultIndexCursor`, `ResultIndexReadBudget`, and `ResultIndexPage`. At this stage a partition is verified bytes plus projection/schema/range/count/digest metadata; Task 6A adds logical membership construction and conflict policy without changing the backend signatures. This ordering avoids a backend/results module cycle.

```rust
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CellId(u32);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorkerId(u32);

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResultProjectionId(String);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResultSchemaVersion(u32);

pub struct ResultPartition {
    pub descriptor: ResultSegmentDescriptor,
    pub bytes: Bytes,
}

pub struct PreparedResultEpoch {
    pub index_root: ContentDigest,
    pub descriptors: Vec<ResultSegmentDescriptor>,
    pub item_count: u64,
    pub byte_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultSegmentDescriptor {
    pub run: StreamRunIdentity,
    pub epoch: CheckpointEpoch,
    pub cell_id: CellId,
    pub worker_id: WorkerId,
    pub projection: ResultProjectionId,
    pub schema: ResultSchemaVersion,
    pub first_sequence: GlobalSequence,
    pub last_sequence: GlobalSequence,
    pub item_count: u64,
    pub byte_length: u64,
    pub membership_root: ContentDigest,
    pub payload_digest: ContentDigest,
}

pub struct ResultSegmentReader {
    pub descriptor: ResultSegmentDescriptor,
    pub bytes: Bytes,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultIndexCursor {
    pub root: ContentDigest,
    pub block: ContentDigest,
    pub item_offset: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResultIndexReadBudget {
    pub max_items: NonZeroUsize,
    pub max_bytes: NonZeroU64,
}

pub struct ResultIndexPage {
    pub descriptors: Vec<ResultSegmentDescriptor>,
    pub next: Option<ResultIndexCursor>,
    pub charged_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointCommitMetadata {
    pub previous: Option<CheckpointGeneration>,
    pub epoch: CheckpointEpoch,
    pub cut: CheckpointCut,
    pub plan_digest: ContentDigest,
    pub result_plan_digest: ContentDigest,
    pub is_final: bool,
    pub terminal_reason: Option<CheckpointTerminalReason>,
}
```

- [ ] **Step 1: Write representative RED tests**

```rust
#[tokio::test(flavor = "current_thread")]
async fn stale_writer_cannot_merge_or_replace_head() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits());
    let first = support::commit_empty(&backend, None, 1).await.unwrap();
    let stale = backend.begin_generation(None).await.unwrap();
    let current = backend.begin_generation(Some(first.generation())).await.unwrap();
    current.commit(support::metadata_at(2)).await.unwrap();
    let error = stale.commit(support::metadata_at(1)).await.unwrap_err();
    assert!(matches!(error, CheckpointError::GenerationConflict { .. }));
}

#[tokio::test(flavor = "current_thread")]
async fn dropped_transaction_publishes_nothing_and_releases_budget() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits());
    let transaction = backend.begin_generation(None).await.unwrap();
    assert_eq!(backend.prepared_transactions(), 1);
    drop(transaction);
    assert_eq!(backend.prepared_transactions(), 0);
    assert!(backend.open_latest(&support::run_id()).await.unwrap().is_none());
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_checkpoint_backend
```

- [ ] **Step 3: Implement the minimal reference backend**

```rust
struct MemoryHead {
    generation: Option<CommittedCheckpointGeneration>,
    objects: BTreeMap<ContentDigest, Bytes>,
}

fn compare_expected(
    head: Option<CheckpointGeneration>,
    expected: Option<CheckpointGeneration>,
) -> Result<(), CheckpointError> {
    if head != expected {
        return Err(CheckpointError::GenerationConflict { expected, actual: head });
    }
    Ok(())
}
```

Keep storage behind `Rc<RefCell<MemoryHead>>`; it is test/reference state on one local runtime, not a shared hot-path lock. The transaction owns prepared permits and releases them in `Drop` unless commit transfers them to the committed generation.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: atomic participant+result publication, stale-writer refusal, immutable read verification, and RAII abort pass.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/checkpoint_backend.rs rust/runtime/src/streaming/checkpoints.rs rust/runtime/src/streaming/checkpoints/memory.rs rust/runtime/src/streaming/results.rs rust/runtime/tests/support/streaming_checkpoint.rs rust/runtime/tests/streaming_checkpoint_backend.rs
git commit -m "feat(runtime): add atomic checkpoint backend contract"
```

### Task 5C: Crash-Durable Local Generation Store

**Depends on:** Task 5B.

**Files:**
- Create: `rust/runtime/src/streaming/checkpoints/local.rs`
- Modify: `rust/runtime/src/streaming/checkpoints.rs`
- Extend: `rust/runtime/tests/support/streaming_checkpoint.rs`
- Create: `rust/runtime/tests/streaming_local_checkpoint.rs`

**Produces:** `LocalCheckpointBackend`, immutable object layout, single-writer lease, expected-head CAS, and injected `LocalCommitFault` used only by tests.

```rust
trait LocalCheckpointFilesystem {
    fn write_new(&self, path: &Path, bytes: &[u8]) -> Result<(), CheckpointError>;
    fn sync_file(&self, path: &Path) -> Result<(), CheckpointError>;
    fn sync_directory(&self, path: &Path) -> Result<(), CheckpointError>;
    fn rename(&self, source: &Path, destination: &Path) -> Result<(), CheckpointError>;
    fn read_no_follow(&self, path: &Path, max_bytes: u64) -> Result<Bytes, CheckpointError>;
}
```

The production implementation delegates each complete operation to `StreamingBlockingExecutor`; the injected test implementation records ordering and fails at one `LocalCommitFault`.

The on-disk layout is fixed before tests are written:

```text
<root>/<run-digest>/objects/blake3-<64-lower-hex>
<root>/<run-digest>/generations/<20-digit-epoch>-<generation-digest>.json
<root>/<run-digest>/CURRENT
<root>/<run-digest>/leases/{writer,prepare-<id>,reader-<id>,report-<id>}
<root>/<run-digest>/tmp/<transaction-id>/
```

`CURRENT` contains strict JSON `{ "epoch": u64, "digest": "blake3:<hex>" }` plus one trailing newline. Object and generation names are derived from verified bytes, never caller paths.

- [ ] **Step 1: Write representative RED test and the full fault table**

```rust
#[tokio::test(flavor = "current_thread")]
async fn every_pre_current_fault_preserves_previous_generation() {
    for fault in LocalCommitFault::before_current_publication() {
        let directory = tempfile::tempdir().unwrap();
        let backend = support::local_backend(directory.path(), None);
        let first = support::commit_empty(&backend, None, 1).await.unwrap();
        backend.inject_fault(fault);
        let transaction = backend.begin_generation(Some(first.generation())).await.unwrap();
        assert!(transaction.commit(support::metadata_at(2)).await.is_err());
        let reopened = support::local_backend(directory.path(), None);
        let latest = reopened.open_latest(&support::run_id()).await.unwrap().unwrap();
    assert_eq!(latest.generation().generation(), first.generation());
    }
}
```

`before_current_publication()` enumerates faults after object write, object fsync, object-parent fsync, generation write, generation fsync, generation-parent fsync, temporary CURRENT write, and temporary CURRENT fsync. Separate tests fault after rename and before CURRENT-parent fsync; reopen must yield either the complete old or complete new head, never mixed metadata.

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_local_checkpoint
```

- [ ] **Step 3: Implement exact commit ordering**

```rust
fn publish_current(
    filesystem: &dyn LocalCheckpointFilesystem,
    paths: &GenerationPaths,
    bytes: &[u8],
) -> Result<(), CheckpointError> {
    filesystem.write_new(&paths.current_tmp, bytes)?;
    filesystem.sync_file(&paths.current_tmp)?;
    filesystem.rename(&paths.current_tmp, &paths.current)?;
    filesystem.sync_directory(&paths.root)?;
    Ok(())
}
```

The blocking job performs: validate writer lease and expected CURRENT; write immutable participant/result/index objects with create-new; fsync each new object and parent; write and fsync `generation-N`; fsync its parent; write/fsync temporary CURRENT; rename over CURRENT; fsync root. Decode and validate the current generation after reopen before returning it.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: every fault exposes a complete prior/next generation, two writers cannot both commit, mutation/no-follow tests fail closed, and no filesystem method runs on `LocalSet`.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/checkpoints.rs rust/runtime/src/streaming/checkpoints/local.rs rust/runtime/tests/support/streaming_checkpoint.rs rust/runtime/tests/streaming_local_checkpoint.rs
git commit -m "feat(runtime): persist atomic local checkpoints"
```

### Task 5D: Leased Readers, Bounded Index Traversal, and Garbage Collection

**Depends on:** Tasks 5C and 6A. Start only after both are integrated.

**Files:**
- Create: `rust/runtime/src/streaming/checkpoints/lease_gc.rs`
- Modify: `rust/runtime/src/streaming/checkpoints/local.rs`
- Modify: `rust/runtime/src/streaming/checkpoints.rs`
- Extend: `rust/runtime/tests/support/streaming_checkpoint.rs`
- Create: `rust/runtime/tests/streaming_checkpoint_gc.rs`

**Consumes these exact limits from Task 5B and adds retention policy:**

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointRetentionPolicy {
    pub resume_roots: NonZeroUsize,
    pub partial_history: usize,
    pub retain_final_until_exported: bool,
    pub retain_source_cache_through_resume_root: bool,
    pub orphan_grace_ns: u64,
    pub prepare_lease_ns: u64,
    pub reader_lease_ns: u64,
}
```

- [ ] **Step 1: Write representative RED tests**

```rust
#[tokio::test(flavor = "current_thread")]
async fn reader_lease_prevents_reachable_object_collection() {
    let fixture = support::local_generation_with_segments(5).await;
    let reader = fixture.backend.open_latest(&fixture.run).await.unwrap().unwrap();
    fixture.backend.retain_last_generations(0).await.unwrap();
    fixture.backend.collect_garbage().await.unwrap();
    let page = reader
        .scan_result_index(None, support::index_budget(2, 4096))
        .await
        .unwrap();
    assert_eq!(page.descriptors.len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn renewal_failure_fences_read_before_gc() {
    let fixture = support::local_generation_with_segments(1).await;
    let reader = fixture.backend.open_latest(&fixture.run).await.unwrap().unwrap();
    fixture.clock.advance(fixture.lease_duration + 1);
    fixture.backend.fail_next_renewal();
    assert!(matches!(
        reader.scan_result_index(None, support::index_budget(1, 1024)).await,
        Err(CheckpointError::LeaseLost { .. })
    ));
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_checkpoint_gc
```

- [ ] **Step 3: Implement lease-safe bounded collection**

Mark committed retention roots, valid generation-reader leases, prepared transaction leases, and final-compaction/report leases by traversing index blocks in bounded pages. Wait the authored grace period through injected `Clock`; sweep only objects still unreachable. Private transaction temporary names are never candidates. A reader checks/renews its lease before every block, participant, or segment read.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: committed, prepared, reader, and compactor objects survive; expired unreachable objects are reclaimed; page item/byte limits are never exceeded.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/checkpoints.rs rust/runtime/src/streaming/checkpoints/lease_gc.rs rust/runtime/src/streaming/checkpoints/local.rs rust/runtime/tests/support/streaming_checkpoint.rs rust/runtime/tests/streaming_checkpoint_gc.rs
git commit -m "feat(runtime): lease and collect checkpoint objects"
```

### Task 5E: Checkpoint Coordinator and Post-CAS Notification

**Depends on:** Task 5B. It may run parallel with 5C and 6A.

**Files:**
- Create: `rust/runtime/src/streaming/checkpoint_coordinator.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Create: `rust/runtime/tests/support/streaming_checkpoint_coordinator.rs`
- Create: `rust/runtime/tests/streaming_checkpoint_coordinator.rs`

**Produces:** `StreamingCheckpointCoordinator::commit_barrier`, exact participant-set enforcement, and idempotent post-CAS notification retry.

```rust
impl StreamingCheckpointCoordinator {
    pub async fn commit_barrier(
        &mut self,
        barrier: CheckpointBarrier,
        result_partitions: Vec<ResultPartition>,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError>;
}
```

- [ ] **Step 1: Write representative RED tests**

```rust
#[path = "support/streaming_checkpoint_coordinator.rs"]
mod coordinator_support;

#[tokio::test(flavor = "current_thread")]
async fn post_commit_failure_does_not_roll_back_authoritative_head() {
    let mut fixture = coordinator_support::coordinator_fixture();
    fixture.participant("session").fail_first_commit_notification();
    let error = fixture.coordinator.commit_barrier(
        coordinator_support::barrier_at(3),
        Vec::new(),
    ).await.unwrap_err();
    assert!(matches!(error, CheckpointError::PostCommitNotification { .. }));
    let latest = fixture.backend.open_latest(&fixture.run).await.unwrap().unwrap();
    assert_eq!(latest.generation().generation(), coordinator_support::generation(1));
    fixture.restore_and_replay_notifications().await.unwrap();
    assert_eq!(fixture.participant("session").commit_notifications(), 1);
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_checkpoint_coordinator
```

- [ ] **Step 3: Implement exact ordering**

```rust
// barrier -> views -> validate -> stage -> CAS -> notifications
let views = self.collect_views(&barrier).await?;
self.plan.validate_exact_set(&views)?;
let mut transaction = self.backend.begin_generation(self.expected).await?;
for view in views {
    transaction.stage_participant(view).await?;
}
transaction.stage_results(result_partitions).await?;
let committed = transaction.commit(self.metadata(&barrier)?).await?;
self.notify_committed(&committed).await?;
Ok(committed)
```

Missing/duplicate participants fail before `begin_generation`. Failed staging/CAS drops the transaction and sends no notifications. Notification failure is surfaced after publication and is replayed from committed receipts during restore.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: exact set, no-notify-before-CAS, frozen order, retry, and overlay reclamation tests pass.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/checkpoint_coordinator.rs rust/runtime/tests/support/streaming_checkpoint_coordinator.rs rust/runtime/tests/streaming_checkpoint_coordinator.rs
git commit -m "feat(runtime): coordinate checkpoint publication"
```

### Task 5F1: Built-In Local and None Backend Factories

**Depends on:** Tasks 2 and 5C.

**Files:**
- Create: `rust/runtime/src/streaming/checkpoints/none.rs`
- Create: `rust/runtime/src/streaming/checkpoint_factories.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Modify: `rust/runtime/src/streaming/checkpoints.rs`
- Modify: `rust/runtime/src/extensions/mod.rs`
- Test: `rust/runtime/tests/streaming_checkpoint_factories.rs`

**Produces these exact built-ins:**

```rust
pub const LOCAL_CHECKPOINT_BACKEND_ID: &str = "local";
pub const NONE_CHECKPOINT_BACKEND_ID: &str = "none";
```

- [ ] **Step 1: Write the RED registry test**

```rust
#[test]
fn local_and_none_factories_are_registered() {
    let registry = frozen_streaming_registry();
    assert!(registry.stream_checkpoint_backend_factory("local").is_some());
    assert!(registry.stream_checkpoint_backend_factory("none").is_some());
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_checkpoint_factories
```

Expected: built-in factories are absent.

- [ ] **Step 3: Implement the two factories**

`local` prepares Task-5C storage. `none` advertises no resume, durable reader, encrypted state, or report-retention capability and refuses `begin_generation`; checkpoint mode `none` bypasses checkpoint coordination without a resume claim.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: registry inventory and `none` capability/refusal cases pass.

- [ ] **Step 5: Review and commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/checkpoints.rs rust/runtime/src/streaming/checkpoints/none.rs rust/runtime/src/streaming/checkpoint_factories.rs rust/runtime/src/extensions/mod.rs rust/runtime/tests/streaming_checkpoint_factories.rs
git commit -m "feat(runtime): register local checkpoint backends"
```

### Task 5F2: Conditional Object-Store CAS Backend

**Depends on:** Tasks 5B, 5E, 5F1, and adapter Task A6.

**Files:**
- Create: `rust/runtime/src/streaming/checkpoints/object_store.rs`
- Modify: `rust/runtime/src/streaming/checkpoints.rs`
- Modify: `rust/runtime/src/streaming/checkpoint_factories.rs`
- Modify: `rust/runtime/src/extensions/mod.rs`
- Modify: `rust/runtime/src/streaming/sources/aws_s3_client.rs` only to expose shared AWS client construction, never source listing authority.
- Test: `rust/runtime/tests/streaming_object_checkpoint.rs`

**Produces the conditional capability and a bounded object I/O contract:**

```rust
pub const OBJECT_STORE_CHECKPOINT_BACKEND_ID: &str = "object_store";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObjectKey(String);
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObjectVersion(String);
pub struct PointerObject { pub bytes: Bytes, pub digest: ContentDigest }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectReadRange { pub offset: u64, pub length: u64 }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectReadBudget { pub max_chunk_bytes: usize }
pub struct BudgetOwnedObjectChunk { pub bytes: Bytes, pub lease: BudgetLease }

#[async_trait(?Send)]
pub trait BudgetOwnedObjectReader {
    fn content_length(&self) -> u64;
    fn content_digest(&self) -> ContentDigest;
    async fn next_chunk(&mut self, max_bytes: usize)
        -> Result<Option<BudgetOwnedObjectChunk>, CheckpointError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointFailureCode {
    ObjectLimitExceeded,
    ConditionalWriteUnsupported,
    StaleWriter,
    Provider,
}

#[async_trait(?Send)]
pub trait ConditionalObjectStore: Debug + Send + Sync {
    async fn put_immutable(
        &self,
        object: Box<dyn BudgetOwnedObjectReader>,
    ) -> Result<ObjectVersion, CheckpointError>;
    async fn compare_and_swap_pointer(
        &self,
        key: &ObjectKey,
        expected: Option<&ObjectVersion>,
        next: PointerObject,
    ) -> Result<ObjectVersion, CheckpointError>;
    async fn get_version_range(
        &self,
        key: &ObjectKey,
        version: &ObjectVersion,
        range: ObjectReadRange,
        budget: ObjectReadBudget,
    ) -> Result<BudgetOwnedObjectChunk, CheckpointError>;
}
```

`BudgetOwnedObjectReader` yields bounded chunks while retaining byte permits. `BudgetOwnedObjectChunk` owns its permit until drop. `ObjectReadRange` is checked before provider I/O, and provider metadata whose declared object/page/chunk length exceeds the configured limit is rejected before allocation. This trait has no list/reconcile operation and is not implemented in terms of the S3 source trait.

- [ ] **Step 1: Write RED bounded-I/O and CAS tests**

```rust
#[tokio::test(flavor = "current_thread")]
async fn object_pointer_cas_publishes_exactly_one_complete_generation() {
    let store = FakeConditionalObjectStore::new(object_io_budget(64 * 1024));
    let backend = object_backend(store.clone());
    let left = prepared_transaction(&backend, None, 1).await;
    let right = prepared_transaction(&backend, None, 1).await;
    assert!(left.commit(metadata(1)).await.is_ok() ^ right.commit(metadata(1)).await.is_ok());
    assert!(store.current_pointer_references_only_verified_objects());
}

#[tokio::test(flavor = "current_thread")]
async fn oversized_metadata_is_rejected_before_allocation() {
    let store = FakeConditionalObjectStore::declaring_length(usize::MAX);
    let error = object_backend(store.clone()).restore_current(read_budget(4096)).await.unwrap_err();
    assert_eq!(error.code(), CheckpointFailureCode::ObjectLimitExceeded);
    assert_eq!(store.allocated_bytes(), 0);
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming-s3 --test streaming_object_checkpoint
```

- [ ] **Step 3: Implement bounded immutable uploads and pointer CAS**

Write and verify immutable participant/result/index/generation objects before conditionally replacing one pointer using the exact prior provider version. Stream uploads and ranged restores under permits; never assemble a complete multi-GiB object in `Bytes`. Register `object_store` only under `streaming-s3`. Providers without exact conditional pointer update fail capability agreement before effects.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: stale-writer, every-upload-fault, CAS, crash-after-CAS, feature inventory, oversized-metadata, and bounded chunk high-water cases pass.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/checkpoints.rs rust/runtime/src/streaming/checkpoints/object_store.rs rust/runtime/src/streaming/checkpoint_factories.rs rust/runtime/src/extensions/mod.rs rust/runtime/src/streaming/sources/aws_s3_client.rs rust/runtime/tests/streaming_object_checkpoint.rs
git commit -m "feat(runtime): add bounded object checkpoint CAS"
```

### Task 5F3: Object-Store Leases, GC, and Encrypted-State Capability

**Depends on:** Tasks 5D, 5F2, and sensitive-state Task P6.

**Files:**
- Modify: `rust/runtime/src/streaming/checkpoints/object_store.rs`
- Modify: `rust/runtime/src/streaming/checkpoint_factories.rs`
- Test: `rust/runtime/tests/streaming_object_checkpoint_retention.rs`

- [ ] **Step 1: Write RED retention tests**

```rust
#[tokio::test(flavor = "current_thread")]
async fn leased_reader_survives_bounded_mark_grace_sweep() {
    let (backend, store, clock) = object_backend_with_manual_clock(gc_page_limit(8));
    let lease = backend.open_generation(committed_generation()).await.unwrap();
    publish_and_age_successors(&backend, &clock).await;
    backend.collect_garbage().await.unwrap();
    assert!(lease.read_manifest(read_budget(4096)).await.is_ok());
    assert!(store.max_list_page_items() <= 8);
    assert!(store.max_live_chunk_bytes() <= 4096);
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming-s3 --test streaming_object_checkpoint_retention
```

- [ ] **Step 3: Implement retention and capability agreement**

Implement renewable generation/prepare leases and bounded mark/grace/sweep traversal. Every list page and read chunk owns permits; traversal state is cursor-bounded. Advertise encrypted-sensitive-state capability only when Task-P6 key resolution and authenticated encryption are available. GC must retain committed, prepared, and reader-leased generations and must not inspect source objects.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: lease renewal/expiry, prepared-generation retention, bounded GC, encryption capability/refusal, and provider-fault cases pass.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/checkpoints/object_store.rs rust/runtime/src/streaming/checkpoint_factories.rs rust/runtime/tests/streaming_object_checkpoint_retention.rs
git commit -m "feat(runtime): retain object checkpoint generations"
```

### Task 6A: Result Vocabulary, Membership, and Persistent Bounded Index

**Depends on:** Task 5B. Task 6B later joins these result identities to Task-4 terminal capture and metrics rotation.

**Files:**
- Modify: `rust/runtime/src/streaming/results.rs`
- Create: `rust/runtime/src/streaming/results/index.rs`
- Modify: `rust/runtime/src/engine/records.rs` after `CapturedRecord`.
- Extend: `rust/runtime/tests/support/streaming_checkpoint.rs`
- Create: `rust/runtime/tests/streaming_result_index.rs`

**Produces these exact interfaces:**

```rust
pub struct StreamingRecordCorrelation {
    pub logical_action_id: StableActionId,
    pub attempt_id: ActionAttemptId,
    pub global_sequence: GlobalSequence,
    pub ownership_epoch: SessionOwnershipEpoch,
    pub membership: ResultMembership,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResultMembership {
    Request,
    EndpointGraphAction,
    SessionStateOnly,
    AttemptTelemetry,
}

pub struct CorrelatedRecordIngest {
    pub correlation: StreamingRecordCorrelation,
    pub record: RecordIngest,
    pub captured: Option<CapturedRecord>,
}

pub struct CheckpointResultPlan {
    pub metrics: MetricsCheckpointProjection,
    pub exact_records: Option<ExactRecordProjection>,
    pub raw_records: Option<RawRecordProjection>,
    pub session_results: SessionResultProjection,
    pub provenance: StreamingProvenanceProjection,
    pub interval: CheckpointInterval,
    pub durability: CheckpointDurability,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetricsCheckpointProjection {
    Exact,
    Sketch,
}

pub struct ExactRecordProjection {
    pub schema: ResultSchemaVersion,
}

pub struct RawRecordProjection {
    pub schema: ResultSchemaVersion,
    pub redaction_policy_digest: ContentDigest,
}

pub struct SessionResultProjection {
    pub schema: ResultSchemaVersion,
}

pub struct StreamingProvenanceProjection {
    pub schema: ResultSchemaVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResultPlaneError {
    MembershipConflict { membership_root: ContentDigest },
    ProvisionalCapacityExceeded { items: u64, bytes: u64 },
    InvalidCoverage,
    SegmentVerification,
    Compaction { message: String },
}
```

Implement `Display` and `std::error::Error` directly; errors expose stable lowercase reason codes through a separate `code()` method.

- [ ] **Step 1: Write representative RED tests**

```rust
#[test]
fn conflicting_payload_for_committed_membership_is_rejected() {
    let mut index = ResultIndexBuilder::new(support::index_limits());
    let first = support::segment("records", &[1, 2], b"first");
    let conflict = support::segment("records", &[1, 2], b"different");
    index.insert(first).unwrap();
    assert!(matches!(
        index.insert(conflict),
        Err(ResultPlaneError::MembershipConflict { .. })
    ));
}

#[test]
fn state_only_terminal_never_enters_request_metrics_membership() {
    let correlation = support::state_only_correlation(9);
    assert!(!MetricsCheckpointProjection::default().accepts(correlation.membership));
    assert!(SessionResultProjection::default().accepts(correlation.membership));
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_result_index
```

- [ ] **Step 3: Implement canonical membership and copy-on-write index blocks**

Sort segment facts by `(global_sequence, logical_action_id)`. Use disjoint intervals only when placement proves them; otherwise write sorted logical action IDs into content-addressed membership blocks. Identical descriptor+payload insertion is idempotent. Same reachable membership with different payload is a conflict. An unreachable orphan is not consulted by the logical index. Build new bounded index blocks and structurally share prior blocks; the generation stores only the new root, counts, and bytes.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: membership, retry/conflict/orphan, bounded page, digest verification, and correlation tests pass.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/results.rs rust/runtime/src/streaming/results/index.rs rust/runtime/src/engine/records.rs rust/runtime/tests/support/streaming_checkpoint.rs rust/runtime/tests/streaming_result_index.rs
git commit -m "feat(runtime): index streaming result segments"
```

### Task 6B: Epoch Rotation, Provisional Holes, and Partial Results

**Depends on:** Tasks 5D, 5E, and 6A.

**Files:**
- Create: `rust/runtime/src/streaming/results/epoch.rs`
- Modify: `rust/runtime/src/streaming/results.rs`
- Modify: `rust/runtime/src/metrics.rs` at `NativeMetricsObserver`.
- Modify: `rust/runtime/src/metrics_core/report.rs` at `NativeReport`.
- Extend: `rust/runtime/tests/support/streaming_checkpoint.rs`
- Create: `rust/runtime/tests/streaming_result_epochs.rs`

**Produces:** `EpochResultCoordinator`, bounded `ProvisionalResultStore`, terminal-contiguous cut selection, worker accumulator rotation, and `CommittedPartialResult`.

```rust
pub struct CommittedPartialResult {
    pub generation: CheckpointGeneration,
    pub cut: CheckpointCut,
    pub authoritative_request_count: u64,
    pub active_session_count: u64,
    pub incomplete_session_count: u64,
    pub metrics: BTreeMap<String, MetricEntry>,
    pub provisional: Option<ProvisionalDashboardSummary>,
}

pub struct WorkerResultEpoch {
    pub generation: CheckpointGeneration,
    pub worker_id: u32,
    pub first_sequence: GlobalSequence,
    pub last_sequence: GlobalSequence,
    pub partitions: Vec<ResultPartition>,
}

impl EpochResultCoordinator {
    pub fn observe_terminal(
        &mut self,
        fact: CorrelatedRecordIngest,
    ) -> Result<(), ResultPlaneError>;
    pub async fn prepare_epoch(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<Vec<ResultPartition>, ResultPlaneError>;
    pub fn committed_partial(
        &self,
        generation: &CommittedCheckpointGeneration,
    ) -> Result<CommittedPartialResult, ResultPlaneError>;
}
```

- [ ] **Step 1: Write representative RED tests**

```rust
#[tokio::test(flavor = "current_thread")]
async fn completion_above_terminal_hole_is_provisional_and_bounded() {
    let mut results = support::epoch_results_with_provisional_limit(2);
    results.observe_terminal(support::record(2)).unwrap();
    results.observe_terminal(support::record(3)).unwrap();
    assert!(matches!(
        results.observe_terminal(support::record(4)),
        Err(ResultPlaneError::ProvisionalCapacityExceeded { .. })
    ));
    let partitions = results.prepare_epoch(&support::barrier_terminal_at(0)).await.unwrap();
    assert!(partitions.iter().all(|partition| !partition.contains_sequence(2)));
}

#[tokio::test(flavor = "current_thread")]
async fn closing_hole_commits_each_logical_action_once() {
    let mut fixture = support::epoch_fixture();
    fixture.observe(2).unwrap();
    fixture.observe(1).unwrap();
    fixture.commit_terminal_at(2).await.unwrap();
    let partial = fixture.latest_partial().await.unwrap();
    assert_eq!(partial.terminal_horizon, TerminalActionHorizon::at(2));
    assert_eq!(partial.authoritative_request_count, 2);
    assert_eq!(partial.provisional_request_count, 0);
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,parquet --test streaming_result_epochs
```

- [ ] **Step 3: Implement bounded epoch ownership**

Rotate each worker `MetricsAccumulator` and configured exact/raw/session/provenance projections at the barrier. Hold completions above `H` in immutable provisional partitions charged to prepare/provisional budgets; never link them from a committed root until the hole closes. On exhaustion, fence new admission and return the authored overload decision. Partial views page and merge only committed segments through `H`; provisional dashboard data is separately labeled and excluded from totals.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: long-hole capacity, backpressure, hole closure, exact/sketch rotation, provenance paging, and partial-authority tests pass.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/results.rs rust/runtime/src/streaming/results/epoch.rs rust/runtime/src/metrics.rs rust/runtime/src/metrics_core/report.rs rust/runtime/tests/support/streaming_checkpoint.rs rust/runtime/tests/streaming_result_epochs.rs
git commit -m "feat(runtime): rotate checkpoint result epochs"
```

### Task 6C: Final/Aborted Results, Compaction Ordering, and Delivery Crash Matrix

**Depends on:** Task 6B and the existing `PreparedRunOutcome`/`PreparedReportCommit` interfaces. This task must merge before source or cellular E2E claims restart correctness.

**Files:**
- Create: `rust/runtime/src/streaming/results/compactor.rs`
- Modify: `rust/runtime/src/streaming/results.rs`
- Extend: `rust/runtime/tests/support/streaming_checkpoint.rs`
- Create: `rust/runtime/tests/streaming_result_finalization.rs`

**Delivery-mode contract:**

| Mode | Durable authority | Restart behavior | Result authority |
|---|---|---|---|
| `terminal` | contiguous terminal cut plus full typed participant state | nonterminal actions reissue; duplicate target window is reported | first terminal receipt reachable from a committed root |
| `admitted` | admitted cut plus pending/inflight participant state | admitted but incomplete actions are not reissued | only terminal facts actually committed contribute |
| `decoded` | decoded cut plus complete decoded/session/pending state | useful for ingestion; fidelity completion claim forbidden | terminal result root remains independently bounded |
| `acquired` | acquired cut plus exact reacquisition/decoder state | useful for ingestion; fidelity completion claim forbidden | terminal result root remains independently bounded |
| `none` | no durable generation | fresh process state | no durable partial/final-result claim |

Target idempotency keys always derive from `StableActionId`; `ActionAttemptId` includes a writer-lease-issued run incarnation and attempt ordinal. Verified endpoint idempotency reuses the logical receipt. Without it, crash after target acceptance but before commit may redeliver; the first receipt made reachable from a committed generation is authoritative and later attempts are non-contributing telemetry.

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointDeliveryMode {
    Terminal,
    Admitted,
    Decoded,
    Acquired,
    None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TargetIdempotencyCapability {
    Unsupported,
    VerifiedLogicalActionKey,
}

pub struct DeliveryRestartDecision {
    pub reissue: Vec<StableActionId>,
    pub authoritative_results: Option<ContentDigest>,
    pub claim: DeliveryClaim,
    pub duplicate_window: DuplicateWindow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeliveryClaim {
    AtLeastOnce,
    AtMostOnce,
    IdempotentAtLeastOnceSubmission,
    IngestionOnly,
    None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DuplicateWindow {
    pub may_duplicate_target_effect: bool,
    pub may_lose_target_effect: bool,
}
```

- [ ] **Step 1: Write representative RED finalization and crash tests**

```rust
#[tokio::test(flavor = "current_thread")]
async fn report_lease_releases_only_after_authoritative_report_commit() {
    let fixture = support::sealed_generation_fixture().await;
    let prepared = fixture.compactor.compact(fixture.reader).await.unwrap();
    fixture.backend.collect_garbage().await.unwrap();
    assert!(fixture.backend.contains(prepared.report_digest()));
    prepared.report_commit.commit().unwrap();
    fixture.backend.collect_garbage().await.unwrap();
    assert!(!fixture.backend.has_report_lease());
}

#[tokio::test(flavor = "current_thread")]
async fn unsafe_abort_preserves_last_partial_without_fabricating_terminal_root() {
    let mut fixture = support::run_with_partial_generation().await;
    fixture.fail_participant_view("session");
    let failure = fixture.abort().await.unwrap_err();
    assert_eq!(failure.diagnostic_generation(), fixture.last_partial_digest());
    assert_eq!(fixture.latest_generation().terminal_reason(), None);
}
```

Add a table-driven `delivery_mode_crash_matrix` that injects crashes: before dispatch; after decode; after acquisition; after admission; after target acceptance; after terminal fact; after segment write; after index write; before CAS; after CAS; during post-CAS notification; during compaction; after report write but before `PreparedReportCommit`. Run each case for all five modes and both endpoint-idempotency capabilities. Assert next action set, logical metric membership, attempt IDs, duplicate/loss window, and reported delivery claim.

Add `restart_rejects_changed_topology_projection_or_membership_scheme`; change worker count, cell topology, placement digest, projection plan, and membership scheme independently and assert restore fails before participant initialization or source polling.

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,parquet --test streaming_result_finalization
```

- [ ] **Step 3: Implement deterministic finalization**

```rust
pub struct PreparedStreamingReport {
    pub native_report: NativeReport,
    pub report_digest: ContentDigest,
    pub report_commit: Box<dyn PreparedReportCommit>,
}

#[async_trait(?Send)]
pub trait StreamingResultCompactor {
    async fn compact(
        &self,
        reader: Box<dyn LeasedGenerationReader>,
    ) -> Result<PreparedStreamingReport, ResultPlaneError>;
}
```

Seal and commit the final result generation first. Traverse its index in fixed `(epoch, cell_id, worker_id, projection_id, first_global_sequence, digest)` order using bounded pages. Merge metrics in that order and stream exact JSONL/CSV/Parquet/outputs/provenance through blocking jobs with byte permits. Return the ordinary `NativeReport` and a synchronous commit object that owns the final-generation/report-retention lease. It releases that lease only after the process coordinator has atomically persisted the report and calls `commit`.

If compaction or report persistence fails, retain the sealed generation and return `PreparedRunFailure` with a content-addressed diagnostic root. On execution failure, commit `aborted` only when every required participant supplies a consistent cut; otherwise retain the latest partial generation and name it without fabricating a terminal generation. User cancellation uses the same safe-cut rule.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: every table-driven crash/mode/capability case, deterministic merge, compaction restart, safe/unsafe abort, cancellation, report-write failure, and lease-ordering test passes.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/results.rs rust/runtime/src/streaming/results/compactor.rs rust/runtime/tests/support/streaming_checkpoint.rs rust/runtime/tests/streaming_result_finalization.rs
git commit -m "feat(runtime): finalize checkpointed streaming results"
```

### Task 6D: Coordinator Report-Persistence and Lease Ordering

**Depends on:** Task 6C.

**Files:**
- Modify: `rust/runtime/src/engine/coordinator.rs:483-538`
- Test in: `rust/runtime/src/engine/coordinator.rs`

**Produces:** the generic non-cellular and cellular ordering final generation CAS → leased compaction → durable report rename → synchronous `PreparedReportCommit::commit` → report-retention lease release.

- [ ] **Step 1: Add the in-module RED test**

```rust
#[test]
fn streaming_report_persists_before_commit_lease_release() {
    let fixture = report_persistence_fixture();
    let events = fixture.events();
    persist_prepared_report(
        fixture.outcome(),
        fixture.report_run_metadata(),
        fixture.report_path(),
        fixture.artifact_dir(),
        fixture.export_config(),
        fixture.exporters(),
    ).unwrap();
    assert_eq!(
        events.borrow().as_slice(),
        ["final_generation", "compact", "report_rename", "report_commit", "lease_release"],
    );
}

#[test]
fn streaming_report_failure_retains_generation_and_skips_commit_hook() {
    let fixture = failing_report_persistence_fixture();
    assert!(persist_prepared_report(
        fixture.outcome(),
        fixture.report_run_metadata(),
        fixture.report_path(),
        fixture.artifact_dir(),
        fixture.export_config(),
        fixture.exporters(),
    ).is_err());
    assert!(fixture.final_generation_is_reconstructable());
    assert_eq!(fixture.report_commit_calls(), 0);
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,parquet --lib engine::coordinator::tests::streaming_report_
```

- [ ] **Step 3: Integrate without exposing private helpers**

Keep `persist_prepared_report` private. Thread `PreparedStreamingReport` through `PreparedRunOutcome`, persist the authoritative native report with the existing atomic file path, and call the synchronous commit hook only after rename succeeds. On failure, preserve the leased generation/diagnostic root for reconstruction and do not call the hook.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: both success ordering and persistence-failure retention tests pass in one suite invocation.

- [ ] **Step 5: Review and commit**

```bash
git add rust/runtime/src/engine/coordinator.rs
git commit -m "feat(engine): commit streaming report lease after persistence"
```

## Completion Audit

Before merging Task 6D, verify the following evidence is present in the named task suite:

- `streaming_checkpoint_participants`: six distinct cuts, exact stable participant plan, one-shot restore, non-destructive view.
- `streaming_checkpoint_backend`: atomic participant+result transaction, stale writer conflict, RAII abort.
- `streaming_local_checkpoint`: create-new immutable objects, every fsync/rename fault, expected-head CAS, reopen.
- `streaming_checkpoint_gc`: bounded index pages, reader/prepare/compactor leases, renewal fencing, grace/sweep.
- `streaming_checkpoint_coordinator`: exact set, no notification before CAS, idempotent post-CAS replay.
- `streaming_result_index`: logical versus attempt identity, membership roots, identical retry, conflicting reachable payload, tolerated unreachable orphan.
- `streaming_result_epochs`: exact/sketch rotation, provenance/session projections, terminal holes, provisional bounds, partial authority.
- `streaming_result_finalization`: all delivery modes, endpoint idempotency on/off, crash matrix, deterministic compaction, final/aborted/cancelled outcomes, report lease ordering.

Run the repository-wide gates only after this subsystem plan is integrated into the master plan; they are not substitutes for the one targeted RED/GREEN command in each task.
