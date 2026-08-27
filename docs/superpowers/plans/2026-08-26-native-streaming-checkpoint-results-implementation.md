<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Streaming Checkpoint and Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the atomic, bounded checkpoint and result plane required by native streaming datasets: typed stage cuts, crash-durable local generations, leased bounded indexes, checkpoint-aligned metric/record/session/provenance epochs, delivery-mode restart semantics, and deterministic partial/final/aborted results.

**Architecture:** Every stateful stage exposes one stable `StreamingCheckpointParticipant`. The coordinator collects non-destructive views at one typed cut, stages participant and result objects in one backend transaction, atomically publishes one generation, and only then sends idempotent commit receipts. Results are immutable content-addressed projections reached through a bounded persistent index; final presentation artifacts are streamed from a leased final generation and never become checkpoint authority.

**Tech Stack:** Rust 2024, Tokio current-thread runtimes, `async_trait(?Send)`, BLAKE3, strict Serde DTOs, existing `Clock`, `MetricsAccumulator`, `RecordIngest`, `NativeReport`, `PreparedRunOutcome`, `PreparedReportCommit`, and the Task-1 `StreamingBlockingExecutor`/resource budgets.

**Spec:** `artifacts/streaming-design/streaming-dataset-shadow-replay-design.md` at base approval `505efc06b0`, amended by `3fea6f2fe0`, especially “Checkpoint and delivery semantics” and “Checkpoint-based results”.

## Global Constraints

- Task 5A prerequisites are foundation Tasks 0 and 1A-1B. The mandatory serial order is `5A -> 1C -> 5A-R -> 5B -> 1D`: Task 5A-R retrofits logical-run authority across the landed checkpoint and blocking-participant APIs before either backend or remaining object-safe contracts proceed. Later tasks declare additional dependencies explicitly.
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
5A typed cuts -> 1C blocking owner -> 5A-R run authority -> 5B backend/memory -> 1D contracts
                                                           |-> 5C local durability -> 5D leases/GC --.
                                                           |-> 6A result index ---------------------+-> 6B epochs/holes/partial
                                                           `-> 5E coordinator/post-CAS -------------'          |
                                                                                                                   `-> 6C1 final/aborted -> 6C2 delivery matrix -> 6D report order

2 + 5C -> 5F1 local/none factories
5B + 5E + 5F1 + A0 -> 5F2 object CAS
5D + 5F2 + P6 -> 5F3 object leases/GC/encryption
```

After 5B merges, two worktrees may run concurrently: one owns 5C; the other owns 6A. A third worktree may run 5E because it owns only `checkpoint_coordinator.rs` and its dedicated support/test files. Merge 5C and 6A before starting 5D. Merge 5D, 5E, and 6A before cutting 6B. Tasks 6C1, 6C2, and 6D serialize after 6B. Each worktree lands the minimal parent module declaration needed to compile; the integration owner resolves declaration conflicts. Tasks 5F1-5F3 follow their explicit cross-plan prerequisites.

---

### Task 5A: Typed Cuts and Stable Checkpoint Participants

**Depends on:** foundation Tasks 0 and 1A-1B.

**Files:**
- Create: `rust/runtime/src/streaming/checkpoint.rs`; Task 5A owns typed cuts, canonical generation authority, and the participant declaration consumed first by foundation Task 1C. Task 5A-R later adds logical-run binding before Task 5B or 1D.
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

macro_rules! typed_horizon {
    ($name:ident, $inner:ty) => {
        #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name($inner);
        impl $name {
            pub const fn new(value: $inner) -> Self { Self(value) }
            pub const fn get(&self) -> &$inner { &self.0 }
        }
    };
}

typed_horizon!(DiscoveryHorizon, SourcePosition);
typed_horizon!(AcquisitionHorizon, SourcePosition);
typed_horizon!(DecodeHorizon, SourcePosition);
typed_horizon!(OrderedActionHorizon, GlobalSequence);
typed_horizon!(AdmissionHorizon, GlobalSequence);
typed_horizon!(TerminalActionHorizon, GlobalSequence);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EventTimeWatermark {
    Unknown,
    Hard { through: EventTimeUtc },
    Estimated { through: EventTimeUtc, late_policy_digest: ContentDigest },
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

`PreparedParticipantState` contains participant ID, schema ID/version, represented cut, non-cloneable budget-owned immutable bytes, BLAKE3 digest, item count, and byte length. `CommittedParticipantState` contains the verified descriptor plus newly budget-owned restored bytes. `CommittedParticipantReceipt` binds generation, participant ID, committed descriptor digest, and represented cut. `CheckpointParticipantPlan::new` sorts by stable ID and rejects duplicates.

The frozen plan must contain stable IDs for source, format, event-time/order policy, session coordinator, every prepared action driver binding, placement policy, placement driver, active-execution set, blocking owner, and result/terminal epoch coordinator. Dynamic jobs/actions/segments aggregate under those owners and never become participant IDs. Add `required_stateful_owner_omission_is_rejected`, including independent omission cases for `blocking_owner` and `result_epoch`.

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

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointGenerationCandidate { /* private canonical fields */ }

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct CommittedCheckpointGeneration(CheckpointGenerationCandidate);

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
    descriptor: ParticipantStateDescriptor,
    payload: BudgetedCheckpointBytes,
}

pub struct CommittedParticipantState {
    descriptor: ParticipantStateDescriptor,
    payload: BudgetedCheckpointBytes,
}

pub struct BudgetedCheckpointBytes { bytes: Bytes, lease: BudgetLease }

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommittedParticipantReceipt {
    generation: CheckpointGeneration,
    participant_id: CheckpointParticipantId,
    descriptor_digest: ContentDigest,
    represented_cut: CheckpointCut,
}

impl CommittedCheckpointGeneration {
    pub fn generation(&self) -> CheckpointGeneration { self.0.generation() }
}

impl CommittedParticipantReceipt {
    pub const fn generation(&self) -> &CheckpointGeneration { &self.generation }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CheckpointError {
    AlreadyInitialized,
    GenerationConflict {
        expected: Option<CheckpointGeneration>,
        actual: Option<CheckpointGeneration>,
    },
    ParticipantSetMismatch,
    CutBlockedByInflight { participant: CheckpointParticipantId, job_count: usize },
    StateBudget {
        participant: CheckpointParticipantId,
        code: StateBudgetFailureCode,
    },
    ObjectVerification,
    LeaseLost { generation: CheckpointGeneration },
    PostCommitNotification { participant: CheckpointParticipantId },
    SourceUnavailableOnResume,
    Storage { message: String },
}
```

Review-hardened authority ruling: `CheckpointGenerationCandidate::new`
canonicalizes descriptors and hashes epoch, predecessor, cut, participant-plan
digest and exact IDs, execution-plan digest, result-plan digest, result-index
root, and terminal state. It is serializable/deserializable and self-verifying,
but it exposes no participant/result state and cannot mint a commit receipt.
`verify_against` additionally requires the frozen participant inventory and
both semantic plan digests. `CommittedCheckpointGeneration` is an opaque,
serialize-only authoritative wrapper with no public constructor or
`Deserialize`; Task 5B promotes a candidate only with an opaque move-only proof
created after successful CAS or a leased current-root read. Candidate
deserialization is a custom private-wire-DTO implementation that performs
self-verification; unchecked derived `Deserialize` is forbidden. Only that wrapper
can construct `CommittedParticipantReceipt`. Task 5A-R adds the run field to
these already-checked authority types after the Task 1C participant exists.

All invariant-bearing state fields are private and checked-construction-only.
`BudgetedCheckpointBytes::new` compact-copies the visible input into exact-sized
owned immutable storage before comparing it with the inseparable move-only
lease; a small `Bytes` slice may not hide a large retained allocation. Restore
verifies charge, exact byte length, and BLAKE3 before producing
`CommittedParticipantState`. Borrowing/consuming accessors preserve the lease
with its bytes.

Implement `Display` and `std::error::Error` directly, following existing runtime library error enums; do not add `thiserror` or another dependency.

`StateBudgetFailureCode` is owned by foundation Task 1A. Foundation Task 1D's
`StableStreamingFailure` mapping reports `CheckpointError::StateBudget` at
`StreamingFailureStage::StateBudget` with the exact nested stable code; it must
not collapse this variant into checkpoint storage failure.

- [ ] **Step 1: Write representative RED tests**

```rust
#[test]
fn horizon_domains_cannot_be_substituted_and_round_trip() {
    let cut = support::cut_at(7);
    let encoded = serde_json::to_vec(&cut).unwrap();
    let restored: CheckpointCut = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(restored, cut);
    assert_eq!(restored.decoded.get(), cut.decoded.get());
    assert_eq!(restored.terminal.get(), cut.terminal.get());
}

#[tokio::test(flavor = "current_thread")]
async fn participant_view_is_non_destructive_before_backend_commit() {
    let mut participant = support::CountingParticipant::new("session", 4);
    participant.initialize(None).await.unwrap();
    let _prepared = participant.checkpoint_view(&support::barrier_at(4)).await.unwrap();
    assert_eq!(participant.released_items(), 0);
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

Run the Step-2 command. Expected: all typed-domain, duplicate-ID, one-shot
initialization, compact byte ownership, candidate verification, plan binding,
and non-destructive-view tests pass. Post-CAS receipt/idempotent-notification
tests belong to Tasks 5B/5E after a backend returns authoritative commitment.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/checkpoint.rs rust/runtime/tests/support/streaming_checkpoint.rs rust/runtime/tests/streaming_checkpoint_participants.rs
git commit -m "feat(runtime): define streaming checkpoint cuts"
```

### Task 5A-R: Logical-Run Authority Retrofit

**Depends on:** landed Tasks 5A and foundation 1C. This task is mandatory and
serial: `5A -> 1C -> 5A-R -> 5B -> 1D`. It resolves the cross-run authority gap
introduced only after the frozen Task 1C contract was implemented.

**Files:**
- Modify: `rust/runtime/src/streaming/checkpoint.rs`
- Modify: `rust/runtime/src/streaming/blocking.rs`
- Extend: `rust/runtime/tests/support/streaming_checkpoint.rs`
- Extend: `rust/runtime/tests/streaming_checkpoint_participants.rs`
- Update all five existing external `CheckpointGenerationCandidate::verify_against`
  calls in `rust/runtime/tests/streaming_checkpoint_participants.rs` to pass the
  expected run; update the internal call in
  `CheckpointGenerationCandidate::promote` in `checkpoint.rs` as part of the
  same signature change, along with the existing `checkpoint.rs` unit callers
  of `promote`.
- Extend: `rust/runtime/tests/streaming_blocking.rs`
- Unit tests: `#[cfg(test)]` in `rust/runtime/src/streaming/checkpoint.rs` owns
  crate-private publication-proof/promotion and authoritative counting-participant
  receipt cases.
- Unit tests: `#[cfg(test)]` in `rust/runtime/src/streaming/blocking.rs` owns the
  authoritative blocking-participant receipt case.

**Produces:** a checked stable run identity around `LogicalReplayRunId`,
run-bound barriers/state/generations/receipts, and participant-side refusal
before any mutation.

```rust
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StreamRunIdentity(LogicalReplayRunId);

impl StreamRunIdentity {
    pub const fn new(logical_replay_run: LogicalReplayRunId) -> Self {
        Self(logical_replay_run)
    }
    pub const fn logical_replay_run(&self) -> &LogicalReplayRunId {
        &self.0
    }
}

pub struct CheckpointBarrier {
    pub run: StreamRunIdentity,
    pub epoch: CheckpointEpoch,
    pub cut: CheckpointCut,
    pub plan_digest: ContentDigest,
}

impl PreparedParticipantState {
    pub fn new(
        run: StreamRunIdentity,
        participant_id: CheckpointParticipantId,
        schema_id: impl Into<String>,
        schema_version: u32,
        represented_cut: CheckpointCut,
        item_count: u64,
        payload: BudgetedCheckpointBytes,
    ) -> Result<Self, CheckpointError>;
    pub const fn run(&self) -> &StreamRunIdentity;
    pub fn into_parts(
        self,
    ) -> (
        StreamRunIdentity,
        ParticipantStateDescriptor,
        BudgetedCheckpointBytes,
    );
}

impl CommittedParticipantState {
    pub fn new(
        run: StreamRunIdentity,
        descriptor: ParticipantStateDescriptor,
        payload: BudgetedCheckpointBytes,
    ) -> Result<Self, CheckpointError>;
    pub const fn run(&self) -> &StreamRunIdentity;
}

impl CheckpointGenerationCandidate {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run: StreamRunIdentity,
        epoch: CheckpointEpoch,
        previous: Option<ContentDigest>,
        cut: CheckpointCut,
        participant_plan: &CheckpointParticipantPlan,
        execution_plan_digest: ContentDigest,
        result_plan_digest: ContentDigest,
        participant_descriptors: Vec<ParticipantStateDescriptor>,
        result_index_root: ContentDigest,
        is_final: bool,
        terminal_reason: Option<CheckpointTerminalReason>,
    ) -> Result<Self, CheckpointError>;
    pub const fn run(&self) -> &StreamRunIdentity;
    pub fn verify_against(
        &self,
        expected_run: &StreamRunIdentity,
        participant_plan: &CheckpointParticipantPlan,
        execution_plan_digest: &ContentDigest,
        result_plan_digest: &ContentDigest,
    ) -> Result<(), CheckpointError>;
    pub(crate) fn promote(
        self,
        expected_run: &StreamRunIdentity,
        participant_plan: &CheckpointParticipantPlan,
        execution_plan_digest: &ContentDigest,
        result_plan_digest: &ContentDigest,
        proof: CheckpointGenerationPublicationProof,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError>;
}

impl CommittedCheckpointGeneration {
    pub const fn run(&self) -> &StreamRunIdentity;
}

impl CommittedParticipantReceipt {
    pub const fn run(&self) -> &StreamRunIdentity;
}
```

`ParticipantStateDescriptor` deliberately remains run-free: its public strict
DTO shape and digest semantics do not change, so byte-identical participant
state in two runs has the same descriptor digest. `PreparedParticipantState`
and `CommittedParticipantState` instead carry private run identity around that
descriptor and payload. Their constructors reject wrapper/context disagreement,
and their borrow accessors cannot detach run from payload authority.
`PreparedParticipantState::into_parts` consumes the wrapper and returns run,
descriptor, and still-budgeted bytes together, so Task 5B staging cannot erase
the run while moving the payload. Canonical candidate
construction takes the run, stores it privately, and includes it as its own
length-framed field in the domain-separated generation digest. The canonical
hash domain is bumped from v2 to exactly
`aiperf.streaming.committed-checkpoint-generation.v3`; immediately after the
length-framed domain, hash one distinct length-framed field containing the raw
`candidate.run.logical_replay_run().as_bytes()`, then the existing epoch,
predecessor, cut, plan, descriptor, result-root, and terminal fields in their
existing order. Do not hash the wrapper's Serde encoding or mix run bytes into
another field. Custom candidate
deserialization verifies the serialized run along with the digest. Promotion
proofs bind the exact run-bound generation. The private run propagates unchanged
through `CommittedCheckpointGeneration` and `CommittedParticipantReceipt`, with
borrow-only accessors.

`verify_against` compares `self.run()` with `expected_run` first, before
self-hash, participant inventory, or semantic-plan checks. A mismatch returns
`CheckpointError::ObjectVerification`. `promote` also requires an explicit
`expected_run`, passes it into `verify_against`, and only then compares the
opaque publication proof with the verified generation. A self-consistent
foreign-run candidate plus its own matching proof therefore cannot be promoted
under the caller's expected run.

Every participant stores the exact initialized/frozen run. It rejects a foreign
barrier before fencing admission, acquiring state budget, or changing any
prepared/committed field. It rejects a foreign receipt before epoch/digest
idempotency checks or state release. The Task 1C blocking owner follows the same
ordering, so foreign input cannot change `is_accepting`, completed horizon,
prepared descriptor, committed receipt, or any budget snapshot.

`StreamRunIdentity` accepts only `LogicalReplayRunId`; it never contains,
derives from, or accepts `RunIncarnationId`. Task 5C remains the sole owner of
incarnation allocation while acquiring durable writer authority.

- [ ] **Step 1: Write representative RED tests**

```rust
#[test]
fn identical_generation_content_in_distinct_runs_has_distinct_digest() {
    let first = support::candidate_for_run(support::run_id(1), 7);
    let second = support::candidate_for_run(support::run_id(2), 7);
    assert_ne!(first.generation().digest(), second.generation().digest());
}

#[test]
fn serialized_candidate_rejects_tampered_run() {
    let first = support::candidate_for_run(support::run_id(1), 7);
    let tampered = support::replace_serialized_run(&first, support::run_id(2));
    assert!(serde_json::from_slice::<CheckpointGenerationCandidate>(&tampered).is_err());
}

/// ```compile_fail
/// # use aiperf_runtime::streaming::{checkpoint::StreamRunIdentity, identity::RunIncarnationId};
/// let _ = StreamRunIdentity::new(RunIncarnationId::from_bytes([1; 32]));
/// ```

#[tokio::test(flavor = "current_thread")]
async fn foreign_barrier_is_rejected_before_blocking_owner_fences() {
    let mut owner = support::blocking_owner_for_run(1);
    let before = owner.snapshot();
    assert!(matches!(
        owner.checkpoint_view(&support::barrier_for_run(2, 7)).await,
        Err(CheckpointError::ObjectVerification)
    ));
    assert_eq!(owner.snapshot(), before);
    assert!(owner.snapshot().is_accepting);
}
```

Keep crate-private authority construction out of integration support. Add these
RED cases inside the production modules' existing `#[cfg(test)]` modules, where
the private proof and participant internals are legitimately visible:

```rust
// rust/runtime/src/streaming/checkpoint.rs
#[test]
fn v3_run_bound_generation_digest_is_stable() {
    // Fixed fixture: run=[0x01;32], epoch=7, no predecessor, participant
    // "session"/"test.v1", cut=7 with unknown event watermark, descriptor
    // content=[0x44;32], item_count=1, byte_length=4, execution=[0x11;32],
    // result plan=[0x12;32], result root=[0x55;32], non-final.
    let candidate = v3_golden_candidate();
    assert_eq!(
        candidate.generation().digest(),
        &ContentDigest::from_bytes([
            0x51, 0x9b, 0xf1, 0x92, 0x51, 0x8f, 0x43, 0xe9,
            0xd4, 0xac, 0xcd, 0x6b, 0xd8, 0xed, 0x38, 0xe8,
            0x85, 0xa1, 0xdc, 0xe0, 0x6d, 0x8d, 0x35, 0x57,
            0x9b, 0xf5, 0xf9, 0x9b, 0x79, 0x4d, 0x10, 0xf1,
        ]),
    );
}

#[test]
fn cross_run_publication_proof_cannot_promote_candidate() {
    let (first, _, _, _) = candidate_fixture_for_run(run_id(1), 7);
    let (second, plan, execution_plan, result_plan) =
        candidate_fixture_for_run(run_id(2), 7);
    let wrong_generation_proof =
        CheckpointGenerationPublicationProof::for_generation(first.generation());
    assert!(second.clone()
        .promote(
            &run_id(2),
            &plan,
            &execution_plan,
            &result_plan,
            wrong_generation_proof,
        )
        .is_err());

    let matching_foreign_proof =
        CheckpointGenerationPublicationProof::for_generation(second.generation());
    assert!(second
        .promote(
            &run_id(1),
            &plan,
            &execution_plan,
            &result_plan,
            matching_foreign_proof,
        )
        .is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn greater_epoch_foreign_receipt_does_not_mutate_counting_participant() {
    let mut counting = counting_participant_for_run(1);
    let (local, foreign) = authoritative_receipts_with_identical_descriptor(
        (run_id(1), 1),
        (run_id(2), 99),
    );
    assert_eq!(local.descriptor_digest(), foreign.descriptor_digest());
    let counting_before = counting.snapshot();
    assert!(counting.checkpoint_committed(&foreign).await.is_err());
    assert_eq!(counting.snapshot(), counting_before);
}

// rust/runtime/src/streaming/blocking.rs
#[tokio::test(flavor = "current_thread")]
async fn greater_epoch_foreign_receipt_does_not_mutate_blocking_owner() {
    let mut blocking = blocking_owner_for_run(1);
    let (local, foreign) = authoritative_receipts_with_identical_descriptor(
        (run_id(1), 1),
        (run_id(2), 99),
    );
    assert_eq!(local.descriptor_digest(), foreign.descriptor_digest());
    let blocking_before = blocking.snapshot();
    assert!(blocking.checkpoint_committed(&foreign).await.is_err());
    assert_eq!(blocking.snapshot(), blocking_before);
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_checkpoint_participants --test streaming_blocking
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --lib
```

Expected: run-bound constructors/accessors are absent and the cross-run tests do
not compile or fail behaviorally against landed 5A+1C. The integration suite
uses only public digest/deserialization/barrier APIs; the full library run owns
private proof/promotion and authoritative receipt construction.

- [ ] **Step 3: Implement the minimal run-bound retrofit**

Add run-bound checked constructors and private fields to participant state,
barrier, canonical candidate, committed generation, and committed receipt. Do
not add run to `ParticipantStateDescriptor`; preserve its strict public DTO and
digest. Make prepared-state consumption transfer the run, descriptor, and
budgeted payload as one tuple.
Update the five external `verify_against` calls and the internal `promote`
verification call to pass an explicit expected run. Update canonical
hashing/deserialization/proof verification and pin the v3 digest. Initialize the
counting and blocking participants with one immutable run and perform run checks
before every fencing or notification mutation. Do not add backend behavior,
incarnation identity, or writer leasing.

- [ ] **Step 4: Verify GREEN**

Run Step 2, including the complete streaming-feature library suite, then run:

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --doc
```

Expected:
the fixed v3 digest, digest separation, tamper/proof refusal, incarnation type exclusion, foreign
barrier no-op, and greater-epoch foreign receipt no-op pass for both participant
fixtures without regressing Task 1C cancellation/checkpoint tests.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/checkpoint.rs rust/runtime/src/streaming/blocking.rs rust/runtime/tests/support/streaming_checkpoint.rs rust/runtime/tests/streaming_checkpoint_participants.rs rust/runtime/tests/streaming_blocking.rs
git commit -m "fix(runtime): bind checkpoint participants to logical runs"
```

### Task 5B: Atomic Backend Contract and In-Memory Reference

**Depends on:** Task 5A-R. Foundation Task 1D starts only after this task lands.

**Budget contract correction:**
`artifacts/streaming-design/checkpoint-backend-budget-contract-correction.md`.
Task 5B owns the backend/read vocabulary below; it does not reopen Task 5A-R's
run-authority behavior.

**Files:**
- Modify: `rust/runtime/src/streaming/checkpoint.rs`; Task 5B adds only the
  stable backend/read budget error vocabulary and `Display` branches below.
- Create: `rust/runtime/src/streaming/checkpoint_backend.rs`
- Create: `rust/runtime/src/streaming/checkpoints.rs`
- Create: `rust/runtime/src/streaming/checkpoints/memory.rs`
- Create: `rust/runtime/src/streaming/results.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Extend: `rust/runtime/tests/support/streaming_checkpoint.rs`; Task 5B owns run-scoped backend fixtures while preserving the Task 5A-R participant fixtures.
- Create: `rust/runtime/tests/streaming_checkpoint_backend.rs`

**Produces these exact interfaces:**

```rust
#[async_trait(?Send)]
pub trait StreamingCheckpointBackend {
    async fn open_latest(
        &self,
        run: &StreamRunIdentity,
        expected: &CheckpointGenerationExpectations,
    ) -> Result<Option<Box<dyn LeasedGenerationReader>>, CheckpointError>;
    async fn begin_generation(
        &self,
        run: StreamRunIdentity,
        expected: Option<CheckpointGeneration>,
        expectations: CheckpointGenerationExpectations,
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
        partitions: &mut Vec<ResultPartition>,
    ) -> Result<PreparedResultEpoch, CheckpointError>;
    async fn commit(
        self: Box<Self>,
        metadata: CheckpointCommitMetadata,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError>;
}
```

Task 5B also defines the content-neutral DTOs needed by those object-safe signatures: `BudgetedResultDescriptor`, `ResultPartition`, `PreparedResultEpoch`, `ResultSegmentDescriptor`, `ResultSegmentReader`, `ResultIndexCursor`, `ResultIndexReadBudget`, and `ResultIndexPage`. At this stage a partition is verified bytes plus projection/schema/range/count/digest metadata; Task 6A adds logical membership construction and conflict policy without changing the backend signatures. This ordering avoids a backend/results module cycle.

```rust
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CellId(u32);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorkerId(u32);

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ResultProjectionId(Box<str>);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResultSchemaVersion(u32);

/// Singular result descriptor with inseparable retained-allocation authority.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::BudgetedResultDescriptor;
/// # fn cannot_separate(value: BudgetedResultDescriptor) {
/// let _descriptor = value.descriptor;
/// let _lease = value.lease;
/// # }
/// ```
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::BudgetedResultDescriptor;
/// # fn cannot_use_backend_transfer(value: BudgetedResultDescriptor) {
/// let _ = value.into_backend_parts();
/// # }
/// ```
pub struct BudgetedResultDescriptor {
    descriptor: ResultSegmentDescriptor,
    lease: BudgetLease,
}

/// Verified result payload and its inseparable budgeted descriptor.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::ResultPartition;
/// # fn cannot_separate(value: ResultPartition) {
/// let _descriptor = value.descriptor;
/// let _payload = value.payload;
/// # }
/// ```
pub struct ResultPartition {
    descriptor: BudgetedResultDescriptor,
    payload: BudgetedCheckpointBytes,
}

pub struct PreparedResultEpoch {
    index_root: ContentDigest,
    descriptors: BudgetedResultDescriptors,
    item_count: u64,
    byte_length: u64,
}

/// Descriptor collection with inseparable aggregate allocation authority.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::BudgetedResultDescriptors;
/// # fn cannot_separate(value: BudgetedResultDescriptors) {
/// let _descriptors = value.descriptors;
/// let _lease = value.lease;
/// # }
/// ```
pub struct BudgetedResultDescriptors {
    descriptors: Box<[ResultSegmentDescriptor]>,
    lease: BudgetLease,
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
    payload: BudgetedCheckpointBytes,
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

/// One budgeted page of reachable result descriptors.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::ResultIndexPage;
/// # fn cannot_separate(value: ResultIndexPage) {
/// let _descriptors = value.descriptors;
/// let _next = value.next;
/// # }
/// ```
pub struct ResultIndexPage {
    descriptors: BudgetedResultDescriptors,
    next: Option<ResultIndexCursor>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointCommitMetadata {
    pub previous: Option<CheckpointGeneration>,
    pub epoch: CheckpointEpoch,
    pub cut: CheckpointCut,
    pub execution_plan_digest: ContentDigest,
    pub result_plan_digest: ContentDigest,
    pub is_final: bool,
    pub terminal_reason: Option<CheckpointTerminalReason>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointGenerationExpectations {
    pub run: StreamRunIdentity,
    pub participant_plan: CheckpointParticipantPlan,
    pub execution_plan_digest: ContentDigest,
    pub result_plan_digest: ContentDigest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryCheckpointLimits {
    pub transactions: BudgetLimits,
    pub prepared_indexes: BudgetLimits,
    pub storage: BudgetLimits,
    pub result_summaries: BudgetLimits,
    pub reads: BudgetLimits,
}

impl MemoryCheckpointBackend {
    pub fn new(limits: MemoryCheckpointLimits) -> Result<Self, CheckpointError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointBackendBudgetKind {
    Transaction,
    PreparedIndex,
    Storage,
    ResultSummary,
    Read,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointBackendBudgetFailureCode {
    ItemCapacity,
    ByteCapacity,
    Closed,
    Unrepresentable,
}

pub enum CheckpointError {
    // Existing Task 5A/5A-R variants remain unchanged.
    BackendBudget {
        budget: CheckpointBackendBudgetKind,
        code: CheckpointBackendBudgetFailureCode,
    },
    ResultIndexReadBudgetTooSmall {
        required_bytes: u64,
        max_bytes: u64,
    },
}
```

The snippets above show private fields intentionally. Implement these exact
checked ownership APIs in `results.rs`:

```rust
impl ResultProjectionId {
    pub fn new(value: impl Into<String>) -> Result<Self, CheckpointError>;
    pub fn as_str(&self) -> &str;
    pub fn retained_allocation_bytes(&self) -> usize;
}

impl<'de> Deserialize<'de> for ResultProjectionId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

impl ResultPartition {
    pub fn new(
        descriptor: BudgetedResultDescriptor,
        payload: BudgetedCheckpointBytes,
    ) -> Result<Self, CheckpointError>;
    pub fn descriptor(&self) -> &ResultSegmentDescriptor;
    pub fn descriptor_charged_bytes(&self) -> usize;
    pub fn payload_bytes(&self) -> &[u8];
    pub fn into_parts(self) -> (BudgetedResultDescriptor, BudgetedCheckpointBytes);
}

impl BudgetedResultDescriptor {
    pub fn new(
        descriptor: ResultSegmentDescriptor,
        lease: BudgetLease,
    ) -> Result<Self, CheckpointError>;
    pub fn descriptor(&self) -> &ResultSegmentDescriptor;
    pub fn charged_bytes(&self) -> usize;
    pub(crate) fn into_backend_parts(self) -> (ResultSegmentDescriptor, BudgetLease);
}

impl BudgetedResultDescriptors {
    pub(crate) fn new(
        descriptors: Box<[ResultSegmentDescriptor]>,
        lease: BudgetLease,
    ) -> Result<Self, CheckpointError>;
    pub fn descriptors(&self) -> &[ResultSegmentDescriptor];
    pub fn charged_bytes(&self) -> usize;
}

impl PreparedResultEpoch {
    pub(crate) fn new(
        index_root: ContentDigest,
        descriptors: BudgetedResultDescriptors,
        item_count: u64,
        byte_length: u64,
    ) -> Result<Self, CheckpointError>;
    pub fn index_root(&self) -> &ContentDigest;
    pub fn descriptors(&self) -> &[ResultSegmentDescriptor];
    pub fn item_count(&self) -> u64;
    pub fn byte_length(&self) -> u64;
    pub fn into_parts(
        self,
    ) -> (ContentDigest, BudgetedResultDescriptors, u64, u64);
}

impl ResultSegmentReader {
    pub(crate) fn new(
        descriptor: &ResultSegmentDescriptor,
        payload: BudgetedCheckpointBytes,
    ) -> Result<Self, CheckpointError>;
    pub fn payload_bytes(&self) -> &[u8];
    pub fn into_payload(self) -> BudgetedCheckpointBytes;
}

impl ResultIndexPage {
    pub(crate) fn new(
        descriptors: BudgetedResultDescriptors,
        next: Option<ResultIndexCursor>,
    ) -> Result<Self, CheckpointError>;
    pub fn descriptors(&self) -> &[ResultSegmentDescriptor];
    pub fn next(&self) -> Option<&ResultIndexCursor>;
    pub fn charged_bytes(&self) -> u64;
    pub fn into_parts(
        self,
    ) -> (BudgetedResultDescriptors, Option<ResultIndexCursor>);
}
```

`ResultProjectionId::new` rejects an empty value and compacts the accepted text
with `String::into_boxed_str`. Its custom `Deserialize` implementation decodes
through that checked constructor; derive must not bypass the nonempty invariant.
`BudgetedResultDescriptor::new` requires exactly one charged item and exact
bytes equal to `size_of::<ResultSegmentDescriptor>()` plus the compact nested
projection length, using the same checked retained-byte helper as descriptor
pages. `ResultPartition::new` accepts that wrapper intact and
`ResultSegmentReader::new` verifies exact payload length and raw BLAKE3 digest
against the descriptor. The segment reader borrows that
descriptor only for verification and retains no descriptor clone, so its payload
lease is sufficient and `into_payload` cannot expose an uncharged allocation.
`BudgetedResultDescriptors::new` requires lease items
equal to descriptor count and lease bytes equal to the checked sum of
`descriptors.len() * size_of::<ResultSegmentDescriptor>()` plus every compact
`descriptor.projection.as_str().len()`. Empty descriptor slices use an exact
`(0, 0)` lease. `PreparedResultEpoch::new` checked-adds descriptor item/byte
totals and requires those computed totals to equal its supplied `item_count`
and `byte_length`. `ResultIndexPage::new` checked-converts the inseparable
descriptor lease charge rather than retaining a second caller-supplied value.
No public accessor or consuming method exposes `BudgetLease` or an unbudgeted
descriptor allocation. `ResultPartition::into_parts` returns the singular
descriptor wrapper intact beside the separately budgeted payload.
`BudgetedResultDescriptors` has no `into_parts` method; the enclosing wrappers'
consuming methods move that wrapper intact. Only the crate-private
`BudgetedResultDescriptor::into_backend_parts` may dismantle the singular
wrapper, and `stage_results` may call it only after acquiring the exact aggregate
prepared-index reservation described below.

`MemoryCheckpointBackend::new` is the only backend constructor. Validate all
five `MemoryCheckpointLimits` entries in field order before retaining state or
budgets. For each kind, precheck zero `max_items` as `ItemCapacity` and zero
`max_bytes` as `ByteCapacity`, then delegate construction to the existing
`StreamingResourceBudget::new` validator. Map its `PermitCountTooLarge` from
either dimension to `Unrepresentable`; do not duplicate a looser
`Semaphore::MAX_PERMITS`-only check. Direct RED calls use `.unwrap()`;
support fixture constructors perform that same unwrap internally and return a
fully initialized backend. Do not add an infallible `new`, `new_unchecked`, or
alternate `try_new`.

`support::invalid_backend_limits()` is an exact 20-case matrix: for each of the
five budget kinds, mutate only that kind to zero items, zero bytes, items above
the existing validator's `u32::MAX` `acquire_many` conversion boundary, or bytes
above that same boundary while every other field remains valid. The expected
tuple names that kind and respectively uses `ItemCapacity`, `ByteCapacity`,
`Unrepresentable`, or `Unrepresentable`. A separate boundary case proves exact
`u32::MAX` item and byte limits are accepted on the supported 64-bit target and
`u32::MAX + 1` is the first conversion refusal.

```rust
fn memory_backend(limits: MemoryCheckpointLimits) -> MemoryCheckpointBackend {
    MemoryCheckpointBackend::new(limits).unwrap()
}
```

`BackendBudget` is exclusively for capacity owned by the checkpoint backend; it
must not be collapsed into participant `StateBudget`, `Storage`, or
`ObjectVerification`. Map one request that exceeds configured item capacity to
`ItemCapacity`, otherwise a request that exceeds configured byte capacity to
`ByteCapacity`; item capacity wins if both are exceeded. Map a closed budget to
`Closed`, and an unrepresentable permit count or accounting transition to
`Unrepresentable`. Ordinary contention waits cancellation-safely and is not an
error. Add these exact `Display` branches to the existing match:

```rust
Self::BackendBudget { budget, code } => write!(
    formatter,
    "checkpoint backend {budget:?} budget failed: {code:?}",
),
Self::ResultIndexReadBudgetTooSmall {
    required_bytes,
    max_bytes,
} => write!(
    formatter,
    "result-index page requires {required_bytes} retained bytes but the caller allowed {max_bytes}",
),
```

`begin_generation` first requires its explicit `run` to equal
`expectations.run`, then freezes both into the transaction before any staging.
Every staged `ResultSegmentDescriptor.run` must equal that transaction run.
`commit` requires metadata's explicitly named semantic digests to match, builds
the canonical candidate with the same run, performs conditional publication
against that run's head, and only then creates the opaque exact-generation proof
used for promotion. The memory reference stores a separate head and immutable
object namespace for each `StreamRunIdentity`; a commit or stale writer on one
run cannot observe, replace, or conflict with another run's head.

`open_latest` first requires its explicit `run` to equal `expected.run`, then
acquires only that run's current-root lease, decodes a candidate, and calls
`verify_against` with the exact run plus supplied expectations before exposing
an authoritative reader. This run binding applies even to empty generations:
identical epochs, cuts, descriptors, and result roots in different logical runs
remain distinct generations. No participant or result bytes are readable before
this promotion boundary.

Task 5B does not allocate or persist `RunIncarnationId` and does not acquire a
durable writer lease. Task 5C owns incarnation allocation and the durable
single-writer/fencing protocol. The in-memory Task 5B CAS is run-scoped reference
semantics, not durable writer authority.

Run discovery is a product-boundary responsibility, not a backend fallback.
Task V1 in the product/verification plan owns the Config-v2/protocol projection
of an explicit fresh-or-resume choice and the resume locator that carries the
exact `StreamRunIdentity`. A fresh invocation allocates its logical identity and
commits the bootstrap generation before any source poll, endpoint issue, or
externally visible result; that generation contains the exact participant
inventory and one canonical zero-partition result epoch. A resume invocation
must receive that same identity through the explicit locator/product projection,
or recover it through a future catalog selected by that locator. Missing or
unresolvable resume identity is a refusal: neither Task 5B nor any caller may
silently allocate a replacement logical run. Task V1 also owns the product
ordering that commits the bootstrap before source polling or endpoint issue.
Task 5E receives the already resolved identity by constructor injection and
uses it for canonical bootstrap/barrier commits; it performs no fresh
allocation, resume lookup, locator parsing, or fallback selection. Task 5C
allocates only a new writer incarnation after the logical run has been resolved.

Every transaction must stage the exact frozen participant inventory and call
`stage_results` exactly once. A zero-partition result epoch is valid, but an
omitted result epoch is not: it produces one canonical empty index root with
zero item/byte totals. A second `stage_results` call is rejected. At commit,
every result descriptor must match both the transaction run and
`CheckpointCommitMetadata.epoch`; run is checked during staging and rechecked at
commit, while epoch is authoritative only when metadata is supplied. No
participant state or result epoch may be inferred from absence.

`stage_results` first validates the input partitions by borrow and checked-sums
their exact singular descriptor charges, result totals, and canonical index
inputs. It then acquires both the backend prepared-index reservation and the
separate returned-summary reservation for the complete descriptor count and
retained-byte sum while the caller's vector and every singular input authority
remain intact. Only after both acquisitions succeed may one synchronous helper
clone the borrowed descriptors into exact-capacity prepared and summary
collections, construct both checked aggregate wrappers, and construct the
returned `PreparedResultEpoch` while the input vector is still unchanged. The
final infallible helper `mem::take`s the vector, invokes the crate-private
`BudgetedResultDescriptor::into_backend_parts`, moves each separately budgeted
payload into transaction storage, drops each original descriptor plus input
lease, and installs the already-built prepared collection. There is no fallible
operation or `.await` after the caller vector or transaction state begins to
mutate.

Refusal or cancellation during either acquisition leaves the transaction and
caller vector unchanged and retryable; dropping the pending future releases any
first aggregate lease already acquired. Any checked construction error releases
the aggregate leases while leaving all input authorities in the caller vector.
The
backend's private prepared-index and returned-summary budgets are never exposed
for caller partition construction, preventing self-contention with input
authority. Commit likewise completes every storage acquisition and validation
before borrowing `MemoryState` mutably; object insertion plus head CAS/publication
is one non-async critical section with no `.await` after publication begins.

A leased reader exposes only descriptors reachable from its exact committed
generation. `read_participant` requires the complete supplied descriptor to be
present in that generation's participant inventory. `read_segment` requires the
complete supplied descriptor to be reachable from that generation's verified
result-index root. A content-addressed object existing elsewhere in the backend,
including another generation or run, is not read authority.

Cursor and reachability validation precede every read-budget decision. Once the
next reachable descriptor is known, compute the actual compact retained
allocation of a one-descriptor page. If it exceeds
`ResultIndexReadBudget.max_bytes`, return
`ResultIndexReadBudgetTooSmall { required_bytes, max_bytes }` before acquiring
backend read capacity. Never return an empty page with an unchanged cursor. If
the caller limit is sufficient but the configured backend read budget cannot
admit that page, return `BackendBudget { budget: Read, ... }`. Neither refusal
advances the cursor, changes the reader, or changes the authoritative head.

- [ ] **Step 1: Write representative RED tests**

```rust
#[test]
fn backend_budget_codes_have_stable_names() {
    assert_eq!(
        serde_json::to_string(&CheckpointBackendBudgetKind::Storage).unwrap(),
        "\"storage\"",
    );
    assert_eq!(
        serde_json::to_string(&CheckpointBackendBudgetFailureCode::ByteCapacity).unwrap(),
        "\"byte_capacity\"",
    );
}

#[test]
fn backend_constructor_rejects_invalid_limits_with_exact_kind_and_code() {
    let cases = support::invalid_backend_limits();
    assert_eq!(cases.len(), 20);
    for (limits, budget, code) in cases {
        assert!(matches!(
            MemoryCheckpointBackend::new(limits),
            Err(CheckpointError::BackendBudget {
                budget: actual_budget,
                code: actual_code,
            }) if actual_budget == budget && actual_code == code
        ));
    }
}

#[test]
fn backend_constructor_uses_existing_acquire_many_conversion_boundary() {
    let boundary = u32::MAX as usize;
    MemoryCheckpointBackend::new(support::backend_limits_with_each_capacity(boundary))
        .unwrap();

    let first_unrepresentable = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
    assert!(support::invalid_backend_limits().iter().any(|(limits, _, code)| {
        support::contains_capacity(*limits, first_unrepresentable)
            && *code == CheckpointBackendBudgetFailureCode::Unrepresentable
    }));
}

#[test]
fn result_projection_id_deserialization_rejects_empty_text() {
    assert!(ResultProjectionId::new("").is_err());
    assert!(serde_json::from_str::<ResultProjectionId>(r#""""#).is_err());
    assert_eq!(
        serde_json::from_str::<ResultProjectionId>(r#""tokens""#)
            .unwrap()
            .as_str(),
        "tokens",
    );
}

#[tokio::test(flavor = "current_thread")]
async fn result_partition_projection_allocation_remains_exactly_charged() {
    let (short_budget, short) = support::result_partition_with_projection("p").await;
    let (long_budget, long) =
        support::result_partition_with_projection("projection-with-retained-bytes").await;
    let projection_delta = "projection-with-retained-bytes".len() - "p".len();

    assert_eq!(
        long.descriptor_charged_bytes() - short.descriptor_charged_bytes(),
        projection_delta,
    );
    assert_eq!(
        short_budget.snapshot().used_bytes,
        short.descriptor_charged_bytes(),
    );
    assert_eq!(
        long_budget.snapshot().used_bytes,
        long.descriptor_charged_bytes(),
    );

    let (wrapped_descriptor, payload) = long.into_parts();
    assert_eq!(long_budget.snapshot().used_bytes, wrapped_descriptor.charged_bytes());
    drop(payload);
    drop(wrapped_descriptor);
    assert_eq!(long_budget.snapshot().used_bytes, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn cancelled_summary_wait_leaves_stage_inputs_and_transaction_retryable() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    let mut transaction = support::transaction_with_all_participants(&backend, run).await;
    let (input_budget, partition) = support::result_partition_with_projection("projection").await;
    let mut partitions = vec![partition];
    let transaction_before = transaction.staged_snapshot();
    let prepared_before = backend.budget_snapshots().prepared_indexes;
    let input_before = input_budget.snapshot();
    let held_summary = backend.hold_all_result_summary_capacity().await;

    let mut pending = Box::pin(transaction.stage_results(&mut partitions));
    assert!(matches!(futures::poll!(&mut pending), std::task::Poll::Pending));
    assert!(backend.budget_snapshots().prepared_indexes.used_items > 0);
    assert_eq!(input_budget.snapshot(), input_before);
    drop(pending);

    assert_eq!(transaction.staged_snapshot(), transaction_before);
    assert_eq!(partitions.len(), 1);
    assert_eq!(input_budget.snapshot(), input_before);
    assert_eq!(backend.budget_snapshots().prepared_indexes, prepared_before);

    drop(held_summary);
    let prepared = transaction.stage_results(&mut partitions).await.unwrap();
    assert!(partitions.is_empty());
    assert_eq!(input_budget.snapshot().used_items, 0);
    assert_eq!(transaction.staged_result_root(), Some(prepared.index_root()));
    assert_eq!(backend.budget_snapshots().result_summaries.used_items, 1);
    drop(prepared);
    assert_eq!(backend.budget_snapshots().result_summaries.used_items, 0);
    assert_eq!(backend.budget_snapshots().prepared_indexes.used_items, 1);
}

#[tokio::test(flavor = "current_thread")]
async fn stale_writer_cannot_merge_or_replace_head() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    let first = support::commit_empty(&backend, run, None, 1).await.unwrap();
    let stale = backend.begin_generation(run, None, support::expectations(run)).await.unwrap();
    let current = backend.begin_generation(run, Some(first.generation()), support::expectations(run)).await.unwrap();
    current.commit(support::metadata_at(2)).await.unwrap();
    let error = stale.commit(support::metadata_at(1)).await.unwrap_err();
    assert!(matches!(error, CheckpointError::GenerationConflict { .. }));
}

#[tokio::test(flavor = "current_thread")]
async fn dropped_transaction_publishes_nothing_and_releases_budget() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    let transaction = backend.begin_generation(run, None, support::expectations(run)).await.unwrap();
    assert_eq!(backend.prepared_transactions(), 1);
    drop(transaction);
    assert_eq!(backend.prepared_transactions(), 0);
    assert!(backend.open_latest(&run, &support::expectations(run)).await.unwrap().is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn empty_generations_and_heads_are_isolated_by_logical_run() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let first_run = support::run_id(1);
    let second_run = support::run_id(2);
    let first = support::commit_empty(&backend, first_run, None, 1).await.unwrap();
    let second = support::commit_empty(&backend, second_run, None, 1).await.unwrap();
    assert_ne!(first.generation().digest(), second.generation().digest());
    assert_eq!(
        backend.open_latest(&first_run, &support::expectations(first_run)).await.unwrap().unwrap().generation(),
        &first.generation(),
    );
    assert_eq!(
        backend.open_latest(&second_run, &support::expectations(second_run)).await.unwrap().unwrap().generation(),
        &second.generation(),
    );
}

#[tokio::test(flavor = "current_thread")]
async fn explicit_expectation_and_result_runs_must_match_transaction_run() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    let other = support::run_id(2);
    assert!(matches!(
        backend.begin_generation(run, None, support::expectations(other)).await,
        Err(CheckpointError::ObjectVerification)
    ));
    assert!(matches!(
        backend.open_latest(&run, &support::expectations(other)).await,
        Err(CheckpointError::ObjectVerification)
    ));

    let mut transaction = backend
        .begin_generation(run, None, support::expectations(run))
        .await
        .unwrap();
    let mut foreign_partitions = vec![support::result_partition(other, 1).await];
    assert!(matches!(
        transaction.stage_results(&mut foreign_partitions).await,
        Err(CheckpointError::ObjectVerification)
    ));
    assert_eq!(foreign_partitions.len(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn commit_requires_exact_participants_and_one_canonical_result_epoch() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);

    let mut omitted_results = support::transaction_with_all_participants(&backend, run).await;
    assert!(omitted_results.commit(support::metadata_at(1)).await.is_err());

    let mut omitted_participant = backend
        .begin_generation(run, None, support::expectations(run))
        .await
        .unwrap();
    let mut no_partitions = Vec::new();
    omitted_participant.stage_results(&mut no_partitions).await.unwrap();
    assert!(matches!(
        omitted_participant.commit(support::metadata_at(1)).await,
        Err(CheckpointError::ParticipantSetMismatch)
    ));

    let mut exact = support::transaction_with_all_participants(&backend, run).await;
    let mut no_partitions = Vec::new();
    let empty = exact.stage_results(&mut no_partitions).await.unwrap();
    assert_eq!(empty.item_count(), 0);
    assert_eq!(empty.byte_length(), 0);
    assert!(exact.stage_results(&mut no_partitions).await.is_err());
    exact.commit(support::metadata_at(1)).await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn result_epoch_must_match_commit_epoch() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    let mut transaction = support::transaction_with_all_participants(&backend, run).await;
    let mut wrong_epoch = vec![support::result_partition(run, 2).await];
    transaction
        .stage_results(&mut wrong_epoch)
        .await
        .unwrap();
    assert!(matches!(
        transaction.commit(support::metadata_at(1)).await,
        Err(CheckpointError::ObjectVerification)
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn existing_immutable_objects_do_not_grant_cross_generation_or_run_read_authority() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    let other = support::run_id(2);
    let superseded = support::commit_generation_with_segment(&backend, run, None, 1)
        .await
        .unwrap();
    let current = support::commit_generation_with_segment(
        &backend,
        run,
        Some(superseded.generation().clone()),
        2,
    )
    .await
    .unwrap();
    let foreign = support::commit_generation_with_segment(&backend, other, None, 1)
        .await
        .unwrap();
    let superseded_descriptor = superseded.only_result_descriptor().clone();
    let foreign_descriptor = foreign.only_result_descriptor().clone();

    assert!(backend
        .immutable_object_inventory(&run)
        .contains(&superseded_descriptor.payload_digest));
    assert!(backend
        .immutable_object_inventory(&other)
        .contains(&foreign_descriptor.payload_digest));

    let reader = backend
        .open_latest(&run, &support::expectations(run))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(reader.generation(), current.generation());
    let reads_before = backend.budget_snapshots().reads;

    assert_eq!(
        reader.read_segment(&superseded_descriptor).await.unwrap_err(),
        CheckpointError::ObjectVerification,
    );
    assert_eq!(backend.budget_snapshots().reads, reads_before);
    assert_eq!(
        reader.read_segment(&foreign_descriptor).await.unwrap_err(),
        CheckpointError::ObjectVerification,
    );
    assert_eq!(backend.budget_snapshots().reads, reads_before);
}

#[tokio::test(flavor = "current_thread")]
async fn storage_capacity_refusal_is_typed_and_publishes_nothing() {
    let limits = support::backend_limits_with_storage_bytes(
        support::one_segment_commit_storage_bytes() - 1,
    );
    let backend = MemoryCheckpointBackend::new(limits).unwrap();
    let run = support::run_id(1);
    let before = backend.budget_snapshots();
    let objects_before = backend.immutable_object_inventory(&run);
    let object_count_before = objects_before.total_count();
    let transaction = support::transaction_with_one_segment(&backend, run).await;

    assert!(matches!(
        transaction.commit(support::metadata_at(1)).await,
        Err(CheckpointError::BackendBudget {
            budget: CheckpointBackendBudgetKind::Storage,
            code: CheckpointBackendBudgetFailureCode::ByteCapacity,
        })
    ));
    assert!(backend
        .open_latest(&run, &support::expectations(run))
        .await
        .unwrap()
        .is_none());
    assert_eq!(backend.immutable_object_inventory(&run), objects_before);
    assert_eq!(
        backend.immutable_object_inventory(&run).total_count(),
        object_count_before,
    );
    assert_eq!(backend.budget_snapshots().storage, before.storage);
    assert_eq!(backend.budget_snapshots().transactions.used_items, 0);
    assert_eq!(backend.budget_snapshots().prepared_indexes.used_items, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn oversized_next_descriptor_refuses_without_empty_cursor_loop() {
    let backend = support::backend_with_one_committed_segment().await;
    let reader = support::latest_reader(&backend).await;
    let required = support::first_descriptor_retained_bytes(&reader);
    let before = backend.budget_snapshots().reads;

    assert!(matches!(
        reader
            .scan_result_index(None, support::index_budget(1, required - 1))
            .await,
        Err(CheckpointError::ResultIndexReadBudgetTooSmall {
            required_bytes,
            max_bytes,
        }) if required_bytes == required && max_bytes == required - 1
    ));
    assert_eq!(backend.budget_snapshots().reads, before);

    let page = reader
        .scan_result_index(None, support::index_budget(1, required))
        .await
        .unwrap();
    assert_eq!(page.descriptors().len(), 1);
    assert!(page.next().is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn projection_allocation_participates_in_result_index_read_charge() {
    let short = support::backend_with_projection("p").await;
    let long = support::backend_with_projection("projection-with-retained-bytes").await;
    let short_page = support::read_first_index_page(&short).await;
    let long_page = support::read_first_index_page(&long).await;

    assert_eq!(
        long_page.charged_bytes() - short_page.charged_bytes(),
        u64::try_from("projection-with-retained-bytes".len() - "p".len()).unwrap(),
    );
    assert_eq!(
        short.budget_snapshots().reads.used_bytes,
        usize::try_from(short_page.charged_bytes()).unwrap(),
    );
    assert_eq!(
        long.budget_snapshots().reads.used_bytes,
        usize::try_from(long_page.charged_bytes()).unwrap(),
    );
}

#[tokio::test(flavor = "current_thread")]
async fn invalid_cursor_refuses_before_page_or_backend_budget() {
    let backend = support::backend_with_one_committed_segment().await;
    let reader = support::latest_reader(&backend).await;
    let before = backend.budget_snapshots().reads;

    for cursor in support::foreign_unreachable_and_out_of_range_cursors(&reader) {
        assert!(matches!(
            reader
                .scan_result_index(
                    Some(cursor),
                    support::index_budget(1, 1),
                )
                .await,
            Err(CheckpointError::ObjectVerification)
        ));
        assert_eq!(backend.budget_snapshots().reads, before);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn sufficient_page_limit_does_not_hide_backend_read_capacity_refusal() {
    let fixture = support::one_segment_fixture();
    let required = fixture.descriptor_retained_bytes();
    let backend = fixture
        .commit_with_read_byte_limit(required - 1)
        .await;
    let reader = support::latest_reader(&backend).await;
    let before = backend.budget_snapshots().reads;

    assert!(matches!(
        reader
            .scan_result_index(None, support::index_budget(1, required))
            .await,
        Err(CheckpointError::BackendBudget {
            budget: CheckpointBackendBudgetKind::Read,
            code: CheckpointBackendBudgetFailureCode::ByteCapacity,
        })
    ));
    assert_eq!(backend.budget_snapshots().reads, before);
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_checkpoint_backend
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --doc
```

- [ ] **Step 3: Implement the minimal reference backend**

```rust
struct MemoryRunHead {
    generation: Option<CommittedCheckpointGeneration>,
    objects: BTreeMap<ContentDigest, BudgetedStoredObject>,
}

struct MemoryState {
    heads: BTreeMap<StreamRunIdentity, MemoryRunHead>,
}

struct StorageCommitBundle {
    _storage_lease: BudgetLease,
}

struct BudgetedStoredObject {
    bytes: Bytes,
    _storage_bundle: Rc<StorageCommitBundle>,
}

async fn prepare_result_partitions(
    &mut self,
    partitions: &mut Vec<ResultPartition>,
) -> Result<PreparedResultEpoch, CheckpointError> {
    self.validate_result_partitions(partitions.as_slice())?;
    let plan = CheckedResultStagePlan::from_partitions(partitions)?;
    let prepared_lease = self
        .prepared_index_budget
        .acquire(plan.descriptor_items, plan.descriptor_bytes)
        .await
        .map_err(map_prepared_index_budget_error)?;
    let summary_lease = self
        .result_summary_budget
        .acquire(plan.descriptor_items, plan.descriptor_bytes)
        .await
        .map_err(map_result_summary_budget_error)?;
    self.install_result_partitions(partitions, plan, prepared_lease, summary_lease)
}

fn install_result_partitions(
    &mut self,
    partitions: &mut Vec<ResultPartition>,
    plan: CheckedResultStagePlan,
    prepared_lease: BudgetLease,
    summary_lease: BudgetLease,
) -> Result<PreparedResultEpoch, CheckpointError> {
    let prepared_descriptors = partitions
        .iter()
        .map(|partition| partition.descriptor().clone())
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let summary_descriptors = prepared_descriptors.to_vec().into_boxed_slice();
    let prepared_descriptors = BudgetedResultDescriptors::new(
        prepared_descriptors,
        prepared_lease,
    )?;
    let summary_descriptors = BudgetedResultDescriptors::new(
        summary_descriptors,
        summary_lease,
    )?;
    let prepared_summary = PreparedResultEpoch::new(
        plan.index_root,
        summary_descriptors,
        plan.item_count,
        plan.byte_length,
    )?;

    let mut payloads = Vec::with_capacity(plan.descriptor_items);
    let inputs = std::mem::take(partitions);
    for partition in inputs {
        let (budgeted_descriptor, payload) = partition.into_parts();
        let (input_descriptor, input_lease) = budgeted_descriptor.into_backend_parts();
        payloads.push(payload);
        drop(input_descriptor);
        drop(input_lease);
    }
    self.staged_results = Some(StagedResultEpoch {
        index_root: plan.index_root,
        descriptors: prepared_descriptors,
        payloads,
        item_count: plan.item_count,
        byte_length: plan.byte_length,
    });
    Ok(prepared_summary)
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

Keep storage behind `Rc<RefCell<MemoryState>>`; it is test/reference state on one
local runtime, not a shared hot-path lock. Resolve the `MemoryRunHead` by the
transaction's exact `StreamRunIdentity` before comparing or publishing the head.
Never compare a writer against a global or different-run generation. The
transaction owns its transaction and prepared-index permits and releases both
through RAII on abort, failed commit, and successful commit; neither lease is
transferred into immutable objects. A successful commit acquires the separate
aggregate storage lease described below before publication. Each reader method
acquires a separate read-budget lease before cheaply cloning underlying
`Bytes`; the returned wrapper owns that full logical-byte charge, so storage and
concurrent readers remain independently bounded.

Keep transaction, prepared-index, immutable-storage, returned-summary, and read
budgets distinct. A returned `PreparedResultEpoch` owns a summary lease separate
from the transaction's prepared-index lease, so dropping either owner cannot
release the other's charge. Precompute the complete set of missing immutable
objects and acquire one aggregate storage reservation before publication; do
not sequentially await per-object reservations while retaining earlier ones.
The memory reference may attach that aggregate charge through one shared
commit-bundle owner to every newly inserted object. It safely over-retains until
the last object from that bundle is reclaimed and cannot undercharge storage.
Concretely, create one private `Rc<StorageCommitBundle>` after the aggregate
acquisition and clone only that handle into each `BudgetedStoredObject` inserted
by the commit. Never clone, expose, or attempt to split `BudgetLease` itself.

Validate cursor root, block reachability, and offset before inspecting page or
backend capacity. Validate the one-descriptor page limit next, and acquire the
backend read lease last. This fixes error precedence and proves every refusal is
side-effect-free.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: atomic participant+result publication, same-run stale-writer
refusal, cross-run independence, empty-generation run binding, transaction/result
run mismatch refusal, cancellation-safe retryable result staging, immutable read
authority verification, DTO privacy doctests, and RAII abort pass.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/checkpoint.rs rust/runtime/src/streaming/checkpoint_backend.rs rust/runtime/src/streaming/checkpoints.rs rust/runtime/src/streaming/checkpoints/memory.rs rust/runtime/src/streaming/results.rs rust/runtime/tests/support/streaming_checkpoint.rs rust/runtime/tests/streaming_checkpoint_backend.rs
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
    fn create_private_dir(&self, path: &Path) -> Result<(), CheckpointError>;
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
        let run = support::run_id(1);
        let first = support::commit_empty(&backend, run, None, 1).await.unwrap();
        backend.inject_fault(fault);
        let transaction = backend.begin_generation(run, Some(first.generation()), support::expectations(run)).await.unwrap();
        assert!(transaction.commit(support::metadata_at(2)).await.is_err());
        let reopened = support::local_backend(directory.path(), None);
        let latest = reopened.open_latest(&run, &support::expectations(run)).await.unwrap().unwrap();
    assert_eq!(latest.generation().generation(), first.generation());
    }
}

#[tokio::test(flavor = "current_thread")]
async fn checkpoint_tree_is_private_no_follow_and_tmp_is_raii_cleaned() {
    let fixture = support::local_filesystem_fixture();
    let transaction = fixture.backend.begin_generation(
        fixture.run,
        None,
        support::expectations(fixture.run),
    ).await.unwrap();
    assert_eq!(fixture.mode(fixture.run_root()), 0o700);
    assert!(fixture.all_regular_file_modes_are(0o600));
    assert!(fixture.symlink_swap_current().is_err());
    let tmp = transaction.tmp_path().to_path_buf();
    drop(transaction);
    assert!(!tmp.exists());
}

#[tokio::test(flavor = "current_thread")]
async fn process_crash_tmp_is_reclaimed_by_bounded_lease_aware_startup_scan() {
    let fixture = support::crashed_local_transaction_fixture();
    fixture.clock.advance_past_prepare_lease();
    let reopened = fixture.reopen_with_gc_page_limit(2).await.unwrap();
    assert!(!fixture.orphan_tmp_path().exists());
    assert!(reopened.gc_high_water().page_items <= 2);
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

Create every run/tmp/lease directory as exact `0700` through no-follow directory descriptors and every regular file as create-new `0600`; reject symlinks and ownership/type/mode drift. A transaction-owned guard removes only its validated private tmp subtree on drop, including cancellation/fault paths; committed immutable objects remain durable. On startup and Task-5D GC, traverse tmp transaction directories in bounded cursor pages and reclaim only those whose prepare/writer lease is absent or expired according to injected `Clock`; never use mtime or delete a live transaction. The blocking job performs: validate writer lease and expected CURRENT; write immutable participant/result/index objects with create-new; fsync each new object and parent; write and fsync `generation-N`; fsync its parent; write/fsync temporary CURRENT; rename over CURRENT; fsync root. Decode and validate the current generation after reopen before returning it.

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
    let reader = fixture.backend.open_latest(&fixture.run, &fixture.expectations).await.unwrap().unwrap();
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
    let reader = fixture.backend.open_latest(&fixture.run, &fixture.expectations).await.unwrap().unwrap();
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

The coordinator constructor requires one already resolved
`StreamRunIdentity`. Product Task V1 is the sole fresh/resume resolver and owns
the no-issue-before-bootstrap lifecycle gate. Task 5E only verifies the injected
run against every barrier, participant state, backend expectation, committed
generation, and receipt; it has no allocator, catalog, or implicit "latest run"
path.

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
    let latest = fixture.backend.open_latest(&fixture.run, &fixture.expectations).await.unwrap().unwrap();
    assert_eq!(latest.generation().generation(), coordinator_support::generation(1));
    fixture.restore_and_replay_notifications().await.unwrap();
    assert_eq!(fixture.participant("session").commit_notifications(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn greater_epoch_receipt_from_another_run_never_reaches_participant() {
    let mut fixture = coordinator_support::coordinator_fixture_for_run(1);
    let foreign = fixture
        .commit_identical_participant_for_run_and_epoch(2, 99)
        .await
        .unwrap();
    let error = fixture
        .coordinator
        .replay_committed_notifications(&foreign)
        .await
        .unwrap_err();
    assert!(matches!(error, CheckpointError::ObjectVerification));
    assert_eq!(fixture.participant("session").commit_notifications(), 0);
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
let mut transaction = self.backend.begin_generation(
    self.run,
    self.expected,
    self.generation_expectations.clone(),
).await?;
for view in views {
    transaction.stage_participant(view).await?;
}
transaction.stage_results(&mut result_partitions).await?;
let committed = transaction.commit(self.metadata(&barrier)?).await?;
self.notify_committed(&committed).await?;
Ok(committed)
```

Missing/duplicate participants fail before `begin_generation`. The coordinator
requires `barrier.run == self.run == generation_expectations.run` before
collecting views, and it checks `committed.run()` and every `receipt.run()` again
before dispatching participant callbacks. Participants independently reject a
receipt whose run differs from their initialized/frozen run before considering
generation ordering or descriptor-digest idempotency. Thus a greater-epoch
receipt from another run cannot be mistaken for progress. Failed staging/CAS
drops the transaction and sends no notifications. Notification failure is
surfaced after publication and is replayed from committed receipts during
restore.

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

**Depends on:** Tasks 5B, 5E, 5F1, and adapter Task A0 (neutral AWS client construction).

**Files:**
- Create: `rust/runtime/src/streaming/checkpoints/object_store.rs`
- Create: `rust/runtime/src/streaming/checkpoints/aws_object_store.rs`
- Modify: `rust/runtime/src/streaming/checkpoints.rs`
- Modify: `rust/runtime/src/streaming/checkpoint_factories.rs`
- Modify: `rust/runtime/src/extensions/mod.rs`
- Consume: `rust/runtime/src/streaming/aws.rs` from Task A0; checkpoint code never imports a source module.
- Test: `rust/runtime/tests/streaming_object_checkpoint.rs`

**Produces the conditional capability and a bounded object I/O contract:**

```rust
pub const OBJECT_STORE_CHECKPOINT_BACKEND_ID: &str = "object_store";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObjectKey(String);
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObjectVersion(String);
pub struct PointerObject { pub bytes: Bytes, pub digest: ContentDigest, pub lease: BudgetLease }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectReadRange { pub offset: u64, pub length: u64 }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectReadBudget { pub max_chunk_bytes: usize }
pub struct ObjectListBudget { pub max_items: NonZeroUsize, pub max_metadata_bytes: NonZeroUsize }
pub struct ObjectListCursor(String);
pub struct ObjectMetadata { pub key: ObjectKey, pub version: ObjectVersion, pub byte_length: u64 }
pub struct BudgetOwnedObjectPage {
    pub objects: Box<[ObjectMetadata]>,
    pub next: Option<ObjectListCursor>,
    pub lease: BudgetLease,
}
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
    async fn list_versions(
        &self,
        prefix: &ObjectKey,
        cursor: Option<&ObjectListCursor>,
        budget: ObjectListBudget,
    ) -> Result<BudgetOwnedObjectPage, CheckpointError>;
    async fn delete_version(
        &self,
        key: &ObjectKey,
        version: &ObjectVersion,
    ) -> Result<(), CheckpointError>;
}
```

`BudgetOwnedObjectReader` yields bounded chunks while retaining byte permits. Chunks and list pages own permits until drop. Read ranges and list budgets are checked before provider I/O, and declared object/page/chunk lengths exceeding limits are rejected before allocation. `list_versions`/`delete_version` are checkpoint-prefix retention authority only; the trait has no source discovery/reconciliation operation and is not implemented in terms of the S3 source trait.

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
git add rust/runtime/src/streaming/checkpoints.rs rust/runtime/src/streaming/checkpoints/object_store.rs rust/runtime/src/streaming/checkpoints/aws_object_store.rs rust/runtime/src/streaming/checkpoint_factories.rs rust/runtime/src/extensions/mod.rs rust/runtime/tests/streaming_object_checkpoint.rs
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
    PartitionDescriptorCapacityExceeded { items: u64, bytes: u64 },
    ProvisionalCapacityExceeded { items: u64, bytes: u64 },
    InvalidCoverage,
    SegmentVerification,
    Compaction { message: String },
}
```

Implement `Display` and `std::error::Error` directly; errors expose stable lowercase reason codes through a separate `code()` method.

Task 6A's `EpochResultCoordinator` also owns a producer-side
`StreamingResourceBudget` for singular partition descriptors; it is distinct
from every private backend budget. Before returning a `ResultPartition`, it
acquires one item and the exact checked inline-descriptor-plus-projection bytes,
constructs `BudgetedResultDescriptor`, and installs that wrapper with the
separately budgeted payload. Producer refusal maps to
`PartitionDescriptorCapacityExceeded { items, bytes }`; it must not be mislabeled
as backend, participant-state, storage, or provisional-hole capacity. The
wrapper remains charged while Task 5E carries partitions into Task 5B staging
and is released only after Task 5B has acquired both aggregate descriptor
authorities.

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
    pub terminal_horizon: TerminalActionHorizon,
    pub authoritative_request_count: u64,
    pub provisional_request_count: u64,
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

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for EpochResultCoordinator {
    fn participant_id(&self) -> CheckpointParticipantId { self.participant_id.clone() }
    async fn checkpoint_view(&mut self, barrier: &CheckpointBarrier)
        -> Result<PreparedParticipantState, CheckpointError> {
        self.prepare_result_state_view(barrier).await
    }
    async fn initialize(&mut self, state: Option<CommittedParticipantState>)
        -> Result<(), CheckpointError> { self.restore_result_state(state).await }
    async fn checkpoint_committed(&mut self, receipt: &CommittedParticipantReceipt)
        -> Result<(), CheckpointError> { self.advance_committed_result_cut(receipt) }
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
    assert_eq!(partial.terminal_horizon, TerminalActionHorizon::new(GlobalSequence::new(2)));
    assert_eq!(partial.authoritative_request_count, 2);
    assert_eq!(partial.provisional_request_count, 0);
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,parquet --test streaming_result_epochs
```

- [ ] **Step 3: Implement bounded epoch ownership**

Rotate each worker `MetricsAccumulator` and configured exact/raw/session/provenance projections at the barrier. The stable result participant checkpoints accumulator epochs, index root, terminal horizon, and all bounded provisional descriptors/leases. Hold completions above `H` in immutable provisional partitions charged to prepare/provisional budgets; never link them from a committed root until the hole closes. On exhaustion, fence new admission and return the authored overload decision. Partial views page and merge only committed segments through `H`; provisional dashboard data is separately labeled and excluded from totals.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: long-hole capacity, backpressure, hole closure, exact/sketch rotation, provenance paging, and partial-authority tests pass.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/results.rs rust/runtime/src/streaming/results/epoch.rs rust/runtime/src/metrics.rs rust/runtime/src/metrics_core/report.rs rust/runtime/tests/support/streaming_checkpoint.rs rust/runtime/tests/streaming_result_epochs.rs
git commit -m "feat(runtime): rotate checkpoint result epochs"
```

### Task 6C1: Deterministic Final/Aborted Generation and Compaction

**Depends on:** Task 6B and the existing `PreparedRunOutcome`/`PreparedReportCommit` interfaces. This task must merge before source or cellular E2E claims restart correctness.

**Files:**
- Create: `rust/runtime/src/streaming/results/compactor.rs`
- Modify: `rust/runtime/src/streaming/results.rs`
- Extend: `rust/runtime/tests/support/streaming_checkpoint.rs`
- Create: `rust/runtime/tests/streaming_result_finalization.rs`

- [ ] **Step 1: Write RED finalization tests**

Add `report_lease_releases_only_after_authoritative_report_commit`, `unsafe_abort_preserves_last_partial_without_fabricating_terminal_root`, `safe_abort_commits_complete_aborted_generation`, `compaction_order_is_stable_across_page_sizes`, and `compaction_failure_retains_reconstructable_generation`.

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,parquet --test streaming_result_finalization
```

- [ ] **Step 3: Implement bounded deterministic finalization**

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

Seal and commit final/aborted generation first. Traverse bounded index pages in fixed `(epoch, cell_id, worker_id, projection_id, first_global_sequence, digest)` order, stream output through the blocking owner, and retain the report lease through durable persistence. Unsafe abort retains the last partial root without fabricating a terminal generation. This task does not implement delivery restart or endpoint idempotency policy.

- [ ] **Step 4: Verify GREEN and commit**

Run Step 2, then commit:

```bash
git add rust/runtime/src/streaming/results.rs rust/runtime/src/streaming/results/compactor.rs rust/runtime/tests/support/streaming_checkpoint.rs rust/runtime/tests/streaming_result_finalization.rs
git commit -m "feat(runtime): finalize checkpointed streaming results"
```

### Task 6C2: Delivery Restart and Target-Idempotency Policy

**Depends on:** Task 6C1.

**Files:**
- Create: `rust/runtime/src/streaming/results/delivery.rs`
- Modify: `rust/runtime/src/streaming/results.rs`
- Extend: `rust/runtime/tests/support/streaming_checkpoint.rs`
- Create: `rust/runtime/tests/streaming_delivery_restart.rs`

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

- [ ] **Step 1: Write the RED delivery crash matrix**

```rust
#[test]
fn delivery_mode_crash_matrix_has_stable_logical_membership() {
    for mode in CheckpointDeliveryMode::ALL {
        for crash in DeliveryCrashPoint::ALL {
            for capability in TargetIdempotencyCapability::ALL {
                let restored = support::delivery_fixture(mode, capability).crash_and_restore(crash);
                assert!(restored.logical_membership_is_unique());
                assert_eq!(restored.claim(), expected_claim(mode, capability));
                assert_eq!(restored.duplicate_window(), expected_window(mode, crash, capability));
            }
        }
    }
}
```

`DeliveryCrashPoint::ALL` covers before dispatch; after decode; acquisition; admission; target acceptance; terminal fact; segment/index write; before/after CAS; post-CAS notification; compaction; and after report write before `PreparedReportCommit`. Assert next action set, logical metric membership, attempt IDs, duplicate/loss window, and delivery claim.

Add `restart_rejects_changed_topology_projection_or_membership_scheme`; change worker count, cell topology, placement digest, projection plan, and membership scheme independently and assert restore fails before participant initialization or source polling.

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,parquet --test streaming_delivery_restart
```

- [ ] **Step 3: Implement restart decisions only**

Derive decisions solely from the selected cut, committed participant/result state, and endpoint capability. Idempotency keys derive from `(LogicalReplayRunId, StableActionId)`; attempt IDs remain telemetry. Reject topology/projection/membership changes before participant initialization or source polling. This task does not modify compaction or final/aborted generation code.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: every crash/mode/capability row and compatibility refusal passes.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/results.rs rust/runtime/src/streaming/results/delivery.rs rust/runtime/tests/support/streaming_checkpoint.rs rust/runtime/tests/streaming_delivery_restart.rs
git commit -m "feat(runtime): enforce streaming delivery restart policy"
```

### Task 6D: Coordinator Report-Persistence and Lease Ordering

**Depends on:** Tasks 6C1 and 6C2.

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
