<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Streaming Pipeline and Sessions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compose the native streaming contracts into a bounded shadow-replay workload that preserves conversation and agent/graph sessions across arbitrary partition boundaries.

**Architecture:** One run-scoped coordinator owns session/causal state; a fused current-thread pipeline pulls only under downstream permits; one action host multiplexes binding drivers into existing transport/dispatch/observation seams. Recorded-input replay is the default, while target-closed-loop state is a separately authorized encrypted policy.

**Tech Stack:** Rust 2024, Tokio current-thread `LocalSet`, AIPerf engine/clock/dispatch/phase/capture seams, XChaCha20-Poly1305, zeroize.

**Spec:** `artifacts/streaming-design/streaming-dataset-shadow-replay-design.md` at base approval `505efc06b0`, amended by `3fea6f2fe0` and `artifacts/streaming-design/reliability-continuation-course-correction.md`.

## Global Constraints

- Requires foundation/runtime through Task 1D-R, checkpoint/results, and adapter Tasks A1-A2 before the first executable workload.
- No `NativeDatasetPlan::Streaming`, complete `GraphTraceProgram`, per-action boxed driver, unbounded queue/task/vector, or Python path.
- One profiling phase in generation 1; warmup/live-profile combinations fail during capability agreement.
- Every accepted action reaches exactly one terminal receipt; checkpoint cuts remain typed by stage.
- Ordinary session and endpoint faults become scoped issue receipts and cannot bubble out as workload-fatal errors. `FailRun` remains restricted to checked authority, conflicting-content, frozen-semantic, truthful-order/cut, and accounting invariants.
- For P1B/P2/P3/P4/P7, each task's RED step includes its row in “Reliability-Continuation Amendment” below and its production step owns the GREEN behavior; the amendment is not a post-task follow-up.
- Cargo commands run from the nested `rust/` workspace; git commands run from the repository root. All builds use the shared `/mnt/4tb` target.
- Each task includes the nearest parent module declaration required for its own GREEN build; declaration conflicts are resolved during integration.

---

### Task P1: Cross-Chunk Conversation Coordinator

**Files:**
- Create: `rust/runtime/src/streaming/session/conversation.rs`
- Modify: `rust/runtime/src/streaming/session.rs`
- Test: `rust/runtime/tests/streaming_session_continuity.rs`

**Interfaces:**
- Consumes: `StreamingSessionProgramFactory`, `StreamingSessionCoordinator`, `StreamingCheckpointParticipant`, `StreamingSessionFragment`.
- Produces and registers session-program ID `conversation`; `StreamingConversationCoordinator`; cross-partition continuity, duplicate/conflict handling, and producer-authored explicit close. It consumes neutral `SessionCausalFrontier` from foundation Task 1A. Task P1B exclusively owns inferred closure, missing-predecessor, completeness, spill, and bounded-state policies.

```rust
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionStateVersion(u64);
```

- [ ] **Step 1: Write the RED continuity test**

```rust
#[tokio::test(flavor = "current_thread")]
async fn one_conversation_spans_partitions_and_checkpoint() {
    let mut sessions = conversation_coordinator(session_budget(2, 4096));
    let mut output = CollectingActionSink::default();
    sessions.ingest(fragment("s", 0, "hello"), &mut output).await.unwrap();
    // Partition EOF is a decoder event and deliberately does not call session seal.
    let saved = sessions.checkpoint_view(&barrier()).await.unwrap();
    let mut restored = restore_conversations(saved).await;
    restored.ingest(fragment("s", 1, "again"), &mut output).await.unwrap();
    assert_eq!(output.actions().last().unwrap().messages(), ["hello", "again"]);
    assert_eq!(restored.active_session_count(), 1);
}
```

- [ ] **Step 2: Verify red**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_session_continuity`

- [ ] **Step 3: Implement the coordinator**

```rust
#[async_trait(?Send)]
impl StreamingSessionCoordinator for StreamingConversationCoordinator {
    async fn ingest(&mut self, fragment: StreamingSessionFragment, output: &mut dyn DatasetActionSink)
        -> Result<(), SessionCoordinatorError> { self.ingest_mutation(fragment, output).await }
    async fn advance_watermark(&mut self, watermark: SessionWatermark, output: &mut dyn DatasetActionSink)
        -> Result<(), SessionCoordinatorError> { self.apply_watermark(watermark, output).await }
    async fn observe_execution(&mut self, event: ActionExecutionEvent, output: &mut dyn DatasetActionSink)
        -> Result<(), SessionCoordinatorError> { self.apply_execution(event, output).await }
    async fn seal(&mut self, seal: SourceSeal, output: &mut dyn DatasetActionSink)
        -> Result<SessionSealReceipt, SessionCoordinatorError> { self.seal_explicit(seal, output).await }
}
```

Key state by `(stream_identity, StableSessionKey)`. Partition EOF never closes a session. Identical producer mutations are idempotent; conflicting content fails. Producer-authored explicit close becomes terminal only after declared actions. Checkpoint complete in-memory state or roll the decoded horizon back before the first unrepresented mutation; P1B adds spill and all inferred closure policies.

- [ ] **Step 4: Verify green**

Run Step 2. Expected: PASS for four-partition continuity, checkpoint restore, duplicate/conflict, and explicit close.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/session.rs rust/runtime/src/streaming/session/conversation.rs rust/runtime/tests/streaming_session_continuity.rs
git commit -m "feat(runtime): preserve conversations across stream chunks"
```

### Task P1B: Session Closure and Bounded Causality Policies

**Depends on:** Task P1 and foundation Task 1D-R.

**Files:**
- Modify: `rust/runtime/src/streaming/session/conversation.rs`
- Create: `rust/runtime/src/streaming/session/spill.rs`
- Modify: `rust/runtime/src/streaming/session.rs`
- Test: `rust/runtime/tests/streaming_session_closure.rs`

**Produces:** `SessionClosurePolicy`, `MissingPredecessorPolicy`, externally
sorted finite completeness receipts, typed `WholeProducerTreeClosureReceipt`,
and explicit refusal when neither a finite bound nor spill/drop/fail policy
exists. The tree receipt binds one root plus the exact closed descendant
inventory and is minted only after the coordinator proves the entire rooted
producer tree complete. P1B also owns the durable budgeted
`SessionQuarantineTombstone` map keyed by exact `(input_domain, session)`.
It consumes the landed `SessionCausalFrontier` and defines the checked
`SessionQuarantineClosureProof` whose only variants are authored close, hard
watermark, verified finite seal, verified complete sorted run, and exhausted
authored missing-predecessor policy. Partition EOF is not a closure proof.

```rust
pub struct SessionQuarantineTombstone {
    run: StreamRunIdentity,
    input_domain: StreamingInputDomainIdentity,
    session_key: StableSessionKey,
    issue_id: ContentDigest,
    causal_frontier: SessionCausalFrontier,
    closure_proof: SessionQuarantineClosureProof,
    encoded: BudgetedCheckpointBytes,
    parsed_lease: BudgetLease,
}
```

Only P1B's checked installer constructs this private-field, non-`Clone` type.
Borrow-only accessors expose identity/frontier/proof; checkpoint/result transfer
never moves the wrapper. P1B's retained map exposes
`SessionQuarantineTombstoneMap::checked_view()` and implements 1D-R's
crate-private sealed `SessionQuarantineTombstoneView` for that borrowed view
only. The checked view captures the map's run/root/revision and borrowed
canonical entries; it cannot outlive or move the map. 1D-R uses the view to
prepare a separately charged move-only install acknowledgement. External
callers cannot implement or forge it.

- [ ] **Step 1: Write the RED policy matrix**

```rust
#[tokio::test(flavor = "current_thread")]
async fn closure_requires_authored_proof_not_partition_eof() {
    let cases = [
        closure_case("inactivity", Evidence::SoftWatermarkBelowDeadline, Outcome::Close),
        closure_case("hard_watermark", Evidence::HardWatermarkPastSession, Outcome::Close),
        closure_case("sealed_incomplete", Evidence::FiniteSealWithGap, Outcome::Fail),
        closure_case("external_sort", Evidence::CompleteSortedRun, Outcome::Close),
        closure_case("follow_gap", Evidence::PartitionEofWithMissingPredecessor, Outcome::Wait),
    ];
    for case in cases {
        assert_eq!(run_closure_case(case).await, case.expected);
    }
}

#[test]
fn whole_tree_receipt_requires_every_descendant_not_individual_session_close() {
    let mut tree = producer_tree_with_open_descendant();
    tree.close_root();
    assert!(tree.whole_tree_receipt().is_none());
    tree.observe_partition_eof();
    assert!(tree.whole_tree_receipt().is_none());
    tree.close_last_descendant_with_complete_inventory();
    assert!(tree.whole_tree_receipt().is_some());
}

#[test]
fn unbounded_session_without_spill_drop_or_fail_is_refused() {
    assert_eq!(validate_session_limits(unbounded_without_policy()).unwrap_err().code(),
        SessionFailureCode::UnboundedCausalityState);
}

#[test]
fn spill_tree_is_private_no_follow_and_cleanup_is_raii() {
    let fixture = private_spill_fixture();
    let spill = fixture.open().unwrap();
    assert_eq!(fixture.root_mode(), 0o700);
    assert!(fixture.all_file_modes_are(0o600));
    assert!(fixture.replace_run_with_symlink().is_err());
    drop(spill);
    assert!(!fixture.run_path().exists());
}


#[test]
fn crashed_spill_run_is_reclaimed_only_after_owner_lease_expiry() {
    let fixture = crashed_spill_fixture_with_manual_clock();
    assert!(!fixture.reclaim().unwrap().removed_live_owner());
    fixture.clock.advance_past_owner_lease();
    assert!(fixture.reclaim_bounded(2).unwrap().removed_orphan());
    assert!(fixture.max_scan_page_items() <= 2);
}

#[tokio::test(flavor = "current_thread")]
async fn quarantined_session_cannot_resurrect_and_tombstone_survives_resume() {
    let committed = quarantine_at_frontier_then_checkpoint(session("s"), frontier(4)).await;
    let mut restored = restore_sessions(committed).await;
    restored.observe(fragment("s", 5)).await.unwrap();
    assert!(!restored.has_live_session("s"));
    assert_eq!(restored.tombstone("s").causal_frontier(), frontier(5));
}

#[tokio::test(flavor = "current_thread")]
async fn quarantine_requires_tombstone_root_ack_in_same_generation() {
    let mut fixture = quarantined_session_fixture_across_chunks();
    fixture.observe(fragment("s", 4)).await.unwrap();
    fixture.quarantine_with_checked_closure("s").await.unwrap();
    assert!(fixture.reporter().handled_cut().is_err());
    let generation = fixture.checkpoint_tombstone_and_receipt().await.unwrap();
    fixture.acknowledge_generation_root(&generation).await.unwrap();
    assert!(fixture.reporter().handled_cut().unwrap().covers("s"));
    let mut restored = fixture.restart_from(generation).await.unwrap();
    restored.observe(fragment("s", 5)).await.unwrap();
    assert!(!restored.has_live_session("s"));
}

#[tokio::test(flavor = "current_thread")]
async fn late_fragment_invalidates_prepared_root_and_requires_fresh_ack() {
    let mut fixture = quarantined_session_fixture_across_chunks();
    fixture.quarantine_with_checked_closure("s").await.unwrap();
    let stale = fixture.prepare_tombstone_install(barrier(1)).await.unwrap();
    assert_eq!(stale.payload_charge_bytes(), fixture.expected_ack_payload_bytes());
    assert_eq!(stale.view_charge_bytes(), fixture.expected_ack_view_bytes());
    fixture.observe(fragment("s", 5)).await.unwrap();
    assert!(fixture.stage_tombstone_install(stale, barrier(1)).await.is_err());
    let fresh = fixture.prepare_tombstone_install(barrier(2)).await.unwrap();
    assert_eq!(fresh.payload_charge_bytes(), fixture.expected_reencoded_ack_payload_bytes());
    assert_eq!(fresh.view_charge_bytes(), fixture.expected_ack_view_bytes());
    fixture.stage_tombstone_install(fresh, barrier(2)).await.unwrap();
    assert_eq!(fixture.tombstone("s").causal_frontier(), frontier(5));
    assert_ne!(fixture.stale_view_revision(), fixture.current_view_revision());
}

#[tokio::test(flavor = "current_thread")]
async fn pre_cas_drop_preserves_owned_tombstone_for_identical_retry() {
    let fixture = quarantined_session_fixture();
    let first = fixture.prepare_tombstone_install(barrier(1)).await.unwrap();
    let root = first.tombstone_root();
    drop(first);
    assert!(fixture.contains_tombstone("s"));
    let retry = fixture.prepare_tombstone_install(barrier(1)).await.unwrap();
    assert_eq!(retry.tombstone_root(), root);
}
```

- [ ] **Step 2: Verify RED**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_session_closure`

- [ ] **Step 3: Implement explicit closure proofs**

Partition EOF is never closure evidence. Inactivity closes only below the soft event-time watermark; hard session watermarks close exactly once; a finite seal fails incomplete sessions; external sort closes only from a verified complete run; indefinite follow waits for a missing predecessor. Charge active frontier, pending predecessor, and spill descriptors to configured item/byte budgets. `PrivateSessionSpill` owns a no-follow `0700` run directory, creates `0600` files, rejects link/type/mode drift, and removes only its validated run subtree through RAII on success, error, and cancellation. A renewable owner lease uses injected `Clock`; startup performs a bounded cursor scan and reclaims crash-orphaned run directories only after lease expiry.

Individual session closure, root discovery, and partition EOF never mint
`WholeProducerTreeClosureReceipt`. Only verified finite seal/external-sort or an
equivalent authored hard completeness proof covering the exact root descendant
inventory may mint it; the receipt is checkpointed with that inventory.
Quarantine atomically retires live/pending/spilled session state and installs a
non-Clone budget-owned tombstone with run, input-domain, session key, issue ID,
`SessionCausalFrontier`, and checked `SessionQuarantineClosureProof`. Later
chunks extend that frontier without recreating the session. The reporter may
advance its handled cut through `Quarantine` only after the separately budgeted
prepared install acknowledgement and issue receipt are reachable at the same
barrier and their tombstone root is acknowledged by the committed generation.
Preparing or dropping an acknowledgement is non-destructive. A later fragment
is excluded, checked-extends the retained `SessionCausalFrontier`, invalidates
the prior root/acknowledgement, and requires a fresh acknowledgement. Retain the tombstone through checkpoint/resume until exact source
no-more-before evidence proves no later fragment, the final tombstone and issue
receipt are reachable in one generation, and generation-reader retention no
longer reaches it.

- [ ] **Step 4: Verify GREEN and commit**

Run Step 2, then commit:

```bash
git add rust/runtime/src/streaming/session.rs rust/runtime/src/streaming/session/conversation.rs rust/runtime/src/streaming/session/spill.rs rust/runtime/tests/streaming_session_closure.rs
git commit -m "feat(runtime): bound streaming session closure"
```

### Task P1C: Deferred Recorded-Content Reconstruction

**Depends on:** Tasks P1B, A5P, and A5.

**Files:**
- Create: `rust/runtime/src/streaming/session/recorded_content.rs`
- Modify: `rust/runtime/src/streaming/session.rs`
- Modify: `rust/runtime/src/streaming/action.rs`
- Test: `rust/runtime/tests/streaming_recorded_content.rs`

**Produces:** a typed session decorator/reconstruction owner that consumes
deferred Dynamo replay descriptors only after root/parent closure, expands them
through the shared pure synthesis profile, and emits ordinary canonical action
content. Its checkpoint state retains exact producer root/tail scope and the
bound session-program semantic digest; it never checkpoints memoized blocks.
The release gate consumes a typed whole-producer-tree closure receipt from P1B;
partition EOF or mere root discovery is insufficient.

- [ ] **Step 1: Write RED parity, closure, and resume tests**

Cover repeated/shared hashes; zero, tiny, full, and full-plus-partial inputs;
checkpoint before and after profile binding; root/tail-scope restore; and the
finite future-descendant/trailing-user-cap case. No action may release while a
later descendant can still alter finite message-role reconstruction.
Add an indefinite-follow case that lacks a whole-tree closure proof and fails
with `SessionFailureCode::UnboundedCausalityState` rather than guessing roles.

- [ ] **Step 2: Verify RED**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_recorded_content`

- [ ] **Step 3: Implement bounded deferred reconstruction**

Keep hash descriptors checkpointed under the session owner until the same
closure evidence needed by the finite future-aware message pass is proven.
Before allocating tokens or decoded text, normalize replay geometry and reserve
a separate action-content lease. The token charge is `input_length` on the
nonzero tiny-prompt path with no complete block, otherwise exactly the retained
complete-block count times block size; partial residual metadata is not
materialized. Checked accounting also includes token-vector capacity,
tokenizer-receipt conservative decoded-text bytes, message/string/vector
capacity, wire-body capacity, and action overhead. Expansion uses the shared
cache-free pure seam.
Generation 1 may run with cache capacity zero; if enabled, the cache is worker-
or cell-local, byte-bounded, evicting, non-waiting, accounts key and value
capacity, skips oversize entries, and never participates in checkpoints.
Capacity zero disables cache construction entirely; it is not passed to the
foundation's nonzero-capacity budget constructor.

- [ ] **Step 4: Verify GREEN and commit**

Run Step 2 plus finite Dynamo parity tests and commit only the named files with
`feat(runtime): reconstruct deferred recorded content`.

### Task P2: Multiplexed Action Host and State-Only Sink

**Depends on:** Task P1B and foundation Task 1D-R.

**Files:**
- Create: `rust/runtime/src/streaming/action/host.rs`
- Create: `rust/runtime/src/streaming/action/session_state.rs`
- Modify: `rust/runtime/src/streaming/action.rs`
- Test: `rust/runtime/tests/streaming_action_binding.rs`

**Interfaces:**
- Produces: `StreamingActionHost`, `ActiveExecutionSet`, exact schema binding map, one submitter/driver/control triple per binding, built-in `session_state` action sink, the sealed pre-receipt `CheckedActionFailureTerminalEvidenceView`, and the finalized `CheckedActionTerminalMembershipView`, each implemented only by P2 private host state.

```rust
pub struct ActiveExecutionSet {
    entries: BTreeMap<StableActionId, ActiveExecution>,
    budget: StreamingResourceBudget,
}

pub struct ActiveExecution {
    submitted: SubmittedAction,
    last_event_ordinal: u64,
    terminal_receipt: Option<BudgetOwnedActionTerminalReceipt>,
    lease: BudgetLease,
}
```

`ActiveExecution` exposes borrow-only submitted/event accessors and
`take_terminal_receipt`, which transfers the complete budget-owned wrapper to
the results plane. No accessor can separate receipt bytes from either lease.

- [ ] **Step 1: Write the RED lifecycle test**

```rust
#[tokio::test(flavor = "current_thread")]
async fn accepted_actions_have_one_terminal_receipt_in_event_order() {
    let (mut host, mut driver) = fake_action_host(active_budget(2, 1024));
    let handle = host.submit(ordered_action("a", 7)).await.unwrap();
    driver.emit(handle.admitted(0)).await;
    driver.emit(handle.first_token(1)).await;
    driver.emit(handle.terminal(2)).await;
    assert_eq!(host.drain_events().await.unwrap().ordinals(), [0, 1, 2]);
    assert_eq!(host.terminal_membership("a"), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn valid_terminal_membership_view_prepares_reporter_fact() {
    let mut fixture = action_reliability_fixture();
    fixture.finish(action(7), sequence(8), ActionTerminalOutcome::Succeeded);
    let terminal = fixture.prepare_action_terminal(action(7)).unwrap();
    fixture.report(IssueSequenceUpdate::CheckedActionTerminal(terminal)).await.unwrap();
    assert_eq!(fixture.reporter().terminal_membership(action(7)), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn failed_action_prepares_receipt_then_finalizes_without_circular_issue_id() {
    let mut fixture = failed_action_reliability_fixture(action(7), sequence(8));
    let queued = fixture.enqueue_failed_action(ordinary_endpoint_issue()).unwrap();
    let prepared = match fixture.poll_failed_action(queued).unwrap() {
        ActionFailureDisposition::TerminalActionReceipt(prepared) => prepared,
        other => panic!("expected terminal action receipt, got {other:?}"),
    };
    let issue_id = prepared.issue_id();
    assert_eq!(fixture.reporter().action_frontier(), sequence(7));
    let receipt = fixture.finish_failed_action(prepared).unwrap();
    assert_eq!(receipt.issue_id(), issue_id);
    let terminal = fixture.prepare_action_terminal(action(7)).unwrap();
    fixture.report(IssueSequenceUpdate::CheckedActionTerminal(terminal)).await.unwrap();
    assert_eq!(fixture.reporter().action_frontier(), sequence(8));
}

#[tokio::test(flavor = "current_thread")]
async fn dropped_failed_action_preparation_retries_same_id_without_double_count() {
    let mut fixture = failed_action_reliability_fixture(action(7), sequence(8));
    let queued = fixture.enqueue_failed_action(ordinary_endpoint_issue()).unwrap();
    let first = fixture.poll_terminal_action_failure(queued).unwrap();
    let issue_id = first.issue_id();
    drop(first);
    let replay = fixture.enqueue_failed_action(ordinary_endpoint_issue()).unwrap();
    let retry = fixture.poll_terminal_action_failure(replay).unwrap();
    assert_eq!(retry.issue_id(), issue_id);
    assert_eq!(fixture.reporter().action_issue_count(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn action_policy_dispositions_are_type_separated_and_nonblocking() {
    for (policy, expected) in [
        (action_retry_policy(), "retry"),
        (action_backpressure_policy(), "backpressure"),
        (action_terminal_policy(), "terminal_action_receipt"),
    ] {
        let mut fixture = ordered_failed_action_fixture(policy);
        let queued = fixture.enqueue_failed_action(ordinary_endpoint_issue()).unwrap();
        assert!(!fixture.reporter_is_borrowed_across_yield());
        tokio::task::yield_now().await;
        let disposition = fixture.poll_after_dense_predecessor(queued).unwrap();
        assert_eq!(disposition.name(), expected);
        assert_eq!(disposition.can_construct_terminal_receipt(), expected == "terminal_action_receipt");
    }
}
```

- [ ] **Step 2: Verify red**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_action_binding`

- [ ] **Step 3: Implement run-scoped multiplexing**

Consume the exact `PreparedStreamingActionBinding` defined by foundation Task 1D; do not redeclare or wrap it. `StreamingActionHost` owns the binding's submitter and driver while phase control retains the separately borrowable control.

Validate exactly one binding per emitted schema before preparation. Bound active handles/items/bytes and driver events. Assign dense global sequence only after causal+event safety. `session_state` produces admitted/terminal membership without endpoint execution. Cancellation uses the separately borrowable control and joins the driver.

- [ ] **Step 4: Verify green**

Run Step 2. Expected: PASS for missing/duplicate binding, event idempotency/order, capacity, cancellation, and exactly-one terminal.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/action.rs rust/runtime/src/streaming/action/host.rs rust/runtime/src/streaming/action/session_state.rs rust/runtime/tests/streaming_action_binding.rs
git commit -m "feat(runtime): multiplex streaming action bindings"
```

### Task P3: Bounded Pipeline and Local Placement

**Depends on:** Tasks P2, 5E, 6B, 7A, and foundation Task 1D-R.

**Files:**
- Create: `rust/runtime/src/streaming/pipeline.rs`
- Create: `rust/runtime/src/streaming/placement.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Test: `rust/runtime/tests/streaming_pipeline_sim.rs`

**Interfaces:**
- Consumes: prepared source/format/session/action components, event-time policy, checkpoint coordinator, result epoch control.
- Produces: `StreamingPipeline::run`, `StreamingPipelineControl`, the exact placement contracts below, `ActiveExecutionSet`, and bounded `LocalStreamingPlacement`.

```rust
pub struct StreamingPhaseContext {
    pub clock: Rc<dyn Clock>,
    pub checkpoint: StreamingCheckpointCoordinator,
    pub results: EpochResultCoordinator,
    pub stop: StreamingStopReceiver,
}

pub enum StreamingTerminalReason {
    Sealed,
    Cancelled,
    DrainedAfterReliabilityFence { issue_id: ContentDigest },
    FailedInvariant { issue_id: ContentDigest },
}

pub struct StreamingRunOutcome {
    pub terminal_reason: StreamingTerminalReason,
    pub last_committed_generation: Option<CommittedCheckpointGeneration>,
}
```

```rust
pub trait StreamingPlacementPolicy: StreamingCheckpointParticipant {
    fn route_admission(&self, action: &OrderedDatasetAction)
        -> Result<Option<PlacementRouteCharge>, PlacementError>;
    fn install_route_reservation(
        &mut self,
        reservation: PlacementRouteReservation,
    ) -> Result<(), PlacementError>;
    fn place(&mut self, action: &OrderedDatasetAction)
        -> Result<PlacementDecision, PlacementError>;
    fn observe_session_terminal(
        &mut self,
        session: StableSessionKey,
        ownership_epoch: SessionOwnershipEpoch,
        causal_frontier: &SessionCausalFrontier,
    ) -> Result<(), PlacementError>;
}

pub struct PlacementRouteCharge {
    pub session: StableSessionKey,
    pub items: usize,
    pub bytes: usize,
}

pub struct PlacementRouteReservation {
    pub session: StableSessionKey,
    pub lease: BudgetLease,
}

/// Separately borrowable async capacity owner for deterministic placement.
#[async_trait(?Send)]
pub trait StreamingPlacementAdmission {
    async fn reserve_route(&mut self, charge: PlacementRouteCharge)
        -> Result<PlacementRouteReservation, PlacementError>;
}

#[async_trait(?Send)]
pub trait StreamingPlacementSubmitter {
    async fn prepare(&mut self, decision: PlacementDecision, action: OrderedDatasetAction)
        -> Result<PlacementHandle, PlacementError>;
    async fn release(&mut self, handle: PlacementHandleId) -> Result<(), PlacementError>;
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

pub struct PreparedStreamingPlacementBinding {
    pub admission: Box<dyn StreamingPlacementAdmission>,
    pub policy: Box<dyn StreamingPlacementPolicy>,
    pub submitter: Box<dyn StreamingPlacementSubmitter>,
    pub driver: Box<dyn StreamingPlacementDriver>,
    pub control: Box<dyn StreamingPlacementControl>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PlacementHandleId(u64);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementDecision {
    pub route_id: u32,
    pub destination_cell: Option<u32>,
    pub ownership_epoch: SessionOwnershipEpoch,
}

pub struct PlacementHandle {
    pub id: PlacementHandleId,
    pub action_id: StableActionId,
    pub global_sequence: GlobalSequence,
    pub ownership_epoch: SessionOwnershipEpoch,
}

pub struct PlacementPreparedReceipt {
    pub handle: PlacementHandleId,
    pub content_digest: ContentDigest,
}

pub struct PlacementReleasedReceipt {
    pub handle: PlacementHandleId,
}

pub struct PlacementFailureReceipt {
    pub handle: Option<PlacementHandleId>,
    pub code: PlacementFailureCode,
}

// `PlacementFailureCode` is the stable failure vocabulary owned by foundation Task 1D.
```

- [ ] **Step 1: Write the RED backpressure/shutdown test**

```rust
#[tokio::test(flavor = "current_thread")]
async fn downstream_pressure_stops_every_upstream_pull_and_stop_wakes_pending() {
    let fixture = PipelineFixture::follow_with_capacity_one();
    let (run, control, probes) = fixture.start();
    probes.block_terminal_lane();
    probes.publish_many(100).await;
    assert!(probes.high_water().within_authored_limits());
    assert_eq!(probes.source_pulls_after_saturation(), 0);
    control.stop().await.unwrap();
    run.await.unwrap();
    assert!(probes.all_owners_joined());
}
```

- [ ] **Step 2: Verify red**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_pipeline_sim`

- [ ] **Step 3: Implement fused composition**

```rust
impl StreamingPipeline {
    pub async fn run(
        self,
        phase: StreamingPhaseContext,
    ) -> Result<StreamingRunOutcome, StreamingPipelineError> {
        self.run_fused(phase).await
    }
}
```

Pull a new unit only when the next stage owns permits. Prefer inline/fused calls on the worker `LocalSet`; bounded leased channels are allowed only at measured concurrency boundaries. `Pending`, `Seal`, and `Cancelled` remain distinct. Shutdown fences admission, wakes pending source/decode/order, drains or cancels accepted actions through phase policy, checkpoints only a valid cut, and joins all owners.

For deferred recorded content, downstream readiness includes a pre-allocation
action-content reservation sized from P1C's normalized tiny/full-block token
geometry, all retained collection/wire capacities, and the prepared tokenizer's
checked conservative decode bound. The pipeline must acquire that lease
before token-vector/text allocation and must stop upstream pulls while it is
pending. A small hash descriptor never authorizes a large uncharged decoded
request. Optional reconstruction-cache admission is non-waiting and separate
from action ownership.

Before calling the synchronous `StreamingPlacementPolicy::place`, the fused
pipeline asks `route_admission` whether that exact action needs a new route.
When it does, the pipeline polls the separately owned
`StreamingPlacementAdmission::reserve_route` future in the same `select!` loop
as placement-driver events and shutdown. A terminal event can therefore call
`observe_session_terminal` on the policy and release route capacity while a
reservation is pending; the capacity wait never holds a borrow of the policy
or route map. Once ready, the pipeline synchronously calls
`install_route_reservation` and `place` for the same action with no intervening
`.await`. Cancellation drops the returned move-only reservation or the pending
budget future without installing a route. The local implementation returns no
route charge because it introduces no persistent route map.

The policy and admission owner may each retain a cheap clone of the same
`StreamingResourceBudget` accounting handle. They never share the route map.
The policy uses its handle only during checkpoint initialization to reacquire
leases for restored routes before polling begins; live capacity waits remain
owned exclusively by the separately borrowable admission object.

`LocalStreamingPlacement` implements the same separately owned admission/policy/submitter/driver/control split as cellular without a transport hop. Placement policy, placement driver, `ActiveExecutionSet`, `StreamingBlockingExecutor`, and `EpochResultCoordinator` are stable checkpoint participants; dynamic handles, blocking jobs, and result segments aggregate beneath them. Pipeline preparation freezes the exact required participant set before source polling. `PlacementEvent::Action` is the only route back into session state.

- [ ] **Step 4: Verify green**

Run Step 2. Expected: PASS for finite seal, quiet follow, source error, permits, cross-partition session, checkpoint, and shutdown.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/pipeline.rs rust/runtime/src/streaming/placement.rs rust/runtime/tests/streaming_pipeline_sim.rs
git commit -m "feat(runtime): compose bounded streaming pipeline"
```

### Task P4: Scheduled-Request Sink and Executable Shadow Workload

**Depends on:** Tasks P3, 4B, 6D, foundation Task 1D-R, plus adapter Tasks A1-A2. The Dynamo product
path additionally depends on A5P, A5, and P1C; capability agreement must omit or
refuse that composition until all three factories are present.

**Files:**
- Create: `rust/runtime/src/streaming/action/scheduled_request.rs`
- Modify: `rust/runtime/src/streaming/action.rs`
- Create: `rust/runtime/src/engine/streaming_execution.rs`
- Modify: `rust/runtime/src/engine/online_execution.rs`
- Modify: `rust/runtime/src/engine/mod.rs`
- Create: `rust/runtime/tests/support/streaming_pipeline.rs`
- Test: `rust/runtime/tests/streaming_shadow_operation.rs`

**Interfaces:**
- Consumes: `NativeTransportExecution::{executor_factory,request_materializer}`, reusable phase/capture service, bounded `ScheduledRuntime`, `PreparedRunnerOperation`.
- Produces: action schema `session_request.v1`; executable registered workload ID `shadow_replay`; `ShadowReplayPreparedOperation`; the immutable action inventory whose borrowed view alone implements sealed `FrozenActionInventoryView`.

- [ ] **Step 1: Write the RED operation test**

```rust
#[tokio::test(flavor = "current_thread")]
async fn local_shadow_run_replays_cross_chunk_session_at_delayed_targets() {
    let request = local_shadow_request_with_sim_anchor(utc_ns(1_000), delay_ns(300));
    let outcome = support::execute_shadow_operation(request).await.unwrap();
    assert_eq!(outcome.records().stable_action_ids(), ["turn-0", "turn-1"]);
    assert_eq!(outcome.records().target_times_ns(), [1_300, 1_500]);
    assert_eq!(outcome.report().checkpoint_generation(), Some(2));
}

#[tokio::test(flavor = "current_thread")]
async fn frozen_action_inventory_prepares_gap_only_after_every_terminal() {
    let mut fixture = shadow_action_inventory_fixture(2);
    fixture.finish(sequence(0));
    assert!(fixture.prepare_no_more_actions_before(sequence(1)).is_err());
    fixture.finish(sequence(1));
    let gap = fixture.prepare_no_more_actions_before(sequence(1)).unwrap();
    fixture.report(IssueSequenceUpdate::CheckedNoMoreActionsBefore(gap)).await.unwrap();
    assert_eq!(fixture.reporter().action_frontier(), sequence(1));
}
```

- [ ] **Step 2: Verify red**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_shadow_operation`

- [ ] **Step 3: Implement through existing execution seams**

```rust
impl PreparedRunnerOperation for ShadowReplayPreparedOperation {
    fn execute(self: Box<Self>) -> anyhow::Result<PreparedRunOutcome> {
        self.execute_streaming_pipeline()
    }
}
```

Materialize requests through the selected endpoint, submit through extracted dispatcher/runtime facilities, and translate observer events to action events without new token-path hooks. Produce and register the `scheduled_request` action-sink factory and the `shadow_replay` workload only in this executable commit. Prepare every selected factory once and initialize participants before polling. Refuse unsupported phase/resource/exporter/accuracy combinations during validation.

Freeze the exact dense action inventory before publishing a no-more-actions
frontier. Its private inventory alone implements the action-module-sealed
`FrozenActionInventoryView`; the borrowed view binds the run, through sequence,
terminal-membership root, and lookup proof for every sequence through that
frontier. P4 passes this view to
`StreamingIssueReporter::prepare_no_more_actions_before` and cannot construct
the returned opaque checked update.

Dynamo actions arrive here only after P1C has reconstructed canonical content;
P4 never interprets Dynamo hashes or owns a second synthesis cache. The static
frozen execution identity binds the authored synthesis profile. The dynamic
checkpoint/session authority separately binds
`SynthesisAuthority::{Unbound, Bound}` plus the bound session-program digest;
resume compares that authority before participant initialization without ever
mutating the frozen execution plan.

- [ ] **Step 4: Verify green**

Run Step 2. Expected: PASS for dry-run/socket-free execution, timing, stable ordinal, cancellation, result checkpoint, and no Python child.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/action.rs rust/runtime/src/streaming/action/scheduled_request.rs rust/runtime/src/engine/streaming_execution.rs rust/runtime/src/engine/online_execution.rs rust/runtime/src/engine/mod.rs rust/runtime/tests/support/streaming_pipeline.rs rust/runtime/tests/streaming_shadow_operation.rs
git commit -m "feat(engine): execute native streaming shadow replay"
```

### Task P5: Incremental Agent/Graph Sessions

**Depends on:** Tasks P1 and P2.

**Files:**
- Create: `rust/runtime/src/streaming/session/agent_graph.rs`
- Create: `rust/runtime/src/streaming/action/graph.rs`
- Modify: `rust/runtime/src/streaming/session.rs`
- Modify: `rust/runtime/src/streaming/action.rs`
- Test: `rust/runtime/tests/streaming_graph_sessions.rs`

**Interfaces:**
- Produces: session program ID `agent_graph`; graph action schema/binding using existing materializer/dispatch primitives.

- [ ] **Step 1: Write the RED hidden-parent/cycle test**

```rust
#[tokio::test(flavor = "current_thread")]
async fn child_waits_across_chunks_and_late_cycle_fails_before_release() {
    let mut graph = streaming_graph_session("s");
    assert!(graph.apply(node("child").depends_on("parent")).unwrap().is_empty());
    let ready = graph.apply(node("parent")).unwrap();
    assert_eq!(ready.ids(), ["parent"]);
    graph.mark_terminal("parent").unwrap();
    assert_eq!(graph.take_ready().ids(), ["child"]);
    assert!(matches!(graph.apply(edge("parent", "child")), Err(GraphStreamError::Cycle { .. })));
}
```

- [ ] **Step 2: Verify red**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_graph_sessions`

- [ ] **Step 3: Implement append-only graph state**

Use immutable node/edge identities, declared predecessor sets, incremental cycle refusal, edge-after-execution refusal, bounded spill, tool-result inertness, and controller-linearized action events. Do not build a complete `GraphTraceProgram` or execute recorded tools.

- [ ] **Step 4: Verify green**

Run Step 2. Expected: PASS for multi-chunk graph, duplicates/conflicts, retry attempts under one logical action, closure, checkpoint/restore, and dependent release.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming/session.rs rust/runtime/src/streaming/action.rs rust/runtime/src/streaming/session/agent_graph.rs rust/runtime/src/streaming/action/graph.rs rust/runtime/tests/streaming_graph_sessions.rs
git commit -m "feat(graph): execute cross-chunk streaming sessions"
```

### Task P6: Recorded and Encrypted Target-Closed-Loop Policies

**Depends on:** Tasks P1 and P4.

**Files:**
- Create: `rust/runtime/src/streaming/sensitive_state.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Create: `rust/runtime/src/engine/streaming_secrets.rs`
- Modify: `rust/runtime/src/engine/mod.rs`
- Modify: `rust/runtime/src/streaming/session.rs`
- Modify: `rust/runtime/src/engine/registry.rs:1113-1160` (`RunContext`)
- Modify: `rust/runtime/src/engine/application.rs`
- Test: `rust/runtime/tests/streaming_sensitive_state.rs`

**Interfaces:**
- Produces: `StreamingSensitiveStateKeyResolver`; `SensitiveStateKey { key_id, key: Zeroizing<[u8; 32]> }`; versioned XChaCha20-Poly1305 envelope; `recorded_inputs` and `target_closed_loop` policies.

```rust
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SensitiveStateKeyId(String);

pub struct SensitiveStateKey {
    pub key_id: SensitiveStateKeyId,
    pub key: Zeroizing<[u8; 32]>,
}
```

- [ ] **Step 1: Write the RED policy/envelope test**

```rust
#[test]
fn target_state_requires_authenticated_external_key_and_bound_aad() {
    let resolver = FakeKeyResolver::one("key-a", [7; 32]);
    let context = sensitive_context(run("r"), generation(4), participant("session"));
    let envelope = encrypt_sensitive(&resolver, "key-a", &context, b"target text").unwrap();
    assert_eq!(decrypt_sensitive(&resolver, &context, &envelope).unwrap(), b"target text");
    assert!(decrypt_sensitive(&resolver, &context.with_generation(5), &envelope).is_err());
    assert!(!format!("{envelope:?}").contains("target text"));
}
```

- [ ] **Step 2: Verify red**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_sensitive_state`

- [ ] **Step 3: Implement narrow authority and policy**

```rust
pub trait StreamingSensitiveStateKeyResolver: Debug + Send + Sync {
    fn resolve(&self, key_id: &SensitiveStateKeyId)
        -> Result<SensitiveStateKey, SensitiveStateError>;
}
```

Resolve keys from inherited private FD or exact `0600` no-follow file; config stores only opaque key ID. Use fresh 24-byte nonce and length-delimited AAD binding run, generation, participant, schema, policy digest, key ID, plaintext length/BLAKE3. Zeroize key/plaintext. `recorded_inputs` rejects/compares target content without mutating later requests. `target_closed_loop` requires encrypted backend capability or checkpoint `none` and no resume claim.

- [ ] **Step 4: Verify green**

Run Step 2. Expected: PASS for divergent target behavior, correct restart, wrong key/tamper, nonce uniqueness, redaction, and backend refusal.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/sensitive_state.rs rust/runtime/src/streaming/session.rs rust/runtime/src/engine/mod.rs rust/runtime/src/engine/streaming_secrets.rs rust/runtime/src/engine/registry.rs rust/runtime/src/engine/application.rs rust/runtime/tests/streaming_sensitive_state.rs
git commit -m "feat(runtime): protect target-closed-loop session state"
```

### Task P7: Streaming-Plane Observability

**Depends on:** Tasks P3, P4, 6B, 7A, and foundation Task 1D-R.

**Files:**
- Create: `rust/runtime/src/streaming/observability.rs`
- Modify: `rust/runtime/src/streaming.rs`
- Modify: `rust/runtime/src/metrics_core/report.rs`
- Test: `rust/runtime/tests/streaming_observability.rs`

**Produces:** one bounded `StreamingPlaneMetrics` snapshot populated at stage boundaries, never from per-token callbacks.

```rust
pub struct StreamingDistributionSnapshot { pub count: u64, pub sum_ns: u128, pub max_ns: u64 }
pub struct QueueHighWater { pub items: usize, pub bytes: usize, pub item_limit: usize, pub byte_limit: usize }
pub enum StreamingStage { Source, Acquire, Decode, Order, Session, Placement, Action, Terminal, Result }
pub enum StreamingDropReason { Late, Overload, AuthoredPolicy, Duplicate }
pub struct ScheduledActionHorizon(GlobalSequence);
pub struct CheckpointHorizonSnapshot {
    pub cut: CheckpointCut,
    pub scheduled: ScheduledActionHorizon,
}

pub struct StreamingPlaneMetrics {
    pub publication_lag_ns: StreamingDistributionSnapshot,
    pub acquisition_duration_ns: StreamingDistributionSnapshot,
    pub decode_duration_ns: StreamingDistributionSnapshot,
    pub watermark_lag_ns: StreamingDistributionSnapshot,
    pub causal_wait_ns: StreamingDistributionSnapshot,
    pub schedule_slip_ns: StreamingDistributionSnapshot,
    pub admission_wait_ns: StreamingDistributionSnapshot,
    pub endpoint_ns: StreamingDistributionSnapshot,
    pub queues: BTreeMap<StreamingStage, QueueHighWater>,
    pub drops_by_reason: BTreeMap<StreamingDropReason, u64>,
    pub duplicate_count: u64,
    pub gap_count: u64,
    pub checkpoint_horizons: CheckpointHorizonSnapshot,
}
```

- [ ] **Step 1: Write the RED timing/high-water test**

```rust
#[tokio::test(flavor = "current_thread")]
async fn stage_metrics_separate_lag_wait_slip_and_endpoint_time() {
    let fixture = observability_fixture_with_sim_clock();
    fixture.run_one_action().await.unwrap();
    let metrics = fixture.snapshot();
    assert_eq!(metrics.schedule_slip_ns.count, 1);
    assert_eq!(metrics.endpoint_ns.count, 1);
    assert!(metrics.queues.values().all(|q| q.items <= q.item_limit && q.bytes <= q.byte_limit));
    assert_eq!(metrics.checkpoint_horizons.cut.terminal,
        TerminalActionHorizon::new(GlobalSequence::new(0)));
}
```

- [ ] **Step 2: Verify RED**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_observability`

- [ ] **Step 3: Instrument stage boundaries**

Observe acquisition, publication, decode, watermark, causal release, schedule admission, endpoint terminal, queue permit high-water, drop reason, duplicate/gap, and all typed checkpoint horizons. Use `Clock` timestamps and bounded mergeable distributions. Aggregate per worker and merge at report/checkpoint boundaries; do not add logging, allocation, locking, or observer work to the token callback.

- [ ] **Step 4: Verify GREEN and commit**

Run Step 2, then commit:

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/observability.rs rust/runtime/src/metrics_core/report.rs rust/runtime/tests/streaming_observability.rs
git commit -m "feat(runtime): report streaming plane metrics"
```

## Reliability-Continuation Amendment

The issue reporter is a separately owned run-scoped participant. Session and
action components borrow it only at explicit event boundaries; they do not hold
that borrow across source, endpoint, checkpoint, or control awaits. These
changes extend the existing tasks and their single RED/GREEN commands:

| Owner | New responsibility | Required RED cases |
|---|---|---|
| P1/P1B | retire a quarantined record/session and every owned pending-predecessor/spill lease at the exact causal frontier; install a durable `(input_domain, session)` tombstone; never treat conflicting stable content as ordinary quarantine | `missing_predecessor_policy_quarantines_only_the_bounded_session`, `quarantined_session_cannot_resurrect_and_tombstone_survives_resume`, `conflicting_stable_content_remains_terminal` |
| P2 | convert endpoint timeout/HTTP/gRPC/application faults through the reporter-owned sealed `ActionFailureDisposition`: enqueue without retaining a reporter borrow, poll dense-order completion at later event boundaries, reschedule `Retry`, pause/fence `Backpressure`, and consume only `TerminalActionReceipt(PreparedActionFailureIdentity)` into exactly one checked `ActionTerminalReceipt`; then expose finalized membership so 1D-R can mint the fact; retry only when pre-acceptance safety or target idempotency proves no uncontrolled duplicate | `endpoint_failure_is_terminal_for_action_not_run`, `retry_exhaustion_emits_one_failed_terminal_receipt`, `failed_action_prepares_receipt_then_finalizes_without_circular_issue_id`, `dropped_failed_action_preparation_retries_same_id_without_double_count`, `action_policy_dispositions_are_type_separated_and_nonblocking`, `action_terminal_receipt_rejects_foreign_run_action_and_success_error_collision`, `valid_terminal_membership_view_prepares_reporter_fact`, `forged_action_success_and_unproved_gap_are_unnameable`, `action_terminal_fact_rejects_mismatched_action_or_sequence`, `tampered_action_terminal_receipt_fails_strict_restore`, `later_action_issues_after_terminal_failure` |
| P3 | apply the exhaustive scope/disposition matrix, accept `Continue` only from the sealed no-membership-loss path, and honor `needs_admission_fence` by stopping new work and truthfully draining; call failed-run shutdown only for the module-private classifier's `FailRun` decision | `ordinary_issue_never_enters_failed_run_shutdown`, `continue_without_no_membership_loss_proof_is_unnameable`, `continuation_threshold_fences_admission_then_drains_truthful_prefix` |
| P4 | inject the resolved frozen reliability policy/reporter into every stage and include its policy digest in execution-plan agreement; expose the private sealed frozen-action-inventory view used by the reporter to prove dense no-more-actions gap closure; no adapter/workload default may replace it | `pipeline_rejects_reliability_policy_digest_mismatch_before_poll_or_issue`, `frozen_action_inventory_view_prepares_gap_only_after_every_terminal` |
| P7 | expose counts by scope/class/disposition, retry ordinals, hole/quarantine membership, failed terminal actions, admission-fence state, and incomplete derived sinks | `observability_separates_failed_action_from_failed_run` |

P2 adds the following outcome without changing stable logical action identity:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum ActionTerminalOutcome {
    Succeeded,
    EndpointFailure { issue_id: ContentDigest },
    Cancelled,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct ActionTerminalReceipt {
    run: StreamRunIdentity,
    action_id: StableActionId,
    global_sequence: GlobalSequence,
    outcome: ActionTerminalOutcome,
    content_digest: ContentDigest,
}

pub struct BudgetOwnedActionTerminalReceipt {
    receipt: ActionTerminalReceipt,
    encoded: BudgetedCheckpointBytes,
    parsed_lease: BudgetLease,
}
```

The checked receipt constructor binds the issue scope to the same run and
action. Its failure overload consumes the payload of
`ActionFailureDisposition::TerminalActionReceipt`; the sealed `Retry` and
`Backpressure` payloads expose no conversion or terminal identity, and there is
no constructor taking a raw issue ID or ordinary issue. It rejects two terminals for one action and a success/error membership
collision. The live receipt and wrapper fields are private; neither type is
`Clone`, and the live receipt does not implement `Deserialize`. A private
unknown-field-denying wire DTO is decoded only by a bounded context-taking
restore function receiving the expected run, expected action, exact terminal
membership context, and budget; it re-runs the checked constructor before any
live state is returned. Compile-fail and serde RED prove callers cannot
literal-construct fields, deserialize a forged live receipt, separate the
encoded bytes or parsed lease, replay an error across run/action, or collide
success/error membership. The wrapper's compact encoded allocation and parsed
heap are charged separately and remain inseparable while owned by
`ActiveExecution`; `take_terminal_receipt` transfers it intact into the
budget-owned result partition.
P2 and P4 do not mint reliability facts. For failure P2 first exposes sealed
terminal evidence without an issue ID, synchronously enqueues it and later
polls without holding a reporter borrow across an await. It handles `Retry` and
`Backpressure` without terminalizing, consumes only reporter-owned
`TerminalActionReceipt(PreparedActionFailureIdentity)` into
`BudgetOwnedActionTerminalReceipt`, and
only then exposes finalized membership. P2/P4 otherwise expose their respective
action-module-sealed borrowed terminal-membership and frozen-inventory views;
`StreamingIssueReporter::prepare_action_terminal` and
`prepare_no_more_actions_before` validate those views and mint private-field
`CheckedActionTerminalFact` and `CheckedNoMoreActionsBefore`. No raw reporter
update accepts an arbitrary `Option<issue>` or frontier. Failure facts must bind
the receipt issue to the exact action and sequence, success facts must bind the
successful terminal membership, and gap proofs cover every sequence they
advance through.
Endpoint retry attempts remain telemetry; only the first reachable
logical terminal receipt contributes to metrics. Retry exhaustion selects
`TerminalActionReceipt`, never `FailRun`. A frozen threshold may
pause admission for truthful draining but cannot reclassify the endpoint fault
as an invariant. The endpoint admission-fence threshold counts cumulative
committed failed-action receipts. Success does not reset it, so source/worker
arrival order cannot change the threshold crossing; add
`endpoint_failure_threshold_is_cumulative_across_success_and_restart`.

Add this integrated RED matrix to `streaming_pipeline_sim` before P3 production
changes:

```rust
#[tokio::test(flavor = "current_thread")]
async fn scoped_faults_continue_and_invariants_stop() {
    for case in reliability_pipeline_cases() {
        let observed = case.run().await;
        assert_eq!(observed.disposition, case.expected_disposition);
        assert_eq!(observed.later_actions_issued, case.expected_later_actions);
        assert_eq!(observed.is_run_failed, case.expected_fail_run);
        assert!(observed.receipts_and_horizons_are_truthful());
        assert!(observed.all_scoped_state_is_settled());
    }
}
```

The matrix includes record quarantine, session quarantine, partition hole,
endpoint timeout, endpoint permanent terminal response, retry exhaustion,
admission fencing, conflicting stable content, foreign run/proof, watermark
regression, frozen plan drift, and lease/membership accounting corruption. Only
the final five invariant/authority rows may set `is_run_failed`.

## Subsystem Completion Gate

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,parquet --test streaming_reliability --test streaming_session_continuity --test streaming_session_closure --test streaming_action_binding --test streaming_pipeline_sim --test streaming_shadow_operation --test streaming_graph_sessions --test streaming_sensitive_state --test streaming_observability
```

Review must confirm bounded high-water diagnostics, no new hot-token callback/allocation, no source/format switches, no placeholder capabilities, and existing finite scheduled/graph behavior unchanged.
