<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Streaming Pipeline and Sessions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compose the native streaming contracts into a bounded shadow-replay workload that preserves conversation and agent/graph sessions across arbitrary partition boundaries.

**Architecture:** One run-scoped coordinator owns session/causal state; a fused current-thread pipeline pulls only under downstream permits; one action host multiplexes binding drivers into existing transport/dispatch/observation seams. Recorded-input replay is the default, while target-closed-loop state is a separately authorized encrypted policy.

**Tech Stack:** Rust 2024, Tokio current-thread `LocalSet`, AIPerf engine/clock/dispatch/phase/capture seams, XChaCha20-Poly1305, zeroize.

**Spec:** `artifacts/streaming-design/streaming-dataset-shadow-replay-design.md` at approved commit `505efc06b0`.

## Global Constraints

- Requires foundation/runtime, checkpoint/results, and adapter Tasks A1-A2 before the first executable workload.
- No `NativeDatasetPlan::Streaming`, complete `GraphTraceProgram`, per-action boxed driver, unbounded queue/task/vector, or Python path.
- One profiling phase in generation 1; warmup/live-profile combinations fail during capability agreement.
- Every accepted action reaches exactly one terminal receipt; checkpoint cuts remain typed by stage.
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

**Depends on:** Task P1.

**Files:**
- Modify: `rust/runtime/src/streaming/session/conversation.rs`
- Create: `rust/runtime/src/streaming/session/spill.rs`
- Modify: `rust/runtime/src/streaming/session.rs`
- Test: `rust/runtime/tests/streaming_session_closure.rs`

**Produces:** `SessionClosurePolicy`, `MissingPredecessorPolicy`, externally sorted finite completeness receipts, and explicit refusal when neither a finite bound nor spill/drop/fail policy exists.

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
```

- [ ] **Step 2: Verify RED**

Run: `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming --test streaming_session_closure`

- [ ] **Step 3: Implement explicit closure proofs**

Partition EOF is never closure evidence. Inactivity closes only below the soft event-time watermark; hard session watermarks close exactly once; a finite seal fails incomplete sessions; external sort closes only from a verified complete run; indefinite follow waits for a missing predecessor. Charge active frontier, pending predecessor, and spill descriptors to configured item/byte budgets. `PrivateSessionSpill` owns a no-follow `0700` run directory, creates `0600` files, rejects link/type/mode drift, and removes only its validated run subtree through RAII on success, error, and cancellation. A renewable owner lease uses injected `Clock`; startup performs a bounded cursor scan and reclaims crash-orphaned run directories only after lease expiry.

- [ ] **Step 4: Verify GREEN and commit**

Run Step 2, then commit:

```bash
git add rust/runtime/src/streaming/session.rs rust/runtime/src/streaming/session/conversation.rs rust/runtime/src/streaming/session/spill.rs rust/runtime/tests/streaming_session_closure.rs
git commit -m "feat(runtime): bound streaming session closure"
```

### Task P2: Multiplexed Action Host and State-Only Sink

**Depends on:** Task P1B.

**Files:**
- Create: `rust/runtime/src/streaming/action/host.rs`
- Create: `rust/runtime/src/streaming/action/session_state.rs`
- Modify: `rust/runtime/src/streaming/action.rs`
- Test: `rust/runtime/tests/streaming_action_binding.rs`

**Interfaces:**
- Produces: `StreamingActionHost`, `ActiveExecutionSet`, exact schema binding map, one submitter/driver/control triple per binding, and built-in `session_state` action sink.

```rust
pub struct ActiveExecutionSet {
    entries: BTreeMap<StableActionId, ActiveExecution>,
    budget: StreamingResourceBudget,
}

pub struct ActiveExecution {
    pub submitted: SubmittedAction,
    pub last_event_ordinal: u64,
    pub terminal_receipt: Option<ActionTerminalReceipt>,
    pub lease: BudgetLease,
}
```

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

**Depends on:** Tasks P2, 5E, 6B, and 7A.

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

pub enum StreamingTerminalReason { Sealed, Cancelled, Failed }

pub struct StreamingRunOutcome {
    pub terminal_reason: StreamingTerminalReason,
    pub last_committed_generation: Option<CommittedCheckpointGeneration>,
}
```

```rust
pub trait StreamingPlacementPolicy: StreamingCheckpointParticipant {
    fn place(&mut self, action: &OrderedDatasetAction)
        -> Result<PlacementDecision, PlacementError>;
    fn observe_session_terminal(
        &mut self,
        session: StableSessionKey,
        ownership_epoch: SessionOwnershipEpoch,
        causal_frontier: &SessionCausalFrontier,
    ) -> Result<(), PlacementError>;
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

`LocalStreamingPlacement` implements the same policy/submitter/driver/control split as cellular without a transport hop. Placement policy, placement driver, `ActiveExecutionSet`, `StreamingBlockingExecutor`, and `EpochResultCoordinator` are stable checkpoint participants; dynamic handles, blocking jobs, and result segments aggregate beneath them. Pipeline preparation freezes the exact required participant set before source polling. `PlacementEvent::Action` is the only route back into session state.

- [ ] **Step 4: Verify green**

Run Step 2. Expected: PASS for finite seal, quiet follow, source error, permits, cross-partition session, checkpoint, and shutdown.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/streaming.rs rust/runtime/src/streaming/pipeline.rs rust/runtime/src/streaming/placement.rs rust/runtime/tests/streaming_pipeline_sim.rs
git commit -m "feat(runtime): compose bounded streaming pipeline"
```

### Task P4: Scheduled-Request Sink and Executable Shadow Workload

**Depends on:** Tasks P3, 4B, and 6D plus adapter Tasks A1-A2.

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
- Produces: action schema `session_request.v1`; executable registered workload ID `shadow_replay`; `ShadowReplayPreparedOperation`.

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

**Depends on:** Tasks P3, P4, 6B, and 7A.

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

## Subsystem Completion Gate

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,parquet --test streaming_session_continuity --test streaming_session_closure --test streaming_action_binding --test streaming_pipeline_sim --test streaming_shadow_operation --test streaming_graph_sessions --test streaming_sensitive_state --test streaming_observability
```

Review must confirm bounded high-water diagnostics, no new hot-token callback/allocation, no source/format switches, no placeholder capabilities, and existing finite scheduled/graph behavior unchanged.
