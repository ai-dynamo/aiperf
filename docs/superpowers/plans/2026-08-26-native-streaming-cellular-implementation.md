# Native Streaming Cellular Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add authenticated, bounded cellular placement and controller-authoritative checkpoint/result convergence for native streaming shadow replay, without early issue, per-action drivers, or reuse of the finite dataset broadcast path.

**Architecture:** The controller remains the sole owner of source decoding, causal session/graph state, dense global sequence, route ownership, and the global checkpoint generation. One run-scoped multiplexed placement binding transfers strict authenticated `PrepareAction`/`ReleaseAction` commands and ordered cell events; cells own only fenced request material and endpoint execution. Worker/cell result partitions become authoritative only when the controller validates the exact barrier membership and commits the global generation last.

**Tech Stack:** Rust 2024, Tokio current-thread runtimes and `LocalSet`, injected `Clock`, existing Ed25519 cellular security/replay ledger, Velo handlers, bounded item/byte permits, BLAKE3, MessagePack with strict Serde DTOs, and the native streaming checkpoint/result traits.

**Spec:** `artifacts/streaming-design/streaming-dataset-shadow-replay-design.md` at approved commit `505efc06b0`, especially lines 1040-1074, 1180-1207, 2076-2087, and 2126-2217.

## Global Constraints

- Cargo commands run from the nested `rust/` workspace; git commands run from the repository root. Builds use `CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target`.
- Each task includes the nearest parent module declaration required for its own GREEN build; declaration conflicts are resolved during integration.
- Every runtime command enables `--features streaming,cellular`; `engine` alone does not enable streaming.
- These tasks consume the non-cellular streaming contracts: `StreamingPlacementPolicy`, `StreamingPlacementSubmitter`, `StreamingPlacementDriver`, `StreamingPlacementControl`, `ActiveExecutionSet`, `StreamingCheckpointParticipant`, `StreamingGenerationTransaction`, and checkpoint result-index types.
- Controller-to-cell and cell-to-controller streaming frames are authenticated for exact role, purpose, run/session nonce, peer/destination, sequence, and payload digest before DTO decode.
- Do not extend or reuse `dataset_velo`, `DatasetSubscribe`, fixed startup dataset pushes, or add-only `CellMessage` history for streaming actions.
- There is one multiplexed placement driver per prepared binding, not one boxed driver, channel, or task per action.
- Every queue/window is bounded by items and bytes. Permit ownership moves with the payload and is released by RAII.
- A cell cannot issue on `PrepareAction`. Only a valid `ReleaseAction` for the same action, global sequence, route, and ownership epoch may issue it.
- Controller `Clock` decides release time. A cell never interprets a controller timestamp and cross-host wall/monotonic readings are never compared.
- Canonical session/graph state never moves to a cell. Every update is linearized at the controller before dependent release.
- No `Arc<Mutex<_>>` on request/token paths, no lock across `.await`, no unbounded channels, no detached tasks, and no direct `Instant::now`, `SystemTime::now`, or Tokio timers.
- Each task below is sequential, begins from the integrated commit of its predecessor, passes its exact RED/GREEN command, receives Graham and independent behavior review, and lands as one focused commit.

## Existing Seams and Anchors

- `rust/runtime/src/engine/cellular_registration.rs:37-205`: process security authority, worker/controller signers, role verifiers, and per-purpose send sequences.
- `rust/runtime/src/engine/cellular_registration.rs:229-370`: `AdmissionPurpose`, authenticated frame decode, fixed rejection classes, and verified-payload boundary.
- `rust/runtime/src/engine/cellular_registration.rs:464-570`: bounded per-role replay ledger; authentication precedes payload decode.
- `rust/runtime/src/cellular/transport/mod.rs:46-58`: feature-gated transport module registration.
- `rust/runtime/src/cellular/transport/mod.rs:69-133`: finite `CellMessage` and fixed terminal partition handlers. Streaming must not be added to this add-only dataset/result vocabulary.
- `rust/runtime/src/cellular/transport/velo_transport.rs:291-588`: controller handler registration and bounded receive channel.
- `rust/runtime/src/cellular/transport/velo_transport.rs:588-915`: cell connection/send implementation.
- `rust/runtime/src/engine/cellular_controller.rs:19-60`: controller dependencies and current cellular orchestration owner.
- `rust/runtime/src/engine/cellular_cell.rs:65-180`: cell bootstrap and controller-coordinate authority.
- `rust/runtime/src/engine/artifact_shipping.rs:1046-1270`: bounded streaming compression/write lifecycle reusable for immutable result bytes only.
- `rust/runtime/src/clock/runtime_clock.rs:11-19`: the only scheduling-time seam.

---

### Task C1: Strict Authenticated Streaming DTOs

**Dependencies:** Non-cellular Tasks 1D, 5A, 6D, P2, P3, and P4.

**Files:**
- Create: `rust/runtime/src/cellular/streaming_protocol.rs`
- Modify: `rust/runtime/src/cellular/mod.rs:12-21`
- Modify: `rust/runtime/src/engine/cellular_registration.rs:37-205`
- Modify: `rust/runtime/src/engine/cellular_registration.rs:229-370`
- Modify: `rust/runtime/src/engine/cellular_registration.rs:394-570`
- Modify: `rust/runtime/src/cellular/transport/velo_transport.rs:67-116`
- Modify: `rust/runtime/src/cellular/transport/velo_transport.rs:198-240`
- Test in: `rust/runtime/src/cellular/streaming_protocol.rs`
- Test in: `rust/runtime/src/engine/cellular_registration.rs`

**Produces:**

```rust
pub const STREAMING_CELLULAR_PROTOCOL_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamingCellularLimits {
    pub max_frame_bytes: usize,
    pub max_payload_bytes: usize,
    pub max_content_items: usize,
    pub max_content_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContentLeaseDescriptor {
    pub content_id: [u8; 32],
    pub byte_length: u64,
    pub digest: [u8; 32],
}

#[derive(Debug, Eq, PartialEq, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedActionContent {
    pub schema: DatasetActionSchema,
    pub canonical_request: Vec<u8>,
    pub content_leases: Vec<ContentLeaseDescriptor>,
    pub item_count: u64,
    pub byte_length: u64,
    pub digest: [u8; 32],
}

#[derive(Debug, Eq, PartialEq, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrepareAction {
    pub version: u16,
    pub plan_digest: [u8; 32],
    pub route_id: u32,
    pub destination_cell: u32,
    pub action_id: StableActionId,
    pub attempt_id: ActionAttemptId,
    pub global_sequence: GlobalSequence,
    pub ownership_epoch: SessionOwnershipEpoch,
    pub prior_session_state_version: SessionStateVersion,
    pub content: PreparedActionContent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseAction {
    pub version: u16,
    pub plan_digest: [u8; 32],
    pub route_id: u32,
    pub action_id: StableActionId,
    pub global_sequence: GlobalSequence,
    pub ownership_epoch: SessionOwnershipEpoch,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CellPlacementEvent {
    Prepared(PlacementPreparedReceipt),
    Released(PlacementReleasedReceipt),
    Action(ActionExecutionEvent),
    Failed(PlacementFailureReceipt),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControllerStreamingPurpose {
    PrepareAction,
    ReleaseAction,
}

#[derive(Debug, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ControllerAuthenticatedFrame {
    version: u16,
    destination: CellularRole,
    controller_session_nonce: [u8; 32],
    sequence: u64,
    peer_info: Vec<u8>,
    payload: Vec<u8>,
    signature: Vec<u8>,
}

pub(crate) struct BudgetOwnedFrame { bytes: Bytes, lease: BudgetLease }
pub(crate) struct AuthenticatedStreamingPayload { bytes: Bytes, lease: BudgetLease }
pub(crate) struct BudgetOwnedPrepareAction { action: PrepareAction, lease: BudgetLease }
pub(crate) struct FrameBudgetReservation { lease: BudgetLease, max_frame_bytes: usize }

```

The registration seam adds controller-signed outbound frames and worker-signed inbound frames without weakening the existing fixed-class rejection boundary:

```rust
impl CellSecurityContext {
    pub(crate) fn seal_streaming_to_cell<T: serde::Serialize>(
        &self,
        purpose: ControllerStreamingPurpose,
        destination: CellularRole,
        peer: &velo::PeerInfo,
        payload: &T,
        reservation: FrameBudgetReservation,
    ) -> anyhow::Result<BudgetOwnedFrame>;

    pub(crate) fn authenticate_streaming_from_controller(
        &self,
        purpose: ControllerStreamingPurpose,
        expected_destination: CellularRole,
        peer: &velo::PeerInfo,
        frame: BudgetOwnedFrame,
        limits: StreamingCellularLimits,
    ) -> Result<AuthenticatedStreamingPayload, AdmissionRejection>;

    pub(crate) fn decode_prepare_action(
        &self,
        payload: AuthenticatedStreamingPayload,
        limits: StreamingCellularLimits,
    ) -> Result<BudgetOwnedPrepareAction, AdmissionRejection>;
}
```

Before calling the synchronous sealer, the async transport acquires one item plus exactly `max_frame_bytes` and moves it into `FrameBudgetReservation`; the sealer rejects overflow and shrinks the lease to actual encoded bytes. It never allocates a frame without a reservation. `BudgetOwnedFrame`, `AuthenticatedStreamingPayload`, and `BudgetOwnedPrepareAction` are non-cloneable and carry that same permit through authentication, typed decode, and queue admission. The transport checks `max_frame_bytes` before outer-frame decoding. `decode_prepare_action` uses a custom Serde visitor with a bounded sequence seed: it refuses payload length and content-item counts before reserving their buffers, charges canonical request bytes while reading, checks every declared length with checked arithmetic, and verifies exact item/byte counts and digest. It never uses `DeserializeOwned` for an action payload.

Add `AdmissionPurpose::StreamingPlacementEvent` and
`AdmissionPurpose::StreamingResultPartition` for worker-signed inbound frames.
Add a separate fixed controller-purpose send-sequence array and worker-local
fixed replay windows for `ControllerAuthenticatedFrame`; do not overload
`AuthenticatedFrame.role`, because `CellularRole` intentionally has no controller
variant. Add `controller_session_nonce: [u8; 32]` to the attested
`RegisterReplyPayload`; `verify_reply` installs that exact nonce into the worker
security context before any streaming handler opens. An unset or changed nonce
fails closed, and a controller restart requires fresh registration.

- [ ] **Step 1: Add the strict DTO RED test**

Add this unit test to `streaming_protocol.rs`:

```rust
#[test]
fn prepare_action_rejects_unknown_fields() {
    let encoded = rmp_serde::to_vec_named(&serde_json::json!({
        "version": STREAMING_CELLULAR_PROTOCOL_VERSION,
        "plan_digest": [0_u8; 32],
        "route_id": 7,
        "destination_cell": 2,
        "action_id": StableActionId::from_bytes([1; 32]),
        "attempt_id": ActionAttemptId::from_bytes([2; 32]),
        "global_sequence": 11,
        "ownership_epoch": 3,
        "prior_session_state_version": 5,
        "content": PreparedActionContent::test_fixture(),
        "credential": "must-not-be-accepted"
    }))
    .expect("test fixture encodes");

    let error = StreamingAuthorityFixture::default()
        .decode_authenticated_prepare_payload(encoded)
        .expect_err("unknown fields must fail closed");
    assert!(error.to_string().contains("unknown field"));
}
```

- [ ] **Step 2: Verify DTO RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,cellular --lib streaming_
```

Expected: FAIL because `cellular::streaming_protocol` and `PrepareAction` do not exist.

- [ ] **Step 3: Add the authenticated-direction RED test**

In `cellular_registration.rs`, construct the existing controller/worker test authorities and assert: correct controller frame opens; wrong destination, purpose, peer, signature, session nonce, replayed sequence, and mutated payload each return `AdmissionRejection`; payload decode is not attempted on authentication failure.

```rust
#[tokio::test(flavor = "current_thread")]
async fn streaming_frame_binds_destination_purpose_and_payload_before_decode() {
    let fixture = StreamingAuthorityFixture::new(CellularRole::Cell(2));
    let frame = fixture.controller.seal_streaming_to_cell(
        ControllerStreamingPurpose::PrepareAction,
        CellularRole::Cell(2),
        &fixture.cell_peer,
        &fixture.prepare,
        fixture.frame_reservation().await,
    ).expect("controller seals test frame");

    let authenticated = fixture.cell.authenticate_streaming_from_controller(
        ControllerStreamingPurpose::PrepareAction,
        CellularRole::Cell(2),
        &fixture.cell_peer,
        frame,
        fixture.limits,
    ).expect("bound frame authenticates once");
    let opened = fixture.cell.decode_prepare_action(authenticated, fixture.limits)
        .expect("bounded typed payload decodes");
    assert_eq!(opened.action(), &fixture.prepare);
}

#[test]
fn oversized_frame_and_nested_content_are_rejected_before_body_allocation() {
    let fixture = StreamingAuthorityFixture::with_limits(StreamingCellularLimits {
        max_frame_bytes: 1024,
        max_payload_bytes: 512,
        max_content_items: 2,
        max_content_bytes: 256,
    });
    assert_eq!(fixture.open_raw(&vec![0_u8; 1025]).unwrap_err(), AdmissionRejection::FrameTooLarge);
    let nested = fixture.authenticated_prepare_with_declared_items(3, 384);
    assert_eq!(fixture.open_raw(&nested).unwrap_err(), AdmissionRejection::ContentLimitExceeded);
    assert_eq!(fixture.body_allocated_bytes(), 0);
}
```

- [ ] **Step 4: Implement the minimal authenticated DTO boundary**

Use canonical MessagePack bytes and a domain-separated transcript containing protocol version, run nonce, signer class, destination role, purpose, session nonce, sequence, peer bytes, payload length, and BLAKE3 payload digest. Keep replay windows fixed-size per route/purpose. Authenticate the bounded raw payload before the purpose-specific bounded visitor decodes it. Do not log payloads or introduce a generic `Any`/`DeserializeOwned` envelope.

- [ ] **Step 5: Verify GREEN and commit**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,cellular --lib streaming_
git add rust/runtime/src/cellular/mod.rs rust/runtime/src/cellular/streaming_protocol.rs rust/runtime/src/cellular/transport/velo_transport.rs rust/runtime/src/engine/cellular_registration.rs
git commit -m "feat(cellular): authenticate strict streaming frames"
```

---

### Task C2: Bounded Multiplexed Placement Transfer

**Dependencies:** C1.

**Files:**
- Create: `rust/runtime/src/cellular/streaming_transport.rs`
- Modify: `rust/runtime/src/cellular/mod.rs`
- Modify: `rust/runtime/src/cellular/transport/mod.rs:46-58`
- Modify: `rust/runtime/src/cellular/transport/velo_transport.rs:291-588`
- Modify: `rust/runtime/src/cellular/transport/velo_transport.rs:588-915`
- Test in: `rust/runtime/src/cellular/streaming_transport.rs`

**Produces:** one binding with separately borrowable handles:

```rust
pub(crate) struct PreparedCellularPlacementBinding {
    pub submitter: CellularPlacementSubmitter,
    pub driver: CellularPlacementDriver,
    pub control: CellularPlacementControl,
    pub cell_endpoint: CellularExecutionEndpoint,
}

pub(crate) fn prepare_cellular_placement_binding(
    routes: Box<[PreparedCellRoute]>,
    budget: StreamingResourceBudget,
    limits: CellularTransferLimits,
) -> Result<PreparedCellularPlacementBinding, CellularStreamingError>;

impl StreamingPlacementSubmitter for CellularPlacementSubmitter {
    async fn prepare(&mut self, decision: PlacementDecision, action: OrderedDatasetAction)
        -> Result<PlacementHandle, PlacementError> { self.prepare_bounded(decision, action).await }
    async fn release(&mut self, handle: PlacementHandleId) -> Result<(), PlacementError> {
        self.release_authenticated(handle).await
    }
}

impl StreamingPlacementDriver for CellularPlacementDriver {
    async fn next_event(&mut self) -> Result<PlacementEvent, PlacementError> {
        self.receive_ordered_event().await
    }
    async fn drain(&mut self) -> Result<(), PlacementError> { self.join_binding_owners().await }
}
```

- [ ] **Step 1: Add the bounded-window RED test**

In the same test module, add `identical_chunk_retransmit_is_idempotent`, `conflicting_sequence_fails_route`, `cancel_pending_wakes_driver`, and `drain_joins_fixed_binding_tasks` before the RED run. Keep the protocol-independent cases in one parameterized harness.

```rust
#[tokio::test(flavor = "current_thread")]
async fn full_route_window_backpressures_without_spawning_per_action_drivers() {
    let fixture = TransferFixture::new(CellularTransferLimits { max_items: 2, max_bytes: 256 });
    let (mut submitter, cell, diagnostics) = fixture.into_parts();
    submitter.prepare(decision(1), action(1, 128)).await.unwrap();
    submitter.prepare(decision(2), action(2, 128)).await.unwrap();
    let mut third = Box::pin(submitter.prepare(decision(3), action(3, 128)));
    assert!(futures::poll!(&mut third).is_pending());
    assert_eq!(diagnostics.driver_count(), 1);

    cell.ack_prepared(1).await.unwrap();
    assert!(third.await.is_ok());
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,cellular --lib cellular::streaming_transport::tests::full_route_window_backpressures_without_spawning_per_action_drivers -- --exact
```

- [ ] **Step 3: Implement one multiplexed binding**

Register dedicated streaming Velo handlers, but keep DTO queues in `streaming_transport.rs`. Use bounded item/byte permits for unacknowledged prepare commands and returned events. Keep one controller driver task and one cell endpoint task per binding; action state lives in an indexed slab keyed by `PlacementHandleId`. Identical retransmission is idempotent, gaps/conflicting duplicate bytes fail the route, and `StreamingPlacementControl` wakes a pending `next_event`. Do not add streaming variants to `CellMessage`, call `dataset_velo`, or clone action bodies into a lifetime history.

- [ ] **Step 4: Verify shutdown/retransmit behavior**

Confirm the Step-1 shutdown/retransmit cases exercise the completed binding without duplicating cases per route.

- [ ] **Step 5: Verify GREEN and commit**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,cellular --lib cellular::streaming_transport::tests
git add rust/runtime/src/cellular/mod.rs rust/runtime/src/cellular/streaming_transport.rs rust/runtime/src/cellular/transport/mod.rs rust/runtime/src/cellular/transport/velo_transport.rs
git commit -m "feat(cellular): multiplex bounded streaming transfer"
```

---

### Task C3: Sticky Placement and Prepare/Release No-Early-Issue

**Dependencies:** C2 and the local `StreamingPlacementPolicy`/`ActiveExecutionSet` implementation.

**Files:**
- Modify: `rust/runtime/src/streaming/placement.rs` at the `StreamingPlacementPolicy` implementation landed by the prerequisite
- Create: `rust/runtime/src/engine/cellular_streaming_controller.rs`
- Create: `rust/runtime/src/engine/cellular_streaming_cell.rs`
- Modify: `rust/runtime/src/engine/mod.rs`
- Modify: `rust/runtime/src/engine/cellular_controller.rs:707-900`
- Modify: `rust/runtime/src/engine/cellular_cell.rs:713-1010`
- Test: `rust/runtime/tests/streaming_cellular_placement.rs`

**Produces:**

```rust
pub struct StickySessionPlacement {
    plan_digest: [u8; 32],
    cell_count: u32,
    routes: BTreeMap<StableSessionKey, BudgetOwnedSessionRoute>,
    route_budget: StreamingResourceBudget,
}

pub struct BudgetOwnedSessionRoute { pub route: SessionRoute, pub lease: BudgetLease }

impl StreamingPlacementPolicy for StickySessionPlacement {
    fn place(&mut self, action: &OrderedDatasetAction)
        -> Result<PlacementDecision, PlacementError> { self.place_sticky(action) }
    fn observe_session_terminal(
        &mut self,
        session: StableSessionKey,
        ownership_epoch: SessionOwnershipEpoch,
        causal_frontier: &SessionCausalFrontier,
    ) -> Result<(), PlacementError> {
        self.retire_route_if_fenced(session, ownership_epoch, causal_frontier)
    }
}

pub(crate) async fn release_at_controller_target(
    clock: std::rc::Rc<dyn Clock>,
    target_ns: i64,
    submitter: &mut CellularPlacementSubmitter,
    handle: PlacementHandleId,
) -> Result<(), PlacementError> {
    let delay = target_ns.checked_sub(clock.now_ns())
        .ok_or_else(|| PlacementError::failure(PlacementFailureCode::TargetOverflow))?;
    clock.sleep(delay).await;
    submitter.release(handle).await
}

impl CellularExecutionEndpoint {
    pub(crate) async fn accept_prepare(&mut self, command: BudgetOwnedPrepareAction)
        -> Result<PlacementPreparedReceipt, CellularStreamingError> { self.stage_without_issue(command).await }
    pub(crate) async fn accept_release(&mut self, command: ReleaseAction)
        -> Result<PlacementReleasedReceipt, CellularStreamingError> { self.issue_if_fenced(command).await }
}
```

- [ ] **Step 1: Add the SimClock no-early-issue RED test**

Also add `same_session_routes_stickily`, `different_sessions_distribute_deterministically`, `prepared_digest_mismatch_never_releases`, and `terminal_event_returns_to_controller_before_dependent_release` before the RED run.

```rust
#[tokio::test(flavor = "current_thread")]
async fn prepare_never_issues_and_release_uses_only_controller_clock() {
    let fixture = PlacementFixture::new(/* controller_now */ 0, /* cell_now */ 9_000_000_000);
    let handle = fixture.controller.prepare(fixture.action_at(500)).await.unwrap();
    fixture.pump_prepare().await.unwrap();
    assert_eq!(fixture.cell.issued_count(), 0);

    let release = release_at_controller_target(fixture.controller_clock(), 500, &mut fixture.submitter, handle);
    tokio::pin!(release);
    assert!(futures::poll!(&mut release).is_pending());
    fixture.advance_controller_to(499);
    assert_eq!(fixture.cell.issued_count(), 0);
    fixture.advance_controller_to(500);
    release.await.unwrap();
    fixture.pump_release().await.unwrap();
    assert_eq!(fixture.cell.issued_count(), 1);
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,cellular --test streaming_cellular_placement prepare_never_issues_and_release_uses_only_controller_clock -- --exact
```

- [ ] **Step 3: Implement sticky routing and fenced cell admission**

Hash `(plan_digest, StableSessionKey)` to the initial cell once. Charge every route entry to an item/byte budget; a terminal session receipt removes its route after the causal frontier and checkpoint participant no longer reference it. `accept_prepare` consumes and stores the non-cloneable `BudgetOwnedPrepareAction` under its original permit through release and terminal/cancel acknowledgement; it cannot call the endpoint action submitter. `release_at_controller_target` uses only `Clock::now_ns`/`Clock::sleep`; `accept_release` validates the exact tuple then submits once. An absent, stale, duplicate-conflicting, or wrong-route release returns a typed failure and leaves the action fenced. Add `million_sequential_closed_sessions_reclaim_routes_with_constant_high_water`.

- [ ] **Step 4: Verify sticky-session and event-return tests**

Add `four_partitions_keep_one_session_on_one_cell`, `stale_release_cannot_issue`, and `cell_action_event_flows_through_active_execution_set`. Assert that controller session state changes only after the ordered cell event is accepted.

- [ ] **Step 5: Verify GREEN and commit**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,cellular --test streaming_cellular_placement
git add rust/runtime/src/streaming/placement.rs rust/runtime/src/engine/mod.rs rust/runtime/src/engine/cellular_streaming_controller.rs rust/runtime/src/engine/cellular_streaming_cell.rs rust/runtime/src/engine/cellular_controller.rs rust/runtime/src/engine/cellular_cell.rs rust/runtime/tests/streaming_cellular_placement.rs
git commit -m "feat(cellular): fence streaming issue until controller release"
```

---

### Task C4: Ownership Epochs and Crash-Safe Route Migration

**Dependencies:** C3 and the global streaming checkpoint coordinator.

**Files:**
- Modify: `rust/runtime/src/streaming/placement.rs` at `StickySessionPlacement` from C3
- Modify: `rust/runtime/src/engine/cellular_streaming_controller.rs`
- Modify: `rust/runtime/src/engine/cellular_streaming_cell.rs`
- Test: `rust/runtime/tests/streaming_cellular_migration.rs`

**Produces:**

```rust
pub enum SessionRouteState {
    Owned(SessionRoute),
    Fencing { old: SessionRoute, through: GlobalSequence },
    Prepared { old: SessionRoute, new: SessionRoute, through: GlobalSequence },
}

impl StickySessionPlacement {
    pub async fn migrate(
        &mut self,
        session: StableSessionKey,
        destination: u32,
        checkpoint: &mut StreamingCheckpointCoordinator,
    ) -> Result<SessionOwnershipEpoch, PlacementError> {
        self.migrate_through_checkpoint(session, destination, checkpoint).await
    }
}
```

- [ ] **Step 1: Add the crash-table RED test**

Also add `late_old_epoch_receipt_is_rejected`, `new_cell_prepare_is_not_authority_before_commit`, `migration_pending_fragments_obey_budget`, identical-retry, and conflicting-receipt cases before the RED run.

```rust
#[tokio::test(flavor = "current_thread")]
async fn restore_uses_only_last_committed_route_epoch() {
    for crash in MigrationCrashPoint::ALL {
        let mut fixture = MigrationFixture::restored_from_generation_one().await;
        fixture.migrate_with_crash(CellId::new(1), crash).await;
        let restored = fixture.restart_controller().await.unwrap();
        let expected = if crash.is_after_route_generation_commit() { 1 } else { 0 };
        assert_eq!(restored.owner_cell(), expected, "crash point {crash:?}");
        assert_eq!(restored.active_owner_count(), 1);
    }
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,cellular --test streaming_cellular_migration restore_uses_only_last_committed_route_epoch -- --exact
```

- [ ] **Step 3: Implement the exact fenced transaction**

Implement: freeze sequence `N`; stop old-epoch prepares; drain or explicitly cancel all old-cell actions `<= N`; commit controller session state, terminal receipts, and old fence; prepare immutable content on the new cell without releasing it; commit the incremented route epoch/owner and next sequence in the global generation; then release actions `> N`. New fragments remain bounded at the controller during handoff. No canonical state is sent cell-to-cell.

- [ ] **Step 4: Verify stale-event and bounded-pending tests**

Confirm the Step-1 stale-event, authority, retry/conflict, and bounded-pending cases pass.

- [ ] **Step 5: Verify GREEN and commit**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,cellular --test streaming_cellular_migration
git add rust/runtime/src/streaming/placement.rs rust/runtime/src/engine/cellular_streaming_controller.rs rust/runtime/src/engine/cellular_streaming_cell.rs rust/runtime/tests/streaming_cellular_migration.rs
git commit -m "feat(cellular): migrate streaming routes by committed epoch"
```

---

### Task C5: Cell Result Partitions and Content-Addressed Retransmission

**Dependencies:** C2, C4, and streaming result epochs/indexes.

**Files:**
- Create: `rust/runtime/src/cellular/streaming_results.rs`
- Modify: `rust/runtime/src/cellular/mod.rs:12-21`
- Modify: `rust/runtime/src/engine/cellular_streaming_cell.rs`
- Modify: `rust/runtime/src/engine/artifact_shipping.rs:1046-1270`
- Test: `rust/runtime/tests/streaming_cellular_results.rs`

**Produces:**

```rust
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CellResultPartitionReceipt {
    pub plan_digest: [u8; 32],
    pub generation: CheckpointGeneration,
    pub cell_id: u32,
    pub worker_id: u32,
    pub projection_id: ResultProjectionId,
    pub first_sequence: GlobalSequence,
    pub last_sequence: GlobalSequence,
    pub item_count: u64,
    pub byte_length: u64,
    pub membership_root: ContentDigest,
    pub payload_digest: [u8; 32],
}

pub(crate) async fn prepare_cell_result_partitions(
    epoch: WorkerResultEpoch,
    writer: &mut dyn BoundedImmutableObjectWriter,
) -> Result<Vec<CellResultPartitionReceipt>, CellularResultError>;

#[async_trait::async_trait(?Send)]
pub(crate) trait BoundedImmutableObjectWriter {
    async fn write(
        &mut self,
        descriptor: &ResultSegmentDescriptor,
        bytes: BudgetedBlockingOutput<Vec<u8>>,
    ) -> Result<(), CellularResultError>;
}

#[async_trait::async_trait(?Send)]
pub(crate) trait CellResultPayloadFetcher {
    async fn fetch_verified_partition(
        &mut self,
        receipt: &CellResultPartitionReceipt,
        budget: ResultIndexReadBudget,
    ) -> Result<ResultPartition, CellularResultError>;
}
```

- [ ] **Step 1: Add the retransmission RED test**

Also add `receipt_binds_cell_range_count_length_and_digest`, `conflicting_retry_is_rejected`, `cell_flush_is_not_globally_visible`, and one over-byte-limit backpressure case before the RED run.

```rust
#[tokio::test(flavor = "current_thread")]
async fn restart_retransmits_same_content_addressed_partition() {
    let epoch = WorkerResultEpoch::fixture(7, 100..=199);
    let checkpoint = epoch.checkpoint_view_for_test().await.unwrap();
    let first = ResultFixture::prepare(epoch).await.unwrap();
    let restored_epoch = WorkerResultEpoch::restore_for_test(checkpoint).await.unwrap();
    let retry = ResultFixture::restart().prepare(restored_epoch).await.unwrap();
    assert_eq!(first.receipts(), retry.receipts());
    assert_eq!(first.object_digests(), retry.object_digests());
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,cellular --test streaming_cellular_results restart_retransmits_same_content_addressed_partition -- --exact
```

- [ ] **Step 3: Implement bounded cell partition preparation**

Rotate worker-local epochs only at the barrier, sort canonical facts, write immutable bytes through the existing bounded artifact writer mechanics, and send only authenticated receipts on the placement/result route. Do not use `RecordsShardPartition`, `ColumnStorePartition`, `CellMessage`, dataset broadcast, or a resident `Vec` of all historical descriptors. An uncommitted cell-local flush is staging only.

- [ ] **Step 4: Verify corruption/membership tests**

Confirm the Step-1 digest, conflict, visibility, and backpressure cases pass.

- [ ] **Step 5: Verify GREEN and commit**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,cellular --test streaming_cellular_results
git add rust/runtime/src/cellular/mod.rs rust/runtime/src/cellular/streaming_results.rs rust/runtime/src/engine/cellular_streaming_cell.rs rust/runtime/src/engine/artifact_shipping.rs rust/runtime/tests/streaming_cellular_results.rs
git commit -m "feat(cellular): prepare content-addressed cell result partitions"
```

---

### Task C6: Controller-Last Checkpoint and Result Convergence

**Dependencies:** C5 and checkpoint/result Task 6D with `PreparedReportCommit` ordering.

**Files:**
- Create: `rust/runtime/src/engine/cellular_streaming_convergence.rs`
- Modify: `rust/runtime/src/engine/mod.rs`
- Modify: `rust/runtime/src/engine/cellular_streaming_controller.rs`
- Modify: `rust/runtime/src/engine/cellular_controller.rs:707-900`
- Modify: `rust/runtime/src/engine/cellular_controller.rs:1419-1605`
- Modify: `rust/runtime/src/engine/coordinator.rs:483-540`
- Test: `rust/runtime/tests/streaming_cellular_convergence.rs`

**Produces:**

```rust
pub(crate) struct CellularBarrierPlan {
    pub generation: CheckpointGeneration,
    pub cut: CheckpointCut,
    pub expected: Box<[ExpectedCellResultPartition]>,
}

pub(crate) async fn commit_cellular_generation(
    plan: CellularBarrierPlan,
    receipts: &mut dyn CellularResultReceiptStream,
    payloads: &mut dyn CellResultPayloadFetcher,
    backend: &dyn StreamingCheckpointBackend,
    participants: &mut CheckpointParticipantSet,
) -> Result<CommittedCheckpointGeneration, CellularConvergenceError>;

#[async_trait::async_trait(?Send)]
pub(crate) trait CellularResultReceiptStream {
    async fn next_receipt(
        &mut self,
    ) -> Result<Option<CellResultPartitionReceipt>, CellularConvergenceError>;
}

pub(crate) struct CheckpointParticipantSet<'a> {
    pub participants: &'a mut [Box<dyn StreamingCheckpointParticipant>],
    pub frozen_ids: &'a [CheckpointParticipantId],
}
```

- [ ] **Step 1: Add the exact-set/CAS RED test**

Also add `cell_restart_retransmit_converges_once`, `controller_restart_ignores_uncommitted_cell_flush`, `gap_or_overlap_prevents_global_commit`, and `final_generation_precedes_compaction_and_report_commit` before the RED run.

```rust
#[tokio::test(flavor = "current_thread")]
async fn controller_commits_only_after_exact_partition_set_is_verified() {
    let mut fixture = ConvergenceFixture::three_cells();
    fixture.deliver_cell(0).await;
    fixture.deliver_cell(2).await;
    assert_eq!(fixture.backend().commit_count(), 0);

    fixture.deliver_cell(1).await;
    let committed = fixture.commit().await.unwrap();
    assert_eq!(fixture.backend().commit_count(), 1);
    assert_eq!(committed.result_root(), fixture.expected_root());
}
```

- [ ] **Step 2: Verify RED**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,cellular --test streaming_cellular_convergence controller_commits_only_after_exact_partition_set_is_verified -- --exact
```

- [ ] **Step 3: Implement controller-last publication**

At the barrier, freeze the expected `(cell, worker, projection, sequence range, membership)` set. Receive receipts through the bounded authenticated stream and validate exact disjoint coverage. For each receipt, `CellResultPayloadFetcher` streams the cell-staged immutable object under the read budget, verifies schema/ownership/count/length/digest/membership, and returns a move-only budget-owned `ResultPartition`. Only after the exact set is present and verified does the controller call `begin_generation`, move those partitions through `stage_results(Vec<ResultPartition>)`, stage every stable participant, and commit the global generation. It then calls `checkpoint_committed` in frozen order. Missing payloads/cells, corruption, gaps, overlap, conflicting duplicates, topology mismatch, or stale CAS leave the previous generation authoritative. On controller restart, ignore unreachable cell staging and restore only the last committed global root.

- [ ] **Step 4: Verify restart and final report-order tests**

Confirm the Step-1 restart/convergence cases pass. The final-order test must observe: final checkpoint CAS; leased compaction; durable report rename; synchronous `PreparedReportCommit`; retention-lease release.

- [ ] **Step 5: Verify runtime GREEN**

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,cellular --test streaming_cellular_convergence
```

- [ ] **Step 6: Graham review, behavior review, and commit**

Review authentication ordering, boundedness, all cancellation joins, `Clock` use, logical membership, final-generation/report ordering, and absence of dataset-broadcast/per-action-driver reuse. Then commit only this slice:

```bash
git add rust/runtime/src/engine/mod.rs rust/runtime/src/engine/cellular_streaming_convergence.rs rust/runtime/src/engine/cellular_streaming_controller.rs rust/runtime/src/engine/cellular_controller.rs rust/runtime/src/engine/coordinator.rs rust/runtime/tests/streaming_cellular_convergence.rs
git commit -m "feat(cellular): commit global streaming results last"
```

## Completion Gate

Run after C1-C6 are integrated:

```bash
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo fmt --check
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo clippy -p aiperf-runtime --all-targets --features streaming,cellular -- -D warnings
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,cellular --lib cellular::streaming_protocol::tests
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,cellular --lib engine::cellular_registration::tests
CARGO_TARGET_DIR=/mnt/4tb/aiperf-streaming-target cargo test -p aiperf-runtime --features streaming,cellular --test streaming_cellular_placement --test streaming_cellular_migration --test streaming_cellular_results --test streaming_cellular_convergence
```

Completion requires fresh evidence that:

- no `PrepareAction` path calls an endpoint submitter;
- skewing the cell clock cannot move issue earlier;
- each prepared binding owns a fixed task/driver count as action count grows;
- late old-epoch events cannot mutate session state or results;
- cell staging cannot become authoritative without the controller's global CAS;
- final generation commit precedes compaction, durable report persistence, and report-commit lease release; and
- `rg -n "DatasetSubscribe|dataset_velo|Unbounded|tokio::spawn" rust/runtime/src/cellular/streaming_* rust/runtime/src/engine/cellular_streaming_*` contains no unreviewed broadcast, unbounded, or detached-task path.
