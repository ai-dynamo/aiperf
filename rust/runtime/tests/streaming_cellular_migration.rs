// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ownership epochs and crash-safe cellular route migration.
//!
//! Every test drives the real seven-step transaction against a real
//! [`StreamingCheckpointCoordinator`] over an in-memory checkpoint store. A
//! "crash" is modelled the way the store makes it observable: the controller is
//! dropped at an exact step and a fresh placement is restored from whatever the
//! last *committed* generation happens to be. Nothing is faked between the
//! placement state machine and the CAS, so the crash matrix is evidence about
//! the product path rather than about a stub.

#![cfg(all(feature = "streaming", feature = "cellular"))]

use std::cell::RefCell;
use std::pin::pin;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Wake, Waker};

use aiperf_runtime::cellular::streaming_migration::{
    CellularStreamingController, SessionRoutePlacementParticipant,
};
use aiperf_runtime::cellular::streaming_placement::{
    ActiveExecutionSet, PlacementRouteCharge, ROUTE_ENTRY_BYTES, ReleaseFence, SessionRouteState,
    StickySessionPlacement, assign_cell,
};
use aiperf_runtime::streaming::budget::{BudgetLimits, StreamingResourceBudget};
use aiperf_runtime::streaming::checkpoint::{
    CheckpointParticipantId, StreamRunIdentity, StreamingCheckpointParticipant,
};
use aiperf_runtime::streaming::checkpoint_backend::{
    LeasedCheckpointGenerationView, StreamingCheckpointBackend,
};
use aiperf_runtime::streaming::checkpoint_coordinator::{
    StreamingCheckpointCoordinator, committed_descriptor,
};
use aiperf_runtime::streaming::checkpoints::memory::MemoryCheckpointBackend;
use aiperf_runtime::streaming::failure::PlacementFailureCode;
use aiperf_runtime::streaming::identity::{
    ContentDigest, GlobalSequence, SessionCausalFrontier, SessionOwnershipEpoch, StableActionId,
    StableSessionKey,
};

#[path = "support/streaming_checkpoint_coordinator.rs"]
mod support;

use support::{
    FakeIssueReporter, PARTICIPANT_ID, PLAN_DIGEST, backend_limits, barrier_at, expectations,
    run_id,
};

/// Two cells: an origin and exactly one destination to migrate to.
const CELL_COUNT: u32 = 2;
/// Freeze point for every migration in this suite.
const FREEZE_THROUGH: u64 = 4;

/// Every observable point at which the controller may die during one migration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MigrationCrashPoint {
    BeforeFreeze,
    AfterFreeze,
    AfterOldDrain,
    AfterFenceCommit,
    AfterNewPrepare,
    AfterRouteGenerationCommit,
    AfterRelease,
}

impl MigrationCrashPoint {
    const ALL: [Self; 7] = [
        Self::BeforeFreeze,
        Self::AfterFreeze,
        Self::AfterOldDrain,
        Self::AfterFenceCommit,
        Self::AfterNewPrepare,
        Self::AfterRouteGenerationCommit,
        Self::AfterRelease,
    ];

    /// True for crash points at or after the route-generation CAS.
    const fn is_after_route_generation_commit(self) -> bool {
        matches!(self, Self::AfterRouteGenerationCommit | Self::AfterRelease)
    }
}

fn action(tag: u8) -> StableActionId {
    let mut bytes = [0u8; 32];
    bytes[31] = tag;
    StableActionId::from_bytes(bytes)
}

fn frontier(through: u64) -> SessionCausalFrontier {
    SessionCausalFrontier {
        through_sequence: GlobalSequence::new(through),
        event_time: None,
        digest: ContentDigest::from_bytes([0u8; 32]),
    }
}

fn budget(max_items: usize) -> StreamingResourceBudget {
    StreamingResourceBudget::new(BudgetLimits {
        max_items,
        max_bytes: 1 << 20,
    })
    .expect("authored budget limits are valid")
}

/// The first session key that the sticky hash pins to cell 0.
///
/// Searched rather than hard-coded so a domain-separator bump reassigns the
/// fixture instead of silently turning the migration into a no-op.
fn session_on_origin_cell() -> StableSessionKey {
    for tag in 0..=u8::MAX {
        let mut bytes = [0u8; 32];
        bytes[0] = tag;
        let candidate = StableSessionKey::from_bytes(bytes);
        if assign_cell(&PLAN_DIGEST, candidate, CELL_COUNT) == 0 {
            return candidate;
        }
    }
    panic!("no session key assigns to cell 0 under the fixture plan digest");
}

/// A waker that records whether it was signalled, so a manual poll loop can
/// distinguish "still parked on capacity" from "was woken and should re-poll".
struct FlagWaker(Arc<AtomicBool>);

impl Wake for FlagWaker {
    fn wake(self: Arc<Self>) {
        self.0.store(true, Ordering::SeqCst);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// One controller, its shared placement, and the store both commit through.
struct MigrationFixture {
    controller: CellularStreamingController,
    placement: Rc<RefCell<StickySessionPlacement>>,
    backend: MemoryCheckpointBackend,
    run: StreamRunIdentity,
    route_budget: StreamingResourceBudget,
}

impl MigrationFixture {
    /// Build a fixture whose route budget admits `route_items` route charges.
    fn new(route_items: usize) -> Self {
        let run = run_id(1);
        let backend = MemoryCheckpointBackend::new(backend_limits()).expect("memory backend");
        let route_budget = budget(route_items);
        let placement = Rc::new(RefCell::new(
            StickySessionPlacement::new(PLAN_DIGEST, CELL_COUNT, route_budget.clone())
                .expect("nonzero cell count"),
        ));
        let participant = SessionRoutePlacementParticipant::new(
            run,
            CheckpointParticipantId::new(PARTICIPANT_ID),
            Rc::clone(&placement),
            budget(64),
        );
        let (reporter, _control) = FakeIssueReporter::new(run);
        let coordinator = StreamingCheckpointCoordinator::new(
            run,
            Box::new(backend.clone()),
            expectations(run),
            vec![Box::new(participant)],
            Box::new(reporter),
            None,
        )
        .expect("valid coordinator over the fixture plan");
        let controller = CellularStreamingController::new(
            Rc::clone(&placement),
            coordinator,
            budget(16),
            PLAN_DIGEST,
        );
        Self {
            controller,
            placement,
            backend,
            run,
            route_budget,
        }
    }

    /// Install one route for `session` at sequence `sequence`.
    async fn install_route(&self, session: StableSessionKey, sequence: u64) {
        let mut admission = self.placement.borrow().admission();
        let reservation = admission
            .reserve_route(PlacementRouteCharge {
                session,
                items: 1,
                bytes: ROUTE_ENTRY_BYTES,
            })
            .await
            .expect("route capacity for the fixture session");
        let mut placement = self.placement.borrow_mut();
        placement
            .install_pending_reservation(reservation)
            .expect("no other pending reservation");
        placement
            .place_session(session, GlobalSequence::new(sequence))
            .expect("first placement installs the route");
    }

    /// Restore a fresh placement from the store's last committed generation.
    ///
    /// This is the crash: the live controller and its in-memory route map are
    /// discarded, and the only thing that survives is whatever the CAS made
    /// durable.
    async fn restore_from_last_committed(&self) -> StickySessionPlacement {
        let restored = StickySessionPlacement::new(PLAN_DIGEST, CELL_COUNT, budget(8))
            .expect("nonzero cell count");
        let shared = Rc::new(RefCell::new(restored));
        let mut participant = SessionRoutePlacementParticipant::new(
            self.run,
            CheckpointParticipantId::new(PARTICIPANT_ID),
            Rc::clone(&shared),
            budget(64),
        );
        let opened = self
            .backend
            .open_latest(&self.run, &expectations(self.run))
            .await
            .expect("open the committed head")
            .expect("a committed head exists");
        let state = match opened.view() {
            LeasedCheckpointGenerationView::CurrentV4(reader) => {
                let descriptor = committed_descriptor(
                    reader.generation(),
                    &CheckpointParticipantId::new(PARTICIPANT_ID),
                )
                .expect("route participant is in the committed plan")
                .clone();
                reader
                    .read_participant(&descriptor)
                    .await
                    .expect("read the verified route participant")
            }
            LeasedCheckpointGenerationView::LegacyV3ReadOnly(_) => {
                panic!("the memory backend writes current-v4 generations")
            }
        };
        participant
            .initialize(Some(state))
            .await
            .expect("restore the committed route set");
        drop(opened);
        drop(participant);
        Rc::try_unwrap(shared)
            .expect("the restored placement has no other owner")
            .into_inner()
    }
}

/// Drive one migration up to (and including) `crash` and return the fixture.
///
/// Barrier 1 is the baseline generation that records the pre-migration owner,
/// barrier 2 is the fence commit, and barrier 3 is the route-generation CAS.
async fn migrate_until(crash: MigrationCrashPoint, session: StableSessionKey) -> MigrationFixture {
    let mut fixture = MigrationFixture::new(4);
    fixture.install_route(session, 1).await;
    fixture
        .controller
        .commit_fence(barrier_at(1))
        .await
        .expect("baseline generation records the pre-migration owner");
    if crash == MigrationCrashPoint::BeforeFreeze {
        return fixture;
    }

    let through = GlobalSequence::new(FREEZE_THROUGH);
    fixture
        .controller
        .freeze_session(session, through)
        .expect("an owned session admits a fence");
    if crash == MigrationCrashPoint::AfterFreeze {
        return fixture;
    }

    fixture
        .controller
        .drain_old_cell(session, &[(0, action(1))])
        .expect("drain a fenced session");
    if crash == MigrationCrashPoint::AfterOldDrain {
        return fixture;
    }

    fixture
        .controller
        .commit_fence(barrier_at(2))
        .await
        .expect("fence generation commits the old owner and its receipts");
    if crash == MigrationCrashPoint::AfterFenceCommit {
        return fixture;
    }

    fixture
        .controller
        .stage_new_cell(session, 1, &[(action(9), GlobalSequence::new(9))])
        .await
        .expect("stage immutable content on the destination");
    if crash == MigrationCrashPoint::AfterNewPrepare {
        return fixture;
    }

    fixture
        .controller
        .commit_route_generation(session, barrier_at(3))
        .await
        .expect("the route generation CAS");
    if crash == MigrationCrashPoint::AfterRouteGenerationCommit {
        return fixture;
    }

    fixture
        .controller
        .promote_owner(session)
        .expect("settle the migration leases and release the hold");
    fixture
}

/// The plan-mandated crash matrix: a restart adopts the new owner if and only if
/// the route generation committed, and never sees two owners.
#[tokio::test(flavor = "current_thread")]
async fn restore_uses_only_last_committed_route_epoch() {
    let session = session_on_origin_cell();
    for crash in MigrationCrashPoint::ALL {
        let fixture = migrate_until(crash, session).await;
        let restored = fixture.restore_from_last_committed().await;
        let route = restored
            .route_for(session)
            .expect("the restored set retains the session");
        let expected_cell = u32::from(crash.is_after_route_generation_commit());
        assert_eq!(
            route.destination_cell, expected_cell,
            "crash at {crash:?} must restore cell {expected_cell}"
        );
        assert_eq!(
            route.ownership_epoch,
            SessionOwnershipEpoch::new(u64::from(crash.is_after_route_generation_commit())),
            "the epoch increment is visible only through the committed generation"
        );
        assert_eq!(
            restored.active_owner_count(),
            1,
            "a restored set names exactly one authoritative owner per session"
        );
        assert_eq!(
            restored.migration_lease_count(),
            0,
            "no migration lease is ever restored"
        );
    }
}

/// A post-migration event stamped with the fenced epoch touches nothing.
#[tokio::test(flavor = "current_thread")]
async fn late_old_epoch_receipt_is_rejected() {
    let session = session_on_origin_cell();
    let mut fixture = migrate_until(MigrationCrashPoint::AfterRelease, session).await;
    let staged_before = fixture.controller.active().staged_count();
    let issued_before = fixture.controller.active().issued_count();

    let refusal = fixture
        .controller
        .admit_placement_event(session, SessionOwnershipEpoch::new(0))
        .expect_err("the pre-migration epoch is fenced");
    assert_eq!(refusal.failure_code(), PlacementFailureCode::StaleOwnershipEpoch);
    assert_eq!(fixture.controller.stale_event_refusals(), 1);
    assert_eq!(fixture.controller.active().staged_count(), staged_before);
    assert_eq!(fixture.controller.active().issued_count(), issued_before);

    fixture
        .controller
        .admit_placement_event(session, SessionOwnershipEpoch::new(1))
        .expect("the committed epoch is current");
    assert_eq!(
        fixture.controller.stale_event_refusals(),
        1,
        "a current-epoch event is not counted as a refusal"
    );

    // The subtle case: a receipt for work committed at the fence is admitted on
    // its sequence, not on its epoch, so committed terminal work is never lost.
    fixture
        .controller
        .admit_fenced_terminal_receipt(session, GlobalSequence::new(1))
        .expect("a receipt at or below the fence is still admissible");
}

/// Content staged on the destination is not authority until the CAS.
#[tokio::test(flavor = "current_thread")]
async fn new_cell_prepare_is_not_authority_before_commit() {
    let session = session_on_origin_cell();
    let fixture = migrate_until(MigrationCrashPoint::AfterNewPrepare, session).await;

    // Staged, and provably not issued.
    assert_eq!(fixture.controller.active().staged_count(), 1);
    assert_eq!(fixture.controller.active().issued_count(), 0);
    assert!(matches!(
        fixture
            .placement
            .borrow()
            .route_state_for(session)
            .expect("the session is installed"),
        SessionRouteState::Prepared { .. }
    ));

    let restored = fixture.restore_from_last_committed().await;
    assert_eq!(
        restored
            .route_for(session)
            .expect("the restored set retains the session")
            .destination_cell,
        0,
        "a crash after staging restores the old owner"
    );

    drop(restored);
    // The placement is shared with the controller's checkpoint participant, so
    // both owners must go before the route leases are actually returned.
    let MigrationFixture {
        controller,
        placement,
        route_budget,
        ..
    } = fixture;
    drop(controller);
    drop(placement);
    assert_eq!(
        route_budget.snapshot().used_items,
        0,
        "discarding the staged migration returns every route charge"
    );
}

/// The transient double route charge is visible to admission waiters.
#[tokio::test(flavor = "current_thread")]
async fn migration_pending_fragments_obey_budget() {
    let session = session_on_origin_cell();
    // Exactly two route entries: one committed owner plus one migration charge.
    let mut fixture = MigrationFixture::new(2);
    fixture.install_route(session, 1).await;
    fixture
        .controller
        .commit_fence(barrier_at(1))
        .await
        .expect("baseline generation");
    fixture
        .controller
        .freeze_session(session, GlobalSequence::new(FREEZE_THROUGH))
        .expect("fence the session");
    fixture
        .controller
        .commit_fence(barrier_at(2))
        .await
        .expect("fence generation");
    fixture
        .controller
        .stage_new_cell(session, 1, &[(action(9), GlobalSequence::new(9))])
        .await
        .expect("stage the destination");
    assert_eq!(fixture.placement.borrow().migration_lease_count(), 1);
    assert_eq!(fixture.route_budget.snapshot().used_items, 2);

    let mut other = [0u8; 32];
    other[0] = 0xEE;
    let contender = StableSessionKey::from_bytes(other);
    let mut admission = fixture.placement.borrow().admission();
    let woken = Arc::new(AtomicBool::new(false));
    let waker = Waker::from(Arc::new(FlagWaker(Arc::clone(&woken))));
    let mut context = Context::from_waker(&waker);
    let mut pending = pin!(admission.reserve_route(PlacementRouteCharge {
        session: contender,
        items: 1,
        bytes: ROUTE_ENTRY_BYTES,
    }));
    assert!(
        pending.as_mut().poll(&mut context).is_pending(),
        "the migration's second charge must make a third route pend"
    );

    fixture
        .controller
        .commit_route_generation(session, barrier_at(3))
        .await
        .expect("route generation CAS");
    fixture
        .controller
        .promote_owner(session)
        .expect("settle the migration leases");
    assert_eq!(fixture.route_budget.snapshot().used_items, 1);
    assert!(woken.load(Ordering::SeqCst), "settling wakes the waiter");
    assert!(matches!(
        pending.as_mut().poll(&mut context),
        Poll::Ready(Ok(_))
    ));
}

/// Re-issuing a completed migration returns the same epoch without a second CAS.
#[tokio::test(flavor = "current_thread")]
async fn identical_migration_retry_is_idempotent() {
    let session = session_on_origin_cell();
    let mut fixture = migrate_until(MigrationCrashPoint::AfterRelease, session).await;
    let head = fixture
        .controller
        .coordinator()
        .expected()
        .cloned()
        .expect("a committed head after the migration");

    let retried = fixture
        .controller
        .migrate(
            session,
            1,
            GlobalSequence::new(FREEZE_THROUGH),
            &[],
            barrier_at(4),
            barrier_at(5),
        )
        .await
        .expect("an identical retry is accepted");

    assert_eq!(retried, SessionOwnershipEpoch::new(1));
    assert_eq!(
        fixture.controller.coordinator().expected(),
        Some(&head),
        "an idempotent retry performs no second CAS"
    );
    assert_eq!(fixture.placement.borrow().migration_lease_count(), 0);
}

/// A second, conflicting migration of the same session fails closed.
#[tokio::test(flavor = "current_thread")]
async fn conflicting_migration_receipt_fails_closed() {
    let session = session_on_origin_cell();
    let mut fixture = migrate_until(MigrationCrashPoint::AfterNewPrepare, session).await;

    let refusal = fixture
        .controller
        .freeze_session(session, GlobalSequence::new(FREEZE_THROUGH + 1))
        .expect_err("a session already migrating admits no second fence");
    assert_eq!(
        refusal.placement_code(),
        Some(PlacementFailureCode::RouteUnavailable)
    );

    fixture
        .controller
        .commit_route_generation(session, barrier_at(3))
        .await
        .expect("the first migration still commits");
    fixture
        .controller
        .promote_owner(session)
        .expect("settle the first migration");
    assert_eq!(
        fixture
            .placement
            .borrow()
            .route_for(session)
            .expect("the session is installed")
            .ownership_epoch,
        SessionOwnershipEpoch::new(1),
        "exactly one epoch increment is committed"
    );
}

/// A terminal arriving mid-migration leaves the route and both leases in place.
#[tokio::test(flavor = "current_thread")]
async fn terminal_retirement_is_refused_during_migration() {
    let session = session_on_origin_cell();
    let fixture = migrate_until(MigrationCrashPoint::AfterNewPrepare, session).await;

    fixture
        .placement
        .borrow_mut()
        .retire_route_if_fenced(session, SessionOwnershipEpoch::new(0), &frontier(100))
        .expect("a terminal mid-migration is accepted, not an error");

    let placement = fixture.placement.borrow();
    assert_eq!(placement.active_owner_count(), 1, "the route stays installed");
    assert_eq!(placement.migration_lease_count(), 1, "both leases are held");
    assert_eq!(fixture.route_budget.snapshot().used_items, 2);
}

/// An aborted migration returns every charge, and so does a restart.
#[tokio::test(flavor = "current_thread")]
async fn aborted_migration_leaks_no_budget_across_restart() {
    let session = session_on_origin_cell();
    let fixture = migrate_until(MigrationCrashPoint::AfterNewPrepare, session).await;
    let mut fixture = fixture;
    assert_eq!(fixture.route_budget.snapshot().used_items, 2);

    fixture
        .controller
        .abort_migration(session)
        .expect("abort reverts to the committed owner");
    assert_eq!(
        fixture.route_budget.snapshot().used_items,
        1,
        "abort returns exactly the transient charge"
    );
    assert!(matches!(
        fixture
            .placement
            .borrow()
            .route_state_for(session)
            .expect("the session is installed"),
        SessionRouteState::Owned(_)
    ));

    let restored = fixture.restore_from_last_committed().await;
    assert_eq!(restored.active_owner_count(), 1);
    assert_eq!(restored.migration_lease_count(), 0);
    drop(restored);

    let MigrationFixture {
        controller,
        placement,
        route_budget,
        ..
    } = fixture;
    drop(controller);
    drop(placement);
    assert_eq!(
        route_budget.snapshot().used_items,
        0,
        "a restart after an abort leaks no route charge"
    );
}

/// A release naming an epoch the staged action was not prepared under is refused
/// in both directions, and the action stays staged for a corrected release.
#[test]
fn greater_release_epoch_is_refused_and_leaves_the_action_staged() {
    let mut active = ActiveExecutionSet::new();
    let staged = ReleaseFence {
        plan_digest: PLAN_DIGEST,
        route_id: 1,
        action_id: action(9),
        global_sequence: GlobalSequence::new(9),
        ownership_epoch: SessionOwnershipEpoch::new(1),
    };
    active.accept_prepare(staged).expect("stage the action");

    let ahead = ReleaseFence {
        ownership_epoch: SessionOwnershipEpoch::new(2),
        ..staged
    };
    let refusal = active
        .accept_release(&ahead)
        .expect_err("a cell holds no epoch counter and infers nothing");
    assert_eq!(
        refusal.failure_code(),
        PlacementFailureCode::StaleOwnershipEpoch
    );
    assert_eq!(active.staged_count(), 1, "the action is still staged");
    assert_eq!(active.issued_count(), 0);

    active
        .accept_release(&staged)
        .expect("the exact release still grants authority");
    assert_eq!(active.issued_count(), 1);
}
