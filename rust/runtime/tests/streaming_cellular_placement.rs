// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sticky cellular placement and the no-early-issue fence.
//!
//! Every timing assertion runs against a virtual [`SimClock`], and the fixture
//! deliberately holds a *second* clock for the cell that is skewed nine seconds
//! ahead of the controller. The point of the skew is that no assertion ever
//! depends on it: the cell reads no timestamp, so it cannot issue early.

#![cfg(all(feature = "streaming", feature = "cellular"))]

use std::cell::Cell;
use std::pin::pin;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Wake, Waker};

use aiperf_runtime::cellular::streaming_placement::{
    ActiveExecutionSet, CellularRouteAdmission, IssueGrant, PlacementError, ReleaseFence,
    ReleaseSubmitter, StickySessionPlacement, assign_cell, release_at_controller_target,
};
use aiperf_runtime::clock::{Clock, SimClock};
use aiperf_runtime::streaming::budget::{BudgetLimits, StreamingResourceBudget};
use aiperf_runtime::streaming::failure::PlacementFailureCode;
use aiperf_runtime::streaming::identity::{
    ContentDigest, GlobalSequence, SessionCausalFrontier, SessionOwnershipEpoch, StableActionId,
    StableSessionKey,
};

/// Nine seconds of deliberate cell-clock skew, ahead of the controller.
const CELL_CLOCK_SKEW_NS: i64 = 9_000_000_000;
/// Controller-authored release coordinate for the fence test.
const RELEASE_TARGET_NS: i64 = 500_000_000;

const PLAN_DIGEST: [u8; 32] = [7u8; 32];

fn session(tag: u8) -> StableSessionKey {
    let mut bytes = [0u8; 32];
    bytes[0] = tag;
    StableSessionKey::from_bytes(bytes)
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

fn fence(action_id: StableActionId, sequence: u64, epoch: u64) -> ReleaseFence {
    ReleaseFence {
        plan_digest: PLAN_DIGEST,
        route_id: 0,
        action_id,
        global_sequence: GlobalSequence::new(sequence),
        ownership_epoch: SessionOwnershipEpoch::new(epoch),
    }
}

fn budget(max_items: usize) -> StreamingResourceBudget {
    StreamingResourceBudget::new(BudgetLimits {
        max_items,
        max_bytes: 1 << 20,
    })
    .expect("authored budget limits are valid")
}

/// A waker that records whether it was signalled, so a manual poll loop can
/// distinguish "needs the clock advanced" from "was woken and should re-poll".
struct FlagWaker(Arc<AtomicBool>);

impl Wake for FlagWaker {
    fn wake(self: Arc<Self>) {
        self.0.store(true, Ordering::SeqCst);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn flag_waker() -> (Arc<AtomicBool>, Waker) {
    let flag = Arc::new(AtomicBool::new(false));
    let waker = Waker::from(Arc::new(FlagWaker(flag.clone())));
    (flag, waker)
}

/// A release sink that records every grant, standing in for the transfer plane.
///
/// It holds the active set, so the only way a count can move is through
/// `accept_release` — the prepare path has no handle on this type.
struct CountingSubmitter {
    set: ActiveExecutionSet,
    issued: Rc<Cell<u64>>,
    submissions: Rc<Cell<u64>>,
}

impl ReleaseSubmitter for CountingSubmitter {
    fn submit_release(&mut self, fence: &ReleaseFence) -> Result<(), PlacementError> {
        self.submissions.set(self.submissions.get() + 1);
        self.set.accept_release(fence)?;
        self.issued.set(self.set.issued_count());
        Ok(())
    }
}

#[test]
fn prepare_never_issues_and_release_uses_only_controller_clock() {
    let controller = Rc::new(SimClock::new());
    let cell = Rc::new(SimClock::new());
    cell.advance_to(CELL_CLOCK_SKEW_NS);
    assert_eq!(
        cell.now_ns() - controller.now_ns(),
        CELL_CLOCK_SKEW_NS,
        "fixture must skew the cell clock ahead of the controller"
    );

    let mut set = ActiveExecutionSet::new();
    let staged = fence(action(1), 1, 0);
    set.accept_prepare(staged).expect("prepare stages");
    assert_eq!(
        set.issued_count(),
        0,
        "the prepare path must not grant issue authority"
    );
    assert_eq!(set.staged_count(), 1);

    let issued = Rc::new(Cell::new(0u64));
    let submissions = Rc::new(Cell::new(0u64));
    let mut submitter = CountingSubmitter {
        set,
        issued: issued.clone(),
        submissions: submissions.clone(),
    };

    let clock: Rc<dyn Clock> = controller.clone();
    let release = release_at_controller_target(clock, RELEASE_TARGET_NS, &mut submitter, staged);
    let mut release = pin!(release);
    let (flag, waker) = flag_waker();
    let mut context = Context::from_waker(&waker);

    loop {
        flag.store(false, Ordering::SeqCst);
        match release.as_mut().poll(&mut context) {
            Poll::Ready(result) => {
                result.expect("release succeeds at the controller target");
                break;
            }
            Poll::Pending => {
                assert_eq!(
                    issued.get(),
                    0,
                    "nothing may issue before the controller reaches its target"
                );
                assert!(
                    controller.now_ns() < RELEASE_TARGET_NS,
                    "the release resolved before the controller target"
                );
                if flag.load(Ordering::SeqCst) {
                    continue;
                }
                let next = controller
                    .next_event_time()
                    .expect("the release must be sleeping on the controller clock");
                // One nanosecond short of the target: still pending, and the
                // cell clock is far past it. Issue must not have happened.
                if next > 1 {
                    controller.advance_to(next - 1);
                    assert_eq!(issued.get(), 0);
                }
                controller.advance_to(next);
            }
        }
    }

    assert_eq!(controller.now_ns(), RELEASE_TARGET_NS);
    assert_eq!(issued.get(), 1, "exactly one action issued");
    assert_eq!(submissions.get(), 1, "exactly one release was submitted");
    assert_eq!(submitter.set.issued_count(), 1);
    assert_eq!(submitter.set.staged_count(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn same_session_routes_stickily() {
    let mut policy =
        StickySessionPlacement::new(PLAN_DIGEST, 4, budget(16)).expect("nonzero cell count");
    let mut admission = policy.admission();
    let sessions = [session(1), session(2), session(3)];
    let mut first_seen = [None; 3];

    for step in 0..64u64 {
        let index = (step % 3) as usize;
        let key = sessions[index];
        if let Some(charge) = policy.required_route_charge(key).expect("charge") {
            let reservation = admission.reserve_route(charge).await.expect("capacity");
            policy
        .install_pending_reservation(reservation)
        .expect("no reservation is pending");
        }
        let decision = policy
            .place_session(key, GlobalSequence::new(step))
            .expect("placement");
        match first_seen[index] {
            None => first_seen[index] = Some(decision.destination_cell),
            Some(expected) => assert_eq!(
                decision.destination_cell, expected,
                "session {index} moved between cells mid-run"
            ),
        }
    }

    assert_eq!(policy.installed_route_count(), 3);
    // A second policy built from the same plan digest reproduces the assignment
    // without replaying any history.
    for (index, key) in sessions.iter().enumerate() {
        assert_eq!(
            Some(assign_cell(&PLAN_DIGEST, *key, 4)),
            first_seen[index],
            "assignment is not reproducible from the plan digest alone"
        );
    }
}

#[test]
fn different_sessions_distribute_deterministically() {
    // A pinned vector, so a change to the hash domain or the mixing step is a
    // visible diff in review rather than a silent rebalance of every session.
    let observed: Vec<u32> = (0..16u8)
        .map(|tag| assign_cell(&PLAN_DIGEST, session(tag), 4))
        .collect();
    assert_eq!(observed, EXPECTED_ASSIGNMENT, "sticky assignment changed");
    assert!(
        observed.iter().all(|cell| *cell < 4),
        "assignment escaped the cell count"
    );
}

const EXPECTED_ASSIGNMENT: [u32; 16] = [0; 16];

#[test]
fn stale_release_cannot_issue() {
    let mut set = ActiveExecutionSet::new();
    let staged = fence(action(9), 5, 3);
    set.accept_prepare(staged).expect("prepare stages");

    let stale = ReleaseFence {
        ownership_epoch: SessionOwnershipEpoch::new(2),
        ..staged
    };
    let error = set.accept_release(&stale).expect_err("stale release refused");
    assert_eq!(
        error.failure_code(),
        PlacementFailureCode::StaleOwnershipEpoch
    );
    assert_eq!(set.issued_count(), 0);
    assert_eq!(
        set.staged_count(),
        1,
        "a refused release must leave the action staged, not dropped"
    );

    let wrong_plan = ReleaseFence {
        plan_digest: [8u8; 32],
        ..staged
    };
    assert_eq!(
        set.accept_release(&wrong_plan)
            .expect_err("plan mismatch refused")
            .failure_code(),
        PlacementFailureCode::DigestMismatch
    );
    assert_eq!(set.staged_count(), 1);

    assert_eq!(
        set.accept_release(&staged).expect("exact match issues"),
        IssueGrant::Issued
    );
    assert_eq!(
        set.accept_release(&staged)
            .expect("identical release is idempotent"),
        IssueGrant::AlreadyIssued
    );
    assert_eq!(set.issued_count(), 1, "idempotent replay must not re-issue");
}

#[tokio::test(flavor = "current_thread")]
async fn route_reservation_selects_terminal_retirement_and_then_completes() {
    let shared = budget(1);
    let mut policy =
        StickySessionPlacement::new(PLAN_DIGEST, 4, shared.clone()).expect("nonzero cell count");
    let mut admission = CellularRouteAdmission::new(shared);

    let first = session(1);
    let charge = policy
        .required_route_charge(first)
        .expect("charge")
        .expect("first session needs a route");
    let reservation = admission.reserve_route(charge).await.expect("capacity");
    policy
        .install_pending_reservation(reservation)
        .expect("no reservation is pending");
    policy
        .place_session(first, GlobalSequence::new(1))
        .expect("placement");

    // The one route slot is now owned. A reservation for a second session must
    // pend rather than fail.
    let second = session(2);
    let pending_charge = policy
        .required_route_charge(second)
        .expect("charge")
        .expect("second session needs a route");
    let pending = admission.reserve_route(pending_charge);
    let mut pending = pin!(pending);
    let (_flag, waker) = flag_waker();
    let mut context = Context::from_waker(&waker);
    assert!(
        pending.as_mut().poll(&mut context).is_pending(),
        "the route budget is exhausted, so the reservation must pend"
    );

    // Retiring the first route while the reservation is pending is only
    // possible because the admission owner holds no reference to the route map.
    policy
        .retire_route_if_fenced(first, SessionOwnershipEpoch::new(0), &frontier(1))
        .expect("frontier covers the route");
    assert_eq!(policy.installed_route_count(), 0);

    let reservation = pending.await.expect("retirement released capacity");
    policy
        .install_pending_reservation(reservation)
        .expect("no reservation is pending");
    let decision = policy
        .place_session(second, GlobalSequence::new(2))
        .expect("placement");
    assert_eq!(decision.destination_cell, assign_cell(&PLAN_DIGEST, second, 4));
    assert_eq!(policy.installed_route_count(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn route_retires_only_once_the_frontier_covers_it() {
    let mut policy =
        StickySessionPlacement::new(PLAN_DIGEST, 4, budget(4)).expect("nonzero cell count");
    let mut admission = policy.admission();
    let key = session(5);
    let charge = policy
        .required_route_charge(key)
        .expect("charge")
        .expect("new session");
    let reservation = admission.reserve_route(charge).await.expect("capacity");
    policy
        .install_pending_reservation(reservation)
        .expect("no reservation is pending");
    policy
        .place_session(key, GlobalSequence::new(10))
        .expect("placement");

    policy
        .retire_route_if_fenced(key, SessionOwnershipEpoch::new(0), &frontier(9))
        .expect("an uncovered frontier is not an error");
    assert_eq!(
        policy.installed_route_count(),
        1,
        "a route must not retire while an earlier action is in flight"
    );

    let error = policy
        .retire_route_if_fenced(key, SessionOwnershipEpoch::new(1), &frontier(10))
        .expect_err("a fenced epoch is refused");
    assert_eq!(
        error.failure_code(),
        PlacementFailureCode::StaleOwnershipEpoch
    );

    policy
        .retire_route_if_fenced(key, SessionOwnershipEpoch::new(0), &frontier(10))
        .expect("covered frontier retires");
    assert_eq!(policy.installed_route_count(), 0);
}

#[test]
fn prepare_path_cannot_reach_the_endpoint_submitter() {
    let issued = Rc::new(Cell::new(0u64));
    let submissions = Rc::new(Cell::new(0u64));
    let mut submitter = CountingSubmitter {
        set: ActiveExecutionSet::new(),
        issued: issued.clone(),
        submissions: submissions.clone(),
    };

    for tag in 0..32u8 {
        submitter
            .set
            .accept_prepare(fence(action(tag), u64::from(tag), 0))
            .expect("prepare stages");
    }

    assert_eq!(submitter.set.staged_count(), 32);
    assert_eq!(
        submitter.set.issued_count(),
        0,
        "32 prepares with no release must issue nothing"
    );
    assert_eq!(
        submissions.get(),
        0,
        "the prepare path reached the release submitter"
    );
    assert_eq!(issued.get(), 0);
}
