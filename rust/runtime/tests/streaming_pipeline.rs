// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded streaming pipeline composition, backpressure, and placement.

#![cfg(feature = "streaming")]

use std::rc::Rc;

use aiperf_runtime::streaming::{
    action::ActionTerminalDisposition,
    checkpoint::{
        CheckpointParticipantId, CheckpointParticipantPlanError, StreamRunIdentity,
        StreamingCheckpointParticipant,
    },
    failure::PlacementFailureCode,
    identity::{
        ContentDigest, GlobalSequence, LogicalReplayRunId, SessionCausalFrontier,
        SessionOwnershipEpoch, StableActionId, StableSessionKey,
    },
    pipeline::{StreamingPipelineError, StreamingTerminalReason},
    placement::{
        LocalPlacementAdmission, LocalStreamingPlacement, PlacementError, PlacementRouteCharge,
        StreamingPlacementAdmission, StreamingPlacementPolicy, StreamingPlacementSubmitter,
        local_placement_binding,
    },
    unit::StateBudgetFailureCode,
};

#[path = "support/streaming_pipeline.rs"]
mod support;

use support::{
    FixtureSpec, OmittedOwner, budget, build, frontiers, frontiers_then_seal, local, request_action,
};

/// Yield enough times for the pipeline task to reach its next parked point.
async fn settle() {
    for _ in 0..64 {
        tokio::task::yield_now().await;
    }
}

/// A saturated innermost permit pins every upstream stage, and stop still wakes.
///
/// The fixture's submitter parks exactly where the terminal-record permit sits,
/// so this asserts the whole nesting at once: with one unit stalled inside
/// `submit`, the pipeline issues no further source pull, admits no second
/// action, and still resolves when the run is stopped.
#[test]
fn downstream_pressure_stops_every_upstream_pull_and_stop_wakes_pending() {
    local(async {
        let fixture = build(FixtureSpec {
            source_events: frontiers(100),
            ..FixtureSpec::default()
        })
        .expect("fixture");
        let probes = Rc::clone(&fixture.probes);
        let stop = fixture.stop.clone();
        probes.block_terminal_lane();

        let pipeline = fixture.pipeline;
        let phase = fixture.phase;
        let handle = tokio::task::spawn_local(async move { pipeline.run(phase).await });

        settle().await;

        assert!(
            probes.is_submit_parked.get(),
            "the submitter must be parked on the blocked lane"
        );
        assert_eq!(
            probes.submitted.borrow().len(),
            0,
            "no action may be accepted while the innermost permit is unavailable"
        );
        assert_eq!(
            probes.source_pulls_after_saturation.get(),
            0,
            "a saturated downstream must stop every upstream pull"
        );
        assert_eq!(
            probes.watermarks.get(),
            1,
            "exactly the stalled unit's watermark may be folded"
        );

        stop.stop();
        let outcome = handle.await.expect("pipeline task").expect("run outcome");
        assert_eq!(outcome.terminal_reason, StreamingTerminalReason::Cancelled);
        assert!(
            probes.is_action_drained.get(),
            "shutdown must join every driver"
        );
        assert!(probes.is_issue_stopped.get(), "shutdown must fence issue");
    });
}

/// Releasing the innermost permit resumes admission and the upstream pulls.
#[test]
fn releasing_the_innermost_permit_resumes_every_upstream_stage() {
    local(async {
        let fixture = build(FixtureSpec {
            source_events: frontiers(3),
            ..FixtureSpec::default()
        })
        .expect("fixture");
        let probes = Rc::clone(&fixture.probes);
        let stop = fixture.stop.clone();
        probes.block_terminal_lane();

        let pipeline = fixture.pipeline;
        let phase = fixture.phase;
        let handle = tokio::task::spawn_local(async move { pipeline.run(phase).await });

        settle().await;
        let stalled_pulls = probes.source_pulls.get();
        assert_eq!(probes.submitted.borrow().len(), 0);

        probes.release_terminal_lane();
        settle().await;

        assert!(
            probes.source_pulls.get() > stalled_pulls,
            "the source must resume once the innermost permit is released"
        );
        assert_eq!(
            probes.submitted.borrow().len(),
            3,
            "every scripted unit must be admitted after the release"
        );

        stop.stop();
        probes.emit_all_terminal();
        let outcome = handle.await.expect("pipeline task").expect("run outcome");
        assert_eq!(outcome.terminal_reason, StreamingTerminalReason::Cancelled);
    });
}

/// A finite source seals, drains its accepted prefix, and commits one generation.
#[test]
fn finite_seal_drains_then_commits_one_terminal_generation() {
    local(async {
        let fixture = build(FixtureSpec {
            source_events: frontiers_then_seal(2),
            ..FixtureSpec::default()
        })
        .expect("fixture");
        let probes = Rc::clone(&fixture.probes);

        let pipeline = fixture.pipeline;
        let phase = fixture.phase;
        let handle = tokio::task::spawn_local(async move { pipeline.run(phase).await });

        settle().await;
        assert_eq!(probes.submitted.borrow().len(), 2);
        assert_eq!(probes.seals.get(), 1, "the source seals exactly once");

        probes.emit_all_terminal();
        let outcome = handle.await.expect("pipeline task").expect("run outcome");

        assert_eq!(outcome.terminal_reason, StreamingTerminalReason::Sealed);
        assert!(
            outcome.last_committed_generation.is_some(),
            "a sealed run must publish exactly one terminal generation"
        );
    });
}

/// Session state is reached only through the settle path, never from the driver.
#[test]
fn placement_event_action_is_the_only_route_into_session_state() {
    local(async {
        let fixture = build(FixtureSpec {
            source_events: frontiers_then_seal(1),
            ..FixtureSpec::default()
        })
        .expect("fixture");
        let probes = Rc::clone(&fixture.probes);

        let pipeline = fixture.pipeline;
        let phase = fixture.phase;
        let handle = tokio::task::spawn_local(async move { pipeline.run(phase).await });

        settle().await;
        assert_eq!(
            probes.observed.borrow().as_slice(),
            &[] as &[&str],
            "no execution event may reach session state before one is emitted"
        );

        probes.emit_admitted(0);
        settle().await;
        assert_eq!(probes.observed.borrow().as_slice(), &["admitted"]);

        probes.emit_terminal(0, ActionTerminalDisposition::Completed);
        let outcome = handle.await.expect("pipeline task").expect("run outcome");

        assert_eq!(
            probes.observed.borrow().as_slice(),
            &["admitted", "terminal"],
            "every fold must arrive in emission order through the settle path"
        );
        assert_eq!(outcome.terminal_reason, StreamingTerminalReason::Sealed);
    });
}

/// The frozen participant set is validated before the first source poll.
#[test]
fn frozen_participant_set_is_required_before_the_first_source_poll() {
    let error = build(FixtureSpec {
        source_events: frontiers(1),
        omit: Some(OmittedOwner::BlockingOwner),
        ..FixtureSpec::default()
    })
    .err()
    .expect("a plan that repeats an owner identity must be refused");

    match error {
        StreamingPipelineError::ParticipantPlan(
            CheckpointParticipantPlanError::DuplicateParticipant(id),
        ) => {
            assert_eq!(id, CheckpointParticipantId::new("source"));
        }
        other => panic!("expected a participant-plan refusal, got {other:?}"),
    }
}

fn placement_run() -> StreamRunIdentity {
    StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x51; 32]))
}

fn frontier() -> SessionCausalFrontier {
    SessionCausalFrontier {
        through_sequence: GlobalSequence::new(1),
        event_time: None,
        digest: ContentDigest::from_bytes([1; 32]),
    }
}

/// A dropped reservation returns its exact charge and installs no route.
#[test]
fn cancelled_route_reservation_installs_no_route() {
    local(async {
        let route_budget = budget(4, 4096);
        let mut admission = LocalPlacementAdmission::new(route_budget.clone());
        let mut policy = LocalStreamingPlacement::new(
            CheckpointParticipantId::new("policy"),
            placement_run(),
            budget(4, 4096),
        );
        let before = route_budget.snapshot().used_bytes;

        let reservation = admission
            .reserve_route(PlacementRouteCharge {
                session: StableSessionKey::from_bytes([1; 32]),
                items: 1,
                bytes: 64,
            })
            .await
            .expect("route capacity");
        assert_eq!(reservation.charged_bytes(), 64);
        assert_eq!(
            route_budget.snapshot().used_bytes,
            before + 64,
            "a held reservation must occupy exact capacity"
        );

        drop(reservation);
        assert_eq!(
            route_budget.snapshot().used_bytes,
            before,
            "dropping a reservation must return the whole charge"
        );
        assert_eq!(
            policy.installed_route_count(),
            0,
            "a reservation that was never installed leaves the route map empty"
        );
        let _ = &mut policy;
    });
}

/// A session terminal fences its route epoch and refuses a stale one afterwards.
#[test]
fn session_terminal_fences_the_route_epoch() {
    local(async {
        let mut policy = LocalStreamingPlacement::new(
            CheckpointParticipantId::new("policy"),
            placement_run(),
            budget(4, 4096),
        );
        let session = StableSessionKey::from_bytes([2; 32]);
        assert_eq!(
            policy.ownership_epoch(session),
            SessionOwnershipEpoch::new(0)
        );

        policy
            .observe_session_terminal(session, SessionOwnershipEpoch::new(0), &frontier())
            .expect("first terminal advances the epoch");
        assert_eq!(
            policy.ownership_epoch(session),
            SessionOwnershipEpoch::new(1)
        );

        let stale = policy
            .observe_session_terminal(session, SessionOwnershipEpoch::new(0), &frontier())
            .expect_err("a fenced epoch must be refused");
        assert_eq!(
            stale,
            PlacementError::placement(PlacementFailureCode::StaleOwnershipEpoch)
        );
    });
}

/// The placement slab is a second, independent bound on in-flight work.
#[test]
fn placement_slab_refuses_beyond_its_authored_bound() {
    local(async {
        let mut binding = local_placement_binding(
            placement_run(),
            CheckpointParticipantId::new("policy"),
            CheckpointParticipantId::new("driver"),
            1,
            budget(4, 4096),
            budget(4, 4096),
        );
        let content = budget(64, 65_536);
        let first = request_action(&content, StableActionId::from_bytes([1; 32])).await;
        let second = request_action(&content, StableActionId::from_bytes([2; 32])).await;

        // A worker-local policy declares no route charge at all.
        assert!(
            binding
                .policy
                .route_admission(&first)
                .expect("route admission")
                .is_none()
        );

        let decision = binding.policy.place(&first).expect("first placement");
        let handle = binding
            .submitter
            .prepare(decision, &first)
            .await
            .expect("first prepare");
        assert_eq!(handle.global_sequence, None, "placement assigns no order");

        let decision = binding.policy.place(&second).expect("second placement");
        let refusal = binding
            .submitter
            .prepare(decision, &second)
            .await
            .expect_err("the slab bound must refuse the second placement");
        assert_eq!(
            refusal,
            PlacementError::state_budget(StateBudgetFailureCode::ItemCapacity)
        );

        binding
            .submitter
            .bind_sequence(handle.id, GlobalSequence::new(7))
            .expect("the host's assigned order binds once");
        assert_eq!(
            binding
                .submitter
                .bind_sequence(handle.id, GlobalSequence::new(8))
                .expect_err("an order binds exactly once"),
            PlacementError::placement(PlacementFailureCode::DigestMismatch)
        );

        binding
            .submitter
            .release(handle.id)
            .await
            .expect("release the slab entry");
        assert_eq!(binding.submitter.prepared_count(), 0);
    });
}

/// The local placement policy and driver both report their frozen identities.
#[test]
fn local_placement_halves_carry_their_frozen_identities() {
    let binding = local_placement_binding(
        placement_run(),
        CheckpointParticipantId::new("placement_policy"),
        CheckpointParticipantId::new("placement_driver"),
        4,
        budget(4, 4096),
        budget(4, 4096),
    );
    assert_eq!(
        binding.policy.participant_id(),
        CheckpointParticipantId::new("placement_policy")
    );
    assert_eq!(
        binding.driver.participant_id(),
        CheckpointParticipantId::new("placement_driver")
    );
}
