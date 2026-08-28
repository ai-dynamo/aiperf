// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checkpoint coordinator publication, idempotency, and post-CAS notification.

#![cfg(feature = "streaming")]

use aiperf_runtime::streaming::{
    checkpoint::CheckpointError,
    checkpoint_coordinator::{
        PreCasFailureRouting, PreparedCheckpointResultInput, StreamingCheckpointCoordinator,
    },
};

#[path = "support/streaming_checkpoint_coordinator.rs"]
mod support;

use support::{
    FakeIssueReporter, NotifyingParticipant, barrier_at, barrier_for_run, coordinator_fixture,
    expectations, run_id,
};

fn local(future: impl Future<Output = ()>) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("current-thread runtime");
    tokio::task::LocalSet::new().block_on(&runtime, future);
}

#[test]
fn one_coordinator_commits_consecutive_barriers_against_its_advanced_head() {
    local(async {
        let mut fixture = coordinator_fixture();
        let first = fixture
            .coordinator
            .commit_barrier(barrier_at(1), &mut PreparedCheckpointResultInput::empty())
            .await
            .expect("publish first barrier");
        let second = fixture
            .coordinator
            .commit_barrier(barrier_at(2), &mut PreparedCheckpointResultInput::empty())
            .await
            .expect("publish second barrier");

        assert_eq!(second.previous(), Some(first.generation_ref().digest()));
        assert_eq!(
            fixture.latest_generation().await.as_ref(),
            Some(second.generation_ref())
        );
        assert_eq!(
            fixture.coordinator.expected(),
            Some(second.generation_ref())
        );
        assert_eq!(fixture.participant.commit_notifications(), 2);
        assert_eq!(fixture.reporter.acknowledged_roots().len(), 2);
    });
}

#[test]
fn post_commit_failure_does_not_roll_back_authoritative_head() {
    local(async {
        let mut fixture = coordinator_fixture();
        fixture.participant.fail_next_commit_notifications(1);
        let error = fixture
            .coordinator
            .commit_barrier(barrier_at(1), &mut PreparedCheckpointResultInput::empty())
            .await
            .expect_err("notification refusal surfaces");
        assert!(matches!(
            error,
            CheckpointError::PostCommitNotification { .. }
        ));

        // The head became authoritative before the callback, so it stands.
        let head = fixture
            .latest_generation()
            .await
            .expect("published head survives the failed notification");
        assert_eq!(fixture.coordinator.expected(), Some(&head));
        assert_eq!(
            fixture.coordinator.pending_notification_generation(),
            Some(&head)
        );
        assert_eq!(fixture.participant.commit_notifications(), 0);
    });
}

#[test]
fn notification_failure_advances_expected_before_same_coordinator_next_barrier() {
    local(async {
        let mut fixture = coordinator_fixture();
        fixture.participant.fail_next_commit_notifications(1);
        fixture
            .coordinator
            .commit_barrier(barrier_at(1), &mut PreparedCheckpointResultInput::empty())
            .await
            .expect_err("first notification refused");
        let first = fixture
            .coordinator
            .expected()
            .cloned()
            .expect("first generation retained");

        let second = fixture
            .coordinator
            .commit_barrier(barrier_at(2), &mut PreparedCheckpointResultInput::empty())
            .await
            .expect("successor publishes after the pending retry");

        // The pending receipt is retried first, then the successor is notified.
        assert_eq!(fixture.participant.commit_notifications(), 2);
        assert_eq!(
            fixture.participant.last_notified_generation().as_ref(),
            Some(second.generation_ref())
        );
        assert_eq!(second.previous(), Some(first.digest()));
        assert_eq!(fixture.coordinator.pending_notification_generation(), None);
    });
}

#[test]
fn exact_barrier_retry_notifies_then_returns_same_generation_without_recommit() {
    local(async {
        let mut fixture = coordinator_fixture();
        fixture.participant.fail_next_commit_notifications(1);
        fixture
            .coordinator
            .commit_barrier(barrier_at(1), &mut PreparedCheckpointResultInput::empty())
            .await
            .expect_err("first notification refused");
        let commits_before = fixture.backend_control.commit_calls();
        let inventory_before = fixture.immutable_object_inventory();

        let repeated = fixture
            .coordinator
            .commit_barrier(barrier_at(1), &mut PreparedCheckpointResultInput::empty())
            .await
            .expect("exact repeat short-circuits");

        assert_eq!(fixture.backend_control.commit_calls(), commits_before);
        assert_eq!(fixture.immutable_object_inventory(), inventory_before);
        assert_eq!(fixture.participant.commit_notifications(), 1);
        assert_eq!(
            fixture.latest_generation().await.as_ref(),
            Some(repeated.generation_ref())
        );
    });
}

#[test]
fn pending_notification_error_preserves_new_partition_authority() {
    local(async {
        let mut fixture = coordinator_fixture();
        fixture.participant.fail_next_commit_notifications(2);
        fixture
            .coordinator
            .commit_barrier(barrier_at(1), &mut PreparedCheckpointResultInput::empty())
            .await
            .expect_err("first notification refused");

        let mut inputs = PreparedCheckpointResultInput::empty();
        let stage_calls_before = fixture.backend_control.stage_results_calls();
        let error = fixture
            .coordinator
            .commit_barrier(barrier_at(2), &mut inputs)
            .await
            .expect_err("pending retry refused again");
        assert!(matches!(
            error,
            CheckpointError::PostCommitNotification { .. }
        ));

        // The successor never reached staging, so the caller's inputs and the
        // pending publication are both exactly as they were.
        assert_eq!(
            fixture.backend_control.stage_results_calls(),
            stage_calls_before
        );
        assert!(inputs.partitions().is_empty());
        assert!(inputs.issue_receipts().is_none());
        assert!(
            fixture
                .coordinator
                .pending_notification_generation()
                .is_some()
        );
    });
}

#[test]
fn foreign_run_barrier_refuses_before_pending_notification_retry() {
    local(async {
        let mut fixture = coordinator_fixture();
        fixture.participant.fail_next_commit_notifications(1);
        fixture
            .coordinator
            .commit_barrier(barrier_at(1), &mut PreparedCheckpointResultInput::empty())
            .await
            .expect_err("first notification refused");

        let foreign = barrier_for_run(run_id(9), 2);
        let error = fixture
            .coordinator
            .commit_barrier(foreign, &mut PreparedCheckpointResultInput::empty())
            .await
            .expect_err("foreign run refuses");
        assert!(matches!(error, CheckpointError::ObjectVerification));
        assert_eq!(fixture.participant.commit_notifications(), 0);
        assert!(
            fixture
                .coordinator
                .pending_notification_generation()
                .is_some()
        );
        assert_eq!(
            fixture.coordinator.last_pre_cas_routing(),
            Some(PreCasFailureRouting::FailRun)
        );
    });
}

#[test]
fn greater_epoch_receipt_from_another_run_never_reaches_participant() {
    local(async {
        let mut foreign = coordinator_fixture_for_other_run();
        let committed = foreign
            .coordinator
            .commit_barrier(
                barrier_for_run(foreign.run, 1),
                &mut PreparedCheckpointResultInput::empty(),
            )
            .await
            .expect("publish in the foreign run");

        let mut fixture = coordinator_fixture();
        let error = fixture
            .coordinator
            .replay_committed_notifications(&committed)
            .await
            .expect_err("foreign-run generation refuses");
        assert!(matches!(error, CheckpointError::ObjectVerification));
        assert_eq!(fixture.participant.commit_notifications(), 0);
    });
}

fn coordinator_fixture_for_other_run() -> support::CoordinatorFixture {
    support::coordinator_fixture_for_run(run_id(7))
}

#[test]
fn no_participant_is_notified_before_cas() {
    local(async {
        let mut fixture = coordinator_fixture();
        fixture.backend_control.fail_next_commit();
        let error = fixture
            .coordinator
            .commit_barrier(barrier_at(1), &mut PreparedCheckpointResultInput::empty())
            .await
            .expect_err("commit refused");
        assert!(matches!(error, CheckpointError::Storage { .. }));

        assert_eq!(fixture.participant.commit_notifications(), 0);
        assert!(fixture.reporter.acknowledged_roots().is_empty());
        assert!(fixture.latest_generation().await.is_none());
        assert_eq!(fixture.coordinator.expected(), None);
        assert_eq!(
            fixture.coordinator.last_pre_cas_routing(),
            Some(PreCasFailureRouting::Retryable)
        );
    });
}

#[test]
fn bind_failure_before_cas_publishes_nothing_and_retains_the_head() {
    local(async {
        let mut fixture = coordinator_fixture();
        fixture.reporter.refuse_bind();
        let error = fixture
            .coordinator
            .commit_barrier(barrier_at(1), &mut PreparedCheckpointResultInput::empty())
            .await
            .expect_err("bind refusal aborts before CAS");
        assert!(matches!(error, CheckpointError::ObjectVerification));

        assert_eq!(fixture.backend_control.commit_calls(), 0);
        assert!(fixture.latest_generation().await.is_none());
        assert_eq!(fixture.coordinator.expected(), None);
        assert!(fixture.reporter.bound_roots().is_empty());
        assert_eq!(
            fixture.coordinator.last_pre_cas_routing(),
            Some(PreCasFailureRouting::FailRun)
        );
    });
}

#[test]
fn missing_participant_refuses_before_begin_generation() {
    local(async {
        let run = run_id(1);
        // The ledger alone cannot represent the frozen two-owner plan.
        let (reporter, _control) = FakeIssueReporter::new(run);
        let error = StreamingCheckpointCoordinator::new(
            run,
            Box::new(
                aiperf_runtime::streaming::checkpoints::memory::MemoryCheckpointBackend::new(
                    support::backend_limits(),
                )
                .expect("valid memory backend"),
            ),
            expectations(run),
            Vec::new(),
            Box::new(reporter),
            None,
        )
        .err()
        .expect("incomplete participant set refuses at construction");
        assert!(matches!(error, CheckpointError::ParticipantSetMismatch));
    });
}

#[test]
fn duplicate_participant_refuses_before_begin_generation() {
    local(async {
        let run = run_id(1);
        let (first, _first_control) = NotifyingParticipant::new(run, support::PARTICIPANT_ID);
        let (second, _second_control) = NotifyingParticipant::new(run, support::PARTICIPANT_ID);
        let (reporter, _reporter_control) = FakeIssueReporter::new(run);
        let error = StreamingCheckpointCoordinator::new(
            run,
            Box::new(
                aiperf_runtime::streaming::checkpoints::memory::MemoryCheckpointBackend::new(
                    support::backend_limits(),
                )
                .expect("valid memory backend"),
            ),
            expectations(run),
            vec![Box::new(first), Box::new(second)],
            Box::new(reporter),
            None,
        )
        .err()
        .expect("duplicate participant refuses at construction");
        assert!(matches!(error, CheckpointError::ParticipantSetMismatch));
    });
}

#[test]
fn stale_expected_head_is_refused_without_adopting_a_concurrent_advance() {
    local(async {
        let mut fixture = coordinator_fixture();
        let first = fixture
            .coordinator
            .commit_barrier(barrier_at(1), &mut PreparedCheckpointResultInput::empty())
            .await
            .expect("publish first barrier");
        let first_generation = first.generation();

        // A second coordinator over the same store advances the head behind us.
        let mut other = support::coordinator_fixture_for_run(fixture.run);
        other.coordinator = StreamingCheckpointCoordinator::new(
            fixture.run,
            Box::new(fixture.backend.clone()),
            expectations(fixture.run),
            vec![Box::new(
                NotifyingParticipant::new(fixture.run, support::PARTICIPANT_ID).0,
            )],
            Box::new(FakeIssueReporter::new(fixture.run).0),
            Some(first_generation.clone()),
        )
        .expect("second writer over the same store");
        other
            .coordinator
            .commit_barrier(barrier_at(2), &mut PreparedCheckpointResultInput::empty())
            .await
            .expect("concurrent advance");

        let mut inputs = PreparedCheckpointResultInput::empty();
        let error = fixture
            .coordinator
            .commit_barrier(barrier_at(2), &mut inputs)
            .await
            .expect_err("stale expectation refuses");
        assert!(matches!(error, CheckpointError::GenerationConflict { .. }));

        // The advance is refused, never adopted.
        assert_eq!(fixture.coordinator.expected(), Some(&first_generation));
        assert!(inputs.partitions().is_empty());
        assert_eq!(
            fixture.coordinator.last_pre_cas_routing(),
            Some(PreCasFailureRouting::FailRun)
        );
    });
}

#[test]
fn staged_index_root_is_bound_pre_cas_and_acknowledged_once_after_cas() {
    local(async {
        let mut fixture = coordinator_fixture();
        let committed = fixture
            .coordinator
            .commit_barrier(barrier_at(1), &mut PreparedCheckpointResultInput::empty())
            .await
            .expect("publish barrier");

        assert_eq!(
            fixture.reporter.bound_roots(),
            vec![*committed.result_index_root()]
        );
        assert_eq!(
            fixture.reporter.acknowledged_roots(),
            vec![*committed.result_index_root()]
        );
    });
}
