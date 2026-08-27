// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "support/streaming_checkpoint.rs"]
mod support;

use aiperf_runtime::streaming::{
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        BudgetedCheckpointBytes, CheckpointCut, CheckpointEpoch, CheckpointGenerationCandidate,
        CheckpointParticipantId, CheckpointParticipantOwners, CheckpointParticipantPlan,
        CheckpointParticipantPlanError, CheckpointTerminalReason, CommittedParticipantState,
        ParticipantInitialization, ParticipantStateDescriptor, PreparedParticipantState,
        RequiredCheckpointOwner, StreamingCheckpointParticipant,
    },
    identity::ContentDigest,
};
use bytes::{Bytes, BytesMut};

fn id(value: &str) -> CheckpointParticipantId {
    CheckpointParticipantId::new(value)
}

fn required_owners() -> CheckpointParticipantOwners {
    CheckpointParticipantOwners {
        source: Some(id("source")),
        format: Some(id("format")),
        event_time_order_policy: Some(id("event-time-order")),
        session_coordinator: Some(id("session")),
        action_driver_bindings: vec![id("driver/z"), id("driver/a")],
        placement_policy: Some(id("placement-policy")),
        placement_driver: Some(id("placement-driver")),
        active_execution_set: Some(id("active-execution")),
        blocking_owner: Some(id("blocking")),
        result_epoch: Some(id("result-epoch")),
    }
}

#[test]
fn horizon_domains_cannot_be_substituted_and_round_trip() {
    let cut = support::cut_at(7);
    let encoded = serde_json::to_vec(&cut).expect("serialize cut");
    let restored: CheckpointCut = serde_json::from_slice(&encoded).expect("restore cut");
    assert_eq!(restored, cut);
    assert_eq!(restored.decoded.get(), cut.decoded.get());
    assert_eq!(restored.terminal.get(), cut.terminal.get());
}

#[test]
fn all_six_horizons_remain_independent() {
    let mut cut = support::cut_at(1);
    cut.discovered = aiperf_runtime::streaming::checkpoint::DiscoveryHorizon::new(
        aiperf_runtime::streaming::unit::SourcePosition::new(2),
    );
    cut.acquired = aiperf_runtime::streaming::checkpoint::AcquisitionHorizon::new(
        aiperf_runtime::streaming::unit::SourcePosition::new(3),
    );
    cut.decoded = aiperf_runtime::streaming::checkpoint::DecodeHorizon::new(
        aiperf_runtime::streaming::unit::SourcePosition::new(4),
    );
    cut.ordered = aiperf_runtime::streaming::checkpoint::OrderedActionHorizon::new(
        aiperf_runtime::streaming::identity::GlobalSequence::new(5),
    );
    cut.admitted = aiperf_runtime::streaming::checkpoint::AdmissionHorizon::new(
        aiperf_runtime::streaming::identity::GlobalSequence::new(6),
    );
    cut.terminal = aiperf_runtime::streaming::checkpoint::TerminalActionHorizon::new(
        aiperf_runtime::streaming::identity::GlobalSequence::new(7),
    );

    assert_eq!(cut.discovered.get().get(), 2);
    assert_eq!(cut.acquired.get().get(), 3);
    assert_eq!(cut.decoded.get().get(), 4);
    assert_eq!(cut.ordered.get().get(), 5);
    assert_eq!(cut.admitted.get().get(), 6);
    assert_eq!(cut.terminal.get().get(), 7);
}

#[test]
fn participant_plan_sorts_stable_ids_and_rejects_duplicates() {
    let plan = CheckpointParticipantPlan::new(vec![id("z"), id("a"), id("m")])
        .expect("distinct participants");
    assert_eq!(plan.ids(), &[id("a"), id("m"), id("z")]);

    assert_eq!(
        CheckpointParticipantPlan::new(vec![id("same"), id("same")]),
        Err(CheckpointParticipantPlanError::DuplicateParticipant(id(
            "same"
        )))
    );
}

#[test]
fn required_stateful_owner_omission_is_rejected() {
    let mut without_blocking = required_owners();
    without_blocking.blocking_owner = None;
    assert_eq!(
        CheckpointParticipantPlan::from_required_owners(without_blocking),
        Err(CheckpointParticipantPlanError::MissingRequiredOwner(
            RequiredCheckpointOwner::BlockingOwner
        ))
    );

    let mut without_results = required_owners();
    without_results.result_epoch = None;
    assert_eq!(
        CheckpointParticipantPlan::from_required_owners(without_results),
        Err(CheckpointParticipantPlanError::MissingRequiredOwner(
            RequiredCheckpointOwner::ResultEpoch
        ))
    );

    let plan = CheckpointParticipantPlan::from_required_owners(required_owners())
        .expect("complete owner inventory");
    assert_eq!(
        plan.ids(),
        &[
            id("active-execution"),
            id("blocking"),
            id("driver/a"),
            id("driver/z"),
            id("event-time-order"),
            id("format"),
            id("placement-driver"),
            id("placement-policy"),
            id("result-epoch"),
            id("session"),
            id("source"),
        ]
    );

    let mut duplicate_across_roles = required_owners();
    duplicate_across_roles.result_epoch = Some(id("source"));
    assert_eq!(
        CheckpointParticipantPlan::from_required_owners(duplicate_across_roles),
        Err(CheckpointParticipantPlanError::DuplicateParticipant(id(
            "source"
        )))
    );
}

#[test]
fn participant_initialization_is_one_shot() {
    let mut initialization = ParticipantInitialization::default();
    assert!(initialization.initialize_once().is_ok());
    assert_eq!(
        initialization.initialize_once(),
        Err(aiperf_runtime::streaming::checkpoint::CheckpointError::AlreadyInitialized)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn participant_restore_is_one_shot() {
    let mut participant = support::CountingParticipant::new("session", 1);
    participant
        .initialize(None)
        .await
        .expect("first initialize");
    assert_eq!(
        participant.initialize(None).await,
        Err(aiperf_runtime::streaming::checkpoint::CheckpointError::AlreadyInitialized)
    );
}

async fn payload(bytes: &[u8]) -> BudgetedCheckpointBytes {
    let budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: bytes.len(),
    })
    .expect("valid payload budget");
    let lease = budget.acquire(1, bytes.len()).await.expect("payload lease");
    BudgetedCheckpointBytes::new(Bytes::copy_from_slice(bytes), lease)
        .expect("exactly charged bytes")
}

#[tokio::test(flavor = "current_thread")]
async fn tiny_slice_is_normalized_to_compact_owned_checkpoint_storage() {
    let mut large = BytesMut::with_capacity(1024 * 1024);
    large.resize(1024 * 1024, 0xaa);
    let large = large.freeze();
    let tiny = large.slice(512..513);
    let budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: 1,
    })
    .expect("valid tiny budget");
    let lease = budget.acquire(1, 1).await.expect("tiny payload lease");

    let compact = BudgetedCheckpointBytes::new(tiny, lease).expect("compact payload");
    drop(large);
    assert_eq!(compact.as_bytes(), &[0xaa]);
    assert_eq!(compact.retained_allocation_bytes(), 1);
    assert_eq!(compact.charged_bytes(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn committed_state_requires_exact_digest_and_length() {
    let prepared = PreparedParticipantState::new(
        support::run_id(1),
        id("session"),
        "session.v1",
        1,
        support::cut_at(2),
        3,
        payload(b"state").await,
    )
    .expect("prepared state");
    let descriptor = prepared.descriptor().clone();
    let (run, prepared_descriptor, prepared_payload) = prepared.into_parts();
    assert_eq!(run, support::run_id(1));
    assert_eq!(prepared_descriptor, descriptor);
    assert!(CommittedParticipantState::new(run, descriptor.clone(), prepared_payload).is_ok());

    let bad_length = ParticipantStateDescriptor {
        byte_length: 4,
        ..descriptor.clone()
    };
    assert!(matches!(
        CommittedParticipantState::new(support::run_id(1), bad_length, payload(b"state").await,),
        Err(aiperf_runtime::streaming::checkpoint::CheckpointError::ObjectVerification)
    ));

    let bad_digest = ParticipantStateDescriptor {
        content_digest: ContentDigest::from_bytes([0xff; 32]),
        ..descriptor
    };
    assert!(matches!(
        CommittedParticipantState::new(support::run_id(1), bad_digest, payload(b"state").await,),
        Err(aiperf_runtime::streaming::checkpoint::CheckpointError::ObjectVerification)
    ));
}

async fn descriptor_for(
    participant: &str,
    represented_cut: CheckpointCut,
    bytes: &[u8],
) -> ParticipantStateDescriptor {
    PreparedParticipantState::new(
        support::run_id(1),
        id(participant),
        format!("{participant}.v1"),
        1,
        represented_cut,
        1,
        payload(bytes).await,
    )
    .expect("prepared descriptor")
    .into_parts()
    .1
}

fn candidate_for_run(run: u8) -> CheckpointGenerationCandidate {
    let cut = support::cut_at(7);
    let descriptor = ParticipantStateDescriptor {
        participant_id: id("session"),
        schema_id: "session.v1".into(),
        schema_version: 1,
        represented_cut: cut.clone(),
        content_digest: ContentDigest::from_bytes([0x44; 32]),
        item_count: 1,
        byte_length: 4,
    };
    let plan = CheckpointParticipantPlan::new([id("session")]).expect("valid plan");
    CheckpointGenerationCandidate::new(
        support::run_id(run),
        CheckpointEpoch::new(7),
        None,
        cut,
        &plan,
        ContentDigest::from_bytes([0x11; 32]),
        ContentDigest::from_bytes([0x12; 32]),
        vec![descriptor],
        ContentDigest::from_bytes([0x55; 32]),
        false,
        None,
    )
    .expect("valid run-bound generation candidate")
}

#[test]
fn identical_generation_content_in_distinct_runs_has_distinct_digest() {
    let first = candidate_for_run(1);
    let second = candidate_for_run(2);

    assert_ne!(first.generation().digest(), second.generation().digest());
}

#[test]
fn serialized_candidate_rejects_tampered_run() {
    let candidate = candidate_for_run(1);
    let mut serialized = serde_json::to_value(&candidate).expect("serialize candidate");
    serialized["run"] =
        serde_json::to_value(support::run_id(2)).expect("serialize replacement run");

    assert!(serde_json::from_value::<CheckpointGenerationCandidate>(serialized).is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn generation_candidate_is_canonical_and_digest_verified() {
    let cut = support::cut_at(9);
    let a = descriptor_for("a", cut.clone(), b"a-state").await;
    let b = descriptor_for("b", cut.clone(), b"b-state").await;
    let plan = CheckpointParticipantPlan::new([id("b"), id("a")]).expect("valid plan");
    let execution_plan = ContentDigest::from_bytes([0x51; 32]);
    let result_plan = ContentDigest::from_bytes([0x52; 32]);
    let result_root = ContentDigest::from_bytes([0x61; 32]);

    let staged_ba = CheckpointGenerationCandidate::new(
        support::run_id(1),
        CheckpointEpoch::new(4),
        Some(ContentDigest::from_bytes([0x41; 32])),
        cut.clone(),
        &plan,
        execution_plan,
        result_plan,
        vec![b.clone(), a.clone()],
        result_root,
        false,
        None,
    )
    .expect("canonical generation");
    let staged_ab = CheckpointGenerationCandidate::new(
        support::run_id(1),
        CheckpointEpoch::new(4),
        Some(ContentDigest::from_bytes([0x41; 32])),
        cut,
        &plan,
        execution_plan,
        result_plan,
        vec![a, b],
        result_root,
        false,
        None,
    )
    .expect("same canonical generation");

    assert_eq!(staged_ba, staged_ab);
    staged_ba
        .verify_against(&support::run_id(1), &plan, &execution_plan, &result_plan)
        .expect("generation digest and plan bindings verify");

    let mut encoded = serde_json::to_value(&staged_ba).expect("serialize generation");
    let restored: CheckpointGenerationCandidate =
        serde_json::from_value(encoded.clone()).expect("restore verified generation");
    assert_eq!(restored, staged_ba);
    encoded["generation"]["digest"] =
        serde_json::to_value(ContentDigest::from_bytes([0xff; 32])).expect("serialize digest");
    assert!(serde_json::from_value::<CheckpointGenerationCandidate>(encoded).is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn semantic_plan_digests_change_generation_identity_and_verify_exactly() {
    let barrier = support::barrier_at(8);
    let cut = barrier.cut;
    let descriptor = descriptor_for("session", cut.clone(), b"state").await;
    let plan = CheckpointParticipantPlan::new([id("session")]).expect("valid plan");
    let execution_plan = barrier.plan_digest;
    let result_plan = ContentDigest::from_bytes([0x72; 32]);
    let construct = |execution_plan, result_plan| {
        CheckpointGenerationCandidate::new(
            support::run_id(1),
            CheckpointEpoch::new(2),
            None,
            cut.clone(),
            &plan,
            execution_plan,
            result_plan,
            vec![descriptor.clone()],
            ContentDigest::from_bytes([0x73; 32]),
            false,
            None,
        )
        .expect("valid candidate")
    };
    let baseline = construct(execution_plan, result_plan);
    let changed_execution = construct(ContentDigest::from_bytes([0x81; 32]), result_plan);
    let changed_result = construct(execution_plan, ContentDigest::from_bytes([0x82; 32]));

    assert_ne!(baseline.generation(), changed_execution.generation());
    assert_ne!(baseline.generation(), changed_result.generation());
    assert_eq!(baseline.execution_plan_digest(), &execution_plan);
    assert_eq!(baseline.result_plan_digest(), &result_plan);
    assert!(
        baseline
            .verify_against(
                &support::run_id(1),
                &plan,
                &ContentDigest::from_bytes([0x81; 32]),
                &result_plan,
            )
            .is_err()
    );
    assert!(
        baseline
            .verify_against(
                &support::run_id(1),
                &plan,
                &execution_plan,
                &ContentDigest::from_bytes([0x82; 32]),
            )
            .is_err()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn self_valid_candidate_from_another_participant_plan_is_refused() {
    let cut = support::cut_at(5);
    let other_descriptor = descriptor_for("other", cut.clone(), b"other").await;
    let other_plan = CheckpointParticipantPlan::new([id("other")]).expect("other plan");
    let expected_plan = CheckpointParticipantPlan::new([id("session")]).expect("expected plan");
    let execution_plan = ContentDigest::from_bytes([0x31; 32]);
    let result_plan = ContentDigest::from_bytes([0x32; 32]);
    let candidate = CheckpointGenerationCandidate::new(
        support::run_id(1),
        CheckpointEpoch::new(1),
        None,
        cut,
        &other_plan,
        execution_plan,
        result_plan,
        vec![other_descriptor],
        ContentDigest::from_bytes([0x33; 32]),
        false,
        None,
    )
    .expect("self-valid candidate from another plan");

    candidate
        .verify_against(
            &support::run_id(1),
            &other_plan,
            &execution_plan,
            &result_plan,
        )
        .expect("candidate is self-valid for its authored plan");
    assert!(matches!(
        candidate.verify_against(
            &support::run_id(1),
            &expected_plan,
            &execution_plan,
            &result_plan,
        ),
        Err(aiperf_runtime::streaming::checkpoint::CheckpointError::ParticipantSetMismatch)
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn generation_candidate_rejects_invalid_participants_and_terminal_state() {
    let cut = support::cut_at(3);
    let a = descriptor_for("a", cut.clone(), b"a").await;
    let b = descriptor_for("b", cut.clone(), b"b").await;
    let plan = CheckpointParticipantPlan::new([id("a"), id("b")]).expect("valid plan");
    let root = ContentDigest::from_bytes([0x21; 32]);

    assert!(matches!(
        CheckpointGenerationCandidate::new(
            support::run_id(1),
            CheckpointEpoch::new(1),
            None,
            cut.clone(),
            &plan,
            ContentDigest::from_bytes([0x11; 32]),
            ContentDigest::from_bytes([0x12; 32]),
            vec![a.clone(), a.clone()],
            root,
            false,
            None,
        ),
        Err(aiperf_runtime::streaming::checkpoint::CheckpointError::ParticipantSetMismatch)
    ));
    let wrong_cut = ParticipantStateDescriptor {
        represented_cut: support::cut_at(2),
        ..a.clone()
    };
    assert!(matches!(
        CheckpointGenerationCandidate::new(
            support::run_id(1),
            CheckpointEpoch::new(1),
            None,
            cut.clone(),
            &plan,
            ContentDigest::from_bytes([0x11; 32]),
            ContentDigest::from_bytes([0x12; 32]),
            vec![wrong_cut, b.clone()],
            root,
            false,
            None,
        ),
        Err(aiperf_runtime::streaming::checkpoint::CheckpointError::ParticipantSetMismatch)
    ));
    assert!(matches!(
        CheckpointGenerationCandidate::new(
            support::run_id(1),
            CheckpointEpoch::new(1),
            None,
            cut.clone(),
            &plan,
            ContentDigest::from_bytes([0x11; 32]),
            ContentDigest::from_bytes([0x12; 32]),
            vec![a.clone()],
            root,
            false,
            None,
        ),
        Err(aiperf_runtime::streaming::checkpoint::CheckpointError::ParticipantSetMismatch)
    ));
    assert!(
        CheckpointGenerationCandidate::new(
            support::run_id(1),
            CheckpointEpoch::new(1),
            None,
            cut.clone(),
            &plan,
            ContentDigest::from_bytes([0x11; 32]),
            ContentDigest::from_bytes([0x12; 32]),
            vec![a.clone(), b.clone()],
            root,
            true,
            None,
        )
        .is_err()
    );
    assert!(
        CheckpointGenerationCandidate::new(
            support::run_id(1),
            CheckpointEpoch::new(1),
            None,
            cut.clone(),
            &plan,
            ContentDigest::from_bytes([0x11; 32]),
            ContentDigest::from_bytes([0x12; 32]),
            vec![a.clone(), b.clone()],
            root,
            false,
            Some(CheckpointTerminalReason::Completed),
        )
        .is_err()
    );
    let terminal = CheckpointGenerationCandidate::new(
        support::run_id(1),
        CheckpointEpoch::new(1),
        None,
        cut,
        &plan,
        ContentDigest::from_bytes([0x11; 32]),
        ContentDigest::from_bytes([0x12; 32]),
        vec![a, b],
        root,
        true,
        Some(CheckpointTerminalReason::Completed),
    )
    .expect("final generation has an exact terminal reason");
    assert!(terminal.is_final());
    assert_eq!(
        terminal.terminal_reason(),
        Some(CheckpointTerminalReason::Completed)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn participant_view_is_non_destructive_before_backend_commit() {
    let mut participant = support::CountingParticipant::new("session", 4);
    participant
        .initialize(None)
        .await
        .expect("fresh initialize");
    let prepared = participant
        .checkpoint_view(&support::barrier_at(4))
        .await
        .expect("checkpoint view");
    assert_eq!(participant.released_items(), 0);
    assert_eq!(prepared.descriptor().participant_id, id("session"));
}
