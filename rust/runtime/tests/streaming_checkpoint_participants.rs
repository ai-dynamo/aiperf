// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "support/streaming_checkpoint.rs"]
mod support;

use aiperf_runtime::streaming::{
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        BudgetedCheckpointBytes, CheckpointCut, CheckpointEpoch, CheckpointParticipantId,
        CheckpointParticipantOwners, CheckpointParticipantPlan, CheckpointParticipantPlanError,
        CheckpointTerminalReason, CommittedCheckpointGeneration, CommittedParticipantReceipt,
        CommittedParticipantState, ParticipantInitialization, ParticipantStateDescriptor,
        PreparedParticipantState, RequiredCheckpointOwner, StreamingCheckpointParticipant,
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
        id("session"),
        "session.v1",
        1,
        support::cut_at(2),
        3,
        payload(b"state").await,
    )
    .expect("prepared state");
    let descriptor = prepared.descriptor().clone();
    let (_, prepared_payload) = prepared.into_parts();
    assert!(CommittedParticipantState::new(descriptor.clone(), prepared_payload).is_ok());

    let bad_length = ParticipantStateDescriptor {
        byte_length: 4,
        ..descriptor.clone()
    };
    assert!(matches!(
        CommittedParticipantState::new(bad_length, payload(b"state").await),
        Err(aiperf_runtime::streaming::checkpoint::CheckpointError::ObjectVerification)
    ));

    let bad_digest = ParticipantStateDescriptor {
        content_digest: ContentDigest::from_bytes([0xff; 32]),
        ..descriptor
    };
    assert!(matches!(
        CommittedParticipantState::new(bad_digest, payload(b"state").await),
        Err(aiperf_runtime::streaming::checkpoint::CheckpointError::ObjectVerification)
    ));
}

async fn descriptor_for(
    participant: &str,
    represented_cut: CheckpointCut,
    bytes: &[u8],
) -> ParticipantStateDescriptor {
    PreparedParticipantState::new(
        id(participant),
        format!("{participant}.v1"),
        1,
        represented_cut,
        1,
        payload(bytes).await,
    )
    .expect("prepared descriptor")
    .into_parts()
    .0
}

#[tokio::test(flavor = "current_thread")]
async fn committed_generation_is_canonical_and_digest_verified() {
    let cut = support::cut_at(9);
    let a = descriptor_for("a", cut.clone(), b"a-state").await;
    let b = descriptor_for("b", cut.clone(), b"b-state").await;
    let plan = CheckpointParticipantPlan::new([id("b"), id("a")]).expect("valid plan");
    let result_root = ContentDigest::from_bytes([0x61; 32]);

    let staged_ba = CommittedCheckpointGeneration::new(
        CheckpointEpoch::new(4),
        Some(ContentDigest::from_bytes([0x41; 32])),
        cut.clone(),
        &plan,
        vec![b.clone(), a.clone()],
        result_root,
        false,
        None,
    )
    .expect("canonical generation");
    let staged_ab = CommittedCheckpointGeneration::new(
        CheckpointEpoch::new(4),
        Some(ContentDigest::from_bytes([0x41; 32])),
        cut,
        &plan,
        vec![a, b],
        result_root,
        false,
        None,
    )
    .expect("same canonical generation");

    assert_eq!(staged_ba, staged_ab);
    assert_eq!(
        staged_ba
            .participant_descriptors()
            .iter()
            .map(|descriptor| descriptor.participant_id.as_str())
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
    staged_ba.verify().expect("generation digest verifies");

    let mut encoded = serde_json::to_value(&staged_ba).expect("serialize generation");
    let restored: CommittedCheckpointGeneration =
        serde_json::from_value(encoded.clone()).expect("restore verified generation");
    assert_eq!(restored, staged_ba);
    encoded["generation"]["digest"] =
        serde_json::to_value(ContentDigest::from_bytes([0xff; 32])).expect("serialize digest");
    assert!(serde_json::from_value::<CommittedCheckpointGeneration>(encoded).is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn committed_generation_rejects_invalid_participants_and_terminal_state() {
    let cut = support::cut_at(3);
    let a = descriptor_for("a", cut.clone(), b"a").await;
    let b = descriptor_for("b", cut.clone(), b"b").await;
    let plan = CheckpointParticipantPlan::new([id("a"), id("b")]).expect("valid plan");
    let root = ContentDigest::from_bytes([0x21; 32]);

    assert!(matches!(
        CommittedCheckpointGeneration::new(
            CheckpointEpoch::new(1),
            None,
            cut.clone(),
            &plan,
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
        CommittedCheckpointGeneration::new(
            CheckpointEpoch::new(1),
            None,
            cut.clone(),
            &plan,
            vec![wrong_cut, b.clone()],
            root,
            false,
            None,
        ),
        Err(aiperf_runtime::streaming::checkpoint::CheckpointError::ParticipantSetMismatch)
    ));
    assert!(matches!(
        CommittedCheckpointGeneration::new(
            CheckpointEpoch::new(1),
            None,
            cut.clone(),
            &plan,
            vec![a.clone()],
            root,
            false,
            None,
        ),
        Err(aiperf_runtime::streaming::checkpoint::CheckpointError::ParticipantSetMismatch)
    ));
    assert!(
        CommittedCheckpointGeneration::new(
            CheckpointEpoch::new(1),
            None,
            cut.clone(),
            &plan,
            vec![a.clone(), b.clone()],
            root,
            true,
            None,
        )
        .is_err()
    );
    assert!(
        CommittedCheckpointGeneration::new(
            CheckpointEpoch::new(1),
            None,
            cut.clone(),
            &plan,
            vec![a.clone(), b.clone()],
            root,
            false,
            Some(CheckpointTerminalReason::Completed),
        )
        .is_err()
    );
    let terminal = CommittedCheckpointGeneration::new(
        CheckpointEpoch::new(1),
        None,
        cut,
        &plan,
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
async fn committed_receipt_requires_the_exact_generation_descriptor() {
    let cut = support::cut_at(6);
    let descriptor = descriptor_for("session", cut, b"state").await;
    let generation = support::generation_for(descriptor.clone(), 7, None);
    let receipt = CommittedParticipantReceipt::new(&generation, &descriptor)
        .expect("generation contains exact descriptor");
    assert_eq!(receipt.generation(), generation.generation_ref());
    assert_eq!(receipt.participant_id(), &descriptor.participant_id);
    assert_eq!(receipt.represented_cut(), &descriptor.represented_cut);

    let mismatched = ParticipantStateDescriptor {
        schema_version: descriptor.schema_version + 1,
        ..descriptor
    };
    assert!(CommittedParticipantReceipt::new(&generation, &mismatched).is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn participant_view_is_non_destructive_until_commit_receipt() {
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
    let receipt = support::receipt_for(&prepared);
    participant
        .checkpoint_committed(&receipt)
        .await
        .expect("commit notification");
    assert_eq!(participant.released_items(), 4);
    participant
        .checkpoint_committed(&receipt)
        .await
        .expect("idempotent commit notification");
    assert_eq!(participant.released_items(), 4);
    assert_eq!(participant.commit_notifications(), 1);

    let next_generation = support::generation_for(
        prepared.descriptor().clone(),
        2,
        Some(receipt.generation().digest().to_owned()),
    );
    let next_receipt = CommittedParticipantReceipt::new(&next_generation, prepared.descriptor())
        .expect("same descriptor committed in a new generation");
    participant
        .checkpoint_committed(&next_receipt)
        .await
        .expect("new generation is not mistaken for a duplicate callback");
    assert_eq!(participant.commit_notifications(), 2);
}
