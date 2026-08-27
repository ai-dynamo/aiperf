// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::streaming::checkpoint::{
    CheckpointBackendBudgetFailureCode, CheckpointBackendBudgetKind, CheckpointError,
};
use aiperf_runtime::streaming::checkpoints::memory::MemoryCheckpointBackend;
use aiperf_runtime::streaming::results::{
    ResultIndexReadBudget, ResultProjectionId, ResultSegmentDescriptor,
};
use std::num::{NonZeroU64, NonZeroUsize};

#[path = "support/streaming_checkpoint.rs"]
mod support;

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
    MemoryCheckpointBackend::new(support::backend_limits_with_each_capacity(boundary)).unwrap();

    let first_unrepresentable = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
    assert!(
        support::invalid_backend_limits()
            .iter()
            .any(|(limits, _, code)| {
                support::contains_capacity(*limits, first_unrepresentable)
                    && *code == CheckpointBackendBudgetFailureCode::Unrepresentable
            })
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
        short.descriptor_charged_bytes()
    );
    assert_eq!(
        long_budget.snapshot().used_bytes,
        long.descriptor_charged_bytes()
    );

    let (wrapped_descriptor, payload) = long.into_parts();
    assert_eq!(
        long_budget.snapshot().used_bytes,
        wrapped_descriptor.charged_bytes()
    );
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
    let held_summary = backend.hold_all_result_summary_capacity().await.unwrap();

    let mut pending = Box::pin(transaction.stage_results(&mut partitions));
    assert!(matches!(
        futures::poll!(&mut pending),
        std::task::Poll::Pending
    ));
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
    assert_eq!(
        transaction.staged_result_root(),
        Some(prepared.index_root())
    );
    assert_eq!(backend.budget_snapshots().result_summaries.used_items, 1);
    drop(prepared);
    assert_eq!(backend.budget_snapshots().result_summaries.used_items, 0);
    assert_eq!(backend.budget_snapshots().prepared_indexes.used_items, 1);
}

async fn fully_staged_after(
    backend: &MemoryCheckpointBackend,
    run: aiperf_runtime::streaming::checkpoint::StreamRunIdentity,
    previous: aiperf_runtime::streaming::checkpoint::CheckpointGeneration,
    epoch: u64,
) -> aiperf_runtime::streaming::checkpoints::memory::MemoryGenerationTransaction {
    let mut transaction = backend
        .begin_generation(run, Some(previous), support::expectations(run))
        .await
        .unwrap();
    transaction
        .stage_participant(support::prepared_participant(run, epoch).await)
        .await
        .unwrap();
    transaction.stage_results(&mut Vec::new()).await.unwrap();
    transaction
}

#[tokio::test(flavor = "current_thread")]
async fn stale_writer_cannot_merge_or_replace_head() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    let first = support::commit_empty(&backend, run, None, 1).await.unwrap();
    let stale = fully_staged_after(&backend, run, first.generation(), 2).await;
    let current = fully_staged_after(&backend, run, first.generation(), 2).await;
    current
        .commit(support::metadata_with_lineage(Some(first.generation()), 2))
        .await
        .unwrap();
    let error = stale
        .commit(support::metadata_with_lineage(Some(first.generation()), 2))
        .await
        .unwrap_err();
    assert!(matches!(error, CheckpointError::GenerationConflict { .. }));
}

#[tokio::test(flavor = "current_thread")]
async fn commit_metadata_must_match_frozen_predecessor_and_exact_next_epoch() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    let baseline = support::commit_empty(&backend, run, None, 1).await.unwrap();
    let head = baseline.generation();
    let before = backend.live_budget_usage();

    for metadata in [
        support::metadata_with_lineage(None, 2),
        support::metadata_with_lineage(Some(support::same_epoch_wrong_digest(&head)), 2),
        support::metadata_with_lineage(Some(head.clone()), 3),
    ] {
        let transaction = fully_staged_after(&backend, run, head.clone(), 2).await;
        assert_eq!(
            transaction.commit(metadata).await.unwrap_err(),
            CheckpointError::ObjectVerification
        );
        assert_eq!(
            backend
                .open_latest(&run, &support::expectations(run))
                .await
                .unwrap()
                .unwrap()
                .generation()
                .generation_ref(),
            &head,
        );
    }
    assert_eq!(backend.live_budget_usage().storage, before.storage);
}

#[tokio::test(flavor = "current_thread")]
async fn dropped_transaction_publishes_nothing_and_releases_budget() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    let transaction = backend
        .begin_generation(run, None, support::expectations(run))
        .await
        .unwrap();
    assert_eq!(backend.prepared_transactions(), 1);
    drop(transaction);
    assert_eq!(backend.prepared_transactions(), 0);
    assert!(
        backend
            .open_latest(&run, &support::expectations(run))
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn empty_generations_and_heads_are_isolated_by_logical_run() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let first_run = support::run_id(1);
    let second_run = support::run_id(2);
    let first = support::commit_empty(&backend, first_run, None, 1)
        .await
        .unwrap();
    let second = support::commit_empty(&backend, second_run, None, 1)
        .await
        .unwrap();
    assert_ne!(first.generation().digest(), second.generation().digest());
    assert_eq!(
        backend
            .open_latest(&first_run, &support::expectations(first_run))
            .await
            .unwrap()
            .unwrap()
            .generation()
            .generation_ref(),
        first.generation_ref(),
    );
    assert_eq!(
        backend
            .open_latest(&second_run, &support::expectations(second_run))
            .await
            .unwrap()
            .unwrap()
            .generation()
            .generation_ref(),
        second.generation_ref(),
    );
}

#[tokio::test(flavor = "current_thread")]
async fn explicit_expectation_and_result_runs_must_match_transaction_run() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    let other = support::run_id(2);
    assert!(matches!(
        backend
            .begin_generation(run, None, support::expectations(other))
            .await,
        Err(CheckpointError::ObjectVerification)
    ));
    assert!(matches!(
        backend
            .open_latest(&run, &support::expectations(other))
            .await,
        Err(CheckpointError::ObjectVerification)
    ));

    let mut transaction = backend
        .begin_generation(run, None, support::expectations(run))
        .await
        .unwrap();
    let mut foreign = vec![support::result_partition(other, 1).await];
    assert!(matches!(
        transaction.stage_results(&mut foreign).await,
        Err(CheckpointError::ObjectVerification)
    ));
    assert_eq!(foreign.len(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn commit_requires_exact_participants_and_one_canonical_result_epoch() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    let omitted_results = support::transaction_with_all_participants(&backend, run).await;
    assert!(
        omitted_results
            .commit(support::metadata_at(1))
            .await
            .is_err()
    );

    let mut omitted_participant = backend
        .begin_generation(run, None, support::expectations(run))
        .await
        .unwrap();
    let empty = omitted_participant
        .stage_results(&mut Vec::new())
        .await
        .unwrap();
    assert_eq!(empty.item_count(), 0);
    assert_eq!(empty.byte_length(), 0);
    assert!(matches!(
        omitted_participant.commit(support::metadata_at(1)).await,
        Err(CheckpointError::ParticipantSetMismatch)
    ));

    let mut exact = support::transaction_with_all_participants(&backend, run).await;
    let mut no_partitions = Vec::new();
    exact.stage_results(&mut no_partitions).await.unwrap();
    assert!(exact.stage_results(&mut no_partitions).await.is_err());
    exact.commit(support::metadata_at(1)).await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn result_epoch_must_match_commit_epoch() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    let mut transaction = support::transaction_with_all_participants(&backend, run).await;
    transaction
        .stage_results(&mut vec![support::result_partition(run, 2).await])
        .await
        .unwrap();
    assert!(matches!(
        transaction.commit(support::metadata_at(1)).await,
        Err(CheckpointError::ObjectVerification)
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn committed_result_can_be_scanned_and_read_under_independent_read_leases() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    support::commit_with_segment(&backend, run, None, 1)
        .await
        .unwrap();
    let reader = backend
        .open_latest(&run, &support::expectations(run))
        .await
        .unwrap()
        .unwrap();
    let page = reader
        .scan_result_index(
            None,
            ResultIndexReadBudget {
                max_items: NonZeroUsize::new(1).unwrap(),
                max_bytes: NonZeroU64::new(u64::MAX).unwrap(),
            },
        )
        .await
        .unwrap();
    assert_eq!(page.descriptors().len(), 1);
    let descriptor: ResultSegmentDescriptor = page.descriptors()[0].clone();
    let segment = reader.read_segment(&descriptor).await.unwrap();
    assert_eq!(segment.payload_bytes(), b"result-payload");
    assert!(backend.live_budget_usage().reads.used_items >= 3);
}

#[tokio::test(flavor = "current_thread")]
async fn oversized_next_descriptor_refuses_before_backend_read_budget() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    support::commit_with_segment(&backend, run, None, 1)
        .await
        .unwrap();
    let reader = backend
        .open_latest(&run, &support::expectations(run))
        .await
        .unwrap()
        .unwrap();
    let before = backend.live_budget_usage().reads;
    let error = reader
        .scan_result_index(
            None,
            ResultIndexReadBudget {
                max_items: NonZeroUsize::new(1).unwrap(),
                max_bytes: NonZeroU64::new(1).unwrap(),
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CheckpointError::ResultIndexReadBudgetTooSmall { max_bytes: 1, .. }
    ));
    assert_eq!(backend.live_budget_usage().reads, before);
}

#[tokio::test(flavor = "current_thread")]
async fn maximum_frozen_epoch_refuses_overflow_before_state_access() {
    use aiperf_runtime::streaming::{
        checkpoint::{CheckpointEpoch, CheckpointGeneration},
        identity::ContentDigest,
    };

    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    let maximum = CheckpointGeneration::new(
        CheckpointEpoch::new(u64::MAX),
        ContentDigest::from_bytes([0xee; 32]),
    );
    let mut transaction = backend
        .begin_generation(run, Some(maximum.clone()), support::expectations(run))
        .await
        .unwrap();
    transaction
        .stage_participant(support::prepared_participant(run, 1).await)
        .await
        .unwrap();
    transaction.stage_results(&mut Vec::new()).await.unwrap();
    backend.reset_test_state_accesses();
    let mut metadata = support::metadata_at(1);
    metadata.previous = Some(maximum.clone());
    metadata.epoch = CheckpointEpoch::new(u64::MAX);

    assert_eq!(
        transaction.commit(metadata).await.unwrap_err(),
        CheckpointError::GenerationEpochOverflow { previous: maximum },
    );
    assert_eq!(backend.test_state_accesses(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn storage_capacity_refusal_is_typed_and_publishes_nothing() {
    let backend =
        MemoryCheckpointBackend::new(support::backend_limits_with_storage_bytes(1)).unwrap();
    let run = support::run_id(1);
    let mut transaction = support::transaction_with_all_participants(&backend, run).await;
    transaction
        .stage_results(&mut vec![support::result_partition(run, 1).await])
        .await
        .unwrap();

    assert!(matches!(
        transaction.commit(support::metadata_at(1)).await,
        Err(CheckpointError::BackendBudget {
            budget: CheckpointBackendBudgetKind::Storage,
            code: CheckpointBackendBudgetFailureCode::ByteCapacity,
        })
    ));
    assert!(
        backend
            .open_latest(&run, &support::expectations(run))
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(backend.immutable_object_inventory(&run).total_count(), 0);
    assert_eq!(backend.live_budget_usage().storage.used_items, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn sufficient_page_limit_does_not_hide_backend_read_capacity_refusal() {
    let probe = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    support::commit_with_segment(&probe, run, None, 1)
        .await
        .unwrap();
    let reader = probe
        .open_latest(&run, &support::expectations(run))
        .await
        .unwrap()
        .unwrap();
    let probe_page = reader
        .scan_result_index(
            None,
            ResultIndexReadBudget {
                max_items: NonZeroUsize::new(1).unwrap(),
                max_bytes: NonZeroU64::new(u64::MAX).unwrap(),
            },
        )
        .await
        .unwrap();
    let required = usize::try_from(probe_page.charged_bytes()).unwrap();
    drop(probe_page);
    drop(reader);

    let backend =
        MemoryCheckpointBackend::new(support::backend_limits_with_read_bytes(required - 1))
            .unwrap();
    support::commit_with_segment(&backend, run, None, 1)
        .await
        .unwrap();
    let reader = backend
        .open_latest(&run, &support::expectations(run))
        .await
        .unwrap()
        .unwrap();
    let before = backend.live_budget_usage().reads;
    assert!(matches!(
        reader
            .scan_result_index(
                None,
                ResultIndexReadBudget {
                    max_items: NonZeroUsize::new(1).unwrap(),
                    max_bytes: NonZeroU64::new(u64::try_from(required).unwrap()).unwrap(),
                },
            )
            .await,
        Err(CheckpointError::BackendBudget {
            budget: CheckpointBackendBudgetKind::Read,
            code: CheckpointBackendBudgetFailureCode::ByteCapacity,
        })
    ));
    assert_eq!(backend.live_budget_usage().reads, before);
}

#[tokio::test(flavor = "current_thread")]
async fn invalid_cursor_refuses_before_page_or_backend_budget() {
    use aiperf_runtime::streaming::{identity::ContentDigest, results::ResultIndexCursor};

    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    support::commit_with_segment(&backend, run, None, 1)
        .await
        .unwrap();
    let reader = backend
        .open_latest(&run, &support::expectations(run))
        .await
        .unwrap()
        .unwrap();
    let root = *reader.generation().result_index_root();
    let before = backend.live_budget_usage().reads;
    for cursor in [
        ResultIndexCursor {
            root: ContentDigest::from_bytes([0xaa; 32]),
            block: root,
            item_offset: 0,
        },
        ResultIndexCursor {
            root,
            block: ContentDigest::from_bytes([0xbb; 32]),
            item_offset: 0,
        },
        ResultIndexCursor {
            root,
            block: root,
            item_offset: 99,
        },
    ] {
        assert!(matches!(
            reader
                .scan_result_index(
                    Some(cursor),
                    ResultIndexReadBudget {
                        max_items: NonZeroUsize::new(1).unwrap(),
                        max_bytes: NonZeroU64::new(1).unwrap(),
                    },
                )
                .await,
            Err(CheckpointError::ObjectVerification)
        ));
        assert_eq!(backend.live_budget_usage().reads, before);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn fault_after_prevalidation_occurs_before_publication_and_changes_nothing() {
    use aiperf_runtime::streaming::checkpoints::memory::TestMemoryFault;

    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    let baseline = support::commit_with_segment(&backend, run, None, 1)
        .await
        .unwrap();
    let head = baseline.generation();
    let inventory = backend.immutable_object_inventory(&run);
    let usage = backend.live_budget_usage();
    let transaction = fully_staged_after(&backend, run, head.clone(), 2).await;
    backend.arm_test_fault(TestMemoryFault::AfterPrevalidationBeforePublication);

    assert!(matches!(
        transaction
            .commit(support::metadata_with_lineage(Some(head.clone()), 2))
            .await,
        Err(CheckpointError::Storage { .. })
    ));
    assert!(backend.test_fault_was_reached(TestMemoryFault::AfterPrevalidationBeforePublication));
    assert_eq!(backend.immutable_object_inventory(&run), inventory);
    assert_eq!(backend.live_budget_usage().storage, usage.storage);
    assert_eq!(
        backend
            .open_latest(&run, &support::expectations(run))
            .await
            .unwrap()
            .unwrap()
            .generation()
            .generation_ref(),
        &head,
    );
}

#[tokio::test(flavor = "current_thread")]
async fn existing_immutable_objects_do_not_grant_cross_generation_or_run_read_authority() {
    let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
    let run = support::run_id(1);
    let other = support::run_id(2);
    let superseded = support::commit_with_segment(&backend, run, None, 1)
        .await
        .unwrap();
    let superseded_reader = backend
        .open_latest(&run, &support::expectations(run))
        .await
        .unwrap()
        .unwrap();
    let superseded_page = superseded_reader
        .scan_result_index(
            None,
            ResultIndexReadBudget {
                max_items: NonZeroUsize::new(1).unwrap(),
                max_bytes: NonZeroU64::new(u64::MAX).unwrap(),
            },
        )
        .await
        .unwrap();
    let superseded_descriptor = superseded_page.descriptors()[0].clone();
    let superseded_participant = superseded.participant_descriptors()[0].clone();
    drop(superseded_page);
    drop(superseded_reader);

    support::commit_with_segment(&backend, run, Some(superseded.generation()), 2)
        .await
        .unwrap();
    let foreign = support::commit_with_segment(&backend, other, None, 1)
        .await
        .unwrap();
    let foreign_reader = backend
        .open_latest(&other, &support::expectations(other))
        .await
        .unwrap()
        .unwrap();
    let foreign_page = foreign_reader
        .scan_result_index(
            None,
            ResultIndexReadBudget {
                max_items: NonZeroUsize::new(1).unwrap(),
                max_bytes: NonZeroU64::new(u64::MAX).unwrap(),
            },
        )
        .await
        .unwrap();
    let foreign_descriptor = foreign_page.descriptors()[0].clone();
    let foreign_participant = foreign.participant_descriptors()[0].clone();
    drop(foreign_page);
    drop(foreign_reader);

    let reader = backend
        .open_latest(&run, &support::expectations(run))
        .await
        .unwrap()
        .unwrap();
    let reads_before = backend.live_budget_usage().reads;
    for descriptor in [&superseded_descriptor, &foreign_descriptor] {
        assert_eq!(
            reader.read_segment(descriptor).await.unwrap_err(),
            CheckpointError::ObjectVerification,
        );
        assert_eq!(backend.live_budget_usage().reads, reads_before);
    }
    for descriptor in [&superseded_participant, &foreign_participant] {
        assert_eq!(
            reader.read_participant(descriptor).await.unwrap_err(),
            CheckpointError::ObjectVerification,
        );
        assert_eq!(backend.live_budget_usage().reads, reads_before);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn projection_allocation_participates_in_result_index_read_charge() {
    async fn page_charge(projection: &str) -> u64 {
        let backend = MemoryCheckpointBackend::new(support::backend_limits()).unwrap();
        let run = support::run_id(1);
        let mut transaction = support::transaction_with_all_participants(&backend, run).await;
        transaction
            .stage_results(&mut vec![
                support::result_partition_with_projection_for(run, 1, projection)
                    .await
                    .1,
            ])
            .await
            .unwrap();
        transaction.commit(support::metadata_at(1)).await.unwrap();
        let reader = backend
            .open_latest(&run, &support::expectations(run))
            .await
            .unwrap()
            .unwrap();
        reader
            .scan_result_index(
                None,
                ResultIndexReadBudget {
                    max_items: NonZeroUsize::new(1).unwrap(),
                    max_bytes: NonZeroU64::new(u64::MAX).unwrap(),
                },
            )
            .await
            .unwrap()
            .charged_bytes()
    }

    let short = page_charge("p").await;
    let long = page_charge("projection-with-retained-bytes").await;
    assert_eq!(
        long - short,
        u64::try_from("projection-with-retained-bytes".len() - "p".len()).unwrap(),
    );
}

#[tokio::test(flavor = "current_thread")]
async fn memory_backend_conforms_to_shared_pre_io_lineage_validation() {
    support::assert_publication_backend_lineage_conformance(
        support::memory_publication_backend_fixture(),
    )
    .await;
}
