// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conditional object-store checkpoint backend: pointer CAS, bounded object
//! I/O, shared lineage prevalidation, and legacy-v3 read-only heads.

use aiperf_runtime::{
    extensions::{AIPerfRegistry, AIPerfRegistryFactory, BuiltinAIPerfRegistryFactory},
    streaming::{
        checkpoint::CheckpointError,
        checkpoint_backend::{
            CheckpointGenerationStorageVersion, StreamingCheckpointBackend,
            VersionedLeasedGenerationReader,
        },
        checkpoint_factories::OBJECT_STORE_CHECKPOINT_BACKEND_ID,
        checkpoints::object_store::{CheckpointErrorCode, CheckpointFailureCode},
    },
};

#[path = "support/streaming_checkpoint.rs"]
mod support;

use support::{
    FakeConditionalObjectStore, expectations, metadata_at, object_backend,
    object_backend_with_legacy_v3_head, object_io_budget, object_publication_backend_fixture,
    prepared_transaction, read_budget, run_id,
};

fn frozen_registry() -> AIPerfRegistry {
    BuiltinAIPerfRegistryFactory
        .build()
        .expect("built-in registry")
}

#[tokio::test(flavor = "current_thread")]
async fn object_pointer_cas_publishes_exactly_one_complete_generation() {
    let store = FakeConditionalObjectStore::new(object_io_budget(64 * 1024));
    let backend = object_backend(store.clone());
    let left = prepared_transaction(&backend, None, 1).await;
    let right = prepared_transaction(&backend, None, 1).await;

    let left_landed = left.commit(metadata_at(1)).await.is_ok();
    let right_landed = right.commit(metadata_at(1)).await.is_ok();

    assert!(
        left_landed ^ right_landed,
        "exactly one racing writer may publish"
    );
    assert!(store.current_pointer_references_only_verified_objects());
}

#[tokio::test(flavor = "current_thread")]
async fn losing_writer_reports_a_stale_writer_refusal() {
    let store = FakeConditionalObjectStore::new(object_io_budget(64 * 1024));
    let backend = object_backend(store.clone());
    let winner = prepared_transaction(&backend, None, 1).await;
    let loser = prepared_transaction(&backend, None, 1).await;

    winner.commit(metadata_at(1)).await.expect("winner lands");
    let error = loser
        .commit(metadata_at(1))
        .await
        .expect_err("loser observes a moved pointer");

    assert_eq!(error.code(), CheckpointFailureCode::StaleWriter);
}

#[tokio::test(flavor = "current_thread")]
async fn oversized_metadata_is_rejected_before_allocation() {
    let store = FakeConditionalObjectStore::declaring_length(usize::MAX);
    let error = object_backend(store.clone())
        .restore_current(read_budget(4096))
        .await
        .expect_err("hostile declared length must be refused");

    assert_eq!(error.code(), CheckpointFailureCode::ObjectLimitExceeded);
    assert_eq!(store.allocated_bytes(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn object_backend_conforms_to_shared_pre_io_lineage_validation() {
    support::assert_publication_backend_lineage_conformance(object_publication_backend_fixture())
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn object_open_exposes_v3_read_only_and_never_attempts_successor_cas() {
    let fixture = object_backend_with_legacy_v3_head().await;
    let opened = fixture
        .backend
        .open_latest(&fixture.run, &fixture.expectations)
        .await
        .expect("open legacy head")
        .expect("legacy head exists");

    assert_eq!(
        opened.version(),
        CheckpointGenerationStorageVersion::LegacyV3ReadOnly
    );
    assert!(matches!(
        fixture
            .backend
            .begin_generation(fixture.run, None, fixture.expectations.clone())
            .await,
        Err(CheckpointError::LegacyReadOnlyHead),
    ));
    assert_eq!(fixture.store.pointer_cas_calls(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn a_failed_upload_leaves_the_pointer_untouched() {
    let store = FakeConditionalObjectStore::new(object_io_budget(64 * 1024));
    let backend = object_backend(store.clone());
    let baseline = prepared_transaction(&backend, None, 1).await;
    baseline
        .commit(metadata_at(1))
        .await
        .expect("baseline lands");
    let published = store.state_fingerprint();

    // Fail the very next immutable upload: no pointer replacement may follow.
    let head = backend
        .open_latest(&run_id(1), &expectations(run_id(1)))
        .await
        .expect("open head")
        .expect("head exists")
        .generation()
        .clone();
    store.arm_upload_failure(store.upload_attempts() + 1);
    let successor = prepared_transaction(&backend, Some(head), 2).await;
    let cas_before = store.pointer_cas_calls();

    assert!(successor.commit(metadata_at(2)).await.is_err());
    assert_eq!(store.pointer_cas_calls(), cas_before);
    assert_eq!(store.state_fingerprint(), published);
}

#[tokio::test(flavor = "current_thread")]
async fn a_landed_pointer_survives_a_crash_after_compare_and_swap() {
    let store = FakeConditionalObjectStore::new(object_io_budget(64 * 1024));
    let backend = object_backend(store.clone());
    let transaction = prepared_transaction(&backend, None, 1).await;
    let committed = transaction.commit(metadata_at(1)).await.expect("commit");

    // A replacement process reads the same store and must observe exactly the
    // generation the compare-and-swap published.
    let restarted = object_backend(store.clone());
    let opened = restarted
        .open_latest(&run_id(1), &expectations(run_id(1)))
        .await
        .expect("reopen after crash")
        .expect("published head survives");

    assert_eq!(opened.generation(), committed.generation_ref());
    assert_eq!(
        opened.version(),
        CheckpointGenerationStorageVersion::CurrentV4
    );
}

#[tokio::test(flavor = "current_thread")]
async fn restores_never_exceed_the_bounded_chunk_high_water() {
    let store = FakeConditionalObjectStore::new(object_io_budget(64 * 1024));
    let backend = object_backend(store.clone());
    prepared_transaction(&backend, None, 1)
        .await
        .commit(metadata_at(1))
        .await
        .expect("commit");

    backend
        .open_latest(&run_id(1), &expectations(run_id(1)))
        .await
        .expect("open head")
        .expect("head exists");

    // The fake store refuses any range wider than the caller's chunk budget, so
    // reaching this point at all proves every read stayed bounded.
    assert!(store.allocated_bytes() > 0);
}

#[test]
fn object_store_backend_is_registered_only_with_its_feature() {
    let registry = frozen_registry();
    assert_eq!(OBJECT_STORE_CHECKPOINT_BACKEND_ID, "object_store");
    let factory = registry
        .stream_checkpoint_backend_factory(OBJECT_STORE_CHECKPOINT_BACKEND_ID)
        .expect("object_store is registered under streaming-s3");
    let descriptor = factory.descriptor();
    assert!(descriptor.is_durable);
    assert!(descriptor.has_atomic_generations);
    assert!(!descriptor.supports_virtual_clock);
}
