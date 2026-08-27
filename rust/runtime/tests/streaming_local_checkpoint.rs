// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Crash-durability, lease, and reclamation coverage for the local checkpoint
//! store. Every test drives the production filesystem seam against a real
//! private temporary tree, so the assertions are about actual on-disk state.

use std::{num::NonZeroUsize, os::unix::fs::PermissionsExt, path::Path, rc::Rc};

use aiperf_runtime::{
    clock::{Clock, SimClock},
    streaming::{
        blocking::StreamingBlockingExecutor,
        budget::BudgetLimits,
        checkpoint::{
            CheckpointBackendBudgetFailureCode, CheckpointBackendBudgetKind, CheckpointError,
            CheckpointGeneration, CommittedCheckpointGeneration, StreamRunIdentity,
        },
        checkpoint_backend::LeasedCheckpointGenerationView,
        checkpoints::local::{
            BlockingLocalFilesystem, LocalCheckpointBackend, LocalCheckpointFilesystem,
            LocalCheckpointLimits, LocalCommitFault, RunPaths,
        },
        results::ResultIndexReadBudget,
    },
};

#[path = "support/streaming_checkpoint.rs"]
mod support;

fn local_limits() -> LocalCheckpointLimits {
    let limits = BudgetLimits {
        max_items: 64,
        max_bytes: 1_048_576,
    };
    LocalCheckpointLimits {
        transactions: limits,
        prepared_indexes: limits,
        storage: limits,
        result_summaries: limits,
        reads: limits,
        gc_page_items: NonZeroUsize::new(2).expect("nonzero page bound"),
        prepare_lease_ns: 1_000,
    }
}

/// One store root plus the clock that drives its lease expiry.
struct LocalStore {
    backend: LocalCheckpointBackend,
    clock: Rc<SimClock>,
    root: std::path::PathBuf,
}

fn filesystem(run: StreamRunIdentity) -> Rc<dyn LocalCheckpointFilesystem> {
    let executor = StreamingBlockingExecutor::for_test(run, 8, 1_048_576, 1_048_576)
        .expect("bounded blocking executor");
    Rc::new(BlockingLocalFilesystem::new(executor))
}

fn open_store(root: &Path, run: StreamRunIdentity, limits: LocalCheckpointLimits) -> LocalStore {
    let clock = Rc::new(SimClock::new());
    let backend = LocalCheckpointBackend::open(
        root.to_path_buf(),
        limits,
        filesystem(run),
        Rc::clone(&clock) as Rc<dyn Clock>,
    )
    .expect("valid local backend");
    LocalStore {
        backend,
        clock,
        root: root.to_path_buf(),
    }
}

async fn commit_generation(
    store: &LocalStore,
    run: StreamRunIdentity,
    previous: Option<&CheckpointGeneration>,
    epoch: u64,
) -> Result<CommittedCheckpointGeneration, CheckpointError> {
    let expected = match previous {
        None => None,
        Some(previous) => {
            let opened = store
                .backend
                .open_latest_local(&run, &support::expectations(run))
                .await?
                .expect("existing head");
            Some(support::current_v4_predecessor(&opened, previous)?)
        }
    };
    let mut transaction = store
        .backend
        .begin_generation_local(run, expected, support::expectations(run))
        .await?;
    transaction
        .stage_participant(support::prepared_participant(run, epoch).await)
        .await?;
    let mut partitions = vec![support::result_partition(run, epoch).await];
    transaction.stage_results(&mut partitions, &mut None).await?;
    transaction
        .commit(support::metadata_with_lineage(
            previous.cloned(),
            epoch,
        ))
        .await
}

async fn head_of(store: &LocalStore, run: StreamRunIdentity) -> Option<CheckpointGeneration> {
    store
        .backend
        .open_latest_local(&run, &support::expectations(run))
        .await
        .expect("open head")
        .map(|opened| opened.generation().clone())
}

#[tokio::test(flavor = "current_thread")]
async fn every_pre_current_fault_preserves_previous_generation() {
    for fault in LocalCommitFault::before_current_publication() {
        let directory = tempfile::tempdir().expect("temporary store root");
        let run = support::run_id(1);
        let store = open_store(directory.path(), run, local_limits());
        let baseline = commit_generation(&store, run, None, 1)
            .await
            .expect("baseline generation");

        store.backend.inject_fault(fault);
        let error = commit_generation(&store, run, Some(&baseline.generation()), 2)
            .await
            .expect_err("armed fault must refuse the commit");

        assert_eq!(error, fault.injected_error(), "fault {fault:?}");
        assert!(
            store.backend.injected_fault_was_reached(fault),
            "fault {fault:?} was never reached"
        );

        // An independently reopened backend must still see the complete
        // previous generation: `CURRENT` was never renamed.
        let reopened = open_store(&store.root, run, local_limits());
        assert_eq!(
            head_of(&reopened, run).await,
            Some(baseline.generation()),
            "fault {fault:?} changed the authoritative head"
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn fault_after_current_rename_yields_the_complete_new_generation() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(2);
    let store = open_store(directory.path(), run, local_limits());
    let baseline = commit_generation(&store, run, None, 1)
        .await
        .expect("baseline generation");

    store
        .backend
        .inject_fault(LocalCommitFault::AfterCurrentRename);
    let error = commit_generation(&store, run, Some(&baseline.generation()), 2)
        .await
        .expect_err("post-fence fault still refuses the caller");
    assert_eq!(error, LocalCommitFault::AfterCurrentRename.injected_error());

    // Past the fence the head is the new one, and every object it references
    // was fsynced before the rename.
    let reopened = open_store(&store.root, run, local_limits());
    let head = head_of(&reopened, run).await.expect("published head");
    assert_ne!(head, baseline.generation());
    assert_eq!(head.epoch().get(), 2);

    let opened = reopened
        .backend
        .open_latest_local(&run, &support::expectations(run))
        .await
        .expect("open published head")
        .expect("published head exists");
    let LeasedCheckpointGenerationView::CurrentV4(reader) = opened.view() else {
        panic!("expected a current-v4 head");
    };
    for descriptor in reader.generation().participant_descriptors() {
        reader
            .read_participant(descriptor)
            .await
            .expect("every referenced participant object is present");
    }
}

#[tokio::test(flavor = "current_thread")]
async fn lineage_refusal_performs_no_filesystem_effect_and_preserves_authority() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(3);
    let store = open_store(directory.path(), run, local_limits());
    let baseline = commit_generation(&store, run, None, 1)
        .await
        .expect("baseline generation");

    for metadata in [
        // Wrong predecessor: the frozen head is not what the caller claims.
        support::metadata_with_lineage(None, 2),
        // Skipped epoch: the successor is not the immediate one.
        support::metadata_with_lineage(Some(baseline.generation()), 3),
    ] {
        let opened = store
            .backend
            .open_latest_local(&run, &support::expectations(run))
            .await
            .expect("open lineage head")
            .expect("lineage head exists");
        let predecessor = support::current_v4_predecessor(&opened, &baseline.generation())
            .expect("verified predecessor");
        drop(opened);
        let mut transaction = store
            .backend
            .begin_generation_local(run, Some(predecessor), support::expectations(run))
            .await
            .expect("begin lineage transaction");
        transaction
            .stage_participant(support::prepared_participant(run, 2).await)
            .await
            .expect("stage participant");
        transaction
            .stage_results(&mut Vec::new(), &mut None)
            .await
            .expect("stage result epoch");

        store.backend.reset_effect_counter();
        assert_eq!(
            transaction
                .commit(metadata)
                .await
                .expect_err("lineage refusal"),
            CheckpointError::ObjectVerification,
        );
        assert_eq!(
            store.backend.effect_counter(),
            0,
            "a lineage refusal must never reach the filesystem"
        );
        assert_eq!(head_of(&store, run).await, Some(baseline.generation()));
    }
}

#[tokio::test(flavor = "current_thread")]
async fn two_writers_cannot_both_commit() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(4);
    let first = open_store(directory.path(), run, local_limits());
    let baseline = commit_generation(&first, run, None, 1)
        .await
        .expect("baseline generation");

    // The first backend keeps its writer lease for its whole lifetime, so the
    // second is refused before it writes any object.
    let _held = first
        .backend
        .begin_generation_local(run, None, support::expectations(run))
        .await;

    let second = open_store(directory.path(), run, local_limits());
    let objects_before = object_names(&directory.path().to_path_buf(), run);
    let error = second
        .backend
        .begin_generation_local(run, None, support::expectations(run))
        .await
        .expect_err("second writer must be refused");

    assert!(
        matches!(error, CheckpointError::GenerationConflict { .. }),
        "unexpected refusal: {error:?}"
    );
    assert_eq!(objects_before, object_names(&directory.path().to_path_buf(), run));
    assert_eq!(head_of(&first, run).await, Some(baseline.generation()));
}

fn object_names(root: &std::path::PathBuf, run: StreamRunIdentity) -> Vec<String> {
    let paths = RunPaths::for_run(root, &run);
    let Ok(entries) = std::fs::read_dir(paths.objects_dir()) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter_map(|entry| entry.file_name().to_str().map(str::to_owned))
        .collect();
    names.sort();
    names
}

#[tokio::test(flavor = "current_thread")]
async fn checkpoint_tree_is_private_and_transaction_scratch_is_reclaimed() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(5);
    let store = open_store(directory.path(), run, local_limits());
    commit_generation(&store, run, None, 1)
        .await
        .expect("baseline generation");

    let paths = RunPaths::for_run(&directory.path().to_path_buf(), &run);
    let mode = std::fs::metadata(paths.root())
        .expect("run root exists")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o700, "the run root must be private");

    let current_mode = std::fs::metadata(paths.current())
        .expect("pointer exists")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(current_mode, 0o600, "every regular file must be private");

    // A cancelled transaction removes its own scratch subtree.
    let transaction = store
        .backend
        .begin_generation_local(run, None, support::expectations(run))
        .await;
    let transaction = match transaction {
        Ok(transaction) => transaction,
        Err(error) => panic!("begin must succeed against the retained head: {error:?}"),
    };
    let scratch = transaction.tmp_path().to_path_buf();
    assert!(scratch.is_dir());
    transaction.cancel().await;
    assert!(!scratch.exists(), "cancellation must remove its scratch");
}

#[tokio::test(flavor = "current_thread")]
async fn orphan_transaction_is_reclaimed_by_a_bounded_lease_aware_scan() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(6);
    let store = open_store(directory.path(), run, local_limits());
    let paths = RunPaths::for_run(&directory.path().to_path_buf(), &run);
    store
        .backend
        .begin_generation_local(run, None, support::expectations(run))
        .await
        .expect("begin transaction");

    // Simulate a crash: the owning process is gone, so its advisory lock is
    // released, but its scratch subtree and expired lease record remain.
    let orphans: Vec<String> = std::fs::read_dir(paths.tmp_dir())
        .expect("scratch directory exists")
        .flatten()
        .filter_map(|entry| entry.file_name().to_str().map(str::to_owned))
        .collect();
    assert_eq!(orphans.len(), 1);
    drop(store);

    let reopened = open_store(directory.path(), run, local_limits());
    // A live lease is never reclaimed on modification time; only the clock
    // crossing the recorded expiry makes the subtree eligible.
    assert_eq!(
        reopened
            .backend
            .reclaim_all_orphan_transactions(&run)
            .await
            .expect("bounded scan"),
        0,
    );
    assert!(paths.tmp_dir().join(&orphans[0]).exists());

    reopened.clock.advance_to(1_000_000);
    assert_eq!(
        reopened
            .backend
            .reclaim_all_orphan_transactions(&run)
            .await
            .expect("bounded scan"),
        1,
    );
    assert!(!paths.tmp_dir().join(&orphans[0]).exists());
    assert!(
        reopened.backend.gc_high_water().page_items <= 2,
        "the scan must stay inside its configured page bound"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn reopen_recovers_exact_storage_charge_for_reachable_objects() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(7);
    let store = open_store(directory.path(), run, local_limits());
    let baseline = commit_generation(&store, run, None, 1)
        .await
        .expect("baseline generation");
    let charged = store.backend.live_budget_usage().storage;
    drop(store);

    let reopened = open_store(directory.path(), run, local_limits());
    assert_eq!(reopened.backend.live_budget_usage().storage.used_items, 0);
    let head = head_of(&reopened, run).await.expect("retained head");
    assert_eq!(head, baseline.generation());

    let recovered = reopened.backend.live_budget_usage().storage;
    assert!(
        recovered.used_items > 0 && recovered.used_bytes > 0,
        "reopening must re-derive the storage charge"
    );
    assert!(
        recovered.used_bytes <= charged.used_bytes,
        "recovery charges reachable objects, never more than the writer did"
    );

    // Opening a second time charges nothing further: the run is already
    // accounted for.
    let after_second_open = {
        head_of(&reopened, run).await;
        reopened.backend.live_budget_usage().storage
    };
    assert_eq!(after_second_open.used_items, recovered.used_items);
}

#[tokio::test(flavor = "current_thread")]
async fn committed_result_segment_is_reachable_through_the_ordinary_result_index() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(8);
    let store = open_store(directory.path(), run, local_limits());
    commit_generation(&store, run, None, 1)
        .await
        .expect("baseline generation");

    let opened = store
        .backend
        .open_latest_local(&run, &support::expectations(run))
        .await
        .expect("open head")
        .expect("head exists");
    let LeasedCheckpointGenerationView::CurrentV4(reader) = opened.view() else {
        panic!("expected a current-v4 head");
    };
    let page = reader
        .scan_result_index(
            None,
            ResultIndexReadBudget {
                max_items: NonZeroUsize::new(8).expect("nonzero"),
                max_bytes: std::num::NonZeroU64::new(65_536).expect("nonzero"),
            },
        )
        .await
        .expect("scan the reachable index");
    let descriptors = page.descriptors().to_vec();
    assert_eq!(descriptors.len(), 1);

    let segment = reader
        .read_segment(&descriptors[0])
        .await
        .expect("read the reachable payload");
    assert_eq!(
        segment.payload_bytes().len() as u64,
        descriptors[0].byte_length
    );
}

#[test]
fn invalid_limits_perform_no_filesystem_effect() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let root = directory.path().join("store");
    let run = support::run_id(9);
    const ZERO: BudgetLimits = BudgetLimits {
        max_items: 0,
        max_bytes: 0,
    };

    for (mutate, kind) in [
        (
            (|limits: &mut LocalCheckpointLimits| limits.transactions = ZERO)
                as fn(&mut LocalCheckpointLimits),
            CheckpointBackendBudgetKind::Transaction,
        ),
        (
            |limits: &mut LocalCheckpointLimits| limits.prepared_indexes = ZERO,
            CheckpointBackendBudgetKind::PreparedIndex,
        ),
        (
            |limits: &mut LocalCheckpointLimits| limits.storage = ZERO,
            CheckpointBackendBudgetKind::Storage,
        ),
        (
            |limits: &mut LocalCheckpointLimits| limits.result_summaries = ZERO,
            CheckpointBackendBudgetKind::ResultSummary,
        ),
        (
            |limits: &mut LocalCheckpointLimits| limits.reads = ZERO,
            CheckpointBackendBudgetKind::Read,
        ),
    ] {
        let mut limits = local_limits();
        mutate(&mut limits);
        let error = LocalCheckpointBackend::open(
            root.clone(),
            limits,
            filesystem(run),
            Rc::new(SimClock::new()) as Rc<dyn Clock>,
        )
        .expect_err("invalid limits must refuse");

        assert_eq!(
            error,
            CheckpointError::BackendBudget {
                budget: kind,
                code: CheckpointBackendBudgetFailureCode::ItemCapacity,
            }
        );
        assert!(!root.exists(), "a refused open must perform no effect");
    }
}
