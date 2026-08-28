// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reachability-lease and garbage-collection coverage for the local checkpoint
//! store. Every test drives the production filesystem seam against a real
//! private temporary tree and an injected clock, so the assertions are about
//! actual on-disk state and never about elapsed wall time.

use std::{num::NonZeroUsize, path::Path, rc::Rc};

use aiperf_runtime::{
    clock::{Clock, SimClock},
    streaming::{
        blocking::StreamingBlockingExecutor,
        budget::BudgetLimits,
        checkpoint::{
            CheckpointBackendBudgetFailureCode, CheckpointBackendBudgetKind, CheckpointError,
            CheckpointGeneration, CommittedCheckpointGeneration, StreamRunIdentity,
        },
        checkpoint_backend::{LeasedCheckpointGenerationView, VersionedLeasedGenerationReader},
        checkpoints::{
            lease_gc::{CheckpointGarbageCollector, CheckpointRetentionPolicy, SweepAuthority},
            local::{
                BlockingLocalFilesystem, LocalCheckpointBackend, LocalCheckpointFilesystem,
                LocalCheckpointLimits, RunPaths,
            },
        },
        identity::ContentDigest,
        reliability::HandledIssueCut,
        results::ResultIndexReadBudget,
    },
};
use bytes::Bytes;

#[path = "support/streaming_checkpoint.rs"]
mod support;

/// Lifetime granted to every reachability lease these tests author.
const LEASE_NS: u64 = 1_000_000;

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

fn retention_policy(grace_ns: u64) -> CheckpointRetentionPolicy {
    CheckpointRetentionPolicy {
        resume_roots: NonZeroUsize::new(1).expect("nonzero"),
        partial_history: 0,
        retain_final_until_exported: true,
        retain_source_cache_through_resume_root: false,
        orphan_grace_ns: grace_ns,
        prepare_lease_ns: 1_000,
        reader_lease_ns: LEASE_NS,
    }
}

fn index_budget(items: usize, bytes: u64) -> ResultIndexReadBudget {
    ResultIndexReadBudget {
        max_items: NonZeroUsize::new(items).expect("nonzero page items"),
        max_bytes: std::num::NonZeroU64::new(bytes).expect("nonzero page bytes"),
    }
}

fn filesystem(run: StreamRunIdentity) -> Rc<dyn LocalCheckpointFilesystem> {
    let executor = StreamingBlockingExecutor::for_test(run, 8, 1_048_576, 1_048_576)
        .expect("bounded blocking executor");
    Rc::new(BlockingLocalFilesystem::new(executor))
}

/// One store root, the clock that drives its lease expiry, and its run.
struct GcStore {
    backend: LocalCheckpointBackend,
    clock: Rc<SimClock>,
    run: StreamRunIdentity,
    root: std::path::PathBuf,
}

fn open_store(root: &Path, run: StreamRunIdentity, limits: LocalCheckpointLimits) -> GcStore {
    let clock = Rc::new(SimClock::new());
    let backend = LocalCheckpointBackend::open(
        root.to_path_buf(),
        limits,
        filesystem(run),
        Rc::clone(&clock) as Rc<dyn Clock>,
    )
    .expect("valid local backend");
    GcStore {
        backend,
        clock,
        run,
        root: root.to_path_buf(),
    }
}

async fn open_store_with_policy(
    root: &Path,
    run: StreamRunIdentity,
    limits: LocalCheckpointLimits,
    grace_ns: u64,
) -> GcStore {
    let store = open_store(root, run, limits);
    store
        .backend
        .set_retention_policy(&run, retention_policy(grace_ns))
        .await
        .expect("valid retention policy");
    store
}

async fn commit_generation(
    store: &GcStore,
    previous: Option<&CheckpointGeneration>,
    epoch: u64,
    segments: usize,
) -> CommittedCheckpointGeneration {
    let run = store.run;
    let expected = match previous {
        None => None,
        Some(previous) => {
            let opened = store
                .backend
                .open_latest_local(&run, &support::expectations(run))
                .await
                .expect("open head")
                .expect("existing head");
            Some(support::current_v4_predecessor(&opened, previous).expect("verified predecessor"))
        }
    };
    let mut transaction = store
        .backend
        .begin_generation_local(run, expected, support::expectations(run))
        .await
        .expect("begin generation");
    transaction
        .stage_participant(support::prepared_participant(run, epoch).await)
        .await
        .expect("stage participant");

    // Retained so each partition's payload and descriptor charges outlive staging.
    let mut budgets = Vec::with_capacity(segments);
    let mut partitions = Vec::with_capacity(segments);
    for index in 0..segments {
        let (budget, partition) = support::result_partition_with_projection_and_bytes_for(
            run,
            epoch,
            &format!("projection-{index}"),
            Bytes::from(format!("result-payload-{epoch}-{index}")),
        )
        .await;
        budgets.push(budget);
        partitions.push(partition);
    }
    transaction
        .stage_results(&mut partitions, &mut None)
        .await
        .expect("stage result epoch");
    transaction
        .commit(support::metadata_with_lineage(previous.cloned(), epoch))
        .await
        .expect("commit generation")
}

fn names_in(directory: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter_map(|entry| entry.file_name().to_str().map(str::to_owned))
        .collect();
    names.sort();
    names
}

fn object_names(root: &Path, run: StreamRunIdentity) -> Vec<String> {
    names_in(&RunPaths::for_run(root, &run).objects_dir())
}

fn generation_names(root: &Path, run: StreamRunIdentity) -> Vec<String> {
    names_in(&RunPaths::for_run(root, &run).generations_dir())
}

fn lease_names(root: &Path, run: StreamRunIdentity) -> Vec<String> {
    names_in(&RunPaths::for_run(root, &run).leases_dir())
}

#[tokio::test(flavor = "current_thread")]
async fn reader_lease_prevents_reachable_object_collection() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(1);
    let store = open_store_with_policy(directory.path(), run, local_limits(), 0).await;
    commit_generation(&store, None, 1, 5).await;

    let reader = store
        .backend
        .open_latest_local(&run, &support::expectations(run))
        .await
        .expect("open head")
        .expect("head exists");
    store
        .backend
        .retain_last_generations(0)
        .await
        .expect("lower retention");
    let report = store.backend.collect_garbage().await.expect("collect");

    assert_eq!(report.authority, SweepAuthority::Held);
    assert_eq!(
        report.swept_objects, 0,
        "everything reachable from the leased generation must survive"
    );
    let page = reader
        .scan_result_index(None, index_budget(2, 4096))
        .await
        .expect("scan the still-reachable index");
    assert_eq!(page.descriptors().len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn renewal_failure_fences_read_before_gc() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(2);
    let store = open_store_with_policy(directory.path(), run, local_limits(), 0).await;
    commit_generation(&store, None, 1, 1).await;

    let reader = store
        .backend
        .open_latest_local(&run, &support::expectations(run))
        .await
        .expect("open head")
        .expect("head exists");
    store
        .clock
        .advance_to(i64::try_from(LEASE_NS).expect("representable lease") + 1);
    store.backend.fail_next_renewal();

    // The fencing is the reader's own decision about its own lease and never
    // depends on observing a sweep: no collection cycle runs in between.
    assert!(matches!(
        reader.scan_result_index(None, index_budget(1, 1024)).await,
        Err(CheckpointError::LeaseLost { .. })
    ));
    assert!(
        matches!(
            reader.scan_result_index(None, index_budget(1, 1024)).await,
            Err(CheckpointError::LeaseLost { .. })
        ),
        "fencing is sticky"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn expired_lease_is_reclaimed_only_once_both_witnesses_agree() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(3);
    let store = open_store_with_policy(directory.path(), run, local_limits(), 0).await;
    let committed = commit_generation(&store, None, 1, 1).await;

    let lease = store
        .backend
        .acquire_report_lease(run, &committed.generation())
        .await
        .expect("report lease");
    store
        .clock
        .advance_to(i64::try_from(LEASE_NS).expect("representable lease") + 1);

    // Expired on the clock, but a live holder still owns the advisory lock.
    let report = store.backend.collect_garbage().await.expect("collect");
    assert_eq!(report.swept_leases, 0);
    assert_eq!(lease_names(directory.path(), run).len(), 2);

    // A crashed holder loses its lock immediately, so both witnesses now agree.
    lease.simulate_holder_crash();
    let report = store.backend.collect_garbage().await.expect("collect");
    assert_eq!(report.swept_leases, 1);
    assert_eq!(
        lease_names(directory.path(), run),
        vec!["writer".to_owned()],
        "only the writer lease remains"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn crashed_lease_holder_releases_its_objects_only_after_the_grace() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(4);
    let store = open_store_with_policy(directory.path(), run, local_limits(), 500_000).await;
    let first = commit_generation(&store, None, 1, 1).await;
    commit_generation(&store, Some(&first.generation()), 2, 1).await;
    store
        .backend
        .retain_last_generations(0)
        .await
        .expect("lower retention");

    let lease = store
        .backend
        .acquire_report_lease(run, &first.generation())
        .await
        .expect("report lease");
    let before = object_names(directory.path(), run);
    let report = store.backend.collect_garbage().await.expect("collect");
    assert_eq!(report.swept_objects, 0, "the lease pins the older root");
    assert_eq!(object_names(directory.path(), run), before);

    lease.simulate_holder_crash();
    store
        .clock
        .advance_to(i64::try_from(LEASE_NS).expect("representable lease") + 1);
    let report = store.backend.collect_garbage().await.expect("collect");
    assert_eq!(report.swept_leases, 1);
    assert_eq!(
        report.swept_objects, 0,
        "an unpinned object serves its grace before it is swept"
    );
    assert!(report.condemned_objects > 0);

    store
        .clock
        .advance_to(i64::try_from(LEASE_NS).expect("representable lease") + 1 + 500_000);
    let report = store.backend.collect_garbage().await.expect("collect");
    assert!(report.swept_objects > 0);
    assert!(object_names(directory.path(), run).len() < before.len());
}

#[tokio::test(flavor = "current_thread")]
async fn one_cycle_condemns_and_a_later_cycle_sweeps() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(5);
    let store = open_store_with_policy(directory.path(), run, local_limits(), 1_000).await;
    let first = commit_generation(&store, None, 1, 1).await;
    commit_generation(&store, Some(&first.generation()), 2, 1).await;
    store
        .backend
        .retain_last_generations(0)
        .await
        .expect("lower retention");

    let before = object_names(directory.path(), run);
    let report = store.backend.collect_garbage().await.expect("first cycle");
    assert_eq!(report.swept_objects, 0);
    assert!(report.condemned_objects > 0);
    assert_eq!(object_names(directory.path(), run), before);

    store.clock.advance_to(1_000);
    let report = store.backend.collect_garbage().await.expect("second cycle");
    assert!(report.swept_objects > 0);
    assert!(report.swept_generations > 0);
    assert_eq!(generation_names(directory.path(), run).len(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn a_re_pinned_object_is_absolved_before_its_grace_elapses() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(6);
    let store = open_store_with_policy(directory.path(), run, local_limits(), 1_000).await;
    let first = commit_generation(&store, None, 1, 1).await;
    commit_generation(&store, Some(&first.generation()), 2, 1).await;
    store
        .backend
        .retain_last_generations(0)
        .await
        .expect("lower retention");

    let before = object_names(directory.path(), run);
    store.backend.collect_garbage().await.expect("condemn");

    // Pinning the older root again returns its objects to the mark set, which
    // clears their condemnation rather than letting the original grace run out.
    let _lease = store
        .backend
        .acquire_report_lease(run, &first.generation())
        .await
        .expect("report lease");
    store.clock.advance_to(10_000);
    let report = store.backend.collect_garbage().await.expect("absolve");

    assert_eq!(report.swept_objects, 0);
    assert_eq!(object_names(directory.path(), run), before);
}

#[tokio::test(flavor = "current_thread")]
async fn the_head_is_never_collectable_even_at_zero_retention() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(7);
    let store = open_store_with_policy(directory.path(), run, local_limits(), 0).await;
    let committed = commit_generation(&store, None, 1, 3).await;
    store
        .backend
        .retain_last_generations(0)
        .await
        .expect("lower retention");

    for _ in 0..3 {
        store.backend.collect_garbage().await.expect("collect");
    }
    drop(store);

    let reopened = open_store(directory.path(), run, local_limits());
    let opened = reopened
        .backend
        .open_latest_local(&run, &support::expectations(run))
        .await
        .expect("reopen head")
        .expect("head survives zero retention");
    assert_eq!(opened.generation(), &committed.generation());

    let LeasedCheckpointGenerationView::CurrentV4(reader) = opened.view() else {
        panic!("expected a current-v4 head");
    };
    for descriptor in reader.generation().participant_descriptors() {
        reader
            .read_participant(descriptor)
            .await
            .expect("every object the head references is still present");
    }
    let page = opened
        .scan_result_index(None, index_budget(8, 65_536))
        .await
        .expect("scan the retained index");
    assert_eq!(page.descriptors().len(), 3);
}

#[tokio::test(flavor = "current_thread")]
async fn sweeping_requires_the_writer_lease_while_marking_does_not() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(8);
    let writer = open_store_with_policy(directory.path(), run, local_limits(), 0).await;
    let first = commit_generation(&writer, None, 1, 1).await;
    commit_generation(&writer, Some(&first.generation()), 2, 1).await;

    let collector = open_store_with_policy(directory.path(), run, local_limits(), 0).await;
    collector
        .backend
        .retain_last_generations(0)
        .await
        .expect("lower retention");
    let before = object_names(directory.path(), run);
    let report = collector
        .backend
        .collect_garbage()
        .await
        .expect("a concurrent writer is not an error");

    assert_eq!(report.authority, SweepAuthority::Unavailable);
    assert_eq!(report.swept_objects, 0);
    assert!(report.marked_objects > 0, "marking needs no authority");
    assert_eq!(object_names(directory.path(), run), before);

    // A crashed writer's advisory lock is released by the kernel, so the next
    // cycle takes authority with no timeout and no heuristic.
    drop(writer);
    let report = collector.backend.collect_garbage().await.expect("collect");
    assert_eq!(report.authority, SweepAuthority::Held);
}

#[tokio::test(flavor = "current_thread")]
async fn handled_issue_roots_are_never_marked_and_never_looked_up() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(9);
    let store = open_store_with_policy(directory.path(), run, local_limits(), 0).await;
    commit_generation(&store, None, 1, 1).await;

    // The three handled-cut roots are digests over ledger state, not names of
    // any byte sequence this store has written. Collection must neither demand
    // them nor refuse because they are absent.
    let cut = HandledIssueCut::empty();
    let store_objects = object_names(directory.path(), run);
    for root in [
        cut.receipt_root(),
        cut.input_frontier_root(),
        cut.quarantine_tombstone_root(),
    ] {
        assert!(!store_objects.contains(&object_entry_name(root)));
    }

    let report = store.backend.collect_garbage().await.expect("collect");
    assert_eq!(report.swept_objects, 0);
    assert_eq!(object_names(directory.path(), run), store_objects);
}

fn object_entry_name(digest: &ContentDigest) -> String {
    let mut name = String::from("blake3-");
    for byte in digest.as_bytes() {
        name.push_str(&format!("{byte:02x}"));
    }
    name
}

#[tokio::test(flavor = "current_thread")]
async fn the_source_cache_flag_has_no_local_effect_in_either_setting() {
    let mut inventories = Vec::new();
    for retain_source_cache in [false, true] {
        let directory = tempfile::tempdir().expect("temporary store root");
        let run = support::run_id(10);
        let store = open_store(directory.path(), run, local_limits());
        let mut policy = retention_policy(0);
        policy.retain_source_cache_through_resume_root = retain_source_cache;
        store
            .backend
            .set_retention_policy(&run, policy)
            .await
            .expect("valid retention policy");
        let first = commit_generation(&store, None, 1, 2).await;
        commit_generation(&store, Some(&first.generation()), 2, 2).await;
        store
            .backend
            .retain_last_generations(0)
            .await
            .expect("lower retention");

        let report = store.backend.collect_garbage().await.expect("collect");
        inventories.push((
            report.swept_objects,
            object_names(directory.path(), run).len(),
        ));
    }

    assert_eq!(
        inventories[0], inventories[1],
        "the local layout has no source-cache object kind, so the flag is a documented no-op"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn no_collection_phase_exceeds_the_configured_page_bound() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(11);
    let store = open_store_with_policy(directory.path(), run, local_limits(), 0).await;
    let first = commit_generation(&store, None, 1, 5).await;
    commit_generation(&store, Some(&first.generation()), 2, 5).await;
    let _reader = store
        .backend
        .open_latest_local(&run, &support::expectations(run))
        .await
        .expect("open head")
        .expect("head exists");

    store
        .backend
        .retain_last_generations(0)
        .await
        .expect("lower retention");
    store.backend.collect_garbage().await.expect("collect");

    let high_water = store.backend.gc_high_water();
    assert!(
        high_water.page_items <= 2,
        "lease, generation, index, and object listings each stay inside the page bound: {high_water:?}"
    );
    assert!(high_water.page_removals <= 2);
}

#[tokio::test(flavor = "current_thread")]
async fn a_mark_set_exceeding_the_read_budget_refuses_and_sweeps_nothing() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(12);
    let store = open_store_with_policy(directory.path(), run, local_limits(), 0).await;
    commit_generation(&store, None, 1, 5).await;
    let before = object_names(directory.path(), run);
    drop(store);

    let mut starved = local_limits();
    starved.reads = BudgetLimits {
        max_items: 2,
        max_bytes: 1_048_576,
    };
    let reopened = open_store_with_policy(directory.path(), run, starved, 0).await;

    assert_eq!(
        reopened.backend.collect_garbage().await,
        Err(CheckpointError::BackendBudget {
            budget: CheckpointBackendBudgetKind::Read,
            code: CheckpointBackendBudgetFailureCode::ItemCapacity,
        }),
        "a partial mark set would delete live data, so collection fails closed"
    );
    assert_eq!(object_names(directory.path(), run), before);
}

#[tokio::test(flavor = "current_thread")]
async fn a_report_lease_pins_its_generation_until_it_is_explicitly_released() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(13);
    let store = open_store_with_policy(directory.path(), run, local_limits(), 0).await;
    let first = commit_generation(&store, None, 1, 1).await;
    commit_generation(&store, Some(&first.generation()), 2, 1).await;
    store
        .backend
        .retain_last_generations(0)
        .await
        .expect("lower retention");

    let lease = store
        .backend
        .acquire_report_lease(run, &first.generation())
        .await
        .expect("report lease");
    assert_eq!(lease.pinned(), &first.generation());
    let before = object_names(directory.path(), run);
    let report = store.backend.collect_garbage().await.expect("collect");
    assert_eq!(report.swept_objects, 0);
    assert_eq!(generation_names(directory.path(), run).len(), 2);

    // The release is an ordered, observable step, so its failure is the caller's
    // rather than a silence swallowed by a drop path.
    lease.release().await.expect("release the report lease");

    let report = store.backend.collect_garbage().await.expect("collect");
    assert!(report.swept_objects > 0);
    assert_eq!(report.swept_generations, 1);
    assert!(object_names(directory.path(), run).len() < before.len());
}

#[tokio::test(flavor = "current_thread")]
async fn an_invalid_retention_policy_performs_no_filesystem_effect() {
    let directory = tempfile::tempdir().expect("temporary store root");
    let run = support::run_id(14);
    let store = open_store(directory.path(), run, local_limits());
    let mut policy = retention_policy(0);
    policy.reader_lease_ns = 0;

    assert_eq!(
        store.backend.set_retention_policy(&run, policy).await,
        Err(CheckpointError::ObjectVerification),
        "a zero-lifetime lease is born expired and pins nothing"
    );
    assert!(
        !store.root.join("objects").exists(),
        "a refused policy must never reach the filesystem"
    );
}
