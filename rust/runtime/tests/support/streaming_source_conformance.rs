// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reusable conformance assertions for `StreamingDatasetSourceFactory`.
//!
//! One implementation of the streaming source contract is exercised end to end:
//! strict configuration validation, discovery of immutable partitions, stable
//! identity across duplicate rediscovery, monotonic frontier, explicit seal,
//! the `Pending`-is-not-`Seal` distinction, host stop wakeup, and idempotent
//! post-commit notification.
//!
//! The caller constructs the reliability reporter and moves it in. No adapter
//! owns it. Every borrow of the reporter is released before the next
//! source, checkpoint, or control `await`.
//!
//! Loaded by an integration test with
//! `#[path = "support/streaming_source_conformance.rs"] mod …;`.

use std::collections::BTreeSet;
use std::rc::Rc;

use aiperf_runtime::clock::RealClock;
use aiperf_runtime::streaming::{
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        AcquisitionHorizon, AdmissionHorizon, CheckpointBarrier, CheckpointCut, CheckpointEpoch,
        CheckpointParticipantPlan, CommittedParticipantReceipt, DecodeHorizon, DiscoveryHorizon,
        EventTimeWatermark, OrderedActionHorizon, StreamRunIdentity, TerminalActionHorizon,
    },
    checkpoint_backend::{CheckpointCommitMetadata, CheckpointGenerationExpectations},
    checkpoints::memory::{MemoryCheckpointBackend, MemoryCheckpointLimits},
    failure::{
        StableStreamingFailure, StreamingFailureStage, StreamingIssueReporter,
        StreamingIssueReporterHandle,
    },
    identity::{ContentDigest, GlobalSequence, ImmutableObjectIdentity, SessionCausalFrontier},
    reliability::HandledIssueCut,
    source::{
        AcquisitionBudget, OpenedStreamingDatasetSource, PartitionAccessRequest, SourceEvent,
        StreamingDatasetSourceFactory, StreamingSourcePrepareContext, streaming_stop_channel,
    },
    unit::{EventTimeUtc, SourcePosition},
};
use serde_json::value::RawValue;

/// Factory-scoped hook that releases exactly one parked discovery step.
///
/// A conformant source is allowed to park in `next_event`. The harness proves
/// that parking is observable and is *not* a seal, then calls this hook to let
/// the scripted adapter make progress. Adapters that never park supply a no-op.
pub type SourceAdvance = Rc<dyn Fn()>;

/// Everything one source implementation contributes to the shared harness.
pub struct SourceConformanceCases {
    /// Strictly authored configuration the factory must accept.
    pub authored: Box<RawValue>,
    /// Authored configuration the factory must refuse before any effect.
    pub rejected_authored: Box<RawValue>,
    /// Resident-memory limits installed for acquisition.
    pub memory_limits: BudgetLimits,
    /// Local-snapshot disk limits installed for acquisition.
    pub disk_limits: BudgetLimits,
    /// Exact number of distinct immutable partitions the script discovers.
    pub expected_partition_count: usize,
    /// Number of times the script re-announces an already-discovered position.
    pub expected_duplicate_count: usize,
    /// Whether the script publishes at least one frontier before its seal.
    pub expects_frontier: bool,
    /// Ordinary issues the script reports through the injected handle.
    pub expected_issue_count: u64,
    /// Logical run the harness binds into checkpoint barriers.
    pub run: StreamRunIdentity,
    /// Hook releasing one parked discovery step.
    pub advance: SourceAdvance,
}

/// Assert the complete streaming source contract for one factory.
///
/// # Panics
///
/// Panics with a described failure on any contract violation.
pub async fn assert_source_conformance(
    factory: &dyn StreamingDatasetSourceFactory,
    mut reporter: Box<dyn StreamingIssueReporter>,
    cases: SourceConformanceCases,
) {
    // The harness owns the reporter for the whole run. Every read below is a
    // borrow taken after a stage future resolved and dropped before the next
    // await; the owned box is never handed to an adapter.
    assert_strict_validation(factory, &cases);
    assert_pending_is_not_seal(factory, reporter.as_mut(), &cases).await;
    assert_discovery_inventory(factory, reporter.as_mut(), &cases).await;
    assert_idempotent_commit_notification(factory, reporter.as_mut(), &cases).await;

    let total = reporter
        .summary()
        .expect("reporter summary is available after conformance")
        .total;
    assert_eq!(
        total, cases.expected_issue_count,
        "scripted ordinary faults are the only reporter receipts"
    );
}

fn assert_strict_validation(
    factory: &dyn StreamingDatasetSourceFactory,
    cases: &SourceConformanceCases,
) {
    let descriptor = factory.descriptor();
    assert!(
        !descriptor.id.is_empty(),
        "a registered source declares a stable identifier"
    );
    factory
        .validate(&cases.authored)
        .expect("authored source configuration validates");
    assert!(
        factory.validate(&cases.rejected_authored).is_err(),
        "unknown or malformed source configuration is refused before preparation"
    );
}

/// A parked `next_event` is not a seal, and host stop wakes it with an
/// unforgeable outcome that creates no issue receipt.
async fn assert_pending_is_not_seal(
    factory: &dyn StreamingDatasetSourceFactory,
    reporter: &mut dyn StreamingIssueReporter,
    cases: &SourceConformanceCases,
) {
    let before = reporter
        .summary()
        .expect("reporter summary before stop")
        .total;
    // Borrow released here, before every later source/control await.

    let mut opened = open_source(factory, reporter, cases).await;
    {
        let pending = opened.source.next_event();
        tokio::pin!(pending);
        assert!(
            futures::poll!(&mut pending).is_pending(),
            "a source with no ready event parks instead of sealing"
        );
        opened.control.stop();
        // `SourceEvent` carries opaque content authority and is not `Debug`, so
        // the failure path is matched rather than unwrapped.
        let error = match pending.await {
            Ok(_) => panic!("stop wakes the pending source"),
            Err(error) => error,
        };
        assert!(error.is_stopped(), "controlled stop is distinguishable");
        assert_eq!(error.stage(), StreamingFailureStage::Source);
        assert_eq!(error.code(), "stopped");
    }
    assert!(opened.control.is_stopped());

    let after = reporter
        .summary()
        .expect("reporter summary after stop")
        .total;
    assert_eq!(
        before, after,
        "a controlled stop advances no seal and mints no issue receipt"
    );
}

/// Drain one opened source to its seal and check the discovered inventory.
async fn assert_discovery_inventory(
    factory: &dyn StreamingDatasetSourceFactory,
    reporter: &mut dyn StreamingIssueReporter,
    cases: &SourceConformanceCases,
) {
    let budget = acquisition_budget(cases);
    let mut opened = open_source(factory, reporter, cases).await;
    let snapshot_digest = opened.source.snapshot().digest;

    let mut discovered: Vec<(SourcePosition, ImmutableObjectIdentity)> = Vec::new();
    let mut distinct: BTreeSet<ImmutableObjectIdentity> = BTreeSet::new();
    let mut duplicates = 0_usize;
    let mut is_frontier_seen = false;
    let mut last_frontier = SourcePosition::new(0);
    let seal = loop {
        (cases.advance)();
        match opened
            .source
            .next_event()
            .await
            .expect("scripted source reaches its seal without failing")
        {
            SourceEvent::Partition(partition) => {
                let position = partition.position();
                let identity = *partition.content().identity();
                // Immutable identity: the announced identity must survive an
                // acquisition round trip unchanged.
                let acquired = partition
                    .content()
                    .acquire(
                        PartitionAccessRequest::Sequential { resume_offset: 0 },
                        &budget,
                    )
                    .await
                    .expect("scripted partition acquires under the harness budget");
                assert_eq!(
                    acquired.identity(),
                    &identity,
                    "acquisition preserves the announced immutable generation"
                );
                assert_eq!(acquired.position(), position);
                drop(acquired);

                if let Some((_, prior)) = discovered.iter().find(|(seen, _)| *seen == position) {
                    duplicates += 1;
                    assert_eq!(
                        *prior, identity,
                        "rediscovering a position never mutates its immutable identity"
                    );
                } else {
                    assert!(
                        distinct.insert(identity),
                        "two distinct positions never share one immutable generation"
                    );
                    discovered.push((position, identity));
                }
            }
            SourceEvent::Frontier(frontier) => {
                assert!(
                    frontier.through.get() >= last_frontier.get(),
                    "source frontiers are monotonic"
                );
                last_frontier = frontier.through;
                is_frontier_seen = true;
            }
            SourceEvent::Seal(seal) => break seal,
        }
    };

    assert_eq!(discovered.len(), cases.expected_partition_count);
    assert_eq!(duplicates, cases.expected_duplicate_count);
    assert_eq!(is_frontier_seen, cases.expects_frontier);
    assert_eq!(
        seal.final_position,
        discovered.last().map(|(position, _)| *position),
        "the seal names the last discovered position"
    );
    assert_eq!(
        seal.digest, snapshot_digest,
        "the seal binds the same immutable inventory as the open snapshot"
    );
    assert!(
        !discovered.is_empty(),
        "a conformant source discovers at least one immutable partition"
    );
}

/// `checkpoint_committed` is idempotent for one exact receipt.
async fn assert_idempotent_commit_notification(
    factory: &dyn StreamingDatasetSourceFactory,
    reporter: &mut dyn StreamingIssueReporter,
    cases: &SourceConformanceCases,
) {
    let handle = reporter.handle();
    // Borrow of the owned reporter ends here, before every await below.
    let (control, stop) = streaming_stop_channel();
    let validated = factory
        .validate(&cases.authored)
        .expect("authored source configuration validates");
    let context = prepare_context(cases, handle);
    let prepared = factory
        .prepare(validated, &context)
        .expect("source preparation succeeds");
    let mut opened = prepared.open(stop).await.expect("source opens");

    opened
        .source
        .initialize(None)
        .await
        .expect("fresh participant initialization");

    let backend = MemoryCheckpointBackend::new(memory_checkpoint_limits())
        .expect("valid in-memory checkpoint backend");
    let expectations = CheckpointGenerationExpectations {
        run: cases.run,
        participant_plan: CheckpointParticipantPlan::new([opened.source.participant_id()])
            .expect("single-participant plan"),
        execution_plan_digest: ContentDigest::from_bytes([0x31; 32]),
        result_plan_digest: ContentDigest::from_bytes([0x32; 32]),
    };
    let barrier = barrier_for(cases.run, 1);
    let prepared_state = opened
        .source
        .checkpoint_view(&barrier)
        .await
        .expect("non-destructive participant view");
    let descriptor = prepared_state.descriptor().clone();

    let mut transaction = backend
        .begin_generation(cases.run, None, expectations)
        .await
        .expect("begin generation");
    transaction
        .stage_participant(prepared_state)
        .await
        .expect("stage the source participant");
    transaction
        .stage_results(&mut Vec::new(), &mut None)
        .await
        .expect("stage empty results");
    let generation = transaction
        .commit(CheckpointCommitMetadata {
            previous: None,
            epoch: CheckpointEpoch::new(1),
            cut: barrier.cut.clone(),
            execution_plan_digest: ContentDigest::from_bytes([0x31; 32]),
            result_plan_digest: ContentDigest::from_bytes([0x32; 32]),
            is_final: false,
            terminal_reason: None,
        })
        .await
        .expect("commit generation");

    let receipt = CommittedParticipantReceipt::new(&generation, &descriptor)
        .expect("receipt for the committed descriptor");
    opened
        .source
        .checkpoint_committed(&receipt)
        .await
        .expect("first post-commit notification");
    opened
        .source
        .checkpoint_committed(&receipt)
        .await
        .expect("post-commit notification is idempotent for one exact receipt");

    control.stop();
}

async fn open_source(
    factory: &dyn StreamingDatasetSourceFactory,
    reporter: &mut dyn StreamingIssueReporter,
    cases: &SourceConformanceCases,
) -> OpenedStreamingDatasetSource {
    let handle = reporter.handle();
    // Borrow of the owned reporter ends on this line; only the cloneable
    // handle crosses the adapter boundary.
    let validated = factory
        .validate(&cases.authored)
        .expect("authored source configuration validates");
    let context = prepare_context(cases, handle);
    let prepared = factory
        .prepare(validated, &context)
        .expect("source preparation succeeds");
    let (_control, stop) = streaming_stop_channel();
    prepared.open(stop).await.expect("source opens")
}

/// Build the one prepare context every harness phase installs.
///
/// Kept in one place so a field added to `StreamingSourcePrepareContext` is a
/// single edit here rather than one per phase.
fn prepare_context(
    cases: &SourceConformanceCases,
    handle: StreamingIssueReporterHandle,
) -> StreamingSourcePrepareContext {
    StreamingSourcePrepareContext {
        run: cases.run,
        stream_semantic_digest: ContentDigest::from_bytes(
            *cases.run.logical_replay_run().as_bytes(),
        ),
        clock: RealClock::new(),
        acquisition_budget: acquisition_budget(cases),
        issue_reporter: handle,
    }
}

fn acquisition_budget(cases: &SourceConformanceCases) -> AcquisitionBudget {
    AcquisitionBudget::new(
        StreamingResourceBudget::new(cases.memory_limits).expect("valid memory limits"),
        StreamingResourceBudget::new(cases.disk_limits).expect("valid disk limits"),
    )
}

fn memory_checkpoint_limits() -> MemoryCheckpointLimits {
    let limits = BudgetLimits {
        max_items: 64,
        max_bytes: 1_048_576,
    };
    MemoryCheckpointLimits {
        transactions: limits,
        prepared_indexes: limits,
        storage: limits,
        result_summaries: limits,
        reads: limits,
    }
}

fn barrier_for(run: StreamRunIdentity, value: u64) -> CheckpointBarrier {
    let event_time = EventTimeUtc::new(
        i64::try_from(value).expect("harness barrier values fit signed nanoseconds"),
    )
    .expect("non-negative event time");
    CheckpointBarrier {
        run,
        epoch: CheckpointEpoch::new(value),
        cut: CheckpointCut {
            discovered: DiscoveryHorizon::new(SourcePosition::new(value)),
            acquired: AcquisitionHorizon::new(SourcePosition::new(value)),
            decoded: DecodeHorizon::new(SourcePosition::new(value)),
            ordered: OrderedActionHorizon::new(GlobalSequence::new(value)),
            admitted: AdmissionHorizon::new(GlobalSequence::new(value)),
            terminal: TerminalActionHorizon::new(GlobalSequence::new(value)),
            handled_issues: HandledIssueCut::empty(),
            event_watermark: EventTimeWatermark::Hard {
                through: event_time,
            },
            causal_frontier: SessionCausalFrontier {
                through_sequence: GlobalSequence::new(value),
                event_time: Some(event_time),
                digest: ContentDigest::from_bytes([0x71; 32]),
            },
        },
        plan_digest: ContentDigest::from_bytes([0x72; 32]),
    }
}
