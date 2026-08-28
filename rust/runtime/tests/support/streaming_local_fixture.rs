// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Filesystem fixture and host doubles for the local streaming source tests.
//!
//! Roots are created under `TMPDIR` and removed on drop, so fixture I/O never
//! touches the system disk. Publication is always by rename: a partial name is
//! written first and moved into place, which is the only publication shape the
//! `local` source accepts.
//!
//! Loaded by an integration test with
//! `#[path = "support/streaming_local_fixture.rs"] mod …;`.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use aiperf_runtime::clock::RealClock;
use aiperf_runtime::streaming::{
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        AcquisitionHorizon, AdmissionHorizon, CheckpointBarrier, CheckpointCut, CheckpointEpoch,
        CheckpointError, CheckpointParticipantId, CheckpointParticipantPlan,
        CommittedParticipantReceipt, CommittedParticipantState, DecodeHorizon, DiscoveryHorizon,
        EventTimeWatermark, OrderedActionHorizon, ParticipantInitialization,
        PreparedParticipantState, StreamRunIdentity, StreamingCheckpointParticipant,
        TerminalActionHorizon,
    },
    checkpoint_backend::{
        CheckpointCommitMetadata, CheckpointGenerationExpectations, LeasedCheckpointGenerationView,
    },
    checkpoints::memory::{MemoryCheckpointBackend, MemoryCheckpointLimits},
    identity::{
        ContentDigest, GlobalSequence, ImmutableObjectIdentity, LogicalReplayRunId,
        SessionCausalFrontier,
    },
    reliability::{
        HandledIssueCut, OrdinaryStreamingIssue, StreamingIssueReportError,
        StreamingIssueReportStatus, StreamingIssueReporter, StreamingIssueReporterEndpoint,
        StreamingIssueReporterHandle, StreamingIssueSummary, StreamingReliabilityError,
    },
    source::{
        AcquiredPartitionAccess, AcquisitionBudget, OpenedStreamingDatasetSource,
        PartitionAccessRequest, SourceEvent, SourcePartition, StreamSourceError,
        StreamingDatasetSourceFactory, StreamingSourcePrepareContext, streaming_stop_channel,
    },
    sources::local::LocalSourceFactory,
    unit::{EventTimeUtc, SourcePosition},
};
use async_trait::async_trait;
use serde_json::value::RawValue;

static ROOT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Temporary publish-by-rename root removed when the fixture drops.
pub struct LocalFixture {
    root: PathBuf,
    /// Logical run bound into every prepared source and checkpoint barrier.
    pub run: StreamRunIdentity,
    /// Semantic namespace every derived partition identity is bound under.
    pub stream_identity: ContentDigest,
}

impl LocalFixture {
    /// Create an empty root under `TMPDIR` with a process-unique name.
    #[must_use]
    pub fn new(tag: &str) -> Self {
        let ordinal = ROOT_COUNTER.fetch_add(1, Ordering::Relaxed);
        let base = std::env::var_os("TMPDIR").map_or_else(std::env::temp_dir, PathBuf::from);
        let root = base.join(format!(
            "aiperf-streaming-local-{tag}-{}-{ordinal}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("fixture root is creatable");
        Self {
            root,
            run: StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x11; 32])),
            stream_identity: ContentDigest::from_bytes([0x51; 32]),
        }
    }

    /// Borrow the absolute fixture root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Publish one immutable object by writing a partial name and renaming it.
    pub fn publish(&self, name: &str, bytes: &[u8]) {
        let staged = self.root.join(format!("{name}.part"));
        let mut file = fs::File::create(&staged).expect("staged partition is creatable");
        file.write_all(bytes).expect("staged partition is writable");
        file.sync_all().expect("staged partition is durable");
        drop(file);
        fs::rename(&staged, self.root.join(name)).expect("publication by rename succeeds");
    }

    /// Remove one published name without touching any other member.
    pub fn unlink(&self, name: &str) {
        fs::remove_file(self.root.join(name)).expect("published partition is removable");
    }

    /// Rewrite one published name in place, retaining its inode.
    pub fn rewrite_in_place(&self, name: &str, bytes: &[u8]) {
        let path = self.root.join(name);
        let mut file = fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&path)
            .expect("published partition is writable in place");
        file.write_all(bytes).expect("in-place rewrite succeeds");
        file.sync_all().expect("in-place rewrite is durable");
    }

    /// Create one symlink inside the root that names an existing member.
    pub fn symlink(&self, link: &str, target: &str) {
        std::os::unix::fs::symlink(self.root.join(target), self.root.join(link))
            .expect("fixture symlink is creatable");
    }

    /// Author a finite-mode configuration over this root.
    #[must_use]
    pub fn finite_config(&self) -> Box<RawValue> {
        self.config(serde_json::json!({ "kind": "finite" }))
    }

    /// Author a follow-mode configuration, optionally naming a seal marker.
    #[must_use]
    pub fn follow_config(&self, seal_marker: Option<&str>) -> Box<RawValue> {
        self.config(serde_json::json!({
            "kind": "follow",
            "seal_marker": seal_marker,
            "accepts_close_write": false,
        }))
    }

    /// Author a reference-manifest configuration naming one root-relative index.
    #[must_use]
    pub fn reference_config(&self, manifest: &str) -> Box<RawValue> {
        self.config(serde_json::json!({
            "kind": "reference",
            "manifest": manifest,
            "max_manifest_bytes": 65_536,
        }))
    }

    fn config(&self, mode: serde_json::Value) -> Box<RawValue> {
        raw(serde_json::json!({
            "root": self.root,
            "mode": mode,
            "suffix": ".jsonl",
            "max_partition_bytes": 1_048_576,
            "max_scan_entries": 64,
            "max_chunk_bytes": 4_096,
            "max_open_attempts": 3,
        }))
    }

    /// Prepare and open one `local` source bound to this fixture's identity.
    pub async fn open(
        &self,
        authored: &RawValue,
        issue_reporter: StreamingIssueReporterHandle,
    ) -> OpenedStreamingDatasetSource {
        self.try_open(authored, issue_reporter)
            .await
            .unwrap_or_else(|error| panic!("local source opens: {error}"))
    }

    /// Prepare and open one `local` source, surfacing an open-time refusal.
    pub async fn try_open(
        &self,
        authored: &RawValue,
        issue_reporter: StreamingIssueReporterHandle,
    ) -> Result<OpenedStreamingDatasetSource, StreamSourceError> {
        let factory = LocalSourceFactory;
        let validated = factory.validate(authored)?;
        let context = StreamingSourcePrepareContext {
            run: self.run,
            stream_semantic_digest: self.stream_identity,
            clock: RealClock::new(),
            acquisition_budget: acquisition_budget(),
            issue_reporter,
        };
        let prepared = factory.prepare(validated, &context)?;
        let (_control, stop) = streaming_stop_channel();
        prepared.open(stop).await
    }
}

impl Drop for LocalFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// Wrap one JSON value as strictly authored source configuration.
#[must_use]
pub fn raw(value: serde_json::Value) -> Box<RawValue> {
    RawValue::from_string(value.to_string()).expect("valid raw configuration")
}

/// Resident-memory and local-snapshot budgets installed for acquisition.
#[must_use]
pub fn acquisition_budget() -> AcquisitionBudget {
    AcquisitionBudget::new(
        StreamingResourceBudget::new(BudgetLimits {
            max_items: 32,
            max_bytes: 1 << 20,
        })
        .expect("valid memory limits"),
        StreamingResourceBudget::new(BudgetLimits {
            max_items: 8,
            max_bytes: 1 << 20,
        })
        .expect("valid disk limits"),
    )
}

/// Acquire one announced partition and return its complete immutable bytes.
///
/// Reading through the sequential authority is what proves the announced
/// identity survives an acquisition round trip.
pub async fn read_partition(partition: &SourcePartition) -> Result<Vec<u8>, StreamSourceError> {
    let budget = acquisition_budget();
    let acquired = partition
        .content()
        .acquire(
            PartitionAccessRequest::Sequential { resume_offset: 0 },
            &budget,
        )
        .await?;
    assert_eq!(
        acquired.identity(),
        partition.content().identity(),
        "acquisition preserves the announced immutable generation"
    );
    let AcquiredPartitionAccess::Sequential(mut reader) = acquired.into_access() else {
        panic!("a sequential request returns sequential access");
    };
    let mut bytes = Vec::new();
    let max = NonZeroUsize::new(1024).expect("nonzero chunk bound");
    while let Some(chunk) = reader.next_chunk(max, &budget).await? {
        bytes.extend_from_slice(chunk.as_bytes());
    }
    Ok(bytes)
}

/// Drive one source until it yields a partition, returning its bytes.
///
/// Panics on any frontier-only or sealed outcome, which the callers treat as a
/// contract violation rather than a skippable event.
pub async fn expect_partition(
    source: &mut dyn aiperf_runtime::streaming::source::StreamingDatasetSource,
) -> (SourcePosition, ImmutableObjectIdentity, Vec<u8>) {
    loop {
        match source.next_event().await {
            Ok(SourceEvent::Partition(partition)) => {
                let position = partition.position();
                let identity = *partition.content().identity();
                let bytes = read_partition(&partition)
                    .await
                    .unwrap_or_else(|error| panic!("partition acquires: {error}"));
                return (position, identity, bytes);
            }
            Ok(SourceEvent::Frontier(_)) => {}
            Ok(SourceEvent::Seal(_)) => panic!("expected a partition, observed a seal"),
            Err(error) => panic!("expected a partition: {error}"),
        }
    }
}

/// Commit one source checkpoint through the in-memory backend and read it back.
///
/// The returned state is exactly what a restart would restore, so the caller
/// can prove the resume contract without forging participant authority.
pub async fn checkpoint_and_commit(
    source: &mut dyn aiperf_runtime::streaming::source::StreamingDatasetSource,
    run: StreamRunIdentity,
) -> CommittedParticipantState {
    let backend =
        MemoryCheckpointBackend::new(memory_checkpoint_limits()).expect("in-memory backend");
    let expectations = CheckpointGenerationExpectations {
        run,
        participant_plan: CheckpointParticipantPlan::new([source.participant_id()])
            .expect("single-participant plan"),
        execution_plan_digest: ContentDigest::from_bytes([0x31; 32]),
        result_plan_digest: ContentDigest::from_bytes([0x32; 32]),
    };
    let barrier = barrier_for(run, 1);
    let prepared = source
        .checkpoint_view(&barrier)
        .await
        .expect("non-destructive participant view");
    let descriptor = prepared.descriptor().clone();

    let mut transaction = backend
        .begin_generation(run, None, expectations.clone())
        .await
        .expect("begin generation");
    transaction
        .stage_participant(prepared)
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
    source
        .checkpoint_committed(&receipt)
        .await
        .expect("post-commit notification");
    source
        .checkpoint_committed(&receipt)
        .await
        .expect("post-commit notification is idempotent");

    let leased = backend
        .open_latest(&run, &expectations)
        .await
        .expect("latest head is readable")
        .expect("a committed head exists");
    let LeasedCheckpointGenerationView::CurrentV4(reader) = leased.view() else {
        panic!("the in-memory backend commits current-v4 generations");
    };
    reader
        .read_participant(&descriptor)
        .await
        .expect("committed participant state is reachable")
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

/// Build one checkpoint barrier whose horizons all sit at `value`.
#[must_use]
pub fn barrier_for(run: StreamRunIdentity, value: u64) -> CheckpointBarrier {
    let event_time = EventTimeUtc::new(i64::try_from(value).expect("barrier value fits i64"))
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

// ---------------------------------------------------------------------------
// Host-owned reliability reporter (test-local, ledger-free)
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct CountingState {
    accepted: Cell<u64>,
    is_closed: Cell<bool>,
}

#[derive(Debug)]
struct CountingEndpoint {
    state: Rc<CountingState>,
}

#[async_trait(?Send)]
impl StreamingIssueReporterEndpoint for CountingEndpoint {
    async fn report(
        &self,
        _issue: OrdinaryStreamingIssue,
    ) -> Result<StreamingIssueReportStatus, StreamingIssueReportError> {
        if self.state.is_closed.get() {
            return Err(StreamingIssueReportError::Closed);
        }
        self.state.accepted.set(self.state.accepted.get() + 1);
        Ok(StreamingIssueReportStatus::Accepted)
    }
}

/// Sole owner of the counting endpoint state used by these tests.
pub struct CountingReporter {
    state: Rc<CountingState>,
    participant_id: CheckpointParticipantId,
    run: StreamRunIdentity,
    initialization: ParticipantInitialization,
}

impl CountingReporter {
    /// Construct a reporter bound to one logical run.
    #[must_use]
    pub fn new(run: StreamRunIdentity) -> Self {
        Self {
            state: Rc::new(CountingState::default()),
            participant_id: CheckpointParticipantId::new("test_issue_reporter"),
            run,
            initialization: ParticipantInitialization::default(),
        }
    }

    /// Return the number of accepted ordinary issues.
    #[must_use]
    pub fn accepted(&self) -> u64 {
        self.state.accepted.get()
    }
}

impl Drop for CountingReporter {
    fn drop(&mut self) {
        self.state.is_closed.set(true);
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for CountingReporter {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        _barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        Err(CheckpointError::ObjectVerification)
    }

    async fn initialize(
        &mut self,
        _state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.initialization.initialize_once()
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        if receipt.run() != &self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(())
    }
}

#[async_trait(?Send)]
impl StreamingIssueReporter for CountingReporter {
    fn handle(&self) -> StreamingIssueReporterHandle {
        StreamingIssueReporterHandle::new(CountingEndpoint {
            state: Rc::clone(&self.state),
        })
    }

    fn summary(&self) -> Result<StreamingIssueSummary, StreamingReliabilityError> {
        Ok(StreamingIssueSummary {
            total: self.state.accepted.get(),
            by_scope: BTreeMap::new(),
            by_class: BTreeMap::new(),
            by_disposition: BTreeMap::new(),
            is_admission_fenced: false,
        })
    }
}
