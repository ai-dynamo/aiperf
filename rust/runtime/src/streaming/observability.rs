// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded stage-boundary observability for the streaming plane.
//!
//! Every observation is recorded where a stage transition already happens in the
//! fused pipeline loop, using a [`Clock`] timestamp the loop has taken anyway. No
//! observation is made inside a token callback, and no path here allocates after
//! construction: distributions are fixed-capacity t-digests and every map is
//! keyed by a closed enum, so the snapshot has a compile-time size bound.
//!
//! The reliability half of the snapshot is not new counting. [`StreamingIssueSummary`]
//! already carries counts by scope, class, and disposition plus the admission-fence
//! flag; this module surfaces that structure and supplies the merge it lacked,
//! rather than defining a parallel one.
//!
//! The split between what is accumulated and what is installed mirrors
//! `HeartbeatAccumulator`: distributions, drop counts, and retry ordinals are
//! accumulated locally by the worker that observes them, while issue counts and
//! checkpoint horizons are *installed* from their authoritative owners at
//! [`StreamingPlaneObserver::refresh_boundary`]. A merge therefore cannot
//! double-count a fact that has a single authoritative issuer.

use std::{collections::BTreeMap, rc::Rc};

use serde::{Deserialize, Serialize};

use crate::{
    cellular::sketch::TDigest,
    clock::Clock,
    metrics_core::report::{
        ReportQueueHighWater, ReportStreamingDistribution, ReportStreamingHorizons,
        ReportStreamingPlane,
    },
    streaming::{
        budget::BudgetSnapshot,
        checkpoint::CheckpointCut,
        failure::StreamingFailureStage,
        identity::GlobalSequence,
        reliability::{
            StreamingIssueClass, StreamingIssueDisposition, StreamingIssueScopeKind,
            StreamingIssueSummary,
        },
    },
};

/// Coarse fixed-size summary of one nanosecond distribution.
///
/// Deliberately not a t-digest: this is the cheap always-on tier, three integers,
/// with no allocation and exact `count`/`sum_ns`/`max_ns`. The same observation
/// also feeds the [`TDigest`] in [`StreamingDistribution`], so the two tiers never
/// disagree about `count`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingDistributionSnapshot {
    /// Number of observations.
    pub count: u64,
    /// Exact sum in nanoseconds. `u128` because a long run can exceed `u64` ns.
    pub sum_ns: u128,
    /// Greatest single observation in nanoseconds.
    pub max_ns: u64,
}

impl StreamingDistributionSnapshot {
    /// Fold another shard's snapshot. Associative, so any reduce order is equal.
    pub fn merge(&mut self, other: &Self) {
        self.count = self.count.saturating_add(other.count);
        self.sum_ns = self.sum_ns.saturating_add(other.sum_ns);
        self.max_ns = self.max_ns.max(other.max_ns);
    }
}

/// One observed nanosecond distribution: exact totals plus a mergeable sketch.
///
/// The sketch is the in-tree [`TDigest`], already `Serialize` and already the
/// cellular sidecar wire format, so a cross-cell merge needs no new machinery.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StreamingDistribution {
    /// Always-on exact totals.
    pub totals: StreamingDistributionSnapshot,
    /// Mergeable quantile sketch, bounded by its compression factor.
    pub sketch: TDigest,
}

impl StreamingDistribution {
    /// Record one nanosecond observation. No allocation past digest capacity.
    pub fn observe_ns(&mut self, elapsed_ns: u64) {
        self.totals.count = self.totals.count.saturating_add(1);
        self.totals.sum_ns = self.totals.sum_ns.saturating_add(u128::from(elapsed_ns));
        self.totals.max_ns = self.totals.max_ns.max(elapsed_ns);
        // `as f64` is lossy past 2^53 ns (~104 days); the exact tier above keeps
        // `count`, `sum_ns`, and `max_ns` authoritative, so only the estimated
        // quantiles are affected.
        self.sketch.add(elapsed_ns as f64);
    }

    /// Fold another shard's distribution.
    pub fn merge(&mut self, other: &Self) {
        self.totals.merge(&other.totals);
        self.sketch.merge(&other.sketch);
    }

    /// Return the exact observation count.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.totals.count
    }

    /// Project this distribution into its feature-independent report shape.
    #[must_use]
    pub fn to_report(&self) -> ReportStreamingDistribution {
        ReportStreamingDistribution {
            count: self.totals.count,
            sum_ns: self.totals.sum_ns,
            max_ns: self.totals.max_ns,
            p50_ns: self.sketch.quantile(0.50),
            p90_ns: self.sketch.quantile(0.90),
            p99_ns: self.sketch.quantile(0.99),
        }
    }
}

/// Permit occupancy and its authored limit for one stage.
///
/// Constructed from [`BudgetSnapshot`], which the budget already maintains, so
/// this reads high-water marks rather than computing them.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueueHighWater {
    /// Peak retained items.
    pub items: usize,
    /// Peak retained bytes.
    pub bytes: usize,
    /// Authored item limit.
    pub item_limit: usize,
    /// Authored byte limit.
    pub byte_limit: usize,
}

impl QueueHighWater {
    /// Build from one budget snapshot and its authored limits.
    #[must_use]
    pub const fn from_budget(snapshot: BudgetSnapshot, item_limit: usize, byte_limit: usize) -> Self {
        Self {
            items: snapshot.high_water_items,
            bytes: snapshot.high_water_bytes,
            item_limit,
            byte_limit,
        }
    }

    /// Whether both peaks are within their authored limits.
    #[must_use]
    pub const fn is_within_limits(&self) -> bool {
        self.items <= self.item_limit && self.bytes <= self.byte_limit
    }

    /// Fold another shard's high-water for the same stage: peaks by max, limits
    /// by max because a heterogeneous shard set has no single authored limit.
    pub fn merge(&mut self, other: &Self) {
        self.items = self.items.max(other.items);
        self.bytes = self.bytes.max(other.bytes);
        self.item_limit = self.item_limit.max(other.item_limit);
        self.byte_limit = self.byte_limit.max(other.byte_limit);
    }

    /// Project into the feature-independent report shape.
    #[must_use]
    pub const fn to_report(&self) -> ReportQueueHighWater {
        ReportQueueHighWater {
            items: self.items,
            bytes: self.bytes,
            item_limit: self.item_limit,
            byte_limit: self.byte_limit,
        }
    }
}

impl From<(BudgetSnapshot, usize, usize)> for QueueHighWater {
    fn from((snapshot, item_limit, byte_limit): (BudgetSnapshot, usize, usize)) -> Self {
        Self::from_budget(snapshot, item_limit, byte_limit)
    }
}

/// Closed set of observable pipeline stages.
///
/// Distinct from [`StreamingFailureStage`], which is the *failure* taxonomy: it
/// has `Acquisition` and `StateBudget` but no `Placement` boundary distinct from
/// `Dispatch`, and no `Terminal`. [`stage_for_failure`] maps between the two, so
/// a failure count and a latency distribution can be read side by side without
/// either enum being bent to fit the other. Adding a stage is a two-place edit.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingStage {
    /// Source discovery and publication.
    Source,
    /// Immutable partition acquisition.
    Acquire,
    /// Format decoding.
    Decode,
    /// Watermark and stable global ordering.
    Order,
    /// Cross-record session coordination.
    Session,
    /// Worker or cell placement and admission.
    Placement,
    /// Action scheduling and issue.
    Action,
    /// Terminal receipt for one issued action.
    Terminal,
    /// Checkpoint-native result commit.
    Result,
}

/// Every observable stage, in declaration order. The snapshot's `queues` map can
/// never hold more keys than this.
pub const STREAMING_STAGES: [StreamingStage; 9] = [
    StreamingStage::Source,
    StreamingStage::Acquire,
    StreamingStage::Decode,
    StreamingStage::Order,
    StreamingStage::Session,
    StreamingStage::Placement,
    StreamingStage::Action,
    StreamingStage::Terminal,
    StreamingStage::Result,
];

impl StreamingStage {
    /// Stable serialized name, identical to this enum's `serde` representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Acquire => "acquire",
            Self::Decode => "decode",
            Self::Order => "order",
            Self::Session => "session",
            Self::Placement => "placement",
            Self::Action => "action",
            Self::Terminal => "terminal",
            Self::Result => "result",
        }
    }

    /// Stable name of the latency distribution this stage folds into.
    #[must_use]
    pub const fn distribution_name(self) -> &'static str {
        match self {
            Self::Source => "publication_lag_ns",
            Self::Acquire => "acquisition_duration_ns",
            Self::Decode => "decode_duration_ns",
            Self::Order => "watermark_lag_ns",
            Self::Session => "causal_wait_ns",
            Self::Placement => "admission_wait_ns",
            Self::Action => "schedule_slip_ns",
            Self::Terminal => "endpoint_ns",
            Self::Result => "result_commit_ns",
        }
    }
}

/// Map a failure stage onto its observable pipeline stage.
///
/// `StateBudget` folds onto `Session` because bounded state admission is charged
/// by the session coordinator; `Checkpoint` folds onto `Result` because a
/// checkpoint failure is observed at the result-commit boundary.
#[must_use]
pub const fn stage_for_failure(stage: StreamingFailureStage) -> StreamingStage {
    match stage {
        StreamingFailureStage::Source => StreamingStage::Source,
        StreamingFailureStage::Acquisition => StreamingStage::Acquire,
        StreamingFailureStage::Decode => StreamingStage::Decode,
        StreamingFailureStage::Ordering => StreamingStage::Order,
        StreamingFailureStage::StateBudget | StreamingFailureStage::Session => {
            StreamingStage::Session
        }
        StreamingFailureStage::Placement => StreamingStage::Placement,
        StreamingFailureStage::Dispatch => StreamingStage::Action,
        StreamingFailureStage::Checkpoint | StreamingFailureStage::Result => StreamingStage::Result,
    }
}

/// Why an observed unit did not become an action.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingDropReason {
    /// Arrived after its event-time window closed.
    Late,
    /// Refused because a bounded capacity was exhausted.
    Overload,
    /// Refused by an authored policy.
    AuthoredPolicy,
    /// Identical to a unit already incorporated.
    Duplicate,
}

/// Every drop reason, in declaration order.
pub const STREAMING_DROP_REASONS: [StreamingDropReason; 4] = [
    StreamingDropReason::Late,
    StreamingDropReason::Overload,
    StreamingDropReason::AuthoredPolicy,
    StreamingDropReason::Duplicate,
];

impl StreamingDropReason {
    /// Stable serialized name, identical to this enum's `serde` representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Late => "late",
            Self::Overload => "overload",
            Self::AuthoredPolicy => "authored_policy",
            Self::Duplicate => "duplicate",
        }
    }
}

/// Greatest sequence for which admission has been scheduled.
///
/// Distinct from `AdmissionHorizon`: scheduling may legitimately lead the
/// committed cut, because a cut is only published at a barrier.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ScheduledActionHorizon(GlobalSequence);

impl Default for ScheduledActionHorizon {
    /// Sequence zero, which is a real position rather than "absent"; absence is
    /// carried by `Option<CheckpointHorizonSnapshot>`.
    fn default() -> Self {
        Self(GlobalSequence::new(0))
    }
}

impl ScheduledActionHorizon {
    /// Construct a scheduling horizon.
    #[must_use]
    pub const fn new(value: GlobalSequence) -> Self {
        Self(value)
    }

    /// Return the underlying global sequence.
    #[must_use]
    pub const fn get(self) -> GlobalSequence {
        self.0
    }
}

/// Every typed horizon the run can truthfully claim.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointHorizonSnapshot {
    /// The complete committed cut, including its terminal horizon.
    pub cut: CheckpointCut,
    /// Scheduling progress, which may lead the committed cut.
    pub scheduled: ScheduledActionHorizon,
}

impl CheckpointHorizonSnapshot {
    /// Project into the feature-independent report shape.
    #[must_use]
    pub fn to_report(&self) -> ReportStreamingHorizons {
        ReportStreamingHorizons {
            ordered: self.cut.ordered.get().get(),
            admitted: self.cut.admitted.get().get(),
            terminal: self.cut.terminal.get().get(),
            scheduled: self.scheduled.get().get(),
        }
    }
}

/// One bounded observability snapshot for the streaming plane.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StreamingPlaneMetrics {
    /// Source publication time to acquisition start.
    pub publication_lag_ns: StreamingDistribution,
    /// Immutable partition acquisition duration.
    pub acquisition_duration_ns: StreamingDistribution,
    /// Decode duration per decode step.
    pub decode_duration_ns: StreamingDistribution,
    /// Event-time watermark lag behind the run clock.
    pub watermark_lag_ns: StreamingDistribution,
    /// Time an action waited for its causal predecessors.
    pub causal_wait_ns: StreamingDistribution,
    /// Time an action waited for an admission permit.
    pub admission_wait_ns: StreamingDistribution,
    /// Difference between an action's scheduler target and its issue time.
    pub schedule_slip_ns: StreamingDistribution,
    /// Endpoint issue to terminal receipt.
    pub endpoint_ns: StreamingDistribution,
    /// Checkpoint-native result commit duration.
    pub result_commit_ns: StreamingDistribution,
    /// Permit high-water by stage.
    pub queues: BTreeMap<StreamingStage, QueueHighWater>,
    /// Drop counts by reason.
    pub drops_by_reason: BTreeMap<StreamingDropReason, u64>,
    /// Units classified as identical duplicates.
    pub duplicate_count: u64,
    /// Proven partition/sequence holes.
    pub gap_count: u64,
    /// Reliability counts by scope, class, and disposition, plus fence state.
    ///
    /// This is the reliability module's own summary type, surfaced rather than
    /// duplicated.
    pub issues: StreamingIssueSummary,
    /// Retry ordinals observed, by ordinal value.
    ///
    /// Retry ordinals are otherwise ephemeral: they exist only at the retry
    /// disposition and nothing retains them, so counting them here is the only
    /// way a run report can show retry pressure.
    pub retry_ordinals: BTreeMap<u32, u64>,
    /// Actions whose single terminal receipt was a failure. Distinct from a
    /// failed run: a failed action is truthful terminal membership, whereas a
    /// failed run has no truthful membership at all.
    pub failed_terminal_actions: u64,
    /// Derived sinks that could not complete for this run, by sink identity.
    pub incomplete_derived_sinks: BTreeMap<String, u64>,
    /// Typed horizons at snapshot time.
    pub checkpoint_horizons: Option<CheckpointHorizonSnapshot>,
}

impl StreamingPlaneMetrics {
    /// Borrow the distribution one stage folds into.
    #[must_use]
    pub const fn distribution(&self, stage: StreamingStage) -> &StreamingDistribution {
        match stage {
            StreamingStage::Source => &self.publication_lag_ns,
            StreamingStage::Acquire => &self.acquisition_duration_ns,
            StreamingStage::Decode => &self.decode_duration_ns,
            StreamingStage::Order => &self.watermark_lag_ns,
            StreamingStage::Session => &self.causal_wait_ns,
            StreamingStage::Placement => &self.admission_wait_ns,
            StreamingStage::Action => &self.schedule_slip_ns,
            StreamingStage::Terminal => &self.endpoint_ns,
            StreamingStage::Result => &self.result_commit_ns,
        }
    }

    fn distribution_mut(&mut self, stage: StreamingStage) -> &mut StreamingDistribution {
        match stage {
            StreamingStage::Source => &mut self.publication_lag_ns,
            StreamingStage::Acquire => &mut self.acquisition_duration_ns,
            StreamingStage::Decode => &mut self.decode_duration_ns,
            StreamingStage::Order => &mut self.watermark_lag_ns,
            StreamingStage::Session => &mut self.causal_wait_ns,
            StreamingStage::Placement => &mut self.admission_wait_ns,
            StreamingStage::Action => &mut self.schedule_slip_ns,
            StreamingStage::Terminal => &mut self.endpoint_ns,
            StreamingStage::Result => &mut self.result_commit_ns,
        }
    }

    /// Fold another worker's or cell's snapshot.
    ///
    /// Associative and commutative except for `checkpoint_horizons`, which takes
    /// the greater committed terminal cut so any reduce order agrees.
    pub fn merge(&mut self, other: &Self) {
        for stage in STREAMING_STAGES {
            let folded = other.distribution(stage).clone();
            self.distribution_mut(stage).merge(&folded);
        }
        for (stage, queue) in &other.queues {
            self.queues.entry(*stage).or_default().merge(queue);
        }
        for (reason, count) in &other.drops_by_reason {
            *self.drops_by_reason.entry(*reason).or_default() += count;
        }
        self.duplicate_count = self.duplicate_count.saturating_add(other.duplicate_count);
        self.gap_count = self.gap_count.saturating_add(other.gap_count);
        self.issues.merge(&other.issues);
        for (ordinal, count) in &other.retry_ordinals {
            *self.retry_ordinals.entry(*ordinal).or_default() += count;
        }
        self.failed_terminal_actions = self
            .failed_terminal_actions
            .saturating_add(other.failed_terminal_actions);
        for (sink, count) in &other.incomplete_derived_sinks {
            *self
                .incomplete_derived_sinks
                .entry(sink.clone())
                .or_default() += count;
        }
        self.checkpoint_horizons = greater_horizons(
            self.checkpoint_horizons.take(),
            other.checkpoint_horizons.clone(),
        );
    }

    /// Project into the feature-independent report shape.
    ///
    /// The `String` keys are the closed enums' `serde` names, so the serialized
    /// report stays a stable wire contract without `metrics_core` importing a
    /// feature-gated type.
    #[must_use]
    pub fn to_report(&self) -> ReportStreamingPlane {
        let mut distributions = BTreeMap::new();
        for stage in STREAMING_STAGES {
            let distribution = self.distribution(stage);
            if distribution.count() > 0 {
                distributions.insert(stage.distribution_name().to_string(), distribution.to_report());
            }
        }
        ReportStreamingPlane {
            distributions,
            queues: self
                .queues
                .iter()
                .map(|(stage, queue)| (stage.as_str().to_string(), queue.to_report()))
                .collect(),
            drops_by_reason: self
                .drops_by_reason
                .iter()
                .map(|(reason, count)| (reason.as_str().to_string(), *count))
                .collect(),
            issues_by_scope: self
                .issues
                .by_scope
                .iter()
                .map(|(scope, count)| (scope_name(*scope).to_string(), *count))
                .collect(),
            issues_by_class: self
                .issues
                .by_class
                .iter()
                .map(|(class, count)| (class_name(*class).to_string(), *count))
                .collect(),
            issues_by_disposition: self
                .issues
                .by_disposition
                .iter()
                .map(|(disposition, count)| (disposition_name(*disposition).to_string(), *count))
                .collect(),
            is_admission_fenced: self.issues.is_admission_fenced,
            retry_ordinals: self.retry_ordinals.clone(),
            failed_terminal_actions: self.failed_terminal_actions,
            duplicate_count: self.duplicate_count,
            gap_count: self.gap_count,
            incomplete_derived_sinks: self.incomplete_derived_sinks.clone(),
            horizons: self
                .checkpoint_horizons
                .as_ref()
                .map(CheckpointHorizonSnapshot::to_report),
        }
    }
}

/// Take the horizon snapshot with the greater committed terminal cut.
///
/// Ties break toward the left operand, which is safe because two snapshots with
/// the same terminal horizon describe the same committed prefix.
fn greater_horizons(
    left: Option<CheckpointHorizonSnapshot>,
    right: Option<CheckpointHorizonSnapshot>,
) -> Option<CheckpointHorizonSnapshot> {
    match (left, right) {
        (None, right) => right,
        (left, None) => left,
        (Some(left), Some(right)) => {
            if right.cut.terminal.get() > left.cut.terminal.get() {
                Some(right)
            } else {
                Some(left)
            }
        }
    }
}

/// Stable serialized name for one issue scope.
#[must_use]
pub const fn scope_name(scope: StreamingIssueScopeKind) -> &'static str {
    match scope {
        StreamingIssueScopeKind::Run => "run",
        StreamingIssueScopeKind::Partition => "partition",
        StreamingIssueScopeKind::Record => "record",
        StreamingIssueScopeKind::Session => "session",
        StreamingIssueScopeKind::Action => "action",
        StreamingIssueScopeKind::Export => "export",
        StreamingIssueScopeKind::CheckpointAttempt => "checkpoint_attempt",
    }
}

/// Stable serialized name for one reliability class.
#[must_use]
pub const fn class_name(class: StreamingIssueClass) -> &'static str {
    match class {
        StreamingIssueClass::Retryable => "retryable",
        StreamingIssueClass::Permanent => "permanent",
        StreamingIssueClass::Invariant => "invariant",
        StreamingIssueClass::Capacity => "capacity",
    }
}

/// Stable serialized name for one host disposition.
#[must_use]
pub const fn disposition_name(disposition: StreamingIssueDisposition) -> &'static str {
    match disposition {
        StreamingIssueDisposition::Retry => "retry",
        StreamingIssueDisposition::Backpressure => "backpressure",
        StreamingIssueDisposition::Quarantine => "quarantine",
        StreamingIssueDisposition::Hole => "hole",
        StreamingIssueDisposition::Continue => "continue",
        StreamingIssueDisposition::TerminalActionReceipt => "terminal_action_receipt",
        StreamingIssueDisposition::ExportIncomplete => "export_incomplete",
        StreamingIssueDisposition::FailRun => "fail_run",
    }
}

/// An open stage interval, closed by [`StreamingPlaneObserver::close_span`].
///
/// Carries only the stage and the clock reading taken when the interval opened,
/// so it is `Copy` and costs no allocation to hold across a stage boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageSpan {
    stage: StreamingStage,
    started_ns: i64,
}

impl StageSpan {
    /// Return the stage this span will fold into.
    #[must_use]
    pub const fn stage(&self) -> StreamingStage {
        self.stage
    }
}

/// Worker-local stage-boundary recorder.
///
/// Held by the fused pipeline loop as a plain field. Every method is synchronous,
/// takes `&mut self`, allocates nothing on an already-seen key, and takes no
/// lock — the hot-path constraint is satisfied structurally, not by discipline.
///
/// This type is deliberately concrete rather than a trait object: there is
/// exactly one implementation, it is worker-local, and a `dyn` call per stage
/// boundary would be pure overhead on the path it exists to measure.
pub struct StreamingPlaneObserver {
    metrics: StreamingPlaneMetrics,
    clock: Rc<dyn Clock>,
}

impl StreamingPlaneObserver {
    /// Build an empty observer bound to the run clock.
    #[must_use]
    pub fn new(clock: Rc<dyn Clock>) -> Self {
        Self {
            metrics: StreamingPlaneMetrics::default(),
            clock,
        }
    }

    /// Open a stage interval at the current clock reading.
    #[must_use]
    pub fn open_span(&self, stage: StreamingStage) -> StageSpan {
        StageSpan {
            stage,
            started_ns: self.clock.now_ns(),
        }
    }

    /// Close a stage interval, folding its elapsed nanoseconds.
    ///
    /// A span whose end reads earlier than its start — which a virtual clock
    /// never produces and a monotonic clock only produces across a restored
    /// incarnation — folds as zero rather than wrapping.
    pub fn close_span(&mut self, span: StageSpan) {
        let elapsed_ns = self.clock.now_ns().saturating_sub(span.started_ns).max(0) as u64;
        self.observe_stage_ns(span.stage, elapsed_ns);
    }

    /// Record elapsed time for one stage-boundary transition.
    pub fn observe_stage_ns(&mut self, stage: StreamingStage, elapsed_ns: u64) {
        self.metrics.distribution_mut(stage).observe_ns(elapsed_ns);
    }

    /// Record one drop with its exact reason.
    ///
    /// `Duplicate` additionally advances `duplicate_count`, so the two views of
    /// duplicate suppression can never disagree.
    pub fn observe_drop(&mut self, reason: StreamingDropReason) {
        *self.metrics.drops_by_reason.entry(reason).or_default() += 1;
        if matches!(reason, StreamingDropReason::Duplicate) {
            self.metrics.duplicate_count = self.metrics.duplicate_count.saturating_add(1);
        }
    }

    /// Record one proven partition or sequence hole.
    pub fn observe_gap(&mut self) {
        self.metrics.gap_count = self.metrics.gap_count.saturating_add(1);
    }

    /// Refresh one stage's permit high-water from its budget.
    pub fn observe_queue(
        &mut self,
        stage: StreamingStage,
        snapshot: BudgetSnapshot,
        item_limit: usize,
        byte_limit: usize,
    ) {
        let observed = QueueHighWater::from_budget(snapshot, item_limit, byte_limit);
        self.metrics
            .queues
            .entry(stage)
            .or_default()
            .merge(&observed);
    }

    /// Record one retry ordinal at the disposition boundary.
    ///
    /// A retry is telemetry, not membership: it never advances
    /// `failed_terminal_actions`, which only a terminal receipt does.
    pub fn observe_retry(&mut self, retry_ordinal: u32) {
        *self.metrics.retry_ordinals.entry(retry_ordinal).or_default() += 1;
    }

    /// Record one action finalized by a failed terminal receipt.
    ///
    /// This is truthful terminal membership for a single action and says nothing
    /// about the run's own disposition; a failed run is the separate `FailRun`
    /// disposition carried by the reliability summary.
    pub fn observe_failed_terminal_action(&mut self) {
        self.metrics.failed_terminal_actions =
            self.metrics.failed_terminal_actions.saturating_add(1);
    }

    /// Record one derived sink that could not complete for this run.
    pub fn observe_incomplete_derived_sink(&mut self, sink: &str) {
        *self
            .metrics
            .incomplete_derived_sinks
            .entry(sink.to_string())
            .or_default() += 1;
    }

    /// Install the reliability summary and horizons at a report/checkpoint boundary.
    ///
    /// Counts installed here come from their authoritative owner rather than
    /// being accumulated locally, which is what makes a cross-shard merge safe.
    pub fn refresh_boundary(
        &mut self,
        issues: StreamingIssueSummary,
        horizons: CheckpointHorizonSnapshot,
    ) {
        self.metrics.issues = issues;
        self.metrics.checkpoint_horizons = Some(horizons);
    }

    /// Install only the reliability summary, when no cut has yet been committed.
    pub fn refresh_issues(&mut self, issues: StreamingIssueSummary) {
        self.metrics.issues = issues;
    }

    /// Produce the current bounded snapshot.
    #[must_use]
    pub fn snapshot(&self) -> StreamingPlaneMetrics {
        self.metrics.clone()
    }

    /// Borrow the accumulated metrics without cloning.
    #[must_use]
    pub const fn metrics(&self) -> &StreamingPlaneMetrics {
        &self.metrics
    }
}
