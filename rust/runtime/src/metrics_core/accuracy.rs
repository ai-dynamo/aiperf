// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Accuracy accumulator and analyzer.
//!
//! This is the first-class accumulator/analyzer pair described in
//! `specs/accuracy.md`: graded
//! responses carry a real correlation id, are summarized over the same phase/time
//! windows as performance metrics, and can be joined with optional metric or energy
//! summaries by an analyzer.

use crate::metrics_core::{AccumulatorSummary, ExportContext, MetricTag, Phase};
use petgraph::algo::toposort;
use petgraph::graphmap::DiGraphMap;
use rustc_hash::FxHashMap;
use serde::Serialize;
use std::any::Any;
use std::collections::BTreeMap;
use std::fmt::{self, Display};

/// Record type routed to [`AccuracyAccumulator`].
pub const ACCURACY_RECORD_TYPE: &str = "accuracy_records";

/// The single threshold used when a grader reports a score instead of a boolean.
pub const LIGHTEVAL_CORRECTNESS_THRESHOLD: f64 = 0.5;

/// Per-request correlation id.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct CorrelationId(String);

impl CorrelationId {
    /// Builds a correlation id.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for CorrelationId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for CorrelationId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

/// Accuracy task id, for example `mmlu_pro.chemistry`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct TaskId(String);

impl TaskId {
    /// Builds a task id.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for TaskId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for TaskId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

/// Full grading result retained for per-record export and aggregation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GradingResult {
    /// Whether the answer is correct under the shared accuracy policy.
    pub correct: bool,
    /// Whether the grader had to fall through to an unparsed/fallback path.
    pub unparsed: bool,
    /// Optional grader confidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// Extracted answer text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extracted: Option<String>,
    /// Public ground-truth answer when the canonical evaluator elects to disclose it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ground_truth: Option<String>,
    /// Optional grader reasoning retained for per-record output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

impl GradingResult {
    /// Builds a grading result from a lighteval-style numeric score.
    pub fn from_score(score: f64, unparsed: bool, ground_truth: impl Into<String>) -> Self {
        Self {
            // Python's lighteval bridge uses a strict `score > 0.5` decision.
            // Keeping that boundary here makes every native score consumer share
            // one policy.
            correct: score.is_finite() && score > LIGHTEVAL_CORRECTNESS_THRESHOLD,
            unparsed,
            confidence: score.is_finite().then_some(score),
            extracted: None,
            ground_truth: Some(ground_truth.into()),
            reasoning: None,
        }
    }
}

/// One graded response.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AccuracyRecord {
    /// Real per-request correlation id from the scheduling/dataset seam.
    pub correlation_id: CorrelationId,
    /// Accuracy task id.
    pub task: TaskId,
    /// Warmup/profiling phase.
    pub phase: Phase,
    /// Request start timestamp in nanoseconds.
    pub start_ns: i64,
    /// Request end timestamp in nanoseconds.
    pub end_ns: i64,
    /// Full grading result.
    pub result: GradingResult,
}

/// Wilson confidence interval for a binomial rate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ConfidenceInterval {
    /// Lower bound.
    pub low: f64,
    /// Upper bound.
    pub high: f64,
}

/// Rollup for one task or the overall population.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AccuracyRollup {
    /// Number of records in the rollup.
    pub n: usize,
    /// Number of correct records.
    pub correct_count: usize,
    /// Number of unparsed records.
    pub unparsed_count: usize,
    /// Correct fraction.
    pub accuracy: Option<f64>,
    /// Unparsed fraction.
    pub unparsed_rate: Option<f64>,
    /// Mean confidence over present confidence values.
    pub mean_confidence: Option<f64>,
    /// Wilson score interval for accuracy.
    pub ci: Option<ConfidenceInterval>,
}

impl AccuracyRollup {
    fn empty() -> Self {
        Self {
            n: 0,
            correct_count: 0,
            unparsed_count: 0,
            accuracy: None,
            unparsed_rate: None,
            mean_confidence: None,
            ci: None,
        }
    }
}

/// Accuracy summary over an export context.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AccuracySummary {
    /// Overall rollup.
    pub overall: AccuracyRollup,
    /// Per-task rollups keyed by task id.
    pub per_task: BTreeMap<TaskId, AccuracyRollup>,
}

impl AccuracySummary {
    /// Number of summarized records.
    pub fn n(&self) -> usize {
        self.overall.n
    }

    /// Number of correct summarized records.
    pub fn correct_count(&self) -> usize {
        self.overall.correct_count
    }
}

/// Rejected accuracy-record reasons.
#[derive(Debug, Clone, PartialEq)]
pub enum AccuracyRecordError {
    /// Correlation ids are required for typed request/evaluator association.
    EmptyCorrelationId,
    /// Task ids are required for per-task rollups.
    EmptyTaskId,
    /// The record's terminal timestamp precedes its start timestamp.
    InvalidInterval {
        /// Request start timestamp.
        start_ns: i64,
        /// Request terminal timestamp.
        end_ns: i64,
    },
    /// A correlation id may identify exactly one graded response.
    DuplicateCorrelationId(CorrelationId),
    /// Confidence values must be finite probabilities.
    InvalidConfidence(f64),
}

impl Display for AccuracyRecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCorrelationId => write!(f, "accuracy correlation id must not be empty"),
            Self::EmptyTaskId => write!(f, "accuracy task id must not be empty"),
            Self::InvalidInterval { start_ns, end_ns } => write!(
                f,
                "accuracy record end_ns ({end_ns}) precedes start_ns ({start_ns})"
            ),
            Self::DuplicateCorrelationId(id) => {
                write!(f, "duplicate accuracy correlation id {:?}", id.as_str())
            }
            Self::InvalidConfidence(value) => write!(
                f,
                "accuracy confidence must be finite and within 0..=1, received {value}"
            ),
        }
    }
}

impl std::error::Error for AccuracyRecordError {}

#[derive(Debug, Default)]
struct AccuracyColumns {
    correlation_ids: Vec<CorrelationId>,
    phases: Vec<Phase>,
    start_ns: Vec<i64>,
    end_ns: Vec<i64>,
    correct: Vec<bool>,
    unparsed: Vec<bool>,
    confidence: Vec<Option<f64>>,
    extracted: Vec<Option<String>>,
    ground_truth: Vec<Option<String>>,
    reasoning: Vec<Option<String>>,
}

impl AccuracyColumns {
    fn len(&self) -> usize {
        self.correlation_ids.len()
    }

    fn push(&mut self, record: AccuracyRecord) -> usize {
        let index = self.len();
        self.correlation_ids.push(record.correlation_id);
        self.phases.push(record.phase);
        self.start_ns.push(record.start_ns);
        self.end_ns.push(record.end_ns);
        self.correct.push(record.result.correct);
        self.unparsed.push(record.result.unparsed);
        self.confidence.push(record.result.confidence);
        self.extracted.push(record.result.extracted);
        self.ground_truth.push(record.result.ground_truth);
        self.reasoning.push(record.result.reasoning);
        debug_assert!(self.columns_have_equal_length());
        index
    }

    fn columns_have_equal_length(&self) -> bool {
        let len = self.len();
        self.phases.len() == len
            && self.start_ns.len() == len
            && self.end_ns.len() == len
            && self.correct.len() == len
            && self.unparsed.len() == len
            && self.confidence.len() == len
            && self.extracted.len() == len
            && self.ground_truth.len() == len
            && self.reasoning.len() == len
    }

    fn record(&self, task: &TaskId, index: usize) -> AccuracyRecord {
        AccuracyRecord {
            correlation_id: self.correlation_ids[index].clone(),
            task: task.clone(),
            phase: self.phases[index],
            start_ns: self.start_ns[index],
            end_ns: self.end_ns[index],
            result: GradingResult {
                correct: self.correct[index],
                unparsed: self.unparsed[index],
                confidence: self.confidence[index],
                extracted: self.extracted[index].clone(),
                ground_truth: self.ground_truth[index].clone(),
                reasoning: self.reasoning[index].clone(),
            },
        }
    }
}

/// Columnar accuracy accumulator.
#[derive(Debug, Default)]
pub struct AccuracyAccumulator {
    tasks: BTreeMap<TaskId, AccuracyColumns>,
    insertion_order: Vec<(TaskId, usize)>,
    by_corr: FxHashMap<CorrelationId, (TaskId, usize)>,
}

impl AccuracyAccumulator {
    /// Record type consumed by this accumulator.
    pub const RECORD_TYPE: &'static str = ACCURACY_RECORD_TYPE;

    /// Builds an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of accumulated responses.
    pub fn len(&self) -> usize {
        self.insertion_order.len()
    }

    /// Returns true when no responses have been accumulated.
    pub fn is_empty(&self) -> bool {
        self.insertion_order.is_empty()
    }

    /// Adds one validated graded response.
    pub fn process_record(&mut self, record: AccuracyRecord) -> Result<(), AccuracyRecordError> {
        validate_record(&record)?;
        if self.by_corr.contains_key(&record.correlation_id) {
            return Err(AccuracyRecordError::DuplicateCorrelationId(
                record.correlation_id,
            ));
        }
        let correlation_id = record.correlation_id.clone();
        let task = record.task.clone();
        let index = self.tasks.entry(task.clone()).or_default().push(record);
        self.by_corr.insert(correlation_id, (task.clone(), index));
        self.insertion_order.push((task, index));
        Ok(())
    }

    /// Returns a cloned record by correlation id.
    pub fn record_for(&self, correlation_id: &CorrelationId) -> Option<AccuracyRecord> {
        self.by_corr.get(correlation_id).and_then(|(task, index)| {
            self.tasks
                .get(task)
                .map(|columns| columns.record(task, *index))
        })
    }

    /// Returns all records in insertion order for typed per-record export.
    pub fn records(&self) -> Vec<AccuracyRecord> {
        self.insertion_order
            .iter()
            .map(|(task, index)| self.tasks[task].record(task, *index))
            .collect()
    }

    /// Merges another per-worker accumulator without partially mutating on a
    /// duplicate correlation id.
    pub fn try_merge(&mut self, other: Self) -> Result<(), AccuracyRecordError> {
        if let Some(duplicate) = other
            .by_corr
            .keys()
            .find(|correlation_id| self.by_corr.contains_key(*correlation_id))
        {
            return Err(AccuracyRecordError::DuplicateCorrelationId(
                duplicate.clone(),
            ));
        }
        for record in other.records() {
            self.process_record(record)?;
        }
        Ok(())
    }

    /// Builds the insertion-order mask for records fully contained in a half-open
    /// time range. A record ending exactly at `end_ns` is contained because its
    /// own interval is also half-open.
    pub fn query_time_range(&self, start_ns: i64, end_ns: i64) -> Vec<bool> {
        self.insertion_order
            .iter()
            .map(|(task, index)| {
                let columns = &self.tasks[task];
                columns.start_ns[*index] >= start_ns && columns.end_ns[*index] <= end_ns
            })
            .collect()
    }

    /// Exports a summary for a phase/time context.
    pub fn export_results(&self, ctx: ExportContext) -> AccuracySummary {
        self.compute_results_for_context(ctx)
    }

    fn compute_results_for_context(&self, ctx: ExportContext) -> AccuracySummary {
        let mut overall = RollupBuilder::default();
        let mut per_task = BTreeMap::<TaskId, RollupBuilder>::new();
        for (task, columns) in &self.tasks {
            for index in 0..columns.len() {
                if !ctx.contains(
                    columns.phases[index],
                    columns.start_ns[index],
                    columns.end_ns[index],
                ) {
                    continue;
                }
                overall.push_values(
                    columns.correct[index],
                    columns.unparsed[index],
                    columns.confidence[index],
                );
                per_task.entry(task.clone()).or_default().push_values(
                    columns.correct[index],
                    columns.unparsed[index],
                    columns.confidence[index],
                );
            }
        }
        AccuracySummary {
            overall: overall.finish(),
            per_task: per_task
                .into_iter()
                .map(|(task, builder)| (task, builder.finish()))
                .collect(),
        }
    }
}

fn validate_record(record: &AccuracyRecord) -> Result<(), AccuracyRecordError> {
    if record.correlation_id.as_str().trim().is_empty() {
        return Err(AccuracyRecordError::EmptyCorrelationId);
    }
    if record.task.as_str().trim().is_empty() {
        return Err(AccuracyRecordError::EmptyTaskId);
    }
    if record.end_ns < record.start_ns {
        return Err(AccuracyRecordError::InvalidInterval {
            start_ns: record.start_ns,
            end_ns: record.end_ns,
        });
    }
    if let Some(confidence) = record.result.confidence
        && (!confidence.is_finite() || !(0.0..=1.0).contains(&confidence))
    {
        return Err(AccuracyRecordError::InvalidConfidence(confidence));
    }
    Ok(())
}

#[derive(Debug, Default)]
struct RollupBuilder {
    n: usize,
    correct_count: usize,
    unparsed_count: usize,
    confidence_sum: f64,
    confidence_count: usize,
}

impl RollupBuilder {
    fn push_values(&mut self, correct: bool, unparsed: bool, confidence: Option<f64>) {
        self.n += 1;
        self.correct_count += usize::from(correct);
        self.unparsed_count += usize::from(unparsed);
        if let Some(confidence) = confidence {
            self.confidence_sum += confidence;
            self.confidence_count += 1;
        }
    }

    fn finish(self) -> AccuracyRollup {
        if self.n == 0 {
            return AccuracyRollup::empty();
        }
        let accuracy = self.correct_count as f64 / self.n as f64;
        AccuracyRollup {
            n: self.n,
            correct_count: self.correct_count,
            unparsed_count: self.unparsed_count,
            accuracy: Some(accuracy),
            unparsed_rate: Some(self.unparsed_count as f64 / self.n as f64),
            mean_confidence: (self.confidence_count > 0)
                .then_some(self.confidence_sum / self.confidence_count as f64),
            ci: Some(wilson_interval(self.correct_count, self.n)),
        }
    }
}

fn wilson_interval(successes: usize, n: usize) -> ConfidenceInterval {
    if n == 0 {
        return ConfidenceInterval {
            low: 0.0,
            high: 0.0,
        };
    }
    let z = 1.959_963_984_540_054_f64;
    let n = n as f64;
    let phat = successes as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = phat + z2 / (2.0 * n);
    let margin = z * ((phat * (1.0 - phat) + z2 / (4.0 * n)) / n).sqrt();
    ConfidenceInterval {
        low: ((center - margin) / denom).clamp(0.0, 1.0),
        high: ((center + margin) / denom).clamp(0.0, 1.0),
    }
}

/// Accumulator identities available to analyzers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AccumulatorType {
    /// Accuracy accumulator output.
    Accuracy,
    /// Metrics accumulator output.
    MetricResults,
    /// GPU/energy telemetry output.
    GpuTelemetry,
}

/// Analyzer output identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AnalyzerType {
    /// Accuracy results analyzer.
    AccuracyResults,
}

/// Type-erased output context passed to analyzers.
#[derive(Default)]
pub struct SummaryContext {
    accumulator_outputs: FxHashMap<AccumulatorType, Box<dyn Any>>,
    analyzer_outputs: FxHashMap<AnalyzerType, Box<dyn Any>>,
}

impl SummaryContext {
    /// Builds an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts an accumulator output.
    pub fn insert_accumulator<T: Any>(&mut self, kind: AccumulatorType, output: T) {
        self.accumulator_outputs.insert(kind, Box::new(output));
    }

    /// Inserts an analyzer output.
    pub fn insert_analyzer<T: Any>(&mut self, kind: AnalyzerType, output: T) {
        self.analyzer_outputs.insert(kind, Box::new(output));
    }

    /// Gets an accumulator output by type.
    pub fn get_accumulator<T: Any>(&self, kind: AccumulatorType) -> Option<&T> {
        self.accumulator_outputs
            .get(&kind)
            .and_then(|output| output.downcast_ref::<T>())
    }

    /// Gets an analyzer output by type.
    pub fn get_analyzer<T: Any>(&self, kind: AnalyzerType) -> Option<&T> {
        self.analyzer_outputs
            .get(&kind)
            .and_then(|output| output.downcast_ref::<T>())
    }

    fn has_accumulator(&self, kind: AccumulatorType) -> bool {
        self.accumulator_outputs.contains_key(&kind)
    }

    fn has_analyzer(&self, kind: AnalyzerType) -> bool {
        self.analyzer_outputs.contains_key(&kind)
    }
}

/// Analyzer trait with real dependency declarations.
pub trait Analyzer {
    /// Typed analyzer output.
    type Output: Clone + 'static;

    /// Analyzer identity.
    const TYPE: AnalyzerType;
    /// Required accumulator outputs.
    const REQUIRED: &'static [AccumulatorType];
    /// Optional accumulator outputs.
    const OPTIONAL: &'static [AccumulatorType];
    /// Required analyzer outputs.
    const REQUIRED_ANALYZERS: &'static [AnalyzerType] = &[];

    /// Summarizes from the context.
    fn summarize(&self, ctx: &SummaryContext) -> Option<Self::Output>;
}

trait ErasedAnalyzer {
    fn ty(&self) -> AnalyzerType;
    fn required_accumulators(&self) -> &'static [AccumulatorType];
    fn required_analyzers(&self) -> &'static [AnalyzerType];
    fn summarize_into(&self, ctx: &mut SummaryContext) -> bool;
}

impl<T> ErasedAnalyzer for T
where
    T: Analyzer,
{
    fn ty(&self) -> AnalyzerType {
        T::TYPE
    }

    fn required_accumulators(&self) -> &'static [AccumulatorType] {
        T::REQUIRED
    }

    fn required_analyzers(&self) -> &'static [AnalyzerType] {
        T::REQUIRED_ANALYZERS
    }

    fn summarize_into(&self, ctx: &mut SummaryContext) -> bool {
        if let Some(output) = self.summarize(ctx) {
            ctx.insert_analyzer(T::TYPE, output);
            true
        } else {
            false
        }
    }
}

/// Runs analyzers in dependency order, skipping missing required accumulators.
pub struct AnalyzerRunner {
    analyzers: Vec<Box<dyn ErasedAnalyzer>>,
}

impl AnalyzerRunner {
    /// Builds an empty analyzer runner.
    pub fn new() -> Self {
        Self {
            analyzers: Vec::new(),
        }
    }

    /// Adds an analyzer.
    pub fn push<A: Analyzer + 'static>(&mut self, analyzer: A) {
        self.analyzers.push(Box::new(analyzer));
    }

    /// Runs analyzers and returns the outputs in the context.
    pub fn run(&self, ctx: &mut SummaryContext) -> Result<Vec<AnalyzerType>, AnalyzerRunError> {
        let mut by_type = FxHashMap::<AnalyzerType, usize>::default();
        for (index, analyzer) in self.analyzers.iter().enumerate() {
            if by_type.insert(analyzer.ty(), index).is_some() {
                return Err(AnalyzerRunError::DuplicateAnalyzer {
                    analyzer: analyzer.ty(),
                });
            }
        }
        let mut graph = DiGraphMap::<AnalyzerType, ()>::new();
        for analyzer in &self.analyzers {
            graph.add_node(analyzer.ty());
            for dep in analyzer.required_analyzers() {
                if !by_type.contains_key(dep) {
                    return Err(AnalyzerRunError::MissingAnalyzerDependency {
                        analyzer: analyzer.ty(),
                        dependency: *dep,
                    });
                }
                graph.add_edge(*dep, analyzer.ty(), ());
            }
        }
        let order = toposort(&graph, None).map_err(|cycle| AnalyzerRunError::Cycle {
            analyzer: cycle.node_id(),
        })?;
        let mut ran = Vec::new();
        for analyzer_type in order {
            let analyzer = &self.analyzers[by_type[&analyzer_type]];
            if analyzer
                .required_accumulators()
                .iter()
                .all(|required| ctx.has_accumulator(*required))
                && analyzer
                    .required_analyzers()
                    .iter()
                    .all(|required| ctx.has_analyzer(*required))
                && analyzer.summarize_into(ctx)
            {
                ran.push(analyzer_type);
            }
        }
        Ok(ran)
    }
}

impl Default for AnalyzerRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyzer dependency execution errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalyzerRunError {
    /// An analyzer identity was registered more than once.
    DuplicateAnalyzer {
        /// Duplicate analyzer identity.
        analyzer: AnalyzerType,
    },
    /// A required analyzer dependency is not registered.
    MissingAnalyzerDependency {
        /// Analyzer that declared the dependency.
        analyzer: AnalyzerType,
        /// Missing dependency.
        dependency: AnalyzerType,
    },
    /// Analyzer dependency graph has a cycle.
    Cycle {
        /// Analyzer in the cycle.
        analyzer: AnalyzerType,
    },
}

impl Display for AnalyzerRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateAnalyzer { analyzer } => {
                write!(f, "analyzer {analyzer:?} was registered more than once")
            }
            Self::MissingAnalyzerDependency {
                analyzer,
                dependency,
            } => write!(
                f,
                "analyzer {analyzer:?} requires unregistered analyzer {dependency:?}"
            ),
            Self::Cycle { analyzer } => {
                write!(f, "analyzer dependency cycle contains {analyzer:?}")
            }
        }
    }
}

impl std::error::Error for AnalyzerRunError {}

/// Optional energy telemetry summary used for accuracy-per-energy joins.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyEfficiencySummary {
    /// Total energy in joules over the summary window.
    pub total_energy_j: f64,
}

/// Accuracy joined to throughput/goodput.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AccuracyAtLoad {
    /// Overall accuracy for the same context.
    pub accuracy: Option<f64>,
    /// Goodput when present.
    pub goodput: Option<f64>,
    /// Request throughput when present.
    pub request_throughput: Option<f64>,
    /// Correct answers per second using goodput if available, otherwise request throughput.
    pub correct_answers_per_second: Option<f64>,
}

/// Accuracy analyzer output.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AccuracyAnalysis {
    /// Base accuracy summary.
    pub summary: AccuracySummary,
    /// Optional quality-at-load join.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy_at_load: Option<AccuracyAtLoad>,
    /// Optional correct answers per kilowatt-hour.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correct_answers_per_kwh: Option<f64>,
}

/// Analyzer that turns an accuracy summary into report-ready analysis.
#[derive(Debug, Default)]
pub struct AccuracyResultsAnalyzer;

impl Analyzer for AccuracyResultsAnalyzer {
    type Output = AccuracyAnalysis;

    const TYPE: AnalyzerType = AnalyzerType::AccuracyResults;
    const REQUIRED: &'static [AccumulatorType] = &[AccumulatorType::Accuracy];
    const OPTIONAL: &'static [AccumulatorType] = &[
        AccumulatorType::MetricResults,
        AccumulatorType::GpuTelemetry,
    ];

    fn summarize(&self, ctx: &SummaryContext) -> Option<Self::Output> {
        let acc = ctx.get_accumulator::<AccuracySummary>(AccumulatorType::Accuracy)?;
        let metrics = ctx.get_accumulator::<AccumulatorSummary>(AccumulatorType::MetricResults);
        let energy = ctx.get_accumulator::<EnergyEfficiencySummary>(AccumulatorType::GpuTelemetry);
        Some(AccuracyAnalysis {
            summary: acc.clone(),
            accuracy_at_load: metrics.map(|summary| quality_vs_throughput(acc, summary)),
            correct_answers_per_kwh: energy.and_then(|summary| {
                safe_div(acc.correct_count() as f64, summary.total_energy_j / 3.6e6)
            }),
        })
    }
}

fn quality_vs_throughput(acc: &AccuracySummary, metrics: &AccumulatorSummary) -> AccuracyAtLoad {
    let goodput = metrics.finite_value(MetricTag::Goodput);
    let request_throughput = metrics.finite_value(MetricTag::RequestThroughput);
    let throughput = goodput.or(request_throughput);
    AccuracyAtLoad {
        accuracy: acc.overall.accuracy,
        goodput,
        request_throughput,
        correct_answers_per_second: acc
            .overall
            .accuracy
            .zip(throughput)
            .map(|(accuracy, tps)| accuracy * tps),
    }
}

fn safe_div(numerator: f64, denominator: f64) -> Option<f64> {
    (numerator.is_finite() && denominator.is_finite() && denominator > 0.0)
        .then_some(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::{
        AccumulatorType, AccuracyAccumulator, AccuracyRecord, AccuracyRecordError,
        AccuracyResultsAnalyzer, AnalyzerRunError, AnalyzerRunner, AnalyzerType, CorrelationId,
        EnergyEfficiencySummary, GradingResult, LIGHTEVAL_CORRECTNESS_THRESHOLD, SummaryContext,
        TaskId,
    };
    use crate::metrics_core::{AccumulatorSummary, ExportContext, MetricTag, Phase};

    fn record(
        id: &str,
        task: &str,
        phase: Phase,
        start_ns: i64,
        end_ns: i64,
        correct: bool,
        unparsed: bool,
    ) -> AccuracyRecord {
        AccuracyRecord {
            correlation_id: CorrelationId::from(id),
            task: TaskId::from(task),
            phase,
            start_ns,
            end_ns,
            result: GradingResult {
                correct,
                unparsed,
                confidence: Some(if correct { 0.9 } else { 0.25 }),
                extracted: Some("answer".to_string()),
                ground_truth: Some("answer".to_string()),
                reasoning: Some("checked".to_string()),
            },
        }
    }

    #[test]
    fn score_policy_has_single_threshold() {
        assert!(
            !GradingResult::from_score(LIGHTEVAL_CORRECTNESS_THRESHOLD - 0.01, false, "x").correct
        );
        assert!(!GradingResult::from_score(LIGHTEVAL_CORRECTNESS_THRESHOLD, false, "x").correct);
        assert!(
            GradingResult::from_score(LIGHTEVAL_CORRECTNESS_THRESHOLD + f64::EPSILON, false, "x")
                .correct
        );
    }

    #[test]
    fn accumulator_summarizes_overall_per_task_phase_and_time() {
        let mut acc = AccuracyAccumulator::new();
        acc.process_record(record("r1", "math", Phase::Warmup, 0, 10, true, false))
            .unwrap();
        acc.process_record(record("r2", "math", Phase::Profiling, 10, 20, false, true))
            .unwrap();
        acc.process_record(record("r3", "chem", Phase::Profiling, 20, 30, true, false))
            .unwrap();

        assert_eq!(
            acc.record_for(&CorrelationId::from("r2"))
                .unwrap()
                .task
                .as_str(),
            "math"
        );
        assert_eq!(acc.query_time_range(10, 31), vec![false, true, true]);
        assert_eq!(acc.query_time_range(10, 30), vec![false, true, true]);

        let profiling = acc.export_results(ExportContext::phase(Phase::Profiling));
        assert_eq!(profiling.overall.n, 2);
        assert_eq!(profiling.overall.correct_count, 1);
        assert_eq!(profiling.overall.unparsed_count, 1);
        assert_eq!(profiling.overall.accuracy, Some(0.5));
        assert_eq!(
            profiling.per_task[&TaskId::from("math")].unparsed_rate,
            Some(1.0)
        );
        assert!(profiling.overall.ci.unwrap().low >= 0.0);
        assert!(profiling.overall.ci.unwrap().high <= 1.0);
    }

    #[test]
    fn time_window_summaries_filter_by_start_timestamp() {
        let mut acc = AccuracyAccumulator::new();
        acc.process_record(record("r1", "math", Phase::Profiling, 9, 19, true, false))
            .unwrap();
        acc.process_record(record("r2", "math", Phase::Profiling, 10, 19, true, false))
            .unwrap();
        acc.process_record(record("r3", "math", Phase::Profiling, 10, 20, true, false))
            .unwrap();
        let summary = acc.export_results(ExportContext::time_range(10, 20));
        assert_eq!(summary.overall.n, 2);
    }

    #[test]
    fn accumulator_rejects_invalid_or_duplicate_records_and_merges_workers() {
        let mut first = AccuracyAccumulator::new();
        first
            .process_record(record("r1", "math", Phase::Profiling, 0, 10, true, false))
            .unwrap();
        assert!(matches!(
            first.process_record(record("r1", "math", Phase::Profiling, 10, 20, false, false,)),
            Err(AccuracyRecordError::DuplicateCorrelationId(_))
        ));

        let mut invalid = record("r-invalid", "math", Phase::Profiling, 20, 10, true, false);
        assert!(matches!(
            first.process_record(invalid.clone()),
            Err(AccuracyRecordError::InvalidInterval { .. })
        ));
        invalid.start_ns = 0;
        invalid.end_ns = 1;
        invalid.result.confidence = Some(f64::NAN);
        assert!(matches!(
            first.process_record(invalid),
            Err(AccuracyRecordError::InvalidConfidence(value)) if value.is_nan()
        ));

        let mut second = AccuracyAccumulator::new();
        second
            .process_record(record("r2", "chem", Phase::Profiling, 10, 20, false, true))
            .unwrap();
        first.try_merge(second).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(first.records()[1].correlation_id.as_str(), "r2");
    }

    #[test]
    fn analyzer_gracefully_runs_with_only_accuracy() {
        let mut acc = AccuracyAccumulator::new();
        acc.process_record(record("r1", "math", Phase::Profiling, 0, 10, true, false))
            .unwrap();
        let summary = acc.export_results(ExportContext::all());
        let mut ctx = SummaryContext::new();
        ctx.insert_accumulator(AccumulatorType::Accuracy, summary);
        let mut runner = AnalyzerRunner::new();
        runner.push(AccuracyResultsAnalyzer);
        let ran = runner.run(&mut ctx).unwrap();
        assert_eq!(ran, vec![AnalyzerType::AccuracyResults]);
        let analysis = ctx
            .get_analyzer::<super::AccuracyAnalysis>(AnalyzerType::AccuracyResults)
            .unwrap();
        assert_eq!(analysis.summary.overall.accuracy, Some(1.0));
        assert_eq!(analysis.accuracy_at_load, None);
    }

    #[test]
    fn analyzer_skips_when_required_accuracy_is_missing() {
        let mut ctx = SummaryContext::new();
        let mut runner = AnalyzerRunner::new();
        runner.push(AccuracyResultsAnalyzer);
        let ran = runner.run(&mut ctx).unwrap();
        assert!(ran.is_empty());
        assert!(
            ctx.get_analyzer::<super::AccuracyAnalysis>(AnalyzerType::AccuracyResults)
                .is_none()
        );
    }

    #[test]
    fn analyzer_runner_rejects_duplicate_identities() {
        let mut ctx = SummaryContext::new();
        let mut runner = AnalyzerRunner::new();
        runner.push(AccuracyResultsAnalyzer);
        runner.push(AccuracyResultsAnalyzer);
        assert_eq!(
            runner.run(&mut ctx),
            Err(AnalyzerRunError::DuplicateAnalyzer {
                analyzer: AnalyzerType::AccuracyResults,
            })
        );
    }

    #[test]
    fn analyzer_adds_optional_metric_and_energy_joins() {
        let mut acc = AccuracyAccumulator::new();
        acc.process_record(record("r1", "math", Phase::Profiling, 0, 10, true, false))
            .unwrap();
        acc.process_record(record("r2", "math", Phase::Profiling, 10, 20, false, false))
            .unwrap();
        let summary = acc.export_results(ExportContext::all());
        let mut metrics = AccumulatorSummary::new();
        metrics.insert_finite(MetricTag::Goodput, 100.0);
        metrics.insert_finite(MetricTag::RequestThroughput, 120.0);
        let mut ctx = SummaryContext::new();
        ctx.insert_accumulator(AccumulatorType::Accuracy, summary);
        ctx.insert_accumulator(AccumulatorType::MetricResults, metrics);
        ctx.insert_accumulator(
            AccumulatorType::GpuTelemetry,
            EnergyEfficiencySummary {
                total_energy_j: 3.6e6,
            },
        );
        let mut runner = AnalyzerRunner::new();
        runner.push(AccuracyResultsAnalyzer);
        runner.run(&mut ctx).unwrap();
        let analysis = ctx
            .get_analyzer::<super::AccuracyAnalysis>(AnalyzerType::AccuracyResults)
            .unwrap();
        let at_load = analysis.accuracy_at_load.as_ref().unwrap();
        assert_eq!(at_load.goodput, Some(100.0));
        assert_eq!(at_load.correct_answers_per_second, Some(50.0));
        assert_eq!(analysis.correct_answers_per_kwh, Some(1.0));
    }
}
