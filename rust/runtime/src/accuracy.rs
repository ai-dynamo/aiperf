// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Accuracy control-plane integration around the ordinary Rust inference path.
//!
//! A long-lived Python/Lighteval worker owns canonical dataset preparation,
//! prompts, hidden tests, and scoring. Rust receives only opaque problem IDs,
//! prompts, generation settings, and grade results. Requests still flow through
//! the same scheduler, endpoint materializer, transport, response parser,
//! observer, and metrics pipeline as every other online run.
//!
//! Only the process boundary changes, so canonical semantics stay on the
//! Python side.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::accuracy_core::{
    AccuracyEvaluator, EvaluatorGrade, EvaluatorGradeItem, EvaluatorIdentity, EvaluatorLoadConfig,
    EvaluatorLoadResult, EvaluatorProblem, ProblemId,
};
use crate::dataset::{
    AccuracyComposer, ComposeConfig, Composer, ConversationContextMode, Dataset, RawRow, RowOrigin,
    SegmentPool, TextTokenizer,
};
use crate::metrics_core::{
    AccumulatorSummary, AccumulatorType, AccuracyAccumulator, AccuracyAnalysis, AccuracyRecord,
    AccuracyResultsAnalyzer, AnalyzerRunner, AnalyzerType, CorrelationId, EnergyEfficiencySummary,
    EvaluatorDatasetReportInfo, EvaluatorReportInfo, ExportContext, GradingResult, MetricTag,
    NativeReport, Phase, ReportError, ReportRunInfo, RunOutcome, SummaryContext, TaskId,
};
use crate::rng::RngRoot;
use anyhow::Context;
use async_trait::async_trait;
use loadgen_core::collector::{ReplayTerminalStatus, TraceSimulationReport};
use serde::Serialize;
use serde_json::json;

use crate::multiturn::IssuedCredit;
use crate::scheduled::{ScheduledRunReport, TurnDispatchOutcome, TurnRecordProcessor};

const PROBLEM_PAGE_SIZE: usize = 256;
const GRADE_BATCH_SIZE: usize = 128;

#[derive(Debug, Clone)]
pub(crate) struct ProblemAssociation {
    problem_id: ProblemId,
    correlation_id: CorrelationId,
    task: TaskId,
}

/// One captured terminal response, carrying only opaque `Send` data (problem id,
/// task, timing, terminal status, response text). It holds no `Rc`/evaluator
/// handle, so a per-shard capture set crosses the thread-per-core spawn boundary
/// back to the coordinator, where the disjoint shard sets concatenate before the
/// single main-thread evaluator grades them (grading is keyed by `problem_id`, so
/// the merged set is order-independent).
#[derive(Debug, Clone)]
pub(crate) struct CapturedResponse {
    sequence: u64,
    problem_id: ProblemId,
    correlation_id: CorrelationId,
    task: TaskId,
    start_ns: i64,
    end_ns: i64,
    terminal: ReplayTerminalStatus,
    response_text: String,
}

/// Frozen evaluator-authored problems plus opaque response associations.
///
/// This is not a dispatch workload. The dataset is consumed by the same
/// [`ConversationSource`](crate::multiturn::ConversationSource) and normal
/// online runner as any other single-turn dataset.
#[derive(Clone)]
pub struct AccuracyDataset {
    dataset: Arc<Dataset>,
    associations: Arc<[ProblemAssociation]>,
}

impl std::fmt::Debug for AccuracyDataset {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AccuracyDataset")
            .field("dataset", &self.dataset)
            .field("associations", &self.associations.len())
            .finish()
    }
}

impl AccuracyDataset {
    /// Lower evaluator-authored prompts into the unified segment-backed dataset.
    pub fn from_evaluator_problems(
        model: &str,
        problems: Vec<EvaluatorProblem>,
        tokenizer: &dyn TextTokenizer,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !problems.is_empty(),
            "accuracy evaluator returned no problems"
        );
        let mut problem_ids = BTreeSet::new();
        let mut segments = SegmentPool::new();
        let rows = problems
            .into_iter()
            .enumerate()
            .map(|(index, problem)| {
                anyhow::ensure!(
                    problem_ids.insert(problem.problem_id.clone()),
                    "accuracy evaluator returned duplicate problem_id {:?}",
                    problem.problem_id.as_str()
                );
                evaluator_problem_row(index, problem)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let compose_config = ComposeConfig::new(model, RngRoot::new(Some(0)));
        let conversations =
            AccuracyComposer.compose(rows, &compose_config, tokenizer, &mut segments)?;
        let dataset = Arc::new(Dataset::new(
            conversations,
            Arc::new(segments.freeze()),
            "sequential",
            ConversationContextMode::MessageArrayWithResponses,
        )?);
        Self::from_dataset(dataset)
    }

    /// Validate opaque evaluator associations on an already-composed dataset.
    pub fn from_dataset(dataset: Arc<Dataset>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !dataset.conversations().is_empty(),
            "accuracy evaluator returned no conversations"
        );
        let mut associations = Vec::with_capacity(dataset.conversations().len());
        let mut problem_ids = BTreeSet::new();
        let mut correlation_ids = BTreeSet::new();
        for conversation in dataset.conversations() {
            anyhow::ensure!(
                conversation.turns.len() == 1,
                "accuracy conversation {:?} must contain exactly one turn, found {}",
                conversation.session_id.as_str(),
                conversation.turns.len()
            );
            let problem_id = ProblemId::new(conversation.session_id.as_str()).map_err(|error| {
                anyhow::anyhow!(
                    "invalid evaluator problem id {:?}: {error}",
                    conversation.session_id.as_str()
                )
            })?;
            anyhow::ensure!(
                problem_ids.insert(problem_id.clone()),
                "duplicate evaluator problem id {:?}",
                problem_id.as_str()
            );
            let association = conversation.accuracy.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "accuracy conversation {:?} has no evaluator association",
                    conversation.session_id.as_str()
                )
            })?;
            anyhow::ensure!(
                association.correlation_id.as_str() == problem_id.as_str(),
                "accuracy problem {:?} has mismatched correlation id {:?}",
                problem_id.as_str(),
                association.correlation_id.as_str()
            );
            anyhow::ensure!(
                !association.task.trim().is_empty(),
                "accuracy problem {:?} has an empty task",
                problem_id.as_str()
            );
            anyhow::ensure!(
                correlation_ids.insert(association.correlation_id.as_str().to_string()),
                "duplicate accuracy correlation id {:?}",
                association.correlation_id.as_str()
            );
            associations.push(ProblemAssociation {
                problem_id,
                correlation_id: CorrelationId::new(association.correlation_id.as_str()),
                task: TaskId::new(association.task.clone()),
            });
        }
        Ok(Self {
            dataset,
            associations: associations.into(),
        })
    }

    /// Borrow the immutable dataset sent through normal inference dispatch.
    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    /// Number of evaluator-authored problems.
    pub fn len(&self) -> usize {
        self.associations.len()
    }

    /// Share the frozen read-only response associations.
    ///
    /// The associations are `Send + Sync` (an `Arc<[…]>` of opaque ids), so each
    /// thread-per-core shard clones this handle and builds its own capture
    /// [`AccuracyRecordProcessor`] over the SAME associations. Only the captured
    /// `Send` responses merge back to the coordinator; the evaluator never moves.
    #[cfg(feature = "engine")]
    pub(crate) fn associations(&self) -> Arc<[ProblemAssociation]> {
        self.associations.clone()
    }

    /// Whether the evaluator authored no problems.
    pub fn is_empty(&self) -> bool {
        self.associations.is_empty()
    }

    /// Build the normal terminal processor that captures response text only.
    pub fn record_processor(&self) -> AccuracyRecordProcessor {
        AccuracyRecordProcessor::new(self.associations.clone())
    }
}

fn evaluator_problem_row(index: usize, problem: EvaluatorProblem) -> anyhow::Result<RawRow> {
    let problem_id = problem.problem_id.as_str();
    anyhow::ensure!(
        !problem.task.trim().is_empty(),
        "accuracy problem {problem_id:?} has an empty task"
    );
    anyhow::ensure!(
        !problem.prompt.trim().is_empty(),
        "accuracy problem {problem_id:?} has an empty prompt"
    );
    anyhow::ensure!(
        !problem.messages.is_empty(),
        "accuracy problem {problem_id:?} has no messages"
    );
    let generation_size = u32::try_from(problem.generation.max_tokens).map_err(|_| {
        anyhow::anyhow!(
            "accuracy problem {problem_id:?} max_tokens {} exceeds u32",
            problem.generation.max_tokens
        )
    })?;
    anyhow::ensure!(
        generation_size > 0,
        "accuracy problem {problem_id:?} max_tokens must be positive"
    );
    anyhow::ensure!(
        problem.generation.temperature.is_finite() && problem.generation.temperature >= 0.0,
        "accuracy problem {problem_id:?} temperature must be finite and non-negative"
    );
    anyhow::ensure!(
        problem.generation.top_p.is_finite() && (0.0..=1.0).contains(&problem.generation.top_p),
        "accuracy problem {problem_id:?} top_p must be in [0, 1]"
    );
    let messages = serde_json::to_value(problem.messages)
        .context("serializing evaluator-authored messages")?;
    let mut extra_body = serde_json::Map::new();
    extra_body.insert("temperature".into(), json!(problem.generation.temperature));
    extra_body.insert("top_p".into(), json!(problem.generation.top_p));
    // Emit `stop` only when non-empty. OpenAI treats an empty stop array as
    // equivalent to no stop, but Dynamo's frontend rejects `stop: []` with
    // HTTP 400 ("Stop sequences array cannot be empty") before generation, so
    // stop-less benchmarks (math_500, aime, gpqa_diamond, mmlu_pro) would fail
    // every request in the native transport.
    if !problem.generation.stop.is_empty() {
        extra_body.insert("stop".into(), json!(problem.generation.stop));
    }
    Ok(RawRow {
        value: json!({
            "prompt": problem.prompt,
            "task": problem.task,
            "session_id": problem_id,
            "correlation_id": problem_id,
            "raw_messages": messages,
            "metadata": {"generation_size": generation_size},
            "extra_body": extra_body,
        }),
        wire: None,
        session_id: None,
        group_key: None,
        origin: RowOrigin::Inline { index },
    })
}

/// Load all opaque problems from a canonical evaluator with strict pagination.
pub async fn load_evaluator_problems(
    evaluator: &mut dyn AccuracyEvaluator,
    benchmark: &str,
    config: &EvaluatorLoadConfig,
) -> anyhow::Result<(EvaluatorLoadResult, Vec<EvaluatorProblem>)> {
    load_evaluator_problems_with_grader(evaluator, benchmark, config, None).await
}

/// Load opaque evaluator problems with an optional Python grader override.
///
/// Config v2 has always allowed explicit grader selection. The override
/// crosses the stdio
/// control plane, but dataset loading, answer extraction, and scoring remain
/// entirely inside the Python plugin implementation.
pub async fn load_evaluator_problems_with_grader(
    evaluator: &mut dyn AccuracyEvaluator,
    benchmark: &str,
    config: &EvaluatorLoadConfig,
    grader: Option<&str>,
) -> anyhow::Result<(EvaluatorLoadResult, Vec<EvaluatorProblem>)> {
    let loaded = evaluator
        .load_with_grader(benchmark, config, grader)
        .await
        .with_context(|| format!("canonical evaluator failed to load {benchmark:?}"))?;
    validate_evaluator_load_identity(&loaded)?;
    anyhow::ensure!(
        loaded.problem_count > 0,
        "canonical evaluator loaded zero problems for {:?}",
        loaded.benchmark
    );
    let mut problems = Vec::with_capacity(loaded.problem_count);
    let mut ids = BTreeSet::new();
    let mut offset = 0;
    loop {
        let page = evaluator
            .next_problems(offset, PROBLEM_PAGE_SIZE)
            .await
            .with_context(|| format!("canonical evaluator problem page at offset {offset}"))?;
        anyhow::ensure!(
            !page.items.is_empty() || page.done,
            "canonical evaluator returned an empty non-terminal page at offset {offset}"
        );
        let expected_next = offset
            .checked_add(page.items.len())
            .ok_or_else(|| anyhow::anyhow!("accuracy problem offset overflow"))?;
        anyhow::ensure!(
            page.next_offset == expected_next,
            "canonical evaluator advanced offset {} to {}, expected {expected_next}",
            offset,
            page.next_offset
        );
        for problem in page.items {
            anyhow::ensure!(
                ids.insert(problem.problem_id.clone()),
                "canonical evaluator returned duplicate problem_id {:?}",
                problem.problem_id.as_str()
            );
            problems.push(problem);
        }
        anyhow::ensure!(
            problems.len() <= loaded.problem_count,
            "canonical evaluator returned more than declared {} problems",
            loaded.problem_count
        );
        offset = page.next_offset;
        if page.done {
            break;
        }
        anyhow::ensure!(
            problems.len() < loaded.problem_count,
            "canonical evaluator did not terminate after its declared problem count"
        );
    }
    anyhow::ensure!(
        problems.len() == loaded.problem_count,
        "canonical evaluator declared {} problems but returned {}",
        loaded.problem_count,
        problems.len()
    );
    Ok((loaded, problems))
}

fn validate_evaluator_load_identity(loaded: &EvaluatorLoadResult) -> anyhow::Result<()> {
    for (field, value) in [
        ("benchmark", loaded.benchmark.as_str()),
        ("grader", loaded.grader.as_str()),
        ("dataset.provider", loaded.dataset.provider.as_str()),
    ] {
        anyhow::ensure!(
            !value.trim().is_empty(),
            "canonical evaluator load identity field {field} was empty"
        );
    }
    let revision = loaded.dataset.revision.as_deref().ok_or_else(|| {
        anyhow::anyhow!("canonical evaluator did not report an immutable dataset revision")
    })?;
    anyhow::ensure!(
        !revision.trim().is_empty(),
        "canonical evaluator reported an empty dataset revision"
    );
    anyhow::ensure!(
        !loaded.dataset.evaluation_splits.is_empty(),
        "canonical evaluator reported no evaluation splits"
    );
    let mut splits = BTreeSet::new();
    for split in &loaded.dataset.evaluation_splits {
        anyhow::ensure!(
            !split.trim().is_empty(),
            "canonical evaluator reported an empty evaluation split"
        );
        anyhow::ensure!(
            splits.insert(split),
            "canonical evaluator reported duplicate evaluation split {split:?}"
        );
    }
    for (field, value) in [
        ("dataset.benchmark", loaded.dataset.benchmark.as_deref()),
        ("dataset.repository", loaded.dataset.repository.as_deref()),
        ("dataset.subset", loaded.dataset.subset.as_deref()),
    ] {
        if let Some(value) = value {
            anyhow::ensure!(
                !value.trim().is_empty(),
                "canonical evaluator load identity field {field} was empty"
            );
        }
    }
    Ok(())
}

/// Terminal response collector attached to the ordinary record-processing seam.
///
/// It cannot dispatch inference or grade answers. Canonical grading is invoked
/// in batches only after the Rust scheduler and transport have drained.
pub struct AccuracyRecordProcessor {
    associations: BTreeMap<String, ProblemAssociation>,
    captures: RefCell<Vec<CapturedResponse>>,
}

impl AccuracyRecordProcessor {
    /// Build a fresh capture processor over the shared read-only associations.
    ///
    /// The single-thread path builds one; each thread-per-core shard builds its
    /// own from the same `Arc<[ProblemAssociation]>`, and the disjoint per-shard
    /// captures concatenate at the coordinator before grading.
    pub(crate) fn new(associations: Arc<[ProblemAssociation]>) -> Self {
        Self {
            associations: associations
                .iter()
                .cloned()
                .map(|association| (association.correlation_id.as_str().to_string(), association))
                .collect(),
            captures: RefCell::new(Vec::new()),
        }
    }

    /// Move this processor's captured responses out (draining it).
    ///
    /// Each thread-per-core shard drains its own processor at shard end and ships
    /// the `Send` captures to the coordinator, which concatenates the disjoint sets
    /// and hands them to [`grade_accuracy_captures`].
    pub(crate) fn take_captures(&self) -> Vec<CapturedResponse> {
        std::mem::take(&mut *self.captures.borrow_mut())
    }
}

/// Order captured responses deterministically and validate uniqueness.
///
/// Called once on the full (merged) capture set — the single processor's captures
/// on the single-thread path, or the concatenation of the per-shard capture sets
/// on the thread-per-core path. The per-request `correlation_id` (a per-request
/// uuid) is globally unique and is the uniqueness guard against double-processing.
///
/// The issue `sequence` is the per-worker monotonic credit id, so it is unique
/// within a worker but COLLIDES across shards (each shard's issuer restarts at 0);
/// it is therefore used only as the primary sort key, with the globally-unique
/// `correlation_id` as the deterministic tiebreak. On the single-thread path this
/// preserves the exact issue order (sequences are already distinct); on the sharded
/// path the per-shard runs interleave by sequence with a stable correlation
/// tiebreak (aggregate-equivalent — grading is keyed by `problem_id`, so the tally
/// is order-independent).
pub(crate) fn validate_captures(
    mut captures: Vec<CapturedResponse>,
) -> anyhow::Result<Vec<CapturedResponse>> {
    anyhow::ensure!(
        !captures.is_empty(),
        "accuracy record pipeline captured no profiling responses"
    );
    captures.sort_by(|a, b| {
        a.sequence
            .cmp(&b.sequence)
            .then_with(|| a.correlation_id.as_str().cmp(b.correlation_id.as_str()))
    });
    let mut correlations = BTreeSet::new();
    for capture in &captures {
        anyhow::ensure!(
            correlations.insert(capture.correlation_id.as_str().to_string()),
            "accuracy record pipeline captured duplicate request correlation {:?}",
            capture.correlation_id.as_str()
        );
    }
    Ok(captures)
}

#[async_trait(?Send)]
impl TurnRecordProcessor for AccuracyRecordProcessor {
    async fn process(
        &self,
        credit: &IssuedCredit,
        outcome: &TurnDispatchOutcome,
    ) -> anyhow::Result<()> {
        let correlation_id = credit.turn.request_correlation_id.as_str();
        let association = self.associations.get(correlation_id).ok_or_else(|| {
            anyhow::anyhow!(
                "terminal request has no accuracy association for correlation id {correlation_id:?}"
            )
        })?;
        let capture = CapturedResponse {
            sequence: credit.id,
            problem_id: association.problem_id.clone(),
            correlation_id: CorrelationId::new(credit.turn.uuid.to_string()),
            task: association.task.clone(),
            start_ns: outcome.start_ns,
            end_ns: outcome.end_ns,
            terminal: outcome.terminal,
            response_text: outcome.response_text.clone(),
        };
        self.captures.borrow_mut().push(capture);
        Ok(())
    }
}

/// A transport/provider failure retained beside scored records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AccuracyFailure {
    /// Opaque benchmark problem id.
    pub problem_id: String,
    /// Stable request correlation id.
    pub correlation_id: String,
    /// Diagnostic suitable for logs and JSON reports.
    pub message: String,
}

/// Canonical grades and report metadata joined to an existing native metric summary.
#[derive(Debug)]
pub struct AccuracyEvaluation {
    /// Exact evaluator runtime identity from the initialization handshake.
    pub evaluator: EvaluatorIdentity,
    /// Canonical benchmark, dataset, and grader identity from load.
    pub evaluator_load: EvaluatorLoadResult,
    /// Typed overall and per-task analysis.
    pub accuracy: AccuracyAnalysis,
    /// Full per-request grades in native issue order.
    pub records: Vec<AccuracyRecord>,
    /// Transport/provider failures retained in the grading denominator.
    pub failures: Vec<AccuracyFailure>,
    /// Native-v2 evaluator provenance block.
    pub evaluator_report: EvaluatorReportInfo,
}

/// Combined native report for one externally evaluated accuracy run.
#[derive(Debug, Serialize)]
pub struct AccuracyRunReport {
    /// Canonical benchmark name.
    pub benchmark: String,
    /// Target model name.
    pub model: String,
    /// Exact evaluator runtime identity from the initialization handshake.
    pub evaluator: EvaluatorIdentity,
    /// Canonical benchmark, dataset, and grader identity from load.
    pub evaluator_load: EvaluatorLoadResult,
    /// Standard performance report over the same requests.
    pub performance: TraceSimulationReport,
    /// Typed accuracy summary and optional performance join.
    pub accuracy: AccuracyAnalysis,
    /// Unified native-v2 performance, accuracy, and evaluator report.
    pub native_report: NativeReport,
    /// Full per-response grading records in deterministic dataset order.
    pub records: Vec<AccuracyRecord>,
    /// Transport/provider failures. Failed requests remain in the denominator.
    pub failures: Vec<AccuracyFailure>,
}

/// Batch captured responses through the canonical evaluator and build reports.
///
/// Evaluator/protocol failures return an infrastructure error and produce no
/// score. Inference terminal failures remain explicit report errors, while the
/// worker still owns the score for whatever partial or empty text was captured.
pub async fn grade_and_finalize_accuracy_report(
    model: &str,
    scheduled: ScheduledRunReport,
    _dataset: &AccuracyDataset,
    processor: &AccuracyRecordProcessor,
    evaluator: &mut dyn AccuracyEvaluator,
    loaded: &EvaluatorLoadResult,
) -> anyhow::Result<AccuracyRunReport> {
    let evaluation =
        grade_accuracy_responses(processor, evaluator, loaded, &scheduled.native_metrics).await?;
    let native_report = NativeReport::from_outcome(
        &scheduled.native_metrics,
        &RunOutcome {
            run: ReportRunInfo {
                mode: Some("accuracy".to_string()),
                model: Some(model.to_string()),
            },
            accuracy: Some(evaluation.accuracy.clone()),
            accuracy_records: evaluation.records.clone(),
            evaluator: Some(evaluation.evaluator_report.clone()),
            errors: accuracy_report_errors(&evaluation.failures),
            ..RunOutcome::default()
        },
    );
    Ok(AccuracyRunReport {
        benchmark: loaded.benchmark.clone(),
        model: model.to_string(),
        evaluator: evaluation.evaluator,
        evaluator_load: evaluation.evaluator_load,
        performance: scheduled.performance,
        accuracy: evaluation.accuracy,
        native_report,
        records: evaluation.records,
        failures: evaluation.failures,
    })
}

/// Grade every profiling response captured by the ordinary scheduled runtime.
///
/// The processor records one row per issued request rather than one row per
/// dataset problem. Sequential cycling therefore preserves Python's
/// `session_num % len(conversations)` multi-pass behavior while retaining a
/// unique request correlation ID for every native accuracy record.
pub async fn grade_accuracy_responses(
    processor: &AccuracyRecordProcessor,
    evaluator: &mut dyn AccuracyEvaluator,
    loaded: &EvaluatorLoadResult,
    native_summary: &AccumulatorSummary,
) -> anyhow::Result<AccuracyEvaluation> {
    grade_accuracy_captures(processor.take_captures(), evaluator, loaded, native_summary).await
}

/// Grade a pre-collected capture set through the canonical evaluator.
///
/// This is the merge point for the thread-per-core path: the coordinator
/// concatenates each shard's [`AccuracyRecordProcessor::take_captures`] output and
/// calls this once. Grading is keyed by `problem_id`, so the merged (order-
/// independent) set produces the same tally as any single-worker capture. The
/// single evaluator stays on the coordinator thread — only the `Send` capture data
/// crossed the shard boundary.
pub(crate) async fn grade_accuracy_captures(
    captures: Vec<CapturedResponse>,
    evaluator: &mut dyn AccuracyEvaluator,
    loaded: &EvaluatorLoadResult,
    native_summary: &AccumulatorSummary,
) -> anyhow::Result<AccuracyEvaluation> {
    let captures = validate_captures(captures)?;

    let mut grades: Vec<Option<EvaluatorGrade>> = vec![None; captures.len()];
    let submitted = captures
        .iter()
        .enumerate()
        .map(|(index, capture)| {
            (
                index,
                EvaluatorGradeItem {
                    problem_id: capture.problem_id.clone(),
                    response: capture.response_text.clone(),
                },
            )
        })
        .collect::<Vec<_>>();

    for chunk in submitted.chunks(GRADE_BATCH_SIZE) {
        let items = chunk
            .iter()
            .map(|(_, item)| item.clone())
            .collect::<Vec<_>>();
        let result = evaluator
            .grade_batch(&items)
            .await
            .context("canonical evaluator grade_batch failed")?;
        anyhow::ensure!(
            result.items.len() == chunk.len(),
            "canonical evaluator returned {} grades for {} submitted responses",
            result.items.len(),
            chunk.len()
        );
        for ((index, submitted), grade) in chunk.iter().zip(result.items) {
            anyhow::ensure!(
                grade.problem_id == submitted.problem_id,
                "canonical evaluator returned problem_id {:?} for submitted {:?}",
                grade.problem_id.as_str(),
                submitted.problem_id.as_str()
            );
            let capture = &captures[*index];
            anyhow::ensure!(
                grade.task == capture.task.as_str(),
                "canonical evaluator returned task {:?} for problem {:?}, expected {:?}",
                grade.task,
                grade.problem_id.as_str(),
                capture.task.as_str()
            );
            anyhow::ensure!(
                grade.confidence.is_finite() && (0.0..=1.0).contains(&grade.confidence),
                "canonical evaluator returned invalid confidence {} for problem {:?}",
                grade.confidence,
                grade.problem_id.as_str()
            );
            anyhow::ensure!(
                grades[*index].replace(grade).is_none(),
                "canonical evaluator returned a duplicate grade at index {index}"
            );
        }
    }

    let mut records = Vec::with_capacity(captures.len());
    let mut failures = Vec::new();
    for (index, capture) in captures.into_iter().enumerate() {
        if capture.terminal != ReplayTerminalStatus::Completed {
            let message = format!("inference request ended {:?}", capture.terminal);
            failures.push(AccuracyFailure {
                problem_id: capture.problem_id.as_str().to_string(),
                correlation_id: capture.correlation_id.as_str().to_string(),
                message,
            });
        }
        let grade = grades[index].take().ok_or_else(|| {
            anyhow::anyhow!(
                "canonical evaluator omitted grade for problem {:?}",
                capture.problem_id.as_str()
            )
        })?;
        records.push(AccuracyRecord {
            correlation_id: capture.correlation_id,
            task: capture.task,
            phase: Phase::Profiling,
            start_ns: capture.start_ns,
            end_ns: capture.end_ns,
            result: grading_result(grade),
        });
    }

    let mut accumulator = AccuracyAccumulator::new();
    for record in records.iter().cloned() {
        accumulator.process_record(record)?;
    }
    let accuracy_summary = accumulator.export_results(ExportContext::phase(Phase::Profiling));
    let mut summary_context = SummaryContext::new();
    summary_context.insert_accumulator(AccumulatorType::Accuracy, accuracy_summary);
    summary_context.insert_accumulator(AccumulatorType::MetricResults, native_summary.clone());
    // Python owns total-GPU-energy production as an externally injected metric.
    // When the native telemetry sidecar has attached that same joule scalar to the metric
    // summary, expose it through the analyzer's existing optional energy seam;
    // accuracy must not scrape or independently recompute telemetry.
    if let Some(total_energy_j) = native_summary.finite_value(MetricTag::TotalGpuEnergy) {
        summary_context.insert_accumulator(
            AccumulatorType::GpuTelemetry,
            EnergyEfficiencySummary { total_energy_j },
        );
    }
    let mut analyzers = AnalyzerRunner::new();
    analyzers.push(AccuracyResultsAnalyzer);
    analyzers.run(&mut summary_context)?;
    let accuracy = summary_context
        .get_analyzer::<AccuracyAnalysis>(AnalyzerType::AccuracyResults)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("accuracy analyzer produced no output"))?;
    let evaluator_identity = evaluator.identity().clone();
    let evaluator_report = evaluator_report_info(&evaluator_identity, loaded);
    Ok(AccuracyEvaluation {
        evaluator: evaluator_identity,
        evaluator_load: loaded.clone(),
        accuracy,
        records,
        failures,
        evaluator_report,
    })
}

/// Convert per-request inference failures into native-v2 report errors.
pub fn accuracy_report_errors(failures: &[AccuracyFailure]) -> Vec<ReportError> {
    failures
        .iter()
        .map(|failure| ReportError {
            code: None,
            error_type: "InferenceTransport".to_string(),
            message: failure.message.clone(),
            count: 1,
        })
        .collect()
}

fn grading_result(grade: EvaluatorGrade) -> GradingResult {
    GradingResult {
        correct: grade.correct,
        unparsed: grade.unparsed,
        confidence: Some(grade.confidence),
        extracted: grade.extracted_answer,
        ground_truth: None,
        reasoning: (!grade.reasoning.is_empty()).then_some(grade.reasoning),
    }
}

fn evaluator_report_info(
    identity: &EvaluatorIdentity,
    loaded: &EvaluatorLoadResult,
) -> EvaluatorReportInfo {
    EvaluatorReportInfo {
        protocol: identity.protocol,
        worker_version: identity.worker_version.clone(),
        python_version: identity.python_version.clone(),
        python_executable: identity.python_executable.clone(),
        packages: identity.packages.clone(),
        worker_source_sha256: identity.worker_source_sha256.clone(),
        dependency_lock_sha256: identity.dependency_lock_sha256.clone(),
        container_digest: identity.container_digest.clone(),
        capabilities: identity.capabilities.clone(),
        benchmark: loaded.benchmark.clone(),
        grader: loaded.grader.clone(),
        dataset: EvaluatorDatasetReportInfo {
            provider: loaded.dataset.provider.clone(),
            benchmark: loaded.dataset.benchmark.clone(),
            repository: loaded.dataset.repository.clone(),
            subset: loaded.dataset.subset.clone(),
            revision: loaded.dataset.revision.clone(),
            evaluation_splits: loaded.dataset.evaluation_splits.clone(),
            task_version: loaded.dataset.task_version,
        },
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::rc::Rc;

    use crate::accuracy_core::{
        EvaluatorDatasetIdentity, EvaluatorGenerationConfig, EvaluatorGradeBatch, EvaluatorMessage,
        EvaluatorProblemPage, EvaluatorWorkerError,
    };
    use crate::dataset::{TextTokenizer, TiktokenTokenizer};
    use axum::{
        Json, Router,
        http::{StatusCode, header},
        response::{IntoResponse, Response},
        routing::post,
    };
    use serde_json::Value;

    use super::*;
    use crate::multiturn::{ConversationSource, NativeDatasetConversationSource};
    use crate::run::run_single_turn_dataset_online;

    async fn accuracy_chat(Json(body): Json<Value>) -> impl IntoResponse {
        let prompt = body["messages"][0]["content"].as_str().unwrap_or_default();
        let answer = if prompt.contains("first fixture") {
            "The answer is (B)"
        } else {
            "The answer is (A)"
        };
        let body = format!(
            "data: {{\"object\":\"chat.completion.chunk\",\"choices\":[{{\"delta\":{{\"content\":{answer:?}}},\"finish_reason\":null}}]}}\n\n\
             data: {{\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{{\"prompt_tokens\":10,\"completion_tokens\":4}}}}\n\n\
             data: [DONE]\n\n"
        );
        ([(header::CONTENT_TYPE, "text/event-stream")], body)
    }

    async fn spawn_accuracy_mock() -> String {
        let app = Router::new().route("/v1/chat/completions", post(accuracy_chat));
        spawn_mock_app(app).await
    }

    async fn accuracy_chat_with_failure(Json(body): Json<Value>) -> Response {
        let prompt = body["messages"][0]["content"].as_str().unwrap_or_default();
        if prompt.contains("second fixture") {
            return (StatusCode::BAD_GATEWAY, "fixture upstream failure").into_response();
        }
        accuracy_chat(Json(body)).await.into_response()
    }

    async fn spawn_accuracy_failure_mock() -> String {
        let app = Router::new().route("/v1/chat/completions", post(accuracy_chat_with_failure));
        spawn_mock_app(app).await
    }

    async fn spawn_mock_app(app: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{address}")
    }

    fn identity() -> EvaluatorIdentity {
        EvaluatorIdentity {
            protocol: 1,
            worker_version: "fixture-worker".to_string(),
            python_version: "3.fixture".to_string(),
            python_executable: "/fixture/python".to_string(),
            packages: BTreeMap::from([("lighteval".to_string(), Some("fixture".to_string()))]),
            worker_source_sha256: "fixture-source".to_string(),
            dependency_lock_sha256: Some("fixture-lock".to_string()),
            container_digest: Some("sha256:fixture".to_string()),
            capabilities: vec!["grade_batch".to_string()],
        }
    }

    fn loaded() -> EvaluatorLoadResult {
        EvaluatorLoadResult {
            benchmark: "mmlu-pro".to_string(),
            problem_count: 2,
            dataset: EvaluatorDatasetIdentity {
                provider: "lighteval".to_string(),
                benchmark: None,
                repository: Some("TIGER-Lab/MMLU-Pro".to_string()),
                subset: Some("default".to_string()),
                revision: Some("fixture-revision".to_string()),
                evaluation_splits: vec!["test".to_string()],
                task_version: Some(1),
            },
            grader: "lighteval task metrics".to_string(),
        }
    }

    fn problem(id: &str, prompt: &str) -> EvaluatorProblem {
        EvaluatorProblem {
            problem_id: ProblemId::new(id).unwrap(),
            task: "math".to_string(),
            prompt: prompt.to_string(),
            messages: vec![EvaluatorMessage {
                role: "user".to_string(),
                content: Value::String(prompt.to_string()),
            }],
            generation: EvaluatorGenerationConfig {
                max_tokens: 16,
                temperature: 0.0,
                top_p: 1.0,
                stop: vec!["Question:".to_string()],
            },
        }
    }

    struct FixtureEvaluator {
        identity: EvaluatorIdentity,
        loaded: EvaluatorLoadResult,
        problems: Vec<EvaluatorProblem>,
        responses: Vec<EvaluatorGradeItem>,
        fail_grading: bool,
    }

    #[async_trait(?Send)]
    impl AccuracyEvaluator for FixtureEvaluator {
        fn identity(&self) -> &EvaluatorIdentity {
            &self.identity
        }

        async fn load(
            &mut self,
            _benchmark: &str,
            _config: &EvaluatorLoadConfig,
        ) -> Result<EvaluatorLoadResult, EvaluatorWorkerError> {
            Ok(self.loaded.clone())
        }

        async fn next_problems(
            &mut self,
            offset: usize,
            limit: usize,
        ) -> Result<EvaluatorProblemPage, EvaluatorWorkerError> {
            let end = offset.saturating_add(limit).min(self.problems.len());
            Ok(EvaluatorProblemPage {
                items: self.problems[offset..end].to_vec(),
                next_offset: end,
                done: end == self.problems.len(),
            })
        }

        async fn grade_batch(
            &mut self,
            items: &[EvaluatorGradeItem],
        ) -> Result<EvaluatorGradeBatch, EvaluatorWorkerError> {
            if self.fail_grading {
                return Err(EvaluatorWorkerError::Remote {
                    kind: "FixtureFailure".to_string(),
                    message: "grading failed".to_string(),
                    retryable: false,
                });
            }
            self.responses.extend_from_slice(items);
            Ok(EvaluatorGradeBatch {
                items: items
                    .iter()
                    .map(|item| {
                        let expected = "(B)";
                        EvaluatorGrade {
                            problem_id: item.problem_id.clone(),
                            task: "math".to_string(),
                            correct: item.response.contains(expected),
                            unparsed: false,
                            confidence: if item.response.contains(expected) {
                                1.0
                            } else {
                                0.0
                            },
                            reasoning: "fixture canonical grade".to_string(),
                            extracted_answer: Some(item.response.clone()),
                        }
                    })
                    .collect(),
            })
        }

        async fn shutdown(&mut self) -> Result<(), EvaluatorWorkerError> {
            Ok(())
        }
    }

    fn evaluator(fail_grading: bool) -> FixtureEvaluator {
        FixtureEvaluator {
            identity: identity(),
            loaded: loaded(),
            problems: vec![
                problem("opaque-1", "first fixture"),
                problem("opaque-2", "second fixture"),
            ],
            responses: Vec::new(),
            fail_grading,
        }
    }

    async fn dispatch_fixture(
        evaluator: &mut FixtureEvaluator,
    ) -> (
        AccuracyDataset,
        Rc<AccuracyRecordProcessor>,
        ScheduledRunReport,
    ) {
        let base_url = spawn_accuracy_mock().await;
        dispatch_fixture_at(evaluator, base_url).await
    }

    async fn dispatch_fixture_at(
        evaluator: &mut FixtureEvaluator,
        base_url: String,
    ) -> (
        AccuracyDataset,
        Rc<AccuracyRecordProcessor>,
        ScheduledRunReport,
    ) {
        let (load, problems) =
            load_evaluator_problems(evaluator, "mmlu-pro", &EvaluatorLoadConfig::default())
                .await
                .unwrap();
        assert_eq!(load, loaded());
        let tokenizer = TiktokenTokenizer::builtin();
        let dataset =
            AccuracyDataset::from_evaluator_problems("fixture-model", problems, &tokenizer)
                .unwrap();
        assert_eq!(
            dataset.dataset().metadata().conversations[0].turns[0].input_tokens,
            tokenizer.count("first fixture").unwrap() as u64
        );
        let processor = Rc::new(dataset.record_processor());
        let endpoint_id = crate::endpoints::EndpointId::new("chat").unwrap();
        let endpoint = crate::endpoints::EndpointRegistry::builtin()
            .unwrap()
            .prepare(
                &endpoint_id,
                crate::endpoints::RawEndpointConfig {
                    streaming: true,
                    use_server_token_count: true,
                    ..crate::endpoints::RawEndpointConfig::default()
                },
            )
            .unwrap();
        let mut table = crate::endpoints::PreparedEndpointTable::new();
        let key = table.push(endpoint).unwrap();
        let table = Rc::new(table);
        let source: Box<dyn ConversationSource> = Box::new(
            NativeDatasetConversationSource::sequential_with_prepared_endpoint(
                dataset.dataset().as_ref().clone(),
                "fixture-model",
                16,
                table.clone(),
                crate::multiturn::PreparedEndpointReference { key, endpoint_id },
            )
            .unwrap(),
        );
        let processors: Vec<Rc<dyn TurnRecordProcessor>> = vec![processor.clone()];
        let scheduled = run_single_turn_dataset_online(
            base_url,
            "fixture-model".to_string(),
            source,
            2,
            false,
            processors,
            table,
        )
        .await
        .unwrap();
        (dataset, processor, scheduled)
    }

    #[tokio::test]
    async fn rust_dispatches_and_external_evaluator_grades_in_batch() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let mut evaluator = evaluator(false);
                let (dataset, processor, mut scheduled) = dispatch_fixture(&mut evaluator).await;
                scheduled
                    .native_metrics
                    .insert_finite(MetricTag::TotalGpuEnergy, 3_600_000.0);
                let report = grade_and_finalize_accuracy_report(
                    "fixture-model",
                    scheduled,
                    &dataset,
                    processor.as_ref(),
                    &mut evaluator,
                    &loaded(),
                )
                .await
                .unwrap();
                assert_eq!(report.performance.request_counts.completed_requests, 2);
                assert_eq!(report.accuracy.summary.overall.n, 2);
                assert_eq!(report.accuracy.summary.overall.correct_count, 1);
                assert_eq!(report.accuracy.summary.overall.accuracy, Some(0.5));
                assert_eq!(report.accuracy.correct_answers_per_kwh, Some(1.0));
                assert_eq!(evaluator.responses.len(), 2);
                assert!(report.failures.is_empty());
                let native = serde_json::to_value(&report.native_report).unwrap();
                assert_eq!(native["evaluator"]["worker_version"], "fixture-worker");
                assert_eq!(native["accuracy"]["correct_answers_per_kwh"], 1.0);
                assert_eq!(
                    native["evaluator"]["dataset"]["revision"],
                    "fixture-revision"
                );
                assert!(
                    native["accuracy_records"][0]["result"]
                        .get("ground_truth")
                        .is_none()
                );
            })
            .await;
    }

    #[tokio::test]
    async fn evaluator_failure_is_infrastructure_error_not_wrong_answer() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let mut evaluator = evaluator(true);
                let (dataset, processor, scheduled) = dispatch_fixture(&mut evaluator).await;
                let error = grade_and_finalize_accuracy_report(
                    "fixture-model",
                    scheduled,
                    &dataset,
                    processor.as_ref(),
                    &mut evaluator,
                    &loaded(),
                )
                .await
                .unwrap_err();
                assert!(error.to_string().contains("grade_batch failed"));
            })
            .await;
    }

    #[tokio::test]
    async fn evaluator_scores_failed_transport_text_instead_of_rust() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let mut evaluator = evaluator(false);
                let base_url = spawn_accuracy_failure_mock().await;
                let (dataset, processor, scheduled) =
                    dispatch_fixture_at(&mut evaluator, base_url).await;
                let report = grade_and_finalize_accuracy_report(
                    "fixture-model",
                    scheduled,
                    &dataset,
                    processor.as_ref(),
                    &mut evaluator,
                    &loaded(),
                )
                .await
                .unwrap();
                assert_eq!(evaluator.responses.len(), 2);
                assert_eq!(report.failures.len(), 1);
                assert_eq!(report.records.len(), 2);
                assert_eq!(
                    report.records[1].result.reasoning.as_deref(),
                    Some("fixture canonical grade")
                );
                assert!(!report.records[1].result.correct);
            })
            .await;
    }

    #[tokio::test]
    async fn load_requires_immutable_dataset_identity_before_dispatch() {
        let mut evaluator = evaluator(false);
        evaluator.loaded.dataset.revision = None;
        let error =
            load_evaluator_problems(&mut evaluator, "mmlu-pro", &EvaluatorLoadConfig::default())
                .await
                .unwrap_err();
        assert!(error.to_string().contains("immutable dataset revision"));
        assert!(evaluator.responses.is_empty());
    }
}
