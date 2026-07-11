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
//! Ownership is grounded in the complete inherited Python flow at
//! `src/aiperf/dataset/loader/accuracy_dataset_loader.py:21-150`,
//! `src/aiperf/accuracy/benchmark_loader.py:14-45`, and
//! `src/aiperf/accuracy/accuracy_record_processor.py:21-147`; only the process
//! boundary changes, so canonical semantics stay on the Python side.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use aiperf_accuracy::{
    AccuracyEvaluator, EvaluatorGrade, EvaluatorGradeItem, EvaluatorIdentity, EvaluatorLoadConfig,
    EvaluatorLoadResult, EvaluatorProblem, ProblemId,
};
use aiperf_dataset::{
    AccuracyComposer, ComposeConfig, Composer, ConversationContextMode, Dataset, RawRow, RowOrigin,
    SegmentPool, TextTokenizer,
};
use aiperf_metrics::{
    AccumulatorType, AccuracyAccumulator, AccuracyAnalysis, AccuracyRecord,
    AccuracyResultsAnalyzer, AnalyzerRunner, AnalyzerType, CorrelationId,
    EvaluatorDatasetReportInfo, EvaluatorReportInfo, ExportContext, GradingResult, NativeReport,
    Phase, ReportError, ReportRunInfo, RunOutcome, SummaryContext, TaskId,
};
use aiperf_rng::RngRoot;
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
struct ProblemAssociation {
    index: usize,
    problem_id: ProblemId,
    correlation_id: CorrelationId,
    task: TaskId,
}

#[derive(Debug, Clone)]
struct CapturedResponse {
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
        for (index, conversation) in dataset.conversations().iter().enumerate() {
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
                index,
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
    Ok(RawRow {
        value: json!({
            "prompt": problem.prompt,
            "task": problem.task,
            "session_id": problem_id,
            "correlation_id": problem_id,
            "raw_messages": messages,
            "metadata": {"generation_size": generation_size},
            "extra_body": {
                "temperature": problem.generation.temperature,
                "top_p": problem.generation.top_p,
                "stop": problem.generation.stop,
            },
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
    let loaded = evaluator
        .load(benchmark, config)
        .await
        .with_context(|| format!("canonical evaluator failed to load {benchmark:?}"))?;
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

/// Terminal response collector attached to the ordinary record-processing seam.
///
/// It cannot dispatch inference or grade answers. Canonical grading is invoked
/// in batches only after the Rust scheduler and transport have drained.
pub struct AccuracyRecordProcessor {
    associations: BTreeMap<String, ProblemAssociation>,
    captures: RefCell<Vec<Option<CapturedResponse>>>,
}

impl AccuracyRecordProcessor {
    fn new(associations: Arc<[ProblemAssociation]>) -> Self {
        let record_count = associations.len();
        Self {
            associations: associations
                .iter()
                .cloned()
                .map(|association| (association.correlation_id.as_str().to_string(), association))
                .collect(),
            captures: RefCell::new(vec![None; record_count]),
        }
    }

    fn finish(&self) -> anyhow::Result<Vec<CapturedResponse>> {
        let captures = self.captures.borrow();
        let missing = captures
            .iter()
            .enumerate()
            .filter_map(|(index, capture)| capture.is_none().then_some(index))
            .collect::<Vec<_>>();
        anyhow::ensure!(
            missing.is_empty(),
            "accuracy record pipeline omitted dataset indices {missing:?}"
        );
        Ok(captures.iter().flatten().cloned().collect())
    }
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
            start_ns: outcome.start_ns,
            end_ns: outcome.end_ns,
            terminal: outcome.terminal,
            response_text: outcome.response_text.clone(),
        };
        let previous = self.captures.borrow_mut()[association.index].replace(capture);
        anyhow::ensure!(
            previous.is_none(),
            "accuracy association {correlation_id:?} was processed more than once"
        );
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
/// score. Only inference terminal failures are represented as incorrect records.
pub async fn grade_and_finalize_accuracy_report(
    model: &str,
    scheduled: ScheduledRunReport,
    dataset: &AccuracyDataset,
    processor: &AccuracyRecordProcessor,
    evaluator: &mut dyn AccuracyEvaluator,
    loaded: &EvaluatorLoadResult,
) -> anyhow::Result<AccuracyRunReport> {
    let captures = processor.finish()?;
    anyhow::ensure!(
        captures.len() == dataset.associations.len(),
        "captured {} responses for {} evaluator problems",
        captures.len(),
        dataset.associations.len()
    );

    let mut grades: Vec<Option<EvaluatorGrade>> = vec![None; captures.len()];
    let completed = captures
        .iter()
        .zip(dataset.associations.iter())
        .filter(|(capture, _)| capture.terminal == ReplayTerminalStatus::Completed)
        .map(|(capture, association)| {
            (
                association.index,
                EvaluatorGradeItem {
                    problem_id: association.problem_id.clone(),
                    response: capture.response_text.clone(),
                },
            )
        })
        .collect::<Vec<_>>();

    for chunk in completed.chunks(GRADE_BATCH_SIZE) {
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
            let association = &dataset.associations[*index];
            anyhow::ensure!(
                grade.task == association.task.as_str(),
                "canonical evaluator returned task {:?} for problem {:?}, expected {:?}",
                grade.task,
                grade.problem_id.as_str(),
                association.task.as_str()
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
    for (association, capture) in dataset.associations.iter().zip(captures) {
        let result = if capture.terminal == ReplayTerminalStatus::Completed {
            let grade = grades[association.index].take().ok_or_else(|| {
                anyhow::anyhow!(
                    "canonical evaluator omitted grade for problem {:?}",
                    association.problem_id.as_str()
                )
            })?;
            grading_result(grade)
        } else {
            let message = format!("inference request ended {:?}", capture.terminal);
            failures.push(AccuracyFailure {
                problem_id: association.problem_id.as_str().to_string(),
                correlation_id: association.correlation_id.as_str().to_string(),
                message: message.clone(),
            });
            failed_inference_result(&message)
        };
        records.push(AccuracyRecord {
            correlation_id: association.correlation_id.clone(),
            task: association.task.clone(),
            phase: Phase::Profiling,
            start_ns: capture.start_ns,
            end_ns: capture.end_ns,
            result,
        });
    }

    let mut accumulator = AccuracyAccumulator::new();
    for record in records.iter().cloned() {
        accumulator.process_record(record)?;
    }
    let native_summary = scheduled.native_metrics;
    let accuracy_summary = accumulator.export_results(ExportContext::phase(Phase::Profiling));
    let mut summary_context = SummaryContext::new();
    summary_context.insert_accumulator(AccumulatorType::Accuracy, accuracy_summary);
    summary_context.insert_accumulator(AccumulatorType::MetricResults, native_summary.clone());
    let mut analyzers = AnalyzerRunner::new();
    analyzers.push(AccuracyResultsAnalyzer);
    analyzers.run(&mut summary_context)?;
    let accuracy = summary_context
        .get_analyzer::<AccuracyAnalysis>(AnalyzerType::AccuracyResults)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("accuracy analyzer produced no output"))?;
    let evaluator_identity = evaluator.identity().clone();
    let evaluator_report = evaluator_report_info(&evaluator_identity, loaded);
    let native_report = NativeReport::from_outcome(
        &native_summary,
        &RunOutcome {
            run: ReportRunInfo {
                mode: Some("accuracy".to_string()),
                model: Some(model.to_string()),
            },
            accuracy: Some(accuracy.clone()),
            accuracy_records: records.clone(),
            evaluator: Some(evaluator_report),
            errors: failures
                .iter()
                .map(|failure| ReportError {
                    code: None,
                    error_type: "InferenceTransport".to_string(),
                    message: failure.message.clone(),
                    count: 1,
                })
                .collect(),
            ..RunOutcome::default()
        },
    );

    Ok(AccuracyRunReport {
        benchmark: loaded.benchmark.clone(),
        model: model.to_string(),
        evaluator: evaluator_identity,
        evaluator_load: loaded.clone(),
        performance: scheduled.performance,
        accuracy,
        native_report,
        records,
        failures,
    })
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

fn failed_inference_result(reason: &str) -> GradingResult {
    GradingResult {
        correct: false,
        unparsed: true,
        confidence: None,
        extracted: None,
        ground_truth: None,
        reasoning: Some(reason.to_string()),
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

    use aiperf_accuracy::{
        EvaluatorDatasetIdentity, EvaluatorGenerationConfig, EvaluatorGradeBatch, EvaluatorMessage,
        EvaluatorProblemPage, EvaluatorWorkerError,
    };
    use aiperf_dataset::{TextTokenizer, TiktokenTokenizer};
    use axum::{Json, Router, http::header, response::IntoResponse, routing::post};
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
                extra: BTreeMap::new(),
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
        let source: Box<dyn ConversationSource> = Box::new(
            NativeDatasetConversationSource::sequential(
                dataset.dataset().as_ref().clone(),
                "fixture-model",
                16,
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
                let (dataset, processor, scheduled) = dispatch_fixture(&mut evaluator).await;
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
                assert_eq!(evaluator.responses.len(), 2);
                assert!(report.failures.is_empty());
                let native = serde_json::to_value(&report.native_report).unwrap();
                assert_eq!(native["evaluator"]["worker_version"], "fixture-worker");
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
}
