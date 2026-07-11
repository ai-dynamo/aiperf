// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Accuracy dataset preparation and post-dispatch record processing.
//!
//! This follows Python AIPerf's ownership split end to end:
//! `dataset/loader/accuracy_dataset_loader.py:1-150` lowers benchmark problems
//! into ordinary conversations, the normal AIPerf pipeline dispatches them, and
//! `accuracy/accuracy_record_processor.py:1-147` grades parsed terminal records.
//! Accuracy owns no scheduler, transport, HTTP request, or run loop.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;

use aiperf_accuracy::{BenchmarkProblem, Grader};
use aiperf_dataset::{
    AccuracyComposer, ComposeConfig, Composer, ConversationContextMode, Dataset, RawRow, RowOrigin,
    SegmentPool, TextTokenizer,
};
use aiperf_metrics::{
    AccumulatorType, AccuracyAccumulator, AccuracyAnalysis, AccuracyRecord,
    AccuracyResultsAnalyzer, AnalyzerRunner, AnalyzerType, CorrelationId, ExportContext,
    GradingResult, NativeReport, Phase, ReportError, ReportRunInfo, RunOutcome, SummaryContext,
    TaskId,
};
use aiperf_rng::RngRoot;
use async_trait::async_trait;
use loadgen_core::collector::{ReplayTerminalStatus, TraceSimulationReport};
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::multiturn::IssuedCredit;
use crate::scheduled::{ScheduledRunReport, TurnDispatchOutcome, TurnRecordProcessor};

#[derive(Debug, Clone)]
struct AccuracyAssociation {
    index: usize,
    problem_id: String,
    correlation_id: CorrelationId,
    task: TaskId,
    ground_truth: String,
}

/// Frozen benchmark dataset plus grading associations.
///
/// It is deliberately not a dispatch workload. Callers pass [`dataset`](Self::dataset)
/// to the same `ConversationSource` and normal online runner used by non-accuracy
/// datasets, then attach an [`AccuracyRecordProcessor`] to the terminal record
/// pipeline.
#[derive(Clone)]
pub struct AccuracyDataset {
    dataset: Arc<Dataset>,
    associations: Arc<[AccuracyAssociation]>,
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
    /// Lower benchmark problems into the unified content-addressed dataset path.
    pub fn from_problems(
        model: &str,
        problems: Vec<BenchmarkProblem>,
        tokenizer: &dyn TextTokenizer,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(!problems.is_empty(), "accuracy benchmark has no problems");
        let mut segments = SegmentPool::new();
        let rows = problems
            .into_iter()
            .enumerate()
            .map(|(index, problem)| problem_row(index, problem))
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

    /// Validate grading associations on an already-composed dataset.
    pub fn from_dataset(dataset: Arc<Dataset>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !dataset.conversations().is_empty(),
            "accuracy benchmark has no conversations"
        );
        let mut associations = Vec::with_capacity(dataset.conversations().len());
        let mut correlation_ids = std::collections::BTreeSet::new();
        for (index, conversation) in dataset.conversations().iter().enumerate() {
            anyhow::ensure!(
                conversation.turns.len() == 1,
                "accuracy conversation {:?} must contain exactly one turn, found {}",
                conversation.session_id.as_str(),
                conversation.turns.len()
            );
            let association = conversation.accuracy.clone().ok_or_else(|| {
                anyhow::anyhow!(
                    "accuracy conversation {:?} has no ground-truth association",
                    conversation.session_id.as_str()
                )
            })?;
            anyhow::ensure!(
                correlation_ids.insert(association.correlation_id.as_str().to_string()),
                "duplicate accuracy correlation id {:?}",
                association.correlation_id.as_str()
            );
            associations.push(AccuracyAssociation {
                index,
                problem_id: conversation.session_id.as_str().to_string(),
                correlation_id: CorrelationId::new(association.correlation_id.as_str()),
                task: TaskId::new(association.task),
                ground_truth: association.ground_truth,
            });
        }
        Ok(Self {
            dataset,
            associations: associations.into(),
        })
    }

    /// Borrow the immutable unified dataset backing this workload.
    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    /// Number of scored benchmark conversations.
    pub fn len(&self) -> usize {
        self.associations.len()
    }

    /// Whether the workload contains no scored requests.
    pub fn is_empty(&self) -> bool {
        self.associations.is_empty()
    }

    /// Build the post-dispatch grader registered with the normal record pipeline.
    pub fn record_processor(&self, grader: Rc<dyn Grader>) -> AccuracyRecordProcessor {
        AccuracyRecordProcessor::new(self.associations.clone(), grader)
    }
}

fn problem_row(index: usize, problem: BenchmarkProblem) -> anyhow::Result<RawRow> {
    anyhow::ensure!(
        !problem.id.is_empty(),
        "accuracy problem at index {index} has an empty id"
    );
    anyhow::ensure!(
        !problem.messages.is_empty(),
        "accuracy problem {:?} has no messages",
        problem.id
    );
    let generation_size = u32::try_from(problem.generation.max_tokens).map_err(|_| {
        anyhow::anyhow!(
            "accuracy problem {:?} max_tokens {} exceeds u32",
            problem.id,
            problem.generation.max_tokens
        )
    })?;
    anyhow::ensure!(
        generation_size > 0,
        "accuracy problem {:?} max_tokens must be positive",
        problem.id
    );
    anyhow::ensure!(
        problem.generation.temperature.is_finite() && problem.generation.temperature >= 0.0,
        "accuracy problem {:?} temperature must be finite and non-negative",
        problem.id
    );
    anyhow::ensure!(
        problem.generation.top_p.is_finite() && (0.0..=1.0).contains(&problem.generation.top_p),
        "accuracy problem {:?} top_p must be in [0, 1]",
        problem.id
    );
    let prompt = problem
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .or_else(|| problem.messages.last())
        .expect("non-empty messages checked above")
        .content
        .clone();
    let mut metadata: Map<String, Value> = problem.metadata.into_iter().collect();
    metadata.insert("generation_size".to_string(), json!(generation_size));
    Ok(RawRow {
        value: json!({
            "prompt": prompt,
            "ground_truth": problem.ground_truth,
            "task": problem.task.as_str(),
            "session_id": problem.id,
            "correlation_id": problem.correlation_id.as_str(),
            "raw_messages": problem.messages,
            "metadata": metadata,
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

/// Accuracy grader attached to the ordinary terminal record-processing seam.
///
/// The processor receives completed `TurnDispatchOutcome`s from
/// [`ScheduledRuntime`](crate::scheduled::ScheduledRuntime); it cannot send a
/// request and has no transport dependency.
pub struct AccuracyRecordProcessor {
    associations: BTreeMap<String, AccuracyAssociation>,
    grader: Rc<dyn Grader>,
    records: RefCell<Vec<Option<AccuracyRecord>>>,
    failures: RefCell<Vec<AccuracyFailure>>,
}

impl AccuracyRecordProcessor {
    fn new(associations: Arc<[AccuracyAssociation]>, grader: Rc<dyn Grader>) -> Self {
        let record_count = associations.len();
        Self {
            associations: associations
                .iter()
                .cloned()
                .map(|association| (association.correlation_id.as_str().to_string(), association))
                .collect(),
            grader,
            records: RefCell::new(vec![None; record_count]),
            failures: RefCell::new(Vec::new()),
        }
    }

    fn finish(&self) -> anyhow::Result<(Vec<AccuracyRecord>, Vec<AccuracyFailure>)> {
        let records = self.records.borrow();
        let missing = records
            .iter()
            .enumerate()
            .filter_map(|(index, record)| record.is_none().then_some(index))
            .collect::<Vec<_>>();
        anyhow::ensure!(
            missing.is_empty(),
            "accuracy record pipeline omitted dataset indices {missing:?}"
        );
        Ok((
            records.iter().flatten().cloned().collect(),
            self.failures.borrow().clone(),
        ))
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
        let (result, failure) = if outcome.terminal == ReplayTerminalStatus::Completed {
            match self
                .grader
                .grade(&outcome.response_text, &association.ground_truth)
                .await
            {
                Ok(result) => (result, None),
                Err(error) => {
                    let message = format!("grader failed: {error}");
                    (
                        failed_grading_result(&association.ground_truth, &message),
                        Some(AccuracyFailure {
                            problem_id: association.problem_id.clone(),
                            correlation_id: correlation_id.to_string(),
                            message,
                        }),
                    )
                }
            }
        } else {
            let message = format!("request ended {:?}", outcome.terminal);
            (
                failed_grading_result(&association.ground_truth, &message),
                Some(AccuracyFailure {
                    problem_id: association.problem_id.clone(),
                    correlation_id: correlation_id.to_string(),
                    message,
                }),
            )
        };
        let record = AccuracyRecord {
            correlation_id: association.correlation_id.clone(),
            task: association.task.clone(),
            phase: Phase::Profiling,
            start_ns: outcome.start_ns,
            end_ns: outcome.end_ns,
            result,
        };
        let mut records = self.records.borrow_mut();
        anyhow::ensure!(
            records[association.index].replace(record).is_none(),
            "accuracy association {correlation_id:?} was processed more than once"
        );
        if let Some(failure) = failure {
            self.failures.borrow_mut().push(failure);
        }
        Ok(())
    }
}

/// A failed/abnormal dispatch retained beside scored records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AccuracyFailure {
    /// Stable benchmark item id.
    pub problem_id: String,
    /// Stable request correlation id.
    pub correlation_id: String,
    /// Diagnostic suitable for logs and JSON reports.
    pub message: String,
}

/// Combined native report for one accuracy benchmark run.
#[derive(Debug, Serialize)]
pub struct AccuracyRunReport {
    /// Stable benchmark name.
    pub benchmark: String,
    /// Target model name.
    pub model: String,
    /// Standard performance report over the same requests.
    pub performance: TraceSimulationReport,
    /// Typed accuracy summary and optional performance join.
    pub accuracy: AccuracyAnalysis,
    /// Unified native-v2 performance and accuracy report.
    pub native_report: NativeReport,
    /// Full per-response grading records in deterministic dataset order.
    pub records: Vec<AccuracyRecord>,
    /// Transport/provider failures. Failed problems remain in the accuracy denominator.
    pub failures: Vec<AccuracyFailure>,
}

/// Combine the ordinary run's performance records with post-dispatch grades.
pub fn finalize_accuracy_report(
    benchmark: &str,
    model: &str,
    scheduled: ScheduledRunReport,
    processor: &AccuracyRecordProcessor,
) -> anyhow::Result<AccuracyRunReport> {
    let (records, failures) = processor.finish()?;
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
    let native_report = NativeReport::from_outcome(
        &native_summary,
        &RunOutcome {
            run: ReportRunInfo {
                mode: Some("accuracy".to_string()),
                model: Some(model.to_string()),
            },
            accuracy: Some(accuracy.clone()),
            accuracy_records: records.clone(),
            errors: failures
                .iter()
                .map(|failure| ReportError {
                    code: None,
                    error_type: "AccuracyDispatch".to_string(),
                    message: failure.message.clone(),
                    count: 1,
                })
                .collect(),
            ..RunOutcome::default()
        },
    );

    Ok(AccuracyRunReport {
        benchmark: benchmark.to_string(),
        model: model.to_string(),
        performance: scheduled.performance,
        accuracy,
        native_report,
        records,
        failures,
    })
}

fn failed_grading_result(ground_truth: &str, reason: &str) -> GradingResult {
    GradingResult {
        correct: false,
        unparsed: true,
        confidence: Some(0.0),
        extracted: None,
        ground_truth: ground_truth.to_string(),
        reasoning: Some(reason.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use aiperf_accuracy::{BenchmarkProblem, ChatMessage, GenerationConfig, MmluProGrader};
    use aiperf_dataset::{TextTokenizer, TiktokenTokenizer};
    use aiperf_metrics::{CorrelationId, TaskId};
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
            "data: {{\"choices\":[{{\"delta\":{{\"content\":{answer:?}}},\"finish_reason\":null}}]}}\n\n\
             data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":10,\"completion_tokens\":4}}}}\n\n\
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

    fn problem(id: &str, prompt: &str, gold: &str) -> BenchmarkProblem {
        BenchmarkProblem {
            id: id.to_string(),
            correlation_id: CorrelationId::new(id),
            task: TaskId::new("mmlu_pro.math"),
            messages: vec![ChatMessage::user(prompt)],
            ground_truth: gold.to_string(),
            generation: GenerationConfig {
                max_tokens: 16,
                temperature: 0.0,
                top_p: 1.0,
                stop: vec!["Question:".to_string()],
            },
            metadata: BTreeMap::new(),
        }
    }

    #[tokio::test]
    async fn mmlu_pro_runs_http_to_grader_to_typed_report() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base_url = spawn_accuracy_mock().await;
                let tokenizer = TiktokenTokenizer::builtin();
                let dataset = AccuracyDataset::from_problems(
                    "fixture-model",
                    vec![
                        problem("q1", "first fixture", "B"),
                        problem("q2", "second fixture", "B"),
                    ],
                    &tokenizer,
                )
                .unwrap();
                assert_eq!(
                    dataset.dataset().metadata().conversations[0].turns[0].input_tokens,
                    tokenizer.count("first fixture").unwrap() as u64
                );
                let processor = Rc::new(dataset.record_processor(Rc::new(MmluProGrader::new())));
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
                let report = finalize_accuracy_report(
                    "mmlu-pro",
                    "fixture-model",
                    scheduled,
                    processor.as_ref(),
                )
                .unwrap();
                assert_eq!(report.performance.request_counts.completed_requests, 2);
                assert_eq!(report.accuracy.summary.overall.n, 2);
                assert_eq!(report.accuracy.summary.overall.correct_count, 1);
                assert_eq!(report.accuracy.summary.overall.accuracy, Some(0.5));
                assert_eq!(report.records.len(), 2);
                assert!(report.failures.is_empty());
                assert!(
                    report
                        .accuracy
                        .accuracy_at_load
                        .as_ref()
                        .and_then(|joined| joined.request_throughput)
                        .is_some(),
                    "accuracy/load join was {:?}; native metric keys were {:?}",
                    report.accuracy.accuracy_at_load,
                    report.native_report.metrics.keys().collect::<Vec<_>>()
                );
            })
            .await;
    }
}
