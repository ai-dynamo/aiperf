// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral accuracy run loop plus the online HTTP dispatcher.
//!
//! Benchmark data and grading come from `aiperf-accuracy`; request execution is
//! injected through [`AccuracyDispatcher`]. The built online implementation uses
//! `aiperf-transport`. A future offline engine implements the same dispatcher and
//! feeds the identical accumulator/analyzer/report path.

use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;

use aiperf_accuracy::{BenchmarkProblem, Grader};
use aiperf_clock::{Clock, RealClock};
use aiperf_core::observer::CollectorObserver;
use aiperf_dataset::{
    AccuracyComposer, ComposeConfig, Composer, ConversationContextMode, ConversationSession,
    Dataset, EndpointRequestMaterializer, Overrides, RawRow, RequestMaterializer, RowOrigin,
    SegmentPool, TextTokenizer,
};
use aiperf_endpoints::{
    ChatEndpoint, CreditPhase, Endpoint, EndpointConfig, EndpointType, ModelEndpoint,
};
use aiperf_metrics::{
    AccumulatorType, AccuracyAccumulator, AccuracyAnalysis, AccuracyRecord,
    AccuracyResultsAnalyzer, AnalyzerRunner, AnalyzerType, CorrelationId, ExportContext,
    GradingResult, HttpTrace, MetricsConfig, NativeReport, Phase, ReportError, ReportRunInfo,
    RunOutcome, SummaryContext, TaskId,
};
use aiperf_rng::RngRoot;
use aiperf_timing::SlotPool;
use async_trait::async_trait;
use bytes::Bytes;
use loadgen_core::collector::{ReplayTerminalStatus, TraceSimulationReport};
use loadgen_core::sink::RequestObserver;
use serde::Serialize;
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::http::{HttpDispatchResult, HttpRequest, TransportSink};
use crate::metrics::{
    NativeMetricsObserver, NativeResponseMetadata, ObserverTee, RequestMetricMetadata,
};

/// One endpoint-formatted benchmark request plus its typed grading association.
///
/// The request bytes and exact input-token count come from `aiperf-dataset`'s
/// content-addressed conversation materializer. Dispatchers consume this neutral
/// shape and never rebuild benchmark-specific JSON.
#[derive(Debug, Clone, PartialEq)]
pub struct AccuracyRequest {
    /// Stable benchmark item identifier.
    pub problem_id: String,
    /// Stable request/ground-truth association.
    pub correlation_id: CorrelationId,
    /// Benchmark sub-task used for rollups.
    pub task: TaskId,
    /// Expected answer supplied to the configured grader.
    pub ground_truth: String,
    /// Exact serialized endpoint request body.
    pub body: Bytes,
    /// Per-request HTTP headers.
    pub headers: BTreeMap<String, String>,
    /// Per-request URL query parameters.
    pub parameters: BTreeMap<String, String>,
    /// Endpoint path or absolute target override.
    pub endpoint_path: Option<String>,
    /// Whether the response is expected to be an SSE stream.
    pub streaming: bool,
    /// Tokenizer-exact input length computed at dataset composition time.
    pub input_tokens: usize,
    /// Requested generation cap.
    pub max_output_tokens: usize,
}

/// Frozen accuracy workload shared by online and future offline dispatchers.
///
/// Benchmark problems lower through the same dataset, segment, tokenizer, and
/// endpoint seams used by ordinary AIPerf workloads. The default constructor
/// targets OpenAI Chat Completions; [`from_dataset`](Self::from_dataset) exposes
/// the endpoint/materializer injection point for another dialect.
#[derive(Clone)]
pub struct AccuracyWorkload {
    dataset: Arc<Dataset>,
    requests: Arc<[AccuracyRequest]>,
}

impl std::fmt::Debug for AccuracyWorkload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AccuracyWorkload")
            .field("dataset", &self.dataset)
            .field("requests", &self.requests.len())
            .finish()
    }
}

impl AccuracyWorkload {
    /// Lower benchmark problems into the unified dataset and OpenAI-chat request path.
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
        let model_endpoint = ModelEndpoint {
            primary_model_name: model.to_string(),
            endpoint: EndpointConfig {
                endpoint_type: EndpointType::Chat,
                streaming: true,
                use_legacy_max_tokens: true,
                use_server_token_count: true,
                ..EndpointConfig::default()
            },
        };
        Self::from_dataset(
            dataset,
            &ChatEndpoint,
            &EndpointRequestMaterializer,
            &model_endpoint,
        )
    }

    /// Materialize a frozen accuracy dataset through injected endpoint seams.
    pub fn from_dataset(
        dataset: Arc<Dataset>,
        endpoint: &dyn Endpoint,
        materializer: &dyn RequestMaterializer,
        model_endpoint: &ModelEndpoint,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !dataset.conversations().is_empty(),
            "accuracy benchmark has no conversations"
        );
        let mut requests = Vec::with_capacity(dataset.conversations().len());
        for conversation in dataset.conversations() {
            anyhow::ensure!(
                conversation.turns.len() == 1,
                "accuracy conversation {:?} must contain exactly one turn, found {}",
                conversation.session_id.as_str(),
                conversation.turns.len()
            );
            let mut session =
                ConversationSession::new(dataset.clone(), conversation.session_id.clone())?;
            session.advance_to(0)?;
            let materialized = session.materialize(
                materializer,
                endpoint,
                model_endpoint,
                CreditPhase::Profiling,
                &Overrides::new(),
            )?;
            let association = materialized.accuracy.ok_or_else(|| {
                anyhow::anyhow!(
                    "accuracy conversation {:?} has no ground-truth association",
                    conversation.session_id.as_str()
                )
            })?;
            let input_tokens = usize::try_from(materialized.input_tokens).map_err(|_| {
                anyhow::anyhow!(
                    "accuracy input token count {} exceeds usize",
                    materialized.input_tokens
                )
            })?;
            let max_output_tokens = materialized
                .max_tokens
                .map(|value| value as usize)
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "accuracy conversation {:?} has no positive generation cap",
                        conversation.session_id.as_str()
                    )
                })?;
            requests.push(AccuracyRequest {
                problem_id: conversation.session_id.as_str().to_string(),
                correlation_id: CorrelationId::new(association.correlation_id.as_str()),
                task: TaskId::new(association.task),
                ground_truth: association.ground_truth,
                body: materialized.body,
                headers: materialized.headers,
                parameters: materialized.parameters,
                endpoint_path: materialized.endpoint_path,
                streaming: materialized.streaming,
                input_tokens,
                max_output_tokens,
            });
        }
        Ok(Self {
            dataset,
            requests: requests.into(),
        })
    }

    /// Borrow the immutable unified dataset backing this workload.
    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    /// Borrow endpoint-formatted requests in deterministic dataset order.
    pub fn requests(&self) -> &[AccuracyRequest] {
        &self.requests
    }

    /// Number of scored benchmark requests.
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Whether the workload contains no scored requests.
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
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

/// Transport-neutral generated response consumed by the accuracy run loop.
#[derive(Debug, Clone)]
pub struct AccuracyDispatchResult {
    /// Clock timestamp when dispatch began.
    pub start_ns: i64,
    /// Clock timestamp when dispatch reached terminal.
    pub end_ns: i64,
    /// Terminal classification.
    pub terminal: ReplayTerminalStatus,
    /// Generated response text.
    pub response_text: String,
    /// HTTP/provider status when available.
    pub status: Option<u16>,
    /// Authoritative server-reported prompt tokens.
    pub prompt_tokens: Option<u64>,
    /// Authoritative server-reported completion tokens.
    pub completion_tokens: Option<u64>,
    /// Fine-grained transport metrics.
    pub http: HttpTrace,
}

/// Accuracy request-dispatch extension seam.
#[async_trait(?Send)]
pub trait AccuracyDispatcher {
    /// Dispatch one endpoint-formatted request and retain generated text for grading.
    async fn dispatch(
        &self,
        request_id: Uuid,
        request: &AccuracyRequest,
        observer: &dyn RequestObserver,
    ) -> anyhow::Result<AccuracyDispatchResult>;
}

/// Online OpenAI-chat dispatcher over the Clock-injected AIPerf transport.
pub struct HttpAccuracyDispatcher {
    sink: TransportSink,
}

impl HttpAccuracyDispatcher {
    /// Builds an online dispatcher on the caller's time line.
    pub fn new(
        clock: Rc<dyn Clock>,
        start_ns: i64,
        base_url: &str,
        model: impl Into<String>,
        http2: bool,
    ) -> Self {
        Self {
            sink: TransportSink::new(clock, start_ns, base_url, model, http2),
        }
    }
}

#[async_trait(?Send)]
impl AccuracyDispatcher for HttpAccuracyDispatcher {
    async fn dispatch(
        &self,
        request_id: Uuid,
        request: &AccuracyRequest,
        observer: &dyn RequestObserver,
    ) -> anyhow::Result<AccuracyDispatchResult> {
        let http_request = HttpRequest {
            uuid: request_id,
            input_length: request.input_tokens,
            max_output_tokens: request.max_output_tokens,
            prompt_text: None,
            request_body: None,
            request_body_bytes: Some(request.body.clone()),
            headers: request.headers.clone(),
            parameters: request.parameters.clone(),
            endpoint_path: request.endpoint_path.clone(),
            streaming: request.streaming,
            x_correlation_id: Some(request.correlation_id.as_str().to_string()),
            is_final_turn: true,
            cancel_after_ns: None,
            url_index: None,
        };
        let result = self
            .sink
            .dispatch_collect_with_hooks(http_request, observer, |_| {})
            .await?;
        Ok(map_http_result(result))
    }
}

fn map_http_result(result: HttpDispatchResult) -> AccuracyDispatchResult {
    AccuracyDispatchResult {
        start_ns: result.start_ns,
        end_ns: result.end_ns,
        terminal: result.terminal,
        response_text: result.response_text,
        status: result.status,
        prompt_tokens: result.prompt_tokens.map(u64::from),
        completion_tokens: result.completion_tokens.map(u64::from),
        http: result.http,
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

struct ProblemOutcome {
    request_id: Uuid,
    record: AccuracyRecord,
    failure: Option<AccuracyFailure>,
    response: NativeResponseMetadata,
}

/// Run an online accuracy benchmark over the real/mock HTTP transport.
///
/// Must be driven inside a current-thread runtime and `LocalSet`.
pub async fn run_accuracy_online(
    base_url: &str,
    model: &str,
    benchmark: &str,
    workload: AccuracyWorkload,
    grader: Rc<dyn Grader>,
    concurrency: usize,
    http2: bool,
) -> anyhow::Result<AccuracyRunReport> {
    let clock: Rc<dyn Clock> = RealClock::new();
    let start_ns = clock.now_ns();
    let dispatcher: Rc<dyn AccuracyDispatcher> = Rc::new(HttpAccuracyDispatcher::new(
        clock.clone(),
        start_ns,
        base_url,
        model,
        http2,
    ));
    run_accuracy_with(
        clock,
        start_ns,
        model,
        benchmark,
        workload,
        grader,
        dispatcher,
        concurrency,
    )
    .await
}

/// Run against an injected clock and dispatcher.
///
/// This is the shared online/offline policy path. A dispatcher must stamp results
/// from the supplied clock's timeline.
#[allow(clippy::too_many_arguments)]
pub async fn run_accuracy_with(
    clock: Rc<dyn Clock>,
    start_ns: i64,
    model: &str,
    benchmark: &str,
    workload: AccuracyWorkload,
    grader: Rc<dyn Grader>,
    dispatcher: Rc<dyn AccuracyDispatcher>,
    concurrency: usize,
) -> anyhow::Result<AccuracyRunReport> {
    anyhow::ensure!(
        concurrency > 0,
        "accuracy concurrency must be greater than zero"
    );
    anyhow::ensure!(!workload.is_empty(), "accuracy benchmark has no requests");

    let collector = Rc::new(CollectorObserver::new(false));
    let native_metrics = Rc::new(NativeMetricsObserver::new(
        clock.clone(),
        start_ns,
        MetricsConfig::default(),
    ));
    let delegates: Vec<Rc<dyn RequestObserver>> = vec![collector.clone(), native_metrics.clone()];
    let observer: Rc<dyn RequestObserver> = Rc::new(ObserverTee::new(delegates));
    let slots = SlotPool::new(concurrency);
    let mut handles = Vec::with_capacity(workload.len());
    for request in workload.requests().iter().cloned() {
        let guard = slots.acquire().await;
        let dispatcher = dispatcher.clone();
        let grader = grader.clone();
        let observer = observer.clone();
        let clock = clock.clone();
        let request_id = Uuid::new_v4();
        native_metrics.register_metadata(
            request_id,
            RequestMetricMetadata {
                conversation_id: Some(request.problem_id.clone()),
                correlation_id: Some(request.correlation_id.as_str().to_string()),
                ..RequestMetricMetadata::default()
            },
        );
        let arrival_ns = clock.now_ns();
        observer.on_arrival(
            request_id,
            (arrival_ns - start_ns) as f64 / 1_000_000.0,
            request.input_tokens,
            request.max_output_tokens,
        );
        handles.push(tokio::task::spawn_local(async move {
            let _guard = guard;
            dispatch_grade_problem(
                clock.as_ref(),
                request_id,
                request,
                grader.as_ref(),
                dispatcher.as_ref(),
                observer.as_ref(),
            )
            .await
        }));
    }

    let mut accumulator = AccuracyAccumulator::new();
    let mut failures = Vec::new();
    for handle in handles {
        let outcome = handle.await??;
        native_metrics.record_response(outcome.request_id, outcome.response);
        if let Some(failure) = outcome.failure {
            failures.push(failure);
        }
        accumulator.process_record(outcome.record)?;
    }

    let wall_ms = (clock.now_ns() - start_ns) as f64 / 1_000_000.0;
    let performance = collector.finish(wall_ms);
    let native_summary = native_metrics.finish();
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
    let records = accumulator.records();
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
        performance,
        accuracy,
        native_report,
        records,
        failures,
    })
}

async fn dispatch_grade_problem(
    clock: &dyn Clock,
    request_id: Uuid,
    request: AccuracyRequest,
    grader: &dyn Grader,
    dispatcher: &dyn AccuracyDispatcher,
    observer: &dyn RequestObserver,
) -> anyhow::Result<ProblemOutcome> {
    let start_ns = clock.now_ns();
    let (record_start_ns, end_ns, result, failure, response) =
        match dispatcher.dispatch(request_id, &request, observer).await {
            Ok(dispatch) if dispatch.terminal == ReplayTerminalStatus::Completed => {
                let response = NativeResponseMetadata {
                    start_ns: Some(dispatch.start_ns),
                    end_ns: Some(dispatch.end_ns),
                    prompt_tokens: dispatch.prompt_tokens,
                    completion_tokens: dispatch.completion_tokens,
                    http: dispatch.http,
                };
                let (result, failure) = match grader
                    .grade(&dispatch.response_text, &request.ground_truth)
                    .await
                {
                    Ok(result) => (result, None),
                    Err(error) => {
                        let message = format!("grader failed: {error}");
                        (
                            failed_grading_result(&request.ground_truth, &message),
                            Some(AccuracyFailure {
                                problem_id: request.problem_id.clone(),
                                correlation_id: request.correlation_id.as_str().to_string(),
                                message,
                            }),
                        )
                    }
                };
                (
                    dispatch.start_ns,
                    dispatch.end_ns,
                    result,
                    failure,
                    response,
                )
            }
            Ok(dispatch) => {
                let message = format!(
                    "request ended {:?} with provider status {:?}",
                    dispatch.terminal, dispatch.status
                );
                let response = NativeResponseMetadata {
                    start_ns: Some(dispatch.start_ns),
                    end_ns: Some(dispatch.end_ns),
                    prompt_tokens: dispatch.prompt_tokens,
                    completion_tokens: dispatch.completion_tokens,
                    http: dispatch.http,
                };
                (
                    dispatch.start_ns,
                    dispatch.end_ns,
                    failed_grading_result(&request.ground_truth, &message),
                    Some(AccuracyFailure {
                        problem_id: request.problem_id.clone(),
                        correlation_id: request.correlation_id.as_str().to_string(),
                        message,
                    }),
                    response,
                )
            }
            Err(error) => {
                observer.on_terminal(request_id, ReplayTerminalStatus::Failed);
                let message = format!("dispatch failed: {error:#}");
                let end_ns = clock.now_ns();
                (
                    start_ns,
                    end_ns,
                    failed_grading_result(&request.ground_truth, &message),
                    Some(AccuracyFailure {
                        problem_id: request.problem_id.clone(),
                        correlation_id: request.correlation_id.as_str().to_string(),
                        message,
                    }),
                    NativeResponseMetadata {
                        start_ns: Some(start_ns),
                        end_ns: Some(end_ns),
                        ..NativeResponseMetadata::default()
                    },
                )
            }
        };
    Ok(ProblemOutcome {
        request_id,
        record: AccuracyRecord {
            correlation_id: request.correlation_id,
            task: request.task,
            phase: Phase::Profiling,
            start_ns: record_start_ns,
            end_ns,
            result,
        },
        failure,
        response,
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
                let workload = AccuracyWorkload::from_problems(
                    "fixture-model",
                    vec![
                        problem("q1", "first fixture", "B"),
                        problem("q2", "second fixture", "B"),
                    ],
                    &tokenizer,
                )
                .unwrap();
                assert_eq!(
                    workload.requests()[0].input_tokens,
                    tokenizer.count("first fixture").unwrap()
                );
                let wire: Value = serde_json::from_slice(&workload.requests()[0].body).unwrap();
                assert_eq!(wire["model"], "fixture-model");
                assert_eq!(wire["stream"], true);
                assert_eq!(wire["stream_options"]["include_usage"], true);
                assert_eq!(wire["max_tokens"], 16);
                assert_eq!(wire["temperature"], 0.0);
                assert_eq!(wire["top_p"], 1.0);
                assert_eq!(wire["stop"], json!(["Question:"]));
                let report = run_accuracy_online(
                    &base_url,
                    "fixture-model",
                    "mmlu-pro",
                    workload,
                    Rc::new(MmluProGrader::new()),
                    2,
                    false,
                )
                .await
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
