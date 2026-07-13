// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2 proof that placement factories sit below the single dispatcher.
//!
//! Both tests enter through [`RunnerV2Coordinator`]. Neither changes workload,
//! phase, metrics, artifact, or report logic: one replaces turn placement with
//! a remote-shaped backend, and one replaces whole-trace graph placement.

use std::cell::Cell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use aiperf::http::{HttpTurnDispatchResult, HttpTurnExecutionBackend, PreparedHttpTurn};
use aiperf::multiturn::TurnToSend;
use aiperf::scheduled::{ModelResponseMetadata, TurnDispatchOutcome};
use aiperf_clock::Clock;
use aiperf_extensions::BuiltinAiperfRegistryFactory;
use aiperf_graph::errors::TraceError;
use aiperf_graph::execution::GraphTraceExecutionBackend;
use aiperf_graph::placement::{GraphPlacementError, GraphTraceExecutionBackendFactory};
use aiperf_metrics::{HttpTrace, InferenceDimensions};
use aiperf_runner::coordinator::{RunnerResponseV2, RunnerV2Coordinator};
use aiperf_runner::dataset_input::BuiltinRunnerDatasetInputAdapterResolver;
use aiperf_runner::graph_input::BuiltinRunnerGraphInputAdapterResolver;
use aiperf_runner::readiness::{
    NativeHttpReadinessPlanFactory, NativeHttpReadinessTransportFactory, ReadinessAttemptRequest,
    ReadinessAttemptResponse, ReadinessTransport, ReadinessTransportFactory,
};
use aiperf_runner::registry::BuiltinRunnerRegistryFactory;
use aiperf_runner::sidecar_input::BuiltinRunnerSidecarInputAdapterResolver;
use aiperf_runner::{
    HttpExecutionBackendConfig, HttpExecutionBackendFactory, NativeHttpExecutionBackendFactory,
    NativeRunnerGraphPlacementFactory, RunnerExecutionFactories, RunnerGraphPlacementFactory,
};
use aiperf_transport_http::models::RequestRecord;
use anyhow::{Result, anyhow, ensure};
use async_trait::async_trait;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{ObservedUsage, RequestObserver};
use serde_json::{Value, json};

const DISTRIBUTION_ID: &str =
    "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

struct FakeRemoteFactory {
    calls: Arc<AtomicUsize>,
    dimension_calls: Arc<AtomicUsize>,
    shutdowns: Arc<AtomicUsize>,
    artifact_target: PathBuf,
}

impl HttpExecutionBackendFactory for FakeRemoteFactory {
    fn build(
        &self,
        config: HttpExecutionBackendConfig,
    ) -> Result<Rc<dyn HttpTurnExecutionBackend>> {
        ensure!(config.workers == 3, "resolved worker count reached factory");
        ensure!(
            !self.artifact_target.exists(),
            "execution backend must be prepared before artifact creation"
        );
        Ok(Rc::new(FakeRemoteBackend {
            calls: self.calls.clone(),
            dimension_calls: self.dimension_calls.clone(),
            shutdowns: self.shutdowns.clone(),
            run_origin_ns: Cell::new(None),
            model: config.model,
        }))
    }
}

struct FakeRemoteBackend {
    calls: Arc<AtomicUsize>,
    dimension_calls: Arc<AtomicUsize>,
    shutdowns: Arc<AtomicUsize>,
    run_origin_ns: Cell<Option<i64>>,
    model: String,
}

#[async_trait(?Send)]
impl HttpTurnExecutionBackend for FakeRemoteBackend {
    fn set_run_origin(&self, start_ns: i64) -> Result<()> {
        ensure!(self.run_origin_ns.replace(Some(start_ns)).is_none());
        Ok(())
    }

    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        self.dimension_calls.fetch_add(1, Ordering::SeqCst);
        InferenceDimensions {
            endpoint_url: Some("zmq://remote-worker".into()),
            model: turn
                .effective_model
                .clone()
                .or_else(|| Some(self.model.clone())),
        }
    }

    async fn execute_turn(
        &self,
        turn: PreparedHttpTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<HttpTurnDispatchResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let start_ns = self
            .run_origin_ns
            .get()
            .expect("run origin is configured before dispatch");
        let uuid = turn.request.uuid;
        observer.on_admit(uuid, 0.0, 0);
        on_first_token(100_000);
        observer.on_token(uuid, 0.1);
        observer.on_usage(
            uuid,
            ObservedUsage {
                prompt_tokens: Some(turn.request.input_length),
                completion_tokens: Some(1),
                ..ObservedUsage::default()
            },
        );
        observer.on_terminal(uuid, ReplayTerminalStatus::Completed);

        let request_payload = turn.request.request_body_bytes.clone().unwrap_or_default();
        Ok(HttpTurnDispatchResult {
            outcome: TurnDispatchOutcome {
                start_ns,
                end_ns: start_ns + 200_000,
                terminal: ReplayTerminalStatus::Completed,
                response_text: "remote response".into(),
                model_response: ModelResponseMetadata::default(),
                prompt_tokens: Some(turn.request.input_length as u64),
                completion_tokens: Some(1),
                http: HttpTrace::default(),
            },
            request_payload: request_payload.clone(),
            record: RequestRecord {
                start_ns,
                request_body: request_payload,
                request_headers: turn.request.headers,
                end_ns: Some(start_ns + 200_000),
                status: Some(200),
                ..RequestRecord::default()
            },
        })
    }

    fn shutdown(&self) -> Result<()> {
        self.shutdowns.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct RecordingGraphPlacement {
    builds: Arc<AtomicUsize>,
    traces: Arc<AtomicUsize>,
    artifact_target: PathBuf,
}

impl RunnerGraphPlacementFactory for RecordingGraphPlacement {
    fn build(
        &self,
        worker_count: usize,
        _worker_factory: Arc<dyn GraphTraceExecutionBackendFactory>,
    ) -> Result<Rc<dyn GraphTraceExecutionBackend>, GraphPlacementError> {
        assert_eq!(worker_count, 3);
        assert!(
            !self.artifact_target.exists(),
            "graph placement must be prepared before artifact creation"
        );
        self.builds.fetch_add(1, Ordering::SeqCst);
        Ok(Rc::new(RecordingGraphBackend {
            traces: self.traces.clone(),
        }))
    }
}

struct RecordingGraphBackend {
    traces: Arc<AtomicUsize>,
}

struct FailingOriginFactory {
    artifact_target: PathBuf,
    shutdowns: Arc<AtomicUsize>,
}

impl HttpExecutionBackendFactory for FailingOriginFactory {
    fn build(
        &self,
        _config: HttpExecutionBackendConfig,
    ) -> Result<Rc<dyn HttpTurnExecutionBackend>> {
        assert!(!self.artifact_target.exists());
        Ok(Rc::new(FailingOriginBackend {
            shutdowns: self.shutdowns.clone(),
        }))
    }
}

struct FailingOriginBackend {
    shutdowns: Arc<AtomicUsize>,
}

#[async_trait(?Send)]
impl HttpTurnExecutionBackend for FailingOriginBackend {
    fn set_run_origin(&self, _start_ns: i64) -> Result<()> {
        Err(anyhow!("intentional remote origin failure"))
    }

    fn inference_dimensions(&self, _turn: &TurnToSend) -> InferenceDimensions {
        unreachable!("origin failure prevents dispatch")
    }

    async fn execute_turn(
        &self,
        _turn: PreparedHttpTurn,
        _observer: &dyn RequestObserver,
        _on_first_token: &dyn Fn(i64),
    ) -> Result<HttpTurnDispatchResult> {
        unreachable!("origin failure prevents dispatch")
    }

    fn shutdown(&self) -> Result<()> {
        self.shutdowns.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Debug)]
struct RecordingReadinessTransportFactory {
    artifact_target: PathBuf,
    attempts: Arc<AtomicUsize>,
}

impl ReadinessTransportFactory for RecordingReadinessTransportFactory {
    fn build(&self, _clock: Rc<dyn Clock>) -> Rc<dyn ReadinessTransport> {
        Rc::new(RecordingReadinessTransport {
            artifact_target: self.artifact_target.clone(),
            attempts: self.attempts.clone(),
        })
    }
}

#[derive(Debug)]
struct RecordingReadinessTransport {
    artifact_target: PathBuf,
    attempts: Arc<AtomicUsize>,
}

#[async_trait(?Send)]
impl ReadinessTransport for RecordingReadinessTransport {
    async fn execute(&self, request: ReadinessAttemptRequest) -> ReadinessAttemptResponse {
        assert!(
            !self.artifact_target.exists(),
            "readiness must finish before the exclusive artifact target is created"
        );
        assert!(request.url().ends_with("/v1/models/remote-model"));
        self.attempts.fetch_add(1, Ordering::SeqCst);
        ReadinessAttemptResponse {
            status: Some(200),
            body: Some("{}".into()),
            error: None,
        }
    }
}

#[async_trait(?Send)]
impl GraphTraceExecutionBackend for RecordingGraphBackend {
    async fn execute_trace(
        &self,
        _plan: aiperf_graph::model::GraphTracePlan,
    ) -> Result<(), TraceError> {
        self.traces.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn coordinator(
    http: Arc<dyn HttpExecutionBackendFactory>,
    graph: Arc<dyn RunnerGraphPlacementFactory>,
) -> RunnerV2Coordinator {
    coordinator_with_readiness(http, graph, Arc::new(NativeHttpReadinessTransportFactory))
}

fn coordinator_with_readiness(
    http: Arc<dyn HttpExecutionBackendFactory>,
    graph: Arc<dyn RunnerGraphPlacementFactory>,
    readiness_transport: Arc<dyn ReadinessTransportFactory>,
) -> RunnerV2Coordinator {
    RunnerV2Coordinator::new(
        DISTRIBUTION_ID,
        &BuiltinRunnerRegistryFactory,
        &BuiltinAiperfRegistryFactory,
        RunnerExecutionFactories::new(
            http,
            graph,
            Arc::new(NativeHttpReadinessPlanFactory),
            readiness_transport,
        ),
        Arc::new(BuiltinRunnerGraphInputAdapterResolver::new()),
        Arc::new(BuiltinRunnerDatasetInputAdapterResolver::new()),
        Arc::new(BuiltinRunnerSidecarInputAdapterResolver::new()),
    )
    .unwrap()
}

fn execute(
    coordinator: &RunnerV2Coordinator,
    run: Value,
) -> aiperf_runner::coordinator::RunnerProcessResultV2 {
    let envelope = serde_json::from_value(json!({
        "protocol_version": 2,
        "operation": "execute",
        "expected_distribution_id": DISTRIBUTION_ID,
        "run": run,
    }))
    .unwrap();
    coordinator.handle(envelope)
}

fn assert_success(result: &aiperf_runner::coordinator::RunnerProcessResultV2) {
    assert_eq!(result.exit_code, 0);
    match &result.response {
        RunnerResponseV2::Terminal(terminal) => {
            assert!(terminal.success, "terminal errors: {:?}", terminal.errors);
        }
        RunnerResponseV2::Validation(_) => panic!("execute returned a validation response"),
    }
}

#[test]
fn v2_scheduled_run_uses_injected_remote_turn_placement() {
    let root = tempfile::tempdir().unwrap();
    let artifact_target = root.path().join("scheduled");
    let calls = Arc::new(AtomicUsize::new(0));
    let dimension_calls = Arc::new(AtomicUsize::new(0));
    let shutdowns = Arc::new(AtomicUsize::new(0));
    let readiness_attempts = Arc::new(AtomicUsize::new(0));
    let coordinator = coordinator_with_readiness(
        Arc::new(FakeRemoteFactory {
            calls: calls.clone(),
            dimension_calls: dimension_calls.clone(),
            shutdowns: shutdowns.clone(),
            artifact_target: artifact_target.clone(),
        }),
        Arc::new(NativeRunnerGraphPlacementFactory),
        Arc::new(RecordingReadinessTransportFactory {
            artifact_target: artifact_target.clone(),
            attempts: readiness_attempts.clone(),
        }),
    );
    let result = execute(
        &coordinator,
        json!({
            "identity": {"benchmark_id": "v2-remote-turn-placement", "random_seed": 17},
            "artifact_target": artifact_target,
            "resources": {
                "models": {"items": [{"name": "remote-model"}]},
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "kserve_v1_predict",
                    "urls": ["http://must-not-be-contacted.invalid"],
                    "streaming": false,
                    "use_server_token_count": true,
                    "wait_for_model_timeout": 1.0,
                    "wait_for_model_interval": 0.01,
                    "wait_for_model_mode": "models"
                }]}
            },
            "transport": {"type": "http", "config": {}},
            "workload": {"type": "scheduled", "config": {
                "worker_count": 3,
                "dataset": {
                    "type": "synthetic",
                    "entries": 3,
                    "sampling": "sequential",
                    "prompts": {
                        "isl": {"value": 4.0},
                        "osl": {"value": 1.0}
                    }
                },
                "tokenizer": {
                    "name": "cl100k_base",
                    "revision": "main",
                    "trust_remote_code": false,
                    "apply_chat_template": false
                },
                "phases": [{
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "requests": 3,
                    "concurrency": 2
                }]
            }}
        }),
    );

    assert_success(&result);
    assert_eq!(readiness_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(dimension_calls.load(Ordering::SeqCst), 3);
    assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
    let report: Value = serde_json::from_slice(
        &std::fs::read(root.path().join("scheduled/native-v2.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"],
        3.0
    );
}

#[test]
fn v2_graph_run_uses_injected_whole_trace_placement() {
    let root = tempfile::tempdir().unwrap();
    let artifact_target = root.path().join("graph");
    let builds = Arc::new(AtomicUsize::new(0));
    let traces = Arc::new(AtomicUsize::new(0));
    let coordinator = coordinator(
        Arc::new(NativeHttpExecutionBackendFactory),
        Arc::new(RecordingGraphPlacement {
            builds: builds.clone(),
            traces: traces.clone(),
            artifact_target: artifact_target.clone(),
        }),
    );
    let result = execute(
        &coordinator,
        json!({
            "identity": {"benchmark_id": "v2-remote-graph-placement", "random_seed": 19},
            "artifact_target": artifact_target,
            "resources": {
                "models": {"items": [{"name": "remote-model"}]},
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "chat",
                    "urls": ["http://must-not-be-contacted.invalid"],
                    "streaming": true,
                    "use_server_token_count": true,
                    "wait_for_model_timeout": 0.0
                }]}
            },
            "transport": {"type": "http", "config": {}},
            "workload": {"type": "graph", "config": {
                "worker_count": 3,
                "dataset": {
                    "type": "file",
                    "format": "dag_jsonl",
                    "sampling": "sequential",
                    "records": [{
                        "session_id": "root",
                        "turns": [{
                            "messages": [{"role": "user", "content": "hello"}],
                            "max_tokens": 1
                        }]
                    }]
                },
                "tokenizer": {
                    "name": "cl100k_base",
                    "revision": "main",
                    "trust_remote_code": false,
                    "apply_chat_template": false
                },
                "phases": [{
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "sessions": 1,
                    "concurrency": 1
                }]
            }}
        }),
    );

    assert_success(&result);
    assert_eq!(builds.load(Ordering::SeqCst), 1);
    assert_eq!(traces.load(Ordering::SeqCst), 1);
    assert!(root.path().join("graph/native-v2.json").is_file());
}

#[test]
fn v2_backend_shutdown_runs_after_pre_artifact_origin_failure() {
    let root = tempfile::tempdir().unwrap();
    let artifact_target = root.path().join("origin-failure");
    let shutdowns = Arc::new(AtomicUsize::new(0));
    let coordinator = coordinator(
        Arc::new(FailingOriginFactory {
            artifact_target: artifact_target.clone(),
            shutdowns: shutdowns.clone(),
        }),
        Arc::new(NativeRunnerGraphPlacementFactory),
    );
    let result = execute(
        &coordinator,
        json!({
            "identity": {"benchmark_id": "v2-origin-failure", "random_seed": 23},
            "artifact_target": artifact_target,
            "resources": {
                "models": {"items": [{"name": "remote-model"}]},
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "chat",
                    "urls": ["http://must-not-be-contacted.invalid"],
                    "streaming": true,
                    "use_server_token_count": true,
                    "wait_for_model_timeout": 0.0
                }]}
            },
            "transport": {"type": "http", "config": {}},
            "workload": {"type": "scheduled", "config": {
                "worker_count": 1,
                "dataset": {
                    "type": "synthetic",
                    "entries": 1,
                    "sampling": "sequential",
                    "prompts": {
                        "isl": {"value": 4.0},
                        "osl": {"value": 1.0}
                    }
                },
                "tokenizer": {
                    "name": "cl100k_base",
                    "revision": "main",
                    "trust_remote_code": false,
                    "apply_chat_template": false
                },
                "phases": [{
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "requests": 1,
                    "concurrency": 1
                }]
            }}
        }),
    );

    assert_eq!(result.exit_code, 1);
    match result.response {
        RunnerResponseV2::Terminal(terminal) => {
            assert!(!terminal.success);
            assert!(terminal.errors[0].message.contains("origin failure"));
        }
        RunnerResponseV2::Validation(_) => panic!("execute returned a validation response"),
    }
    assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
    assert!(!root.path().join("origin-failure").exists());
}
