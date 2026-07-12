// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2 proof that placement factories sit below the single dispatcher.
//!
//! Both tests enter through [`RunnerV2Coordinator`]. Neither changes workload,
//! phase, metrics, artifact, or report logic: one replaces turn placement with
//! a remote-shaped backend, and one replaces whole-trace graph placement.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use aiperf::http::{HttpTurnDispatchResult, HttpTurnExecutionBackend, PreparedHttpTurn};
use aiperf::multiturn::TurnToSend;
use aiperf::scheduled::{ModelResponseMetadata, TurnDispatchOutcome};
use aiperf_extensions::BuiltinAiperfRegistryFactory;
use aiperf_graph::errors::TraceError;
use aiperf_graph::execution::GraphTraceExecutionBackend;
use aiperf_graph::placement::{GraphPlacementError, GraphTraceExecutionBackendFactory};
use aiperf_metrics::{HttpTrace, InferenceDimensions};
use aiperf_runner::coordinator::{RunnerResponseV2, RunnerV2Coordinator};
use aiperf_runner::dataset_input::BuiltinRunnerDatasetInputAdapterResolver;
use aiperf_runner::graph_input::BuiltinRunnerGraphInputAdapterResolver;
use aiperf_runner::readiness::{
    NativeHttpReadinessPlanFactory, NativeHttpReadinessTransportFactory,
};
use aiperf_runner::registry::BuiltinRunnerRegistryFactory;
use aiperf_runner::sidecar_input::BuiltinRunnerSidecarInputAdapterResolver;
use aiperf_runner::{
    HttpExecutionBackendConfig, HttpExecutionBackendFactory, NativeHttpExecutionBackendFactory,
    NativeRunnerGraphPlacementFactory, RunnerExecutionFactories, RunnerGraphPlacementFactory,
};
use aiperf_transport_http::models::RequestRecord;
use anyhow::{Result, ensure};
use async_trait::async_trait;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{ObservedUsage, RequestObserver};
use serde_json::{Value, json};

const DISTRIBUTION_ID: &str =
    "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

struct FakeRemoteFactory {
    calls: Arc<AtomicUsize>,
    dimension_calls: Arc<AtomicUsize>,
}

impl HttpExecutionBackendFactory for FakeRemoteFactory {
    fn build(
        &self,
        config: HttpExecutionBackendConfig,
    ) -> Result<Rc<dyn HttpTurnExecutionBackend>> {
        ensure!(config.workers == 3, "resolved worker count reached factory");
        Ok(Rc::new(FakeRemoteBackend {
            calls: self.calls.clone(),
            dimension_calls: self.dimension_calls.clone(),
            run_origin_ns: Cell::new(None),
            model: config.model,
        }))
    }
}

struct FakeRemoteBackend {
    calls: Arc<AtomicUsize>,
    dimension_calls: Arc<AtomicUsize>,
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
}

struct RecordingGraphPlacement {
    builds: Arc<AtomicUsize>,
    traces: Arc<AtomicUsize>,
}

impl RunnerGraphPlacementFactory for RecordingGraphPlacement {
    fn build(
        &self,
        worker_count: usize,
        _worker_factory: Arc<dyn GraphTraceExecutionBackendFactory>,
    ) -> Result<Rc<dyn GraphTraceExecutionBackend>, GraphPlacementError> {
        assert_eq!(worker_count, 3);
        self.builds.fetch_add(1, Ordering::SeqCst);
        Ok(Rc::new(RecordingGraphBackend {
            traces: self.traces.clone(),
        }))
    }
}

struct RecordingGraphBackend {
    traces: Arc<AtomicUsize>,
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
    RunnerV2Coordinator::new(
        DISTRIBUTION_ID,
        &BuiltinRunnerRegistryFactory,
        &BuiltinAiperfRegistryFactory,
        RunnerExecutionFactories::new(
            http,
            graph,
            Arc::new(NativeHttpReadinessPlanFactory),
            Arc::new(NativeHttpReadinessTransportFactory),
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
    let coordinator = coordinator(
        Arc::new(FakeRemoteFactory {
            calls: calls.clone(),
            dimension_calls: dimension_calls.clone(),
        }),
        Arc::new(NativeRunnerGraphPlacementFactory),
    );
    let result = execute(
        &coordinator,
        json!({
            "identity": {"benchmark_id": "v2-remote-turn-placement", "random_seed": 17},
            "artifact_target": artifact_target,
            "models": {"items": [{"name": "remote-model"}]},
            "endpoints": {"profiles": [{
                "id": "default",
                "type": "chat",
                "urls": ["http://must-not-be-contacted.invalid"],
                "streaming": true,
                "use_server_token_count": true,
                "wait_for_model_timeout": 0.0
            }]},
            "backend": {"type": "online_http", "config": {}},
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
            }},
            "metrics": {},
            "artifacts": {},
            "sidecars": {}
        }),
    );

    assert_success(&result);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(dimension_calls.load(Ordering::SeqCst), 3);
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
        }),
    );
    let result = execute(
        &coordinator,
        json!({
            "identity": {"benchmark_id": "v2-remote-graph-placement", "random_seed": 19},
            "artifact_target": artifact_target,
            "models": {"items": [{"name": "remote-model"}]},
            "endpoints": {"profiles": [{
                "id": "default",
                "type": "chat",
                "urls": ["http://must-not-be-contacted.invalid"],
                "streaming": true,
                "use_server_token_count": true,
                "wait_for_model_timeout": 0.0
            }]},
            "backend": {"type": "online_http", "config": {}},
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
            }},
            "metrics": {},
            "artifacts": {},
            "sidecars": {}
        }),
    );

    assert_success(&result);
    assert_eq!(builds.load(Ordering::SeqCst), 1);
    assert_eq!(traces.load(Ordering::SeqCst), 1);
    assert!(root.path().join("graph/native-v2.json").is_file());
}
