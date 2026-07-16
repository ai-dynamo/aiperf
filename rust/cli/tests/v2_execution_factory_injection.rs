// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2 proof that placement factories sit below the single dispatcher.
//!
//! Both tests enter through [`RunnerV2Coordinator`]. Neither changes workload,
//! phase, metrics, artifact, or report logic: one replaces turn placement with
//! a remote-shaped backend, and one replaces whole-trace graph placement.

use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use aiperf_runtime::engine::coordinator::{RunnerResponseV2, RunnerV2Coordinator};
use aiperf_runtime::engine::dataset_input::BuiltinRunnerDatasetInputAdapterResolver;
use aiperf_runtime::engine::execution_factories::RunnerExecutionFactories;
use aiperf_runtime::engine::graph_execution::{
    NativeRunnerGraphPlacementFactory, RunnerGraphPlacementFactory,
};
use aiperf_runtime::engine::graph_input::BuiltinRunnerGraphInputAdapterResolver;
use aiperf_runtime::engine::readiness::{
    NativeHttpReadinessPlanFactory, NativeHttpReadinessTransportFactory, ReadinessTransportFactory,
};
use aiperf_runtime::engine::sidecar_input::BuiltinRunnerSidecarInputAdapterResolver;
use aiperf_runtime::engine::turn_execution::{
    ExecutionBackendConfig, HttpExecutionFactory, RequestExecutorFactory,
};
use aiperf_runtime::extensions::BuiltinAIPerfRegistryFactory;
use aiperf_runtime::graph::errors::TraceError;
use aiperf_runtime::graph::execution::TracePlacement;
use aiperf_runtime::graph::placement::{GraphPlacementError, TracePlacementFactory};
use aiperf_runtime::metrics_core::InferenceDimensions;
use aiperf_runtime::multiturn::TurnToSend;
use aiperf_runtime::transport::core::{MeasuredContext, MeasuredOutcome};
use aiperf_runtime::transport::http::{PreparedTurn, RequestExecutor};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde_json::{Value, json};

const DISTRIBUTION_ID: &str =
    "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

struct RecordingGraphPlacement {
    builds: Arc<AtomicUsize>,
    traces: Arc<AtomicUsize>,
    artifact_target: PathBuf,
}

impl RunnerGraphPlacementFactory for RecordingGraphPlacement {
    fn build(
        &self,
        worker_count: usize,
        _worker_factory: Arc<dyn TracePlacementFactory>,
    ) -> Result<Rc<dyn TracePlacement>, GraphPlacementError> {
        assert_eq!(worker_count, 3);
        assert!(
            self.artifact_target.exists(),
            "BenchmarkRun commits the artifact target before graph placement"
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

impl RequestExecutorFactory for FailingOriginFactory {
    fn build(&self, _config: ExecutionBackendConfig) -> Result<Rc<dyn RequestExecutor>> {
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
impl RequestExecutor for FailingOriginBackend {
    fn set_run_origin(&self, _start_ns: i64) -> Result<()> {
        Err(anyhow!("intentional remote origin failure"))
    }

    fn inference_dimensions(&self, _turn: &TurnToSend) -> InferenceDimensions {
        unreachable!("origin failure prevents dispatch")
    }

    async fn execute_measured(
        &self,
        _turn: PreparedTurn,
        _context: MeasuredContext,
        _on_first_token: &dyn Fn(i64),
    ) -> Result<MeasuredOutcome> {
        unreachable!("origin failure prevents dispatch")
    }

    fn shutdown(&self) -> Result<()> {
        self.shutdowns.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[async_trait(?Send)]
impl TracePlacement for RecordingGraphBackend {
    async fn execute_trace(
        &self,
        _plan: aiperf_runtime::graph::model::GraphTracePlan,
    ) -> Result<(), TraceError> {
        self.traces.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn coordinator(
    http: Arc<dyn RequestExecutorFactory>,
    graph: Arc<dyn RunnerGraphPlacementFactory>,
) -> RunnerV2Coordinator {
    coordinator_with_readiness(http, graph, Arc::new(NativeHttpReadinessTransportFactory))
}

fn coordinator_with_readiness(
    http: Arc<dyn RequestExecutorFactory>,
    graph: Arc<dyn RunnerGraphPlacementFactory>,
    readiness_transport: Arc<dyn ReadinessTransportFactory>,
) -> RunnerV2Coordinator {
    RunnerV2Coordinator::new(
        DISTRIBUTION_ID,
        &BuiltinAIPerfRegistryFactory,
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

fn benchmark_run(legacy: Value) -> Value {
    let mut endpoint = legacy["resources"]["endpoints"]["profiles"][0].clone();
    endpoint.as_object_mut().unwrap().remove("id");
    json!({
        "benchmark_id": legacy["identity"]["benchmark_id"],
        "artifact_dir": legacy["artifact_target"],
        "random_seed": legacy["identity"]["random_seed"],
        "cfg": {
            "models": legacy["resources"]["models"],
            "endpoint": endpoint,
            "datasets": [legacy["workload"]["config"]["dataset"]],
            "phases": legacy["workload"]["config"]["phases"],
            "tokenizer": legacy["workload"]["config"]["tokenizer"],
            "transport": {"type": legacy["transport"]["type"]},
            "runtime": {"workers": legacy["workload"]["config"]["worker_count"]}
        }
    })
}

fn execute(
    coordinator: &RunnerV2Coordinator,
    run: Value,
) -> aiperf_runtime::engine::coordinator::RunnerProcessResultV2 {
    let envelope = serde_json::from_value(json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": benchmark_run(run),
    }))
    .unwrap();
    coordinator.handle(envelope)
}

fn assert_success(result: &aiperf_runtime::engine::coordinator::RunnerProcessResultV2) {
    assert_eq!(result.exit_code, 0, "{:?}", result.response);
    match &result.response {
        RunnerResponseV2::Terminal(terminal) => {
            assert!(terminal.success, "terminal errors: {:?}", terminal.errors);
        }
        RunnerResponseV2::Validation(_) => panic!("execute returned a validation response"),
    }
}

#[test]
fn v2_graph_run_uses_injected_whole_trace_placement() {
    let root = tempfile::tempdir().unwrap();
    let artifact_target = root.path().join("graph");
    let builds = Arc::new(AtomicUsize::new(0));
    let traces = Arc::new(AtomicUsize::new(0));
    let coordinator = coordinator(
        Arc::new(HttpExecutionFactory),
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
