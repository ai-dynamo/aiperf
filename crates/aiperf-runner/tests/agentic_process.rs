// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process proof for canonical worker + Rust gateway + the one HTTP dispatcher.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use aiperf::http::{HttpTurnDispatchResult, HttpTurnExecutionBackend, PreparedHttpTurn};
use aiperf::multiturn::TurnToSend;
use aiperf_endpoints::{
    ChatEndpoint, EffectiveEndpointConfig, EndpointDescriptor, EndpointFactory, EndpointId,
    EndpointResult, ExtractedPayload, Modality, ParsedResponse, PreparedEndpoint, PreparedRequest,
    RawEndpointConfig, ReadinessPolicy, RequestRecord as EndpointRequestRecord, ServerResponse,
    StatelessEndpointFactory, Turn,
};
use aiperf_extensions::{AiperfRegistry, AiperfRegistryFactory, BuiltinAiperfRegistryFactory};
use aiperf_runner::agentic_execution::{
    AgenticOnlineEndpointSpec, AgenticOnlineExecutionSpec, AgenticWorkloadConfigV2,
    NativeAgenticEvaluatorProcessFactory, NativeAgenticGatewayFactory,
    NativeAgenticTokenizerFactory, execute_agentic_online_with_factories,
};
use aiperf_runner::turn_execution::{
    HttpExecutionBackendConfig, HttpExecutionBackendFactory, NativeHttpExecutionBackendFactory,
};
use anyhow::Result;
use async_trait::async_trait;
use axum::extract::State;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use loadgen_core::sink::RequestObserver;
use serde_json::{Value, json};
use tempfile::TempDir;

const PROCESS_EVALUATOR: &str = r#"
import json
import os
import sys
import threading
import urllib.parse
import urllib.request

episodes = [
    {"episode_id": "episode-scored", "task": "fixture.scored-zero", "source": "fixture/agentic"},
    {"episode_id": "episode-infra", "task": "fixture.infrastructure", "source": "fixture/agentic"},
    {"episode_id": "episode-cancelled", "task": "fixture.cancelled", "source": "fixture/agentic"},
]
events = []
results = {}
lock = threading.Lock()
gateway = None
auxiliary_thread = None

def append_event(event):
    with lock:
        events.append(event)

def primary_call(episode):
    return {
        "kind": "model_call",
        "call": {
            "episode_id": episode["episode_id"],
            "call_id": episode["episode_id"] + ":call:00000000",
            "turn_index": 0,
            "prompt": "Use the canonical fixture tool",
            "messages": [{"role": "user", "content": "Use the canonical fixture tool"}],
            "generation": {"max_tokens": 31, "temperature": 0.2, "top_p": 0.9, "stop": ["END"]},
            "tools": [{"type": "function", "function": {"name": "answer", "parameters": {"type": "object"}}}],
            "tool_choice": "auto",
        },
    }

def auxiliary_calls():
    episode_id = "episode-scored"
    encoded = urllib.parse.quote(episode_id, safe="")
    try:
        for purpose, model in (("environment", "environment-model"), ("verifier", "verifier-model")):
            url = f'{gateway["base_url"]}/episodes/{encoded}/{purpose}/v1/chat/completions'
            body = {
                "model": model,
                "messages": [
                    {"role": "assistant", "content": None, "tool_calls": [{"id": "prior", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "prior", "content": "ready"},
                    {"role": "user", "content": f"Run canonical {purpose}"},
                ],
                "tools": [{"type": "function", "function": {"name": "answer", "parameters": {"type": "object"}}}],
                "tool_choice": "auto",
                "max_tokens": 29,
                "stream": False,
            }
            request = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers={"Authorization": f'Bearer {gateway["api_key"]}', "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            assert payload["choices"][0]["message"]["tool_calls"][0]["id"] == "answer-call"
            assert payload["usage"]["prompt_tokens"] == 7
            assert payload["usage"]["completion_tokens"] == 3
        result = {
            "episode_id": episode_id,
            "task": "fixture.scored-zero",
            "outcome": "completed",
            "rewards": {"reward": 0.0},
            "primary_reward": "reward",
            "duration_seconds": 0.25,
            "model_calls": 1,
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "cached_tokens": 2,
            "artifact_path": "fixture/scored",
        }
    except Exception as error:
        result = {
            "episode_id": episode_id,
            "task": "fixture.scored-zero",
            "outcome": "infrastructure_error",
            "rewards": {},
            "duration_seconds": 0.25,
            "model_calls": 1,
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "cached_tokens": 2,
            "error_kind": type(error).__name__,
            "error_message": str(error),
        }
    results[episode_id] = result
    append_event({"kind": "episode_completed", "result": result})

for line in sys.stdin:
    request = json.loads(line)
    operation = request["op"]
    if operation == "hello":
        result = {
            "protocol": 1,
            "worker_version": "runner-agentic-process-fixture",
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "packages": {"fixture-harness": "1.0.0"},
            "worker_source_sha256": "a" * 64,
            "dependency_lock_sha256": "b" * 64,
            "container_digest": None,
            "capabilities": ["load", "next_problems", "grade_batch", "shutdown", "agentic", "agentic_inference_gateway"],
        }
    elif operation == "load_agentic":
        assert request["dataset"] == "fixture/agentic@locked"
        assert request["model"] == "primary-model"
        assert request["config"]["task_concurrency"] == 2
        expected_absent = os.environ.get("AIPERF_EXPECT_ARTIFACT_ABSENT")
        if expected_absent:
            assert not os.path.exists(expected_absent), expected_absent
        gateway = request["config"]["inference_gateway"]
        assert gateway["base_url"].startswith("http://127.0.0.1:")
        result = {
            "harness": "fixture-canonical",
            "harness_version": "1.0.0",
            "harness_source_sha256": "c" * 64,
            "dataset": {"provider": "fixture", "benchmark": "fixture/agentic", "repository": "fixture/agentic", "revision": "d" * 64, "evaluation_splits": ["tasks"]},
            "agent": "fixture-agent",
            "agent_version": "1.0.0",
            "environment": "fixture",
            "verifier": "fixture canonical verifier",
            "episode_count": 3,
            "primary_reward": "reward",
        }
    elif operation == "next_episodes":
        offset = request["offset"]
        page = episodes[offset:offset + request["limit"]]
        result = {"items": page, "next_offset": offset + len(page), "done": offset + len(page) >= len(episodes)}
    elif operation == "start_episodes":
        for episode_id in request["episode_ids"]:
            append_event(primary_call(next(item for item in episodes if item["episode_id"] == episode_id)))
        result = {"started": request["episode_ids"]}
    elif operation == "poll_agentic":
        with lock:
            page = events[:request["limit"]]
            del events[:len(page)]
        result = {"events": page}
    elif operation == "submit_model_results":
        accepted = []
        for item in request["items"]:
            assert item["status"] == "completed"
            assert item["prompt_tokens"] == 7
            assert item["completion_tokens"] == 3
            accepted.append(item["call_id"])
            if item["episode_id"] == "episode-scored":
                auxiliary_thread = threading.Thread(target=auxiliary_calls, daemon=True)
                auxiliary_thread.start()
            elif item["episode_id"] == "episode-infra":
                terminal = {
                    "episode_id": "episode-infra",
                    "task": "fixture.infrastructure",
                    "outcome": "infrastructure_error",
                    "rewards": {},
                    "duration_seconds": 0.1,
                    "model_calls": 1,
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "cached_tokens": 2,
                    "error_kind": "SandboxStartup",
                    "error_message": "fixture sandbox unavailable",
                    "artifact_path": "fixture/infra",
                }
                results["episode-infra"] = terminal
                append_event({"kind": "episode_completed", "result": terminal})
            else:
                terminal = {
                    "episode_id": "episode-cancelled",
                    "task": "fixture.cancelled",
                    "outcome": "cancelled",
                    "rewards": {},
                    "duration_seconds": 0.05,
                    "model_calls": 1,
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "cached_tokens": 2,
                    "error_kind": "CancelledByScheduler",
                    "error_message": "fixture cancellation",
                    "artifact_path": "fixture/cancelled",
                }
                results["episode-cancelled"] = terminal
                append_event({"kind": "episode_completed", "result": terminal})
        result = {"accepted": accepted}
    elif operation == "cancel_episodes":
        result = {"cancelled": request["episode_ids"]}
    elif operation == "finish_agentic":
        result = {"items": [results[item["episode_id"]] for item in episodes]}
    elif operation == "shutdown":
        if auxiliary_thread is not None:
            auxiliary_thread.join(timeout=5)
        result = {"shutdown": True}
    else:
        raise RuntimeError(operation)
    print(json.dumps({"id": request["id"], "ok": True, "result": result}), flush=True)
    if operation == "shutdown":
        break
"#;

static PREPARED_ONLY_AGENTIC_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "fixture_agentic",
    aliases: &["fixture_agentic_alias"],
    description: "prepared-only agentic process fixture",
    endpoint_path: Some("/v1/chat/completions"),
    streaming_path: Some("/v1/chat/completions"),
    supports_streaming: true,
    produces_tokens: true,
    tokenizes_input: true,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Tokens],
    metrics_title: "Fixture LLM Metrics",
    service_kind: "fixture-llm",
};

#[derive(Clone, Copy, Debug)]
struct PreparedOnlyAgenticEndpointFactory;

impl EndpointFactory for PreparedOnlyAgenticEndpointFactory {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &PREPARED_ONLY_AGENTIC_DESCRIPTOR
    }

    fn prepare(
        &self,
        config: EffectiveEndpointConfig,
    ) -> EndpointResult<Box<dyn PreparedEndpoint>> {
        let inner = StatelessEndpointFactory::new(ChatEndpoint).prepare(config)?;
        Ok(Box::new(PreparedOnlyAgenticEndpoint { inner }))
    }
}

#[derive(Debug)]
struct PreparedOnlyAgenticEndpoint {
    inner: Box<dyn PreparedEndpoint>,
}

impl PreparedEndpoint for PreparedOnlyAgenticEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &PREPARED_ONLY_AGENTIC_DESCRIPTOR
    }

    fn config(&self) -> &EffectiveEndpointConfig {
        self.inner.config()
    }

    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<Value> {
        self.inner.format_payload(request)
    }

    fn headers(&self) -> &BTreeMap<String, String> {
        self.inner.headers()
    }

    fn readiness_policy(&self, model: &str) -> EndpointResult<ReadinessPolicy> {
        self.inner.readiness_policy(model)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        self.inner.parse_response(response)
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        self.inner.extract_payload_inputs(body)
    }

    fn extract_response_data(
        &self,
        record: &EndpointRequestRecord,
    ) -> EndpointResult<Vec<ParsedResponse>> {
        self.inner.extract_response_data(record)
    }

    fn build_assistant_turn(&self, record: &EndpointRequestRecord) -> EndpointResult<Option<Turn>> {
        self.inner.build_assistant_turn(record)
    }

    fn captures_assistant_turn(&self) -> bool {
        self.inner.captures_assistant_turn()
    }
}

#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<Value>>>);

async fn completion(
    State(captured): State<Captured>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let response_id = {
        let mut captured = captured.0.lock().unwrap();
        let response_id = format!("provider-{}", captured.len());
        captured.push(body);
        response_id
    };
    let first = json!({
        "id": response_id,
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{"index": 0, "id": "answer-call", "type": "function", "function": {"name": "answer", "arguments": "{\"value\":"}}]
            },
            "finish_reason": null
        }]
    });
    let second = json!({
        "id": response_id,
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]},
            "finish_reason": "tool_calls"
        }]
    });
    let usage = json!({
        "id": response_id,
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 2}}
    });
    (
        [(header::CONTENT_TYPE, "text/event-stream")],
        format!("data: {first}\n\ndata: {second}\n\ndata: {usage}\n\ndata: [DONE]\n\n"),
    )
}

struct MockServer {
    base_url: String,
    captured: Captured,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl MockServer {
    fn spawn() -> Self {
        let captured = Captured::default();
        let captured_for_thread = captured.clone();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let thread = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let app = Router::new()
                    .route("/v1/chat/completions", post(completion))
                    .with_state(captured_for_thread);
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let address = listener.local_addr().unwrap();
                let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
                ready_tx.send((address, shutdown_tx)).unwrap();
                axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
                        let _ = shutdown_rx.await;
                    })
                    .await
                    .unwrap();
            });
        });
        let (address, shutdown) = ready_rx.recv().unwrap();
        Self {
            base_url: format!("http://{address}"),
            captured,
            shutdown: Some(shutdown),
            thread: Some(thread),
        }
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

#[derive(Clone, Default)]
struct CountingRegistryFactory {
    builds: Arc<AtomicUsize>,
    prepared_only_agentic: bool,
}

impl AiperfRegistryFactory for CountingRegistryFactory {
    fn build(&self) -> std::result::Result<AiperfRegistry, aiperf_extensions::ExtensionError> {
        self.builds.fetch_add(1, Ordering::Relaxed);
        let mut registry = BuiltinAiperfRegistryFactory.build()?;
        if self.prepared_only_agentic {
            registry.register_endpoint_factory(PreparedOnlyAgenticEndpointFactory)?;
        }
        Ok(registry)
    }
}

#[derive(Clone, Default)]
struct CountingBackendFactory {
    builds: Arc<AtomicUsize>,
    turns: Arc<AtomicUsize>,
    expected_absent: Option<PathBuf>,
}

impl HttpExecutionBackendFactory for CountingBackendFactory {
    fn build(
        &self,
        config: HttpExecutionBackendConfig,
    ) -> Result<Rc<dyn HttpTurnExecutionBackend>> {
        if let Some(path) = &self.expected_absent {
            assert!(
                !path.exists(),
                "artifact target existed before HTTP worker preparation: {}",
                path.display()
            );
        }
        self.builds.fetch_add(1, Ordering::Relaxed);
        let inner = NativeHttpExecutionBackendFactory.build(config)?;
        Ok(Rc::new(CountingBackend {
            inner,
            turns: self.turns.clone(),
        }))
    }
}

struct CountingBackend {
    inner: Rc<dyn HttpTurnExecutionBackend>,
    turns: Arc<AtomicUsize>,
}

#[async_trait(?Send)]
impl HttpTurnExecutionBackend for CountingBackend {
    fn set_run_origin(&self, start_ns: i64) -> Result<()> {
        self.inner.set_run_origin(start_ns)
    }

    fn inference_dimensions(&self, turn: &TurnToSend) -> aiperf_metrics::InferenceDimensions {
        self.inner.inference_dimensions(turn)
    }

    async fn execute_turn(
        &self,
        turn: PreparedHttpTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<HttpTurnDispatchResult> {
        self.turns.fetch_add(1, Ordering::Relaxed);
        self.inner
            .execute_turn(turn, observer, on_first_token)
            .await
    }

    fn shutdown(&self) -> Result<()> {
        self.inner.shutdown()
    }
}

fn python_executable() -> PathBuf {
    if let Some(path) = std::env::var_os("PYTHON").map(PathBuf::from)
        && path.is_absolute()
        && path.is_file()
    {
        return path;
    }
    for directory in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        for name in ["python3", "python"] {
            let candidate = directory.join(name);
            if candidate.is_file() {
                return candidate.canonicalize().unwrap();
            }
        }
    }
    panic!("no Python interpreter found for agentic process test");
}

fn write_worker(root: &TempDir) {
    let package = root.path().join("fixture_agentic");
    std::fs::create_dir_all(&package).unwrap();
    std::fs::write(package.join("__init__.py"), "").unwrap();
    std::fs::write(package.join("worker.py"), PROCESS_EVALUATOR).unwrap();
}

fn runner_capabilities() -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_aiperf-runner"))
        .arg("--capabilities")
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    one_json_line(&output.stdout)
}

fn run_runner(request: &Value) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_aiperf-runner"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(request).unwrap().as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

fn one_json_line(stdout: &[u8]) -> Value {
    let lines = stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 1, "stdout={}", String::from_utf8_lossy(stdout));
    serde_json::from_slice(lines[0]).unwrap()
}

#[test]
fn process_worker_and_auxiliary_gateway_share_the_injected_dispatcher() {
    let server = MockServer::spawn();
    let root = TempDir::new().unwrap();
    write_worker(&root);
    let python = python_executable();
    let artifact_target = root.path().join("run");
    let workload: AgenticWorkloadConfigV2 = serde_json::from_value(json!({
        "worker_count": 2,
        "dataset": "fixture/agentic@locked",
        "tokenizer": {"name": "builtin"},
        "phases": [{
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "concurrency": 2
        }],
        "evaluator": {
            "python_executable": python,
            "worker_module": "fixture_agentic.worker",
            "environment": {
                "PYTHONPATH": root.path(),
                "AIPERF_EXPECT_ARTIFACT_ABSENT": artifact_target.clone()
            }
        },
        "task_concurrency": 2,
        "environment": "fixture",
        "output_dir": "episodes",
        "max_tokens": 64,
        "context_window": 4096,
        "inference_gateway_host": "127.0.0.1"
    }))
    .unwrap();
    let registry = CountingRegistryFactory {
        prepared_only_agentic: true,
        ..CountingRegistryFactory::default()
    };
    let backend = CountingBackendFactory {
        expected_absent: Some(artifact_target.clone()),
        ..CountingBackendFactory::default()
    };
    let spec = AgenticOnlineExecutionSpec {
        benchmark_id: "agentic-process-proof".to_string(),
        artifact_target,
        model: "primary-model".to_string(),
        workload,
        tokenizer: Default::default(),
        endpoint: AgenticOnlineEndpointSpec {
            endpoint_id: EndpointId::new("fixture_agentic_alias").unwrap(),
            config: RawEndpointConfig {
                urls: vec![server.base_url.clone()],
                streaming: true,
                ..RawEndpointConfig::default()
            },
            connection_reuse: Default::default(),
            session_header: None,
            client: Default::default(),
        },
    };
    let outcome = execute_agentic_online_with_factories(
        spec,
        &registry,
        &backend,
        &NativeAgenticEvaluatorProcessFactory,
        &NativeAgenticGatewayFactory,
        &NativeAgenticTokenizerFactory,
    )
    .unwrap();

    assert!(outcome.report_path.is_file());
    assert_eq!(registry.builds.load(Ordering::Relaxed), 1);
    assert_eq!(backend.builds.load(Ordering::Relaxed), 1);
    assert_eq!(backend.turns.load(Ordering::Relaxed), 5);
    assert_eq!(outcome.report.evaluation.summary.episode_count, 3);
    assert_eq!(outcome.report.evaluation.summary.completed_count, 1);
    assert_eq!(
        outcome.report.evaluation.summary.infrastructure_error_count,
        1
    );
    assert_eq!(outcome.report.evaluation.summary.cancelled_count, 1);
    assert_eq!(outcome.report.evaluation.summary.model_calls, 5);
    assert_eq!(outcome.report.evaluation.summary.primary_model_calls, 3);
    assert_eq!(outcome.report.evaluation.summary.auxiliary_model_calls, 2);
    assert_eq!(outcome.report.evaluation.summary.environment_model_calls, 1);
    assert_eq!(outcome.report.evaluation.summary.verifier_model_calls, 1);
    assert_eq!(outcome.report.evaluation.summary.primary_score, Some(0.0));
    let reward = &outcome.report.evaluation.summary.rewards["reward"];
    assert_eq!(reward.n, 1);
    assert_eq!(reward.avg, 0.0);

    let report_bytes = std::fs::read(&outcome.report_path).unwrap();
    let report: Value = serde_json::from_slice(&report_bytes).unwrap();
    assert_eq!(report["schema_version"], "2.0");
    assert_eq!(report["run"]["mode"], "agentic_accuracy");
    assert_eq!(report["agentic"]["records"][0]["rewards"]["reward"], 0.0);
    assert_eq!(
        report["agentic"]["records"][1]["outcome"],
        "infrastructure_error"
    );
    assert_eq!(report["agentic"]["records"][2]["outcome"], "cancelled");
    assert_eq!(
        report["errors"][0]["type"],
        "AgenticCancelled:CancelledByScheduler"
    );
    assert_eq!(
        report["errors"][1]["type"],
        "AgenticInfrastructure:SandboxStartup"
    );
    let metric_requests = report["metrics"]["request_count"]["series"]
        .as_array()
        .unwrap()
        .iter()
        .map(|series| series["stats"]["total"].as_f64().unwrap())
        .sum::<f64>();
    assert_eq!(metric_requests, 5.0);
    assert!(!String::from_utf8_lossy(&report_bytes).contains("api_key"));

    let requests = server.captured.0.lock().unwrap();
    assert_eq!(requests.len(), 5);
    assert!(requests.iter().all(|body| body["stream"] == true));
    assert!(
        requests
            .iter()
            .all(|body| body["stream_options"]["include_usage"] == true)
    );
    let models = requests
        .iter()
        .map(|body| body["model"].as_str().unwrap())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        models,
        BTreeSet::from(["environment-model", "primary-model", "verifier-model"])
    );
    let auxiliary = requests
        .iter()
        .find(|body| body["model"] == "environment-model")
        .unwrap();
    assert_eq!(auxiliary["messages"][0]["tool_calls"][0]["id"], "prior");
    assert_eq!(auxiliary["messages"][1]["tool_call_id"], "prior");
}

#[test]
fn protocol_v2_stdio_validates_without_side_effects_then_executes_agentic_pair() {
    let server = MockServer::spawn();
    let root = TempDir::new().unwrap();
    write_worker(&root);
    let python = python_executable();
    let artifact_target = root.path().join("runner-v2");
    let capabilities = runner_capabilities();
    let distribution_id = capabilities["distribution_id"].as_str().unwrap();
    assert!(
        capabilities["supported_pairs"]
            .as_array()
            .unwrap()
            .contains(&json!(["online_http", "agentic"]))
    );
    let mut request = json!({
        "protocol_version": 2,
        "operation": "validate",
        "expected_distribution_id": distribution_id,
        "run": {
            "identity": {"benchmark_id": "agentic-v2-stdio"},
            "artifact_target": artifact_target.clone(),
            "resources": {
                "models": {"items": [{"name": "primary-model"}]},
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "chat_completions",
                    "urls": [server.base_url.clone()],
                    "streaming": true,
                    "use_legacy_max_tokens": false,
                    "use_server_token_count": true,
                    "timeout_seconds": 30.0,
                    "connection_reuse": "pooled",
                    "download_video_content": false,
                    "extra": {},
                    "headers": {},
                    "http2": false,
                    "wait_for_model_timeout": 0.0,
                    "wait_for_model_interval": 5.0,
                    "wait_for_model_mode": "inference"
                }]}
            },
            "backend": {"type": "online_http", "config": {}},
            "workload": {"type": "agentic", "config": {
                "worker_count": 2,
                "dataset": "fixture/agentic@locked",
                "tokenizer": {"name": "builtin"},
                "phases": [{
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "concurrency": 2
                }],
                "evaluator": {
                    "python_executable": python,
                    "worker_module": "fixture_agentic.worker",
                    "environment": {
                        "PYTHONPATH": root.path(),
                        "AIPERF_EXPECT_ARTIFACT_ABSENT": artifact_target.clone()
                    }
                },
                "task_concurrency": 2,
                "environment": "fixture",
                "output_dir": "episodes",
                "max_tokens": 64,
                "context_window": 4096,
                "inference_gateway_host": "127.0.0.1"
            }}
        }
    });

    let validation_output = run_runner(&request);
    let validation = one_json_line(&validation_output.stdout);
    assert_eq!(validation_output.status.code(), Some(0));
    assert_eq!(validation["event"], "run_validation");
    assert_eq!(validation["success"], true);
    assert_eq!(validation["distribution_id"], distribution_id);
    assert!(!artifact_target.exists());
    assert!(server.captured.0.lock().unwrap().is_empty());

    request["operation"] = json!("execute");
    let execution_output = run_runner(&request);
    let terminal = one_json_line(&execution_output.stdout);
    assert_eq!(
        execution_output.status.code(),
        Some(0),
        "stderr={}",
        String::from_utf8_lossy(&execution_output.stderr)
    );
    assert_eq!(terminal["event"], "run_terminal");
    assert_eq!(terminal["success"], true);
    assert_eq!(terminal["distribution_id"], distribution_id);
    assert_eq!(terminal["provenance"]["backend"], "online_http");
    assert_eq!(terminal["provenance"]["workload"], "agentic");
    let report_path = PathBuf::from(terminal["report_path"].as_str().unwrap());
    assert_eq!(report_path, artifact_target.join("native-v2.json"));
    let report: Value = serde_json::from_slice(&std::fs::read(report_path).unwrap()).unwrap();
    assert_eq!(report["schema_version"], "2.0");
    assert_eq!(report["run"]["distribution_id"], distribution_id);
    assert_eq!(report["run"]["backend"], "online_http");
    assert_eq!(report["run"]["workload"], "agentic");
    assert_eq!(report["run"]["extensions"], json!([]));
    assert_eq!(
        report["run"]["endpoint_profiles"],
        json!([{"profile_id": "default", "endpoint_id": "chat"}])
    );
    assert_eq!(report["run"]["mode"], "agentic_accuracy");
    assert_eq!(report["agentic"]["summary"]["episode_count"], 3);
    assert_eq!(report["agentic"]["summary"]["completed_count"], 1);
    assert_eq!(
        report["agentic"]["summary"]["infrastructure_error_count"],
        1
    );
    assert_eq!(report["agentic"]["summary"]["cancelled_count"], 1);
    assert_eq!(report["agentic"]["summary"]["rewards"]["reward"]["n"], 1);
    assert_eq!(server.captured.0.lock().unwrap().len(), 5);
}
