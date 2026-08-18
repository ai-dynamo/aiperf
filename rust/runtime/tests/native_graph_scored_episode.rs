// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NativeGraph task-environment callback ordering contracts.

use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    fs,
    io::{self, Read},
    num::NonZeroUsize,
    rc::Rc,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use aiperf_runtime::eval::{
    AgentVariantRef, ArtifactDigest, AttemptId, DockerBuildRequest, DockerCopyRequest,
    DockerCreateRequest, DockerExecRequest, DockerNativeGraphEpisodeExecutor, DockerProcessSandbox,
    DockerRemoveRequest, DockerRuntime, DockerStartRequest, EngineNativeGraphEpisodeCallback,
    EpisodeAssignment, EpisodeExecutionError, EvalExecutionError, EvalNodeRecordArtifact,
    EvidenceEvent, EvidenceKind, FrozenAttemptBundle, HarborEpisodeEvaluatorFactory,
    HarborEvaluationCoordinator, HarborImporter, HarborLifecycleAgentContract,
    HarborLifecycleRequest, HarborLifecycleScoreRequest, HarborSandboxRecipe, HarborSource,
    LocalNativeGraphSuiteScheduler, ModelCapacityKey, ModelEndpointIsolationProof, ModelIdentity,
    ModelRuntimeConfig, NativeGraphCompletedAttempt, NativeGraphEpisodeCallback,
    NativeGraphEpisodeExecutor, NativeGraphEpisodeLease, NativeGraphEpisodeRunner,
    NativeGraphPackagePlan, NativeGraphSuiteManifest, NativeSourceAcquirer, PolicyIdentity,
    ProviderCapabilities, ProviderCapability, ProviderProfile, RegradeRequest,
    ResourceLeaseRequest, RewardDocument, RuntimeIdentity, ScoreVersion, SecretProvider,
    SuiteRunId, SuiteTrialSpec, TrialBudget, TrialSpec, VerifierResult, regrade,
    run_native_graph_episode_callback, run_resolved_suite,
};
use aiperf_runtime::{engine::application::Application, eval::EnvName};
use async_trait::async_trait;
use axum::{Json, Router, extract::State, routing::post};

#[tokio::test(flavor = "current_thread")]
async fn callback_runs_only_after_authorized_environment_acquisition_and_before_collection() {
    let callback_ran = Cell::new(false);
    let collection_started = Cell::new(false);
    let mut lease = RecordingLease {
        authorized: true,
        environment_acquired: true,
        _collection_started: &collection_started,
    };
    let mut callback = RecordingCallback {
        callback_ran: &callback_ran,
    };

    run_native_graph_episode_callback(&mut callback, &mut lease, &mut || {
        assert!(callback_ran.get());
        collection_started.set(true);
        Ok(())
    })
    .await
    .expect("authorized callback completes before collection begins");

    assert!(callback_ran.get());
    assert!(collection_started.get());
}

#[tokio::test(flavor = "current_thread")]
async fn callback_failure_is_returned_before_collection_can_start() {
    let collection_started = Cell::new(false);
    let mut lease = RecordingLease {
        authorized: true,
        environment_acquired: true,
        _collection_started: &collection_started,
    };
    let mut callback = FailingCallback;

    let error = run_native_graph_episode_callback(&mut callback, &mut lease, &mut || {
        collection_started.set(true);
        Ok(())
    })
    .await
    .expect_err("callback failure must prevent artifact collection and verifier execution");

    assert!(
        matches!(error, EvalExecutionError::ProcessFailure(reason) if reason == "graph failed")
    );
    assert!(!collection_started.get());
}

#[tokio::test(flavor = "current_thread")]
async fn imported_acyclic_graph_callback_emits_current_transport_observer_evidence() {
    async fn completion() -> Json<serde_json::Value> {
        Json(serde_json::json!({
            "id": "native-graph-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "completed"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        }))
    }

    let app = Router::new().route("/v1/chat/completions", post(completion));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback model endpoint");
    let address = listener.local_addr().expect("read loopback address");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve loopback model endpoint");
    });

    let imported = import_live_graph_fixture(&format!("http://{address}"));
    let native = imported
        .package
        .native_graph()
        .expect("fixture has the NativeGraph profile");
    let application = Application::stock(format!("blake3:{}", "1".repeat(64)))
        .expect("compose the stock application once");
    let runtime: ModelRuntimeConfig =
        toml::from_str("version = 1\n").expect("empty host model-secret mapping is valid");
    let mut callback =
        EngineNativeGraphEpisodeCallback::new(&application, native, &runtime, &EmptySecrets, None)
            .expect("resolve the immutable model binding through the stock application");
    let collection_started = Cell::new(false);
    let mut lease = RecordingLease {
        authorized: true,
        environment_acquired: true,
        _collection_started: &collection_started,
    };

    callback
        .run(&mut lease)
        .await
        .expect("the imported acyclic model node dispatches through EngineGraphSink");

    let evidence = callback
        .transport_evidence()
        .expect("the callback retains observer evidence after graph completion");
    assert_eq!(evidence.model_records(), 1);
    assert_eq!(evidence.completed_traces(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn imported_generation_defaults_reach_the_outbound_graph_request() {
    async fn completion(
        State(requests): State<Arc<Mutex<Vec<serde_json::Value>>>>,
        Json(request): Json<serde_json::Value>,
    ) -> Json<serde_json::Value> {
        requests
            .lock()
            .expect("test request capture lock is not poisoned")
            .push(request);
        Json(serde_json::json!({
            "id": "native-graph-generation",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "completed"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        }))
    }

    let requests = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/chat/completions", post(completion))
        .with_state(requests.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback model endpoint");
    let address = listener.local_addr().expect("read loopback address");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve loopback model endpoint");
    });

    let imported = import_live_graph_fixture(&format!("http://{address}"));
    let native = imported
        .package
        .native_graph()
        .expect("fixture has the NativeGraph profile");
    let application = Application::stock(format!("blake3:{}", "7".repeat(64)))
        .expect("compose the stock application once");
    let runtime: ModelRuntimeConfig =
        toml::from_str("version = 1\n").expect("empty host model-secret mapping is valid");
    let mut callback =
        EngineNativeGraphEpisodeCallback::new(&application, native, &runtime, &EmptySecrets, None)
            .expect("resolve the immutable model binding through the stock application");
    let collection_started = Cell::new(false);
    let mut lease = RecordingLease {
        authorized: true,
        environment_acquired: true,
        _collection_started: &collection_started,
    };

    callback
        .run(&mut lease)
        .await
        .expect("the graph callback dispatches the authored generation request");

    let requests = requests
        .lock()
        .expect("test request capture lock is not poisoned");
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request["max_completion_tokens"], serde_json::json!(17));
    assert_eq!(request["min_tokens"], serde_json::json!(3));
    assert_eq!(request["temperature"], serde_json::json!(0.37));
    assert_eq!(request["top_p"], serde_json::json!(0.73));
    assert_eq!(request["top_k"], serde_json::json!(42));
    assert_eq!(request["seed"], serde_json::json!(99));
    assert_eq!(request["presence_penalty"], serde_json::json!(0.25));
    assert_eq!(request["frequency_penalty"], serde_json::json!(-0.5));
    assert_eq!(request["repetition_penalty"], serde_json::json!(1.1));
}

#[tokio::test(flavor = "current_thread")]
async fn runner_unit_maps_explicitly_stubbed_frozen_facts_after_observed_model_dispatch() {
    async fn completion() -> Json<serde_json::Value> {
        Json(serde_json::json!({
            "id": "native-graph-matrix-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "completed"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        }))
    }

    let app = Router::new().route("/v1/chat/completions", post(completion));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback model endpoint");
    let address = listener.local_addr().expect("read loopback address");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve loopback model endpoint");
    });

    let imported = import_live_graph_fixture(&format!("http://{address}"));
    let native = imported
        .package
        .native_graph()
        .expect("fixture has NativeGraph package")
        .clone();
    let application = Application::stock(format!("blake3:{}", "2".repeat(64)))
        .expect("compose the stock application once");
    let runtime: ModelRuntimeConfig =
        toml::from_str("version = 1\n").expect("empty host model-secret mapping is valid");
    let callback =
        EngineNativeGraphEpisodeCallback::new(&application, &native, &runtime, &EmptySecrets, None)
            .expect("resolve the imported graph callback");
    let binding = native
        .model_bindings()
        .first()
        .expect("fixture has one model binding");
    let capacity_key = ModelCapacityKey::from_task_binding(&imported.task, binding);
    let resources = ResourceLeaseRequest::new(1, 64, BTreeMap::from([(capacity_key.clone(), 1)]))
        .expect("finite episode resources");
    let trial = TrialSpec::new(
        imported.task.clone(),
        AgentVariantRef::new("native-graph").expect("agent variant"),
        ModelIdentity::new("provider-default", "example-model").expect("model identity"),
        7,
        PolicyIdentity::new(ArtifactDigest::from_bytes(b"policy")),
        TrialBudget::new(30.0, 30.0).expect("trial budget"),
        ArtifactDigest::from_bytes(b"environment"),
        ArtifactDigest::from_bytes(b"verifier"),
        RuntimeIdentity::new("native").expect("runtime identity"),
    )
    .expect("trial identity");
    let suite = NativeGraphSuiteManifest::new(vec![
        SuiteTrialSpec::from_imported(
            imported,
            trial,
            NonZeroUsize::new(1).expect("one repetition"),
            resources,
        )
        .expect("suite trial"),
    ])
    .expect("one-trial suite")
    .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(b"matrix-run")))
    .expect("resolve one matrix assignment");
    let scheduler = LocalNativeGraphSuiteScheduler::new(
        aiperf_runtime::eval::ResourceLimits::new(1, 1, 64, BTreeMap::from([(capacity_key, 1)]))
            .expect("finite scheduler resources"),
    )
    .expect("scheduler");
    let executor = Rc::new(StubFrozenFactsExecutor {
        callback: RefCell::new(callback),
    });
    let runner = Rc::new(NativeGraphEpisodeRunner::new(
        executor,
        Rc::new(HarborEpisodeEvaluatorFactory),
    ));

    let results = run_resolved_suite(&scheduler, suite, runner)
        .await
        .expect("matrix runner returns one scored result");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].verified_reward(), Some(1.0));
}

#[tokio::test(flavor = "current_thread")]
async fn docker_episode_executor_appends_two_suite_episodes_to_one_record_artifact() {
    async fn completion() -> Json<serde_json::Value> {
        Json(serde_json::json!({
            "id": "native-graph-docker-e2e",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "completed"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        }))
    }

    let app = Router::new().route("/v1/chat/completions", post(completion));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback model endpoint");
    let address = listener.local_addr().expect("read loopback address");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve loopback model endpoint");
    });

    let imported = import_live_graph_fixture(&format!("http://{address}"));
    let lifecycle = native_graph_lifecycle_request();
    let trial = HarborEvaluationCoordinator::resolve_trial(&imported, &lifecycle)
        .expect("resolve lifecycle trial before environment provisioning");
    let native = imported
        .package
        .native_graph()
        .expect("fixture has NativeGraph package");
    let binding = native
        .model_bindings()
        .first()
        .expect("fixture has one model binding");
    let capacity_key = ModelCapacityKey::from_task_binding(&imported.task, binding);
    let resources = ResourceLeaseRequest::new(1, 64, BTreeMap::from([(capacity_key.clone(), 1)]))
        .expect("finite episode resources");
    let suite = NativeGraphSuiteManifest::new(vec![
        SuiteTrialSpec::from_imported(
            imported.clone(),
            trial,
            NonZeroUsize::new(2).expect("two repetitions"),
            resources,
        )
        .expect("suite trial"),
    ])
    .expect("one-trial suite")
    .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(
        b"docker-matrix-run",
    )))
    .expect("resolve one matrix assignment");
    let runtime = Rc::new(GraphDockerRuntime::default());
    let application = Rc::new(
        Application::stock(format!("blake3:{}", "5".repeat(64)))
            .expect("compose the stock application once"),
    );
    let model_runtime: ModelRuntimeConfig =
        toml::from_str("version = 1\n").expect("empty host model-secret mapping is valid");
    let output = tempfile::tempdir().expect("temporary suite output directory");
    let records_path = output.path().join("records.jsonl");
    let artifact =
        EvalNodeRecordArtifact::open(&records_path).expect("open suite-owned node record artifact");
    let executor = Rc::new(
        DockerNativeGraphEpisodeExecutor::new_with_runtime(
            DockerProcessSandbox::new(),
            runtime.clone(),
            HarborSandboxRecipe::for_standard_task(
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                None,
            )
            .expect("immutable Docker recipe"),
            imported,
            lifecycle,
            application,
            model_runtime,
            Rc::new(EmptySecrets),
            Some(artifact.clone()),
        )
        .expect("construct the concrete Docker NativeGraph executor"),
    );
    let scheduler = LocalNativeGraphSuiteScheduler::new(
        aiperf_runtime::eval::ResourceLimits::new(1, 1, 64, BTreeMap::from([(capacity_key, 1)]))
            .expect("finite scheduler resources"),
    )
    .expect("scheduler");
    let runner = Rc::new(NativeGraphEpisodeRunner::new(
        executor,
        Rc::new(HarborEpisodeEvaluatorFactory),
    ));

    let results = run_resolved_suite(&scheduler, suite, runner)
        .await
        .expect("Docker callback, declared verifier, freeze, and evaluator all complete");

    artifact.finish().expect("flush suite-owned artifact");

    assert_eq!(results.len(), 2);
    assert!(
        results
            .iter()
            .all(|result| result.verified_reward() == Some(1.0))
    );
    assert_eq!(
        runtime.events.borrow().as_slice(),
        [
            "build", "create", "start", "verifier", "remove", "build", "create", "start",
            "verifier", "remove"
        ]
    );
    let rows = fs::read_to_string(&records_path).expect("read suite node record artifact");
    assert_eq!(rows.lines().count(), 2);
}

#[cfg(target_os = "linux")]
#[tokio::test(flavor = "current_thread")]
async fn episode_fails_on_record_append_error_without_exposing_record_content() {
    const PRIVATE_OUTPUT_MARKER: &str = "private-model-output-marker";

    async fn completion() -> Json<serde_json::Value> {
        Json(serde_json::json!({
            "id": "native-graph-writer-error",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": format!("{}{}", PRIVATE_OUTPUT_MARKER, "x".repeat(32 * 1024))
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        }))
    }

    let app = Router::new().route("/v1/chat/completions", post(completion));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback model endpoint");
    let address = listener.local_addr().expect("read loopback address");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve loopback model endpoint");
    });

    let imported = import_live_graph_fixture(&format!("http://{address}"));
    let lifecycle = native_graph_lifecycle_request();
    let trial = HarborEvaluationCoordinator::resolve_trial(&imported, &lifecycle)
        .expect("resolve lifecycle trial before environment provisioning");
    let binding = imported
        .package
        .native_graph()
        .expect("fixture has NativeGraph package")
        .model_bindings()
        .first()
        .expect("fixture has one model binding");
    let capacity_key = ModelCapacityKey::from_task_binding(&imported.task, binding);
    let resources = ResourceLeaseRequest::new(1, 64, BTreeMap::from([(capacity_key.clone(), 1)]))
        .expect("finite episode resources");
    let suite = NativeGraphSuiteManifest::new(vec![
        SuiteTrialSpec::from_imported(
            imported.clone(),
            trial,
            NonZeroUsize::new(1).expect("one repetition"),
            resources,
        )
        .expect("suite trial"),
    ])
    .expect("one-trial suite")
    .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(
        b"writer-error-run",
    )))
    .expect("resolve one matrix assignment");
    let application = Rc::new(
        Application::stock(format!("blake3:{}", "a".repeat(64)))
            .expect("compose the stock application once"),
    );
    let model_runtime: ModelRuntimeConfig =
        toml::from_str("version = 1\n").expect("empty host model-secret mapping is valid");
    let artifact = EvalNodeRecordArtifact::open(std::path::Path::new("/dev/full"))
        .expect("open deterministic failing destination");
    let executor = Rc::new(
        DockerNativeGraphEpisodeExecutor::new_with_runtime(
            DockerProcessSandbox::new(),
            Rc::new(GraphDockerRuntime::default()),
            HarborSandboxRecipe::for_standard_task(
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                None,
            )
            .expect("immutable Docker recipe"),
            imported,
            lifecycle,
            application,
            model_runtime,
            Rc::new(EmptySecrets),
            Some(artifact),
        )
        .expect("construct executor with failing record artifact"),
    );
    let scheduler = LocalNativeGraphSuiteScheduler::new(
        aiperf_runtime::eval::ResourceLimits::new(1, 1, 64, BTreeMap::from([(capacity_key, 1)]))
            .expect("finite scheduler resources"),
    )
    .expect("scheduler");
    let runner = Rc::new(NativeGraphEpisodeRunner::new(
        executor,
        Rc::new(HarborEpisodeEvaluatorFactory),
    ));

    let error = run_resolved_suite(&scheduler, suite, runner)
        .await
        .expect_err("record append failure must fail the episode before scoring");
    let message = error.to_string();
    assert!(
        message.contains("full"),
        "error names the sanitized destination"
    );
    assert!(!message.contains("/dev/full"), "error omits the raw path");
    assert!(
        !message.contains(PRIVATE_OUTPUT_MARKER),
        "error omits raw model output"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn unknown_source_model_binding_fails_before_any_model_dispatch() {
    let requests = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route(
        "/v1/chat/completions",
        post({
            let requests = requests.clone();
            move || {
                let requests = requests.clone();
                async move {
                    requests.fetch_add(1, Ordering::SeqCst);
                    Json(serde_json::json!({
                        "object": "chat.completion",
                        "choices": [{"message": {"role": "assistant", "content": "unexpected"}}]
                    }))
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback model endpoint");
    let address = listener.local_addr().expect("read loopback address");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve loopback model endpoint");
    });

    let imported = import_live_graph_fixture_with_binding(&format!("http://{address}"), "unknown");
    let native = imported
        .package
        .native_graph()
        .expect("fixture has NativeGraph package");
    let application = Application::stock(format!("blake3:{}", "3".repeat(64)))
        .expect("compose the stock application once");
    let runtime: ModelRuntimeConfig =
        toml::from_str("version = 1\n").expect("empty host model-secret mapping is valid");

    let error = match EngineNativeGraphEpisodeCallback::new(
        &application,
        native,
        &runtime,
        &EmptySecrets,
        None,
    ) {
        Ok(_) => panic!("unknown model binding must reject before graph dispatch"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("undeclared binding \"unknown\""));
    assert_eq!(requests.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn two_declared_model_bindings_prepare_and_dispatch_without_default_fallback() {
    let requests = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route(
        "/v1/chat/completions",
        post({
            let requests = requests.clone();
            move || {
                let requests = requests.clone();
                async move {
                    requests.fetch_add(1, Ordering::SeqCst);
                    Json(serde_json::json!({
                        "id": "native-graph-two-binding-test",
                        "object": "chat.completion",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "completed"},
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
                    }))
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback model endpoint");
    let address = listener.local_addr().expect("read loopback address");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve loopback model endpoint");
    });

    let imported = import_two_binding_graph_fixture(&format!("http://{address}"));
    let native = imported
        .package
        .native_graph()
        .expect("fixture has NativeGraph package");
    let application = Application::stock(format!("blake3:{}", "4".repeat(64)))
        .expect("compose the stock application once");
    let runtime: ModelRuntimeConfig =
        toml::from_str("version = 1\n").expect("empty host model-secret mapping is valid");
    let mut callback =
        EngineNativeGraphEpisodeCallback::new(&application, native, &runtime, &EmptySecrets, None)
            .expect("every declared binding prepares through the frozen application");
    let collection_started = Cell::new(false);
    let mut lease = RecordingLease {
        authorized: true,
        environment_acquired: true,
        _collection_started: &collection_started,
    };

    callback
        .run(&mut lease)
        .await
        .expect("both declared bindings dispatch through their selected profiles");

    let evidence = callback
        .transport_evidence()
        .expect("the callback retains observer evidence after graph completion");
    assert_eq!(evidence.model_records(), 2);
    assert_eq!(evidence.completed_traces(), 1);
    assert_eq!(requests.load(Ordering::SeqCst), 2);
}

struct RecordingLease<'a> {
    authorized: bool,
    environment_acquired: bool,
    _collection_started: &'a Cell<bool>,
}

impl NativeGraphEpisodeLease for RecordingLease<'_> {
    fn is_authorized(&self) -> bool {
        self.authorized
    }

    fn is_environment_acquired(&self) -> bool {
        self.environment_acquired
    }

    fn instruction(&self) -> &str {
        "write a result"
    }
}

struct RecordingCallback<'a> {
    callback_ran: &'a Cell<bool>,
}

#[async_trait(?Send)]
impl NativeGraphEpisodeCallback for RecordingCallback<'_> {
    async fn run(
        &mut self,
        lease: &mut dyn NativeGraphEpisodeLease,
    ) -> Result<(), EvalExecutionError> {
        assert!(lease.is_authorized());
        assert!(lease.is_environment_acquired());
        assert_eq!(lease.instruction(), "write a result");
        self.callback_ran.set(true);
        Ok(())
    }
}

struct FailingCallback;

#[async_trait(?Send)]
impl NativeGraphEpisodeCallback for FailingCallback {
    async fn run(&mut self, _: &mut dyn NativeGraphEpisodeLease) -> Result<(), EvalExecutionError> {
        Err(EvalExecutionError::ProcessFailure(
            "graph failed".to_owned(),
        ))
    }
}

/// Test-only runner seam for matrix/evaluator mechanics. This does not exercise
/// Harbor execution and must not be used as NativeGraph end-to-end evidence.
struct StubFrozenFactsExecutor {
    callback: RefCell<EngineNativeGraphEpisodeCallback>,
}

#[async_trait(?Send)]
impl NativeGraphEpisodeExecutor for StubFrozenFactsExecutor {
    async fn execute(
        &self,
        assignment: &EpisodeAssignment,
    ) -> Result<NativeGraphCompletedAttempt, EpisodeExecutionError> {
        let collection_started = Cell::new(false);
        let mut lease = RecordingLease {
            authorized: true,
            environment_acquired: true,
            _collection_started: &collection_started,
        };
        let mut callback = self.callback.borrow_mut();
        callback
            .run(&mut lease)
            .await
            .map_err(EpisodeExecutionError::Callback)?;
        let evidence = callback
            .transport_evidence()
            .ok_or(EpisodeExecutionError::MissingTransportEvidence)?;
        if evidence.model_records() != 1 || evidence.completed_traces() != 1 {
            return Err(EpisodeExecutionError::UnexpectedTransportEvidence {
                model_records: evidence.model_records(),
                completed_traces: evidence.completed_traces(),
            });
        }
        let verifier_input = ArtifactDigest::from_bytes(b"declared artifact");
        let verifier = ArtifactDigest::from_bytes(b"verifier");
        let rationale = ArtifactDigest::from_bytes(b"rationale");
        let reward = RewardDocument::new(BTreeMap::from([("reward".to_owned(), 1.0)]))
            .map_err(|error| EpisodeExecutionError::Facts(error.to_string()))?;
        let verifier_result = VerifierResult::new(
            assignment.attempt_id().clone(),
            verifier.clone(),
            vec![verifier_input.clone()],
            reward,
            rationale.clone(),
        )
        .map_err(|error| EpisodeExecutionError::Facts(error.to_string()))?;
        let initial = ScoreVersion::initial(
            assignment.attempt_id().clone(),
            verifier,
            vec![verifier_input],
            "reward",
            1.0,
            rationale,
        )
        .map_err(|error| EpisodeExecutionError::Facts(error.to_string()))?;
        let regrade_result = regrade(
            RegradeRequest::new(initial.clone(), verifier_result.clone(), "reward")
                .map_err(|error| EpisodeExecutionError::Facts(error.to_string()))?,
        )
        .map_err(|error| EpisodeExecutionError::Facts(error.to_string()))?;
        let frozen = FrozenAttemptBundle::new(
            assignment.trial_digest().clone(),
            verifier_result,
            vec![EvidenceEvent::new(
                assignment.attempt_id().clone(),
                0,
                EvidenceKind::Llm,
                ArtifactDigest::from_bytes(b"one EngineGraphSink record"),
                None,
            )],
            vec![initial, regrade_result],
        )
        .map_err(|error| EpisodeExecutionError::Facts(error.to_string()))?;
        Ok(NativeGraphCompletedAttempt::from_frozen(frozen))
    }
}

struct EmptySecrets;

impl SecretProvider for EmptySecrets {
    fn resolve(
        &self,
        name: &EnvName,
    ) -> Result<aiperf_runtime::eval::SecretValue, EvalExecutionError> {
        Err(EvalExecutionError::MissingSecret(name.clone()))
    }
}

fn native_graph_lifecycle_request() -> HarborLifecycleRequest {
    HarborLifecycleRequest {
        version: 1,
        agent_variant: AgentVariantRef::new("native-graph").expect("native graph variant"),
        model: ModelIdentity::new("provider-default", "example-model")
            .expect("selected model identity"),
        seed: 11,
        policy: PolicyIdentity::new(ArtifactDigest::from_bytes(b"native-graph-policy")),
        runtime: RuntimeIdentity::new("native").expect("native runtime identity"),
        attempt: AttemptId::new("caller-attempt").expect("caller attempt identity"),
        budget: TrialBudget::new(30.0, 30.0).expect("finite trial budget"),
        agent_contract: HarborLifecycleAgentContract::NativeGraph,
        command: vec!["aiperf-native-graph".to_owned()],
        initial_score: HarborLifecycleScoreRequest {
            metric: "reward".to_owned(),
            rationale: ArtifactDigest::from_bytes(b"initial rationale"),
        },
        regrade: HarborLifecycleScoreRequest {
            metric: "reward".to_owned(),
            rationale: ArtifactDigest::from_bytes(b"regrade rationale"),
        },
    }
}

#[derive(Default)]
struct GraphDockerRuntime {
    events: RefCell<Vec<&'static str>>,
}

impl DockerRuntime for GraphDockerRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_users()
            .with_phase_env()
            .with_workdir()
            .with_phase_timeouts()
            .with_separate_verifier()
            .with_healthchecks()
            .with_no_network()
            .with_public_network()
            .with_model_endpoint_isolation()
    }

    fn supports_phase_network_transitions(&self) -> bool {
        true
    }

    fn native_graph_provider_profile(
        &self,
        _: &NativeGraphPackagePlan,
    ) -> Result<ProviderProfile, EvalExecutionError> {
        ProviderProfile::new(
            "test-native-graph-docker",
            vec![ProviderCapability::ModelEndpointIsolation],
        )
        .and_then(|profile| {
            profile.with_model_endpoint_isolation(ModelEndpointIsolationProof::NoAdapterEgress)
        })
        .map_err(|error| {
            EvalExecutionError::UnsupportedEnforcement(match error {
                _ => "test native graph provider profile",
            })
        })
    }

    fn native_graph_model_secret_environment(
        &self,
        _: &NativeGraphPackagePlan,
    ) -> Result<BTreeMap<aiperf_runtime::eval::ModelSecretId, EnvName>, EvalExecutionError> {
        Ok(BTreeMap::new())
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("build");
        Ok(())
    }

    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("create");
        Ok(())
    }

    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start");
        Ok(())
    }

    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        if request.phase().to_string() == "verifier"
            && request
                .public_arguments()
                .iter()
                .any(|argument| argument == "/tests/test.sh")
        {
            self.events.borrow_mut().push("verifier");
        }
        Ok(())
    }

    fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        Ok(())
    }

    fn container_workdir(&self, _: &str) -> Result<String, EvalExecutionError> {
        Ok("/work".to_owned())
    }

    fn copy_archive(&self, _: &str, source: &str) -> Result<Box<dyn Read>, EvalExecutionError> {
        if source.ends_with("reward.json") {
            return Err(EvalExecutionError::ProcessFailure(
                "reward.json is absent in the fixture".to_owned(),
            ));
        }
        let entry = if source.ends_with("result.txt") {
            "result.txt"
        } else {
            "reward.txt"
        };
        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_size(2);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, entry, io::Cursor::new(b"1\n"))
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        let bytes = builder
            .into_inner()
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        Ok(Box::new(io::Cursor::new(bytes)))
    }

    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("remove");
        Ok(())
    }
}

fn import_live_graph_fixture(endpoint: &str) -> aiperf_runtime::eval::ImportedTask {
    import_live_graph_fixture_with_binding(endpoint, "primary")
}

fn import_live_graph_fixture_with_binding(
    endpoint: &str,
    binding: &str,
) -> aiperf_runtime::eval::ImportedTask {
    import_graph_fixture(endpoint, binding, false)
}

fn import_two_binding_graph_fixture(endpoint: &str) -> aiperf_runtime::eval::ImportedTask {
    import_graph_fixture(endpoint, "primary", true)
}

fn import_graph_fixture(
    endpoint: &str,
    binding: &str,
    has_secondary_binding: bool,
) -> aiperf_runtime::eval::ImportedTask {
    let task = tempfile::tempdir().expect("temporary NativeGraph task root");
    fs::create_dir_all(task.path().join("environment")).expect("task environment directory");
    fs::create_dir_all(task.path().join("tests")).expect("task verifier directory");
    fs::write(
        task.path().join("environment/Dockerfile"),
        b"FROM scratch\n",
    )
    .expect("task Dockerfile");
    fs::write(task.path().join("instruction.md"), b"Complete the graph.\n")
        .expect("task instruction");
    fs::write(task.path().join("tests/test.sh"), b"exit 0\n").expect("task verifier");
    fs::write(
        task.path().join("task.toml"),
        r#"schema_version = "1.1"
artifacts = ["/work/result.txt"]

[task]
name = "example/native-graph-live-episode"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#,
    )
    .expect("task manifest");
    let graph = if has_secondary_binding {
        r#"{
  "schema_version": "1.0", "trace_id": "live-episode-two-bindings", "stage_bound": 1,
  "channels": {
    "first": { "type": "messages", "reducer": "add_messages" },
    "second": { "type": "messages", "reducer": "add_messages" }
  },
  "nodes": [
    { "id": "primary-model", "kind": "model", "binding": "primary", "output": "first", "streaming": false },
    { "id": "secondary-model", "kind": "model", "binding": "secondary", "inputs": ["first"], "output": "second", "streaming": false }
  ],
  "edges": [
    { "source": "START", "target": "primary-model" },
    { "source": "primary-model", "target": "secondary-model" },
    { "source": "secondary-model", "target": "END" }
  ],
  "terminal_outputs": []
}"#
        .to_owned()
    } else {
        format!(
            r#"{{
  "schema_version": "1.0", "trace_id": "live-episode", "stage_bound": 1,
  "channels": {{ "output": {{ "type": "messages", "reducer": "add_messages" }} }},
  "nodes": [{{ "id": "model", "kind": "model", "binding": "{binding}", "output": "output", "streaming": false }}],
  "edges": [{{ "source": "START", "target": "model" }}, {{ "source": "model", "target": "END" }}],
  "terminal_outputs": []
}}"#
        )
    };
    fs::write(task.path().join("agent_graph.json"), graph).expect("acyclic graph source");
    let model_bindings = if has_secondary_binding {
        format!(
            r#"[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-primary"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "example-primary"
urls = ["{endpoint}"]
streaming = false
request_timeout_ms = 30000
capture = "metadata"

[model_bindings.tokenizer]
type = "local"
name = "builtin"
revision = "main"
apply_chat_template = false

[model_bindings.generation]

[[model_bindings]]
id = "secondary"
endpoint_profile_id = "provider-secondary"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "example-secondary"
urls = ["{endpoint}"]
streaming = false
request_timeout_ms = 30000
capture = "metadata"

[model_bindings.tokenizer]
type = "local"
name = "builtin"
revision = "main"
apply_chat_template = false

[model_bindings.generation]
temperature = 0.37
top_p = 0.73
top_k = 42
max_tokens = 17
min_tokens = 3
seed = 99
presence_penalty = 0.25
frequency_penalty = -0.5
repetition_penalty = 1.1
"#
        )
    } else {
        format!(
            r#"[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-default"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "example-model"
urls = ["{endpoint}"]
streaming = false
request_timeout_ms = 30000
capture = "metadata"

[model_bindings.tokenizer]
type = "local"
name = "builtin"
revision = "main"
apply_chat_template = false

[model_bindings.generation]
temperature = 0.37
top_p = 0.73
top_k = 42
max_tokens = 17
min_tokens = 3
seed = 99
presence_penalty = 0.25
frequency_penalty = -0.5
repetition_penalty = 1.1
"#
        )
    };
    fs::write(task.path().join("models.toml"), model_bindings).expect("model binding manifest");
    fs::write(task.path().join("adapters.toml"), b"").expect("empty adapter manifest");
    let source = HarborSource::local(task.path().to_string_lossy())
        .expect("temporary task path is a valid source");
    HarborImporter::new(&NativeSourceAcquirer)
        .import(&source)
        .expect("live NativeGraph fixture imports")
}
