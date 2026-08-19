// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NativeGraph task-environment callback ordering contracts.

use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, VecDeque},
    fs,
    future::Future,
    io::{self, Read},
    num::NonZeroUsize,
    pin::Pin,
    rc::Rc,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use aiperf_runtime::eval::{
    AdapterEnvelope, AdapterExit, AdapterMessage, AdapterProcess, AdapterSpawnRequest,
    AdapterSpawnTransaction, AdapterSpawner, AdapterSupervisionError, AgentVariantRef,
    ArtifactDigest, ArtifactDownloadHandle, AttemptId, AuthorizedExternalDriverSpawn, CancelReason,
    CompatibilityFidelity, CompatibilityTerminalReceipt, DockerAdapterSpawnerRequest,
    DockerBuildRequest, DockerCopyRequest, DockerCreateRequest, DockerExecRequest,
    DockerExternallyDrivenEpisodeExecutor, DockerNativeGraphEpisodeExecutor, DockerProcessSandbox,
    DockerRemoveRequest, DockerRuntime, DockerStartRequest, EngineNativeGraphEpisodeCallback,
    EpisodeAssignment, EpisodeExecutionError, EvalExecutionError, EvalNodeRecordArtifact,
    EvidenceEvent, EvidenceKind, ExternalDriverDockerSpawner, ExternalDriverError,
    ExternalDriverSession, ExternalDriverSpawnExecutor, FrozenArtifactReference,
    FrozenAttemptBundle, HarborEpisodeEvaluatorFactory, HarborEvaluationCoordinator,
    HarborImporter, HarborLifecycleAgentContract, HarborLifecycleRequest,
    HarborLifecycleScoreRequest, HarborSandboxRecipe, HarborSource, HostEnvelope, HostMessage,
    LocalNativeGraphSuiteScheduler, ModelCapacityKey, ModelEndpointIsolationProof, ModelIdentity,
    ModelRuntimeConfig, NativeGraphAttemptAuthority, NativeGraphCompletedAttempt,
    NativeGraphEnvironmentAdapterStart, NativeGraphEpisodeBackendLease, NativeGraphEpisodeCallback,
    NativeGraphEpisodeExecutor, NativeGraphEpisodeLease, NativeGraphEpisodeRunner,
    NativeGraphExternalDriverFactory, NativeGraphLeaseRolloutStart, NativeGraphPackagePlan,
    NativeGraphSuiteManifest, NativeSourceAcquirer, PolicyIdentity, PreparedExternalDriver,
    ProtocolCapability, ProviderCapabilities, ProviderCapability, ProviderProfile, RegradeRequest,
    ResourceLeaseRequest, RewardDocument, RuntimeIdentity, ScoreVersion, SecretProvider,
    SuiteRunId, SuiteTrialSpec, TrialBudget, TrialSpec, VerifierResult,
    bind_native_graph_environment_stepper, regrade, run_native_graph_episode_callback,
    run_resolved_suite,
};
use aiperf_runtime::{engine::application::Application, eval::EnvName};
use async_trait::async_trait;
use axum::{Json, Router, extract::State, http::header, routing::post};
use base64::{Engine as _, engine::general_purpose::STANDARD};

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
async fn callback_failure_reaps_a_resource_owning_non_docker_lease() {
    let collection_started = Cell::new(false);
    let adapter_started = Cell::new(false);
    let adapter_reaped = Cell::new(false);
    let mut lease = ResourceOwningLease {
        adapter: ResourceOwningAdapterStart {
            started: &adapter_started,
        },
        reaped: &adapter_reaped,
    };
    let mut callback = StartsAdapterThenFails;

    let error = run_native_graph_episode_callback(&mut callback, &mut lease, &mut || {
        collection_started.set(true);
        Ok(())
    })
    .await
    .expect_err("a failed callback must reap an adapter owned by every backend lease");

    assert!(matches!(
        error,
        EvalExecutionError::ProcessFailure(reason) if reason == "resource-owning callback failed"
    ));
    assert!(
        adapter_started.get(),
        "the callback started the owned adapter"
    );
    assert!(
        adapter_reaped.get(),
        "the backend reaped the started adapter"
    );
    assert!(
        !collection_started.get(),
        "callback failure must not begin collection before cleanup"
    );
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
    assert!(!format!("{results:?}").contains("terminal payload"));
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

#[tokio::test(flavor = "current_thread")]
async fn docker_external_episode_preserves_harbor_score_and_compatibility_fidelity() {
    let imported = import_external_driver_fixture();
    let lifecycle = external_driver_lifecycle_request();
    let trial = HarborEvaluationCoordinator::resolve_trial(&imported, &lifecycle)
        .expect("resolve external lifecycle trial before environment provisioning");
    let suite = NativeGraphSuiteManifest::new(vec![
        SuiteTrialSpec::from_imported(
            imported.clone(),
            trial,
            NonZeroUsize::new(1).expect("one external episode"),
            ResourceLeaseRequest::new(1, 64, BTreeMap::new())
                .expect("external episode needs no model capacity"),
        )
        .expect("external suite trial"),
    ])
    .expect("one external suite trial")
    .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(
        b"external-driver-scored-run",
    )))
    .expect("resolve one external matrix assignment");
    let prepared = ScoredExternalDriverFactory
        .prepare(
            &imported.package,
            suite
                .trials()
                .first()
                .expect("resolved suite retains its one external trial"),
        )
        .expect("prepare the exact external trial before provisioning");
    let events = Rc::new(RefCell::new(Vec::new()));
    let runtime = Rc::new(GraphDockerRuntime {
        events: Rc::clone(&events),
        external_driver_spawn_executor: Some(Rc::new(ScoredExternalDriverSpawnExecutor {
            events: Rc::clone(&events),
        })),
        ..GraphDockerRuntime::default()
    });
    let executor = Rc::new(
        DockerExternallyDrivenEpisodeExecutor::new_with_runtime(
            DockerProcessSandbox::new(),
            runtime,
            HarborSandboxRecipe::for_standard_task(
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                Some("/work".to_owned()),
            )
            .expect("immutable external Docker recipe"),
            imported,
            lifecycle,
            prepared,
            Rc::new(EmptySecrets),
        )
        .expect("construct the consuming external episode executor"),
    );
    let runner = Rc::new(NativeGraphEpisodeRunner::new(
        executor,
        Rc::new(HarborEpisodeEvaluatorFactory),
    ));
    let scheduler = LocalNativeGraphSuiteScheduler::new(
        aiperf_runtime::eval::ResourceLimits::new(1, 1, 64, BTreeMap::new())
            .expect("finite no-model scheduler resources"),
    )
    .expect("external scheduler");

    let consumed_suite = suite.clone();
    let results = run_resolved_suite(&scheduler, suite, runner.clone())
        .await
        .expect("external terminal, Harbor verifier, freeze, and evaluator all complete");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].verified_reward(), Some(1.0));
    assert_eq!(
        results[0].fidelity(),
        aiperf_runtime::eval::EpisodeFidelity::ExternallyDriven(CompatibilityFidelity::Missing)
    );
    assert_eq!(
        events.borrow().as_slice(),
        [
            "build",
            "create",
            "start",
            "driver-spawn",
            "terminal",
            "cancel",
            "reap",
            "create",
            "start",
            "verifier",
            "remove",
            "remove",
        ]
    );
    let completed_events = events.borrow().clone();
    let error = run_resolved_suite(&scheduler, consumed_suite, runner)
        .await
        .expect_err("the sealed external capability cannot be reconstructed after consumption");
    assert!(
        error
            .to_string()
            .contains("external Driver preparation was already consumed")
    );
    assert_eq!(events.borrow().as_slice(), completed_events.as_slice());
}

#[tokio::test(flavor = "current_thread")]
async fn docker_rollout_only_episode_appends_bounded_policy_lifecycle_evidence() {
    let execution = tokio::task::LocalSet::new()
        .run_until(execute_rollout_only_episode())
        .await;
    let completed = execution.completed;

    assert_eq!(
        execution.model_calls, 1,
        "one reset observation drives one selected model call"
    );
    let rollout = completed
        .rollout()
        .expect("the rollout evidence is retained");
    let policy_calls = rollout
        .live_policy_calls()
        .expect("a live rollout retains bounded policy timing facts");
    assert_eq!(policy_calls.call_count(), 1);
    assert_eq!(policy_calls.first_token_count(), 1);
    assert!(policy_calls.total_first_token_ns() >= policy_calls.max_first_token_ns());
    let lifecycle = completed.frozen_attempt().lifecycle_evidence();
    assert!(
        lifecycle.len() >= 2,
        "the ordinary Harbor lifecycle precedes rollout evidence"
    );
    assert_eq!(
        lifecycle[lifecycle.len() - 2..]
            .iter()
            .map(|event| event.kind.clone())
            .collect::<Vec<_>>(),
        vec![EvidenceKind::Artifact, EvidenceKind::Llm],
        "the live call is represented only by one following lifecycle digest"
    );
    let policy_digest = &lifecycle
        .last()
        .expect("the policy lifecycle event exists")
        .payload;
    assert!(
        !completed
            .frozen_attempt()
            .verifier_result()
            .evidence
            .contains(policy_digest),
        "live policy facts do not become verifier evidence"
    );
    assert_eq!(
        completed
            .frozen_attempt()
            .verifier_result()
            .reward
            .metrics
            .get("reward"),
        Some(&1.0)
    );

    let labels = execution.runtime.adapter_spawner_labels.borrow();
    let labels = labels
        .first()
        .expect("the trusted rollout adapter is bound to one ownership-labelled container");
    let creates = execution.runtime.creates.borrow();
    let task_create = creates
        .first()
        .expect("the trusted rollout provisions its task container");
    let adapter_create = creates
        .iter()
        .find(|create| {
            create
                .windows(2)
                .any(|arguments| arguments == ["--network", "none"])
        })
        .expect("the rollout adapter runs in one no-network sidecar");
    let mounted_workspace = |create: &[String]| {
        create
            .windows(2)
            .find_map(|arguments| (arguments[0] == "--volume").then(|| arguments[1].clone()))
            .expect("container has one mounted workspace")
    };
    assert_ne!(
        mounted_workspace(task_create),
        mounted_workspace(adapter_create),
        "the adapter cannot write into the verifier workspace"
    );
    assert!(
        !adapter_create
            .iter()
            .any(|argument| argument.starts_with("AIPERF_EVAL_INSTRUCTION=")),
        "the adapter sidecar receives no task instruction environment"
    );
    for (name, value) in labels {
        assert!(
            adapter_create
                .windows(2)
                .any(|arguments| arguments == ["--label", &format!("{name}={value}")]),
            "the adapter sidecar carries the adapter's exact ownership label"
        );
    }

    let timeline = execution.runtime.events.borrow();
    let position = |event: &str| {
        timeline
            .iter()
            .position(|actual| *actual == event)
            .expect("trusted rollout lifecycle event")
    };
    assert!(position("client-cancel") < position("client-reap"));
    assert!(position("client-reap") < position("verifier"));
    assert!(position("verifier") < position("remove"));
    assert_eq!(
        timeline
            .iter()
            .filter(|event| **event == "client-cancel")
            .count(),
        1,
        "the lease cancels its trusted child exactly once"
    );
    assert_eq!(
        timeline
            .iter()
            .filter(|event| **event == "client-reap")
            .count(),
        1,
        "the lease reaps its trusted child exactly once"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn failed_workspace_patch_taints_the_live_session_before_any_followup_operation() {
    let model_calls = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route(
        "/v1/chat/completions",
        post({
            let model_calls = Arc::clone(&model_calls);
            move || {
                let model_calls = Arc::clone(&model_calls);
                async move {
                    model_calls.fetch_add(1, Ordering::SeqCst);
                    let frame = serde_json::json!({
                        "id": "tainted-workspace-policy", "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": r#"{"kind":"move","direction":"north"}"#},
                            "finish_reason": null
                        }]
                    });
                    (
                        [(header::CONTENT_TYPE, "text/event-stream")],
                        format!("data: {frame}\n\ndata: [DONE]\n\n"),
                    )
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind tainted-workspace policy endpoint");
    let address = listener
        .local_addr()
        .expect("read tainted-workspace policy address");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve tainted-workspace policy endpoint");
    });

    let imported = import_rollout_only_graph_fixture(&format!("http://{address}"));
    let lifecycle = native_graph_lifecycle_request();
    let trial = HarborEvaluationCoordinator::resolve_trial(&imported, &lifecycle)
        .expect("resolve tainted-workspace trial");
    let binding = imported
        .package
        .native_graph()
        .expect("fixture has the NativeGraph package")
        .model_bindings()
        .first()
        .expect("fixture has the selected policy binding");
    let capacity_key = ModelCapacityKey::from_task_binding(&imported.task, binding);
    let suite = NativeGraphSuiteManifest::new(vec![
        SuiteTrialSpec::from_imported(
            imported.clone(),
            trial,
            NonZeroUsize::new(1).expect("one tainted-workspace repetition"),
            ResourceLeaseRequest::new(1, 64, BTreeMap::from([(capacity_key, 1)]))
                .expect("finite tainted-workspace resources"),
        )
        .expect("tainted-workspace suite trial"),
    ])
    .expect("one tainted-workspace suite trial")
    .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(
        b"tainted-workspace-suite",
    )))
    .expect("resolve tainted-workspace assignment");
    let resolved_trial = suite
        .trials()
        .first()
        .expect("fixture has one tainted-workspace resolved trial");
    let application = Application::stock(format!("blake3:{}", "c".repeat(64)))
        .expect("compose stock application");
    let native = imported
        .package
        .native_graph()
        .expect("fixture retains its NativeGraph package");
    let runtime: ModelRuntimeConfig =
        toml::from_str("version = 1\n").expect("empty model runtime configuration");
    let mut engine_callback =
        EngineNativeGraphEpisodeCallback::new(&application, native, &runtime, &EmptySecrets, None)
            .expect("prepare the exact selected policy runtime");
    let bound =
        bind_native_graph_environment_stepper(application.product_registry(), resolved_trial)
            .expect("bind the exact selected environment stepper");
    engine_callback
        .bind_live_rollout(
            bound,
            NativeGraphAttemptAuthority::from_resolved_trial(resolved_trial),
            native
                .rollout()
                .expect("fixture retains its rollout plan")
                .workspace_patch()
                .clone(),
        )
        .expect("bind the sealed rollout start before Docker provisioning");
    let observations = Rc::new(RefCell::new(Vec::new()));
    let mut callback = TaintedWorkspacePatchCallback {
        rollout_start: Some(
            engine_callback
                .take_lease_rollout_start()
                .expect("callback transfers one sealed rollout start"),
        ),
        observations: Rc::clone(&observations),
    };
    let runtime_events = Rc::new(RefCell::new(Vec::new()));
    let runtime = Rc::new(GraphDockerRuntime {
        events: Rc::clone(&runtime_events),
        adapter_spawner: Some(Rc::new(RolloutAdapterSpawner {
            events: Rc::clone(&runtime_events),
            workspace_patch: b"not a tar archive".to_vec(),
        })),
        ..GraphDockerRuntime::default()
    });

    let error = tokio::task::LocalSet::new()
        .run_until(async {
            DockerProcessSandbox::new()
                .execute_native_graph_with_runtime(
                    runtime.as_ref(),
                    &HarborSandboxRecipe::for_standard_task(
                        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        Some("/work".to_owned()),
                    )
                    .expect("immutable tainted-workspace Docker recipe"),
                    &imported.package,
                    imported.package.execution_plan(),
                    &EmptySecrets,
                    &mut callback,
                )
                .await
                .expect_err("a rejected workspace patch must fail the live rollout")
        })
        .await;

    assert!(
        error.to_string().contains("tainted"),
        "the primary patch failure seals the session: {error}"
    );
    assert_eq!(
        observations.borrow().len(),
        4,
        "the callback observed every session boundary"
    );
    assert_eq!(observations.borrow()[0], "step");
    assert!(observations.borrow()[1].starts_with("reset: "));
    assert!(observations.borrow()[2].starts_with("step: "));
    assert!(observations.borrow()[3].starts_with("freeze: "));
    assert_eq!(
        model_calls.load(Ordering::SeqCst),
        1,
        "a tainted session must reject before another policy request"
    );
    for observation in observations.borrow().iter().skip(1) {
        assert!(
            observation.contains("tainted"),
            "every follow-up live-session operation must refuse after a failed patch: {observation}"
        );
    }
    let events = runtime.events.borrow();
    assert_eq!(
        events
            .iter()
            .filter(|event| **event == "client-cancel")
            .count(),
        1,
        "the failed patch cancels the isolated adapter exactly once"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| **event == "client-reap")
            .count(),
        1,
        "the failed patch reaps the isolated adapter exactly once"
    );
    assert!(
        !events.contains(&"verifier"),
        "a tainted workspace cannot reach artifact collection or verification"
    );
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

impl NativeGraphEpisodeBackendLease for RecordingLease<'_> {
    fn reap_environment_adapter<'lease>(
        &'lease mut self,
    ) -> Pin<Box<dyn Future<Output = Result<(), EvalExecutionError>> + 'lease>> {
        Box::pin(async { Ok(()) })
    }
}

struct ResourceOwningLease<'a> {
    adapter: ResourceOwningAdapterStart<'a>,
    reaped: &'a Cell<bool>,
}

impl NativeGraphEpisodeLease for ResourceOwningLease<'_> {
    fn is_authorized(&self) -> bool {
        true
    }

    fn is_environment_acquired(&self) -> bool {
        true
    }

    fn instruction(&self) -> &str {
        "resource-owning lease"
    }

    fn environment_adapter_start(
        &mut self,
    ) -> Result<&mut dyn NativeGraphEnvironmentAdapterStart, EvalExecutionError> {
        Ok(&mut self.adapter)
    }
}

impl NativeGraphEpisodeBackendLease for ResourceOwningLease<'_> {
    fn reap_environment_adapter<'lease>(
        &'lease mut self,
    ) -> Pin<Box<dyn Future<Output = Result<(), EvalExecutionError>> + 'lease>> {
        let reaped = self.reaped;
        Box::pin(async move {
            reaped.set(true);
            Ok(())
        })
    }
}

struct ResourceOwningAdapterStart<'a> {
    started: &'a Cell<bool>,
}

#[async_trait(?Send)]
impl NativeGraphEnvironmentAdapterStart for ResourceOwningAdapterStart<'_> {
    async fn start(&mut self) -> Result<(), EvalExecutionError> {
        self.started.set(true);
        Ok(())
    }
}

struct StartsAdapterThenFails;

#[async_trait(?Send)]
impl NativeGraphEpisodeCallback for StartsAdapterThenFails {
    async fn run(
        &mut self,
        lease: &mut dyn NativeGraphEpisodeLease,
    ) -> Result<(), EvalExecutionError> {
        lease.environment_adapter_start()?.start().await?;
        Err(EvalExecutionError::ProcessFailure(
            "resource-owning callback failed".to_owned(),
        ))
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
                live_policy_calls: evidence.live_policy_calls(),
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

fn external_driver_lifecycle_request() -> HarborLifecycleRequest {
    HarborLifecycleRequest {
        version: 1,
        agent_variant: AgentVariantRef::new("external-driver").expect("external driver variant"),
        model: ModelIdentity::new("compatibility", "opaque-driver")
            .expect("external compatibility identity"),
        seed: 17,
        policy: PolicyIdentity::new(ArtifactDigest::from_bytes(b"external-driver-policy")),
        runtime: RuntimeIdentity::new("external-compatibility").expect("external runtime identity"),
        attempt: AttemptId::new("caller-attempt").expect("caller attempt identity"),
        budget: TrialBudget::new(30.0, 30.0).expect("finite trial budget"),
        agent_contract: HarborLifecycleAgentContract::ExternallyDriven,
        command: vec!["tools/driver.sh".to_owned()],
        initial_score: HarborLifecycleScoreRequest {
            metric: "reward".to_owned(),
            rationale: ArtifactDigest::from_bytes(b"initial external rationale"),
        },
        regrade: HarborLifecycleScoreRequest {
            metric: "reward".to_owned(),
            rationale: ArtifactDigest::from_bytes(b"external regrade rationale"),
        },
    }
}

struct ScoredExternalDriverFactory;

impl NativeGraphExternalDriverFactory for ScoredExternalDriverFactory {
    fn id(&self) -> &str {
        "scored-fixture"
    }

    fn prepare_driver(
        &self,
        _: &NativeGraphPackagePlan,
        _: &aiperf_runtime::eval::ResolvedEpisodeTrial,
    ) -> Result<Box<dyn PreparedExternalDriver>, ExternalDriverError> {
        Ok(Box::new(ScoredPreparedExternalDriver))
    }
}

struct ScoredPreparedExternalDriver;

#[async_trait(?Send)]
impl PreparedExternalDriver for ScoredPreparedExternalDriver {
    async fn run(
        &mut self,
        session: &mut dyn ExternalDriverSession,
    ) -> Result<CompatibilityTerminalReceipt, ExternalDriverError> {
        session.request_terminal().await
    }
}

struct ScoredExternalDriverSpawnExecutor {
    events: Rc<RefCell<Vec<&'static str>>>,
}

impl ExternalDriverSpawnExecutor for ScoredExternalDriverSpawnExecutor {
    fn begin_spawn(
        &self,
        _: AuthorizedExternalDriverSpawn,
    ) -> Result<Box<dyn AdapterSpawnTransaction>, AdapterSupervisionError> {
        self.events.borrow_mut().push("driver-spawn");
        Ok(Box::new(ScoredExternalDriverSpawnTransaction {
            process: Some(Box::new(ScoredExternalDriverProcess {
                events: Rc::clone(&self.events),
                stdout: VecDeque::new(),
            })),
        }))
    }
}

struct ScoredExternalDriverSpawnTransaction {
    process: Option<Box<dyn AdapterProcess>>,
}

#[async_trait(?Send)]
impl AdapterSpawnTransaction for ScoredExternalDriverSpawnTransaction {
    async fn await_process(&mut self) -> Result<Box<dyn AdapterProcess>, AdapterSupervisionError> {
        self.process
            .take()
            .ok_or(AdapterSupervisionError::AlreadyReaped)
    }

    async fn abort(
        &mut self,
        deadline: std::time::Duration,
    ) -> Result<(), AdapterSupervisionError> {
        let Some(mut process) = self.process.take() else {
            return Ok(());
        };
        process.cancel(CancelReason::HostShutdown, deadline).await?;
        process.reap(deadline).await?;
        Ok(())
    }

    fn fence(&mut self) {
        if let Some(process) = self.process.as_deref_mut() {
            process.fence();
        }
    }
}

struct ScoredExternalDriverProcess {
    events: Rc<RefCell<Vec<&'static str>>>,
    stdout: VecDeque<Vec<u8>>,
}

impl ScoredExternalDriverProcess {
    fn push(&mut self, envelope: AdapterEnvelope) -> Result<(), AdapterSupervisionError> {
        let mut frame = serde_json::to_vec(&envelope).map_err(|error| {
            AdapterSupervisionError::Process(format!(
                "scored external Driver frame is invalid: {error}"
            ))
        })?;
        frame.push(b'\n');
        self.stdout.push_back(frame);
        Ok(())
    }
}

#[async_trait(?Send)]
impl AdapterProcess for ScoredExternalDriverProcess {
    async fn write_frame(
        &mut self,
        frame: &[u8],
        _: std::time::Duration,
    ) -> Result<(), AdapterSupervisionError> {
        let host: HostEnvelope = serde_json::from_slice(frame).map_err(|error| {
            AdapterSupervisionError::Process(format!(
                "scored external Driver host frame is invalid: {error}"
            ))
        })?;
        match host.message {
            HostMessage::Hello { .. } => self.push(AdapterEnvelope::new(
                host.episode,
                "startup",
                0,
                host.operation,
                AdapterMessage::Ready {
                    protocol_version: 1,
                    capabilities: vec![ProtocolCapability::Driver],
                    implementation_digest: ArtifactDigest::from_bytes(b"scored-external-driver"),
                },
            )),
            HostMessage::RequestEpisodeTerminal { .. } => {
                self.events.borrow_mut().push("terminal");
                self.push(AdapterEnvelope::new(
                    host.episode,
                    "external-driver-terminal",
                    1,
                    host.operation,
                    AdapterMessage::EpisodeTerminalCandidate {
                        output: serde_json::json!({"discarded": "terminal payload"}),
                    },
                ))
            }
            _ => Err(AdapterSupervisionError::InvalidResetTransition),
        }
    }

    async fn read_stdout_frame(
        &mut self,
        _: usize,
        _: std::time::Duration,
    ) -> Result<Vec<u8>, AdapterSupervisionError> {
        self.stdout.pop_front().ok_or_else(|| {
            AdapterSupervisionError::Process("scored external Driver returned no frame".to_owned())
        })
    }

    async fn drain_stderr(&mut self, _: usize) -> Result<Vec<u8>, AdapterSupervisionError> {
        Ok(Vec::new())
    }

    async fn cancel(
        &mut self,
        _: CancelReason,
        _: std::time::Duration,
    ) -> Result<(), AdapterSupervisionError> {
        self.events.borrow_mut().push("cancel");
        Ok(())
    }

    async fn reap(
        &mut self,
        _: std::time::Duration,
    ) -> Result<AdapterExit, AdapterSupervisionError> {
        self.events.borrow_mut().push("reap");
        Ok(AdapterExit::Reaped)
    }

    fn fence(&mut self) {
        self.events.borrow_mut().push("fence");
    }
}

#[derive(Default)]
struct GraphDockerRuntime {
    events: Rc<RefCell<Vec<&'static str>>>,
    adapter_spawner_labels: Rc<RefCell<Vec<BTreeMap<String, String>>>>,
    creates: Rc<RefCell<Vec<Vec<String>>>>,
    adapter_spawner: Option<Rc<dyn AdapterSpawner>>,
    external_driver_spawn_executor: Option<Rc<dyn ExternalDriverSpawnExecutor>>,
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

    fn adapter_spawner(
        &self,
        request: &DockerAdapterSpawnerRequest,
        _: &aiperf_runtime::eval::NativeGraphAdapterAuthorization,
    ) -> Result<Rc<dyn AdapterSpawner>, EvalExecutionError> {
        self.adapter_spawner_labels
            .borrow_mut()
            .push(request.project().ownership_labels());
        self.adapter_spawner
            .clone()
            .ok_or(EvalExecutionError::UnsupportedEnforcement(
                "streaming Docker adapter spawn",
            ))
    }

    fn external_driver_spawner(
        &self,
        request: &DockerAdapterSpawnerRequest,
    ) -> Result<ExternalDriverDockerSpawner, EvalExecutionError> {
        let executor = self.external_driver_spawn_executor.clone().ok_or(
            EvalExecutionError::UnsupportedEnforcement("external Driver Docker adapter spawn"),
        )?;
        Ok(ExternalDriverDockerSpawner::new(request, executor))
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("build");
        Ok(())
    }

    fn create(&self, request: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("create");
        self.creates
            .borrow_mut()
            .push(request.public_arguments().to_vec());
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

struct RolloutAdapterSpawner {
    events: Rc<RefCell<Vec<&'static str>>>,
    workspace_patch: Vec<u8>,
}

impl AdapterSpawner for RolloutAdapterSpawner {
    fn begin_spawn(
        &self,
        _: AdapterSpawnRequest,
    ) -> Result<Box<dyn AdapterSpawnTransaction>, AdapterSupervisionError> {
        Ok(Box::new(RolloutAdapterTransaction {
            process: Some(Box::new(RolloutAdapterChild {
                events: Rc::clone(&self.events),
                workspace_patch: self.workspace_patch.clone(),
                ..RolloutAdapterChild::default()
            })),
        }))
    }
}

struct RolloutAdapterTransaction {
    process: Option<Box<dyn AdapterProcess>>,
}

#[async_trait(?Send)]
impl AdapterSpawnTransaction for RolloutAdapterTransaction {
    async fn await_process(&mut self) -> Result<Box<dyn AdapterProcess>, AdapterSupervisionError> {
        self.process
            .take()
            .ok_or(AdapterSupervisionError::AlreadyReaped)
    }

    async fn abort(&mut self, _: std::time::Duration) -> Result<(), AdapterSupervisionError> {
        self.process.take();
        Ok(())
    }

    fn fence(&mut self) {}
}

#[derive(Default)]
struct RolloutAdapterChild {
    events: Rc<RefCell<Vec<&'static str>>>,
    stdout: VecDeque<Vec<u8>>,
    sequence: u64,
    parent: Option<String>,
    pending_operation: Option<String>,
    pending_bytes: Option<Vec<u8>>,
    pending_read_parent: Option<String>,
    pending_download: Option<ArtifactDownloadHandle>,
    outputs: VecDeque<Vec<u8>>,
    workspace_patch: Vec<u8>,
    references: Vec<FrozenArtifactReference>,
    is_reset_complete: bool,
}

#[async_trait(?Send)]
impl AdapterProcess for RolloutAdapterChild {
    async fn write_frame(
        &mut self,
        frame: &[u8],
        _: std::time::Duration,
    ) -> Result<(), AdapterSupervisionError> {
        let host: HostEnvelope = serde_json::from_slice(frame).map_err(|error| {
            AdapterSupervisionError::Process(format!("fixture host frame is invalid: {error}"))
        })?;
        match host.message {
            HostMessage::Hello { .. } => self.push(
                host.episode,
                "startup",
                host.operation,
                AdapterMessage::Ready {
                    protocol_version: 1,
                    capabilities: vec![
                        ProtocolCapability::Environment,
                        ProtocolCapability::Artifacts,
                    ],
                    implementation_digest: ArtifactDigest::from_bytes(b"rollout-only-child"),
                },
            )?,
            HostMessage::ResetEnvironment { .. } => {
                if self.parent.is_some() || self.is_reset_complete {
                    return Err(AdapterSupervisionError::InvalidResetTransition);
                }
                self.parent = Some(host.operation);
                self.outputs = VecDeque::from([b"rollout-reset-observation".to_vec()]);
                self.request_output(host.episode)?;
            }
            HostMessage::StepEnvironment { action_ref } => {
                if self.parent.is_some() || !self.is_reset_complete {
                    return Err(AdapterSupervisionError::InvalidResetTransition);
                }
                self.pending_read_parent = Some(host.operation.clone());
                self.push(
                    host.episode,
                    "native-graph-rollout",
                    format!("{}-read", host.operation),
                    AdapterMessage::GetArtifactRequest {
                        parent_operation: host.operation,
                        request: serde_json::to_value(action_ref).map_err(|error| {
                            AdapterSupervisionError::Process(format!(
                                "fixture action reference is invalid: {error}"
                            ))
                        })?,
                    },
                )?;
            }
            HostMessage::GetArtifactHandle { download, .. } => {
                if self.pending_read_parent.is_none() || self.pending_download.is_some() {
                    return Err(AdapterSupervisionError::InvalidResetTransition);
                }
                self.pending_download = Some(download);
            }
            HostMessage::ArtifactDownloadChunk { download, .. } => {
                if self.pending_download.as_ref() != Some(&download) {
                    return Err(AdapterSupervisionError::InvalidResetTransition);
                }
            }
            HostMessage::ArtifactDownloadComplete { download } => {
                if self.pending_download.take() != Some(download) {
                    return Err(AdapterSupervisionError::InvalidResetTransition);
                }
                self.parent = Some(
                    self.pending_read_parent
                        .take()
                        .ok_or(AdapterSupervisionError::InvalidResetTransition)?,
                );
                self.outputs = VecDeque::from([
                    b"rollout-transition-observation".to_vec(),
                    b"rollout-transition-info".to_vec(),
                    self.workspace_patch.clone(),
                ]);
                self.request_output(host.episode)?;
            }
            HostMessage::PutArtifactHandle { upload, .. } => {
                let operation = self
                    .pending_operation
                    .clone()
                    .ok_or(AdapterSupervisionError::InvalidResetTransition)?;
                if operation != host.operation {
                    return Err(AdapterSupervisionError::InvalidResetTransition);
                }
                let bytes = self
                    .pending_bytes
                    .as_deref()
                    .ok_or(AdapterSupervisionError::InvalidResetTransition)?;
                self.push(
                    host.episode.clone(),
                    "native-graph-rollout",
                    operation.clone(),
                    AdapterMessage::ArtifactUploadChunk {
                        upload: upload.clone(),
                        bytes_base64: STANDARD.encode(bytes),
                    },
                )?;
                self.push(
                    host.episode,
                    "native-graph-rollout",
                    operation,
                    AdapterMessage::ArtifactUploadComplete { upload },
                )?;
            }
            HostMessage::ArtifactCommitted { reference, .. } => {
                let operation = self
                    .pending_operation
                    .take()
                    .ok_or(AdapterSupervisionError::InvalidResetTransition)?;
                if operation != host.operation {
                    return Err(AdapterSupervisionError::InvalidResetTransition);
                }
                self.pending_bytes = None;
                self.references.push(reference);
                if self.outputs.is_empty() {
                    let parent = self
                        .parent
                        .clone()
                        .ok_or(AdapterSupervisionError::InvalidResetTransition)?;
                    self.finish(host.episode, parent)?;
                } else {
                    self.request_output(host.episode)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    async fn read_stdout_frame(
        &mut self,
        _: usize,
        _: std::time::Duration,
    ) -> Result<Vec<u8>, AdapterSupervisionError> {
        self.stdout
            .pop_front()
            .ok_or(AdapterSupervisionError::EndOfStream)
    }

    async fn drain_stderr(&mut self, _: usize) -> Result<Vec<u8>, AdapterSupervisionError> {
        Ok(Vec::new())
    }

    async fn cancel(
        &mut self,
        _: CancelReason,
        _: std::time::Duration,
    ) -> Result<(), AdapterSupervisionError> {
        self.events.borrow_mut().push("client-cancel");
        Ok(())
    }

    async fn reap(
        &mut self,
        _: std::time::Duration,
    ) -> Result<AdapterExit, AdapterSupervisionError> {
        self.events.borrow_mut().push("client-reap");
        Ok(AdapterExit::Reaped)
    }

    fn fence(&mut self) {}
}

impl RolloutAdapterChild {
    fn workspace_patch_archive() -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_size(6);
        header.set_mode(0o600);
        header.set_cksum();
        builder
            .append_data(&mut header, "result.txt", io::Cursor::new(b"south\n"))
            .expect("test workspace patch writes");
        builder.into_inner().expect("test workspace patch finishes")
    }

    fn request_output(&mut self, episode: String) -> Result<(), AdapterSupervisionError> {
        let parent = self
            .parent
            .as_ref()
            .ok_or(AdapterSupervisionError::InvalidResetTransition)?;
        let bytes = self
            .outputs
            .pop_front()
            .ok_or(AdapterSupervisionError::InvalidResetTransition)?;
        let operation = format!("{parent}-output-{}", self.references.len());
        let declared_bytes = u64::try_from(bytes.len())
            .map_err(|_| AdapterSupervisionError::InvalidResetTransition)?;
        self.pending_operation = Some(operation.clone());
        self.pending_bytes = Some(bytes);
        self.push(
            episode,
            "native-graph-rollout",
            operation,
            AdapterMessage::PutArtifactRequest {
                parent_operation: parent.clone(),
                declared_bytes,
            },
        )
    }

    fn finish(
        &mut self,
        episode: String,
        operation: String,
    ) -> Result<(), AdapterSupervisionError> {
        let parent = self
            .parent
            .take()
            .ok_or(AdapterSupervisionError::InvalidResetTransition)?;
        if parent != operation {
            return Err(AdapterSupervisionError::InvalidResetTransition);
        }
        let references = std::mem::take(&mut self.references);
        if !self.is_reset_complete {
            let [observation]: [FrozenArtifactReference; 1] = references
                .try_into()
                .map_err(|_| AdapterSupervisionError::InvalidResetTransition)?;
            self.is_reset_complete = true;
            self.push(
                episode,
                "native-graph-rollout",
                parent,
                AdapterMessage::EnvironmentReset {
                    observation_ref: observation,
                },
            )
        } else {
            let [observation, info, workspace_patch]: [FrozenArtifactReference; 3] = references
                .try_into()
                .map_err(|_| AdapterSupervisionError::InvalidResetTransition)?;
            self.push(
                episode,
                "native-graph-rollout",
                parent,
                AdapterMessage::Transition {
                    observation_ref: observation,
                    reward: 1.0,
                    terminated: true,
                    truncated: false,
                    info_ref: info,
                    workspace_patch_ref: workspace_patch,
                },
            )
        }
    }

    fn push(
        &mut self,
        episode: String,
        span: impl Into<String>,
        operation: String,
        message: AdapterMessage,
    ) -> Result<(), AdapterSupervisionError> {
        let envelope = AdapterEnvelope::new(episode, span, self.sequence, operation, message);
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or(AdapterSupervisionError::InvalidResetTransition)?;
        let mut frame = serde_json::to_vec(&envelope).map_err(|error| {
            AdapterSupervisionError::Process(format!("fixture adapter frame is invalid: {error}"))
        })?;
        frame.push(b'\n');
        self.stdout.push_back(frame);
        Ok(())
    }
}

fn import_live_graph_fixture(endpoint: &str) -> aiperf_runtime::eval::ImportedTask {
    import_live_graph_fixture_with_binding(endpoint, "primary")
}

fn import_external_driver_fixture() -> aiperf_runtime::eval::ImportedTask {
    let task = tempfile::tempdir().expect("temporary external Driver task root");
    fs::create_dir_all(task.path().join("environment"))
        .expect("external Driver environment directory");
    fs::create_dir_all(task.path().join("tests")).expect("external Driver verifier directory");
    fs::create_dir_all(task.path().join("tools")).expect("external Driver tools directory");
    fs::write(
        task.path().join("environment/Dockerfile"),
        b"FROM scratch\n",
    )
    .expect("external Driver Dockerfile");
    fs::write(
        task.path().join("instruction.md"),
        b"Complete the external episode.\n",
    )
    .expect("external Driver instruction");
    fs::write(task.path().join("tests/test.sh"), b"exit 0\n").expect("external Driver verifier");
    fs::write(task.path().join("tools/driver.sh"), b"#!/bin/sh\nexit 0\n")
        .expect("external Driver executable");
    fs::write(
        task.path().join("task.toml"),
        r#"schema_version = "1.1"
artifacts = ["/work/result.txt"]

[task]
name = "example/scored-external-driver"

[native_graph]
profile = "externally_driven"
adapter_manifest = "adapters.toml"
driver = "driver-adapter"
external_driver_factory_id = "scored-fixture"

[verifier]
environment_mode = "separate"
"#,
    )
    .expect("external Driver task manifest");
    fs::write(
        task.path().join("adapters.toml"),
        r#"[[adapters]]
id = "driver-adapter"
role = "driver"
argv = ["tools/driver.sh"]
executable = "tools/driver.sh"
"#,
    )
    .expect("external Driver adapter manifest");
    let source = HarborSource::local(task.path().to_string_lossy())
        .expect("temporary external task path is a valid source");
    HarborImporter::new(&NativeSourceAcquirer)
        .import(&source)
        .expect("external Driver fixture imports")
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

struct RolloutOnlyExecution {
    completed: NativeGraphCompletedAttempt,
    model_calls: usize,
    runtime: Rc<GraphDockerRuntime>,
}

async fn execute_rollout_only_episode() -> RolloutOnlyExecution {
    let model_calls = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route(
        "/v1/chat/completions",
        post({
            let model_calls = Arc::clone(&model_calls);
            move || {
                let model_calls = Arc::clone(&model_calls);
                async move {
                    model_calls.fetch_add(1, Ordering::SeqCst);
                    let frame = serde_json::json!({
                        "id": "rollout-only-policy", "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": r#"{"kind":"move","direction":"north"}"#},
                            "finish_reason": null
                        }],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
                    });
                    (
                        [(header::CONTENT_TYPE, "text/event-stream")],
                        format!("data: {frame}\n\ndata: [DONE]\n\n"),
                    )
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind rollout-only model endpoint");
    let address = listener
        .local_addr()
        .expect("read rollout-only model endpoint address");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve rollout-only model endpoint");
    });

    let imported = import_rollout_only_graph_fixture(&format!("http://{address}"));
    let lifecycle = native_graph_lifecycle_request();
    let trial = HarborEvaluationCoordinator::resolve_trial(&imported, &lifecycle)
        .expect("resolve rollout-only trial");
    let binding = imported
        .package
        .native_graph()
        .expect("rollout-only fixture has NativeGraph package")
        .model_bindings()
        .first()
        .expect("rollout-only fixture has its selected policy binding");
    let capacity_key = ModelCapacityKey::from_task_binding(&imported.task, binding);
    let resources = ResourceLeaseRequest::new(1, 64, BTreeMap::from([(capacity_key.clone(), 1)]))
        .expect("finite rollout-only resources");
    let suite = NativeGraphSuiteManifest::new(vec![
        SuiteTrialSpec::from_imported(
            imported.clone(),
            trial,
            NonZeroUsize::new(1).expect("one rollout-only repetition"),
            resources,
        )
        .expect("rollout-only suite trial"),
    ])
    .expect("one rollout-only suite trial")
    .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(
        b"rollout-only-suite",
    )))
    .expect("resolve one rollout-only assignment");
    let runtime_events = Rc::new(RefCell::new(Vec::new()));
    let runtime = Rc::new(GraphDockerRuntime {
        events: Rc::clone(&runtime_events),
        adapter_spawner: Some(Rc::new(RolloutAdapterSpawner {
            events: runtime_events,
            workspace_patch: RolloutAdapterChild::workspace_patch_archive(),
        })),
        ..GraphDockerRuntime::default()
    });
    let runtime_for_executor: Rc<dyn DockerRuntime> = runtime.clone();
    let inner = DockerNativeGraphEpisodeExecutor::new_with_runtime(
        DockerProcessSandbox::new(),
        runtime_for_executor,
        HarborSandboxRecipe::for_standard_task(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            Some("/work".to_owned()),
        )
        .expect("immutable rollout-only Docker recipe"),
        imported,
        lifecycle,
        Rc::new(
            Application::stock(format!("blake3:{}", "b".repeat(64)))
                .expect("compose stock application"),
        ),
        toml::from_str("version = 1\n").expect("empty model secret mapping"),
        Rc::new(EmptySecrets),
        None,
    )
    .expect("construct Docker rollout-only executor");
    let completed = Rc::new(RefCell::new(None));
    let executor = Rc::new(CapturingExecutor {
        inner,
        completed: Rc::clone(&completed),
    });
    let scheduler = LocalNativeGraphSuiteScheduler::new(
        aiperf_runtime::eval::ResourceLimits::new(1, 1, 64, BTreeMap::from([(capacity_key, 1)]))
            .expect("finite rollout-only scheduler resources"),
    )
    .expect("rollout-only scheduler");
    let runner = Rc::new(NativeGraphEpisodeRunner::new(
        executor,
        Rc::new(HarborEpisodeEvaluatorFactory),
    ));
    let results = run_resolved_suite(&scheduler, suite, runner)
        .await
        .expect("rollout-only episode completes and scores");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].verified_reward(), Some(1.0));
    let completed = completed
        .borrow_mut()
        .take()
        .expect("capturing executor retains the sealed completed attempt");
    RolloutOnlyExecution {
        completed,
        model_calls: model_calls.load(Ordering::SeqCst),
        runtime,
    }
}

struct CapturingExecutor {
    inner: DockerNativeGraphEpisodeExecutor,
    completed: Rc<RefCell<Option<NativeGraphCompletedAttempt>>>,
}

struct TaintedWorkspacePatchCallback {
    rollout_start: Option<NativeGraphLeaseRolloutStart>,
    observations: Rc<RefCell<Vec<String>>>,
}

#[async_trait(?Send)]
impl NativeGraphEpisodeCallback for TaintedWorkspacePatchCallback {
    async fn run(
        &mut self,
        lease: &mut dyn NativeGraphEpisodeLease,
    ) -> Result<(), EvalExecutionError> {
        let adapter = lease.environment_adapter_start()?;
        adapter.start_rollout().await?;
        let session = adapter.rollout_session()?;
        let observation = session.reset().await?;
        let primary = session
            .step(&observation)
            .await
            .expect_err("the fixture's malformed workspace patch must be rejected");
        self.observations.borrow_mut().push("step".to_owned());
        let reset = session
            .reset()
            .await
            .expect_err("a failed workspace patch must taint reset");
        self.observations
            .borrow_mut()
            .push(format!("reset: {reset}"));
        let step = session
            .step(&observation)
            .await
            .expect_err("a failed workspace patch must taint step");
        self.observations.borrow_mut().push(format!("step: {step}"));
        let freeze = session
            .freeze()
            .expect_err("a failed workspace patch must taint freeze");
        self.observations
            .borrow_mut()
            .push(format!("freeze: {freeze}"));
        Err(primary)
    }

    fn take_lease_rollout_start(&mut self) -> Option<NativeGraphLeaseRolloutStart> {
        self.rollout_start.take()
    }
}

#[async_trait(?Send)]
impl NativeGraphEpisodeExecutor for CapturingExecutor {
    async fn execute(
        &self,
        assignment: &EpisodeAssignment,
    ) -> Result<NativeGraphCompletedAttempt, EpisodeExecutionError> {
        let completed = self.inner.execute(assignment).await?;
        self.completed.replace(Some(completed.clone()));
        Ok(completed)
    }
}

fn import_rollout_only_graph_fixture(endpoint: &str) -> aiperf_runtime::eval::ImportedTask {
    let task = tempfile::tempdir().expect("temporary rollout-only task root");
    fs::create_dir_all(task.path().join("environment"))
        .expect("rollout-only environment directory");
    fs::create_dir_all(task.path().join("tests")).expect("rollout-only verifier directory");
    fs::create_dir_all(task.path().join("tools")).expect("rollout-only tools directory");
    fs::create_dir_all(task.path().join("rollout")).expect("rollout-only policy directory");
    fs::write(
        task.path().join("environment/Dockerfile"),
        b"FROM scratch\n",
    )
    .expect("rollout-only Dockerfile");
    fs::write(
        task.path().join("instruction.md"),
        b"Complete the rollout.\n",
    )
    .expect("rollout-only instruction");
    fs::write(task.path().join("tests/test.sh"), b"exit 0\n").expect("rollout-only verifier");
    fs::write(
        task.path().join("tools/environment.sh"),
        b"#!/bin/sh\nexit 0\n",
    )
    .expect("rollout-only adapter executable");
    fs::write(task.path().join("rollout/reset.json"), b"{}\n").expect("rollout-only reset");
    fs::write(task.path().join("rollout/policy.json"), b"{}\n").expect("rollout-only prompt");
    fs::write(
        task.path().join("task.toml"),
        r#"schema_version = "1.1"
artifacts = ["/work/result.txt"]

[task]
name = "example/native-graph-rollout-only"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#,
    )
    .expect("rollout-only task manifest");
    fs::write(
        task.path().join("agent_graph.json"),
        r#"{
  "schema_version": "1.0", "trace_id": "rollout-only", "stage_bound": 1,
  "channels": { "rollout": { "type": "text", "reducer": "overwrite" } },
  "nodes": [],
  "edges": [{ "source": "START", "target": "END" }],
  "terminal_outputs": []
}"#,
    )
    .expect("rollout-only zero-model graph");
    fs::write(
        task.path().join("models.toml"),
        format!(
            r#"[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-default"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "example-model"
urls = ["{endpoint}"]
streaming = true
request_timeout_ms = 30000
capture = "metadata"

[model_bindings.tokenizer]
type = "local"
name = "builtin"
revision = "main"
apply_chat_template = false

[model_bindings.generation]
max_tokens = 17
"#,
        ),
    )
    .expect("rollout-only model binding");
    fs::write(
        task.path().join("adapters.toml"),
        r#"[[adapters]]
id = "environment-adapter"
role = "environment"
argv = ["tools/environment.sh"]
executable = "tools/environment.sh"
"#,
    )
    .expect("rollout-only adapter manifest");
    fs::write(
        task.path().join("rollout.toml"),
        r#"[environment]
adapter_id = "environment-adapter"
protocol_factory_id = "strict_jsonl"
runtime_provider_id = "strict_supervised"
stepper_factory_id = "supervised_environment"
action_encoder_id = "move_v1"
operation_deadline_ms = 5000
reset_source = "rollout/reset.json"
max_frame_bytes = 4096
max_identifier_bytes = 128
max_json_bytes = 2048
max_json_depth = 4
max_json_array_entries = 8
max_json_object_entries = 8
max_operation_ledger_entries = 16
max_model_call_lineage_entries = 4
max_session_model_call_lineage_entries = 16
max_session_model_call_lineage_bytes = 2048
max_artifact_handles = 4
max_artifact_bytes = 4096

[artifacts]
max_artifacts = 8
max_total_bytes = 16384
max_artifact_bytes = 3072
max_download_handles = 4

[policy]
environment = "counter-v1"
model_binding_id = "primary"
prompt_source = "rollout/policy.json"
max_decision_bytes = 256
horizon = 1
gamma = 0.75

[limits]
max_environment_bytes = 256
max_horizon = 8
max_prompt_bytes = 256

[workspace_patch]
mutable_paths = ["result.txt"]
max_patches = 4
max_patch_bytes = 4096
max_total_patch_bytes = 8192
"#,
    )
    .expect("rollout-only policy manifest");
    let source =
        HarborSource::local(task.path().to_string_lossy()).expect("rollout-only source path");
    HarborImporter::new(&NativeSourceAcquirer)
        .import(&source)
        .expect("rollout-only NativeGraph fixture imports")
}
