// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Package-bound live-rollout model-decision admission contracts.

use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    rc::Rc,
    sync::{Arc, Mutex},
};

use aiperf_runtime::{
    engine::application::Application,
    eval::{
        AdapterProtocolConfig, AdapterProtocolFactory, AdapterRuntimeFactory, AdapterSpawnRequest,
        AdapterSpawner, AdapterSupervisionError, AgentVariantRef, ArtifactDigest, ArtifactQuota,
        CurrentNativeGraphModelBindingResolver, EpisodeArtifactStore, HarborImporter, HarborSource,
        ModelBindingId, ModelIdentity, ModelRuntimeConfig, MoveV1ActionEncoderFactory,
        NativeGraphAdapterRuntimeProvider, NativeGraphAdapterRuntimeResolution,
        NativeGraphEnvironmentStepperFactory, NativeGraphFactoryError,
        NativeGraphLiveRolloutCoordinator, NativeGraphModelBindingResolver,
        NativeGraphModelDecisionError, NativeGraphPolicyModelRuntime, NativeGraphSuiteManifest,
        NativeSourceAcquirer, PolicyIdentity, ResourceLeaseRequest, RuntimeIdentity,
        StrictAdapterProtocolFactory, SuiteRunId, SuiteTrialSpec, TrialBudget, TrialSpec,
        bind_native_graph_environment_stepper,
    },
    graph::{
        agent::{LiveAgentPolicyDecisionReader, LiveAgentPolicyDecisionRequest},
        tools::{
            EnvironmentResetRequest, EnvironmentSessionAuthority, EnvironmentStepRequest,
            EnvironmentStepper, EnvironmentStepperBinding, EnvironmentStepperError,
            EnvironmentStepperFactory,
        },
    },
};
use async_trait::async_trait;
use axum::{Json, Router, extract::State, http::header, routing::post};

struct NoModelSecrets;

impl aiperf_runtime::eval::SecretProvider for NoModelSecrets {
    fn resolve(
        &self,
        name: &aiperf_runtime::eval::EnvName,
    ) -> Result<aiperf_runtime::eval::SecretValue, aiperf_runtime::eval::EvalExecutionError> {
        Err(aiperf_runtime::eval::EvalExecutionError::MissingSecret(
            name.clone(),
        ))
    }
}

#[tokio::test(flavor = "current_thread")]
async fn engine_selected_runtime_uses_exact_binding_generation_and_bounded_reader() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let endpoint = policy_model_endpoint(
        Arc::clone(&requests),
        r#"{"kind":"move","direction":"north"}"#,
    )
    .await;
    let imported = import_rollout_fixture_at(&endpoint, 35);
    let application = Application::stock(format!("blake3:{}", "2".repeat(64)))
        .expect("stock application composes the selected product seams");
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains the imported NativeGraph package");
    let runtime: ModelRuntimeConfig =
        toml::from_str("version = 1\n").expect("empty runtime secret mapping is valid");
    let bindings = CurrentNativeGraphModelBindingResolver::from_registry(
        application.product_registry().clone(),
    )
    .resolve(native.model_bindings(), &runtime, &NoModelSecrets)
    .expect("the package model bindings resolve through the frozen application");
    let selected = native
        .rollout()
        .expect("fixture has a rollout policy")
        .policy()
        .model_binding_id();
    let engine_runtime = bindings
        .engine_selected_policy_runtime(&application, selected)
        .expect("the exact imported binding prepares a bounded engine policy reader");
    let mut coordinator = selected_coordinator(&imported, engine_runtime)
        .await
        .expect("the engine runtime keeps the imported model selector");
    let root = tempfile::tempdir().expect("artifact-store root");
    let mut store = EpisodeArtifactStore::new(root.path(), quota()).expect("store opens");
    let observation = freeze(&mut store, br#"{"position":0}"#);

    let decision = tokio::task::LocalSet::new()
        .run_until(async {
            coordinator
                .decide_policy_decision(&observation, &mut store)
                .await
        })
        .await
        .expect("an exact-limit engine response reaches the bounded policy reader");

    assert!(format!("{decision:?}").contains("move_v1"));
    let requests = requests
        .lock()
        .expect("test request capture is not poisoned");
    assert_eq!(requests.len(), 1, "only the selected binding dispatches");
    assert_eq!(requests[0]["model"], "example-model");
    assert_eq!(requests[0]["max_completion_tokens"], 17);
    assert_eq!(requests[0]["temperature"], 0.37);
    assert!(
        requests[0].get("response").is_none(),
        "bounded policy capture must not retain a response record"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn engine_selected_runtime_rejects_an_unresolved_selector_before_dispatch() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let endpoint = policy_model_endpoint(
        Arc::clone(&requests),
        r#"{"kind":"move","direction":"north"}"#,
    )
    .await;
    let imported = import_rollout_fixture_at(&endpoint, 35);
    let application = Application::stock(format!("blake3:{}", "3".repeat(64)))
        .expect("stock application composes the selected product seams");
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains the imported NativeGraph package");
    let runtime: ModelRuntimeConfig =
        toml::from_str("version = 1\n").expect("empty runtime secret mapping is valid");
    let bindings = CurrentNativeGraphModelBindingResolver::from_registry(
        application.product_registry().clone(),
    )
    .resolve(native.model_bindings(), &runtime, &NoModelSecrets)
    .expect("the package model bindings resolve through the frozen application");
    let foreign: ModelBindingId =
        serde_json::from_str("\"foreign\"").expect("fixture selector is canonical");

    let error = match bindings.engine_selected_policy_runtime(&application, &foreign) {
        Ok(_) => panic!("an unresolved package selector must fail before environment provisioning"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("selected model binding"));
    assert!(
        requests
            .lock()
            .expect("test request capture is not poisoned")
            .is_empty(),
        "selector preflight cannot reach the selected model transport"
    );
}

#[tokio::test]
async fn reset_observation_dispatches_exactly_one_selected_model_call_before_action_admission() {
    let imported = import_rollout_fixture();
    let calls = Rc::new(RefCell::new(Vec::new()));
    let runtime = RecordingModelRuntime::new(
        "primary",
        Rc::clone(&calls),
        ModelReply::bytes(br#"{"kind":"move","direction":"north"}"#),
    );
    let mut coordinator = selected_coordinator(&imported, Box::new(runtime))
        .await
        .expect("the selected model runtime admits the package model selector");
    let root = tempfile::tempdir().expect("artifact-store root");
    let mut store = EpisodeArtifactStore::new(root.path(), quota()).expect("store opens");
    let observation = freeze(&mut store, br#"{"position":0}"#);

    let _decision = coordinator
        .decide_policy_decision(&observation, &mut store)
        .await
        .expect("the selected bounded model decision admits one action capability");

    let calls = calls.borrow();
    assert_eq!(
        calls.len(),
        1,
        "reset observation must issue one model call"
    );
    assert_eq!(calls[0].binding.as_str(), "primary");
    assert_eq!(calls[0].prompt, b"{\"instruction\":\"choose a move\"}\n");
    assert_eq!(calls[0].observation, br#"{"position":0}"#);
    assert_eq!(calls[0].max_decision_bytes, 256);
}

#[tokio::test]
async fn worker_selected_encoder_admits_the_coordinator_decision_before_one_environment_step() {
    let imported = import_rollout_fixture();
    let runtime = RecordingModelRuntime::new(
        "primary",
        Rc::new(RefCell::new(Vec::new())),
        ModelReply::bytes(br#"{"kind":"move","direction":"north"}"#),
    );
    let mut coordinator = selected_coordinator(&imported, Box::new(runtime))
        .await
        .expect("the selected model runtime admits the package model selector");
    let root = tempfile::tempdir().expect("artifact-store root");
    let mut store = EpisodeArtifactStore::new(root.path(), quota()).expect("store opens");
    let observation = freeze(&mut store, br#"{"position":0}"#);

    let decision = coordinator
        .decide_policy_decision(&observation, &mut store)
        .await
        .expect("the coordinator returns only a sealed policy decision");
    assert!(format!("{decision:?}").contains("move_v1"));
}

#[tokio::test]
async fn oversized_model_decision_never_dispatches_environment_step() {
    let requested = Rc::new(RefCell::new(Vec::new()));
    let materialized = Rc::new(Cell::new(false));
    let (mut coordinator, mut store, observation, _root) = coordinator_with_response(
        ModelReply::oversized_frame(257, Rc::clone(&requested), Rc::clone(&materialized)),
    )
    .await;
    let error = coordinator
        .decide_policy_decision(&observation, &mut store)
        .await
        .expect_err("a response collected above the selected decision bound must be refused");

    assert!(
        error.to_string().contains("frame"),
        "oversized reader refusal must retain its bounded-read context: {error}"
    );
    assert_eq!(requested.borrow().as_slice(), &[256]);
    assert!(
        !materialized.get(),
        "the provider must reject an oversized frame before materializing bytes outside the host cap"
    );
}

#[tokio::test]
async fn malformed_model_decision_never_dispatches_environment_step() {
    let (mut coordinator, mut store, observation, _root) =
        coordinator_with_response(ModelReply::bytes(br#"{"kind":"move","direction":"north"#)).await;
    coordinator
        .decide_policy_decision(&observation, &mut store)
        .await
        .expect_err("malformed policy bytes must fail before a step dispatch");
}

#[tokio::test]
async fn mismatched_model_binding_never_dispatches_environment_step() {
    let imported = import_rollout_fixture();
    let error = match selected_coordinator(
        &imported,
        Box::new(RecordingModelRuntime::new(
            "secondary",
            Rc::new(RefCell::new(Vec::new())),
            ModelReply::bytes(br#"{"kind":"move","direction":"north"}"#),
        )),
    )
    .await
    {
        Ok(_) => panic!("a model runtime from another package binding must be refused"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("model binding"));
}

#[tokio::test]
async fn absent_selected_model_binding_refuses_before_any_environment_provisioning() {
    let imported = import_rollout_fixture();
    let calls = Rc::new(RefCell::new(Vec::new()));
    let runtime = RecordingModelRuntime::new(
        "secondary",
        Rc::clone(&calls),
        ModelReply::bytes(br#"{"kind":"move","direction":"north"}"#),
    );
    let error = match selected_coordinator(&imported, Box::new(runtime)).await {
        Ok(_) => panic!("an absent model selector must refuse before a stepper can be provisioned"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("model binding"));
    assert!(
        calls.borrow().is_empty(),
        "pre-provisioning selector refusal cannot dispatch"
    );
}

struct SeenModelCall {
    binding: ModelBindingId,
    prompt: Vec<u8>,
    observation: Vec<u8>,
    max_decision_bytes: usize,
}

struct RecordingModelRuntime {
    binding: ModelBindingId,
    calls: Rc<RefCell<Vec<SeenModelCall>>>,
    reply: Option<ModelReply>,
}

impl RecordingModelRuntime {
    fn new(binding: &str, calls: Rc<RefCell<Vec<SeenModelCall>>>, reply: ModelReply) -> Self {
        Self {
            binding: serde_json::from_str(&format!("\"{binding}\""))
                .expect("fixture binding is valid"),
            calls,
            reply: Some(reply),
        }
    }
}

enum ModelReply {
    Bytes(Vec<u8>),
    OversizedFrame {
        declared_bytes: usize,
        requested: Rc<RefCell<Vec<usize>>>,
        materialized: Rc<Cell<bool>>,
    },
}

impl ModelReply {
    fn bytes(bytes: &[u8]) -> Self {
        Self::Bytes(bytes.to_vec())
    }

    fn oversized_frame(
        declared_bytes: usize,
        requested: Rc<RefCell<Vec<usize>>>,
        materialized: Rc<Cell<bool>>,
    ) -> Self {
        Self::OversizedFrame {
            declared_bytes,
            requested,
            materialized,
        }
    }
}

#[async_trait(?Send)]
impl NativeGraphPolicyModelRuntime for RecordingModelRuntime {
    fn binding(&self) -> &ModelBindingId {
        &self.binding
    }

    async fn open_decision(
        &mut self,
        request: &LiveAgentPolicyDecisionRequest,
    ) -> Result<Box<dyn LiveAgentPolicyDecisionReader>, NativeGraphModelDecisionError> {
        self.calls.borrow_mut().push(SeenModelCall {
            binding: self.binding.clone(),
            prompt: request.prompt().to_vec(),
            observation: request.observation().to_vec(),
            max_decision_bytes: request.max_decision_bytes(),
        });
        let reply = self.reply.take().ok_or_else(|| {
            NativeGraphModelDecisionError::new("test runtime was asked for a second decision")
        })?;
        Ok(Box::new(RecordingModelReader { reply, offset: 0 }))
    }
}

struct RecordingModelReader {
    reply: ModelReply,
    offset: usize,
}

#[async_trait(?Send)]
impl LiveAgentPolicyDecisionReader for RecordingModelReader {
    async fn read(
        &mut self,
        destination: &mut [u8],
    ) -> Result<usize, aiperf_runtime::graph::agent::AgentLoopError> {
        match &mut self.reply {
            ModelReply::Bytes(bytes) => {
                let remaining = &bytes[self.offset..];
                let count = remaining.len().min(destination.len());
                destination[..count].copy_from_slice(&remaining[..count]);
                self.offset += count;
                Ok(count)
            }
            ModelReply::OversizedFrame {
                declared_bytes,
                requested,
                materialized,
            } => {
                requested.borrow_mut().push(destination.len());
                if *declared_bytes > destination.len() {
                    return Err(aiperf_runtime::graph::agent::AgentLoopError::new(
                        "provider frame exceeds host-owned bounded read",
                    ));
                }
                materialized.set(true);
                destination[..*declared_bytes].fill(b'x');
                Ok(*declared_bytes)
            }
        }
    }
}

async fn coordinator_with_response(
    reply: ModelReply,
) -> (
    NativeGraphLiveRolloutCoordinator,
    EpisodeArtifactStore,
    aiperf_runtime::eval::FrozenArtifact,
    tempfile::TempDir,
) {
    let imported = import_rollout_fixture();
    let runtime = RecordingModelRuntime::new("primary", Rc::new(RefCell::new(Vec::new())), reply);
    let coordinator = selected_coordinator(&imported, Box::new(runtime))
        .await
        .expect("the selected model runtime admits the package model selector");
    let root = tempfile::tempdir().expect("artifact-store root");
    let mut store = EpisodeArtifactStore::new(root.path(), quota()).expect("store opens");
    let observation = freeze(&mut store, br#"{"position":0}"#);
    (coordinator, store, observation, root)
}

async fn selected_coordinator(
    imported: &aiperf_runtime::eval::ImportedTask,
    runtime: Box<dyn NativeGraphPolicyModelRuntime>,
) -> Result<NativeGraphLiveRolloutCoordinator, aiperf_runtime::eval::NativeGraphFactoryError> {
    let mut registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();
    registry
        .register_native_graph_protocol("strict_jsonl", Arc::new(StrictAdapterProtocolFactory))
        .expect("test protocol registration succeeds");
    registry
        .register_native_graph_adapter_runtime(
            "strict_supervised",
            Arc::new(LiveTestRuntimeProvider),
        )
        .expect("test runtime registration succeeds");
    registry
        .register_native_graph_environment_stepper(
            "supervised_environment",
            Arc::new(LiveTestStepperBinder),
        )
        .expect("test stepper registration succeeds");
    registry
        .register_native_graph_action_encoder("move_v1", Arc::new(MoveV1ActionEncoderFactory))
        .expect("test encoder registration succeeds");
    let trial = resolve_rollout_trial(imported.clone());
    let bound = bind_native_graph_environment_stepper(&registry, &trial)?;
    let prepared = bound.prepare_live_rollout_coordinator(runtime)?;
    let root = tempfile::tempdir().expect("live-rollout environment-store root");
    let store = Rc::new(RefCell::new(
        EpisodeArtifactStore::new(root.path(), quota())
            .expect("live-rollout environment store opens"),
    ));
    let started = bound
        .start(
            "live-rollout",
            store,
            Rc::new(LiveTestSpawner) as Rc<dyn AdapterSpawner>,
        )
        .await?;
    started.bind_live_rollout_coordinator(prepared)
}

struct LiveTestRuntimeProvider;

impl NativeGraphAdapterRuntimeProvider for LiveTestRuntimeProvider {
    fn resolve(
        &self,
        config: AdapterProtocolConfig,
        _: Rc<dyn AdapterProtocolFactory>,
    ) -> Result<Rc<dyn NativeGraphAdapterRuntimeResolution>, NativeGraphFactoryError> {
        Ok(Rc::new(LiveTestRuntimeResolution { config }))
    }
}

struct LiveTestRuntimeResolution {
    config: AdapterProtocolConfig,
}

impl NativeGraphAdapterRuntimeResolution for LiveTestRuntimeResolution {
    fn protocol_config(&self) -> &AdapterProtocolConfig {
        &self.config
    }

    fn bind(
        &self,
        _: Rc<dyn AdapterSpawner>,
    ) -> Result<Rc<dyn AdapterRuntimeFactory>, NativeGraphFactoryError> {
        Ok(Rc::new(LiveTestAdapterRuntime {
            config: self.config.clone(),
        }))
    }
}

struct LiveTestAdapterRuntime {
    config: AdapterProtocolConfig,
}

#[async_trait(?Send)]
impl AdapterRuntimeFactory for LiveTestAdapterRuntime {
    fn protocol_config(&self) -> Option<&AdapterProtocolConfig> {
        Some(&self.config)
    }

    async fn start(
        &self,
        _: AdapterSpawnRequest,
    ) -> Result<Box<dyn aiperf_runtime::eval::SupervisedAdapter>, AdapterSupervisionError> {
        Err(AdapterSupervisionError::Process(
            "live-rollout test adapter runtime must not start".to_owned(),
        ))
    }
}

struct LiveTestStepperBinder;

impl NativeGraphEnvironmentStepperFactory for LiveTestStepperBinder {
    fn bind(
        &self,
        _: Rc<dyn AdapterRuntimeFactory>,
    ) -> Result<Rc<dyn EnvironmentStepperFactory>, NativeGraphFactoryError> {
        Ok(Rc::new(LiveTestStepperFactory))
    }
}

struct LiveTestStepperFactory;

#[async_trait(?Send)]
impl EnvironmentStepperFactory for LiveTestStepperFactory {
    async fn start(
        &self,
        _: EnvironmentStepperBinding,
        _: EnvironmentSessionAuthority,
        _: AdapterSpawnRequest,
    ) -> Result<Box<dyn EnvironmentStepper>, EnvironmentStepperError> {
        Ok(Box::new(LiveTestStepper))
    }
}

struct LiveTestStepper;

#[async_trait(?Send)]
impl EnvironmentStepper for LiveTestStepper {
    async fn reset(
        &mut self,
        _: EnvironmentResetRequest,
    ) -> Result<aiperf_runtime::graph::tools::EnvironmentResetRecord, EnvironmentStepperError> {
        Err(EnvironmentStepperError::AlreadyReset)
    }

    async fn step(
        &mut self,
        _: EnvironmentStepRequest,
    ) -> Result<aiperf_runtime::eval::EnvironmentTransitionRecord, EnvironmentStepperError> {
        Err(EnvironmentStepperError::EpisodeTerminal)
    }

    async fn cancel_and_reap(&mut self) -> Result<(), EnvironmentStepperError> {
        // This in-memory fake owns no adapter child.
        Ok(())
    }
}

struct LiveTestSpawner;

impl AdapterSpawner for LiveTestSpawner {
    fn begin_spawn(
        &self,
        _: AdapterSpawnRequest,
    ) -> Result<Box<dyn aiperf_runtime::eval::AdapterSpawnTransaction>, AdapterSupervisionError>
    {
        Err(AdapterSupervisionError::Process(
            "live-rollout test spawner must not start".to_owned(),
        ))
    }
}

fn resolve_rollout_trial(
    imported: aiperf_runtime::eval::ImportedTask,
) -> aiperf_runtime::eval::ResolvedEpisodeTrial {
    let trial = TrialSpec::new(
        imported.task.clone(),
        AgentVariantRef::new("native-graph").expect("fixture agent is valid"),
        ModelIdentity::new("provider-default", "example-model").expect("fixture model is valid"),
        7,
        PolicyIdentity::new(ArtifactDigest::from_bytes(b"policy")),
        TrialBudget::new(30.0, 30.0).expect("fixture budget is valid"),
        ArtifactDigest::from_bytes(b"environment"),
        ArtifactDigest::from_bytes(b"verifier"),
        RuntimeIdentity::new("native").expect("fixture runtime is valid"),
    )
    .expect("fixture trial is valid");
    NativeGraphSuiteManifest::new(vec![
        SuiteTrialSpec::from_imported(
            imported,
            trial,
            NonZeroUsize::new(1).expect("one repetition is nonzero"),
            ResourceLeaseRequest::new(1, 64, BTreeMap::new()).expect("fixture resources are valid"),
        )
        .expect("fixture suite trial resolves"),
    ])
    .expect("fixture suite manifest resolves")
    .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(
        b"live-rollout-run",
    )))
    .expect("fixture suite expands")
    .trials()
    .first()
    .expect("fixture contains exactly one resolved trial")
    .clone()
}

fn freeze(store: &mut EpisodeArtifactStore, bytes: &[u8]) -> aiperf_runtime::eval::FrozenArtifact {
    let declared_bytes = u64::try_from(bytes.len()).expect("fixture length fits u64");
    let upload = store.begin_upload(declared_bytes).expect("upload reserves");
    store
        .write_upload(&upload, &mut std::io::Cursor::new(bytes))
        .expect("upload writes");
    store.commit_upload(&upload).expect("upload freezes")
}

const fn quota() -> ArtifactQuota {
    ArtifactQuota {
        max_artifacts: 8,
        max_total_bytes: 16_384,
        max_artifact_bytes: 3_072,
        max_download_handles: 4,
    }
}

fn import_rollout_fixture() -> aiperf_runtime::eval::ImportedTask {
    import_rollout_fixture_at("https://provider.example/v1", 256)
}

async fn policy_model_endpoint(
    requests: Arc<Mutex<Vec<serde_json::Value>>>,
    decision: &'static str,
) -> String {
    #[derive(Clone)]
    struct PolicyModelState {
        requests: Arc<Mutex<Vec<serde_json::Value>>>,
        decision: &'static str,
    }

    async fn completion(
        State(state): State<PolicyModelState>,
        Json(request): Json<serde_json::Value>,
    ) -> ([(axum::http::HeaderName, &'static str); 1], String) {
        state
            .requests
            .lock()
            .expect("test request capture is not poisoned")
            .push(request);
        let frame = serde_json::json!({
            "id": "native-graph-policy-runtime",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"content": state.decision},
                "finish_reason": null
            }]
        });
        (
            [(header::CONTENT_TYPE, "text/event-stream")],
            format!("data: {frame}\n\ndata: [DONE]\n\n"),
        )
    }

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback policy model endpoint");
    let address = listener
        .local_addr()
        .expect("read loopback policy model endpoint address");
    tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new()
                .route("/v1/chat/completions", post(completion))
                .with_state(PolicyModelState { requests, decision }),
        )
        .await
        .expect("serve loopback policy model endpoint");
    });
    format!("http://{address}")
}

fn import_rollout_fixture_at(
    endpoint: &str,
    max_decision_bytes: usize,
) -> aiperf_runtime::eval::ImportedTask {
    let task = tempfile::tempdir().expect("temporary task root");
    fs::create_dir_all(task.path().join("environment")).expect("task environment directory");
    fs::create_dir_all(task.path().join("tests")).expect("task tests directory");
    fs::create_dir_all(task.path().join("tools")).expect("adapter directory");
    fs::create_dir_all(task.path().join("rollout")).expect("rollout directory");
    fs::write(
        task.path().join("environment/Dockerfile"),
        b"FROM scratch\n",
    )
    .expect("task Dockerfile");
    fs::write(task.path().join("instruction.md"), b"Do work.\n").expect("task instruction");
    fs::write(task.path().join("tests/test.sh"), b"exit 0\n").expect("task verifier");
    fs::write(
        task.path().join("task.toml"),
        r#"schema_version = "1.1"

[task]
name = "example/native-graph-live-rollout"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#,
    )
    .expect("task manifest");
    fs::write(
        task.path().join("agent_graph.json"),
        br#"{
  "schema_version": "1.0", "trace_id": "live-rollout", "stage_bound": 1,
  "channels": { "output": { "type": "messages", "reducer": "add_messages" } },
  "nodes": [{ "id": "model", "kind": "model", "binding": "primary", "output": "output" }],
  "edges": [{ "source": "START", "target": "model" }, { "source": "model", "target": "END" }],
  "terminal_outputs": []
}"#,
    )
    .expect("graph source");
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
capture = "redacted_raw"

[model_bindings.tokenizer]
type = "local"
name = "tiktoken"
revision = "main"
apply_chat_template = false

[model_bindings.generation]
max_tokens = 17
temperature = 0.37
"#,
        ),
    )
    .expect("model bindings");
    fs::write(
        task.path().join("adapters.toml"),
        r#"[[adapters]]
id = "environment-adapter"
role = "environment"
argv = ["tools/environment.sh"]
executable = "tools/environment.sh"
"#,
    )
    .expect("adapter manifest");
    fs::write(
        task.path().join("tools/environment.sh"),
        b"#!/bin/sh\nexit 0\n",
    )
    .expect("adapter executable");
    fs::write(task.path().join("rollout/reset.json"), b"{\"seed\":7}\n").expect("reset source");
    fs::write(
        task.path().join("rollout/policy.json"),
        b"{\"instruction\":\"choose a move\"}\n",
    )
    .expect("policy prompt source");
    fs::write(
        task.path().join("rollout.toml"),
        format!(
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
max_decision_bytes = {max_decision_bytes}
horizon = 4
gamma = 0.75

[limits]
max_environment_bytes = 256
max_horizon = 8
max_prompt_bytes = 256
"#,
        ),
    )
    .expect("rollout manifest");
    let source = HarborSource::local(task.path().to_string_lossy())
        .expect("temporary task path is a valid source");
    HarborImporter::new(&NativeSourceAcquirer)
        .import(&source)
        .expect("fixture imports")
}
