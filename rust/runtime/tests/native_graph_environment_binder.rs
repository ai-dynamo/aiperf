// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NativeGraph rollout environment-binder selection contracts.

use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    fs,
    num::NonZeroUsize,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use aiperf_runtime::graph::tools::EnvironmentStepperFactory;
use aiperf_runtime::{
    eval::{
        AdapterExit, AdapterLifecycleDeadlines, AdapterProtocolConfig, AdapterProtocolFactory,
        AdapterRole, AdapterRuntimeFactory, AdapterSpawnRequest, AdapterSpawner,
        AdapterSupervisionError, AgentVariantRef, ArtifactDigest, ArtifactQuota, CancelReason,
        HarborImporter, HarborSource, ModelIdentity, MoveV1ActionEncoderFactory,
        NativeGraphAdapterRuntimeProvider, NativeGraphAdapterRuntimeResolution,
        NativeGraphEnvironmentStepperFactory, NativeGraphFactoryError, NativeGraphSuiteManifest,
        NativeSourceAcquirer, PolicyIdentity, ProtocolCapability, ProtocolLimits,
        ResourceLeaseRequest, RuntimeIdentity, StrictAdapterProtocolFactory, SuiteRunId,
        SuiteTrialSpec, SupervisedAdapter, SupervisedEnvironmentStepperBinder, TrialBudget,
        TrialSpec, bind_native_graph_environment_stepper,
    },
    extensions::{AIPerfRegistryFactory, BuiltinAIPerfRegistryFactory},
};
use async_trait::async_trait;

#[test]
fn builtin_registry_resolves_every_sealed_rollout_environment_selector() {
    let imported = import_rollout_fixture();
    let rollout = imported
        .package
        .native_graph()
        .and_then(|native| native.rollout())
        .expect("fixture imports one sealed rollout selection");
    let environment = rollout.environment();
    let registry = BuiltinAIPerfRegistryFactory
        .build()
        .expect("the built-in registry is available");

    assert!(
        registry
            .native_graph_protocol(environment.protocol_factory_id().as_str())
            .is_some(),
        "the sealed protocol selector must resolve from the frozen registry"
    );
    assert!(
        registry
            .native_graph_adapter_runtime(environment.runtime_provider_id().as_str())
            .is_some(),
        "the sealed runtime selector must resolve from the frozen registry"
    );
    assert!(
        registry
            .native_graph_environment_stepper(environment.stepper_factory_id().as_str())
            .is_some(),
        "the sealed stepper selector must resolve from the frozen registry"
    );
    assert!(
        registry
            .native_graph_action_encoder(environment.action_encoder_id().as_str())
            .is_some(),
        "the sealed action encoder selector must resolve from the frozen registry"
    );
}

#[test]
fn unknown_rollout_selector_refuses_before_any_adapter_start() {
    let imported = import_rollout_fixture();
    let trial = resolve_rollout_trial(imported.clone());
    let registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();

    let error = bind_native_graph_environment_stepper(&registry, &trial)
        .err()
        .expect("an unregistered protocol selection must fail closed");

    assert!(error.to_string().contains("protocol factory"));
}

#[test]
fn incompatible_runtime_admission_refuses_before_stepper_or_adapter_start() {
    let imported = import_rollout_fixture();
    let trial = resolve_rollout_trial(imported);
    let runtime_starts = Arc::new(AtomicUsize::new(0));
    let runtime_binds = Arc::new(AtomicUsize::new(0));
    let stepper_binds = Arc::new(AtomicUsize::new(0));
    let mut registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();
    registry
        .register_native_graph_protocol("strict_jsonl", Arc::new(StrictAdapterProtocolFactory))
        .expect("test protocol registration succeeds");
    registry
        .register_native_graph_adapter_runtime(
            "strict_supervised",
            Arc::new(WrongRoleRuntimeProvider {
                starts: Arc::clone(&runtime_starts),
                binds: Arc::clone(&runtime_binds),
            }),
        )
        .expect("test runtime registration succeeds");
    registry
        .register_native_graph_environment_stepper(
            "supervised_environment",
            Arc::new(CountingStepperBinder {
                binds: Arc::clone(&stepper_binds),
            }),
        )
        .expect("test stepper registration succeeds");
    registry
        .register_native_graph_action_encoder("move_v1", Arc::new(MoveV1ActionEncoderFactory))
        .expect("test action encoder registration succeeds");
    let error = bind_native_graph_environment_stepper(&registry, &trial)
        .err()
        .expect("a non-environment runtime configuration must be rejected");

    assert!(error.to_string().contains("runtime protocol configuration"));
    assert_eq!(runtime_starts.load(Ordering::Relaxed), 0);
    assert_eq!(runtime_binds.load(Ordering::Relaxed), 0);
    assert_eq!(stepper_binds.load(Ordering::Relaxed), 0);
}

#[test]
fn missing_selected_action_encoder_refuses_before_runtime_or_stepper_binding() {
    let imported = import_rollout_fixture();
    let trial = resolve_rollout_trial(imported);
    let runtime_starts = Arc::new(AtomicUsize::new(0));
    let stepper_binds = Arc::new(AtomicUsize::new(0));
    let mut registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();
    registry
        .register_native_graph_protocol("strict_jsonl", Arc::new(StrictAdapterProtocolFactory))
        .expect("test protocol registration succeeds");
    registry
        .register_native_graph_adapter_runtime(
            "strict_supervised",
            Arc::new(RecordingRuntimeProvider {
                starts: Arc::clone(&runtime_starts),
            }),
        )
        .expect("test runtime registration succeeds");
    registry
        .register_native_graph_environment_stepper(
            "supervised_environment",
            Arc::new(CountingStepperBinder {
                binds: Arc::clone(&stepper_binds),
            }),
        )
        .expect("test stepper registration succeeds");

    let error = bind_native_graph_environment_stepper(&registry, &trial)
        .err()
        .expect("a missing package-selected action encoder must fail closed");

    assert!(error.to_string().contains("action encoder"));
    assert_eq!(runtime_starts.load(Ordering::Relaxed), 0);
    assert_eq!(stepper_binds.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn selected_worker_local_components_preserve_rollout_admission_through_stepper_start() {
    let imported = import_rollout_fixture();
    let trial = resolve_rollout_trial(imported.clone());
    let runtime_starts = Arc::new(AtomicUsize::new(0));
    let mut registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();
    registry
        .register_native_graph_protocol("strict_jsonl", Arc::new(StrictAdapterProtocolFactory))
        .expect("test protocol registration succeeds");
    registry
        .register_native_graph_adapter_runtime(
            "strict_supervised",
            Arc::new(RecordingRuntimeProvider {
                starts: Arc::clone(&runtime_starts),
            }),
        )
        .expect("test runtime registration succeeds");
    registry
        .register_native_graph_environment_stepper(
            "supervised_environment",
            Arc::new(SupervisedEnvironmentStepperBinder),
        )
        .expect("test stepper registration succeeds");
    registry
        .register_native_graph_action_encoder("move_v1", Arc::new(MoveV1ActionEncoderFactory))
        .expect("test action encoder registration succeeds");
    let spawner = Rc::new(CountingSpawner::default());
    let bound = bind_native_graph_environment_stepper(&registry, &trial)
        .expect("the selected components bind one worker-local environment stepper");
    let store_root = tempfile::tempdir().expect("artifact-store root");
    let store = Rc::new(RefCell::new(
        aiperf_runtime::eval::EpisodeArtifactStore::new(
            store_root.path(),
            ArtifactQuota {
                max_artifacts: 8,
                max_total_bytes: 16_384,
                max_artifact_bytes: 3_072,
                max_download_handles: 4,
            },
        )
        .expect("the package-selected store quota is valid"),
    ));
    let started = bound
        .start(
            "root",
            Rc::clone(&store),
            Rc::clone(&spawner) as Rc<dyn AdapterSpawner>,
        )
        .await
        .expect("the selected runtime and supervised stepper start with sealed admission");

    assert_eq!(bound.adapter().id.as_str(), "environment-adapter");
    assert_eq!(bound.package_identity(), &imported.task.digest);
    assert_eq!(bound.action_encoder_id().as_str(), "move_v1");
    assert_eq!(bound.action_encoder().id(), "move_v1");
    assert_eq!(bound.operation_deadline(), Duration::from_secs(5));
    assert_eq!(runtime_starts.load(Ordering::Relaxed), 1);
    assert_eq!(spawner.starts.get(), 0);
    assert_eq!(
        store
            .borrow()
            .read_frozen(started.reset_input().artifact())
            .expect("reset input is frozen through the Rust-owned store"),
        b"{\"seed\":7}\n"
    );
    drop(started.into_stepper());
}

#[tokio::test]
async fn sealed_start_mints_adapter_request_without_caller_argv_or_model_secret() {
    let imported = import_rollout_fixture();
    let trial = resolve_rollout_trial(imported);
    let mut registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();
    registry
        .register_native_graph_protocol("strict_jsonl", Arc::new(StrictAdapterProtocolFactory))
        .expect("test protocol registration succeeds");
    registry
        .register_native_graph_adapter_runtime(
            "strict_supervised",
            Arc::new(CapturingRuntimeProvider),
        )
        .expect("test runtime registration succeeds");
    registry
        .register_native_graph_environment_stepper(
            "supervised_environment",
            Arc::new(SupervisedEnvironmentStepperBinder),
        )
        .expect("test stepper registration succeeds");
    registry
        .register_native_graph_action_encoder("move_v1", Arc::new(MoveV1ActionEncoderFactory))
        .expect("test action encoder registration succeeds");
    let spawner = Rc::new(CapturingSpawner::default());
    let bound =
        bind_native_graph_environment_stepper(&registry, &trial).expect("selected components bind");
    let store_root = tempfile::tempdir().expect("artifact-store root");
    let store = Rc::new(RefCell::new(
        aiperf_runtime::eval::EpisodeArtifactStore::new(
            store_root.path(),
            ArtifactQuota {
                max_artifacts: 8,
                max_total_bytes: 16_384,
                max_artifact_bytes: 3_072,
                max_download_handles: 4,
            },
        )
        .expect("store opens"),
    ));
    let error = match bound
        .start("root", store, Rc::clone(&spawner) as Rc<dyn AdapterSpawner>)
        .await
    {
        Ok(_) => panic!("the capturing runtime must refuse the test start"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("capturing test spawner"));
    let requests = spawner.requests.borrow();
    assert_eq!(requests.len(), 1);
    let request = requests.first().expect("the sealed request is retained");
    assert_eq!(request.argv(), ["tools/environment.sh"]);
    assert!(request.environment().is_empty());
    assert!(!request.environment().contains_key("MODEL_SECRET"));
    assert_eq!(request.deadlines().operation(), Duration::from_secs(5));
}

#[derive(Default)]
struct CountingSpawner {
    starts: Cell<usize>,
}

impl AdapterSpawner for CountingSpawner {
    fn begin_spawn(
        &self,
        _: AdapterSpawnRequest,
    ) -> Result<Box<dyn aiperf_runtime::eval::AdapterSpawnTransaction>, AdapterSupervisionError>
    {
        self.starts.set(self.starts.get() + 1);
        Err(AdapterSupervisionError::Process(
            "test spawner must not start".to_owned(),
        ))
    }
}

#[derive(Default)]
struct CapturingSpawner {
    requests: RefCell<Vec<AdapterSpawnRequest>>,
}

impl AdapterSpawner for CapturingSpawner {
    fn begin_spawn(
        &self,
        request: AdapterSpawnRequest,
    ) -> Result<Box<dyn aiperf_runtime::eval::AdapterSpawnTransaction>, AdapterSupervisionError>
    {
        self.requests.borrow_mut().push(request);
        Err(AdapterSupervisionError::Process(
            "capturing test spawner refuses launch".to_owned(),
        ))
    }
}

struct RecordingRuntimeProvider {
    starts: Arc<AtomicUsize>,
}

struct CapturingRuntimeProvider;

impl NativeGraphAdapterRuntimeProvider for CapturingRuntimeProvider {
    fn resolve(
        &self,
        config: AdapterProtocolConfig,
        _: Rc<dyn AdapterProtocolFactory>,
    ) -> Result<Rc<dyn NativeGraphAdapterRuntimeResolution>, NativeGraphFactoryError> {
        Ok(Rc::new(CapturingRuntimeResolution { config }))
    }
}

struct CapturingRuntimeResolution {
    config: AdapterProtocolConfig,
}

impl NativeGraphAdapterRuntimeResolution for CapturingRuntimeResolution {
    fn protocol_config(&self) -> &AdapterProtocolConfig {
        &self.config
    }

    fn bind(
        &self,
        spawner: Rc<dyn AdapterSpawner>,
    ) -> Result<Rc<dyn AdapterRuntimeFactory>, NativeGraphFactoryError> {
        Ok(Rc::new(CapturingRuntime {
            config: self.config.clone(),
            spawner,
        }))
    }
}

struct CapturingRuntime {
    config: AdapterProtocolConfig,
    spawner: Rc<dyn AdapterSpawner>,
}

#[async_trait(?Send)]
impl AdapterRuntimeFactory for CapturingRuntime {
    fn protocol_config(&self) -> Option<&AdapterProtocolConfig> {
        Some(&self.config)
    }

    async fn start(
        &self,
        request: AdapterSpawnRequest,
    ) -> Result<Box<dyn SupervisedAdapter>, AdapterSupervisionError> {
        let _ = self.spawner.begin_spawn(request)?;
        Err(AdapterSupervisionError::Process(
            "capturing test transaction unexpectedly started".to_owned(),
        ))
    }
}

impl NativeGraphAdapterRuntimeProvider for RecordingRuntimeProvider {
    fn resolve(
        &self,
        config: AdapterProtocolConfig,
        _: Rc<dyn AdapterProtocolFactory>,
    ) -> Result<Rc<dyn NativeGraphAdapterRuntimeResolution>, NativeGraphFactoryError> {
        Ok(Rc::new(RecordingRuntimeResolution {
            config,
            starts: Arc::clone(&self.starts),
        }))
    }
}

struct RecordingRuntimeResolution {
    config: AdapterProtocolConfig,
    starts: Arc<AtomicUsize>,
}

impl NativeGraphAdapterRuntimeResolution for RecordingRuntimeResolution {
    fn protocol_config(&self) -> &AdapterProtocolConfig {
        &self.config
    }

    fn bind(
        &self,
        _: Rc<dyn AdapterSpawner>,
    ) -> Result<Rc<dyn AdapterRuntimeFactory>, NativeGraphFactoryError> {
        Ok(Rc::new(RecordingRuntime {
            config: self.config.clone(),
            starts: Arc::clone(&self.starts),
        }))
    }
}

struct WrongRoleRuntimeProvider {
    starts: Arc<AtomicUsize>,
    binds: Arc<AtomicUsize>,
}

impl NativeGraphAdapterRuntimeProvider for WrongRoleRuntimeProvider {
    fn resolve(
        &self,
        config: AdapterProtocolConfig,
        _: Rc<dyn AdapterProtocolFactory>,
    ) -> Result<Rc<dyn NativeGraphAdapterRuntimeResolution>, NativeGraphFactoryError> {
        let config = AdapterProtocolConfig::new(
            AdapterRole::Tool,
            config.episode(),
            [ProtocolCapability::Tool].into_iter().collect(),
            BTreeSet::new(),
            ProtocolLimits::default(),
        )
        .map_err(|error| NativeGraphFactoryError::new(error.to_string()))?;
        Ok(Rc::new(WrongRoleRuntimeResolution {
            config,
            starts: Arc::clone(&self.starts),
            binds: Arc::clone(&self.binds),
        }))
    }
}

struct WrongRoleRuntimeResolution {
    config: AdapterProtocolConfig,
    starts: Arc<AtomicUsize>,
    binds: Arc<AtomicUsize>,
}

impl NativeGraphAdapterRuntimeResolution for WrongRoleRuntimeResolution {
    fn protocol_config(&self) -> &AdapterProtocolConfig {
        &self.config
    }

    fn bind(
        &self,
        spawner: Rc<dyn AdapterSpawner>,
    ) -> Result<Rc<dyn AdapterRuntimeFactory>, NativeGraphFactoryError> {
        self.binds.fetch_add(1, Ordering::Relaxed);
        let _ = spawner.begin_spawn(adapter_spawn_request(Duration::from_secs(5)));
        Ok(Rc::new(RecordingRuntime {
            config: self.config.clone(),
            starts: Arc::clone(&self.starts),
        }))
    }
}

struct RecordingRuntime {
    config: AdapterProtocolConfig,
    starts: Arc<AtomicUsize>,
}

#[async_trait(?Send)]
impl AdapterRuntimeFactory for RecordingRuntime {
    fn protocol_config(&self) -> Option<&AdapterProtocolConfig> {
        Some(&self.config)
    }

    async fn start(
        &self,
        _: AdapterSpawnRequest,
    ) -> Result<Box<dyn SupervisedAdapter>, AdapterSupervisionError> {
        self.starts.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(RecordingAdapter))
    }
}

struct RecordingAdapter;

#[async_trait(?Send)]
impl SupervisedAdapter for RecordingAdapter {
    async fn send(
        &mut self,
        _: aiperf_runtime::eval::HostEnvelope,
    ) -> Result<(), AdapterSupervisionError> {
        Ok(())
    }

    async fn receive(
        &mut self,
    ) -> Result<aiperf_runtime::eval::AdapterEnvelope, AdapterSupervisionError> {
        Err(AdapterSupervisionError::EndOfStream)
    }

    async fn receive_heartbeat(
        &mut self,
    ) -> Result<aiperf_runtime::eval::AdapterEnvelope, AdapterSupervisionError> {
        self.receive().await
    }

    async fn receive_idle(
        &mut self,
    ) -> Result<aiperf_runtime::eval::AdapterEnvelope, AdapterSupervisionError> {
        self.receive().await
    }

    async fn reset(
        &mut self,
        _: aiperf_runtime::eval::HostEnvelope,
    ) -> Result<(), AdapterSupervisionError> {
        Ok(())
    }

    fn release_download_handle(
        &mut self,
        _: &aiperf_runtime::eval::ArtifactDownloadHandle,
    ) -> Result<(), AdapterSupervisionError> {
        Ok(())
    }

    async fn cancel_and_reap(
        &mut self,
        _: CancelReason,
    ) -> Result<AdapterExit, AdapterSupervisionError> {
        Ok(AdapterExit::Reaped)
    }
}

struct CountingStepperBinder {
    binds: Arc<AtomicUsize>,
}

impl NativeGraphEnvironmentStepperFactory for CountingStepperBinder {
    fn bind(
        &self,
        _: Rc<dyn AdapterRuntimeFactory>,
    ) -> Result<Rc<dyn EnvironmentStepperFactory>, NativeGraphFactoryError> {
        self.binds.fetch_add(1, Ordering::Relaxed);
        Err(NativeGraphFactoryError::new(
            "test stepper binder must not run",
        ))
    }
}

fn adapter_spawn_request(operation: Duration) -> AdapterSpawnRequest {
    let deadlines = AdapterLifecycleDeadlines::new(
        Duration::from_secs(1),
        Duration::from_secs(1),
        Duration::from_secs(1),
        Duration::from_secs(1),
        operation,
        Duration::from_secs(1),
        Duration::from_secs(1),
    )
    .expect("test deadlines are valid");
    AdapterSpawnRequest::for_non_model_adapter(
        ["environment-adapter".to_owned()],
        BTreeMap::new(),
        deadlines,
    )
    .expect("test spawn request is valid")
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
        b"environment-binder-run",
    )))
    .expect("fixture suite expands")
    .trials()
    .first()
    .expect("fixture contains exactly one resolved trial")
    .clone()
}

fn import_rollout_fixture() -> aiperf_runtime::eval::ImportedTask {
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
name = "example/native-graph-environment-binder"

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
  "schema_version": "1.0", "trace_id": "environment-binder", "stage_bound": 1,
  "channels": { "output": { "type": "messages", "reducer": "add_messages" } },
  "nodes": [{ "id": "model", "kind": "model", "binding": "primary", "output": "output" }],
  "edges": [{ "source": "START", "target": "model" }, { "source": "model", "target": "END" }],
  "terminal_outputs": []
}"#,
    )
    .expect("graph source");
    fs::write(
        task.path().join("models.toml"),
        r#"[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-default"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "example-model"
urls = ["https://provider.example/v1"]
streaming = true
request_timeout_ms = 30000
capture = "none"

[model_bindings.tokenizer]
type = "local"
name = "tiktoken"
revision = "main"
apply_chat_template = false

[model_bindings.generation]
"#,
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
horizon = 4
gamma = 0.75

[limits]
max_environment_bytes = 256
max_horizon = 8
"#,
    )
    .expect("rollout manifest");
    let source = HarborSource::local(task.path().to_string_lossy())
        .expect("temporary task path is a valid source");
    HarborImporter::new(&NativeSourceAcquirer)
        .import(&source)
        .expect("fixture imports")
}
