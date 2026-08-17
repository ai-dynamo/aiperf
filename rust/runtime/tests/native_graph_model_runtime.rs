// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NativeGraph model-runtime resolution contracts.

use std::{
    fs,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use aiperf_runtime::eval::{
    CurrentNativeGraphModelBindingResolver, EngineNativeGraphEpisodeCallback, EnvName,
    EvalExecutionError, GraphLowererFactory, HarborImporter, HarborSource, ModelCapturePolicy,
    ModelRuntimeConfig, NativeGraphFactoryError, NativeGraphLowererProvider,
    NativeGraphModelBindingResolver, NativeGraphPackagePlan, NativeSourceAcquirer, SecretProvider,
    SecretValue,
};
use aiperf_runtime::{
    engine::{
        application::Application, dataset_input::BuiltinRunnerDatasetInputAdapterResolver,
        execution_factories::native_execution_factories,
        graph_input::BuiltinRunnerGraphInputAdapterResolver, registry::HttpExtension,
        sidecar_input::BuiltinRunnerSidecarInputAdapterResolver,
    },
    extensions::{
        AIPerfExtension, AIPerfRegistry, AIPerfRegistryFactory, BuiltinAIPerfRegistryFactory,
        ExtensionError,
    },
};

#[test]
fn resolves_binding_through_the_current_endpoint_transport_and_tokenizer_seams() {
    let imported = import_fixture(false);
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let runtime: ModelRuntimeConfig =
        toml::from_str("version = 1\n").expect("empty runtime secret mapping is valid");
    let resolver = CurrentNativeGraphModelBindingResolver::from_registry(
        BuiltinAIPerfRegistryFactory
            .build()
            .expect("stock registry is available"),
    );

    let bindings = resolver
        .resolve(native.model_bindings(), &runtime, &FixedSecrets)
        .expect("declared current endpoint, transport, and tokenizer resolve");
    let binding = bindings
        .binding("primary")
        .expect("primary binding resolves");

    assert_eq!(binding.profile().endpoint_id.as_str(), "chat");
    assert_eq!(binding.transport_id(), "http");
    assert_eq!(binding.tokenizer_id(), "tiktoken");
    assert_eq!(binding.max_connect_retries(), 2);
    assert_eq!(binding.generation().max_tokens, Some(17));
    assert_eq!(binding.generation().min_tokens, Some(3));
    assert_eq!(binding.generation().temperature, Some(0.37));
    assert_eq!(binding.generation().top_p, Some(0.73));
    assert_eq!(binding.generation().top_k, Some(42));
    assert_eq!(binding.generation().seed, Some(99));
    assert_eq!(binding.generation().presence_penalty, Some(0.25));
    assert_eq!(binding.generation().frequency_penalty, Some(-0.5));
    assert_eq!(binding.generation().repetition_penalty, Some(1.1));
    assert_eq!(binding.capture(), ModelCapturePolicy::RedactedRaw);
}

#[test]
fn resolved_model_secret_never_enters_adapter_environment() {
    let imported = import_fixture(true);
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let runtime: ModelRuntimeConfig =
        toml::from_str("version = 1\n[secrets]\nprovider_key = \"NATIVE_GRAPH_PROVIDER_KEY\"\n")
            .expect("runtime maps logical secret ids to host names");
    let resolver = CurrentNativeGraphModelBindingResolver::from_registry(
        BuiltinAIPerfRegistryFactory
            .build()
            .expect("stock registry is available"),
    );

    let bindings = resolver
        .resolve(native.model_bindings(), &runtime, &FixedSecrets)
        .expect("declared model secret resolves before adapter provisioning");
    let binding = bindings
        .binding("primary")
        .expect("primary binding resolves");

    assert!(binding.model_headers().contains_key("authorization"));
    assert!(
        binding
            .adapter_environment()
            .values()
            .all(|value| value != "secret-value"),
        "model secret must stay outside the adapter environment"
    );
}

#[test]
fn duplicate_native_graph_factory_names_reject_transactionally() {
    let mut registry = AIPerfRegistry::empty_or_base();
    let error = registry
        .register_extension(&DuplicateLowererExtension)
        .expect_err("a duplicate lowerer name must reject the staged extension");

    assert!(error.to_string().contains("duplicate registry name"));
    assert!(
        registry.native_graph_lowerer("test").is_none(),
        "the first staged insertion must not survive a later duplicate"
    );
}

#[test]
fn callback_selects_the_registered_lowerer_before_any_environment_provisioning() {
    let imported = import_fixture(false);
    let native = imported
        .package
        .native_graph()
        .expect("fixture contains a NativeGraph package");
    let calls = Arc::new(AtomicUsize::new(0));
    let application = Application::new(
        format!("blake3:{}", "6".repeat(64)),
        &RejectingLowererRegistryFactory {
            calls: calls.clone(),
        },
        native_execution_factories(),
        Arc::new(BuiltinRunnerGraphInputAdapterResolver::new()),
        Arc::new(BuiltinRunnerDatasetInputAdapterResolver::new()),
        Arc::new(BuiltinRunnerSidecarInputAdapterResolver::new()),
    )
    .expect("compose a product application with an injected lowerer provider");
    let runtime: ModelRuntimeConfig =
        toml::from_str("version = 1\n").expect("empty runtime secret mapping is valid");

    let error = match EngineNativeGraphEpisodeCallback::new(
        &application,
        native,
        &runtime,
        &FixedSecrets,
    ) {
        Ok(_) => {
            panic!("the selected lowerer must be consulted before a Docker environment exists")
        }
        Err(error) => error,
    };

    assert!(error.to_string().contains("test lowerer selected"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

struct DuplicateLowererExtension;

impl AIPerfExtension for DuplicateLowererExtension {
    fn name(&self) -> &str {
        "test.duplicate-native-graph-lowerer"
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        registry
            .register_native_graph_lowerer("test", Arc::new(RejectingLowererProvider::new(0)))?;
        registry.register_native_graph_lowerer("test", Arc::new(RejectingLowererProvider::new(0)))
    }
}

struct RejectingLowererRegistryFactory {
    calls: Arc<AtomicUsize>,
}

impl AIPerfRegistryFactory for RejectingLowererRegistryFactory {
    fn build(&self) -> Result<AIPerfRegistry, ExtensionError> {
        AIPerfRegistry::builtin()?.with_extensions([
            &HttpExtension as &dyn AIPerfExtension,
            &RejectingLowererExtension {
                calls: self.calls.clone(),
            },
        ])
    }
}

struct RejectingLowererExtension {
    calls: Arc<AtomicUsize>,
}

impl AIPerfExtension for RejectingLowererExtension {
    fn name(&self) -> &str {
        "test.rejecting-native-graph-lowerer"
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        registry.register_native_graph_lowerer(
            "native_graph",
            Arc::new(RejectingLowererProvider {
                calls: self.calls.clone(),
            }),
        )
    }
}

struct RejectingLowererProvider {
    calls: Arc<AtomicUsize>,
}

impl RejectingLowererProvider {
    fn new(calls: usize) -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(calls)),
        }
    }
}

impl NativeGraphLowererProvider for RejectingLowererProvider {
    fn bind(
        &self,
        _: &NativeGraphPackagePlan,
    ) -> Result<Arc<dyn GraphLowererFactory>, NativeGraphFactoryError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(NativeGraphFactoryError::new("test lowerer selected"))
    }
}

struct FixedSecrets;

impl SecretProvider for FixedSecrets {
    fn resolve(&self, name: &EnvName) -> Result<SecretValue, EvalExecutionError> {
        if name == "NATIVE_GRAPH_PROVIDER_KEY" {
            Ok(SecretValue::new("secret-value"))
        } else {
            Err(EvalExecutionError::MissingSecret(name.clone()))
        }
    }
}

fn import_fixture(with_secret: bool) -> aiperf_runtime::eval::ImportedTask {
    let task = tempfile::tempdir().expect("temporary task root");
    fs::create_dir_all(task.path().join("environment")).expect("task environment directory");
    fs::create_dir_all(task.path().join("tests")).expect("task tests directory");
    fs::create_dir_all(task.path().join("tools")).expect("adapter directory");
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
name = "example/native-graph-model-runtime"

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
  "schema_version": "1.0", "trace_id": "model-runtime", "stage_bound": 1,
  "channels": { "output": { "type": "messages", "reducer": "add_messages" } },
  "nodes": [{ "id": "model", "kind": "model", "binding": "primary", "output": "output" }],
  "edges": [{ "source": "START", "target": "model" }, { "source": "model", "target": "END" }],
  "terminal_outputs": []
}"#,
    )
    .expect("graph source");
    let authentication = with_secret.then_some(
        "\n[[model_bindings.authentication]]\nheader = \"authorization\"\nsecret = \"provider_key\"\n",
    );
    fs::write(
        task.path().join("models.toml"),
        format!(
            r#"[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-default"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "example-model"
urls = ["https://provider.example/v1"]
streaming = true
max_connect_retries = 2
request_timeout_ms = 30000
capture = "redacted_raw"

[model_bindings.tokenizer]
type = "local"
name = "tiktoken"
revision = "main"
apply_chat_template = true

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
{}"#,
            authentication.unwrap_or_default()
        ),
    )
    .expect("model bindings");
    fs::write(
        task.path().join("adapters.toml"),
        r#"[[adapters]]
id = "tool-adapter"
role = "tool"
argv = ["tools/adapter.sh"]
executable = "tools/adapter.sh"
"#,
    )
    .expect("adapter manifest");
    fs::write(task.path().join("tools/adapter.sh"), b"#!/bin/sh\nexit 0\n")
        .expect("adapter executable");
    let source = HarborSource::local(task.path().to_string_lossy())
        .expect("temporary task path is a valid source");
    HarborImporter::new(&NativeSourceAcquirer)
        .import(&source)
        .expect("fixture imports")
}
