// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::{fs, path::Path};

use aiperf_runtime::eval::{
    HarborImporter, HarborSource, ModelCapturePolicy, NativeSourceAcquirer, TokenizerBindingSpec,
};

const LEGACY_DIGEST: &str =
    "blake3:f6e65dd2abc7f38df7d68eb599f81326c6d4a3850d1ef775ac8b449654ac7584";

#[test]
fn model_binding_retains_every_runtime_selection_and_logical_secret_identifiers() {
    let task = native_task_fixture(b"print('adapter')\n");

    let imported = import_native_task(task.path()).unwrap();
    let binding = &imported.package.native_graph().unwrap().model_bindings()[0];

    assert_eq!(binding.id.as_str(), "primary");
    assert_eq!(binding.endpoint_profile_id, "provider-default");
    assert_eq!(binding.endpoint_factory_id().as_str(), "chat");
    assert_eq!(binding.transport_factory_id(), "http");
    assert_eq!(binding.model, "example-model");
    assert_eq!(
        binding.urls,
        ["https://provider.example/v1", "https://fallback.example/v1",]
    );
    assert!(binding.streaming);
    assert!(matches!(
        &binding.tokenizer,
        TokenizerBindingSpec::Local {
            name,
            revision,
            apply_chat_template: true,
        } if name == "builtin" && revision == "main"
    ));
    assert_eq!(
        binding
            .authentication()
            .iter()
            .map(|reference| (reference.header.as_str(), reference.secret.as_str()))
            .collect::<Vec<_>>(),
        vec![
            ("authorization", "provider-key"),
            ("x-secondary-authorization", "provider-secondary-key"),
        ]
    );
    assert_eq!(binding.generation.temperature, Some(0.4));
    assert_eq!(binding.generation.top_p, Some(0.9));
    assert_eq!(binding.generation.top_k, Some(32));
    assert_eq!(binding.generation.max_tokens, Some(128));
    assert_eq!(binding.generation.min_tokens, Some(8));
    assert_eq!(binding.generation.seed, Some(7));
    assert_eq!(binding.generation.presence_penalty, Some(0.1));
    assert_eq!(binding.generation.frequency_penalty, Some(0.2));
    assert_eq!(binding.generation.repetition_penalty, Some(1.1));
    assert_eq!(binding.max_connect_retries, 2);
    assert_eq!(binding.request_timeout_ms.get(), 30000);
    assert_eq!(binding.capture, ModelCapturePolicy::RedactedRaw);
}

#[test]
fn every_declared_model_binding_selection_changes_the_resolved_binding() {
    let baseline = native_task_fixture(b"print('adapter')\n");
    let baseline_binding = import_native_task(baseline.path())
        .unwrap()
        .package
        .native_graph()
        .unwrap()
        .model_bindings()[0]
        .clone();

    for (selection, from, to) in [
        ("binding id", "id = \"primary\"", "id = \"primary-v2\""),
        (
            "endpoint profile",
            "endpoint_profile_id = \"provider-default\"",
            "endpoint_profile_id = \"provider-alternate\"",
        ),
        (
            "endpoint factory",
            "endpoint_factory_id = \"chat\"",
            "endpoint_factory_id = \"completion\"",
        ),
        (
            "transport factory",
            "transport_factory_id = \"http\"",
            "transport_factory_id = \"grpc\"",
        ),
        (
            "model",
            "model = \"example-model\"",
            "model = \"example-model-v2\"",
        ),
        (
            "ordered URLs",
            "urls = [\"https://provider.example/v1\", \"https://fallback.example/v1\"]",
            "urls = [\"https://fallback.example/v1\", \"https://provider.example/v1\"]",
        ),
        ("streaming", "streaming = true", "streaming = false"),
        (
            "local tokenizer name",
            "name = \"builtin\"",
            "name = \"custom\"",
        ),
        (
            "local tokenizer revision",
            "revision = \"main\"",
            "revision = \"v2\"",
        ),
        (
            "local tokenizer template",
            "apply_chat_template = true",
            "apply_chat_template = false",
        ),
        (
            "first authentication header",
            "header = \"Authorization\"",
            "header = \"X-Authorization\"",
        ),
        (
            "first authentication secret id",
            "secret = \"provider-key\"",
            "secret = \"provider-key-v2\"",
        ),
        (
            "second authentication header",
            "header = \"X-Secondary-Authorization\"",
            "header = \"X-Secondary-Authorization-V2\"",
        ),
        (
            "second authentication secret id",
            "secret = \"provider-secondary-key\"",
            "secret = \"provider-secondary-key-v2\"",
        ),
        ("temperature", "temperature = 0.4", "temperature = 0.5"),
        ("top p", "top_p = 0.9", "top_p = 0.8"),
        ("top k", "top_k = 32", "top_k = 16"),
        ("max tokens", "max_tokens = 128", "max_tokens = 64"),
        ("min tokens", "min_tokens = 8", "min_tokens = 4"),
        ("seed", "seed = 7", "seed = 8"),
        (
            "presence penalty",
            "presence_penalty = 0.1",
            "presence_penalty = 0.2",
        ),
        (
            "frequency penalty",
            "frequency_penalty = 0.2",
            "frequency_penalty = 0.3",
        ),
        (
            "repetition penalty",
            "repetition_penalty = 1.1",
            "repetition_penalty = 1.2",
        ),
        (
            "connect retries",
            "max_connect_retries = 2",
            "max_connect_retries = 3",
        ),
        (
            "request timeout",
            "request_timeout_ms = 30000",
            "request_timeout_ms = 30001",
        ),
        (
            "capture policy",
            "capture = \"redacted_raw\"",
            "capture = \"metadata\"",
        ),
    ] {
        let task = native_task_fixture(b"print('adapter')\n");
        replace(&task.path().join("models.toml"), from, to);
        let imported = import_native_task(task.path()).unwrap();
        let binding = &imported.package.native_graph().unwrap().model_bindings()[0];

        assert_ne!(
            binding, &baseline_binding,
            "{selection} must change the resolved NativeGraph binding"
        );
    }

    let server = server_tokenizer_task_fixture();
    let server_binding = import_native_task(server.path())
        .unwrap()
        .package
        .native_graph()
        .unwrap()
        .model_bindings()[0]
        .clone();
    assert!(matches!(
        &server_binding.tokenizer,
        TokenizerBindingSpec::Server {
            url,
            apply_chat_template: true,
        } if url == "https://tokenizer.example/v1"
    ));
    assert_ne!(
        server_binding.tokenizer, baseline_binding.tokenizer,
        "the server tokenizer variant must be retained distinctly from local tokenization"
    );
    for (selection, from, to) in [
        (
            "server tokenizer URL",
            "https://tokenizer.example/v1",
            "https://tokenizer.example/v2",
        ),
        (
            "server tokenizer template",
            "apply_chat_template = true",
            "apply_chat_template = false",
        ),
    ] {
        let server = server_tokenizer_task_fixture();
        replace(&server.path().join("models.toml"), from, to);
        let imported = import_native_task(server.path()).unwrap();
        let binding = &imported.package.native_graph().unwrap().model_bindings()[0];

        assert_ne!(
            binding, &server_binding,
            "{selection} must change the resolved server-tokenizer binding"
        );
    }
}

#[test]
fn executable_adapter_mutation_changes_package_identity() {
    let first = native_task_fixture(b"print('a')\n");
    let second = native_task_fixture(b"print('b')\n");

    assert_ne!(
        import_native_task(first.path()).unwrap().task.digest,
        import_native_task(second.path()).unwrap().task.digest
    );
}

#[test]
fn native_graph_plan_retains_the_validated_program_source_snapshot() {
    let task = native_task_fixture(b"print('adapter')\n");
    let program_path = task.path().join("agent_graph.json");
    let imported = import_native_task(task.path()).unwrap();

    fs::write(&program_path, b"{\"mutated\":true}\n").unwrap();

    let program = imported
        .package
        .native_graph()
        .unwrap()
        .program_source()
        .unwrap();
    assert_eq!(program.path(), "agent_graph.json");
    assert_eq!(program.bytes(), b"{}\n");
}

#[test]
fn native_graph_rejects_strict_schema_and_binding_boundary_violations() {
    let unknown = native_task_fixture(b"print('adapter')\n");
    append(&unknown.path().join("task.toml"), "unknown_field = true\n");
    assert_invalid_package(unknown.path());

    let profile = native_task_fixture(b"print('adapter')\n");
    replace(
        &profile.path().join("task.toml"),
        "profile = \"native_graph\"",
        "profile = \"unknown\"",
    );
    assert_invalid_package(profile.path());

    let steps = native_task_fixture(b"print('adapter')\n");
    append(
        &steps.path().join("task.toml"),
        "[[steps]]\nname = \"forbidden\"\n",
    );
    let error = import_native_task(steps.path()).unwrap_err();
    assert!(error.to_string().contains("must not declare steps"));

    let path = native_task_fixture(b"print('adapter')\n");
    replace(
        &path.path().join("adapters.toml"),
        "argv = [\"tools/adapter.py\"]\nexecutable = \"tools/adapter.py\"",
        "argv = [\"tools/../adapter.py\"]\nexecutable = \"tools/../adapter.py\"",
    );
    let error = import_native_task(path.path()).unwrap_err();
    assert!(
        error.to_string().contains("canonical relative path"),
        "unexpected canonical-path rejection: {error}"
    );

    let id = native_task_fixture(b"print('adapter')\n");
    replace(
        &id.path().join("models.toml"),
        "id = \"primary\"",
        "id = \"1primary\"",
    );
    assert_invalid_package(id.path());

    let userinfo = native_task_fixture(b"print('adapter')\n");
    replace(
        &userinfo.path().join("models.toml"),
        "https://provider.example/v1",
        "https://actual-secret@provider.example/v1",
    );
    assert_invalid_package(userinfo.path());

    let query = native_task_fixture(b"print('adapter')\n");
    replace(
        &query.path().join("models.toml"),
        "https://provider.example/v1",
        "https://provider.example/v1?api_key=actual-secret",
    );
    assert_invalid_package(query.path());

    let server_tokenizer = native_task_fixture(b"print('adapter')\n");
    replace(
        &server_tokenizer.path().join("models.toml"),
        "type = \"local\"\nname = \"builtin\"\nrevision = \"main\"\napply_chat_template = true",
        "type = \"server\"\nurl = \"https://actual-secret@tokenizer.example/v1\"\napply_chat_template = true",
    );
    assert_invalid_package(server_tokenizer.path());

    let finite = native_task_fixture(b"print('adapter')\n");
    replace(
        &finite.path().join("models.toml"),
        "temperature = 0.4",
        "temperature = nan",
    );
    assert_invalid_package(finite.path());

    let header = native_task_fixture(b"print('adapter')\n");
    replace(
        &header.path().join("models.toml"),
        "header = \"Authorization\"",
        "header = \"Authorization Value\"",
    );
    assert_invalid_package(header.path());

    let duplicate_adapter = native_task_fixture(b"print('adapter')\n");
    append(
        &duplicate_adapter.path().join("adapters.toml"),
        "\n[[adapters]]\nid = \"tool-adapter\"\nrole = \"heuristic\"\nargv = [\"tools/adapter.py\"]\nexecutable = \"tools/adapter.py\"\n",
    );
    assert_invalid_package(duplicate_adapter.path());

    let duplicate_model = native_task_fixture(b"print('adapter')\n");
    append(
        &duplicate_model.path().join("models.toml"),
        r#"
[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-secondary"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "second-example-model"
urls = ["https://second-provider.example/v1"]
streaming = false
request_timeout_ms = 30000
capture = "none"

[model_bindings.tokenizer]
type = "local"
name = "builtin"
revision = "main"

[model_bindings.generation]
"#,
    );
    assert_invalid_package(duplicate_model.path());

    let duplicate_header = native_task_fixture(b"print('adapter')\n");
    replace(
        &duplicate_header.path().join("models.toml"),
        "header = \"X-Secondary-Authorization\"",
        "header = \"authorization\"",
    );
    assert_invalid_package(duplicate_header.path());

    let empty_urls = native_task_fixture(b"print('adapter')\n");
    replace(
        &empty_urls.path().join("models.toml"),
        "urls = [\"https://provider.example/v1\", \"https://fallback.example/v1\"]",
        "urls = []",
    );
    assert_invalid_package(empty_urls.path());

    let empty_argv = native_task_fixture(b"print('adapter')\n");
    replace(
        &empty_argv.path().join("adapters.toml"),
        "argv = [\"tools/adapter.py\"]",
        "argv = []",
    );
    assert_invalid_package(empty_argv.path());

    let local_tokenizer_url = native_task_fixture(b"print('adapter')\n");
    replace(
        &local_tokenizer_url.path().join("models.toml"),
        "revision = \"main\"\napply_chat_template = true",
        "revision = \"main\"\nurl = \"https://tokenizer.example/v1\"\napply_chat_template = true",
    );
    assert_invalid_package(local_tokenizer_url.path());

    let server_tokenizer_name = native_task_fixture(b"print('adapter')\n");
    replace(
        &server_tokenizer_name.path().join("models.toml"),
        "type = \"local\"\nname = \"builtin\"\nrevision = \"main\"\napply_chat_template = true",
        "type = \"server\"\nname = \"builtin\"\nurl = \"https://tokenizer.example/v1\"\napply_chat_template = true",
    );
    assert_invalid_package(server_tokenizer_name.path());

    let zero_timeout = native_task_fixture(b"print('adapter')\n");
    replace(
        &zero_timeout.path().join("models.toml"),
        "request_timeout_ms = 30000",
        "request_timeout_ms = 0",
    );
    assert_invalid_package(zero_timeout.path());

    let native_driver = native_task_fixture(b"print('adapter')\n");
    replace(
        &native_driver.path().join("adapters.toml"),
        "role = \"tool\"",
        "role = \"driver\"",
    );
    let error = import_native_task(native_driver.path()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("native_graph profile must not declare a driver adapter"),
        "unexpected native profile-role rejection: {error}"
    );

    let external_role = externally_driven_task_fixture();
    assert!(
        import_native_task(external_role.path()).is_ok(),
        "externally driven baseline must be valid"
    );
    replace(
        &external_role.path().join("adapters.toml"),
        "role = \"driver\"",
        "role = \"tool\"",
    );
    let error = import_native_task(external_role.path()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("externally_driven profile requires exactly one declared driver adapter"),
        "unexpected external profile-role rejection: {error}"
    );
}

#[test]
fn adapter_invocation_starts_at_the_declared_executable_source() {
    let task = native_task_fixture(b"print('adapter')\n");
    replace(
        &task.path().join("adapters.toml"),
        "argv = [\"tools/adapter.py\"]",
        "argv = [\"python3\", \"tools/adapter.py\"]",
    );

    assert_invalid_package(task.path());
}

#[test]
fn schema_1_0_digest_golden_is_unchanged() {
    let task = tempfile::tempdir().unwrap();
    write_standard_task(task.path(), "1.0");

    assert_eq!(
        import_native_task(task.path())
            .unwrap()
            .task
            .digest
            .as_str(),
        LEGACY_DIGEST
    );
}

fn import_native_task(
    task_root: &Path,
) -> Result<aiperf_runtime::eval::ImportedTask, aiperf_runtime::eval::HarborImportError> {
    let source = HarborSource::local(task_root.to_string_lossy()).unwrap();
    HarborImporter::new(&NativeSourceAcquirer).import(&source)
}

fn assert_invalid_package(task_root: &Path) {
    assert!(matches!(
        import_native_task(task_root),
        Err(aiperf_runtime::eval::HarborImportError::InvalidPackage(_))
    ));
}

fn append(path: &Path, content: &str) {
    let mut current = fs::read_to_string(path).unwrap();
    current.push_str(content);
    fs::write(path, current).unwrap();
}

fn replace(path: &Path, from: &str, to: &str) {
    let current = fs::read_to_string(path).unwrap();
    assert!(current.contains(from));
    fs::write(path, current.replacen(from, to, 1)).unwrap();
}

fn native_task_fixture(adapter: &[u8]) -> tempfile::TempDir {
    let task = tempfile::tempdir().unwrap();
    write_standard_task(task.path(), "1.1");
    fs::write(
        task.path().join("task.toml"),
        r#"schema_version = "1.1"

[task]
name = "example/native-graph"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#,
    )
    .unwrap();
    fs::write(task.path().join("agent_graph.json"), b"{}\n").unwrap();
    fs::write(
        task.path().join("models.toml"),
        r#"[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-default"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "example-model"
urls = ["https://provider.example/v1", "https://fallback.example/v1"]
streaming = true
max_connect_retries = 2
request_timeout_ms = 30000
capture = "redacted_raw"

[model_bindings.tokenizer]
type = "local"
name = "builtin"
revision = "main"
apply_chat_template = true

[[model_bindings.authentication]]
header = "Authorization"
secret = "provider-key"

[[model_bindings.authentication]]
header = "X-Secondary-Authorization"
secret = "provider-secondary-key"

[model_bindings.generation]
temperature = 0.4
top_p = 0.9
top_k = 32
max_tokens = 128
min_tokens = 8
seed = 7
presence_penalty = 0.1
frequency_penalty = 0.2
repetition_penalty = 1.1
"#,
    )
    .unwrap();
    fs::create_dir_all(task.path().join("tools")).unwrap();
    fs::write(
        task.path().join("adapters.toml"),
        r#"[[adapters]]
id = "tool-adapter"
role = "tool"
argv = ["tools/adapter.py"]
executable = "tools/adapter.py"
config = ["tools/config.toml"]
policy = ["tools/policy.toml"]
"#,
    )
    .unwrap();
    fs::write(task.path().join("tools/adapter.py"), adapter).unwrap();
    fs::write(task.path().join("tools/config.toml"), b"enabled = true\n").unwrap();
    fs::write(task.path().join("tools/policy.toml"), b"allow = true\n").unwrap();
    task
}

fn externally_driven_task_fixture() -> tempfile::TempDir {
    let task = tempfile::tempdir().unwrap();
    write_standard_task(task.path(), "1.1");
    fs::write(
        task.path().join("task.toml"),
        r#"schema_version = "1.1"

[task]
name = "example/external-driver"

[native_graph]
profile = "externally_driven"
adapter_manifest = "adapters.toml"
driver = "driver-adapter"
"#,
    )
    .unwrap();
    fs::create_dir_all(task.path().join("tools")).unwrap();
    fs::write(
        task.path().join("adapters.toml"),
        r#"[[adapters]]
id = "driver-adapter"
role = "driver"
argv = ["tools/driver.sh"]
executable = "tools/driver.sh"
"#,
    )
    .unwrap();
    fs::write(task.path().join("tools/driver.sh"), b"#!/bin/sh\nexit 0\n").unwrap();
    task
}

fn server_tokenizer_task_fixture() -> tempfile::TempDir {
    let task = native_task_fixture(b"print('adapter')\n");
    replace(
        &task.path().join("models.toml"),
        "type = \"local\"\nname = \"builtin\"\nrevision = \"main\"\napply_chat_template = true",
        "type = \"server\"\nurl = \"https://tokenizer.example/v1\"\napply_chat_template = true",
    );
    task
}

fn write_standard_task(task_root: &Path, schema_version: &str) {
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(task_root.join("environment/Dockerfile"), b"FROM scratch\n").unwrap();
    fs::write(task_root.join("instruction.md"), b"Do work.\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), b"exit 0\n").unwrap();
    if schema_version == "1.0" {
        fs::write(
            task_root.join("task.toml"),
            "schema_version = \"1.0\"\nartifacts = [\"/work/result.txt\"]\n[task]\nname = \"example/legacy-golden\"\n",
        )
        .unwrap();
    }
}
