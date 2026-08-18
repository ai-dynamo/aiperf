// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::{fs, path::Path};

use aiperf_runtime::eval::{
    ArtifactDigest, HarborImporter, HarborSource, ModelCapturePolicy, NativeSourceAcquirer,
    TokenizerBindingSpec, select_native_graph_external_driver,
};
use aiperf_runtime::extensions::{AIPerfRegistryFactory, BuiltinAIPerfRegistryFactory};

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
fn native_graph_rollout_import_retains_environment_selection_and_seals_every_authored_fact() {
    let baseline = rollout_task_fixture(b"{\"seed\":7}\n");
    let baseline_imported = import_native_task(baseline.path()).unwrap();
    let rollout = baseline_imported
        .package
        .native_graph()
        .and_then(|native| native.rollout())
        .expect("the strict rollout selection is retained from the acquired package snapshot");
    assert_eq!(
        rollout.environment().adapter_id().as_str(),
        "environment-adapter"
    );
    assert_eq!(
        rollout.environment().protocol_factory_id().as_str(),
        "strict_jsonl"
    );
    assert_eq!(
        rollout.environment().runtime_provider_id().as_str(),
        "strict_supervised"
    );
    assert_eq!(
        rollout.environment().stepper_factory_id().as_str(),
        "supervised_environment"
    );
    assert_eq!(
        rollout.environment().action_encoder_id().as_str(),
        "move_v1"
    );
    assert_eq!(rollout.environment().operation_deadline_ms().get(), 5_000);
    assert_eq!(rollout.policy().environment(), "counter-v1");
    assert_eq!(rollout.policy().model_binding_id().as_str(), "primary");
    assert_eq!(
        rollout.policy().prompt_source().path(),
        "rollout/policy.json"
    );
    assert_eq!(
        rollout.policy().prompt_source().bytes(),
        b"{\"instruction\":\"choose a move\"}\n"
    );
    assert_eq!(
        rollout.policy().prompt_source().digest(),
        &ArtifactDigest::from_bytes(b"{\"instruction\":\"choose a move\"}\n")
    );
    assert_eq!(rollout.policy().max_decision_bytes(), 256);
    assert_eq!(rollout.policy().horizon(), 4);
    assert_eq!(rollout.policy().gamma(), 0.75);
    assert_eq!(rollout.limits().max_environment_bytes(), 256);
    assert_eq!(rollout.limits().max_horizon(), 8);
    assert_eq!(rollout.limits().max_prompt_bytes(), 256);
    assert_eq!(
        rollout.environment().reset_source().bytes(),
        b"{\"seed\":7}\n"
    );
    assert_eq!(
        rollout.environment().protocol_limits().max_frame_bytes(),
        4_096
    );
    assert_eq!(
        rollout.environment().artifact_limits().max_total_bytes(),
        16_384
    );

    for (selection, path, from, to) in [
        (
            "environment policy identity",
            "rollout.toml",
            "environment = \"counter-v1\"",
            "environment = \"counter-v2\"",
        ),
        (
            "rollout horizon",
            "rollout.toml",
            "horizon = 4",
            "horizon = 3",
        ),
        (
            "discount factor",
            "rollout.toml",
            "gamma = 0.75",
            "gamma = 0.5",
        ),
        (
            "policy model binding selection",
            "rollout.toml",
            "model_binding_id = \"primary\"",
            "model_binding_id = \"secondary\"",
        ),
        (
            "policy prompt selection",
            "rollout.toml",
            "prompt_source = \"rollout/policy.json\"",
            "prompt_source = \"rollout/alternate-policy.json\"",
        ),
        (
            "policy decision byte cap",
            "rollout.toml",
            "max_decision_bytes = 256",
            "max_decision_bytes = 128",
        ),
        (
            "environment byte cap",
            "rollout.toml",
            "max_environment_bytes = 256",
            "max_environment_bytes = 128",
        ),
        (
            "environment horizon cap",
            "rollout.toml",
            "max_horizon = 8",
            "max_horizon = 4",
        ),
        (
            "policy prompt cap",
            "rollout.toml",
            "max_prompt_bytes = 256",
            "max_prompt_bytes = 128",
        ),
        (
            "workspace patch mutable path",
            "rollout.toml",
            "mutable_paths = [\"result.txt\"]",
            "mutable_paths = [\"state.txt\"]",
        ),
        (
            "workspace patch count cap",
            "rollout.toml",
            "max_patches = 4",
            "max_patches = 3",
        ),
        (
            "workspace patch byte cap",
            "rollout.toml",
            "max_patch_bytes = 4096",
            "max_patch_bytes = 2048",
        ),
        (
            "workspace patch total byte cap",
            "rollout.toml",
            "max_total_patch_bytes = 8192",
            "max_total_patch_bytes = 4096",
        ),
        (
            "environment adapter selection",
            "rollout.toml",
            "adapter_id = \"environment-adapter\"",
            "adapter_id = \"alternate-environment-adapter\"",
        ),
        (
            "operation deadline",
            "rollout.toml",
            "operation_deadline_ms = 5000",
            "operation_deadline_ms = 4000",
        ),
        (
            "protocol factory selection",
            "rollout.toml",
            "protocol_factory_id = \"strict_jsonl\"",
            "protocol_factory_id = \"alternate_jsonl\"",
        ),
        (
            "runtime provider selection",
            "rollout.toml",
            "runtime_provider_id = \"strict_supervised\"",
            "runtime_provider_id = \"alternate_supervised\"",
        ),
        (
            "stepper factory selection",
            "rollout.toml",
            "stepper_factory_id = \"supervised_environment\"",
            "stepper_factory_id = \"alternate_environment\"",
        ),
        (
            "action encoder selection",
            "rollout.toml",
            "action_encoder_id = \"move_v1\"",
            "action_encoder_id = \"alternate_move\"",
        ),
        (
            "protocol frame cap",
            "rollout.toml",
            "max_frame_bytes = 4096",
            "max_frame_bytes = 2048",
        ),
        (
            "protocol identifier cap",
            "rollout.toml",
            "max_identifier_bytes = 128",
            "max_identifier_bytes = 64",
        ),
        (
            "protocol JSON byte cap",
            "rollout.toml",
            "max_json_bytes = 2048",
            "max_json_bytes = 1024",
        ),
        (
            "protocol JSON depth cap",
            "rollout.toml",
            "max_json_depth = 4",
            "max_json_depth = 3",
        ),
        (
            "protocol JSON array cap",
            "rollout.toml",
            "max_json_array_entries = 8",
            "max_json_array_entries = 7",
        ),
        (
            "protocol JSON object cap",
            "rollout.toml",
            "max_json_object_entries = 8",
            "max_json_object_entries = 7",
        ),
        (
            "protocol operation-ledger cap",
            "rollout.toml",
            "max_operation_ledger_entries = 16",
            "max_operation_ledger_entries = 15",
        ),
        (
            "protocol per-operation lineage cap",
            "rollout.toml",
            "max_model_call_lineage_entries = 4",
            "max_model_call_lineage_entries = 3",
        ),
        (
            "protocol session lineage entry cap",
            "rollout.toml",
            "max_session_model_call_lineage_entries = 16",
            "max_session_model_call_lineage_entries = 15",
        ),
        (
            "protocol session lineage byte cap",
            "rollout.toml",
            "max_session_model_call_lineage_bytes = 2048",
            "max_session_model_call_lineage_bytes = 1024",
        ),
        (
            "protocol artifact-handle cap",
            "rollout.toml",
            "max_artifact_handles = 4",
            "max_artifact_handles = 3",
        ),
        (
            "protocol artifact-byte cap",
            "rollout.toml",
            "max_artifact_bytes = 4096",
            "max_artifact_bytes = 2048",
        ),
        (
            "artifact-count cap",
            "rollout.toml",
            "max_artifacts = 8",
            "max_artifacts = 7",
        ),
        (
            "artifact total-byte cap",
            "rollout.toml",
            "max_total_bytes = 16384",
            "max_total_bytes = 8192",
        ),
        (
            "artifact per-entry cap",
            "rollout.toml",
            "max_artifact_bytes = 3072",
            "max_artifact_bytes = 2048",
        ),
        (
            "artifact download-handle cap",
            "rollout.toml",
            "max_download_handles = 4",
            "max_download_handles = 3",
        ),
        (
            "reset source path",
            "rollout.toml",
            "reset_source = \"rollout/reset.json\"",
            "reset_source = \"rollout/alternate-reset.json\"",
        ),
        (
            "reset bytes",
            "rollout/reset.json",
            "{\"seed\":7}",
            "{\"seed\":8}",
        ),
        (
            "policy prompt bytes",
            "rollout/policy.json",
            "choose a move",
            "choose a different move",
        ),
    ] {
        let altered = rollout_task_fixture(b"{\"seed\":7}\n");
        replace(&altered.path().join(path), from, to);
        let altered_imported = import_native_task(altered.path()).unwrap();

        assert_ne!(
            baseline_imported.task.digest, altered_imported.task.digest,
            "{selection} must be sealed into the NativeGraph package identity"
        );
    }
}

#[test]
fn native_graph_rollout_import_retains_sealed_workspace_patch_contract() {
    let task = rollout_task_fixture(b"{\"seed\":7}\n");
    let imported = import_native_task(task.path())
        .expect("sealed workspace patch authoring is retained from the package snapshot");
    let rollout = imported
        .package
        .native_graph()
        .and_then(|native| native.rollout())
        .expect("fixture selects a rollout");

    assert_eq!(rollout.workspace_patch().mutable_paths(), ["result.txt"]);
    assert_eq!(rollout.workspace_patch().max_patches(), 4);
    assert_eq!(rollout.workspace_patch().max_patch_bytes(), 4_096);
    assert_eq!(rollout.workspace_patch().max_total_patch_bytes(), 8_192);
}

#[test]
fn rollout_policy_prompt_exceeding_the_selected_cap_is_refused_at_import() {
    let task = rollout_task_fixture(b"{\"seed\":7}\n");
    replace(
        &task.path().join("rollout.toml"),
        "max_prompt_bytes = 256",
        "max_prompt_bytes = 8",
    );

    let error = import_native_task(task.path())
        .expect_err("a retained policy prompt above the selected cap must fail import");

    assert!(
        error
            .to_string()
            .contains("rollout.policy.prompt_source exceeds rollout.limits.max_prompt_bytes"),
        "unexpected selected prompt-cap rejection: {error}"
    );
}

#[test]
fn native_graph_rollout_rejects_invalid_or_incompatible_authoring_before_provisioning() {
    let external = externally_driven_task_fixture();
    fs::write(external.path().join("rollout.toml"), valid_rollout_toml()).unwrap();
    assert_invalid_package(external.path());

    let malformed = rollout_task_fixture(b"{\"seed\":7}\n");
    append(&malformed.path().join("rollout.toml"), "unknown = true\n");
    assert_invalid_package(malformed.path());

    let zero_deadline = rollout_task_fixture(b"{\"seed\":7}\n");
    replace(
        &zero_deadline.path().join("rollout.toml"),
        "operation_deadline_ms = 5000",
        "operation_deadline_ms = 0",
    );
    assert_invalid_package(zero_deadline.path());

    let invalid_gamma = rollout_task_fixture(b"{\"seed\":7}\n");
    replace(
        &invalid_gamma.path().join("rollout.toml"),
        "gamma = 0.75",
        "gamma = nan",
    );
    assert_invalid_package(invalid_gamma.path());

    let missing_policy_model = rollout_task_fixture(b"{\"seed\":7}\n");
    replace(
        &missing_policy_model.path().join("rollout.toml"),
        "model_binding_id = \"primary\"\n",
        "",
    );
    assert_invalid_package(missing_policy_model.path());

    let unknown_policy_model = rollout_task_fixture(b"{\"seed\":7}\n");
    replace(
        &unknown_policy_model.path().join("rollout.toml"),
        "model_binding_id = \"primary\"",
        "model_binding_id = \"not-declared\"",
    );
    assert_invalid_package(unknown_policy_model.path());

    let missing_policy_prompt = rollout_task_fixture(b"{\"seed\":7}\n");
    replace(
        &missing_policy_prompt.path().join("rollout.toml"),
        "prompt_source = \"rollout/policy.json\"\n",
        "",
    );
    assert_invalid_package(missing_policy_prompt.path());

    let missing_prompt_snapshot = rollout_task_fixture(b"{\"seed\":7}\n");
    replace(
        &missing_prompt_snapshot.path().join("rollout.toml"),
        "prompt_source = \"rollout/policy.json\"",
        "prompt_source = \"rollout/missing-policy.json\"",
    );
    assert_invalid_package(missing_prompt_snapshot.path());

    let zero_decision_limit = rollout_task_fixture(b"{\"seed\":7}\n");
    replace(
        &zero_decision_limit.path().join("rollout.toml"),
        "max_decision_bytes = 256",
        "max_decision_bytes = 0",
    );
    assert_invalid_package(zero_decision_limit.path());

    let unsafe_patch_path = rollout_task_fixture(b"{\"seed\":7}\n");
    replace(
        &unsafe_patch_path.path().join("rollout.toml"),
        "mutable_paths = [\"result.txt\"]",
        "mutable_paths = [\"../result.txt\"]",
    );
    assert_invalid_package(unsafe_patch_path.path());

    let duplicate_patch_path = rollout_task_fixture(b"{\"seed\":7}\n");
    replace(
        &duplicate_patch_path.path().join("rollout.toml"),
        "mutable_paths = [\"result.txt\"]",
        "mutable_paths = [\"result.txt\", \"result.txt\"]",
    );
    assert_invalid_package(duplicate_patch_path.path());

    let zero_patch_limit = rollout_task_fixture(b"{\"seed\":7}\n");
    replace(
        &zero_patch_limit.path().join("rollout.toml"),
        "max_patches = 4",
        "max_patches = 0",
    );
    assert_invalid_package(zero_patch_limit.path());

    let noncanonical_reset = rollout_task_fixture(b"{\"seed\":7}\n");
    replace(
        &noncanonical_reset.path().join("rollout.toml"),
        "reset_source = \"rollout/reset.json\"",
        "reset_source = \"rollout/../rollout/reset.json\"",
    );
    assert_invalid_package(noncanonical_reset.path());

    let external_reset = rollout_task_fixture(b"{\"seed\":7}\n");
    replace(
        &external_reset.path().join("rollout.toml"),
        "reset_source = \"rollout/reset.json\"",
        "reset_source = \"/outside-the-package/reset.json\"",
    );
    assert_invalid_package(external_reset.path());

    let missing_reset = rollout_task_fixture(b"{\"seed\":7}\n");
    replace(
        &missing_reset.path().join("rollout.toml"),
        "reset_source = \"rollout/reset.json\"",
        "reset_source = \"rollout/missing.json\"",
    );
    assert_invalid_package(missing_reset.path());

    let role_mismatch = rollout_task_fixture(b"{\"seed\":7}\n");
    replace(
        &role_mismatch.path().join("rollout.toml"),
        "adapter_id = \"environment-adapter\"",
        "adapter_id = \"tool-adapter\"",
    );
    assert_invalid_package(role_mismatch.path());
}

#[test]
fn rollout_manifest_and_reset_bytes_each_change_the_executable_source_projection() {
    let baseline = rollout_task_fixture(b"{\"seed\":7}\n");
    let baseline_digest = import_native_task(baseline.path())
        .unwrap()
        .package
        .native_graph()
        .unwrap()
        .executable_source_digest()
        .clone();

    let manifest_mutation = rollout_task_fixture(b"{\"seed\":7}\n");
    replace(
        &manifest_mutation.path().join("rollout.toml"),
        "gamma = 0.75",
        "gamma = 0.5",
    );
    let manifest_digest = import_native_task(manifest_mutation.path())
        .unwrap()
        .package
        .native_graph()
        .unwrap()
        .executable_source_digest()
        .clone();
    assert_ne!(baseline_digest, manifest_digest);

    let reset_mutation = rollout_task_fixture(b"{\"seed\":8}\n");
    let reset_digest = import_native_task(reset_mutation.path())
        .unwrap()
        .package
        .native_graph()
        .unwrap()
        .executable_source_digest()
        .clone();
    assert_ne!(baseline_digest, reset_digest);
}

#[test]
fn native_graph_rollout_retains_the_acquired_reset_snapshot_after_origin_mutation() {
    let task = rollout_task_fixture(b"{\"seed\":7}\n");
    let imported = import_native_task(task.path()).unwrap();

    fs::write(task.path().join("rollout.toml"), b"not valid TOML\n").unwrap();
    fs::write(task.path().join("rollout/reset.json"), b"{\"seed\":999}\n").unwrap();
    fs::write(
        task.path().join("rollout/policy.json"),
        b"{\"instruction\":\"choose a different move\"}\n",
    )
    .unwrap();

    let rollout = imported
        .package
        .native_graph()
        .and_then(|native| native.rollout())
        .expect("the imported rollout snapshot remains available after origin mutation");
    assert_eq!(rollout.policy().gamma(), 0.75);
    assert_eq!(
        rollout.environment().reset_source().bytes(),
        b"{\"seed\":7}\n"
    );
    assert_eq!(
        rollout.policy().prompt_source().bytes(),
        b"{\"instruction\":\"choose a move\"}\n"
    );
}

#[test]
fn native_graph_without_rollout_preserves_the_pre_rollout_identity_golden() {
    let task = native_task_fixture(b"print('adapter')\n");

    assert_eq!(
        import_native_task(task.path())
            .unwrap()
            .task
            .digest
            .as_str(),
        "blake3:29a90e3a6e30ac5c9d103912ce3c95baa4bfb3699427f7b575810bb4edb00965"
    );
}

#[test]
fn externally_driven_driver_selection_and_argv_are_immutable_package_identity() {
    let baseline = externally_driven_task_fixture();
    let baseline_imported = import_native_task(baseline.path()).unwrap();
    let baseline_native = baseline_imported.package.native_graph().unwrap();
    let baseline_driver = baseline_native
        .driver_adapter()
        .expect("the declared external driver is retained in the imported snapshot");

    assert_eq!(baseline_driver.id.as_str(), "driver-adapter");
    assert_eq!(baseline_driver.argv, ["tools/driver.sh"]);

    let altered = externally_driven_task_fixture();
    replace(
        &altered.path().join("adapters.toml"),
        "argv = [\"tools/driver.sh\"]",
        "argv = [\"tools/driver.sh\", \"--strict\"]",
    );
    let altered_imported = import_native_task(altered.path()).unwrap();
    let altered_driver = altered_imported
        .package
        .native_graph()
        .unwrap()
        .driver_adapter()
        .expect("the altered driver is retained in the imported snapshot");

    assert_eq!(altered_driver.argv, ["tools/driver.sh", "--strict"]);
    assert_ne!(baseline_imported.task.digest, altered_imported.task.digest);
}

#[test]
fn externally_driven_factory_selection_is_required_and_identity_bound() {
    let missing = externally_driven_task_fixture();
    replace(
        &missing.path().join("task.toml"),
        "external_driver_factory_id = \"refuse\"\n",
        "",
    );
    let error = import_native_task(missing.path()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("externally_driven profile requires native_graph.external_driver_factory_id"),
        "an external package without an explicit factory selector must refuse: {error}"
    );

    let baseline = externally_driven_task_fixture();
    let baseline_imported = import_native_task(baseline.path()).unwrap();

    let altered = externally_driven_task_fixture();
    replace(
        &altered.path().join("task.toml"),
        "external_driver_factory_id = \"refuse\"",
        "external_driver_factory_id = \"other-driver\"",
    );
    let altered_imported = import_native_task(altered.path()).unwrap();

    assert_ne!(
        baseline_imported.task.digest, altered_imported.task.digest,
        "the immutable package identity must bind the exact external-driver factory selector"
    );
}

#[test]
fn externally_driven_factory_preflight_resolves_only_the_exact_immutable_selector() {
    let registry = BuiltinAIPerfRegistryFactory
        .build()
        .expect("the built-in registry is available");
    let selected = externally_driven_task_fixture();
    let selected_package = import_native_task(selected.path())
        .unwrap()
        .package
        .native_graph()
        .unwrap()
        .clone();

    let factory = select_native_graph_external_driver(&registry, &selected_package)
        .expect("the sealed built-in selector resolves exactly");
    assert_eq!(factory.id(), "refuse");

    let unknown = externally_driven_task_fixture();
    replace(
        &unknown.path().join("task.toml"),
        "external_driver_factory_id = \"refuse\"",
        "external_driver_factory_id = \"unregistered\"",
    );
    let unknown_package = import_native_task(unknown.path())
        .unwrap()
        .package
        .native_graph()
        .unwrap()
        .clone();
    let error = match select_native_graph_external_driver(&registry, &unknown_package) {
        Ok(_) => panic!("an unregistered selector must not fall back to another driver"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("unknown external driver factory")
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

fn rollout_task_fixture(reset: &[u8]) -> tempfile::TempDir {
    let task = native_task_fixture(b"print('adapter')\n");
    append(
        &task.path().join("models.toml"),
        r#"
[[model_bindings]]
id = "secondary"
endpoint_profile_id = "provider-secondary"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "alternate-model"
urls = ["https://alternate.example/v1"]
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
    append(
        &task.path().join("adapters.toml"),
        r#"
[[adapters]]
id = "environment-adapter"
role = "environment"
argv = ["tools/environment.py"]
executable = "tools/environment.py"

[[adapters]]
id = "alternate-environment-adapter"
role = "environment"
argv = ["tools/alternate-environment.py"]
executable = "tools/alternate-environment.py"
"#,
    );
    fs::write(
        task.path().join("tools/environment.py"),
        b"print('environment')\n",
    )
    .unwrap();
    fs::write(
        task.path().join("tools/alternate-environment.py"),
        b"print('alternate environment')\n",
    )
    .unwrap();
    fs::create_dir_all(task.path().join("rollout")).unwrap();
    fs::write(task.path().join("rollout/reset.json"), reset).unwrap();
    fs::write(task.path().join("rollout/alternate-reset.json"), reset).unwrap();
    fs::write(
        task.path().join("rollout/policy.json"),
        b"{\"instruction\":\"choose a move\"}\n",
    )
    .unwrap();
    fs::write(
        task.path().join("rollout/alternate-policy.json"),
        b"{\"instruction\":\"choose an alternate move\"}\n",
    )
    .unwrap();
    fs::write(task.path().join("rollout.toml"), valid_rollout_toml()).unwrap();
    task
}

fn valid_rollout_toml() -> &'static str {
    r#"
[environment]
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
horizon = 4
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
"#
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
external_driver_factory_id = "refuse"
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
