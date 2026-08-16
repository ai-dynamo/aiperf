// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Pure contracts for normalized Harbor benchmark execution plans.

use std::{cell::RefCell, collections::BTreeMap, fs};

use aiperf_runtime::eval::{
    EnvBinding, EnvName, EvalExecutionError, HarborImporter, HarborSandboxRecipe, HarborSource,
    LocalProcessSandbox, NativeSourceAcquirer, NetworkPolicy, SecretProvider, SecretValue,
    VerifierMode, resolve_phase_environment,
};

#[test]
fn normalizes_equivalent_allowlist_spelling_deterministically() {
    let policy =
        NetworkPolicy::allowlist(["EXAMPLE.com", "*.Example.ORG", "2001:DB8::1", "10.0.0.0/24"])
            .unwrap();

    assert_eq!(
        policy
            .allowed_hosts()
            .unwrap()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["*.example.org", "10.0.0.0/24", "2001:db8::1", "example.com"]
    );
}

#[test]
fn environment_binding_never_captures_the_host_secret_value() {
    let binding = EnvBinding::parse("${HOST_API_TOKEN}").unwrap();

    assert_eq!(binding.secret_reference(), Some("HOST_API_TOKEN"));
    assert!(!format!("{binding:?}").contains("super-secret"));
}

#[derive(Default)]
struct RecordingSecrets {
    requested: RefCell<Vec<String>>,
    values: BTreeMap<String, String>,
}

impl SecretProvider for RecordingSecrets {
    fn resolve(&self, name: &EnvName) -> Result<SecretValue, EvalExecutionError> {
        self.requested.borrow_mut().push(name.to_string());
        self.values
            .get(name.as_str())
            .cloned()
            .map(SecretValue::new)
            .ok_or_else(|| EvalExecutionError::MissingSecret(name.to_string()))
    }
}

#[test]
fn resolves_only_active_phase_secrets_and_keeps_later_phase_bindings_independent() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(
        &temporary,
        r#"[environment.env]
BASE = "base"
TOKEN = "${BASE_TOKEN}"

[agent.env]
BASE = "agent"
AGENT_TOKEN = "${AGENT_TOKEN}"

[verifier.env]
BASE = "verifier"
VERIFY_TOKEN = "${VERIFY_TOKEN}"
"#,
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let plan = imported.package.execution_plan();
    let secrets = RecordingSecrets {
        values: BTreeMap::from([
            ("BASE_TOKEN".to_owned(), "base-secret".to_owned()),
            ("AGENT_TOKEN".to_owned(), "agent-secret".to_owned()),
            ("VERIFY_TOKEN".to_owned(), "verify-secret".to_owned()),
        ]),
        ..Default::default()
    };

    let agent = resolve_phase_environment(plan.environment(), plan.agent(), &secrets).unwrap();
    assert_eq!(
        secrets.requested.borrow().as_slice(),
        ["AGENT_TOKEN", "BASE_TOKEN"]
    );
    assert_eq!(
        agent.public().get("BASE").map(String::as_str),
        Some("agent")
    );
    assert_eq!(agent.secret_names(), ["AGENT_TOKEN", "TOKEN"]);

    let verifier = resolve_phase_environment(
        plan.verifier().environment(),
        plan.verifier().phase(),
        &secrets,
    )
    .unwrap();
    assert_eq!(
        secrets.requested.borrow().as_slice(),
        ["AGENT_TOKEN", "BASE_TOKEN", "BASE_TOKEN", "VERIFY_TOKEN"]
    );
    assert_eq!(
        verifier.public().get("BASE").map(String::as_str),
        Some("verifier")
    );
    assert_eq!(verifier.secret_names(), ["TOKEN", "VERIFY_TOKEN"]);
}

#[test]
fn missing_secret_names_the_binding_without_rendering_secret_values() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "[agent.env]\nTOKEN = \"${MISSING_TOKEN}\"\n");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let secrets = RecordingSecrets::default();

    let error = resolve_phase_environment(
        imported.package.execution_plan().environment(),
        imported.package.execution_plan().agent(),
        &secrets,
    )
    .unwrap_err();

    assert_eq!(
        error,
        EvalExecutionError::MissingSecret("MISSING_TOKEN".to_owned())
    );
    assert!(error.to_string().contains("MISSING_TOKEN"));
    assert!(!error.to_string().contains("secret-value"));
}

#[test]
fn local_sandbox_refuses_standard_tasks_before_running_commands() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "[agent]\nnetwork = \"no-network\"\n");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();

    assert_eq!(
        LocalProcessSandbox::new().execute(&recipe, &imported.package, VerifierMode::Shared),
        Err(EvalExecutionError::UnsupportedEnforcement("docker"))
    );
}

fn standard_task_root(temporary: &tempfile::TempDir, manifest_suffix: &str) -> std::path::PathBuf {
    let task_root = temporary.path().join("task");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        format!("schema_version = \"1.0\"\n\n[task]\nname = \"example/task\"\n\n{manifest_suffix}"),
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Do work.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();
    task_root
}
