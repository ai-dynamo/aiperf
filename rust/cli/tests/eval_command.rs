// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;
use std::sync::Mutex;
use std::time::Duration;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use aiperf_runtime::{
    engine::application::Application,
    eval::{
        ArtifactDigest, EngineNativeGraphEpisodeCallback, EnvName, EvalExecutionError,
        EvalNodeRecordArtifact, HarborImporter, HarborSource, LocalExecutionResult,
        ModelRuntimeConfig, MultiStepExecutionResult, NativeGraphEpisodeCallback,
        NativeGraphEpisodeLease, NativeSourceAcquirer, RewardDocument, SecretProvider, SecretValue,
        StepExecutionResult,
    },
};
use serde_json::json;

static DOCKER_TIMEOUT_TEST_LOCK: Mutex<()> = Mutex::new(());

fn write_externally_driven_task(task_root: &Path) {
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::create_dir_all(task_root.join("tools")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.1\"\n[task]\nname = \"example/external-driver\"\n[native_graph]\nprofile = \"externally_driven\"\nadapter_manifest = \"adapters.toml\"\ndriver = \"driver-adapter\"\nexternal_driver_factory_id = \"terminal_v1\"\n[verifier]\nenvironment_mode = \"separate\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Do work.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();
    fs::write(
        task_root.join("adapters.toml"),
        "[[adapters]]\nid = \"driver-adapter\"\nrole = \"driver\"\nargv = [\"tools/driver.sh\"]\nexecutable = \"tools/driver.sh\"\n",
    )
    .unwrap();
    fs::write(task_root.join("tools/driver.sh"), "#!/bin/sh\nexit 0\n").unwrap();
}

fn write_externally_driven_lifecycle(path: &Path, command: &[&str]) {
    let policy = ArtifactDigest::from_bytes(b"external-policy");
    let command = serde_json::to_string(command).unwrap();
    fs::write(
        path,
        format!(
            r#"{{"version":1,"agent_variant":"external-driver","model":{{"provider":"external","model":"opaque"}},"seed":7,"policy":"{policy}","runtime":"external:v1","attempt":"attempt-1","budget":{{"execution_seconds":2.0,"verifier_seconds":3.0}},"agent_contract":"externally_driven","command":{command},"initial_score":{{"metric":"reward","rationale":"{policy}"}},"regrade":{{"metric":"reward","rationale":"{policy}"}}}}"#,
            policy = policy.as_str(),
        ),
    )
    .unwrap();
}

fn reward<const N: usize>(metrics: [(&str, f64); N]) -> RewardDocument {
    RewardDocument::new(
        metrics
            .into_iter()
            .map(|(name, value)| (name.to_owned(), value))
            .collect::<BTreeMap<_, _>>(),
    )
    .expect("test reward is finite and nonempty")
}

#[test]
fn native_eval_single_step_serialization_retains_its_exact_json_contract() {
    let result = LocalExecutionResult {
        artifacts: vec![(
            "result.txt".to_owned(),
            ArtifactDigest::from_bytes(b"result"),
        )],
        reward: reward([("score", 1.0)]),
        verifier: ArtifactDigest::from_bytes(b"verifier"),
    };

    let output = aiperf_cli::eval::serialize_eval_result(
        "example/single",
        aiperf_cli::eval::EvalExecutionResult::Single(result),
    )
    .expect("single-step evaluation result serializes");

    assert_eq!(
        output,
        json!({
            "task": "example/single",
            "artifacts": [["result.txt", ArtifactDigest::from_bytes(b"result").as_str()]],
            "reward": {"score": 1.0},
        })
    );
}

#[test]
fn native_eval_multi_step_serialization_reports_ordered_sanitized_step_results() {
    let first_artifact = ArtifactDigest::from_bytes(b"first");
    let final_artifact = ArtifactDigest::from_bytes(b"final");
    let result = MultiStepExecutionResult {
        steps: vec![
            StepExecutionResult {
                name: "prepare".to_owned(),
                artifacts: vec![("prepare.txt".to_owned(), first_artifact.clone())],
                reward: reward([("quality", 0.5)]),
            },
            StepExecutionResult {
                name: "finish".to_owned(),
                artifacts: vec![("result.txt".to_owned(), final_artifact.clone())],
                reward: reward([("quality", 1.0), ("speed", 0.75)]),
            },
        ],
        reward: reward([("quality", 0.75), ("speed", 0.375)]),
        verifier: ArtifactDigest::from_bytes(b"verifier"),
    };

    let output = aiperf_cli::eval::serialize_eval_result(
        "example/multi",
        aiperf_cli::eval::EvalExecutionResult::MultiStep(result),
    )
    .expect("multi-step evaluation result serializes");

    assert_eq!(
        output,
        json!({
            "task": "example/multi",
            "artifacts": [["result.txt", final_artifact.as_str()]],
            "reward": {"quality": 0.75, "speed": 0.375},
            "steps": [
                {
                    "name": "prepare",
                    "artifacts": [["prepare.txt", first_artifact.as_str()]],
                    "reward": {"quality": 0.5},
                },
                {
                    "name": "finish",
                    "artifacts": [["result.txt", final_artifact.as_str()]],
                    "reward": {"quality": 1.0, "speed": 0.75},
                },
            ],
        })
    );
    let serialized = output.to_string();
    assert!(!serialized.contains("instruction"));
    assert!(!serialized.contains("secret"));
}

#[test]
fn native_eval_refuses_standard_multi_step_tasks_locally_before_starting_the_agent() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("multi-step-local");
    let started = temporary.path().join("agent-started");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::create_dir_all(task_root.join("steps/prepare/tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/multi-step-local\"\n[[steps]]\nname = \"prepare\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Root instruction.\n").unwrap();
    fs::write(
        task_root.join("steps/prepare/instruction.md"),
        "Prepare the result.\n",
    )
    .unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        "--agent-command".to_owned(),
        format!("touch {}", started.display()),
        "--sandbox".to_owned(),
        "local".to_owned(),
    ])
    .expect_err("local execution cannot enforce a standard multi-step task");

    assert!(matches!(
        error.downcast_ref::<aiperf_runtime::eval::EvalExecutionError>(),
        Some(aiperf_runtime::eval::EvalExecutionError::UnsupportedMultiStep)
    ));
    assert!(!started.exists());
}

#[test]
fn eval_command_without_records_output_retains_reward_json_contract() {
    let temporary = tempfile::tempdir().unwrap();
    let package_path = temporary.path().join("task.json");
    fs::write(
        &package_path,
        br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["true"],"verifier_command":["sh","-c","printf '{\"reward\":1.0}' > reward.json"],"declared_artifacts":[]}"#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .args(["eval", "--task"])
        .arg(&package_path)
        .args([
            "--image",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--verifier-mode",
            "shared",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "eval failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reward: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        reward
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["artifacts", "reward", "task"])
    );
    assert_eq!(reward["reward"]["reward"], 1.0);
}

#[test]
fn eval_command_rejects_records_output_for_non_native_graph_before_provisioning() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("standard-task");
    let started = temporary.path().join("agent-started");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/standard\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Do work.\n").unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "not a Dockerfile\n",
    )
    .unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--agent-command".to_owned(),
        format!("touch {}", started.display()),
        "--records-output".to_owned(),
        temporary
            .path()
            .join("records.jsonl")
            .to_string_lossy()
            .into_owned(),
    ])
    .expect_err("record output is available only to schema-1.1 NativeGraph evaluation");

    assert!(
        error
            .to_string()
            .contains("--records-output is available only for schema-1.1 NativeGraph evaluation"),
        "unexpected records-output refusal: {error:#}"
    );
    assert!(!started.exists());
}

#[test]
fn eval_command_rejects_invalid_records_output_before_docker_provisioning() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("native-graph");
    write_native_graph_task(
        &task_root,
        "https://provider.example/v1",
        "not a Dockerfile\n",
    );
    let runtime_path = temporary.path().join("model-runtime.toml");
    fs::write(&runtime_path, "version = 1\n").unwrap();
    let lifecycle_path = temporary.path().join("lifecycle.json");
    write_native_graph_lifecycle(&lifecycle_path);
    let parent_file = temporary.path().join("not-a-directory");
    fs::write(&parent_file, "blocking file\n").unwrap();
    let records_path = parent_file.join("records.jsonl");

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--model-runtime".to_owned(),
        runtime_path.to_string_lossy().into_owned(),
        "--lifecycle-request".to_owned(),
        lifecycle_path.to_string_lossy().into_owned(),
        "--records-output".to_owned(),
        records_path.to_string_lossy().into_owned(),
    ])
    .expect_err("an invalid record destination must fail before Docker provisioning");

    assert!(
        error
            .to_string()
            .contains("native eval node record export directory"),
        "unexpected record destination error: {error:#}"
    );
}

struct EmptyNativeGraphSecrets;

impl SecretProvider for EmptyNativeGraphSecrets {
    fn resolve(&self, name: &EnvName) -> Result<SecretValue, EvalExecutionError> {
        Err(EvalExecutionError::MissingSecret(name.clone()))
    }
}

struct AcquiredNativeGraphLease;

impl NativeGraphEpisodeLease for AcquiredNativeGraphLease {
    fn is_authorized(&self) -> bool {
        true
    }

    fn is_environment_acquired(&self) -> bool {
        true
    }

    fn instruction(&self) -> &str {
        "Complete the graph."
    }
}

#[test]
fn eval_command_native_graph_records_output_is_parseable_without_docker() {
    use std::io::{Read as _, Write as _};
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0_u8; 8192];
        let _ = stream.read(&mut request).unwrap();
        let body = br#"{"id":"completion-1","object":"chat.completion","created":1,"model":"example-model","choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
        write!(
            stream,
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            body.len()
        )
        .unwrap();
        stream.write_all(body).unwrap();
    });

    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("native-graph");
    write_native_graph_task(&task_root, &endpoint, "FROM scratch\n");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let native = imported.package.native_graph().unwrap();
    let model_runtime: ModelRuntimeConfig = toml::from_str("version = 1\n").unwrap();
    let application = Application::stock(format!("blake3:{}", "5".repeat(64))).unwrap();
    let records_path = temporary.path().join("records.jsonl");
    let artifact = EvalNodeRecordArtifact::open(&records_path).unwrap();
    let mut callback = EngineNativeGraphEpisodeCallback::new(
        &application,
        native,
        &model_runtime,
        &EmptyNativeGraphSecrets,
        Some(artifact.clone()),
    )
    .unwrap();
    let mut lease = AcquiredNativeGraphLease;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime
        .block_on(callback.run(&mut lease))
        .expect("the NativeGraph callback writes its suite-owned record artifact");
    artifact.finish().unwrap();

    let rows = fs::read_to_string(&records_path).unwrap();
    let rows = rows
        .lines()
        .map(serde_json::from_str::<serde_json::Value>)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(rows.len(), 1);
}

#[test]
fn native_eval_requires_model_runtime_for_schema_1_1_before_provisioning() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("native-graph");
    let started = temporary.path().join("agent-started");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::create_dir_all(task_root.join("tools")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.1\"\n[task]\nname = \"example/native-graph\"\n[native_graph]\nprofile = \"native_graph\"\nprogram = \"agent_graph.json\"\nmodel_bindings = \"models.toml\"\nadapter_manifest = \"adapters.toml\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Do work.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();
    fs::write(task_root.join("agent_graph.json"), "{}\n").unwrap();
    fs::write(
        task_root.join("models.toml"),
        "[[model_bindings]]\nid = \"primary\"\nendpoint_profile_id = \"provider-default\"\nendpoint_factory_id = \"chat\"\ntransport_factory_id = \"http\"\nmodel = \"example-model\"\nurls = [\"https://provider.example/v1\"]\nstreaming = true\nrequest_timeout_ms = 30000\ncapture = \"metadata\"\n[model_bindings.tokenizer]\ntype = \"local\"\nname = \"builtin\"\nrevision = \"main\"\n[model_bindings.generation]\n",
    )
    .unwrap();
    fs::write(
        task_root.join("adapters.toml"),
        "[[adapters]]\nid = \"tool-adapter\"\nrole = \"tool\"\nargv = [\"tools/adapter.py\"]\nexecutable = \"tools/adapter.py\"\n",
    )
    .unwrap();
    fs::write(task_root.join("tools/adapter.py"), "#!/bin/sh\nexit 0\n").unwrap();

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--agent-command".to_owned(),
        format!("touch {}", started.display()),
    ])
    .expect_err("schema-1.1 NativeGraph must reject before Docker without host runtime mapping");

    assert!(
        error
            .to_string()
            .contains("--model-runtime is required for schema-1.1 NativeGraph evaluation"),
        "unexpected NativeGraph refusal: {error:#}"
    );
    assert!(!started.exists());
}

#[test]
fn externally_driven_eval_rejects_agent_command_before_model_runtime_or_provisioning() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("external-driver");
    let started = temporary.path().join("agent-started");
    let lifecycle_path = temporary.path().join("lifecycle.json");
    write_externally_driven_task(&task_root);
    write_externally_driven_lifecycle(&lifecycle_path, &["tools/driver.sh"]);

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--lifecycle-request".to_owned(),
        lifecycle_path.to_string_lossy().into_owned(),
        "--agent-command".to_owned(),
        format!("touch {}", started.display()),
    ])
    .expect_err("the immutable external driver argv must reject a caller command before Docker");

    assert!(
        error
            .to_string()
            .contains("NativeGraph lifecycle contracts do not accept --agent-command"),
        "unexpected external-profile refusal: {error:#}"
    );
    assert!(!started.exists());
}

#[test]
fn externally_driven_eval_enters_the_runner_without_model_runtime_before_provisioning() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("external-driver");
    let lifecycle_path = temporary.path().join("lifecycle.json");
    write_externally_driven_task(&task_root);
    write_externally_driven_lifecycle(&lifecycle_path, &["tools/driver.sh"]);

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--lifecycle-request".to_owned(),
        lifecycle_path.to_string_lossy().into_owned(),
        "--image".to_owned(),
        "sha256:bad".to_owned(),
    ])
    .expect_err("the invalid immutable image must fail only after external runner composition");

    assert!(
        error
            .to_string()
            .contains("invalid sandbox recipe image digest"),
        "unexpected external-profile refusal: {error:#}"
    );
    assert!(
        !error.to_string().contains("--model-runtime is required"),
        "external compatibility preflight must not require a Rust model runtime: {error:#}"
    );
}

#[test]
fn externally_driven_eval_rejects_shared_verifier_before_provisioning() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("external-driver");
    let lifecycle_path = temporary.path().join("lifecycle.json");
    write_externally_driven_task(&task_root);
    let manifest = fs::read_to_string(task_root.join("task.toml")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        manifest.replace(
            "environment_mode = \"separate\"",
            "environment_mode = \"shared\"",
        ),
    )
    .unwrap();
    write_externally_driven_lifecycle(&lifecycle_path, &["tools/driver.sh"]);

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--lifecycle-request".to_owned(),
        lifecycle_path.to_string_lossy().into_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
    ])
    .expect_err("a shared verifier cannot be isolated from the external Driver container");

    assert!(
        error
            .to_string()
            .contains("external Driver shared verifier isolation"),
        "unexpected external-profile refusal: {error:#}"
    );
}

#[test]
fn externally_driven_eval_rejects_model_runtime_before_provisioning() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("external-driver");
    let lifecycle_path = temporary.path().join("lifecycle.json");
    let runtime_path = temporary.path().join("model-runtime.toml");
    write_externally_driven_task(&task_root);
    write_externally_driven_lifecycle(&lifecycle_path, &["tools/driver.sh"]);
    fs::write(&runtime_path, "version = 1\n").unwrap();

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--lifecycle-request".to_owned(),
        lifecycle_path.to_string_lossy().into_owned(),
        "--model-runtime".to_owned(),
        runtime_path.to_string_lossy().into_owned(),
    ])
    .expect_err("an external task must not accept a Rust model runtime");

    assert!(
        error
            .to_string()
            .contains("externally driven NativeGraph evaluation does not accept --model-runtime"),
        "unexpected external-profile refusal: {error:#}"
    );
}

#[test]
fn externally_driven_eval_rejects_an_unregistered_factory_before_runner_or_provisioning() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("external-driver");
    let lifecycle_path = temporary.path().join("lifecycle.json");
    write_externally_driven_task(&task_root);
    let task_toml = fs::read_to_string(task_root.join("task.toml")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        task_toml.replace(
            "external_driver_factory_id = \"terminal_v1\"",
            "external_driver_factory_id = \"unregistered\"",
        ),
    )
    .unwrap();
    write_externally_driven_lifecycle(&lifecycle_path, &["tools/driver.sh"]);

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--lifecycle-request".to_owned(),
        lifecycle_path.to_string_lossy().into_owned(),
    ])
    .expect_err("an unregistered factory must refuse before the compatibility runner boundary");

    assert!(
        error
            .to_string()
            .contains("unknown external driver factory \"unregistered\""),
        "the selected factory must fail closed before generic execution: {error:#}"
    );
    assert!(
        !error
            .to_string()
            .contains("compatibility runner is not enabled"),
        "the generic compatibility runner must not be reached: {error:#}"
    );
}

#[test]
fn externally_driven_eval_refuses_authored_suite_execution_before_provisioning() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("external-driver");
    let lifecycle_path = temporary.path().join("lifecycle.json");
    let runtime_path = temporary.path().join("model-runtime.toml");
    let suite_path = temporary.path().join("external-suite.toml");
    write_externally_driven_task(&task_root);
    write_externally_driven_lifecycle(&lifecycle_path, &["tools/driver.sh"]);
    fs::write(&runtime_path, "version = 1\n").unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    fs::write(
        &suite_path,
        format!(
            r#"[defaults]
runtime = "external:v1"
execution_seconds = 2.0
verifier_seconds = 3.0
environment = "{}"
verifier = "{}"

[limits]
parallelism = 1
cpu_units = 1
memory_bytes = 1
max_expanded_trials = 1
model_binding_units = {{}}

[[tasks]]
source = {{ kind = "local", path = {:?} }}
task_id = "{}"
task_digest = "{}"
graph_axes = ["external-driver"]
model_axes = ["opaque"]
policy_axes = ["{}"]
seeds = [7]
repetitions = 1

[tasks.resources]
cpu_units = 1
memory_bytes = 1
model_binding_units = {{}}
"#,
            ArtifactDigest::from_bytes(b"environment").as_str(),
            ArtifactDigest::from_bytes(b"verifier").as_str(),
            task_root.to_string_lossy(),
            imported.task.id.as_str(),
            imported.task.digest.as_str(),
            ArtifactDigest::from_bytes(b"policy").as_str(),
        ),
    )
    .unwrap();

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--suite".to_owned(),
        suite_path.to_string_lossy().into_owned(),
        "--model-runtime".to_owned(),
        runtime_path.to_string_lossy().into_owned(),
        "--lifecycle-request".to_owned(),
        lifecycle_path.to_string_lossy().into_owned(),
    ])
    .expect_err("authored external suites remain outside the compatibility slice");

    assert!(
        error
            .to_string()
            .contains("externally driven NativeGraph --suite execution is not supported"),
        "unexpected external-suite refusal: {error:#}"
    );
}

#[test]
fn externally_driven_eval_rejects_lifecycle_command_that_disagrees_with_manifest_driver() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("external-driver");
    let lifecycle_path = temporary.path().join("lifecycle.json");
    write_externally_driven_task(&task_root);
    write_externally_driven_lifecycle(&lifecycle_path, &["tools/other-driver.sh"]);

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--lifecycle-request".to_owned(),
        lifecycle_path.to_string_lossy().into_owned(),
    ])
    .expect_err("the lifecycle record must bind the manifest-selected external driver argv");

    assert!(
        error
            .to_string()
            .contains("lifecycle command provenance disagrees with the manifest driver"),
        "unexpected external-profile refusal: {error:#}"
    );
}

#[test]
fn native_eval_suite_rejects_multiple_lifecycles_before_provisioning() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("native-graph-suite");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.1\"\n[task]\nname = \"example/native-graph-suite\"\n[native_graph]\nprofile = \"native_graph\"\nprogram = \"agent_graph.json\"\nmodel_bindings = \"models.toml\"\nadapter_manifest = \"adapters.toml\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Do work.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();
    fs::write(task_root.join("agent_graph.json"), "{}\n").unwrap();
    fs::write(
        task_root.join("models.toml"),
        "[[model_bindings]]\nid = \"primary\"\nendpoint_profile_id = \"provider-default\"\nendpoint_factory_id = \"chat\"\ntransport_factory_id = \"http\"\nmodel = \"example-model\"\nurls = [\"https://provider.example/v1\"]\nstreaming = true\nrequest_timeout_ms = 30000\ncapture = \"metadata\"\n[model_bindings.tokenizer]\ntype = \"local\"\nname = \"builtin\"\nrevision = \"main\"\n[model_bindings.generation]\n",
    )
    .unwrap();
    fs::write(task_root.join("adapters.toml"), "").unwrap();

    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .expect("the suite fixture imports once into its owned snapshot");
    let suite_path = temporary.path().join("native-graph-suite.toml");
    let policy = ArtifactDigest::from_bytes(b"policy");
    fs::write(
        &suite_path,
        format!(
            r#"[defaults]
runtime = "native:v1"
execution_seconds = 2.0
verifier_seconds = 3.0
environment = "{environment}"
verifier = "{verifier}"

[limits]
parallelism = 1
cpu_units = 1
memory_bytes = 1
max_expanded_trials = 2

[limits.model_binding_units]
primary = 1

[[tasks]]
source = {{ kind = "local", path = {task_path:?} }}
task_id = "{task_id}"
task_digest = "{task_digest}"
graph_axes = ["graph-a"]
model_axes = ["primary"]
policy_axes = ["{policy}"]
seeds = [7]
repetitions = 2

[tasks.resources]
cpu_units = 1
memory_bytes = 1
model_binding_units = {{ primary = 1 }}
"#,
            environment = ArtifactDigest::from_bytes(b"environment").as_str(),
            verifier = ArtifactDigest::from_bytes(b"verifier").as_str(),
            task_path = task_root.to_string_lossy(),
            task_id = imported.task.id.as_str(),
            task_digest = imported.task.digest.as_str(),
            policy = policy.as_str(),
        ),
    )
    .unwrap();
    let runtime_path = temporary.path().join("model-runtime.toml");
    fs::write(&runtime_path, "version = 1\n").unwrap();
    let lifecycle_path = temporary.path().join("lifecycle.json");
    fs::write(
        &lifecycle_path,
        format!(
            r#"{{"version":1,"agent_variant":"native-graph","model":{{"provider":"provider","model":"model"}},"seed":7,"policy":"{policy}","runtime":"native:v1","attempt":"attempt-1","budget":{{"execution_seconds":2.0,"verifier_seconds":3.0}},"agent_contract":"native_graph","command":["native-graph"],"initial_score":{{"metric":"reward","rationale":"{policy}"}},"regrade":{{"metric":"reward","rationale":"{policy}"}}}}"#,
            policy = policy.as_str(),
        ),
    )
    .unwrap();

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--suite".to_owned(),
        suite_path.to_string_lossy().into_owned(),
        "--model-runtime".to_owned(),
        runtime_path.to_string_lossy().into_owned(),
        "--lifecycle-request".to_owned(),
        lifecycle_path.to_string_lossy().into_owned(),
    ])
    .expect_err("a multi-trial suite has no singular lifecycle provenance");

    assert!(
        error
            .to_string()
            .contains("requires exactly one lifecycle-addressable trial"),
        "unexpected suite refusal: {error:#}"
    );
}

#[test]
fn native_eval_rejects_an_explicit_mode_that_conflicts_with_a_later_multi_step_verifier() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("mixed-verifier-modes");
    let started = temporary.path().join("agent-started");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::create_dir_all(task_root.join("steps/prepare")).unwrap();
    fs::create_dir_all(task_root.join("steps/finish")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/mixed-verifier-modes\"\n[verifier]\nenvironment_mode = \"separate\"\n[[steps]]\nname = \"prepare\"\n[[steps]]\nname = \"finish\"\n[steps.verifier]\nenvironment_mode = \"shared\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Root instruction.\n").unwrap();
    fs::write(
        task_root.join("steps/prepare/instruction.md"),
        "Prepare the result.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("steps/finish/instruction.md"),
        "Finish the result.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "not a Dockerfile\n",
    )
    .unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--agent-command".to_owned(),
        format!("touch {}", started.display()),
        "--verifier-mode".to_owned(),
        "separate".to_owned(),
    ])
    .expect_err("an explicit mode must match every multi-step verifier");

    assert!(
        error
            .to_string()
            .contains("--verifier-mode conflicts with the standard task"),
        "unexpected verifier-mode error: {error:#}"
    );
    assert!(!started.exists());
}

#[test]
fn native_eval_command_runs_a_local_harbor_package() {
    let temporary = tempfile::tempdir().unwrap();
    let package_path = temporary.path().join("task.json");
    fs::write(
        &package_path,
        br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","printf patch > \"$AIPERF_EVAL_ROOT/results/patch.diff\""],"verifier_command":["sh","-c","test -f results/patch.diff && printf '{\"reward\":1.0}' > reward.json"],"declared_artifacts":["/results/patch.diff"]}"#,
    )
    .unwrap();

    let exit = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        package_path.to_string_lossy().into_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        "--verifier-mode".to_owned(),
        "shared".to_owned(),
    ])
    .unwrap();

    assert_eq!(exit, 0);
}

#[test]
fn native_eval_refuses_a_local_separate_verifier_before_running_the_agent() {
    let temporary = tempfile::tempdir().unwrap();
    let package_path = temporary.path().join("task.json");
    let started = temporary.path().join("agent-started");
    fs::write(
        &package_path,
        format!(
            r#"{{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","touch {}"],"verifier_command":["sh","-c","true"],"declared_artifacts":[]}}"#,
            started.display(),
        ),
    )
    .unwrap();

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        package_path.to_string_lossy().into_owned(),
        "--sandbox".to_owned(),
        "local".to_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
    ])
    .expect_err("local execution must not claim separate verifier isolation");

    assert!(matches!(
        error.downcast_ref::<aiperf_runtime::eval::EvalExecutionError>(),
        Some(
            aiperf_runtime::eval::EvalExecutionError::UnsupportedEnforcement(
                "separate verifier isolation"
            )
        )
    ));
    assert!(!started.exists());
}

#[test]
fn native_eval_requires_an_image_for_default_separate_legacy_verification() {
    let temporary = tempfile::tempdir().unwrap();
    let package_path = temporary.path().join("task.json");
    fs::write(
        &package_path,
        br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["true"],"verifier_command":["true"],"declared_artifacts":[]}"#,
    )
    .unwrap();

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        package_path.to_string_lossy().into_owned(),
    ])
    .expect_err("separate legacy verification requires a concrete Docker image");

    assert!(error.to_string().contains("--image is required"));
}

#[test]
fn native_eval_command_runs_a_pinned_git_harbor_package() {
    let temporary = tempfile::tempdir().unwrap();
    let repository = temporary.path().join("tasks");
    fs::create_dir(&repository).unwrap();
    run_git(&repository, ["init"]);
    run_git(
        &repository,
        ["config", "user.email", "eval@example.invalid"],
    );
    run_git(&repository, ["config", "user.name", "Native Eval"]);
    fs::write(
        repository.join("task.json"),
        br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","printf patch > \"$AIPERF_EVAL_ROOT/results/patch.diff\""],"verifier_command":["sh","-c","test -f results/patch.diff && printf '{\"reward\":1.0}' > reward.json"],"declared_artifacts":["/results/patch.diff"]}"#,
    )
    .unwrap();
    run_git(&repository, ["add", "task.json"]);
    run_git(&repository, ["commit", "-m", "pinned task"]);
    let revision = git_output(&repository, ["rev-parse", "HEAD"]);

    let exit = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--git-repository".to_owned(),
        repository.to_string_lossy().into_owned(),
        "--git-revision".to_owned(),
        revision,
        "--git-path".to_owned(),
        "task.json".to_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        "--verifier-mode".to_owned(),
        "shared".to_owned(),
    ])
    .unwrap();

    assert_eq!(exit, 0);
}

#[test]
fn native_eval_command_runs_a_pinned_git_package_from_a_remote_repository() {
    let temporary = tempfile::tempdir().unwrap();
    let repository = temporary.path().join("tasks");
    fs::create_dir(&repository).unwrap();
    run_git(&repository, ["init"]);
    run_git(
        &repository,
        ["config", "user.email", "eval@example.invalid"],
    );
    run_git(&repository, ["config", "user.name", "Native Eval"]);
    fs::write(
        repository.join("task.json"),
        br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","printf patch > \"$AIPERF_EVAL_ROOT/results/patch.diff\""],"verifier_command":["sh","-c","test -f results/patch.diff && printf '{\"reward\":1.0}' > reward.json"],"declared_artifacts":["/results/patch.diff"]}"#,
    )
    .unwrap();
    run_git(&repository, ["add", "task.json"]);
    run_git(&repository, ["commit", "-m", "pinned task"]);
    let revision = git_output(&repository, ["rev-parse", "HEAD"]);
    let remote = temporary.path().join("tasks.git");
    let status = Command::new("git")
        .args(["clone", "--bare"])
        .arg(&repository)
        .arg(&remote)
        .status()
        .unwrap();
    assert!(status.success());

    let exit = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--git-repository".to_owned(),
        format!("file://{}", remote.display()),
        "--git-revision".to_owned(),
        revision,
        "--git-path".to_owned(),
        "task.json".to_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        "--verifier-mode".to_owned(),
        "shared".to_owned(),
    ])
    .unwrap();

    assert_eq!(exit, 0);
}

#[test]
#[ignore = "requires a Docker daemon and the local openclaw sandbox image"]
fn native_eval_command_runs_a_pinned_standard_task_directory_in_docker() {
    let temporary = tempfile::tempdir().unwrap();
    let repository = temporary.path().join("tasks");
    fs::create_dir(&repository).unwrap();
    run_git(&repository, ["init"]);
    run_git(
        &repository,
        ["config", "user.email", "eval@example.invalid"],
    );
    run_git(&repository, ["config", "user.name", "Native Eval"]);
    fs::create_dir_all(repository.join("task/environment")).unwrap();
    fs::create_dir_all(repository.join("task/tests")).unwrap();
    fs::write(
        repository.join("task/task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/pinned-standard\"\n[environment]\nworkdir = \"/work\"\n",
    )
    .unwrap();
    fs::write(repository.join("task/instruction.md"), "Write a result.\n").unwrap();
    fs::write(
        repository.join("task/environment/Dockerfile"),
        "FROM openclaw-sandbox:bookworm-slim\n",
    )
    .unwrap();
    fs::write(
        repository.join("task/tests/test.sh"),
        "test -f /work/result.txt\nmkdir -p /logs/verifier\nprintf 1 > /logs/verifier/reward.txt\n",
    )
    .unwrap();
    run_git(&repository, ["add", "."]);
    run_git(&repository, ["commit", "-m", "standard task"]);
    let revision = git_output(&repository, ["rev-parse", "HEAD"]);

    let exit = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--git-repository".to_owned(),
        repository.to_string_lossy().into_owned(),
        "--git-revision".to_owned(),
        revision,
        "--git-path".to_owned(),
        "task/task.toml".to_owned(),
        "--agent-command".to_owned(),
        "printf result > result.txt".to_owned(),
    ])
    .unwrap();
    assert_eq!(exit, 0);
}

#[test]
fn native_eval_refuses_standard_task_directories_locally() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("repair-1");
    let started = temporary.path().join("agent-started");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/repair-1\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Write the result.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "test -f result.txt && printf 1 > reward.txt\n",
    )
    .unwrap();

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        "--agent-command".to_owned(),
        format!("touch {}", started.display()),
        "--sandbox".to_owned(),
        "local".to_owned(),
    ])
    .expect_err("local execution cannot enforce standard task guarantees");

    assert!(matches!(
        error.downcast_ref::<aiperf_runtime::eval::EvalExecutionError>(),
        Some(aiperf_runtime::eval::EvalExecutionError::UnsupportedEnforcement("docker"))
    ));
    assert!(!started.exists());
}

#[test]
fn native_eval_rejects_standard_task_verifier_mode_override() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("mode-conflict");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/mode-conflict\"\n[verifier]\nenvironment_mode = \"separate\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Write a result.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--verifier-mode".to_owned(),
        "shared".to_owned(),
        "--agent-command".to_owned(),
        "true".to_owned(),
    ])
    .expect_err("standard task mode must not be silently overridden");

    assert!(
        error
            .to_string()
            .contains("--verifier-mode conflicts with the standard task")
    );
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn native_eval_command_explicit_workdir_overrides_a_standard_task_manifest() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("docker-workdir-override");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/docker-workdir-override\"\n[environment]\nworkdir = \"/manifest-work\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Record the workdir.\n").unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN mkdir -p /logs/verifier\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "test \"$(cat /cli-work/pwd.txt)\" = /cli-work\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    )
    .unwrap();

    let exit = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        "--workdir".to_owned(),
        "/cli-work".to_owned(),
        "--agent-command".to_owned(),
        "pwd > pwd.txt".to_owned(),
    ])
    .expect("an explicit CLI workdir must override the normalized manifest workdir");

    assert_eq!(exit, 0);
}

#[test]
#[ignore = "requires a Docker daemon and the local openclaw sandbox image"]
fn native_eval_command_runs_a_standard_task_directory_in_docker() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("docker-repair-1");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\nartifacts = [\"/work/result.txt\"]\n[task]\nname = \"example/docker-repair-1\"\n[environment]\nworkdir = \"/work\"\n[verifier]\nenvironment_mode = \"separate\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Write the result.\n").unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM openclaw-sandbox:bookworm-slim\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "test -f /work/result.txt\ntest ! -e /work/agent-secret\nmkdir -p /logs/verifier\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    )
    .unwrap();

    let exit = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        "--agent-command".to_owned(),
        "test ! -e /tests/test.sh && printf secret > agent-secret && printf result > result.txt"
            .to_owned(),
    ])
    .unwrap();

    assert_eq!(exit, 0);
}

#[test]
#[ignore = "requires a Docker daemon and the local openclaw sandbox image"]
fn native_eval_command_transfers_only_declared_directory_artifacts_to_a_separate_verifier() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("docker-directory-artifacts");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\nartifacts = [{ source = \"/work/output\", destination = \"published\", exclude = [\"*.tmp\"] }]\n[task]\nname = \"example/docker-directory-artifacts\"\n[environment]\nworkdir = \"/work\"\n[agent.env]\nAGENT_ONLY_SECRET = \"agent-secret\"\n[verifier]\nenvironment_mode = \"separate\"\nuser = \"root\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Write a result.\n").unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM openclaw-sandbox:bookworm-slim\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "test -f /work/published/result.txt\ntest ! -e /work/published/drop.tmp\ntest ! -e /work/agent-only\ntest ! -e /work/tests/agent-only\ntest -z \"${AGENT_ONLY_SECRET+x}\"\nmkdir -p /logs/verifier\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    )
    .unwrap();

    let exit = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        "--agent-command".to_owned(),
        "mkdir -p output tests && printf result > output/result.txt && printf temporary > output/drop.tmp && printf agent > agent-only && printf agent > tests/agent-only"
            .to_owned(),
    ])
    .unwrap();

    assert_eq!(exit, 0);
}

#[test]
#[ignore = "requires a Docker daemon and the local openclaw sandbox image"]
fn native_eval_command_allows_a_non_root_separate_verifier_to_read_artifacts() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("docker-non-root-artifacts");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\nartifacts = [{ source = \"/work/output\", destination = \"published\" }]\n[task]\nname = \"example/docker-non-root-artifacts\"\n[environment]\nworkdir = \"/work\"\n[agent]\nuser = \"root\"\n[verifier]\nenvironment_mode = \"separate\"\nuser = \"nobody\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Write a result.\n").unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM openclaw-sandbox:bookworm-slim\nUSER root\nRUN mkdir -p /logs/verifier && chmod 0777 /logs/verifier\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "set -eu\ntest \"$(cat /work/published/nested/result.txt)\" = result\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    )
    .unwrap();

    let exit = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        "--agent-command".to_owned(),
        "mkdir -p output/nested && printf result > output/nested/result.txt".to_owned(),
    ])
    .unwrap();

    assert_eq!(exit, 0);
}

#[test]
#[ignore = "requires a Docker daemon and the local openclaw sandbox image"]
fn docker_timeout_removes_agent_container_after_descendant_command() {
    let _docker_test_lock = DOCKER_TIMEOUT_TEST_LOCK.lock().unwrap();
    let task_root = docker_timeout_task(
        "agent",
        "mkdir -p /logs/verifier\nprintf 1 > /logs/verifier/reward.txt\n",
        "shared",
        0.2,
        2.0,
    );

    let error = aiperf_cli::dispatch::run(&docker_eval_arguments(
        task_root.path(),
        "sleep 300 & sleep 2",
    ))
    .expect_err("an agent command exceeding its configured timeout must fail");

    let execution_error = error.downcast_ref::<aiperf_runtime::eval::EvalExecutionError>();
    assert!(
        matches!(
            execution_error,
            Some(aiperf_runtime::eval::EvalExecutionError::Timeout {
                phase: aiperf_runtime::eval::EvalExecutionPhase::Agent,
                timeout,
            }) if *timeout == Duration::from_millis(200)
        ),
        "unexpected agent timeout result: {execution_error:?}"
    );
    assert_task_containers_absent();
}

#[test]
#[ignore = "requires a Docker daemon and the local openclaw sandbox image"]
fn docker_timeout_removes_separate_verifier_container_after_descendant_command() {
    let _docker_test_lock = DOCKER_TIMEOUT_TEST_LOCK.lock().unwrap();
    let task_root = docker_timeout_task(
        "verifier",
        "sleep 300 & sleep 2\nmkdir -p /logs/verifier\nprintf 1 > /logs/verifier/reward.txt\n",
        "separate",
        2.0,
        0.2,
    );

    let error = aiperf_cli::dispatch::run(&docker_eval_arguments(task_root.path(), "true"))
        .expect_err("a verifier command exceeding its configured timeout must fail");

    let execution_error = error
        .downcast_ref::<aiperf_runtime::eval::EvalExecutionError>()
        .expect("Docker evaluation errors must preserve their typed execution cause");
    assert!(
        matches!(
            execution_error,
            aiperf_runtime::eval::EvalExecutionError::Timeout {
                phase: aiperf_runtime::eval::EvalExecutionPhase::Verifier,
                timeout,
            } if *timeout == Duration::from_millis(200)
        ),
        "unexpected verifier timeout result: {execution_error:?}"
    );
    assert_task_containers_absent();
}

fn docker_timeout_task(
    name: &str,
    verifier_script: &str,
    verifier_mode: &str,
    agent_timeout: f64,
    verifier_timeout: f64,
) -> tempfile::TempDir {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path();
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        format!(
            "schema_version = \"1.0\"\n[task]\nname = \"example/docker-timeout-{name}\"\n[agent]\ntimeout_sec = {agent_timeout}\n[verifier]\ntimeout_sec = {verifier_timeout}\nenvironment_mode = \"{verifier_mode}\"\n"
        ),
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Complete the task.\n").unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM openclaw-sandbox:bookworm-slim\n",
    )
    .unwrap();
    fs::write(task_root.join("tests/test.sh"), verifier_script).unwrap();
    temporary
}

fn docker_eval_arguments(task_root: &std::path::Path, agent_command: &str) -> Vec<String> {
    vec![
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        "--agent-command".to_owned(),
        agent_command.to_owned(),
    ]
}

fn assert_task_containers_absent() {
    let prefix = format!("aiperf-eval-{}-", std::process::id());
    let output = Command::new("docker")
        .args(["container", "ls", "--all", "--format", "{{.Names}}"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "unable to inspect Docker containers: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let names = String::from_utf8_lossy(&output.stdout);
    let remaining = names
        .lines()
        .filter(|name| name.starts_with(&prefix))
        .collect::<Vec<_>>();
    assert!(
        remaining.is_empty(),
        "task containers remained after the evaluation API returned: {remaining:?}"
    );
}

fn write_native_graph_task(task_root: &Path, endpoint: &str, dockerfile: &str) {
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.1"
[task]
name = "example/native-graph-records"
[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#,
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Complete the graph.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), dockerfile).unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "mkdir -p /logs/verifier\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    )
    .unwrap();
    fs::write(
        task_root.join("agent_graph.json"),
        r#"{
  "schema_version": "1.0", "trace_id": "cli-records", "stage_bound": 1,
  "channels": { "output": { "type": "messages", "reducer": "add_messages" } },
  "nodes": [{ "id": "model", "kind": "model", "binding": "primary", "output": "output", "streaming": false }],
  "edges": [{ "source": "START", "target": "model" }, { "source": "model", "target": "END" }],
  "terminal_outputs": []
}"#,
    )
    .unwrap();
    fs::write(
        task_root.join("models.toml"),
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
"#
        ),
    )
    .unwrap();
    fs::write(task_root.join("adapters.toml"), "").unwrap();
}

fn write_native_graph_lifecycle(path: &Path) {
    let policy = ArtifactDigest::from_bytes(b"native-graph-policy");
    fs::write(
        path,
        format!(
            r#"{{"version":1,"agent_variant":"native-graph","model":{{"provider":"provider-default","model":"example-model"}},"seed":11,"policy":"{policy}","runtime":"native","attempt":"caller-attempt","budget":{{"execution_seconds":30.0,"verifier_seconds":30.0}},"agent_contract":"native_graph","command":["aiperf-native-graph"],"initial_score":{{"metric":"reward","rationale":"{initial}"}},"regrade":{{"metric":"reward","rationale":"{regrade}"}}}}"#,
            policy = policy.as_str(),
            initial = ArtifactDigest::from_bytes(b"initial rationale").as_str(),
            regrade = ArtifactDigest::from_bytes(b"regrade rationale").as_str(),
        ),
    )
    .unwrap();
}

fn run_git<const N: usize>(repository: &std::path::Path, arguments: [&str; N]) {
    let status = Command::new("git")
        .arg("-c")
        .arg("commit.gpgsign=false")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .status()
        .unwrap();
    assert!(status.success());
}

fn git_output<const N: usize>(repository: &std::path::Path, arguments: [&str; N]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}
