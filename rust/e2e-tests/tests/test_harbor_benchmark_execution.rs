// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Black-box Docker execution coverage for standard Harbor benchmark tasks.

mod common;

use std::{
    collections::BTreeSet,
    fs,
    io::{Read, Write},
    net::TcpListener,
    process::{Command, Output},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use common::exec_binary;

const IMAGE_DIGEST: &str =
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

static DOCKER_E2E_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn standard_task_execution_preserves_image_workdir_without_a_cli_override() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("image-workdir");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/image-workdir\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Verify image defaults.\n").unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN mkdir -p /work /logs/verifier && printf image > /work/preloaded.txt\nWORKDIR /image-work\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "test \"$(cat /image-work/pwd.txt)\" = /image-work\ntest \"$(cat /work/preloaded.txt)\" = image\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    )
    .unwrap();

    let output = Command::new(exec_binary())
        .args([
            "eval",
            "--task",
            task_root.to_string_lossy().as_ref(),
            "--image",
            IMAGE_DIGEST,
            "--agent-command",
            "pwd > pwd.txt",
        ])
        .output()
        .expect("start native aiperf eval");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_summary(&summary, "example/image-workdir");
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn standard_task_exposes_only_declared_host_secrets_to_each_active_phase() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("declared-secrets");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/declared-secrets\"\n[environment]\nworkdir = \"/work\"\n[environment.env]\nBASE_TOKEN = \"${BENCHMARK_BASE_TOKEN}\"\n[agent.env]\nAGENT_TOKEN = \"${BENCHMARK_AGENT_TOKEN}\"\n[verifier.env]\nVERIFIER_TOKEN = \"${BENCHMARK_VERIFIER_TOKEN}\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Use scoped secrets.\n").unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN mkdir -p /work /logs/verifier\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "test \"$BASE_TOKEN\" = base-token\ntest \"$VERIFIER_TOKEN\" = verifier-token\ntest -z \"${AGENT_TOKEN+x}\"\ntest -z \"${BENCHMARK_UNDECLARED_TOKEN+x}\"\ntest -f /work/agent-ran\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    )
    .unwrap();

    let output = Command::new(exec_binary())
        .env("BENCHMARK_BASE_TOKEN", "base-token")
        .env("BENCHMARK_AGENT_TOKEN", "agent-token")
        .env("BENCHMARK_VERIFIER_TOKEN", "verifier-token")
        .env("BENCHMARK_UNDECLARED_TOKEN", "must-not-reach-docker")
        .args([
            "eval",
            "--task",
            task_root.to_string_lossy().as_ref(),
            "--image",
            IMAGE_DIGEST,
            "--agent-command",
            "test \"$BASE_TOKEN\" = base-token && test \"$AGENT_TOKEN\" = agent-token && test -z \"${VERIFIER_TOKEN+x}\" && test -z \"${BENCHMARK_UNDECLARED_TOKEN+x}\" && touch agent-ran",
        ])
        .output()
        .expect("start native aiperf eval");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    for value in [
        "base-token",
        "agent-token",
        "verifier-token",
        "must-not-reach-docker",
    ] {
        assert!(!stdout.contains(value), "summary leaked {value}");
        assert!(!stderr.contains(value), "diagnostic leaked {value}");
    }
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn missing_declared_secret_stops_before_the_agent_without_leaking_host_values() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let recorder = RequestRecorder::new();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("missing-secret");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/missing-secret\"\n[agent.env]\nREQUIRED_TOKEN = \"${BENCHMARK_MISSING_TOKEN}\"\n[verifier]\nenvironment_mode = \"separate\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Require a host secret.\n").unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN mkdir -p /logs/verifier\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        &format!(
            "wget -qO /dev/null {}/verifier\nprintf '{{\"reward\":1.0}}' > /logs/verifier/reward.json\n",
            recorder.url()
        ),
    )
    .unwrap();

    let output = Command::new(exec_binary())
        .env_remove("BENCHMARK_MISSING_TOKEN")
        .env("BENCHMARK_UNRELATED_SECRET", "host-value-must-not-leak")
        .args([
            "eval",
            "--task",
            task_root.to_string_lossy().as_ref(),
            "--image",
            IMAGE_DIGEST,
            "--agent-command",
            &format!("wget -qO /dev/null {}/agent", recorder.url()),
        ])
        .output()
        .expect("start native aiperf eval");

    assert!(
        !output.status.success(),
        "missing secret must stop the agent"
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("BENCHMARK_MISSING_TOKEN"),
        "stderr: {stderr}"
    );
    assert!(!stdout.contains("host-value-must-not-leak"));
    assert!(!stderr.contains("host-value-must-not-leak"));
    assert!(
        stdout.trim().is_empty(),
        "failed runs must not emit a summary"
    );
    recorder.assert_no_request("missing secret must prevent both agent and verifier startup");
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn standard_task_enforces_phase_network_user_environment_artifact_and_verifier_isolation() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let recorder = RequestRecorder::new();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task(
        &temporary,
        "network-user-env-artifacts",
        "artifacts = [{ source = \"/workspace/output\", destination = \"published\", exclude = [\"*.tmp\"] }]\n[environment]\nworkdir = \"/workspace\"\nuser = \"bench\"\nnetwork = \"no-network\"\n[environment.env]\nSCOPE = \"baseline\"\n[agent]\nuser = \"bench\"\nnetwork = \"no-network\"\n[agent.env]\nSCOPE = \"agent\"\n[verifier]\nenvironment_mode = \"separate\"\nuser = \"root\"\nnetwork = \"public\"\n[verifier.env]\nSCOPE = \"verifier\"\n[verifier.environment]\nnetwork = \"public\"\n",
        "FROM alpine:3.20\nRUN adduser -D bench && mkdir -p /workspace /logs/verifier && chown -R bench:bench /workspace && chmod 0777 /logs/verifier\nWORKDIR /image-default\n",
        &format!(
            "test \"$SCOPE\" = verifier\ntest \"$(id -un)\" = root\nwget -qO /dev/null {}/verifier\ntest \"$(cat /workspace/published/result.txt)\" = exact-bytes\ntest ! -e /workspace/published/drop.tmp\ntest \"$(cat /workspace/published/agent-scope.txt)\" = agent\ntest \"$(cat /workspace/published/agent-user.txt)\" = bench\ntest \"$(cat /workspace/published/agent-pwd.txt)\" = /workspace\ntest ! -e /workspace/agent-only\ntest -f /tests/test.sh\nprintf '{{\"reward\":1.0}}' > /logs/verifier/reward.json\n",
            recorder.url()
        ),
    );
    let output = run_eval(
        &task_root,
        &format!(
            "test ! -e /tests/test.sh && ! wget -qO /dev/null {}/agent && test \"$SCOPE\" = agent && test \"$(id -un)\" = bench && test \"$(pwd)\" = /workspace && mkdir -p output && printf exact-bytes > output/result.txt && printf excluded > output/drop.tmp && printf \"$SCOPE\" > output/agent-scope.txt && id -un > output/agent-user.txt && pwd > output/agent-pwd.txt && printf private > agent-only",
            recorder.url()
        ),
    );

    assert_success(&output);
    assert_eq!(
        recorder.next_request("the public verifier phase must reach the test server"),
        "/verifier"
    );
    recorder.assert_no_request("the no-network agent phase must not reach the test server");
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_summary(&summary, "example/network-user-env-artifacts");
    let artifacts = summary["artifacts"].as_array().unwrap();
    assert_eq!(artifacts.len(), 4);
    let paths = artifacts
        .iter()
        .map(|artifact| {
            let pair = artifact.as_array().expect("artifact must be a pair");
            assert_eq!(pair.len(), 2);
            assert!(pair[0].as_str().is_some());
            let digest = pair[1].as_str().expect("artifact digest must be a string");
            assert!(
                digest.starts_with("blake3:")
                    && digest.len() == "blake3:".len() + 64
                    && digest["blake3:".len()..]
                        .chars()
                        .all(|character| character.is_ascii_hexdigit()),
                "unexpected artifact digest {digest}"
            );
            pair[0].as_str().unwrap().to_owned()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        paths,
        BTreeSet::from([
            "published/agent-pwd.txt".to_owned(),
            "published/agent-scope.txt".to_owned(),
            "published/agent-user.txt".to_owned(),
            "published/result.txt".to_owned(),
        ])
    );
    let result_digest = artifacts
        .iter()
        .find_map(|artifact| {
            let pair = artifact.as_array()?;
            (pair.first()?.as_str()? == "published/result.txt")
                .then(|| pair.get(1)?.as_str())
                .flatten()
        })
        .expect("result artifact digest");
    assert_eq!(
        result_digest,
        format!("blake3:{}", blake3::hash(b"exact-bytes").to_hex())
    );
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn standard_task_default_public_agent_connects_to_the_controlled_host() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let recorder = RequestRecorder::new();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task(
        &temporary,
        "public-agent",
        "[environment]\nworkdir = \"/work\"\nnetwork = \"public\"\n",
        "FROM alpine:3.20\nRUN mkdir -p /work /logs/verifier\n",
        "test -f /work/agent-ran\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    );
    let output = run_eval(
        &task_root,
        &format!(
            "wget -qO /dev/null {}/agent && touch agent-ran",
            recorder.url()
        ),
    );

    assert_success(&output);
    assert_eq!(
        recorder.next_request("the default public agent must reach the controlled host"),
        "/agent"
    );
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn standard_task_retries_readiness_before_starting_the_agent() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task(
        &temporary,
        "readiness-retry",
        "[environment]\nworkdir = \"/work\"\n[environment.healthcheck]\ncommand = [\"/bin/sh\", \"-c\", \"if test -f /tmp/ready; then exit 0; fi; touch /tmp/ready; exit 1\"]\nstart_interval_sec = 0.01\ninterval_sec = 0.01\ntimeout_sec = 1\nretries = 2\n",
        "FROM alpine:3.20\nRUN mkdir -p /work /logs/verifier\n",
        "test -f /work/agent-ran\ntest -f /tmp/ready\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    );
    let output = run_eval(&task_root, "test -f /tmp/ready && touch agent-ran");

    assert_success(&output);
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_summary(&summary, "example/readiness-retry");
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn exhausted_readiness_prevents_agent_and_verifier_network_sentinels() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let recorder = RequestRecorder::new();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task(
        &temporary,
        "readiness-exhaustion",
        "[environment.healthcheck]\ncommand = [\"false\"]\ntimeout_sec = 1\nretries = 1\n",
        "FROM alpine:3.20\nRUN mkdir -p /logs/verifier\n",
        &format!("wget -qO /dev/null {}/verifier\n", recorder.url()),
    );
    let output = run_eval(
        &task_root,
        &format!("wget -qO /dev/null {}/agent", recorder.url()),
    );

    assert_failure_without_summary(&output, "unhealthy");
    recorder.assert_no_request("readiness exhaustion must prevent agent and verifier startup");
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn agent_timeout_cleans_up_the_docker_task_container_before_verifier_startup() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let recorder = RequestRecorder::new();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task(
        &temporary,
        "agent-timeout",
        "[agent]\ntimeout_sec = 0.2\n[verifier]\ntimeout_sec = 2\n",
        "FROM alpine:3.20\nRUN mkdir -p /logs/verifier\n",
        &format!("wget -qO /dev/null {}/verifier\n", recorder.url()),
    );
    let before = task_container_names();
    let output = run_eval(&task_root, "sleep 300 & sleep 2");

    assert_failure_without_summary(&output, "timed out");
    assert_eq!(
        task_container_names(),
        before,
        "timeout leaked a task container"
    );
    recorder.assert_no_request("agent timeout must prevent verifier startup");
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn separate_verifier_timeout_removes_every_task_container_before_returning() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task(
        &temporary,
        "separate-verifier-timeout",
        "[agent]\ntimeout_sec = 2\n[verifier]\ntimeout_sec = 0.2\nenvironment_mode = \"separate\"\n",
        "FROM alpine:3.20\nRUN mkdir -p /logs/verifier\n",
        "sleep 300 & sleep 2\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    );
    let before = task_container_names();
    let output = run_eval(&task_root, "true");

    assert_failure_without_summary(&output, "timed out");
    assert_eq!(
        task_container_names(),
        before,
        "separate verifier timeout leaked a task container"
    );
}

fn standard_task(
    temporary: &tempfile::TempDir,
    name: &str,
    manifest_suffix: &str,
    dockerfile: &str,
    verifier: &str,
) -> std::path::PathBuf {
    let task_root = temporary.path().join(name);
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        format!("schema_version = \"1.0\"\n{manifest_suffix}\n[task]\nname = \"example/{name}\"\n"),
    )
    .unwrap();
    fs::write(
        task_root.join("instruction.md"),
        "Run the benchmark task.\n",
    )
    .unwrap();
    fs::write(task_root.join("environment/Dockerfile"), dockerfile).unwrap();
    fs::write(task_root.join("tests/test.sh"), verifier).unwrap();
    task_root
}

fn run_eval(task_root: &std::path::Path, agent_command: &str) -> Output {
    Command::new(exec_binary())
        .args([
            "eval",
            "--task",
            task_root.to_string_lossy().as_ref(),
            "--image",
            IMAGE_DIGEST,
            "--agent-command",
            agent_command,
        ])
        .output()
        .expect("start native aiperf eval")
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_failure_without_summary(output: &Output, diagnostic: &str) {
    assert!(
        !output.status.success(),
        "evaluation unexpectedly succeeded"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(diagnostic),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).trim().is_empty(),
        "failed evaluation emitted a JSON summary: {}",
        String::from_utf8_lossy(&output.stdout)
    );
}

fn assert_summary(summary: &serde_json::Value, task: &str) {
    let object = summary.as_object().expect("summary must be a JSON object");
    assert_eq!(
        object.keys().map(String::as_str).collect::<Vec<_>>(),
        ["task", "artifacts", "reward"]
    );
    assert_eq!(summary["task"], task);
    assert_eq!(summary["reward"], serde_json::json!({ "reward": 1.0 }));
    assert!(summary["artifacts"].is_array());
}

fn task_container_names() -> BTreeSet<String> {
    let output = Command::new("docker")
        .args(["container", "ls", "--all", "--format", "{{.Names}}"])
        .output()
        .expect("inspect Docker containers");
    assert!(
        output.status.success(),
        "docker container listing failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .filter(|name| name.starts_with("aiperf-eval-"))
        .map(str::to_owned)
        .collect()
}

struct RequestRecorder {
    url: String,
    requests: Receiver<String>,
    shutdown: Arc<AtomicBool>,
    server: Option<JoinHandle<()>>,
}

impl RequestRecorder {
    fn new() -> Self {
        let listener = TcpListener::bind(("0.0.0.0", 0)).expect("bind request recorder");
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let gateway = public_network_gateway();
        let (sender, requests) = mpsc::channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let is_shutdown = shutdown.clone();
        let server = thread::spawn(move || {
            while !is_shutdown.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let mut request = [0; 1024];
                        let count = stream.read(&mut request).unwrap_or(0);
                        let path = String::from_utf8_lossy(&request[..count])
                            .lines()
                            .next()
                            .and_then(|line| line.split_whitespace().nth(1))
                            .unwrap_or("/")
                            .to_owned();
                        let _ = sender.send(path);
                        let _ = stream.write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK",
                        );
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            url: format!("http://{gateway}:{port}"),
            requests,
            shutdown,
            server: Some(server),
        }
    }

    fn url(&self) -> &str {
        &self.url
    }

    fn next_request(&self, context: &str) -> String {
        self.requests
            .recv_timeout(Duration::from_secs(3))
            .unwrap_or_else(|error| panic!("{context}: {error}"))
    }

    fn assert_no_request(&self, context: &str) {
        assert!(
            self.requests
                .recv_timeout(Duration::from_millis(300))
                .is_err(),
            "{context}"
        );
    }
}

impl Drop for RequestRecorder {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(server) = self.server.take() {
            server.join().unwrap();
        }
    }
}

fn public_network_gateway() -> String {
    let network = "aiperf-eval-public";
    let inspect = Command::new("docker")
        .args([
            "network",
            "inspect",
            network,
            "--format",
            "{{(index .IPAM.Config 0).Gateway}}",
        ])
        .output()
        .expect("inspect public Docker network");
    if !inspect.status.success() {
        let created = Command::new("docker")
            .args(["network", "create", network])
            .status()
            .expect("create public Docker network");
        assert!(created.success(), "create public Docker network");
    }
    let gateway = Command::new("docker")
        .args([
            "network",
            "inspect",
            network,
            "--format",
            "{{(index .IPAM.Config 0).Gateway}}",
        ])
        .output()
        .expect("inspect public Docker network gateway");
    assert!(gateway.status.success());
    String::from_utf8(gateway.stdout).unwrap().trim().to_owned()
}
