// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Imported Codex session sets ship over the authenticated cellular TLS artifact plane.
//!
//! The multi-cell run uses the real `aiperf` controller and cells, while the
//! in-process mock server receives every reconstructed imported request. The raw
//! record set must match a one-cell run over the same exact source set.

mod common;
use std::fs::{self, File};
use std::net::{SocketAddr, TcpListener};
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use common::*;
use ed25519_dalek::SigningKey;
use serde_json::{Value, json};

const SESSIONS: u32 = 3;
const CELLS: u32 = 3;
const CONCURRENCY: u32 = 3;
const PRIVATE_SOURCE_SENTINEL: &str = "PRIVATE_SOURCE_SENTINEL";

fn write_codex_session_set(root: &Path) -> PathBuf {
    let sessions = root.join("sessions");
    fs::create_dir_all(&sessions).expect("create imported-session directory");
    for session in 0..SESSIONS {
        let session_id = format!("cellular-import-{session}");
        let body = [
            json!({"type": "session_meta", "payload": {"id": session_id}}),
            json!({"type": "response_item", "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": format!("imported cellular prompt {session}")}],
            }}),
            json!({"type": "response_item", "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": format!("recorded reply {session}")}],
            }}),
        ]
        .into_iter()
        .map(|record| record.to_string())
        .collect::<Vec<_>>()
        .join("\n")
            + "\n";
        fs::write(sessions.join(format!("session-{session}.jsonl")), body)
            .expect("write imported Codex session");
    }
    fs::write(root.join("credentials.txt"), PRIVATE_SOURCE_SENTINEL)
        .expect("write ignored credentials decoy");
    fs::write(
        root.join("ignored-source.jsonl"),
        format!("{{\"private\":\"{PRIVATE_SOURCE_SENTINEL}\"}}\n"),
    )
    .expect("write ignored JSONL decoy");
    sessions
}

fn config(url: &str, source: &Path, replay_root: &Path, cells: u32) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 model: {DEFAULT_MODEL}\n\
        \x20 endpoint:\n\
        \x20   url: {url}\n\
        \x20   type: chat\n\
        \x20   streaming: true\n\
        \x20 dataset:\n\
        \x20   type: file\n\
        \x20   path: {}\n\
        \x20   format: agent_recording\n\
        \x20   graph:\n\
        \x20     source_format: codex\n\
        \x20     replay_root: {}\n\
        \x20 profiling:\n\
        \x20   type: concurrency\n\
        \x20   sessions: {SESSIONS}\n\
        \x20   concurrency: {CONCURRENCY}\n\
        \x20 artifacts:\n\
        \x20   records:\n\
        \x20     - jsonl\n\
        \x20   raw: true\n\
         runtime:\n\
        \x20 cells: {cells}\n",
        source.display(),
        replay_root.display(),
    )
}

fn run_imported_sessions(
    harness: &AIPerfHarness,
    source: &Path,
    replay_root: &Path,
    cells: u32,
    force_http: bool,
) -> RunResult {
    let temporary = tempfile::tempdir().expect("config temporary directory");
    let path = temporary.path().join("imported-sessions.yaml");
    fs::write(&path, config(&harness.mock.url, source, replay_root, cells))
        .expect("write imported-session config");
    let mut env = vec![(
        "AIPERF_LOG",
        "warn,aiperf=info,aiperf_cellular_artifact=debug",
    )];
    if force_http {
        env.push(("AIPERF_CELL_ARTIFACT_HTTP_FORCE", "1"));
    }
    harness.run_env(&format!("--config {} --ui simple", path.display()), &env)
}

fn raw_projection(record: &Value) -> String {
    json!({
        "payload": record["payload"],
        "error": record["error"],
        "status": record["status"],
    })
    .to_string()
}

fn raw_response_text(record: &Value) -> String {
    record["responses"]
        .as_array()
        .expect("raw record responses must be an array")
        .iter()
        .flat_map(|response| {
            response["packets"]
                .as_array()
                .expect("raw response packets must be an array")
        })
        .filter(|packet| packet["name"] == "data")
        .filter_map(|packet| packet["value"].as_str())
        .filter_map(|chunk| serde_json::from_str::<Value>(chunk).ok())
        .filter_map(|chunk| {
            chunk
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect()
}

fn sorted_raw_records(result: &RunResult) -> Vec<String> {
    let mut records: Vec<_> = result
        .artifacts
        .raw_records()
        .iter()
        .map(raw_projection)
        .collect();
    records.sort();
    records
}

fn dataset_serve_observables(result: &RunResult) -> Vec<String> {
    let log = result
        .artifacts
        .find_file("**/aiperf.log")
        .expect("logs/aiperf.log should exist");
    fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .filter(|line| line.contains("served dataset source over TLS/authenticated transfer"))
        .map(str::to_owned)
        .collect()
}

fn regular_artifact_texts(root: &Path) -> Vec<String> {
    let mut pending = vec![root.to_path_buf()];
    let mut texts = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).expect("read artifact directory") {
            let entry = entry.expect("read artifact entry");
            let path = entry.path();
            let kind = entry.file_type().expect("inspect artifact entry");
            if kind.is_dir() {
                pending.push(path);
            } else if kind.is_file() {
                texts.push(
                    String::from_utf8_lossy(&fs::read(&path).expect("read regular artifact"))
                        .into_owned(),
                );
            }
        }
    }
    texts
}

fn regular_artifact_bytes(root: &Path) -> Vec<Vec<u8>> {
    let mut pending = vec![root.to_path_buf()];
    let mut bytes = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).expect("read artifact directory") {
            let entry = entry.expect("read artifact entry");
            let path = entry.path();
            if entry.file_type().expect("inspect artifact entry").is_dir() {
                pending.push(path);
            } else if path.is_file() {
                bytes.push(fs::read(path).expect("read artifact bytes"));
            }
        }
    }
    bytes
}

fn assert_private_material_absent(result: &RunResult) {
    let secret = [0xC3; 32];
    let base64 = STANDARD.encode(secret);
    let hex = secret
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let artifact_bytes = regular_artifact_bytes(&result.artifacts.dir);
    assert!(
        artifact_bytes
            .iter()
            .all(|bytes| !bytes.windows(secret.len()).any(|window| window == secret)),
        "raw fixture signing material leaked into artifacts"
    );
    for text in [result.stdout.as_str(), result.stderr.as_str()] {
        assert!(
            !text.contains(&base64) && !text.contains(&hex),
            "encoded fixture signing material leaked into controller output"
        );
    }
    assert!(
        artifact_bytes.iter().all(|bytes| {
            let text = String::from_utf8_lossy(bytes);
            !text.contains(&base64) && !text.contains(&hex)
        }),
        "encoded fixture signing material leaked into artifacts"
    );
}

fn assert_fixture_private_material_absent(root: &Path) {
    let private_keys = [
        [0xC3_u8; 32],
        [1_u8; 32],
        [2_u8; 32],
        [0x51_u8; 32],
        [0x52_u8; 32],
    ];
    let mut outputs = Vec::new();
    for entry in fs::read_dir(root).expect("read fixture root") {
        let path = entry.expect("read fixture entry").path();
        if matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("stdout" | "stderr")
        ) {
            outputs.push(fs::read(path).expect("read fixture child output"));
        }
    }
    let artifact_root = root.join("artifacts");
    if artifact_root.is_dir() {
        outputs.extend(regular_artifact_bytes(&artifact_root));
    }
    for private_key in private_keys {
        let base64 = STANDARD.encode(private_key);
        let hex = private_key
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert!(
            outputs.iter().all(|bytes| !bytes
                .windows(private_key.len())
                .any(|window| window == private_key)),
            "raw fixture private key leaked to child output or artifact"
        );
        assert!(
            outputs.iter().all(|bytes| {
                let text = String::from_utf8_lossy(bytes);
                !text.contains(&base64) && !text.contains(&hex)
            }),
            "encoded fixture private key leaked to child output or artifact"
        );
    }
}

fn fixture_dataset_transfer_count(root: &Path) -> usize {
    let controller_stderr = fs::read(root.join("controller.stderr")).unwrap_or_default();
    String::from_utf8_lossy(&controller_stderr)
        .lines()
        .filter(|line| line.contains("served dataset source over TLS/authenticated transfer"))
        .count()
}

fn has_execution_output(root: &Path) -> bool {
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        if !directory.is_dir() {
            continue;
        }
        for entry in fs::read_dir(directory).expect("read execution namespace") {
            let entry = entry.expect("read execution entry");
            let path = entry.path();
            if entry.file_type().expect("inspect execution entry").is_dir() {
                pending.push(path);
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == "native-v2.json"
                || name == "profile_export.jsonl"
                || name == "profile_export_raw.jsonl"
            {
                return true;
            }
        }
    }
    false
}

fn mock_chat_request_arrivals(state: &aiperf_mock_server::AppState) -> u64 {
    state
        .recorder
        .metrics
        .aiperf
        .REQUESTS_BY_MODEL
        .with_label_values(&[DEFAULT_MODEL, "/v1/chat/completions"])
        .get()
}

fn has_cell_local_native_report(root: &Path) -> bool {
    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        for entry in fs::read_dir(directory).expect("read cell-local namespace") {
            let entry = entry.expect("read cell-local entry");
            if entry
                .file_type()
                .expect("inspect cell-local entry")
                .is_dir()
            {
                directories.push(entry.path());
            } else if entry.file_name() == "native-v2.json" {
                return true;
            }
        }
    }
    false
}

const CONTROLLER_LISTENER_FD: i32 = 3;
const ARTIFACT_LISTENER_FD: i32 = 4;
const SCRATCH_HOLD_FD: i32 = 5;

static CELLULAR_PROCESS_FIXTURE_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

async fn cellular_process_fixture_lock() -> tokio::sync::MutexGuard<'static, ()> {
    CELLULAR_PROCESS_FIXTURE_LOCK
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

trait ProcessOps {
    fn id(&self) -> u32;
    fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>>;
    fn kill(&mut self) -> std::io::Result<()>;
    fn wait(&mut self) -> std::io::Result<std::process::ExitStatus>;
}

impl ProcessOps for Child {
    fn id(&self) -> u32 {
        Child::id(self)
    }

    fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        Child::try_wait(self)
    }

    fn kill(&mut self) -> std::io::Result<()> {
        Child::kill(self)
    }

    fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        Child::wait(self)
    }
}

struct DeploymentChild {
    child: Option<Box<dyn ProcessOps>>,
    output_paths: Option<(PathBuf, PathBuf)>,
}

struct DeploymentChildren {
    children: Vec<DeploymentChild>,
}
impl DeploymentChildren {
    fn new() -> Self {
        Self {
            children: Vec::new(),
        }
    }
    fn push(&mut self, child: Child) {
        self.children.push(DeploymentChild {
            child: Some(Box::new(child)),
            output_paths: None,
        });
    }
    fn push_with_output_files(&mut self, child: Child, stdout: PathBuf, stderr: PathBuf) {
        self.children.push(DeploymentChild {
            child: Some(Box::new(child)),
            output_paths: Some((stdout, stderr)),
        });
    }
    fn live_child_count(&self) -> usize {
        self.children
            .iter()
            .filter(|child| child.child.is_some())
            .count()
    }

    async fn wait_until(&mut self, deadline: Instant) -> Result<Vec<(usize, Output)>, String> {
        let mut outputs = Vec::with_capacity(self.children.len());
        while self.live_child_count() != 0 {
            for index in 0..self.children.len() {
                let exited = match self.children[index].child.as_mut() {
                    Some(child) => child
                        .try_wait()
                        .map_err(|error| error.to_string())?
                        .is_some(),
                    None => false,
                };
                if exited {
                    let status = self.children[index]
                        .child
                        .as_mut()
                        .expect("exited child remains owned")
                        .wait()
                        .map_err(|error| error.to_string())?;
                    let output = self.collect_output(index, status)?;
                    self.children[index].child.take();
                    outputs.push((index, output));
                }
            }
            if self.live_child_count() == 0 {
                break;
            }
            if Instant::now() >= deadline {
                let diagnostics = self.all_child_diagnostics();
                let cleanup = self.kill_and_reap().err();
                return Err(format!(
                    "deployment child deadline elapsed:\n{diagnostics}{}",
                    cleanup.map_or_else(String::new, |error| format!("\ncleanup: {error}"))
                ));
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        Ok(outputs)
    }

    async fn wait_indices_until(
        &mut self,
        indices: &[usize],
        deadline: Instant,
    ) -> Result<Vec<(usize, Output)>, String> {
        let mut outputs = Vec::with_capacity(indices.len());
        while outputs.len() != indices.len() {
            for &index in indices {
                if outputs.iter().any(|(finished, _)| *finished == index) {
                    continue;
                }
                let exited = match self.children[index].child.as_mut() {
                    Some(child) => child
                        .try_wait()
                        .map_err(|error| error.to_string())?
                        .is_some(),
                    None => false,
                };
                if exited {
                    let status = self.children[index]
                        .child
                        .as_mut()
                        .expect("exited child remains owned")
                        .wait()
                        .map_err(|error| error.to_string())?;
                    let output = self.collect_output(index, status)?;
                    self.children[index].child.take();
                    outputs.push((index, output));
                }
            }
            if outputs.len() == indices.len() {
                break;
            }
            if Instant::now() >= deadline {
                let diagnostics = self.all_child_diagnostics();
                let cleanup = self.kill_and_reap().err();
                return Err(format!(
                    "deployment child deadline elapsed:\n{diagnostics}{}",
                    cleanup.map_or_else(String::new, |error| format!("\ncleanup: {error}"))
                ));
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        Ok(outputs)
    }

    fn kill_and_reap(&mut self) -> Result<(), String> {
        let mut errors = Vec::new();
        for (index, entry) in self.children.iter_mut().enumerate() {
            let Some(child) = entry.child.as_mut() else {
                continue;
            };
            if let Err(error) = child.kill() {
                if error.kind() != std::io::ErrorKind::InvalidInput {
                    errors.push(format!("kill child {index}: {error}"));
                }
            }
            match child.wait() {
                Ok(_) => {
                    entry.child.take();
                }
                Err(error) => errors.push(format!("wait child {index}: {error}")),
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }

    fn collect_output(
        &self,
        index: usize,
        status: std::process::ExitStatus,
    ) -> Result<Output, String> {
        let Some((stdout, stderr)) = self.children[index].output_paths.as_ref() else {
            return Ok(Output {
                status,
                stdout: Vec::new(),
                stderr: Vec::new(),
            });
        };
        Ok(Output {
            status,
            stdout: fs::read(stdout).map_err(|error| error.to_string())?,
            stderr: fs::read(stderr).map_err(|error| error.to_string())?,
        })
    }

    fn all_child_diagnostics(&mut self) -> String {
        let mut diagnostics = Vec::with_capacity(self.children.len());
        for (index, entry) in self.children.iter_mut().enumerate() {
            let state = match entry.child.as_mut() {
                Some(child) => match child.try_wait() {
                    Ok(Some(status)) => format!("exited with {status}"),
                    Ok(None) => format!("live pid {}", child.id()),
                    Err(error) => format!("try_wait failed: {error}"),
                },
                None => "already reaped".to_owned(),
            };
            let (stdout, stderr) = entry
                .output_paths
                .as_ref()
                .map(|(stdout, stderr)| {
                    (
                        String::from_utf8_lossy(&fs::read(stdout).unwrap_or_default()).into_owned(),
                        String::from_utf8_lossy(&fs::read(stderr).unwrap_or_default()).into_owned(),
                    )
                })
                .unwrap_or_default();
            diagnostics.push(format!(
                "child {index} {state}:\nstdout:\n{stdout}\nstderr:\n{stderr}"
            ));
        }
        diagnostics.join("\n")
    }
}
impl Drop for DeploymentChildren {
    fn drop(&mut self) {
        let _ = self.kill_and_reap();
    }
}

#[tokio::test]
#[cfg(unix)]
async fn deployment_children_deadline_kills_reaps_and_releases_ports() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind deadline fixture listener");
    let address = listener.local_addr().expect("deadline fixture address");
    let mut children = DeploymentChildren::new();
    children.push(
        Command::new("/bin/sh")
            .args(["-c", "exec sleep 0.1"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn deliberately non-terminating child"),
    );
    let deadline = Instant::now() + Duration::from_millis(25);
    assert!(
        children.wait_until(deadline).await.is_err(),
        "the outer deadline must interrupt child waiting"
    );
    assert_eq!(
        children.live_child_count(),
        0,
        "deadline must reap every child"
    );
    drop(listener);
    assert!(
        TcpListener::bind(address).is_ok(),
        "deadline cleanup must release ports"
    );
}

#[cfg(unix)]
#[test]
fn cleanup_failure_retains_child_bookkeeping() {
    struct FailingWaitProcess {
        kills: Arc<AtomicUsize>,
        waits: Arc<AtomicUsize>,
    }

    impl ProcessOps for FailingWaitProcess {
        fn id(&self) -> u32 {
            0
        }

        fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
            Ok(None)
        }

        fn kill(&mut self) -> std::io::Result<()> {
            self.kills.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
            self.waits.fetch_add(1, Ordering::SeqCst);
            Err(std::io::Error::other("injected wait failure"))
        }
    }

    let kills = Arc::new(AtomicUsize::new(0));
    let waits = Arc::new(AtomicUsize::new(0));
    let mut children = DeploymentChildren::new();
    children.children.push(DeploymentChild {
        child: Some(Box::new(FailingWaitProcess {
            kills: Arc::clone(&kills),
            waits: Arc::clone(&waits),
        })),
        output_paths: None,
    });
    assert!(
        children.kill_and_reap().is_err(),
        "cleanup must report a child it could not reap"
    );
    assert_eq!(
        children.live_child_count(),
        1,
        "failed cleanup must retain child bookkeeping"
    );
    assert_eq!(kills.load(Ordering::SeqCst), 1);
    assert_eq!(waits.load(Ordering::SeqCst), 1);
}
fn encoded_role(role: Option<u32>) -> [u8; 9] {
    let mut bytes = [0; 9];
    if let Some(id) = role {
        bytes[0] = 1;
        bytes[5..].copy_from_slice(&id.to_le_bytes());
    }
    bytes
}
fn security_material(
    role: Option<u32>,
    run_nonce: [u8; 32],
    signer: &SigningKey,
    controller_verifier: [u8; 32],
    roles: &[SigningKey],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(118 + roles.len() * 41);
    bytes.extend_from_slice(b"AIPRFSEC");
    bytes.push(1);
    bytes.extend_from_slice(&encoded_role(role));
    bytes.extend_from_slice(&run_nonce);
    bytes.extend_from_slice(&signer.to_bytes());
    bytes.extend_from_slice(&controller_verifier);
    bytes.extend_from_slice(&(roles.len() as u32).to_le_bytes());
    for (id, role) in roles.iter().enumerate() {
        bytes.extend_from_slice(&encoded_role(Some(id as u32)));
        bytes.extend_from_slice(role.verifying_key().as_bytes());
    }
    bytes
}
fn write_private(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).expect("write private binary material");
    #[cfg(unix)]
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
        .expect("restrict private binary material");
}
fn provision_binary_security(
    root: &Path,
    has_wrong_controller_role_keys: bool,
) -> (PathBuf, Vec<PathBuf>) {
    let nonce = [0xA6; 32];
    let controller = SigningKey::from_bytes(&[0xC3; 32]);
    let roles: Vec<_> = (0..2)
        .map(|id| SigningKey::from_bytes(&[id + 1; 32]))
        .collect();
    let controller_roles = if has_wrong_controller_role_keys {
        (0..2)
            .map(|id| SigningKey::from_bytes(&[0x51 + id; 32]))
            .collect()
    } else {
        roles.clone()
    };
    let controller_path = root.join("controller.security");
    write_private(
        &controller_path,
        &security_material(
            None,
            nonce,
            &controller,
            controller.verifying_key().to_bytes(),
            &controller_roles,
        ),
    );
    let role_paths = roles
        .iter()
        .enumerate()
        .map(|(id, signer)| {
            let path = root.join(format!("cell-{id}.security"));
            write_private(
                &path,
                &security_material(
                    Some(id as u32),
                    nonce,
                    signer,
                    controller.verifying_key().to_bytes(),
                    &roles,
                ),
            );
            path
        })
        .collect();
    (controller_path, role_paths)
}
struct CrossHostCellularFixture {
    harness: AIPerfHarness,
    temporary: tempfile::TempDir,
    controller_listener: Option<TcpListener>,
    artifact_listener: Option<TcpListener>,
    scratch_gate_reader: Option<File>,
    scratch_gate_writer: Option<File>,
    controller_addr: SocketAddr,
    artifact_addr: SocketAddr,
    controller_cwd: PathBuf,
    cell_cwds: Vec<PathBuf>,
    start_marker: PathBuf,
    children: DeploymentChildren,
}
impl CrossHostCellularFixture {
    async fn new() -> Self {
        let harness = AIPerfHarness::new().await;
        let temporary = tempfile::tempdir().expect("fixture root");
        let controller_listener =
            TcpListener::bind("127.0.0.1:0").expect("bind controller listener");
        let artifact_listener = TcpListener::bind("127.0.0.1:0").expect("bind artifact listener");
        let controller_addr = controller_listener
            .local_addr()
            .expect("controller address");
        let artifact_addr = artifact_listener.local_addr().expect("artifact address");
        let controller_cwd = temporary.path().join("controller-cwd");
        let start_marker = temporary.path().join("controller-start.events");
        fs::create_dir_all(controller_cwd.join("tmp")).expect("create controller cwd");
        let cell_cwds = (0..2)
            .map(|id| {
                let cwd = temporary.path().join(format!("cell-{id}-cwd"));
                fs::create_dir_all(cwd.join("tmp")).expect("create cell cwd");
                cwd
            })
            .collect();
        let mut fds = [0; 2];
        // SAFETY: pipe2 initializes both local descriptor slots; this fixture owns
        // each endpoint until it transfers the reader and closes the writer.
        assert_eq!(
            unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) },
            0,
            "create scratch hold pipe"
        );
        let scratch_gate_reader = Some(unsafe { File::from_raw_fd(fds[0]) });
        let scratch_gate_writer = Some(unsafe { File::from_raw_fd(fds[1]) });
        Self {
            harness,
            temporary,
            controller_listener: Some(controller_listener),
            artifact_listener: Some(artifact_listener),
            scratch_gate_reader,
            scratch_gate_writer,
            controller_addr,
            artifact_addr,
            controller_cwd,
            cell_cwds,
            start_marker,
            children: DeploymentChildren::new(),
        }
    }
    fn spawn_controller(&mut self) {
        self.spawn_controller_with_roles(false);
    }
    fn spawn_controller_with_roles(&mut self, has_wrong_controller_role_keys: bool) {
        let source = write_codex_session_set(self.temporary.path());
        let config_path = self.temporary.path().join("deployment.yaml");
        fs::write(
            &config_path,
            config(&self.harness.mock.url, &source, self.temporary.path(), 2),
        )
        .expect("write deployment config");
        let (controller_file, role_files) =
            provision_binary_security(self.temporary.path(), has_wrong_controller_role_keys);
        fs::write(
            self.temporary.path().join("roles"),
            role_files
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .expect("record roles");
        let mut command = Command::new(exec_binary());
        let (stdout, stderr) = child_output_paths(self.temporary.path(), "controller");
        command
            .args(["controller", "--config"])
            .arg(config_path)
            .args(["--artifact-dir"])
            .arg(self.temporary.path().join("artifacts"))
            .env("AIPERF_CELL_LAUNCHER", "k8s")
            .env(
                "AIPERF_CELL_CONTROLLER_ADDR",
                format!("tcp://{}", self.controller_addr),
            )
            .env("AIPERF_CONTROLLER_BOOTSTRAP_FILE", controller_file)
            .env("AIPERF_CONTROLLER_LISTENER_FD", "3")
            .env("AIPERF_CONTROLLER_ARTIFACT_LISTENER_FD", "4")
            .env(
                "AIPERF_CONTROLLER_SCRATCH_HOLD_FD",
                SCRATCH_HOLD_FD.to_string(),
            )
            .env("AIPERF_CELL_REGISTER_TIMEOUT_SECS", "2")
            .env("AIPERF_CELL_ARTIFACT_HTTP_FORCE", "1")
            .env("AIPERF_E2E_CONTROLLER_START_MARKER", &self.start_marker)
            .env("AIPERF_LOG", "aiperf=trace")
            .env("TMPDIR", "tmp")
            .current_dir(&self.controller_cwd)
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .stdout(Stdio::from(
                File::create(&stdout).expect("create controller stdout"),
            ))
            .stderr(Stdio::from(
                File::create(&stderr).expect("create controller stderr"),
            ));
        let inherited_descriptors = prepare_inherited_descriptor_remap(
            &mut command,
            &[
                (
                    self.controller_listener.as_ref().unwrap().as_raw_fd(),
                    CONTROLLER_LISTENER_FD,
                ),
                (
                    self.artifact_listener.as_ref().unwrap().as_raw_fd(),
                    ARTIFACT_LISTENER_FD,
                ),
                (
                    self.scratch_gate_reader.as_ref().unwrap().as_raw_fd(),
                    SCRATCH_HOLD_FD,
                ),
            ],
        )
        .expect("prepare inherited controller descriptors");
        self.children.push_with_output_files(
            command.spawn().expect("spawn controller"),
            stdout,
            stderr,
        );
        drop(inherited_descriptors);
        self.controller_listener.take();
        self.artifact_listener.take();
        self.scratch_gate_reader.take();
    }
    fn spawn_cells(&mut self) {
        for (id, role) in fs::read_to_string(self.temporary.path().join("roles"))
            .expect("read roles")
            .lines()
            .enumerate()
        {
            let mut command = Command::new(exec_binary());
            let (stdout, stderr) = child_output_paths(self.temporary.path(), &format!("cell-{id}"));
            command
                .arg("cell")
                .env("AIPERF_CELL_LAUNCHER", "k8s")
                .env(
                    "AIPERF_CELL_CONTROLLER_ADDR",
                    format!("tcp://{}", self.controller_addr),
                )
                .env("AIPERF_CELL_ARTIFACT_ADDR", self.artifact_addr.to_string())
                .env("AIPERF_CELL_ID", id.to_string())
                .env("AIPERF_CELL_COUNT", "2")
                .env("AIPERF_ROLE_BOOTSTRAP_FILE", role)
                .env("AIPERF_CELL_REGISTER_TIMEOUT_SECS", "2")
                .env("AIPERF_CELL_ARTIFACT_HTTP_FORCE", "1")
                .env("AIPERF_LOG", "aiperf=trace")
                .env("TMPDIR", "tmp")
                .current_dir(&self.cell_cwds[id])
                .env("HF_HUB_OFFLINE", "1")
                .env("TRANSFORMERS_OFFLINE", "1")
                .stdout(Stdio::from(
                    File::create(&stdout).expect("create cell stdout"),
                ))
                .stderr(Stdio::from(
                    File::create(&stderr).expect("create cell stderr"),
                ));
            self.children.push_with_output_files(
                command.spawn().expect("spawn cell"),
                stdout,
                stderr,
            );
        }
    }
    fn spawn_one_cell(&mut self, id: usize, role: &Path) {
        let mut command = Command::new(exec_binary());
        let (stdout, stderr) = child_output_paths(
            self.temporary.path(),
            &format!("negative-cell-{id}-{}", self.children.live_child_count()),
        );
        command
            .arg("cell")
            .env("AIPERF_CELL_LAUNCHER", "k8s")
            .env(
                "AIPERF_CELL_CONTROLLER_ADDR",
                format!("tcp://{}", self.controller_addr),
            )
            .env("AIPERF_CELL_ARTIFACT_ADDR", self.artifact_addr.to_string())
            .env("AIPERF_CELL_ID", id.to_string())
            .env("AIPERF_CELL_COUNT", "2")
            .env("AIPERF_ROLE_BOOTSTRAP_FILE", role)
            .env("AIPERF_CELL_REGISTER_TIMEOUT_SECS", "2")
            .env("AIPERF_CELL_ARTIFACT_HTTP_FORCE", "1")
            .env("AIPERF_LOG", "aiperf=trace")
            .env("TMPDIR", "tmp")
            .current_dir(&self.cell_cwds[id])
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .stdout(Stdio::from(
                File::create(&stdout).expect("create cell stdout"),
            ))
            .stderr(Stdio::from(
                File::create(&stderr).expect("create cell stderr"),
            ));
        self.children
            .push_with_output_files(command.spawn().expect("spawn cell"), stdout, stderr);
    }

    async fn wait_for_failure(
        &mut self,
        indices: &[usize],
        deadline: Instant,
    ) -> Vec<(usize, Output)> {
        let outputs = self
            .children
            .wait_indices_until(indices, deadline)
            .await
            .expect("negative cells finished before deadline");
        assert!(
            outputs.iter().all(|(_, output)| !output.status.success()),
            "every rejected deployment cell must terminate unsuccessfully"
        );
        outputs
    }

    async fn release_gate_and_wait_for_controller_failure(&mut self, deadline: Instant) -> Output {
        self.scratch_gate_writer.take();
        let output = self
            .children
            .wait_indices_until(&[0], deadline)
            .await
            .expect("controller negative completion before deadline")
            .pop()
            .expect("controller output")
            .1;
        assert!(
            !output.status.success(),
            "controller unexpectedly accepted negative deployment"
        );
        output
    }

    fn assert_no_controller_connection(&self) {
        let listener = self
            .controller_listener
            .as_ref()
            .expect("fixture still owns listener");
        listener
            .set_nonblocking(true)
            .expect("make probe listener nonblocking");
        let result = listener.accept();
        listener
            .set_nonblocking(false)
            .expect("restore probe listener blocking mode");
        assert!(
            matches!(result, Err(error) if error.kind() == std::io::ErrorKind::WouldBlock),
            "missing role material must fail before the cell dials Velo"
        );
    }

    fn assert_no_dispatched_records(&self) {
        assert!(
            ArtifactReader {
                dir: self.temporary.path().join("artifacts"),
            }
            .raw_records()
            .is_empty(),
            "rejected admission must not materialize profile records"
        );
    }

    fn assert_admission_side_effects(&self, expected_dataset_transfers: usize) {
        self.assert_no_dispatched_records();
        assert_eq!(
            mock_chat_request_arrivals(&self.harness.mock.state),
            0,
            "rejected admission must not dispatch an inference request"
        );
        assert!(
            !self.start_marker.exists(),
            "rejected admission must not trigger controller START state"
        );
        for namespace in std::iter::once(&self.controller_cwd).chain(self.cell_cwds.iter()) {
            assert!(
                !has_execution_output(namespace),
                "rejected admission wrote raw/native output in {}",
                namespace.display()
            );
        }
        let dataset_transfers = fixture_dataset_transfer_count(self.temporary.path());
        assert_eq!(
            dataset_transfers, expected_dataset_transfers,
            "only the winning registration may subscribe to the dataset transfer"
        );
    }

    async fn wait_for_success(&mut self) -> RunResult {
        let deadline = Instant::now() + Duration::from_secs(15);
        let cells = self
            .children
            .wait_indices_until(&[1, 2], deadline)
            .await
            .expect("cells finished before deadline");
        let cell_failures: Vec<_> = cells
            .into_iter()
            .filter_map(|(index, output)| {
                (!output.status.success()).then(|| {
                    format!(
                        "cell {index} failed:\nstdout:\n{}\nstderr:\n{}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    )
                })
            })
            .collect();
        assert!(cell_failures.is_empty(), "{}", cell_failures.join("\n"));
        assert!(
            has_cell_local_native_report(&self.controller_cwd),
            "cell reports must remain present while the inherited scratch gate is open"
        );
        self.scratch_gate_writer.take();
        let output = self
            .children
            .wait_indices_until(&[0], deadline)
            .await
            .expect("controller finished after scratch gate release")
            .pop()
            .expect("controller output")
            .1;
        assert!(
            output.status.success(),
            "controller failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            fs::read_to_string(&self.start_marker)
                .expect("read controller START marker")
                .lines()
                .filter(|line| *line == "START")
                .count(),
            1,
            "successful deployment must trigger controller START exactly once"
        );
        RunResult {
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            artifacts: ArtifactReader {
                dir: self.temporary.path().join("artifacts"),
            },
        }
    }
}
fn child_output_paths(root: &Path, name: &str) -> (PathBuf, PathBuf) {
    (
        root.join(format!("{name}.stdout")),
        root.join(format!("{name}.stderr")),
    )
}
impl Drop for CrossHostCellularFixture {
    fn drop(&mut self) {
        self.scratch_gate_writer.take();
        let _ = self.children.kill_and_reap();
    }
}
#[cfg(unix)]
fn prepare_inherited_descriptor_remap(
    command: &mut Command,
    descriptors: &[(i32, i32)],
) -> std::io::Result<Vec<OwnedFd>> {
    let safe_min = descriptors
        .iter()
        .flat_map(|(source, target)| [*source, *target])
        .max()
        .unwrap_or(SCRATCH_HOLD_FD)
        .checked_add(1)
        .ok_or_else(|| std::io::Error::other("inherited descriptor range overflow"))?;
    let mut temporary = Vec::with_capacity(descriptors.len());
    for &(source, target) in descriptors {
        let duplicate = unsafe { libc::fcntl(source, libc::F_DUPFD_CLOEXEC, safe_min) };
        if duplicate < 0 {
            return Err(std::io::Error::last_os_error());
        }
        temporary.push((unsafe { OwnedFd::from_raw_fd(duplicate) }, target));
    }
    let child_descriptors: Vec<_> = temporary
        .iter()
        .map(|(source, target)| (source.as_raw_fd(), *target))
        .collect();
    unsafe {
        command.pre_exec(move || {
            for &(source, target) in &child_descriptors {
                if libc::dup2(source, target) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::fcntl(target, libc::F_SETFD, 0) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            for &(source, _) in &child_descriptors {
                if libc::close(source) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    Ok(temporary.into_iter().map(|(source, _)| source).collect())
}

#[cfg(unix)]
#[test]
fn inherited_descriptor_remap_handles_source_target_collisions() {
    fn pipe_with_byte(byte: u8) -> OwnedFd {
        let mut fds = [0; 2];
        assert_eq!(
            unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) },
            0,
            "create descriptor fixture pipe"
        );
        assert_eq!(
            unsafe { libc::write(fds[1], &byte as *const u8 as *const libc::c_void, 1) },
            1,
            "seed descriptor fixture pipe"
        );
        assert_eq!(
            unsafe { libc::close(fds[1]) },
            0,
            "close descriptor fixture writer"
        );
        unsafe { OwnedFd::from_raw_fd(fds[0]) }
    }

    let first = pipe_with_byte(b'A');
    let second = pipe_with_byte(b'B');
    let first_fd = first.as_raw_fd();
    let second_fd = second.as_raw_fd();
    let mut command = Command::new("/bin/sh");
    command.args([
        "-c",
        &format!(
            "a=$(dd bs=1 count=1 <&{first_fd} 2>/dev/null); \
             b=$(dd bs=1 count=1 <&{second_fd} 2>/dev/null); test \"$a$b\" = BA"
        ),
    ]);
    let inherited = prepare_inherited_descriptor_remap(
        &mut command,
        &[(first_fd, second_fd), (second_fd, first_fd)],
    )
    .expect("prepare collision-safe remap");
    let output = command.output().expect("spawn collision child");
    drop(inherited);
    assert!(
        output.status.success(),
        "collision-safe descriptor remap must preserve both sources: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
#[cfg(not(unix))]
fn inherit_listener(_command: &mut Command, _source_fd: i32, _target_fd: i32) {}
#[tokio::test]
async fn test_cross_host_cellular_security_uses_dynamic_listeners_and_cleans_up() {
    let _fixture_lock = cellular_process_fixture_lock().await;
    if cfg!(target_os = "macos") {
        return;
    }
    {
        let mut fixture = CrossHostCellularFixture::new().await;
        assert_ne!(fixture.controller_addr.port(), 9500);
        assert_ne!(fixture.artifact_addr.port(), 9600);
        fixture.spawn_controller();
        fixture.spawn_cells();
        let result = fixture.wait_for_success().await;
        assert_eq!(result.artifacts.raw_records().len(), SESSIONS as usize);
        assert!(!dataset_serve_observables(&result).is_empty());
        assert_private_material_absent(&result);
        assert_fixture_private_material_absent(fixture.temporary.path());
        assert!(
            !has_cell_local_native_report(&fixture.controller_cwd),
            "controller scratch must be released after the gate writer closes"
        );
        assert_eq!(fixture.children.live_child_count(), 0);
        assert!(TcpListener::bind(fixture.controller_addr).is_ok());
        assert!(TcpListener::bind(fixture.artifact_addr).is_ok());
    }
}

#[tokio::test]
async fn test_cross_host_wrong_controller_role_key_rejects_before_dataset_dispatch() {
    let _fixture_lock = cellular_process_fixture_lock().await;
    let mut fixture = CrossHostCellularFixture::new().await;
    fixture.spawn_controller_with_roles(true);
    let role = fs::read_to_string(fixture.temporary.path().join("roles"))
        .expect("read role")
        .lines()
        .next()
        .expect("first role")
        .to_owned();
    fixture.spawn_one_cell(0, Path::new(&role));
    let deadline = Instant::now() + Duration::from_secs(15);
    let failures = fixture.wait_for_failure(&[1], deadline).await;
    assert!(
        failures.iter().any(|(_, output)| {
            String::from_utf8_lossy(&output.stderr).contains("AdmissionRejected")
        }),
        "wrong role verification key must be rejected by the production registration handler"
    );
    fixture
        .release_gate_and_wait_for_controller_failure(deadline)
        .await;
    fixture.assert_admission_side_effects(0);
    assert_fixture_private_material_absent(fixture.temporary.path());
    assert!(TcpListener::bind(fixture.controller_addr).is_ok());
    assert!(TcpListener::bind(fixture.artifact_addr).is_ok());
}

#[tokio::test]
async fn test_cross_host_missing_role_material_fails_before_velo_dial() {
    let _fixture_lock = cellular_process_fixture_lock().await;
    let mut fixture = CrossHostCellularFixture::new().await;
    fixture.spawn_one_cell(0, &fixture.temporary.path().join("missing.security"));
    let deadline = Instant::now() + Duration::from_secs(15);
    let failures = fixture.wait_for_failure(&[0], deadline).await;
    assert!(
        failures
            .iter()
            .any(|(_, output)| { String::from_utf8_lossy(&output.stderr).contains("security") }),
        "missing role material must fail in the production bootstrap"
    );
    fixture.assert_no_controller_connection();
    fixture.assert_admission_side_effects(0);
    assert_fixture_private_material_absent(fixture.temporary.path());
}

#[tokio::test]
async fn test_cross_host_duplicate_registration_does_not_start_controller_state() {
    let _fixture_lock = cellular_process_fixture_lock().await;
    let mut fixture = CrossHostCellularFixture::new().await;
    fixture.spawn_controller();
    let role = fs::read_to_string(fixture.temporary.path().join("roles"))
        .expect("read role")
        .lines()
        .next()
        .expect("first role")
        .to_owned();
    fixture.spawn_one_cell(0, Path::new(&role));
    fixture.spawn_one_cell(0, Path::new(&role));
    let deadline = Instant::now() + Duration::from_secs(15);
    let failures = fixture.wait_for_failure(&[1, 2], deadline).await;
    assert_eq!(
        failures
            .iter()
            .filter(|(_, output)| {
                String::from_utf8_lossy(&output.stderr).contains("AdmissionRejected")
            })
            .count(),
        1,
        "exactly the duplicate loser must be rejected by the production registration ledger"
    );
    fixture
        .release_gate_and_wait_for_controller_failure(deadline)
        .await;
    fixture.assert_admission_side_effects(SESSIONS as usize);
    assert_fixture_private_material_absent(fixture.temporary.path());
    assert!(TcpListener::bind(fixture.controller_addr).is_ok());
    assert!(TcpListener::bind(fixture.artifact_addr).is_ok());
}

#[tokio::test]
async fn test_cellular_imported_session_exact_set_shipping_matches_single_cell_raw_records() {
    let _fixture_lock = cellular_process_fixture_lock().await;
    if cfg!(target_os = "macos") {
        return;
    }

    let temporary = tempfile::tempdir().expect("session fixture root");
    let source = write_codex_session_set(temporary.path());

    let baseline_harness = AIPerfHarness::new().await;
    let baseline = run_imported_sessions(&baseline_harness, &source, temporary.path(), 1, false);
    assert!(
        baseline.success(),
        "single-cell imported-session run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        baseline.exit_code,
        baseline.stdout,
        baseline.stderr
    );

    let cellular_harness = AIPerfHarness::new().await;
    let cellular = run_imported_sessions(&cellular_harness, &source, temporary.path(), CELLS, true);
    assert!(
        cellular.success(),
        "{CELLS}-cell imported-session run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        cellular.exit_code,
        cellular.stdout,
        cellular.stderr
    );
    assert!(
        cellular
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "multi-cell imported-session run must go through the cellular controller"
    );

    let observables = dataset_serve_observables(&cellular);
    assert_eq!(
        observables.len(),
        (SESSIONS * CELLS) as usize,
        "each cell must fetch every exact-set source once: {observables:?}"
    );
    assert!(
        observables.iter().all(|line| {
            (line.contains("content_encoding=\"zstd\"") || line.contains("content_encoding=zstd"))
                && line.contains("TLS/authenticated")
        }),
        "imported-session sources must be served over authenticated TLS + zstd: {observables:?}"
    );
    assert!(
        observables.iter().all(|line| {
            line.contains("session-0.jsonl")
                || line.contains("session-1.jsonl")
                || line.contains("session-2.jsonl")
        }),
        "only selected session sources may cross the channel: {observables:?}"
    );
    assert!(
        !observables
            .iter()
            .any(|line| line.contains("credentials.txt") || line.contains("ignored-source.jsonl")),
        "decoy sources must never be served: {observables:?}"
    );
    for text in [cellular.stdout.as_str(), cellular.stderr.as_str()] {
        assert!(
            !text.contains(PRIVATE_SOURCE_SENTINEL),
            "private ignored source content leaked into a cellular artifact"
        );
    }
    assert!(
        regular_artifact_texts(&cellular.artifacts.dir)
            .iter()
            .all(|text| !text.contains(PRIVATE_SOURCE_SENTINEL)),
        "private ignored source content leaked into a cellular artifact"
    );
    assert!(
        dataset_serve_observables(&baseline).is_empty(),
        "single-process run must not expose an artifact server"
    );

    let baseline_raw = sorted_raw_records(&baseline);
    let cellular_raw = sorted_raw_records(&cellular);
    assert_eq!(baseline_raw.len(), SESSIONS as usize);
    for record in cellular.artifacts.raw_records() {
        assert_eq!(
            record["status"], 200,
            "raw imported-session response: {record}"
        );
        assert!(
            !raw_response_text(&record).is_empty(),
            "raw imported-session response has no generated content: {record}"
        );
    }
    assert_eq!(
        baseline_raw, cellular_raw,
        "raw imported request/response set diverged"
    );
}
