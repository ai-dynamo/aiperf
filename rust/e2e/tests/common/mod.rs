// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full-stack integration-test harness for the native `aiperf` CLI.
//!
//! The harness runs an in-process mock server on loopback, invokes `aiperf`
//! as a subprocess, and reads its artifact tree as `serde_json::Value`.
//! `AIPERF_RUNTIME_ENGINE=python` runs through `python -m aiperf.cli` because
//! that execution-engine selector is handled by the Python frontend.

#![allow(dead_code)]

mod raw_jsonl;
// `mod common` is compiled separately for each test binary.
#[allow(unused_imports)]
pub use raw_jsonl::{
    RawRecordTiming, TunedExpectations, assert_raw_records_timing_and_data,
    assert_raw_records_timing_self_consistent, assert_raw_records_timing_self_consistent_model,
    extract_timing, timing_fast_forwarded, tuned_mock_config,
};

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener as StdTcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub use aiperf_mock_server::config::MockServerConfig;
use aiperf_mock_server::{AppState, build_router};

/// Default model / tokenizer used by every harness run.
pub const DEFAULT_MODEL: &str = "openai/gpt-oss-120b";
/// Default concurrency used by simple smoke tests.
pub const DEFAULT_CONCURRENCY: u32 = 2;
/// Default request count used by simple smoke tests.
pub const DEFAULT_REQUEST_COUNT: u32 = 10;

/// Default subprocess timeout, in seconds.
const DEFAULT_TIMEOUT_SECS: u64 = 300;

/// An in-process `aiperf-mock-server` server bound to a random loopback port.
///
/// The Axum router runs on the `MockServer`'s own multi-threaded tokio runtime,
/// not the ambient `#[tokio::test]` runtime. That decoupling is deliberate: the
/// harness blocks the test thread on the `aiperf` subprocess, so a server that
/// depended on the ambient (typically `current_thread`) runtime being polled
/// would stall and never accept connections. The owned runtime's worker threads
/// drive the accept loop independently; dropping the `MockServer` shuts it down.
pub struct MockServer {
    /// Base URL, e.g. `http://127.0.0.1:<port>`.
    pub url: String,
    /// The bound loopback port.
    pub port: u16,
    /// KServe gRPC URL, e.g. `grpc://127.0.0.1:<grpc_port>`, when the server was
    /// started with the gRPC listener enabled ([`MockServer::start_with_grpc`]).
    pub grpc_url: Option<String>,
    /// The bound gRPC loopback port, when enabled.
    pub grpc_port: Option<u16>,
    // Owned runtime whose worker threads drive the accept loop. Dropping it
    // shuts the server down. Kept last so it drops after everything else.
    runtime: Option<tokio::runtime::Runtime>,
}

impl MockServer {
    /// Start a fast, tokenizer-free mock server on a random port.
    pub fn start() -> Self {
        let mut cfg = MockServerConfig::default();
        cfg.fast = true;
        cfg.workers = 8;
        cfg.no_tokenizer = true;
        Self::start_with(cfg)
    }

    /// Start a mock server from an explicit config. The `port`/`host` fields are
    /// overridden: the harness always binds `127.0.0.1:0` (random free port).
    pub fn start_with(cfg: MockServerConfig) -> Self {
        Self::start_inner(cfg, false)
    }

    /// Like [`start_with`](Self::start_with) but additionally serves the KServe
    /// OIP v2 gRPC service on a second random loopback port, exposed as
    /// [`grpc_url`](Self::grpc_url). Both listeners share one `AppState`.
    pub fn start_with_grpc(cfg: MockServerConfig) -> Self {
        Self::start_inner(cfg, true)
    }

    fn start_inner(mut cfg: MockServerConfig, with_grpc: bool) -> Self {
        cfg = cfg.apply_flags();

        // Bind synchronously so the port is known and already listening before
        // we hand the socket to axum's accept loop.
        let std_listener = StdTcpListener::bind("127.0.0.1:0").expect("bind mock server listener");
        let port = std_listener.local_addr().expect("listener addr").port();
        std_listener
            .set_nonblocking(true)
            .expect("set listener nonblocking");

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("build mock server runtime");

        // `AppState::build` needs a runtime handle to start the batch scheduler.
        let state: Arc<AppState> = {
            let _guard = runtime.enter();
            AppState::build(cfg)
        };

        // Optional KServe gRPC listener on its own port, sharing the same state.
        let (grpc_url, grpc_port) = if with_grpc {
            let grpc_std =
                StdTcpListener::bind("127.0.0.1:0").expect("bind mock server grpc listener");
            let grpc_port = grpc_std.local_addr().expect("grpc listener addr").port();
            drop(grpc_std);
            let grpc_addr = std::net::SocketAddr::from(([127, 0, 0, 1], grpc_port));
            let grpc_state = state.clone();
            runtime.spawn(async move {
                let _ = aiperf_mock_server::grpc::serve_grpc(grpc_addr, grpc_state).await;
            });
            (
                Some(format!("grpc://127.0.0.1:{grpc_port}")),
                Some(grpc_port),
            )
        } else {
            (None, None)
        };

        let router = build_router(state);
        runtime.spawn(async move {
            let listener = tokio::net::TcpListener::from_std(std_listener)
                .expect("adopt std listener into tokio");
            // A serve error only matters to a test that is watching the socket;
            // it will surface as a connection failure there.
            let _ = axum::serve(listener, router).await;
        });

        let url = format!("http://127.0.0.1:{port}");
        wait_for_health(port);
        if let Some(grpc_port) = grpc_port {
            wait_for_tcp(grpc_port);
        }
        Self {
            url,
            port,
            grpc_url,
            grpc_port,
            runtime: Some(runtime),
        }
    }

    /// DCGM metrics scrape URLs (`/dcgm1/metrics`, `/dcgm2/metrics`).
    pub fn dcgm_urls(&self) -> Vec<String> {
        vec![
            format!("{}/dcgm1/metrics", self.url),
            format!("{}/dcgm2/metrics", self.url),
        ]
    }

    /// Per-backend server-metrics scrape URLs, keyed by backend name.
    pub fn server_metrics_urls(&self) -> HashMap<String, String> {
        let u = &self.url;
        let mut m = HashMap::new();
        m.insert("aiperf".to_string(), format!("{u}/metrics"));
        m.insert("vllm".to_string(), format!("{u}/vllm/metrics"));
        m.insert("sglang".to_string(), format!("{u}/sglang/metrics"));
        m.insert("trtllm".to_string(), format!("{u}/trtllm/metrics"));
        m.insert(
            "dynamo_frontend".to_string(),
            format!("{u}/dynamo_frontend/metrics"),
        );
        m.insert(
            "dynamo_prefill".to_string(),
            format!("{u}/dynamo_component/prefill/metrics"),
        );
        m.insert(
            "dynamo_decode".to_string(),
            format!("{u}/dynamo_component/decode/metrics"),
        );
        m
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        // Don't block the test thread waiting for in-flight tasks to unwind.
        if let Some(rt) = self.runtime.take() {
            rt.shutdown_background();
        }
    }
}

/// Poll `GET /health` up to 50 times (100ms apart) until it returns HTTP 200.
///
/// Uses a raw synchronous HTTP/1.0 request so this works from either a
/// `current_thread` or multi-thread tokio test runtime without needing to await.
fn wait_for_health(port: u16) {
    for _ in 0..50 {
        if health_ok(port) {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("mock server on port {port} never became healthy");
}

/// Poll a raw TCP connect up to 50 times (100ms apart) until the port accepts.
/// Used for the gRPC listener, which has no HTTP `/health` route.
fn wait_for_tcp(port: u16) {
    for _ in 0..50 {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("mock gRPC server on port {port} never became reachable");
}

fn health_ok(port: u16) -> bool {
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let req = "GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = Vec::new();
    if stream.read_to_end(&mut buf).is_err() {
        return false;
    }
    let head = String::from_utf8_lossy(&buf);
    head.starts_with("HTTP/1.") && head.contains(" 200 ")
}

/// A full-stack harness: an in-process mock server plus a fresh artifact dir.
pub struct AIPerfHarness {
    /// The in-process mock target.
    pub mock: MockServer,
    /// A fresh temp dir passed to `--artifact-dir`.
    pub artifact_dir: tempfile::TempDir,
}

impl AIPerfHarness {
    /// Start a default (fast, 8-worker, tokenizer-free) mock plus a temp dir.
    pub async fn new() -> Self {
        Self::from_mock(MockServer::start())
    }

    /// Start a mock from an explicit config plus a temp dir.
    pub async fn new_with(cfg: MockServerConfig) -> Self {
        Self::from_mock(MockServer::start_with(cfg))
    }

    /// Start a default mock with the KServe gRPC listener enabled, plus a temp
    /// dir. The gRPC target URL is `self.mock.grpc_url`.
    pub async fn new_with_grpc() -> Self {
        let mut cfg = MockServerConfig::default();
        cfg.fast = true;
        cfg.workers = 8;
        cfg.no_tokenizer = true;
        Self::from_mock(MockServer::start_with_grpc(cfg))
    }

    fn from_mock(mock: MockServer) -> Self {
        let artifact_dir = tempfile::TempDir::new().expect("create artifact tempdir");
        Self { mock, artifact_dir }
    }

    /// Path of the artifact directory.
    pub fn artifact_path(&self) -> &Path {
        self.artifact_dir.path()
    }

    /// Run `aiperf profile <args> --artifact-dir <dir> --tokenizer <model>`.
    pub fn run(&self, profile_args: &str) -> RunResult {
        self.run_timeout(profile_args, DEFAULT_TIMEOUT_SECS)
    }

    /// Like [`run`](Self::run) but with an explicit timeout in seconds.
    pub fn run_timeout(&self, profile_args: &str, secs: u64) -> RunResult {
        let mut args = vec!["profile".to_string()];
        args.extend(shell_split(profile_args));
        args.push("--artifact-dir".to_string());
        args.push(self.artifact_path().display().to_string());
        // An explicit tokenizer always takes precedence over the harness default.
        if !args.iter().any(|a| a == "--tokenizer") {
            args.push("--tokenizer".to_string());
            args.push(DEFAULT_MODEL.to_string());
        }
        self.exec(args, secs)
    }

    /// Like [`run`](Self::run) but with extra environment variables set on the
    /// `aiperf` subprocess. Applies [`run`](Self::run)'s tokenizer and artifact
    /// arguments.
    pub fn run_env(&self, profile_args: &str, extra_env: &[(&str, &str)]) -> RunResult {
        let mut args = vec!["profile".to_string()];
        args.extend(shell_split(profile_args));
        args.push("--artifact-dir".to_string());
        args.push(self.artifact_path().display().to_string());
        if !args.iter().any(|a| a == "--tokenizer") {
            args.push("--tokenizer".to_string());
            args.push(DEFAULT_MODEL.to_string());
        }
        self.exec_env(args, DEFAULT_TIMEOUT_SECS, extra_env)
    }

    /// Run an arbitrary non-profile subcommand (e.g. `plot ...`). No
    /// `--artifact-dir`/`--tokenizer` are appended and no server is required.
    pub fn run_no_server(&self, args: &str) -> RunResult {
        self.exec(shell_split(args), DEFAULT_TIMEOUT_SECS)
    }

    fn exec(&self, args: Vec<String>, timeout_secs: u64) -> RunResult {
        self.exec_env(args, timeout_secs, &[])
    }

    fn exec_env(
        &self,
        args: Vec<String>,
        timeout_secs: u64,
        extra_env: &[(&str, &str)],
    ) -> RunResult {
        // The Python frontend owns the `AIPERF_RUNTIME_ENGINE=python` selector.
        let wants_python_engine = extra_env
            .iter()
            .any(|(k, v)| *k == "AIPERF_RUNTIME_ENGINE" && *v == "python");

        let (program, mut cmd) = if wants_python_engine {
            let python = python_binary();
            let mut c = Command::new(&python);
            c.arg("-m").arg("aiperf.cli");
            (format!("{python} -m aiperf.cli"), c)
        } else {
            let bin = exec_binary();
            let c = Command::new(&bin);
            (bin, c)
        };
        cmd.args(&args);
        cmd.env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .env("PYTHONUNBUFFERED", "1")
            .env("MALLOC_ARENA_MAX", "2")
            .env("AIPERF_EXEC_BIN", exec_binary())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (key, value) in extra_env {
            cmd.env(key, value);
        }

        let mut child = cmd.spawn().unwrap_or_else(|e| {
            panic!("failed to spawn `{program}`: {e}");
        });

        // Drain both pipes on dedicated threads so a full OS pipe buffer can
        // never deadlock the child while we poll for exit.
        let mut out_pipe = child.stdout.take().expect("child stdout");
        let mut err_pipe = child.stderr.take().expect("child stderr");
        let out_thread = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = out_pipe.read_to_string(&mut s);
            s
        });
        let err_thread = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = err_pipe.read_to_string(&mut s);
            s
        });

        let deadline = Instant::now() + Duration::from_secs(timeout_secs);
        let status = loop {
            match child.try_wait().expect("try_wait on aiperf child") {
                Some(status) => break status,
                None => {
                    if Instant::now() >= deadline {
                        // Escalate: SIGINT, wait 10s, then SIGKILL.
                        cancel_child(&child);
                        let hard = Instant::now() + Duration::from_secs(10);
                        break loop {
                            if let Some(s) = child.try_wait().expect("try_wait after SIGINT") {
                                break s;
                            }
                            if Instant::now() >= hard {
                                let _ = child.kill();
                                break child.wait().expect("wait after SIGKILL");
                            }
                            std::thread::sleep(Duration::from_millis(100));
                        };
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        };

        let stdout = out_thread.join().unwrap_or_default();
        let stderr = err_thread.join().unwrap_or_default();
        let exit_code = status.code().unwrap_or(-1);

        RunResult {
            exit_code,
            stdout,
            stderr,
            artifacts: ArtifactReader {
                dir: self.artifact_path().to_path_buf(),
            },
        }
    }
}

/// Send SIGINT to the child so aiperf's handler can flush partial artifacts.
#[cfg(unix)]
fn cancel_child(child: &std::process::Child) {
    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;
    let pid = Pid::from_raw(child.id() as i32);
    let _ = kill(pid, Signal::SIGINT);
}

/// Non-unix stub: no POSIX SIGINT, so the caller falls through to the hard
/// `child.kill()` escalation. Keeps the crate compiling on windows-msvc.
#[cfg(not(unix))]
fn cancel_child(_child: &std::process::Child) {}

/// Resolve the Python interpreter for `AIPERF_RUNTIME_ENGINE=python`.
fn python_binary() -> String {
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        let candidate = PathBuf::from(&venv).join("bin").join("python");
        if candidate.exists() {
            return candidate.display().to_string();
        }
    }
    "python3".to_string()
}

/// Resolve the unified `aiperf` execution binary (entry point + execution engine).
///
/// The harness spawns this binary directly as the entry point; it re-execs itself
/// (`current_exe()`) in the internal `--execute` mode, or honors `AIPERF_EXEC_BIN`
/// as an override. The Python frontend reads the same variable. Priority:
/// 1. `AIPERF_EXEC_BIN` env var (already set by the user or outer harness)
/// 2. `target/release/aiperf` then `target/debug/aiperf` relative to the Cargo
///    workspace root (derived from this file's `CARGO_MANIFEST_DIR` at compile time)
/// 3. `aiperf` on PATH as last resort
pub fn exec_binary() -> String {
    if let Ok(v) = std::env::var("AIPERF_EXEC_BIN") {
        if !v.is_empty() {
            return v;
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or(&manifest);
    let suffix = std::env::consts::EXE_SUFFIX;
    for profile in ["release", "debug"] {
        let candidate = workspace
            .join("target")
            .join(profile)
            .join(format!("aiperf{suffix}"));
        if candidate.exists() {
            return candidate.display().to_string();
        }
    }
    format!("aiperf{suffix}")
}

/// Result of one `aiperf` subprocess run.
pub struct RunResult {
    /// Process exit code (`-1` if terminated by signal / unknown).
    pub exit_code: i32,
    /// Captured stdout.
    pub stdout: String,
    /// Captured stderr.
    pub stderr: String,
    /// Reader over the emitted artifact tree.
    pub artifacts: ArtifactReader,
}

impl RunResult {
    /// True when the process exited 0.
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }
}

/// Reads and parses artifacts under an artifact directory using glob patterns.
pub struct ArtifactReader {
    /// The artifact root directory.
    pub dir: PathBuf,
}

impl ArtifactReader {
    /// Load `*aiperf.json` as a `Value`; `Value::Null` when absent.
    pub fn json(&self) -> serde_json::Value {
        match self.find_file("**/*aiperf.json") {
            Some(p) => read_json(&p),
            None => serde_json::Value::Null,
        }
    }

    /// Load `profile_export.jsonl`, one `Value` per non-empty line.
    pub fn jsonl(&self) -> Vec<serde_json::Value> {
        self.read_jsonl("**/*profile_export.jsonl")
    }

    /// Load `profile_export_raw.jsonl`, one `Value` per non-empty line.
    pub fn raw_records(&self) -> Vec<serde_json::Value> {
        self.read_jsonl("**/*profile_export_raw.jsonl")
    }

    /// Load `inputs.json`; `Value::Null` when absent.
    pub fn inputs(&self) -> serde_json::Value {
        match self.find_file("**/inputs.json") {
            Some(p) => read_json(&p),
            None => serde_json::Value::Null,
        }
    }

    /// Load `*aiperf.csv` as text; empty string when absent.
    pub fn csv(&self) -> String {
        match self.find_file("**/*aiperf.csv") {
            Some(p) => std::fs::read_to_string(p).unwrap_or_default(),
            None => String::new(),
        }
    }

    /// Load `*server_metrics_export.json`; `Value::Null` when absent.
    pub fn server_metrics_json(&self) -> serde_json::Value {
        match self.find_file("**/*server_metrics_export.json") {
            Some(p) => read_json(&p),
            None => serde_json::Value::Null,
        }
    }

    /// Load `*server_metrics_export.jsonl`, one `Value` per non-empty line.
    pub fn server_metrics_jsonl(&self) -> Vec<serde_json::Value> {
        self.read_jsonl("**/*server_metrics_export.jsonl")
    }

    /// First file matching a glob (relative to the artifact dir), if any.
    pub fn find_file(&self, glob_pat: &str) -> Option<PathBuf> {
        let pattern = self.dir.join(glob_pat);
        let pattern = pattern.to_string_lossy();
        glob::glob(&pattern)
            .ok()?
            .filter_map(Result::ok)
            .find(|p| p.is_file())
    }

    /// Find a file by glob and parse it as JSON; `Value::Null` when absent.
    pub fn read_json_file(&self, glob_pat: &str) -> serde_json::Value {
        match self.find_file(glob_pat) {
            Some(p) => read_json(&p),
            None => serde_json::Value::Null,
        }
    }

    /// `json()["request_count"]["avg"]` as f64, or 0.0.
    pub fn request_count(&self) -> f64 {
        self.json()
            .get("request_count")
            .and_then(|v| v.get("avg"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
    }

    /// `json()["was_cancelled"]` as bool, or false.
    pub fn was_cancelled(&self) -> bool {
        self.json()
            .get("was_cancelled")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    fn read_jsonl(&self, glob_pat: &str) -> Vec<serde_json::Value> {
        let Some(path) = self.find_file(glob_pat) else {
            return Vec::new();
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Vec::new();
        };
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    }
}

fn read_json(path: &Path) -> serde_json::Value {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
        Err(_) => serde_json::Value::Null,
    }
}

/// Write JSONL records to `dir/filename`, one JSON object per line.
pub fn write_jsonl(dir: &Path, filename: &str, records: &[serde_json::Value]) -> PathBuf {
    let path = dir.join(filename);
    let mut body = String::new();
    for r in records {
        body.push_str(&serde_json::to_string(r).expect("serialize jsonl record"));
        body.push('\n');
    }
    std::fs::write(&path, body).expect("write jsonl file");
    path
}

/// Write a simple CSV (comma-joined, no quoting) to `dir/filename`.
pub fn write_csv(dir: &Path, filename: &str, headers: &[&str], rows: &[Vec<String>]) -> PathBuf {
    let path = dir.join(filename);
    let mut body = String::new();
    body.push_str(&headers.join(","));
    body.push('\n');
    for row in rows {
        body.push_str(&row.join(","));
        body.push('\n');
    }
    std::fs::write(&path, body).expect("write csv file");
    path
}

/// Write raw text to `dir/filename`.
pub fn write_text(dir: &Path, filename: &str, content: &str) -> PathBuf {
    let path = dir.join(filename);
    std::fs::write(&path, content).expect("write text file");
    path
}

/// Minimal shell-style splitter: whitespace-delimited, honoring single and
/// double quotes (quotes are stripped). Backslash escaping is not interpreted,
/// matching the POSIX-path-friendly behavior the Python harness relies on.
fn shell_split(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_token = false;
    let mut quote: Option<char> = None;

    for ch in input.chars() {
        match quote {
            Some(q) => {
                if ch == q {
                    quote = None;
                } else {
                    cur.push(ch);
                }
            }
            None => {
                if ch == '\'' || ch == '"' {
                    quote = Some(ch);
                    in_token = true;
                } else if ch.is_whitespace() {
                    if in_token {
                        out.push(std::mem::take(&mut cur));
                        in_token = false;
                    }
                } else {
                    cur.push(ch);
                    in_token = true;
                }
            }
        }
    }
    if in_token {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::shell_split;

    #[test]
    fn splits_plain_args() {
        assert_eq!(
            shell_split("--model foo --concurrency 2"),
            vec!["--model", "foo", "--concurrency", "2"]
        );
    }

    #[test]
    fn honors_quotes() {
        assert_eq!(
            shell_split("--seq \"64|10,32|8\" --x 'a b'"),
            vec!["--seq", "64|10,32|8", "--x", "a b"]
        );
    }

    #[test]
    fn empty_is_empty() {
        assert!(shell_split("   ").is_empty());
    }
}
