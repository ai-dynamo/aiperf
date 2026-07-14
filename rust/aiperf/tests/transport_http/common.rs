// rust/transport-http/tests/common/mod.rs
//! Spawns the workspace `aiperf-mock-server` binary for integration tests.
#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Stdio;

use tokio::net::TcpStream;
use tokio::process::{Child, Command};

pub struct MockServer {
    child: Child,
    pub base_url: String,
}

fn free_port() -> u16 {
    // Bind :0, read the assigned port, drop the listener.
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}

fn binary() -> PathBuf {
    if let Some(path) = std::env::var_os("AIPERF_MOCK_RS_BIN") {
        return PathBuf::from(path);
    }

    let binary_name = format!("aiperf-mock-server{}", std::env::consts::EXE_SUFFIX);
    if let Ok(current_exe) = std::env::current_exe()
        && let Some(profile_dir) = current_exe.parent().and_then(|deps_dir| deps_dir.parent())
    {
        let candidate = profile_dir.join(&binary_name);
        if candidate.is_file() {
            return candidate;
        }
    }

    let target_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target");
    for profile in ["debug", "release"] {
        let candidate = target_dir.join(profile).join(&binary_name);
        if candidate.is_file() {
            return candidate;
        }
    }

    PathBuf::from(binary_name)
}

impl MockServer {
    /// Spawn the mock on a free port with `extra_args`. Returns `None` (and
    /// prints a skip note) if the binary can't be launched. `--no-tokenizer` is
    /// always passed so the server starts fast and offline (no HF download).
    pub async fn spawn(extra_args: &[&str]) -> Option<MockServer> {
        let port = free_port();
        let bin = binary();
        let mut cmd = Command::new(&bin);
        cmd.arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--no-tokenizer")
            .args(extra_args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "SKIP: cannot launch {}: {e} (set AIPERF_MOCK_RS_BIN)",
                    bin.display()
                );
                return None;
            }
        };
        let base_url = format!("http://127.0.0.1:{port}");
        // Poll for readiness (up to ~5s).
        for _ in 0..250 {
            if TcpStream::connect(("127.0.0.1", port)).await.is_ok() {
                return Some(MockServer { child, base_url });
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        eprintln!("SKIP: mock server did not become ready on port {port}");
        let mut c = child;
        let _ = c.start_kill();
        None
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

/// Run `fut` to completion on a fresh current-thread runtime + `LocalSet`.
/// Mirrors the per-test harness (`new_current_thread().enable_all()` +
/// `LocalSet::block_on`) so behavior is identical to the inline version.
pub fn run_local<F: std::future::Future>(fut: F) -> F::Output {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, fut)
}

/// Build the streaming chat-completions request body used by the tests.
/// Byte-identical to the per-file `chat_body(model)` builders.
pub fn chat_body(model: &str) -> bytes::Bytes {
    bytes::Bytes::from(
        serde_json::to_vec(&serde_json::json!({
            "model": model,
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hello"}],
        }))
        .unwrap(),
    )
}
