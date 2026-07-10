// crates/aiperf-transport/tests/common/mod.rs
//! Spawns the external `aiperf-mock-rs` binary for integration tests.
#![allow(dead_code)]

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

fn binary() -> String {
    std::env::var("AIPERF_MOCK_RS_BIN").unwrap_or_else(|_| "aiperf-mock-rs".to_string())
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
                eprintln!("SKIP: cannot launch {bin}: {e} (set AIPERF_MOCK_RS_BIN)");
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
