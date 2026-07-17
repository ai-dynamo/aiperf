// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2 credential-redaction process coverage.

use std::io::Write;
use std::process::{Command, Output, Stdio};

use serde_json::{Value, json};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_aiperf")
}

fn one_line(output: &Output) -> Value {
    let lines = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    assert_eq!(
        lines.len(),
        1,
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(lines[0]).unwrap()
}

fn run(input: &Value) -> Output {
    let mut child = Command::new(binary())
        .arg("--execute")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(&input["run"]).unwrap().as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn protocol_diagnostics_redact_secret_assignments_and_url_userinfo() {
    for (authored_field, secret) in [
        ("api_key=super-secret-value", "super-secret-value"),
        ("https://user:password@host.test/private", "user:password"),
    ] {
        let output = run(&json!({
            "protocol_version": 2,
            "operation": "execute",
            "run": {
                "benchmark_id": "redaction",
                "artifact_dir": "/tmp/aiperf-redaction-never-created",
                authored_field: true,
                "cfg": {
                    "models": {"items": [{"name": "mock-model"}]},
                    "endpoint": {
                        "type": "chat",
                        "urls": ["http://127.0.0.1:9"],
                        "streaming": true
                    },
                    "datasets": [{"type": "synthetic", "entries": 1}],
                    "phases": [{
                        "name": "profiling",
                        "type": "concurrency",
                        "exclude_from_results": false,
                        "concurrency": 1
                    }],
                    "transport": {"type": "http"},
                    "runtime": {"workers": 1}
                }
            }
        }));
        assert_eq!(output.status.code(), Some(2));
        let stdout = String::from_utf8(output.stdout.clone()).unwrap();
        assert!(!stdout.contains(secret), "credential leaked: {stdout}");
        let terminal = one_line(&output);
        assert_eq!(terminal["event"], "run_terminal");
        assert_eq!(terminal["success"], false);
        assert_eq!(terminal["stage"], "protocol");
        assert!(
            terminal["errors"][0]["message"]
                .as_str()
                .unwrap()
                .contains("<redacted>")
        );
    }
}
