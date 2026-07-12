// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process proof that typed protocol-v2 failures never echo authored credentials.

use std::io::Write;
use std::process::{Command, Output, Stdio};

use serde_json::{Value, json};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_aiperf-runner")
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

fn distribution_id() -> String {
    let output = Command::new(binary())
        .arg("--capabilities")
        .output()
        .unwrap();
    assert!(output.status.success());
    one_line(&output)["distribution_id"]
        .as_str()
        .unwrap()
        .to_owned()
}

fn run(input: &Value) -> Output {
    let mut child = Command::new(binary())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(input).unwrap().as_bytes())
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
            "expected_distribution_id": distribution_id(),
            "run": {
                authored_field: true,
                "identity": {"benchmark_id": "redaction-proof"}
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
