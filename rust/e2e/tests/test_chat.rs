// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

fn prompt_token_counts(stdout: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for (idx, _) in stdout.match_indices("prompt tokens cached") {
        let prefix = &stdout[..idx];
        let prefix = prefix.trim_end();
        if let Some(slash) = prefix.rfind('/') {
            let digits: String = prefix[slash + 1..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(n) = digits.parse::<u32>() {
                out.push(n);
            }
        }
    }
    out
}

#[tokio::test]
async fn test_quick_prints_stats() {
    let h = AIPerfHarness::new().await;
    let r = h.run_no_server(&format!(
        "chat --model mock-model --url {} --tokenizer builtin --quick \"hello there, who are you?\"",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0, "{}", r.stderr);
    assert!(r.stdout.contains("TTFT:"));
    assert!(r.stdout.contains("TPS:"));
    assert!(r.stdout.contains("Cache:"));
}

/// requires: interactive `aiperf chat` stdin feeding (harness runs stdin=null).
#[tokio::test]
#[ignore]
async fn test_multi_turn_resends_history() {
    let h = AIPerfHarness::new().await;
    let stdout = run_chat_over_stdin(
        &h.mock.url,
        "tell me a short story\ncontinue the story\n",
        &[],
    );
    assert_eq!(stdout.matches("TTFT:").count(), 2);
    for label in ["TPS:", "ITL:", "Cache:"] {
        assert!(stdout.contains(label));
    }
    let prompts = prompt_token_counts(&stdout);
    assert_eq!(prompts.len(), 2);
    assert!(prompts[1] > prompts[0]);
}

/// requires: interactive `aiperf chat` stdin feeding (harness runs stdin=null).
#[tokio::test]
#[ignore]
async fn test_no_history_is_stateless() {
    let h = AIPerfHarness::new().await;
    let stdout = run_chat_over_stdin(
        &h.mock.url,
        "repeat this exactly\nrepeat this exactly\n",
        &["--no-history"],
    );
    assert_eq!(stdout.matches("TTFT:").count(), 2);
    assert!(stdout.contains("ITL:"));
    let prompts = prompt_token_counts(&stdout);
    assert_eq!(prompts.len(), 2);
    assert_eq!(prompts[0], prompts[1]);
}

/// Bypasses the shared harness because it wires stdin to `/dev/null`.
fn run_chat_over_stdin(url: &str, stdin_text: &str, extra_args: &[&str]) -> String {
    use std::io::{Read, Write};
    use std::process::{Command, Stdio};

    let mut cmd = Command::new(exec_binary());
    cmd.arg("chat")
        .arg("--model")
        .arg("mock-model")
        .arg("--url")
        .arg(url)
        .arg("--tokenizer")
        .arg("builtin");
    for a in extra_args {
        cmd.arg(a);
    }
    cmd.env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("spawn aiperf chat");
    child
        .stdin
        .take()
        .expect("child stdin")
        .write_all(stdin_text.as_bytes())
        .expect("write stdin");

    let mut stdout = String::new();
    child
        .stdout
        .take()
        .expect("child stdout")
        .read_to_string(&mut stdout)
        .expect("read stdout");
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .expect("child stderr")
        .read_to_string(&mut stderr)
        .expect("read stderr");

    let status = child.wait().expect("wait aiperf chat");
    assert_eq!(status.code().unwrap_or(-1), 0, "{stderr}");
    stdout
}
