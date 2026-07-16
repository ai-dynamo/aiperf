// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

// Integration tests for the `aiperf chat` command against the mock server.
//
// Complements the unit tests (which cover the pure parsing/metric logic) by
// exercising the full path end to end: real HTTP streaming, the metric
// pipeline, and the printed stats block. Uses the in-repo mock server, which
// reports `prompt_tokens_details.cached_tokens` so the cache-hit line resolves.
//
// Statefulness is asserted via the prompt-token count in each turn's `Cache:`
// line (the server counts the real prompt it received): with history it grows
// turn over turn; with `--no-history` and an identical message it does not.

/// Per-turn prompt-token counts parsed from `Cache: <hit>/<prompt> ...` lines.
///
/// Captures the prompt-token count (ISL) from each
/// `<hit>/<prompt> prompt tokens cached` fragment, in order.
fn prompt_token_counts(stdout: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for (idx, _) in stdout.match_indices("prompt tokens cached") {
        // Walk back over " prompt tokens cached" to the "<hit>/<prompt>" pair
        // and pull the digits immediately after the '/'.
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

/// `--quick` streams one reply and prints the stats block, including the cache
/// line (prefix caches are server-side, so even a single-shot request reports a
/// hit rate when the server surfaces cached tokens).
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

/// Default mode prints the ITL/decode + cache lines per turn, and the prompt
/// grows turn over turn because history is resent.
///
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
    // The full per-turn block (all four metrics) prints for each turn.
    assert_eq!(stdout.matches("TTFT:").count(), 2);
    for label in ["TPS:", "ITL:", "Cache:"] {
        assert!(stdout.contains(label));
    }
    let prompts = prompt_token_counts(&stdout);
    assert_eq!(prompts.len(), 2);
    // Turn 2 resends turn 1 (user + assistant), so its prompt is larger.
    assert!(prompts[1] > prompts[0]);
}

/// `--no-history` sends each message independently: an identical message yields
/// an identical prompt size across turns (no history).
///
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
    // No history resent -> identical message -> identical prompt size.
    assert_eq!(prompts[0], prompts[1]);
}

/// Run `aiperf chat` interactively, feeding `stdin_text` then EOF.
///
/// Returns captured stdout; asserts a clean exit. The shared harness always
/// wires stdin to `/dev/null`, so this helper shells out directly to reproduce
/// the Python `asyncio.create_subprocess_exec` + `communicate(input=...)` path.
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
