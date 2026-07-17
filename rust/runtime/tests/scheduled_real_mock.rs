// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wall-clock timing against the workspace `aiperf-mock-server` process.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use aiperf_runtime::body_plan::{BodyPlan, JsonBodyMaterializer};
use aiperf_runtime::dataset::materialize::Overrides;
use aiperf_runtime::dataset::segment::SegmentPool;
use aiperf_runtime::fixed_schedule::FixedScheduleConfig;
use aiperf_runtime::multiturn::ConversationSource;
use aiperf_runtime::timing::StopConfig;
use aiperf_runtime::user_centric::UserCentricConfig;
use bytes::Bytes;

mod common;

struct RealMock {
    child: Child,
    base_url: String,
}

impl RealMock {
    fn spawn() -> Option<Self> {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .ok()?
            .local_addr()
            .ok()?
            .port();
        let binary = mock_binary();
        let child = match Command::new(&binary)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--no-tokenizer")
            .arg("--ttft")
            .arg("12")
            .arg("--itl")
            .arg("3")
            .arg("--disable-prefix-cache")
            .arg("--random-seed")
            .arg("7")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => child,
            Err(error) => {
                eprintln!(
                    "SKIP: cannot launch {}: {error} (set AIPERF_MOCK_RS_BIN)",
                    binary.display()
                );
                return None;
            }
        };
        for _ in 0..250 {
            if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
                return Some(Self {
                    child,
                    base_url: format!("http://127.0.0.1:{port}"),
                });
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let mut child = child;
        let _ = child.kill();
        eprintln!("SKIP: aiperf-mock-server did not become ready");
        None
    }
}

impl Drop for RealMock {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn mock_binary() -> PathBuf {
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

fn run_local<F: std::future::Future>(future: F) -> F::Output {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let local = tokio::task::LocalSet::new();
    local.block_on(&runtime, future)
}

/// Verify `BodyPlan::raw` and `JsonBodyMaterializer` preserve authored bytes and
/// tail-splice dispatch overrides as required by segment-unification §4 and
/// endpoint-body-construction §4.
#[test]
fn raw_payload_body_plan_dispatches_byte_exactly_to_the_real_mock() {
    let Some(mock) = RealMock::spawn() else {
        return;
    };

    // An authored chat request body with deliberate whitespace and key order the
    // splicer must preserve verbatim. The model is already the mock's served
    // model so the body is valid wire on its own.
    let authored = Bytes::from_static(
        b"{ \"model\":\"openai/gpt-oss-120b\", \"messages\":[{\"role\":\"user\",\"content\":\"ping\"}] }",
    );
    let mut pool = SegmentPool::new();
    let raw = pool.intern_raw(None, authored.clone()).unwrap();
    let store = pool.freeze();

    // No overrides: byte-identical to the authored body.
    let plan = BodyPlan::raw(raw);
    let verbatim = JsonBodyMaterializer::materialize(&plan, &store, &Overrides::new()).unwrap();
    assert_eq!(
        verbatim, authored,
        "raw body must be byte-identical without overrides"
    );

    // Dispatch overrides for fields absent from the authored body: the tail is
    // spliced immediately before the closing brace, and the authored bytes and
    // trailing whitespace are preserved verbatim (concat, never re-serialize).
    let mut overrides = Overrides::new();
    overrides.set_stream(false);
    overrides.set_max_tokens("max_tokens", 4);
    let dispatched = JsonBodyMaterializer::materialize(&plan, &store, &overrides).unwrap();
    assert_eq!(
        dispatched,
        Bytes::from_static(
            b"{ \"model\":\"openai/gpt-oss-120b\", \"messages\":[{\"role\":\"user\",\"content\":\"ping\"}] ,\"stream\":false,\"max_tokens\":4}"
        ),
    );

    // Use a raw HTTP/1.1 POST to avoid an additional client dev dependency.
    let (status_line, response_body) =
        post_raw(&mock.base_url, "/v1/chat/completions", &dispatched);
    assert!(
        status_line.contains("200"),
        "mock rejected materialized raw body: {status_line}"
    );
    assert!(
        response_body.contains("choices"),
        "mock response missing choices: {response_body}"
    );
}

/// Minimal blocking HTTP/1.1 POST returning `(status_line, body)`; reads the
/// full response and splits headers from body on the blank line.
fn post_raw(base_url: &str, path: &str, body: &[u8]) -> (String, String) {
    use std::io::{Read, Write};

    let hostport = base_url.strip_prefix("http://").expect("http base url");
    let mut stream = std::net::TcpStream::connect(hostport).expect("connect to mock");
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {hostport}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(request.as_bytes()).expect("write headers");
    stream.write_all(body).expect("write body");
    stream.flush().expect("flush");

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read response");
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((text.as_str(), ""));
    let status_line = head.lines().next().unwrap_or_default().to_string();
    (status_line, body.to_string())
}

fn assert_real_ttft_and_lateness(report: &aiperf_runtime::scheduled::ScheduledRunReport) {
    assert_eq!(report.schedule_timing.early_turns, 0);
    assert!(
        report.schedule_timing.max_issue_lateness_ms < 50.0,
        "unexpected scheduler lateness: {:?}",
        report.schedule_timing
    );
    let mean_ttft = report.schedule_timing.mean_ttft_ms.unwrap();
    assert!(
        (8.0..80.0).contains(&mean_ttft),
        "12ms mock TTFT should remain recognizable, got {mean_ttft:.3}ms"
    );
    for turn in &report.turns {
        let ttft_ms = turn.ttft_ns.unwrap() as f64 / 1_000_000.0;
        assert!(
            (8.0..100.0).contains(&ttft_ms),
            "turn {} TTFT {ttft_ms:.3}ms outside real-clock tolerance",
            turn.uuid
        );
        assert!(turn.terminal_offset_ns.unwrap() >= turn.first_token_offset_ns.unwrap());
    }
}

#[test]
fn both_scheduled_strategies_match_real_mock_timing() {
    let Some(mock) = RealMock::spawn() else {
        return;
    };

    let fixed_source: Box<dyn ConversationSource> =
        run_local(common::prepared_source_from_conversations(
            serde_json::json!([
                {"session_id":"a","turns":[
                    {"timestamp":0,"text":"a0","input_length":1,"output_length":2},
                    {"timestamp":140,"text":"a1","input_length":1,"output_length":2}
                ]},
                {"session_id":"b","turns":[
                    {"timestamp":60,"text":"b0","input_length":1,"output_length":2},
                    {"delay":30,"text":"b1","input_length":1,"output_length":2}
                ]}
            ]),
            "model",
            2,
        ));
    let fixed = run_local(common::run_fixed_schedule_online(
        mock.base_url.clone(),
        "model".to_string(),
        fixed_source,
        FixedScheduleConfig {
            auto_offset_timestamps: true,
            start_offset_ms: None,
        },
        false,
    ))
    .unwrap();
    assert_eq!(fixed.performance.request_counts.num_requests, 4);
    assert_eq!(fixed.performance.request_counts.completed_requests, 4);
    assert_real_ttft_and_lateness(&fixed);
    let a = fixed
        .turns
        .iter()
        .filter(|turn| turn.conversation_id == "a")
        .collect::<Vec<_>>();
    // The scheduler anchors the whole grid at `now + SCHEDULE_START_LEAD_NS`
    // (a fixed warm-start lead, `fixed_schedule.rs`), so absolute
    // `scheduled_offset_ns` carries that lead plus sub-millisecond setup jitter.
    // Assert the trace-derived RELATIVE spacing instead — it is invariant to the
    // lead and the jitter cancels in the differences.
    let base = a[0].scheduled_offset_ns;
    assert_eq!(a[1].scheduled_offset_ns - base, 140_000_000);
    let b = fixed
        .turns
        .iter()
        .filter(|turn| turn.conversation_id == "b")
        .collect::<Vec<_>>();
    assert_eq!(b[0].scheduled_offset_ns - base, 60_000_000);
    assert_eq!(
        b[1].scheduled_offset_ns - b[0].terminal_offset_ns.unwrap(),
        30_000_000,
        "relative delay must be anchored to response terminal"
    );

    let user_source: Box<dyn ConversationSource> =
        run_local(common::synthetic_prepared_source(3, 2, 2, None, "model"));
    let user = run_local(common::run_user_centric_online(
        mock.base_url.clone(),
        "model".to_string(),
        user_source,
        UserCentricConfig {
            num_users: 2,
            request_rate: 20.0,
            concurrency: None,
        },
        StopConfig {
            total_expected_requests: Some(8),
            expected_num_sessions: None,
            expected_duration_ns: None,
        },
        false,
    ))
    .unwrap();
    assert_eq!(user.performance.request_counts.num_requests, 8);
    assert_eq!(user.performance.request_counts.completed_requests, 8);
    assert_real_ttft_and_lateness(&user);
    let first_targets = user
        .turns
        .iter()
        .filter(|turn| turn.turn_index == 0)
        .map(|turn| turn.scheduled_offset_ns / 1_000_000)
        .collect::<Vec<_>>();
    // Relative to the first target — invariant to the fixed warm-start grid lead
    // (`SCHEDULE_START_LEAD_NS`) that shifts every absolute offset equally.
    let base = first_targets[0];
    let relative_targets: Vec<i64> = first_targets.iter().map(|t| t - base).collect();
    assert_eq!(relative_targets, vec![0, 50, 150, 300]);
    for session_id in user
        .turns
        .iter()
        .map(|turn| turn.x_correlation_id.as_str())
        .collect::<std::collections::HashSet<_>>()
    {
        let turns = user
            .turns
            .iter()
            .filter(|turn| turn.x_correlation_id == session_id)
            .collect::<Vec<_>>();
        for pair in turns.windows(2) {
            let gap_ms = (pair[1].issued_offset_ns - pair[0].issued_offset_ns) as f64 / 1_000_000.0;
            assert!(
                (95.0..150.0).contains(&gap_ms),
                "per-user gap should track 100ms, got {gap_ms:.3}ms"
            );
            assert!(
                pair[1].issued_offset_ns >= pair[0].terminal_offset_ns.unwrap(),
                "a user's turns must never overlap"
            );
        }
    }
}
