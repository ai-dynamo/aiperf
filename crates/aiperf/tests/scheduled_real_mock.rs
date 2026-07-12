// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wall-clock timing proof against the workspace `aiperf-mock-rs` process.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use aiperf::fixed_schedule::FixedScheduleConfig;
use aiperf::multiturn::{
    ConversationDataset, ConversationSource, DatasetConversationSource, SyntheticConversationSource,
};
use aiperf::run::{run_fixed_schedule_online, run_user_centric_online};
use aiperf::user_centric::UserCentricConfig;
use aiperf::workload::SkeletonWorkload;
use aiperf_timing::StopConfig;

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
        eprintln!("SKIP: aiperf-mock-rs did not become ready");
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

    let binary_name = format!("aiperf-mock-rs{}", std::env::consts::EXE_SUFFIX);
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

fn assert_real_ttft_and_lateness(report: &aiperf::scheduled::ScheduledRunReport) {
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

    let fixed_dataset = ConversationDataset::from_json_or_jsonl(
        r#"{
          "conversations": [
            {"conversation_id":"a","turns":[
              {"timestamp_ms":0,"prompt_text":"a0","input_length":1,"max_output_tokens":2},
              {"timestamp_ms":140,"prompt_text":"a1","input_length":1,"max_output_tokens":2}
            ]},
            {"conversation_id":"b","turns":[
              {"timestamp_ms":60,"prompt_text":"b0","input_length":1,"max_output_tokens":2},
              {"delay_ms":30,"prompt_text":"b1","input_length":1,"max_output_tokens":2}
            ]}
          ]
        }"#,
        1,
        2,
    )
    .unwrap();
    let fixed_source: Box<dyn ConversationSource> =
        Box::new(DatasetConversationSource::new(fixed_dataset));
    let fixed = run_local(run_fixed_schedule_online(
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
    assert_eq!(a[0].scheduled_offset_ns, 0);
    assert_eq!(a[1].scheduled_offset_ns, 140_000_000);
    let b = fixed
        .turns
        .iter()
        .filter(|turn| turn.conversation_id == "b")
        .collect::<Vec<_>>();
    assert_eq!(b[0].scheduled_offset_ns, 60_000_000);
    assert_eq!(
        b[1].scheduled_offset_ns - b[0].terminal_offset_ns.unwrap(),
        30_000_000,
        "relative delay must be anchored to response terminal"
    );

    let user_source: Box<dyn ConversationSource> = Box::new(
        SyntheticConversationSource::new(SkeletonWorkload {
            num_requests: 0,
            input_tokens: 2,
            output_tokens: 2,
            turns: 3,
            think_time_ms: None,
        })
        .unwrap(),
    );
    let user = run_local(run_user_centric_online(
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
    assert_eq!(first_targets, vec![0, 50, 150, 300]);
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
