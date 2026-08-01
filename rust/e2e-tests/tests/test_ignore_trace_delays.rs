// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use serde_json::{Value, json};

// `--ignore-trace-delays` runs a graph-ir trace's structure without its recorded
// pacing: every node fires as soon as its inputs are ready. The trace below records
// one session whose second and third turns each sit behind a 1.5s delay, so the
// flag's effect is visible in the issued-request gaps rather than inferred.

/// Authored inter-turn delay, in milliseconds, on each continuation turn.
const TURN_DELAY_MS: f64 = 1500.0;
/// Gap floor proving the control run actually waited. Well under the authored
/// delay so ordinary scheduling overhead cannot fail it.
const HONORED_GAP_FLOOR_MS: f64 = 1000.0;
/// Gap ceiling proving the flagged run did not wait. Well above process-level
/// jitter but far below the authored delay.
const IGNORED_GAP_CEILING_MS: f64 = 500.0;

const TURN_CONTENTS: [&str; 3] = ["turn zero", "turn one", "turn two"];

fn turn(content: &str, delay_ms: Option<f64>) -> Value {
    let mut turn = json!({
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4,
    });
    if let Some(delay_ms) = delay_ms {
        turn["delay"] = json!(delay_ms);
    }
    turn
}

fn delayed_trace(h: &AIPerfHarness) -> String {
    let session = json!({
        "session_id": "delayed",
        "turns": [
            turn(TURN_CONTENTS[0], None),
            turn(TURN_CONTENTS[1], Some(TURN_DELAY_MS)),
            turn(TURN_CONTENTS[2], Some(TURN_DELAY_MS)),
        ],
    });
    write_jsonl(h.artifact_path(), "delayed.dag.jsonl", &[session])
        .display()
        .to_string()
}

/// Issue-time gaps between consecutive requests, in milliseconds, plus the user
/// content of each request in issue order.
fn gaps_and_contents(r: &RunResult) -> (Vec<f64>, Vec<String>) {
    let mut records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        TURN_CONTENTS.len(),
        "expected one record per turn, got {}:\nstderr:\n{}",
        records.len(),
        r.stderr
    );
    records.sort_by_key(|record| {
        record["start_perf_ns"]
            .as_i64()
            .unwrap_or_else(|| panic!("record has no start_perf_ns: {record}"))
    });

    let starts: Vec<i64> = records
        .iter()
        .map(|record| record["start_perf_ns"].as_i64().expect("start_perf_ns"))
        .collect();
    let gaps = starts
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) as f64 / 1e6)
        .collect();
    let contents = records
        .iter()
        .map(|record| {
            let messages = &record["payload"]["messages"];
            let last = messages
                .as_array()
                .and_then(|m| m.last())
                .unwrap_or_else(|| panic!("record has no messages: {record}"));
            last["content"]
                .as_str()
                .unwrap_or_else(|| panic!("record message has no content: {record}"))
                .to_string()
        })
        .collect();
    (gaps, contents)
}

fn run_trace(h: &AIPerfHarness, extra: &str) -> RunResult {
    let trace = delayed_trace(h);
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --input-file {trace} --custom-dataset-type dag_jsonl \
             --num-conversations 1 --concurrency 1 --export-level raw {extra} --ui none",
            h.mock.url
        ),
        120,
    );
    assert!(
        r.success(),
        "trace run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );
    r
}

/// Without the flag the authored delays are honored; with it they are dropped.
/// Both halves are needed: the fast run alone would also pass if the fixture's
/// delays never reached the scheduler in the first place.
#[tokio::test]
async fn ignore_trace_delays_drops_authored_pacing_but_keeps_the_trace() {
    let h = AIPerfHarness::new().await;
    let (honored_gaps, honored_contents) = gaps_and_contents(&run_trace(&h, ""));
    for (index, gap) in honored_gaps.iter().enumerate() {
        assert!(
            *gap >= HONORED_GAP_FLOOR_MS,
            "control gap {index} was {gap:.1}ms, expected the authored \
             {TURN_DELAY_MS:.0}ms delay to be honored; gaps: {honored_gaps:?}"
        );
    }

    let h = AIPerfHarness::new().await;
    let (ignored_gaps, ignored_contents) =
        gaps_and_contents(&run_trace(&h, "--ignore-trace-delays"));
    for (index, gap) in ignored_gaps.iter().enumerate() {
        assert!(
            *gap < IGNORED_GAP_CEILING_MS,
            "gap {index} was {gap:.1}ms under --ignore-trace-delays, expected the \
             authored delay to be dropped; gaps: {ignored_gaps:?}"
        );
    }

    // Only the pacing is dropped. The same turns must still be sent, in order.
    assert_eq!(
        honored_contents, TURN_CONTENTS,
        "the control run did not replay the trace in order"
    );
    assert_eq!(
        ignored_contents, TURN_CONTENTS,
        "--ignore-trace-delays must preserve trace structure and order"
    );
}

/// The flag contradicts `--use-think-time-only`, which replays exactly the pacing
/// this one discards. The CLI must reject the pair rather than silently pick one.
#[tokio::test]
async fn ignore_trace_delays_conflicts_with_use_think_time_only() {
    let h = AIPerfHarness::new().await;
    let trace = delayed_trace(&h);
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --input-file {trace} --custom-dataset-type dag_jsonl \
         --num-conversations 1 --concurrency 1 \
         --ignore-trace-delays --use-think-time-only --ui none",
        h.mock.url
    ));

    assert_ne!(
        r.exit_code, 0,
        "conflicting flags were accepted:\n{}",
        r.stdout
    );
    assert!(
        r.stderr
            .contains("--use-think-time-only and --ignore-trace-delays are mutually exclusive"),
        "expected a mutual-exclusion diagnostic; stderr:\n{}",
        r.stderr
    );
}
