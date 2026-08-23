// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use serde_json::{Value, json};

// `--system-idle-gap-cap-seconds` bounds how long a graph-ir replay may sit idle
// waiting on a recorded firing gate, without rewriting the trace's timing. The
// trace below records one session whose second and third turns each sit behind a
// 1.5s delay; the cap is set below that, so its effect is visible in the issued
// gaps rather than inferred.
//
// This is the named guard for the field's *projection*, not only its arithmetic:
// `into_authored` used to attach `system_idle_gap_cap_seconds` to the workload
// config only under `weka_semantics` legacy/agentx, which made the flag a silent
// no-op on the graph-ir arm even though `resolve.rs` validates it there and
// `lower_graph` reads it into `NativeGraphDatasetPlan`. Losing the projection
// again produces the uncapped gaps this test rejects.

/// Authored inter-turn delay, in milliseconds, on each continuation turn.
const TURN_DELAY_MS: f64 = 1500.0;
/// Idle cap under test, in seconds. Comfortably below the authored delay so a
/// capped run is distinguishable from an honored one.
const CAP_SECONDS: f64 = 0.5;
/// Gap floor proving the uncapped control run actually waited.
const HONORED_GAP_FLOOR_MS: f64 = 1000.0;
/// Gap ceiling for the capped run: the cap plus generous transport and
/// process-scheduling overhead, still far below the authored delay.
const CAPPED_GAP_CEILING_MS: f64 = 1100.0;

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
        "session_id": "idle-capped",
        "turns": [
            turn(TURN_CONTENTS[0], None),
            turn(TURN_CONTENTS[1], Some(TURN_DELAY_MS)),
            turn(TURN_CONTENTS[2], Some(TURN_DELAY_MS)),
        ],
    });
    write_jsonl(h.artifact_path(), "idle-capped.dag.jsonl", &[session])
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
             --weka-semantics graph-ir \
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

/// Under graph-ir semantics the cap bounds each recorded idle gap while leaving
/// the trace's structure and order intact. Both halves are needed: the capped run
/// alone would also pass if the fixture's delays never reached the scheduler.
#[tokio::test]
async fn system_idle_gap_cap_bounds_graph_ir_replay_waits() {
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
    let (capped_gaps, capped_contents) = gaps_and_contents(&run_trace(
        &h,
        &format!("--system-idle-gap-cap-seconds {CAP_SECONDS}"),
    ));
    for (index, gap) in capped_gaps.iter().enumerate() {
        assert!(
            *gap < CAPPED_GAP_CEILING_MS,
            "gap {index} was {gap:.1}ms under --system-idle-gap-cap-seconds \
             {CAP_SECONDS}, expected the wait to be capped; gaps: {capped_gaps:?}"
        );
    }

    // Only the pacing is bounded. The same turns must still be sent, in order.
    assert_eq!(
        honored_contents, TURN_CONTENTS,
        "the control run did not replay the trace in order"
    );
    assert_eq!(
        capped_contents, TURN_CONTENTS,
        "--system-idle-gap-cap-seconds must preserve trace structure and order"
    );
}

/// The cap is a Weka-replay knob. Outside a replay mode it is rejected rather
/// than silently ignored -- and the diagnostic names both supported arms.
#[tokio::test]
async fn system_idle_gap_cap_requires_a_weka_replay_mode() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count 1 --concurrency 1 --system-idle-gap-cap-seconds 1 --ui none",
        h.mock.url
    ));

    assert_ne!(
        r.exit_code, 0,
        "the cap was accepted outside a replay mode:\n{}",
        r.stdout
    );
    assert!(
        r.stderr
            .contains("--system-idle-gap-cap-seconds requires a Weka replay mode"),
        "expected a replay-mode diagnostic; stderr:\n{}",
        r.stderr
    );
}
