// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native-binary integration coverage for TraceLab recorded graph input.

mod common;

use std::io::Write;

use common::{Artifacts, assert_credits_balanced, assert_request_count, run};
use flate2::{Compression, write::GzEncoder};
use serde_json::json;

fn row(round: u32, second: u32, input: u64, prefix: u64, output: u64) -> serde_json::Value {
    json!({
        "session_id": "claude:native-binary",
        "round_index": round,
        "model": common::MODEL,
        "input_tokens_total": input,
        "prefix_tokens": prefix,
        "newly_append_tokens": input - prefix,
        "output_tokens": output,
        "timing_events": [
            {
                "event_type": if round == 0 { "user_message" } else { "tool_result" },
                "timestamp": format!("2026-08-25T00:00:{second:02}Z")
            },
            {
                "event_type": "text",
                "timestamp": format!("2026-08-25T00:00:{second:02}.010Z")
            }
        ]
    })
}

#[test]
fn native_binary_replays_a_real_gzip_tracelab_graph() {
    let files = tempfile::tempdir().expect("TraceLab fixture directory");
    let path = files.path().join("trace.jsonl.gz");
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    for row in [row(0, 0, 64, 0, 4), row(1, 1, 128, 64, 5)] {
        writeln!(encoder, "{row}").expect("encode TraceLab row");
    }
    std::fs::write(&path, encoder.finish().expect("finish TraceLab gzip"))
        .expect("write TraceLab fixture");
    let path = path.to_str().expect("UTF-8 TraceLab path");

    let run = run(&[
        "--custom-dataset-type",
        "tracelab",
        "--input-file",
        path,
        "--num-conversations",
        "1",
        "--concurrency",
        "1",
        "--isl-block-size",
        "64",
        "--ignore-trace-delays",
    ]);
    run.assert_success();
    assert_request_count(&run, 2, "TraceLab Graph-IR nodes").expect("both rounds execute");
    assert_credits_balanced(&run).expect("TraceLab graph credits balance");
    let output_lengths = run
        .artifacts
        .jsonl()
        .iter()
        .map(|record| Artifacts::metric(record, "output_sequence_length") as u64)
        .collect::<Vec<_>>();
    assert_eq!(output_lengths, [4, 5]);
}
