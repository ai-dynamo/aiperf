// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A single-node `dag_jsonl` program must produce the same deterministic records
//! through the flat-graph and general executors. Timing metrics and
//! `x_request_id` are excluded because they vary between online runs.

mod common;
use common::*;

use serde_json::Value;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/dag/single_node.dag.jsonl"
);

/// Deterministic per-record projection: everything a correct flat/full run must
/// reproduce byte-for-byte, excluding wall-clock timing and the random request id.
fn deterministic_projection(r: &RunResult) -> Vec<String> {
    let metric = |record: &Value, tag: &str| -> Option<i64> {
        record["metrics"][tag]["value"].as_f64().map(|v| v as i64)
    };
    let mut rows: Vec<String> = r
        .artifacts
        .jsonl()
        .iter()
        .filter(|record| record["metadata"]["benchmark_phase"] == "profiling")
        .map(|record| {
            let meta = &record["metadata"];
            format!(
                "isl={:?} osl={:?} reasoning={:?} osl_mismatch={:?} \
                 conv={} turn={} xcorr={} cancelled={} error={}",
                metric(record, "input_sequence_length"),
                metric(record, "output_sequence_length"),
                metric(record, "reasoning_token_count"),
                metric(record, "osl_mismatch_diff_pct"),
                meta["conversation_id"],
                meta["turn_index"],
                meta["x_correlation_id"],
                meta["was_cancelled"],
                record["error"],
            )
        })
        .collect();
    rows.sort_unstable();
    rows
}

/// Common profile arguments for the single-node graph program. `--custom-dataset-type
/// dag_jsonl` forces the graph path (a no-fork single-turn session is otherwise
/// ambiguous with a linear dataset); `--num-conversations 3` bounds the run to a
/// single deterministic pass over the fixture's three sessions.
fn args(url: &str) -> String {
    format!(
        "--model openai/gpt-oss-120b --url {url} --endpoint-type chat \
         --input-file {FIXTURE} --custom-dataset-type dag_jsonl \
         --num-conversations 3 --random-seed 7 --tokenizer cl100k_base --ui simple"
    )
}

#[tokio::test]
async fn flatgraph_fast_path_is_byte_identical_to_the_general_executor() {
    let h = AIPerfHarness::new().await;

    let flat = h.run(&args(&h.mock.url));
    assert!(flat.success(), "flat-arm profile failed:\n{}", flat.stderr);
    let flat_rows = deterministic_projection(&flat);
    assert!(
        !flat_rows.is_empty(),
        "flat run produced no profiling records"
    );

    let full = h.run_env(&args(&h.mock.url), &[("AIPERF_DISABLE_FLATGRAPH", "1")]);
    assert!(full.success(), "full-arm profile failed:\n{}", full.stderr);
    let full_rows = deterministic_projection(&full);

    assert_eq!(
        flat_rows.len(),
        full_rows.len(),
        "flat and full arms must produce the same number of records"
    );
    assert_eq!(
        flat_rows, full_rows,
        "the flat fast path must be byte-identical to the general executor on the \
         deterministic per-record surface (ISL/OSL/reasoning/osl_mismatch/conversation/\
         turn/correlation/cancelled/error)"
    );
}
