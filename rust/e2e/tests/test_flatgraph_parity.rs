// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

// Byte-parity of the flat-graph fast path against the general executor, proven
// through the real `aiperf` binary (an external `aiperf profile` process).
//
// A single-LLM-node `dag_jsonl` program routes through `FlatGraphActor`; the same
// program with `AIPERF_DISABLE_FLATGRAPH=1` routes through the general
// `TraceExecutor`. Both runs use the identical seed/config against the same mock,
// so every deterministic per-record field must be identical. Timing metrics
// (`*_ns`, `http_req_*`, latencies) and the minted `x_request_id` are wall-clock /
// random and are intentionally excluded from the comparison.

use serde_json::Value;

const FIXTURE: &str =
    "/home/anthony/nvidia/projects/aiperf/ajc/rust/tests/fixtures/dag/single_node.dag.jsonl";

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
         --num-conversations 3 --random-seed 7 --ui simple"
    )
}

#[tokio::test]
async fn flatgraph_fast_path_is_byte_identical_to_the_general_executor() {
    let h = AIPerfHarness::new().await;

    // Flat arm: single-node traces route through FlatGraphActor.
    let flat = h.run(&args(&h.mock.url));
    assert!(flat.success(), "flat-arm profile failed:\n{}", flat.stderr);
    let flat_rows = deterministic_projection(&flat);
    assert!(!flat_rows.is_empty(), "flat run produced no profiling records");

    // Full arm: the kill-switch forces every trace onto the general TraceExecutor.
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
