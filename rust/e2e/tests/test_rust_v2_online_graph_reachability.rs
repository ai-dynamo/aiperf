// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

// Online graph wire checks requiring protocol capture and a body-recording server.

use serde_json::json;

/// The three-session graph fixture (root -> fork/spawn -> continuation).
fn graph_rows() -> Vec<serde_json::Value> {
    vec![
        json!({
            "session_id": "root",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "root-0"}],
                    "forks": [{"child": "fork", "background": true}],
                    "spawns": [{"children": ["spawn"], "join_at": 1}],
                    "max_tokens": 1,
                },
                {
                    "messages": [{"role": "user", "content": "root-1"}],
                    "max_tokens": 1,
                },
            ],
        }),
        json!({
            "session_id": "fork",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "fork-0"}],
                    "max_tokens": 1,
                }
            ],
        }),
        json!({
            "session_id": "spawn",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "spawn-0"}],
                    "max_tokens": 1,
                }
            ],
        }),
    ]
}

/// Write the graph fixture to a `dag_jsonl` dataset file the runner can load.
fn write_graph_dataset(dir: &std::path::Path) -> std::path::PathBuf {
    write_jsonl(dir, "graph.jsonl", &graph_rows())
}

// requires: protocol-v2 request-wire capture + custom body-recording graph
// chat server (Python orchestrator internals: Installation /
// RustSubprocessExecutor). The Rust harness cannot observe the v2 request wire
// or per-dispatch request bodies.
#[tokio::test]
#[ignore]
async fn test_python_config_v2_reaches_online_graph_adapter_without_dual_conversion() {
    let h = AIPerfHarness::new().await;
    let dataset = write_graph_dataset(h.artifact_dir.path());
    let r = h.run(&format!(
        "--model mock-model --url {} --input-file {} --custom-dataset-type dag_jsonl \
         --concurrency 2 --request-count 4 --workers 2 --ui none --streaming --server-token-count",
        h.mock.url,
        dataset.display(),
    ));
    assert!(r.success(), "{}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, 4);

    // Required protocol-capture assertions:
    //   request["protocol_version"] == 2 && request["operation"] == "execute"
    //   request["run"]["cfg"]["transport"] == {"type": "http"}
    //   dataset["format"] == "dag_jsonl" && dataset["records"] == graph_rows()
    //   terminal["provenance"] == {"transport":"http","workload":"graph"}
    //   native["run"]["graph"] == {input_format: dag_jsonl, root_count: 1,
    //       node_count: 4, worker_count: 2, phase_count: 1}
    //   captured chat histories == {
    //       "root-0": ["root-0"],
    //       "fork-0": ["root-0","answer-root-0","fork-0"],
    //       "spawn-0": ["spawn-0"],
    //       "root-1": ["root-0","answer-root-0","root-1"]}
}

// requires: protocol-v2 request-wire capture + custom body-recording graph
// chat server (Python orchestrator internals). Shared phase/ramp/adaptive/
// session-policy projection is only observable on the v2 request wire.
#[tokio::test]
#[ignore]
async fn test_python_config_v2_graph_uses_shared_phase_ramp_adaptive_and_session_policy() {
    let h = AIPerfHarness::new().await;
    let dataset = write_graph_dataset(h.artifact_dir.path());
    let r = h.run(&format!(
        "--model mock-model --url {} --input-file {} --custom-dataset-type dag_jsonl \
         --concurrency 2 --request-count 4 --workers 2 --ui none --streaming --server-token-count",
        h.mock.url,
        dataset.display(),
    ));
    assert!(r.success(), "{}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, 4);

    // Required phase-projection assertions:
    //   projected_phases[0]["seamless"] == false
    //   projected_phases[1]["seamless"] == true
    //   projected_phases[0]["concurrency_ramp"] == {duration: 0.01, strategy: linear}
    //   projected_phases[1]["adaptive_scale"]["control_variable"] == "prefill_concurrency"
    //   artifact_dir/adaptive_scale_events.jsonl && adaptive_scale_summary.json exist
    //   captured bodies count == 8
}
