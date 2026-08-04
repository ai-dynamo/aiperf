// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for single-reactor dry-run virtual workers.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::{Artifacts, run_config};
use serde_json::Value;

fn config(runtime: &str, virtual_workers: &str, profiling: &str) -> String {
    let virtual_workers = virtual_workers
        .lines()
        .map(|line| format!("  {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        r#"schemaVersion: "2.0"
randomSeed: 7
runtime:
{runtime}
benchmark:
  model: openai/gpt-oss-120b
  tokenizer: {{name: cl100k_base}}
  transport:
    type: dry_run
    clock: sim
    ttft_ms: 10
    itl_ms: 2
    virtual_workers:
{virtual_workers}
  endpoint:
    type: chat
    url: http://127.0.0.1:8000
    streaming: true
  dataset:
    type: synthetic
    prompts: {{isl: 20, osl: 4}}
  profiling:
{profiling}
  artifacts:
    dir: $ARTIFACT_DIR
"#
    )
}

fn records_by_session(records: &[Value]) -> BTreeMap<String, BTreeSet<String>> {
    let mut assignments = BTreeMap::<String, BTreeSet<String>>::new();
    for record in records {
        assignments
            .entry(
                record["metadata"]["x_correlation_id"]
                    .as_str()
                    .expect("correlation id")
                    .to_owned(),
            )
            .or_default()
            .insert(
                record["metadata"]["worker_id"]
                    .as_str()
                    .expect("worker id")
                    .to_owned(),
            );
    }
    assignments
}

#[test]
fn global_hop_round_robin_uses_authored_width_and_global_sequence() {
    let yaml = config(
        "  workers: 8\n  dispatch: global-hop\n  hop_routing: round-robin",
        "    enabled: true\n    width: 8",
        "    type: concurrency\n    requests: 8\n    concurrency: 4",
    );
    let run = run_config(&yaml);
    run.assert_success();
    let records = run.artifacts.jsonl();
    let assignments: BTreeMap<u64, String> = records
        .iter()
        .map(|record| {
            (
                record["metadata"]["worker_assignment_index"]
                    .as_u64()
                    .expect("assignment index"),
                record["metadata"]["worker_id"]
                    .as_str()
                    .expect("worker id")
                    .to_owned(),
            )
        })
        .collect();
    assert_eq!(
        assignments.into_values().collect::<Vec<_>>(),
        (0..8)
            .map(|worker| format!("dry-run-{worker}"))
            .collect::<Vec<_>>()
    );
}

#[test]
fn sticky_and_least_loaded_keep_multiturn_sessions_affine() {
    for routing in ["sticky", "least-loaded"] {
        let yaml = config(
            &format!("  workers: 4\n  dispatch: global-hop\n  hop_routing: {routing}"),
            "    enabled: true",
            "    type: concurrency\n    sessions: 8\n    concurrency: 4",
        )
        .replace(
            "    prompts: {isl: 20, osl: 4}",
            "    prompts: {isl: 20, osl: 4}\n    turns: {mean: 3, stddev: 0}",
        );
        let run = run_config(&yaml);
        run.assert_success();
        assert!(
            records_by_session(&run.artifacts.jsonl())
                .values()
                .all(|workers| workers.len() == 1),
            "{routing} moved a session between workers"
        );
    }
}

#[test]
fn worker_profiles_scale_ttft_and_itl_after_placement() {
    let yaml = config(
        "  workers: 2\n  dispatch: global-hop\n  hop_routing: round-robin",
        "    enabled: true\n    profiles:\n      - worker: 1\n        ttft_multiplier: 2.0\n        itl_multiplier: 1.5",
        "    type: concurrency\n    requests: 2\n    concurrency: 2",
    );
    let run = run_config(&yaml);
    run.assert_success();
    for record in run.artifacts.jsonl() {
        let worker = record["metadata"]["worker_id"].as_str().expect("worker id");
        let expected = if worker == "dry-run-1" {
            (20.0, 3.0)
        } else {
            (10.0, 2.0)
        };
        assert_eq!(
            Artifacts::metric(&record, "time_to_first_token"),
            expected.0
        );
        assert_eq!(
            Artifacts::metric(&record, "inter_token_latency"),
            expected.1
        );
    }
}

/// With virtual workers off, records carry no virtual placement at all: no
/// `worker_assignment_index`, and a `worker_id` naming the REAL shard that ran the
/// request rather than a modeled `dry-run-{n}` slot.
///
/// This used to assert `worker_id == "rust-0"` for every record of a 4-worker run.
/// That constant was the export's fallback for a record no executing worker had
/// stamped, not an observation: `Sharded`/`Global` never reached the stamping path,
/// so the artifact claimed one worker had executed all four shards' requests.
#[test]
fn disabled_mode_reports_the_real_shard_and_no_virtual_placement() {
    const WORKERS: u64 = 4;
    let yaml = config(
        "  workers: 4\n  dispatch: global",
        "    enabled: false",
        "    type: concurrency\n    requests: 3\n    concurrency: 2",
    );
    let run = run_config(&yaml);
    run.assert_success();
    for record in run.artifacts.jsonl() {
        let worker = record["metadata"]["worker_id"]
            .as_str()
            .expect("worker id")
            .to_owned();
        let index: u64 = worker
            .strip_prefix("rust-")
            .and_then(|suffix| suffix.parse().ok())
            .unwrap_or_else(|| {
                panic!("a non-virtual run names a real shard as rust-{{n}}, got {worker}")
            });
        assert!(
            index < WORKERS,
            "worker_id {worker} is outside the run's 0..{WORKERS} worker grid"
        );
        assert!(record["metadata"].get("worker_assignment_index").is_none());
    }
}

#[test]
fn cancellation_records_are_terminal_and_later_assignments_continue() {
    let yaml = config(
        "  workers: 3\n  dispatch: global-hop\n  hop_routing: round-robin",
        "    enabled: true",
        "    type: concurrency\n    requests: 6\n    concurrency: 2\n    cancellation: {rate: 100, delay: 0}",
    );
    let run = run_config(&yaml);
    run.assert_success();
    let records = run.artifacts.jsonl();
    assert_eq!(records.len(), 6);
    for (index, record) in records.iter().enumerate() {
        assert_eq!(record["metadata"]["was_cancelled"], true);
        assert_eq!(record["error"]["code"], 499);
        assert_eq!(record["metadata"]["worker_assignment_index"], index as u64);
    }
}

#[test]
fn virtual_workers_reject_sharded_and_real_clock() {
    let sharded = config(
        "  workers: 2\n  dispatch: sharded",
        "    enabled: true",
        "    type: concurrency\n    requests: 2\n    concurrency: 1",
    );
    let run = run_config(&sharded);
    assert!(!run.success());
    assert!(
        String::from_utf8_lossy(&run.output.stderr).contains("do not yet support runtime.dispatch")
    );

    let real = config(
        "  workers: 2\n  dispatch: global",
        "    enabled: true",
        "    type: concurrency\n    requests: 2\n    concurrency: 1",
    )
    .replace("clock: sim", "clock: real");
    let run = run_config(&real);
    assert!(!run.success());
    assert!(String::from_utf8_lossy(&run.output.stderr).contains("require clock: sim"));
}
