// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Sweep expansion parity: controller-conversion path vs direct construction.
//!
//! Three tests:
//!
//! 1. `test_sweep_expansion_is_identical_between_local_and_controller_paths` —
//!    pure logic, no binary, no cluster.  Verifies that the
//!    `kube::sweep_controller::convert_axes` path (how the sweep-controller
//!    converts contract-layer `SweepAxis` objects into plan-layer `SweepAxis`
//!    objects) produces a byte-identical `Vec<BenchmarkRun>` compared with
//!    constructing equivalent `sweep::plan::SweepAxis` values directly.
//!
//! 2. `test_local_sweep_parity` — full-stack: mock server + subprocess.
//!    Verifies per-run artifact directory structure and `request_count_avg`
//!    correctness for a two-value concurrency sweep.
//!
//! 3. `test_kube_sweep_parity_kind` — `#[ignore]`d kind cluster stub.
//!    Runs with `-- --ignored --test-threads=1` in CI after kind provisioning.
mod common;
use common::*;

use std::fs;
use std::path::Path;

use aiperf_cli::kube::contract::SweepAxis as ContractSweepAxis;
use aiperf_cli::kube::sweep_controller::convert_axes;
use aiperf_cli::model::BenchmarkConfig;
use aiperf_cli::sweep::plan::{Sweep, SweepAxis, build_benchmark_plan};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Test 1: pure expansion parity (no binary, no mock server)
// ---------------------------------------------------------------------------

/// Minimal JSON that produces a `BenchmarkConfig` with both
/// `phases.profiling.concurrency` and `endpoint.type` addressable by
/// `sweep::plan::set_dotted`.  Both paths must pre-exist in the serialized
/// form for the sweep plan to apply values to them.
fn base_config_for_parity() -> BenchmarkConfig {
    let json = json!({
        "endpoint": {
            "urls": ["http://127.0.0.1:9999"],
            "type": "chat"
        },
        "phases": [
            {
                "type": "concurrency",
                "concurrency": 2,
                "name": "profiling",
                "requests": 4
            }
        ]
    });
    serde_json::from_value(json).expect("base_config_for_parity round-trip")
}

#[test]
fn test_sweep_expansion_is_identical_between_local_and_controller_paths() {
    let base = base_config_for_parity();

    // Contract-layer axes — as the sweep-controller receives them from the
    // operator-mounted ConfigMap.
    let contract_axes = vec![
        ContractSweepAxis {
            parameter: "phases.profiling.concurrency".to_string(),
            values: vec![json!(2), json!(4)],
        },
        ContractSweepAxis {
            parameter: "endpoint.type".to_string(),
            values: vec![json!("chat"), json!("openai-chat")],
        },
    ];

    // Path 1: via convert_axes — the path the sweep-controller takes.
    let converted = convert_axes(&contract_axes);
    let sweep_via_controller = Sweep::grid(converted);
    let runs_via_controller =
        build_benchmark_plan(&base, &sweep_via_controller, None)
            .expect("expand via controller conversion path");

    // Path 2: direct SweepAxis construction — what local sweep code does.
    let direct_axes = vec![
        SweepAxis {
            path: "phases.profiling.concurrency".to_string(),
            seg: "concurrency".to_string(),
            values: vec![json!(2), json!(4)],
        },
        SweepAxis {
            path: "endpoint.type".to_string(),
            seg: "type".to_string(),
            values: vec![json!("chat"), json!("openai-chat")],
        },
    ];
    let sweep_direct = Sweep::grid(direct_axes);
    let runs_direct =
        build_benchmark_plan(&base, &sweep_direct, None)
            .expect("expand via direct construction path");

    // A 2×2 grid yields 4 combinations.
    assert_eq!(
        runs_via_controller.len(),
        4,
        "controller path: expected 4 combinations for a 2×2 grid"
    );
    assert_eq!(
        runs_direct.len(),
        4,
        "direct path: expected 4 combinations for a 2×2 grid"
    );

    // Strict equality on the serialized configs: both paths must produce
    // byte-identical expanded BenchmarkConfig values.
    let controller_cfgs: Vec<Value> = runs_via_controller
        .iter()
        .map(|r| serde_json::to_value(&r.cfg).expect("serialize controller run cfg"))
        .collect();
    let direct_cfgs: Vec<Value> = runs_direct
        .iter()
        .map(|r| serde_json::to_value(&r.cfg).expect("serialize direct run cfg"))
        .collect();

    assert_eq!(
        controller_cfgs,
        direct_cfgs,
        "controller conversion path must produce byte-identical configs \
         to direct SweepAxis construction"
    );
}

// ---------------------------------------------------------------------------
// Test 2: local sweep end-to-end (binary + mock server)
// ---------------------------------------------------------------------------

fn jload(path: &Path) -> Value {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn request_count_avg_from(path: &Path) -> f64 {
    jload(path)["request_count"]["avg"]
        .as_f64()
        .expect("request_count.avg must be a number")
}

#[tokio::test]
async fn test_local_sweep_parity() {
    let h = AIPerfHarness::new().await;
    let root = h.artifact_path().to_path_buf();

    // A two-value concurrency sweep; one trial per value (no num-profile-runs).
    // --request-count 4 so each run completes quickly against the fast mock.
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} \
         --endpoint-type chat --concurrency 2,4 \
         --request-count 4 --workers-max 1 --ui simple",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0, "sweep must exit 0; stderr:\n{}", r.stderr);

    // Both concurrency point directories must be present at the artifact root.
    for c in [2u32, 4u32] {
        let cdir = root.join(format!("concurrency_{c}"));
        assert!(
            cdir.exists(),
            "concurrency_{c} directory must exist under artifact root"
        );
        let json_file = cdir.join("profile_export_aiperf.json");
        assert!(
            json_file.exists(),
            "concurrency_{c}/profile_export_aiperf.json must exist"
        );
        assert_eq!(
            request_count_avg_from(&json_file),
            4.0,
            "concurrency_{c}: request_count_avg must equal the requested 4"
        );
        // isl (input sequence length) is produced for every run regardless of
        // streaming mode and is a reliable presence check for the metrics block.
        let data = jload(&json_file);
        assert!(
            data.get("isl").is_some(),
            "concurrency_{c}: profile export must contain isl metric"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: kind cluster stub (ignored; wired by native-cli-kind CI job)
// ---------------------------------------------------------------------------

#[ignore]
#[tokio::test]
async fn test_kube_sweep_parity_kind() {
    // Requires: a workflow-provisioned kind cluster and KUBECONFIG.
    // Wired in CI by the `native-cli-kind` job.
    todo!("kind sweep e2e not yet implemented")
}
