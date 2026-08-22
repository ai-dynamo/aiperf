// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Byte-exact single-trial sweep aggregate golden coverage.

use std::path::PathBuf;

use aiperf_cli::flags::ProfileFlags;
use aiperf_cli::sweep::aggregate::{CellOutcome, finish};
use aiperf_cli::sweep::artifact_dir::IterationOrder;

mod common;

fn repo_file(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .join(rel)
}

#[test]
fn single_trial_sweep_aggregate_matches_python() {
    common::on_profile_flags_stack(single_trial_sweep_aggregate_matches_python_on_larger_stack);
}

fn single_trial_sweep_aggregate_matches_python_on_larger_stack() {
    let spec: serde_json::Value = serde_json::from_slice(
        &std::fs::read(repo_file("tools/parity/sweep_agg_spec.json")).unwrap(),
    )
    .unwrap();
    let golden: serde_json::Value = serde_json::from_slice(
        &std::fs::read(repo_file("tools/parity/sweep_golden/sweep_agg.json")).unwrap(),
    )
    .unwrap();

    let base = std::env::temp_dir().join(format!("aiperf-sweep-agg-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();

    let mut outcomes = Vec::new();
    for (i, cell) in spec["cells"].as_array().unwrap().iter().enumerate() {
        let variation = &cell["variation"];
        let dir = base.join(format!("cell_{i}"));
        std::fs::create_dir_all(&dir).unwrap();
        let report_path = dir.join("native-v2.json");
        std::fs::write(&report_path, serde_json::to_vec(&cell["report"]).unwrap()).unwrap();
        outcomes.push(CellOutcome {
            label: variation["label"].as_str().unwrap().to_string(),
            values: Some(serde_json::json!({
                "index": variation["index"],
                "label": variation["label"],
                "values": variation["values"],
            })),
            artifact_dir: dir,
            report_path: Some(report_path.to_str().unwrap().to_string()),
            success: true,
            trial: 0,
            error: None,
        });
    }

    let flags = ProfileFlags::parse_from_args(&[
        "--model".to_string(),
        "m".to_string(),
        "--url".to_string(),
        "127.0.0.1:8000".to_string(),
        "--no-sweep-table".to_string(),
        "--artifact-dir".to_string(),
        base.to_str().unwrap().to_string(),
    ])
    .unwrap();

    finish(&flags, &outcomes, true, IterationOrder::Repeated).unwrap();

    let got_json =
        std::fs::read_to_string(base.join("sweep_aggregate/profile_export_aiperf_sweep.json"))
            .unwrap();
    let got_csv =
        std::fs::read_to_string(base.join("sweep_aggregate/profile_export_aiperf_sweep.csv"))
            .unwrap();

    assert_eq!(
        got_json,
        golden["json"].as_str().unwrap(),
        "sweep aggregate JSON diverges"
    );
    assert_eq!(
        got_csv,
        golden["csv"].as_str().unwrap(),
        "sweep aggregate CSV diverges"
    );

    let _ = std::fs::remove_dir_all(&base);
}
