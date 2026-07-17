// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Byte-exact sweep expansion and cell-planning golden coverage.

use aiperf_cli::flags::ProfileFlags;
use aiperf_cli::load;
use aiperf_cli::sweep::{self, artifact_dir::IterationOrder, run};

fn load_golden(name: &str) -> serde_json::Value {
    let path = format!("../../tools/parity/sweep_golden/{name}.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_slice(&bytes).expect("golden json")
}

fn fixture_args(name: &str) -> Vec<String> {
    let path = format!("../../tools/parity/fixtures/{name}.args");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {path}: {e}"))
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

const SWEEP_FIXTURES: &[(&str, sweep::SweepType)] = &[
    ("sweep_grid", sweep::SweepType::Grid),
    ("sweep_isl", sweep::SweepType::Grid),
    ("sweep_zip", sweep::SweepType::Zip),
];

/// Multi-run fixtures as `(name, sweep_type, trials)`.
const MULTI_FIXTURES: &[(&str, sweep::SweepType, u32)] = &[
    ("multi_run", sweep::SweepType::Grid, 2),
    ("sweep_multi", sweep::SweepType::Grid, 2),
];

#[test]
fn sweep_cells_match_oracle() {
    for (name, sweep_type) in SWEEP_FIXTURES {
        let golden = load_golden(name);
        let cells_g = golden["cells"].as_array().expect("cells array");

        let flags = ProfileFlags::parse_from_args(&fixture_args(name))
            .unwrap_or_else(|e| panic!("[{name}] flags: {e}"));
        let expansion =
            sweep::expand(&flags, *sweep_type).unwrap_or_else(|e| panic!("[{name}] expand: {e}"));
        let cells = run::plan_cells(
            &flags,
            &expansion,
            1,
            IterationOrder::Repeated,
            "parity-sweep",
            run::SeedPolicy {
                base: Some(run::DEFAULT_SWEEP_SEED),
                same_seed: false,
            },
            true,
            load::resolve,
        )
        .unwrap_or_else(|e| panic!("[{name}] plan_cells: {e}"));

        assert_eq!(
            cells.len(),
            cells_g.len(),
            "[{name}] cell count: got {} want {}",
            cells.len(),
            cells_g.len()
        );

        let modeled = [
            "phases", "datasets", "endpoint", "models", "runtime", "metrics",
        ];
        for (i, (cell, want)) in cells.iter().zip(cells_g).enumerate() {
            assert_eq!(cell.label, want["label"], "[{name}] cell {i} label");
            assert_eq!(
                cell.run.artifact_dir.to_str().unwrap(),
                want["artifact_dir"].as_str().unwrap(),
                "[{name}] cell {i} artifact_dir"
            );
            assert_eq!(
                cell.run.random_seed,
                want["random_seed"].as_u64(),
                "[{name}] cell {i} random_seed"
            );
            let built = serde_json::to_value(&cell.run).expect("serialize");
            for section in modeled {
                assert_eq!(
                    built["cfg"][section], want["request"]["run"]["cfg"][section],
                    "[{name}] cell {i} cfg.{section} diverges\n got {:#}\nwant {:#}",
                    built["cfg"][section], want["request"]["run"]["cfg"][section]
                );
            }
        }
    }
}

#[test]
fn yaml_sweep_cells_match_oracle() {
    let golden = load_golden("sweep_yaml");
    let cells_g = golden["cells"].as_array().expect("cells array");

    let path = std::path::Path::new("../../tools/parity/configs/sweep_dist.yaml");
    let mut base = aiperf_cli::yaml::read_env_substituted(path).expect("read config");
    let sweep = aiperf_cli::sweep::yaml_sweep::parse(&base)
        .expect("parse sweep")
        .expect("sweep block present");
    aiperf_cli::sweep::yaml_sweep::normalize_benchmark(&mut base);
    let cells = aiperf_cli::profile::plan_yaml_cells(
        Some(std::path::PathBuf::from("/tmp/aiperf-parity/sweep_dist")),
        &base,
        &sweep,
        "parity-sweep",
        None,
    )
    .expect("plan yaml cells");

    assert_eq!(cells.len(), cells_g.len(), "cell count");
    let modeled = [
        "phases", "datasets", "endpoint", "models", "runtime", "metrics",
    ];
    for (i, (cell, want)) in cells.iter().zip(cells_g).enumerate() {
        assert_eq!(cell.label, want["label"], "cell {i} label");
        assert_eq!(
            cell.run.artifact_dir.to_str().unwrap(),
            want["artifact_dir"].as_str().unwrap(),
            "cell {i} artifact_dir"
        );
        assert_eq!(
            cell.run.random_seed,
            want["random_seed"].as_u64(),
            "cell {i} random_seed"
        );
        let built = serde_json::to_value(&cell.run).expect("serialize");
        for section in modeled {
            assert_eq!(
                built["cfg"][section], want["request"]["run"]["cfg"][section],
                "cell {i} cfg.{section} diverges\n got {:#}\nwant {:#}",
                built["cfg"][section], want["request"]["run"]["cfg"][section]
            );
        }
    }
}

#[test]
fn search_recipe_cells_match_oracle() {
    for name in [
        "recipe_ramp",
        "recipe_prefill",
        "recipe_decode",
        "recipe_pareto",
        "recipe_maxconc",
    ] {
        let golden = load_golden(name);
        let cells_g = golden["cells"].as_array().expect("cells array");
        let flags = ProfileFlags::parse_from_args(&fixture_args(name))
            .unwrap_or_else(|e| panic!("[{name}] flags: {e}"));
        let recipe = aiperf_cli::search::expand_recipe(&flags)
            .expect("recipe")
            .expect("recipe present");
        let cells = aiperf_cli::profile::plan_recipe_cells(&flags, &recipe, "parity-sweep")
            .unwrap_or_else(|e| panic!("[{name}] plan_recipe_cells: {e}"));
        assert_eq!(cells.len(), cells_g.len(), "[{name}] cell count");
        let modeled = ["phases", "datasets", "endpoint", "models"];
        for (i, (cell, want)) in cells.iter().zip(cells_g).enumerate() {
            assert_eq!(cell.label, want["label"], "[{name}] cell {i} label");
            assert_eq!(
                cell.run.random_seed,
                want["random_seed"].as_u64(),
                "[{name}] cell {i} seed"
            );
            let built = serde_json::to_value(&cell.run).expect("serialize");
            for section in modeled {
                assert_eq!(
                    built["cfg"][section], want["request"]["run"]["cfg"][section],
                    "[{name}] cell {i} cfg.{section} diverges\n got {:#}\nwant {:#}",
                    built["cfg"][section], want["request"]["run"]["cfg"][section]
                );
            }
        }
    }
}

#[test]
fn seed_knob_cells_match_oracle() {
    for name in ["sweep_sameseed", "sweep_noseed"] {
        let golden = load_golden(name);
        let cells_g = golden["cells"].as_array().expect("cells array");
        let flags = ProfileFlags::parse_from_args(&fixture_args(name))
            .unwrap_or_else(|e| panic!("[{name}] flags: {e}"));
        let expansion = sweep::expand(&flags, sweep::SweepType::Grid)
            .unwrap_or_else(|e| panic!("[{name}] expand: {e}"));
        let cells = run::plan_cells(
            &flags,
            &expansion,
            1,
            IterationOrder::Repeated,
            "parity-sweep",
            aiperf_cli::profile::seed_policy(&flags),
            true,
            load::resolve,
        )
        .unwrap_or_else(|e| panic!("[{name}] plan_cells: {e}"));
        for (i, (cell, want)) in cells.iter().zip(cells_g).enumerate() {
            assert_eq!(
                cell.run.random_seed,
                want["random_seed"].as_u64(),
                "[{name}] cell {i} random_seed"
            );
        }
    }
}

#[test]
fn multi_run_cells_match_oracle() {
    for (name, sweep_type, trials) in MULTI_FIXTURES {
        let golden = load_golden(name);
        let cells_g = golden["cells"].as_array().expect("cells array");
        assert_eq!(
            golden["trials"].as_u64(),
            Some(*trials as u64),
            "[{name}] golden trials",
        );

        let flags = ProfileFlags::parse_from_args(&fixture_args(name))
            .unwrap_or_else(|e| panic!("[{name}] flags: {e}"));
        let expansion =
            sweep::expand(&flags, *sweep_type).unwrap_or_else(|e| panic!("[{name}] expand: {e}"));
        let cells = run::plan_cells(
            &flags,
            &expansion,
            *trials,
            IterationOrder::Repeated,
            "parity-sweep",
            run::SeedPolicy {
                base: Some(run::DEFAULT_SWEEP_SEED),
                same_seed: false,
            },
            true,
            load::resolve,
        )
        .unwrap_or_else(|e| panic!("[{name}] plan_cells: {e}"));

        assert_eq!(
            cells.len(),
            cells_g.len(),
            "[{name}] cell count: got {} want {}",
            cells.len(),
            cells_g.len()
        );

        let modeled = [
            "phases", "datasets", "endpoint", "models", "runtime", "metrics",
        ];
        for (i, (cell, want)) in cells.iter().zip(cells_g).enumerate() {
            assert_eq!(
                cell.trial,
                want["trial"].as_u64().unwrap() as u32,
                "[{name}] cell {i} trial"
            );
            assert_eq!(
                cell.run.artifact_dir.to_str().unwrap(),
                want["artifact_dir"].as_str().unwrap(),
                "[{name}] cell {i} artifact_dir"
            );
            assert_eq!(
                cell.run.random_seed,
                want["random_seed"].as_u64(),
                "[{name}] cell {i} random_seed"
            );
            let built = serde_json::to_value(&cell.run).expect("serialize");
            // Sweep envelope (sweep_id/variation/trial) is stamped on every cell,
            // including the non-sweep multi-run base variation.
            assert_eq!(
                built["variation"], want["request"]["run"]["variation"],
                "[{name}] cell {i} variation diverges\n got {:#}\nwant {:#}",
                built["variation"], want["request"]["run"]["variation"]
            );
            for section in modeled {
                assert_eq!(
                    built["cfg"][section], want["request"]["run"]["cfg"][section],
                    "[{name}] cell {i} cfg.{section} diverges\n got {:#}\nwant {:#}",
                    built["cfg"][section], want["request"]["run"]["cfg"][section]
                );
            }
        }
    }
}
