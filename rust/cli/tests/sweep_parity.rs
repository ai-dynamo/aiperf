// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Byte-exact parity of the native sweep expander against `dump_sweep.py`.
//!
//! For each sweep fixture the native `expand` + `plan_cells` must reproduce the
//! oracle's per-cell list: same order, labels, artifact dirs, seeds, and the
//! swept scalar landing in the right phase field.

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

/// The sweep fixtures whose native cells match the oracle.
const SWEEP_FIXTURES: &[(&str, sweep::SweepType)] = &[
    ("sweep_grid", sweep::SweepType::Grid),
    ("sweep_isl", sweep::SweepType::Grid),
    ("sweep_zip", sweep::SweepType::Zip),
];

/// Multi-run (`--num-profile-runs N`) fixtures: `(name, sweep_type, trials)`.
/// Trial iteration order defaults to REPEATED (Python `parameter_sweep_mode`).
const MULTI_FIXTURES: &[(&str, sweep::SweepType, u32)] = &[
    // No sweep, 2 trials: `profile_runs/run_000{1,2}`, both seed 42.
    ("multi_run", sweep::SweepType::Grid, 2),
    // Sweep x 2 trials (REPEATED): `profile_runs/trial_000{1,2}/concurrency_{2,4}`,
    // seeds 42/43 (base + variation.index, constant across trials).
    ("sweep_multi", sweep::SweepType::Grid, 2),
];

#[test]
fn sweep_cells_match_oracle() {
    for (name, sweep_type) in SWEEP_FIXTURES {
        let golden = load_golden(name);
        let cells_g = golden["cells"].as_array().expect("cells array");

        let flags = ProfileFlags::parse_from_args(&fixture_args(name))
            .unwrap_or_else(|e| panic!("[{name}] flags: {e}"));
        let expansion = sweep::expand(&flags, *sweep_type)
            .unwrap_or_else(|e| panic!("[{name}] expand: {e}"));
        let cells = run::plan_cells(
            &flags,
            &expansion,
            1,
            IterationOrder::Repeated,
            "parity-sweep",
            run::SeedPolicy { base: Some(run::DEFAULT_SWEEP_SEED), same_seed: false },
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

        // The `cfg` sections the single-run parity already asserts; here we check
        // the swept scalar landed and the per-cell coordinates match the oracle.
        let ported = ["phases", "datasets", "endpoint", "models", "runtime", "metrics"];
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
            for section in ported {
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
    // A config-authored `sweep:` block (grid over dotted-path parameters) must
    // reproduce the oracle's per-cell list: labels, dir names, artifact dirs,
    // seeds, and the swept scalar landing in the right config subtree.
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
    )
    .expect("plan yaml cells");

    assert_eq!(cells.len(), cells_g.len(), "cell count");
    let ported = ["phases", "datasets", "endpoint", "models", "runtime", "metrics"];
    for (i, (cell, want)) in cells.iter().zip(cells_g).enumerate() {
        assert_eq!(cell.label, want["label"], "cell {i} label");
        assert_eq!(
            cell.run.artifact_dir.to_str().unwrap(),
            want["artifact_dir"].as_str().unwrap(),
            "cell {i} artifact_dir"
        );
        assert_eq!(cell.run.random_seed, want["random_seed"].as_u64(), "cell {i} random_seed");
        let built = serde_json::to_value(&cell.run).expect("serialize");
        for section in ported {
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
    // A grid `--search-recipe` expands its log-spaced search space into a static
    // sweep; native must reproduce the oracle's per-cell list byte-exact.
    for name in ["recipe_ramp"] {
        let golden = load_golden(name);
        let cells_g = golden["cells"].as_array().expect("cells array");
        let flags = ProfileFlags::parse_from_args(&fixture_args(name))
            .unwrap_or_else(|e| panic!("[{name}] flags: {e}"));
        // Apply the recipe expansion (as `profile::run` does) before sweeping.
        let flags = match aiperf_cli::search::expand_grid_recipe(&flags).expect("recipe") {
            Some(exp) => exp.apply(&flags),
            None => flags,
        };
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
        assert_eq!(cells.len(), cells_g.len(), "[{name}] cell count");
        let ported = ["phases", "datasets", "endpoint", "models"];
        for (i, (cell, want)) in cells.iter().zip(cells_g).enumerate() {
            assert_eq!(cell.label, want["label"], "[{name}] cell {i} label");
            assert_eq!(cell.run.random_seed, want["random_seed"].as_u64(), "[{name}] cell {i} seed");
            let built = serde_json::to_value(&cell.run).expect("serialize");
            for section in ported {
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
    // The seed-policy flags (`--parameter-sweep-same-seed`, `--no-set-consistent-seed`)
    // change the per-cell random_seed: same-seed → all `base`; no-consistent → None.
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
        let expansion = sweep::expand(&flags, *sweep_type)
            .unwrap_or_else(|e| panic!("[{name}] expand: {e}"));
        let cells = run::plan_cells(
            &flags,
            &expansion,
            *trials,
            IterationOrder::Repeated,
            "parity-sweep",
            run::SeedPolicy { base: Some(run::DEFAULT_SWEEP_SEED), same_seed: false },
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

        let ported = ["phases", "datasets", "endpoint", "models", "runtime", "metrics"];
        for (i, (cell, want)) in cells.iter().zip(cells_g).enumerate() {
            assert_eq!(cell.trial, want["trial"].as_u64().unwrap() as u32, "[{name}] cell {i} trial");
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
            for section in ported {
                assert_eq!(
                    built["cfg"][section], want["request"]["run"]["cfg"][section],
                    "[{name}] cell {i} cfg.{section} diverges\n got {:#}\nwant {:#}",
                    built["cfg"][section], want["request"]["run"]["cfg"][section]
                );
            }
        }
    }
}
