// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Outer-loop behavior layered above a single profile run: multi-axis sweep
//! combination (`--sweep-type`) and the adaptive SLA search loop.
//!
//! The sibling `test_parameter_sweep.rs` covers the single-axis artifact tree and
//! aggregate shape. What is only reachable here is *how multiple axes combine*,
//! which no single-axis sweep can distinguish, and the ask/tell search loop that
//! chooses its own load between iterations instead of enumerating it up front.

mod common;
use common::*;

use std::collections::BTreeSet;
use std::path::Path;

const WORKERS_MAX: u32 = 1;
const UI: &str = "simple";

/// Grid (the default) takes the cartesian product: 2 concurrencies x 2 ISLs = 4 runs.
///
/// Every live sweep test varies a single axis, where grid and zip are
/// indistinguishable. Two axes is the smallest case that pins the product.
#[tokio::test]
async fn test_grid_sweep_runs_the_cartesian_product_of_axes() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&multi_axis_args(&h, ""));
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);

    assert_eq!(
        variation_dirs(h.artifact_path()),
        set([
            "mean_8__concurrency_2",
            "mean_8__concurrency_4",
            "mean_16__concurrency_2",
            "mean_16__concurrency_4",
        ]),
        "grid must run every (isl, concurrency) pair"
    );

    // The aggregate must agree with the directory tree, not merely exist.
    let agg = sweep_aggregate(h.artifact_path());
    assert_eq!(agg["metadata"]["num_combinations"].as_u64(), Some(4));
    assert_eq!(agg["num_successful_runs"].as_u64(), Some(4));
    assert_eq!(
        combination_parameters(&agg),
        set([
            "concurrency=2,mean=8",
            "concurrency=4,mean=8",
            "concurrency=2,mean=16",
            "concurrency=4,mean=16",
        ]),
    );

    // Each cell must actually have applied its own ISL, not just be named for it.
    for (dir, isl) in [
        ("mean_8__concurrency_2", 8.0),
        ("mean_16__concurrency_4", 16.0),
    ] {
        let export = read_json(
            &h.artifact_path()
                .join(dir)
                .join("profile_export_aiperf.json"),
        );
        assert_eq!(
            export["input_sequence_length"]["avg"].as_f64(),
            Some(isl),
            "{dir} should have run at ISL {isl}"
        );
    }
}

/// `--sweep-type zip` pairs axes positionally: 2 values per axis = 2 runs, not 4.
#[tokio::test]
async fn test_zip_sweep_preserves_authored_pairing() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&multi_axis_args(&h, "--sweep-type zip"));
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);

    // First-with-first and second-with-second. The two grid-only cells
    // (8 with 4, 16 with 2) must be absent.
    assert_eq!(
        variation_dirs(h.artifact_path()),
        set(["mean_8__concurrency_2", "mean_16__concurrency_4"]),
        "zip must pair axes positionally rather than cross them"
    );
    let agg = sweep_aggregate(h.artifact_path());
    assert_eq!(
        combination_parameters(&agg),
        set(["concurrency=2,mean=8", "concurrency=4,mean=16"]),
    );
    // Only the two paired runs were executed and aggregated. Note that
    // `metadata.num_combinations` is NOT asserted here: it is computed as the
    // product of per-axis distinct values (cli/src/sweep/aggregate.rs:346), which
    // assumes a cartesian plan and so reports 4 for this 2-run zip. That is a
    // product-side inaccuracy, not something this test should encode as correct.
    assert_eq!(agg["num_profile_runs"].as_u64(), Some(2));
    assert_eq!(agg["num_successful_runs"].as_u64(), Some(2));
}

/// Zip cannot pair axes of unequal length, and says so instead of truncating.
#[tokio::test]
async fn test_zip_sweep_rejects_axes_of_unequal_length() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4,6 --synthetic-input-tokens-mean 8,16 --output-tokens-mean 2 \
         --request-count 4 --workers-max {WORKERS_MAX} --ui {UI} --sweep-type zip",
        h.mock.url
    ));
    assert_ne!(r.exit_code, 0, "unequal zip axes must not run");
    let combined = format!("{}{}", r.stderr, r.stdout);
    assert!(
        combined.contains("zip sweep requires all axes to have the same number of values"),
        "expected the zip length diagnostic, got: {combined}"
    );
    assert!(
        variation_dirs(h.artifact_path()).is_empty(),
        "a rejected plan must not run any variation"
    );
}

/// The adaptive search loop picks each concurrency from the previous verdict.
///
/// This is the ask/tell path (`--search-style monotonic`), distinct from a sweep:
/// the probe sequence is not enumerable from the flags, so the assertion is that
/// `search_history.json` records a real trajectory consistent with the boundary
/// summary — not that it hit specific values.
#[tokio::test]
async fn test_adaptive_search_records_its_probe_trajectory() {
    let h = AIPerfHarness::new().await;
    // A generous TTFT SLA against a fast mock keeps every probe feasible, so the
    // planner climbs and the run stays deterministic.
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --search-recipe max-concurrency-under-sla --search-style monotonic \
             --concurrency-min 1 --concurrency-max 16 --ttft-sla-ms 1000 \
             --search-max-iterations 4 --synthetic-input-tokens-mean 8 \
             --output-tokens-mean 2 --request-count 4 \
             --workers-max {WORKERS_MAX} --ui {UI}",
            h.mock.url
        ),
        600,
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);

    // One artifact directory per probe, and nothing else: the loop must not fall
    // back to enumerating a static grid.
    let iters = dirs_with_prefix(h.artifact_path(), "search_iter_");
    assert_eq!(
        iters.len(),
        4,
        "expected one directory per probe, got {iters:?}"
    );
    assert_eq!(iters.first().map(String::as_str), Some("search_iter_0000"));

    let history = read_json(&h.artifact_path().join("search_history.json"));
    assert_eq!(
        history["recipe"].as_str(),
        Some("max-concurrency-under-sla")
    );
    assert_eq!(history["config"]["planner"].as_str(), Some("monotonic_sla"));
    let space = &history["config"]["search_space"][0];
    assert_eq!(space["path"].as_str(), Some("phases.profiling.concurrency"));
    assert_eq!(
        (space["lo"].as_i64(), space["hi"].as_i64()),
        (Some(1), Some(16))
    );
    // The SLA that defined feasibility must be echoed, or the verdicts are unattributable.
    assert_eq!(
        history["config"]["sla_filters"][0]["metric_tag"].as_str(),
        Some("time_to_first_token")
    );

    let iterations = history["iterations"].as_array().expect("iterations array");
    assert_eq!(
        iterations.len(),
        iters.len(),
        "every probe must be recorded"
    );
    let mut probes = Vec::new();
    for (idx, it) in iterations.iter().enumerate() {
        assert_eq!(it["iteration_idx"].as_u64(), Some(idx as u64));
        let value = it["variation_values"]["phases.profiling.concurrency"]
            .as_i64()
            .unwrap_or_else(|| panic!("iteration {idx} has no probed concurrency: {it}"));
        assert!(
            (1..=16).contains(&value),
            "probe {idx} left the search space: {value}"
        );
        // A probe that ran must have produced an objective; a bare `null` here
        // would mean the report was never read back.
        assert!(
            it["objective_values"][0]
                .as_f64()
                .is_some_and(f64::is_finite),
            "iteration {idx} recorded no finite objective: {it}"
        );
        assert_eq!(
            it["feasible"].as_bool(),
            Some(true),
            "a 1s TTFT SLA against the mock should stay feasible: {it}"
        );
        probes.push(value);
    }
    assert!(
        probes.iter().any(|&v| v > probes[0]),
        "an all-feasible trajectory must climb, got {probes:?}"
    );

    // The boundary summary is derived from the trajectory; it must not disagree with it.
    let feasible_max = history["boundary_summary"]["feasible_max"]["value"].as_i64();
    assert_eq!(
        feasible_max,
        probes.iter().copied().max(),
        "feasible_max must be the highest feasible probe"
    );
    let boundary = read_json(&h.artifact_path().join("search_boundary.json"));
    assert_eq!(boundary["feasible_max"].as_i64(), feasible_max);
    assert_eq!(
        boundary["convergence_reason"].as_str(),
        history["convergence_reason"].as_str(),
        "the two artifacts must agree on why the search stopped"
    );
}

/// Two axes, two values each: `extra` selects the combination strategy.
fn multi_axis_args(h: &AIPerfHarness, extra: &str) -> String {
    format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 2,4 --synthetic-input-tokens-mean 8,16 --output-tokens-mean 2 \
         --request-count 4 --workers-max {WORKERS_MAX} --ui {UI} {extra}",
        h.mock.url
    )
}

/// Per-variation artifact directory names (the `mean_N__concurrency_N` cells),
/// excluding the fixed `logs`/`sweep_aggregate` siblings.
fn variation_dirs(root: &Path) -> BTreeSet<String> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return BTreeSet::new();
    };
    entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name != "logs" && name != "sweep_aggregate")
        .collect()
}

/// Sorted directory names under `root` starting with `prefix`.
fn dirs_with_prefix(root: &Path, prefix: &str) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with(prefix))
        .collect();
    out.sort();
    out
}

fn sweep_aggregate(root: &Path) -> serde_json::Value {
    read_json(
        &root
            .join("sweep_aggregate")
            .join("profile_export_aiperf_sweep.json"),
    )
}

/// `"mean=8,concurrency=2"` per aggregated combination, order-independent.
fn combination_parameters(aggregate: &serde_json::Value) -> BTreeSet<String> {
    aggregate["per_combination_metrics"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .map(|entry| {
            let params = entry["parameters"]
                .as_object()
                .expect("combination parameters");
            let mut parts: Vec<String> = params.iter().map(|(k, v)| format!("{k}={v}")).collect();
            // Serialized order is the sweep's own; sort so the assertion does not
            // depend on it (the ordering contract is covered by the dir names).
            parts.sort();
            parts.join(",")
        })
        .collect()
}

fn read_json(path: &Path) -> serde_json::Value {
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()))
}

fn set<'a>(items: impl IntoIterator<Item = &'a str>) -> BTreeSet<String> {
    items.into_iter().map(str::to_string).collect()
}
