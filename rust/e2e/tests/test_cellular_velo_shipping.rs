// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A cellular run ships its per-record artifacts between real processes over the
//! **velo streaming plane** (`AIPERF_ARTIFACT_TRANSPORT=velo`) — the shared cellular
//! velo endpoint, NOT a second HTTP port — and the controller reassembles them
//! byte-for-byte, matching both a single-cell baseline and the HTTP shipping path.
//!
//! This mirrors `test_cellular_http_shipping.rs` (the Stage-F HTTP+zstd proof) but
//! flips the transport: `AIPERF_CELL_ARTIFACT_HTTP_FORCE` drives the cross-host path
//! on same-host loopback, and `AIPERF_ARTIFACT_TRANSPORT=velo` routes the per-record
//! artifact bytes over velo's native ordered/backpressured stream primitive (zstd
//! chunks) rather than the raw-hyper HTTP server. The controller's velo receiver logs
//! `received artifact stream over velo … transport="velo"` per cell × file.
//!
//! Parity keys on `conversation_id` + dataset-deterministic metrics and EXCLUDES the
//! per-cell-local `session_num` counter, an accepted cellular characteristic.

mod common;
use common::*;
use serde_json::{Value, json};

/// Full coverage: every synthetic conversation dispatches exactly once.
const ENTRIES: u32 = 18;
/// Fixed seed so baseline and cellular synthesize the identical dataset.
const SEED: u32 = 20260716;
/// Cells the forced multi-process run partitions across (uneven round-robin).
const CELLS: u32 = 3;
/// Concurrency cap (>= CELLS so it splits per cell without flooring to 1).
const CONCURRENCY: u32 = 6;

/// A single-turn synthetic config with the always-on per-record artifacts
/// (records jsonl + raw + outputs) enabled, seeded, partitioned across `cells`.
fn config(url: &str, cells: u32) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         \n\
         benchmark:\n\
        \x20 model: {DEFAULT_MODEL}\n\
        \x20 endpoint:\n\
        \x20   url: {url}/v1/chat/completions\n\
        \x20   type: chat\n\
        \x20   streaming: true\n\
        \x20 dataset:\n\
        \x20   type: synthetic\n\
        \x20   entries: {ENTRIES}\n\
        \x20   random_seed: {SEED}\n\
        \x20   prompts:\n\
        \x20     isl: 32\n\
        \x20     osl: 16\n\
        \x20 phases:\n\
        \x20   type: concurrency\n\
        \x20   requests: {ENTRIES}\n\
        \x20   concurrency: {CONCURRENCY}\n\
        \x20 artifacts:\n\
        \x20   records:\n\
        \x20     - jsonl\n\
        \x20   raw: true\n\
        \x20   export_outputs_json: true\n\
         \n\
         runtime:\n\
        \x20 cells: {cells}\n"
    )
}

/// Deterministic, run-independent projection of one `profile_export.jsonl` record.
/// Excludes wall-clock timing, per-request UUIDs, and the per-cell-local `session_num`.
fn record_projection(r: &Value) -> String {
    let m = &r["metadata"];
    let met = &r["metrics"];
    json!({
        "conversation_id": m["conversation_id"],
        "turn_index": m["turn_index"],
        "benchmark_phase": m["benchmark_phase"],
        "input_sequence_length": met["input_sequence_length"],
        "output_sequence_length": met["output_sequence_length"],
        "error": r["error"],
    })
    .to_string()
}

/// Deterministic projection of one `outputs.json` row.
fn output_projection(row: &Value) -> String {
    json!({
        "conversation_id": row["conversation_id"],
        "turn_index": row["turn_index"],
        "response_text": row["response_text"],
        "reasoning_text": row["reasoning_text"],
    })
    .to_string()
}

fn sorted<T, F: Fn(&T) -> String>(items: &[T], f: F) -> Vec<String> {
    let mut v: Vec<String> = items.iter().map(f).collect();
    v.sort();
    v
}

fn outputs(r: &RunResult) -> Vec<Value> {
    let p = r
        .artifacts
        .find_file("**/outputs.json")
        .expect("outputs.json");
    let v: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    v["data"].as_array().cloned().unwrap_or_default()
}

fn aiperf_log(r: &RunResult) -> String {
    let path = r
        .artifacts
        .find_file("**/aiperf.log")
        .expect("logs/aiperf.log should exist");
    std::fs::read_to_string(&path).unwrap_or_default()
}

/// The velo artifact-stream observable lines: one per received cell × file.
fn velo_observables(r: &RunResult) -> Vec<String> {
    aiperf_log(r)
        .lines()
        .filter(|l| l.contains("received artifact stream over velo"))
        .map(str::to_string)
        .collect()
}

/// Run the config against `h`'s mock at `cells` cells. When `velo`, additionally set
/// the force seam + the velo transport toggle + the `info`-level artifact observable
/// filter, so a multi-cell run ships its artifacts over the velo plane. When `hub`,
/// also set `AIPERF_CELLULAR_HUB=1` so the controller stands up ONE velo hub (the
/// cell↔controller + `/artifact` + discovery plugins on one anchor) instead of the
/// standalone planes — cells reach it by the identical `tcp://` coordinate.
fn run_modes(h: &AIPerfHarness, cells: u32, velo: bool, hub: bool) -> RunResult {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg = tmp.path().join("velo_coverage.yaml");
    std::fs::write(&cfg, config(&h.mock.url, cells)).unwrap();
    let mut env: Vec<(&str, &str)> = vec![("AIPERF_LOG", "warn,aiperf_cellular_artifact=info")];
    if velo {
        env.push(("AIPERF_CELL_ARTIFACT_HTTP_FORCE", "1"));
        env.push(("AIPERF_ARTIFACT_TRANSPORT", "velo"));
    }
    if hub {
        env.push(("AIPERF_CELLULAR_HUB", "1"));
    }
    h.run_env(&format!("--config {} --ui simple", cfg.display()), &env)
}

/// Run the config against `h`'s mock at `cells` cells (standalone-plane path).
fn run(h: &AIPerfHarness, cells: u32, velo: bool) -> RunResult {
    run_modes(h, cells, velo, false)
}

/// A same-host multi-process `--cells N` run with `AIPERF_ARTIFACT_TRANSPORT=velo`
/// ships every per-record artifact over the velo stream plane between real cell
/// subprocesses and the controller, and the merged result matches a single-cell
/// exact-fold baseline.
#[tokio::test]
async fn test_cellular_velo_shipping_matches_single_cell() {
    // Flaky on macOS CI like the other artifact e2es; skip there.
    if cfg!(target_os = "macos") {
        return;
    }

    let h_base = AIPerfHarness::new().await;
    let baseline = run(&h_base, 1, false);
    assert!(
        baseline.success(),
        "1-cell baseline run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        baseline.exit_code,
        baseline.stdout,
        baseline.stderr
    );

    let h_cell = AIPerfHarness::new().await;
    let cellular = run(&h_cell, CELLS, true);
    assert!(
        cellular.success(),
        "velo {CELLS}-cell run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        cellular.exit_code,
        cellular.stdout,
        cellular.stderr
    );

    // The multi-cell run must have gone through the controller.
    assert!(
        cellular
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "{CELLS}-cell run must go through the controller (cellular-heartbeat.json sidecar)"
    );

    // The bytes really crossed the velo plane (not HTTP, not shared-FS): one observable
    // per cell × file, each tagged transport="velo".
    let observables = velo_observables(&cellular);
    assert!(
        !observables.is_empty(),
        "no velo artifact-stream observable in logs/aiperf.log — the bytes did not go \
         over velo (or the toggle did not engage). Log tail:\n{}",
        aiperf_log(&cellular)
            .lines()
            .rev()
            .take(40)
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(
        velo_observables(&baseline).is_empty(),
        "single-process baseline must not ship artifacts over velo, but observed: {:?}",
        velo_observables(&baseline)
    );
    for line in &observables {
        assert!(
            line.contains("transport=\"velo\"") || line.contains("transport=velo"),
            "artifact-stream observable is not tagged velo: {line}"
        );
    }
    for cell_id in 0..CELLS {
        let cell_lines: Vec<&String> = observables
            .iter()
            .filter(|l| l.contains(&format!("cell_id={cell_id}")))
            .collect();
        assert!(
            !cell_lines.is_empty(),
            "cell {cell_id} streamed no artifacts over velo; observables:\n{}",
            observables.join("\n")
        );
        assert!(
            cell_lines.iter().any(|l| l.contains("inputs.json")),
            "cell {cell_id} did not stream inputs.json over velo; its streams:\n{}",
            cell_lines
                .iter()
                .map(|l| l.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    eprintln!(
        "velo shipping observed: {} artifact streams across {} cells:\n{}",
        observables.len(),
        CELLS,
        observables.join("\n")
    );

    // inputs.json must be byte-identical (seeded, timing-free).
    let inputs_base = std::fs::read(
        baseline
            .artifacts
            .find_file("**/inputs.json")
            .expect("baseline inputs.json"),
    )
    .unwrap();
    let inputs_cell = std::fs::read(
        cellular
            .artifacts
            .find_file("**/inputs.json")
            .expect("cellular inputs.json"),
    )
    .unwrap();
    assert_eq!(
        inputs_base, inputs_cell,
        "inputs.json must be byte-identical between the baseline and the velo-shipped \
         cellular run"
    );

    // records.jsonl deterministic row set.
    let recs_base = baseline.artifacts.jsonl();
    let recs_cell = cellular.artifacts.jsonl();
    assert_eq!(
        recs_base.len(),
        ENTRIES as usize,
        "full-coverage baseline must emit one record per conversation"
    );
    assert_eq!(
        recs_base.len(),
        recs_cell.len(),
        "baseline and velo-shipped cellular must emit the same records.jsonl count"
    );
    assert_eq!(
        sorted(&recs_base, record_projection),
        sorted(&recs_cell, record_projection),
        "records.jsonl deterministic row SET diverged after velo shipping"
    );

    // raw.jsonl request-payload set.
    let raw_base = baseline.artifacts.raw_records();
    let raw_cell = cellular.artifacts.raw_records();
    let raw_key = |r: &Value| r["payload"]["messages"].to_string();
    assert_eq!(
        sorted(&raw_base, raw_key),
        sorted(&raw_cell, raw_key),
        "raw.jsonl request-payload SET diverged after velo shipping"
    );

    // outputs.json deterministic text set.
    let ob = outputs(&baseline);
    let oc = outputs(&cellular);
    assert_eq!(ob.len(), oc.len(), "outputs.json row count diverged");
    assert_eq!(
        sorted(&ob, |r| output_projection(r)),
        sorted(&oc, |r| output_projection(r)),
        "outputs.json deterministic (text) SET diverged after velo shipping"
    );
}

/// The SAME velo-shipping run but with `AIPERF_CELLULAR_HUB=1`: the controller stands
/// up one velo hub (cell↔controller + `/artifact` + discovery plugins on a single
/// anchor) instead of the standalone transport + velo artifact receiver. The cells
/// reach the hub by the identical `tcp://` coordinate, the artifact bytes ride the
/// hub-mounted `/artifact` plugin (same `received artifact stream over velo`
/// observable), and the merged per-record output matches the single-cell baseline —
/// proving the hub path is wire- and data-equivalent to the default path.
#[tokio::test]
async fn test_cellular_hub_mode_velo_shipping_matches_single_cell() {
    // Flaky on macOS CI like the other artifact e2es; skip there.
    if cfg!(target_os = "macos") {
        return;
    }

    let h_base = AIPerfHarness::new().await;
    let baseline = run(&h_base, 1, false);
    assert!(
        baseline.success(),
        "1-cell baseline run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        baseline.exit_code,
        baseline.stdout,
        baseline.stderr
    );

    let h_cell = AIPerfHarness::new().await;
    let cellular = run_modes(&h_cell, CELLS, true, true);
    assert!(
        cellular.success(),
        "hub-mode velo {CELLS}-cell run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        cellular.exit_code,
        cellular.stdout,
        cellular.stderr
    );

    // The multi-cell run went through the controller.
    assert!(
        cellular
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "{CELLS}-cell hub-mode run must go through the controller (cellular-heartbeat.json)"
    );

    // The bytes crossed the velo plane via the hub-mounted `/artifact` plugin: one
    // observable per cell × file, each tagged transport="velo".
    let observables = velo_observables(&cellular);
    assert!(
        !observables.is_empty(),
        "no velo artifact-stream observable in hub-mode logs — the bytes did not go over \
         the hub's /artifact plugin (or the toggle did not engage). Log tail:\n{}",
        aiperf_log(&cellular)
            .lines()
            .rev()
            .take(40)
            .collect::<Vec<_>>()
            .join("\n")
    );
    for cell_id in 0..CELLS {
        assert!(
            observables
                .iter()
                .any(|l| l.contains(&format!("cell_id={cell_id}"))
                    && l.contains("transport=\"velo\"")),
            "cell {cell_id} streamed no artifacts over the hub in hub mode; observables:\n{}",
            observables.join("\n")
        );
    }

    // inputs.json byte-identical (seeded, timing-free).
    let inputs_base = std::fs::read(
        baseline
            .artifacts
            .find_file("**/inputs.json")
            .expect("baseline inputs.json"),
    )
    .unwrap();
    let inputs_cell = std::fs::read(
        cellular
            .artifacts
            .find_file("**/inputs.json")
            .expect("hub-mode inputs.json"),
    )
    .unwrap();
    assert_eq!(
        inputs_base, inputs_cell,
        "inputs.json must be byte-identical between the baseline and the hub-mode run"
    );

    // records.jsonl deterministic row set (excludes session_num, per the module doc).
    let recs_base = baseline.artifacts.jsonl();
    let recs_cell = cellular.artifacts.jsonl();
    assert_eq!(
        recs_base.len(),
        recs_cell.len(),
        "baseline and hub-mode cellular must emit the same records.jsonl count"
    );
    assert_eq!(
        sorted(&recs_base, record_projection),
        sorted(&recs_cell, record_projection),
        "records.jsonl deterministic row SET diverged in hub mode"
    );

    // outputs.json deterministic text set.
    let ob = outputs(&baseline);
    let oc = outputs(&cellular);
    assert_eq!(ob.len(), oc.len(), "outputs.json row count diverged in hub mode");
    assert_eq!(
        sorted(&ob, |r| output_projection(r)),
        sorted(&oc, |r| output_projection(r)),
        "outputs.json deterministic (text) SET diverged in hub mode"
    );

    eprintln!(
        "hub-mode velo shipping observed: {} artifact streams across {} cells",
        observables.len(),
        CELLS
    );
}
