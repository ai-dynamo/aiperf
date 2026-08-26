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

/// A single-turn synthetic config with every per-record artifact this suite compares
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
    // The controller forwards each velo receiver line under the `aiperf` binary target,
    // so `aiperf=info` (not a per-target `aiperf_cellular_artifact=info` directive, which
    // only matches the un-forwarded target) reliably surfaces the `received artifact
    // stream over velo` observable into `logs/aiperf.log`. Matches `run_velo_cells`.
    let mut env: Vec<(&str, &str)> = vec![("AIPERF_LOG", "warn,aiperf=info")];
    if velo {
        env.push(("AIPERF_CELL_ARTIFACT_HTTP_FORCE", "1"));
        env.push(("AIPERF_ARTIFACT_TRANSPORT", "velo"));
    }
    if hub {
        env.push(("AIPERF_CELLULAR_HUB", "1"));
    }
    // Pin `--random-seed`, which sets `run.random_seed`. Cross-topology byte parity
    // (1-cell baseline vs N-cell cellular) is contractually keyed to `run.random_seed`:
    // every cell inherits it verbatim, whereas a seedless cellular run auto-derives a
    // shared seed from the per-run `benchmark_id` (see `resolve_cellular_seed` in
    // `cellular_controller.rs`), which differs between two independent harness runs and
    // yields divergent run-seed-governed sampling. The config's `dataset.random_seed`
    // deterministically pins only the dataset sampler and is deliberately kept separate
    // from the run seed, so it alone cannot make the baseline and cellular runs
    // byte-identical. Matches the seed pin the hub-parity test relies on.
    h.run_env(
        &format!(
            "--config {} --ui simple --random-seed {SEED}",
            cfg.display()
        ),
        &env,
    )
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

/// Run the config against `h`'s mock at `CELLS` cells over the velo plane, with the
/// controller-forwarded observable surfaced (`warn,aiperf=info`) and, when `hub`, the
/// hub anchor (`AIPERF_CELLULAR_HUB=1`). When `fanout`, the dataset fan-out + phaser
/// control planes are enabled too (`AIPERF_CELL_DATASET_FANOUT` / `AIPERF_CELL_PHASER_
/// START`) — under `hub` those planes ride the one hub anchor as the `/dataset` and
/// `/phaser` plugins. The paths are driven through the identical config + force seam +
/// velo transport; only the anchor (and, per `fanout`, the enabled planes) differ.
fn run_velo_cells(h: &AIPerfHarness, hub: bool, fanout: bool) -> RunResult {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg = tmp.path().join("velo_coverage.yaml");
    std::fs::write(&cfg, config(&h.mock.url, CELLS)).unwrap();
    // The controller forwards each receiver line under the `aiperf` binary target, so an
    // `aiperf=info` level (rather than a per-target `aiperf_cellular_artifact=info`
    // directive that only matches the un-forwarded target) reliably surfaces the
    // `received artifact stream over velo` observable into `logs/aiperf.log`.
    let mut env: Vec<(&str, &str)> = vec![
        ("AIPERF_LOG", "warn,aiperf=info"),
        ("AIPERF_CELL_ARTIFACT_HTTP_FORCE", "1"),
        ("AIPERF_ARTIFACT_TRANSPORT", "velo"),
    ];
    if hub {
        env.push(("AIPERF_CELLULAR_HUB", "1"));
    }
    if fanout {
        // Fan-out requires the phaser availability interlock (ShardsAvailable per chunk).
        env.push(("AIPERF_CELL_DATASET_FANOUT", "1"));
        env.push(("AIPERF_CELL_PHASER_START", "1"));
    }
    // `--random-seed` sets `run.random_seed`, which every cell inherits verbatim (the
    // controller only auto-derives a per-identity seed when none is authored), so both
    // the default and hub runs synthesize the byte-identical dataset — the seed pin is
    // what makes two independent cellular runs comparable.
    h.run_env(
        &format!(
            "--config {} --ui simple --random-seed {SEED}",
            cfg.display()
        ),
        &env,
    )
}

/// Assert a velo-shipping cellular run streamed every cell's artifacts over the velo
/// plane (one `transport="velo"` observable per cell), and return its records + outputs.
fn assert_velo_run(label: &str, r: &RunResult) {
    assert!(
        r.success(),
        "{label} velo {CELLS}-cell run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );
    assert!(
        r.artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "{label} {CELLS}-cell run must go through the controller (cellular-heartbeat.json)"
    );
    let observables = velo_observables(r);
    assert!(
        !observables.is_empty(),
        "{label}: no velo artifact-stream observable — the bytes did not go over velo. \
         Log tail:\n{}",
        aiperf_log(r)
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
                    && (l.contains("transport=\"velo\"") || l.contains("transport=velo"))),
            "{label}: cell {cell_id} streamed no artifacts over velo; observables:\n{}",
            observables.join("\n")
        );
    }
}

/// A hub-mode cellular run (`AIPERF_CELLULAR_HUB=1`) stands up ONE velo hub as the
/// cellular anchor — the cell↔controller plugin (register/heartbeat/partition), the
/// `/artifact` plugin (the streaming-zstd receiver), and the discovery plugin on a
/// single velo instance — instead of the standalone transport + separate artifact
/// receiver. This asserts the hub path is WIRE- and DATA-equivalent to the default
/// (standalone) velo path: both 3-cell runs ship every cell's artifacts over velo, and
/// their merged per-record output (records.jsonl / raw.jsonl / outputs.json) is
/// identical up to the accepted per-cell-local `session_num`. Comparing the two
/// cellular paths (not a 1-cell baseline) isolates the anchor change from the unrelated
/// single-vs-multi-cell session-id/synthesis differences.
#[tokio::test]
async fn test_cellular_hub_mode_matches_default_velo_path() {
    // Flaky on macOS CI like the other artifact e2es; skip there.
    if cfg!(target_os = "macos") {
        return;
    }

    let h_default = AIPerfHarness::new().await;
    let default_path = run_velo_cells(&h_default, false, false);
    assert_velo_run("default", &default_path);

    let h_hub = AIPerfHarness::new().await;
    let hub_path = run_velo_cells(&h_hub, true, false);
    assert_velo_run("hub", &hub_path);

    // inputs.json byte-identical: both are 3-cell velo runs over the same seeded
    // dataset; only the control-plane anchor differs, never the dataset.
    let inputs_default = std::fs::read(
        default_path
            .artifacts
            .find_file("**/inputs.json")
            .expect("default inputs.json"),
    )
    .unwrap();
    let inputs_hub = std::fs::read(
        hub_path
            .artifacts
            .find_file("**/inputs.json")
            .expect("hub inputs.json"),
    )
    .unwrap();
    assert_eq!(
        inputs_default, inputs_hub,
        "inputs.json must be byte-identical between the default and hub velo paths"
    );

    // records.jsonl deterministic row set (excludes session_num per the module doc).
    let recs_default = default_path.artifacts.jsonl();
    let recs_hub = hub_path.artifacts.jsonl();
    assert_eq!(
        recs_default.len(),
        ENTRIES as usize,
        "full-coverage default velo run must emit one record per conversation"
    );
    assert_eq!(
        recs_default.len(),
        recs_hub.len(),
        "default and hub velo paths must emit the same records.jsonl count"
    );
    assert_eq!(
        sorted(&recs_default, record_projection),
        sorted(&recs_hub, record_projection),
        "records.jsonl deterministic row SET diverged between the default and hub paths"
    );

    // raw.jsonl request-payload set.
    let raw_default = default_path.artifacts.raw_records();
    let raw_hub = hub_path.artifacts.raw_records();
    let raw_key = |r: &Value| r["payload"]["messages"].to_string();
    assert_eq!(
        sorted(&raw_default, raw_key),
        sorted(&raw_hub, raw_key),
        "raw.jsonl request-payload SET diverged between the default and hub paths"
    );

    // outputs.json deterministic text set.
    let od = outputs(&default_path);
    let oh = outputs(&hub_path);
    assert_eq!(
        od.len(),
        oh.len(),
        "outputs.json row count diverged (default vs hub)"
    );
    assert_eq!(
        sorted(&od, |r| output_projection(r)),
        sorted(&oh, |r| output_projection(r)),
        "outputs.json deterministic (text) SET diverged between the default and hub paths"
    );

    eprintln!(
        "hub-mode parity confirmed: {} default + {} hub velo artifact streams across {} cells",
        velo_observables(&default_path).len(),
        velo_observables(&hub_path).len(),
        CELLS
    );
}

/// A COMPLETE hub-mode run: `AIPERF_CELLULAR_HUB=1` WITH the dataset fan-out + phaser
/// control planes enabled, so the hub carries the cell↔controller, `/artifact`,
/// `/dataset`, and `/phaser` plugins on ONE velo anchor — a full replacement of the
/// standalone control/data planes for the run. This asserts the hub anchor is WIRE- and
/// DATA-equivalent to the standalone anchor while BOTH drive the dataset fan-out +
/// phaser planes: the controller generates the request-ids once and broadcasts them, the
/// phaser gates chunk availability, and each cell subscribes over its anchor and
/// dispatches its owned slice. Comparing hub-fanout to standalone-fanout (both fan-out,
/// so the dispatch source is identical) isolates the anchor change from the fan-out
/// dispatch-source change, keeping the strong per-record row-set parity.
#[tokio::test]
async fn test_cellular_hub_mode_dataset_fanout_and_phaser_matches_standalone() {
    // Flaky on macOS CI like the other artifact e2es; skip there.
    if cfg!(target_os = "macos") {
        return;
    }

    // Standalone anchor, fan-out + phaser planes bound directly on the control-plane velo.
    let h_std = AIPerfHarness::new().await;
    let std_path = run_velo_cells(&h_std, false, true);
    assert_velo_run("standalone-fanout", &std_path);

    // Hub anchor, fan-out + phaser planes mounted as the `/dataset` + `/phaser` plugins.
    let h_hub = AIPerfHarness::new().await;
    let hub_path = run_velo_cells(&h_hub, true, true);
    assert_velo_run("hub-fanout", &hub_path);

    // inputs.json byte-identical: both are 3-cell fan-out runs over the same seeded
    // dataset; only the anchor differs, never the controller-generated dataset.
    let inputs_std = std::fs::read(
        std_path
            .artifacts
            .find_file("**/inputs.json")
            .expect("standalone inputs.json"),
    )
    .unwrap();
    let inputs_hub = std::fs::read(
        hub_path
            .artifacts
            .find_file("**/inputs.json")
            .expect("hub inputs.json"),
    )
    .unwrap();
    assert_eq!(
        inputs_std, inputs_hub,
        "inputs.json must be byte-identical between the standalone and hub fan-out paths"
    );

    // records.jsonl deterministic row set (excludes session_num per the module doc).
    let recs_std = std_path.artifacts.jsonl();
    let recs_hub = hub_path.artifacts.jsonl();
    assert_eq!(
        recs_std.len(),
        recs_hub.len(),
        "standalone and hub fan-out paths must emit the same records.jsonl count"
    );
    assert_eq!(
        sorted(&recs_std, record_projection),
        sorted(&recs_hub, record_projection),
        "records.jsonl deterministic row SET diverged between the standalone and hub fan-out paths"
    );

    // raw.jsonl request-payload set.
    let raw_std = std_path.artifacts.raw_records();
    let raw_hub = hub_path.artifacts.raw_records();
    let raw_key = |r: &Value| r["payload"]["messages"].to_string();
    assert_eq!(
        sorted(&raw_std, raw_key),
        sorted(&raw_hub, raw_key),
        "raw.jsonl request-payload SET diverged between the standalone and hub fan-out paths"
    );

    // outputs.json deterministic text set.
    let os = outputs(&std_path);
    let oh = outputs(&hub_path);
    assert_eq!(
        os.len(),
        oh.len(),
        "outputs.json row count diverged (standalone vs hub fan-out)"
    );
    assert_eq!(
        sorted(&os, |r| output_projection(r)),
        sorted(&oh, |r| output_projection(r)),
        "outputs.json deterministic (text) SET diverged between the standalone and hub fan-out paths"
    );

    eprintln!(
        "hub-mode dataset+phaser parity confirmed: {} standalone + {} hub velo artifact streams across {} cells",
        velo_observables(&std_path).len(),
        velo_observables(&hub_path).len(),
        CELLS
    );
}
