// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for cellular (multi-process) mode over the **gRPC**
//! transport, reached through the ordinary Python frontend via `--cells N`.
//!
//! gRPC and HTTP cellular runs share ONE executor: the coordinator selects a
//! different `RequestExecutorFactory` (`NativeGrpcExecutionBackendFactory` vs the
//! HTTP one), but the cell-issuer injection (`CellularAutonomousIssuer`) and the
//! records shipper (`CellRecordsShipper`) live in the shared `execute_native_inner`
//! loop above the transport — so a gRPC cell ships its partition exactly as an HTTP
//! cell does, and the controller merges them identically. This test proves that end
//! to end: `aiperf profile --config <grpc> --cells N` runs the mock's KServe OIP v2
//! gRPC target across N cells and reproduces the single-cell run's
//! dataset-deterministic metrics byte-for-byte.
//!
//! Requires the launched runner (`AIPERF_EXEC_BIN`) to include the `velo` cell
//! transport (default build), and the mock to serve its gRPC listener.

mod common;
use common::*;

const REQUEST_COUNT: u32 = 24;
const CONCURRENCY: u32 = 6;
const CELLS: u32 = 3;

/// The dataset-deterministic metrics a cellular merge must reproduce exactly: they
/// depend only on the seeded synthetic dataset (input tokens) and the deterministic
/// mock's response (output tokens), not on wall-clock timing.
const DETERMINISTIC_METRICS: &[&str] = &["input_sequence_length", "output_sequence_length"];

/// A Config-v2 YAML selecting the native gRPC KServe transport against `grpc_url`.
/// The harness appends `--artifact-dir` and `--tokenizer`, which override the
/// corresponding config fields; `--cells`/`--random-seed` are added per run.
fn grpc_config(grpc_url: &str) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 models: [{DEFAULT_MODEL}]\n\
        \x20 endpoint:\n\
        \x20   urls: [\"{grpc_url}\"]\n\
        \x20   type: kserve_v2_infer\n\
        \x20   streaming: false\n\
        \x20   waitForModelTimeout: 0.0\n\
        \x20 dataset:\n\
        \x20   type: synthetic\n\
        \x20   entries: {REQUEST_COUNT}\n\
        \x20   prompts:\n\
        \x20     isl: 32\n\
        \x20     osl: 16\n\
        \x20 phases:\n\
        \x20   - name: profiling\n\
        \x20     type: concurrency\n\
        \x20     requests: {REQUEST_COUNT}\n\
        \x20     concurrency: {CONCURRENCY}\n\
        \x20 gpuTelemetry: {{enabled: false}}\n\
        \x20 serverMetrics: {{enabled: false}}\n\
        \x20 transport:\n\
        \x20   type: grpc\n\
        \x20 runtime:\n\
        \x20   ui: none\n"
    )
}

/// `aiperf profile --config <grpc> --cells 3` runs end-to-end over gRPC and reports
/// the full request budget, via the multi-cell controller.
#[tokio::test]
async fn test_grpc_cellular_run_from_python_frontend() {
    let h = AIPerfHarness::new_with_grpc().await;
    let grpc_url = h
        .mock
        .grpc_url
        .clone()
        .expect("mock started with grpc listener");
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_file = tmp.path().join("grpc_cellular.yaml");
    std::fs::write(&cfg_file, grpc_config(&grpc_url)).unwrap();

    let r = h.run(&format!(
        "--config {} --cells {CELLS} --random-seed 42",
        cfg_file.display()
    ));
    assert!(
        r.success(),
        "grpc cellular run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );
    assert_eq!(
        r.artifacts.request_count() as u32,
        REQUEST_COUNT,
        "merged gRPC cellular report must carry every cell's records"
    );
    // Non-vacuous proof the CONTROLLER (multi-cell) path actually ran: the
    // cellular-heartbeat.json sidecar is written only by the controller after
    // aggregating the cells' shipped heartbeats. Its presence proves --cells reached
    // the runner and gRPC cells shipped partitions back — not a single-process run.
    assert!(
        r.artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "gRPC cellular run must emit the controller's cellular-heartbeat.json sidecar; \
         its absence means --cells did not reach the runner (single-process run)"
    );
}

/// A 3-cell gRPC run reproduces the 1-cell gRPC run's dataset-deterministic metrics
/// exactly. Same seed → same instance space; the 3-cell run partitions it across
/// cells and the controller merges, so the input/output sequence-length
/// distributions must be byte-identical. Wall-clock metrics are intentionally not
/// compared. This proves the gRPC cell-ship + merge path is correct, not just live.
#[tokio::test]
async fn test_grpc_cellular_matches_single_cell() {
    let run = |cells: u32| async move {
        let h = AIPerfHarness::new_with_grpc().await;
        let grpc_url = h
            .mock
            .grpc_url
            .clone()
            .expect("mock started with grpc listener");
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_file = tmp.path().join("grpc_cellular.yaml");
        std::fs::write(&cfg_file, grpc_config(&grpc_url)).unwrap();
        let r = h.run(&format!(
            "--config {} --cells {cells} --random-seed 42",
            cfg_file.display()
        ));
        assert!(
            r.success(),
            "{cells}-cell gRPC run failed (exit {}):\nstderr:\n{}",
            r.exit_code,
            r.stderr
        );
        // Keep the harness alive until the caller finishes reading (its TempDir owns
        // the artifacts); return the run result to read its report + sidecars.
        (h, r)
    };

    let (_h1, baseline) = run(1).await;
    let (_h3, cellular) = run(CELLS).await;

    // Guard against a vacuous pass: the 3-cell run must go through the controller.
    // (The 1-cell run is single-process; only the multi-cell run emits the sidecar.)
    assert!(
        cellular
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "the 3-cell gRPC run must go multi-cell (controller sidecar present)"
    );

    let baseline_json = baseline.artifacts.json();
    let cellular_json = cellular.artifacts.json();
    for metric in DETERMINISTIC_METRICS {
        let single = &baseline_json[metric];
        let multi = &cellular_json[metric];
        assert!(
            !single.is_null(),
            "baseline gRPC report missing deterministic metric {metric}"
        );
        assert_eq!(
            single, multi,
            "gRPC {metric} must be byte-identical between the 1-cell and {CELLS}-cell \
             runs (same seed → same instance space → same merged distribution)"
        );
    }
}
