// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native CLI/config coverage for synthesis YAML overrides.

use std::fs;
use std::path::Path;
use std::process::Command;

use serde_json::{Value, json};

fn write_trace(path: &Path) {
    fs::write(
        path,
        concat!(
            r#"{"timestamp":0,"input_length":48,"output_length":8,"hash_ids":[1,2,3]}"#,
            "\n"
        ),
    )
    .expect("write trace fixture");
}

fn write_config(path: &Path, trace: &Path, artifact_dir: &Path) {
    fs::write(
        path,
        format!(
            "schemaVersion: \"2.0\"\n\
             benchmark:\n\
             \x20 model: mock-model\n\
             \x20 endpoint:\n\
             \x20   type: chat\n\
             \x20   url: http://127.0.0.1:9\n\
             \x20   streaming: true\n\
             \x20 transport:\n\
             \x20   type: dry_run\n\
             \x20   ttft_ms: 1\n\
             \x20   itl_ms: 1\n\
             \x20 dataset:\n\
             \x20   type: file\n\
             \x20   path: {}\n\
             \x20   format: mooncake_trace\n\
             \x20   synthesis:\n\
             \x20     corpus: sonnet\n\
             \x20     speedupRatio: 2.0\n\
             \x20     prefixLenMultiplier: 2.5\n\
             \x20     prefixRootMultiplier: 3\n\
             \x20     promptLenMultiplier: 4.0\n\
             \x20     outputLenMultiplier: 5.0\n\
             \x20     maxIsl: 6000\n\
             \x20     maxOsl: 7000\n\
             \x20     maxContextLength: 8000\n\
             \x20     allowDatasetWrap: false\n\
             \x20     idleGapCapSeconds: 9.0\n\
             \x20     trajectoryStartMinRatio: 0.1\n\
             \x20     trajectoryStartMaxRatio: 0.2\n\
             \x20     tStarRandomSeed: 7\n\
             \x20     datasetSamplingStrategy: sequential\n\
             \x20     cacheBustTarget: system_prefix\n\
             \x20 phases:\n\
             \x20   type: concurrency\n\
             \x20   requests: 1\n\
             \x20   concurrency: 1\n\
             \x20 artifacts:\n\
             \x20   dir: {}\n",
            trace.display(),
            artifact_dir.display(),
        ),
    )
    .expect("write config fixture");
}

fn run_profile(config: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .args([
            "profile",
            "--config",
            config.to_str().expect("config path utf8"),
            "--synthesis-speedup-ratio",
            "1.5",
            "--synthesis-prefix-len-multiplier",
            "1.6",
            "--synthesis-prefix-root-multiplier",
            "8",
            "--synthesis-prompt-len-multiplier",
            "1.7",
            "--synthesis-output-len-multiplier",
            "1.8",
            "--synthesis-max-isl",
            "6100",
            "--synthesis-max-osl",
            "7100",
            "--trace-idle-gap-cap-seconds",
            "11.5",
            "--max-context-length",
            "8100",
            "--allow-dataset-wrap",
            "--cache-bust",
            "first_turn_prefix",
            "--dataset-sampling-strategy",
            "random",
            "--trajectory-start-min-ratio",
            "0.3",
            "--trajectory-start-max-ratio",
            "0.4",
            "--random-seed",
            "99",
        ])
        .output()
        .expect("spawn aiperf profile --config")
}

fn summary_synthesis(artifact_dir: &Path) -> Value {
    let summary: Value = serde_json::from_slice(
        &fs::read(artifact_dir.join("profile_export_aiperf.json")).expect("read summary export"),
    )
    .expect("summary export is valid JSON");
    summary["input_config"]["datasets"][0]["synthesis"].clone()
}

#[test]
fn profile_config_cli_synthesis_flags_override_yaml_and_preserve_yaml_only_fields() {
    let scratch = tempfile::tempdir().expect("tempdir");
    let trace = scratch.path().join("trace.jsonl");
    let config = scratch.path().join("config.yaml");
    let artifact_dir = scratch.path().join("artifacts");

    write_trace(&trace);
    write_config(&config, &trace, &artifact_dir);

    let output = run_profile(&config);
    assert!(
        output.status.success(),
        "native YAML profile failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let synthesis = summary_synthesis(&artifact_dir);
    assert_eq!(
        synthesis,
        json!({
            "corpus": "sonnet",
            "speedup_ratio": 1.5,
            "prefix_len_multiplier": 1.6,
            "prefix_root_multiplier": 8,
            "prompt_len_multiplier": 1.7,
            "output_len_multiplier": 1.8,
            "max_isl": 6100,
            "max_osl": 7100,
            "max_context_length": 8100,
            "allow_dataset_wrap": true,
            "idle_gap_cap_seconds": 11.5,
            "trajectory_start_min_ratio": 0.3,
            "trajectory_start_max_ratio": 0.4,
            "t_star_random_seed": 99,
            "dataset_sampling_strategy": "random",
            "cache_bust_target": "first_turn_prefix"
        }),
        "CLI-authored synthesis fields must override YAML while preserving YAML-only corpus"
    );
}
