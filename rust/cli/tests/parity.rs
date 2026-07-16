// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Byte-exact parity of the native `BenchmarkRun` against Python golden vectors.
//!
//! The native type is both the domain object and the wire request. This suite
//! asserts two things per golden:
//!
//! 1. The `aiperf` stdin wire type (`BenchmarkRunWireV2`, the bare run) accepts
//!    the golden's `run` object — structural validity against the real consumer.
//! 2. Each already-ported `cfg` section round-trips byte-exact: deserializing
//!    the golden through the native `BenchmarkRun` (which drops not-yet-ported
//!    keys) and re-serializing reproduces that section's subtree exactly.
//!
//! As sections land, add their key to `PORTED_CFG_SECTIONS`. A section not in
//! the list is intentionally dropped by the native type and not asserted yet.

use aiperf_cli::flags::ProfileFlags;
use aiperf_cli::load;
use aiperf_cli::model::BenchmarkRun;
use aiperf_runtime::engine::protocol_v2::BenchmarkRunWireV2;

/// `cfg` sections the native type currently models; asserted for byte-exact
/// round-trip. Extend as each section is ported.
const PORTED_CFG_SECTIONS: &[&str] = &[
    "endpoint",
    "models",
    "tokenizer",
    "transport",
    "runtime",
    "metrics",
    "artifacts",
    "datasets",
    "phases",
    "gpu_telemetry",
    "server_metrics",
    "network_latency",
    "slos",
    "sidecars",
    "accuracy",
    "synthesis",
    "endpoint_profiles",
    "failure_policy",
    "scenario",
    "trajectory_start_max_ratio",
    "trajectory_start_min_ratio",
    "unsafe_override",
];

/// Run-level (non-`cfg`) fields the native type currently models byte-exact.
const PORTED_RUN_FIELDS: &[&str] = &["resolved"];

/// Golden fixtures the native path reproduces byte-exact (name = fixture stem).
const FIXTURES: &[&str] = &[
    "minimal_chat",
    "rate_chat",
    "completions",
    "duration",
    "warmup",
    "num_conv",
    "tokenizer",
    "isl_osl",
    "auth",
    "file_ds",
    "public_ds",
    "fixed_sched",
    "rate_constant",
    "rate_gamma",
    "tuning",
    "ramp",
    "cancel",
    "multi_turn",
    "user_centric",
    "endpoint_extra",
    "prefill",
    "warmup_extra",
    "no_telemetry",
    "sm_formats",
    "goodput",
    "netlat_fixed",
    "netlat_probe",
    "otel",
    "config_extra",
    "sched_offset",
    "sketch",
    "image",
    "image_format",
    "dataset_filter",
    "audio",
    "video",
    "video_audio",
    "video_codec",
    "adaptive",
    "hf_subset",
    "interturn_cap",
    "video_audio",
    "prefix_shared",
    "prefix_pool",
    "mlflow_wandb",
    "endpoint_extra2",
    "sm_urls",
    "arrival",
    "gpu_urls",
    "dataset_entries",
    "custom_endpoint",
    "export_raw",
    "cells",
    "seq_dist",
    "warmup_arrival",
    "export_extras",
    "alias_genai",
    "toplevel",
    "agentic",
    "rankings",
    "accuracy",
    "synthesis",
];

/// Load a golden request JSON (paths are relative to the crate dir `rust/cli`).
fn load_golden(name: &str) -> serde_json::Value {
    let path = format!("../../tools/parity/golden/{name}.request.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read golden {path}: {e}"));
    serde_json::from_slice(&bytes).expect("golden is valid JSON")
}

/// Re-serialize the golden `run` through the native `BenchmarkRun` type.
fn runner_view(golden: &serde_json::Value) -> serde_json::Value {
    let run: BenchmarkRun =
        serde_json::from_value(golden["run"].clone()).expect("golden run deserializes as native");
    serde_json::to_value(&run).expect("native run serializes")
}

/// Read a `.args` fixture and split into argv tokens (fixtures use simple
/// whitespace-separated tokens, no quoting).
fn fixture_args(name: &str) -> Vec<String> {
    let path = format!("../../tools/parity/fixtures/{name}.args");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    text.split_whitespace().map(str::to_owned).collect()
}

/// Assert the built run's consumed sections + run fields + artifact_dir match a
/// golden, and that it deserializes as valid runner input.
fn assert_matches_golden(fixture: &str, built: &serde_json::Value, golden: &serde_json::Value) {
    for section in PORTED_CFG_SECTIONS {
        // A conditional section (e.g. `slos`) is null/absent on both sides for
        // fixtures that don't exercise it; `null == null` passes.
        assert_eq!(
            &built["cfg"][section], &golden["run"]["cfg"][section],
            "[{fixture}] cfg.{section} diverges\n got: {:#}\nwant: {:#}",
            built["cfg"][section], golden["run"]["cfg"][section]
        );
    }
    for field in PORTED_RUN_FIELDS {
        assert_eq!(
            &built[field], &golden["run"][field],
            "[{fixture}] run.{field} diverges"
        );
    }
    // The `export` section's static, registry-derived metadata (console_txt +
    // genai_perf header/tag maps) is byte-exact; the genai_perf envelope is a
    // best-effort echo (Python exclude_unset) and is compared only structurally.
    assert_export_static(
        fixture,
        &built["cfg"]["export"],
        &golden["run"]["cfg"]["export"],
    );

    // The stdin wire is now the bare run object (no `{protocol_version, operation}`
    // wrapper): the built run must deserialize directly as the real consumer type.
    let _: BenchmarkRunWireV2 = serde_json::from_value(built.clone())
        .unwrap_or_else(|e| panic!("[{fixture}] invalid runner input: {e}"));
}

/// Compare the byte-exact static parts of `export`; skip the best-effort
/// genai_perf envelope (opaque, echoed by the runner).
fn assert_export_static(fixture: &str, built: &serde_json::Value, golden: &serde_json::Value) {
    if golden.is_null() {
        return;
    }
    assert_eq!(
        built["console_txt"], golden["console_txt"],
        "[{fixture}] export.console_txt diverges"
    );
    for key in ["enabled", "header_map", "filtered_tags", "scalar_tags"] {
        assert_eq!(
            built["genai_perf"][key], golden["genai_perf"][key],
            "[{fixture}] export.genai_perf.{key} diverges"
        );
    }
    assert!(
        built["genai_perf"]["envelope"].is_object(),
        "[{fixture}] export.genai_perf.envelope must be present"
    );
    // MLflow/W&B config fields are byte-exact; params/config_json/cli_command
    // are best-effort (invocation-specific) and not compared.
    if !golden["mlflow"].is_null() {
        for key in [
            "enabled",
            "artifact_globs",
            "tracking_uri",
            "experiment",
            "run_name",
            "parent_run_id",
            "total_expected_requests",
            "tags",
        ] {
            assert_eq!(
                built["mlflow"][key], golden["mlflow"][key],
                "[{fixture}] export.mlflow.{key} diverges"
            );
        }
    }
    if !golden["wandb"].is_null() {
        for key in ["project", "entity", "run_name", "aiperf_version", "tags"] {
            assert_eq!(
                built["wandb"][key], golden["wandb"][key],
                "[{fixture}] export.wandb.{key} diverges"
            );
        }
    }
    // The OTLP sink is byte-exact except for the invocation-specific
    // `aiperf.benchmark.id` resource attribute (fresh uuid vs pinned golden).
    if !golden["otel"].is_null() {
        assert_eq!(
            built["otel"]["endpoint"], golden["otel"]["endpoint"],
            "[{fixture}] export.otel.endpoint diverges"
        );
        assert_eq!(
            built["otel"]["provider"], golden["otel"]["provider"],
            "[{fixture}] export.otel.provider diverges"
        );
        for attr in [
            "aiperf.endpoint.type",
            "aiperf.model.name",
            "service.instance.id",
        ] {
            assert_eq!(
                built["otel"]["resource_attributes"][attr],
                golden["otel"]["resource_attributes"][attr],
                "[{fixture}] export.otel.resource_attributes.{attr} diverges"
            );
        }
    }
}

#[test]
fn goldens_are_valid_runner_input() {
    // The golden files retain the historical `{operation, protocol_version, run}`
    // shape (Python-generated reference data), but the stdin wire is now the bare
    // `run` object — so validate the `run` subtree against the real consumer type.
    for fixture in FIXTURES {
        let golden = load_golden(fixture);
        let _: BenchmarkRunWireV2 = serde_json::from_value(golden["run"].clone())
            .unwrap_or_else(|e| panic!("[{fixture}] golden run is not valid BenchmarkRunWireV2: {e}"));
    }
}

#[test]
fn goldens_roundtrip_through_native_type() {
    // Deserializing a golden through the native type and re-serializing must
    // reproduce every ported section byte-exact (the type is correct).
    for fixture in FIXTURES {
        let golden = load_golden(fixture);
        let view = runner_view(&golden);
        assert_matches_golden(fixture, &view, &golden);
    }
}

#[test]
fn loader_reproduces_goldens() {
    // The pre-translation (flags -> native run) must reproduce each golden's
    // consumed sections + resolved + artifact_dir. benchmark_id/cli_command are
    // invocation-specific and excluded.
    for fixture in FIXTURES {
        let golden = load_golden(fixture);
        let flags = ProfileFlags::parse_from_args(&fixture_args(fixture))
            .unwrap_or_else(|e| panic!("[{fixture}] flags parse: {e}"));
        let run = load::resolve(&flags).unwrap_or_else(|e| panic!("[{fixture}] loader: {e}"));
        let built = serde_json::to_value(&run).expect("serialize built run");
        assert_matches_golden(fixture, &built, &golden);
        assert_eq!(
            built["artifact_dir"], golden["run"]["artifact_dir"],
            "[{fixture}] artifact_dir diverges"
        );
    }
}

/// YAML config fixtures: (config file stem, golden stem, artifact_dir).
const YAML_FIXTURES: &[(&str, &str, &str)] = &[
    ("basic", "yaml_basic", "/tmp/aiperf-parity/yaml_basic"),
    ("grpc", "grpc", "/tmp/aiperf-parity/grpc"),
    (
        "dynosim_offline",
        "dynosim_offline",
        "/tmp/aiperf-parity/dynosim_offline",
    ),
    (
        "dynosim_online",
        "dynosim_online",
        "/tmp/aiperf-parity/dynosim_online",
    ),
    // camelCase authoring (template-style): aliases resolve, wire is snake_case,
    // required_features sorted + deduped.
    (
        "dynosim_camel",
        "dynosim_camel",
        "/tmp/aiperf-parity/dynosim_camel",
    ),
    // YAML config surface parity (matching the flag surface section by section).
    (
        "yaml_endpoint",
        "yaml_endpoint",
        "/tmp/aiperf-parity/yaml_endpoint",
    ),
    ("yaml_infra", "yaml_infra", "/tmp/aiperf-parity/yaml_infra"),
    ("yaml_synth", "yaml_synth", "/tmp/aiperf-parity/yaml_synth"),
    ("yaml_media", "yaml_media", "/tmp/aiperf-parity/media"),
    ("yaml_phase", "yaml_phase", "/tmp/aiperf-parity/yaml_phase"),
    ("yaml_uc", "yaml_uc", "/tmp/aiperf-parity/yaml_uc"),
    (
        "yaml_public",
        "yaml_public",
        "/tmp/aiperf-parity/yaml_public",
    ),
    ("yaml_sched", "yaml_sched", "/tmp/aiperf-parity/yaml_sched"),
    (
        "yaml_adaptive",
        "yaml_adaptive",
        "/tmp/aiperf-parity/yaml_adaptive",
    ),
    (
        "yaml_warmup",
        "yaml_warmup",
        "/tmp/aiperf-parity/yaml_warmup",
    ),
    ("yaml_camel", "yaml_camel", "/tmp/aiperf-parity/yaml_camel"),
    (
        "yaml_records",
        "yaml_records",
        "/tmp/aiperf-parity/yaml_records",
    ),
    // `${ENV:default}` substitution + Jinja2 `variables:` expansion parity.
    ("jinja_vars", "yaml_jinja", "/tmp/aiperf-parity/jinja_vars"),
    ("env_prod", "yaml_envprod", "/tmp/aiperf-parity/env_prod"),
    // Inline `records:` file dataset (materialized on the wire, not a path).
    ("inline_ds", "yaml_inline", "/tmp/aiperf-parity/inline_ds"),
];

#[test]
fn yaml_configs_reproduce_goldens() {
    // The native YAML surface must reproduce the same consumed sections as
    // Python's resolve_config for each config file.
    for (config, golden_stem, artifact_dir) in YAML_FIXTURES {
        let golden = load_golden(golden_stem);
        let cfg = format!("../../tools/parity/configs/{config}.yaml");
        let run = aiperf_cli::yaml::resolve(
            std::path::Path::new(&cfg),
            Some(std::path::PathBuf::from(artifact_dir)),
        )
        .unwrap_or_else(|e| panic!("[{golden_stem}] yaml resolve: {e}"));
        let built = serde_json::to_value(&run).expect("serialize built run");
        assert_matches_golden(golden_stem, &built, &golden);
    }
}
