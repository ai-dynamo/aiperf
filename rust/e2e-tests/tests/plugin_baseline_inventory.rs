// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract tests for the immutable native-plugin parity inventory.

use std::{collections::BTreeSet, fs, path::PathBuf};

use serde_yaml::Value;

const REQUIRED: &[&str] = &[
    "schema_version",
    "host_commit",
    "rustc",
    "target",
    "cargo_profile",
    "feature_sets",
    "build_commands",
    "runtime_scenarios",
    "artifacts",
    "allocation_probe",
    "raw_samples",
];

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn mapping<'a>(value: &'a Value, name: &str) -> &'a serde_yaml::Mapping {
    value
        .as_mapping()
        .unwrap_or_else(|| panic!("{name} must be a mapping"))
}

fn field<'a>(mapping: &'a serde_yaml::Mapping, name: &str) -> &'a Value {
    mapping
        .get(Value::String(name.to_owned()))
        .unwrap_or_else(|| panic!("missing required field `{name}`"))
}

fn identity_text<'a>(mapping: &'a serde_yaml::Mapping, name: &str) -> &'a str {
    let text = field(mapping, name)
        .as_str()
        .unwrap_or_else(|| panic!("identity `{name}` must be text"));
    assert!(!text.is_empty(), "identity `{name}` must not be empty");
    assert!(
        !text.contains("pending") && !text.contains("placeholder"),
        "identity `{name}` must be measured, got {text}"
    );
    text
}

fn assert_blake3(value: &str, name: &str) {
    let digest = value
        .strip_prefix("blake3:")
        .unwrap_or_else(|| panic!("{name} must have a blake3: prefix"));
    assert_eq!(digest.len(), 64, "{name} must contain 32 digest bytes");
    assert!(
        digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{name} must be hexadecimal"
    );
}

#[test]
fn baseline_inventory_has_complete_comparable_scenarios() {
    let path = repository_root().join("rust/benchmarks/plugin-parity.yaml");
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let document: Value = serde_yaml::from_str(&contents).expect("baseline inventory must be YAML");
    let inventory = mapping(&document, "inventory");

    for required in REQUIRED {
        assert!(
            inventory.contains_key(Value::String((*required).to_owned())),
            "missing `{required}`"
        );
    }
    assert_eq!(field(inventory, "schema_version").as_i64(), Some(1));
    for identity in [
        "host_commit",
        "rustc",
        "target",
        "cargo_profile",
        "canonical_inventory_digest",
    ] {
        identity_text(inventory, identity);
    }

    let build_commands = mapping(field(inventory, "build_commands"), "build_commands");
    for feature in ["default", "engine", "grpc", "parquet", "dynosim", "full"] {
        let build = mapping(field(build_commands, feature), feature);
        identity_text(build, "command");
        identity_text(build, "artifact_digest");
        assert!(field(build, "first_build_nanoseconds").as_u64().is_some());
        assert!(field(build, "second_build_nanoseconds").as_u64().is_some());
    }

    let scenarios = field(inventory, "runtime_scenarios")
        .as_sequence()
        .expect("runtime_scenarios must be a sequence");
    assert!(
        !scenarios.is_empty(),
        "at least one runtime scenario is required"
    );
    let expected_scenarios = BTreeSet::from([
        "http_non_streaming_c1",
        "http_non_streaming_c64",
        "http_streaming_c1",
        "http_streaming_c64",
        "grpc_unary_c1",
        "grpc_unary_c64",
        "grpc_streaming_c1",
        "grpc_streaming_c64",
        "http_streaming_workers4",
        "otlp_disabled_capture",
        "otlp_enabled_capture",
        "exporter_100k",
    ]);
    let mut actual_scenarios = BTreeSet::new();
    let legal_primary = BTreeSet::from([
        "successful_requests_per_second",
        "output_tokens_per_second",
        "cpu_nanoseconds_per_successful_request",
        "exporter_nanoseconds_per_record",
    ]);
    for scenario in scenarios {
        let scenario = mapping(scenario, "runtime scenario");
        for required in [
            "request_budget",
            "minimum_duration_seconds",
            "core_assignment",
            "mock_placement",
            "artifact_digest",
            "response_shape",
            "warmups",
            "estimator",
            "bootstrap_seed",
            "primary_metric",
            "ratio_direction",
            "measured_metrics",
            "invalidation_classifier",
            "harness_mock_digest",
            "firmware",
            "memory_topology",
            "canonical_inventory_digest",
        ] {
            assert!(
                !field(scenario, required).is_null(),
                "scenario `{required}` must not be null"
            );
        }
        let name = identity_text(scenario, "name");
        assert!(actual_scenarios.insert(name), "duplicate scenario `{name}`");
        assert!(field(scenario, "request_budget").as_u64().is_some());
        assert!(
            field(scenario, "minimum_duration_seconds")
                .as_u64()
                .is_some_and(|duration| duration >= 30)
        );
        assert_eq!(field(scenario, "warmups").as_u64(), Some(5));
        for digest in [
            "artifact_digest",
            "harness_mock_digest",
            "canonical_inventory_digest",
        ] {
            assert_blake3(identity_text(scenario, digest), digest);
        }
        let primary = field(scenario, "primary_metric")
            .as_str()
            .expect("primary_metric must be text");
        assert!(
            legal_primary.contains(primary),
            "illegal primary metric `{primary}`"
        );
        let expected_direction = match primary {
            "successful_requests_per_second" | "output_tokens_per_second" => "dynamic_over_static",
            "cpu_nanoseconds_per_successful_request" | "exporter_nanoseconds_per_record" => {
                "static_over_dynamic"
            }
            _ => unreachable!("primary metric legality was checked"),
        };
        assert_eq!(
            field(scenario, "ratio_direction").as_str(),
            Some(expected_direction),
            "primary metric `{primary}` has the wrong ratio direction"
        );
        let measured = field(scenario, "measured_metrics")
            .as_sequence()
            .expect("measured_metrics must be a sequence");
        let measured: BTreeSet<_> = measured.iter().filter_map(Value::as_str).collect();
        for metric in [
            "ttft_p50", "ttft_p90", "ttft_p99", "itl_p50", "itl_p90", "itl_p99",
        ] {
            assert!(
                measured.contains(metric),
                "missing mandatory secondary metric `{metric}`"
            );
            assert_ne!(primary, metric, "latency percentile cannot be primary");
        }
    }
    assert_eq!(actual_scenarios, expected_scenarios);

    let raw_samples = field(inventory, "raw_samples")
        .as_sequence()
        .expect("raw_samples must be a sequence");
    assert!(!raw_samples.is_empty());
    for sample in raw_samples {
        let path = sample.as_str().expect("raw sample paths must be text");
        assert!(path.starts_with("artifacts/native-plugin-baseline/raw/"));
        assert!(!PathBuf::from(path).is_absolute());
    }
}
