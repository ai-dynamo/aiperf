// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Static streaming config validation integration tests.
//!
//! These drive the two stages `aiperf config validate` performs: YAML
//! normalization (which now raises the streaming Config-v2 cross-field rules)
//! and the in-process static registry stage.

use std::path::PathBuf;

use aiperf_cli::{streaming_preflight, yaml};

fn artifact_dir() -> Option<PathBuf> {
    Some(PathBuf::from("/tmp/aiperf-validate"))
}

/// A stream config whose `reliability` block is authored inline.
fn reliability_yaml() -> &'static str {
    include_str!("fixtures/streaming-shadow-reliability.yaml")
}

/// A stream config that leaves `reliability` at its documented defaults.
fn shadow_yaml() -> &'static str {
    include_str!("fixtures/streaming-shadow.yaml")
}

/// Build one stream config with the given `reliability` block body.
fn yaml_with_reliability(block: &str) -> String {
    let base = shadow_yaml();
    let (head, tail) = base
        .split_once("  shadow_replay:")
        .unwrap_or_else(|| panic!("fixture must contain a shadow_replay section"));
    format!("{head}    reliability:\n{block}  shadow_replay:{tail}")
}

#[test]
fn streaming_yaml_normalizes_without_python_or_secrets() {
    let inputs =
        yaml::normalize_str(shadow_yaml(), artifact_dir()).expect("streaming-shadow.yaml resolves");
    let streams = inputs
        .dataset_streams
        .as_ref()
        .expect("dataset_streams survives normalization");
    assert_eq!(streams.items.len(), 1);
    assert_eq!(streams.items[0].id, "s1");
    assert!(inputs.shadow_replay.is_some());
    let serialized = serde_json::to_string(&inputs).expect("inputs serialize");
    assert!(
        !serialized.contains("secret-value"),
        "serialized inputs must carry no secret material"
    );
}

#[test]
fn authored_reliability_block_normalizes_verbatim() {
    let inputs = yaml::normalize_str(reliability_yaml(), artifact_dir())
        .expect("streaming-shadow-reliability.yaml resolves");
    let reliability = inputs
        .dataset_streams
        .as_ref()
        .expect("dataset_streams survives normalization")
        .reliability;
    assert_eq!(reliability.partition_retry_limit, 5);
    assert_eq!(reliability.endpoint_retry_limit, 0);
    assert_eq!(reliability.checkpoint_retry_limit, 2);
    assert_eq!(reliability.export_retry_limit, 2);
    assert_eq!(reliability.retry_backoff_ms, 200);
}

#[test]
fn absent_reliability_block_normalizes_the_documented_defaults() {
    let inputs =
        yaml::normalize_str(shadow_yaml(), artifact_dir()).expect("streaming-shadow.yaml resolves");
    let reliability = inputs
        .dataset_streams
        .as_ref()
        .expect("dataset_streams survives normalization")
        .reliability;
    assert_eq!(reliability.partition_retry_limit, 3);
    assert_eq!(reliability.endpoint_retry_limit, 0);
    assert_eq!(reliability.checkpoint_retry_limit, 3);
    assert_eq!(reliability.export_retry_limit, 3);
    assert_eq!(reliability.retry_backoff_ms, 100);
    assert_eq!(reliability.partition_holes_before_admission_fence, None);
    assert_eq!(reliability.quarantines_before_admission_fence, None);
    assert_eq!(reliability.endpoint_failures_before_admission_fence, None);
    assert_eq!(
        reliability
            .checkpoint_failures_before_admission_fence
            .map(std::num::NonZeroU64::get),
        Some(3)
    );
}

#[test]
fn config_validate_refuses_a_stream_resource_the_selected_workload_forbids() {
    // No `shadow_replay` workload is compiled into the stock distribution, so a
    // stream resource on a scheduled workload is refused by the presence rule
    // before any streaming-internal message can fire. This pins the ordering:
    // the resource-requirement message wins over a stream-internal one.
    let inputs =
        yaml::normalize_str(shadow_yaml(), artifact_dir()).expect("streaming-shadow.yaml resolves");
    let result =
        streaming_preflight::validate_statically(&inputs).expect("static validation completes");
    assert!(
        !result.is_valid,
        "a stream resource on a scheduled workload must be refused"
    );
    let joined = result
        .errors
        .iter()
        .map(|error| error.message.as_str())
        .collect::<Vec<_>>()
        .join(" | ");
    assert!(
        joined.contains("dataset_streams"),
        "refusal must name the offending resource: {joined}"
    );
}

#[test]
fn config_validate_rejects_mixed_datasets_and_dataset_streams() {
    let yaml = shadow_yaml().replace(
        "  dataset_streams:",
        "  datasets:\n    - {type: synthetic, prompts: {isl: 128, osl: 16}}\n  dataset_streams:",
    );
    let error = yaml::normalize_str(&yaml, artifact_dir())
        .expect_err("datasets and dataset_streams must not coexist");
    let message = format!("{error:#}");
    assert!(
        message.contains("mutually exclusive"),
        "expected a mutual-exclusion refusal, got: {message}"
    );
}

#[test]
fn config_validate_rejects_accuracy_with_dataset_streams() {
    let yaml = shadow_yaml().replace(
        "  dataset_streams:",
        "  accuracy:\n    evaluator: exact_match\n  dataset_streams:",
    );
    let error =
        yaml::normalize_str(&yaml, artifact_dir()).expect_err("accuracy needs a finite dataset");
    let message = format!("{error:#}");
    assert!(
        message.contains("accuracy"),
        "expected an accuracy refusal, got: {message}"
    );
}

#[test]
fn reliability_policy_unknown_fields_are_rejected() {
    let yaml = yaml_with_reliability("      fail_run: true\n");
    let error = yaml::normalize_str(&yaml, artifact_dir())
        .expect_err("an unknown reliability key must be refused");
    let message = format!("{error:#}");
    assert!(
        message.contains("fail_run") || message.contains("unknown field"),
        "expected an unknown-field refusal naming the key, got: {message}"
    );
}

#[test]
fn zero_backoff_with_nonzero_retry_limit_is_rejected() {
    let yaml =
        yaml_with_reliability("      partition_retry_limit: 3\n      retry_backoff_ms: 0\n");
    let error = yaml::normalize_str(&yaml, artifact_dir())
        .expect_err("a zero backoff cannot pace a nonzero retry limit");
    let message = format!("{error:#}");
    assert!(
        message.contains("retry_backoff_ms"),
        "expected a refusal naming the backoff field, got: {message}"
    );
}

#[test]
fn retry_backoff_overflow_is_rejected() {
    let yaml = yaml_with_reliability(&format!("      retry_backoff_ms: {}\n", u64::MAX));
    let error = yaml::normalize_str(&yaml, artifact_dir())
        .expect_err("an unrepresentable backoff must be refused");
    let message = format!("{error:#}");
    assert!(
        message.contains("retry_backoff_ms") && message.contains("duration"),
        "expected a representability refusal naming the field, got: {message}"
    );
}

#[test]
fn ordinary_fail_run_policy_is_not_authorable() {
    use aiperf_runtime::engine::protocol_v2::StreamingReliabilityPolicyV2;
    use aiperf_runtime::engine::streaming_policy::prepare_streaming_policy;
    use aiperf_runtime::streaming::reliability::{
        StreamingIssueClass, StreamingIssueComponentId, StreamingIssueDisposition,
        StreamingIssueScopeKind, StreamingIssueThresholdRule, StreamingReliabilityError,
    };

    let policy = StreamingReliabilityPolicyV2 {
        partition_retry_limit: 3,
        endpoint_retry_limit: 0,
        checkpoint_retry_limit: 3,
        export_retry_limit: 3,
        retry_backoff_ms: 100,
        partition_holes_before_admission_fence: None,
        quarantines_before_admission_fence: None,
        endpoint_failures_before_admission_fence: None,
        checkpoint_failures_before_admission_fence: std::num::NonZeroU64::new(3),
    };
    let prepared = prepare_streaming_policy(&policy).expect("the default policy prepares");
    for rule in prepared.rules() {
        assert_ne!(
            rule.exhausted_disposition(),
            StreamingIssueDisposition::FailRun,
            "rule {:?} must not reach FailRun",
            rule.rule_id()
        );
    }

    let rule_id = StreamingIssueComponentId::new("authored_fail_run").expect("valid component id");
    let refused = StreamingIssueThresholdRule::new(
        rule_id,
        StreamingIssueScopeKind::Partition,
        StreamingIssueClass::Retryable,
        None,
        3,
        StreamingIssueDisposition::FailRun,
        None,
    );
    assert_eq!(refused.err(), Some(StreamingReliabilityError::IllegalDisposition));
}

#[test]
fn endpoint_failure_threshold_is_cumulative_across_every_authorable_class() {
    use aiperf_runtime::engine::protocol_v2::StreamingReliabilityPolicyV2;
    use aiperf_runtime::engine::streaming_policy::prepare_streaming_policy;
    use aiperf_runtime::streaming::reliability::StreamingIssueScopeKind;

    let fence = std::num::NonZeroU64::new(7);
    let policy = StreamingReliabilityPolicyV2 {
        partition_retry_limit: 3,
        endpoint_retry_limit: 0,
        checkpoint_retry_limit: 3,
        export_retry_limit: 3,
        retry_backoff_ms: 100,
        partition_holes_before_admission_fence: None,
        quarantines_before_admission_fence: None,
        endpoint_failures_before_admission_fence: fence,
        checkpoint_failures_before_admission_fence: None,
    };
    let prepared = prepare_streaming_policy(&policy).expect("policy prepares");
    let action_rules: Vec<_> = prepared
        .rules()
        .iter()
        .filter(|rule| rule.scope() == StreamingIssueScopeKind::Action)
        .collect();
    assert_eq!(action_rules.len(), 3, "one wildcard rule per authorable class");
    for rule in action_rules {
        assert_eq!(
            rule.admission_fence_count(),
            fence,
            "every action class shares the one cumulative threshold"
        );
    }
}

#[test]
fn config_validate_succeeds_and_lists_deferred_checks() {
    let yaml = "schemaVersion: \"2.0\"\n\
        benchmark:\n\
        \x20 model: my-model\n\
        \x20 endpoint: {type: chat, url: \"127.0.0.1:8000\"}\n\
        \x20 phases: {type: concurrency, requests: 10, concurrency: 1}\n";
    let inputs = yaml::normalize_str(yaml, artifact_dir()).expect("minimal config resolves");
    let result =
        streaming_preflight::validate_statically(&inputs).expect("static validation completes");
    assert!(
        result.is_valid,
        "a non-streaming config must still validate: {:?}",
        result.errors
    );
    assert!(
        !result.deferred.is_empty(),
        "static validation reports the checks it deferred"
    );
}
