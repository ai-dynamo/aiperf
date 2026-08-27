// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen inventory and observed-experiment authority for plugin parity.

use std::collections::BTreeMap;

use aiperf_bench_tools::plugin_stats::{
    ExperimentObservationReceipt, MachineObservation, ObservedExperimentAuthority, PairedCase,
    PairedSample, RatioDirection, SimultaneousGateInput, SimultaneousGatePolicy, Variant,
    checked_in_case_plans, checked_in_inventory_digest, evaluate_simultaneous_gate,
};

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const FORGED_DIGEST: &str =
    "blake3:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const INVENTORY_DIGEST: &str =
    "blake3:298ee51d2c9d319b9db3571681e4056d5b57e264661e6892d189a1179c901222";

struct AuthorityFixture {
    directory: tempfile::TempDir,
    receipt_path: std::path::PathBuf,
    authority: ObservedExperimentAuthority,
    input: SimultaneousGateInput,
}

fn metric_direction(metric: &str) -> RatioDirection {
    match metric {
        "successful_requests_per_second" | "output_tokens_per_second" => {
            RatioDirection::DynamicOverStatic
        }
        _ => RatioDirection::StaticOverDynamic,
    }
}

fn authority_fixture() -> AuthorityFixture {
    let directory = tempfile::tempdir().expect("temporary authority directory");
    let write = |name: &str, bytes: &[u8]| {
        let path = directory.path().join(name);
        std::fs::write(&path, bytes).expect("observed fixture is written");
        path
    };
    let receipt = ExperimentObservationReceipt {
        source_commit: COMMIT.to_owned(),
        target: "x86_64-unknown-linux-gnu".to_owned(),
        profile: "release-fat-lto".to_owned(),
        source_tree_receipt_path: write("source-tree.receipt", b"source tree receipt"),
        cargo_lock_path: write("Cargo.lock", b"lockfile bytes"),
        rustc_receipt_path: write("rustc.receipt", b"rustc 1.98.0\n"),
        sysroot_receipt_path: write("sysroot.receipt", b"sysroot tree receipt"),
        static_artifact_path: write("static-artifact", b"static artifact bytes"),
        dynamic_artifact_path: write("dynamic-artifact", b"dynamic artifact bytes"),
        harness_artifact_path: write("harness-artifact", b"harness artifact bytes"),
        mock_server_artifact_path: write("mock-artifact", b"mock artifact bytes"),
        machine: MachineObservation {
            cpu_model: "paper-rig".to_owned(),
            cpu_stepping: "1".to_owned(),
            microcode: "0x1".to_owned(),
            core_topology: "cores=0-7".to_owned(),
            memory_topology: "node0=0-7".to_owned(),
            firmware: "firmware-1".to_owned(),
            kernel: "linux-6".to_owned(),
            allocator_provider: "mimalloc-provider".to_owned(),
            frequency_governor: "performance".to_owned(),
            affinity_isolation: "mock=0-3;client=4-7".to_owned(),
            mock_server_placement: "disjoint-local".to_owned(),
        },
        environment: BTreeMap::from([
            ("EMPTY_VALUE".to_owned(), Some(String::new())),
            (
                "RUSTC_WRAPPER".to_owned(),
                Some("/usr/bin/sccache".to_owned()),
            ),
            ("UNSET_VALUE".to_owned(), None),
        ]),
        bootstrap_seed: 20260826,
    };
    let receipt_path = directory.path().join("observation.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_vec(&receipt).expect("observation serializes"),
    )
    .expect("observation receipt is written");
    let authority =
        ObservedExperimentAuthority::acquire(&receipt).expect("observed authority validates");
    let plans = checked_in_case_plans().expect("checked-in inventory validates");
    let cases = plans
        .iter()
        .map(|plan| {
            let is_pattern_case = matches!(
                plan.scenario.as_str(),
                "http_streaming_c1" | "http_streaming_c64"
            );
            let mut samples = Vec::new();
            for (pair, scheduled) in authority.pair_schedule().iter().enumerate() {
                for metric in &plan.measured_metrics {
                    let static_value = 100.0 + (pair % 5) as f64 * 0.1;
                    let is_allocation_metric = matches!(
                        metric.as_str(),
                        "allocated_bytes_per_successful_request"
                            | "allocation_count_per_successful_request"
                    );
                    let ratio = if is_pattern_case && !is_allocation_metric {
                        0.996 + (pair % 3) as f64 * 0.0005
                    } else {
                        1.0
                    };
                    let dynamic_value = match metric_direction(metric) {
                        RatioDirection::DynamicOverStatic => static_value * ratio,
                        RatioDirection::StaticOverDynamic => static_value / ratio,
                    };
                    for variant in scheduled.member_order {
                        samples.push(PairedSample {
                            scenario: plan.scenario.clone(),
                            pair_id: scheduled.pair_id.clone(),
                            variant,
                            metric: metric.clone(),
                            value: match variant {
                                Variant::Static => static_value,
                                Variant::Dynamic => dynamic_value,
                            },
                            unit: "ratio-source".to_owned(),
                            commit: COMMIT.to_owned(),
                            artifact_digest: authority.artifact_digest(variant).to_owned(),
                            experiment_identity_digest: authority.identity_digest().to_owned(),
                        });
                    }
                }
            }
            PairedCase {
                scenario: plan.scenario.clone(),
                primary_metric: plan.primary_metric.clone(),
                samples,
                invalidation_attempts: Vec::new(),
            }
        })
        .collect();
    AuthorityFixture {
        directory,
        receipt_path,
        authority,
        input: SimultaneousGateInput { cases },
    }
}

#[test]
fn checked_in_task1_inventory_is_the_only_complete_authority() {
    assert_eq!(
        checked_in_inventory_digest().expect("checked-in digest validates"),
        INVENTORY_DIGEST
    );
    let plans = checked_in_case_plans().expect("checked-in plans validate");
    assert_eq!(plans.len(), 12);
    let case = plans
        .iter()
        .find(|case| case.scenario == "http_non_streaming_c1")
        .expect("HTTP non-streaming C1 is normative");
    assert_eq!(case.request_budget, 1_000);
    assert_eq!(case.warmups, 5);
    assert_eq!(case.retained_pairs, 30);
    assert_eq!(case.minimum_duration_seconds, 30);
    assert_eq!(case.bootstrap_seed, 20260826);
    assert_eq!(case.measured_metrics.len(), 9);
    assert!(case.response_shape.contains("ITL is zero"));
    assert!(case.core_assignment.contains("mock=0-3"));
    assert!(case.mock_placement.contains("paper-rig"));
    assert!(
        case.invalidation_classifier
            .contains("max_replacement_pairs=5")
    );
    assert!(case.complete_case_digest.starts_with("blake3:"));
}

#[test]
fn missing_or_extra_cases_and_metrics_reject_before_resampling() {
    let fixture = authority_fixture();
    let mut subset = fixture.input.clone();
    subset.cases.pop();
    assert!(
        evaluate_simultaneous_gate(
            &subset,
            &fixture.authority,
            &SimultaneousGatePolicy::normative(),
        )
        .is_err()
    );

    let mut missing_metric = fixture.input.clone();
    let removed_metric = missing_metric.cases[0].samples[0].metric.clone();
    missing_metric.cases[0]
        .samples
        .retain(|sample| sample.metric != removed_metric);
    assert!(
        evaluate_simultaneous_gate(
            &missing_metric,
            &fixture.authority,
            &SimultaneousGatePolicy::normative(),
        )
        .is_err()
    );

    let mut extra_metric = fixture.input.clone();
    let mut extra_sample = extra_metric.cases[0].samples[0].clone();
    extra_sample.metric = "caller_added_metric".to_owned();
    extra_metric.cases[0].samples.push(extra_sample);
    assert!(
        evaluate_simultaneous_gate(
            &extra_metric,
            &fixture.authority,
            &SimultaneousGatePolicy::normative(),
        )
        .is_err()
    );
}

#[test]
fn row_identity_mutation_and_caller_sealed_identity_reject() {
    let fixture = authority_fixture();
    let mut one_row = fixture.input.clone();
    one_row.cases[0].samples[0].experiment_identity_digest = FORGED_DIGEST.to_owned();
    assert!(
        evaluate_simultaneous_gate(
            &one_row,
            &fixture.authority,
            &SimultaneousGatePolicy::normative(),
        )
        .is_err()
    );

    let mut forged = fixture.input.clone();
    for sample in forged.cases.iter_mut().flat_map(|case| &mut case.samples) {
        sample.experiment_identity_digest = FORGED_DIGEST.to_owned();
        sample.artifact_digest = FORGED_DIGEST.to_owned();
    }
    assert!(
        evaluate_simultaneous_gate(
            &forged,
            &fixture.authority,
            &SimultaneousGatePolicy::normative(),
        )
        .is_err()
    );

    let mut injected_identity = serde_json::to_value(&fixture.input).expect("input serializes");
    injected_identity
        .as_object_mut()
        .expect("gate input is an object")
        .insert(
            "experiment_identity".to_owned(),
            serde_json::json!({"identity_digest": FORGED_DIGEST}),
        );
    assert!(
        serde_json::from_value::<SimultaneousGateInput>(injected_identity).is_err(),
        "the authoritative input cannot carry a caller-sealed identity"
    );
}

#[test]
fn observed_artifact_mutation_changes_authority_and_rejects_stale_rows() {
    let fixture = authority_fixture();
    let receipt: ExperimentObservationReceipt = serde_json::from_slice(
        &std::fs::read(&fixture.receipt_path).expect("observation receipt is readable"),
    )
    .expect("observation receipt parses");
    std::fs::write(
        &receipt.dynamic_artifact_path,
        b"mutated dynamic artifact bytes",
    )
    .expect("observed artifact mutation is written");
    let reacquired =
        ObservedExperimentAuthority::acquire(&receipt).expect("changed observations reacquire");
    assert_ne!(
        reacquired.identity_digest(),
        fixture.authority.identity_digest()
    );
    assert!(
        evaluate_simultaneous_gate(
            &fixture.input,
            &reacquired,
            &SimultaneousGatePolicy::normative(),
        )
        .is_err()
    );
}

#[test]
fn static_and_dynamic_artifact_assignments_must_be_distinct() {
    let fixture = authority_fixture();
    let mut receipt: ExperimentObservationReceipt = serde_json::from_slice(
        &std::fs::read(&fixture.receipt_path).expect("observation receipt is readable"),
    )
    .expect("observation receipt parses");
    receipt.dynamic_artifact_path = receipt.static_artifact_path.clone();
    assert!(ObservedExperimentAuthority::acquire(&receipt).is_err());
}

#[test]
fn every_metric_enforces_the_seeded_exact_member_order() {
    let fixture = authority_fixture();
    let mut drifted = fixture.input.clone();
    drifted.cases[0].samples.swap(0, 1);
    assert!(
        evaluate_simultaneous_gate(
            &drifted,
            &fixture.authority,
            &SimultaneousGatePolicy::normative(),
        )
        .is_err()
    );
}

#[test]
fn production_cli_has_no_inventory_or_expected_digest_input() {
    let fixture = authority_fixture();
    let mut subset = fixture.input;
    subset.cases.pop();
    let input_path = fixture.directory.path().join("subset.json");
    std::fs::write(
        &input_path,
        serde_json::to_vec(&subset).expect("subset input serializes"),
    )
    .expect("subset input is written");
    let command = env!("CARGO_BIN_EXE_plugin_runtime_bench");
    let rejected = std::process::Command::new(command)
        .args([
            "evaluate",
            fixture
                .receipt_path
                .to_str()
                .expect("temporary path is UTF-8"),
            input_path.to_str().expect("temporary path is UTF-8"),
        ])
        .output()
        .expect("production acceptance seam executes");
    assert!(!rejected.status.success());

    let old_self_authorizing_shape = std::process::Command::new(command)
        .args([
            "evaluate",
            "caller-inventory.json",
            FORGED_DIGEST,
            input_path.to_str().expect("temporary path is UTF-8"),
        ])
        .output()
        .expect("legacy invocation executes");
    assert!(!old_self_authorizing_shape.status.success());
    assert!(
        String::from_utf8_lossy(&old_self_authorizing_shape.stderr)
            .contains("harness-observation.json")
    );
}

#[test]
fn production_cli_refuses_consistently_forged_receipt_files_and_rows() {
    let fixture = authority_fixture();
    let input_path = fixture.directory.path().join("forged-complete-input.json");
    std::fs::write(
        &input_path,
        serde_json::to_vec(&fixture.input).expect("forged input serializes"),
    )
    .expect("forged input is written");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_plugin_runtime_bench"))
        .args([
            "evaluate",
            fixture
                .receipt_path
                .to_str()
                .expect("temporary path is UTF-8"),
            input_path.to_str().expect("temporary path is UTF-8"),
        ])
        .output()
        .expect("production acceptance seam executes");
    assert!(
        !output.status.success(),
        "a caller-controlled receipt, files, authority, and matching rows must not self-authorize"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("same-process controlled measurement capability")
    );
}

#[test]
fn full_authoritative_joint_bootstrap_retains_the_golden_distribution() {
    let fixture = authority_fixture();
    let report = evaluate_simultaneous_gate(
        &fixture.input,
        &fixture.authority,
        &SimultaneousGatePolicy::normative(),
    )
    .expect("full checked-in matrix evaluates");
    assert!(report.passed);
    assert_eq!(report.metric_reports.len(), 108);
    let patterned = report
        .metric_reports
        .iter()
        .find(|metric| {
            metric.scenario == "http_streaming_c1" && metric.metric == "output_tokens_per_second"
        })
        .expect("patterned metric is retained");
    assert_eq!(patterned.lower_confidence_bound, 0.996383333333333);
    let distribution_digest = format!(
        "blake3:{}",
        blake3::hash(
            &serde_json::to_vec(&report.maximum_degradation_bootstrap_distribution)
                .expect("golden distribution serializes")
        )
        .to_hex()
    );
    assert_eq!(
        distribution_digest,
        "blake3:e9096ef04a23ffe2f9bdeb9495b367611dbe9b796c25acc49b120de924a466fd"
    );

    let mut decorrelated = fixture.input;
    let case = decorrelated
        .cases
        .iter_mut()
        .find(|case| case.scenario == "http_streaming_c64")
        .expect("second patterned case exists");
    for metric in ["cpu_nanoseconds_per_successful_request", "itl_p99"] {
        for variant in [Variant::Static, Variant::Dynamic] {
            let first = case
                .samples
                .iter()
                .position(|sample| {
                    sample.pair_id == "pair-00"
                        && sample.metric == metric
                        && sample.variant == variant
                })
                .expect("first correlated member exists");
            let second = case
                .samples
                .iter()
                .position(|sample| {
                    sample.pair_id == "pair-01"
                        && sample.metric == metric
                        && sample.variant == variant
                })
                .expect("second correlated member exists");
            let first_value = case.samples[first].value;
            case.samples[first].value = case.samples[second].value;
            case.samples[second].value = first_value;
        }
    }
    let decorrelated_report = evaluate_simultaneous_gate(
        &decorrelated,
        &fixture.authority,
        &SimultaneousGatePolicy::normative(),
    )
    .expect("same marginal vectors with changed joint correlation evaluate");
    assert_eq!(
        patterned.positive_paired_ratios,
        decorrelated_report
            .metric_reports
            .iter()
            .find(|metric| {
                metric.scenario == "http_streaming_c1"
                    && metric.metric == "output_tokens_per_second"
            })
            .expect("unmodified patterned metric exists")
            .positive_paired_ratios
    );
    assert_ne!(
        report.maximum_degradation_bootstrap_distribution,
        decorrelated_report.maximum_degradation_bootstrap_distribution,
        "case-wise independent resampling would miss changed joint correlation"
    );
}
