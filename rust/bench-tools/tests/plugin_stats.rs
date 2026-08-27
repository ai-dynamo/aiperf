// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavioral contract for native-plugin paired performance statistics.

use std::collections::BTreeMap;

use aiperf_bench_tools::plugin_stats::{
    AttemptDisposition, ExperimentAttempt, ExperimentIdentity, ExporterRepetition,
    ExporterSampleContract, InvalidationAttempt, NonInferiorityGate, NormativeCase,
    NormativeInventory, NormativeMetric, PairSchedule, PairedCase, PairedSample, RatioDirection,
    SimultaneousGateInput, SimultaneousGatePolicy, Variant, balanced_pair_orders,
    decode_samples_jsonl, encode_samples_jsonl, evaluate_exporter_sample, evaluate_paired_gate,
    evaluate_simultaneous_gate, validate_experiment_attempts,
};

const DIGEST: &str = "blake3:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const DYNAMIC_DIGEST: &str =
    "blake3:1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn direction(metric: &str) -> RatioDirection {
    match metric {
        "successful_requests_per_second" | "output_tokens_per_second" => {
            RatioDirection::DynamicOverStatic
        }
        _ => RatioDirection::StaticOverDynamic,
    }
}

fn authoritative_input(
    mut cases: Vec<PairedCase>,
    seed: u64,
) -> (NormativeInventory, SimultaneousGateInput) {
    let inventory = NormativeInventory::new(
        "test-component",
        cases
            .iter()
            .map(|case| {
                let mut metrics = case
                    .samples
                    .iter()
                    .map(|sample| sample.metric.clone())
                    .collect::<Vec<_>>();
                metrics.sort();
                metrics.dedup();
                NormativeCase {
                    scenario: case.scenario.clone(),
                    primary_metric: case.primary_metric.clone(),
                    metrics: metrics
                        .into_iter()
                        .map(|metric| NormativeMetric {
                            direction: direction(&metric),
                            metric,
                        })
                        .collect(),
                }
            })
            .collect(),
    )
    .expect("test inventory is canonical");
    let pair_schedule = balanced_pair_orders(seed)
        .into_iter()
        .enumerate()
        .map(|(pair, member_order)| PairSchedule {
            pair_id: format!("pair-{pair:02}"),
            member_order,
        })
        .collect::<Vec<_>>();
    for case in &mut cases {
        let mut reordered = Vec::with_capacity(case.samples.len());
        let mut metrics = case
            .samples
            .iter()
            .map(|sample| sample.metric.clone())
            .collect::<Vec<_>>();
        metrics.sort();
        metrics.dedup();
        for planned in &pair_schedule {
            for metric in &metrics {
                for variant in planned.member_order {
                    let mut sample = case
                        .samples
                        .iter()
                        .find(|sample| {
                            sample.pair_id == planned.pair_id
                                && sample.metric == *metric
                                && sample.variant == variant
                        })
                        .expect("fixture has both members")
                        .clone();
                    sample.artifact_digest = match variant {
                        Variant::Static => DIGEST.to_owned(),
                        Variant::Dynamic => DYNAMIC_DIGEST.to_owned(),
                    };
                    reordered.push(sample);
                }
            }
        }
        case.samples = reordered;
        for invalidation in &mut case.invalidation_attempts {
            let planned = pair_schedule
                .iter()
                .find(|planned| planned.pair_id == invalidation.pair_id)
                .expect("invalidation pair is planned");
            invalidation.member_order = planned.member_order;
            invalidation.members.sort_by_key(|sample| {
                planned
                    .member_order
                    .iter()
                    .position(|variant| variant == &sample.variant)
            });
            for sample in &mut invalidation.members {
                sample.artifact_digest = match sample.variant {
                    Variant::Static => DIGEST.to_owned(),
                    Variant::Dynamic => DYNAMIC_DIGEST.to_owned(),
                };
            }
        }
    }
    let identity = ExperimentIdentity {
        schema_version: 1,
        source_commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
        source_tree_digest: DIGEST.to_owned(),
        cargo_lock_digest: DIGEST.to_owned(),
        rustc: "rustc 1.97.1".to_owned(),
        sysroot_digest: DIGEST.to_owned(),
        target: "x86_64-unknown-linux-gnu".to_owned(),
        profile: "release".to_owned(),
        static_artifact_digest: DIGEST.to_owned(),
        dynamic_artifact_digest: DYNAMIC_DIGEST.to_owned(),
        harness_artifact_digest: DIGEST.to_owned(),
        mock_server_artifact_digest: DIGEST.to_owned(),
        inventory_digest: inventory.digest.clone(),
        cpu_model: "test-cpu".to_owned(),
        cpu_stepping: "1".to_owned(),
        microcode: "1".to_owned(),
        core_topology: "cores=0-7".to_owned(),
        memory_topology: "node0=0-7".to_owned(),
        firmware: "test-firmware".to_owned(),
        kernel: "test-kernel".to_owned(),
        allocator_provider: "test-mimalloc".to_owned(),
        frequency_governor: "performance".to_owned(),
        affinity_isolation: "mock=0-3;client=4-7".to_owned(),
        mock_server_placement: "disjoint".to_owned(),
        environment: BTreeMap::from([("RUSTC_WRAPPER".to_owned(), "sccache".to_owned())]),
        bootstrap_seed: seed,
        pair_schedule,
        identity_digest: String::new(),
    }
    .seal()
    .expect("test identity is complete");
    for case in &mut cases {
        for sample in &mut case.samples {
            sample.experiment_identity_digest = identity.identity_digest.clone();
        }
        for invalidation in &mut case.invalidation_attempts {
            for sample in &mut invalidation.members {
                sample.experiment_identity_digest = identity.identity_digest.clone();
            }
        }
    }
    let input = SimultaneousGateInput {
        experiment_identity: identity,
        cases,
    };
    (inventory, input)
}

fn sample(pair: usize, variant: Variant, metric: &str, value: f64) -> PairedSample {
    PairedSample {
        scenario: "http_streaming_c64".to_owned(),
        pair_id: format!("pair-{pair:02}"),
        variant,
        metric: metric.to_owned(),
        value,
        unit: "ratio-source".to_owned(),
        commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
        artifact_digest: DIGEST.to_owned(),
        experiment_identity_digest: DIGEST.to_owned(),
    }
}

fn balanced_samples(metrics: &[&str], dynamic_ratio: f64) -> Vec<PairedSample> {
    let mut samples = Vec::new();
    for pair in 0..30 {
        let order = if pair % 2 == 0 {
            [Variant::Static, Variant::Dynamic]
        } else {
            [Variant::Dynamic, Variant::Static]
        };
        for metric in metrics {
            let static_value = 100.0 + pair as f64 * 0.01;
            let dynamic_value = match *metric {
                "successful_requests_per_second" | "output_tokens_per_second" => {
                    static_value * dynamic_ratio
                }
                _ => static_value / dynamic_ratio,
            };
            for variant in order {
                samples.push(sample(
                    pair,
                    variant,
                    metric,
                    match variant {
                        Variant::Static => static_value,
                        Variant::Dynamic => dynamic_value,
                    },
                ));
            }
        }
    }
    samples
}

#[test]
fn paired_gate_is_one_sided_and_bootstrap_is_seeded() {
    let samples = balanced_samples(&["successful_requests_per_second"], 0.995);
    let gate = NonInferiorityGate {
        metric: "successful_requests_per_second".to_owned(),
        max_relative_regression: 0.01,
        confidence: 0.95,
    };
    let first = evaluate_paired_gate(&samples, &gate, 20260826).expect("fixed vector is valid");
    let second = evaluate_paired_gate(&samples, &gate, 20260826).expect("fixed vector is valid");

    assert_eq!(first, second);
    assert!(first.passed);
    assert_eq!(first.paired_ratios.len(), 30);
    assert!(first.lower_confidence_bound >= 0.99);

    let failing = NonInferiorityGate {
        max_relative_regression: 0.004,
        ..gate
    };
    assert!(
        !evaluate_paired_gate(&samples, &failing, 20260826)
            .expect("a valid regression is a report, not an error")
            .passed
    );
}

#[test]
fn balanced_pair_order_is_seeded_and_exactly_fifteen_each_way() {
    let first = balanced_pair_orders(20260826);
    let second = balanced_pair_orders(20260826);
    assert_eq!(first, second);
    assert_eq!(first.len(), 30);
    assert_eq!(
        first
            .iter()
            .filter(|order| order[0] == Variant::Static)
            .count(),
        15
    );
}

#[test]
fn canonical_jsonl_has_the_exact_field_order_and_round_trips() {
    let samples = vec![sample(
        0,
        Variant::Static,
        "successful_requests_per_second",
        100.0,
    )];
    let encoded = encode_samples_jsonl(&samples).expect("sample is canonicalizable");
    let text = std::str::from_utf8(&encoded).expect("JSONL is UTF-8");
    assert!(text.starts_with(
        "{\"scenario\":\"http_streaming_c64\",\"pair_id\":\"pair-00\",\"variant\":\"static\",\"metric\":\"successful_requests_per_second\",\"value\":100.0,\"unit\":\"ratio-source\",\"commit\":"
    ));
    assert_eq!(
        decode_samples_jsonl(&encoded).expect("canonical JSONL parses"),
        samples
    );
    let wrong_order = format!(
        "{{\"pair_id\":\"pair-00\",\"scenario\":\"http_streaming_c64\",\"variant\":\"static\",\"metric\":\"successful_requests_per_second\",\"value\":100.0,\"unit\":\"ratio-source\",\"commit\":\"0123456789abcdef0123456789abcdef01234567\",\"artifact_digest\":\"{DIGEST}\",\"experiment_identity_digest\":\"{DIGEST}\"}}\n"
    );
    assert!(decode_samples_jsonl(wrong_order.as_bytes()).is_err());
}

#[test]
fn statistics_reject_non_finite_and_negative_inputs_but_accept_zero_zero() {
    let mut samples = balanced_samples(&["cpu_nanoseconds_per_successful_request"], 1.0);
    samples[0].value = f64::NAN;
    assert!(
        evaluate_paired_gate(
            &samples,
            &NonInferiorityGate::standard("cpu_nanoseconds_per_successful_request"),
            7,
        )
        .is_err()
    );

    let mut samples = balanced_samples(&["cpu_nanoseconds_per_successful_request"], 1.0);
    samples[0].value = 0.0;
    samples[1].value = 0.0;
    let report = evaluate_paired_gate(
        &samples,
        &NonInferiorityGate::standard("cpu_nanoseconds_per_successful_request"),
        7,
    )
    .expect("zero/zero latency is a neutral valid pair");
    assert_eq!(report.paired_ratios[0], 1.0);

    samples[0].value = -1.0;
    assert!(
        evaluate_paired_gate(
            &samples,
            &NonInferiorityGate::standard("cpu_nanoseconds_per_successful_request"),
            7,
        )
        .is_err()
    );
}

#[test]
fn simultaneous_gate_covers_primary_and_secondary_matrix_deterministically() {
    let metrics = [
        "output_tokens_per_second",
        "cpu_nanoseconds_per_successful_request",
        "ttft_p50",
        "ttft_p90",
        "ttft_p99",
        "itl_p50",
        "itl_p90",
        "itl_p99",
    ];
    let case = PairedCase {
        scenario: "http_streaming_c64".to_owned(),
        primary_metric: "output_tokens_per_second".to_owned(),
        samples: balanced_samples(&metrics, 0.997),
        invalidation_attempts: Vec::new(),
    };
    let policy = SimultaneousGatePolicy::normative();
    let (inventory, input) = authoritative_input(vec![case], 20260826);
    let first = evaluate_simultaneous_gate(&input, &inventory, &inventory.digest, &policy)
        .expect("full metric matrix is valid");
    let second = evaluate_simultaneous_gate(&input, &inventory, &inventory.digest, &policy)
        .expect("full metric matrix is valid");

    assert_eq!(first, second);
    assert!(first.passed);
    assert!(!first.is_invalid);
    assert_eq!(first.metric_reports.len(), metrics.len());
    assert_eq!(
        first.maximum_degradation_bootstrap_distribution.len(),
        100_000
    );
    for report in &first.metric_reports {
        assert_eq!(report.static_summaries.len(), 30);
        assert_eq!(report.dynamic_summaries.len(), 30);
        assert_eq!(report.positive_paired_ratios.len(), 30);
    }
    assert_eq!(
        serde_json::to_vec(&first).expect("serializes"),
        serde_json::to_vec(&second).expect("serializes")
    );
}

#[test]
fn runtime_bench_invocations_emit_byte_identical_summaries() {
    let case = PairedCase {
        scenario: "http_streaming_c64".to_owned(),
        primary_metric: "output_tokens_per_second".to_owned(),
        samples: balanced_samples(
            &[
                "output_tokens_per_second",
                "cpu_nanoseconds_per_successful_request",
                "ttft_p99",
                "itl_p99",
            ],
            0.997,
        ),
        invalidation_attempts: Vec::new(),
    };
    let (inventory, input) = authoritative_input(vec![case], 20260826);
    let directory = tempfile::tempdir().expect("temporary fixture directory");
    let cases_path = directory.path().join("cases.json");
    let inventory_path = directory.path().join("inventory.json");
    std::fs::write(
        &cases_path,
        serde_json::to_vec(&input).expect("fixture serializes"),
    )
    .expect("fixture is written");
    std::fs::write(
        &inventory_path,
        serde_json::to_vec(&inventory).expect("inventory serializes"),
    )
    .expect("inventory is written");
    let run = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_plugin_runtime_bench"))
            .args([
                "evaluate",
                inventory_path.to_str().expect("temporary path is UTF-8"),
                &inventory.digest,
                cases_path.to_str().expect("temporary path is UTF-8"),
            ])
            .output()
            .expect("runtime benchmark executes")
    };
    let first = run();
    let second = run();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(
        second.status.success(),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );
    assert_eq!(first.stdout, second.stdout);
}

#[test]
fn secondary_metric_is_rejected_as_a_primary_name() {
    assert!(
        NormativeInventory::new(
            "bad-primary",
            vec![NormativeCase {
                scenario: "bad-primary".to_owned(),
                primary_metric: "ttft_p99".to_owned(),
                metrics: vec![NormativeMetric {
                    metric: "ttft_p99".to_owned(),
                    direction: RatioDirection::StaticOverDynamic,
                }],
            }],
        )
        .is_err()
    );
}

#[test]
fn cv_noise_invalidates_without_converting_a_valid_failure_into_a_retry() {
    let mut samples = balanced_samples(&["successful_requests_per_second"], 0.98);
    for member in &mut samples {
        if member.pair_id == "pair-29" && member.variant == Variant::Dynamic {
            member.value *= 2.0;
        }
    }
    let case = PairedCase {
        scenario: "http_streaming_c64".to_owned(),
        primary_metric: "successful_requests_per_second".to_owned(),
        samples,
        invalidation_attempts: Vec::new(),
    };
    let (inventory, input) = authoritative_input(vec![case], 9);
    let report = evaluate_simultaneous_gate(
        &input,
        &inventory,
        &inventory.digest,
        &SimultaneousGatePolicy::normative(),
    )
    .expect("noise is an invalid report");
    assert!(report.is_invalid);
    assert!(!report.passed);
    assert!(
        report
            .invalidation_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("coefficient of variation"))
    );

    let valid_failure = vec![
        ExperimentAttempt::invalid(1, "host reboot"),
        ExperimentAttempt::valid_failure(2, "lower bound below threshold"),
        ExperimentAttempt::invalid(3, "must not replace a valid failure"),
    ];
    assert!(validate_experiment_attempts(&valid_failure).is_err());
    assert!(
        validate_experiment_attempts(&[
            ExperimentAttempt::invalid(1, "host reboot"),
            ExperimentAttempt::invalid(2, "affinity loss"),
            ExperimentAttempt::valid_pass(3),
        ])
        .is_ok()
    );
    assert!(
        validate_experiment_attempts(&[
            ExperimentAttempt::invalid(1, "one"),
            ExperimentAttempt::invalid(2, "two"),
            ExperimentAttempt::invalid(3, "three"),
            ExperimentAttempt::invalid(4, "four"),
        ])
        .is_err()
    );
}

#[test]
fn pair_replacement_retains_raw_members_reason_order_and_caps_at_five() {
    let samples = balanced_samples(&["successful_requests_per_second"], 1.0);
    let invalid_member_samples = samples
        .iter()
        .filter(|sample| sample.pair_id == "pair-00")
        .cloned()
        .collect::<Vec<_>>();
    let invalidation = InvalidationAttempt {
        pair_id: "pair-00".to_owned(),
        experiment_attempt: 1,
        replacement_ordinal: 1,
        member_order: [Variant::Static, Variant::Dynamic],
        members: invalid_member_samples,
        reason: "mock death unrelated to either member".to_owned(),
        disposition: AttemptDisposition::InfrastructureInvalid,
    };
    let case = PairedCase {
        scenario: "http_streaming_c64".to_owned(),
        primary_metric: "successful_requests_per_second".to_owned(),
        samples: samples.clone(),
        invalidation_attempts: vec![invalidation.clone()],
    };
    let (inventory, input) = authoritative_input(vec![case], 10);
    let report = evaluate_simultaneous_gate(
        &input,
        &inventory,
        &inventory.digest,
        &SimultaneousGatePolicy::normative(),
    )
    .expect("one same-order infrastructure replacement is valid");
    assert_eq!(report.invalidation_attempts.len(), 1);

    let mut wrong_input = input.clone();
    wrong_input.cases[0].invalidation_attempts[0]
        .member_order
        .reverse();
    assert!(
        evaluate_simultaneous_gate(
            &wrong_input,
            &inventory,
            &inventory.digest,
            &SimultaneousGatePolicy::normative(),
        )
        .is_err()
    );

    let mut product_input = input.clone();
    product_input.cases[0].invalidation_attempts[0].disposition =
        AttemptDisposition::ProductFailure;
    assert!(
        evaluate_simultaneous_gate(
            &product_input,
            &inventory,
            &inventory.digest,
            &SimultaneousGatePolicy::normative(),
        )
        .is_err()
    );

    let too_many = (1..=6)
        .map(|ordinal| InvalidationAttempt {
            replacement_ordinal: ordinal,
            ..invalidation.clone()
        })
        .collect();
    let capped_case = PairedCase {
        scenario: "http_streaming_c64".to_owned(),
        primary_metric: "successful_requests_per_second".to_owned(),
        samples,
        invalidation_attempts: too_many,
    };
    let (capped_inventory, capped_input) = authoritative_input(vec![capped_case], 10);
    assert!(
        evaluate_simultaneous_gate(
            &capped_input,
            &capped_inventory,
            &capped_inventory.digest,
            &SimultaneousGatePolicy::normative(),
        )
        .is_err()
    );
}

#[test]
fn exporter_contract_is_exact_and_uses_only_summed_active_duration() {
    let repetitions = (0..16)
        .map(|ordinal| ExporterRepetition {
            ordinal,
            emitted_records: 100_000,
            output_digest: DIGEST.to_owned(),
            active_duration_nanoseconds: 2_000_000_000,
        })
        .collect::<Vec<_>>();
    let summary = evaluate_exporter_sample(&ExporterSampleContract::normative(), &repetitions)
        .expect("fixed exporter vector is valid");
    assert_eq!(summary.active_duration_nanoseconds, 32_000_000_000);
    assert_eq!(summary.processed_records, 1_600_000);
    assert_eq!(summary.retained_artifact_records, 100_000);
    assert_eq!(summary.exporter_nanoseconds_per_record, 20_000.0);

    let mut changed = ExporterSampleContract::normative();
    changed.corpus_records = 99_999;
    assert!(evaluate_exporter_sample(&changed, &repetitions).is_err());
    let mut changed = ExporterSampleContract::normative();
    changed.sample_repetitions = 15;
    assert!(evaluate_exporter_sample(&changed, &repetitions).is_err());
    let mut changed = ExporterSampleContract::normative();
    changed.processed_records = 100_000;
    assert!(evaluate_exporter_sample(&changed, &repetitions).is_err());
    let mut changed = ExporterSampleContract::normative();
    changed.retained_artifact_records = 1_600_000;
    assert!(evaluate_exporter_sample(&changed, &repetitions).is_err());

    let short = repetitions
        .iter()
        .cloned()
        .map(|mut repetition| {
            repetition.active_duration_nanoseconds = 1;
            repetition
        })
        .collect::<Vec<_>>();
    assert!(evaluate_exporter_sample(&ExporterSampleContract::normative(), &short).is_err());
}
