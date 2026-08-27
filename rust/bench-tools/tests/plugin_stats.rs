// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Statistical fixture and evidence-record contracts for plugin parity.

use aiperf_bench_tools::plugin_stats::{
    ExperimentAttempt, ExporterRepetition, ExporterSampleContract, NonInferiorityGate,
    PairedSample, RatioDirection, Variant, balanced_pair_orders, decode_samples_jsonl,
    encode_samples_jsonl, evaluate_non_authoritative_exporter_fixture,
    evaluate_non_authoritative_paired_fixture, validate_experiment_attempts,
};

const DIGEST: &str = "blake3:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn sample(pair: usize, variant: Variant, metric: &str, value: f64) -> PairedSample {
    PairedSample {
        scenario: "statistical-fixture".to_owned(),
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

fn paired_samples(metric: &str, ratios: &[f64]) -> Vec<PairedSample> {
    let mut samples = Vec::new();
    for (pair, ratio) in ratios.iter().copied().enumerate() {
        let static_value = 100.0 + pair as f64 * 0.01;
        let dynamic_value = match metric {
            "successful_requests_per_second" | "output_tokens_per_second" => static_value * ratio,
            _ => static_value / ratio,
        };
        for variant in [Variant::Static, Variant::Dynamic] {
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
    samples
}

fn two_pair_zero_fixture(metric: &str, first_static: f64, first_dynamic: f64) -> Vec<PairedSample> {
    vec![
        sample(0, Variant::Static, metric, first_static),
        sample(0, Variant::Dynamic, metric, first_dynamic),
        sample(1, Variant::Static, metric, 1.0),
        sample(1, Variant::Dynamic, metric, 1.0),
    ]
}

#[test]
fn explicitly_non_authoritative_fixture_is_seeded_and_one_sided() {
    let ratios = (0..30)
        .map(|pair| 0.984 + (pair % 3) as f64 * 0.0005)
        .collect::<Vec<_>>();
    let samples = paired_samples("successful_requests_per_second", &ratios);
    let gate = NonInferiorityGate::standard("successful_requests_per_second");
    let first = evaluate_non_authoritative_paired_fixture(&samples, &gate, 20260826)
        .expect("fixed statistical fixture is valid");
    let second = evaluate_non_authoritative_paired_fixture(&samples, &gate, 20260826)
        .expect("fixed statistical fixture is valid");
    assert_eq!(first, second);
    assert!(!first.passed);
    assert_eq!(first.paired_ratios.len(), ratios.len());
    assert!(
        first
            .paired_ratios
            .windows(2)
            .any(|pair| pair[0] != pair[1])
    );
}

#[test]
fn balanced_pair_order_has_an_exact_seeded_golden() {
    assert_eq!(
        balanced_pair_orders(20260826),
        vec![
            [Variant::Dynamic, Variant::Static],
            [Variant::Static, Variant::Dynamic],
            [Variant::Static, Variant::Dynamic],
            [Variant::Static, Variant::Dynamic],
            [Variant::Static, Variant::Dynamic],
            [Variant::Static, Variant::Dynamic],
            [Variant::Dynamic, Variant::Static],
            [Variant::Static, Variant::Dynamic],
            [Variant::Dynamic, Variant::Static],
            [Variant::Dynamic, Variant::Static],
            [Variant::Static, Variant::Dynamic],
            [Variant::Dynamic, Variant::Static],
            [Variant::Dynamic, Variant::Static],
            [Variant::Static, Variant::Dynamic],
            [Variant::Dynamic, Variant::Static],
            [Variant::Dynamic, Variant::Static],
            [Variant::Static, Variant::Dynamic],
            [Variant::Dynamic, Variant::Static],
            [Variant::Dynamic, Variant::Static],
            [Variant::Static, Variant::Dynamic],
            [Variant::Static, Variant::Dynamic],
            [Variant::Static, Variant::Dynamic],
            [Variant::Static, Variant::Dynamic],
            [Variant::Static, Variant::Dynamic],
            [Variant::Dynamic, Variant::Static],
            [Variant::Dynamic, Variant::Static],
            [Variant::Dynamic, Variant::Static],
            [Variant::Static, Variant::Dynamic],
            [Variant::Dynamic, Variant::Static],
            [Variant::Dynamic, Variant::Static],
        ]
    );
    assert_ne!(
        balanced_pair_orders(20260826),
        balanced_pair_orders(20260827)
    );
}

#[test]
fn zero_ratio_semantics_are_finite_in_all_four_directional_cases() {
    let throughput_gate = NonInferiorityGate::standard("successful_requests_per_second");
    let latency_gate = NonInferiorityGate::standard("ttft_p99");

    let throughput_static_zero = evaluate_non_authoritative_paired_fixture(
        &two_pair_zero_fixture("successful_requests_per_second", 0.0, 1.0),
        &throughput_gate,
        7,
    )
    .expect("dynamic/static with a zero denominator is finite");
    assert_eq!(
        throughput_static_zero.ratio_direction,
        RatioDirection::DynamicOverStatic
    );
    assert_eq!(throughput_static_zero.paired_ratios[0], 1.0 / f64::EPSILON);

    let throughput_dynamic_zero = evaluate_non_authoritative_paired_fixture(
        &two_pair_zero_fixture("successful_requests_per_second", 1.0, 0.0),
        &throughput_gate,
        7,
    )
    .expect("dynamic/static with a zero numerator is finite");
    assert_eq!(throughput_dynamic_zero.paired_ratios[0], f64::EPSILON);

    let latency_static_zero = evaluate_non_authoritative_paired_fixture(
        &two_pair_zero_fixture("ttft_p99", 0.0, 1.0),
        &latency_gate,
        7,
    )
    .expect("static/dynamic with a zero numerator is finite");
    assert_eq!(
        latency_static_zero.ratio_direction,
        RatioDirection::StaticOverDynamic
    );
    assert_eq!(latency_static_zero.paired_ratios[0], f64::EPSILON);

    let latency_dynamic_zero = evaluate_non_authoritative_paired_fixture(
        &two_pair_zero_fixture("ttft_p99", 1.0, 0.0),
        &latency_gate,
        7,
    )
    .expect("static/dynamic with a zero denominator is finite");
    assert_eq!(latency_dynamic_zero.paired_ratios[0], 1.0 / f64::EPSILON);
    for report in [
        throughput_static_zero,
        throughput_dynamic_zero,
        latency_static_zero,
        latency_dynamic_zero,
    ] {
        assert!(report.observed_ratio.is_finite());
        assert!(report.lower_confidence_bound.is_finite());
        assert!(
            report
                .bootstrap_distribution
                .iter()
                .all(|value| value.is_finite())
        );
    }
}

#[test]
fn zero_zero_is_neutral_and_non_finite_or_negative_values_reject() {
    let gate = NonInferiorityGate::standard("ttft_p99");
    let report = evaluate_non_authoritative_paired_fixture(
        &two_pair_zero_fixture("ttft_p99", 0.0, 0.0),
        &gate,
        9,
    )
    .expect("zero/zero latency is neutral");
    assert_eq!(report.paired_ratios[0], 1.0);

    for invalid in [f64::NAN, f64::INFINITY, -1.0] {
        let samples = two_pair_zero_fixture("ttft_p99", invalid, 1.0);
        assert!(evaluate_non_authoritative_paired_fixture(&samples, &gate, 9).is_err());
    }
}

#[test]
fn canonical_jsonl_includes_the_row_identity_and_rejects_field_reordering() {
    let samples = vec![sample(
        0,
        Variant::Static,
        "successful_requests_per_second",
        100.0,
    )];
    let encoded = encode_samples_jsonl(&samples).expect("sample is canonicalizable");
    assert_eq!(
        decode_samples_jsonl(&encoded).expect("canonical JSONL parses"),
        samples
    );
    let wrong_order = format!(
        "{{\"pair_id\":\"pair-00\",\"scenario\":\"statistical-fixture\",\"variant\":\"static\",\"metric\":\"successful_requests_per_second\",\"value\":100.0,\"unit\":\"ratio-source\",\"commit\":\"0123456789abcdef0123456789abcdef01234567\",\"artifact_digest\":\"{DIGEST}\",\"experiment_identity_digest\":\"{DIGEST}\"}}\n"
    );
    assert!(decode_samples_jsonl(wrong_order.as_bytes()).is_err());
}

#[test]
fn experiment_attempt_history_preserves_first_valid_authority() {
    assert!(
        validate_experiment_attempts(&[
            ExperimentAttempt::invalid(1, "host reboot"),
            ExperimentAttempt::valid_failure(2, "lower bound below threshold"),
            ExperimentAttempt::invalid(3, "must not replace a valid failure"),
        ])
        .is_err()
    );
    assert!(
        validate_experiment_attempts(&[
            ExperimentAttempt::invalid(1, "host reboot"),
            ExperimentAttempt::invalid(2, "affinity loss"),
            ExperimentAttempt::valid_pass(3),
        ])
        .is_ok()
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
    let summary = evaluate_non_authoritative_exporter_fixture(
        &ExporterSampleContract::normative(),
        &repetitions,
    )
    .expect("fixed exporter vector is valid");
    assert_eq!(summary.active_duration_nanoseconds, 32_000_000_000);
    assert_eq!(summary.processed_records, 1_600_000);
    assert_eq!(summary.exporter_nanoseconds_per_record, 20_000.0);

    let mut changed = ExporterSampleContract::normative();
    changed.sample_repetitions = 15;
    assert!(evaluate_non_authoritative_exporter_fixture(&changed, &repetitions).is_err());

    let short_paired_repetitions = (0..16)
        .map(|ordinal| ExporterRepetition {
            ordinal,
            emitted_records: 100_000,
            output_digest: DIGEST.to_owned(),
            active_duration_nanoseconds: 1,
        })
        .collect::<Vec<_>>();
    assert!(
        evaluate_non_authoritative_exporter_fixture(
            &ExporterSampleContract::normative(),
            &short_paired_repetitions,
        )
        .is_ok(),
        "only authoritative static-calibration evidence has a 30-second minimum"
    );
}
