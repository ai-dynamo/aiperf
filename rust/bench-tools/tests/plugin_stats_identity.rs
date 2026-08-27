// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Identity and complete-inventory contract for plugin parity statistics.

use std::collections::BTreeMap;

use aiperf_bench_tools::plugin_stats::{
    ExperimentIdentity, NormativeCase, NormativeInventory, NormativeMetric, PairSchedule,
    PairedCase, PairedSample, RatioDirection, SimultaneousGateInput, SimultaneousGatePolicy,
    Variant, balanced_pair_orders, evaluate_simultaneous_gate,
};

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const DIGEST_A: &str = "blake3:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const DIGEST_B: &str = "blake3:1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn inventory() -> NormativeInventory {
    NormativeInventory::new(
        "transport-http",
        vec![
            NormativeCase {
                scenario: "case-a".to_owned(),
                primary_metric: "successful_requests_per_second".to_owned(),
                metrics: vec![
                    NormativeMetric {
                        metric: "successful_requests_per_second".to_owned(),
                        direction: RatioDirection::DynamicOverStatic,
                    },
                    NormativeMetric {
                        metric: "ttft_p99".to_owned(),
                        direction: RatioDirection::StaticOverDynamic,
                    },
                ],
            },
            NormativeCase {
                scenario: "case-b".to_owned(),
                primary_metric: "cpu_nanoseconds_per_successful_request".to_owned(),
                metrics: vec![
                    NormativeMetric {
                        metric: "cpu_nanoseconds_per_successful_request".to_owned(),
                        direction: RatioDirection::StaticOverDynamic,
                    },
                    NormativeMetric {
                        metric: "itl_p99".to_owned(),
                        direction: RatioDirection::StaticOverDynamic,
                    },
                ],
            },
        ],
    )
    .expect("inventory is canonical")
}

fn identity(inventory: &NormativeInventory, seed: u64) -> ExperimentIdentity {
    let schedule = balanced_pair_orders(seed)
        .into_iter()
        .enumerate()
        .map(|(pair, member_order)| PairSchedule {
            pair_id: format!("pair-{pair:02}"),
            member_order,
        })
        .collect();
    ExperimentIdentity {
        schema_version: 1,
        source_commit: COMMIT.to_owned(),
        source_tree_digest: DIGEST_A.to_owned(),
        cargo_lock_digest: DIGEST_A.to_owned(),
        rustc: "rustc 1.97.1".to_owned(),
        sysroot_digest: DIGEST_A.to_owned(),
        target: "x86_64-unknown-linux-gnu".to_owned(),
        profile: "release-fat-lto".to_owned(),
        static_artifact_digest: DIGEST_A.to_owned(),
        dynamic_artifact_digest: DIGEST_B.to_owned(),
        harness_artifact_digest: DIGEST_A.to_owned(),
        mock_server_artifact_digest: DIGEST_A.to_owned(),
        inventory_digest: inventory.digest.clone(),
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
        environment: BTreeMap::from([("RUSTC_WRAPPER".to_owned(), "/usr/bin/sccache".to_owned())]),
        bootstrap_seed: seed,
        pair_schedule: schedule,
        identity_digest: String::new(),
    }
    .seal()
    .expect("identity is canonical")
}

fn input() -> (NormativeInventory, SimultaneousGateInput) {
    let inventory = inventory();
    let identity = identity(&inventory, 20260826);
    let cases = inventory
        .cases
        .iter()
        .map(|case| {
            let mut samples = Vec::new();
            for (pair, planned) in identity.pair_schedule.iter().enumerate() {
                for metric in &case.metrics {
                    let static_value = if metric.metric == "itl_p99" {
                        0.0
                    } else {
                        100.0 + (pair % 5) as f64 * 0.1
                    };
                    let ratio = 0.996 + (pair % 3) as f64 * 0.0005;
                    let dynamic_value = match metric.direction {
                        RatioDirection::DynamicOverStatic => static_value * ratio,
                        RatioDirection::StaticOverDynamic if static_value == 0.0 => 0.0,
                        RatioDirection::StaticOverDynamic => static_value / ratio,
                    };
                    for variant in planned.member_order {
                        samples.push(PairedSample {
                            scenario: case.scenario.clone(),
                            pair_id: planned.pair_id.clone(),
                            variant,
                            metric: metric.metric.clone(),
                            value: match variant {
                                Variant::Static => static_value,
                                Variant::Dynamic => dynamic_value,
                            },
                            unit: "ratio-source".to_owned(),
                            commit: COMMIT.to_owned(),
                            artifact_digest: match variant {
                                Variant::Static => DIGEST_A.to_owned(),
                                Variant::Dynamic => DIGEST_B.to_owned(),
                            },
                            experiment_identity_digest: identity.identity_digest.clone(),
                        });
                    }
                }
            }
            PairedCase {
                scenario: case.scenario.clone(),
                primary_metric: case.primary_metric.clone(),
                samples,
                invalidation_attempts: Vec::new(),
            }
        })
        .collect();
    let input = SimultaneousGateInput {
        experiment_identity: identity,
        cases,
    };
    (inventory, input)
}

#[test]
fn complete_authenticated_inventory_is_exact() {
    let (inventory, valid) = input();
    let report = evaluate_simultaneous_gate(
        &valid,
        &inventory,
        &inventory.digest,
        &SimultaneousGatePolicy::normative(),
    )
    .expect("complete authenticated input evaluates");
    assert_eq!(report.metric_reports.len(), 4);
    assert_eq!(report.inventory_digest, inventory.digest);
    assert_eq!(
        report.experiment_identity_digest,
        valid.experiment_identity.identity_digest
    );

    let mut missing_case = valid.clone();
    missing_case.cases.pop();
    assert!(
        evaluate_simultaneous_gate(
            &missing_case,
            &inventory,
            &inventory.digest,
            &SimultaneousGatePolicy::normative(),
        )
        .is_err()
    );

    let mut missing_metric = valid.clone();
    missing_metric.cases[0]
        .samples
        .retain(|sample| sample.metric != "ttft_p99");
    assert!(
        evaluate_simultaneous_gate(
            &missing_metric,
            &inventory,
            &inventory.digest,
            &SimultaneousGatePolicy::normative(),
        )
        .is_err()
    );

    let mut extra_metric = valid.clone();
    let mut extra = extra_metric.cases[0].samples[0].clone();
    extra.metric = "itl_p50".to_owned();
    extra_metric.cases[0].samples.push(extra);
    assert!(
        evaluate_simultaneous_gate(
            &extra_metric,
            &inventory,
            &inventory.digest,
            &SimultaneousGatePolicy::normative(),
        )
        .is_err()
    );

    let recomputed_subset = NormativeInventory::new(
        inventory.component.clone(),
        vec![inventory.cases[0].clone()],
    )
    .expect("attacker can make a self-consistent subset");
    assert!(
        evaluate_simultaneous_gate(
            &valid,
            &recomputed_subset,
            &inventory.digest,
            &SimultaneousGatePolicy::normative(),
        )
        .is_err()
    );

    let directory = tempfile::tempdir().expect("temporary fixture directory");
    let inventory_path = directory.path().join("recomputed-subset.json");
    let input_path = directory.path().join("samples.json");
    std::fs::write(
        &inventory_path,
        serde_json::to_vec(&recomputed_subset).expect("subset inventory serializes"),
    )
    .expect("subset inventory is written");
    std::fs::write(
        &input_path,
        serde_json::to_vec(&valid).expect("sample input serializes"),
    )
    .expect("sample input is written");
    let acceptance = std::process::Command::new(env!("CARGO_BIN_EXE_plugin_runtime_bench"))
        .args([
            "evaluate",
            inventory_path.to_str().expect("temporary path is UTF-8"),
            &inventory.digest,
            input_path.to_str().expect("temporary path is UTF-8"),
        ])
        .output()
        .expect("production acceptance seam executes");
    assert!(!acceptance.status.success());
    assert!(
        String::from_utf8_lossy(&acceptance.stderr).contains("independently bound expected digest")
    );
}

#[test]
fn every_sample_is_bound_to_identity_and_artifact_assignment() {
    let (inventory, valid) = input();
    for mutation in 0..5 {
        let mut changed = valid.clone();
        match mutation {
            0 => {
                changed.cases[1].samples[0].commit =
                    "1123456789abcdef0123456789abcdef01234567".to_owned()
            }
            1 => {
                changed.cases[0]
                    .samples
                    .iter_mut()
                    .find(|sample| sample.variant == Variant::Static)
                    .expect("fixture contains a static member")
                    .artifact_digest = DIGEST_B.to_owned()
            }
            2 => changed.cases[0].samples[0].experiment_identity_digest = DIGEST_A.to_owned(),
            3 => changed.experiment_identity.kernel = "linux-7".to_owned(),
            4 => std::mem::swap(
                &mut changed.experiment_identity.static_artifact_digest,
                &mut changed.experiment_identity.dynamic_artifact_digest,
            ),
            _ => unreachable!(),
        }
        assert!(
            evaluate_simultaneous_gate(
                &changed,
                &inventory,
                &inventory.digest,
                &SimultaneousGatePolicy::normative(),
            )
            .is_err(),
            "identity mutation {mutation} must reject"
        );
    }
}

#[test]
fn seeded_exact_schedule_is_retained_and_enforced() {
    let (inventory, valid) = input();
    let expected = balanced_pair_orders(20260826);
    assert_eq!(
        expected,
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
    assert_ne!(expected, balanced_pair_orders(20260827));

    let report = evaluate_simultaneous_gate(
        &valid,
        &inventory,
        &inventory.digest,
        &SimultaneousGatePolicy::normative(),
    )
    .expect("exact schedule evaluates");
    assert_eq!(report.bootstrap_seed, 20260826);
    assert_eq!(
        report.pair_schedule,
        valid.experiment_identity.pair_schedule
    );

    let mut balanced_but_wrong = valid;
    for case in &mut balanced_but_wrong.cases {
        for sample in &mut case.samples {
            if sample.pair_id == "pair-00" || sample.pair_id == "pair-03" {
                sample.variant = match sample.variant {
                    Variant::Static => Variant::Dynamic,
                    Variant::Dynamic => Variant::Static,
                };
                sample.artifact_digest = match sample.variant {
                    Variant::Static => DIGEST_A.to_owned(),
                    Variant::Dynamic => DIGEST_B.to_owned(),
                };
            }
        }
    }
    assert!(
        evaluate_simultaneous_gate(
            &balanced_but_wrong,
            &inventory,
            &inventory.digest,
            &SimultaneousGatePolicy::normative(),
        )
        .is_err()
    );
}

#[test]
fn nonconstant_joint_bootstrap_has_a_pinned_golden_vector() {
    let (inventory, input) = input();
    let report = evaluate_simultaneous_gate(
        &input,
        &inventory,
        &inventory.digest,
        &SimultaneousGatePolicy::normative(),
    )
    .expect("nonconstant correlated vector evaluates");
    assert!(report.passed);
    assert_eq!(
        report.metric_reports[0].positive_paired_ratios[..6],
        [
            0.996,
            0.9964999999999999,
            0.997,
            0.996,
            0.9964999999999998,
            0.997,
        ]
    );
    assert_eq!(
        report.metric_reports[0].lower_confidence_bound,
        0.996383333333333
    );
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

    let mut decorrelated = input;
    for metric in ["cpu_nanoseconds_per_successful_request", "itl_p99"] {
        for variant in [Variant::Static, Variant::Dynamic] {
            let first = decorrelated.cases[1]
                .samples
                .iter()
                .position(|sample| {
                    sample.pair_id == "pair-00"
                        && sample.metric == metric
                        && sample.variant == variant
                })
                .expect("first member exists");
            let second = decorrelated.cases[1]
                .samples
                .iter()
                .position(|sample| {
                    sample.pair_id == "pair-01"
                        && sample.metric == metric
                        && sample.variant == variant
                })
                .expect("second member exists");
            let first_value = decorrelated.cases[1].samples[first].value;
            decorrelated.cases[1].samples[first].value =
                decorrelated.cases[1].samples[second].value;
            decorrelated.cases[1].samples[second].value = first_value;
        }
    }
    let decorrelated_report = evaluate_simultaneous_gate(
        &decorrelated,
        &inventory,
        &inventory.digest,
        &SimultaneousGatePolicy::normative(),
    )
    .expect("same marginal vectors with different joint correlation evaluate");
    assert_eq!(
        report.metric_reports[0].positive_paired_ratios,
        decorrelated_report.metric_reports[0].positive_paired_ratios
    );
    assert_ne!(
        report.maximum_degradation_bootstrap_distribution,
        decorrelated_report.maximum_degradation_bootstrap_distribution,
        "case-wise independent resampling would miss the changed joint correlation"
    );

    let mut failing = decorrelated;
    for case in &mut failing.cases {
        let static_values = case
            .samples
            .iter()
            .filter(|sample| sample.variant == Variant::Static)
            .map(|sample| {
                (
                    (sample.pair_id.clone(), sample.metric.clone()),
                    sample.value,
                )
            })
            .collect::<BTreeMap<_, _>>();
        for sample in &mut case.samples {
            if sample.variant == Variant::Dynamic {
                let static_value = static_values[&(sample.pair_id.clone(), sample.metric.clone())];
                sample.value = match case.scenario.as_str() {
                    "case-a" if sample.metric == "successful_requests_per_second" => {
                        static_value * 0.98
                    }
                    _ => static_value / 0.98,
                };
            }
        }
    }
    let failure_report = evaluate_simultaneous_gate(
        &failing,
        &inventory,
        &inventory.digest,
        &SimultaneousGatePolicy::normative(),
    )
    .expect("a valid statistical failure is a report");
    assert!(!failure_report.is_invalid);
    assert!(!failure_report.passed);
}
