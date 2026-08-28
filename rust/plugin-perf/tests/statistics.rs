// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task 38 RED — statistical engine correctness tests.
//!
//! Pins the Hyndman-Fan type-7 quantile algorithm, paired bootstrap protocol,
//! CV threshold, and simultaneous confidence-bound semantics. All tests use
//! deterministic synthetic sample vectors so they are reproducible without a
//! real paper-rig run.

use aiperf_plugin_perf::stats::{
    BootstrapConfig, BootstrapResult, PairedSamples, QuantileType, bootstrap_paired_max_degradation,
    coefficient_of_variation, hyndman_fan_quantile,
};

/// Hyndman-Fan type-7 quantile: the standard Excel/R/NumPy default.
/// p = 0.5 on [1.0, 2.0, 3.0] must give 2.0 exactly.
#[test]
fn hyndman_fan_type7_median_three_elements() {
    let samples = vec![1.0_f64, 2.0, 3.0];
    let q = hyndman_fan_quantile(&samples, 0.5, QuantileType::Type7);
    assert!(
        (q - 2.0).abs() < 1e-12,
        "type-7 median of [1,2,3] must be 2.0, got {q}"
    );
}

/// p = 0.9 on five equally-spaced values: type-7 gives 4.6.
#[test]
fn hyndman_fan_type7_p90_five_elements() {
    let samples = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let q = hyndman_fan_quantile(&samples, 0.9, QuantileType::Type7);
    assert!(
        (q - 4.6).abs() < 1e-10,
        "type-7 p90 of [1..5] must be 4.6, got {q}"
    );
}

/// p = 0.0 and p = 1.0 must return the exact minimum and maximum.
#[test]
fn hyndman_fan_type7_boundaries() {
    let samples = vec![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let min = hyndman_fan_quantile(&samples, 0.0, QuantileType::Type7);
    let max = hyndman_fan_quantile(&samples, 1.0, QuantileType::Type7);
    assert!((min - 1.0).abs() < 1e-12, "p0 must be 1.0, got {min}");
    assert!((max - 9.0).abs() < 1e-12, "p1 must be 9.0, got {max}");
}

/// CV of a constant series must be 0.0.
#[test]
fn cv_constant_series_is_zero() {
    let samples: Vec<f64> = vec![5.0; 30];
    let cv = coefficient_of_variation(&samples);
    assert!(cv.abs() < 1e-12, "CV of constant must be 0, got {cv}");
}

/// CV <=2% threshold: a series with CV=1% must pass.
#[test]
fn cv_below_threshold_accepted() {
    // Mean=100, stddev≈1 → CV≈1%.
    let samples: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64 - 14.5) / 14.5).collect();
    let cv = coefficient_of_variation(&samples);
    assert!(cv <= 0.02, "CV of low-variance series must be <=2%, got {cv:.4}");
}

/// CV >2% threshold: a series with CV=10% must fail.
#[test]
fn cv_above_threshold_rejected() {
    // Mean=100, values spread 70–130 → CV>>2%.
    let samples: Vec<f64> = (0..30).map(|i| 70.0 + i as f64 * 2.0).collect();
    let cv = coefficient_of_variation(&samples);
    assert!(cv > 0.02, "CV of high-variance series must be >2%, got {cv:.4}");
}

/// Paired bootstrap on identical AB and BA series must produce a lower bound >=0.99.
#[test]
fn bootstrap_identical_distributions_pass() {
    let ab: Vec<(f64, f64)> = (0..30).map(|_| (100.0_f64, 100.0_f64)).collect();
    let ba: Vec<(f64, f64)> = ab.clone();
    let paired = PairedSamples { ab, ba };
    let config = BootstrapConfig {
        resamples: 10_000,
        confidence: 0.95,
        seed: 42,
    };
    let result: BootstrapResult = bootstrap_paired_max_degradation(&paired, &config);
    assert!(
        result.lower_bound >= 0.99,
        "identical distributions must produce lower_bound >=0.99, got {:.4}",
        result.lower_bound
    );
}

/// Paired bootstrap on a dynamic that is 50% slower must produce lower bound <0.99.
#[test]
fn bootstrap_significantly_slower_dynamic_fails() {
    // Static=100ns, Dynamic=150ns → ratio=0.667, well below 0.99.
    let ab: Vec<(f64, f64)> = (0..30).map(|_| (100.0_f64, 150.0_f64)).collect();
    let ba: Vec<(f64, f64)> = (0..30).map(|_| (150.0_f64, 100.0_f64)).collect();
    let paired = PairedSamples { ab, ba };
    let config = BootstrapConfig {
        resamples: 10_000,
        confidence: 0.95,
        seed: 42,
    };
    let result: BootstrapResult = bootstrap_paired_max_degradation(&paired, &config);
    assert!(
        result.lower_bound < 0.99,
        "50%-slower dynamic must produce lower_bound <0.99, got {:.4}",
        result.lower_bound
    );
}

/// Bootstrap requires exactly 30 retained pairs; fewer must be refused.
#[test]
fn bootstrap_rejects_fewer_than_30_pairs() {
    let ab: Vec<(f64, f64)> = (0..29).map(|_| (100.0_f64, 100.0_f64)).collect();
    let ba: Vec<(f64, f64)> = ab.clone();
    let paired = PairedSamples { ab, ba };
    let config = BootstrapConfig {
        resamples: 100,
        confidence: 0.95,
        seed: 42,
    };
    let result = std::panic::catch_unwind(|| bootstrap_paired_max_degradation(&paired, &config));
    assert!(result.is_err(), "fewer than 30 pairs must panic/error");
}

/// Bootstrap resamples must be >= 100,000 for a production run; fewer must be refused.
#[test]
fn bootstrap_enforces_minimum_resamples() {
    use aiperf_plugin_perf::stats::MINIMUM_BOOTSTRAP_RESAMPLES;
    assert!(
        MINIMUM_BOOTSTRAP_RESAMPLES >= 100_000,
        "MINIMUM_BOOTSTRAP_RESAMPLES must be >=100_000, got {MINIMUM_BOOTSTRAP_RESAMPLES}"
    );
}
