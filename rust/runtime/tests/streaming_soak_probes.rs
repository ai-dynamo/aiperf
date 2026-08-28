// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Non-ignored tests for streaming soak process probes, configuration parsing,
//! and the budget high-water oracle.

#[allow(dead_code)]
#[path = "support/streaming_soak.rs"]
mod support;

use std::sync::{Mutex, MutexGuard, PoisonError};

use support::{MIB, SoakError};

/// Serializes the process-wide environment across the configuration tests.
///
/// `libtest` runs test functions as threads of one process, so the env-var
/// mutations below would otherwise race each other.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn lock_env() -> MutexGuard<'static, ()> {
    // A panicking env test still leaves the environment cleared by `clear_soak_env`,
    // so a poisoned lock carries no unsafe state to recover from.
    ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Remove every soak variable so each test starts from a known environment.
fn clear_soak_env() {
    for name in [
        "AIPERF_STREAM_SOAK_DIR",
        "AIPERF_STREAM_SOAK_GIB",
        "AIPERF_STREAM_SOAK_LOGICAL_HOURS",
        "AIPERF_STREAM_SOAK_SOURCE",
        "AIPERF_STREAM_SOAK_FORMAT",
        "AIPERF_STREAM_SOAK_OBJECT_CONCURRENCY",
        "AIPERF_STREAM_SOAK_FAULT_PERIOD",
    ] {
        // Safe under `ENV_LOCK`: no other test thread reads or writes these.
        unsafe { std::env::remove_var(name) };
    }
}

fn set_required_env(dir: &str, gib: &str, hours: &str) {
    unsafe {
        std::env::set_var("AIPERF_STREAM_SOAK_DIR", dir);
        std::env::set_var("AIPERF_STREAM_SOAK_GIB", gib);
        std::env::set_var("AIPERF_STREAM_SOAK_LOGICAL_HOURS", hours);
    }
}

#[test]
#[cfg(target_os = "linux")]
fn sample_process_reports_plausible_rss_and_descriptors() {
    let sample = support::sample_process().expect("procfs probe readable");

    assert!(
        sample.rss_bytes >= MIB,
        "resident size unrealistically small: {}",
        sample.rss_bytes
    );
    assert!(
        sample.rss_bytes <= 256 * 1024 * MIB,
        "resident size unrealistically large: {}",
        sample.rss_bytes
    );
    // Every process holds at least stdin, stdout, and stderr.
    assert!(
        sample.open_fds >= 3,
        "descriptor count unrealistically small: {}",
        sample.open_fds
    );
}

#[test]
#[cfg(target_os = "linux")]
fn peak_rss_is_at_least_the_sampled_rss() {
    // The sample is read first so that any growth from a concurrently running
    // test thread lands inside the later peak rather than outside it.
    let sample = support::sample_process().expect("procfs probe readable");
    let peak = support::peak_rss_bytes().expect("VmHWM readable");

    // `VmHWM` is refreshed at kernel accounting points rather than on every
    // page fault, so it can trail the current resident size by a small margin.
    const HIGH_WATER_LAG_TOLERANCE: u64 = MIB;
    assert!(
        peak + HIGH_WATER_LAG_TOLERANCE >= sample.rss_bytes,
        "peak {peak} below sampled {}",
        sample.rss_bytes
    );
}

#[test]
fn from_env_refuses_missing_required_variable() {
    let _guard = lock_env();
    clear_soak_env();

    assert!(matches!(
        support::SoakConfig::from_env(),
        Err(SoakError::MissingEnv { .. })
    ));
}

#[test]
fn from_env_refuses_zero_input_volume() {
    let _guard = lock_env();
    clear_soak_env();
    set_required_env("/tmp/aiperf-stream-soak-probe", "0", "1");

    let result = support::SoakConfig::from_env();
    clear_soak_env();
    assert!(matches!(result, Err(SoakError::InvalidEnv { .. })));
}

#[test]
fn from_env_refuses_relative_scratch_path() {
    let _guard = lock_env();
    clear_soak_env();
    set_required_env("relative/path", "1", "1");

    let result = support::SoakConfig::from_env();
    clear_soak_env();
    assert!(matches!(result, Err(SoakError::UnsafeScratch { .. })));
}

#[test]
fn from_env_accepts_required_variables_with_defaults() {
    let _guard = lock_env();
    clear_soak_env();
    set_required_env("/tmp/aiperf-stream-soak-probe", "4", "2");

    let config = support::SoakConfig::from_env();
    clear_soak_env();
    let config = config.expect("validated soak configuration");

    assert_eq!(config.input_gib.get(), 4);
    assert_eq!(config.logical_hours.get(), 2);
    assert_eq!(config.source_id, "hf_hub");
    assert_eq!(config.format_id, "baseten_trace");
    assert_eq!(config.object_concurrency, 8);
    assert_eq!(config.fault_period.get(), 997);
}

#[test]
fn drained_budget_reports_within_budget() {
    use aiperf_runtime::streaming::budget::{BudgetLimits, StreamingResourceBudget};

    let limits = BudgetLimits {
        max_items: 4,
        max_bytes: 1024,
    };
    let budget = StreamingResourceBudget::new(limits).expect("valid budget");
    let lease = budget.try_acquire(2, 512).expect("available capacity");
    drop(lease);

    let record = support::StateHighWater::from_snapshot(limits, budget.snapshot());

    assert_eq!(record.high_water_items, 2);
    assert_eq!(record.high_water_bytes, 512);
    assert!(record.is_within_budget());
}

#[test]
fn retained_charge_is_reported_as_a_leak() {
    use aiperf_runtime::streaming::budget::{BudgetLimits, StreamingResourceBudget};

    let limits = BudgetLimits {
        max_items: 4,
        max_bytes: 1024,
    };
    let budget = StreamingResourceBudget::new(limits).expect("valid budget");
    let lease = budget.try_acquire(2, 512).expect("available capacity");

    // The lease is still held, so the snapshot carries a residual charge.
    let record = support::StateHighWater::from_snapshot(limits, budget.snapshot());

    assert_eq!(record.residual_items, 2);
    assert_eq!(record.residual_bytes, 512);
    assert!(
        !record.is_within_budget(),
        "an outstanding charge must not report within budget"
    );
    drop(lease);
}
