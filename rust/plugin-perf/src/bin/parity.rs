// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `parity` — runs one AB/BA performance-parity experiment between a statically
//! linked comparator binary and a dynamically loading candidate binary.
//!
//! The gate this binary enforces is *zero loss*: after moving a capability
//! behind the plugin ABI, the candidate must retain the comparator's
//! performance. "Retain" is made precise by a one-sided 95% lower confidence
//! bound on the ratio `static / dynamic`, which must sit at or above 0.99,
//! computed from 30 retained pairs per orientation after 5 discarded warmups.
//!
//! Both binaries are digested before anything runs, and an experiment that
//! names the same binary twice is refused: comparing a build against itself
//! measures the rig, not the boundary.
//!
//! Exits 0 when the experiment clears the gate and 1 when it does not, so a
//! caller can branch on the process status without parsing the document.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use tracing::{debug, info, warn};

use aiperf_plugin_perf::experiment::{
    AttemptOutcome, Digest, ExperimentRunner, ExperimentSpec, PairOrder, PairSchedule,
};
use aiperf_plugin_perf::report::{
    PairedSample, ParityResult, side_coefficients_of_variation, to_paired_samples,
};
use aiperf_plugin_perf::stats::{
    BootstrapConfig, MINIMUM_BOOTSTRAP_RESAMPLES, MINIMUM_RETAINED_PAIRS, WARMUP_ITERATIONS,
    try_bootstrap_paired_max_degradation,
};

/// One AB/BA performance-parity experiment.
#[derive(Debug, Parser)]
#[command(name = "parity", about = "Native-plugin performance-parity gate")]
struct Args {
    /// Statically linked comparator binary.
    #[arg(long)]
    static_bin: PathBuf,

    /// Dynamically loading candidate binary.
    #[arg(long)]
    dynamic_bin: PathBuf,

    /// Metric under comparison, such as `ttft_p50` or `e2e_p50`.
    #[arg(long)]
    metric: String,

    /// Warmup pairs to run and discard.
    #[arg(long, default_value_t = WARMUP_ITERATIONS)]
    warmups: usize,

    /// Pairs to retain in each orientation.
    #[arg(long, default_value_t = MINIMUM_RETAINED_PAIRS)]
    pairs: usize,

    /// Path the result document is written to.
    #[arg(long)]
    output: PathBuf,

    /// Workspace lockfile both binaries were built from.
    #[arg(long)]
    cargo_lock: Option<PathBuf>,

    /// Bootstrap resamples to draw.
    #[arg(long, default_value_t = MINIMUM_BOOTSTRAP_RESAMPLES)]
    resamples: usize,

    /// Seed for the deterministic bootstrap.
    #[arg(long, default_value_t = 0x5EED_0000_0000_0001)]
    seed: u64,

    /// Allocations counted for the static build, when instrumentation supplied
    /// them.
    #[arg(long, default_value_t = 0)]
    static_allocations: i64,

    /// Allocations counted for the dynamic build, when instrumentation supplied
    /// them.
    #[arg(long, default_value_t = 0)]
    dynamic_allocations: i64,

    /// Arguments passed to both binaries on every invocation.
    #[arg(long = "bin-arg")]
    bin_args: Vec<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let result = run(&args)?;
    let document =
        serde_json::to_string_pretty(&result).context("serializing the parity result document")?;
    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating output directory {}", parent.display()))?;
    }
    fs::write(&args.output, document)
        .with_context(|| format!("writing parity result to {}", args.output.display()))?;

    if result.is_zero_loss {
        info!(
            experiment = %result.identity.experiment_id,
            lower_bound = result.bootstrap_lower_bound,
            "parity gate passed"
        );
        Ok(())
    } else {
        warn!(
            experiment = %result.identity.experiment_id,
            lower_bound = result.bootstrap_lower_bound,
            cv_static = result.cv_static,
            cv_dynamic = result.cv_dynamic,
            allocation_delta = result.allocation_delta,
            "parity gate failed"
        );
        std::process::exit(1);
    }
}

/// Freezes the experiment, runs it, and computes its verdict.
fn run(args: &Args) -> Result<ParityResult> {
    let static_digest = digest_file(&args.static_bin)?;
    let dynamic_digest = digest_file(&args.dynamic_bin)?;
    let cargo_lock_digest = match &args.cargo_lock {
        Some(path) => digest_file(path)?,
        None => [0u8; 32],
    };
    let harness_digest = match std::env::current_exe() {
        Ok(path) => digest_file(&path)?,
        // A harness that cannot name itself still measures correctly; the
        // result document records the absence rather than inventing a digest.
        Err(error) => {
            warn!(error = %error, "harness binary path is unavailable; recording a null digest");
            [0u8; 32]
        }
    };

    let spec = ExperimentSpec {
        static_binary_digest: static_digest,
        dynamic_binary_digest: dynamic_digest,
        cargo_lock_digest,
        harness_digest,
        cpu_model: read_cpu_model(),
        memory_topology: read_memory_topology(),
        rust_version: read_rust_version(),
        timestamp_utc: timestamp_utc(),
        metric: args.metric.clone(),
        warmups: args.warmups,
        retained_pairs: args.pairs,
    };
    let mut runner = ExperimentRunner::new(spec).context("freezing the parity experiment")?;
    let identity = runner.freeze_identity();
    info!(
        experiment = %identity.experiment_id,
        metric = %args.metric,
        warmups = args.warmups,
        pairs = args.pairs,
        "parity experiment frozen"
    );

    let warmup_schedule = PairSchedule::balanced(args.warmups);
    let mut warmup_pairs = Vec::with_capacity(args.warmups);
    for (index, order) in warmup_schedule.iter().enumerate() {
        let pair = measure_pair(args, *order, index as u32, true)?;
        debug!(pair = index, "discarded warmup pair");
        warmup_pairs.push(pair);
    }

    let mut retained_pairs = Vec::with_capacity(args.pairs);
    // The schedule is fixed at freeze time, so it is taken by value here and
    // the runner stays free to accept the pairs it produces.
    let retained_schedule = runner.schedule().clone();
    for (index, order) in retained_schedule.iter().enumerate() {
        let pair = measure_pair(args, *order, index as u32, false)?;
        // Every AB pair is followed by its BA counterpart in the same retained
        // slot, so the runner is fed both orientations together.
        let counter_order = counterpart(*order);
        let counterpart_pair = measure_pair(args, counter_order, index as u32, false)?;
        let outcome = runner.record_valid_pair(
            pair.static_value_ns as f64,
            pair.dynamic_value_ns as f64,
            counterpart_pair.dynamic_value_ns as f64,
            counterpart_pair.static_value_ns as f64,
        );
        if let AttemptOutcome::ImmediateFailure { reason } = outcome {
            bail!("parity experiment abandoned: {reason}");
        }
        retained_pairs.push(pair);
        retained_pairs.push(counterpart_pair);
    }

    let samples = to_paired_samples(&retained_pairs);
    let bootstrap = try_bootstrap_paired_max_degradation(
        &samples,
        &BootstrapConfig {
            resamples: args.resamples,
            confidence: 0.95,
            seed: args.seed,
        },
    )
    .context("bootstrapping the retention bound")?;

    let (cv_static, cv_dynamic) = side_coefficients_of_variation(&retained_pairs);
    let mut result = ParityResult {
        identity,
        warmup_pairs,
        retained_pairs,
        bootstrap_lower_bound: bootstrap.lower_bound,
        point_estimate: bootstrap.point_estimate,
        bootstrap_resamples: bootstrap.resamples,
        bootstrap_seed: args.seed,
        cv_static,
        cv_dynamic,
        allocation_delta: args
            .dynamic_allocations
            .saturating_sub(args.static_allocations),
        is_zero_loss: false,
    };
    result.is_zero_loss = result.evaluate_zero_loss();
    Ok(result)
}

/// The opposite orientation, so a retained slot covers both.
fn counterpart(order: PairOrder) -> PairOrder {
    if order.is_ab() {
        PairOrder::DynamicFirst
    } else {
        PairOrder::StaticFirst
    }
}

/// Runs both binaries once, in the orientation `order` specifies.
fn measure_pair(
    args: &Args,
    order: PairOrder,
    pair_index: u32,
    is_warmup: bool,
) -> Result<PairedSample> {
    let (static_value_ns, dynamic_value_ns) = if order.is_ab() {
        let s = measure_once(&args.static_bin, &args.bin_args)?;
        let d = measure_once(&args.dynamic_bin, &args.bin_args)?;
        (s, d)
    } else {
        let d = measure_once(&args.dynamic_bin, &args.bin_args)?;
        let s = measure_once(&args.static_bin, &args.bin_args)?;
        (s, d)
    };
    Ok(PairedSample {
        pair_index,
        is_ab: order.is_ab(),
        static_value_ns,
        dynamic_value_ns,
        metric: args.metric.clone(),
        is_warmup,
    })
}

/// Wall-clock nanoseconds one invocation took, start to exit.
///
/// A non-zero exit is a product error and aborts the experiment: a build that
/// cannot complete the workload has not measured it, and retrying until it
/// succeeds would hide the defect.
fn measure_once(binary: &Path, bin_args: &[String]) -> Result<u64> {
    let started = Instant::now();
    let status = Command::new(binary)
        .args(bin_args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("launching {}", binary.display()))?;
    let elapsed = started.elapsed();
    if !status.success() {
        bail!("{} exited with {status}", binary.display());
    }
    // A sub-nanosecond invocation is impossible on real hardware, but flooring
    // at 1 keeps the value a usable ratio denominator regardless.
    Ok(u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX).max(1))
}

/// BLAKE3 digest of a file's contents.
fn digest_file(path: &Path) -> Result<Digest> {
    let bytes =
        fs::read(path).with_context(|| format!("reading {} for digesting", path.display()))?;
    Ok(*blake3::hash(&bytes).as_bytes())
}

/// The CPU model string, or a placeholder when the platform does not expose one.
fn read_cpu_model() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim().to_owned())
        })
        .unwrap_or_else(|| "unknown-cpu".to_owned())
}

/// A short socket, NUMA-node, and capacity summary of the machine.
fn read_memory_topology() -> String {
    let total_kb = fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("MemTotal"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim().to_owned())
        })
        .unwrap_or_else(|| "unknown".to_owned());
    let numa_nodes = fs::read_dir("/sys/devices/system/node")
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| {
                    entry
                        .file_name()
                        .to_str()
                        .is_some_and(|name| name.starts_with("node"))
                })
                .count()
        })
        .unwrap_or(0);
    format!("numa_nodes={numa_nodes}, mem_total={total_kb}")
}

/// The toolchain version, or a placeholder when `rustc` is not on the path.
fn read_rust_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|text| text.trim().to_owned())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "unknown-toolchain".to_owned())
}

/// The current instant as RFC 3339 UTC.
fn timestamp_utc() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    // Seconds since the epoch is unambiguous and needs no calendar dependency;
    // the `Z` suffix marks it as UTC for a reader.
    format!("epoch-seconds:{now}Z")
}
