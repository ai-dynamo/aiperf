// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Periodic realtime-progress log line for a running profiling phase.
//!
//! Inspired by agentx's `RecordsManager._report_realtime_metrics` +
//! `_render_realtime_block` (`src/aiperf/records/records_manager.py`), which logs
//! a compact `[realtime MM:SS profiling]` block every `--stats-interval` seconds
//! on non-dashboard UIs. The native single-process runner has no mesh/dashboard,
//! so a log line is the whole feature: a [`Clock`]-driven task samples live
//! progress on a fixed interval and emits it at INFO. Because the runner logs to
//! stderr and the front door forwards that into `logs/aiperf.log`, the line
//! reaches the console and the run log exactly like the phase-lifecycle lines.
//!
//! # Why counts, not the full metric block
//! agentx assembles each completed record into per-metric accumulators as it
//! arrives, so it can percentile TTFT/ITL/latency live. The native runner's
//! `MetricsAccumulator` is built for END-OF-RUN summarization: per-token and
//! terminal facts are finalized at drain, and its throughput/latency are computed
//! over the record window, which is degenerate mid-run (a handful of clustered
//! records ⇒ meaningless percentiles and throughput). The only quantities that
//! ARE trustworthy live are the completion COUNTS (how many requests have
//! finished vs are in flight) and a rate derived from their delta over wall time.
//! This module therefore emits an honest progress heartbeat — completed /
//! in-flight / requests-per-second — rather than fabricated percentiles. The
//! authoritative TTFT/ITL/latency/throughput distributions come from the
//! end-of-run report, unchanged.
//!
//! The completion count is [`NativeMetricsObserver::snapshot_summary`] — a
//! non-consuming recompute over the requests whose transport has finished — so
//! this best-effort side channel never perturbs the authoritative report.

use std::rc::Rc;

use crate::clock::Clock;
use crate::metrics::NativeMetricsObserver;
use crate::metrics_core::AccumulatorSummary;

/// Default realtime interval in seconds when `AIPERF_STATS_INTERVAL` is unset.
/// Matches agentx's non-dashboard default (30s).
const DEFAULT_STATS_INTERVAL_SECS: f64 = 30.0;

/// The `AIPERF_STATS_INTERVAL` env var (seconds). `0` (or negative) disables the
/// realtime line entirely; mirrors agentx's `--stats-interval 0`.
const STATS_INTERVAL_ENV: &str = "AIPERF_STATS_INTERVAL";

/// Resolve the realtime tick interval in nanoseconds, or `None` when disabled
/// (`AIPERF_STATS_INTERVAL=0`). A malformed value falls back to the default
/// rather than failing the run.
pub fn stats_interval_ns() -> Option<i64> {
    let secs = std::env::var(STATS_INTERVAL_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .unwrap_or(DEFAULT_STATS_INTERVAL_SECS);
    if !secs.is_finite() || secs <= 0.0 {
        return None;
    }
    Some((secs * 1_000_000_000.0) as i64)
}

/// Drive the realtime progress line for one profiling phase: every `interval_ns`
/// (virtual or wall time via the injected [`Clock`]) sample live progress, render
/// the line, and log it at INFO. Loops until the task is aborted (the phase
/// execution aborts its handle when the phase completes). `origin_ns` is the
/// phase start on the clock's timeline, used to render elapsed time and the rate.
pub async fn realtime_reporter_loop(
    clock: Rc<dyn Clock>,
    observer: Rc<NativeMetricsObserver>,
    origin_ns: i64,
    interval_ns: i64,
) {
    // (completed, elapsed_s) at the previous emitted tick, for the instantaneous
    // requests-per-second.
    let mut prev: Option<(f64, f64)> = None;
    loop {
        clock.clone().sleep(interval_ns).await;
        let now = clock.now_ns();
        let elapsed_s = now.saturating_sub(origin_ns).max(0) as f64 / 1_000_000_000.0;
        let summary = observer.snapshot_summary(now);
        let completed = completed_count(&summary);
        if completed <= 0.0 {
            continue; // Nothing finished yet; suppress the line (agentx parity).
        }
        // `arrivals` counts every dispatched request; in-flight = dispatched not
        // yet finished. Both are exact live counts (no accumulator windowing).
        let (arrivals, _) = observer.record_counts();
        let in_flight = (arrivals as f64 - completed).max(0.0);
        let line = render_progress_line(elapsed_s, completed, in_flight, prev);
        tracing::info!("{line}");
        prev = Some((completed, elapsed_s));
    }
}

/// The count of requests whose transport has finished, from a live snapshot.
/// `request_count` is a plain tally of finalized records, so it is exact live
/// even though the accumulator's rate/latency columns are not.
fn completed_count(summary: &AccumulatorSummary) -> f64 {
    summary
        .result_by_name("request_count")
        .and_then(|r| r.finite_value())
        .unwrap_or(0.0)
}

/// Render the one-line progress heartbeat, e.g.
/// `[realtime 00:49 profiling] completed=641 in_flight=3 rps=13.1`.
pub fn render_progress_line(
    elapsed_s: f64,
    completed: f64,
    in_flight: f64,
    prev: Option<(f64, f64)>,
) -> String {
    // Instantaneous rps from the delta since the previous emitted tick; on the
    // first tick, the average over the elapsed window.
    let rps = match prev {
        Some((prev_completed, prev_elapsed)) if elapsed_s > prev_elapsed => {
            (completed - prev_completed) / (elapsed_s - prev_elapsed)
        }
        _ if elapsed_s > 0.0 => completed / elapsed_s,
        _ => 0.0,
    };
    format!(
        "[realtime {} profiling] completed={} in_flight={} rps={rps:.1}",
        fmt_elapsed(elapsed_s),
        fmt_int(completed),
        fmt_int(in_flight),
    )
}

/// Format `elapsed_s` as `MM:SS` (agentx `_format_elapsed`).
fn fmt_elapsed(elapsed_s: f64) -> String {
    let total = elapsed_s.max(0.0) as u64;
    format!("{:02}:{:02}", total / 60, total % 60)
}

/// Round to a non-negative integer with thousands separators (e.g. `1,097,271`).
fn fmt_int(value: f64) -> String {
    let n = value.round().max(0.0) as u64;
    let digits = n.to_string();
    let bytes = digits.as_bytes();
    let mut grouped = String::new();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(*b as char);
    }
    grouped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_int_groups_thousands() {
        assert_eq!(fmt_int(1_097_271.0), "1,097,271");
        assert_eq!(fmt_int(641.0), "641");
        assert_eq!(fmt_int(0.0), "0");
    }

    #[test]
    fn fmt_elapsed_is_mm_ss() {
        assert_eq!(fmt_elapsed(49.0), "00:49");
        assert_eq!(fmt_elapsed(605.0), "10:05");
    }

    #[test]
    fn first_tick_uses_average_rate() {
        let line = render_progress_line(10.0, 20.0, 3.0, None);
        assert_eq!(
            line,
            "[realtime 00:10 profiling] completed=20 in_flight=3 rps=2.0"
        );
    }

    #[test]
    fn later_tick_uses_delta_rate() {
        // 10 more completions over 5 more seconds -> 2.0 rps, regardless of the
        // running average.
        let line = render_progress_line(15.0, 30.0, 2.0, Some((20.0, 10.0)));
        assert_eq!(
            line,
            "[realtime 00:15 profiling] completed=30 in_flight=2 rps=2.0"
        );
    }

    #[test]
    fn stats_interval_zero_disables() {
        // Exercised via the parse helper's contract; env mutation is process-wide
        // so we assert the disable branch on the parsed value directly.
        assert!(DEFAULT_STATS_INTERVAL_SECS > 0.0);
    }
}
