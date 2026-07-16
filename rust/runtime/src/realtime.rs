// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Periodic realtime-metrics log block for a running profiling phase.
//!
//! Ports agentx's `RecordsManager._report_realtime_metrics` +
//! `_render_realtime_block` (`src/aiperf/records/records_manager.py`), which logs
//! a compact `[realtime MM:SS profiling]` block every `--stats-interval` seconds
//! on non-dashboard UIs. The native single-process runner has no mesh/dashboard,
//! so the log block is the whole feature.
//!
//! # How it gets correct live percentiles
//! agentx assembles each completed record into per-metric accumulators as it
//! arrives, so it can percentile TTFT/ITL/latency live. This module does the
//! same: [`LiveMetricsProcessor`] is a [`TurnRecordProcessor`] the profiling
//! runtime invokes once per completed request; it snapshots that request's fully
//! assembled record (terminal status + token arrivals + usage — all present at
//! completion, see the scheduled dispatch path) and folds it into a persistent
//! [`MetricsAccumulator`]. The snapshot is non-consuming
//! ([`NativeMetricsObserver::snapshot_record`]), so the authoritative end-of-run
//! report is untouched. A [`Clock`]-driven [`realtime_reporter_loop`] summarizes
//! that accumulator on a fixed interval and logs the block at INFO — reaching the
//! console and `logs/aiperf.log` like the phase-lifecycle lines.
//!
//! Because the accumulator holds every completed record (not a degenerate
//! window), the latency/TTFT/ITL/throughput percentiles are meaningful mid-run;
//! only the headline requests-per-second is additionally shown as an
//! instantaneous delta since the previous tick.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use anyhow::Result;
use async_trait::async_trait;

use crate::clock::Clock;
use crate::metrics::NativeMetricsObserver;
use crate::metrics_core::{AccumulatorSummary, MetricsAccumulator, MetricsConfig, RecordIngest};
use crate::multiturn::IssuedCredit;
use crate::scheduled::{TurnDispatchOutcome, TurnRecordProcessor};

/// Default realtime interval in seconds when `AIPERF_STATS_INTERVAL` is unset.
/// Matches agentx's non-dashboard default (30s).
const DEFAULT_STATS_INTERVAL_SECS: f64 = 30.0;

/// The `AIPERF_STATS_INTERVAL` env var (seconds). `0` (or negative) disables the
/// realtime block entirely; mirrors agentx's `--stats-interval 0`.
const STATS_INTERVAL_ENV: &str = "AIPERF_STATS_INTERVAL";

/// Resolve the realtime tick interval in nanoseconds, or `None` when disabled
/// (`AIPERF_STATS_INTERVAL=0`). A malformed value falls back to the default.
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

/// Persistent live-metrics accumulator, shared (cheaply cloned `Rc`) between the
/// per-completion [`LiveMetricsProcessor`] writer and the [`realtime_reporter_loop`]
/// reader. Single-threaded (thread-per-core), so a `RefCell` suffices; the writer
/// borrows briefly per completion and the reader borrows per tick, never across
/// an await, so the borrows cannot overlap.
#[derive(Clone)]
pub struct LiveMetrics(Rc<RefCell<MetricsAccumulator>>);

impl LiveMetrics {
    /// Build a live accumulator with the phase's metrics configuration.
    pub fn new(config: MetricsConfig) -> Self {
        Self(Rc::new(RefCell::new(MetricsAccumulator::with_config(
            config,
        ))))
    }

    /// Fold one completed record into the accumulator.
    fn ingest(&self, record: &RecordIngest) {
        self.0.borrow_mut().process_record(record);
    }

    /// Summarize everything folded so far (non-consuming).
    fn summarize(&self) -> AccumulatorSummary {
        self.0.borrow().summarize()
    }
}

/// Per-completion record processor feeding [`LiveMetrics`]. Registered on the
/// profiling phase's scheduled runtime; the runtime calls [`Self::process`] once
/// per completed request, after `on_terminal`/token callbacks and
/// `record_response` have all fired for that request — so the snapshot is a
/// complete record.
pub struct LiveMetricsProcessor {
    observer: Rc<NativeMetricsObserver>,
    live: LiveMetrics,
    /// Monotonic record ordinal for the snapshot's request-index/session fields.
    ordinal: Cell<u64>,
}

impl LiveMetricsProcessor {
    /// Create a processor that folds each completed request into `live`.
    pub fn new(observer: Rc<NativeMetricsObserver>, live: LiveMetrics) -> Self {
        Self {
            observer,
            live,
            ordinal: Cell::new(0),
        }
    }
}

#[async_trait(?Send)]
impl TurnRecordProcessor for LiveMetricsProcessor {
    async fn process(&self, credit: &IssuedCredit, _outcome: &TurnDispatchOutcome) -> Result<()> {
        let ordinal = self.ordinal.get();
        self.ordinal.set(ordinal + 1);
        // `snapshot_record` clones (never removes) the just-finished request's
        // fully assembled facts; a `None` (unknown/incomplete uuid) is skipped.
        if let Some(record) = self.observer.snapshot_record(credit.turn.uuid, ordinal) {
            self.live.ingest(&record);
        }
        Ok(())
    }
}

/// Drive the realtime block for one profiling phase: every `interval_ns` (virtual
/// or wall time via the injected [`Clock`]) summarize the live accumulator,
/// render the block, and log it at INFO. Loops until the task is aborted (the
/// phase execution aborts its handle when the phase completes). `origin_ns` is the
/// phase start on the clock's timeline, for elapsed time and the rate delta.
pub async fn realtime_reporter_loop(
    clock: Rc<dyn Clock>,
    live: LiveMetrics,
    origin_ns: i64,
    interval_ns: i64,
) {
    // (completed, elapsed_s) at the previous emitted tick, for instantaneous rps.
    let mut prev: Option<(f64, f64)> = None;
    loop {
        clock.clone().sleep(interval_ns).await;
        let now = clock.now_ns();
        let elapsed_s = now.saturating_sub(origin_ns).max(0) as f64 / 1_000_000_000.0;
        let summary = live.summarize();
        let done = finite(&summary, "request_count").unwrap_or(0.0);
        if done <= 0.0 {
            continue; // Nothing finished yet; suppress the block (agentx parity).
        }
        if let Some(block) = render_realtime_block(&summary, elapsed_s, prev) {
            tracing::info!("{block}");
            prev = Some((done, elapsed_s));
        }
    }
}

/// Latency rows rendered from per-request distributions (label, metric tag).
const LATENCY_ROWS: &[(&str, &str)] = &[
    ("ttft", "time_to_first_token"),
    ("itl", "inter_token_latency"),
    ("e2e", "request_latency"),
];

/// Percentiles shown per row, matching the native console summary columns.
const PERCENTILES: &[u32] = &[50, 90, 99];

/// Render the compact realtime block, or `None` when nothing has completed. The
/// shape mirrors agentx's `_render_realtime_block` (a header, a counter row, then
/// one labeled percentile row per latency metric plus output-sequence-length).
pub fn render_realtime_block(
    summary: &AccumulatorSummary,
    elapsed_s: f64,
    prev: Option<(f64, f64)>,
) -> Option<String> {
    let done = finite(summary, "request_count").unwrap_or(0.0);
    if done <= 0.0 {
        return None;
    }
    let err = finite(summary, "error_request_count").unwrap_or(0.0);
    let ok = (done - err).max(0.0);

    let rps_avg = finite(summary, "request_throughput");
    let rps_avg_str = rps_avg.map_or_else(|| "-".to_owned(), |v| format!("{v:.1}"));
    let rps_str = match prev {
        Some((prev_done, prev_elapsed)) if elapsed_s > prev_elapsed => {
            format!("{:.1}", (done - prev_done) / (elapsed_s - prev_elapsed))
        }
        _ => rps_avg_str.clone(),
    };
    let tput_in = finite(summary, "input_token_throughput").map_or_else(|| "-".to_owned(), fmt_int);
    let tput_out =
        finite(summary, "output_token_throughput").map_or_else(|| "-".to_owned(), fmt_int);

    let mut lines = vec![
        format!("[realtime {} profiling]", fmt_elapsed(elapsed_s)),
        format!(
            "  rps={rps_str} (avg {rps_avg_str})  tput_in={tput_in}/s  tput_out={tput_out}/s  \
             done={} ok={} err={}",
            fmt_int(done),
            fmt_int(ok),
            fmt_int(err),
        ),
    ];
    for (label, tag) in LATENCY_ROWS {
        let cells: Vec<String> = PERCENTILES
            .iter()
            .map(|p| {
                format!(
                    "p{p}={}",
                    percentile(summary, tag, *p).map_or_else(|| "-".to_owned(), fmt_ms)
                )
            })
            .collect();
        lines.push(format!("  {label:<5} {}", cells.join("  ")));
    }
    let osl_cells: Vec<Option<f64>> = PERCENTILES
        .iter()
        .map(|p| percentile(summary, "output_sequence_length", *p))
        .collect();
    if osl_cells.iter().any(Option::is_some) {
        let cells: Vec<String> = PERCENTILES
            .iter()
            .zip(&osl_cells)
            .map(|(p, v)| format!("p{p}={}", v.map_or_else(|| "-".to_owned(), fmt_int)))
            .collect();
        lines.push(format!("  {:<5} {}  (tokens)", "osl", cells.join("  ")));
    }
    Some(lines.join("\n"))
}

/// The scalar/representative value of a metric, if finite.
fn finite(summary: &AccumulatorSummary, tag: &str) -> Option<f64> {
    summary.result_by_name(tag).and_then(|r| r.finite_value())
}

/// One percentile of a distribution metric, if present and finite.
fn percentile(summary: &AccumulatorSummary, tag: &str, p: u32) -> Option<f64> {
    summary
        .result_by_name(tag)
        .and_then(|r| r.distribution())
        .and_then(|d| d.percentiles.get(&p))
        .and_then(|v| v.as_f64())
        .filter(|v| v.is_finite())
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

/// Format a millisecond latency value as `<int>ms` with thousands separators.
fn fmt_ms(value: f64) -> String {
    format!("{}ms", fmt_int(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics_core::MetricTag;

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
    fn empty_run_renders_nothing() {
        let summary = AccumulatorSummary::new();
        assert!(render_realtime_block(&summary, 1.0, None).is_none());
    }

    #[test]
    fn block_has_header_counters_and_latency_rows() {
        let mut summary = AccumulatorSummary::new();
        summary.insert_finite(MetricTag::RequestCount, 641.0);
        summary.insert_finite(MetricTag::RequestThroughput, 13.1);
        summary.insert_finite(MetricTag::InputTokenThroughput, 1_097_271.0);
        summary.insert_finite(MetricTag::OutputTokenThroughput, 10_441.0);

        let block = render_realtime_block(&summary, 49.0, None).expect("non-empty run renders");
        assert!(block.starts_with("[realtime 00:49 profiling]"));
        assert!(block.contains("done=641 ok=641 err=0"));
        assert!(block.contains("tput_in=1,097,271/s"));
        assert!(block.contains("tput_out=10,441/s"));
        assert!(block.contains("\n  ttft "));
        assert!(block.contains("\n  itl "));
        assert!(block.contains("\n  e2e "));
    }

    #[test]
    fn rps_uses_delta_when_prev_present() {
        let mut summary = AccumulatorSummary::new();
        summary.insert_finite(MetricTag::RequestCount, 30.0);
        // 10 completions over 5s since the last tick -> 2.0 instantaneous rps.
        let block = render_realtime_block(&summary, 15.0, Some((20.0, 10.0))).unwrap();
        assert!(block.contains("rps=2.0 "), "block was: {block}");
    }
}
