// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Env-gated live cellular heartbeat output.
//!
//! The NDJSON carries a **percentile projection**, not raw centroids: an empty
//! t-digest anchors `min = +inf`, which JSON cannot encode, and percentiles are the
//! directly comparable form for validating live→report convergence.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::rc::Rc;

use crate::cellular::{
    HeartbeatAccumulator, HeartbeatCounters, HeartbeatSaturation, MetricsHeartbeat, TDigest,
};
use crate::clock::Clock;
use crate::metrics_core::{PERCENTILES, RecordIngest};
use crate::timing::{PhaseBranchStats, PhaseConfig, PhaseObserver, PhaseStats};
use anyhow::{Context, Result};
use serde::Serialize;

const HEARTBEAT_LOG_ENV: &str = "AIPERF_CELLULAR_HEARTBEAT_LOG";
const HEARTBEAT_PROTOCOL_VERSION: u32 = 1;

/// Feeds one completed record's latency facts into a heartbeat accumulator, matching
/// the report's exact record-metric definitions (`metrics_core::accumulator`) so the
/// live sketch converges to the exact report. Valid (non-errored, non-cancelled)
/// records only. TTFT is the non-negative first-token gap, latency the whole request,
/// and ITL `(latency − ttft)/(osl − 1)` with authoritative
/// `osl = usage.completion_tokens` (else observed output+reasoning) for `osl ≥ 2` —
/// one value per request, matching the report's `inter_token_latency` distribution
/// rather than a per-token one. All milliseconds. Shared by the single-process live
/// lane and the cellular cell's final heartbeat.
pub(crate) fn observe_ingest(accumulator: &mut HeartbeatAccumulator, ingest: &RecordIngest) {
    if ingest.errored || ingest.canceled {
        return;
    }
    let ttft_ms = ingest
        .first_token_ns
        .map(|first| first - ingest.start_ns)
        .filter(|delta| *delta >= 0)
        .map(|delta| delta as f64 / 1e6);
    let latency_ms =
        (ingest.end_ns >= ingest.start_ns).then(|| (ingest.end_ns - ingest.start_ns) as f64 / 1e6);
    let osl = ingest
        .usage
        .completion_tokens
        .or_else(|| ingest.tokens.output_sequence_length());
    // `Option<f64>` yields 0 or 1 value: one mean-ITL sample per request.
    let inter_token_ms = match (ttft_ms, latency_ms, osl) {
        (Some(ttft), Some(latency), Some(osl)) if osl >= 2 => {
            Some((latency - ttft) / (osl - 1) as f64)
        }
        _ => None,
    };
    accumulator.observe(ttft_ms, inter_token_ms, latency_ms);
}

/// A JSON-safe percentile projection of one t-digest sketch.
#[derive(Serialize)]
struct SketchProjection {
    count: u64,
    min: Option<f64>,
    max: Option<f64>,
    /// Percentile band → value; empty when the sketch has no observations.
    percentiles: BTreeMap<u32, f64>,
}

impl SketchProjection {
    fn from_sketch(sketch: &TDigest) -> Self {
        // One clustering for the whole band, not one per percentile.
        let quantiles: Vec<f64> = PERCENTILES.iter().map(|&p| p as f64 / 100.0).collect();
        let percentiles = PERCENTILES
            .iter()
            .zip(sketch.quantiles(&quantiles))
            .filter_map(|(&percentile, value)| value.map(|value| (percentile, value)))
            .collect();
        Self {
            count: sketch.count(),
            min: sketch.min(),
            max: sketch.max(),
            percentiles,
        }
    }
}

/// One NDJSON heartbeat line: counters, saturation, and percentile projections.
#[derive(Serialize)]
struct HeartbeatEvent {
    protocol_version: u32,
    event: &'static str,
    observed_at_ns: i64,
    counters: HeartbeatCounters,
    saturation: HeartbeatSaturation,
    ttft_ms: SketchProjection,
    itl_ms: SketchProjection,
    latency_ms: SketchProjection,
}

/// The live heartbeat lane: an accumulator fed per-record and snapshotted on the
/// phase cadence into an NDJSON file.
pub(crate) struct HeartbeatLane {
    accumulator: RefCell<HeartbeatAccumulator>,
    writer: RefCell<BufWriter<File>>,
    clock: Rc<dyn Clock>,
    origin_ns: i64,
}

impl HeartbeatLane {
    /// Whether `AIPERF_CELLULAR_HEARTBEAT_LOG` names a non-empty path, i.e. whether
    /// [`Self::from_env`] would build a lane. Lets a caller decide (e.g. exact-fold
    /// gating) whether the per-record heartbeat consumer is active without paying the
    /// file-truncating construction.
    pub(crate) fn enabled_by_env() -> bool {
        std::env::var_os(HEARTBEAT_LOG_ENV).is_some_and(|path| !path.is_empty())
    }

    /// Builds the lane when `AIPERF_CELLULAR_HEARTBEAT_LOG` names a writable path,
    /// else `None`. Truncates the target so each run starts a fresh stream.
    pub(crate) fn from_env(clock: Rc<dyn Clock>, origin_ns: i64) -> Result<Option<Rc<Self>>> {
        let Some(path) = std::env::var_os(HEARTBEAT_LOG_ENV) else {
            return Ok(None);
        };
        if path.is_empty() {
            return Ok(None);
        }
        let path = PathBuf::from(path);
        let file = File::create(&path)
            .with_context(|| format!("create heartbeat log {}", path.display()))?;
        Ok(Some(Rc::new(Self {
            accumulator: RefCell::new(HeartbeatAccumulator::new()),
            writer: RefCell::new(BufWriter::new(file)),
            clock,
            origin_ns,
        })))
    }

    /// Feeds one completed record's latency facts into the live sketches.
    pub(crate) fn observe_record(&self, ingest: &RecordIngest) {
        observe_ingest(&mut self.accumulator.borrow_mut(), ingest);
    }

    /// Snapshots the sketches with the cadence's counters and writes one NDJSON
    /// line. Skips emission until at least one record has been observed (an empty
    /// heartbeat is uninformative and not JSON-representable as raw centroids).
    fn emit(&self, stats: &PhaseStats) {
        let counters = HeartbeatCounters {
            issued: stats.requests_sent,
            completed: stats.requests_completed,
            errored: stats.request_errors + stats.requests_cancelled,
        };
        let saturation = HeartbeatSaturation {
            in_flight: stats.in_flight_requests,
            concurrency_limit: 0,
        };
        let observed_at_ns = self.clock.now_ns().saturating_sub(self.origin_ns);
        let heartbeat = self
            .accumulator
            .borrow()
            .snapshot(observed_at_ns, counters, saturation);
        self.write(&heartbeat);
    }

    fn write(&self, heartbeat: &MetricsHeartbeat) {
        if heartbeat.latency_ms.is_empty()
            && heartbeat.ttft_ms.is_empty()
            && heartbeat.itl_ms.is_empty()
        {
            return;
        }
        let Some(mut line) = heartbeat_event_line(heartbeat) else {
            return;
        };
        line.push(b'\n');
        let mut writer = self.writer.borrow_mut();
        if let Err(error) = writer.write_all(&line).and_then(|()| writer.flush()) {
            tracing::warn!(error = %error, "failed to write cellular heartbeat line");
        }
    }
}

/// Serializes one [`MetricsHeartbeat`] into its NDJSON `metrics_heartbeat` line
/// (counters + saturation + percentile-projected TTFT/ITL/latency sketches), or
/// `None` on a serialization error. The full native-v2-level metric snapshot the
/// single-process lane writes and the cellular controller emits as its live
/// cross-cell aggregate, so both surfaces (and the k8s CR-status snapshot the
/// frontend patches from it) carry the identical shape. No trailing newline —
/// the caller frames the stream.
pub(crate) fn heartbeat_event_line(heartbeat: &MetricsHeartbeat) -> Option<Vec<u8>> {
    let event = HeartbeatEvent {
        protocol_version: HEARTBEAT_PROTOCOL_VERSION,
        event: "metrics_heartbeat",
        observed_at_ns: heartbeat.observed_at_ns,
        counters: heartbeat.counters,
        saturation: heartbeat.saturation,
        ttft_ms: SketchProjection::from_sketch(&heartbeat.ttft_ms),
        itl_ms: SketchProjection::from_sketch(&heartbeat.itl_ms),
        latency_ms: SketchProjection::from_sketch(&heartbeat.latency_ms),
    };
    match serde_json::to_vec(&event) {
        Ok(line) => Some(line),
        Err(error) => {
            tracing::warn!(error = %error, "failed to serialize cellular heartbeat");
            None
        }
    }
}

/// [`PhaseObserver`] that emits a heartbeat on every progress tick and at each
/// phase's completion (the last tick has the fullest sketches).
pub(crate) struct HeartbeatPhaseObserver {
    lane: Rc<HeartbeatLane>,
}

impl HeartbeatPhaseObserver {
    pub(crate) fn new(lane: Rc<HeartbeatLane>) -> Self {
        Self { lane }
    }
}

impl PhaseObserver for HeartbeatPhaseObserver {
    fn on_phase_start(&self, _config: &PhaseConfig, _stats: PhaseStats) {}

    fn on_progress(&self, stats: PhaseStats) {
        self.lane.emit(&stats);
    }

    fn on_sending_complete(&self, _stats: PhaseStats) {}

    fn on_phase_complete(&self, stats: PhaseStats, _branch_stats: Option<PhaseBranchStats>) {
        self.lane.emit(&stats);
    }
}

/// Fans one phase lifecycle out to several observers (e.g. the Python live sink and
/// the cellular heartbeat lane) so both consume the same cadence.
pub(crate) struct CompositePhaseObserver {
    observers: Vec<Rc<dyn PhaseObserver>>,
}

impl CompositePhaseObserver {
    /// Collapses a list into the cheapest observer: a no-op is left to the caller
    /// (an empty list should not reach here); a single observer is returned as-is;
    /// otherwise the fan-out wrapper.
    pub(crate) fn compose(mut observers: Vec<Rc<dyn PhaseObserver>>) -> Rc<dyn PhaseObserver> {
        if observers.len() == 1 {
            observers.remove(0)
        } else {
            Rc::new(Self { observers })
        }
    }
}

impl PhaseObserver for CompositePhaseObserver {
    fn on_phase_start(&self, config: &PhaseConfig, stats: PhaseStats) {
        for observer in &self.observers {
            observer.on_phase_start(config, stats.clone());
        }
    }

    fn on_progress(&self, stats: PhaseStats) {
        for observer in &self.observers {
            observer.on_progress(stats.clone());
        }
    }

    fn on_sending_complete(&self, stats: PhaseStats) {
        for observer in &self.observers {
            observer.on_sending_complete(stats.clone());
        }
    }

    fn on_phase_complete(&self, stats: PhaseStats, branch_stats: Option<PhaseBranchStats>) {
        for observer in &self.observers {
            observer.on_phase_complete(stats.clone(), branch_stats.clone());
        }
    }

    fn on_phases_complete(&self, stats: Vec<PhaseStats>) {
        for observer in &self.observers {
            observer.on_phases_complete(stats.clone());
        }
    }
}
