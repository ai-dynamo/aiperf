// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Env-gated single-process live cellular heartbeat lane.
//!
//! When `AIPERF_CELLULAR_HEARTBEAT_LOG` names a writable path, each completed
//! record's TTFT / ITL / latency feeds a [`HeartbeatAccumulator`], and the
//! phase-progress cadence writes a percentile-projected [`MetricsHeartbeat`] as one
//! NDJSON line to that file. Live percentiles are **sketch-derived**; the final
//! report stays exact from the S2 partitions (roadmap S3). This is the
//! single-process realization of the S3 lane — the same
//! [`HeartbeatAccumulator`]/`t-digest` the cellular controller aggregates across
//! cells over the transport in Phase 2.
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

use aiperf::cellular::{
    HeartbeatAccumulator, HeartbeatCounters, HeartbeatSaturation, MetricsHeartbeat, TDigest,
};
use aiperf::clock::Clock;
use aiperf::metrics_core::{PERCENTILES, RecordIngest};
use aiperf::timing::{PhaseBranchStats, PhaseConfig, PhaseObserver, PhaseStats};
use anyhow::{Context, Result};
use serde::Serialize;

const HEARTBEAT_LOG_ENV: &str = "AIPERF_CELLULAR_HEARTBEAT_LOG";
const HEARTBEAT_PROTOCOL_VERSION: u32 = 1;

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
        let mut percentiles = BTreeMap::new();
        if !sketch.is_empty() {
            for percentile in PERCENTILES {
                if let Some(value) = sketch.quantile(percentile as f64 / 100.0) {
                    percentiles.insert(percentile, value);
                }
            }
        }
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

    /// Feeds one completed record's latency facts into the live sketches. TTFT is
    /// the first-token gap and latency the whole request. ITL is the request's mean
    /// inter-token latency — one value per request, `sum(gaps)/n == (last−first)/n`
    /// — matching the report's per-request `inter_token_latency` distribution, so
    /// the live sketch converges to the exact report rather than a per-token
    /// distribution. All milliseconds.
    pub(crate) fn observe_record(&self, ingest: &RecordIngest) {
        let ttft_ms = ingest
            .first_token_ns
            .map(|first| (first - ingest.start_ns) as f64 / 1e6);
        let latency_ms = Some((ingest.end_ns - ingest.start_ns) as f64 / 1e6);
        let arrivals = &ingest.token_arrival_ns;
        let gap_count = arrivals.len().saturating_sub(1);
        // `Option<f64>` yields 0 or 1 value: one mean-ITL sample per request.
        let inter_token_ms = (gap_count > 0)
            .then(|| (arrivals[arrivals.len() - 1] - arrivals[0]) as f64 / 1e6 / gap_count as f64);
        self.accumulator
            .borrow_mut()
            .observe(ttft_ms, inter_token_ms, latency_ms);
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
        let mut writer = self.writer.borrow_mut();
        match serde_json::to_vec(&event) {
            Ok(mut line) => {
                line.push(b'\n');
                if let Err(error) = writer.write_all(&line).and_then(|()| writer.flush()) {
                    tracing::warn!(error = %error, "failed to write cellular heartbeat line");
                }
            }
            Err(error) => tracing::warn!(error = %error, "failed to serialize cellular heartbeat"),
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
