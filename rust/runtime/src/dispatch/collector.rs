// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rustc_hash::FxHashMap;
use serde::Serialize;
use serde::ser::{SerializeMap, Serializer};
use std::fmt::{Display, Formatter, Result as FmtResult};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct TraceSimulationReport {
    pub request_counts: TraceRequestCounts,
    pub throughput: TraceThroughputStats,
    pub prefix_cache_reused_ratio: f64,
    pub first_admission_prefix_cache_reused_ratio: f64,
    pub latency: TraceLatencyStats,
    /// SLA-goodput statistics, or `None` when no SLA was supplied.
    pub goodput: Option<TraceGoodputStats>,
    /// Per-request records omitted from summary serialization.
    pub per_request: Vec<PerRequestRecord>,
}

#[derive(Debug, Clone)]
pub struct TraceRequestCounts {
    pub num_requests: usize,
    pub completed_requests: usize,
    pub total_input_tokens: usize,
    pub total_output_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct TraceThroughputStats {
    pub duration_ms: f64,
    pub wall_time_ms: f64,
    pub request_throughput_rps: f64,
    pub input_throughput_tok_s: f64,
    pub output_throughput_tok_s: f64,
    pub total_throughput_tok_s: f64,
    /// Time-integral of provisioned workers, including startup and drain, in
    /// worker-seconds.
    /// Multiply by GPUs-per-worker for GPU-seconds (/3600 for GPU-hours).
    /// Aggregated replay reports through `decode_worker_seconds`, leaving
    /// `prefill_worker_seconds` at 0.0.
    pub prefill_worker_seconds: f64,
    pub decode_worker_seconds: f64,
    /// GPUs per worker per role; 0 when unavailable.
    pub prefill_gpus_per_worker: usize,
    pub decode_gpus_per_worker: usize,
    /// Provisioned GPU-hours: Σ_role `worker_seconds × gpus_per_worker / 3600`.
    pub gpu_hours: f64,
}

/// Throughput restricted to completed requests that satisfy the SLA.
#[derive(Debug, Clone)]
pub struct TraceGoodputStats {
    /// Completed requests that satisfied the SLA.
    pub completed_requests: usize,
    /// Good requests per second, over the simulated `duration_s`.
    pub request_throughput_rps: f64,
    /// Output tokens from good requests per second, over `duration_s`.
    pub output_throughput_tok_s: f64,
}

#[derive(Debug, Clone, Default)]
pub struct TraceDistributionStats {
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub median_ms: f64,
    pub p75_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub std_ms: f64,
}

#[derive(Debug, Clone)]
pub struct TraceLatencyStats {
    pub ttft: TraceDistributionStats,
    pub ttst: TraceDistributionStats,
    pub tpot: TraceDistributionStats,
    pub itl: TraceInterTokenLatencyStats,
    pub e2e: TraceDistributionStats,
    pub output_token_throughput_per_user: TraceDistributionStats,
}

#[derive(Debug, Clone)]
pub struct TraceInterTokenLatencyStats {
    pub distribution: TraceDistributionStats,
    pub max_ms: f64,
}

impl TraceSimulationReport {
    pub fn with_wall_time_ms(mut self, wall_time_ms: f64) -> Self {
        self.throughput.wall_time_ms = wall_time_ms;
        self
    }

    pub fn processed_tokens(&self) -> usize {
        self.request_counts.total_input_tokens + self.request_counts.total_output_tokens
    }

    pub fn processed_tokens_per_s(&self) -> f64 {
        if self.throughput.wall_time_ms <= 0.0 {
            return 0.0;
        }
        self.processed_tokens() as f64 / self.throughput.wall_time_ms * 1000.0
    }

    pub fn processed_output_tokens_per_s(&self) -> f64 {
        if self.throughput.wall_time_ms <= 0.0 {
            return 0.0;
        }
        self.request_counts.total_output_tokens as f64 / self.throughput.wall_time_ms * 1000.0
    }
}

impl Display for TraceSimulationReport {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        writeln!(
            f,
            "  completed_requests: {}",
            self.request_counts.completed_requests
        )?;
        writeln!(
            f,
            "  request_throughput_rps: {:.6}",
            self.throughput.request_throughput_rps
        )?;
        writeln!(
            f,
            "  output_throughput_tok_s: {:.6}",
            self.throughput.output_throughput_tok_s
        )?;
        writeln!(
            f,
            "  total_input_tokens: {}",
            self.request_counts.total_input_tokens
        )?;
        writeln!(
            f,
            "  total_output_tokens: {}",
            self.request_counts.total_output_tokens
        )?;
        writeln!(
            f,
            "  processed_tokens_per_s: {:.6}",
            self.processed_tokens_per_s()
        )?;
        writeln!(
            f,
            "  processed_output_tokens_per_s: {:.6}",
            self.processed_output_tokens_per_s()
        )?;
        writeln!(f, "  mean_ttft_ms: {:.6}", self.latency.ttft.mean_ms)?;
        writeln!(f, "  mean_e2e_latency_ms: {:.6}", self.latency.e2e.mean_ms)?;
        writeln!(
            f,
            "  prefix_cache_reused_ratio: {:.6}",
            self.prefix_cache_reused_ratio
        )?;
        writeln!(
            f,
            "  first_admission_prefix_cache_reused_ratio: {:.6}",
            self.first_admission_prefix_cache_reused_ratio
        )?;
        write!(f, "  wall_time_ms: {:.6}", self.throughput.wall_time_ms)
    }
}

impl Serialize for TraceSimulationReport {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // `goodput_*` entries are conditional, so the map has no fixed length.
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("num_requests", &self.request_counts.num_requests)?;
        map.serialize_entry(
            "completed_requests",
            &self.request_counts.completed_requests,
        )?;
        map.serialize_entry(
            "total_input_tokens",
            &self.request_counts.total_input_tokens,
        )?;
        map.serialize_entry(
            "total_output_tokens",
            &self.request_counts.total_output_tokens,
        )?;
        map.serialize_entry("duration_ms", &self.throughput.duration_ms)?;
        map.serialize_entry("wall_time_ms", &self.throughput.wall_time_ms)?;
        map.serialize_entry(
            "request_throughput_rps",
            &self.throughput.request_throughput_rps,
        )?;
        map.serialize_entry(
            "input_throughput_tok_s",
            &self.throughput.input_throughput_tok_s,
        )?;
        map.serialize_entry(
            "output_throughput_tok_s",
            &self.throughput.output_throughput_tok_s,
        )?;
        map.serialize_entry(
            "total_throughput_tok_s",
            &self.throughput.total_throughput_tok_s,
        )?;
        map.serialize_entry(
            "prefill_worker_seconds",
            &self.throughput.prefill_worker_seconds,
        )?;
        map.serialize_entry(
            "decode_worker_seconds",
            &self.throughput.decode_worker_seconds,
        )?;
        map.serialize_entry(
            "prefill_gpus_per_worker",
            &self.throughput.prefill_gpus_per_worker,
        )?;
        map.serialize_entry(
            "decode_gpus_per_worker",
            &self.throughput.decode_gpus_per_worker,
        )?;
        map.serialize_entry("gpu_hours", &self.throughput.gpu_hours)?;
        if let Some(goodput) = &self.goodput {
            map.serialize_entry("goodput_completed_requests", &goodput.completed_requests)?;
            map.serialize_entry(
                "goodput_request_throughput_rps",
                &goodput.request_throughput_rps,
            )?;
            map.serialize_entry(
                "goodput_output_throughput_tok_s",
                &goodput.output_throughput_tok_s,
            )?;
        }
        map.serialize_entry("processed_tokens", &self.processed_tokens())?;
        map.serialize_entry("processed_tokens_per_s", &self.processed_tokens_per_s())?;
        map.serialize_entry(
            "processed_output_tokens_per_s",
            &self.processed_output_tokens_per_s(),
        )?;
        map.serialize_entry("prefix_cache_reused_ratio", &self.prefix_cache_reused_ratio)?;
        map.serialize_entry(
            "first_admission_prefix_cache_reused_ratio",
            &self.first_admission_prefix_cache_reused_ratio,
        )?;
        serialize_distribution(&mut map, "ttft", "_ms", &self.latency.ttft)?;
        serialize_distribution(&mut map, "ttst", "_ms", &self.latency.ttst)?;
        serialize_distribution(&mut map, "tpot", "_ms", &self.latency.tpot)?;
        serialize_distribution(&mut map, "itl", "_ms", &self.latency.itl.distribution)?;
        map.serialize_entry("max_itl_ms", &self.latency.itl.max_ms)?;
        serialize_distribution(&mut map, "e2e_latency", "_ms", &self.latency.e2e)?;
        serialize_distribution(
            &mut map,
            "output_token_throughput_per_user",
            "",
            &self.latency.output_token_throughput_per_user,
        )?;
        map.end()
    }
}

/// Serialize a distribution as `{stat}_{prefix}{suffix}` keys. Latency
/// distributions pass `suffix = "_ms"` (e.g. `mean_ttft_ms`); rate
/// distributions pass `suffix = ""` (e.g. `mean_output_token_throughput_per_user`).
fn serialize_distribution<S>(
    map: &mut S,
    prefix: &str,
    suffix: &str,
    stats: &TraceDistributionStats,
) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    map.serialize_entry(&format!("mean_{prefix}{suffix}"), &stats.mean_ms)?;
    map.serialize_entry(&format!("min_{prefix}{suffix}"), &stats.min_ms)?;
    map.serialize_entry(&format!("max_{prefix}{suffix}"), &stats.max_ms)?;
    map.serialize_entry(&format!("median_{prefix}{suffix}"), &stats.median_ms)?;
    map.serialize_entry(&format!("p75_{prefix}{suffix}"), &stats.p75_ms)?;
    map.serialize_entry(&format!("p90_{prefix}{suffix}"), &stats.p90_ms)?;
    map.serialize_entry(&format!("p95_{prefix}{suffix}"), &stats.p95_ms)?;
    map.serialize_entry(&format!("p99_{prefix}{suffix}"), &stats.p99_ms)?;
    map.serialize_entry(&format!("std_{prefix}{suffix}"), &stats.std_ms)?;
    Ok(())
}

#[derive(Debug)]
struct TraceRequestStats {
    arrival_time_ms: f64,
    first_admit_ms: Option<f64>,
    token_times_ms: Vec<f64>,
    input_length: usize,
    requested_output_length: usize,
    reused_input_tokens: usize,
    first_admission_reused_input_tokens: usize,
    /// Prefill worker index. In disaggregated mode, `None` denotes bypass.
    prefill_worker_idx: Option<usize>,
    decode_worker_idx: Option<usize>,
    session_id: Option<String>,
    turn_index: Option<usize>,
    // Retained without export detail so partial terminal streams never count as
    // completed.
    terminal_status: Option<ReplayTerminalStatus>,
    detail: Option<Box<PerRequestDetail>>,
}

#[derive(Debug, Default)]
struct PerRequestDetail {
    prefill_reused_input_tokens: Option<usize>,
    prefill_admit_ms: Option<f64>,
    source_held_ms: Option<f64>,
    destination_reserved_ms: Option<f64>,
    destination_activated_ms: Option<f64>,
    decode_admit_ms: Option<f64>,
    source_released_ms: Option<f64>,
    decode_reused_input_tokens: Option<usize>,
    prefill_route_overlap_tokens: Option<usize>,
    decode_route_overlap_tokens: Option<usize>,
    terminal_status: Option<ReplayTerminalStatus>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayTerminalStatus {
    Completed,
    Rejected,
    Canceled,
    Failed,
}

/// Flat record emitted by `--report-jsonl`.
#[derive(Debug, Clone, Serialize)]
pub struct PerRequestRecord {
    /// Session identifier, serialized first with `turn_index`.
    pub session_id: Option<String>,
    /// Zero-based turn index within `session_id`, when present.
    pub turn_index: Option<usize>,
    pub uuid: String,
    pub arrival_time_ms: f64,
    pub first_admit_ms: Option<f64>,
    pub first_token_ms: Option<f64>,
    pub last_token_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub ttst_ms: Option<f64>,
    pub e2e_latency_ms: Option<f64>,
    /// Inter-token latency for this request, in milliseconds. Matches
    /// AIPerf's `inter_token_latency` field — one scalar per request.
    pub itl_ms: Option<f64>,
    pub input_length: usize,
    /// Number of output tokens requested by the workload trace.
    pub requested_output_length: usize,
    /// Number of output-token observations recorded for the request.
    pub output_length: usize,
    pub reused_input_tokens: usize,
    pub prefill_worker_idx: Option<usize>,
    pub decode_worker_idx: Option<usize>,
    pub prefill_admit_ms: Option<f64>,
    pub source_held_ms: Option<f64>,
    pub destination_reserved_ms: Option<f64>,
    pub destination_activated_ms: Option<f64>,
    pub decode_admit_ms: Option<f64>,
    pub source_released_ms: Option<f64>,
    pub decode_reused_input_tokens: Option<usize>,
    pub prefill_route_overlap_tokens: Option<usize>,
    pub decode_route_overlap_tokens: Option<usize>,
    pub terminal_status: ReplayTerminalStatus,
}

/// SLA thresholds used to classify requests for goodput.
/// Only the thresholds that are set are checked, so an e2e-only SLA gates on
/// e2e and a ttft+itl SLA gates on both. All-`None` (the default) means "no
/// SLA", which suppresses goodput entirely.
#[derive(Debug, Clone, Copy, Default)]
pub struct SlaThresholds {
    pub ttft_ms: Option<f64>,
    pub itl_ms: Option<f64>,
    pub e2e_ms: Option<f64>,
}

impl SlaThresholds {
    pub(crate) fn is_set(&self) -> bool {
        self.ttft_ms.is_some() || self.itl_ms.is_some() || self.e2e_ms.is_some()
    }

    /// Whether a completed request satisfies the SLA. Each *set* threshold must
    /// hold; unset thresholds are ignored.
    ///
    /// - `ttft_ms`: time-to-first-token ≤ bound.
    /// - `e2e_ms`: end-to-end latency ≤ bound.
    /// - `itl_ms`: the per-request **average inter-token latency** ≤ bound,
    ///   computed the same way as aiperf / genai-perf:
    ///   `avg_itl = (e2e_ms − ttft_ms) / (output_length − 1)`. When
    ///   `output_length ≤ 1` there is no inter-token interval, so the ITL check
    ///   is skipped (treated as satisfied).
    fn is_good(&self, ttft_ms: f64, e2e_ms: f64, output_length: usize) -> bool {
        if let Some(bound) = self.e2e_ms
            && e2e_ms > bound
        {
            return false;
        }
        if let Some(bound) = self.ttft_ms
            && ttft_ms > bound
        {
            return false;
        }
        if let Some(bound) = self.itl_ms
            && output_length > 1
        {
            let avg_itl_ms = (e2e_ms - ttft_ms) / (output_length as f64 - 1.0);
            if avg_itl_ms > bound {
                return false;
            }
        }
        true
    }
}

/// Accumulates per-request measurement events received through
/// [`RequestObserver`] and produces a [`TraceSimulationReport`].
///
/// [`RequestObserver`]: super::sink::RequestObserver
#[derive(Debug, Default)]
pub struct TraceCollector {
    requests: FxHashMap<Uuid, TraceRequestStats>,
    // Disabled by default to avoid the record pass and allocation.
    capture_per_request: bool,
    sla: SlaThresholds,
    // Integrated over the simulation clock for variable worker counts.
    prefill_worker_seconds: f64,
    decode_worker_seconds: f64,
    // Fixed counts use `count × duration_s` instead of the accumulator.
    static_worker_count: Option<(usize, usize)>,
    prefill_gpus_per_worker: usize,
    decode_gpus_per_worker: usize,
}

impl TraceRequestStats {
    fn first_token_ms(&self) -> Option<f64> {
        self.token_times_ms.first().copied()
    }

    fn last_token_ms(&self) -> Option<f64> {
        self.token_times_ms.last().copied()
    }

    fn actual_output_length(&self) -> usize {
        self.token_times_ms.len()
    }

    fn mean_tpot_ms(&self) -> Option<f64> {
        let num_gaps = self.token_times_ms.len().saturating_sub(1);
        if num_gaps == 0 {
            return None;
        }

        let first_token_ms = self.first_token_ms()?;
        let last_token_ms = self.last_token_ms()?;
        Some((last_token_ms - first_token_ms).max(0.0) / num_gaps as f64)
    }

    fn itls_ms(&self) -> impl Iterator<Item = f64> + '_ {
        self.token_times_ms
            .windows(2)
            .map(|window| (window[1] - window[0]).max(0.0))
    }

    fn ttst_ms(&self) -> Option<f64> {
        let [first_token_ms, second_token_ms, ..] = self.token_times_ms.as_slice() else {
            return None;
        };
        Some((second_token_ms - first_token_ms).max(0.0))
    }
}

impl TraceCollector {
    /// Configure whether `finish()` builds per-request records.
    pub fn set_capture_per_request(&mut self, value: bool) {
        self.capture_per_request = value;
    }

    /// Set SLA thresholds used to classify goodput.
    pub(crate) fn set_sla_thresholds(&mut self, sla: SlaThresholds) {
        self.sla = sla;
    }

    /// Add provisioned worker-seconds for an elapsed interval, including
    /// starting and draining workers. Aggregated replay uses only `decode`.
    pub(crate) fn add_worker_seconds(&mut self, prefill: f64, decode: f64) {
        self.prefill_worker_seconds += prefill;
        self.decode_worker_seconds += decode;
    }

    /// Set fixed provisioned worker counts when no event loop integrates them.
    pub(crate) fn set_static_worker_count(&mut self, prefill: usize, decode: usize) {
        self.static_worker_count = Some((prefill, decode));
    }

    /// Set GPUs per worker for GPU-hour accounting.
    pub(crate) fn set_gpus_per_worker(&mut self, prefill: usize, decode: usize) {
        self.prefill_gpus_per_worker = prefill;
        self.decode_gpus_per_worker = decode;
    }

    pub fn on_arrival(
        &mut self,
        uuid: Uuid,
        arrival_time_ms: f64,
        input_length: usize,
        requested_output_length: usize,
    ) {
        self.requests.insert(
            uuid,
            TraceRequestStats {
                arrival_time_ms,
                first_admit_ms: None,
                token_times_ms: Vec::with_capacity(requested_output_length),
                input_length,
                requested_output_length,
                reused_input_tokens: 0,
                prefill_worker_idx: None,
                decode_worker_idx: None,
                session_id: None,
                turn_index: None,
                first_admission_reused_input_tokens: 0,
                terminal_status: None,
                detail: self
                    .capture_per_request
                    .then(|| Box::new(PerRequestDetail::default())),
            },
        );
    }

    /// Record that `uuid` was dispatched to `worker_idx` on the prefill pool
    /// (offline disagg replay only). Idempotent — subsequent calls are no-ops
    /// once a value is set, so the first dispatch wins. Aggregated replay does
    /// not call this; for those requests `prefill_worker_idx` stays `None`.
    pub(crate) fn on_prefill_assigned(&mut self, uuid: Uuid, worker_idx: usize) {
        if let Some(stats) = self.requests.get_mut(&uuid)
            && stats.prefill_worker_idx.is_none()
        {
            stats.prefill_worker_idx = Some(worker_idx);
        }
    }

    /// Record that `uuid` was dispatched to `worker_idx` on the decode pool
    /// (offline disagg replay), or to the only pool (aggregated replay).
    /// Idempotent.
    pub(crate) fn on_decode_assigned(&mut self, uuid: Uuid, worker_idx: usize) {
        if let Some(stats) = self.requests.get_mut(&uuid)
            && stats.decode_worker_idx.is_none()
        {
            stats.decode_worker_idx = Some(worker_idx);
        }
    }

    pub fn on_admit(&mut self, uuid: Uuid, admit_time_ms: f64, reused_input_tokens: usize) {
        if let Some(stats) = self.requests.get_mut(&uuid) {
            if stats.first_admit_ms.is_none() {
                stats.first_admission_reused_input_tokens = reused_input_tokens;
                stats.first_admit_ms = Some(admit_time_ms);
            }
            stats.reused_input_tokens = stats.reused_input_tokens.max(reused_input_tokens);
        }
    }

    pub(crate) fn on_prefill_admit(
        &mut self,
        uuid: Uuid,
        admit_time_ms: f64,
        reused_input_tokens: usize,
    ) {
        self.on_admit(uuid, admit_time_ms, reused_input_tokens);
        if let Some(detail) = self.detail_mut(uuid) {
            record_role_admit(
                &mut detail.prefill_admit_ms,
                &mut detail.prefill_reused_input_tokens,
                admit_time_ms,
                reused_input_tokens,
            );
        }
    }

    pub(crate) fn on_decode_admit(
        &mut self,
        uuid: Uuid,
        admit_time_ms: f64,
        reused_input_tokens: usize,
    ) {
        self.on_admit(uuid, admit_time_ms, reused_input_tokens);
        if let Some(detail) = self.detail_mut(uuid) {
            record_role_admit(
                &mut detail.decode_admit_ms,
                &mut detail.decode_reused_input_tokens,
                admit_time_ms,
                reused_input_tokens,
            );
        }
    }

    pub(crate) fn on_source_held(&mut self, uuid: Uuid, at_ms: f64) {
        if let Some(detail) = self.detail_mut(uuid) {
            detail.source_held_ms.get_or_insert(at_ms);
        }
    }

    pub(crate) fn on_destination_reserved(&mut self, uuid: Uuid, at_ms: f64) {
        if let Some(detail) = self.detail_mut(uuid) {
            detail.destination_reserved_ms.get_or_insert(at_ms);
        }
    }

    pub(crate) fn on_destination_activated(&mut self, uuid: Uuid, at_ms: f64) {
        if let Some(detail) = self.detail_mut(uuid) {
            detail.destination_activated_ms.get_or_insert(at_ms);
        }
    }

    pub(crate) fn on_source_released(&mut self, uuid: Uuid, at_ms: f64) {
        if let Some(detail) = self.detail_mut(uuid) {
            detail.source_released_ms.get_or_insert(at_ms);
        }
    }

    pub(crate) fn on_prefill_route_overlap(&mut self, uuid: Uuid, tokens: usize) {
        if let Some(detail) = self.detail_mut(uuid) {
            detail.prefill_route_overlap_tokens.get_or_insert(tokens);
        }
    }

    pub(crate) fn on_decode_route_overlap(&mut self, uuid: Uuid, tokens: usize) {
        if let Some(detail) = self.detail_mut(uuid) {
            detail.decode_route_overlap_tokens.get_or_insert(tokens);
        }
    }

    pub fn on_terminal(&mut self, uuid: Uuid, status: ReplayTerminalStatus) {
        if let Some(stats) = self.requests.get_mut(&uuid) {
            stats.terminal_status.get_or_insert(status);
            if let Some(detail) = stats.detail.as_deref_mut() {
                detail.terminal_status.get_or_insert(status);
            }
        }
    }

    fn detail_mut(&mut self, uuid: Uuid) -> Option<&mut PerRequestDetail> {
        if !self.capture_per_request {
            return None;
        }
        self.requests.get_mut(&uuid)?.detail.as_deref_mut()
    }

    pub fn on_token(&mut self, uuid: Uuid, token_time_ms: f64) {
        if let Some(stats) = self.requests.get_mut(&uuid)
            && stats.terminal_status.is_none()
        {
            stats.token_times_ms.push(token_time_ms);
        }
    }

    pub fn finish(self) -> TraceSimulationReport {
        // Build records before moving requests into aggregation. This avoids the
        // record pass and allocation unless `--report-jsonl` requested them.
        let per_request = if self.capture_per_request {
            self.per_request_records()
        } else {
            Vec::new()
        };
        let sla = self.sla;
        let static_worker_count = self.static_worker_count;
        let accumulated_prefill_worker_seconds = self.prefill_worker_seconds;
        let accumulated_decode_worker_seconds = self.decode_worker_seconds;
        let prefill_gpus_per_worker = self.prefill_gpus_per_worker;
        let decode_gpus_per_worker = self.decode_gpus_per_worker;
        let requests = self.requests;
        let request_count = requests.len();

        // Float accumulation order affects report values.
        let agg = accumulate_requests(&requests, sla);

        let duration_s = (agg.duration_ms / 1000.0).max(1e-9);
        let resources = derive_resource_stats(
            &agg,
            duration_s,
            static_worker_count,
            accumulated_prefill_worker_seconds,
            accumulated_decode_worker_seconds,
            prefill_gpus_per_worker,
            decode_gpus_per_worker,
            sla,
        );

        let itl_distribution = build_distribution_stats(agg.itls);
        TraceSimulationReport {
            request_counts: TraceRequestCounts {
                num_requests: request_count,
                completed_requests: agg.completed_requests,
                total_input_tokens: agg.total_input_tokens,
                total_output_tokens: agg.total_output_tokens,
            },
            throughput: TraceThroughputStats {
                duration_ms: agg.duration_ms,
                wall_time_ms: 0.0,
                request_throughput_rps: agg.completed_requests as f64 / duration_s,
                input_throughput_tok_s: agg.total_input_tokens as f64 / duration_s,
                output_throughput_tok_s: agg.total_output_tokens as f64 / duration_s,
                total_throughput_tok_s: (agg.total_input_tokens + agg.total_output_tokens) as f64
                    / duration_s,
                prefill_worker_seconds: resources.prefill_worker_seconds,
                decode_worker_seconds: resources.decode_worker_seconds,
                prefill_gpus_per_worker,
                decode_gpus_per_worker,
                gpu_hours: resources.gpu_hours,
            },
            prefix_cache_reused_ratio: if agg.total_input_tokens == 0 {
                0.0
            } else {
                agg.total_reused_tokens as f64 / agg.total_input_tokens as f64
            },
            first_admission_prefix_cache_reused_ratio: if agg.total_input_tokens == 0 {
                0.0
            } else {
                agg.total_first_admission_reused_tokens as f64 / agg.total_input_tokens as f64
            },
            latency: TraceLatencyStats {
                ttft: build_distribution_stats(agg.ttfts),
                ttst: build_distribution_stats(agg.ttsts),
                tpot: build_distribution_stats(agg.tpots),
                itl: TraceInterTokenLatencyStats {
                    max_ms: itl_distribution.max_ms,
                    distribution: itl_distribution,
                },
                e2e: build_distribution_stats(agg.e2e_latencies),
                output_token_throughput_per_user: build_distribution_stats(
                    agg.output_token_throughput_per_user,
                ),
            },
            goodput: resources.goodput,
            per_request,
        }
    }

    /// Flatten retained requests for `--report-jsonl`.
    ///
    /// Only requests with a terminal outcome are emitted. Requests truncated
    /// by a simulation-time cap have no terminal outcome and remain omitted.
    pub fn per_request_records(&self) -> Vec<PerRequestRecord> {
        let mut records = Vec::with_capacity(self.requests.len());
        for (uuid, stats) in &self.requests {
            let Some(detail) = stats.detail.as_deref() else {
                continue;
            };
            let Some(terminal_status) = stats.terminal_status else {
                continue;
            };
            let first_token_ms = stats.first_token_ms();
            let last_token_ms = stats.last_token_ms();
            records.push(PerRequestRecord {
                session_id: stats.session_id.clone(),
                turn_index: stats.turn_index,
                uuid: uuid.to_string(),
                arrival_time_ms: stats.arrival_time_ms,
                first_admit_ms: stats.first_admit_ms,
                first_token_ms,
                last_token_ms,
                ttft_ms: first_token_ms.map(|time| (time - stats.arrival_time_ms).max(0.0)),
                ttst_ms: stats.ttst_ms(),
                e2e_latency_ms: last_token_ms.map(|time| (time - stats.arrival_time_ms).max(0.0)),
                itl_ms: stats.mean_tpot_ms(),
                input_length: stats.input_length,
                requested_output_length: stats.requested_output_length,
                output_length: stats.actual_output_length(),
                reused_input_tokens: detail
                    .prefill_reused_input_tokens
                    .unwrap_or(stats.reused_input_tokens),
                prefill_worker_idx: stats.prefill_worker_idx,
                decode_worker_idx: stats.decode_worker_idx,
                prefill_admit_ms: detail.prefill_admit_ms,
                source_held_ms: detail.source_held_ms,
                destination_reserved_ms: detail.destination_reserved_ms,
                destination_activated_ms: detail.destination_activated_ms,
                decode_admit_ms: detail.decode_admit_ms,
                source_released_ms: detail.source_released_ms,
                decode_reused_input_tokens: detail.decode_reused_input_tokens,
                prefill_route_overlap_tokens: detail.prefill_route_overlap_tokens,
                decode_route_overlap_tokens: detail.decode_route_overlap_tokens,
                terminal_status,
            });
        }
        // Stable ordering makes JSONL reproducible across hash-map iteration.
        records.sort_by(|a, b| {
            a.arrival_time_ms
                .total_cmp(&b.arrival_time_ms)
                .then_with(|| a.uuid.cmp(&b.uuid))
        });
        records
    }
}

/// Summary series and totals produced by [`accumulate_requests`].
struct RequestAggregate {
    ttfts: Vec<f64>,
    ttsts: Vec<f64>,
    tpots: Vec<f64>,
    itls: Vec<f64>,
    e2e_latencies: Vec<f64>,
    output_token_throughput_per_user: Vec<f64>,
    duration_ms: f64,
    total_input_tokens: usize,
    total_output_tokens: usize,
    completed_requests: usize,
    total_reused_tokens: usize,
    total_first_admission_reused_tokens: usize,
    goodput_requests: usize,
    goodput_output_tokens: usize,
}

/// Accumulate requests without changing map iteration or floating-point sum order.
fn accumulate_requests(
    requests: &FxHashMap<Uuid, TraceRequestStats>,
    sla: SlaThresholds,
) -> RequestAggregate {
    let request_count = requests.len();
    let mut ttfts = Vec::with_capacity(request_count);
    let mut ttsts = Vec::with_capacity(request_count);
    let mut tpots = Vec::with_capacity(request_count);
    let mut itls = Vec::new();
    let mut e2e_latencies = Vec::with_capacity(request_count);
    let mut output_token_throughput_per_user = Vec::new();
    let mut duration_ms = 0.0_f64;
    let mut total_input_tokens = 0usize;
    let mut total_output_tokens = 0usize;
    let mut completed_requests = 0usize;
    let mut total_reused_tokens = 0usize;
    let mut total_first_admission_reused_tokens = 0usize;
    let mut goodput_requests = 0usize;
    let mut goodput_output_tokens = 0usize;

    for stats in requests.values() {
        if stats.terminal_status != Some(ReplayTerminalStatus::Completed) {
            continue;
        }
        if stats.first_admit_ms.is_none() {
            continue;
        }
        let Some(first_token_ms) = stats.first_token_ms() else {
            continue;
        };
        let Some(last_token_ms) = stats.last_token_ms() else {
            continue;
        };

        completed_requests += 1;
        total_input_tokens += stats.input_length;
        let output_length = stats.actual_output_length();
        total_output_tokens += output_length;
        total_reused_tokens += stats.reused_input_tokens;
        total_first_admission_reused_tokens += stats.first_admission_reused_input_tokens;
        duration_ms = duration_ms.max(last_token_ms);

        let ttft_ms = (first_token_ms - stats.arrival_time_ms).max(0.0);
        let e2e_ms = (last_token_ms - stats.arrival_time_ms).max(0.0);
        ttfts.push(ttft_ms);
        e2e_latencies.push(e2e_ms);

        if sla.is_set() && sla.is_good(ttft_ms, e2e_ms, output_length) {
            goodput_requests += 1;
            goodput_output_tokens += output_length;
        }

        if let Some(ttst_ms) = stats.ttst_ms() {
            ttsts.push(ttst_ms);
        }

        if let Some(tpot_ms) = stats.mean_tpot_ms() {
            tpots.push(tpot_ms);
            for itl_ms in stats.itls_ms() {
                if itl_ms > 0.0 {
                    output_token_throughput_per_user.push(1000.0 / itl_ms);
                }
                itls.push(itl_ms);
            }
        }
    }

    RequestAggregate {
        ttfts,
        ttsts,
        tpots,
        itls,
        e2e_latencies,
        output_token_throughput_per_user,
        duration_ms,
        total_input_tokens,
        total_output_tokens,
        completed_requests,
        total_reused_tokens,
        total_first_admission_reused_tokens,
        goodput_requests,
        goodput_output_tokens,
    }
}

struct ResourceStats {
    prefill_worker_seconds: f64,
    decode_worker_seconds: f64,
    gpu_hours: f64,
    goodput: Option<TraceGoodputStats>,
}

#[allow(clippy::too_many_arguments)]
fn derive_resource_stats(
    agg: &RequestAggregate,
    duration_s: f64,
    static_worker_count: Option<(usize, usize)>,
    accumulated_prefill_worker_seconds: f64,
    accumulated_decode_worker_seconds: f64,
    prefill_gpus_per_worker: usize,
    decode_gpus_per_worker: usize,
    sla: SlaThresholds,
) -> ResourceStats {
    let (prefill_worker_seconds, decode_worker_seconds) = match static_worker_count {
        Some((prefill, decode)) => (prefill as f64 * duration_s, decode as f64 * duration_s),
        None => (
            accumulated_prefill_worker_seconds,
            accumulated_decode_worker_seconds,
        ),
    };
    let gpu_hours = (prefill_worker_seconds * prefill_gpus_per_worker as f64
        + decode_worker_seconds * decode_gpus_per_worker as f64)
        / 3600.0;
    let goodput = sla.is_set().then(|| TraceGoodputStats {
        completed_requests: agg.goodput_requests,
        request_throughput_rps: agg.goodput_requests as f64 / duration_s,
        output_throughput_tok_s: agg.goodput_output_tokens as f64 / duration_s,
    });
    ResourceStats {
        prefill_worker_seconds,
        decode_worker_seconds,
        gpu_hours,
        goodput,
    }
}

fn record_role_admit(
    admit_ms: &mut Option<f64>,
    reused_input_tokens_slot: &mut Option<usize>,
    admit_time_ms: f64,
    reused_input_tokens: usize,
) {
    admit_ms.get_or_insert(admit_time_ms);
    *reused_input_tokens_slot = Some(
        reused_input_tokens_slot
            .unwrap_or_default()
            .max(reused_input_tokens),
    );
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn build_distribution_stats(mut values: Vec<f64>) -> TraceDistributionStats {
    if values.is_empty() {
        return TraceDistributionStats::default();
    }

    let min_ms = values
        .iter()
        .copied()
        .min_by(|left, right| left.total_cmp(right))
        .expect("non-empty values must have a minimum");
    let max_ms = values
        .iter()
        .copied()
        .max_by(|left, right| left.total_cmp(right))
        .expect("non-empty values must have a maximum");

    TraceDistributionStats {
        mean_ms: mean(&values),
        min_ms,
        max_ms,
        median_ms: percentile_in_place(&mut values, 50.0),
        p75_ms: percentile_in_place(&mut values, 75.0),
        p90_ms: percentile_in_place(&mut values, 90.0),
        p95_ms: percentile_in_place(&mut values, 95.0),
        p99_ms: percentile_in_place(&mut values, 99.0),
        std_ms: std_dev(&values),
    }
}

fn percentile_in_place(values: &mut [f64], percentile: f64) -> f64 {
    let rank = percentile_rank(values.len(), percentile);
    let (_, selected, _) = values.select_nth_unstable_by(rank, |left, right| left.total_cmp(right));
    *selected
}

/// Rounded zero-based index for `percentile` (0–100) over `len` elements.
///
/// `len` must be at least one.
pub fn percentile_rank(len: usize, percentile: f64) -> usize {
    let rank = ((len - 1) as f64 * percentile / 100.0).round() as usize;
    rank.min(len - 1)
}

fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = mean(values);
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_distribution_stats_sorted(values: &[f64]) -> TraceDistributionStats {
        if values.is_empty() {
            return TraceDistributionStats::default();
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|left, right| left.total_cmp(right));
        TraceDistributionStats {
            mean_ms: mean(values),
            min_ms: sorted[0],
            max_ms: *sorted.last().expect("sorted values must be non-empty"),
            median_ms: sorted[percentile_rank(sorted.len(), 50.0)],
            p75_ms: sorted[percentile_rank(sorted.len(), 75.0)],
            p90_ms: sorted[percentile_rank(sorted.len(), 90.0)],
            p95_ms: sorted[percentile_rank(sorted.len(), 95.0)],
            p99_ms: sorted[percentile_rank(sorted.len(), 99.0)],
            std_ms: std_dev(values),
        }
    }

    #[test]
    fn build_distribution_stats_matches_sorted_baseline() {
        let values = vec![
            0.0, 1.0, 1.0, 2.5, 4.0, 4.0, 7.25, 9.5, 15.0, 22.0, 22.0, 100.0,
        ];

        let expected = build_distribution_stats_sorted(&values);
        let actual = build_distribution_stats(values);

        assert_eq!(actual.mean_ms, expected.mean_ms);
        assert_eq!(actual.min_ms, expected.min_ms);
        assert_eq!(actual.max_ms, expected.max_ms);
        assert_eq!(actual.median_ms, expected.median_ms);
        assert_eq!(actual.p75_ms, expected.p75_ms);
        assert_eq!(actual.p90_ms, expected.p90_ms);
        assert_eq!(actual.p95_ms, expected.p95_ms);
        assert_eq!(actual.p99_ms, expected.p99_ms);
        assert_eq!(actual.std_ms, expected.std_ms);
    }

    #[test]
    fn per_request_disagg_record_populates_all_fields() {
        let mut collector = TraceCollector::default();
        collector.set_capture_per_request(true);
        let uuid = Uuid::from_u128(1);
        collector.on_arrival(uuid, 0.0, 100, 4);
        collector.on_prefill_route_overlap(uuid, 64);
        collector.on_prefill_admit(uuid, 5.0, 30);
        collector.on_source_held(uuid, 10.0);
        collector.on_destination_reserved(uuid, 12.0);
        collector.on_destination_activated(uuid, 20.0);
        collector.on_source_released(uuid, 21.0);
        collector.on_decode_route_overlap(uuid, 32);
        collector.on_decode_admit(uuid, 25.0, 40);
        collector.on_prefill_assigned(uuid, 2);
        collector.on_decode_assigned(uuid, 7);
        collector.on_token(uuid, 50.0);
        collector.on_token(uuid, 60.0);
        collector.on_token(uuid, 75.0);
        collector.on_token(uuid, 95.0);
        collector.on_terminal(uuid, ReplayTerminalStatus::Completed);

        let report = collector.finish();
        assert_eq!(report.per_request.len(), 1);
        let rec = &report.per_request[0];
        assert_eq!(rec.uuid, uuid.to_string());
        assert_eq!(rec.arrival_time_ms, 0.0);
        assert_eq!(rec.first_admit_ms, Some(5.0));
        assert_eq!(rec.first_token_ms, Some(50.0));
        assert_eq!(rec.last_token_ms, Some(95.0));
        assert_eq!(rec.ttft_ms, Some(50.0));
        assert_eq!(rec.ttst_ms, Some(10.0));
        assert_eq!(rec.e2e_latency_ms, Some(95.0));
        assert_eq!(rec.itl_ms, Some(15.0));
        assert_eq!(rec.input_length, 100);
        assert_eq!(rec.output_length, 4);
        assert_eq!(rec.reused_input_tokens, 30);
        assert_eq!(rec.prefill_worker_idx, Some(2));
        assert_eq!(rec.decode_worker_idx, Some(7));
        assert_eq!(rec.prefill_admit_ms, Some(5.0));
        assert_eq!(rec.source_held_ms, Some(10.0));
        assert_eq!(rec.destination_reserved_ms, Some(12.0));
        assert_eq!(rec.destination_activated_ms, Some(20.0));
        assert_eq!(rec.source_released_ms, Some(21.0));
        assert_eq!(rec.decode_admit_ms, Some(25.0));
        assert_eq!(rec.decode_reused_input_tokens, Some(40));
        assert_eq!(rec.prefill_route_overlap_tokens, Some(64));
        assert_eq!(rec.decode_route_overlap_tokens, Some(32));
        assert_eq!(rec.terminal_status, ReplayTerminalStatus::Completed);
    }

    #[test]
    fn per_request_bypass_leaves_prefill_worker_idx_none() {
        let mut collector = TraceCollector::default();
        collector.set_capture_per_request(true);
        let uuid = Uuid::from_u128(42);
        collector.on_arrival(uuid, 0.0, 100, 2);
        collector.on_admit(uuid, 5.0, 0);
        collector.on_decode_assigned(uuid, 1);
        collector.on_token(uuid, 30.0);
        collector.on_token(uuid, 45.0);
        collector.on_terminal(uuid, ReplayTerminalStatus::Completed);

        let report = collector.finish();
        assert_eq!(report.per_request.len(), 1);
        let rec = &report.per_request[0];
        assert!(
            rec.prefill_worker_idx.is_none(),
            "bypassed request must have prefill_worker_idx = None"
        );
        assert_eq!(rec.decode_worker_idx, Some(1));
    }

    #[test]
    fn per_request_default_off() {
        let mut collector = TraceCollector::default();
        let uuid = Uuid::from_u128(1);
        collector.on_arrival(uuid, 0.0, 100, 2);
        collector.on_admit(uuid, 5.0, 0);
        collector.on_decode_assigned(uuid, 0);
        collector.on_token(uuid, 50.0);
        collector.on_token(uuid, 60.0);
        collector.on_terminal(uuid, ReplayTerminalStatus::Completed);

        assert!(collector.requests[&uuid].detail.is_none());

        let report = collector.finish();
        assert!(report.per_request.is_empty());
        assert_eq!(report.request_counts.completed_requests, 1);
    }

    fn add_completed(
        collector: &mut TraceCollector,
        uuid_n: u128,
        arrival_ms: f64,
        output_length: usize,
        token_times_ms: &[f64],
    ) {
        let uuid = Uuid::from_u128(uuid_n);
        collector.on_arrival(uuid, arrival_ms, 100, output_length);
        collector.on_admit(uuid, arrival_ms, 0);
        collector.on_decode_assigned(uuid, 0);
        for &t in token_times_ms {
            collector.on_token(uuid, t);
        }
        collector.on_terminal(uuid, ReplayTerminalStatus::Completed);
    }

    #[test]
    fn goodput_classifies_by_aiperf_avg_itl() {
        let mut collector = TraceCollector::default();
        collector.set_sla_thresholds(SlaThresholds {
            ttft_ms: Some(150.0),
            itl_ms: Some(30.0),
            e2e_ms: None,
        });
        add_completed(&mut collector, 1, 0.0, 3, &[100.0, 150.0, 200.0]);
        add_completed(&mut collector, 2, 0.0, 3, &[100.0, 120.0, 140.0]);
        add_completed(&mut collector, 3, 0.0, 1, &[100.0]);

        let goodput = collector
            .finish()
            .goodput
            .expect("SLA set → goodput present");
        assert_eq!(goodput.completed_requests, 2);
        assert!((goodput.output_throughput_tok_s - 4.0 / 0.2).abs() < 1e-6);
        assert!((goodput.request_throughput_rps - 2.0 / 0.2).abs() < 1e-6);
    }

    #[test]
    fn goodput_itl_boundary_is_inclusive() {
        let sla = SlaThresholds {
            ttft_ms: None,
            itl_ms: Some(50.0),
            e2e_ms: None,
        };
        let mut at_bound = TraceCollector::default();
        at_bound.set_sla_thresholds(sla);
        add_completed(&mut at_bound, 1, 0.0, 3, &[100.0, 150.0, 200.0]);
        assert_eq!(at_bound.finish().goodput.unwrap().completed_requests, 1);
        let mut over = TraceCollector::default();
        over.set_sla_thresholds(sla);
        add_completed(&mut over, 1, 0.0, 3, &[100.0, 150.0, 201.0]);
        assert_eq!(over.finish().goodput.unwrap().completed_requests, 0);
    }

    #[test]
    fn goodput_e2e_only_sla() {
        let mut collector = TraceCollector::default();
        collector.set_sla_thresholds(SlaThresholds {
            ttft_ms: None,
            itl_ms: None,
            e2e_ms: Some(150.0),
        });
        add_completed(&mut collector, 1, 0.0, 2, &[100.0, 200.0]);
        add_completed(&mut collector, 2, 0.0, 2, &[60.0, 120.0]);
        assert_eq!(collector.finish().goodput.unwrap().completed_requests, 1);
    }

    #[test]
    fn goodput_absent_without_sla() {
        let mut collector = TraceCollector::default();
        add_completed(&mut collector, 1, 0.0, 2, &[10.0, 20.0]);
        assert!(collector.finish().goodput.is_none());
    }

    #[test]
    fn worker_seconds_accumulated_and_static() {
        let mut accumulated = TraceCollector::default();
        add_completed(&mut accumulated, 1, 0.0, 2, &[10.0, 20.0]);
        accumulated.add_worker_seconds(1.5, 4.0);
        accumulated.add_worker_seconds(0.5, 1.0);
        let report = accumulated.finish();
        assert!((report.throughput.prefill_worker_seconds - 2.0).abs() < 1e-9);
        assert!((report.throughput.decode_worker_seconds - 5.0).abs() < 1e-9);

        let mut static_single = TraceCollector::default();
        static_single.set_static_worker_count(0, 1);
        add_completed(&mut static_single, 1, 0.0, 2, &[100.0, 200.0]);
        let report = static_single.finish();
        assert!(report.throughput.prefill_worker_seconds.abs() < 1e-9);
        assert!((report.throughput.decode_worker_seconds - 0.2).abs() < 1e-9);
    }

    #[test]
    fn gpu_hours_from_worker_seconds_and_gpus_per_worker() {
        let mut collector = TraceCollector::default();
        collector.set_gpus_per_worker(2, 4);
        add_completed(&mut collector, 1, 0.0, 2, &[100.0, 200.0]);
        collector.add_worker_seconds(10.0, 5.0);
        let report = collector.finish();
        assert_eq!(report.throughput.prefill_gpus_per_worker, 2);
        assert_eq!(report.throughput.decode_gpus_per_worker, 4);
        assert!((report.throughput.gpu_hours - 40.0 / 3600.0).abs() < 1e-9);
    }

    #[test]
    fn per_request_records_are_sorted_by_arrival_time() {
        let mut collector = TraceCollector::default();
        collector.set_capture_per_request(true);
        for (uuid_n, arrival) in [(3u128, 30.0), (1, 0.0), (2, 10.0)] {
            let uuid = Uuid::from_u128(uuid_n);
            collector.on_arrival(uuid, arrival, 100, 1);
            collector.on_admit(uuid, arrival + 1.0, 0);
            collector.on_decode_assigned(uuid, 0);
            collector.on_token(uuid, arrival + 5.0);
            collector.on_terminal(uuid, ReplayTerminalStatus::Completed);
        }
        let report = collector.finish();
        let arrivals: Vec<f64> = report
            .per_request
            .iter()
            .map(|r| r.arrival_time_ms)
            .collect();
        assert_eq!(arrivals, vec![0.0, 10.0, 30.0]);
    }

    #[test]
    fn per_request_record_serializes_to_json_object() {
        let mut collector = TraceCollector::default();
        collector.set_capture_per_request(true);
        let uuid = Uuid::from_u128(123);
        collector.on_arrival(uuid, 0.0, 50, 2);
        collector.on_admit(uuid, 1.0, 10);
        collector.on_prefill_assigned(uuid, 0);
        collector.on_decode_assigned(uuid, 1);
        collector.on_token(uuid, 20.0);
        collector.on_token(uuid, 25.0);
        collector.on_terminal(uuid, ReplayTerminalStatus::Completed);

        let report = collector.finish();
        let line = serde_json::to_string(&report.per_request[0])
            .expect("PerRequestRecord must serialize cleanly");
        let parsed: serde_json::Value =
            serde_json::from_str(&line).expect("emitted JSON must parse");
        assert!(parsed.is_object());
        assert_eq!(parsed["uuid"], uuid.to_string());
        assert_eq!(parsed["input_length"], 50);
        assert_eq!(parsed["output_length"], 2);
        assert_eq!(parsed["prefill_worker_idx"], 0);
        assert_eq!(parsed["decode_worker_idx"], 1);
        assert!(parsed["itl_ms"].is_number());
        assert_eq!(parsed["terminal_status"], "completed");
    }

    #[test]
    fn terminal_failures_emit_nullable_latencies_and_unfinished_requests_are_omitted() {
        let mut collector = TraceCollector::default();
        collector.set_capture_per_request(true);
        for (uuid_n, status) in [
            (1, ReplayTerminalStatus::Rejected),
            (2, ReplayTerminalStatus::Canceled),
            (3, ReplayTerminalStatus::Failed),
        ] {
            let uuid = Uuid::from_u128(uuid_n);
            collector.on_arrival(uuid, uuid_n as f64, 64, 2);
            collector.on_terminal(uuid, status);
        }
        collector.on_arrival(Uuid::from_u128(4), 4.0, 64, 2);

        let report = collector.finish();

        assert_eq!(report.per_request.len(), 3);
        assert_eq!(
            report
                .per_request
                .iter()
                .map(|record| record.terminal_status)
                .collect::<Vec<_>>(),
            vec![
                ReplayTerminalStatus::Rejected,
                ReplayTerminalStatus::Canceled,
                ReplayTerminalStatus::Failed,
            ]
        );
        assert!(report.per_request.iter().all(|record| {
            record.first_admit_ms.is_none()
                && record.first_token_ms.is_none()
                && record.last_token_ms.is_none()
                && record.ttft_ms.is_none()
                && record.e2e_latency_ms.is_none()
        }));
    }

    #[test]
    fn cancellation_freezes_tokens_and_excludes_partial_stream_from_completion() {
        let uuid = Uuid::from_u128(91);
        let mut collector = TraceCollector::default();
        collector.set_capture_per_request(true);
        collector.on_arrival(uuid, 0.0, 64, 4);
        collector.on_admit(uuid, 1.0, 0);
        collector.on_token(uuid, 10.0);
        collector.on_terminal(uuid, ReplayTerminalStatus::Canceled);
        collector.on_token(uuid, 20.0);

        let report = collector.finish();

        assert_eq!(report.request_counts.num_requests, 1);
        assert_eq!(report.request_counts.completed_requests, 0);
        assert_eq!(report.request_counts.total_output_tokens, 0);
        assert_eq!(report.per_request.len(), 1);
        assert_eq!(
            report.per_request[0].terminal_status,
            ReplayTerminalStatus::Canceled
        );
        assert_eq!(report.per_request[0].output_length, 1);
    }

    #[test]
    fn first_admission_reuse_ignores_later_readmission_self_reuse() {
        let uuid = Uuid::from_u128(1);
        let mut collector = TraceCollector::default();
        collector.on_arrival(uuid, 0.0, 100, 1);
        collector.on_admit(uuid, 1.0, 0);
        collector.on_admit(uuid, 2.0, 80);
        collector.on_token(uuid, 3.0);
        collector.on_terminal(uuid, ReplayTerminalStatus::Completed);

        let report = collector.finish();

        assert_eq!(report.prefix_cache_reused_ratio, 0.8);
        assert_eq!(report.first_admission_prefix_cache_reused_ratio, 0.0);
    }
}
