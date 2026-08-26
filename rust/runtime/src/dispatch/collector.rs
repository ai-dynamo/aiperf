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
    /// Compatibility-only prefill worker-seconds; the current runtime has no
    /// disaggregated producer, so this is always zero.
    pub prefill_worker_seconds: f64,
    /// Compatibility-only decode worker-seconds; the current runtime has no
    /// disaggregated producer, so this is always zero.
    pub decode_worker_seconds: f64,
    /// Compatibility-only prefill GPU count; always zero without a
    /// disaggregated producer.
    pub prefill_gpus_per_worker: usize,
    /// Compatibility-only decode GPU count; always zero without a
    /// disaggregated producer.
    pub decode_gpus_per_worker: usize,
    /// Compatibility-only GPU-hours; always zero without a disaggregated
    /// producer.
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
    session_id: Option<String>,
    turn_index: Option<usize>,
    // Retained without export detail so partial terminal streams never count as
    // completed.
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

/// Flat record emitted when `artifacts.per_request_jsonl` is configured.
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
    /// Compatibility-only prefill-worker assignment; always absent because
    /// the current runtime has no disaggregated producer.
    pub prefill_worker_idx: Option<usize>,
    /// Compatibility-only decode-worker assignment; always absent because
    /// the current runtime has no disaggregated producer.
    pub decode_worker_idx: Option<usize>,
    /// Compatibility-only prefill admission time; always absent.
    pub prefill_admit_ms: Option<f64>,
    /// Compatibility-only source-held time; always absent.
    pub source_held_ms: Option<f64>,
    /// Compatibility-only destination-reserved time; always absent.
    pub destination_reserved_ms: Option<f64>,
    /// Compatibility-only destination-activated time; always absent.
    pub destination_activated_ms: Option<f64>,
    /// Compatibility-only decode admission time; always absent.
    pub decode_admit_ms: Option<f64>,
    /// Compatibility-only source-released time; always absent.
    pub source_released_ms: Option<f64>,
    /// Compatibility-only decode reused-input count; always absent.
    pub decode_reused_input_tokens: Option<usize>,
    /// Compatibility-only prefill route-overlap count; always absent.
    pub prefill_route_overlap_tokens: Option<usize>,
    /// Compatibility-only decode route-overlap count; always absent.
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
    #[allow(dead_code)]
    pub(crate) fn set_sla_thresholds(&mut self, sla: SlaThresholds) {
        self.sla = sla;
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
                session_id: None,
                turn_index: None,
                first_admission_reused_input_tokens: 0,
                terminal_status: None,
            },
        );
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

    pub fn on_terminal(&mut self, uuid: Uuid, status: ReplayTerminalStatus) {
        if let Some(stats) = self.requests.get_mut(&uuid) {
            stats.terminal_status.get_or_insert(status);
        }
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
        // record pass and allocation unless per-request capture requested them.
        let per_request = if self.capture_per_request {
            self.per_request_records()
        } else {
            Vec::new()
        };
        let sla = self.sla;
        let requests = self.requests;
        let request_count = requests.len();

        // Float accumulation order affects report values.
        let agg = accumulate_requests(&requests, sla);

        let duration_s = (agg.duration_ms / 1000.0).max(1e-9);
        let goodput = derive_goodput(&agg, duration_s, sla);

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
                prefill_worker_seconds: 0.0,
                decode_worker_seconds: 0.0,
                prefill_gpus_per_worker: 0,
                decode_gpus_per_worker: 0,
                gpu_hours: 0.0,
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
            goodput,
            per_request,
        }
    }

    /// Flatten retained requests for the per-request JSONL artifact.
    ///
    /// Only requests with a terminal outcome are emitted. Requests truncated
    /// by a simulation-time cap have no terminal outcome and remain omitted.
    pub fn per_request_records(&self) -> Vec<PerRequestRecord> {
        if !self.capture_per_request {
            return Vec::new();
        }
        let mut records = Vec::with_capacity(self.requests.len());
        for (uuid, stats) in &self.requests {
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
                reused_input_tokens: stats.reused_input_tokens,
                prefill_worker_idx: None,
                decode_worker_idx: None,
                prefill_admit_ms: None,
                source_held_ms: None,
                destination_reserved_ms: None,
                destination_activated_ms: None,
                decode_admit_ms: None,
                source_released_ms: None,
                decode_reused_input_tokens: None,
                prefill_route_overlap_tokens: None,
                decode_route_overlap_tokens: None,
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

fn derive_goodput(
    agg: &RequestAggregate,
    duration_s: f64,
    sla: SlaThresholds,
) -> Option<TraceGoodputStats> {
    sla.is_set().then(|| TraceGoodputStats {
        completed_requests: agg.goodput_requests,
        request_throughput_rps: agg.goodput_requests as f64 / duration_s,
        output_throughput_tok_s: agg.goodput_output_tokens as f64 / duration_s,
    })
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
    fn per_request_default_off() {
        let mut collector = TraceCollector::default();
        let uuid = Uuid::from_u128(1);
        collector.on_arrival(uuid, 0.0, 100, 2);
        collector.on_admit(uuid, 5.0, 0);
        collector.on_token(uuid, 50.0);
        collector.on_token(uuid, 60.0);
        collector.on_terminal(uuid, ReplayTerminalStatus::Completed);

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
    fn per_request_records_are_sorted_by_arrival_time() {
        let mut collector = TraceCollector::default();
        collector.set_capture_per_request(true);
        for (uuid_n, arrival) in [(3u128, 30.0), (1, 0.0), (2, 10.0)] {
            let uuid = Uuid::from_u128(uuid_n);
            collector.on_arrival(uuid, arrival, 100, 1);
            collector.on_admit(uuid, arrival + 1.0, 0);
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
        assert!(parsed["prefill_worker_idx"].is_null());
        assert!(parsed["decode_worker_idx"].is_null());
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
