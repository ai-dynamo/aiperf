// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Online aggregation of content-server transfers into media-fetch metrics.
//!
//! Each completed [`ContentRequestRecord`] is joined to its originating request
//! via the `rid`/`mi`/`td` tag carried in the served query string, yielding a
//! per-fetch [`MediaRecord`] streamed to the artifact and folded into six
//! distributions. Records that carry no parseable tag are counted and logged,
//! never silently dropped. The aggregator holds only compact numeric samples,
//! not the records themselves, so it scales with request volume.

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;

use crate::content_server::model::ContentRequestRecord;
use crate::content_server::parse_media_tag;
use crate::metrics_core::kernel::linear_distribution;
use crate::metrics_core::sidecar::{SidecarSeries, SidecarStats};
use crate::metrics_core::{SidecarMetric, Unit};

/// Metric name constants (verbatim from the spec).
pub const TIME_TO_MEDIA_FETCH: &str = "time_to_media_fetch";
pub const MEDIA_SERVING_LATENCY: &str = "media_serving_latency";
pub const MEDIA_TIME_TO_FIRST_BYTE: &str = "media_time_to_first_byte";
pub const MEDIA_TRANSFER_DURATION: &str = "media_transfer_duration";
pub const MEDIA_BYTES_SERVED: &str = "media_bytes_served";
pub const MEDIA_FETCH_COUNT: &str = "media_fetch_count";

/// One content-server fetch joined to the request/slot that carried it. Streamed
/// to the `media_records` artifact, one JSON object per line.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MediaRecord {
    /// Originating request id (`X-Request-ID`).
    pub rid: String,
    /// Zero-based media slot ordinal within the originating turn.
    pub mi: u32,
    /// Served path (`/content/...`).
    pub path: String,
    /// HTTP response status.
    pub status: u16,
    /// Body bytes served.
    pub bytes: u64,
    /// Dispatch to content-server arrival (clamped to 0 on clock skew).
    pub time_to_media_fetch_ns: i64,
    /// Content-server arrival through terminal body.
    pub serving_latency_ns: u64,
    /// Arrival until response status/headers.
    pub time_to_first_byte_ns: u64,
    /// Arrival until first non-empty body chunk.
    pub time_to_first_body_byte_ns: u64,
    /// First through last non-empty body chunk.
    pub transfer_duration_ns: u64,
}

/// Finalized media-fetch metrics and drain accounting.
#[derive(Debug, Default)]
pub struct MediaMetricsSummary {
    /// Metric name to its single-series gauge distribution.
    pub metrics: BTreeMap<String, SidecarMetric>,
    /// Fetches successfully joined to a request.
    pub total_fetches: u64,
    /// Records excluded because they carried no parseable tag.
    pub unmatched: u64,
    /// Fetches whose negative `time_to_media_fetch` was clamped to 0.
    pub negative_ttmf: u64,
}

/// Folds content-server records into media-fetch distributions online.
#[derive(Debug, Default)]
pub struct MediaFetchAggregator {
    time_to_fetch: Vec<f64>,
    serving_latency: Vec<f64>,
    time_to_first_byte: Vec<f64>,
    transfer_duration: Vec<f64>,
    per_request_count: HashMap<String, u32>,
    per_request_bytes: HashMap<String, u64>,
    total_fetches: u64,
    unmatched: u64,
    negative_ttmf: u64,
}

impl MediaFetchAggregator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest one served record. Returns the joined [`MediaRecord`] for the
    /// artifact, or `None` when the record carried no parseable tag.
    pub fn ingest(&mut self, record: &ContentRequestRecord) -> Option<MediaRecord> {
        let Some(tag) = parse_media_tag(&record.query_string) else {
            self.unmatched += 1;
            tracing::warn!(
                path = %record.path,
                query = %record.query_string,
                "content-server record has no media tag; excluded from media metrics"
            );
            return None;
        };
        self.total_fetches += 1;

        let mut time_to_media_fetch_ns =
            record.timestamp_ns as i64 - tag.dispatch_wall_ns as i64;
        if time_to_media_fetch_ns < 0 {
            self.negative_ttmf += 1;
            tracing::debug!(
                rid = %tag.rid,
                mi = tag.mi,
                raw = time_to_media_fetch_ns,
                "negative time_to_media_fetch clamped to 0 (clock skew)"
            );
            time_to_media_fetch_ns = 0;
        }

        self.time_to_fetch.push(time_to_media_fetch_ns as f64);
        self.serving_latency.push(record.latency_ns as f64);
        self.time_to_first_byte.push(record.time_to_first_byte_ns as f64);
        self.transfer_duration.push(record.transfer_duration_ns as f64);
        *self.per_request_count.entry(tag.rid.clone()).or_insert(0) += 1;
        *self.per_request_bytes.entry(tag.rid.clone()).or_insert(0) += record.body_bytes;

        Some(MediaRecord {
            rid: tag.rid,
            mi: tag.mi,
            path: record.path.clone(),
            status: record.status_code,
            bytes: record.body_bytes,
            time_to_media_fetch_ns,
            serving_latency_ns: record.latency_ns,
            time_to_first_byte_ns: record.time_to_first_byte_ns,
            time_to_first_body_byte_ns: record.time_to_first_body_byte_ns,
            transfer_duration_ns: record.transfer_duration_ns,
        })
    }

    /// Build the six distributions. Per-fetch metrics use one sample per fetch;
    /// `media_bytes_served` and `media_fetch_count` roll up to one sample per
    /// request.
    pub fn finish(self) -> MediaMetricsSummary {
        let per_request_bytes: Vec<f64> =
            self.per_request_bytes.values().map(|&b| b as f64).collect();
        let per_request_count: Vec<f64> =
            self.per_request_count.values().map(|&c| c as f64).collect();

        let mut metrics = BTreeMap::new();
        let series = [
            (TIME_TO_MEDIA_FETCH, Unit::Nanosecond, self.time_to_fetch),
            (MEDIA_SERVING_LATENCY, Unit::Nanosecond, self.serving_latency),
            (MEDIA_TIME_TO_FIRST_BYTE, Unit::Nanosecond, self.time_to_first_byte),
            (MEDIA_TRANSFER_DURATION, Unit::Nanosecond, self.transfer_duration),
            (MEDIA_BYTES_SERVED, Unit::Byte, per_request_bytes),
            (MEDIA_FETCH_COUNT, Unit::Count, per_request_count),
        ];
        for (name, unit, samples) in series {
            if let Some(metric) = gauge_metric(name, unit, samples) {
                metrics.insert(name.to_string(), metric);
            }
        }

        MediaMetricsSummary {
            metrics,
            total_fetches: self.total_fetches,
            unmatched: self.unmatched,
            negative_ttmf: self.negative_ttmf,
        }
    }
}

/// Build a single-series gauge `SidecarMetric` from samples, or `None` if empty.
fn gauge_metric(tag: &str, unit: Unit, samples: Vec<f64>) -> Option<SidecarMetric> {
    // ddof=1 matches the sample standard deviation telemetry series use.
    let stats = linear_distribution(tag, samples, f64::NAN, 1)?;
    let series = SidecarSeries {
        labels: None,
        endpoint_url: None,
        stats: SidecarStats::Gauge(stats),
        timeslices: Vec::new(),
    };
    Some(SidecarMetric::new(Some(unit), vec![series]))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn record(query: &str, arrival_ns: u64, latency_ns: u64, bytes: u64) -> ContentRequestRecord {
        ContentRequestRecord {
            timestamp_ns: arrival_ns,
            method: "GET".into(),
            path: "/content/images/x.png".into(),
            query_string: query.into(),
            http_version: "1.1".into(),
            client_host: String::new(),
            client_port: 0,
            request_headers: BTreeMap::new(),
            status_code: 200,
            content_type: "image/png".into(),
            response_headers: BTreeMap::new(),
            body_bytes: bytes,
            body_chunk_count: 1,
            latency_ns,
            time_to_first_byte_ns: latency_ns / 2,
            time_to_first_body_byte_ns: latency_ns / 2,
            transfer_duration_ns: latency_ns / 4,
            error: None,
        }
    }

    fn gauge(summary: &MediaMetricsSummary, name: &str) -> DistributionSnapshot {
        let metric = summary.metrics.get(name).expect("metric present");
        match &metric.series[0].stats {
            SidecarStats::Gauge(stats) => DistributionSnapshot {
                avg: stats.avg.as_f64().unwrap(),
                sum: stats.sum.as_f64().unwrap(),
                count: stats.count,
            },
            other => panic!("expected gauge, got {other:?}"),
        }
    }

    struct DistributionSnapshot {
        avg: f64,
        sum: f64,
        count: usize,
    }

    #[test]
    fn joins_records_and_rolls_up_per_request() {
        let mut agg = MediaFetchAggregator::new();
        // rid=A carries two media (mi 0,1); rid=B one.
        agg.ingest(&record("rid=A&mi=0&td=1000", 1300, 40, 100));
        agg.ingest(&record("rid=A&mi=1&td=1000", 1500, 60, 200));
        agg.ingest(&record("rid=B&mi=0&td=2000", 2400, 80, 500));
        // Missing tag -> excluded, counted.
        agg.ingest(&record("garbage=1", 9999, 10, 10));

        let summary = agg.finish();
        assert_eq!(summary.total_fetches, 3);
        assert_eq!(summary.unmatched, 1);

        // time_to_media_fetch per fetch: 300, 500, 400 -> avg 400, 3 samples.
        let ttmf = gauge(&summary, TIME_TO_MEDIA_FETCH);
        assert_eq!(ttmf.count, 3);
        assert!((ttmf.avg - 400.0).abs() < 1e-9);

        // serving latency per fetch: 40, 60, 80 -> sum 180.
        let serving = gauge(&summary, MEDIA_SERVING_LATENCY);
        assert!((serving.sum - 180.0).abs() < 1e-9);

        // bytes summed per request: A=300, B=500 -> 2 samples, sum 800.
        let bytes = gauge(&summary, MEDIA_BYTES_SERVED);
        assert_eq!(bytes.count, 2);
        assert!((bytes.sum - 800.0).abs() < 1e-9);

        // fetch count per request: A=2, B=1 -> 2 samples, avg 1.5.
        let fetches = gauge(&summary, MEDIA_FETCH_COUNT);
        assert_eq!(fetches.count, 2);
        assert!((fetches.avg - 1.5).abs() < 1e-9);
    }

    #[test]
    fn negative_time_to_fetch_is_clamped_and_counted() {
        let mut agg = MediaFetchAggregator::new();
        // arrival before dispatch (clock skew).
        let rec = agg.ingest(&record("rid=A&mi=0&td=5000", 4000, 10, 10)).unwrap();
        assert_eq!(rec.time_to_media_fetch_ns, 0);
        let summary = agg.finish();
        assert_eq!(summary.negative_ttmf, 1);
    }

    #[test]
    fn empty_aggregator_yields_no_metrics() {
        let summary = MediaFetchAggregator::new().finish();
        assert!(summary.metrics.is_empty());
        assert_eq!(summary.total_fetches, 0);
    }
}
