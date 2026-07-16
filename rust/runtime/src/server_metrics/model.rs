// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parsed server-metrics records.
//!
//! The family/sample representation uses Rust
//! enums to make scalar and histogram payloads mutually exclusive by type.

use std::collections::BTreeMap;

use crate::metrics_core::Phase;
use serde::Serialize;

/// Prometheus family semantics preserved through accumulation and export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PrometheusMetricType {
    /// Monotonic scalar counter.
    Counter,
    /// Point-in-time scalar gauge.
    Gauge,
    /// Classic cumulative-bucket histogram.
    Histogram,
    /// Server-lifetime summary; parsed but excluded from benchmark exports.
    Summary,
    /// Untyped family, accumulated with gauge-shaped statistics.
    Unknown,
}

/// Structured value for one histogram label set.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HistogramValue {
    /// Cumulative counts keyed by `le` upper bound.
    pub buckets: BTreeMap<String, f64>,
    /// Cumulative observation sum, when emitted.
    pub sum: Option<f64>,
    /// Cumulative observation count, when emitted.
    pub count: Option<f64>,
}

/// One scalar or histogram sample with its dimensional labels.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MetricSample {
    /// Counter, gauge, or unknown scalar.
    Scalar {
        /// Stable labels; empty means unlabeled.
        labels: BTreeMap<String, String>,
        /// Finite sample value.
        value: f64,
    },
    /// Histogram sum/count/buckets grouped by labels excluding `le`.
    Histogram {
        /// Stable base labels; empty means unlabeled.
        labels: BTreeMap<String, String>,
        /// Structured cumulative histogram value.
        value: HistogramValue,
    },
}

impl MetricSample {
    /// Returns this sample's labels independent of value shape.
    pub fn labels(&self) -> &BTreeMap<String, String> {
        match self {
            Self::Scalar { labels, .. } | Self::Histogram { labels, .. } => labels,
        }
    }
}

/// One parsed metric family.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MetricFamily {
    /// Semantic family type.
    pub metric_type: PrometheusMetricType,
    /// HELP text, or an empty string when absent.
    pub description: String,
    /// Last-value-deduplicated label series.
    pub samples: Vec<MetricSample>,
}

/// Complete metrics snapshot from one endpoint.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ServerMetricsRecord {
    /// Credential-free endpoint used for the successful scrape.
    pub endpoint_url: String,
    /// Clock timestamp representing the scrape snapshot.
    pub timestamp_ns: i64,
    /// Optional HTTP round-trip duration.
    pub endpoint_latency_ns: Option<i64>,
    /// Clock timestamp when the control-plane GET began.
    pub request_sent_ns: Option<i64>,
    /// Clock timestamp of the first response byte.
    pub first_byte_ns: Option<i64>,
    /// Whether this body equals the preceding body from this source.
    pub is_duplicate: bool,
    /// Benchmark phase active for this scrape; absent for a boundary baseline.
    pub benchmark_phase: Option<Phase>,
    /// Parsed families keyed by normalized name (counter `_total` removed).
    pub metrics: BTreeMap<String, MetricFamily>,
}
