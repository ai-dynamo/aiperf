// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inference-server `/metrics` telemetry for native AIPerf.
//!
//! This leaf crate keeps Prometheus parsing, endpoint compatibility policy,
//! histogram estimation, and server-specific aggregation outside the IO-free
//! `aiperf-metrics` engine. Runtime code owns phase barriers and the sequential
//! scrape task. Parser, unit-inference, source, and realtime-atlas policies are
//! traits so additional exposition dialects and engines remain injectable.

pub mod accumulator;
pub mod atlas;
pub mod histogram;
pub mod model;
pub mod parser;
pub(crate) mod prom_text;
pub mod source;
pub mod units;

pub use atlas::{DerivedServerMetric, ServerMetricAtlas, ServerMetricView, VllmSglangMetricAtlas};
pub use histogram::{
    BucketStatistics, HistogramSnapshot, accumulate_bucket_statistics,
    compute_estimated_percentiles, compute_prometheus_percentiles,
};
pub use model::{
    HistogramValue, MetricFamily, MetricSample, PrometheusMetricType, ServerMetricsRecord,
};
pub use parser::{MetricsParseError, MetricsTextParser, PrometheusTextParser};
pub use source::{
    PrometheusHttpSource, ServerMetricsError, ServerMetricsScrapeMode, ServerMetricsScrapeOutcome,
    ServerMetricsSource,
};
pub use units::{PrometheusUnitInferer, UnitInferer, infer_unit};

/// Sequential server scrape cadence: 333 milliseconds.
pub const DEFAULT_COLLECTION_INTERVAL_NS: i64 = 333_000_000;

/// Connection/reachability timeout: 10 seconds.
pub const DEFAULT_REACHABILITY_TIMEOUT_NS: i64 = 10_000_000_000;
pub use accumulator::{
    ServerMetricsAccumulator, ServerMetricsEndpointInfo, ServerMetricsMergeError,
    ServerMetricsPhaseBoundary, ServerMetricsSummary,
};
