// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Rust AIPerf metrics engine.
//!
//! This crate is the IO-free metrics plane described by
//! `specs/2026-07-10-aiperf-rust-metrics-accumulator-sweepline-design.md` and the
//! first-class accuracy accumulator/analyzer from
//! `specs/2026-07-10-aiperf-rust-accuracy-accumulator-design.md`.

pub mod accumulator;
pub mod accuracy;
pub mod catalog;
pub mod derived;
pub mod ingest;
pub mod kernel;
pub mod report;
pub mod store;
pub mod sweepline;
pub mod units;
pub mod value;
pub mod window;

pub use accumulator::{
    Accumulator, AccumulatorSummary, MetricResult, MetricResultData, MetricTimeslice,
    MetricsAccumulator, MetricsConfig, MetricsMergeError, SloThreshold,
};
pub use accuracy::{
    ACCURACY_RECORD_TYPE, AccumulatorType, AccuracyAccumulator, AccuracyAnalysis, AccuracyAtLoad,
    AccuracyRecord, AccuracyRecordError, AccuracyResultsAnalyzer, AccuracyRollup, AccuracySummary,
    Analyzer, AnalyzerRunError, AnalyzerRunner, AnalyzerType, ConfidenceInterval, CorrelationId,
    EnergyEfficiencySummary, GradingResult, LIGHTEVAL_CORRECTNESS_THRESHOLD, SummaryContext,
    TaskId,
};
pub use catalog::{
    AggregationKind, CATALOG, MetricConsoleGroup, MetricFlags, MetricSpec, MetricTag, MetricType,
    PlotMetricDirection, validate_catalog,
};
pub use derived::{delta_ms, error_adjusted_result, network_adjusted_ms};
pub use ingest::{HttpTrace, RecordIngest, TokenCounts, UsageMetrics};
pub use kernel::{DistributionStats, PERCENTILES, linear_distribution, nearest_distribution};
pub use report::{
    MetricEntry, MetricSeries, NATIVE_REPORT_SCHEMA_VERSION, NativeReport, NativeReporter,
    ReportCounterStats, ReportDistributionStats, ReportError, ReportRunInfo, ReportScalarStats,
    ReportStats, ReportSummary, ReportTimeslice, ReportValue, Reporter, RunOutcome,
};
pub use store::{
    CategoryInterner, ColumnStore, ListMetricBackend, NumericColumn, RaggedReplay, RaggedSeries,
};
pub use units::{MetricValueType, Unit, UnitConversionError};
pub use value::MetricValue;
pub use window::{ExportContext, Phase, Timeslice};
