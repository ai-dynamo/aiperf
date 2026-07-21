// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! IO-free AIPerf metrics engine with accuracy accumulation and analysis.

pub mod accumulator;
pub mod accuracy;
pub mod catalog;
pub mod counter;
pub mod derived;
pub mod ingest;
pub mod kernel;
pub mod report;
pub mod sidecar;
pub mod steady_state;
pub mod store;
pub mod sweepline;
pub mod units;
pub mod value;
pub mod window;

pub use accumulator::{
    Accumulator, AccumulatorSummary, InferenceMetricSeriesSummary, MetricResult, MetricResultData,
    MetricTimeslice, MetricsAccumulator, MetricsConfig, MetricsMergeError, SloThreshold,
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
    PlotMetricDirection, RecordMetricColumn, record_metric_columns, validate_catalog,
};
pub use counter::{CounterDelta, boundary_counter_delta};
pub use derived::{delta_ms, network_adjusted_ms};
pub use ingest::{InferenceDimensions, RecordIngest, RequestTrace, TokenCounts, UsageMetrics};
pub use kernel::{DistributionStats, PERCENTILES, linear_distribution, nearest_distribution};
pub use report::{
    EvaluatorDatasetReportInfo, EvaluatorReportInfo, FiniteReportValue, MetricEntry, MetricSeries,
    NATIVE_REPORT_SCHEMA_VERSION, NativeReport, NativeReportInput, NativeReporter, ReportClockKind,
    ReportCounterStats, ReportDistributionStats, ReportDynamoCapacityInfo, ReportDynamoParityInfo,
    ReportDynamoRouter, ReportDynamoRunInfo, ReportDynamoTopology, ReportEndpointProfileIdentity,
    ReportError, ReportExtensionIdentity, ReportGraphOutcomeInfo, ReportGraphRunInfo,
    ReportMetadataError, ReportPairRunFacts, ReportRun, ReportRunInfo, ReportRunMetadata,
    ReportScalarStats, ReportServerMetricsEndpointInfo, ReportServerMetricsMetadata,
    ReportServerMetricsPhaseRange, ReportStats, ReportSteadyState, ReportSummary, ReportTimeslice,
    ReportValue, Reporter, RunOutcome,
};
pub use sidecar::{SidecarMetric, SidecarSeries, SidecarStats, SidecarTimeslice};
pub use steady_state::{
    DEFAULT_STEADY_STATE_FRACTION, SteadyStateConfig, SteadyStateOutcome, SteadyWindow,
    detect_steady_window, steady_state_summary,
};
pub use store::{
    CategoryInterner, ColumnStore, ListMetricBackend, MetricsStorageMode, NumericColumn,
    RaggedReplay, RaggedSeries, SketchColumns, TagSketch,
};
// Re-export the shared t-digest so the sketch retention path has a single source.
pub use crate::cellular::sketch::{DEFAULT_COMPRESSION as SKETCH_DEFAULT_COMPRESSION, TDigest};
pub use units::{MetricValueType, Unit, UnitConversionError};
pub use value::MetricValue;
pub use window::{ExportContext, Phase, Timeslice};
