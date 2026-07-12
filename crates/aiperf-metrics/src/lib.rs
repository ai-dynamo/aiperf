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
pub mod counter;
pub mod derived;
pub mod ingest;
pub mod kernel;
pub mod report;
pub mod sidecar;
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
    PlotMetricDirection, validate_catalog,
};
pub use counter::{CounterDelta, boundary_counter_delta};
pub use derived::{delta_ms, error_adjusted_result, network_adjusted_ms};
pub use ingest::{HttpTrace, InferenceDimensions, RecordIngest, TokenCounts, UsageMetrics};
pub use kernel::{DistributionStats, PERCENTILES, linear_distribution, nearest_distribution};
pub use report::{
    AgenticEpisodeReport, AgenticEpisodeReportOutcome, AgenticEvaluationReport,
    AgenticEvaluationSummary, AgenticEvaluatorReportInfo, AgenticRewardSummary,
    AgenticRunConfigReport, EvaluationAggregateMetricReport, EvaluationArtifactReport,
    EvaluationCaseErrorReport, EvaluationCaseOutcomeKind, EvaluationCaseReport,
    EvaluationIdentityReport, EvaluationPublicScoreReport, EvaluationReport, EvaluationRouteReport,
    EvaluationRouteSummaryReport, EvaluatorDatasetReportInfo, EvaluatorReportInfo,
    FiniteReportValue, MetricEntry, MetricSeries, NATIVE_REPORT_SCHEMA_VERSION, NativeReport,
    NativeReportInput, NativeReporter, ReportClockKind, ReportCounterStats,
    ReportDistributionStats, ReportDynamoCapacityInfo, ReportDynamoParityInfo, ReportDynamoRouter,
    ReportDynamoRunInfo, ReportDynamoTopology, ReportEndpointProfileIdentity, ReportError,
    ReportEvaluationCompatibilityGrantLimits, ReportEvaluationCompatibilityInfo,
    ReportExtensionIdentity, ReportGraphOutcomeInfo, ReportGraphRunInfo, ReportPairRunFacts,
    ReportProvenanceError, ReportRun, ReportRunInfo, ReportRunProvenance, ReportScalarStats,
    ReportServerMetricsEndpointInfo, ReportServerMetricsMetadata, ReportServerMetricsPhaseRange,
    ReportStats, ReportSummary, ReportTelemetryArchive, ReportTelemetryArchiveHead,
    ReportTelemetryArchiveHealth, ReportTelemetryArchiveSpoolBudget,
    ReportTelemetryArchiveState, ReportTelemetryBoundaryReference,
    ReportTelemetryBoundaryRole, ReportTelemetryLossKind, ReportTelemetryLossRange,
    ReportTelemetryLossReason, ReportTelemetryLossSaturationSummary, ReportTimeslice, ReportValue,
    Reporter, RunOutcome, TELEMETRY_ARCHIVE_REPORT_SCHEMA_VERSION,
};
pub use sidecar::{SidecarMetric, SidecarSeries, SidecarStats, SidecarTimeslice};
pub use store::{
    CategoryInterner, ColumnStore, ListMetricBackend, NumericColumn, RaggedReplay, RaggedSeries,
};
pub use units::{MetricValueType, Unit, UnitConversionError};
pub use value::MetricValue;
pub use window::{ExportContext, Phase, Timeslice};
