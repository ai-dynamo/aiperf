// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DCGM-first GPU telemetry for native AIPerf.
//!
//! The runtime owns the async collection task and phase barriers. This module owns DCGM decoding,
//! exact counter snapshots, gauge accumulation, and GPU efficiency rollups.

pub mod accumulator;
pub mod amdsmi;
pub mod collector;
pub mod custom_metrics;
pub mod fields;
pub mod model;
pub mod nvml;
pub mod parser;
pub mod source;
mod vendor_worker;

pub use accumulator::{
    GpuMergeError, GpuMetricRegistrationError, GpuPhaseBoundary, GpuTelemetryAccumulator,
    GpuTelemetrySummary,
};
pub use collector::GpuTelemetryCollector;
pub use custom_metrics::{
    CustomDcgmField, CustomMetricsError, LoadedCustomMetrics, load_custom_dcgm_metrics,
};
pub use fields::{
    AMD_METRICS, DCGM_METRICS, GpuMetricKind, GpuMetricSpec, LEGACY_NVIDIA_METRIC_ALIASES,
    RuntimeGpuMetricSpec, amd_metric_spec, dcgm_metric_spec, metric_spec,
    normalize_legacy_nvidia_metric_names,
};
pub use model::{
    AMD_GPU_TELEMETRY_PLATFORM, GpuBoundarySnapshot, GpuMetadata, GpuScrape, GpuSeriesKey,
    GpuTelemetryRecord, NVIDIA_GPU_TELEMETRY_PLATFORM, UNKNOWN_GPU_TELEMETRY_PLATFORM,
};
pub use parser::{DcgmPrometheusDecoder, GpuTelemetryDecoder};
pub use source::{DcgmTelemetrySource, GpuScrapeMode, GpuTelemetryError, GpuTelemetrySource};
