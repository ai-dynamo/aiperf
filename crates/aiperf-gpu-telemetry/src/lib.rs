// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DCGM-first GPU telemetry for native AIPerf.
//!
//! The crate is a leaf over the IO-free `aiperf-metrics` seam. A runtime owns
//! the async collection task and phase barriers; this crate owns DCGM decoding,
//! exact counter snapshots, gauge accumulation, and GPU efficiency rollups.
//! Local NVML/AMDSMI implementations can be added behind [`GpuTelemetrySource`]
//! without changing the accumulator or reporter.

pub mod accumulator;
pub mod collector;
pub mod fields;
pub mod model;
pub mod parser;
pub mod source;

pub use accumulator::{
    GpuMergeError, GpuPhaseBoundary, GpuTelemetryAccumulator, GpuTelemetrySummary,
};
pub use collector::GpuTelemetryCollector;
pub use fields::{
    AMD_METRICS, DCGM_METRICS, GpuMetricKind, GpuMetricSpec, amd_metric_spec, dcgm_metric_spec,
    metric_spec,
};
pub use model::{GpuBoundarySnapshot, GpuMetadata, GpuScrape, GpuSeriesKey, GpuTelemetryRecord};
pub use parser::{DcgmPrometheusDecoder, GpuTelemetryDecoder};
pub use source::{DcgmTelemetrySource, GpuScrapeMode, GpuTelemetryError, GpuTelemetrySource};

/// Default GPU telemetry collection cadence: 333 milliseconds.
pub const DEFAULT_COLLECTION_INTERVAL_NS: i64 = 333_000_000;
