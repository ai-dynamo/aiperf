// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Domain-neutral metric series supplied by telemetry side channels.
//!
//! GPU, server, and network collectors depend on this IO-free representation;
//! the core metrics engine never depends back on those producer domains. New
//! telemetry implementations can populate the same structs without changing
//! the native reporter.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::metrics_core::{DistributionStats, MetricConsoleGroup, MetricValue, Unit};

/// Type-specific statistics for one side-channel metric series.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum SidecarStats {
    /// Sampled gauge distribution, including telemetry's `ddof=1` standard deviation.
    Gauge(DistributionStats),
    /// Cumulative counter delta and optional authoritative-window rate.
    Counter {
        /// Reset-clamped counter delta.
        total: MetricValue,
        /// Delta divided by the exact phase duration, when meaningful.
        rate: Option<MetricValue>,
    },
    /// Prometheus histogram boundary delta and polynomial percentile estimates.
    Histogram {
        /// Reset-clamped observation-count delta.
        count: u64,
        /// Reset-clamped observation-sum delta.
        sum: MetricValue,
        /// `sum / count`, when count is positive.
        avg: Option<MetricValue>,
        /// Observations per exact phase second.
        count_rate: Option<MetricValue>,
        /// Observation-value sum per exact phase second.
        sum_rate: Option<MetricValue>,
        /// Estimated percentiles keyed by integer percentile.
        percentiles: BTreeMap<u32, MetricValue>,
        /// Reset-clamped cumulative bucket deltas keyed by upper bound.
        buckets: BTreeMap<String, u64>,
    },
}

/// One chronological side-channel timeslice.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SidecarTimeslice {
    /// Inclusive window start in nanoseconds.
    pub start_ns: i64,
    /// Exclusive window end in nanoseconds.
    pub end_ns: i64,
    /// Whether this spans the full configured duration.
    pub complete: bool,
    /// Type-appropriate statistics for this slice.
    pub stats: SidecarStats,
}

/// One labeled side-channel series from one optional source endpoint.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SidecarSeries {
    /// Stable label dimensions, absent for an unlabeled series.
    pub labels: Option<BTreeMap<String, String>>,
    /// Credential-free source endpoint.
    pub endpoint_url: Option<String>,
    /// Overall series statistics.
    pub stats: SidecarStats,
    /// Chronological non-empty timeslices.
    pub timeslices: Vec<SidecarTimeslice>,
}

/// One externally produced metric with one or more labeled series.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SidecarMetric {
    /// Display unit inferred or declared by the producer.
    pub unit: Option<Unit>,
    /// Console grouping used by presentation consumers.
    pub console_group: MetricConsoleGroup,
    /// Plot/SLO direction.
    pub higher_is_better: bool,
    /// Deterministically ordered endpoint/label series.
    pub series: Vec<SidecarSeries>,
}

impl SidecarMetric {
    /// Builds a metric with the default console group and no preferred direction.
    pub fn new(unit: Option<Unit>, series: Vec<SidecarSeries>) -> Self {
        Self {
            unit,
            console_group: MetricConsoleGroup::Default,
            higher_is_better: false,
            series,
        }
    }
}
