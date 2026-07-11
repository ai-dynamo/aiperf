// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native report DTOs produced by metrics accumulators and analyzers.

use crate::{AccumulatorSummary, AccuracyAnalysis, MetricValue};
use serde::Serialize;

/// One metric entry in the native report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MetricReportEntry {
    /// Stable metric tag.
    pub tag: String,
    /// Boundary-safe value.
    pub value: MetricValue,
}

/// Native report shape used by the v2 exporter design.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NativeReport {
    /// Summarized inference metrics.
    pub metrics: Vec<MetricReportEntry>,
    /// Optional accuracy analysis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<AccuracyAnalysis>,
}

impl NativeReport {
    /// Builds a native report from a metrics summary and optional accuracy analysis.
    pub fn new(metrics: &AccumulatorSummary, accuracy: Option<AccuracyAnalysis>) -> Self {
        let metrics = metrics
            .iter()
            .map(|(tag, value)| MetricReportEntry {
                tag: tag.as_str().to_string(),
                value,
            })
            .collect();
        Self { metrics, accuracy }
    }
}
