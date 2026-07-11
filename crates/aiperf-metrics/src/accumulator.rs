// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal metric accumulator summary types shared with analyzers.
//!
//! The full sweep-line metrics engine grows from these types. Accuracy analyzers only
//! need typed access to already-summarized metric scalars, so this module keeps the
//! join seam small and IO-free.

use crate::{MetricTag, MetricValue};
use std::collections::BTreeMap;

/// A summarized metric value keyed by catalog tag.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricResult {
    /// Metric identity.
    pub tag: MetricTag,
    /// Boundary-safe metric value.
    pub value: MetricValue,
}

impl MetricResult {
    /// Builds a finite metric result.
    pub fn finite(tag: MetricTag, value: f64) -> Self {
        Self {
            tag,
            value: MetricValue::from_f64(value, false),
        }
    }
}

/// Summary produced by the metrics accumulator.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AccumulatorSummary {
    values: BTreeMap<MetricTag, MetricValue>,
}

impl AccumulatorSummary {
    /// Builds an empty summary.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces a metric value.
    pub fn insert(&mut self, tag: MetricTag, value: MetricValue) {
        self.values.insert(tag, value);
    }

    /// Inserts a finite metric value.
    pub fn insert_finite(&mut self, tag: MetricTag, value: f64) {
        self.insert(tag, MetricValue::from_f64(value, false));
    }

    /// Returns a boundary value by tag.
    pub fn value(&self, tag: MetricTag) -> Option<MetricValue> {
        self.values.get(&tag).copied()
    }

    /// Returns a finite value by tag.
    pub fn finite_value(&self, tag: MetricTag) -> Option<f64> {
        self.value(tag).and_then(MetricValue::as_f64)
    }

    /// Iterates over summarized values in tag order.
    pub fn iter(&self) -> impl Iterator<Item = (MetricTag, MetricValue)> + '_ {
        self.values.iter().map(|(tag, value)| (*tag, *value))
    }
}

/// Placeholder for the full metrics accumulator implementation.
#[derive(Debug, Default)]
pub struct MetricsAccumulator {
    summary: AccumulatorSummary,
}

impl MetricsAccumulator {
    /// Builds an empty metrics accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Injects a summarized scalar. Runtime record ingestion will replace this seam
    /// with the full columnar engine while preserving analyzer access.
    pub fn insert_summary_value(&mut self, tag: MetricTag, value: MetricValue) {
        self.summary.insert(tag, value);
    }

    /// Exports the current summary.
    pub fn export_results(&self) -> AccumulatorSummary {
        self.summary.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::AccumulatorSummary;
    use crate::{MetricTag, MetricValue};

    #[test]
    fn summary_returns_only_finite_values_for_joins() {
        let mut summary = AccumulatorSummary::new();
        summary.insert_finite(MetricTag::Goodput, 42.0);
        summary.insert(MetricTag::RequestLatency, MetricValue::Absent);
        assert_eq!(summary.finite_value(MetricTag::Goodput), Some(42.0));
        assert_eq!(summary.finite_value(MetricTag::RequestLatency), None);
    }
}
