// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-owned exporter capture requirements.
//!
//! An [`ExportCapturePlan`] expresses which per-record fields and histogram
//! projections a set of exporters requires, annotated with the exporter that
//! drives each requirement. The engine validates the plan against the run's
//! metrics mode (exact vs sketch) before any effects, so capture failures
//! surface before sockets are opened.

use std::fmt;

/// A per-record field that one or more exporters need retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExactRecordField {
    /// Sequential 0-based request index assigned at dispatch.
    RequestIndex,
    /// Model-generated text payload.
    ResponseContent,
    /// Client-visible input token count.
    InputTokens,
    /// Client-visible output token count.
    OutputTokens,
    /// Time from request send to first SSE/streaming token.
    TimeToFirstToken,
    /// Per-output-token inter-token latency.
    InterTokenLatency,
    /// End-to-end request latency.
    RequestLatency,
    /// Raw HTTP exchange bytes.
    RawExchange,
}

/// Annotation identifying which exporter requires a given field or histogram.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetentionReason {
    /// The named exporter (canonical ID string) requires this retention.
    RequiredByExporter(String),
}

impl fmt::Display for RetentionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequiredByExporter(id) => write!(f, "required by exporter {id}"),
        }
    }
}

/// One exact-record field requirement with its reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactRecordRequirement {
    /// Which field must be retained.
    pub field: ExactRecordField,
    /// Which exporter requires it.
    pub reason: RetentionReason,
}

/// A named GenAI-semconv histogram that an exporter needs populated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistogramRequirement {
    /// Semantic-convention histogram name
    /// (e.g. `gen_ai.client.operation.duration`).
    pub name: String,
    /// Canonical exporter ID that drives this requirement.
    pub exporter_id: String,
}

/// Error returned when an [`ExportCapturePlan`] cannot be constructed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportCapturePlanError {
    /// One or more exact-record requirements conflict with sketch-metrics mode.
    SketchIncompatible {
        /// The exporter IDs whose requirements cannot be satisfied.
        conflicting_exporters: Vec<String>,
    },
}

impl fmt::Display for ExportCapturePlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SketchIncompatible {
                conflicting_exporters,
            } => write!(
                f,
                "sketch metrics mode is incompatible with exact-record requirements from: {}",
                conflicting_exporters.join(", ")
            ),
        }
    }
}

impl std::error::Error for ExportCapturePlanError {}

/// Combined capture requirements across all exporters for one run.
///
/// Constructed by merging per-exporter plans before runtime effects begin.
/// The engine validates this against the run's metrics mode (exact vs sketch)
/// so incompatibilities surface before any socket or file is opened.
#[derive(Debug, Clone, Default)]
pub struct ExportCapturePlan {
    /// Which per-record fields must be retained and why. Sorted by
    /// [`ExactRecordField`] discriminant for deterministic iteration order.
    pub exact_record_requirements: Vec<ExactRecordRequirement>,
    /// Named histograms that exporters need populated from per-record data.
    pub histogram_requirements: Vec<HistogramRequirement>,
    /// Whether any exporter requires per-record exact data. `false` when
    /// `exact_record_requirements` is empty.
    pub requires_exact_records: bool,
}

impl ExportCapturePlan {
    /// Build a plan with a single field requirement.
    pub fn with_requirement(field: ExactRecordField, reason: RetentionReason) -> Self {
        Self {
            requires_exact_records: true,
            exact_record_requirements: vec![ExactRecordRequirement { field, reason }],
            histogram_requirements: Vec::new(),
        }
    }

    /// Merge an iterable of plans into one sorted-union plan.
    ///
    /// When a field appears in multiple plans the first requirement (by
    /// iteration order) is kept. Requirements are sorted by field discriminant.
    pub fn merge<I>(plans: I) -> Result<Self, ExportCapturePlanError>
    where
        I: IntoIterator<Item = Self>,
    {
        Self::merge_with_sketch_check(plans, false)
    }

    /// Merge plans, rejecting the result if `sketch` mode is enabled and any
    /// exact-record requirement is present.
    pub fn merge_with_sketch_check<I>(
        plans: I,
        sketch: bool,
    ) -> Result<Self, ExportCapturePlanError>
    where
        I: IntoIterator<Item = Self>,
    {
        let mut seen_fields = std::collections::BTreeSet::new();
        let mut requirements: Vec<ExactRecordRequirement> = Vec::new();
        let mut histograms: Vec<HistogramRequirement> = Vec::new();

        for plan in plans {
            for req in plan.exact_record_requirements {
                if seen_fields.insert(req.field) {
                    requirements.push(req);
                }
            }
            histograms.extend(plan.histogram_requirements);
        }

        // Sort by field discriminant for stable ordering.
        requirements.sort_by_key(|r| r.field);

        if sketch && !requirements.is_empty() {
            let conflicting_exporters: Vec<String> = requirements
                .iter()
                .filter_map(|r| match &r.reason {
                    RetentionReason::RequiredByExporter(id) => Some(id.clone()),
                })
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            return Err(ExportCapturePlanError::SketchIncompatible {
                conflicting_exporters,
            });
        }

        let requires_exact_records = !requirements.is_empty();
        Ok(Self {
            exact_record_requirements: requirements,
            histogram_requirements: histograms,
            requires_exact_records,
        })
    }
}
