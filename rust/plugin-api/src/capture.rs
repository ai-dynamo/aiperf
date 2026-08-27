// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What an exporter requires from a finished run, and how it is then invoked.
//!
//! An exporter declares its requirements before the run starts, so the host
//! knows whether to retain exact records at all — retention is expensive and is
//! only paid for when some configured exporter asked for it. The requirement
//! vocabulary is sealed: an exporter selects from [`CaptureRequirementV1`] and
//! cannot invent a projection identifier, because a projection the host does not
//! produce is a silently empty artifact rather than an error.
//!
//! [`ExporterCaptureRequirementsV1`] is a sorted set and always contains
//! [`CaptureRequirementV1::FinalReport`]: the finalized report is what makes a
//! run's output a report at all, so it is available unconditionally.
//!
//! The projection types themselves are `aiperf-core`'s. This module references
//! them; it does not redefine them.

use core::fmt::{self, Display, Formatter};

use aiperf_core::artifact::ArtifactAccess;
use aiperf_core::capture::{ExactRecordsV1, FinalReportV1, GenAiClientHistogramsV1};

use crate::id::RegistryId;
use crate::validation::{PluginCategory, ValidationError};

/// The folded projections the host can produce.
///
/// Folded projections are merged across workers and cells before an exporter
/// sees them, so an exporter never merges partial state itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FoldedProjectionV1 {
    /// `aiperf_core::capture::GenAiClientHistogramsV1`.
    GenAiClientHistogramsV1,
}

impl FoldedProjectionV1 {
    /// The projection identifier used in receipts and diagnostics.
    pub const fn label(self) -> &'static str {
        match self {
            Self::GenAiClientHistogramsV1 => "GenAiClientHistogramsV1",
        }
    }
}

/// One capture an exporter requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CaptureRequirementV1 {
    /// `aiperf_core::capture::FinalReportV1`; always available.
    FinalReport,
    /// `aiperf_core::capture::ExactRecordsV1`; retained only on request.
    ExactRecordsV1,
    /// A folded projection, merged across workers and cells.
    FoldedProjectionV1(FoldedProjectionV1),
}

/// Every capture requirement in canonical (sorted) order.
pub const CAPTURE_REQUIREMENTS_V1: &[CaptureRequirementV1] = &[
    CaptureRequirementV1::FinalReport,
    CaptureRequirementV1::ExactRecordsV1,
    CaptureRequirementV1::FoldedProjectionV1(FoldedProjectionV1::GenAiClientHistogramsV1),
];

impl CaptureRequirementV1 {
    /// The requirement identifier used in receipts and diagnostics.
    pub const fn label(self) -> &'static str {
        match self {
            Self::FinalReport => "FinalReport",
            Self::ExactRecordsV1 => "ExactRecordsV1",
            Self::FoldedProjectionV1(projection) => projection.label(),
        }
    }

    /// Parse a requirement identifier, refusing anything outside the sealed set.
    pub fn parse(label: &str) -> Result<Self, ValidationError> {
        CAPTURE_REQUIREMENTS_V1
            .iter()
            .copied()
            .find(|requirement| requirement.label() == label)
            .ok_or_else(|| ValidationError::UnknownCaptureProjection(label.to_owned()))
    }
}

impl Display for CaptureRequirementV1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

/// A sorted set of the captures one exporter requires.
///
/// [`CaptureRequirementV1::FinalReport`] is inserted unconditionally, so an
/// exporter that declares nothing still receives a report, and two exporters
/// that declare the same set in different orders produce identical receipts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExporterCaptureRequirementsV1 {
    requirements: Vec<CaptureRequirementV1>,
}

impl Default for ExporterCaptureRequirementsV1 {
    fn default() -> Self {
        Self::new([])
    }
}

impl ExporterCaptureRequirementsV1 {
    /// Build the sorted set, always including the finalized report.
    pub fn new(requirements: impl IntoIterator<Item = CaptureRequirementV1>) -> Self {
        let mut requirements: Vec<CaptureRequirementV1> = requirements
            .into_iter()
            .chain([CaptureRequirementV1::FinalReport])
            .collect();
        requirements.sort_unstable();
        requirements.dedup();
        Self { requirements }
    }

    /// The requirements in canonical order.
    pub fn as_slice(&self) -> &[CaptureRequirementV1] {
        &self.requirements
    }

    /// Whether a capture was required.
    pub fn contains(&self, requirement: CaptureRequirementV1) -> bool {
        self.requirements.contains(&requirement)
    }

    /// Whether exact per-record retention must be paid for.
    pub fn needs_exact_records(&self) -> bool {
        self.contains(CaptureRequirementV1::ExactRecordsV1)
    }
}

/// The captures handed to one prepared exporter after the report is finalized.
///
/// A projection an exporter did not require is `None` even when the host
/// produced it for a sibling exporter: an exporter reads exactly what its own
/// receipt says it asked for.
#[derive(Debug, Clone, Copy)]
pub struct ExportInputV1<'a> {
    report: &'a FinalReportV1,
    exact_records: Option<&'a ExactRecordsV1>,
    histograms: Option<&'a GenAiClientHistogramsV1>,
}

impl<'a> ExportInputV1<'a> {
    /// Bind the finalized report, which is always present.
    pub const fn new(report: &'a FinalReportV1) -> Self {
        Self {
            report,
            exact_records: None,
            histograms: None,
        }
    }

    /// Attach the exact per-record projection.
    pub const fn with_exact_records(mut self, records: &'a ExactRecordsV1) -> Self {
        self.exact_records = Some(records);
        self
    }

    /// Attach the folded GenAI client histogram projection.
    pub const fn with_histograms(mut self, histograms: &'a GenAiClientHistogramsV1) -> Self {
        self.histograms = Some(histograms);
        self
    }

    /// The finalized report projection.
    pub const fn report(&self) -> &'a FinalReportV1 {
        self.report
    }

    /// The exact per-record projection, when required and retained.
    pub const fn exact_records(&self) -> Option<&'a ExactRecordsV1> {
        self.exact_records
    }

    /// The folded GenAI client histogram projection, when required.
    pub const fn histograms(&self) -> Option<&'a GenAiClientHistogramsV1> {
        self.histograms
    }
}

/// Why an exporter did not produce its output.
///
/// An exporter failure is a typed value, never a panic: every artifact on both
/// sides of the boundary is built `panic = abort`, so an unwinding exporter
/// would take the run's other output with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExporterError {
    /// A required capture was not supplied.
    MissingCapture(CaptureRequirementV1),
    /// Writing through the artifact capability failed.
    Artifact(String),
    /// The exporter refused for a backend-specific reason.
    Backend(String),
}

impl Display for ExporterError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCapture(requirement) => {
                write!(formatter, "required capture {requirement} was not supplied")
            }
            Self::Artifact(reason) => write!(formatter, "exporter artifact write failed: {reason}"),
            Self::Backend(reason) => write!(formatter, "exporter refused: {reason}"),
        }
    }
}

impl core::error::Error for ExporterError {}

/// The post-report execution boundary of one validated exporter.
///
/// The host calls [`PreparedExporterV1::export`] once, after the report is
/// finalized and after ordering has placed this exporter relative to its peers.
/// The only I/O it is given is [`ArtifactAccess`]: there is no artifact
/// directory path to reconstruct.
pub trait PreparedExporterV1 {
    /// The registered identifier of the exporter that produced this value.
    fn id(&self) -> &RegistryId;

    /// The captures this exporter's receipt bound it to.
    fn requirements(&self) -> &ExporterCaptureRequirementsV1;

    /// Write this exporter's output.
    fn export(
        &self,
        input: ExportInputV1<'_>,
        artifacts: &dyn ArtifactAccess,
    ) -> Result<(), ExporterError>;
}

/// The category every value in this module belongs to.
pub const EXPORTER_CATEGORY: PluginCategory = PluginCategory::Exporter;
