// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Building and satisfying the post-report exporter boundary.
//!
//! [`aiperf_plugin_api::PreparedExporterV1`] is the only thing the host calls
//! after a run finalizes. Two things every implementation of it does are shared
//! here: turning a declared-but-absent capture into the exact
//! [`ExporterError::MissingCapture`] the boundary expects, and wrapping a plain
//! closure as a prepared exporter so a leaf that is one write does not need a
//! hand-written trait impl.
//!
//! The requirement checks are deliberately not silent: a projection an exporter
//! declared and did not receive is a host-side bug, and an exporter that treats
//! it as "nothing to write" produces a silently empty artifact instead of a
//! diagnosable failure.

use aiperf_core::artifact::ArtifactAccess;
use aiperf_core::capture::{ExactRecordsV1, GenAiClientHistogramsV1};
use aiperf_plugin_api::capture::FoldedProjectionV1;
use aiperf_plugin_api::{
    CaptureRequirementV1, ExportInputV1, ExporterCaptureRequirementsV1, ExporterError,
    PreparedExporterV1, RegistryId,
};
use serde_json::Value;

/// The finalized report's payload: the exact projection the host commits.
///
/// [`CaptureRequirementV1::FinalReport`] is unconditional, so this never fails;
/// it exists so a leaf reads the report the same way it reads the optional
/// projections.
pub fn require_report<'a>(input: ExportInputV1<'a>) -> &'a Value {
    &input.report().report
}

/// The exact per-record projection, or the refusal for a capture that was
/// declared and not supplied.
pub fn require_exact_records<'a>(
    input: ExportInputV1<'a>,
) -> Result<&'a ExactRecordsV1, ExporterError> {
    input.exact_records().ok_or(ExporterError::MissingCapture(
        CaptureRequirementV1::ExactRecordsV1,
    ))
}

/// The folded GenAI client histogram projection, or the refusal for a capture
/// that was declared and not supplied.
pub fn require_histograms<'a>(
    input: ExportInputV1<'a>,
) -> Result<&'a GenAiClientHistogramsV1, ExporterError> {
    input.histograms().ok_or(ExporterError::MissingCapture(
        CaptureRequirementV1::FoldedProjectionV1(FoldedProjectionV1::GenAiClientHistogramsV1),
    ))
}

/// A prepared exporter whose whole behavior is one closure.
///
/// The identity and requirements are stored rather than recomputed so the value
/// the host reads back is exactly the one the factory bound to its receipt.
pub struct ClosureExporter<F> {
    id: RegistryId,
    requirements: ExporterCaptureRequirementsV1,
    export: F,
}

impl<F> ClosureExporter<F>
where
    F: Fn(ExportInputV1<'_>, &dyn ArtifactAccess) -> Result<(), ExporterError>,
{
    /// Bind one identifier, requirement set, and export closure.
    pub const fn new(
        id: RegistryId,
        requirements: ExporterCaptureRequirementsV1,
        export: F,
    ) -> Self {
        Self {
            id,
            requirements,
            export,
        }
    }
}

impl<F> PreparedExporterV1 for ClosureExporter<F>
where
    F: Fn(ExportInputV1<'_>, &dyn ArtifactAccess) -> Result<(), ExporterError>,
{
    fn id(&self) -> &RegistryId {
        &self.id
    }

    fn requirements(&self) -> &ExporterCaptureRequirementsV1 {
        &self.requirements
    }

    fn export(
        &self,
        input: ExportInputV1<'_>,
        artifacts: &dyn ArtifactAccess,
    ) -> Result<(), ExporterError> {
        (self.export)(input, artifacts)
    }
}

#[cfg(test)]
mod tests {
    use aiperf_core::artifact::DirectoryArtifacts;
    use aiperf_core::capture::FinalReportV1;

    use super::*;

    fn identifier() -> RegistryId {
        RegistryId::new(
            "closure-exporter",
            aiperf_plugin_api::REGISTRY_ID_NORMALIZATION_VERSION,
        )
        .expect("valid identifier")
    }

    #[test]
    fn a_declared_but_absent_capture_names_itself_in_the_refusal() {
        let report = FinalReportV1::new(serde_json::json!({ "metrics": {} }));
        let input = ExportInputV1::new(&report);
        assert_eq!(require_report(input), &serde_json::json!({ "metrics": {} }));
        assert!(matches!(
            require_exact_records(input),
            Err(ExporterError::MissingCapture(
                CaptureRequirementV1::ExactRecordsV1
            ))
        ));
        assert!(matches!(
            require_histograms(input),
            Err(ExporterError::MissingCapture(
                CaptureRequirementV1::FoldedProjectionV1(_)
            ))
        ));
    }

    #[test]
    fn a_supplied_capture_is_handed_through_unchanged() {
        let report = FinalReportV1::new(Value::Null);
        let records = ExactRecordsV1::from_records(Vec::new());
        let input = ExportInputV1::new(&report).with_exact_records(&records);
        assert_eq!(require_exact_records(input).expect("records"), &records);
    }

    #[test]
    fn a_closure_exporter_writes_through_the_capability() {
        let root = tempfile::tempdir().expect("temporary artifact root");
        let artifacts = DirectoryArtifacts::new(root.path());
        let exporter = ClosureExporter::new(
            identifier(),
            ExporterCaptureRequirementsV1::default(),
            |input, artifacts| crate::write_json(artifacts, "report.json", require_report(input)),
        );
        let report = FinalReportV1::new(serde_json::json!({ "ok": true }));
        exporter
            .export(ExportInputV1::new(&report), &artifacts)
            .expect("export");
        assert_eq!(exporter.id(), &identifier());
        assert!(
            exporter
                .requirements()
                .contains(CaptureRequirementV1::FinalReport)
        );
        assert!(!artifacts.read("report.json").expect("read").is_empty());
    }
}
