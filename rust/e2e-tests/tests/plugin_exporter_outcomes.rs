// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract tests for the deterministic exporter runner and its structured
//! per-exporter outcomes.
//!
//! The native-v2 report is the committed authority: it is written and renamed
//! before the exporter plane runs at all. These tests pin the three properties a
//! plugin-supplied exporter must be able to rely on:
//!
//! 1. one failing exporter never short-circuits the ones after it,
//! 2. an already-committed report is never reverted by an exporter failure, and
//! 3. outcomes come back in the registry's deterministic descriptor order.

use std::path::Path;
use std::sync::Arc;

use aiperf_runtime::export::{ExportConfig, Exporter, ExporterRegistry};
use aiperf_runtime::metrics_core::{AccumulatorSummary, NativeReport, ReportView};

/// Exporter that always fails with a fixed message.
struct FailingExporter {
    name: &'static str,
}

impl Exporter for FailingExporter {
    fn name(&self) -> &'static str {
        self.name
    }

    fn enabled(&self, _cfg: &ExportConfig) -> bool {
        true
    }

    fn export(
        &self,
        _report: &dyn ReportView,
        _artifact_dir: &Path,
        _cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        anyhow::bail!("deliberate exporter failure from {}", self.name)
    }
}

/// Exporter that writes one marker file into the artifact directory.
struct MarkerExporter {
    name: &'static str,
}

impl Exporter for MarkerExporter {
    fn name(&self) -> &'static str {
        self.name
    }

    fn enabled(&self, _cfg: &ExportConfig) -> bool {
        true
    }

    fn export(
        &self,
        _report: &dyn ReportView,
        artifact_dir: &Path,
        _cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        std::fs::write(artifact_dir.join(format!("{}.marker", self.name)), b"ok")?;
        Ok(())
    }
}

/// Exporter that is registered but never selected by the policy.
struct DisabledExporter {
    name: &'static str,
}

impl Exporter for DisabledExporter {
    fn name(&self) -> &'static str {
        self.name
    }

    fn enabled(&self, _cfg: &ExportConfig) -> bool {
        false
    }

    fn export(
        &self,
        _report: &dyn ReportView,
        _artifact_dir: &Path,
        _cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        anyhow::bail!("disabled exporter must never run")
    }
}

fn temp_artifact_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("temp artifact dir")
}

fn empty_report() -> NativeReport {
    NativeReport::new(&AccumulatorSummary::new(), None)
}

#[test]
fn exporter_failure_continues_to_next() {
    let mut registry = ExporterRegistry::new();
    registry
        .register(0, Arc::new(FailingExporter { name: "first_fail" }))
        .expect("register failing exporter");
    registry
        .register(1, Arc::new(MarkerExporter { name: "second_ok" }))
        .expect("register marker exporter");

    let tmp = temp_artifact_dir();
    let dir = tmp.path();
    let outcomes = registry.run_collect(&empty_report(), dir, &ExportConfig::default());

    assert_eq!(outcomes.len(), 2, "both exporters must record an outcome");
    assert_eq!(outcomes[0].descriptor_id, "first_fail");
    assert!(!outcomes[0].success);
    let message = outcomes[0]
        .error_message
        .as_deref()
        .expect("failed outcome carries an error message");
    assert!(
        message.contains("deliberate exporter failure"),
        "error detail preserved: {message}"
    );

    assert_eq!(outcomes[1].descriptor_id, "second_ok");
    assert!(outcomes[1].success);
    assert!(outcomes[1].error_message.is_none());
    assert!(
        dir.join("second_ok.marker").exists(),
        "the exporter after a failure still ran"
    );
}

#[test]
fn failing_exporter_does_not_revert_report() {
    let tmp = temp_artifact_dir();
    let dir = tmp.path();
    // Stand in for the authoritative native-v2 report, which the coordinator has
    // already written and renamed by the time the exporter plane runs.
    let report_path = dir.join("profile_export_aiperf.json");
    let committed = br#"{"committed":true}"#;
    std::fs::write(&report_path, committed).expect("write committed report");

    let mut registry = ExporterRegistry::new();
    registry
        .register(0, Arc::new(FailingExporter { name: "blows_up" }))
        .expect("register failing exporter");

    let outcomes = registry.run_collect(&empty_report(), dir, &ExportConfig::default());

    assert_eq!(outcomes.len(), 1);
    assert!(!outcomes[0].success);
    assert_eq!(
        std::fs::read(&report_path).expect("committed report still readable"),
        committed,
        "an exporter failure must never revert the committed report"
    );
}

#[test]
fn exporter_outcomes_ordered_by_descriptor() {
    let mut registry = ExporterRegistry::new();
    // Registration order is deliberately scrambled relative to emit order.
    registry
        .register(20, Arc::new(MarkerExporter { name: "gamma" }))
        .expect("register gamma");
    registry
        .register(0, Arc::new(FailingExporter { name: "alpha" }))
        .expect("register alpha");
    registry
        .register(10, Arc::new(MarkerExporter { name: "beta" }))
        .expect("register beta");
    registry
        .register(5, Arc::new(DisabledExporter { name: "skipped" }))
        .expect("register skipped");

    let tmp = temp_artifact_dir();
    let dir = tmp.path();
    let cfg = ExportConfig::default();
    let first = registry.run_collect(&empty_report(), dir, &cfg);
    let second = registry.run_collect(&empty_report(), dir, &cfg);

    let ids: Vec<&str> = first
        .iter()
        .map(|outcome| outcome.descriptor_id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec!["alpha", "beta", "gamma"],
        "outcomes follow descriptor emit order, and a disabled descriptor is not run"
    );

    let repeated: Vec<&str> = second
        .iter()
        .map(|outcome| outcome.descriptor_id.as_str())
        .collect();
    assert_eq!(ids, repeated, "descriptor order is stable across runs");
}
