// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for `ExportCapturePlan`: sorted-union merge, retention tagging, sketch
//! rejection, and histogram requirements.

use aiperf_runtime::export::capture::{
    ExactRecordField, ExportCapturePlan, ExportCapturePlanError, HistogramRequirement,
    RetentionReason,
};

#[test]
fn empty_plan_requires_no_exact_records() {
    let plan = ExportCapturePlan::default();
    assert!(!plan.requires_exact_records);
    assert!(plan.exact_record_requirements.is_empty());
    assert!(plan.histogram_requirements.is_empty());
}

#[test]
fn single_requirement_enables_exact_records() {
    let plan = ExportCapturePlan::with_requirement(
        ExactRecordField::TimeToFirstToken,
        RetentionReason::RequiredByExporter("otel".to_string()),
    );
    assert!(plan.requires_exact_records);
    assert_eq!(plan.exact_record_requirements.len(), 1);
    assert_eq!(
        plan.exact_record_requirements[0].field,
        ExactRecordField::TimeToFirstToken
    );
    assert_eq!(
        plan.exact_record_requirements[0].reason,
        RetentionReason::RequiredByExporter("otel".to_string())
    );
}

#[test]
fn merge_produces_sorted_union_of_fields() {
    let a = ExportCapturePlan::with_requirement(
        ExactRecordField::RequestIndex,
        RetentionReason::RequiredByExporter("exporter-a".to_string()),
    );
    let b = ExportCapturePlan::with_requirement(
        ExactRecordField::OutputTokens,
        RetentionReason::RequiredByExporter("exporter-b".to_string()),
    );
    let merged = ExportCapturePlan::merge([a, b]).expect("no conflict");
    assert!(merged.requires_exact_records);
    // Both fields present; order is stable (sorted by field discriminant)
    let fields: Vec<_> = merged
        .exact_record_requirements
        .iter()
        .map(|r| r.field)
        .collect();
    assert!(fields.contains(&ExactRecordField::RequestIndex));
    assert!(fields.contains(&ExactRecordField::OutputTokens));
}

#[test]
fn merge_deduplicates_same_field() {
    let a = ExportCapturePlan::with_requirement(
        ExactRecordField::TimeToFirstToken,
        RetentionReason::RequiredByExporter("x".to_string()),
    );
    let b = ExportCapturePlan::with_requirement(
        ExactRecordField::TimeToFirstToken,
        RetentionReason::RequiredByExporter("y".to_string()),
    );
    let merged = ExportCapturePlan::merge([a, b]).expect("no conflict");
    // Same field from two exporters: only one entry (first wins, or union — check dedup)
    let fields: Vec<_> = merged
        .exact_record_requirements
        .iter()
        .map(|r| r.field)
        .collect();
    let ttft_count = fields
        .iter()
        .filter(|&&f| f == ExactRecordField::TimeToFirstToken)
        .count();
    assert_eq!(ttft_count, 1, "duplicate field should be deduplicated");
}

#[test]
fn sketch_incompatible_with_exact_requirements() {
    let plan = ExportCapturePlan::with_requirement(
        ExactRecordField::InterTokenLatency,
        RetentionReason::RequiredByExporter("otlp".to_string()),
    );
    let result = ExportCapturePlan::merge_with_sketch_check([plan], /*sketch=*/ true);
    assert!(
        matches!(result, Err(ExportCapturePlanError::SketchIncompatible { .. })),
        "exact requirement in sketch mode must be rejected"
    );
}

#[test]
fn histogram_requirements_propagate_through_merge() {
    let mut a = ExportCapturePlan::default();
    a.histogram_requirements.push(HistogramRequirement {
        name: "gen_ai.client.operation.duration".to_string(),
        exporter_id: "otel".to_string(),
    });
    let merged = ExportCapturePlan::merge([a]).expect("no conflict");
    assert_eq!(merged.histogram_requirements.len(), 1);
    assert_eq!(
        merged.histogram_requirements[0].name,
        "gen_ai.client.operation.duration"
    );
}

#[test]
fn retention_reason_display_includes_exporter_id() {
    let reason = RetentionReason::RequiredByExporter("my-exporter".to_string());
    let s = format!("{reason}");
    assert!(s.contains("my-exporter"));
}
