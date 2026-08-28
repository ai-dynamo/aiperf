// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract tests for plugin lock digest and catalog provenance in reports.

use aiperf_runtime::metrics_core::{AccumulatorSummary, NativeReport, PluginCatalogEntry};

fn make_catalog_entry(id: &str) -> PluginCatalogEntry {
    PluginCatalogEntry {
        package_id: id.to_string(),
        version: "1.0.0".to_string(),
        status: "winner".to_string(),
        manifest_digest: "abc123def456".to_string(),
    }
}

#[test]
fn plugin_catalog_entry_no_absolute_paths() {
    let entry = make_catalog_entry("vendor/my-exporter");
    let json = serde_json::to_string(&entry).unwrap();
    assert!(!json.contains("/usr/"));
    assert!(!json.contains("/home/"));
    assert!(!json.contains("/etc/"));
    assert!(!json.contains("/var/"));
    // package_id may contain a slash (namespace separator) — that's OK
    assert!(json.contains("vendor/my-exporter"));
}

#[test]
fn report_omits_plugin_fields_when_none() {
    let report = NativeReport::new(&AccumulatorSummary::new(), None);
    let json = serde_json::to_value(&report).unwrap();
    assert!(
        json.get("plugin_lock_digest").is_none(),
        "plugin_lock_digest should be absent when None"
    );
    assert!(
        json.get("plugin_catalog").is_none(),
        "plugin_catalog should be absent when None"
    );
}

#[test]
fn report_includes_plugin_fields_when_set() {
    let report = NativeReport::new(&AccumulatorSummary::new(), None)
        .with_plugin_provenance(
            "blake3:aabbccdd".to_string(),
            vec![make_catalog_entry("acme/exporter")],
        );
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(
        json["plugin_lock_digest"],
        serde_json::Value::String("blake3:aabbccdd".to_string())
    );
    let catalog = json["plugin_catalog"].as_array().unwrap();
    assert_eq!(catalog.len(), 1);
    assert_eq!(catalog[0]["package_id"], "acme/exporter");
    assert_eq!(catalog[0]["status"], "winner");
}

#[test]
fn report_with_empty_catalog_omits_field() {
    let report = NativeReport::new(&AccumulatorSummary::new(), None)
        .with_plugin_provenance("blake3:aabb".to_string(), vec![]);
    let json = serde_json::to_value(&report).unwrap();
    // lock digest is set
    assert!(json.get("plugin_lock_digest").is_some());
    // empty catalog skipped
    assert!(
        json.get("plugin_catalog").is_none(),
        "empty catalog should be skipped"
    );
}

#[test]
fn distribution_id_unchanged_when_plugin_lock_set() {
    use aiperf_runtime::metrics_core::{
        ReportEndpointProfileIdentity, ReportPairRunFacts, ReportRunMetadata,
    };

    let run_metadata = ReportRunMetadata::new(
        format!("blake3:{}", "a".repeat(64)),
        "online_http",
        "scheduled",
        Vec::new(),
        vec![ReportEndpointProfileIdentity::new("default", "chat").unwrap()],
    )
    .unwrap();

    let report_a = NativeReport::new(&AccumulatorSummary::new(), None)
        .finalize_run(
            run_metadata.clone(),
            ReportPairRunFacts::default(),
        )
        .unwrap();

    let report_b = NativeReport::new(&AccumulatorSummary::new(), None)
        .with_plugin_provenance("blake3:deadbeef".to_string(), vec![])
        .finalize_run(run_metadata, ReportPairRunFacts::default())
        .unwrap();

    let json_a = serde_json::to_value(&report_a).unwrap();
    let json_b = serde_json::to_value(&report_b).unwrap();

    // distribution_id must be identical
    assert_eq!(
        json_a["run"]["distribution_id"],
        json_b["run"]["distribution_id"],
        "plugin_lock_digest must not affect distribution_id"
    );

    // plugin_lock_digest only appears in b
    assert!(json_a.get("plugin_lock_digest").is_none());
    assert!(json_b.get("plugin_lock_digest").is_some());
}
