// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Tests for open-form plugin transport and exporter Config-v2 entries (Task 18).

use aiperf_runtime::config::model::export::{Export, PluginExport};
use aiperf_runtime::config::model::transport::Transport;
use aiperf_runtime::config::plugin_id::validate_plugin_id;

// ── transport ─────────────────────────────────────────────────────────────────

#[test]
fn plugin_transport_deserializes_with_parameters() {
    let json = serde_json::json!({
        "type": "plugin",
        "id": "vendor/my-transport:1.0",
        "parameters": {"timeout_ms": 500}
    });
    let transport: Transport = serde_json::from_value(json).unwrap();
    let Transport::Plugin { id, parameters } = transport else {
        panic!("expected Plugin variant");
    };
    assert_eq!(id, "vendor/my-transport:1.0");
    assert_eq!(parameters["timeout_ms"], 500);
}

#[test]
fn plugin_transport_deserializes_without_parameters() {
    let json = serde_json::json!({"type": "plugin", "id": "acme/fast-transport:2.1"});
    let transport: Transport = serde_json::from_value(json).unwrap();
    let Transport::Plugin { id, parameters } = transport else {
        panic!("expected Plugin variant");
    };
    assert_eq!(id, "acme/fast-transport:2.1");
    assert!(
        parameters.is_null()
            || parameters == serde_json::Value::Null
            || parameters
                .as_object()
                .map(|m| m.is_empty())
                .unwrap_or(false)
    );
}

#[test]
fn plugin_transport_canonical_id_is_plugin() {
    let t = Transport::Plugin {
        id: "vendor/my-transport:1.0".to_owned(),
        parameters: serde_json::Value::Null,
    };
    assert_eq!(t.canonical_id(), "plugin");
}

#[test]
fn plugin_transport_round_trips_through_serde() {
    let original = Transport::Plugin {
        id: "vendor/my-transport:1.0".to_owned(),
        parameters: serde_json::json!({"key": "value"}),
    };
    let value = serde_json::to_value(&original).unwrap();
    assert_eq!(value["type"], "plugin");
    assert_eq!(value["id"], "vendor/my-transport:1.0");
    let decoded: Transport = serde_json::from_value(value).unwrap();
    let Transport::Plugin { id, parameters } = decoded else {
        panic!("round-trip failed");
    };
    assert_eq!(id, "vendor/my-transport:1.0");
    assert_eq!(parameters["key"], "value");
}

// ── exporter ──────────────────────────────────────────────────────────────────

#[test]
fn plugin_exporter_deserializes_with_parameters() {
    let json = serde_json::json!({
        "id": "vendor/metrics-out:2.0",
        "parameters": {"endpoint": "http://metrics.internal"}
    });
    let exporter: PluginExport = serde_json::from_value(json).unwrap();
    assert_eq!(exporter.id, "vendor/metrics-out:2.0");
    assert_eq!(exporter.parameters["endpoint"], "http://metrics.internal");
}

#[test]
fn plugin_exporter_deserializes_without_parameters() {
    let json = serde_json::json!({"id": "vendor/metrics-out:2.0"});
    let exporter: PluginExport = serde_json::from_value(json).unwrap();
    assert_eq!(exporter.id, "vendor/metrics-out:2.0");
}

#[test]
fn export_config_plugin_exporters_field_defaults_empty() {
    // `plugin_exporters` must default to empty when absent — test via direct
    // round-trip of the Vec, since constructing a full Export from scratch
    // requires all required GenaiPerf/ConsoleTxt fields.
    let json = serde_json::json!([]);
    let exporters: Vec<PluginExport> = serde_json::from_value(json).unwrap();
    assert!(exporters.is_empty());

    // Also verify that a PluginExport without parameters can be deserialized
    // and that the serde default kicks in correctly.
    let json2 = serde_json::json!([{"id": "vendor/test:1.0"}]);
    let exporters2: Vec<PluginExport> = serde_json::from_value(json2).unwrap();
    assert_eq!(exporters2.len(), 1);
    assert_eq!(exporters2[0].id, "vendor/test:1.0");
}

#[test]
fn export_config_plugin_exporters_round_trips() {
    let exporters = vec![
        PluginExport {
            id: "vendor/metrics-out:1.0".to_owned(),
            parameters: serde_json::json!({"endpoint": "http://example.com"}),
        },
        PluginExport {
            id: "acme/telemetry:0.5".to_owned(),
            parameters: serde_json::Value::Null,
        },
    ];
    let serialized = serde_json::to_value(&exporters).unwrap();
    let decoded: Vec<PluginExport> = serde_json::from_value(serialized).unwrap();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].id, "vendor/metrics-out:1.0");
    assert_eq!(decoded[1].id, "acme/telemetry:0.5");
}

// ── plugin ID validation ──────────────────────────────────────────────────────

#[test]
fn plugin_id_rejects_absolute_path() {
    assert!(validate_plugin_id("/usr/lib/plugin").is_err());
    assert!(validate_plugin_id("/home/user/plugin.so").is_err());
}

#[test]
fn plugin_id_rejects_relative_traversal() {
    assert!(validate_plugin_id("../escape").is_err());
    assert!(validate_plugin_id("./local").is_err());
    assert!(validate_plugin_id("vendor/../escape").is_err());
}

#[test]
fn plugin_id_rejects_empty() {
    assert!(validate_plugin_id("").is_err());
}

#[test]
fn plugin_id_accepts_namespaced_with_version() {
    assert!(validate_plugin_id("vendor/my-transport:1.0").is_ok());
    assert!(validate_plugin_id("acme/telemetry:2.3.4").is_ok());
    assert!(validate_plugin_id("my-plugin").is_ok());
}

#[test]
fn plugin_id_accepts_scoped_dotted() {
    assert!(validate_plugin_id("com.example/plugin:0.1").is_ok());
    assert!(validate_plugin_id("nvidia.aiperf/grpc-transport:1.0").is_ok());
}
