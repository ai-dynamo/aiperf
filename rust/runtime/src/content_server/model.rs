// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serializable content-server status and request facts.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Sink for streaming each completed request record out of the tracker to a
/// live consumer (the media-fetch aggregator), independent of the bounded
/// retention buffer. Unbounded because the producer ([`RequestTracker::record`])
/// runs synchronously inside the HTTP response path and must not block on a full
/// queue; the consumer only parses and folds, so it keeps pace, and the run's own
/// request rate bounds the in-flight volume.
///
/// [`RequestTracker::record`]: crate::content_server::RequestTracker::record
pub type ContentRecordSender = tokio::sync::mpsc::UnboundedSender<ContentRequestRecord>;

/// One completed HTTP request served by the content server.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentRequestRecord {
    /// Wall-clock arrival time in nanoseconds since the Unix epoch.
    pub timestamp_ns: u64,
    /// HTTP method.
    pub method: String,
    /// Percent-decoded URL path.
    pub path: String,
    /// Raw query string without `?`.
    #[serde(default)]
    pub query_string: String,
    /// HTTP version (`0.9`, `1.0`, `1.1`, `2`, `3`, or a future debug spelling).
    #[serde(default = "default_http_version")]
    pub http_version: String,
    /// Client IP address.
    #[serde(default)]
    pub client_host: String,
    /// Client ephemeral port.
    #[serde(default)]
    pub client_port: u16,
    /// Lowercase request headers; duplicate values are joined with `, `.
    #[serde(default)]
    pub request_headers: BTreeMap<String, String>,
    /// HTTP response status.
    #[serde(default)]
    pub status_code: u16,
    /// Response content type.
    #[serde(default = "default_content_type")]
    pub content_type: String,
    /// Lowercase response headers; duplicate values are joined with `, `.
    #[serde(default)]
    pub response_headers: BTreeMap<String, String>,
    /// Actual non-empty response body bytes yielded to the HTTP connection.
    #[serde(default)]
    pub body_bytes: u64,
    /// Number of non-empty response body chunks yielded.
    #[serde(default)]
    pub body_chunk_count: u64,
    /// Total monotonic latency from arrival through terminal body polling.
    #[serde(default)]
    pub latency_ns: u64,
    /// Monotonic interval from arrival until response status/headers are ready.
    #[serde(default)]
    pub time_to_first_byte_ns: u64,
    /// Monotonic interval from arrival until the first non-empty body chunk.
    #[serde(default)]
    pub time_to_first_body_byte_ns: u64,
    /// Monotonic interval from first through last non-empty body chunk.
    #[serde(default)]
    pub transfer_duration_ns: u64,
    /// Body/connection failure observed after response construction.
    #[serde(default)]
    pub error: Option<String>,
}

fn default_http_version() -> String {
    "1.1".into()
}

fn default_content_type() -> String {
    "application/octet-stream".into()
}

/// Current server availability and serving root.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentServerStatus {
    /// Whether the HTTP listener is live.
    pub enabled: bool,
    /// Advertised base URL.
    #[serde(default)]
    pub base_url: String,
    /// Canonical directory being served.
    #[serde(default)]
    pub content_dir: PathBuf,
    /// Startup or terminal failure reason when disabled.
    #[serde(default)]
    pub reason: Option<String>,
}

/// Atomic snapshot of lifetime counters and the bounded recent-record buffer.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestTrackerSnapshot {
    /// Lifetime request count, including evicted records.
    pub total_requests: u64,
    /// Lifetime served body bytes, including evicted records.
    pub total_bytes_served: u64,
    /// Recent records in completion order.
    pub records: Vec<ContentRequestRecord>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn request_record_defaults_match_the_python_model() {
        let record: ContentRequestRecord = serde_json::from_value(json!({
            "timestamp_ns": 1,
            "method": "GET",
            "path": "/",
            "status_code": 404
        }))
        .unwrap();

        assert_eq!(record.query_string, "");
        assert_eq!(record.http_version, "1.1");
        assert_eq!(record.client_host, "");
        assert_eq!(record.client_port, 0);
        assert!(record.request_headers.is_empty());
        assert_eq!(record.content_type, "application/octet-stream");
        assert!(record.response_headers.is_empty());
        assert_eq!(record.body_bytes, 0);
        assert_eq!(record.body_chunk_count, 0);
        assert_eq!(record.latency_ns, 0);
        assert_eq!(record.time_to_first_byte_ns, 0);
        assert_eq!(record.time_to_first_body_byte_ns, 0);
        assert_eq!(record.transfer_duration_ns, 0);
        assert_eq!(record.error, None);
    }

    #[test]
    fn models_serialize_and_restore_without_losing_request_facts() {
        let record: ContentRequestRecord = serde_json::from_value(json!({
            "timestamp_ns": 1000,
            "method": "GET",
            "path": "/content/test.txt",
            "status_code": 200,
            "body_bytes": 42,
            "request_headers": {"host": "localhost"},
            "response_headers": {"content-type": "text/plain"}
        }))
        .unwrap();
        let restored: ContentRequestRecord =
            serde_json::from_value(serde_json::to_value(&record).unwrap()).unwrap();

        assert_eq!(restored, record);
        assert_eq!(RequestTrackerSnapshot::default().records, Vec::new());

        let disabled: ContentServerStatus = serde_json::from_value(json!({
            "enabled": false,
            "reason": "not initialized"
        }))
        .unwrap();
        assert_eq!(disabled.base_url, "");
        assert!(disabled.content_dir.as_os_str().is_empty());
        assert_eq!(disabled.reason.as_deref(), Some("not initialized"));
    }
}
