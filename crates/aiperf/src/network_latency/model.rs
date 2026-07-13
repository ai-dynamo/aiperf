// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed probe records and aggregate result shapes.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use serde::Serialize;

/// A unique TCP target derived from the first endpoint URL for one host/port.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NetworkLatencyTarget {
    /// Credential-redacted endpoint URL retained in compatibility artifacts.
    pub target_url: String,
    /// Parsed hostname used for DNS and TCP connect.
    pub target_host: String,
    /// Parsed or scheme-defaulted TCP port.
    pub target_port: u16,
}

impl NetworkLatencyTarget {
    /// Parse an endpoint using the Python manager's scheme/default-port rules.
    ///
    /// Sources without a host, such as Unix-domain socket URLs, return `None`
    /// and are not probe targets.
    pub fn from_endpoint_url(
        endpoint_url: &str,
    ) -> Result<Option<Self>, NetworkLatencyTargetParseError> {
        if endpoint_url.starts_with("unix:") {
            return Ok(None);
        }
        let parse_url = if endpoint_url.contains("://") {
            endpoint_url.to_string()
        } else {
            format!("http://{endpoint_url}")
        };
        let parsed =
            url::Url::parse(&parse_url).map_err(|source| NetworkLatencyTargetParseError {
                endpoint_url: endpoint_url.to_string(),
                message: source.to_string(),
            })?;
        let Some(host) = parsed.host_str() else {
            return Ok(None);
        };
        let port = parsed.port().unwrap_or_else(|| match parsed.scheme() {
            "https" => 443,
            _ => 80,
        });
        Ok(Some(Self {
            target_url: redact_url_userinfo(endpoint_url),
            target_host: host.to_string(),
            target_port: port,
        }))
    }

    /// Stable Python-compatible deduplication key.
    pub fn key(&self) -> String {
        format!("{}:{}", self.target_host, self.target_port)
    }
}

fn redact_url_userinfo(value: &str) -> String {
    let Some(scheme_end) = value.find("://") else {
        if let Some(at) = value.find('@')
            && value[..at].contains(':')
        {
            return format!("<redacted>@{}", &value[at + 1..]);
        }
        return value.to_string();
    };
    let authority_start = scheme_end + 3;
    let authority_end = value[authority_start..]
        .find(['/', '?', '#'])
        .map_or(value.len(), |offset| authority_start + offset);
    let authority = &value[authority_start..authority_end];
    let Some(at) = authority.rfind('@') else {
        return value.to_string();
    };
    format!(
        "{}<redacted>@{}",
        &value[..authority_start],
        &value[authority_start + at + 1..]
    )
}

/// Invalid endpoint URL supplied to target discovery.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NetworkLatencyTargetParseError {
    endpoint_url: String,
    message: String,
}

impl fmt::Display for NetworkLatencyTargetParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid network-latency endpoint {:?}: {}",
            self.endpoint_url, self.message
        )
    }
}

impl Error for NetworkLatencyTargetParseError {}

/// Python-compatible structured details for a failed connect probe.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NetworkLatencyErrorDetails {
    /// Optional platform error number.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<i32>,
    /// Stable exception-style category.
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
    /// Human-readable failure text.
    pub message: String,
    /// Nested cause text when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<String>,
    /// Ordered exception categories in a nested cause chain.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause_chain: Option<Vec<String>>,
}

/// One fresh TCP-handshake RTT observation.
///
/// Optional fields are omitted to match the Python buffered JSONL writer's
/// `exclude_none=True` behavior.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NetworkLatencySample {
    /// Clock timestamp immediately before connect issuance.
    pub timestamp_ns: i64,
    /// Credential-redacted source endpoint.
    pub target_url: String,
    /// Parsed target host.
    pub target_host: String,
    /// Parsed target port.
    pub target_port: u16,
    /// Probe mechanism; always `tcp_connect` in the built-in source.
    pub probe_type: &'static str,
    /// Successful connect duration in nanoseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rtt_ns: Option<i64>,
    /// Whether the TCP connection completed before its deadline.
    pub success: bool,
    /// Captured connection failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<NetworkLatencyErrorDetails>,
}

/// Distribution statistics over successful RTTs.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct NetworkLatencyStats {
    /// Minimum RTT.
    pub min_ns: Option<f64>,
    /// Arithmetic mean RTT.
    pub mean_ns: Option<f64>,
    /// Median RTT using NumPy-compatible linear interpolation.
    pub median_ns: Option<f64>,
    /// 90th percentile RTT using linear interpolation.
    pub p90_ns: Option<f64>,
    /// 99th percentile RTT using linear interpolation.
    pub p99_ns: Option<f64>,
    /// Population standard deviation (`ddof=0`).
    pub stddev_ns: Option<f64>,
}

/// Aggregate RTT statistics for one target.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NetworkLatencyTargetSummary {
    /// Credential-redacted source endpoint.
    pub target_url: String,
    /// Parsed target host.
    pub target_host: String,
    /// Parsed target port.
    pub target_port: u16,
    /// Total probes issued.
    pub count: usize,
    /// Successful probes.
    pub success_count: usize,
    /// Failed probes.
    pub failure_count: usize,
    /// Successful RTT distribution.
    #[serde(flatten)]
    pub stats: NetworkLatencyStats,
}

/// Count for one distinct structured probe failure.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NetworkLatencyErrorDetailsCount {
    /// Failure identity.
    pub error_details: NetworkLatencyErrorDetails,
    /// Number of matching failures.
    pub count: usize,
}

/// Full per-target and aggregate calibration result.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NetworkLatencyResults {
    /// Benchmark identifier shared across run artifacts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub benchmark_id: Option<String>,
    /// Per-target results keyed by `host:port`.
    pub target_summaries: BTreeMap<String, NetworkLatencyTargetSummary>,
    /// Total probes issued.
    pub count: usize,
    /// Successful probes.
    pub success_count: usize,
    /// Failed probes.
    pub failure_count: usize,
    /// Aggregate successful RTT distribution.
    #[serde(flatten)]
    pub stats: NetworkLatencyStats,
    /// Counts for distinct failures.
    pub error_summary: Vec<NetworkLatencyErrorDetailsCount>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_discovery_defaults_ports_and_redacts_userinfo() {
        let http = NetworkLatencyTarget::from_endpoint_url("host.test/v1")
            .unwrap()
            .unwrap();
        assert_eq!(http.target_host, "host.test");
        assert_eq!(http.target_port, 80);

        let https =
            NetworkLatencyTarget::from_endpoint_url("https://user:super-secret@host.test/v1")
                .unwrap()
                .unwrap();
        assert_eq!(https.target_port, 443);
        assert_eq!(https.target_url, "https://<redacted>@host.test/v1");
    }

    #[test]
    fn target_discovery_skips_hostless_urls() {
        assert!(
            NetworkLatencyTarget::from_endpoint_url("unix:/tmp/aiperf.sock")
                .unwrap()
                .is_none()
        );
    }
}
