// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GPU telemetry records and exact phase-boundary snapshots.
//!
//! A dynamic metric map lets sources share one wire schema.

use std::collections::BTreeMap;

/// Canonical platform identifier for NVIDIA GPU telemetry.
pub const NVIDIA_GPU_TELEMETRY_PLATFORM: &str = "nvidia";
/// Canonical platform identifier for AMD GPU telemetry.
pub const AMD_GPU_TELEMETRY_PLATFORM: &str = "amd";
/// Platform identifier used when a source does not report its vendor.
pub const UNKNOWN_GPU_TELEMETRY_PLATFORM: &str = "unknown";

/// Static identity and placement metadata for one GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuMetadata {
    /// Device index local to the source node.
    pub gpu_index: i32,
    /// Stable vendor-provided GPU identifier.
    pub gpu_uuid: String,
    /// Human-readable model name.
    pub gpu_model_name: String,
    /// Optional PCI bus identifier.
    pub pci_bus_id: Option<String>,
    /// Optional device node/name.
    pub device: Option<String>,
    /// Optional source hostname.
    pub hostname: Option<String>,
    /// Optional Kubernetes namespace.
    pub namespace: Option<String>,
    /// Optional Kubernetes pod name.
    pub pod_name: Option<String>,
    /// Canonical telemetry platform (`nvidia`, `amd`, or `unknown`).
    pub platform: String,
}

/// Stable key for one GPU series at one source endpoint.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GpuSeriesKey {
    /// Credential-free source endpoint.
    pub endpoint_url: String,
    /// GPU UUID within the endpoint.
    pub gpu_uuid: String,
}

/// All available signals for one GPU in one scrape.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuTelemetryRecord {
    /// Clock timestamp shared by every sample in the scrape.
    pub timestamp_ns: i64,
    /// Credential-free source endpoint.
    pub endpoint_url: String,
    /// Static GPU metadata.
    pub metadata: GpuMetadata,
    /// Finite, scaled values keyed by normalized telemetry name.
    pub metrics: BTreeMap<String, f64>,
}

impl GpuTelemetryRecord {
    /// Returns the stable endpoint/GPU key for this record.
    pub fn series_key(&self) -> GpuSeriesKey {
        GpuSeriesKey {
            endpoint_url: self.endpoint_url.clone(),
            gpu_uuid: self.metadata.gpu_uuid.clone(),
        }
    }
}

/// One decoded scrape from a telemetry source.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuScrape {
    /// One Clock timestamp for the complete scrape.
    pub timestamp_ns: i64,
    /// Credential-free source endpoint.
    pub endpoint_url: String,
    /// One record per GPU with at least one supported signal.
    pub records: Vec<GpuTelemetryRecord>,
}

/// Exact counter values captured by a synchronous phase barrier.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuBoundarySnapshot {
    /// Clock timestamp of the forced scrape.
    pub timestamp_ns: i64,
    /// Per-GPU counter values keyed by normalized signal name.
    pub counters: BTreeMap<GpuSeriesKey, BTreeMap<String, f64>>,
}

impl GpuBoundarySnapshot {
    /// Extracts all known finite counters from a decoded scrape.
    pub fn from_scrape(scrape: &GpuScrape) -> Self {
        let counters = scrape
            .records
            .iter()
            .filter_map(|record| {
                let values = record
                    .metrics
                    .iter()
                    .filter(|(name, value)| {
                        value.is_finite()
                            && crate::gpu_telemetry::fields::metric_spec(name).is_some_and(|spec| {
                                spec.kind == crate::gpu_telemetry::GpuMetricKind::Counter
                            })
                    })
                    .map(|(name, value)| (name.clone(), *value))
                    .collect::<BTreeMap<_, _>>();
                (!values.is_empty()).then(|| (record.series_key(), values))
            })
            .collect();
        Self {
            timestamp_ns: scrape.timestamp_ns,
            counters,
        }
    }

    /// Returns one counter value for an endpoint/GPU series.
    pub fn counter(&self, key: &GpuSeriesKey, name: &str) -> Option<f64> {
        self.counters
            .get(key)
            .and_then(|values| values.get(name))
            .copied()
            .filter(|value| value.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_telemetry::fields::normalize_legacy_nvidia_metric_names;
    use std::collections::BTreeMap;

    #[test]
    fn platform_constants_match_python() {
        assert_eq!(NVIDIA_GPU_TELEMETRY_PLATFORM, "nvidia");
        assert_eq!(AMD_GPU_TELEMETRY_PLATFORM, "amd");
        assert_eq!(UNKNOWN_GPU_TELEMETRY_PLATFORM, "unknown");
    }

    #[test]
    fn legacy_nvidia_aliases_normalize_at_ingest_only() {
        let input = BTreeMap::from([
            ("gpu_power_usage".to_string(), 75.5),
            ("energy_consumption".to_string(), 1.25),
            ("nvidia_sm_utilization".to_string(), 91.0),
        ]);
        let normalized = normalize_legacy_nvidia_metric_names(input);
        assert_eq!(normalized["nvidia_power_usage"], 75.5);
        assert_eq!(normalized["nvidia_energy_consumption"], 1.25);
        assert_eq!(normalized["nvidia_sm_utilization"], 91.0);
        assert!(!normalized.contains_key("gpu_power_usage"));
        assert!(!normalized.contains_key("energy_consumption"));
    }
}
