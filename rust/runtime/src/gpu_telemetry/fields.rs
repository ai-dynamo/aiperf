// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static GPU signal, unit, and collector-scale tables.
//!
//! These tables define the DCGM signal set and scaling constants plus the AMD
//! units/scales.

use crate::metrics_core::Unit;
use std::collections::BTreeMap;

/// Legacy unprefixed NVIDIA names accepted only while reading telemetry.
pub const LEGACY_NVIDIA_METRIC_ALIASES: &[(&str, &str)] = &[
    ("gpu_power_usage", "nvidia_power_usage"),
    ("energy_consumption", "nvidia_energy_consumption"),
    ("gpu_utilization", "nvidia_gpu_utilization"),
    ("mem_utilization", "nvidia_memory_utilization"),
    ("gpu_memory_used", "nvidia_memory_used"),
    ("gpu_temperature", "nvidia_temperature"),
    ("decoder_utilization", "nvidia_decoder_utilization"),
    ("encoder_utilization", "nvidia_encoder_utilization"),
    ("jpg_utilization", "nvidia_jpg_utilization"),
    ("sm_utilization", "nvidia_sm_utilization"),
    ("xid_errors", "nvidia_xid_errors"),
    ("power_violation", "nvidia_power_violation"),
];

/// Normalizes legacy NVIDIA keys at an ingest seam without re-emitting aliases.
///
/// A canonical value wins when both forms are present.
pub fn normalize_legacy_nvidia_metric_names(
    mut metrics: BTreeMap<String, f64>,
) -> BTreeMap<String, f64> {
    for (legacy, canonical) in LEGACY_NVIDIA_METRIC_ALIASES {
        if let Some(value) = metrics.remove(*legacy) {
            metrics.entry((*canonical).to_string()).or_insert(value);
        }
    }
    metrics
}

/// Whether a GPU signal is sampled as a gauge or snapshotted as a counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMetricKind {
    /// Point-in-time value summarized over the authoritative phase window.
    Gauge,
    /// Monotonic value summarized from exact phase-boundary snapshots.
    Counter,
}

/// One static source-field mapping and its normalized representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuMetricSpec {
    /// Exporter or collector field name.
    pub source_field: &'static str,
    /// Stable AIPerf telemetry name.
    pub name: &'static str,
    /// Human-readable display name.
    pub header: &'static str,
    /// Unit after collector scaling.
    pub unit: Unit,
    /// Multiplier applied to the source value.
    pub scale: f64,
    /// Gauge/counter accumulation policy.
    pub kind: GpuMetricKind,
}

/// Owned metric description supplied by a runtime or Python extension worker.
///
/// Static DCGM/AMDSMI fields and Config-v2 custom metric files converge on
/// this representation before accumulation, so report construction never
/// branches on the source implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeGpuMetricSpec {
    /// Stable normalized telemetry field name.
    pub name: String,
    /// Human-readable report header.
    pub header: String,
    /// Unit after source-side scaling.
    pub unit: Unit,
    /// Gauge/counter accumulation policy.
    pub kind: GpuMetricKind,
}

impl From<&GpuMetricSpec> for RuntimeGpuMetricSpec {
    fn from(spec: &GpuMetricSpec) -> Self {
        Self {
            name: spec.name.to_string(),
            header: spec.header.to_string(),
            unit: spec.unit,
            kind: spec.kind,
        }
    }
}

/// DCGM Prometheus field table.
pub const DCGM_METRICS: &[GpuMetricSpec] = &[
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_POWER_USAGE",
        name: "nvidia_power_usage",
        header: "NVIDIA GPU Power Usage",
        unit: Unit::Watt,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
        name: "nvidia_energy_consumption",
        header: "NVIDIA Energy Consumption",
        unit: Unit::Megajoule,
        scale: 1e-9,
        kind: GpuMetricKind::Counter,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_GPU_UTIL",
        name: "nvidia_gpu_utilization",
        header: "NVIDIA GPU Utilization",
        unit: Unit::Percent,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_MEM_COPY_UTIL",
        name: "nvidia_memory_utilization",
        header: "NVIDIA Memory Utilization",
        unit: Unit::Percent,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_FB_USED",
        name: "nvidia_memory_used",
        header: "NVIDIA GPU Memory Used",
        unit: Unit::Gigabyte,
        scale: 1.048_576e-3,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_GPU_TEMP",
        name: "nvidia_temperature",
        header: "NVIDIA GPU Temperature",
        unit: Unit::Celsius,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_ENC_UTIL",
        name: "nvidia_encoder_utilization",
        header: "NVIDIA Encoder Utilization",
        unit: Unit::Percent,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_DEC_UTIL",
        name: "nvidia_decoder_utilization",
        header: "NVIDIA Decoder Utilization",
        unit: Unit::Percent,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_PROF_SM_ACTIVE",
        name: "nvidia_sm_utilization",
        header: "NVIDIA SM Utilization",
        unit: Unit::Percent,
        scale: 100.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_XID_ERRORS",
        name: "nvidia_xid_errors",
        header: "NVIDIA XID Errors",
        unit: Unit::Count,
        scale: 1.0,
        kind: GpuMetricKind::Counter,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_POWER_VIOLATION",
        name: "nvidia_power_violation",
        header: "NVIDIA Power Violation",
        unit: Unit::Microsecond,
        scale: 1e-3,
        kind: GpuMetricKind::Counter,
    },
];

/// AMD/ROCm metric definitions.
pub const AMD_METRICS: &[GpuMetricSpec] = &[
    GpuMetricSpec {
        source_field: "amd_power",
        name: "amd_power",
        header: "AMD GPU Power",
        unit: Unit::Watt,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "amd_energy_consumption",
        name: "amd_energy_consumption",
        header: "AMD Energy Consumption",
        unit: Unit::Megajoule,
        scale: 1e-12,
        kind: GpuMetricKind::Counter,
    },
    GpuMetricSpec {
        source_field: "amd_gfx_activity",
        name: "amd_gfx_activity",
        header: "AMD GFX Activity",
        unit: Unit::Percent,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "amd_umc_activity",
        name: "amd_umc_activity",
        header: "AMD UMC Activity",
        unit: Unit::Percent,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "amd_mm_activity",
        name: "amd_mm_activity",
        header: "AMD MM Activity",
        unit: Unit::Percent,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "amd_memory_used",
        name: "amd_memory_used",
        header: "AMD GPU Memory Used",
        unit: Unit::Gigabyte,
        scale: 1e-9,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "amd_temperature",
        name: "amd_temperature",
        header: "AMD GPU Temperature",
        unit: Unit::Celsius,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "amd_ecc_uncorrectable",
        name: "amd_ecc_uncorrectable",
        header: "AMD ECC Uncorrectable",
        unit: Unit::Count,
        scale: 1.0,
        kind: GpuMetricKind::Counter,
    },
    GpuMetricSpec {
        source_field: "amd_throttle_status",
        name: "amd_throttle_status",
        header: "AMD Throttle Status",
        unit: Unit::Count,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
];

/// Resolves a DCGM exporter field after an optional `_total` suffix is removed.
pub fn dcgm_metric_spec(source_field: &str) -> Option<&'static GpuMetricSpec> {
    DCGM_METRICS
        .iter()
        .find(|spec| spec.source_field == source_field)
}

/// Resolves an AMD collector field.
pub fn amd_metric_spec(source_field: &str) -> Option<&'static GpuMetricSpec> {
    AMD_METRICS
        .iter()
        .find(|spec| spec.source_field == source_field)
}

/// Resolves a normalized telemetry metric name across supported vendors.
pub fn metric_spec(name: &str) -> Option<&'static GpuMetricSpec> {
    DCGM_METRICS
        .iter()
        .chain(AMD_METRICS)
        .find(|spec| spec.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dcgm_scale_table_retains_source_units() {
        assert_eq!(
            dcgm_metric_spec("DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION")
                .map(|spec| (spec.name, spec.unit, spec.scale, spec.kind)),
            Some((
                "nvidia_energy_consumption",
                Unit::Megajoule,
                1e-9,
                GpuMetricKind::Counter,
            ))
        );
        assert_eq!(
            dcgm_metric_spec("DCGM_FI_PROF_SM_ACTIVE").map(|spec| spec.scale),
            Some(100.0)
        );
        assert_eq!(
            dcgm_metric_spec("DCGM_FI_DEV_FB_USED").map(|spec| spec.scale),
            Some(1.048_576e-3)
        );
        assert_eq!(
            dcgm_metric_spec("DCGM_FI_DEV_POWER_VIOLATION").map(|spec| spec.scale),
            Some(1e-3)
        );
    }
}
