// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static GPU signal, unit, and collector-scale tables.
//!
//! These tables port `src/aiperf/gpu_telemetry/constants.py:22-79`, the DCGM
//! scaling constants in `src/aiperf/gpu_telemetry/dcgm_collector.py:19-25`, and
//! the AMD units/scales documented by
//! `src/aiperf/gpu_telemetry/amdsmi_collector.py:51-60`.

use aiperf_metrics::Unit;

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

/// DCGM Prometheus field table.
pub const DCGM_METRICS: &[GpuMetricSpec] = &[
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_POWER_USAGE",
        name: "gpu_power_usage",
        header: "GPU Power Usage",
        unit: Unit::Watt,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
        name: "energy_consumption",
        header: "Energy Consumption",
        unit: Unit::Megajoule,
        scale: 1e-9,
        kind: GpuMetricKind::Counter,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_GPU_UTIL",
        name: "gpu_utilization",
        header: "GPU Utilization",
        unit: Unit::Percent,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_MEM_COPY_UTIL",
        name: "mem_utilization",
        header: "Memory Utilization",
        unit: Unit::Percent,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_FB_USED",
        name: "gpu_memory_used",
        header: "GPU Memory Used",
        unit: Unit::Gigabyte,
        scale: 1.048_576e-3,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_GPU_TEMP",
        name: "gpu_temperature",
        header: "GPU Temperature",
        unit: Unit::Celsius,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_ENC_UTIL",
        name: "encoder_utilization",
        header: "Encoder Utilization",
        unit: Unit::Percent,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_DEC_UTIL",
        name: "decoder_utilization",
        header: "Decoder Utilization",
        unit: Unit::Percent,
        scale: 1.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_PROF_SM_ACTIVE",
        name: "sm_utilization",
        header: "SM Utilization",
        unit: Unit::Percent,
        scale: 100.0,
        kind: GpuMetricKind::Gauge,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_XID_ERRORS",
        name: "xid_errors",
        header: "XID Errors",
        unit: Unit::Count,
        scale: 1.0,
        kind: GpuMetricKind::Counter,
    },
    GpuMetricSpec {
        source_field: "DCGM_FI_DEV_POWER_VIOLATION",
        name: "power_violation",
        header: "Power Violation",
        unit: Unit::Microsecond,
        scale: 1e-3,
        kind: GpuMetricKind::Counter,
    },
];

/// AMD/ROCm mirror table retained for a future local AMDSMI source.
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
                "energy_consumption",
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
