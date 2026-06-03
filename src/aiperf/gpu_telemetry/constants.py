# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants for GPU telemetry collection (DCGM, pynvml, and amdsmi)."""

from aiperf.common.enums import (
    EnergyMetricUnit,
    GenericMetricUnit,
    MetricSizeUnit,
    MetricTimeUnit,
    MetricUnitT,
    PowerMetricUnit,
    TemperatureMetricUnit,
)

# Source identifier for pynvml collector (used in TelemetryRecord.dcgm_url field)
PYNVML_SOURCE_IDENTIFIER = "pynvml://localhost"

# Source identifier for amdsmi collector (used in TelemetryRecord.dcgm_url field)
AMDSMI_SOURCE_IDENTIFIER = "amdsmi://localhost"

NVIDIA_GPU_TELEMETRY_PLATFORM = "nvidia"
AMD_GPU_TELEMETRY_PLATFORM = "amd"
UNKNOWN_GPU_TELEMETRY_PLATFORM = "unknown"

# Canonical namespaced NVIDIA telemetry field names. After AIP-905 namespacing,
# DCGM/pynvml samples are stored under these keys. The power-efficiency
# accumulator (GPUTelemetryAccumulator.compute_efficiency_metrics) resolves
# power and energy through these same constants, so a future rename updates one
# place and the producer (the mappings below) can never desync from the
# consumer (the accumulator).
NVIDIA_POWER_USAGE_FIELD = "nvidia_power_usage"
NVIDIA_ENERGY_CONSUMPTION_FIELD = "nvidia_energy_consumption"

# Canonical AMD telemetry field names, the AMD-side counterparts to the NVIDIA
# constants above. amdsmi populates these (power already in W, energy in MJ), and
# the power-efficiency accumulator resolves AMD power/energy through them.
AMD_POWER_FIELD = "amd_power"
AMD_ENERGY_CONSUMPTION_FIELD = "amd_energy_consumption"

NVIDIA_TELEMETRY_FIELD_ALIASES = {
    "gpu_power_usage": NVIDIA_POWER_USAGE_FIELD,
    "energy_consumption": NVIDIA_ENERGY_CONSUMPTION_FIELD,
    "gpu_utilization": "nvidia_gpu_utilization",
    "mem_utilization": "nvidia_memory_utilization",
    "gpu_memory_used": "nvidia_memory_used",
    "gpu_temperature": "nvidia_temperature",
    "decoder_utilization": "nvidia_decoder_utilization",
    "encoder_utilization": "nvidia_encoder_utilization",
    "jpg_utilization": "nvidia_jpg_utilization",
    "sm_utilization": "nvidia_sm_utilization",
    "xid_errors": "nvidia_xid_errors",
    "power_violation": "nvidia_power_violation",
}

# DCGM field mapping to telemetry record fields
DCGM_TO_FIELD_MAPPING = {
    "DCGM_FI_DEV_POWER_USAGE": NVIDIA_POWER_USAGE_FIELD,
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION": NVIDIA_ENERGY_CONSUMPTION_FIELD,
    "DCGM_FI_DEV_GPU_UTIL": "nvidia_gpu_utilization",
    "DCGM_FI_DEV_MEM_COPY_UTIL": "nvidia_memory_utilization",
    "DCGM_FI_DEV_FB_USED": "nvidia_memory_used",
    "DCGM_FI_DEV_GPU_TEMP": "nvidia_temperature",
    "DCGM_FI_DEV_ENC_UTIL": "nvidia_encoder_utilization",
    "DCGM_FI_DEV_DEC_UTIL": "nvidia_decoder_utilization",
    "DCGM_FI_PROF_SM_ACTIVE": "nvidia_sm_utilization",
    "DCGM_FI_DEV_XID_ERRORS": "nvidia_xid_errors",
    "DCGM_FI_DEV_POWER_VIOLATION": "nvidia_power_violation",
}

# GPU Telemetry Metrics Configuration
# Format: (display_name, field_name, unit_enum)
# - display_name: Human-readable metric name shown in outputs
# - field_name: Corresponds to TelemetryMetrics model field name
# - unit_enum: MetricUnitT enum (use .value in exporters to get string)
GPU_TELEMETRY_METRICS_CONFIG: list[tuple[str, str, MetricUnitT]] = [
    ("NVIDIA GPU Power Usage", NVIDIA_POWER_USAGE_FIELD, PowerMetricUnit.WATT),
    (
        "NVIDIA Energy Consumption",
        NVIDIA_ENERGY_CONSUMPTION_FIELD,
        EnergyMetricUnit.MEGAJOULE,
    ),
    ("NVIDIA GPU Utilization", "nvidia_gpu_utilization", GenericMetricUnit.PERCENT),
    (
        "NVIDIA Memory Utilization",
        "nvidia_memory_utilization",
        GenericMetricUnit.PERCENT,
    ),
    ("NVIDIA GPU Memory Used", "nvidia_memory_used", MetricSizeUnit.GIGABYTES),
    ("NVIDIA GPU Temperature", "nvidia_temperature", TemperatureMetricUnit.CELSIUS),
    (
        "NVIDIA SM Utilization",
        "nvidia_sm_utilization",
        GenericMetricUnit.PERCENT,
    ),
    (
        "NVIDIA Decoder Utilization",
        "nvidia_decoder_utilization",
        GenericMetricUnit.PERCENT,
    ),
    (
        "NVIDIA Encoder Utilization",
        "nvidia_encoder_utilization",
        GenericMetricUnit.PERCENT,
    ),
    (
        "NVIDIA JPEG Utilization",
        "nvidia_jpg_utilization",
        GenericMetricUnit.PERCENT,
    ),
    ("NVIDIA XID Errors", "nvidia_xid_errors", GenericMetricUnit.COUNT),
    (
        "NVIDIA Power Violation",
        "nvidia_power_violation",
        MetricTimeUnit.MICROSECONDS,
    ),
    # AMD ROCm telemetry (collected by AMDSMITelemetryCollector). These mirror
    # the amdsmi field names rather than NVML semantics, since the underlying
    # signals do not always measure the same physical quantity. Registered here
    # so accumulator/exporter/dashboard surface them end-to-end.
    ("AMD GPU Power", AMD_POWER_FIELD, PowerMetricUnit.WATT),
    (
        "AMD Energy Consumption",
        AMD_ENERGY_CONSUMPTION_FIELD,
        EnergyMetricUnit.MEGAJOULE,
    ),
    ("AMD GFX Activity", "amd_gfx_activity", GenericMetricUnit.PERCENT),
    ("AMD UMC Activity", "amd_umc_activity", GenericMetricUnit.PERCENT),
    ("AMD MM Activity", "amd_mm_activity", GenericMetricUnit.PERCENT),
    ("AMD GPU Memory Used", "amd_memory_used", MetricSizeUnit.GIGABYTES),
    ("AMD GPU Temperature", "amd_temperature", TemperatureMetricUnit.CELSIUS),
    ("AMD ECC Uncorrectable", "amd_ecc_uncorrectable", GenericMetricUnit.COUNT),
    ("AMD Throttle Status", "amd_throttle_status", GenericMetricUnit.COUNT),
]

# Metrics that are cumulative counters (need delta calculation).
# These metrics accumulate over time (e.g., total energy consumed since boot),
# so we compute the delta between baseline and final values rather than statistics.
GPU_TELEMETRY_COUNTER_METRICS: set[str] = {
    NVIDIA_ENERGY_CONSUMPTION_FIELD,
    "nvidia_xid_errors",
    "nvidia_power_violation",
    AMD_ENERGY_CONSUMPTION_FIELD,
    "amd_ecc_uncorrectable",
}


def get_gpu_telemetry_metrics_config() -> list[tuple[str, str, MetricUnitT]]:
    """Get the current GPU telemetry metrics configuration."""
    return GPU_TELEMETRY_METRICS_CONFIG
