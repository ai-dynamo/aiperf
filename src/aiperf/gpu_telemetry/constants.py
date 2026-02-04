# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants specific to GPU telemetry collection."""

from aiperf.common.enums import (
    EnergyMetricUnit,
    GenericMetricUnit,
    MetricSizeUnit,
    MetricTimeUnit,
    MetricUnitT,
    PowerMetricUnit,
    TemperatureMetricUnit,
)

# Unit conversion scaling factors
SCALING_FACTORS = {
    "energy_consumption": 1e-9,  # mJ to MJ
    "gpu_memory_used": 1.048576 * 1e-3,  # MiB to GB
}

# DCGM field mapping to telemetry record fields
DCGM_TO_FIELD_MAPPING = {
    "DCGM_FI_DEV_POWER_USAGE": "gpu_power_usage",
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION": "energy_consumption",
    "DCGM_FI_DEV_GPU_UTIL": "gpu_utilization",
    "DCGM_FI_DEV_FB_USED": "gpu_memory_used",
    "DCGM_FI_DEV_GPU_TEMP": "gpu_temperature",
    "DCGM_FI_DEV_XID_ERRORS": "xid_errors",
    "DCGM_FI_DEV_POWER_VIOLATION": "power_violation",
}

# GPU Telemetry Metrics Configuration
# Format: (display_name, field_name, unit_enum)
# - display_name: Human-readable metric name shown in outputs
# - field_name: Corresponds to TelemetryMetrics model field name
# - unit_enum: MetricUnitT enum (use .value in exporters to get string)
GPU_TELEMETRY_METRICS_CONFIG: list[tuple[str, str, MetricUnitT]] = [
    ("GPU Power Usage", "gpu_power_usage", PowerMetricUnit.WATT),
    ("Energy Consumption", "energy_consumption", EnergyMetricUnit.MEGAJOULE),
    ("GPU Utilization", "gpu_utilization", GenericMetricUnit.PERCENT),
    ("GPU Memory Used", "gpu_memory_used", MetricSizeUnit.GIGABYTES),
    ("GPU Temperature", "gpu_temperature", TemperatureMetricUnit.CELSIUS),
    ("XID Errors", "xid_errors", GenericMetricUnit.COUNT),
    ("Power Violation", "power_violation", MetricTimeUnit.MICROSECONDS),
]

# Metrics that are cumulative counters (need delta calculation).
# These metrics accumulate over time (e.g., total energy consumed since boot),
# so we compute the delta between baseline and final values rather than statistics.
GPU_TELEMETRY_COUNTER_METRICS: set[str] = {
    "energy_consumption",
    "xid_errors",
    "power_violation",
}


def get_gpu_telemetry_metrics_config() -> list[tuple[str, str, MetricUnitT]]:
    """Get the current GPU telemetry metrics configuration."""
    return GPU_TELEMETRY_METRICS_CONFIG
