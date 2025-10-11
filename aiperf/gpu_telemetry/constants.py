# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants specific to GPU telemetry collection."""

from aiperf.common.enums.metric_enums import (
    EnergyMetricUnit,
    FrequencyMetricUnit,
    GenericMetricUnit,
    MetricSizeUnit,
    MetricTimeUnit,
    MetricUnitT,
    PowerMetricUnit,
    TemperatureMetricUnit,
)

# Default telemetry configuration
DEFAULT_DCGM_ENDPOINT = "http://localhost:9401/metrics"
DEFAULT_COLLECTION_INTERVAL = 0.33  # in seconds, 330ms (~3Hz)

# Timeouts for telemetry operations (seconds)
URL_REACHABILITY_TIMEOUT = 5
THREAD_JOIN_TIMEOUT = 5.0

# Unit conversion scaling factors
SCALING_FACTORS = {
    "energy_consumption": 1e-9,  # mJ to MJ
    "gpu_memory_used": 1.048576 * 1e-3,  # MiB to GB
    "gpu_memory_free": 1.048576 * 1e-3,  # MiB to GB
    "gpu_memory_total": 1.048576 * 1e-3,  # MiB to GB
}

# DCGM field mapping to telemetry record fields
DCGM_TO_FIELD_MAPPING = {
    "DCGM_FI_DEV_POWER_USAGE": "gpu_power_usage",
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION": "energy_consumption",
    "DCGM_FI_DEV_GPU_UTIL": "gpu_utilization",
    "DCGM_FI_DEV_FB_USED": "gpu_memory_used",
    "DCGM_FI_DEV_SM_CLOCK": "sm_clock_frequency",
    "DCGM_FI_DEV_MEM_CLOCK": "memory_clock_frequency",
    "DCGM_FI_DEV_MEMORY_TEMP": "memory_temperature",
    "DCGM_FI_DEV_GPU_TEMP": "gpu_temperature",
    "DCGM_FI_DEV_MEM_COPY_UTIL": "memory_copy_utilization",
    "DCGM_FI_DEV_XID_ERRORS": "xid_errors",
    "DCGM_FI_DEV_POWER_VIOLATION": "power_violation",
    "DCGM_FI_DEV_THERMAL_VIOLATION": "thermal_violation",
    "DCGM_FI_DEV_POWER_MGMT_LIMIT": "power_management_limit",
    "DCGM_FI_DEV_FB_FREE": "gpu_memory_free",
    "DCGM_FI_DEV_FB_TOTAL": "gpu_memory_total",
}

# GPU Telemetry Metrics Configuration
# Format: (display_name, field_name, unit_enum)
# - display_name: Human-readable metric name shown in outputs
# - field_name: Corresponds to TelemetryMetrics model field name
# - unit_enum: MetricUnitT enum (use .value in exporters to get string)
GPU_TELEMETRY_METRICS_CONFIG: list[tuple[str, str, MetricUnitT]] = [
    ("GPU Power Usage", "gpu_power_usage", PowerMetricUnit.WATT),
    ("GPU Power Limit", "power_management_limit", PowerMetricUnit.WATT),
    ("Energy Consumption", "energy_consumption", EnergyMetricUnit.MEGAJOULE),
    ("GPU Utilization", "gpu_utilization", GenericMetricUnit.PERCENT),
    ("Memory Copy Utilization", "memory_copy_utilization", GenericMetricUnit.PERCENT),
    ("GPU Memory Used", "gpu_memory_used", MetricSizeUnit.GIGABYTES),
    ("GPU Memory Free", "gpu_memory_free", MetricSizeUnit.GIGABYTES),
    ("GPU Memory Total", "gpu_memory_total", MetricSizeUnit.GIGABYTES),
    ("SM Clock Frequency", "sm_clock_frequency", FrequencyMetricUnit.MEGAHERTZ),
    ("Memory Clock Frequency", "memory_clock_frequency", FrequencyMetricUnit.MEGAHERTZ),
    ("Memory Temperature", "memory_temperature", TemperatureMetricUnit.CELSIUS),
    ("GPU Temperature", "gpu_temperature", TemperatureMetricUnit.CELSIUS),
    ("XID Errors", "xid_errors", GenericMetricUnit.COUNT),
    ("Power Violation", "power_violation", MetricTimeUnit.MICROSECONDS),
    ("Thermal Violation", "thermal_violation", MetricTimeUnit.MICROSECONDS),
]
