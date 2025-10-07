# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants specific to GPU telemetry collection."""

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
}
