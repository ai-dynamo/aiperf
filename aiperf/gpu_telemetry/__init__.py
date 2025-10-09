# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU telemetry collection module for AIPerf.

This module provides GPU telemetry collection capabilities through DCGM endpoints.
"""

from aiperf.gpu_telemetry.constants import (
    DCGM_TO_FIELD_MAPPING,
    DEFAULT_COLLECTION_INTERVAL,
    DEFAULT_DCGM_ENDPOINT,
    SCALING_FACTORS,
    THREAD_JOIN_TIMEOUT,
    URL_REACHABILITY_TIMEOUT,
)
from aiperf.gpu_telemetry.telemetry_data_collector import (
    TelemetryDataCollector,
)
from aiperf.gpu_telemetry.telemetry_manager import (
    TelemetryManager,
)
from aiperf.gpu_telemetry.telemetry_metric_generator import (
    register_telemetry_metrics,
)

__all__ = [
    "DCGM_TO_FIELD_MAPPING",
    "DEFAULT_COLLECTION_INTERVAL",
    "DEFAULT_DCGM_ENDPOINT",
    "SCALING_FACTORS",
    "THREAD_JOIN_TIMEOUT",
    "TelemetryDataCollector",
    "TelemetryManager",
    "URL_REACHABILITY_TIMEOUT",
    "register_telemetry_metrics",
]
