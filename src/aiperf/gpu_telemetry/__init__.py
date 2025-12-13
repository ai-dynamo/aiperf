# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU telemetry collection module for AIPerf.

This module provides GPU telemetry collection capabilities through DCGM endpoints.
"""

from aiperf.gpu_telemetry.constants import (
    DCGM_TO_FIELD_MAPPING,
    SCALING_FACTORS,
    get_gpu_telemetry_metrics_config,
)
from aiperf.gpu_telemetry.metrics_config import (
    MetricsConfigLoader,
)
from aiperf.gpu_telemetry.telemetry_data_collector import (
    TelemetryDataCollector,
)
from aiperf.gpu_telemetry.telemetry_manager import (
    TelemetryManager,
)

__all__ = [
    "DCGM_TO_FIELD_MAPPING",
    "MetricsConfigLoader",
    "SCALING_FACTORS",
    "TelemetryDataCollector",
    "TelemetryManager",
    "get_gpu_telemetry_metrics_config",
]
