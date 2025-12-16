# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU telemetry collection module for AIPerf.

This module provides GPU telemetry collection capabilities through DCGM endpoints.
"""

from aiperf.gpu_telemetry.accumulator import (
    GPUTelemetryAccumulator,
)
from aiperf.gpu_telemetry.constants import (
    DCGM_TO_FIELD_MAPPING,
    GPU_TELEMETRY_METRICS_CONFIG,
    SCALING_FACTORS,
    get_gpu_telemetry_metrics_config,
)
from aiperf.gpu_telemetry.data_collector import (
    GPUTelemetryDataCollector,
)
from aiperf.gpu_telemetry.jsonl_writer import (
    GPUTelemetryJSONLWriter,
)
from aiperf.gpu_telemetry.manager import (
    GPUTelemetryManager,
)
from aiperf.gpu_telemetry.metrics_config import (
    MetricsConfigLoader,
)

__all__ = [
    "DCGM_TO_FIELD_MAPPING",
    "GPUTelemetryAccumulator",
    "GPUTelemetryDataCollector",
    "GPUTelemetryJSONLWriter",
    "GPUTelemetryManager",
    "GPU_TELEMETRY_METRICS_CONFIG",
    "MetricsConfigLoader",
    "SCALING_FACTORS",
    "get_gpu_telemetry_metrics_config",
]
