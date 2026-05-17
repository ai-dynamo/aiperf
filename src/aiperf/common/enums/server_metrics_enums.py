# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum
from aiperf.common.enums.enums import ConvergenceStat as ConvergenceStat
from aiperf.common.enums.enums import GPUTelemetryMode as GPUTelemetryMode
from aiperf.common.enums.enums import PrometheusMetricType as PrometheusMetricType
from aiperf.common.enums.enums import (
    ServerMetricsDiscoveryMode as ServerMetricsDiscoveryMode,
)


class GpuTelemetryType(CaseInsensitiveStrEnum):
    """Defines the type of GPU telemetry source."""

    DASHBOARD = "dashboard"
    """Built-in dashboard metrics collection."""

    DCGM = "dcgm"
    """NVIDIA DCGM (Data Center GPU Manager) integration."""

    CSV = "csv"
    """Export GPU metrics to CSV file."""
