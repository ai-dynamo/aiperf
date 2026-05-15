# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class OptimizationDirection(CaseInsensitiveStrEnum):
    """Direction of optimization for a metric.

    Members:
        MAXIMIZE: Higher values are preferred (e.g. throughput, goodput).
        MINIMIZE: Lower values are preferred (e.g. latency, TTFT, p99).

    Defined here for the parameter-sweeping feature. Ideally this would
    be a property of BaseMetric itself; until then sweep objectives carry
    it explicitly.

    Example:
        >>> OptimizationDirection.MAXIMIZE == "maximize"
        True
    """

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
