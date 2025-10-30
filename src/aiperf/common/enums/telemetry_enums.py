# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class GPUTelemetryMode(CaseInsensitiveStrEnum):
    """GPU telemetry display mode."""

    SUMMARY = "summary"
    REALTIME_DASHBOARD = "realtime_dashboard"
