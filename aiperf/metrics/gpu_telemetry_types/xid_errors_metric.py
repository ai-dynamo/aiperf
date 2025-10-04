# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.metric_enums import GenericMetricUnit, MetricFlags
from aiperf.common.models import TelemetryRecord
from aiperf.metrics.base_telemetry_metric import BaseTelemetryMetric


class XidErrorsMetric(BaseTelemetryMetric[float]):
    """GPU XID error metric.

    Tracks the value of the last XID error encountered by the GPU.
    XID errors indicate various GPU hardware or driver issues.
    """

    tag = "xid_errors"
    header = "XID Errors"
    unit = GenericMetricUnit.COUNT
    display_order = None
    required_metrics = None
    flags = MetricFlags.GPU_TELEMETRY

    def _extract_value(self, record: TelemetryRecord) -> float | None:
        """Extract XID error value from telemetry record.

        Args:
            record: The telemetry record containing GPU metrics

        Returns:
            XID error value, or None if not available
        """
        return record.telemetry_data.xid_errors
