# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic metric class generator for GPU telemetry metrics.

This module generates metric classes at import time based on GPU_TELEMETRY_METRICS_CONFIG.
Each generated class inherits from BaseTelemetryMetric and automatically registers
with the MetricRegistry via the __init_subclass__ hook.
"""

from aiperf.common.enums import MetricFlags
from aiperf.common.models import TelemetryRecord
from aiperf.gpu_telemetry.constants import GPU_TELEMETRY_METRICS_CONFIG
from aiperf.metrics.base_telemetry_metric import BaseTelemetryMetric


def _create_extract_value_method(field_name: str):
    """Create the _extract_value method for a telemetry metric class.

    Args:
        field_name: The field name from TelemetryMetrics to extract

    Returns:
        A method that extracts the value from a TelemetryRecord
    """

    def _extract_value(self, record: TelemetryRecord) -> float | None:
        return getattr(record.telemetry_data, field_name)

    return _extract_value


def _generate_metric_classes():
    """Generate metric classes dynamically from GPU_TELEMETRY_METRICS_CONFIG.

    For each metric in the config, creates a class that:
    - Inherits from BaseTelemetryMetric[float]
    - Has proper tag, header, unit, and flags attributes
    - Implements _extract_value to extract the metric from TelemetryRecord
    - Auto-registers with MetricRegistry via __init_subclass__
    """
    for display_name, field_name, unit_enum in GPU_TELEMETRY_METRICS_CONFIG:
        class_name = (
            "".join(word.capitalize() for word in field_name.split("_")) + "Metric"
        )

        class_attrs = {
            "tag": field_name,
            "header": display_name,
            "unit": unit_enum,
            "display_order": None,
            "required_metrics": None,
            "flags": MetricFlags.GPU_TELEMETRY,
            "_extract_value": _create_extract_value_method(field_name),
            "__module__": __name__,
        }

        type(class_name, (BaseTelemetryMetric,), class_attrs)


def register_telemetry_metrics():
    """Register all GPU telemetry metric classes with MetricRegistry.

    This function generates and registers metric classes dynamically based on
    GPU_TELEMETRY_METRICS_CONFIG. It should be called once during module initialization.
    """
    _generate_metric_classes()
