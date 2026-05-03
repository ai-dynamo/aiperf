# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import STAT_KEYS
from aiperf.common.exceptions import MetricUnitError
from aiperf.common.models import MetricResult
from aiperf.metrics.metric_registry import MetricRegistry

_logger = AIPerfLogger(__name__)


def to_display_unit(result: MetricResult, registry: MetricRegistry) -> MetricResult:
    """
    Return a new MetricResult converted to its display unit (if different).
    """
    metric_cls = registry.get_class(result.tag)
    if result.unit and result.unit != metric_cls.unit.value:
        _logger.error(
            f"Metric {result.tag} has a unit ({result.unit}) that does not match the expected unit ({metric_cls.unit.value}). "
            f"({metric_cls.unit.value}) will be used for conversion."
        )

    display_unit = metric_cls.display_unit or metric_cls.unit

    if display_unit == metric_cls.unit:
        return result

    record = result.model_copy(deep=True)
    record.unit = display_unit.value

    for stat in STAT_KEYS:
        val = getattr(record, stat, None)
        if val is None:
            continue
        # Only convert numeric values
        if isinstance(val, int | float):
            try:
                new_value = metric_cls.unit.convert_to(display_unit, val)
            except MetricUnitError as e:
                _logger.warning(
                    f"Error converting {stat} for {result.tag} from {metric_cls.unit.value} to {display_unit.value}: {e}"
                )
                continue
            setattr(record, stat, new_value)
    return record
