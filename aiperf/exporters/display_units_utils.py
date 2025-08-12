# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.exceptions import MetricUnitError
from aiperf.common.models import MetricResult
from aiperf.common.types import MetricTagT
from aiperf.metrics.metric_registry import MetricRegistry

_logger = AIPerfLogger(__name__)
STAT_KEYS = [
    "avg",
    "min",
    "max",
    "p1",
    "p5",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
    "std",
]


def to_display_unit(result: MetricResult, registry: MetricRegistry) -> MetricResult:
    """
    Return a new MetricResult converted to its display unit (if different).
    """
    metric_cls = registry.get_class(result.tag)
    if result.unit != metric_cls.unit:
        _logger.error(
            f"Metric {result.tag} has a unit ({result.unit}) that does not match the expected unit ({metric_cls.unit.value})."
        )

    display_unit = metric_cls.display_unit or metric_cls.unit

    # Counts do not need to be converted
    if display_unit == metric_cls.unit:
        return result

    record = result.model_copy(deep=True)
    record.unit = display_unit.value

    for stat in STAT_KEYS:
        val = getattr(record, stat, None)
        if val is None:
            continue
        # Only convert numerics
        if isinstance(val, int | float):
            try:
                setattr(
                    record,
                    stat,
                    metric_cls.unit.convert_to(display_unit, val),
                )
            except MetricUnitError as e:
                _logger.warning(
                    f"Error converting {stat} for {result.tag} from {metric_cls.unit.value} to {display_unit.value}: {e}"
                )
    return record


def convert_all_metrics_to_display_units(
    records: Iterable[MetricResult], registry: MetricRegistry
) -> dict[MetricTagT, MetricResult]:
    """Helper for exporters that want a tag->result mapping in display units."""
    out: dict[MetricTagT, MetricResult] = {}
    for r in records:
        out[r.tag] = to_display_unit(r, registry)
    return out
