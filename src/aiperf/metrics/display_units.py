# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import STAT_KEYS
from aiperf.common.exceptions import MetricTypeError, MetricUnitError
from aiperf.common.models import MetricResult
from aiperf.metrics.metric_registry import MetricRegistry

_logger = AIPerfLogger(__name__)

_ADJ_PREFIX = "adj_"


def to_display_unit(result: MetricResult, registry: MetricRegistry) -> MetricResult:
    """
    Return a new MetricResult converted to its display unit (if different).

    Returns the result unchanged if the tag is not in the metric registry
    (e.g. sweep metrics injected by analyzers). For ``adj_<tag>`` derived
    metrics (failure-inflated percentiles, see issue #688), looks up the
    parent tag's unit metadata so the standard conversion path applies.
    """
    metric_cls = _resolve_metric_class(registry, result.tag)
    if metric_cls is None:
        return result

    unit = getattr(metric_cls, "unit", None)
    unit_value = _unit_value(unit)
    if unit_value is None:
        return result

    if result.unit and result.unit != unit_value:
        _logger.error(
            f"Metric {result.tag} has a unit ({result.unit}) that does not match the expected unit ({unit_value}). "
            f"({unit_value}) will be used for conversion."
        )

    display_unit = getattr(metric_cls, "display_unit", None) or unit
    display_unit_value = _unit_value(display_unit)
    if display_unit_value is None or display_unit == unit:
        return result

    convert_to = getattr(unit, "convert_to", None)
    if not callable(convert_to):
        return result

    record = result.model_copy(deep=True)
    record.unit = display_unit_value

    for stat in STAT_KEYS:
        val = getattr(record, stat, None)
        if val is None:
            continue
        # Only convert numeric values. ``+inf`` (failure-inflation sentinel
        # from ``adj_<tag>`` derived metrics) divides correctly through the
        # linear time/byte conversions used here, so no special-casing
        # required — the convert_to call returns ``inf`` unchanged.
        if isinstance(val, int | float):
            try:
                new_value = convert_to(display_unit, val)
            except MetricUnitError as e:
                _logger.warning(
                    f"Error converting {stat} for {result.tag} from {unit_value} to {display_unit_value}: {e}"
                )
                continue
            setattr(record, stat, new_value)
    return record


def _unit_value(unit: Any) -> str | None:
    value = getattr(unit, "value", None)
    return value if isinstance(value, str) else None


def _resolve_metric_class(registry: MetricRegistry, tag: str) -> Any | None:
    """Look up the metric class for ``tag``, falling back to the parent tag for
    ``adj_<tag>`` synthetic derived metrics so they inherit unit metadata."""
    try:
        return registry.get_class(tag)
    except (MetricTypeError, KeyError):
        if tag.startswith(_ADJ_PREFIX):
            try:
                return registry.get_class(tag[len(_ADJ_PREFIX) :])
            except (MetricTypeError, KeyError):
                return None
        return None
