# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, TypeAlias, TypeVar

from aiperf.common.enums.metric_enums import BaseMetricUnit as BaseMetricUnit
from aiperf.common.enums.metric_enums import BaseMetricUnitInfo as BaseMetricUnitInfo

if TYPE_CHECKING:
    from aiperf.metrics.metric_dicts import MetricSeriesProtocol

MetricValueTypeT: TypeAlias = int | float | list[float] | list[int]
MetricValueTypeVarT = TypeVar("MetricValueTypeVarT", bound=MetricValueTypeT)
MetricDictValueTypeT: TypeAlias = (
    "MetricValueTypeT | list[MetricValueTypeT] | MetricSeriesProtocol"
)


# We allow either an actual enum unit, or an info object that can act like a unit.
MetricUnitT: TypeAlias = BaseMetricUnit | BaseMetricUnitInfo
