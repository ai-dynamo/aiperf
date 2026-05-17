# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.enums.metric_base import (
    BaseMetricUnitInfo,
)
from aiperf.common.enums.metric_enums import EnergyMetricUnit as EnergyMetricUnit
from aiperf.common.enums.metric_enums import (
    EnergyMetricUnitInfo as EnergyMetricUnitInfo,
)
from aiperf.common.enums.metric_enums import FrequencyMetricUnit as FrequencyMetricUnit
from aiperf.common.enums.metric_enums import (
    FrequencyMetricUnitInfo as FrequencyMetricUnitInfo,
)
from aiperf.common.enums.metric_enums import GenericMetricUnit as GenericMetricUnit
from aiperf.common.enums.metric_enums import MetricOverTimeUnit as MetricOverTimeUnit
from aiperf.common.enums.metric_enums import (
    MetricOverTimeUnitInfo as MetricOverTimeUnitInfo,
)
from aiperf.common.enums.metric_enums import MetricSizeUnit as MetricSizeUnit
from aiperf.common.enums.metric_enums import MetricSizeUnitInfo as MetricSizeUnitInfo
from aiperf.common.enums.metric_enums import MetricTimeUnit as MetricTimeUnit
from aiperf.common.enums.metric_enums import MetricTimeUnitInfo as MetricTimeUnitInfo
from aiperf.common.enums.metric_enums import PowerMetricUnit as PowerMetricUnit
from aiperf.common.enums.metric_enums import PowerMetricUnitInfo as PowerMetricUnitInfo
from aiperf.common.enums.metric_enums import (
    TemperatureMetricUnit as TemperatureMetricUnit,
)
from aiperf.common.enums.metric_enums import (
    TemperatureMetricUnitInfo as TemperatureMetricUnitInfo,
)


# Syntactic sugar for creating BaseMetricUnitInfo instances with a tag
def _unit(tag: str) -> BaseMetricUnitInfo:
    return BaseMetricUnitInfo(tag=tag)
