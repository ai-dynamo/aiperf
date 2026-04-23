# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-run plot type handlers split into focused submodules."""

from aiperf.plot.handlers.single_run._area import AreaHandler
from aiperf.plot.handlers.single_run._base import (
    BaseSingleRunHandler,
    _is_single_stat_metric,
)
from aiperf.plot.handlers.single_run._dual_axis import DualAxisHandler
from aiperf.plot.handlers.single_run._histogram import HistogramHandler
from aiperf.plot.handlers.single_run._percentile_bands import PercentileBandsHandler
from aiperf.plot.handlers.single_run._request_timeline import RequestTimelineHandler
from aiperf.plot.handlers.single_run._scatter import ScatterHandler
from aiperf.plot.handlers.single_run._scatter_percentiles import (
    ScatterWithPercentilesHandler,
)
from aiperf.plot.handlers.single_run._timeslice import TimeSliceHandler

__all__ = [
    "AreaHandler",
    "BaseSingleRunHandler",
    "DualAxisHandler",
    "HistogramHandler",
    "PercentileBandsHandler",
    "RequestTimelineHandler",
    "ScatterHandler",
    "ScatterWithPercentilesHandler",
    "TimeSliceHandler",
    "_is_single_stat_metric",
]
