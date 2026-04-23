# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-run plot type handlers.

This module re-exports handlers from the `single_run` subpackage to preserve
the historical import path ``aiperf.plot.handlers.single_run_handlers``.
Handler implementations live in focused submodules under
``aiperf.plot.handlers.single_run``.
"""

from aiperf.plot.handlers.single_run import (
    AreaHandler,
    BaseSingleRunHandler,
    DualAxisHandler,
    HistogramHandler,
    PercentileBandsHandler,
    RequestTimelineHandler,
    ScatterHandler,
    ScatterWithPercentilesHandler,
    TimeSliceHandler,
    _is_single_stat_metric,
)

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
