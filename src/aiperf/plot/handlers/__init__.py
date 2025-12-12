# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot type handlers for extensible plot creation."""

from aiperf.plot.handlers.multi_run_handlers import (
    BaseMultiRunHandler,
    ParetoHandler,
    ScatterLineHandler,
)
from aiperf.plot.handlers.single_run_handlers import (
    AreaHandler,
    BaseSingleRunHandler,
    DualAxisHandler,
    HistogramHandler,
    RequestTimelineHandler,
    ScatterHandler,
    ScatterWithPercentilesHandler,
    TimeSliceHandler,
)

__all__ = [
    "AreaHandler",
    "BaseMultiRunHandler",
    "BaseSingleRunHandler",
    "DualAxisHandler",
    "HistogramHandler",
    "ParetoHandler",
    "RequestTimelineHandler",
    "ScatterHandler",
    "ScatterLineHandler",
    "ScatterWithPercentilesHandler",
    "TimeSliceHandler",
]
