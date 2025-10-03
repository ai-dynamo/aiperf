# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.post_processors.metric_results_processor import MetricResultsProcessor


class TimesliceMetricResultsProcessor(MetricResultsProcessor):
    """Description"""

    def get_results(self) -> MetricResultsDict | None:
        return None
