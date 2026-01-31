# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.mixins import (
    ProgressTrackerMixin,
    RealtimeMetricsMixin,
    RealtimeTelemetryMetricsMixin,
    WorkerTrackerMixin,
)


class BaseAIPerfUI(
    ProgressTrackerMixin,
    WorkerTrackerMixin,
    RealtimeMetricsMixin,
    RealtimeTelemetryMetricsMixin,
):
    """Base class for AIPerf UI implementations.

    This class provides a simple starting point for a UI for AIPerf components.
    It inherits from the :class:`ProgressTrackerMixin`, :class:`WorkerTrackerMixin`,
    :class:`RealtimeMetricsMixin`, and :class:`RealtimeTelemetryMetricsMixin`
    to provide a simple starting point for a UI for AIPerf components.

    Now, you can use the various hooks defined in the :class:`ProgressTrackerMixin`,
    :class:`WorkerTrackerMixin`, :class:`RealtimeMetricsMixin`, and
    :class:`RealtimeTelemetryMetricsMixin` to create a UI for AIPerf components.

    Example:
    ```python
    class MyUI(BaseAIPerfUI):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        @on_records_progress
        def _on_records_progress(self, records_stats: CombinedPhaseStats):
            '''Callback for records progress updates.'''
            pass

        @on_requests_phase_progress
        def _on_requests_phase_progress(self, phase_progress: CombinedPhaseStats):
            '''Callback for requests phase progress updates.'''
            pass

        @on_worker_update
        def _on_worker_update(self, worker_id: str, worker_stats: WorkerStats):
            '''Callback for worker updates.'''
            pass

        @on_realtime_metrics
        def _on_realtime_metrics(self, metrics: list[MetricResult]):
            '''Callback for real-time metrics updates.'''
            pass
    ```
    """
