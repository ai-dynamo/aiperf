# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.mixins.progress_tracker_mixin import ProgressTrackerMixin
from aiperf.common.mixins.worker_tracker_mixin import WorkerTrackerMixin


class BaseAIPerfUI(ProgressTrackerMixin, WorkerTrackerMixin):
    """Base class for AIPerf UI implementations.

    This class provides a simple starting point for a UI for AIPerf components.
    It inherits from the :class:`ProgressTrackerMixin` and :class:`WorkerTrackerMixin`
    to provide a simple starting point for a UI for AIPerf components.

    Now, you can use the various hooks defined in the :class:`ProgressTrackerMixin` and :class:`WorkerTrackerMixin`
    to create a UI for AIPerf components.

    Example:
    ```python
    @AIPerfUIFactory.register("custom")
    class MyUI(BaseAIPerfUI):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        @on_records_progress
        def _on_records_progress(self, records_stats: RecordsStats):
            '''Callback for records progress updates.'''
            pass

        @on_requests_phase_progress
        def _on_requests_phase_progress(self, phase: CreditPhase, requests_stats: RequestsStats):
            '''Callback for requests phase progress updates.'''
            pass

        @on_worker_update
        def _on_worker_update(self, worker_id: str, worker_stats: WorkerStats):
            '''Callback for worker updates.'''
            pass
    ```
    """
