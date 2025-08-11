# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

import psutil

from aiperf.common.mixins.base_mixin import BaseMixin
from aiperf.common.models import CPUTimes, CtxSwitches, ProcessHealth


class ProcessHealthMixin(BaseMixin):
    """Mixin to provide process health information."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize process-specific CPU monitoring
        self.process: psutil.Process = psutil.Process()
        self.process.cpu_percent()  # throw away the first result (will be 0)
        self.create_time: float = self.process.create_time()

        self.process_health: ProcessHealth | None = None
        self.previous: ProcessHealth | None = None

    def get_process_health(self) -> ProcessHealth:
        """Get the process health information for the current process."""

        # Get process-specific CPU and memory usage
        raw_cpu_times = self.process.cpu_times()
        cpu_times = CPUTimes(
            user=raw_cpu_times[0],
            system=raw_cpu_times[1],
            iowait=raw_cpu_times[4] if len(raw_cpu_times) > 4 else 0.0,  # type: ignore
        )

        self.previous = self.process_health

        self.process_health = ProcessHealth(
            pid=self.process.pid,
            create_time=self.create_time,
            uptime=time.time() - self.create_time,
            cpu_usage=self.process.cpu_percent(),
            memory_usage=self.process.memory_info().rss,
            io_counters=self.process.io_counters(),
            cpu_times=cpu_times,
            num_ctx_switches=CtxSwitches(*self.process.num_ctx_switches()),
            num_threads=self.process.num_threads(),
        )
        return self.process_health
