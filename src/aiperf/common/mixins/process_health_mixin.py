# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

import psutil

from aiperf.common.mixins.base_mixin import BaseMixin
from aiperf.common.models import CPUTimes, CtxSwitches, ProcessHealth


def read_pss_self() -> int | None:
    """Read the proportional set size (PSS) for the current process.

    PSS reflects shared-memory contributions proportionally so a worker that
    inherits a large copy-on-write parent doesn't get blamed for the full
    parent RSS. Returns None when the kernel doesn't expose ``smaps_rollup``
    (non-Linux, sandboxed kernels, container restrictions).
    """
    try:
        with open(f"/proc/{psutil.Process().pid}/smaps_rollup") as fh:
            for line in fh:
                if line.startswith("Pss:"):
                    # ``Pss:  12345 kB``
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
        return None
    except (OSError, ValueError):
        return None


class ProcessHealthMixin(BaseMixin):
    """Mixin to provide process health information."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize process-specific CPU monitoring
        self._process: psutil.Process = psutil.Process()
        self._process.cpu_percent()  # throw away the first result (will be 0)
        self._create_time: float = self._process.create_time()

        self._process_health: ProcessHealth | None = None
        self._previous: ProcessHealth | None = None

    def get_pss_memory(self) -> int | None:
        """Read the per-process PSS memory (Linux-only).

        Thin wrapper around the module-level :func:`read_pss_self` so tests can
        monkey-patch the read site (rather than calling /proc directly in tests).
        Returns None on platforms where the kernel doesn't expose PSS.
        """
        return read_pss_self()

    def get_process_health(self) -> ProcessHealth:
        """Get the process health information for the current process."""

        # Get process-specific CPU and memory usage
        raw_cpu_times = self._process.cpu_times()
        cpu_times = CPUTimes(
            user=raw_cpu_times[0],
            system=raw_cpu_times[1],
            iowait=raw_cpu_times[4] if len(raw_cpu_times) > 4 else 0.0,  # type: ignore
        )

        self._previous = self._process_health

        self._process_health = ProcessHealth(
            pid=self._process.pid,
            create_time=self._create_time,
            uptime=time.time() - self._create_time,
            cpu_usage=self._process.cpu_percent(),
            memory_usage=self._process.memory_info().rss,
            io_counters=self._process.io_counters() if hasattr(self._process, "io_counters") else None,
            cpu_times=cpu_times,
            num_ctx_switches=CtxSwitches(*self._process.num_ctx_switches()),
            num_threads=self._process.num_threads(),
        )  # fmt: skip
        return self._process_health
