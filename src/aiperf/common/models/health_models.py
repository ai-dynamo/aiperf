# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import namedtuple

from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel

# TODO: These can be potentially different for each platform. (below is linux)
IOCounters = namedtuple(
    "IOCounters",
    [
        "read_count",  # system calls io read
        "write_count",  # system calls io write
        "read_bytes",  # bytes read (disk io)
        "write_bytes",  # bytes written (disk io)
        "read_chars",  # io read bytes (system calls)
        "write_chars",  # io write bytes (system calls)
    ],
)

CPUTimes = namedtuple(
    "CPUTimes",
    ["user", "system", "iowait"],
)

CtxSwitches = namedtuple("CtxSwitches", ["voluntary", "involuntary"])


class ProcessHealth(AIPerfBaseModel):
    """Model for process health data."""

    pid: int | None = Field(
        default=None,
        description="The PID of the process",
    )
    create_time: float = Field(
        ..., description="The creation time of the process in seconds"
    )
    uptime: float = Field(..., description="The uptime of the process in seconds")
    cpu_usage: float = Field(
        ..., description="The current CPU usage of the process in %"
    )
    memory_usage: int = Field(
        ..., description="The current memory usage of the process in bytes (rss)"
    )
    io_counters: IOCounters | tuple | None = Field(
        default=None,
        description="The current I/O counters of the process (read_count, write_count, read_bytes, write_bytes, read_chars, write_chars)",
    )
    cpu_times: CPUTimes | tuple | None = Field(
        default=None,
        description="The current CPU times of the process (user, system, iowait)",
    )
    num_ctx_switches: CtxSwitches | tuple | None = Field(
        default=None,
        description="The current number of context switches (voluntary, involuntary)",
    )
    num_threads: int | None = Field(
        default=None,
        description="The current number of threads",
    )
