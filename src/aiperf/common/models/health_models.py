# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import namedtuple
from dataclasses import dataclass, field
from typing import ClassVar

from pydantic import ConfigDict, Field

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
        ge=0,
        description="The PID of the process",
    )
    create_time: float = Field(
        ..., ge=0, description="The creation time of the process in seconds"
    )
    uptime: float = Field(..., ge=0, description="The uptime of the process in seconds")
    cpu_usage: float = Field(
        ..., ge=0, description="The current CPU usage of the process in %"
    )
    memory_usage: int = Field(
        ..., ge=0, description="The current memory usage of the process in bytes (rss)"
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
        ge=0,
        description="The current number of threads",
    )


@dataclass(slots=True, kw_only=True)
class NumericAggregate:
    """Aggregates for a single numeric value over time.

    Mutable slotted dataclass: ``update()`` rewrites ``min``/``max``/``sum``/``count``
    in place, so this type intentionally is not frozen.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None
    sum: float = 0.0
    count: int = 0

    @property
    def avg(self) -> float | None:
        """Average of all observed values."""
        return self.sum / self.count if self.count > 0 else None

    def update(self, value: float | int | None) -> None:
        """Update aggregates with a new observed value."""
        if value is None:
            return
        val = float(value)
        self.min = val if self.min is None else min(self.min, val)
        self.max = val if self.max is None else max(self.max, val)
        self.sum += val
        self.count += 1


@dataclass(slots=True, kw_only=True)
class ProcessHealthAggregates:
    """Aggregated statistics for process health metrics over time.

    Holds mutable ``NumericAggregate`` sub-fields updated in place each tick.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    memory_usage: NumericAggregate = field(default_factory=NumericAggregate)
    cpu_usage: NumericAggregate = field(default_factory=NumericAggregate)
    num_threads: NumericAggregate = field(default_factory=NumericAggregate)
    voluntary_ctx_switches: NumericAggregate = field(default_factory=NumericAggregate)
    involuntary_ctx_switches: NumericAggregate = field(default_factory=NumericAggregate)
    io_read_bytes: NumericAggregate = field(default_factory=NumericAggregate)
    io_write_bytes: NumericAggregate = field(default_factory=NumericAggregate)
    cpu_time_user: NumericAggregate = field(default_factory=NumericAggregate)
    cpu_time_system: NumericAggregate = field(default_factory=NumericAggregate)
    cpu_time_iowait: NumericAggregate = field(default_factory=NumericAggregate)
