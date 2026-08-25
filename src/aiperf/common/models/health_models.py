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
    """Running min/max/sum/count for a single numeric health metric.

    Mutable slotted dataclass: ``update()`` rewrites the fields in place on
    every health tick, so this type intentionally is not frozen. Carries
    ``__pydantic_config__`` because it is nested inside models that cross a
    Pydantic validation boundary.

    Example:
        >>> agg = NumericAggregate()
        >>> agg.update(12.5)
        >>> agg.update(7.5)
        >>> agg.avg
        10.0
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    min: float | None = None
    """Smallest value observed so far, or None before the first sample."""
    max: float | None = None
    """Largest value observed so far, or None before the first sample."""
    sum: float = 0.0
    """Sum of every observed value."""
    count: int = 0
    """Number of observed values."""

    @property
    def avg(self) -> float | None:
        """Mean of all observed values, or None when nothing was observed."""
        return self.sum / self.count if self.count > 0 else None

    def update(self, value: float | int | None) -> None:
        """Fold a newly observed value into the aggregate. ``None`` is ignored."""
        if value is None:
            return
        val = float(value)
        self.min = val if self.min is None else min(self.min, val)
        self.max = val if self.max is None else max(self.max, val)
        self.sum += val
        self.count += 1


@dataclass(slots=True, kw_only=True)
class ProcessHealthAggregates:
    """Aggregated statistics for process-health metrics over the run.

    Holds mutable :class:`NumericAggregate` sub-fields updated in place each
    tick, giving a whole-run min/max/avg view alongside the latest
    :class:`ProcessHealth` sample.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    memory_usage: NumericAggregate = field(default_factory=NumericAggregate)
    """Resident memory in bytes."""
    cpu_usage: NumericAggregate = field(default_factory=NumericAggregate)
    """CPU utilization percentage."""
    num_threads: NumericAggregate = field(default_factory=NumericAggregate)
    """Thread count."""
    voluntary_ctx_switches: NumericAggregate = field(default_factory=NumericAggregate)
    """Voluntary context switches."""
    involuntary_ctx_switches: NumericAggregate = field(default_factory=NumericAggregate)
    """Involuntary context switches."""
    io_read_bytes: NumericAggregate = field(default_factory=NumericAggregate)
    """Bytes read from disk."""
    io_write_bytes: NumericAggregate = field(default_factory=NumericAggregate)
    """Bytes written to disk."""
    cpu_time_user: NumericAggregate = field(default_factory=NumericAggregate)
    """Seconds of user-mode CPU time."""
    cpu_time_system: NumericAggregate = field(default_factory=NumericAggregate)
    """Seconds of kernel-mode CPU time."""
    cpu_time_iowait: NumericAggregate = field(default_factory=NumericAggregate)
    """Seconds spent waiting on I/O."""
