# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import msgspec


class NumericAggregate(
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
):
    """Aggregates for a single numeric value over time.

    Mutable accumulator: ``update()`` rewrites ``min``/``max``/``sum``/``count``
    in place, so this struct intentionally omits ``frozen``.
    """

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


class ProcessHealthAggregates(
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
):
    """Aggregated statistics for process health metrics over time.

    Holds mutable ``NumericAggregate`` sub-fields updated in place each tick.
    """

    memory_usage: NumericAggregate = msgspec.field(default_factory=NumericAggregate)
    cpu_usage: NumericAggregate = msgspec.field(default_factory=NumericAggregate)
    num_threads: NumericAggregate = msgspec.field(default_factory=NumericAggregate)
    voluntary_ctx_switches: NumericAggregate = msgspec.field(
        default_factory=NumericAggregate
    )
    involuntary_ctx_switches: NumericAggregate = msgspec.field(
        default_factory=NumericAggregate
    )
    io_read_bytes: NumericAggregate = msgspec.field(default_factory=NumericAggregate)
    io_write_bytes: NumericAggregate = msgspec.field(default_factory=NumericAggregate)
    cpu_time_user: NumericAggregate = msgspec.field(default_factory=NumericAggregate)
    cpu_time_system: NumericAggregate = msgspec.field(default_factory=NumericAggregate)
    cpu_time_iowait: NumericAggregate = msgspec.field(default_factory=NumericAggregate)


# IO / CPU / ctx-switch tuples: msgspec.Struct frozen replacements for the
# prior ``collections.namedtuple`` types. Named attribute access (e.g.
# ``io_counters.read_bytes``) matches the namedtuple surface that downstream
# consumers (UI, tests) depend on. Constructed via positional splat from the
# psutil namedtuples — see ``ProcessHealthMixin.get_process_health``.
class IOCounters(
    msgspec.Struct,
    frozen=True,
):
    read_count: int
    write_count: int
    read_bytes: int
    write_bytes: int
    read_chars: int
    write_chars: int


class CPUTimes(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
):
    user: float
    system: float
    iowait: float


class CtxSwitches(
    msgspec.Struct,
    frozen=True,
):
    voluntary: int
    involuntary: int


class ProcessHealth(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """Immutable snapshot of process health for a single tick."""

    pid: int | None = None
    create_time: float
    uptime: float
    cpu_usage: float
    memory_usage: int
    pss_memory: int | None = None
    io_counters: IOCounters | None = None
    cpu_times: CPUTimes | None = None
    num_ctx_switches: CtxSwitches | None = None
    num_threads: int | None = None
