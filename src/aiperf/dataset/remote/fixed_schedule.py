# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finish fixed-schedule auto-promotion for a remote trace once its first record is known."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.config.loader.errors import ConfigurationError

if TYPE_CHECKING:
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.dataset.remote_ref import RemoteFixedScheduleCheck

_FIXED_SCHEDULE_ONLY_FIELDS = ("type", "auto_offset", "start_offset", "end_offset")
_STOP_CONDITION_FIELDS = ("requests", "duration", "sessions")


def finalize_remote_fixed_schedule(
    cfg: BenchmarkConfig, check: RemoteFixedScheduleCheck, *, promote: bool
) -> None:
    """Apply the decision the CLI converter would have made with the file on disk."""
    if promote and check.conflict_error:
        raise ConfigurationError(check.conflict_error)
    if check.promoted and not promote:
        _demote_profiling_phase(cfg)


def _demote_profiling_phase(cfg: BenchmarkConfig) -> None:
    from aiperf.config.phases import (
        DEFAULT_UNBOUNDED_REQUESTS,
        ConcurrencyPhase,
        PhaseType,
    )

    for i, phase in enumerate(cfg.phases):
        if phase.name != "profiling" or str(phase.type) != str(
            PhaseType.FIXED_SCHEDULE
        ):
            continue
        data = phase.model_dump(exclude_unset=True)
        for key in _FIXED_SCHEDULE_ONLY_FIELDS:
            data.pop(key, None)
        # The same bound the converter gives an unbounded run it did not promote.
        if not any(data.get(k) is not None for k in _STOP_CONDITION_FIELDS):
            data["requests"] = DEFAULT_UNBOUNDED_REQUESTS
        cfg.phases[i] = ConcurrencyPhase(type=PhaseType.CONCURRENCY, **data)
        return
