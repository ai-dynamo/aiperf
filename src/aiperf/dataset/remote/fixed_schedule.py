# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finish fixed-schedule auto-promotion for a remote trace once its first record is known."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.config.loader.errors import ConfigurationError

if TYPE_CHECKING:
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.dataset.remote_ref import RemoteFixedScheduleCheck

_logger = AIPerfLogger(__name__)

_FIXED_SCHEDULE_ONLY_FIELDS = ("type", "auto_offset", "start_offset", "end_offset")
_STOP_CONDITION_FIELDS = ("requests", "duration", "sessions")


def finalize_remote_fixed_schedule(
    cfg: BenchmarkConfig, check: RemoteFixedScheduleCheck, *, promote: bool
) -> None:
    """Apply the decision the CLI converter would have made with the file on disk."""
    if promote and check.conflict_error:
        raise ConfigurationError(check.conflict_error)
    if check.promoted and not promote:
        _demote_profiling_phase(cfg, check)


def _demoted_stop_condition(
    cfg: BenchmarkConfig, check: RemoteFixedScheduleCheck
) -> tuple[dict[str, Any], str]:
    """The default a local file would have had: a full pass for a suite format, else the
    bound the converter computed before it promoted."""
    from aiperf.config.flags._converter_profiling import full_pass_stop_condition
    from aiperf.config.phases import DEFAULT_UNBOUNDED_REQUESTS
    from aiperf.plugin import plugins

    fmt = getattr(cfg.get_default_dataset(), "format", None)
    # An untyped remote learns its format from the manifest, after the converter ran.
    if (
        fmt is not None
        and plugins.get_dataset_loader_metadata(str(fmt)).full_pass_by_default
    ):
        return full_pass_stop_condition(str(fmt))
    if check.demoted_stop_condition:
        return dict(check.demoted_stop_condition), check.demoted_stop_warning or ""
    # A check serialized before the bound was carried on it.
    return {"requests": DEFAULT_UNBOUNDED_REQUESTS}, (
        f"No stop condition given; defaulting to {DEFAULT_UNBOUNDED_REQUESTS} requests. "
        "Pass --request-count, --num-conversations, or --benchmark-duration to change it."
    )


def _demote_profiling_phase(
    cfg: BenchmarkConfig, check: RemoteFixedScheduleCheck
) -> None:
    from aiperf.config.phases import ConcurrencyPhase, PhaseType

    for i, phase in enumerate(cfg.phases):
        if phase.name != "profiling" or str(phase.type) != str(
            PhaseType.FIXED_SCHEDULE
        ):
            continue
        data = phase.model_dump(exclude_unset=True)
        for key in _FIXED_SCHEDULE_ONLY_FIELDS:
            data.pop(key, None)
        if not any(data.get(k) is not None for k in _STOP_CONDITION_FIELDS):
            stop, message = _demoted_stop_condition(cfg, check)
            data.update(stop)
            if message:
                _logger.warning(message)
        cfg.phases[i] = ConcurrencyPhase(type=PhaseType.CONCURRENCY, **data)
        return
