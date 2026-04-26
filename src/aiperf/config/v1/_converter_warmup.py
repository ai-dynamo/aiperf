# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""build_warmup: convert v1 UserConfig.loadgen warmup_* fields into a phase dict.

Returns None when the user did not explicitly set any warmup_request_count /
warmup_num_sessions / warmup_duration on the load-generator section.

Ported from aiperf.config._cli_sections.build_warmup, but reads from
UserConfig.loadgen.* (with model_fields_set on the LoadGeneratorConfig sub-model)
instead of the legacy CLIModel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.config.phases import PhaseType
from aiperf.plugin.enums import ArrivalPattern

if TYPE_CHECKING:
    from aiperf.config.v1 import UserConfig
    from aiperf.config.v1._loadgen import LoadGeneratorConfig


def _warmup_count_field(w: dict[str, Any], lg: LoadGeneratorConfig) -> None:
    if lg.warmup_request_count is not None:
        w["requests"] = lg.warmup_request_count
    elif lg.warmup_num_sessions is not None:
        w["sessions"] = lg.warmup_num_sessions
    elif lg.warmup_duration is not None:
        w["duration"] = lg.warmup_duration


def _warmup_pattern_type(
    w: dict[str, Any], lg: LoadGeneratorConfig, s: set[str]
) -> None:
    warmup_rate = (
        lg.warmup_request_rate if "warmup_request_rate" in s else lg.request_rate
    )
    warmup_pattern = (
        lg.warmup_arrival_pattern
        if "warmup_arrival_pattern" in s
        else lg.arrival_pattern
    )
    warmup_concurrency = (
        lg.warmup_concurrency if "warmup_concurrency" in s else lg.concurrency
    ) or 1

    if warmup_rate is not None:
        w["rate"] = warmup_rate
        match warmup_pattern:
            case ArrivalPattern.GAMMA:
                w["type"] = PhaseType.GAMMA
                w["smoothness"] = lg.arrival_smoothness
            case ArrivalPattern.CONSTANT:
                w["type"] = PhaseType.CONSTANT
            case _:
                w["type"] = PhaseType.POISSON
    else:
        w["type"] = PhaseType.CONCURRENCY
    # Why: warmup phase always emits concurrency; ConcurrencyPhase defaults
    # this to 1 anyway, but rate phases (POISSON/GAMMA/CONSTANT) need it as
    # a cap and origin/main's behavior derives it from main concurrency or 1.
    w["concurrency"] = warmup_concurrency


def _warmup_ramps(w: dict[str, Any], lg: LoadGeneratorConfig, s: set[str]) -> None:
    def _pick(warmup_field: str, fallback_field: str) -> Any:
        if warmup_field in s:
            return getattr(lg, warmup_field)
        if fallback_field in s:
            return getattr(lg, fallback_field)
        return None

    cr = _pick("warmup_concurrency_ramp_duration", "concurrency_ramp_duration")
    pr = _pick(
        "warmup_prefill_concurrency_ramp_duration",
        "prefill_concurrency_ramp_duration",
    )
    rr = _pick("warmup_request_rate_ramp_duration", "request_rate_ramp_duration")
    if cr is not None:
        w["concurrency_ramp"] = {"duration": cr}
    if pr is not None:
        w["prefill_ramp"] = {"duration": pr}
    if rr is not None:
        w["rate_ramp"] = {"duration": rr}


def build_warmup(user: UserConfig) -> dict[str, Any] | None:
    """Build a warmup phase dict from UserConfig.loadgen, or return None.

    The warmup phase is only emitted when the caller explicitly set one of the
    "trigger" fields (warmup_request_count / warmup_num_sessions /
    warmup_duration) on LoadGeneratorConfig. Other warmup_* fields without a
    trigger are intentionally ignored — matches the legacy CLIModel behaviour.

    Example::

        user = UserConfig.model_validate({"loadgen": {
            "warmup_request_count": 50, "warmup_concurrency": 10,
        }})
        build_warmup(user)
        # -> {"exclude_from_results": True, "type": PhaseType.CONCURRENCY,
        #     "concurrency": 10, "requests": 50}
    """
    lg = user.loadgen
    if lg is None:
        return None
    s = lg.model_fields_set
    if not ({"warmup_request_count", "warmup_num_sessions", "warmup_duration"} & s):
        return None
    w: dict[str, Any] = {"exclude_from_results": True}
    _warmup_count_field(w, lg)
    _warmup_pattern_type(w, lg, s)
    _warmup_ramps(w, lg, s)
    if lg.warmup_prefill_concurrency is not None:
        w["prefill_concurrency"] = lg.warmup_prefill_concurrency
    if lg.warmup_grace_period is not None:
        w["grace_period"] = lg.warmup_grace_period
    return w
