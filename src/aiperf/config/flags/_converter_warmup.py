# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""build_warmup: convert CLIConfig warmup_* fields into a phase dict.

Returns None when the user did not explicitly set any warmup_request_count /
warmup_num_sessions / warmup_duration on CLIConfig.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.config.phases import PhaseType
from aiperf.plugin.enums import ArrivalPattern

if TYPE_CHECKING:
    from aiperf.config.flags import CLIConfig


def _warmup_count_field(w: dict[str, Any], cli: CLIConfig) -> None:
    if cli.warmup_request_count is not None:
        w["requests"] = cli.warmup_request_count
    elif cli.warmup_num_sessions is not None:
        w["sessions"] = cli.warmup_num_sessions
    elif cli.warmup_duration is not None:
        w["duration"] = cli.warmup_duration


def _warmup_pattern_type(w: dict[str, Any], cli: CLIConfig, s: set[str]) -> None:
    warmup_rate = (
        cli.warmup_request_rate if "warmup_request_rate" in s else cli.request_rate
    )
    warmup_pattern = (
        cli.warmup_arrival_pattern
        if "warmup_arrival_pattern" in s
        else cli.arrival_pattern
    )
    warmup_concurrency = (
        cli.warmup_concurrency if "warmup_concurrency" in s else cli.concurrency
    )
    # If --concurrency was a sweep list (e.g. 10,20,30), the warmup phase must
    # NOT inherit the list — warmup runs once before the sweep with a scalar
    # concurrency. Fall back to None so the sweep promoter doesn't accidentally
    # also sweep warmup.
    if isinstance(warmup_concurrency, list):
        warmup_concurrency = None

    if warmup_rate is not None:
        w["rate"] = warmup_rate
        match warmup_pattern:
            case ArrivalPattern.GAMMA:
                w["type"] = PhaseType.GAMMA
                w["smoothness"] = cli.arrival_smoothness
            case ArrivalPattern.CONSTANT:
                w["type"] = PhaseType.CONSTANT
            case _:
                w["type"] = PhaseType.POISSON
        # Rate phases: concurrency acts as an optional cap. Only set it when the
        # user explicitly provided one — otherwise leave None (no cap) so the
        # configured rate is the sole bound. A duration-based warmup at rate R
        # for T seconds emits ~R*T credits, not throttled.
        if warmup_concurrency is not None:
            w["concurrency"] = warmup_concurrency
    else:
        w["type"] = PhaseType.CONCURRENCY
        # Concurrency phase needs a positive concurrency. ConcurrencyPhase
        # defaults to 1 already, but be explicit for clarity.
        w["concurrency"] = warmup_concurrency if warmup_concurrency is not None else 1


def _warmup_ramps(w: dict[str, Any], cli: CLIConfig, s: set[str]) -> None:
    def _pick(warmup_field: str, fallback_field: str) -> Any:
        if warmup_field in s:
            return getattr(cli, warmup_field)
        if fallback_field in s:
            return getattr(cli, fallback_field)
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


def build_warmup(cli: CLIConfig) -> dict[str, Any] | None:
    """Build a warmup phase dict from CLIConfig, or return None.

    The warmup phase is only emitted when the caller explicitly set one of the
    "trigger" fields (warmup_request_count / warmup_num_sessions /
    warmup_duration) on CLIConfig. Other warmup_* fields without a
    trigger are intentionally ignored.

    Example::

        cli = CLIConfig.model_validate({
            "warmup_request_count": 50, "warmup_concurrency": 10,
        })
        build_warmup(cli)
        # -> {"exclude_from_results": True, "type": PhaseType.CONCURRENCY,
        #     "concurrency": 10, "requests": 50}
    """
    s = cli.model_fields_set
    if not ({"warmup_request_count", "warmup_num_sessions", "warmup_duration"} & s):
        # No warmup trigger -> no warmup phase. Refuse to silently drop
        # secondary warmup-only flags the user supplied.
        if cli.warmup_grace_period is not None:
            raise ValueError(
                "--warmup-grace-period was supplied without any warmup "
                "trigger; warmup runs only when --warmup-request-count, "
                "--warmup-num-sessions, or --warmup-duration is set. Pass "
                "--warmup-duration to enable a duration-bounded warmup with "
                "the grace period, or drop --warmup-grace-period."
            )
        return None
    w: dict[str, Any] = {"exclude_from_results": True}
    _warmup_count_field(w, cli)
    _warmup_pattern_type(w, cli, s)
    _warmup_ramps(w, cli, s)
    if "warmup_prefill_concurrency" in s:
        w["prefill_concurrency"] = cli.warmup_prefill_concurrency
    elif "prefill_concurrency" in s:
        w["prefill_concurrency"] = cli.prefill_concurrency
    if cli.warmup_grace_period is not None:
        # grace_period is a duration-phase concept (a tail on top of ``duration``);
        # PhaseConfig rejects it without ``duration`` set. Raise a targeted error
        # here rather than letting the user hit a confusing ValidationError
        # buried in the AIPerfConfig load. Unlike --benchmark-grace-period (which
        # has a non-None default and so cannot be reliably distinguished from
        # explicit user input downstream), warmup_grace_period defaults to None,
        # so a non-None value means the user explicitly asked for it.
        if "duration" not in w:
            raise ValueError(
                "--warmup-grace-period requires --warmup-duration; "
                "grace_period applies only to duration-bounded warmup phases. "
                "Either set --warmup-duration, or drop --warmup-grace-period "
                "when using --warmup-request-count / --warmup-num-sessions."
            )
        w["grace_period"] = cli.warmup_grace_period
    return w
