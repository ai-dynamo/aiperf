# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIConfig -> profiling phase dict.

Reads load-generator fields, dataset/schedule fields, and session-turn count
directly off the flat ``CLIConfig``.

Each entry in ``_PROF_FIELD_ROUTES`` declares ``(output_key, attr_name)``
where ``attr_name`` is a top-level field on ``CLIConfig``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.flags import CLIConfig


# (output_key, attr_name) — attr_name is the top-level field on CLIConfig
# whose user-set status we test via ``cli.model_fields_set``.
_PROF_FIELD_ROUTES: tuple[tuple[str, str], ...] = (
    ("duration", "benchmark_duration"),
    ("grace_period", "benchmark_grace_period"),
    ("concurrency", "concurrency"),
    ("prefill_concurrency", "prefill_concurrency"),
    ("requests", "request_count"),
    ("sessions", "conversation_num"),
    ("users", "num_users"),
    ("rate", "request_rate"),
    ("rate", "user_centric_rate"),
)


# Routes whose output keys only exist on GammaPhase. Routed here only when the
# resolved phase type is GAMMA; otherwise the user-supplied value is rejected
# with a clear error rather than crashing PhaseConfig with extra_forbidden.
_GAMMA_ONLY_ROUTES: tuple[tuple[str, str], ...] = (
    ("smoothness", "arrival_smoothness"),
)


# Routes whose output keys only exist on FixedSchedulePhase. Routed here only
# when the resolved phase type is FIXED_SCHEDULE; otherwise we fail loud
# instead of silently dropping the offsets the user passed.
_FIXED_SCHEDULE_ONLY_ROUTES: tuple[tuple[str, str], ...] = (
    ("auto_offset", "fixed_schedule_auto_offset"),
    ("start_offset", "fixed_schedule_start_offset"),
    ("end_offset", "fixed_schedule_end_offset"),
)


_RAMP_FIELDS: tuple[tuple[str, str], ...] = (
    ("concurrency_ramp_duration", "concurrency_ramp"),
    ("prefill_concurrency_ramp_duration", "prefill_ramp"),
    ("request_rate_ramp_duration", "rate_ramp"),
)


def _profiling_phase_type(cli: CLIConfig) -> Any:
    from aiperf.config.phases import PhaseType
    from aiperf.plugin.enums import ArrivalPattern

    if cli.fixed_schedule:
        return PhaseType.FIXED_SCHEDULE
    if cli.user_centric_rate is not None:
        return PhaseType.USER_CENTRIC
    if cli.request_rate is not None:
        match cli.arrival_pattern:
            case ArrivalPattern.GAMMA:
                return PhaseType.GAMMA
            case ArrivalPattern.CONSTANT:
                return PhaseType.CONSTANT
            case _:
                return PhaseType.POISSON
    return PhaseType.CONCURRENCY


def _apply_profiling_ramps(prof: dict[str, Any], cli: CLIConfig) -> None:
    fields_set = cli.model_fields_set
    for field, key in _RAMP_FIELDS:
        if field in fields_set:
            prof[key] = {"duration": getattr(cli, field)}


def _apply_phase_specific_routes(prof: dict[str, Any], cli: CLIConfig) -> None:
    """Apply routes whose output keys only exist on a specific phase subclass.

    Errors out with a clear message when the user supplied a phase-specific
    flag that doesn't match the resolved phase type, instead of letting the
    flag silently no-op (fixed-schedule offsets) or crash PhaseConfig with
    ``extra_forbidden`` (gamma smoothness).
    """
    from aiperf.config.phases import PhaseType

    phase_type = prof["type"]
    fields_set = cli.model_fields_set

    for output_key, attr_name in _GAMMA_ONLY_ROUTES:
        if attr_name not in fields_set:
            continue
        if phase_type != PhaseType.GAMMA:
            raise ValueError(
                "--arrival-smoothness is only supported with --arrival-pattern gamma. "
                "Pass --arrival-pattern gamma to enable smoothness, or drop "
                "--arrival-smoothness to use the default arrival pattern."
            )
        prof[output_key] = getattr(cli, attr_name)

    for output_key, attr_name in _FIXED_SCHEDULE_ONLY_ROUTES:
        if attr_name not in fields_set:
            continue
        if phase_type != PhaseType.FIXED_SCHEDULE:
            raise ValueError(
                "--fixed-schedule-{auto,start,end}-offset requires --fixed-schedule. "
                "Pass --fixed-schedule with a trace file to enable offsets, or drop "
                "these flags."
            )
        prof[output_key] = getattr(cli, attr_name)


def _validate_profiling(prof: dict[str, Any], cli: CLIConfig) -> None:
    from aiperf.config.phases import PhaseType

    # `--conversation-turn-mean` may be a list when used as a magic-list
    # sweep. User-centric mode requires every variation to satisfy
    # turn_mean >= 2, so check the floor of the swept range.
    raw_turn_mean = cli.conversation_turn_mean or 1
    if isinstance(raw_turn_mean, list):
        turn_mean = min(raw_turn_mean) if raw_turn_mean else 1
    else:
        turn_mean = raw_turn_mean
    if prof["type"] == PhaseType.USER_CENTRIC and turn_mean < 2:
        raise ValueError(
            "User-centric rate mode requires --session-turns-mean >= 2. "
            "For single-turn workloads, use --request-rate instead."
        )
    if (
        not any(k in prof for k in ("requests", "duration", "sessions"))
        and prof["type"] != PhaseType.FIXED_SCHEDULE
    ):
        # Why: when no bound is given for an unbounded run, default to
        # 10 requests so the run terminates in a reasonable time.
        # Deliberate override of the PhaseConfig default (which would
        # leave it unbounded).
        prof.setdefault("requests", 10)
    delay_set = "request_cancellation_delay" in cli.model_fields_set
    if cli.request_cancellation_rate:
        cancel: dict[str, Any] = {"rate": cli.request_cancellation_rate}
        if delay_set:
            cancel["delay"] = cli.request_cancellation_delay
        prof["cancellation"] = cancel
    elif delay_set:
        # Mirror --arrival-smoothness gating: refuse to silently drop a
        # user-supplied flag whose dependency wasn't met.
        raise ValueError(
            "--request-cancellation-delay requires --request-cancellation-rate "
            "to be set (cancellation is disabled when rate is unset). "
            "Pass --request-cancellation-rate > 0 to enable cancellation, or "
            "drop --request-cancellation-delay."
        )


def build_profiling(cli: CLIConfig) -> dict[str, Any]:
    """Produce the canonical profiling-phase dict from ``cli``.

    Reads load-generator settings (concurrency, rate, ramps, cancellation),
    schedule/replay flags, and session-turn count directly from ``cli``
    (all fields are top-level post-Task-13). Returns a dict whose ``type``
    is one of ``PhaseType.{CONCURRENCY, POISSON, GAMMA, CONSTANT,
    USER_CENTRIC, FIXED_SCHEDULE}`` plus the keys mapped by
    ``_PROF_FIELD_ROUTES`` and any ramp/cancellation sub-dicts.

    Raises:
        ValueError: when USER_CENTRIC mode is selected but
            ``conversation_turn_mean`` is < 2.
    """
    from aiperf.config.phases import PhaseType

    fields_set = cli.model_fields_set
    prof: dict[str, Any] = {}
    for output_key, attr_name in _PROF_FIELD_ROUTES:
        if attr_name in fields_set:
            prof[output_key] = getattr(cli, attr_name)

    _apply_profiling_ramps(prof, cli)

    prof["type"] = _profiling_phase_type(cli)

    _apply_phase_specific_routes(prof, cli)

    if prof["type"] == PhaseType.FIXED_SCHEDULE and "start_offset" in prof:
        prof.setdefault("auto_offset", False)

    # grace_period is a duration-phase concept (a tail on top of ``duration``);
    # PhaseConfig rejects it without ``duration`` set. Drop it silently when no
    # duration is supplied so users don't hit a confusing validation error from
    # ``--benchmark-grace-period N`` alone.
    if "grace_period" in prof and prof.get("duration") is None:
        prof.pop("grace_period")

    _validate_profiling(prof, cli)
    return prof
