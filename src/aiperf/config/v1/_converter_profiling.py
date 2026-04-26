# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1 UserConfig -> profiling phase dict.

Ports the timing-mode discriminator and field-mapping logic from
``aiperf.config._cli_sections.build_profiling`` (the flat-CLI converter)
to read from the structured ``UserConfig`` shape: load-generator fields
on ``user.loadgen``, dataset/schedule fields on ``user.input``, and
session-turn-count on ``user.input.conversation.turn``.

Each entry in ``_PROF_FIELD_ROUTES`` declares (output_key, attribute_chain),
where the chain is resolved against ``user`` and the originating model's
``model_fields_set`` is checked for "explicitly set" semantics. Mirrors the
"if field in s" iteration of the flat converter without losing the
distinction between defaulted and user-supplied values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

    from aiperf.config.v1 import UserConfig


# (output_key, model_path, attr_name) — model_path is the dotted path from
# UserConfig to the BaseModel that owns attr_name (so we can check its
# model_fields_set), and attr_name is the field on that model.
_PROF_FIELD_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("duration", "loadgen", "benchmark_duration"),
    ("grace_period", "loadgen", "benchmark_grace_period"),
    ("concurrency", "loadgen", "concurrency"),
    ("prefill_concurrency", "loadgen", "prefill_concurrency"),
    ("smoothness", "loadgen", "arrival_smoothness"),
    ("requests", "loadgen", "request_count"),
    ("sessions", "input.conversation", "num"),
    ("users", "loadgen", "num_users"),
    ("rate", "loadgen", "request_rate"),
    ("rate", "loadgen", "user_centric_rate"),
    ("auto_offset", "input", "fixed_schedule_auto_offset"),
    ("start_offset", "input", "fixed_schedule_start_offset"),
    ("end_offset", "input", "fixed_schedule_end_offset"),
)


_RAMP_FIELDS: tuple[tuple[str, str], ...] = (
    ("concurrency_ramp_duration", "concurrency_ramp"),
    ("prefill_concurrency_ramp_duration", "prefill_ramp"),
    ("request_rate_ramp_duration", "rate_ramp"),
)


def _resolve_model(user: UserConfig, model_path: str) -> BaseModel | None:
    """Walk a dotted attribute path on ``user``, returning ``None`` if any
    intermediate node is unset (``None``)."""
    obj: Any = user
    for part in model_path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _profiling_phase_type(user: UserConfig) -> Any:
    from aiperf.config.phases import PhaseType
    from aiperf.plugin.enums import ArrivalPattern

    input_cfg = user.input
    loadgen = user.loadgen

    if input_cfg is not None and input_cfg.fixed_schedule:
        return PhaseType.FIXED_SCHEDULE
    if loadgen is not None and loadgen.user_centric_rate is not None:
        return PhaseType.USER_CENTRIC
    if loadgen is not None and loadgen.request_rate is not None:
        match loadgen.arrival_pattern:
            case ArrivalPattern.GAMMA:
                return PhaseType.GAMMA
            case ArrivalPattern.CONSTANT:
                return PhaseType.CONSTANT
            case _:
                return PhaseType.POISSON
    return PhaseType.CONCURRENCY


def _apply_profiling_ramps(prof: dict[str, Any], user: UserConfig) -> None:
    loadgen = user.loadgen
    if loadgen is None:
        return
    fields_set = loadgen.model_fields_set
    for field, key in _RAMP_FIELDS:
        if field in fields_set:
            prof[key] = {"duration": getattr(loadgen, field)}


def _validate_profiling(prof: dict[str, Any], user: UserConfig) -> None:
    from aiperf.config.phases import PhaseType

    turn_mean = 1
    if user.input is not None:
        turn_mean = user.input.conversation.turn.mean or 1
    if prof["type"] == PhaseType.USER_CENTRIC and turn_mean < 2:
        raise ValueError(
            "User-centric rate mode requires --session-turns-mean >= 2. "
            "For single-turn workloads, use --request-rate instead."
        )
    if (
        not any(k in prof for k in ("requests", "duration", "sessions"))
        and prof["type"] != PhaseType.FIXED_SCHEDULE
    ):
        # Why: mirrors origin/main's user-friendlier convention — when no
        # bound was given for an unbounded run, default to 10 requests so
        # the run terminates in a reasonable time. Deliberate override of
        # the v2 PhaseConfig default (which would leave it unbounded).
        prof.setdefault("requests", 10)
    loadgen = user.loadgen
    if loadgen is not None and loadgen.request_cancellation_rate:
        cancel: dict[str, Any] = {"rate": loadgen.request_cancellation_rate}
        if "request_cancellation_delay" in loadgen.model_fields_set:
            cancel["delay"] = loadgen.request_cancellation_delay
        prof["cancellation"] = cancel


def build_profiling(user: UserConfig) -> dict[str, Any]:
    """Produce the canonical profiling-phase dict from ``user``.

    Reads load-generator settings (concurrency, rate, ramps, cancellation)
    from ``user.loadgen``, schedule/replay flags from ``user.input``, and
    session-turn count from ``user.input.conversation.turn``. Returns a
    dict whose ``type`` is one of ``PhaseType.{CONCURRENCY, POISSON,
    GAMMA, CONSTANT, USER_CENTRIC, FIXED_SCHEDULE}`` plus the keys mapped
    by ``_PROF_FIELD_ROUTES`` and any ramp/cancellation sub-dicts.

    Raises:
        ValueError: when USER_CENTRIC mode is selected but
            ``input.conversation.turn.mean`` is < 2.
    """
    from aiperf.config.phases import PhaseType

    prof: dict[str, Any] = {}
    for output_key, model_path, attr_name in _PROF_FIELD_ROUTES:
        model = _resolve_model(user, model_path)
        if model is None:
            continue
        if attr_name in model.model_fields_set:
            prof[output_key] = getattr(model, attr_name)

    _apply_profiling_ramps(prof, user)

    prof["type"] = _profiling_phase_type(user)
    if prof["type"] == PhaseType.FIXED_SCHEDULE and "start_offset" in prof:
        prof.setdefault("auto_offset", False)

    _validate_profiling(prof, user)
    return prof
