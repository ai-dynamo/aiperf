# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import secrets
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aiperf.common.scenario.base import (
    ScenarioLockError,
    ScenarioViolation,
)
from aiperf.common.scenario.registry import get_scenario

if TYPE_CHECKING:
    from aiperf.common.config.user_config import UserConfig

_logger = logging.getLogger(__name__)

_AGENTX_SCENARIO = "inferencex-agentx-mvp"
_AGENTX_WEKA_HF_REPO = "semianalysisai/cc-traces-weka-062126"


@dataclass
class ValidationOutcome:
    """Result of running scenario validation against a user config."""

    violations: list[ScenarioViolation] = field(default_factory=list)
    """All scenario invariant conflicts collected in one validation pass."""

    submission_valid: bool | None = None
    """True if scenario lock is satisfied, False under --unsafe-override with violations, None when no scenario set."""

    submission_invalid_reasons: list[str] = field(default_factory=list)
    """Short tags explaining why submission_valid is False (e.g. 'unsafe_override')."""


def _extract_extra_inputs(user_config: Any) -> dict:
    """Return the parsed extra_inputs as a dict regardless of underlying shape."""
    raw = getattr(user_config.input, "extra_inputs_parsed", None)
    if raw is None:
        raw = getattr(user_config.input, "extra", None)
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    try:
        return dict(raw)
    except (TypeError, ValueError):
        return {}


def _is_truthy_extra_input(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _is_falsy_extra_input(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value.strip().lower() in ("false", "0", "no")
    if isinstance(value, (int, float)):
        return value == 0
    return False


# Fields whose presence in `loadgen.model_fields_set` indicates the user
# explicitly drove `_timing_mode` selection in `UserConfig.validate_timing_mode`.
# Keep in sync with that validator (user_config.py).
_TIMING_MODE_DRIVERS: tuple[str, ...] = (
    "request_rate",
    "arrival_pattern",
    "user_centric_rate",
    "request_rate_ramp_duration",
)
_INPUT_TIMING_MODE_DRIVERS: tuple[str, ...] = (
    "fixed_schedule",
    "fixed_schedule_auto_offset",
    "fixed_schedule_start_offset",
    "fixed_schedule_end_offset",
)


def _derive_timing_mode_explicit(user_config: Any) -> bool:
    """True iff the user explicitly drove `_timing_mode` selection.

    Reads `model_fields_set` on the relevant sub-configs. Falls back to the
    `_timing_mode_explicitly_set` attribute that some MagicMock test fixtures
    stamp directly.
    """
    loadgen = getattr(user_config, "loadgen", None)
    loadgen_fields = getattr(loadgen, "model_fields_set", None)
    if isinstance(loadgen_fields, (set, frozenset)) and any(
        name in loadgen_fields for name in _TIMING_MODE_DRIVERS
    ):
        return True
    input_cfg = getattr(user_config, "input", None)
    input_fields = getattr(input_cfg, "model_fields_set", None)
    if isinstance(input_fields, (set, frozenset)) and any(
        name in input_fields for name in _INPUT_TIMING_MODE_DRIVERS
    ):
        return True
    # Fallback: MagicMock test fixtures stamp this directly.
    return bool(getattr(user_config, "_timing_mode_explicitly_set", False))


def validate_scenario(
    user_config: UserConfig | Any,
    *,
    timing_mode_explicit: bool | None = None,
) -> ValidationOutcome:
    """Validate user_config against the locked scenario invariants.

    Run from UserConfig.model_post_init AFTER extra_inputs parsing and AFTER
    loader auto-detection. If --scenario is unset, returns a no-op outcome.

    Args:
        user_config: The fully-validated UserConfig (or test mock).
        timing_mode_explicit: When provided, overrides the auto-derivation of
            "did the user explicitly set timing-mode-driving fields?". The
            production caller computes this from `model_fields_set` and passes
            it in; test callers may omit it to use auto-derivation.
    """
    scenario_name = getattr(user_config, "scenario", None)
    if scenario_name is None:
        return ValidationOutcome()

    spec = get_scenario(scenario_name)
    violations: list[ScenarioViolation] = []
    extra_inputs = _extract_extra_inputs(user_config)

    actual_mode = user_config.timing_mode
    if actual_mode != spec.timing_mode:
        explicit = (
            timing_mode_explicit
            if timing_mode_explicit is not None
            else _derive_timing_mode_explicit(user_config)
        )
        if explicit:
            violations.append(
                ScenarioViolation(
                    flag="--request-rate / --user-centric-rate / --fixed-schedule",
                    current_value=str(actual_mode),
                    required_value=str(spec.timing_mode),
                    message=(
                        f"scenario {spec.name!r} requires timing_mode={spec.timing_mode}; "
                        "do not pass --request-rate / --arrival-pattern / "
                        "--user-centric-rate / --fixed-schedule (or related flags) "
                        "alongside --scenario"
                    ),
                )
            )
        else:
            # `timing_mode` is a read-only property on UserConfig backed by
            # `_timing_mode`. With --scenario alone, the property falls through
            # to REQUEST_RATE default; override the underlying storage.
            user_config._timing_mode = spec.timing_mode
            _logger.info(
                "Scenario %r: setting timing_mode=%s (was at default %s).",
                spec.name,
                spec.timing_mode,
                actual_mode,
            )

    if spec.require_ignore_eos:
        ignore_eos = extra_inputs.get("ignore_eos")
        if ignore_eos is None:
            extra_inputs["ignore_eos"] = True
            user_config.input.extra_inputs_parsed = extra_inputs
            # Mirror into the user-facing `extra` so the wire payload includes
            # the injection. EndpointInfo.from_user_config passes input.extra
            # straight to EndpointInfo.extra: list[tuple[str, Any]].
            with suppress(TypeError, ValueError):
                user_config.input.extra = list(extra_inputs.items())
            _logger.info(
                "Scenario %r: injecting extra_inputs.ignore_eos=true (was absent).",
                spec.name,
            )
        elif _is_falsy_extra_input(ignore_eos):
            violations.append(
                ScenarioViolation(
                    flag="extra_inputs.ignore_eos",
                    current_value=ignore_eos,
                    required_value=True,
                    message=f"scenario {spec.name!r} requires ignore_eos=true",
                )
            )

    if spec.require_use_think_time_only:
        explicit = getattr(
            user_config.input, "_use_think_time_only_explicitly_set", False
        )
        if not user_config.input.use_think_time_only:
            if explicit:
                violations.append(
                    ScenarioViolation(
                        flag="--use-think-time-only",
                        current_value=False,
                        required_value=True,
                        message=f"scenario {spec.name!r} requires --use-think-time-only=true",
                    )
                )
            else:
                user_config.input.use_think_time_only = True
                _logger.info(
                    "Scenario %r: forcing --use-think-time-only=true (was unset).",
                    spec.name,
                )

    if spec.require_use_end_to_start_delays:
        explicit = "use_end_to_start_delays" in user_config.input.model_fields_set
        if not user_config.input.use_end_to_start_delays:
            if explicit:
                violations.append(
                    ScenarioViolation(
                        flag="--use-end-to-start-delays",
                        current_value=False,
                        required_value=True,
                        message=f"scenario {spec.name!r} requires --use-end-to-start-delays=true",
                    )
                )
            else:
                user_config.input.use_end_to_start_delays = True
                _logger.info(
                    "Scenario %r: forcing --use-end-to-start-delays=true (was unset).",
                    spec.name,
                )

    if spec.require_streaming:
        explicit = getattr(user_config.endpoint, "_streaming_explicitly_set", False)
        if not user_config.endpoint.streaming:
            if explicit:
                violations.append(
                    ScenarioViolation(
                        flag="--streaming",
                        current_value=False,
                        required_value=True,
                        message=(
                            f"scenario {spec.name!r} requires --streaming; the "
                            "per-token latency metrics (TTFT, ITL) are core to "
                            "this benchmark and need streaming responses"
                        ),
                    )
                )
            else:
                user_config.endpoint.streaming = True
                _logger.info(
                    "Scenario %r: forcing --streaming=true (was unset).",
                    spec.name,
                )

    if user_config.input.ignore_trace_delays and spec.forbid_ignore_trace_delays:
        violations.append(
            ScenarioViolation(
                flag="--ignore-trace-delays",
                current_value=True,
                required_value=False,
                message=(
                    f"scenario {spec.name!r} replays recorded trace timing; "
                    "--ignore-trace-delays would null every per-turn "
                    "timestamp/delay and dispatch all turns back-to-back, "
                    "falsifying the workload"
                ),
            )
        )

    if spec.forbid_input_truncation:
        synthesis = getattr(user_config.input, "synthesis", None)
        max_isl = getattr(synthesis, "max_isl", None)
        if max_isl is not None:
            violations.append(
                ScenarioViolation(
                    flag="--synthesis-max-isl",
                    current_value=max_isl,
                    required_value=None,
                    message=(
                        f"scenario {spec.name!r} forbids client-side input "
                        "truncation; --synthesis-max-isl drops traces whose "
                        "input length exceeds the cap, falsifying the workload"
                    ),
                )
            )

    detected = getattr(user_config.input, "detected_loader", None)
    if spec.require_loader is not None:
        allowed = (
            (spec.require_loader,)
            if isinstance(spec.require_loader, str)
            else tuple(spec.require_loader)
        )
        if detected not in allowed:
            display = allowed[0] if len(allowed) == 1 else f"any of {sorted(allowed)}"
            violations.append(
                ScenarioViolation(
                    flag="--input-file (loader)",
                    current_value=detected,
                    required_value=display,
                    message=f"scenario {spec.name!r} requires loader={display}",
                )
            )
        if spec.name == _AGENTX_SCENARIO and detected == "weka_hf":
            hf_weka_dataset = getattr(user_config.input, "hf_weka_dataset", None)
            if hf_weka_dataset != _AGENTX_WEKA_HF_REPO:
                violations.append(
                    ScenarioViolation(
                        flag="--hf-weka-dataset",
                        current_value=hf_weka_dataset,
                        required_value=_AGENTX_WEKA_HF_REPO,
                        message=(
                            f"scenario {spec.name!r} only allows --public-dataset "
                            f"weka_hf with hf_weka_dataset={_AGENTX_WEKA_HF_REPO}"
                        ),
                    )
                )

    if spec.require_cache_bust is not None:
        cache_bust_cfg = getattr(
            getattr(getattr(user_config, "input", None), "prompt", None),
            "cache_bust",
            None,
        )
        actual_cache_bust = getattr(cache_bust_cfg, "target", None)
        cache_bust_explicit = getattr(cache_bust_cfg, "_target_explicitly_set", False)
        if actual_cache_bust != spec.require_cache_bust:
            if cache_bust_explicit:
                violations.append(
                    ScenarioViolation(
                        flag="--cache-bust",
                        current_value=str(actual_cache_bust),
                        required_value=str(spec.require_cache_bust),
                        message=(
                            f"scenario {spec.name!r} requires "
                            f"cache_bust.target={spec.require_cache_bust}; "
                            f"got {actual_cache_bust}"
                        ),
                    )
                )
            elif cache_bust_cfg is not None:
                cache_bust_cfg.target = spec.require_cache_bust
                _logger.info(
                    "Scenario %r: auto-set --cache-bust=%s (was at default %s).",
                    spec.name,
                    spec.require_cache_bust,
                    actual_cache_bust,
                )

    # Reject parameter sweeps for fixed-spec scenarios. `--concurrency`
    # accepts comma-separated lists for sweeping; list-shape values must be
    # rejected here — a scenario locks one fixed configuration
    # and a sweep would multiply it into N runs with diverging settings.
    concurrency = getattr(user_config.loadgen, "concurrency", None)
    if isinstance(concurrency, list):
        violations.append(
            ScenarioViolation(
                flag="--concurrency",
                current_value=concurrency,
                required_value="int",
                message=(
                    f"scenario {spec.name!r} does not support parameter sweeps; "
                    "pass a single --concurrency value instead of a list"
                ),
            )
        )

    if (
        spec.default_benchmark_duration_seconds is not None
        and user_config.loadgen.benchmark_duration is None
    ):
        user_config.loadgen.benchmark_duration = float(
            spec.default_benchmark_duration_seconds
        )
        _logger.info(
            "Scenario %r: auto-set --benchmark-duration=%s (was unset).",
            spec.name,
            spec.default_benchmark_duration_seconds,
        )

    duration = user_config.loadgen.benchmark_duration or 0.0
    if duration < spec.min_benchmark_duration_seconds:
        violations.append(
            ScenarioViolation(
                flag="--benchmark-duration",
                current_value=duration,
                required_value=f">={spec.min_benchmark_duration_seconds}",
                message=(
                    f"scenario {spec.name!r} requires duration >= "
                    f"{spec.min_benchmark_duration_seconds}s to reach steady "
                    "state and trigger KV offloading"
                ),
            )
        )

    for ratio_field, spec_default in (
        ("trajectory_start_min_ratio", spec.default_trajectory_start_min_ratio),
        ("trajectory_start_max_ratio", spec.default_trajectory_start_max_ratio),
    ):
        if spec_default is None:
            continue
        explicit = getattr(user_config.loadgen, f"_{ratio_field}_explicitly_set", False)
        if not explicit:
            current = getattr(user_config.loadgen, ratio_field, None)
            if current != spec_default:
                setattr(user_config.loadgen, ratio_field, spec_default)
                _logger.info(
                    "Scenario %r: auto-set --%s=%s (was at default %s).",
                    spec.name,
                    ratio_field.replace("_", "-"),
                    spec_default,
                    current,
                )

    if user_config.input.random_seed is None:
        seed = secrets.randbits(63)
        user_config.input.random_seed = seed
        _logger.info(
            "Scenario %r: auto-set random_seed=%d (was unset).", spec.name, seed
        )

    if spec.inter_turn_delay_cap_seconds is not None:
        cap_explicit = getattr(
            user_config.loadgen, "_inter_turn_delay_cap_explicitly_set", False
        )
        cap = user_config.loadgen.inter_turn_delay_cap_seconds
        if cap_explicit:
            if cap != spec.inter_turn_delay_cap_seconds:
                violations.append(
                    ScenarioViolation(
                        flag="--inter-turn-delay-cap-seconds",
                        current_value=cap,
                        required_value=spec.inter_turn_delay_cap_seconds,
                        message=f"scenario {spec.name!r} locks the cap to {spec.inter_turn_delay_cap_seconds}",
                    )
                )
        elif cap is None:
            user_config.loadgen.inter_turn_delay_cap_seconds = (
                spec.inter_turn_delay_cap_seconds
            )
            _logger.info(
                "Scenario %r: auto-set --inter-turn-delay-cap-seconds=%s (was unset).",
                spec.name,
                spec.inter_turn_delay_cap_seconds,
            )

    if spec.trace_idle_gap_cap_seconds is not None:
        idle_explicit = getattr(
            user_config.loadgen, "_trace_idle_gap_cap_explicitly_set", False
        )
        idle = user_config.loadgen.trace_idle_gap_cap_seconds
        if idle_explicit:
            if idle != spec.trace_idle_gap_cap_seconds:
                violations.append(
                    ScenarioViolation(
                        flag="--trace-idle-gap-cap-seconds",
                        current_value=idle,
                        required_value=spec.trace_idle_gap_cap_seconds,
                        message=f"scenario {spec.name!r} locks the per-trace idle-gap cap to {spec.trace_idle_gap_cap_seconds}",
                    )
                )
        elif idle is None:
            user_config.loadgen.trace_idle_gap_cap_seconds = (
                spec.trace_idle_gap_cap_seconds
            )
            _logger.info(
                "Scenario %r: auto-set --trace-idle-gap-cap-seconds=%s (was unset).",
                spec.name,
                spec.trace_idle_gap_cap_seconds,
            )

    unsafe = bool(getattr(user_config, "unsafe_override", False))
    if violations and not unsafe:
        raise ScenarioLockError(violations)

    if violations and unsafe:
        for v in violations:
            _logger.warning("Scenario violation (override active): %s", v)
        return ValidationOutcome(
            violations=violations,
            submission_valid=False,
            submission_invalid_reasons=["unsafe_override"],
        )

    return ValidationOutcome(violations=[], submission_valid=True)
