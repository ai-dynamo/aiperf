# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenario invariant lock applied as a v2 config-resolver step.

``apply_scenario(run)`` reads ``run.cfg.scenario``; when set, it looks up the
:class:`ScenarioSpec` and applies/validates each invariant against the v2
``BenchmarkConfig`` (``run.cfg``) and the resolved runtime state
(``run.resolved``). Defaults are auto-filled (info log); explicit user
conflicts raise :class:`ScenarioLockError` unless ``run.cfg.unsafe_override``
downgrades them to warnings and stamps ``submission_valid=False``.

This is the v2 successor to the old ``validate_scenario(user_config)`` model
validator that mutated a v1 ``UserConfig``. It is invoked from the
``ScenarioResolver`` step in the pre-bootstrap resolver chain (after the
``DatasetResolver`` populates ``run.resolved.dataset_types`` and before the
``TimingResolver`` sums phase durations), so the locked ``timing_mode`` and
``duration`` are in place before downstream resolution reads them.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.scenario.base import (
    ScenarioLockError,
    ScenarioOutcome,
    ScenarioSpec,
    ScenarioViolation,
)
from aiperf.common.scenario.registry import get_scenario
from aiperf.plugin.enums import PhaseType, TimingMode

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

_logger = AIPerfLogger(__name__)

_AGENTX_SCENARIO = "inferencex-agentx-mvp"

# Phase types that only arise from an explicit scheduling request
# (--request-rate / --user-centric-rate / --fixed-schedule or a YAML phase
# declaration). A scenario that locks its own timing_mode conflicts with
# these rather than silently overriding them; the default concurrency shape
# is the only one the lock may stamp.
_CONFLICTING_PHASE_TYPES = frozenset(
    {
        PhaseType.POISSON,
        PhaseType.GAMMA,
        PhaseType.CONSTANT,
        PhaseType.USER_CENTRIC,
        PhaseType.FIXED_SCHEDULE,
    }
)
_AGENTX_WEKA_HF_REPO = "semianalysisai/cc-traces-weka-062126"
_WEKA_HF_LOADER = "weka_hf"
_WEKA_TRACE_LOADER = "weka_trace"


def apply_scenario(run: BenchmarkRun) -> ScenarioOutcome:
    """Apply the locked scenario invariants to ``run.cfg`` / ``run.resolved``.

    Reads ``run.cfg.scenario``; when None returns a no-op outcome. Otherwise
    looks up the :class:`ScenarioSpec`, mutates ``run.cfg`` for each auto-filled
    default and validates each hard invariant, collecting violations. Raises
    :class:`ScenarioLockError` on a hard conflict unless ``run.cfg.unsafe_override``
    is set (which downgrades violations to warnings and stamps
    ``submission_valid=False``). One exception: a synthetic default dataset
    under ``require_loader`` always raises, even with ``unsafe_override``,
    because continuing would silently build unusable 1-turn sessions.
    The result is stored on ``run.resolved.scenario_outcome`` and also returned.
    """
    scenario_name = getattr(run.cfg, "scenario", None)
    if scenario_name is None:
        outcome = ScenarioOutcome()
        run.resolved.scenario_outcome = outcome
        return outcome

    spec = get_scenario(scenario_name)
    violations: list[ScenarioViolation] = []
    applied: list[str] = []

    _apply_timing_mode(run, spec, violations, applied)
    _apply_require_streaming(run, spec, violations, applied)
    _apply_require_ignore_eos(run, spec, violations, applied)
    _apply_forbid_ignore_trace_delays(run, spec, violations, applied)
    _apply_use_think_time_only(run, spec, violations, applied)
    _apply_forbid_input_truncation(run, spec, violations, applied)
    _apply_require_loader(run, spec, violations, applied)
    _apply_require_cache_bust(run, spec, violations, applied)
    _apply_duration(run, spec, violations, applied)
    _apply_trajectory_ratios(run, spec, applied)
    _apply_inter_turn_delay_cap(run, spec, violations, applied)
    _apply_trace_idle_gap_cap(run, spec, violations, applied)
    _apply_system_idle_gap_cap(run, spec, violations, applied)
    _apply_random_seed(run, spec, applied)

    # Synthetic (CLI default when no --public-dataset/--input-file) under a
    # require_loader scenario is never a meaningful override — it silently
    # builds 1-turn Shakespeare sessions that empty the AgentX trajectory
    # pool. Fail hard even with --unsafe-override.
    hard = _non_overridable_violations(run, violations)
    if hard:
        raise ScenarioLockError(violations, bypassable=False)

    unsafe = bool(getattr(run.cfg, "unsafe_override", False))
    if violations and not unsafe:
        raise ScenarioLockError(violations)

    if violations and unsafe:
        for v in violations:
            _logger.warning("Scenario violation (override active): %s", v)
        outcome = ScenarioOutcome(
            scenario_name=spec.name,
            applied_locks=applied,
            violations=violations,
            submission_valid=False,
            submission_invalid_reasons=["unsafe_override"],
        )
        run.resolved.scenario_outcome = outcome
        return outcome

    outcome = ScenarioOutcome(
        scenario_name=spec.name,
        applied_locks=applied,
        violations=[],
        submission_valid=True,
    )
    run.resolved.scenario_outcome = outcome
    return outcome


def _non_overridable_violations(
    run: BenchmarkRun,
    violations: list[ScenarioViolation],
) -> list[ScenarioViolation]:
    """Return loader violations that ``--unsafe-override`` must not downgrade.

    Explicit wrong loaders (e.g. ``sharegpt``) stay overridable for ablations.
    A synthetic default dataset means the user omitted a weka source entirely;
    continuing would look like a corpus load while producing unusable 1-turn
    sessions.
    """
    from aiperf.config.dataset import SyntheticDataset

    dataset = run.cfg.get_default_dataset()
    if not isinstance(dataset, SyntheticDataset):
        return []
    return [v for v in violations if v.flag == "--input-file (loader)"]


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


def _apply_timing_mode(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Stamp ``spec.timing_mode`` onto every profiling phase.

    v2 has no top-level timing_mode: the credit pipeline derives it per-phase
    from ``phase.type`` unless ``phase.timing_mode`` overrides it (read by
    ``aiperf.timing.config._is_agentic_replay`` / ``_build_profiling_config``).
    The scenario locks AGENTIC_REPLAY by setting that per-phase override.

    An explicitly requested scheduling mode conflicts with the lock rather
    than being silently overridden: a rate / user-centric / fixed-schedule
    phase only arises from the corresponding CLI flags (``--request-rate`` /
    ``--user-centric-rate`` / ``--fixed-schedule``) or an explicit YAML phase
    type, so such a phase raises a violation (v1 parity), and the phase is
    left un-stamped so an ``--unsafe-override`` run keeps the user's mode.
    Only the default concurrency shape accepts the scenario's override.
    """
    changed = False
    for phase in run.cfg.get_profiling_phases():
        if phase.type in _CONFLICTING_PHASE_TYPES:
            violations.append(
                ScenarioViolation(
                    flag="--request-rate / --user-centric-rate / --fixed-schedule",
                    current_value=str(phase.type),
                    required_value=str(spec.timing_mode),
                    message=(
                        f"scenario {spec.name!r} requires timing_mode={spec.timing_mode}; "
                        "do not pass --request-rate / --arrival-pattern / "
                        "--user-centric-rate / --fixed-schedule (or declare a "
                        "rate-shaped phase) alongside --scenario"
                    ),
                )
            )
            continue
        # --adaptive-scale keeps the default concurrency phase type and flips
        # a per-phase flag instead, so the phase-type gate above cannot see
        # it. Without this check the lock's explicit per-phase timing_mode
        # would silently win over ADAPTIVE_SCALE in _build_profiling_config —
        # the flag would be accepted and ignored.
        if getattr(phase, "adaptive_scale", False):
            violations.append(
                ScenarioViolation(
                    flag="--adaptive-scale",
                    current_value=str(TimingMode.ADAPTIVE_SCALE),
                    required_value=str(spec.timing_mode),
                    message=(
                        f"scenario {spec.name!r} requires timing_mode={spec.timing_mode}; "
                        "do not pass --adaptive-scale (or set adaptive_scale on a "
                        "phase) alongside --scenario"
                    ),
                )
            )
            continue
        if getattr(phase, "timing_mode", None) != spec.timing_mode:
            phase.timing_mode = spec.timing_mode
            changed = True
    if changed:
        _logger.info(
            "Scenario %r: setting profiling-phase timing_mode=%s.",
            spec.name,
            spec.timing_mode,
        )
    applied.append("timing_mode")


def _apply_require_streaming(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    if not spec.require_streaming:
        return
    endpoint = run.cfg.endpoint
    if endpoint.streaming:
        applied.append("streaming")
        return
    explicit = getattr(endpoint, "_streaming_explicitly_set", False)
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
        endpoint.streaming = True
        _logger.info("Scenario %r: forcing --streaming=true (was unset).", spec.name)
        applied.append("streaming")


def _apply_require_ignore_eos(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Inject ``ignore_eos=true`` into ``endpoint.extra`` (the wire body).

    v2 has no ``extra_inputs_parsed``: ``--extra-inputs`` lands directly on
    ``EndpointConfig.extra`` (a dict merged into every request body by the
    endpoint formatters). Auto-injects when absent; raises when explicitly
    falsy.
    """
    if not spec.require_ignore_eos:
        return
    extra = run.cfg.endpoint.extra
    ignore_eos = extra.get("ignore_eos")
    if ignore_eos is None:
        extra["ignore_eos"] = True
        _logger.info(
            "Scenario %r: injecting extra_inputs.ignore_eos=true (was absent).",
            spec.name,
        )
        applied.append("ignore_eos")
    elif _is_falsy_extra_input(ignore_eos):
        violations.append(
            ScenarioViolation(
                flag="extra_inputs.ignore_eos",
                current_value=ignore_eos,
                required_value=True,
                message=f"scenario {spec.name!r} requires ignore_eos=true",
            )
        )
    else:
        applied.append("ignore_eos")


def _apply_forbid_ignore_trace_delays(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    if not spec.forbid_ignore_trace_delays:
        return
    dataset = run.cfg.get_default_dataset()
    if getattr(dataset, "ignore_trace_delays", False):
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
    else:
        applied.append("forbid_ignore_trace_delays")


def _apply_use_think_time_only(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    if not spec.require_use_think_time_only:
        return
    dataset = run.cfg.get_default_dataset()
    if not hasattr(dataset, "use_think_time_only"):
        # Synthetic datasets have no think-time-only concept; the authoritative
        # violation comes from _apply_require_loader, not an auto-fill crash.
        return
    if getattr(dataset, "use_think_time_only", False):
        applied.append("use_think_time_only")
        return
    explicit = getattr(dataset, "_use_think_time_only_explicitly_set", False)
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
        dataset.use_think_time_only = True
        _logger.info(
            "Scenario %r: forcing --use-think-time-only=true (was unset).", spec.name
        )
        applied.append("use_think_time_only")


def _apply_forbid_input_truncation(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Reject client-side input truncation.

    v2 home: ``FileDataset.synthesis.max_isl`` (the synthesis-time ISL filter
    that drops over-length traces). ``SyntheticDataset`` has no ``synthesis``
    block; ``PublicDataset`` has no ``synthesis`` either, so this is a no-op for
    those dataset kinds (the lock only bites on file-based trace synthesis).
    """
    if not spec.forbid_input_truncation:
        return
    dataset = run.cfg.get_default_dataset()
    synthesis = getattr(dataset, "synthesis", None)
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
    else:
        applied.append("forbid_input_truncation")


def _detect_loader(run: BenchmarkRun) -> str | None:
    """Derive the active dataset's loader identity for the require_loader check.

    For PublicDataset, the loader is the ``dataset`` enum value (e.g.
    ``weka_hf`` / ``semianalysis_cc_traces_weka_*``). For FileDataset, the
    DatasetResolver populated ``run.resolved.dataset_types`` keyed by dataset
    name with a ``CustomDatasetType`` (e.g. ``weka_trace``). Returns the string
    form, or None when no loader could be resolved.
    """
    from aiperf.config.dataset import FileDataset, PublicDataset

    dataset = run.cfg.get_default_dataset()
    if isinstance(dataset, PublicDataset):
        return str(dataset.dataset)
    if isinstance(dataset, FileDataset):
        types = run.resolved.dataset_types or {}
        detected = types.get(dataset.name)
        if detected is not None:
            return str(detected)
    return None


def _apply_require_loader(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    if spec.require_loader is None:
        return
    allowed = (
        (spec.require_loader,)
        if isinstance(spec.require_loader, str)
        else tuple(spec.require_loader)
    )
    detected = _detect_loader(run)
    if detected not in allowed:
        display = allowed[0] if len(allowed) == 1 else f"any of {sorted(allowed)}"
        from aiperf.config.dataset import SyntheticDataset

        dataset = run.cfg.get_default_dataset()
        if isinstance(dataset, SyntheticDataset):
            current: str | None = "synthetic"
            message = (
                f"scenario {spec.name!r} requires loader={display}; "
                "got synthetic (CLI default when --public-dataset / "
                "--input-file is omitted). --unsafe-override cannot bypass "
                "a missing weka loader"
            )
        else:
            current = detected
            message = f"scenario {spec.name!r} requires loader={display}"
        violations.append(
            ScenarioViolation(
                flag="--input-file (loader)",
                current_value=current,
                required_value=display,
                message=message,
            )
        )
    else:
        applied.append("require_loader")

    # AgentX submission_valid requires a pinned corpus identity. Public-dataset
    # aliases are pinned by name; weka_hf is pinned to a known HF repo below.
    # Local weka_trace only proves format compatibility — any directory of
    # Weka-shaped JSON passes — so it cannot stamp submission_valid=true.
    if spec.name == _AGENTX_SCENARIO and detected == _WEKA_TRACE_LOADER:
        violations.append(
            ScenarioViolation(
                flag="--custom-dataset-type / --input-file",
                current_value="weka_trace (local, unpinned)",
                required_value=(
                    "pinned --public-dataset alias or "
                    f"--hf-weka-dataset {_AGENTX_WEKA_HF_REPO}"
                ),
                message=(
                    f"scenario {spec.name!r} cannot verify corpus identity for a "
                    "local weka_trace directory; use a with-subagents "
                    "--public-dataset alias or "
                    f"--hf-weka-dataset {_AGENTX_WEKA_HF_REPO} for "
                    "submission_valid=true, or pass --unsafe-override for "
                    "offline smoke tests (submission_valid=false)"
                ),
            )
        )

    if spec.name == _AGENTX_SCENARIO and detected == _WEKA_HF_LOADER:
        dataset = run.cfg.get_default_dataset()
        hf_weka_dataset = getattr(dataset, "hf_weka_dataset", None)
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


def _apply_require_cache_bust(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Validate / auto-fill the active dataset's cache-bust target.

    The home differs by dataset kind: synthetic -> ``prompts.cache_bust``,
    file/public trace -> ``cache_bust``. ``get_cache_bust_target()`` reads the
    same precedence; here we resolve the concrete ``CacheBustConfig`` to mutate
    it (auto-fill) or read its explicit-set flag.
    """
    if spec.require_cache_bust is None:
        return
    cache_bust_cfg = _resolve_cache_bust_config(run)
    actual = run.cfg.get_cache_bust_target()
    if actual == spec.require_cache_bust:
        applied.append("cache_bust")
        return
    explicit = getattr(cache_bust_cfg, "_target_explicitly_set", False)
    if explicit:
        violations.append(
            ScenarioViolation(
                flag="--cache-bust",
                current_value=str(actual),
                required_value=str(spec.require_cache_bust),
                message=(
                    f"scenario {spec.name!r} requires "
                    f"cache_bust.target={spec.require_cache_bust}; got {actual}"
                ),
            )
        )
    elif cache_bust_cfg is not None:
        cache_bust_cfg.target = spec.require_cache_bust
        _logger.info(
            "Scenario %r: auto-set --cache-bust=%s (was at default %s).",
            spec.name,
            spec.require_cache_bust,
            actual,
        )
        applied.append("cache_bust")


def _resolve_cache_bust_config(run: BenchmarkRun) -> Any | None:
    """Return the active dataset's ``CacheBustConfig`` instance to mutate.

    Synthetic datasets store it on ``prompts.cache_bust`` (materializing a
    default ``PromptConfig`` is out of scope here -- a synthetic scenario run
    without prompts would carry no cache-bust home); file/public datasets store
    it directly on ``cache_bust``.
    """
    dataset = run.cfg.get_default_dataset()
    prompts = getattr(dataset, "prompts", None)
    if prompts is not None:
        cache_bust = getattr(prompts, "cache_bust", None)
        if cache_bust is not None:
            return cache_bust
    return getattr(dataset, "cache_bust", None)


def _apply_duration(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Auto-fill / enforce the profiling-phase duration floor.

    v2 duration lives per-phase (``BasePhaseConfig.duration``); the scenario
    applies its default to every profiling phase whose duration is unset and
    enforces the floor on each.
    """
    profiling_phases = run.cfg.get_profiling_phases()
    if spec.default_benchmark_duration_seconds is not None:
        for phase in profiling_phases:
            if phase.duration is None:
                phase.duration = float(spec.default_benchmark_duration_seconds)
                _logger.info(
                    "Scenario %r: auto-set --benchmark-duration=%s (was unset).",
                    spec.name,
                    spec.default_benchmark_duration_seconds,
                )
        applied.append("default_benchmark_duration")

    for phase in profiling_phases:
        duration = phase.duration or 0.0
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
    if not any(
        (phase.duration or 0.0) < spec.min_benchmark_duration_seconds
        for phase in profiling_phases
    ):
        applied.append("min_benchmark_duration")


def _apply_trajectory_ratios(
    run: BenchmarkRun, spec: ScenarioSpec, applied: list[str]
) -> None:
    """Auto-fill trajectory start ratios on profiling phases (explicit honored)."""
    fields = (
        ("trajectory_start_min_ratio", spec.default_trajectory_start_min_ratio),
        ("trajectory_start_max_ratio", spec.default_trajectory_start_max_ratio),
    )
    for phase in run.cfg.get_profiling_phases():
        for ratio_field, spec_default in fields:
            if spec_default is None:
                continue
            explicit = getattr(phase, f"_{ratio_field}_explicitly_set", False)
            if explicit:
                continue
            current = getattr(phase, ratio_field, None)
            if current != spec_default:
                setattr(phase, ratio_field, spec_default)
                _logger.info(
                    "Scenario %r: auto-set --%s=%s (was at default %s).",
                    spec.name,
                    ratio_field.replace("_", "-"),
                    spec_default,
                    current,
                )
            # Deliberately NOT marking _{ratio_field}_explicitly_set here: that
            # flag means "the user set this before scenario apply ran", and the
            # conflict check above depends on it staying frozen. Downstream
            # readers of deliberateness use ``model_fields_set``, which the
            # ``setattr`` above already updates.
    if (
        spec.default_trajectory_start_min_ratio is not None
        or spec.default_trajectory_start_max_ratio is not None
    ):
        applied.append("trajectory_start_ratios")


def _apply_inter_turn_delay_cap(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    dataset = run.cfg.get_default_dataset()
    if not hasattr(dataset, "inter_turn_delay_cap_seconds"):
        # Synthetic datasets have no inter-turn delay-cap concept; nothing to lock.
        return
    cap = dataset.inter_turn_delay_cap_seconds
    if spec.forbid_inter_turn_delay_cap and cap is not None:
        violations.append(
            ScenarioViolation(
                flag="--inter-turn-delay-cap-seconds",
                current_value=cap,
                required_value=None,
                message=(
                    f"scenario {spec.name!r} preserves original inter-turn "
                    "timing and forbids a per-turn delay cap"
                ),
            )
        )
        return
    if spec.inter_turn_delay_cap_seconds is None:
        return
    explicit = "inter_turn_delay_cap_seconds" in getattr(
        dataset, "model_fields_set", ()
    )
    if explicit:
        if cap != spec.inter_turn_delay_cap_seconds:
            violations.append(
                ScenarioViolation(
                    flag="--inter-turn-delay-cap-seconds",
                    current_value=cap,
                    required_value=spec.inter_turn_delay_cap_seconds,
                    message=(
                        f"scenario {spec.name!r} locks the inter-turn delay cap "
                        f"to {spec.inter_turn_delay_cap_seconds}"
                    ),
                )
            )
        else:
            applied.append("inter_turn_delay_cap")
    elif cap is None:
        dataset.inter_turn_delay_cap_seconds = spec.inter_turn_delay_cap_seconds
        _logger.info(
            "Scenario %r: auto-set --inter-turn-delay-cap-seconds=%s (was unset).",
            spec.name,
            spec.inter_turn_delay_cap_seconds,
        )
        applied.append("inter_turn_delay_cap")


def _apply_trace_idle_gap_cap(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    dataset = run.cfg.get_default_dataset()
    if not hasattr(dataset, "trace_idle_gap_cap_seconds"):
        # Synthetic datasets have no trace idle-gap concept; nothing to lock.
        return
    idle = dataset.trace_idle_gap_cap_seconds
    if spec.forbid_trace_idle_gap_cap and idle is not None:
        violations.append(
            ScenarioViolation(
                flag="--trace-idle-gap-cap-seconds",
                current_value=idle,
                required_value=None,
                message=(
                    f"scenario {spec.name!r} preserves original per-trace "
                    "request timing and forbids timeline compression"
                ),
            )
        )
        return
    if spec.trace_idle_gap_cap_seconds is None:
        return
    explicit = "trace_idle_gap_cap_seconds" in getattr(dataset, "model_fields_set", ())
    if explicit:
        if idle != spec.trace_idle_gap_cap_seconds:
            violations.append(
                ScenarioViolation(
                    flag="--trace-idle-gap-cap-seconds",
                    current_value=idle,
                    required_value=spec.trace_idle_gap_cap_seconds,
                    message=(
                        f"scenario {spec.name!r} locks the per-trace idle-gap cap "
                        f"to {spec.trace_idle_gap_cap_seconds}"
                    ),
                )
            )
        else:
            applied.append("trace_idle_gap_cap")
    elif idle is None:
        dataset.trace_idle_gap_cap_seconds = spec.trace_idle_gap_cap_seconds
        _logger.info(
            "Scenario %r: auto-set --trace-idle-gap-cap-seconds=%s (was unset).",
            spec.name,
            spec.trace_idle_gap_cap_seconds,
        )
        applied.append("trace_idle_gap_cap")


def _apply_system_idle_gap_cap(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Lock the global idle cap without altering dataset timestamps."""
    required = spec.system_idle_gap_cap_seconds
    if required is None:
        return

    valid = True
    for phase in run.cfg.get_profiling_phases():
        current = phase.system_idle_gap_cap_seconds
        explicit = phase._system_idle_gap_cap_seconds_explicitly_set
        if explicit and current != required:
            valid = False
            violations.append(
                ScenarioViolation(
                    flag="--system-idle-gap-cap-seconds",
                    current_value=current,
                    required_value=required,
                    message=(
                        f"scenario {spec.name!r} limits only globally idle "
                        f"replay time to {required} seconds"
                    ),
                )
            )
            continue
        if current != required:
            phase.system_idle_gap_cap_seconds = required
            _logger.info(
                "Scenario %r: auto-set --system-idle-gap-cap-seconds=%s (was unset).",
                spec.name,
                required,
            )
    if valid:
        applied.append("system_idle_gap_cap")


def _apply_random_seed(
    run: BenchmarkRun, spec: ScenarioSpec, applied: list[str]
) -> None:
    """Auto-fill a fresh per-run ``random_seed`` when the user left it unset.

    A scenario submission must be reproducible after the fact: the resolved
    seed is recorded in provenance (``RunInfo.random_seed`` ->
    ``profile_export_aiperf.json``) and drives every downstream RNG consumer
    that falls back to the run-level seed -- the bootstrap RNG init
    (``rng.init(run.random_seed)``), synthetic dataset composers, and the
    trace loaders' shuffle/sample seed.

    v2 home: the operative single source for a run's seed is the
    ``BenchmarkRun.random_seed`` field (sourced from ``plan.variation_seeds``).
    The envelope-level ``BenchmarkConfig._apply_consistent_seed_default`` only
    stamps a fixed 42 for sweeps / multi-run trials and returns early for a
    single run, so a single ``--scenario`` run with no ``--random-seed`` would
    otherwise reach the orchestrator with ``random_seed=None`` -- a
    non-reproducible workload. This step closes that gap by stamping a fresh
    random seed (matching the per-run randomization of the v1
    ``validate_scenario`` auto-fill).
    """
    if run.random_seed is None:
        seed = secrets.randbits(63)
        run.random_seed = seed
        _logger.info(
            "Scenario %r: auto-set random_seed=%d (was unset).", spec.name, seed
        )
    applied.append("random_seed")


# PORT-NOTE (require_ignore_eos): v1 also mirrored the injection into the
# user-facing ``input.extra`` list so EndpointInfo.from_user_config picked it
# up. In v2 ``endpoint.extra`` IS the single dict consumed by the endpoint
# formatters (see e.g. ``aiperf/endpoints/openai_chat.py`` reading
# ``model_endpoint.endpoint.extra``), so one write covers the wire payload.
#
# PORT-NOTE (concurrency-sweep rejection): the v1 validator rejected a
# list-shaped ``--concurrency`` here. In v2, list-shaped magic-list flags are
# hoisted to a top-level ``sweep:`` block by the CLI converter
# (``_promote_magic_lists_to_sweep_block``) BEFORE a BenchmarkConfig is ever
# built, and ``phase.concurrency`` is a scalar ``int`` that rejects lists at
# validation time -- so a per-phase scalar is all that reaches this resolver.
# By the time ``apply_scenario`` runs, each variation has already been expanded
# into its own single-cell BenchmarkRun, so a sweep would individually satisfy
# the lock N times. The sweep-vs-scenario rejection therefore lives at the
# envelope level instead, on ``AIPerfConfig._reject_scenario_with_sweep``
# (config/config.py), which errors when ``benchmark.scenario`` is set alongside
# a non-empty ``sweep`` block (downgraded to a warning by
# ``benchmark.unsafe_override``).
