# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenario invariant lock applied as a config-resolver step (graph-IR path).

``apply_scenario(run)`` reads ``run.cfg.scenario``; when set, it looks up the
:class:`ScenarioSpec`, auto-fills unset defaults (info log) and validates each
invariant against ``run.cfg`` / ``run.resolved``. Explicit user conflicts raise
:class:`ScenarioLockError` unless ``run.cfg.unsafe_override`` downgrades them to
warnings and stamps ``submission_valid=False``.

The validator derives
``TimingMode.GRAPH_IR`` from weka graph-workload detection (no per-phase
``timing_mode`` override consumer), stores cache-bust as a bare
``CacheBustTarget`` on ``EndpointConfig.cache_bust`` (no ``CacheBustConfig``
wrapper), and treats the trajectory-start window + idle-gap cap as per-run
config -- their locks (``_env_locks.py``) auto-apply the spec value when the
field is unset and raise only on a user-explicit mismatch.

User-explicit vs default is read from ``model_fields_set`` membership on the
LIVE converted config (the resolver chain mutates ``run`` in place), so
membership is faithful here -- no explicit-set sentinel is required.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aiperf.common.enums import CacheBustTarget
from aiperf.common.scenario._env_locks import (
    apply_concurrency_sweep,
    apply_trace_idle_gap_cap,
    apply_trajectory_ratios,
    pin_weka_hf_repo,
)
from aiperf.common.scenario.base import (
    ScenarioLockError,
    ScenarioOutcome,
    ScenarioSpec,
    ScenarioViolation,
)
from aiperf.common.scenario.registry import get_scenario

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


def _is_graph_workload(run: BenchmarkRun) -> bool:
    """Lazily import + delegate to the memoized graph-workload resolution.

    The import is local to avoid a module-import cycle: the dataset graph
    loader pulls heavy deps, and ``aiperf.common.scenario`` is imported early
    by the config/resolver chain.
    """
    from aiperf.dataset.graph.workload_detect import resolve_graph_workload

    return resolve_graph_workload(run) is not None


_logger = logging.getLogger(__name__)

# Synthetic loader identity returned for any weka graph workload.
_WEKA_GRAPH_LOADER = "weka_trace"


def apply_scenario(run: BenchmarkRun) -> ScenarioOutcome:
    """Apply the locked scenario invariants to ``run.cfg`` / ``run.resolved``.

    Reads ``run.cfg.scenario``; when None returns a no-op outcome. Otherwise
    looks up the spec, auto-fills defaults + validates each invariant, and
    raises :class:`ScenarioLockError` on a conflict unless
    ``run.cfg.unsafe_override`` downgrades to warnings + ``submission_valid=False``.
    The result is stored on ``run.resolved.scenario_outcome`` and returned.
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
    _apply_ignore_eos(run, spec, violations, applied)
    _apply_forbid_input_truncation(run, spec, violations, applied)
    _apply_require_loader(run, spec, violations, applied)
    _apply_require_cache_bust(run, spec, violations, applied)
    apply_concurrency_sweep(run, spec, violations, applied)
    _apply_duration(run, spec, violations, applied)
    apply_trajectory_ratios(run, spec, violations, applied)
    apply_trace_idle_gap_cap(run, spec, violations, applied)

    # There is deliberately NO random_seed auto-fill here (no secrets.randbits
    # fallback when unset). The graph-IR config has no input.random_seed field: the run
    # seed lives on BenchmarkRun.random_seed, threaded deterministically by the
    # orchestrator (resolve_run_seed -> variation_seeds / derive_variation_seed)
    # BEFORE this resolver step runs. Synthesized content is already seed-
    # invariant via resolve_graph_content_seed (which returns --random-seed
    # verbatim, with no weka-specific fallback). A non-deterministic
    # secrets.randbits here would collide with the
    # orchestrator's deterministic derivation and break the content-determinism
    # contract (every parse of the same run must synthesize identical bytes);
    # the observable behavior (a seed is always assigned, the run is
    # reproducible) is identical-or-better without it.

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


def _is_falsy_extra_input(value: Any) -> bool:
    """True when ``value`` is an explicit falsy ``ignore_eos`` extra-input."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value.strip().lower() in ("false", "0", "no")
    if isinstance(value, (int, float)):
        return value == 0
    return False


def _has_config_field(model: Any, field: str) -> bool:
    """True when ``field`` is a real declared Pydantic field on ``model``.

    Used to keep dormant locks honest: a check whose backing config field does
    not exist must NOT claim to be "applied". ``getattr`` alone can't tell a
    real field from an absent one, so this inspects the model's declared fields.
    """
    return field in getattr(type(model), "model_fields", {})


def _apply_timing_mode(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Verify the run is a graph workload (graph-IR is derived, not stamped).

    The target derives ``GRAPH_IR`` from ``resolve_graph_workload(run)`` with
    no per-phase ``timing_mode`` consumer, so this verifies the workload IS
    graph and raises otherwise rather than stamping phases.
    """
    if _is_graph_workload(run):
        applied.append("timing_mode")
        return
    violations.append(
        ScenarioViolation(
            flag="--input-file (timing_mode)",
            current_value="non-graph workload",
            required_value=str(spec.timing_mode),
            message=(
                f"scenario {spec.name!r} requires a weka graph workload "
                f"(timing_mode={spec.timing_mode}); the input is not one"
            ),
        )
    )


def _apply_require_streaming(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Auto-enable ``--streaming`` when unset; violation on explicit ``False``."""
    if not spec.require_streaming:
        return
    endpoint = run.cfg.endpoint
    if endpoint.streaming:
        applied.append("streaming")
        return
    explicit = "streaming" in endpoint.model_fields_set
    if explicit:
        violations.append(
            ScenarioViolation(
                flag="--streaming",
                current_value=False,
                required_value=True,
                message=(
                    f"scenario {spec.name!r} requires --streaming; the "
                    "per-token latency metrics (TTFT, ITL) need streaming"
                ),
            )
        )
    else:
        endpoint.streaming = True
        _logger.info("Scenario %r: forcing --streaming=true (was unset).", spec.name)
        applied.append("streaming")


def _apply_ignore_eos(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Inject ``ignore_eos=true`` into ``endpoint.extra`` (the wire body).

    ``EndpointConfig.extra`` is merged into every request body by the formatters.
    Auto-injects when absent; raises when explicitly falsy.
    """
    if not spec.require_ignore_eos:
        return
    extra = run.cfg.endpoint.extra
    ignore_eos = extra.get("ignore_eos")
    if ignore_eos is None:
        extra["ignore_eos"] = True
        _logger.info("Scenario %r: injecting ignore_eos=true (was absent).", spec.name)
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


def _apply_forbid_input_truncation(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Reject ``--synthesis-max-isl`` (the ``FileDataset.synthesis.max_isl`` ISL
    filter). A weka graph dataset without a ``synthesis`` block has no such
    field, so this degrades to a graceful no-op.
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
                    "truncation; --synthesis-max-isl drops over-length traces, "
                    "falsifying the workload"
                ),
            )
        )
    else:
        applied.append("forbid_input_truncation")


def _apply_require_loader(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Require the active loader to be a canonical weka graph workload.

    The DatasetResolver does NOT populate ``dataset_types`` for weka inputs, so
    detection is the graph-workload sniff: a weka graph workload reports
    ``"weka_trace"``; else None. When the input is a HuggingFace dataset id
    (rather than a local file) the resolved repo is additionally pinned to the
    canonical SemiAnalysis weka org/repo prefix, so a submission cannot swap in
    a foreign corpus. A foreign HF id that merely carries the "weka"
    marker is flagged submission_invalid.
    """
    if spec.require_loader is None:
        return
    allowed = (
        (spec.require_loader,)
        if isinstance(spec.require_loader, str)
        else tuple(spec.require_loader)
    )
    detected = _WEKA_GRAPH_LOADER if _is_graph_workload(run) else None
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
        return
    if not pin_weka_hf_repo(run, spec, violations):
        applied.append("require_loader")


def _apply_require_cache_bust(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Auto-fill the bare ``endpoint.cache_bust`` to the required value when at
    default ``NONE`` and not user-set; violation on an explicit different value.
    The auto-filled value drives the already-wired worker stamping.
    """
    if spec.require_cache_bust is None:
        return
    endpoint = run.cfg.endpoint
    actual = endpoint.cache_bust
    if actual == spec.require_cache_bust:
        applied.append("cache_bust")
        return
    explicit = "cache_bust" in endpoint.model_fields_set
    if explicit:
        violations.append(
            ScenarioViolation(
                flag="--cache-bust",
                current_value=str(actual),
                required_value=str(spec.require_cache_bust),
                message=(
                    f"scenario {spec.name!r} requires "
                    f"cache_bust={spec.require_cache_bust}; got {actual}"
                ),
            )
        )
    elif actual == CacheBustTarget.NONE:
        endpoint.cache_bust = spec.require_cache_bust
        _logger.info(
            "Scenario %r: auto-set --cache-bust=%s (was default).",
            spec.name,
            spec.require_cache_bust,
        )
        applied.append("cache_bust")


def _apply_duration(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Auto-fill / enforce the profiling-phase duration floor.

    Graph-IR completion is trace/stop-condition driven, so this is a
    submission-validity + run-length lock: auto-fill unset durations from the
    default; violation when an explicit duration is below the floor.
    """
    profiling_phases = run.cfg.get_profiling_phases()
    if spec.default_benchmark_duration_seconds is not None:
        for phase in profiling_phases:
            if phase.duration is None:
                phase.duration = float(spec.default_benchmark_duration_seconds)
                _logger.info(
                    "Scenario %r: auto-set duration=%ss (was unset).",
                    spec.name,
                    spec.default_benchmark_duration_seconds,
                )
        applied.append("default_benchmark_duration")

    below_floor = False
    for phase in profiling_phases:
        duration = phase.duration or 0.0
        if duration < spec.min_benchmark_duration_seconds:
            below_floor = True
            violations.append(
                ScenarioViolation(
                    flag="--benchmark-duration",
                    current_value=duration,
                    required_value=f">={spec.min_benchmark_duration_seconds}",
                    message=(
                        f"scenario {spec.name!r} requires duration >= "
                        f"{spec.min_benchmark_duration_seconds}s to reach "
                        "steady state and trigger KV offloading"
                    ),
                )
            )
    if not below_floor:
        applied.append("min_benchmark_duration")
