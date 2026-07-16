# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenario invariant lock applied as a config-resolver step (adapted).

Ported from ``ajc/aiperf-graph-ir:src/aiperf/common/scenario/validator.py``.

``apply_scenario(run)`` reads ``run.cfg.scenario``; when set, it looks up the
:class:`ScenarioSpec`, auto-fills unset defaults (info log) and validates each
invariant against ``run.cfg`` / ``run.resolved``. Explicit user conflicts raise
:class:`ScenarioLockError` unless ``run.cfg.unsafe_override`` downgrades them to
warnings and stamps ``submission_valid=False``.

Adaptations for ajc/rust:

* The graph-IR branch derived ``TimingMode.GRAPH_IR`` from a memoized
  ``resolve_graph_workload(run)`` sniff. ajc/rust has no such helper and no
  ``TimingMode.GRAPH_IR``: a graph workload is selected by dataset FORMAT
  (``weka_trace``). ``_is_weka_workload`` inspects the already-resolved
  ``run.resolved.dataset_types`` (populated by ``DatasetResolver`` earlier in
  the chain) for a ``weka_trace`` type. ``_apply_timing_mode`` verifies the run
  IS a weka graph workload and raises otherwise.
* ajc/rust has no ``endpoint.cache_bust`` knob; instead
  ``_apply_require_cache_bust`` auto-fills the per-run ``cfg.cache_bust_target``
  from the scenario lock, which ``rust_wire`` projects onto the recorded
  dataset's synthesis block for the native runner's first-turn marker.

User-explicit vs default is read from ``model_fields_set`` membership on the
LIVE converted config (the resolver chain mutates ``run`` in place), so
membership is faithful here -- no explicit-set sentinel is required.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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

_logger = logging.getLogger(__name__)

# Synthetic loader identity returned for any weka graph workload.
_WEKA_GRAPH_LOADER = "weka_trace"


def _is_weka_workload(run: BenchmarkRun) -> bool:
    """True when the run targets a weka graph workload.

    On ajc/rust a graph workload is selected by dataset FORMAT, not a timing
    mode. Two detection signals, in preference order:

    1. ``run.resolved.dataset_types`` — populated by ``DatasetResolver`` with
       the detected ``CustomDatasetType`` per file dataset; a weka input
       resolves to ``CustomDatasetType.WEKA_TRACE``. This is the signal when
       the resolver chain ran.
    2. ``run.cfg.datasets[*].format`` — the direct-pair execute path
       (``rust_executor``) never runs ``DatasetResolver``, so ``dataset_types``
       is empty there. Fall back to the explicitly-set ``weka_trace`` format on
       any configured ``FileDataset`` (what ``--custom-dataset-type weka-trace``
       lands on ``run.cfg``). ``format`` defaults to ``single_turn``, so an
       explicit-set check (``model_fields_set``) is required to avoid matching
       an unset default.

    Both match by string value so this stays independent of the dynamic-enum
    import.
    """
    dataset_types = getattr(run.resolved, "dataset_types", None)
    if dataset_types and any(
        str(dt) == _WEKA_GRAPH_LOADER for dt in dataset_types.values()
    ):
        return True
    for ds in getattr(run.cfg, "datasets", ()):
        fmt = getattr(ds, "format", None)
        if (
            fmt is not None
            and str(fmt) == _WEKA_GRAPH_LOADER
            and "format" in getattr(ds, "model_fields_set", set())
        ):
            return True
    return False


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
    # fallback when unset). The run seed lives on BenchmarkRun.random_seed,
    # threaded deterministically by the orchestrator BEFORE this resolver step
    # runs; a non-deterministic fallback here would break the content-
    # determinism contract.

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


def _apply_timing_mode(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Verify the run is a weka graph workload (graph-IR is derived, not stamped).

    ajc/rust selects a graph workload by dataset format; this verifies the
    workload IS a weka graph and raises otherwise rather than stamping phases.
    """
    if _is_weka_workload(run):
        applied.append("timing_mode")
        return
    violations.append(
        ScenarioViolation(
            flag="--input-file (timing_mode)",
            current_value="non-graph workload",
            required_value=spec.timing_mode,
            message=(
                f"scenario {spec.name!r} requires a weka graph workload "
                f"(timing_mode={spec.timing_mode}); the input is not one "
                "(set a dataset with format: weka_trace)"
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
    """Reject ``--synthesis-max-isl`` (the ``synthesis.max_isl`` ISL filter).

    A weka graph dataset without a ``synthesis`` block has no such field, so
    this degrades to a graceful no-op.
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

    Detection is the weka-workload sniff over ``run.resolved.dataset_types``: a
    weka graph workload reports ``"weka_trace"``; else None. ``pin_weka_hf_repo``
    is a documented no-op on ajc/rust (weka inputs are local file/dir paths that
    carry no HF repo id at resolution time).
    """
    if spec.require_loader is None:
        return
    allowed = (
        (spec.require_loader,)
        if isinstance(spec.require_loader, str)
        else tuple(spec.require_loader)
    )
    detected = _WEKA_GRAPH_LOADER if _is_weka_workload(run) else None
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
    """Auto-fill the recorded-graph cache-bust marker target from the scenario.

    The native runner materializes a first-turn cache-bust marker
    (``graph_execution::GraphCacheBust``) gated on ``cfg.cache_bust_target``.
    Auto-fill that per-run knob from the scenario's ``require_cache_bust`` lock
    (e.g. ``"first_turn_prefix"``) so the scenario-required marker + its ISL
    accounting engage. Unset ``require_cache_bust`` leaves the config default
    (``"none"``) untouched. An explicit user-set target that disagrees with the
    scenario is a violation (the marker changes wire bytes and ISL, so it is not
    a soft mismatch).
    """
    required = spec.require_cache_bust
    if required is None:
        return
    current = getattr(run.cfg, "cache_bust_target", "none")
    if current in (None, "none", ""):
        run.cfg.cache_bust_target = required
        applied.append("require_cache_bust")
        return
    if current != required:
        violations.append(
            ScenarioViolation(
                flag="--cache-bust-target",
                current_value=current,
                required_value=required,
                message=(
                    f"scenario {spec.name!r} requires cache_bust_target={required!r}"
                ),
            )
        )
        return
    applied.append("require_cache_bust")


def _apply_duration(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Auto-fill / enforce the profiling-phase duration floor.

    Graph completion is trace/stop-condition driven, so this is a
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
