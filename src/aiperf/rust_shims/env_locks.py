# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config apply-or-lock scenario locks for recorded-graph runtime knobs (adapted).

Ported from ``ajc/aiperf-graph-ir:src/aiperf/common/scenario/_env_locks.py``.

The per-trace idle-gap cap (``synthesis.idle_gap_cap_seconds``) auto-applies the
scenario value when the user left the field unset and raises a violation only
when a user-explicit value differs. The trajectory-start (t*) window
(``cfg.trajectory_start_min/max_ratio``) is a default, not a lock: it
auto-applies the scenario value when unset and HONORS an explicit user value
(parity with the official agentx validator). Scenario application performs NO
process-global writes.

This module also holds the concurrency-sweep rejection (reads the run's
``SweepVariation``) and the weka HF-repo pin.

Adaptation for ajc/rust: ``pin_weka_hf_repo`` is a documented no-op. The
graph-IR branch sniffed a resolved HF dataset id off ``resolve_graph_workload``;
ajc/rust weka inputs are local file/dir paths (``FileDataset.path``) that carry
no HF org/repo id at resolution time, and there is no ``resolve_graph_workload``
helper or HF-hosted weka public dataset to pin. The pin therefore never fires.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiperf.common.scenario.base import ScenarioSpec, ScenarioViolation

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

_logger = logging.getLogger(__name__)


def apply_trajectory_ratios(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Auto-fill-or-honor for the trajectory-start (t*) window on the run config.

    The window lives at ``cfg.trajectory_start_min_ratio`` /
    ``cfg.trajectory_start_max_ratio`` (``--trajectory-start-min-ratio`` /
    ``--trajectory-start-max-ratio``), per-run config threaded natively to the
    runner. Parity with the official agentx validator
    (``ajc/agentx:validator.py`` ~403-420): an unset field is auto-applied from
    the spec default; a user-explicit value is HONORED (no violation), so the
    scenario supplies a default t* window rather than locking it. ``violations``
    is retained in the signature for a uniform lock-callback shape but is never
    appended to here.
    """
    checks = (
        (
            "trajectory_start_min_ratio",
            "--trajectory-start-min-ratio",
            spec.default_trajectory_start_min_ratio,
        ),
        (
            "trajectory_start_max_ratio",
            "--trajectory-start-max-ratio",
            spec.default_trajectory_start_max_ratio,
        ),
    )
    cfg = run.cfg
    any_checked = False
    for field, flag, required in checks:
        if required is None:
            continue
        any_checked = True
        if field in cfg.model_fields_set:
            # User-explicit value is honored (parity with the agentx validator,
            # which only auto-fills when the field was left unset).
            continue
        current = getattr(cfg, field, None)
        if current != required:
            setattr(cfg, field, required)
            _logger.info(
                f"Scenario {spec.name!r}: auto-set {flag}={required} (was unset)."
            )
    if any_checked:
        applied.append("trajectory_start_ratios")


def apply_trace_idle_gap_cap(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Apply-or-lock for the per-trace idle-gap cap.

    The cap lives at ``synthesis.idle_gap_cap_seconds``
    (``--synthesis-idle-gap-cap``). A user-explicit value — including an
    explicit null (warp disabled) — is LOCKED: a violation is raised when it
    differs from ``spec.trace_idle_gap_cap_seconds``. When the field is unset
    the scenario auto-applies the spec value onto the run config, so the bare
    60s adapter default never leaks into a scenario run (apply when unset, lock
    when explicit).
    """
    if spec.trace_idle_gap_cap_seconds is None:
        return
    dataset = run.cfg.get_default_dataset()
    if dataset is None or "synthesis" not in type(dataset).model_fields:
        # Synthetic datasets have no trace idle-gap concept; nothing to lock.
        return
    synthesis = getattr(dataset, "synthesis", None)
    explicit = (
        synthesis is not None and "idle_gap_cap_seconds" in synthesis.model_fields_set
    )
    if explicit:
        actual = synthesis.idle_gap_cap_seconds
        if actual != spec.trace_idle_gap_cap_seconds:
            violations.append(
                ScenarioViolation(
                    flag="--synthesis-idle-gap-cap",
                    current_value=actual,
                    required_value=spec.trace_idle_gap_cap_seconds,
                    message=(
                        f"scenario {spec.name!r} locks the per-trace idle-gap cap "
                        f"to {spec.trace_idle_gap_cap_seconds}; "
                        "--synthesis-idle-gap-cap must match"
                    ),
                )
            )
        else:
            applied.append("trace_idle_gap_cap")
        return
    if synthesis is None:
        from aiperf.config.dataset.trace import SynthesisConfig

        dataset.synthesis = SynthesisConfig(
            idle_gap_cap_seconds=spec.trace_idle_gap_cap_seconds
        )
    else:
        synthesis.idle_gap_cap_seconds = spec.trace_idle_gap_cap_seconds
    _logger.info(
        f"Scenario {spec.name!r}: auto-set "
        f"--synthesis-idle-gap-cap={spec.trace_idle_gap_cap_seconds} (was unset)."
    )
    applied.append("trace_idle_gap_cap")


# Final dotted-path segments whose presence in a run's SweepVariation marks a
# concurrency sweep (a scenario locks ONE config; a sweep would multiply it).
_CONCURRENCY_SWEEP_SUFFIXES: tuple[str, ...] = (
    "concurrency",
    "prefill_concurrency",
    "warmup_concurrency",
    "warmup_prefill_concurrency",
)


def apply_concurrency_sweep(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    applied: list[str],
) -> None:
    """Reject a swept ``--concurrency`` (or related) under a fixed-spec scenario.

    A scenario locks ONE fixed configuration; a swept concurrency
    (``--concurrency 10,20,30``) would multiply it into N runs with diverging
    settings, so it must be rejected outright. A raw list never reaches this
    check: the sweep is expanded one level UP into a grid sweep, and each
    ``BenchmarkRun`` carries a single concrete concurrency plus a
    ``SweepVariation`` whose ``values`` record the swept dotted-path key.
    Detecting that variation key here flags every run minted from a concurrency
    sweep as a violation (downgradable under ``--unsafe-override``). A single
    non-swept run has ``variation=None`` (or a variation with no concurrency
    key) and is untouched.
    """
    variation = getattr(run, "variation", None)
    values = getattr(variation, "values", None)
    if not isinstance(values, dict):
        return
    swept = [
        key for key in values if key.rsplit(".", 1)[-1] in _CONCURRENCY_SWEEP_SUFFIXES
    ]
    if not swept:
        return
    violations.append(
        ScenarioViolation(
            flag="--concurrency",
            current_value=sorted(swept),
            required_value="single value (no sweep)",
            message=(
                f"scenario {spec.name!r} does not support parameter sweeps; "
                "pass a single --concurrency value instead of a list "
                "(a sweep multiplies the locked config into N diverging runs)"
            ),
        )
    )


def pin_weka_hf_repo(
    run: BenchmarkRun,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
) -> bool:
    """Documented no-op on ajc/rust.

    The graph-IR branch pinned a HuggingFace-id weka input to the canonical
    SemiAnalysis org/repo prefix by sniffing ``resolve_graph_workload(run)``.
    ajc/rust has no ``resolve_graph_workload`` helper and no HF-hosted weka
    public dataset: weka inputs are local ``FileDataset.path`` file/dir paths
    that carry no HF repo id at resolution time. There is nothing to pin here,
    so this always returns False (no violation) and the caller treats the loader
    lock as satisfied.
    """
    return False
