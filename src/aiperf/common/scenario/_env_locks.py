# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config apply-or-lock + submission scenario locks for graph-IR runtime knobs.

The trajectory-start (t*) window (``cfg.trajectory_start_min/max_ratio``) and
the per-trace idle-gap cap (``synthesis.idle_gap_cap_seconds``) are per-run
config: their locks AUTO-APPLY the scenario value when the user left the field
unset and raise a violation only when a user-explicit value differs (mirroring
AgentX's semantics). Scenario application performs NO process-global writes.

This module also holds the two scenario locks that do not fit
``validator.py``'s per-``_apply_*`` config-mutation shape: the concurrency-sweep
rejection (reads the run's ``SweepVariation``) and the canonical weka HF-repo
pin (sniffs the resolved input path). All are split out of ``validator.py`` to
keep that module under the file-size budget.
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
    """Apply-or-lock for the trajectory-start (t*) window on the run config.

    The window lives at ``cfg.trajectory_start_min_ratio`` /
    ``cfg.trajectory_start_max_ratio`` (``--trajectory-start-min-ratio`` /
    ``--trajectory-start-max-ratio``), per-run config threaded natively to the
    Dataset/Timing services. A user-explicit value is LOCKED (violation on
    mismatch); an unset field is auto-applied from the spec. A violated bound
    never blocks its unset sibling from being auto-applied (under
    ``--unsafe-override`` the run proceeds with the mixed window).
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
    added = False
    for field, flag, required in checks:
        if required is None:
            continue
        any_checked = True
        if field in cfg.model_fields_set:
            actual = getattr(cfg, field)
            if actual != required:
                added = True
                violations.append(
                    ScenarioViolation(
                        flag=flag,
                        current_value=actual,
                        required_value=required,
                        message=(
                            f"scenario {spec.name!r} locks the trajectory-start "
                            f"window; {flag} must equal {required}"
                        ),
                    )
                )
            continue
        setattr(cfg, field, required)
        _logger.info(f"Scenario {spec.name!r}: auto-set {flag}={required} (was unset).")
    if any_checked and not added:
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
    60s adapter default never leaks into a scenario run (agentx-on-main
    parity: apply when unset, lock when explicit).
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
        from aiperf.config.dataset import SynthesisConfig

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

# Canonical SemiAnalysis weka HF dataset org/repo prefix. The published weka
# corpora are all ``semianalysisai/cc-traces-weka-<date>[-256k]`` repos.
# Workload detection cannot distinguish a specific dated repo (it sniffs
# only "is this a weka graph workload?"), so the lock pins the
# org/repo PREFIX: an HF-id input must live under this prefix to be a recognized
# canonical weka workload. A non-canonical HF id (a foreign org, or a repo that
# merely carries the "weka" marker) is flagged submission_invalid.
_WEKA_HF_REPO_PREFIX = "semianalysisai/cc-traces-weka"
# The corpus repo the scenario contract requires; surfaced as the
# example/required value so the violation message points at the canonical corpus.
_WEKA_HF_CANONICAL_REPO = "semianalysisai/cc-traces-weka-062126"


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
    check, though: the sweep is expanded
    one level UP -- ``--concurrency 10,20,30`` is promoted into a grid sweep and
    each ``BenchmarkRun`` carries a single concrete concurrency plus a
    ``SweepVariation`` whose ``values`` record the swept dotted-path key
    (e.g. ``phases.profiling.concurrency``). Detecting that variation key here
    flags every run minted from a concurrency sweep
    as a violation (a ``ScenarioLockError``, downgradable under
    ``--unsafe-override``). A single non-swept run has ``variation=None`` (or a
    variation with no concurrency key) and is untouched.
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
    """Pin a HuggingFace-id weka input to the canonical org/repo prefix.

    Returns True (and appends a violation) when the input is an HF id NOT under
    :data:`_WEKA_HF_REPO_PREFIX`; returns False (no violation) for a local-file
    weka workload or a canonical HF id. Local imports avoid pulling the heavy
    dataset graph loader at ``aiperf.common.scenario`` import time.
    """
    from aiperf.dataset.graph.adapters.weka.trace import (
        _hf_dataset_id_str,
        _looks_like_hf_dataset_id,
    )
    from aiperf.dataset.graph.workload_detect import resolve_graph_workload

    ref = resolve_graph_workload(run)
    if ref is None:
        return False
    repo = _hf_dataset_id_str(ref.path)
    if not _looks_like_hf_dataset_id(repo):
        return False
    if repo.lower().startswith(_WEKA_HF_REPO_PREFIX):
        return False
    violations.append(
        ScenarioViolation(
            flag="--input-file (hf-repo)",
            current_value=repo,
            required_value=f"{_WEKA_HF_REPO_PREFIX}-* (e.g. {_WEKA_HF_CANONICAL_REPO})",
            message=(
                f"scenario {spec.name!r} only allows the canonical SemiAnalysis "
                f"weka HF corpora ({_WEKA_HF_REPO_PREFIX}-*); the resolved "
                f"dataset {repo!r} is not a recognized weka workload"
            ),
        )
    )
    return True
