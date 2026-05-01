# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit case definitions — one per workflow shape exercised by the suite."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AuditCase:
    """One workflow audited by ``test_audit.py``.

    ``profile_args`` is the canonical CLI form. Both deployers translate it:
    the bare deployer passes the args verbatim to ``aiperf profile``; the
    operator runner translates them to an ``AIPerfJobConfig`` (see
    ``operator_runner.py``).
    """

    case_id: str
    """Stable id used for parametrize ids and artifact directory names."""

    endpoint_type: str
    """chat, completions, embeddings, etc."""

    concurrency: int
    """--concurrency value passed to both sides."""

    request_count: int
    """--request-count value passed to both sides."""

    num_conversations: int | None = None
    """--num-conversations override; None lets aiperf pick its default."""

    epochs: int = 1
    """Number of epochs (1 means a single run on each side)."""

    trials: int = 1
    """Number of trials per variation (multi_run.trials in AIPerfSweep). 1 means no multi-run."""

    sweep: dict[str, list[int | float | str]] | None = None
    """Optional sweep dimension; e.g. {'concurrency': [4, 16]}. None disables."""

    seed: int = 42
    """--random-seed value pinned for determinism."""

    metric_tolerance_overrides: dict[str, float] = field(default_factory=dict)
    """Per-metric tolerance overrides (relative diff, e.g. 0.30 = 30%)."""

    expected_artifacts: tuple[str, ...] = (
        "inputs.json",
        "profile_export_aiperf.json",
        "profile_export_aiperf.csv",
        "profile_export.jsonl",
    )
    """Filenames that MUST exist on both sides for the structural diff.

    Per-record JSONL is included: v2 ArtifactsConfig.records defaults to
    ['jsonl'] (matches v1's ExportLevel.RECORDS default), and the operator
    no longer races sub-second benchmarks (the trigger gate on JobProgress.
    results_exported plus the listing-endpoint marker gate close the race).
    """


AUDIT_CASES: tuple[AuditCase, ...] = (
    AuditCase(
        case_id="baseline-chat",
        endpoint_type="chat",
        concurrency=4,
        # Sub-second by design: the trigger + listing-endpoint gates close
        # the prior CompletedBeforeMonitor → ResultsFetchFailed race, so
        # 64-request 0.04s benchmarks must converge cleanly. Bumping back up
        # would mask the regression.
        request_count=64,
    ),
    AuditCase(
        case_id="baseline-completions",
        endpoint_type="completions",
        concurrency=4,
        request_count=64,
    ),
    AuditCase(
        case_id="concurrency-scale",
        endpoint_type="chat",
        concurrency=16,
        request_count=64,
        # Higher concurrency on operator side spreads across worker pods;
        # tail latency is structurally noisier than a single bare process.
        metric_tolerance_overrides={
            "p99": 0.40,
            "p95": 0.35,
        },
    ),
)
# Deferred cases (need operator-side helper extensions before they can audit):
#   - multi-epoch: AIPerfJobConfig.epochs + bare-side multi-run loop
#   - small-sweep: AIPerfSweep runner in tests/kubernetes/helpers/
# Add to AUDIT_CASES once the helpers land; no harness changes required.


SWEEP_AUDIT_CASES: tuple[AuditCase, ...] = (
    AuditCase(
        case_id="sweep-3x2",
        endpoint_type="chat",
        concurrency=4,  # base value; overridden per-variation by sweep
        request_count=32,
        num_conversations=16,
        sweep={"concurrency": [4, 8, 16]},
        trials=2,
        # Multi-pod orchestration + 6 cells = noisier tail latency than
        # 6 sequential single-process runs. Bands match concurrency-scale.
        metric_tolerance_overrides={
            "p99": 0.40,
            "p95": 0.35,
        },
    ),
)
