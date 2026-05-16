# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto-plot on-complete callback factory.

Exposes :func:`build_auto_plot_callback` so the CLI runner can request a
post-benchmark callback that runs ``aiperf plot`` against the just-written
artifact directory. The implementation delegates to ``plot.cli_runner`` so
the plot module stays the single source of truth for plot orchestration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiperf.config import BenchmarkRun


def build_auto_plot_callback(
    *,
    plot_required: bool,
    plot_envelope: Any | None = None,
) -> Callable[[BenchmarkRun], None]:
    """Build an on-complete callback that runs ``aiperf plot`` post-benchmark.

    Args:
        plot_required: When True, a plot failure surfaces as a non-zero exit
            from the CLI; when False, plot failures degrade to a logged warning
            so they don't mask the benchmark's own success.
        plot_envelope: Optional envelope-level plot config (only set when
            running through the top-level ``BenchmarkPlan.plot`` path); when
            None, the per-run ``BenchmarkConfig.plot`` is used.

    Returns:
        A callable accepting a finished ``BenchmarkRun`` that triggers the
        plot run against its artifact directory.
    """
    from aiperf.plot.cli_runner import run_plot_for_artifacts

    def _callback(run: BenchmarkRun) -> None:
        try:
            run_plot_for_artifacts(
                artifact_dir=run.artifact_dir,
                plot_envelope=plot_envelope,
            )
        except Exception:  # noqa: BLE001 - plot error boundary
            if plot_required:
                raise

    return _callback
