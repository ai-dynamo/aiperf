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
    from pathlib import Path

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
    from aiperf.plot.cli_runner import run_plot_controller

    def _callback(run: BenchmarkRun) -> None:
        try:
            config_path: str | None = None
            if plot_envelope is not None:
                config_path = str(
                    _materialize_envelope(run.artifact_dir, plot_envelope)
                )
            run_plot_controller(paths=[str(run.artifact_dir)], config=config_path)
        except Exception:  # noqa: BLE001 - plot error boundary
            if plot_required:
                raise

    return _callback


def _materialize_envelope(artifact_dir: Path, plot_envelope: Any) -> Path:
    """Write the envelope plot config to ``<artifact_dir>/.aiperf-plot-config.yaml``.

    ``run_plot_controller`` consumes the envelope only as a YAML file path, so the
    resolved ``PlotEnvelopeConfig`` is dumped to disk (snake_case field names,
    mirroring ``default_plot_config.yaml``) and the path is threaded back through
    ``config=``. The materialized file also lets ``aiperf plot <dir>`` reproduce
    the run.

    ``by_alias=False`` is load-bearing: ``PlotConfig`` reads snake_case keys
    (``multi_run_defaults``, ``server_metrics_downsampling``,
    ``experiment_classification``), so dumping the camelCase aliases would make
    the reader silently find nothing and emit zero plots.
    """
    from ruamel.yaml import YAML

    from aiperf.plot.constants import MATERIALIZED_PLOT_CONFIG_NAME

    target = artifact_dir / MATERIALIZED_PLOT_CONFIG_NAME
    yaml = YAML()
    with target.open("w") as f:
        yaml.dump(
            plot_envelope.model_dump(by_alias=False, mode="json", exclude_none=True), f
        )
    return target
