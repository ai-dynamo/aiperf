# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-run callback that invokes ``aiperf plot`` against an artifact dir.

CLI-time helper, not part of the service lifecycle, so it uses stdlib
:mod:`logging` rather than :class:`AIPerfLogger`. Imported lazily by
``run_benchmark`` only when ``--auto-plot`` resolves to True.

When the envelope ships a ``plot:`` section, the callback materializes the
resolved ``PlotEnvelopeConfig`` to ``<artifact_dir>/.aiperf-plot-config.yaml``
and passes that path to ``run_plot_controller`` via its ``config=`` arg. The
materialized file becomes a run artifact: re-running ``aiperf plot <run>``
later picks it up via the existing ``--config`` priority chain, making the
run's plots reproducible without the original envelope or the user's
``~/.aiperf/plot_config.yaml``.
"""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path

from aiperf.cli_runner import CompletedRun, OnComplete
from aiperf.config.plot import PlotEnvelopeConfig
from aiperf.plot.cli_runner import run_plot_controller

logger = logging.getLogger(__name__)

_MATERIALIZED_PLOT_CONFIG_NAME = ".aiperf-plot-config.yaml"


def _materialize_plot_envelope(envelope: PlotEnvelopeConfig, dest: Path) -> None:
    """Round-trip the envelope to ruamel YAML at ``dest``."""
    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    payload = envelope.model_dump(by_alias=False, exclude_none=True)
    buf = io.StringIO()
    yaml.dump(payload, buf)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(buf.getvalue(), encoding="utf-8")


def _run_auto_plot(
    *,
    artifact_dir: Path,
    plot_envelope: PlotEnvelopeConfig | None,
    input_paths: list[Path] | None = None,
    output_dir: Path | None = None,
) -> None:
    """Materialize an optional envelope and synchronously render its plots."""
    config_path: Path | None = None
    if plot_envelope is not None:
        config_path = artifact_dir / _MATERIALIZED_PLOT_CONFIG_NAME
        _materialize_plot_envelope(plot_envelope, config_path)

    paths = input_paths or [artifact_dir]
    kwargs = {
        "paths": [str(path) for path in paths],
        "config": str(config_path) if config_path is not None else None,
    }
    if output_dir is not None:
        kwargs["output"] = str(output_dir)
    run_plot_controller(**kwargs)


def _warn_auto_plot_failed(artifact_dir: Path, exc: Exception) -> None:
    logger.warning(
        "auto-plot failed (run artifacts intact at %s): %s; "
        "see %s for details. Re-run `aiperf plot %s` manually if needed.",
        artifact_dir,
        exc,
        artifact_dir / "plots" / "aiperf_plot.log",
        artifact_dir,
    )


def build_auto_plot_callback(
    *,
    plot_required: bool,
    plot_envelope: PlotEnvelopeConfig | None = None,
) -> OnComplete:
    """Return a post-run callback that invokes ``aiperf plot`` on the run dir.

    Args:
        plot_required: When True, plot failures re-raise so the caller exits
            non-zero. When False, plot failures are logged as warnings and
            the run is still considered successful.
        plot_envelope: Resolved envelope-level plot configuration. When set,
            it is materialized to ``<artifact_dir>/.aiperf-plot-config.yaml``
            and passed to ``run_plot_controller`` via ``config=``. When None,
            ``run_plot_controller`` falls back to its existing chain
            (CLI ``--config`` -> ``~/.aiperf/plot_config.yaml`` -> shipped default).
    """

    def _callback(run: CompletedRun) -> None:
        artifact_dir = Path(run.artifact_dir)
        try:
            _run_auto_plot(
                artifact_dir=artifact_dir,
                plot_envelope=plot_envelope,
            )
        except Exception as exc:
            if plot_required:
                raise
            _warn_auto_plot_failed(artifact_dir, exc)

    return _callback


async def run_auto_plot_async(
    *,
    artifact_dir: Path,
    plot_required: bool,
    plot_envelope: PlotEnvelopeConfig | None = None,
    input_paths: list[Path] | None = None,
    output_dir: Path | None = None,
) -> None:
    """Render plots off the event loop for Kubernetes completion paths.

    ``artifact_dir`` owns the materialized plot envelope and ``plots/`` output.
    ``input_paths`` may point elsewhere, which lets the sweep controller read
    its full aggregate tree while keeping generated artifacts under the durable
    parent-sweep epoch directory.
    """
    artifact_dir = Path(artifact_dir)
    try:
        await asyncio.to_thread(
            _run_auto_plot,
            artifact_dir=artifact_dir,
            plot_envelope=plot_envelope,
            input_paths=input_paths,
            output_dir=output_dir,
        )
    except Exception as exc:
        if plot_required:
            raise
        _warn_auto_plot_failed(artifact_dir, exc)
