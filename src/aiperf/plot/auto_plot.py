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
        config_path: Path | None = None
        if plot_envelope is not None:
            config_path = Path(run.artifact_dir) / _MATERIALIZED_PLOT_CONFIG_NAME
            _materialize_plot_envelope(plot_envelope, config_path)

        try:
            run_plot_controller(
                paths=[str(run.artifact_dir)],
                config=str(config_path) if config_path is not None else None,
            )
        except Exception:
            if plot_required:
                raise
            logger.warning(
                "auto-plot failed (run artifacts intact at %s); "
                "see %s for details. Re-run "
                "`aiperf plot %s` manually if needed.",
                run.artifact_dir,
                Path(run.artifact_dir) / "plots" / "aiperf_plot.log",
                run.artifact_dir,
            )

    return _callback
