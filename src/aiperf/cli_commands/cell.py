# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for the native cellular CELL role (Kubernetes)."""

from __future__ import annotations

from cyclopts import App

from aiperf.config.flags import CLIConfig

app = App(name="cell")


@app.default
def cell(
    *,
    cli_config: CLIConfig,
) -> None:
    """Run this pod as one native cellular cell.

    The Kubernetes counterpart of ``aiperf profile`` for a cell pod: resolve Config
    v2 and launch ``aiperf-runner`` in cell mode, which runs this pod's
    ``(cell_id, cell_count)`` budget slice (from the ``CELL_*`` env the operator sets)
    and streams its records shard to the controller. A cell reports nothing to
    Kubernetes -- the controller owns the aggregate progress and completion. Launched
    by the operator's JobSet, not by users directly.

    Args:
        cli_config: Cyclopts-populated CLIConfig; ``--config`` names the mounted
            Config v2 file.
    """
    from aiperf.cli_commands._cellular_role import run_cellular_role

    run_cellular_role(cli_config, role="cell")
