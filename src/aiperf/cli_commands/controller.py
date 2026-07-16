# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for the native cellular CONTROLLER role (Kubernetes)."""

from __future__ import annotations

from cyclopts import App

from aiperf.config.flags import CLIConfig

app = App(name="controller")


@app.default
def controller(
    *,
    cli_config: CLIConfig,
) -> None:
    """Run this pod as the native cellular controller.

    The Kubernetes counterpart of ``aiperf profile`` for the controller pod: resolve
    Config v2, launch ``aiperf`` in controller mode (it binds the cell
    transport, merges the cells' record shards, and runs the native export plane),
    push live progress to the owning AIPerfJob ``.status`` while the run is in flight,
    and signal completion when it finishes. Launched by the operator's JobSet, not by
    users directly.

    Args:
        cli_config: Cyclopts-populated CLIConfig; ``--config`` names the mounted
            Config v2 file.
    """
    from aiperf.cli_commands._cellular_role import run_cellular_role

    run_cellular_role(cli_config, role="controller")
