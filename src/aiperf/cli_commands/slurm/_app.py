# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SLURM CLI subcommand group with lazy-loaded subcommands."""

from __future__ import annotations

from cyclopts import App

app = App(name="slurm", help="SLURM job-script generation for native cellular runs")

app.command(
    "aiperf.cli_commands.slurm.generate:app",
    name="generate",
    help="Generate an sbatch job script for a native cellular AIPerf benchmark",
)
