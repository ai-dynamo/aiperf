# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``aiperf analyze`` — tools that operate on completed AIPerf run artifacts."""

from cyclopts import App

app = App(
    name="analyze",
    help=(
        "Post-run analysis of completed AIPerf artifacts: visualize per-session "
        "timelines, concurrency curves, and other derived views over the "
        "``profile_export.jsonl`` produced by ``aiperf profile``."
    ),
)

app.command(
    "aiperf.cli_commands.swim_lane:app",
    name="swim-lane",
    help="Render a per-session swim-lane PNG with concurrency curve underneath.",
)

app.command(
    "aiperf.cli_commands.turn_messages:app",
    name="turn-messages",
    help="Render a collapsible HTML viewer of per-turn input messages (needs --export-level raw).",
)
