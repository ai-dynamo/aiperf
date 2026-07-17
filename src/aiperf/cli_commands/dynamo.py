# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI subcommand family for Dynamo agent-trace tooling."""

import sys

from cyclopts import App

app = App(
    name="dynamo",
    help=(
        "Dynamo agent-trace tooling. Use `aiperf dynamo trace-report` to "
        "aggregate metrics from a captured trace."
    ),
)


@app.default
def dynamo() -> None:
    """Dynamo agent-trace tooling namespace."""
    app.help_print([])
    sys.exit(2)


app.command("aiperf.cli_commands.dynamo_trace_report:app", name="trace-report")
