# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube CLI subcommand group with lazy-loaded subcommands."""

from __future__ import annotations

from cyclopts import App

app = App(name="kube", help="Kubernetes deployment and management commands")

app.command(
    "aiperf.cli_commands.kube.init:app",
    name="init",
    help="Generate a starter configuration template",
)
app.command(
    "aiperf.cli_commands.kube.validate:app",
    name="validate",
    help="Validate AIPerfJob YAML files against the CRD schema",
)
app.command(
    "aiperf.cli_commands.kube.profile:app",
    name="profile",
    help="Run a benchmark in Kubernetes",
)
app.command(
    "aiperf.cli_commands.kube.generate:app",
    name="generate",
    help="Generate Kubernetes YAML manifests",
)
app.command(
    "aiperf.cli_commands.kube.attach:app",
    name="attach",
    help="Attach to a running benchmark and stream progress",
)
app.command(
    "aiperf.cli_commands.kube.list_:app",
    name="list",
    help="List benchmark jobs and their status",
)
app.command(
    "aiperf.cli_commands.kube.logs:app",
    name="logs",
    help="Retrieve logs from benchmark pods",
)
app.command(
    "aiperf.cli_commands.kube.results:app",
    name="results",
    help="Retrieve benchmark results",
)
app.command(
    "aiperf.cli_commands.kube.debug:app",
    name="debug",
    help="Run diagnostic analysis on a deployment",
)
app.command(
    "aiperf.cli_commands.kube.watch:app",
    name="watch",
    help="Watch a running benchmark with live status and diagnostics",
)
app.command(
    "aiperf.cli_commands.kube.preflight:app",
    name="preflight",
    help="Run pre-flight checks against the target Kubernetes cluster",
)
app.command(
    "aiperf.cli_commands.kube.dashboard:app",
    name="dashboard",
    help="Open the operator results server UI in your browser",
)
