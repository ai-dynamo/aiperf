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
    help="Validate AIPerfJob and AIPerfSweep YAML files against the CRD schema",
)
app.command(
    "aiperf.cli_commands.kube.profile:app",
    name="profile",
    help="Run a benchmark in Kubernetes",
)
app.command(
    "aiperf.cli_commands.kube.sweep:app",
    name="sweep",
    help="Run a parameter sweep or multi-run benchmark in Kubernetes",
)
app.command(
    "aiperf.cli_commands.kube.generate:app",
    name="generate",
    help="Generate Kubernetes YAML manifests",
)
app.command(
    "aiperf.cli_commands.kube.setup:app",
    name="setup",
    help="Install cluster prerequisites (JobSet CRD, namespaces, AIPerf operator)",
)
app.command(
    "aiperf.cli_commands.kube.delete:app",
    name="delete",
    help="Delete a benchmark and its backing Kubernetes resources",
)
app.command(
    "aiperf.cli_commands.kube.cleanup:app",
    name="cleanup",
    help="Bulk-remove finished benchmarks from a namespace",
)
app.command(
    "aiperf.cli_commands.kube.shutdown:app",
    name="shutdown",
    help="Retire a finished benchmark's controller pod",
)
app.command(
    "aiperf.cli_commands.kube.cancel:app",
    name="cancel",
    help="Cancel a running benchmark (patches spec.cancel on the CR)",
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
    "aiperf.cli_commands.kube.show:app",
    name="show",
    help="Render an AIPerfJob CR with Jinja2/env-vars resolved",
)
app.command(
    "aiperf.cli_commands.kube.debug:app",
    name="debug",
    help="Run diagnostic analysis on a deployment",
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
