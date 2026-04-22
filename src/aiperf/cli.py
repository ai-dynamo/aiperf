# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for the AIPerf system."""

from cyclopts import App


def _get_help_text() -> str:
    """Generate help text with installed plugin information."""
    # Get aiperf version for the title
    from aiperf import __version__ as aiperf_version
    from aiperf.plugin import plugins

    packages = plugins.list_packages()
    plugin_list = []
    for pkg in packages:
        meta = plugins.get_package_metadata(pkg)
        plugin_list.append(f"{pkg} (v{meta.version})")

    plugins_str = ", ".join(plugin_list) if plugin_list else "none"
    return f"NVIDIA AIPerf v{aiperf_version} - AI Performance Benchmarking Tool\n\nInstalled Plugin Packages: {plugins_str}"


app = App(name="aiperf", help=_get_help_text())

# Register --install-completion flag to install completion for the current shell
app.register_install_completion_command()


# Register all CLI commands (lazily loaded at invocation time)
# NOTE: The order here determines the order they will appear in docs/cli_options.md
app.command("aiperf.cli_commands.analyze_trace:app", name="analyze-trace")
app.command("aiperf.cli_commands.profile:app", name="profile")
app.command("aiperf.cli_commands.plot:app", name="plot")
app.command("aiperf.cli_commands.plugins:app", name="plugins")
app.command("aiperf.cli_commands.service:app", name="service")
app.command("aiperf.cli_commands.speed_bench_report:app", name="speed-bench-report")
app.command("aiperf.cli_commands.synthesize:app", name="synthesize")
app.command("aiperf.cli_commands.validate:app", name="validate")
