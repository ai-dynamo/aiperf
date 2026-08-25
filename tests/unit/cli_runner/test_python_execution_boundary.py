# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python profile execution must remain outside the native binary."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from aiperf.orchestrator.models import RunResult


def _run() -> MagicMock:
    run = MagicMock()
    run.label = "python-boundary"
    run.trial = 1
    run.artifact_dir = Path("/tmp/python-boundary")
    return run


def test_python_run_resolves_before_bootstrapping_the_service_mesh() -> None:
    from aiperf.cli_runner import _single_run

    run = _run()
    events: list[tuple[str, object]] = []
    resolver = Mock()
    resolver.resolve_all.side_effect = lambda resolved: events.append(("resolve", resolved))

    with (
        patch(
            "aiperf.config.resolution.resolvers.build_default_resolver_chain",
            return_value=resolver,
        ),
        patch(
            "aiperf.common.bootstrap.bootstrap_and_run_service",
            side_effect=lambda service, *, run: events.append(("bootstrap", run)),
        ),
    ):
        result = _single_run._execute_python_run(run)

    assert result == RunResult(
        label="python-boundary",
        success=True,
        artifacts_path=Path("/tmp/python-boundary"),
    )
    assert events == [("resolve", run), ("bootstrap", run)]


def test_single_python_run_does_not_launch_the_native_binary() -> None:
    from aiperf.cli_runner import _single_run

    run = _run()
    resolver = Mock()
    forbidden = Mock(side_effect=AssertionError("native process launch"))

    with (
        patch(
            "aiperf.config.resolution.resolvers.build_default_resolver_chain",
            return_value=resolver,
        ),
        patch("aiperf.common.bootstrap.bootstrap_and_run_service"),
        patch("aiperf.common.logging.setup_rich_logging"),
        patch("os._exit"),
        patch("subprocess.run", forbidden),
        patch("subprocess.Popen", forbidden),
        patch("os.execv", forbidden),
        patch("os.execve", forbidden),
        patch("os.execvp", forbidden),
        patch("os.execvpe", forbidden),
    ):
        _single_run._run_single_benchmark(run)

    forbidden.assert_not_called()
    resolver.resolve_all.assert_called_once_with(run)
