# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard that the ``aiperf kube`` tree stays out of the non-Kubernetes hot path.

``aiperf.cli.app`` registers every subcommand by import string, so cyclopts
resolves a command module only when that command is actually invoked. A plain
``aiperf profile`` run must therefore never pay for ``kubernetes_asyncio`` (a
large, slow-importing dependency) or for the ``aiperf.cli_commands.kube`` tree
that pulls it in.

These run in a subprocess because ``sys.modules`` in the pytest process is
already polluted by sibling Kubernetes tests.
"""

from __future__ import annotations

import subprocess
import sys

_MODULES_THAT_MUST_STAY_LAZY = (
    "kubernetes_asyncio",
    "aiperf.cli_commands.kube",
    "aiperf.kubernetes.client",
    "aiperf.operator",
    "aiperf.sweep_controller",
)

# ``aiperf.kubernetes`` itself is deliberately absent from the list: its
# ``__init__`` is empty, and ``aiperf.config.deployment`` (on the plain-profile
# path, for CRD round-tripping) reaches the pure-enum leaf
# ``aiperf.kubernetes.enums``. That costs nothing. What must stay lazy is the
# API-client layer and everything above it.


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def test_importing_cli_does_not_import_kubernetes() -> None:
    """Importing the CLI entry point must not drag in the Kubernetes stack."""
    code = (
        "import sys, aiperf.cli\n"
        f"leaked = [m for m in {_MODULES_THAT_MUST_STAY_LAZY!r} if m in sys.modules]\n"
        "assert not leaked, f'eagerly imported: {leaked}'\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr


def test_profile_help_does_not_import_kubernetes() -> None:
    """Rendering ``aiperf profile --help`` must not drag in the Kubernetes stack."""
    code = (
        "import sys, aiperf.cli\n"
        "try:\n"
        "    aiperf.cli.app(['profile', '--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
        f"leaked = [m for m in {_MODULES_THAT_MUST_STAY_LAZY!r} if m in sys.modules]\n"
        "assert not leaked, f'eagerly imported: {leaked}'\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr


def test_kube_command_is_registered() -> None:
    """The lazy registration must still expose a working ``kube`` command."""
    code = (
        "import aiperf.cli\n"
        "assert 'kube' in aiperf.cli.app, 'kube subcommand not registered'\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
