# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""pytest configuration for operator UI unit tests.

Every test under ``tests/ui`` is marked ``ui`` (see the ``ui`` marker in
``pyproject.toml``). Only the modules that shell out to Node.js (``node
--input-type=module``) to execute JavaScript require Node.js on PATH --
either via the shared ``node_utils.run_node`` helper or a module-local
``_run_node`` helper that wraps ``subprocess.run(["node", ...])`` directly.
The many pure-Python static-analysis modules in this directory
(regex/pathlib checks over the UI source and docs) never touch Node and must
keep running even when Node.js isn't installed.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from aiperf.config import BenchmarkRun
from aiperf.config.flags.cli_config import CLIConfig

_THIS_DIR = Path(__file__).resolve().parent


@pytest.fixture
def cli_config() -> CLIConfig:
    """Minimal CLIConfig fixture for UI tests that need a BenchmarkRun."""
    return CLIConfig(model_names=["test-model"])


@pytest.fixture
def benchmark_run(cli_config: CLIConfig) -> BenchmarkRun:
    """Build a v2 ``BenchmarkRun`` from :fixture:`cli_config`."""
    from aiperf.config.flags.resolver import resolve_config

    aiperf_config = resolve_config(cli_config, cli_config.config_file)
    return BenchmarkRun(
        benchmark_id=uuid.uuid4().hex,
        cfg=aiperf_config.benchmark,
        artifact_dir=aiperf_config.benchmark.artifacts.dir,
        random_seed=aiperf_config.random_seed,
        variables=dict(aiperf_config.variables),
    )


_NODE_AVAILABLE = shutil.which("node") is not None

# Names a module may bind its Node-shellout helper to: the shared
# ``node_utils.run_node`` import, or a module-local ``_run_node`` reimplementing
# the same "subprocess.run(['node', ...])" pattern. Any module defining or
# importing either name is treated as Node-dependent.
_NODE_HELPER_NAMES = ("run_node", "_run_node")


def _module_uses_node(item: pytest.Item) -> bool:
    module = getattr(item, "module", None)
    if module is None:
        return False
    module_vars = vars(module)
    return any(name in module_vars for name in _NODE_HELPER_NAMES)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    skip = pytest.mark.skip(
        reason="node not found on PATH; install Node.js to run UI tests"
    )
    ui_marker = pytest.mark.ui
    for item in items:
        if not Path(item.fspath).resolve().is_relative_to(_THIS_DIR):
            continue
        item.add_marker(ui_marker)
        if not _NODE_AVAILABLE and _module_uses_node(item):
            item.add_marker(skip)
