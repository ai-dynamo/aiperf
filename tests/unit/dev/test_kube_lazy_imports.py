# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard that ``dev/kube.py``'s function-local aiperf imports actually resolve.

The CLI defers most ``aiperf.*`` imports into command bodies for startup speed,
so a module that is renamed or moved stays green until someone runs the command
that touches it. This walks every ``from aiperf... import ...`` in the file and
resolves each imported name eagerly.
"""

import ast
import importlib
from pathlib import Path

import pytest
from pytest import param

_REPO_ROOT = Path(__file__).resolve().parents[3]
_KUBE_DEV_CLI = _REPO_ROOT / "dev" / "kube.py"


def _aiperf_import_targets() -> list[tuple[str, str, int]]:
    tree = ast.parse(_KUBE_DEV_CLI.read_text(), filename=str(_KUBE_DEV_CLI))
    return [
        (node.module, alias.name, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module is not None
        and node.module.split(".")[0] == "aiperf"
        for alias in node.names
    ]


@pytest.mark.parametrize(
    ("module_name", "symbol", "lineno"),
    [
        param(module_name, symbol, lineno, id=f"{module_name}.{symbol}")
        for module_name, symbol, lineno in _aiperf_import_targets()
    ],
)  # fmt: skip
def test_dev_kube_aiperf_import_resolves(
    module_name: str, symbol: str, lineno: int
) -> None:
    module = importlib.import_module(module_name)
    assert hasattr(module, symbol), (
        f"dev/kube.py:{lineno} imports {symbol!r} from {module_name!r}, "
        "which does not define it"
    )
