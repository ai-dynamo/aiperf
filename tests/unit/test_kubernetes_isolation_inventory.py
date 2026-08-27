# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression locks for the native Kubernetes control-plane boundary."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]
PYTHON_OWNERSHIP_PATHS = (
    "src/aiperf/cli_commands/kube",
    "src/aiperf/kubernetes",
    "src/aiperf/operator",
    "src/aiperf/sweep_controller",
    "src/aiperf/config/kube.py",
    "src/aiperf/config/deployment.py",
    "src/aiperf/api/models/results.py",
    "tests/unit/cli_commands/kube",
    "tests/unit/kubernetes",
    "tests/unit/operator",
    "tests/unit/sweep_controller",
    "tests/component_integration/operator",
    "tests/integration/operator",
    "tests/harness/k8s.py",
)


def _holds_source(path: Path) -> bool:
    """Report whether a path still carries Python source.

    A bare ``exists()`` check reports a directory left behind by stale
    ``__pycache__`` as surviving ownership, which produced a false failure
    after the tree was reset. Ownership means source, not build artifacts.
    """
    if path.is_file():
        return True
    if not path.is_dir():
        return False
    return any(
        source.is_file()
        for source in path.rglob("*.py")
        if "__pycache__" not in source.parts
    )


def test_python_kubernetes_ownership_paths_are_absent() -> None:
    """The root Python package cannot reclaim native/operator ownership."""
    survivors = [path for path in PYTHON_OWNERSHIP_PATHS if _holds_source(ROOT / path)]
    assert not survivors, f"legacy Kubernetes ownership remains: {survivors}"


def test_root_package_has_no_operator_or_kubernetes_imports() -> None:
    """Native and standalone boundaries may communicate only by contracts."""
    offenders: list[str] = []
    for source in (ROOT / "src/aiperf").rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            names = (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else
                [name.name for name in node.names] if isinstance(node, ast.Import) else []
            )
            if any(
                name == "aiperf_k8s_operator"
                or name.startswith("aiperf_k8s_operator.")
                or name == "aiperf.kubernetes"
                or name.startswith("aiperf.kubernetes.")
                or name == "aiperf.operator"
                or name.startswith("aiperf.operator.")
                for name in names
            ):
                offenders.append(str(source.relative_to(ROOT)))
                break
    assert not offenders, f"root Python package crosses Kubernetes boundary: {offenders}"


def test_standalone_operator_never_imports_root_aiperf_package() -> None:
    """The independently installable operator has no ``aiperf.*`` dependency."""
    offenders: list[str] = []
    package = ROOT / "aiperf-k8s-operator/src/aiperf_k8s_operator"
    for source in package.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            names = (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else
                [name.name for name in node.names] if isinstance(node, ast.Import) else []
            )
            if any(name == "aiperf" or name.startswith("aiperf.") for name in names):
                offenders.append(str(source.relative_to(ROOT)))
                break
    assert not offenders, f"standalone operator imports root aiperf package: {offenders}"
