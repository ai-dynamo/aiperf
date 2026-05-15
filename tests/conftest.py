# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared test configuration and fixtures for all test types.

ONLY ADD FIXTURES HERE THAT ARE USED IN ALL TEST TYPES.
DO NOT ADD FIXTURES THAT ARE ONLY USED IN A SPECIFIC TEST TYPE.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from _pytest.mark.expression import IDENT_PREFIX, Scanner, expression

_DEFAULT_EXCLUDED_MARKERS = {
    "performance",
    "ffmpeg",
    "stress",
    "slow",
    "statistical",
    "component_integration",
    "integration",
    "server_unit",
    "fern",
}
_MARKER_PATH_MAP = {
    "integration": "tests/integration",
    "component_integration": "tests/component_integration",
}
_AUTO_MARKER_MAP = {
    "tests/integration": "integration",
    "tests/component_integration": "component_integration",
    "tests/unit/server": "server_unit",
    "tests/unit/fern": "fern",
}
_PATH_MARKER_MAP: list[tuple[str, list[str]]] = [
    ("tests/integration", ["integration"]),
    ("tests/component_integration", ["component_integration"]),
    ("tests/unit/server", ["server_unit"]),
    ("tests/unit/fern", ["fern"]),
]


def pytest_configure(config: pytest.Config) -> None:
    config._aiperf_enabled_markers = set()
    if _apply_marker_default_path(config):
        return
    if _apply_default_unit_path(config):
        return
    config._aiperf_enabled_markers = _enabled_markers_for_args(
        config.args, config.invocation_params.dir
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    _auto_mark_items(config, items)
    enabled_markers = getattr(
        config, "_aiperf_enabled_markers", set()
    ) | _positive_markers_from_expression(getattr(config.option, "markexpr", "") or "")
    deselected: list[pytest.Item] = []
    selected: list[pytest.Item] = []
    for item in items:
        marker_names = {marker.name for marker in item.iter_markers()}
        excluded_markers = marker_names & _DEFAULT_EXCLUDED_MARKERS
        if excluded_markers - enabled_markers:
            deselected.append(item)
        else:
            selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def _auto_mark_items(config: pytest.Config, items: list[pytest.Item]) -> None:
    rootpath = config.invocation_params.dir
    for item in items:
        path = _relative_item_path(item, rootpath)
        for path_prefix, marker in _AUTO_MARKER_MAP.items():
            if _path_matches_prefix(path, path_prefix):
                item.add_marker(getattr(pytest.mark, marker))


def _path_matches_prefix(path: str, path_prefix: str) -> bool:
    return path == path_prefix or path.startswith(f"{path_prefix}/")


def _relative_item_path(item: pytest.Item, rootpath: Path) -> str:
    try:
        return str(item.path.resolve().relative_to(rootpath))
    except (OSError, ValueError):
        return str(item.path)


def _apply_marker_default_path(config: pytest.Config) -> bool:
    if config.args and not _paths_match_tests_root(
        config.args, config.invocation_params.dir
    ):
        return False
    markexpr = getattr(config.option, "markexpr", "") or ""
    matched_markers = (
        _positive_markers_from_expression(markexpr) & _MARKER_PATH_MAP.keys()
    )
    if not matched_markers:
        return False
    config.args = [
        path for marker, path in _MARKER_PATH_MAP.items() if marker in matched_markers
    ]
    config._aiperf_enabled_markers = matched_markers
    return True


def _positive_markers_from_expression(markexpr: str) -> set[str]:
    positive_markers: set[str] = set()
    parsed = expression(Scanner(markexpr))

    def visit(node: ast.AST, positive: bool) -> None:
        if isinstance(node, ast.Expression):
            visit(node.body, positive)
        elif isinstance(node, ast.BoolOp):
            for value in node.values:
                visit(value, positive)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            visit(node.operand, not positive)
        elif isinstance(node, ast.Name):
            if positive and node.id.startswith(IDENT_PREFIX):
                positive_markers.add(node.id[len(IDENT_PREFIX) :])
        elif isinstance(node, ast.Call):
            visit(node.func, positive)

    visit(parsed, True)
    return positive_markers & _DEFAULT_EXCLUDED_MARKERS


def _apply_default_unit_path(config: pytest.Config) -> bool:
    args = [str(arg) for arg in config.args]
    if not args or _paths_match_tests_root(args, config.invocation_params.dir):
        config.args = ["tests/unit"]
        return True
    return False


def _paths_match_tests_root(args: list[str], rootpath: Path) -> bool:
    if len(args) != 1:
        return False
    path = args[0].rstrip("/")
    if path == "tests":
        return True
    try:
        return Path(path).resolve().relative_to(rootpath) == Path("tests")
    except (OSError, ValueError):
        return False


def _enabled_markers_for_args(args: list[str], rootpath: Path) -> set[str]:
    raw_args = [str(arg) for arg in args]
    if not raw_args:
        return set()

    rel_args: list[str] = []
    for arg in raw_args:
        path_part = arg.split("::")[0]
        try:
            rel_args.append(str(Path(path_part).resolve().relative_to(rootpath)))
        except (OSError, ValueError):
            rel_args.append(path_part)

    enabled: set[str] = set()
    for path_prefix, markers in _PATH_MARKER_MAP:
        if any(_path_matches_prefix(arg, path_prefix) for arg in rel_args):
            enabled.update(markers)
    return enabled
