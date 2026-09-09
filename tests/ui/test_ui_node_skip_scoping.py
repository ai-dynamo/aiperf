# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifies the Node-availability skip in conftest.py only touches Node-dependent tests."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from tests.ui import conftest as ui_conftest
from tests.ui import node_utils
from tests.ui import test_operator_compare_filters as local_helper_node_module
from tests.ui import test_operator_docs_media_edges as non_node_module


def _fake_item(module: types.ModuleType, fspath: Path) -> pytest.Item:
    item = types.SimpleNamespace()
    item.module = module
    item.fspath = fspath
    item.own_markers = []
    item.add_marker = lambda marker: item.own_markers.append(marker)
    return item


def _is_skipped(item: pytest.Item) -> bool:
    return any(marker.name == "skip" for marker in item.own_markers)


def test_pytest_collection_modifyitems_node_unavailable_skips_only_node_dependent_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ui_conftest, "_NODE_AVAILABLE", False)

    node_using_item = _fake_item(node_utils, Path(node_utils.__file__))
    local_helper_item = _fake_item(
        local_helper_node_module, Path(local_helper_node_module.__file__)
    )
    non_node_item = _fake_item(non_node_module, Path(non_node_module.__file__))

    ui_conftest.pytest_collection_modifyitems(
        [node_using_item, local_helper_item, non_node_item]
    )

    assert _is_skipped(node_using_item), (
        "test module that imports run_node() must be skipped when node is unavailable"
    )
    assert _is_skipped(local_helper_item), (
        "test module that defines its own module-local _run_node() helper "
        "wrapping subprocess.run(['node', ...]) must also be skipped when "
        "node is unavailable"
    )
    assert not _is_skipped(non_node_item), (
        "test module with no run_node()/_run_node() usage must NOT be skipped "
        "when node is unavailable"
    )


def test_pytest_collection_modifyitems_node_available_skips_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ui_conftest, "_NODE_AVAILABLE", True)

    node_using_item = _fake_item(node_utils, Path(node_utils.__file__))
    non_node_item = _fake_item(non_node_module, Path(non_node_module.__file__))

    ui_conftest.pytest_collection_modifyitems([node_using_item, non_node_item])

    assert not _is_skipped(node_using_item)
    assert not _is_skipped(non_node_item)
