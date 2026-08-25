# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fire-and-forget tasks must be strongly referenced until they finish.

asyncio keeps only a weak reference to a running task. A bare
``asyncio.create_task(...)`` whose result nobody holds is collectable the
moment it yields -- and a collected task never runs its ``finally`` block.
Both sites here have a user-visible consequence when that happens: the API's
delayed stop silently never stops the pod, and the dashboard's refresh leaves
``dashboard_refresh_inflight`` stuck True so every later refresh answers
``already_rebuilding`` forever.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pytest import param

SITES = [
    param(
        "src/aiperf/api/routers/core.py",
        "_delayed_stop",
        id="api-shutdown",
    ),
    param(
        "src/aiperf/operator/dashboard_server.py",
        "_refresh_task",
        id="dashboard-refresh",
    ),
]


@pytest.mark.parametrize("path,coro_name", SITES)  # fmt: skip
def test_create_task_result_is_retained(path: str, coro_name: str) -> None:
    """The create_task call for this coroutine must not be a bare statement."""
    tree = ast.parse(Path(path).read_text())
    bare = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "create_task"):
            continue
        inner = call.args[0] if call.args else None
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == coro_name
        ):
            bare.append(node.lineno)
    assert not bare, (
        f"{path}:{bare} calls create_task({coro_name}()) as a bare statement; "
        "the task is GC-eligible. Keep a strong reference until it completes."
    )
