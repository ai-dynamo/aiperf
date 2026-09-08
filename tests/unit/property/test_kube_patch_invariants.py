# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mechanical "global invariant" test: every patch call names its content type.

``kubernetes_asyncio`` picks JSON Patch when ``_content_type`` is omitted, so a
merge-patch dict sent that way is rejected by the apiserver. An ``AsyncMock``
accepts the call, making the break visible only against a real cluster.
"""

from __future__ import annotations

import ast
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src" / "aiperf"


def _patch_call_sites() -> list[tuple[str, int, set[str]]]:
    """Every ``patch_namespaced_*`` call in src, with the kwargs it passes."""
    sites: list[tuple[str, int, set[str]]] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else ""
            if not name.startswith("patch_namespaced_"):
                continue
            rel = str(path.relative_to(PROJECT_ROOT))
            sites.append((rel, node.lineno, {kw.arg for kw in node.keywords if kw.arg}))
    return sites


def test_every_patch_names_its_content_type() -> None:
    sites = _patch_call_sites()
    # Not an inventory: any non-zero count is fine. Zero means the scan itself
    # broke, which would let the assertion below pass without checking anything.
    assert sites, f"no patch_namespaced_* calls found under {SRC_ROOT}"

    unguarded = [
        f"{path}:{line}"
        for path, line, kwargs in sites
        if "_content_type" not in kwargs
    ]
    assert not unguarded, (
        "These patch calls rely on the kubernetes_asyncio default content type "
        "(application/json-patch+json) and will 400 if they send a dict body. "
        "Pass _content_type=JSON_PATCH_CONTENT_TYPE or MERGE_PATCH_CONTENT_TYPE "
        "from aiperf.kubernetes.constants:\n  " + "\n  ".join(unguarded)
    )
