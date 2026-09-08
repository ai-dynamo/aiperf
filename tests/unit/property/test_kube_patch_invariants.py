# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mechanical "global invariant" test for Kubernetes patch content types.

``kubernetes_asyncio`` picks the first content type each ``patch_*`` method
advertises whenever the caller omits ``_content_type``, and for every one of
them that is ``application/json-patch+json``. A merge-patch dict sent that way
reaches the apiserver under an RFC 6902 header and is rejected:

    400 error decoding patch: json: cannot unmarshal object into Go value of
    type []handlers.jsonPatchOp

This is invisible to unit tests built on ``AsyncMock`` and only fails against a
real cluster, which is how it shipped twice -- once in ``aiperf kube cancel``
and once in the operator's namespace-lease renewal, where the 400 was swallowed
by a best-effort handler and silently dropped every namespace claim.

The contract: every ``patch_namespaced_*`` call in ``src/aiperf`` names its
content type explicitly. Pair ``JSON_PATCH_CONTENT_TYPE`` with a list of RFC
6902 operations and ``MERGE_PATCH_CONTENT_TYPE`` with a dict.
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


def test_patch_call_sites_exist() -> None:
    """Guard the guard: a scan that finds nothing would pass vacuously."""
    assert len(_patch_call_sites()) >= 20


def test_every_patch_names_its_content_type() -> None:
    unguarded = [
        f"{path}:{line}"
        for path, line, kwargs in _patch_call_sites()
        if "_content_type" not in kwargs
    ]
    assert not unguarded, (
        "These patch calls rely on the kubernetes_asyncio default content type "
        "(application/json-patch+json) and will 400 if they send a dict body. "
        "Pass _content_type=JSON_PATCH_CONTENT_TYPE or MERGE_PATCH_CONTENT_TYPE "
        "from aiperf.kubernetes.constants:\n  " + "\n  ".join(unguarded)
    )
