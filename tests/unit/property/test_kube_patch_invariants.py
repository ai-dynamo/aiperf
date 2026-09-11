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

# The body shape each content-type constant obliges the caller to send, keyed by
# the constant's name as it appears at the call site. Mirrors the runtime rule in
# tests.harness.k8s.decode_patch_like_apiserver.
BODY_SHAPE_BY_CONTENT_TYPE = {
    "JSON_PATCH_CONTENT_TYPE": ast.List,
    "MERGE_PATCH_CONTENT_TYPE": ast.Dict,
}


class _PatchCall:
    """One ``patch_*`` call site, reduced to what the invariants can check."""

    def __init__(self, path: str, node: ast.Call) -> None:
        self.location = f"{path}:{node.lineno}"
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        self.names_content_type = "_content_type" in keywords
        self.content_type = _constant_name(keywords.get("_content_type"))
        self.body = keywords.get("body")

    @property
    def mispaired(self) -> bool:
        """True when a statically known content type contradicts the body shape."""
        shape = BODY_SHAPE_BY_CONTENT_TYPE.get(self.content_type or "")
        if shape is None or not isinstance(self.body, ast.Dict | ast.List):
            return False
        return not isinstance(self.body, shape)


def _constant_name(node: ast.expr | None) -> str | None:
    """The bare name of a ``FOO`` or ``module.FOO`` reference, else ``None``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _patch_call_sites() -> list[_PatchCall]:
    """Every ``patch_*`` call in src, reduced to its checkable parts.

    The prefix is deliberately wider than ``patch_namespaced_``: cluster-scoped
    verbs such as ``patch_cluster_custom_object`` advertise the same JSON-Patch
    default and would otherwise be outside the gate by construction.
    """
    sites: list[_PatchCall] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else ""
            if not name.startswith("patch_"):
                continue
            sites.append(_PatchCall(str(path.relative_to(PROJECT_ROOT)), node))
    return sites


def test_every_patch_names_its_content_type() -> None:
    sites = _patch_call_sites()
    # Not an inventory: any non-zero count is fine. Zero means the scan itself
    # broke, which would let the assertion below pass without checking anything.
    assert sites, f"no patch_* calls found under {SRC_ROOT}"

    unguarded = [site.location for site in sites if not site.names_content_type]
    assert not unguarded, (
        "These patch calls rely on the kubernetes_asyncio default content type "
        "(application/json-patch+json) and will 400 if they send a dict body. "
        "Pass _content_type=JSON_PATCH_CONTENT_TYPE or MERGE_PATCH_CONTENT_TYPE "
        "from aiperf.kubernetes.constants:\n  " + "\n  ".join(unguarded)
    )


def test_every_patch_pairs_its_body_shape_with_its_content_type() -> None:
    mispaired = [site.location for site in _patch_call_sites() if site.mispaired]
    assert not mispaired, (
        "These patch calls name a content type that contradicts their literal "
        "body: JSON Patch takes a list of RFC 6902 operations and merge patch "
        "takes a dict. The apiserver answers 400 for either mismatch:\n  "
        + "\n  ".join(mispaired)
    )
