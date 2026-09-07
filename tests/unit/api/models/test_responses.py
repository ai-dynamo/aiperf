# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests guarding against duplicate class-body field definitions in
``aiperf.api.models.responses``.

Pydantic silently keeps the last of two identical class-level assignments to
the same name, so a duplicate ``Field()`` definition is dead code that never
raises or warns at runtime. The only reliable way to catch it is to inspect
the source AST for the class body rather than instantiate the model.
"""

import ast
import inspect

import aiperf.api.models.responses as responses_module


def _field_assignment_names(class_node: ast.ClassDef) -> list[str]:
    """Return the field name for every top-level annotated assignment in a
    class body, in source order, including duplicates."""
    names = []
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return names


def _get_class_node(tree: ast.Module, class_name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name!r} not found in module source")


def test_progress_response_workers_field_defined_once_no_duplicates() -> None:
    source = inspect.getsource(responses_module)
    tree = ast.parse(source)
    class_node = _get_class_node(tree, "ProgressResponse")

    names = _field_assignment_names(class_node)
    duplicates = {name for name in names if names.count(name) > 1}

    assert not duplicates, (
        f"ProgressResponse defines field(s) {duplicates} more than once in "
        "its class body; a later duplicate assignment silently shadows the "
        "earlier one, making the first definition dead code that Pydantic "
        "never processes."
    )
