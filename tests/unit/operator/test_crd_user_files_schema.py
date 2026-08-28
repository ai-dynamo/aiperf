# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""artifacts.userFiles[].content must stay expressible for every legal format.

``UserFile.content`` is ``Any``: a bare string for ``format: text``, a
dict/list/scalar otherwise. The generator used to emit ``type: object`` for the
untyped annotation, so the apiserver rejected every ``format: text`` entry.
"""

from __future__ import annotations

from typing import Any

import jsonschema
import pytest
from pytest import param

from tools.generate_crd import CRDDocumentBuilder, CRDSchemaSource


def _user_files_node(kind: str) -> dict[str, Any]:
    """Return the ``artifacts.userFiles`` schema node from a generated CRD."""
    source = CRDSchemaSource()
    builder = CRDDocumentBuilder()
    if kind == "AIPerfJob":
        crd = builder.aiperfjob_crd(source.job_schema())
        spec = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"][
            "spec"
        ]
    else:
        crd = builder.aiperfsweep_crd(source.sweep_schema())
        spec = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"][
            "spec"
        ]
    return spec["properties"]["benchmark"]["properties"]["artifacts"]["properties"][
        "userFiles"
    ]


def _structural_schema(node: Any) -> Any:
    """Strip ``x-kubernetes-*`` keys, leaving the schema the apiserver enforces.

    ``x-kubernetes-preserve-unknown-fields`` only tells the apiserver to keep
    unknown *properties*; it never relaxes a declared ``type``. Dropping the
    extensions therefore models structural validation of this node faithfully
    enough for jsonschema to evaluate.
    """
    if isinstance(node, dict):
        return {
            key: _structural_schema(value)
            for key, value in node.items()
            if not key.startswith("x-kubernetes-")
        }
    if isinstance(node, list):
        return [_structural_schema(value) for value in node]
    return node


@pytest.mark.parametrize("kind", ["AIPerfJob", "AIPerfSweep"])
def test_user_files_content_node_is_typeless_on_both_kinds(kind: str) -> None:
    content = _user_files_node(kind)["items"]["properties"]["content"]

    assert content["x-kubernetes-preserve-unknown-fields"] is True
    assert "type" not in content


@pytest.mark.parametrize(
    "entry",
    [
        param(
            {"path": "notes.md", "format": "text", "content": "Run {{ job_name }}.\n"},
            id="format_text_string_content",
        ),
        param(
            {"path": "cfg.json", "format": "json", "content": {"isl": 1024}},
            id="format_json_dict_content",
        ),
        param(
            {"path": "cfg.json", "format": "json", "content": [1, 2]},
            id="format_json_list_content",
        ),
        param(
            {"path": "cfg.yaml", "format": "yaml", "content": 42},
            id="format_yaml_scalar_content",
        ),
    ],
)  # fmt: skip
def test_user_files_entry_validates_against_generated_job_crd(
    entry: dict[str, Any],
) -> None:
    schema = _structural_schema(_user_files_node("AIPerfJob"))

    jsonschema.validate([entry], schema)


def test_user_files_entry_missing_path_rejected_by_generated_job_crd() -> None:
    """The relaxed ``content`` type must not relax the required-key contract."""
    schema = _structural_schema(_user_files_node("AIPerfJob"))

    with pytest.raises(jsonschema.ValidationError, match="path"):
        jsonschema.validate([{"content": "x", "format": "text"}], schema)
