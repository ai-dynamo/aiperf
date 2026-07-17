# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Path-level graph-workload predicate tests.

:func:`is_graph_workload_path` is the path-level companion to
:func:`resolve_graph_workload` (which takes a run). It uses the SAME registry
detection (``_detect_graph_workload_format``, which excludes ``native``), so a
local trace file (dynamo ``.jsonl.gz``) is recognized while a plain conversation
``.jsonl`` is not.
"""

from __future__ import annotations

from pathlib import Path

from aiperf.dataset.graph.workload_detect import is_graph_workload_path

_DYNAMO_FIXTURE = (
    Path(__file__).resolve().parent
    / "adapters/fixtures/dynamo_nested/nested_2_level.jsonl.gz"
)


def test_is_graph_workload_path_true_for_dynamo_fixture() -> None:
    assert is_graph_workload_path(_DYNAMO_FIXTURE) is True


def test_is_graph_workload_path_false_for_plain_jsonl(tmp_path: Path) -> None:
    plain = tmp_path / "plain.jsonl"
    plain.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')
    assert is_graph_workload_path(plain) is False
