# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``is_graph_workload_path`` accepts adapter-detectable captures and rejects ordinary chat JSONL."""

from __future__ import annotations

from pathlib import Path

from aiperf.dataset.graph.workload_detect import is_graph_workload_path
from tests.unit.dataset.graph.conftest import DYNAMO_NESTED_FIXTURE

# The predicate shares `_detect_graph_workload_format` with config resolution,
# which deliberately excludes the `native` format -- hence a recorded dynamo
# capture is the positive case here, not a hand-authored native graph.


def test_is_graph_workload_path_true_for_dynamo_fixture() -> None:
    """A recorded dynamo trace capture is recognized as a graph workload."""
    assert is_graph_workload_path(DYNAMO_NESTED_FIXTURE) is True


def test_is_graph_workload_path_false_for_plain_jsonl(tmp_path: Path) -> None:
    """A plain chat-messages JSONL file is not a graph workload."""
    plain = tmp_path / "plain.jsonl"
    plain.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')
    assert is_graph_workload_path(plain) is False
