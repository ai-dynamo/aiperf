# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugin-registration smoke tests for DagJsonlLoader."""

from pathlib import Path

import orjson
import pytest

from aiperf.common.enums import ConversationContextMode
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    CustomDatasetType,
    DatasetSamplingStrategy,
    PluginType,
)


def test_dag_jsonl_registered_as_custom_dataset_loader():
    assert plugins.has_entry(
        PluginType.CUSTOM_DATASET_LOADER, CustomDatasetType.DAG_JSONL
    )
    LoaderClass = plugins.get_class(
        PluginType.CUSTOM_DATASET_LOADER, CustomDatasetType.DAG_JSONL
    )
    assert LoaderClass is DagJsonlLoader


def test_dag_jsonl_custom_dataset_type_enum_value():
    assert CustomDatasetType.DAG_JSONL.value == "dag_jsonl"


def test_dag_jsonl_preferred_sampling_and_context_mode():
    assert (
        DagJsonlLoader.get_preferred_sampling_strategy()
        == DatasetSamplingStrategy.RANDOM
    )
    assert (
        DagJsonlLoader.get_default_context_mode()
        == ConversationContextMode.DELTAS_WITHOUT_RESPONSES
    )


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "x"}],
                        "forks": ["child"],
                    }
                ],
            },
            True,
        ),
        (
            {
                "session_id": "leaf",
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            },
            True,
        ),
        # Raw payload format (no session_id / turns wrapper) must not match.
        (
            {"messages": [{"role": "user", "content": "x"}]},
            False,
        ),
        # Multi-turn format (session_id + turns but no messages/forks/spawns).
        (
            {
                "session_id": "s",
                "turns": [{"text": "hi", "delay": 0}],
            },
            False,
        ),
        (None, False),
    ],
)
def test_dag_jsonl_can_load_detection(data, expected):
    assert DagJsonlLoader.can_load(data=data) is expected


def test_dag_jsonl_load_dataset_and_convert(tmp_path):
    lines = [
        {
            "session_id": "root",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "p"}],
                    "forks": ["child"],
                }
            ],
        },
        {
            "session_id": "child",
            "turns": [{"messages": [{"role": "user", "content": "c"}]}],
        },
    ]
    path: Path = tmp_path / "dag.jsonl"
    path.write_bytes(b"\n".join(orjson.dumps(line) for line in lines))

    loader = DagJsonlLoader(path)
    data = loader.load_dataset()
    assert set(data) == {"root", "child"}
    conversations = loader.convert_to_conversations(data)
    by_id = {c.session_id: c for c in conversations}
    # ``agent_depth == 0`` is the root predicate (replaces the old
    # ``is_root`` field). The metadata projection must preserve
    # ``agent_depth`` so the sampler can filter roots.
    assert by_id["root"].agent_depth == 0
    assert by_id["child"].agent_depth == 1
    assert by_id["root"].metadata().agent_depth == 0
    assert by_id["child"].metadata().agent_depth == 1


def test_loader_stamps_static_depth(tmp_path: Path) -> None:
    """Static-depth pin: the loader BFS-walks the DAG and stamps
    ``conv.agent_depth`` on every conversation. Roots get 0; each
    child gets ``parent_depth + 1``. Pre-fix ``agent_depth`` defaulted
    to 0 for every conversation including non-roots, leaving the
    runtime to recompute via ``parent_depth + 1`` on every spawn —
    which means a runtime drift would never be caught.
    """
    lines = [
        {
            "session_id": "root",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "r"}],
                    "forks": ["c1", "c2"],
                }
            ],
        },
        {
            "session_id": "c1",
            "turns": [{"messages": [{"role": "user", "content": "c1"}]}],
        },
        {
            "session_id": "c2",
            "turns": [{"messages": [{"role": "user", "content": "c2"}]}],
        },
    ]
    path: Path = tmp_path / "dag.jsonl"
    path.write_bytes(b"\n".join(orjson.dumps(line) for line in lines))

    loader = DagJsonlLoader(path)
    data = loader.load_dataset()
    by_id = {c.session_id: c for c in loader.convert_to_conversations(data)}

    # Root is depth 0; both children are depth 1.
    assert by_id["root"].agent_depth == 0
    assert by_id["c1"].agent_depth == 1
    assert by_id["c2"].agent_depth == 1
    # Metadata projection carries it through (consumed at runtime).
    assert by_id["root"].metadata().agent_depth == 0
    assert by_id["c1"].metadata().agent_depth == 1


def test_loader_depth_matches_topology_walk(tmp_path: Path) -> None:
    """Multi-tree pin: a file with two independent root trees gets
    depth=0 stamped on both roots and depth=1 on each child. This
    catches a regression where the BFS would only seed from one root
    or accidentally fall through to a single-root topology.
    """
    lines = [
        {
            "session_id": "r1",
            "turns": [
                {"messages": [{"role": "user", "content": "r1"}], "forks": ["c1"]}
            ],
        },
        {
            "session_id": "c1",
            "turns": [{"messages": [{"role": "user", "content": "c1"}]}],
        },
        {
            "session_id": "r2",
            "turns": [{"messages": [{"role": "user", "content": "r2"}]}],
        },
    ]
    path: Path = tmp_path / "dag.jsonl"
    path.write_bytes(b"\n".join(orjson.dumps(line) for line in lines))

    loader = DagJsonlLoader(path)
    data = loader.load_dataset()
    by_id = {c.session_id: c for c in loader.convert_to_conversations(data)}
    assert by_id["r1"].agent_depth == 0
    assert by_id["r2"].agent_depth == 0
    assert by_id["c1"].agent_depth == 1
