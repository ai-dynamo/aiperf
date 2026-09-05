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
    assert by_id["root"].is_root is True
    assert by_id["child"].is_root is False
    # The metadata projection must preserve is_root so the sampler can filter roots.
    assert by_id["root"].metadata().is_root is True
    assert by_id["child"].metadata().is_root is False
