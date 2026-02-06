# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from pytest import param

from aiperf.dataset.loader.aa_rwlt import AARWLTDatasetLoader
from aiperf.dataset.loader.models import AARWLTEntry
from aiperf.plugin.enums import DatasetSamplingStrategy


def _make_entry(
    conv_id: str = "traj-001",
    idx: int = 0,
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
) -> dict:
    """Helper to build a valid AA-RWLT entry dict."""
    return {
        "conversation_id": conv_id,
        "conversation_idx": idx,
        "messages": messages
        or [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        "tools": tools,
    }


class TestAARWLTCanLoad:
    """Tests for AARWLTDatasetLoader.can_load()."""

    @pytest.mark.parametrize(
        "data,expected",
        [
            param(_make_entry(), True, id="valid_entry"),
            param(_make_entry(tools=[{"type": "function", "function": {"name": "f"}}]), True, id="with_tools"),
            param({"type": "aa_rwlt", "conversation_id": "x", "conversation_idx": 0, "messages": [{"role": "user", "content": "hi"}]}, True, id="explicit_type"),
            param({"text": "Hello world"}, False, id="single_turn_data"),
            param({"turns": [{"text": "Turn 1"}]}, False, id="multi_turn_data"),
            param({"input_length": 100}, False, id="mooncake_data"),
            param({"conversation_id": "x", "conversation_idx": 0}, False, id="missing_messages"),
            param({"conversation_id": "x", "messages": [{"role": "user", "content": "hi"}]}, False, id="missing_idx"),
            param(None, False, id="none_data"),
        ],
    )  # fmt: skip
    def test_can_load(self, data, expected):
        assert AARWLTDatasetLoader.can_load(data) is expected

    def test_can_load_rejects_negative_idx(self):
        data = _make_entry()
        data["conversation_idx"] = -1
        assert AARWLTDatasetLoader.can_load(data) is False

    def test_can_load_rejects_empty_messages(self):
        data = _make_entry()
        data["messages"] = []
        assert AARWLTDatasetLoader.can_load(data) is False


class TestAARWLTDatasetLoader:
    """Tests for AARWLTDatasetLoader load_dataset and convert_to_conversations."""

    def test_get_preferred_sampling_strategy(self):
        assert (
            AARWLTDatasetLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    def test_load_dataset_groups_by_conversation_id(
        self, create_jsonl_file, default_user_config
    ):
        content = [
            json.dumps(_make_entry("traj-A", 0)),
            json.dumps(_make_entry("traj-A", 1)),
            json.dumps(_make_entry("traj-B", 0)),
        ]
        filename = create_jsonl_file(content)
        loader = AARWLTDatasetLoader(filename=filename, user_config=default_user_config)
        dataset = loader.load_dataset()

        assert len(dataset) == 2
        assert "traj-A" in dataset
        assert "traj-B" in dataset
        assert len(dataset["traj-A"]) == 2
        assert len(dataset["traj-B"]) == 1

    def test_load_dataset_sorts_by_conversation_idx(
        self, create_jsonl_file, default_user_config
    ):
        """Entries arrive out of order but get sorted."""
        content = [
            json.dumps(
                _make_entry(
                    "traj-A", 2, messages=[{"role": "user", "content": "step2"}]
                )
            ),
            json.dumps(
                _make_entry(
                    "traj-A", 0, messages=[{"role": "user", "content": "step0"}]
                )
            ),
            json.dumps(
                _make_entry(
                    "traj-A", 1, messages=[{"role": "user", "content": "step1"}]
                )
            ),
        ]
        filename = create_jsonl_file(content)
        loader = AARWLTDatasetLoader(filename=filename, user_config=default_user_config)
        dataset = loader.load_dataset()

        entries = dataset["traj-A"]
        assert [e.conversation_idx for e in entries] == [0, 1, 2]
        assert entries[0].messages[0]["content"] == "step0"

    def test_load_dataset_validates_sequential_indexing(
        self, create_jsonl_file, default_user_config
    ):
        """Gap in indexing raises ValueError."""
        content = [
            json.dumps(_make_entry("traj-A", 0)),
            json.dumps(_make_entry("traj-A", 2)),  # gap: missing idx=1
        ]
        filename = create_jsonl_file(content)
        loader = AARWLTDatasetLoader(filename=filename, user_config=default_user_config)

        with pytest.raises(ValueError, match="non-sequential indexing"):
            loader.load_dataset()

    def test_load_dataset_validates_duplicate_indexing(
        self, create_jsonl_file, default_user_config
    ):
        """Duplicate indices raise ValueError."""
        content = [
            json.dumps(_make_entry("traj-A", 0)),
            json.dumps(_make_entry("traj-A", 0)),  # duplicate
        ]
        filename = create_jsonl_file(content)
        loader = AARWLTDatasetLoader(filename=filename, user_config=default_user_config)

        with pytest.raises(ValueError, match="non-sequential indexing"):
            loader.load_dataset()

    def test_load_dataset_skips_empty_lines(
        self, create_jsonl_file, default_user_config
    ):
        content = [
            json.dumps(_make_entry("traj-A", 0)),
            "",
            json.dumps(_make_entry("traj-A", 1)),
        ]
        filename = create_jsonl_file(content)
        loader = AARWLTDatasetLoader(filename=filename, user_config=default_user_config)
        dataset = loader.load_dataset()
        assert len(dataset["traj-A"]) == 2


class TestAARWLTConvertToConversations:
    """Tests for convert_to_conversations."""

    def test_creates_correct_structure(self, default_user_config):
        tools = [{"type": "function", "function": {"name": "search"}}]
        data = {
            "traj-001": [
                AARWLTEntry(
                    conversation_id="traj-001",
                    conversation_idx=0,
                    messages=[
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "hi"},
                    ],
                    tools=tools,
                ),
                AARWLTEntry(
                    conversation_id="traj-001",
                    conversation_idx=1,
                    messages=[
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "hi"},
                        {
                            "role": "assistant",
                            "content": "hello",
                            "tool_calls": [{"id": "1", "function": {"name": "search"}}],
                        },
                        {"role": "tool", "tool_call_id": "1", "content": "result"},
                    ],
                    tools=tools,
                ),
            ]
        }
        loader = AARWLTDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        conv = conversations[0]
        assert conv.session_id == "traj-001"
        assert conv.discard_assistant_response is True
        assert conv.tools == tools
        assert len(conv.turns) == 2

        # Turn 0 has 2 messages (system + user)
        assert conv.turns[0].raw_messages is not None
        assert len(conv.turns[0].raw_messages) == 2

        # Turn 1 has 2 delta messages (assistant + tool), not the 4 cumulative
        assert conv.turns[1].raw_messages is not None
        assert len(conv.turns[1].raw_messages) == 2
        assert conv.turns[1].raw_messages[0]["role"] == "assistant"
        assert conv.turns[1].raw_messages[1]["role"] == "tool"

    def test_zero_delay(self, default_user_config):
        data = {
            "traj-001": [
                AARWLTEntry(
                    conversation_id="traj-001",
                    conversation_idx=0,
                    messages=[{"role": "user", "content": "hi"}],
                ),
                AARWLTEntry(
                    conversation_id="traj-001",
                    conversation_idx=1,
                    messages=[
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hey"},
                    ],
                ),
            ]
        }
        loader = AARWLTDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )
        conversations = loader.convert_to_conversations(data)
        for turn in conversations[0].turns:
            assert turn.delay == 0

    def test_no_tools(self, default_user_config):
        data = {
            "traj-001": [
                AARWLTEntry(
                    conversation_id="traj-001",
                    conversation_idx=0,
                    messages=[{"role": "user", "content": "hi"}],
                ),
            ]
        }
        loader = AARWLTDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )
        conversations = loader.convert_to_conversations(data)
        assert conversations[0].tools is None

    def test_multiple_trajectories(self, default_user_config):
        data = {
            "traj-A": [
                AARWLTEntry(
                    conversation_id="traj-A",
                    conversation_idx=0,
                    messages=[{"role": "user", "content": "a"}],
                ),
            ],
            "traj-B": [
                AARWLTEntry(
                    conversation_id="traj-B",
                    conversation_idx=0,
                    messages=[{"role": "user", "content": "b"}],
                ),
                AARWLTEntry(
                    conversation_id="traj-B",
                    conversation_idx=1,
                    messages=[
                        {"role": "user", "content": "b"},
                        {"role": "assistant", "content": "b-reply"},
                        {"role": "user", "content": "b2"},
                    ],
                ),
            ],
        }
        loader = AARWLTDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2
        ids = {c.session_id for c in conversations}
        assert ids == {"traj-A", "traj-B"}
