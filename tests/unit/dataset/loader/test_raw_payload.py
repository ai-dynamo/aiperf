# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.common.enums import ConversationContextMode
from aiperf.dataset.loader.raw_payload import RawPayloadDatasetLoader


@pytest.fixture
def single_payload():
    return {"messages": [{"role": "user", "content": "Hello"}], "model": "test"}


@pytest.fixture
def jsonl_file(tmp_path, single_payload):
    """Create a JSONL file with raw payloads."""
    path = tmp_path / "payloads.jsonl"
    lines = [
        orjson.dumps(single_payload),
        orjson.dumps(
            {"messages": [{"role": "user", "content": "World"}], "model": "test"}
        ),
    ]
    path.write_bytes(b"\n".join(lines) + b"\n")
    return path


@pytest.fixture
def jsonl_directory(tmp_path):
    """Create a directory with JSONL files (multi-turn sessions)."""
    d = tmp_path / "sessions"
    d.mkdir()
    for i in range(2):
        lines = [
            orjson.dumps(
                {"messages": [{"role": "user", "content": f"Turn {j} of session {i}"}]}
            )
            for j in range(3)
        ]
        (d / f"session_{i}.jsonl").write_bytes(b"\n".join(lines) + b"\n")
    return d


class TestCanLoad:
    def test_accepts_chat_payload(self, single_payload):
        assert RawPayloadDatasetLoader.can_load(data=single_payload) is True

    def test_rejects_no_messages(self):
        assert RawPayloadDatasetLoader.can_load(data={"model": "x"}) is False

    def test_rejects_agentic_trajectory(self, single_payload):
        single_payload["conversation_id"] = "abc"
        assert RawPayloadDatasetLoader.can_load(data=single_payload) is False

    def test_rejects_inputs_file_format(self, single_payload):
        single_payload["data"] = [{"payloads": []}]
        assert RawPayloadDatasetLoader.can_load(data=single_payload) is False

    def test_accepts_directory(self, jsonl_directory):
        assert RawPayloadDatasetLoader.can_load(filename=jsonl_directory) is True

    def test_rejects_empty_directory(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert RawPayloadDatasetLoader.can_load(filename=d) is False

    def test_returns_false_for_none(self):
        assert RawPayloadDatasetLoader.can_load() is False


class TestLoadDataset:
    def _make_loader(self, filename):
        loader = RawPayloadDatasetLoader.__new__(RawPayloadDatasetLoader)
        loader.filename = str(filename)
        loader.session_id_generator = MagicMock()
        loader.session_id_generator.next.side_effect = [f"s{i}" for i in range(100)]
        loader.info = MagicMock()
        loader.debug = MagicMock()
        return loader

    def test_load_single_file(self, jsonl_file):
        loader = self._make_loader(jsonl_file)
        data = loader.load_dataset()
        assert len(data) == 2
        for payloads in data.values():
            assert len(payloads) == 1
            assert "messages" in payloads[0].payload

    def test_load_directory(self, jsonl_directory):
        loader = self._make_loader(jsonl_directory)
        data = loader.load_dataset()
        assert len(data) == 2
        for payloads in data.values():
            assert len(payloads) == 3

    def test_convert_to_conversations(self, jsonl_file):
        loader = self._make_loader(jsonl_file)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        assert len(conversations) == 2
        for conv in conversations:
            assert len(conv.turns) == 1
            assert conv.turns[0].raw_payload is not None
            assert conv.turns[0].role == "user"
            assert (
                conv.context_mode
                == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
            )


class TestContextMode:
    def test_default_context_mode_is_message_array_with_responses(self):
        assert (
            RawPayloadDatasetLoader.get_default_context_mode()
            == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        )
