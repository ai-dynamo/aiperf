# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import orjson
import pytest

from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import ConversationContextMode
from aiperf.dataset.loader.raw_payload import RawPayloadDatasetLoader


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")


@pytest.fixture
def jsonl_file(tmp_path: Path) -> Path:
    p = tmp_path / "payloads.jsonl"
    _write_jsonl(
        p,
        [
            {
                "messages": [{"role": "user", "content": "hi"}],
                "model": "Qwen/Qwen3-0.6B",
                "max_tokens": 16,
            },
            {
                "messages": [{"role": "user", "content": "bye"}],
                "model": "Qwen/Qwen3-0.6B",
                "max_tokens": 16,
            },
        ],
    )
    return p


@pytest.fixture
def jsonl_dir(tmp_path: Path) -> Path:
    d = tmp_path / "convs"
    d.mkdir()
    _write_jsonl(
        d / "session_001.jsonl",
        [
            {"messages": [{"role": "user", "content": "t1"}]},
            {
                "messages": [
                    {"role": "user", "content": "t1"},
                    {"role": "assistant", "content": "r"},
                    {"role": "user", "content": "t2"},
                ]
            },
        ],
    )
    _write_jsonl(
        d / "session_002.jsonl",
        [{"messages": [{"role": "user", "content": "single"}]}],
    )
    return d


class TestRawPayloadCanLoad:
    def test_messages_array_accepted(self):
        assert RawPayloadDatasetLoader.can_load(
            data={"messages": [{"role": "user", "content": "hi"}]}
        )

    def test_conversation_id_rejected(self):
        assert not RawPayloadDatasetLoader.can_load(
            data={
                "messages": [{"role": "user", "content": "hi"}],
                "conversation_id": "abc",
            }
        )

    def test_data_list_rejected(self):
        assert not RawPayloadDatasetLoader.can_load(
            data={"messages": [], "data": [{"session_id": "s", "payloads": []}]}
        )

    def test_directory_with_jsonl_accepted(self, jsonl_dir: Path):
        assert RawPayloadDatasetLoader.can_load(filename=jsonl_dir)


class TestRawPayloadLoad:
    def test_single_file_one_session_per_line(
        self, jsonl_file: Path, default_user_config: UserConfig
    ):
        loader = RawPayloadDatasetLoader(
            filename=jsonl_file, user_config=default_user_config
        )
        data = loader.load_dataset()
        assert len(data) == 2
        for payloads in data.values():
            assert len(payloads) == 1

    def test_directory_one_file_per_session_multi_turn(
        self, jsonl_dir: Path, default_user_config: UserConfig
    ):
        loader = RawPayloadDatasetLoader(
            filename=jsonl_dir, user_config=default_user_config
        )
        data = loader.load_dataset()
        assert len(data) == 2
        turn_counts = sorted(len(payloads) for payloads in data.values())
        assert turn_counts == [1, 2]

    def test_convert_produces_raw_payload_turns(
        self, jsonl_file: Path, default_user_config: UserConfig
    ):
        loader = RawPayloadDatasetLoader(
            filename=jsonl_file, user_config=default_user_config
        )
        conversations = loader.convert_to_conversations(loader.load_dataset())
        assert all(
            c.context_mode == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
            for c in conversations
        )
        for c in conversations:
            for turn in c.turns:
                assert turn.raw_payload is not None
                assert "messages" in turn.raw_payload
