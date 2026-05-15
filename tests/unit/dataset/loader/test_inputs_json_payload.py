# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import orjson
import pytest

from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import ConversationContextMode
from aiperf.dataset.loader.inputs_json import InputsJsonPayloadLoader


@pytest.fixture
def inputs_json_file(tmp_path: Path) -> Path:
    data = {
        "data": [
            {
                "session_id": "session-001",
                "payloads": [
                    {
                        "messages": [{"role": "user", "content": "Hello"}],
                        "model": "Qwen/Qwen3-0.6B",
                        "max_tokens": 32,
                    },
                    {
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi"},
                            {"role": "user", "content": "How are you?"},
                        ],
                        "model": "Qwen/Qwen3-0.6B",
                        "max_tokens": 64,
                    },
                ],
            },
            {
                "session_id": "session-002",
                "payloads": [
                    {
                        "messages": [{"role": "user", "content": "Bye"}],
                        "model": "Qwen/Qwen3-0.6B",
                    }
                ],
            },
        ]
    }
    p = tmp_path / "inputs.json"
    p.write_bytes(orjson.dumps(data))
    return p


class TestInputsJsonCanLoad:
    def test_can_load_with_data_key(self):
        assert InputsJsonPayloadLoader.can_load(
            data={"data": [{"session_id": "s", "payloads": [{}]}]}
        )

    def test_rejects_non_dict_data(self):
        assert not InputsJsonPayloadLoader.can_load(data={"data": "not a list"})

    def test_rejects_missing_payloads_key(self):
        assert not InputsJsonPayloadLoader.can_load(
            data={"data": [{"session_id": "s"}]}
        )

    def test_can_load_from_file(self, inputs_json_file: Path):
        assert InputsJsonPayloadLoader.can_load(filename=inputs_json_file)

    def test_rejects_empty_dict(self):
        assert not InputsJsonPayloadLoader.can_load(data={})


class TestInputsJsonLoad:
    def test_load_preserves_session_ids_and_turns(
        self, inputs_json_file: Path, default_user_config: UserConfig
    ):
        loader = InputsJsonPayloadLoader(
            filename=inputs_json_file, user_config=default_user_config
        )
        data = loader.load_dataset()
        assert set(data.keys()) == {"session-001", "session-002"}
        assert len(data["session-001"][0].payloads) == 2
        assert len(data["session-002"][0].payloads) == 1

    def test_convert_produces_raw_payload_turns(
        self, inputs_json_file: Path, default_user_config: UserConfig
    ):
        loader = InputsJsonPayloadLoader(
            filename=inputs_json_file, user_config=default_user_config
        )
        conversations = loader.convert_to_conversations(loader.load_dataset())
        assert len(conversations) == 2
        conv = next(c for c in conversations if c.session_id == "session-001")
        assert conv.context_mode == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        assert len(conv.turns) == 2
        for turn in conv.turns:
            assert turn.raw_payload is not None
            assert "messages" in turn.raw_payload
