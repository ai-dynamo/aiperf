# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock

import orjson
import pytest
from pydantic import ValidationError

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader


def test_mooncake_trace_accepts_extra():
    t = MooncakeTrace(
        text_input="Hello",
        extra={"vendor_top_k": 5, "ignore_eos": True},
    )
    assert t.extra == {"vendor_top_k": 5, "ignore_eos": True}


def test_mooncake_trace_extra_defaults_to_none():
    t = MooncakeTrace(text_input="Hello")
    assert t.extra is None


@pytest.fixture
def default_user_config() -> UserConfig:
    return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))


@pytest.fixture
def mock_prompt_generator():
    generator = Mock()
    generator.generate.return_value = "Generated prompt text"
    generator._decoded_cache = {}
    generator._build_token_sequence.return_value = [1, 2, 3, 4, 5]
    return generator


class TestMooncakeTracePayloadMode:
    def test_payload_field_accepted(self):
        t = MooncakeTrace(
            payload={"prompt": "Hello", "max_tokens": 50},
            timestamp=1000,
        )
        assert t.payload == {"prompt": "Hello", "max_tokens": 50}

    def test_payload_mutually_exclusive_with_input_length(self):
        with pytest.raises(ValidationError):
            MooncakeTrace(
                payload={"prompt": "Hello"},
                input_length=10,
            )

    def test_payload_mutually_exclusive_with_messages(self):
        with pytest.raises(ValidationError):
            MooncakeTrace(
                payload={"prompt": "Hello"},
                messages=[{"role": "user", "content": "x"}],
            )

    def test_payload_mutually_exclusive_with_text_input(self):
        with pytest.raises(ValidationError):
            MooncakeTrace(
                payload={"prompt": "Hello"},
                text_input="Hello",
            )

    def test_empty_payload_rejected(self):
        with pytest.raises(ValidationError):
            MooncakeTrace(payload={})

    def test_payload_with_hash_ids_rejected(self):
        with pytest.raises(ValidationError):
            MooncakeTrace(
                payload={"prompt": "Hello"},
                hash_ids=[123],
            )


class TestMooncakeTraceLoaderPayload:
    def test_payload_traces_produce_raw_payload_turns(
        self,
        tmp_path: Path,
        default_user_config: UserConfig,
        mock_prompt_generator,
    ):
        file = tmp_path / "trace.jsonl"
        with open(file, "wb") as f:
            for i in range(3):
                f.write(
                    orjson.dumps(
                        {
                            "timestamp": 100 * i,
                            "payload": {
                                "prompt": f"prompt-{i}",
                                "max_tokens": 40,
                            },
                        }
                    )
                )
                f.write(b"\n")

        loader = MooncakeTraceDatasetLoader(
            filename=file,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        conversations = loader.convert_to_conversations(loader.load_dataset())
        assert len(conversations) >= 1
        for conv in conversations:
            for turn in conv.turns:
                assert turn.raw_payload is not None
                assert turn.raw_payload["prompt"].startswith("prompt-")
                assert turn.raw_payload["max_tokens"] == 40

    def test_mixed_payload_and_messages_in_session_rejected(
        self,
        tmp_path: Path,
        default_user_config: UserConfig,
        mock_prompt_generator,
    ):
        file = tmp_path / "mixed.jsonl"
        with open(file, "wb") as f:
            f.write(
                orjson.dumps(
                    {
                        "session_id": "s1",
                        "payload": {"prompt": "p"},
                    }
                )
            )
            f.write(b"\n")
            f.write(
                orjson.dumps(
                    {
                        "session_id": "s1",
                        "messages": [{"role": "user", "content": "m"}],
                    }
                )
            )
            f.write(b"\n")

        loader = MooncakeTraceDatasetLoader(
            filename=file,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        with pytest.raises(ValueError, match="payload.*messages|messages.*payload"):
            loader.convert_to_conversations(loader.load_dataset())

    def test_extra_propagates_to_turn_in_payload_mode(
        self,
        tmp_path: Path,
        default_user_config: UserConfig,
        mock_prompt_generator,
    ):
        file = tmp_path / "trace.jsonl"
        with open(file, "wb") as f:
            f.write(
                orjson.dumps(
                    {
                        "timestamp": 0,
                        "payload": {"prompt": "p", "max_tokens": 40},
                        "extra": {"vendor_x": 1, "stream": False},
                    }
                )
            )
            f.write(b"\n")

        loader = MooncakeTraceDatasetLoader(
            filename=file,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        conversations = loader.convert_to_conversations(loader.load_dataset())
        turn = conversations[0].turns[0]
        assert turn.extra_body == {"vendor_x": 1, "stream": False}
