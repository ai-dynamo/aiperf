# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock

import orjson
import pytest
from pydantic import ValidationError

from aiperf.common.models import Turn
from aiperf.config.flags import CLIConfig
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader


def test_mooncake_trace_accepts_extra() -> None:
    t = MooncakeTrace(
        text_input="Hello",
        extra={"vendor_top_k": 5, "ignore_eos": True},
    )
    assert t.extra == {"vendor_top_k": 5, "ignore_eos": True}


def test_mooncake_trace_extra_defaults_to_none() -> None:
    t = MooncakeTrace(text_input="Hello")
    assert t.extra is None


def test_mooncake_trace_accepts_replay_fields() -> None:
    t = MooncakeTrace(
        text_input="Hello",
        output_length=3,
        output_token_ids=[10, 11, 12],
        request_id="request-a",
    )
    assert t.output_token_ids == [10, 11, 12]
    assert t.request_id == "request-a"


def test_mooncake_trace_rejects_replay_length_mismatch() -> None:
    with pytest.raises(ValidationError, match="output_length.*len"):
        MooncakeTrace(
            text_input="Hello",
            output_length=2,
            output_token_ids=[10, 11, 12],
        )


def test_mooncake_trace_rejects_replay_tokens_without_output_length() -> None:
    with pytest.raises(ValidationError, match="output_length.*required"):
        MooncakeTrace(text_input="Hello", output_token_ids=[10])


def test_mooncake_trace_rejects_negative_replay_token_ids() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        MooncakeTrace(text_input="Hello", output_length=1, output_token_ids=[-1])


@pytest.fixture
def default_cfg() -> CLIConfig:
    return CLIConfig(model_names=["test-model"], url="http://localhost:8000")


@pytest.fixture
def mock_prompt_generator() -> Mock:
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
    @staticmethod
    def _write_jsonl(
        file: Path, rows: list[dict], *, leading_blank: bool = False
    ) -> None:
        with open(file, "wb") as f:
            if leading_blank:
                f.write(b"\n")
            for row in rows:
                f.write(orjson.dumps(row))
                f.write(b"\n")

    @staticmethod
    def _load_turns(
        file: Path,
        default_cfg: CLIConfig,
        mock_prompt_generator: Mock,
    ) -> list[Turn]:
        loader = MooncakeTraceDatasetLoader(
            filename=file,
            cfg=default_cfg,
            prompt_generator=mock_prompt_generator,
        )
        conversations = loader.convert_to_conversations(loader.load_dataset())
        return [turn for conv in conversations for turn in conv.turns]

    def test_payload_traces_produce_raw_payload_turns(
        self,
        tmp_path: Path,
        default_cfg: CLIConfig,
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
            cfg=default_cfg,
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
        default_cfg: CLIConfig,
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
            cfg=default_cfg,
            prompt_generator=mock_prompt_generator,
        )
        with pytest.raises(ValueError, match="payload.*messages|messages.*payload"):
            loader.convert_to_conversations(loader.load_dataset())

    def test_extra_propagates_to_turn_in_payload_mode(
        self,
        tmp_path: Path,
        default_cfg: CLIConfig,
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
            cfg=default_cfg,
            prompt_generator=mock_prompt_generator,
        )
        conversations = loader.convert_to_conversations(loader.load_dataset())
        turn = conversations[0].turns[0]
        assert turn.extra_body == {"vendor_x": 1, "stream": False}

    def test_replay_request_id_injects_output_replay_annotation(
        self,
        tmp_path: Path,
        default_cfg: CLIConfig,
        mock_prompt_generator,
    ):
        file = tmp_path / "trace.jsonl"
        self._write_jsonl(
            file,
            [
                {
                    "request_id": "req-1",
                    "text_input": "hello",
                    "output_length": 2,
                    "output_token_ids": [100, 101],
                    "extra": {"nvext": {"annotations": ["existing"]}},
                }
            ],
        )

        turns = self._load_turns(file, default_cfg, mock_prompt_generator)
        assert turns[0].extra_body == {
            "nvext": {"annotations": ["existing", "output_replay_id:req-1"]}
        }

    def test_replay_key_uses_session_turn_index_without_request_id(
        self,
        tmp_path: Path,
        default_cfg: CLIConfig,
        mock_prompt_generator,
    ):
        file = tmp_path / "trace.jsonl"
        self._write_jsonl(
            file,
            [
                {
                    "session_id": "session-a",
                    "text_input": "first",
                    "output_length": 1,
                    "output_token_ids": [100],
                },
                {
                    "session_id": "session-a",
                    "text_input": "second",
                    "output_length": 1,
                    "output_token_ids": [101],
                },
            ],
        )

        turns = self._load_turns(file, default_cfg, mock_prompt_generator)
        annotations = [
            turn.extra_body["nvext"]["annotations"][0]
            for turn in turns
            if turn.extra_body is not None
        ]
        assert annotations == [
            "output_replay_id:session-a:0",
            "output_replay_id:session-a:1",
        ]

    def test_replay_key_uses_physical_line_index_without_request_or_session(
        self,
        tmp_path: Path,
        default_cfg: CLIConfig,
        mock_prompt_generator,
    ):
        file = tmp_path / "trace.jsonl"
        self._write_jsonl(
            file,
            [
                {
                    "text_input": "hello",
                    "output_length": 1,
                    "output_token_ids": [100],
                }
            ],
            leading_blank=True,
        )

        turns = self._load_turns(file, default_cfg, mock_prompt_generator)
        assert turns[0].extra_body == {
            "nvext": {"annotations": ["output_replay_id:line:1"]}
        }

    def test_replay_annotation_is_injected_into_payload_mode_raw_payload(
        self,
        tmp_path: Path,
        default_cfg: CLIConfig,
        mock_prompt_generator,
    ):
        file = tmp_path / "trace.jsonl"
        self._write_jsonl(
            file,
            [
                {
                    "request_id": "payload-req",
                    "payload": {
                        "messages": [{"role": "user", "content": "hello"}],
                        "model": "test-model",
                        "nvext": {"annotations": ["existing"]},
                    },
                    "output_length": 1,
                    "output_token_ids": [100],
                }
            ],
        )

        turns = self._load_turns(file, default_cfg, mock_prompt_generator)
        assert turns[0].raw_payload["nvext"]["annotations"] == [
            "existing",
            "output_replay_id:payload-req",
        ]
