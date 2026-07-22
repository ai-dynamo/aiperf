# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DatasetManager payload-bytes pathways.

Covers:
- Fallback conversation serving from a PAYLOAD_BYTES store (workers whose
  local mmap client is not ready fall back to the DatasetManager; that path
  must reconstruct turns from per-turn payload bytes instead of crashing with
  MemoryMapSerializationError).
- ``_preformat_payloads`` gating (opt-in env, self-contained-only, endpoint
  NotImplementedError skip).
- ``_select_mmap_format`` (PAYLOAD_BYTES only when all turns have raw_payload;
  global mix falls back to CONVERSATION; per-conversation mix raises).
- ``_generate_input_payloads`` verbatim export for raw-payload datasets.
- ``_run_mmap_paths`` compressed (Kubernetes) variants.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from aiperf.common.enums import ConversationContextMode, MemoryMapFormat
from aiperf.common.environment import Environment
from aiperf.common.exceptions import MemoryMapSerializationError, ServiceError
from aiperf.common.messages import (
    ConversationRequestMessage,
    ConversationTurnRequestMessage,
    ProfileConfigureCommand,
)
from aiperf.common.models import Conversation, ModelEndpointInfo, Text, Turn
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.conftest import make_run_from_cli

RAW_PAYLOAD = {
    "messages": [{"role": "user", "content": "raw-payload body"}],
    "model": "test-model",
    "stream": False,
    "max_tokens": 7,
    "vendor_flag": {"preserve": True},
}


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pin the cache + run mmap dirs to tmp so tests never touch ~/.cache."""
    monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_ENABLED", True)
    monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path / "mmap")


@pytest.fixture
def mock_tokenizer(mock_tokenizer_cls):
    with patch("aiperf.common.tokenizer.Tokenizer.from_pretrained") as mock:
        mock.return_value = mock_tokenizer_cls.from_pretrained("test-model")
        yield mock


def _make_raw_payload_run(tmp_path: Path) -> BenchmarkRun:
    payload_file = tmp_path / "payloads.jsonl"
    payload_file.write_bytes(orjson.dumps(RAW_PAYLOAD) + b"\n")
    return make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            tokenizer_name="test-tokenizer",
            input_file=str(payload_file),
            custom_dataset_type=CustomDatasetType.RAW_PAYLOAD,
        )
    )


async def _configure_manager(run: BenchmarkRun) -> DatasetManager:
    dataset_manager = DatasetManager(run=run, service_id="dm-test")
    await dataset_manager.initialize()
    dataset_manager.publish = AsyncMock()
    await dataset_manager._profile_configure_command(
        ProfileConfigureCommand(service_id="dm-test")
    )
    return dataset_manager


class TestPayloadBytesFallbackServing:
    """The DM fallback request path must serve PAYLOAD_BYTES datasets.

    Regression for the worker-misses-DatasetConfiguredNotification race: with
    the mmap cache HIT making dataset configure nearly instant, a slow-starting
    worker can subscribe after the broadcast, keep ``_dataset_client=None``,
    and fall back to the DM conversation request. That request used to crash
    with MemoryMapSerializationError on PAYLOAD_BYTES stores, failing the run.
    """

    @pytest.mark.asyncio
    async def test_conversation_request_reconstructs_from_payload_bytes(
        self, tmp_path: Path, mock_tokenizer
    ) -> None:
        dm = await _configure_manager(_make_raw_payload_run(tmp_path))
        assert dm.dataset_metadata is not None
        conversation_id = dm.dataset_metadata.conversations[0].conversation_id

        # Precondition: the store really is PAYLOAD_BYTES, so the plain
        # get_conversation path raises and the handler MUST reconstruct.
        assert dm._dataset_client is not None
        with pytest.raises(MemoryMapSerializationError):
            await dm._dataset_client.get_conversation(conversation_id)

        response = await dm._handle_conversation_request(
            ConversationRequestMessage(
                service_id="worker-test", conversation_id=conversation_id
            )
        )

        conversation = response.conversation
        assert conversation.session_id == conversation_id
        assert len(conversation.turns) == 1
        assert conversation.turns[0].raw_payload == RAW_PAYLOAD
        await dm.stop()

    @pytest.mark.asyncio
    async def test_turn_request_reconstructs_from_payload_bytes(
        self, tmp_path: Path, mock_tokenizer
    ) -> None:
        dm = await _configure_manager(_make_raw_payload_run(tmp_path))
        assert dm.dataset_metadata is not None
        conversation_id = dm.dataset_metadata.conversations[0].conversation_id

        response = await dm._handle_conversation_turn_request(
            ConversationTurnRequestMessage(
                service_id="worker-test",
                conversation_id=conversation_id,
                turn_index=0,
            )
        )

        assert response.turn.raw_payload == RAW_PAYLOAD
        await dm.stop()

    @pytest.mark.asyncio
    async def test_turn_request_out_of_range_raises_service_error(
        self, tmp_path: Path, mock_tokenizer
    ) -> None:
        dm = await _configure_manager(_make_raw_payload_run(tmp_path))
        assert dm.dataset_metadata is not None
        conversation_id = dm.dataset_metadata.conversations[0].conversation_id

        with pytest.raises(ServiceError, match="out of range"):
            await dm._handle_conversation_turn_request(
                ConversationTurnRequestMessage(
                    service_id="worker-test",
                    conversation_id=conversation_id,
                    turn_index=99,
                )
            )
        await dm.stop()

    @pytest.mark.asyncio
    async def test_unknown_conversation_raises_service_error(
        self, tmp_path: Path, mock_tokenizer
    ) -> None:
        dm = await _configure_manager(_make_raw_payload_run(tmp_path))

        with pytest.raises(ServiceError, match="not found"):
            await dm._handle_conversation_request(
                ConversationRequestMessage(
                    service_id="worker-test", conversation_id="no-such-conversation"
                )
            )
        await dm.stop()

    @pytest.mark.asyncio
    async def test_reconstruction_returns_none_without_payload_bytes_api(
        self, tmp_path: Path
    ) -> None:
        dm = DatasetManager(run=_make_raw_payload_run(tmp_path), service_id="dm-test")
        dm._dataset_client = object()  # no get_payload_bytes attribute
        assert await dm._conversation_from_payload_bytes("any") is None


def _make_synthetic_run() -> BenchmarkRun:
    return make_run_from_cli(CLIConfig(model_names=["test-model"]))


def _single_turn_conversation(session_id: str = "s1") -> Conversation:
    return Conversation(
        session_id=session_id,
        turns=[Turn(role="user", texts=[Text(contents=["hello"])])],
    )


def _raw_conversation(session_id: str = "raw1", num_turns: int = 1) -> Conversation:
    return Conversation(
        session_id=session_id,
        turns=[
            Turn(role="user", raw_payload={"p": i, "session": session_id})
            for i in range(num_turns)
        ],
    )


class TestPreformatPayloads:
    """Opt-in pre-encoding of structured conversations to raw_payload."""

    def _manager(self) -> DatasetManager:
        return DatasetManager(run=_make_synthetic_run(), service_id="dm-test")

    def test_disabled_by_default_is_noop(self) -> None:
        dm = self._manager()
        conversations = [_single_turn_conversation()]
        dm._preformat_payloads(conversations)
        assert conversations[0].turns[0].raw_payload is None

    def test_all_raw_conversations_skip_formatting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Environment.DATASET, "PREFORMAT_PAYLOADS", True)
        dm = self._manager()
        conversations = [_raw_conversation()]
        before = conversations[0].turns[0].raw_payload

        dm._preformat_payloads(conversations)

        assert conversations[0].turns[0].raw_payload == before

    def test_multi_turn_delta_conversation_skips_entire_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One non-self-contained multi-turn conversation disqualifies ALL
        conversations (mixed raw_payload state is rejected downstream)."""
        monkeypatch.setattr(Environment.DATASET, "PREFORMAT_PAYLOADS", True)
        dm = self._manager()
        single = _single_turn_conversation()
        multi = Conversation(
            session_id="m1",
            turns=[
                Turn(role="user", texts=[Text(contents=["a"])]),
                Turn(role="user", texts=[Text(contents=["b"])]),
            ],
        )

        dm._preformat_payloads([single, multi])

        assert single.turns[0].raw_payload is None
        assert all(t.raw_payload is None for t in multi.turns)

    def test_self_contained_multi_turn_is_eligible(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Environment.DATASET, "PREFORMAT_PAYLOADS", True)
        dm = self._manager()
        conversation = Conversation(
            session_id="sc1",
            context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
            turns=[
                Turn(role="user", texts=[Text(contents=["a"])]),
                Turn(role="user", texts=[Text(contents=["b"])]),
            ],
        )

        mock_endpoint = MagicMock()
        mock_endpoint.format_payload.side_effect = [{"p": 1}, {"p": 2}]
        mock_endpoint.get_endpoint_headers.return_value = {}
        mock_endpoint.get_endpoint_params.return_value = {}

        with patch(
            "aiperf.dataset.payload_formatting.plugins.get_class",
            return_value=lambda **kwargs: mock_endpoint,
        ):
            dm._preformat_payloads([conversation])

        assert conversation.turns[0].raw_payload == {"p": 1}
        assert conversation.turns[1].raw_payload == {"p": 2}

    def test_endpoint_without_format_payload_skips(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Environment.DATASET, "PREFORMAT_PAYLOADS", True)
        dm = self._manager()
        conversation = _single_turn_conversation()

        with patch(
            "aiperf.dataset.payload_formatting.format_conversation_payloads",
            side_effect=NotImplementedError("no format_payload"),
        ):
            dm._preformat_payloads([conversation])

        assert conversation.turns[0].raw_payload is None

    def test_formats_single_turn_conversations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Environment.DATASET, "PREFORMAT_PAYLOADS", True)
        dm = self._manager()
        conversations = [
            _single_turn_conversation("s1"),
            _single_turn_conversation("s2"),
        ]

        mock_endpoint = MagicMock()
        mock_endpoint.format_payload.side_effect = [{"p": "s1"}, {"p": "s2"}]
        mock_endpoint.get_endpoint_headers.return_value = {}
        mock_endpoint.get_endpoint_params.return_value = {}

        with patch(
            "aiperf.dataset.payload_formatting.plugins.get_class",
            return_value=lambda **kwargs: mock_endpoint,
        ):
            dm._preformat_payloads(conversations)

        assert conversations[0].turns[0].raw_payload == {"p": "s1"}
        assert conversations[1].turns[0].raw_payload == {"p": "s2"}


class TestSelectMmapFormat:
    def _manager(self) -> DatasetManager:
        return DatasetManager(run=_make_synthetic_run(), service_id="dm-test")

    def test_all_raw_selects_payload_bytes(self) -> None:
        dm = self._manager()
        assert (
            dm._select_mmap_format([_raw_conversation("r1"), _raw_conversation("r2")])
            == MemoryMapFormat.PAYLOAD_BYTES
        )

    def test_no_raw_selects_conversation(self) -> None:
        dm = self._manager()
        assert (
            dm._select_mmap_format([_single_turn_conversation()])
            == MemoryMapFormat.CONVERSATION
        )

    def test_mixed_across_conversations_selects_conversation(self) -> None:
        """Global mix is allowed: fall back to CONVERSATION rather than hard-fail."""
        dm = self._manager()
        assert (
            dm._select_mmap_format(
                [_raw_conversation("r1"), _single_turn_conversation("s1")]
            )
            == MemoryMapFormat.CONVERSATION
        )

    def test_mixed_within_conversation_raises(self) -> None:
        dm = self._manager()
        mixed = Conversation(
            session_id="mixed",
            turns=[
                Turn(role="user", raw_payload={"p": 0}),
                Turn(role="user", texts=[Text(contents=["no payload"])]),
            ],
        )
        with pytest.raises(ValueError, match="mixed[\\s\\S]*raw_payload"):
            dm._select_mmap_format([mixed])


class TestGenerateInputPayloadsVerbatim:
    """Raw-payload datasets are exported to inputs.json VERBATIM."""

    def _manager_with_dataset(
        self, conversations: list[Conversation]
    ) -> DatasetManager:
        dm = DatasetManager(run=_make_synthetic_run(), service_id="dm-test")
        dm.dataset = {c.session_id: c for c in conversations}
        return dm

    def test_raw_payloads_exported_verbatim(self) -> None:
        dm = self._manager_with_dataset(
            [_raw_conversation("r1", num_turns=2), _raw_conversation("r2")]
        )
        model_endpoint = ModelEndpointInfo.from_run(dm.run)

        inputs = dm._generate_input_payloads(model_endpoint)

        by_session = {s.session_id: s.payloads for s in inputs.data}
        assert by_session["r1"] == [
            {"p": 0, "session": "r1"},
            {"p": 1, "session": "r1"},
        ]
        assert by_session["r2"] == [{"p": 0, "session": "r2"}]

    def test_mixed_raw_state_within_conversation_raises(self) -> None:
        mixed = Conversation(
            session_id="mixed",
            turns=[
                Turn(role="user", raw_payload={"p": 0}),
                Turn(role="user", texts=[Text(contents=["no payload"])]),
            ],
        )
        dm = self._manager_with_dataset([mixed])
        model_endpoint = ModelEndpointInfo.from_run(dm.run)

        with pytest.raises(ValueError, match="mixed[\\s\\S]*raw_payload"):
            dm._generate_input_payloads(model_endpoint)


class TestRunMmapPaths:
    def test_local_paths_are_uncompressed(self) -> None:
        dm = DatasetManager(run=_make_synthetic_run(), service_id="dm-test")
        dm._compress_only = False
        data_p, index_p = dm._run_mmap_paths()
        assert data_p.name == "dataset.dat"
        assert index_p.name == "index.dat"

    def test_kubernetes_paths_are_compressed(self) -> None:
        dm = DatasetManager(run=_make_synthetic_run(), service_id="dm-test")
        dm._compress_only = True
        data_p, index_p = dm._run_mmap_paths()
        assert data_p.name == "dataset.dat.zst"
        assert index_p.name == "index.dat.zst"
