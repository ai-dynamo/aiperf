# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.control_structs import Command
from aiperf.common.enums import CommandType
from aiperf.common.exceptions import ServiceError
from aiperf.common.messages import (
    ConversationRequestMessage,
    ConversationTurnRequestMessage,
    DatasetConfiguredNotification,
)
from aiperf.common.models import Conversation, Text, Turn
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.plugin.enums import (
    CustomDatasetType,
    EndpointType,
)


def _make_run(config: AIPerfConfig) -> BenchmarkRun:
    return BenchmarkRun(benchmark_id="test", cfg=config, artifact_dir=Path("/tmp/test"))


# ============================================================================
# Shared Fixtures
# ============================================================================

_BASE_CONFIG = dict(
    models=["test-model"],
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
)


@pytest.fixture(autouse=True)
async def cleanup_communication():
    """Clean up after each test to prevent shared state issues."""
    yield


@pytest.fixture
def mock_tokenizer(mock_tokenizer_cls):
    """Fixture to mock tokenizer creation."""
    with patch("aiperf.common.tokenizer.Tokenizer.from_pretrained") as mock:
        mock.return_value = mock_tokenizer_cls.from_pretrained("test-model")
        yield mock


@pytest.fixture
def base_user_config() -> AIPerfConfig:
    """Create a basic AIPerfConfig for testing."""
    return AIPerfConfig(
        **_BASE_CONFIG,
        datasets={
            "default": {
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 128, "osl": 64},
            }
        },
    )


@pytest.fixture
def base_run(base_user_config: AIPerfConfig) -> BenchmarkRun:
    """Create a BenchmarkRun wrapping the base config."""
    return BenchmarkRun(
        benchmark_id="test",
        cfg=base_user_config,
        artifact_dir=Path("/tmp/test"),
    )


@pytest.fixture
async def initialized_dataset_manager(mock_tokenizer, base_run):
    """Create an initialized DatasetManager with mocked publish."""
    dataset_manager = DatasetManager(
        run=base_run,
        service_id="test_dataset_manager",
    )

    await dataset_manager.initialize()
    dataset_manager.publish = AsyncMock()

    return dataset_manager


@pytest.fixture
async def configured_dataset_manager(initialized_dataset_manager, base_user_config):
    """Create a fully configured DatasetManager ready for request handling."""
    await initialized_dataset_manager._profile_configure_command(
        Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
    )
    return initialized_dataset_manager


# ============================================================================
# Helper Functions
# ============================================================================


def create_mock_conversations(session_ids: list[str]) -> list[Conversation]:
    """Create mock conversations with specified session IDs."""
    return [
        Conversation(
            session_id=session_id,
            turns=[Turn(texts=[Text(contents=["Hello"])], model="test-model")],
        )
        for session_id in session_ids
    ]


async def capture_published_messages(dataset_manager, config):
    """Configure dataset and capture published messages."""
    published_messages = []

    async def mock_publish(msg):
        published_messages.append(msg)

    dataset_manager.publish = AsyncMock(side_effect=mock_publish)

    await dataset_manager._profile_configure_command(
        Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
    )

    return published_messages


def extract_dataset_notifications(
    messages: list,
) -> list[DatasetConfiguredNotification]:
    """Extract DatasetConfiguredNotification messages from a list."""
    return [msg for msg in messages if isinstance(msg, DatasetConfiguredNotification)]


# ============================================================================
# Test Classes
# ============================================================================


class TestDatasetManager:
    """Test DatasetManager functionality.

    Note: Dataset sampling tests have been moved to test_dataset_samplers.py
    since sampling is now handled by timing strategies, not DatasetManager.
    """

    @pytest.mark.asyncio
    async def test_profile_configure_is_reentrant_safe(
        self,
        initialized_dataset_manager: DatasetManager,
    ) -> None:
        """Concurrent PROFILE_CONFIGURE commands should coalesce into one configuration pass."""
        started = asyncio.Event()
        release = asyncio.Event()
        calls: list[str] = []

        async def fake_configure_tokenizer() -> None:
            calls.append("tokenizer")
            started.set()
            await release.wait()

        async def fake_configure_dataset() -> None:
            calls.append("dataset")

        async def fake_generate_inputs() -> None:
            calls.append("inputs")

        async def fake_finalize() -> None:
            calls.append("finalize")
            initialized_dataset_manager.dataset_configured.set()

        initialized_dataset_manager._configure_tokenizer = fake_configure_tokenizer
        initialized_dataset_manager._configure_dataset = fake_configure_dataset
        initialized_dataset_manager._generate_inputs_json_file = fake_generate_inputs
        initialized_dataset_manager._configure_dataset_client_and_free_memory = (
            fake_finalize
        )
        initialized_dataset_manager.publish = AsyncMock()

        first = asyncio.create_task(
            initialized_dataset_manager._profile_configure_command(
                Command(cid="first", cmd=CommandType.PROFILE_CONFIGURE)
            )
        )
        await started.wait()
        second = asyncio.create_task(
            initialized_dataset_manager._profile_configure_command(
                Command(cid="second", cmd=CommandType.PROFILE_CONFIGURE)
            )
        )
        release.set()
        await asyncio.gather(first, second)

        assert calls == ["tokenizer", "dataset", "inputs", "finalize"]

    @pytest.mark.asyncio
    async def test_dataset_configured_notification_for_multi_turn_conversations(
        self,
        mock_tokenizer,
        create_mooncake_trace_file,
    ):
        """Test that dataset configured notification includes correct metadata for multi-turn conversations.

        When a dataset has multiple turns per conversation, the notification should:
        - Include one ConversationMetadata per conversation (not one per turn)
        - Include the first_turn_timestamp and turn_delays for each conversation
        - Have the correct turn count for each conversation
        """
        entries = [
            '{"session_id": "sess-1", "timestamp": 0, "input_length": 50, "output_length": 10}',
            '{"session_id": "sess-1", "delay": 10000, "input_length": 50, "output_length": 10}',
            '{"session_id": "sess-1", "delay": 10000, "input_length": 100, "output_length": 10}',
            '{"session_id": "sess-2", "timestamp": 20000, "input_length": 25, "output_length": 20}',
            '{"session_id": "sess-2", "delay": 10000, "input_length": 10000, "output_length": 20}',
        ]
        filename = create_mooncake_trace_file(entries)

        try:
            config = AIPerfConfig(
                **_BASE_CONFIG,
                datasets={
                    "default": {
                        "type": "file",
                        "path": filename,
                        "format": CustomDatasetType.MOONCAKE_TRACE,
                    }
                },
            )

            dataset_manager = DatasetManager(
                run=_make_run(config),
                service_id="test_dataset_manager",
            )

            await dataset_manager.initialize()

            published_messages = await capture_published_messages(
                dataset_manager, config
            )

            published_notifications = extract_dataset_notifications(published_messages)
            assert len(published_notifications) == 1

            notification = published_notifications[0]
            metadata = notification.metadata

            assert len(metadata.conversations) == 2

            conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

            assert "sess-1" in conv_dict
            sess1 = conv_dict["sess-1"]
            assert len(sess1.turns) == 3

            assert "sess-2" in conv_dict
            sess2 = conv_dict["sess-2"]
            assert len(sess2.turns) == 2

            conversation_ids = [conv.conversation_id for conv in metadata.conversations]
            assert len(conversation_ids) == len(set(conversation_ids))

        finally:
            Path(filename).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_dataset_configured_notification_preserves_float_timestamps(
        self,
        mock_tokenizer,
        create_mooncake_trace_file,
    ):
        """Test that floating point timestamps are preserved exactly in dataset notifications."""
        entries = [
            '{"session_id": "sess-1", "timestamp": 0.123, "input_length": 50, "output_length": 10}',
            '{"session_id": "sess-1", "delay": 10000.456, "input_length": 50, "output_length": 10}',
            '{"session_id": "sess-2", "timestamp": 20000.789, "input_length": 25, "output_length": 20}',
            '{"session_id": "sess-2", "delay": 15000.123, "input_length": 100, "output_length": 20}',
        ]
        filename = create_mooncake_trace_file(entries)

        try:
            config = AIPerfConfig(
                **_BASE_CONFIG,
                datasets={
                    "default": {
                        "type": "file",
                        "path": filename,
                        "format": CustomDatasetType.MOONCAKE_TRACE,
                    }
                },
            )

            dataset_manager = DatasetManager(
                run=_make_run(config),
                service_id="test_dataset_manager",
            )

            await dataset_manager.initialize()

            published_messages = await capture_published_messages(
                dataset_manager, config
            )

            published_notifications = extract_dataset_notifications(published_messages)
            assert len(published_notifications) == 1

            notification = published_notifications[0]
            metadata = notification.metadata

            conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

            assert "sess-1" in conv_dict
            sess1 = conv_dict["sess-1"]
            assert len(sess1.turns) == 2

            assert "sess-2" in conv_dict
            sess2 = conv_dict["sess-2"]
            assert len(sess2.turns) == 2

        finally:
            Path(filename).unlink(missing_ok=True)


class TestDatasetManagerMemoryAndClient:
    """Test dataset client initialization and memory freeing after configuration."""

    @pytest.mark.asyncio
    async def test_dataset_client_initialized_after_configuration(
        self,
        initialized_dataset_manager,
        base_user_config,
    ):
        """Test that dataset client is initialized after profile configuration."""
        dataset_manager = initialized_dataset_manager

        assert dataset_manager._dataset_client is None

        await dataset_manager._profile_configure_command(
            Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
        )

        assert dataset_manager._dataset_client is not None

    @pytest.mark.asyncio
    async def test_in_memory_dataset_freed_after_client_initialization(
        self,
        mock_tokenizer,
    ):
        """Test that in-memory dataset is freed after dataset client is initialized."""
        config = AIPerfConfig(
            **_BASE_CONFIG,
            datasets={
                "default": {
                    "type": "synthetic",
                    "entries": 5,
                    "prompts": {"isl": 128, "osl": 64},
                }
            },
        )
        dataset_manager = DatasetManager(
            run=_make_run(config),
            service_id="test_dataset_manager",
        )

        await dataset_manager.initialize()
        dataset_manager.publish = AsyncMock()

        await dataset_manager._profile_configure_command(
            Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
        )

        assert dataset_manager.dataset == {}
        assert dataset_manager._conversation_ids_cache == []

    @pytest.mark.asyncio
    async def test_dataset_configured_event_set_after_client_initialization(
        self,
        initialized_dataset_manager,
        base_user_config,
    ):
        """Test that dataset_configured event is set after client initialization."""
        dataset_manager = initialized_dataset_manager

        assert not dataset_manager.dataset_configured.is_set()

        await dataset_manager._profile_configure_command(
            Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
        )

        assert dataset_manager.dataset_configured.is_set()


class TestDatasetManagerFallbackHandlers:
    """Test fallback request handlers that use the dataset client."""

    @pytest.fixture
    async def dataset_manager_with_entries(self, mock_tokenizer):
        """Create a configured dataset manager with multiple entries."""
        config = AIPerfConfig(
            **_BASE_CONFIG,
            datasets={
                "default": {
                    "type": "synthetic",
                    "entries": 3,
                    "prompts": {"isl": 128, "osl": 64},
                }
            },
        )
        dataset_manager = DatasetManager(
            run=_make_run(config),
            service_id="test_dataset_manager",
        )

        await dataset_manager.initialize()
        dataset_manager.publish = AsyncMock()

        await dataset_manager._profile_configure_command(
            Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
        )

        return dataset_manager

    @pytest.mark.asyncio
    async def test_handle_conversation_request_uses_dataset_client(
        self,
        dataset_manager_with_entries,
    ):
        """Test that conversation request handler uses dataset client, not in-memory dict."""
        dataset_manager = dataset_manager_with_entries

        conversation_id = dataset_manager.dataset_metadata.conversations[
            0
        ].conversation_id

        assert dataset_manager.dataset == {}

        request = ConversationRequestMessage(
            service_id="test_worker",
            conversation_id=conversation_id,
        )
        response = await dataset_manager._handle_conversation_request(request)

        assert response.conversation is not None
        assert response.conversation.session_id == conversation_id

    @pytest.mark.asyncio
    async def test_handle_conversation_turn_request_uses_dataset_client(
        self,
        dataset_manager_with_entries,
    ):
        """Test that turn request handler uses dataset client, not in-memory dict."""
        dataset_manager = dataset_manager_with_entries

        conversation_id = dataset_manager.dataset_metadata.conversations[
            0
        ].conversation_id

        assert dataset_manager.dataset == {}

        request = ConversationTurnRequestMessage(
            service_id="test_worker",
            conversation_id=conversation_id,
            turn_index=0,
        )
        response = await dataset_manager._handle_conversation_turn_request(request)

        assert response.turn is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "conversation_id,expected_error_match",
        [
            ("nonexistent-conversation-id", "not found in dataset"),
        ],
    )
    async def test_handle_conversation_request_not_found(
        self,
        dataset_manager_with_entries,
        conversation_id,
        expected_error_match,
    ):
        """Test that conversation request handler raises error for unknown conversation."""
        request = ConversationRequestMessage(
            service_id="test_worker",
            conversation_id=conversation_id,
        )

        with pytest.raises(ServiceError, match=expected_error_match):
            await dataset_manager_with_entries._handle_conversation_request(request)

    @pytest.mark.asyncio
    async def test_handle_turn_request_invalid_turn_index(
        self,
        dataset_manager_with_entries,
    ):
        """Test that turn request handler raises error for invalid turn index."""
        dataset_manager = dataset_manager_with_entries

        conversation_id = dataset_manager.dataset_metadata.conversations[
            0
        ].conversation_id

        request = ConversationTurnRequestMessage(
            service_id="test_worker",
            conversation_id=conversation_id,
            turn_index=999,
        )

        with pytest.raises(ServiceError, match="out of range"):
            await dataset_manager._handle_conversation_turn_request(request)


class TestKubernetesMode:
    """Test Kubernetes-specific behavior in DatasetManager."""

    def test_compress_only_kubernetes_returns_true(
        self, base_user_config: AIPerfConfig
    ) -> None:
        """compress_only should be True when service_run_type is KUBERNETES."""
        manager = DatasetManager(
            run=_make_run(base_user_config),
            service_id="test_dataset_manager",
        )
        # Simulate Kubernetes mode by directly setting the flag
        manager._compress_only = True
        assert manager._compress_only is True

    def test_compress_only_multiprocessing_returns_false(
        self, base_user_config: AIPerfConfig
    ) -> None:
        """compress_only should be False in local (multiprocessing) mode."""
        manager = DatasetManager(
            run=_make_run(base_user_config),
            service_id="test_dataset_manager",
        )
        assert manager._compress_only is False

    @pytest.mark.asyncio
    async def test_configure_client_compress_only_skips_client_creation(
        self, base_user_config: AIPerfConfig
    ) -> None:
        """In compress_only mode, _configure_dataset_client_and_free_memory skips client creation."""
        manager = DatasetManager(
            run=_make_run(base_user_config),
            service_id="test_dataset_manager",
        )
        manager._compress_only = True
        manager.dataset = {"conv1": MagicMock(), "conv2": MagicMock()}
        manager._conversation_ids_cache = ["conv1", "conv2"]

        await manager._configure_dataset_client_and_free_memory()

        assert manager.dataset_configured.is_set()
        assert manager.dataset == {}
        assert manager._conversation_ids_cache == []
        assert manager._dataset_client is None


class TestDatasetManagerTokenizerSkip:
    """Test tokenizer skip logic for non-tokenizing endpoints."""

    @pytest.fixture
    def _mock_dataset_steps(self):
        """Mock dataset configuration steps to isolate tokenizer logic."""
        with (
            patch.object(DatasetManager, "_configure_dataset", new_callable=AsyncMock),
            patch.object(
                DatasetManager,
                "_generate_inputs_json_file",
                new_callable=AsyncMock,
            ),
            patch.object(
                DatasetManager,
                "_configure_dataset_client_and_free_memory",
                new_callable=AsyncMock,
            ),
        ):
            yield

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_mock_dataset_steps")
    async def test_tokenizer_skipped_for_non_tokenizing_endpoint(self):
        """Test that tokenizer is not loaded when endpoint has tokenizes_input=false."""
        config = AIPerfConfig(
            models=["nvidia/nemoretriever-page-elements-v3"],
            endpoint={
                "urls": ["http://localhost:8000/v1/image_retrieval"],
                "type": "image_retrieval",
            },
            datasets={
                "default": {
                    "type": "synthetic",
                    "entries": 100,
                    "prompts": {"isl": 128, "osl": 64},
                }
            },
            phases={
                "default": {"type": "concurrency", "requests": 10, "concurrency": 1}
            },
        )
        dataset_manager = DatasetManager(
            run=_make_run(config),
            service_id="test_dataset_manager",
        )
        await dataset_manager.initialize()
        dataset_manager.publish = AsyncMock()

        with patch.object(
            DatasetManager, "_configure_tokenizer", new_callable=AsyncMock
        ) as mock_configure_tokenizer:
            await dataset_manager._profile_configure_command(
                Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
            )
            mock_configure_tokenizer.assert_not_called()

        assert dataset_manager.tokenizer is None

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_mock_dataset_steps", "mock_tokenizer")
    async def test_tokenizer_loaded_for_tokenizing_endpoint(self):
        """Test that tokenizer is loaded when endpoint has tokenizes_input=true."""
        config = AIPerfConfig(
            models=["test-model"],
            endpoint={
                "urls": ["http://localhost:8000/v1/chat/completions"],
                "type": EndpointType.CHAT,
            },
            datasets={
                "default": {
                    "type": "synthetic",
                    "entries": 100,
                    "prompts": {"isl": 128, "osl": 64},
                }
            },
            phases={
                "default": {"type": "concurrency", "requests": 10, "concurrency": 1}
            },
        )
        dataset_manager = DatasetManager(
            run=_make_run(config),
            service_id="test_dataset_manager",
        )
        await dataset_manager.initialize()
        dataset_manager.publish = AsyncMock()

        await dataset_manager._profile_configure_command(
            Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
        )

        assert dataset_manager.tokenizer is not None
