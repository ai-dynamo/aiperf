# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from aiperf.common.config import EndpointConfig, InputConfig, ServiceConfig, UserConfig
from aiperf.common.config.config_defaults import InputDefaults
from aiperf.common.config.tokenizer_config import TokenizerConfig
from aiperf.common.exceptions import ServiceError
from aiperf.common.messages import (
    ConversationRequestMessage,
    ConversationTurnRequestMessage,
    DatasetConfiguredNotification,
)
from aiperf.common.messages.command_messages import ProfileConfigureCommand
from aiperf.common.models import Conversation, Image, Text, Turn
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.plugin.enums import (
    CustomDatasetType,
    DatasetSamplingStrategy,
    PublicDatasetType,
    ServiceRunType,
)

# ============================================================================
# Shared Fixtures
# ============================================================================


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
def base_user_config():
    """Create a basic UserConfig for testing."""
    return UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"]),
        input=InputConfig(),
    )


@pytest.fixture
async def initialized_dataset_manager(mock_tokenizer, base_user_config):
    """Create an initialized DatasetManager with mocked publish."""
    service_config = ServiceConfig()
    dataset_manager = DatasetManager(service_config, base_user_config)

    await dataset_manager.initialize()
    dataset_manager.publish = AsyncMock()

    return dataset_manager


@pytest.fixture
async def configured_dataset_manager(initialized_dataset_manager, base_user_config):
    """Create a fully configured DatasetManager ready for request handling."""
    await initialized_dataset_manager._profile_configure_command(
        ProfileConfigureCommand(config=base_user_config, service_id="test_service")
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


async def capture_published_messages(dataset_manager, user_config):
    """Configure dataset and capture published messages."""
    published_messages = []

    async def mock_publish(msg):
        published_messages.append(msg)

    dataset_manager.publish = AsyncMock(side_effect=mock_publish)

    await dataset_manager._profile_configure_command(
        ProfileConfigureCommand(config=user_config, service_id="test_service")
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
        # Create a file with multi-turn conversations
        entries = [
            '{"session_id": "sess-1", "timestamp": 0, "input_length": 50, "output_length": 10}',
            '{"session_id": "sess-1", "delay": 10000, "input_length": 50, "output_length": 10}',
            '{"session_id": "sess-1", "delay": 10000, "input_length": 100, "output_length": 10}',
            '{"session_id": "sess-2", "timestamp": 20000, "input_length": 25, "output_length": 20}',
            '{"session_id": "sess-2", "delay": 10000, "input_length": 10000, "output_length": 20}',
        ]
        filename = create_mooncake_trace_file(entries)

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=filename, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )

            service_config = ServiceConfig()
            dataset_manager = DatasetManager(service_config, user_config)

            await dataset_manager.initialize()

            published_messages = await capture_published_messages(
                dataset_manager, user_config
            )

            # Verify the notification was published
            published_notifications = extract_dataset_notifications(published_messages)
            assert len(published_notifications) == 1

            notification = published_notifications[0]
            metadata = notification.metadata

            # Verify dataset metadata structure
            assert len(metadata.conversations) == 2  # 2 conversations, not 5 turns

            # Extract conversation metadata for easier testing
            conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

            # Verify session 1 metadata
            assert "sess-1" in conv_dict
            sess1 = conv_dict["sess-1"]
            assert len(sess1.turns) == 3

            # Verify session 2 metadata
            assert "sess-2" in conv_dict
            sess2 = conv_dict["sess-2"]
            assert len(sess2.turns) == 2

            # Verify no duplicate conversation IDs (one per conversation, not per turn)
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
        """Test that floating point timestamps are preserved exactly in dataset notifications.

        This test verifies that high-precision floating point timestamps from trace data
        are maintained throughout the dataset loading and notification process.
        """
        # Create a file with floating point timestamps (in milliseconds)
        entries = [
            '{"session_id": "sess-1", "timestamp": 0.123, "input_length": 50, "output_length": 10}',
            '{"session_id": "sess-1", "delay": 10000.456, "input_length": 50, "output_length": 10}',
            '{"session_id": "sess-2", "timestamp": 20000.789, "input_length": 25, "output_length": 20}',
            '{"session_id": "sess-2", "delay": 15000.123, "input_length": 100, "output_length": 20}',
        ]
        filename = create_mooncake_trace_file(entries)

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=filename, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )

            service_config = ServiceConfig()
            dataset_manager = DatasetManager(service_config, user_config)

            await dataset_manager.initialize()

            published_messages = await capture_published_messages(
                dataset_manager, user_config
            )

            # Verify the notification was published
            published_notifications = extract_dataset_notifications(published_messages)
            assert len(published_notifications) == 1

            notification = published_notifications[0]
            metadata = notification.metadata

            # Extract conversation metadata
            conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

            # Verify conversations are loaded correctly
            assert "sess-1" in conv_dict
            sess1 = conv_dict["sess-1"]
            assert len(sess1.turns) == 2

            assert "sess-2" in conv_dict
            sess2 = conv_dict["sess-2"]
            assert len(sess2.turns) == 2

        finally:
            Path(filename).unlink(missing_ok=True)


class TestDatasetManagerSamplingStrategyDefaults:
    """Test default sampling strategy behavior for different dataset types."""

    @pytest.mark.asyncio
    @patch("aiperf.dataset.loader.sharegpt.ShareGPTLoader.load_dataset")
    @patch("aiperf.dataset.loader.sharegpt.ShareGPTLoader.convert_to_conversations")
    async def test_public_dataset_uses_loader_recommended_strategy(
        self,
        mock_convert,
        mock_load,
        mock_tokenizer,
    ):
        """Test that public datasets use the loader's recommended sampling strategy."""
        # Mock dataset loading
        mock_load.return_value = {}
        mock_convert.return_value = create_mock_conversations(
            ["session-1", "session-2"]
        )

        # Create config with public dataset and NO explicit sampling strategy
        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(public_dataset=PublicDatasetType.SHAREGPT),
        )
        assert user_config.input.dataset_sampling_strategy is None

        service_config = ServiceConfig()
        dataset_manager = DatasetManager(service_config, user_config)

        await dataset_manager.initialize()
        await dataset_manager._profile_configure_command(
            ProfileConfigureCommand(config=user_config, service_id="test_service")
        )

        # Verify the loader's recommended strategy was used (SEQUENTIAL for ShareGPT)
        assert (
            user_config.input.dataset_sampling_strategy
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    @pytest.mark.asyncio
    async def test_fallback_default_when_strategy_not_set(
        self,
        mock_tokenizer,
    ):
        """Test that InputDefaults.DATASET_SAMPLING_STRATEGY is used as fallback."""
        # Create config with NO public dataset and NO explicit sampling strategy
        # This will use synthetic dataset generation
        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(),
        )

        service_config = ServiceConfig()
        dataset_manager = DatasetManager(service_config, user_config)

        await dataset_manager.initialize()
        await dataset_manager._profile_configure_command(
            ProfileConfigureCommand(config=user_config, service_id="test_service")
        )

        # Synthetic composer sets its own default, which should be the same as InputDefaults
        assert user_config.input.dataset_sampling_strategy is not None
        assert (
            user_config.input.dataset_sampling_strategy
            == InputDefaults.DATASET_SAMPLING_STRATEGY
        )

    @pytest.mark.asyncio
    @patch("aiperf.dataset.loader.sharegpt.ShareGPTLoader.load_dataset")
    @patch("aiperf.dataset.loader.sharegpt.ShareGPTLoader.convert_to_conversations")
    async def test_explicit_strategy_overrides_loader_recommendation(
        self,
        mock_convert,
        mock_load,
        mock_tokenizer,
    ):
        """Test that explicitly set strategy is not overridden by loader recommendation."""
        # Mock dataset loading
        mock_load.return_value = {}
        mock_convert.return_value = create_mock_conversations(["session-1"])

        # Create config with explicit SHUFFLE strategy (different from loader's SEQUENTIAL)
        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                public_dataset=PublicDatasetType.SHAREGPT,
                dataset_sampling_strategy=DatasetSamplingStrategy.SHUFFLE,
            ),
        )

        service_config = ServiceConfig()
        dataset_manager = DatasetManager(service_config, user_config)

        await dataset_manager.initialize()
        await dataset_manager._profile_configure_command(
            ProfileConfigureCommand(config=user_config, service_id="test_service")
        )

        # Verify the explicit strategy was preserved, not overwritten by loader's SEQUENTIAL
        assert (
            user_config.input.dataset_sampling_strategy
            == DatasetSamplingStrategy.SHUFFLE
        )


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

        # Before configuration, client should be None
        assert dataset_manager._dataset_client is None

        await dataset_manager._profile_configure_command(
            ProfileConfigureCommand(config=base_user_config, service_id="test_service")
        )

        # After configuration, client should be initialized
        assert dataset_manager._dataset_client is not None

    @pytest.mark.asyncio
    async def test_in_memory_dataset_freed_after_client_initialization(
        self,
        mock_tokenizer,
    ):
        """Test that in-memory dataset is freed after dataset client is initialized."""
        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(num_dataset_entries=5),
        )
        service_config = ServiceConfig()
        dataset_manager = DatasetManager(service_config, user_config)

        await dataset_manager.initialize()
        dataset_manager.publish = AsyncMock()

        await dataset_manager._profile_configure_command(
            ProfileConfigureCommand(config=user_config, service_id="test_service")
        )

        # After configuration, in-memory dataset should be empty
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

        # Before configuration, event should not be set
        assert not dataset_manager.dataset_configured.is_set()

        await dataset_manager._profile_configure_command(
            ProfileConfigureCommand(config=base_user_config, service_id="test_service")
        )

        # After configuration, event should be set
        assert dataset_manager.dataset_configured.is_set()


class TestDatasetManagerFallbackHandlers:
    """Test fallback request handlers that use the dataset client."""

    @pytest.fixture
    async def dataset_manager_with_entries(self, mock_tokenizer):
        """Create a configured dataset manager with multiple entries."""
        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(num_dataset_entries=3),
        )
        service_config = ServiceConfig()
        dataset_manager = DatasetManager(service_config, user_config)

        await dataset_manager.initialize()
        dataset_manager.publish = AsyncMock()

        await dataset_manager._profile_configure_command(
            ProfileConfigureCommand(config=user_config, service_id="test_service")
        )

        return dataset_manager

    @pytest.mark.asyncio
    async def test_handle_conversation_request_uses_dataset_client(
        self,
        dataset_manager_with_entries,
    ):
        """Test that conversation request handler uses dataset client, not in-memory dict."""
        dataset_manager = dataset_manager_with_entries

        # Get a valid conversation ID from the metadata
        conversation_id = dataset_manager.dataset_metadata.conversations[
            0
        ].conversation_id

        # Verify in-memory dataset is empty (freed)
        assert dataset_manager.dataset == {}

        # Request should still work via dataset client
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

        # Get a valid conversation ID from the metadata
        conversation_id = dataset_manager.dataset_metadata.conversations[
            0
        ].conversation_id

        # Verify in-memory dataset is empty (freed)
        assert dataset_manager.dataset == {}

        # Request should still work via dataset client
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
            turn_index=999,  # Invalid index
        )

        with pytest.raises(ServiceError, match="out of range"):
            await dataset_manager._handle_conversation_turn_request(request)


class TestKubernetesMode:
    """Test Kubernetes-specific behavior in DatasetManager."""

    def test_compress_only_kubernetes_returns_true(
        self, base_user_config: UserConfig
    ) -> None:
        """compress_only should be True when service_run_type is KUBERNETES."""
        service_config = ServiceConfig(service_run_type=ServiceRunType.KUBERNETES)
        manager = DatasetManager(service_config, base_user_config)
        assert manager._compress_only is True

    def test_compress_only_multiprocessing_returns_false(
        self, base_user_config: UserConfig
    ) -> None:
        """compress_only should be False in local (multiprocessing) mode."""
        service_config = ServiceConfig(service_run_type=ServiceRunType.MULTIPROCESSING)
        manager = DatasetManager(service_config, base_user_config)
        assert manager._compress_only is False

    @pytest.mark.asyncio
    async def test_configure_client_compress_only_skips_client_creation(
        self, base_user_config: UserConfig
    ) -> None:
        """In compress_only mode, _configure_dataset_client_and_free_memory skips client creation."""
        service_config = ServiceConfig(service_run_type=ServiceRunType.KUBERNETES)
        manager = DatasetManager(service_config, base_user_config)
        # Simulate some dataset entries
        manager.dataset = {"conv1": MagicMock(), "conv2": MagicMock()}
        manager._conversation_ids_cache = ["conv1", "conv2"]

        await manager._configure_dataset_client_and_free_memory()

        # Should have set dataset_configured event
        assert manager.dataset_configured.is_set()
        # Should have freed memory (cleared dataset)
        assert manager.dataset == {}
        assert manager._conversation_ids_cache == []
        # Should NOT have created a dataset client
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
        user_config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["nvidia/nemoretriever-page-elements-v3"],
                type="image_retrieval",
            ),
            input=InputConfig(),
        )
        service_config = ServiceConfig()
        dataset_manager = DatasetManager(service_config, user_config)
        await dataset_manager.initialize()
        dataset_manager.publish = AsyncMock()

        with patch.object(
            DatasetManager, "_configure_tokenizer", new_callable=AsyncMock
        ) as mock_configure_tokenizer:
            await dataset_manager._profile_configure_command(
                ProfileConfigureCommand(config=user_config, service_id="test_service")
            )
            mock_configure_tokenizer.assert_not_called()

        assert dataset_manager.tokenizer is None

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_mock_dataset_steps", "mock_tokenizer")
    async def test_tokenizer_loaded_for_tokenizing_endpoint(self):
        """Test that tokenizer is loaded when endpoint has tokenizes_input=true."""
        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"], type="chat"),
            input=InputConfig(),
        )
        service_config = ServiceConfig()
        dataset_manager = DatasetManager(service_config, user_config)
        await dataset_manager.initialize()
        dataset_manager.publish = AsyncMock()

        await dataset_manager._profile_configure_command(
            ProfileConfigureCommand(config=user_config, service_id="test_service")
        )

        assert dataset_manager.tokenizer is not None

    def test_tokenizer_rejected_when_explicitly_set_on_non_tokenizing_endpoint(self):
        """Tokenizer options are rejected for endpoints that don't tokenize input or produce tokens."""
        with pytest.raises(ValidationError, match="Tokenizer options cannot be used"):
            UserConfig(
                endpoint=EndpointConfig(
                    model_names=["nvidia/nemoretriever-page-elements-v3"],
                    type="image_retrieval",
                ),
                input=InputConfig(),
                tokenizer=TokenizerConfig(name="test-model"),
            )


# ============================================================================
# Media URL Inline Conversion Tests
# ============================================================================

# 1x1 red PNG image bytes
_TINY_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01"
    b"\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)


class TestConvertMediaUrlsToInline:
    """Tests for DatasetManager._convert_media_urls_to_inline."""

    @pytest.fixture
    async def dataset_manager(self, mock_tokenizer):
        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"], type="image_retrieval"),
            input=InputConfig(),
        )
        service_config = ServiceConfig()
        dm = DatasetManager(service_config, user_config)
        await dm.initialize()
        dm.publish = AsyncMock()
        return dm

    @pytest.mark.asyncio
    async def test_converts_http_urls_to_data_urls(self, dataset_manager):
        """HTTP image URLs are downloaded and replaced with base64 data URLs."""
        url = "https://example.com/image.png"
        dataset_manager.dataset = {
            "s1": Conversation(
                session_id="s1",
                turns=[Turn(images=[Image(contents=[url])], model="test-model")],
            )
        }

        with patch(
            "aiperf.dataset.dataset_manager.aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_content = AsyncMock()
            mock_content.read = AsyncMock(return_value=_TINY_PNG_BYTES)

            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.content_length = len(_TINY_PNG_BYTES)
            mock_resp.content = mock_content
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            await dataset_manager._convert_media_urls_to_inline()

        contents = dataset_manager.dataset["s1"].turns[0].images[0].contents
        assert len(contents) == 1
        assert contents[0].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_skips_already_encoded_data_urls(self, dataset_manager):
        """Already-encoded data URLs are left unchanged."""
        data_url = "data:image/png;base64,iVBORw0KGgo="
        dataset_manager.dataset = {
            "s1": Conversation(
                session_id="s1",
                turns=[Turn(images=[Image(contents=[data_url])], model="test-model")],
            )
        }

        await dataset_manager._convert_media_urls_to_inline()

        assert dataset_manager.dataset["s1"].turns[0].images[0].contents[0] == data_url

    @pytest.mark.asyncio
    async def test_deduplicates_same_url_across_turns(self, dataset_manager):
        """Same URL appearing in multiple turns is downloaded only once."""
        url = "https://example.com/image.png"
        dataset_manager.dataset = {
            "s1": Conversation(
                session_id="s1",
                turns=[
                    Turn(images=[Image(contents=[url])], model="test-model"),
                    Turn(images=[Image(contents=[url])], model="test-model"),
                ],
            )
        }

        with patch(
            "aiperf.dataset.dataset_manager.aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_content = AsyncMock()
            mock_content.read = AsyncMock(return_value=_TINY_PNG_BYTES)

            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.content_length = len(_TINY_PNG_BYTES)
            mock_resp.content = mock_content
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            await dataset_manager._convert_media_urls_to_inline()

        # Both turns should have the same data URL
        t0 = dataset_manager.dataset["s1"].turns[0].images[0].contents[0]
        t1 = dataset_manager.dataset["s1"].turns[1].images[0].contents[0]
        assert t0.startswith("data:image/png;base64,")
        assert t0 == t1
        # Only one GET request should have been made
        mock_session.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_on_download_failure(self, dataset_manager):
        """RuntimeError is raised when a URL download fails."""
        url = "https://example.com/missing.png"
        dataset_manager.dataset = {
            "s1": Conversation(
                session_id="s1",
                turns=[Turn(images=[Image(contents=[url])], model="test-model")],
            )
        }

        with patch(
            "aiperf.dataset.dataset_manager.aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_resp = AsyncMock()
            mock_resp.status = 404
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            with pytest.raises(RuntimeError, match="HTTP 404"):
                await dataset_manager._convert_media_urls_to_inline()

    @pytest.mark.asyncio
    async def test_noop_when_no_urls(self, dataset_manager):
        """No error or download when dataset has no HTTP URLs."""
        dataset_manager.dataset = {
            "s1": Conversation(
                session_id="s1",
                turns=[
                    Turn(
                        images=[Image(contents=["data:image/png;base64,abc"])],
                        model="test-model",
                    )
                ],
            )
        }

        # Should complete without any HTTP calls
        await dataset_manager._convert_media_urls_to_inline()


class TestConfigureDatasetInlineMediaGating:
    """Tests that endpoint metadata correctly declares requires_inline_media."""

    def test_image_retrieval_requires_inline_media(self):
        """image_retrieval endpoint metadata has requires_inline_media=True."""
        from aiperf.plugin import plugins

        meta = plugins.get_endpoint_metadata("image_retrieval")
        assert meta.requires_inline_media is True

    def test_chat_does_not_require_inline_media(self):
        """chat endpoint metadata has requires_inline_media=False."""
        from aiperf.plugin import plugins

        meta = plugins.get_endpoint_metadata("chat")
        assert meta.requires_inline_media is False
