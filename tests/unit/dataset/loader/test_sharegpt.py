# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import warnings

import pytest

from aiperf.common.models import Conversation
from aiperf.dataset.loader import ShareGPTLoader
from aiperf.dataset.loader.base_public_dataset import BasePublicDatasetLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


@pytest.mark.asyncio
class TestShareGPTLoader:
    """Test suite for ShareGPTLoader class"""

    @pytest.fixture
    async def sharegpt_loader(self, user_config, mock_tokenizer_cls, tmp_path):
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        filename = tmp_path / "sharegpt.json"
        return ShareGPTLoader(user_config, tokenizer, str(filename))

    async def test_initialization(self, sharegpt_loader: ShareGPTLoader):
        """Test initialization of ShareGPTLoader"""
        assert sharegpt_loader.tokenizer is not None
        assert sharegpt_loader.user_config is not None
        assert sharegpt_loader.turn_count == 0
        assert sharegpt_loader.tag == "ShareGPT"

    async def test_convert_to_conversations(self, sharegpt_loader: ShareGPTLoader):
        """Test converting single entry dataset to conversations"""
        dataset = [
            {
                "conversations": [
                    {"value": "Hello how are you"},
                    {"value": "This is test output"},
                ]
            }
        ]
        conversations = sharegpt_loader.convert_to_conversations(dataset)

        assert len(conversations) == 1
        assert isinstance(conversations[0], Conversation)

        turn = conversations[0].turns[0]
        assert turn.texts[0].contents[0] == "Hello how are you"
        assert turn.max_tokens == len(["This", "is", "test", "output"])
        assert turn.model == "test-model"

    async def test_convert_to_conversations_validation(
        self, sharegpt_loader: ShareGPTLoader
    ):
        """Test converting multiple entries dataset to conversations with validation"""

        dataset = [
            {
                "conversations": [
                    {"value": "Hello"},  # 1 prompt token (too short)
                    {"value": "This is test output"},  # 4 completion tokens
                ]
            },
            {
                "conversations": [
                    {"value": "Hello how are you"},  # 4 prompt tokens
                    {"value": "This is test output"},  # 4 completion tokens
                ]
            },
            {
                "conversations": [
                    {"value": "Hello how are you"},  # 4 prompt tokens
                    {"value": "This"},  # 1 completion tokens (too short)
                ]
            },
        ]
        conversations = sharegpt_loader.convert_to_conversations(dataset)

        assert len(conversations) == 1
        assert isinstance(conversations[0], Conversation)

        turn = conversations[0].turns[0]
        assert turn.texts[0].contents[0] == "Hello how are you"
        assert turn.max_tokens == len(["This", "is", "test", "output"])
        assert turn.model == "test-model"

    async def test_load_dataset(self, user_config, mock_tokenizer_cls, tmp_path):
        """Test that load_dataset reads and parses JSON from the given file."""
        data = [{"conversations": [{"value": "Hello"}, {"value": "World"}]}]
        filename = tmp_path / "sharegpt.json"
        filename.write_text(json.dumps(data))

        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = ShareGPTLoader(user_config, tokenizer, str(filename))

        result = loader.load_dataset()
        assert result == data

    async def test_get_preferred_sampling_strategy(self):
        """Test that ShareGPTLoader returns the correct preferred sampling strategy."""
        strategy = ShareGPTLoader.get_preferred_sampling_strategy()
        assert strategy == DatasetSamplingStrategy.SEQUENTIAL


class TestBasePublicDatasetLoaderDeprecation:
    """Test that BasePublicDatasetLoader emits a deprecation warning on instantiation."""

    @pytest.mark.asyncio
    async def test_deprecation_warning_on_init(self, user_config):
        class ConcreteLoader(BasePublicDatasetLoader):
            tag = "Test"
            url = "http://example.com/data.json"
            filename = "data.json"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ConcreteLoader(user_config=user_config)

        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        assert any("BaseFileLoader" in str(w.message) for w in caught)
