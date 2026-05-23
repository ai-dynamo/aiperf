# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from aiperf.common.models import Conversation
from aiperf.dataset.loader import ShareGPTLoader
from aiperf.plugin.enums import DatasetSamplingStrategy
from tests.unit.conftest import make_run_from_cli


@pytest.mark.asyncio
class TestShareGPTLoader:
    """Test suite for ShareGPTLoader class"""

    @pytest.fixture
    async def sharegpt_loader(self, cli_config, mock_tokenizer_cls):
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        return ShareGPTLoader(run=make_run_from_cli(cli_config), tokenizer=tokenizer)

    async def test_initialization(self, sharegpt_loader: ShareGPTLoader):
        """Test initialization of ShareGPTLoader"""
        assert sharegpt_loader.tokenizer is not None
        assert sharegpt_loader.run is not None
        assert sharegpt_loader.turn_count == 0
        assert sharegpt_loader.tag == "ShareGPT"
        assert (
            sharegpt_loader.url
            == "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
        )
        assert sharegpt_loader.filename == "ShareGPT_V3_unfiltered_cleaned_split.json"
        assert isinstance(sharegpt_loader.cache_filepath, Path)

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
        conversations = await sharegpt_loader.convert_to_conversations(dataset)

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
        conversations = await sharegpt_loader.convert_to_conversations(dataset)

        assert len(conversations) == 1
        assert isinstance(conversations[0], Conversation)

        turn = conversations[0].turns[0]
        assert turn.texts[0].contents[0] == "Hello how are you"
        assert turn.max_tokens == len(["This", "is", "test", "output"])
        assert turn.model == "test-model"

    async def test_convert_multi_turn_sharegpt_roles(
        self, sharegpt_loader: ShareGPTLoader
    ):
        """Entries with human/gpt roles beyond the first pair become extra turns."""
        dataset = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello how are you"},
                    {"from": "gpt", "value": "This is test output"},
                    {"from": "human", "value": "Follow up question here"},
                    {"from": "gpt", "value": "Second assistant reply text"},
                ]
            }
        ]
        conversations = await sharegpt_loader.convert_to_conversations(dataset)

        assert len(conversations) == 1
        conv = conversations[0]
        assert len(conv.turns) == 2
        assert conv.turns[0].texts[0].contents[0] == "Hello how are you"
        assert conv.turns[0].max_tokens == len(["This", "is", "test", "output"])
        assert conv.turns[1].texts[0].contents[0] == "Follow up question here"
        assert conv.turns[1].max_tokens == len(["Second", "assistant", "reply", "text"])

    async def test_skips_pair_with_missing_value_key(
        self, sharegpt_loader: ShareGPTLoader
    ):
        """Pairs where a message lacks the 'value' key are silently skipped."""
        dataset = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello how are you"},
                    {"from": "gpt"},
                    {"from": "human", "value": "Follow up question here"},
                    {"from": "gpt", "value": "Second assistant reply text"},
                ]
            }
        ]
        conversations = await sharegpt_loader.convert_to_conversations(dataset)
        assert len(conversations) == 1
        assert len(conversations[0].turns) == 1
        assert (
            conversations[0].turns[0].texts[0].contents[0] == "Follow up question here"
        )

    async def test_get_preferred_sampling_strategy(
        self, sharegpt_loader: ShareGPTLoader
    ):
        """Test that ShareGPTLoader returns the correct preferred sampling strategy."""
        strategy = ShareGPTLoader.get_preferred_sampling_strategy()
        assert strategy == DatasetSamplingStrategy.SEQUENTIAL

    async def test_skips_row_with_non_dict_messages_without_crashing(
        self, sharegpt_loader: ShareGPTLoader
    ):
        """A malformed row whose 'conversations' list contains non-dict items
        (strings, None, ints) must be skipped — not abort the whole load via
        AttributeError on the next .get() call. Also covers the case where a
        row has one valid dict + non-dict noise (fewer than two usable dicts)."""
        dataset = [
            {"conversations": ["raw_string", None, 42]},
            {"conversations": [None, {"from": "gpt", "value": "orphan reply text"}]},
            {
                "conversations": [
                    {"from": "human", "value": "Hello can you help me with a question"},
                    {"from": "gpt", "value": "Yes I can help you with that question"},
                ]
            },
        ]
        conversations = await sharegpt_loader.convert_to_conversations(dataset)
        assert len(conversations) == 1
        assert (
            conversations[0].turns[0].texts[0].contents[0]
            == "Hello can you help me with a question"
        )
