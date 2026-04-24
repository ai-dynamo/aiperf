# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from aiperf.common.models import Conversation
from aiperf.config import BenchmarkRun
from aiperf.dataset.loader import ShareGPTLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


@pytest.mark.asyncio
class TestShareGPTLoader:
    """Test suite for ShareGPTLoader class"""

    @pytest.fixture
    async def sharegpt_loader(self, user_config, mock_tokenizer_cls):
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        run = BenchmarkRun(
            benchmark_id="test",
            cfg=user_config,
            artifact_dir=Path("/tmp/test"),
        )
        return ShareGPTLoader(run, tokenizer)

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
        """Test converting multiple entries dataset to conversations with validation.

        Entry 1 (short prompt "Hello" = 1 token) is always filtered.
        Entry 2 (4 prompt tokens, 4 completion tokens) always passes.
        Entry 3 (4 prompt tokens, 1 completion token) passes because
        output_tokens_mean is set (osl=64), which skips the min output length check.
        """

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
                    {
                        "value": "This"
                    },  # 1 completion token (passes: osl skips min check)
                ]
            },
        ]
        conversations = await sharegpt_loader.convert_to_conversations(dataset)

        assert len(conversations) == 2

        turn0 = conversations[0].turns[0]
        assert turn0.texts[0].contents[0] == "Hello how are you"
        assert turn0.max_tokens == len(["This", "is", "test", "output"])
        assert turn0.model == "test-model"

        turn1 = conversations[1].turns[0]
        assert turn1.texts[0].contents[0] == "Hello how are you"
        assert turn1.max_tokens == len(["This"])
        assert turn1.model == "test-model"

    async def test_get_preferred_sampling_strategy(
        self, sharegpt_loader: ShareGPTLoader
    ):
        """Test that ShareGPTLoader returns the correct preferred sampling strategy."""
        strategy = ShareGPTLoader.get_preferred_sampling_strategy()
        assert strategy == DatasetSamplingStrategy.SEQUENTIAL
