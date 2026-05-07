# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.config import (
    ConversationConfig,
    EndpointConfig,
    InputConfig,
    InputTokensConfig,
    PromptConfig,
    UserConfig,
)
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.composer.public import PublicDatasetComposer
from aiperf.plugin.enums import DatasetSamplingStrategy, PublicDatasetType


@pytest.fixture
def user_config() -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"]),
        input=InputConfig(
            conversation=ConversationConfig(num_dataset_entries=5),
            prompt=PromptConfig(input_tokens=InputTokensConfig(mean=10, stddev=2)),
        ),
    )


@pytest.fixture
def aimo_config(user_config: UserConfig) -> UserConfig:
    user_config.input.public_dataset = PublicDatasetType.AIMO
    return user_config


def _make_conversations(n: int = 2) -> list[Conversation]:
    return [
        Conversation(
            session_id=f"conv-{i}",
            turns=[Turn(texts=[Text(contents=[f"What is {i} + {i}?"])])],
        )
        for i in range(n)
    ]


class TestPublicDatasetComposerInit:
    def test_stores_tokenizer(self, aimo_config, mock_tokenizer_cls):
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        composer = PublicDatasetComposer(aimo_config, tokenizer)
        assert composer.tokenizer is tokenizer

    def test_stores_config(self, aimo_config):
        composer = PublicDatasetComposer(aimo_config, None)
        assert composer.config is aimo_config

    def test_create_dataset_raises(self, aimo_config):
        composer = PublicDatasetComposer(aimo_config, None)
        with pytest.raises(NotImplementedError):
            composer.create_dataset()


class TestSetSamplingStrategy:
    def test_sets_strategy_when_not_configured(self, aimo_config):
        aimo_config.input.dataset_sampling_strategy = None
        composer = PublicDatasetComposer(aimo_config, None)

        mock_loader_class = MagicMock()
        mock_loader_class.get_preferred_sampling_strategy.return_value = (
            DatasetSamplingStrategy.SEQUENTIAL
        )

        composer._set_sampling_strategy(PublicDatasetType.AIMO, mock_loader_class)

        assert (
            aimo_config.input.dataset_sampling_strategy
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    def test_does_not_override_user_strategy(self, aimo_config):
        aimo_config.input.dataset_sampling_strategy = DatasetSamplingStrategy.RANDOM
        composer = PublicDatasetComposer(aimo_config, None)

        mock_loader_class = MagicMock()
        mock_loader_class.get_preferred_sampling_strategy.return_value = (
            DatasetSamplingStrategy.SEQUENTIAL
        )

        composer._set_sampling_strategy(PublicDatasetType.AIMO, mock_loader_class)

        assert (
            aimo_config.input.dataset_sampling_strategy
            == DatasetSamplingStrategy.RANDOM
        )
        mock_loader_class.get_preferred_sampling_strategy.assert_not_called()


class TestBuildLoaderKwargs:
    def test_hf_kwargs_populated_from_metadata(self, aimo_config):
        composer = PublicDatasetComposer(aimo_config, None)
        kwargs = composer._build_loader_kwargs(PublicDatasetType.AIMO)

        assert kwargs["hf_dataset_name"] == "AI-MO/NuminaMath-TIR"
        assert kwargs["hf_split"] == "train"
        assert kwargs["prompt_column"] == "problem"

    def test_no_subset_when_metadata_lacks_it(self, aimo_config):
        composer = PublicDatasetComposer(aimo_config, None)
        kwargs = composer._build_loader_kwargs(PublicDatasetType.AIMO)
        assert "hf_subset" not in kwargs

    def test_no_kwargs_when_no_hf_metadata(self, aimo_config):
        """Loaders without HF metadata (e.g. ShareGPT) receive no unexpected kwargs."""
        from aiperf.plugin.schema.schemas import PublicDatasetLoaderMetadata

        composer = PublicDatasetComposer(aimo_config, None)
        with patch(
            "aiperf.dataset.composer.public.plugins.get_public_dataset_loader_metadata",
            return_value=PublicDatasetLoaderMetadata(),
        ):
            kwargs = composer._build_loader_kwargs(PublicDatasetType.AIMO)
        assert kwargs == {}


@pytest.mark.asyncio
class TestCreateDatasetAsync:
    async def test_returns_conversations_with_finalized_turns(self, aimo_config):
        conversations = _make_conversations(3)
        mock_loader = AsyncMock()
        mock_loader.load_dataset = AsyncMock(return_value={"dataset": []})
        mock_loader.convert_to_conversations = AsyncMock(return_value=conversations)

        mock_loader_class = MagicMock()
        mock_loader_class.get_preferred_sampling_strategy.return_value = (
            DatasetSamplingStrategy.SEQUENTIAL
        )
        mock_loader_class.return_value = mock_loader

        composer = PublicDatasetComposer(aimo_config, None)
        with (
            patch(
                "aiperf.dataset.composer.public.plugins.get_class",
                return_value=mock_loader_class,
            ),
            patch(
                "aiperf.dataset.composer.public.plugins.get_public_dataset_loader_metadata",
                return_value=MagicMock(
                    hf_dataset_name="test/dataset",
                    hf_split="train",
                    hf_subset=None,
                    prompt_column="problem",
                ),
            ),
        ):
            result = await composer.create_dataset_async()

        assert len(result) == 3
        assert all(isinstance(c, Conversation) for c in result)
        # _finalize_turn sets model name on each turn
        for conv in result:
            for turn in conv.turns:
                assert turn.model == "test-model"

    async def test_sets_sampling_strategy_from_loader(self, aimo_config):
        aimo_config.input.dataset_sampling_strategy = None
        conversations = _make_conversations(1)
        mock_loader = AsyncMock()
        mock_loader.load_dataset = AsyncMock(return_value={"dataset": []})
        mock_loader.convert_to_conversations = AsyncMock(return_value=conversations)

        mock_loader_class = MagicMock()
        mock_loader_class.get_preferred_sampling_strategy.return_value = (
            DatasetSamplingStrategy.SEQUENTIAL
        )
        mock_loader_class.return_value = mock_loader

        composer = PublicDatasetComposer(aimo_config, None)
        with (
            patch(
                "aiperf.dataset.composer.public.plugins.get_class",
                return_value=mock_loader_class,
            ),
            patch(
                "aiperf.dataset.composer.public.plugins.get_public_dataset_loader_metadata",
                return_value=MagicMock(
                    hf_dataset_name="test/dataset",
                    hf_split="train",
                    hf_subset=None,
                    prompt_column="problem",
                ),
            ),
        ):
            await composer.create_dataset_async()

        assert (
            aimo_config.input.dataset_sampling_strategy
            == DatasetSamplingStrategy.SEQUENTIAL
        )
