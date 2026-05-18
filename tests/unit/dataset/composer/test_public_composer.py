# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.models import Conversation, Text, Turn
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.composer.public import PublicDatasetComposer
from aiperf.plugin.enums import DatasetSamplingStrategy, PublicDatasetType
from tests.unit.dataset.composer.conftest import make_run


@pytest.fixture
def cli_config() -> CLIConfig:
    # Public datasets do not accept synthetic prompt config in v2; keep input
    # minimal so the v1->v2 resolver doesn't try to attach prompts to a
    # PublicDataset.
    return CLIConfig(
        model_names=["test-model"],
        conversation_num_dataset_entries=5,
    )


@pytest.fixture
def aimo_config(cli_config: CLIConfig) -> CLIConfig:
    cli_config.public_dataset = PublicDatasetType.AIMO
    return cli_config


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
        composer = PublicDatasetComposer(run=make_run(aimo_config), tokenizer=tokenizer)
        assert composer.tokenizer is tokenizer

    def test_create_dataset_raises(self, aimo_config):
        composer = PublicDatasetComposer(run=make_run(aimo_config), tokenizer=None)
        with pytest.raises(NotImplementedError):
            composer.create_dataset()


class TestBuildLoaderKwargs:
    def test_hf_kwargs_populated_from_metadata(self, aimo_config):
        composer = PublicDatasetComposer(run=make_run(aimo_config), tokenizer=None)
        kwargs = composer._build_loader_kwargs(PublicDatasetType.AIMO)

        assert kwargs["hf_dataset_name"] == "AI-MO/NuminaMath-TIR"
        assert kwargs["hf_split"] == "train"
        assert kwargs["prompt_column"] == "problem"

    def test_no_subset_when_metadata_lacks_it(self, aimo_config):
        composer = PublicDatasetComposer(run=make_run(aimo_config), tokenizer=None)
        kwargs = composer._build_loader_kwargs(PublicDatasetType.AIMO)
        assert "hf_subset" not in kwargs

    def test_no_kwargs_when_no_hf_metadata(self, aimo_config):
        """Loaders without HF metadata (e.g. ShareGPT) receive no unexpected kwargs."""
        from aiperf.plugin.schema.schemas import PublicDatasetLoaderMetadata

        composer = PublicDatasetComposer(run=make_run(aimo_config), tokenizer=None)
        with patch(
            "aiperf.dataset.composer.public.plugins.get_public_dataset_loader_metadata",
            return_value=PublicDatasetLoaderMetadata(),
        ):
            kwargs = composer._build_loader_kwargs(PublicDatasetType.AIMO)
        assert kwargs == {}

    def test_category_forwarded_when_set(self, aimo_config):
        from aiperf.plugin.schema.schemas import PublicDatasetLoaderMetadata

        composer = PublicDatasetComposer(run=make_run(aimo_config), tokenizer=None)
        with patch(
            "aiperf.dataset.composer.public.plugins.get_public_dataset_loader_metadata",
            return_value=PublicDatasetLoaderMetadata(
                hf_dataset_name="nvidia/SPEED-Bench",
                hf_split="test",
                hf_subset="qualitative",
                category="coding",
            ),
        ):
            kwargs = composer._build_loader_kwargs(PublicDatasetType.AIMO)
        assert kwargs["category"] == "coding"

    def test_no_category_in_kwargs_when_none(self, aimo_config):
        from aiperf.plugin.schema.schemas import PublicDatasetLoaderMetadata

        composer = PublicDatasetComposer(run=make_run(aimo_config), tokenizer=None)
        with patch(
            "aiperf.dataset.composer.public.plugins.get_public_dataset_loader_metadata",
            return_value=PublicDatasetLoaderMetadata(
                hf_dataset_name="nvidia/SPEED-Bench",
                hf_split="test",
            ),
        ):
            kwargs = composer._build_loader_kwargs(PublicDatasetType.AIMO)
        assert "category" not in kwargs


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

        composer = PublicDatasetComposer(run=make_run(aimo_config), tokenizer=None)
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
        # The composer no longer mutates the v1 cli_config sampling strategy;
        # this assertion was a v1 reverse-flow artifact. Verify instead that
        # create_dataset_async runs to completion when the user did not
        # configure a sampling strategy.
        conversations = _make_conversations(1)
        mock_loader = AsyncMock()
        mock_loader.load_dataset = AsyncMock(return_value={"dataset": []})
        mock_loader.convert_to_conversations = AsyncMock(return_value=conversations)

        mock_loader_class = MagicMock()
        mock_loader_class.get_preferred_sampling_strategy.return_value = (
            DatasetSamplingStrategy.SEQUENTIAL
        )
        mock_loader_class.return_value = mock_loader

        composer = PublicDatasetComposer(run=make_run(aimo_config), tokenizer=None)
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

        assert len(result) == 1
