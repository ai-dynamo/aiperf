# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.models import Conversation, Text, Turn
from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.dataset.composer.public import PublicDatasetComposer
from aiperf.plugin.enums import PublicDatasetType  # dynamic enum (has AIMO etc.)

_MINIMAL_CONFIG_KWARGS: dict[str, Any] = {
    "models": ["test-model"],
    "endpoint": {
        "type": "chat",
        "urls": ["http://localhost:8000/v1/chat/completions"],
    },
    "datasets": {
        "default": {
            "type": "public",
            "name": "sharegpt",
        }
    },
    "phases": {"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
}


def _make_config(**overrides: Any) -> BenchmarkConfig:
    kwargs = {**_MINIMAL_CONFIG_KWARGS, **overrides}
    return BenchmarkConfig(**kwargs)


def _make_run(**overrides: Any) -> BenchmarkRun:
    config = _make_config(**overrides)
    return BenchmarkRun(
        benchmark_id="test",
        cfg=config,
        artifact_dir=Path("/tmp/test"),
    )


@pytest.fixture
def aimo_run() -> BenchmarkRun:
    return _make_run()


def _make_conversations(n: int = 2) -> list[Conversation]:
    return [
        Conversation(
            session_id=f"conv-{i}",
            turns=[Turn(texts=[Text(contents=[f"What is {i} + {i}?"])])],
        )
        for i in range(n)
    ]


class TestPublicDatasetComposerInit:
    def test_stores_tokenizer(self, aimo_run, mock_tokenizer_cls):
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        composer = PublicDatasetComposer(aimo_run, tokenizer)
        assert composer.tokenizer is tokenizer

    def test_stores_run(self, aimo_run):
        composer = PublicDatasetComposer(aimo_run, None)
        assert composer.run is aimo_run

    def test_create_dataset_raises(self, aimo_run):
        composer = PublicDatasetComposer(aimo_run, None)
        with pytest.raises(NotImplementedError):
            composer.create_dataset()


class TestBuildLoaderKwargs:
    def test_hf_kwargs_populated_from_metadata(self, aimo_run):
        composer = PublicDatasetComposer(aimo_run, None)
        kwargs = composer._build_loader_kwargs("aimo")

        assert kwargs["hf_dataset_name"] == "AI-MO/NuminaMath-TIR"
        assert kwargs["hf_split"] == "train"
        assert kwargs["prompt_column"] == "problem"

    def test_no_subset_when_metadata_lacks_it(self, aimo_run):
        composer = PublicDatasetComposer(aimo_run, None)
        kwargs = composer._build_loader_kwargs("aimo")
        assert "hf_subset" not in kwargs

    def test_no_kwargs_when_no_hf_metadata(self, aimo_run):
        """Loaders without HF metadata (e.g. ShareGPT) receive no unexpected kwargs."""
        from aiperf.plugin.schema.schemas import PublicDatasetLoaderMetadata

        composer = PublicDatasetComposer(aimo_run, None)
        with patch(
            "aiperf.dataset.composer.public.plugins.get_public_dataset_loader_metadata",
            return_value=PublicDatasetLoaderMetadata(),
        ):
            kwargs = composer._build_loader_kwargs("aimo")
        assert kwargs == {}

    def test_category_forwarded_when_set(self, aimo_run):
        from aiperf.plugin.schema.schemas import PublicDatasetLoaderMetadata

        composer = PublicDatasetComposer(aimo_run, None)
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

    def test_no_category_in_kwargs_when_none(self, aimo_run):
        from aiperf.plugin.schema.schemas import PublicDatasetLoaderMetadata

        composer = PublicDatasetComposer(aimo_run, None)
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
    async def test_returns_conversations_with_finalized_turns(self, aimo_run):
        conversations = _make_conversations(3)
        mock_loader = AsyncMock()
        mock_loader.load_dataset = AsyncMock(return_value={"dataset": []})
        mock_loader.convert_to_conversations = AsyncMock(return_value=conversations)

        mock_loader_class = MagicMock()
        mock_loader_class.return_value = mock_loader

        composer = PublicDatasetComposer(aimo_run, None)
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
        for conv in result:
            for turn in conv.turns:
                assert turn.model == "test-model"
