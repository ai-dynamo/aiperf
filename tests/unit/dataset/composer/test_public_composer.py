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
from aiperf.common.enums import PromptCorpus
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.composer.public import PublicDatasetComposer
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from aiperf.plugin.enums import DatasetSamplingStrategy, PublicDatasetType
from aiperf.plugin.schema.schemas import PublicDatasetLoaderMetadata


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

    def test_category_forwarded_when_set(self, aimo_config):
        from aiperf.plugin.schema.schemas import PublicDatasetLoaderMetadata

        composer = PublicDatasetComposer(aimo_config, None)
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

        composer = PublicDatasetComposer(aimo_config, None)
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
        from aiperf.plugin.schema.schemas import PublicDatasetLoaderMetadata

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
                return_value=PublicDatasetLoaderMetadata(
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
        from aiperf.plugin.schema.schemas import PublicDatasetLoaderMetadata

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
                return_value=PublicDatasetLoaderMetadata(
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


# ============================================================================
# Trace-loader kwarg injection (_inject_trace_kwargs)
# ============================================================================


def _trace_metadata(
    *,
    default_prompt_corpus: PromptCorpus = PromptCorpus.CODING,
    default_block_size: int | None = 64,
) -> PublicDatasetLoaderMetadata:
    """Build a PublicDatasetLoaderMetadata flagged as is_trace=True."""
    return PublicDatasetLoaderMetadata(
        hf_dataset_name="semianalysisai/cc-traces-weka-no-subagents-051226",
        hf_split="train",
        is_trace=True,
        default_block_size=default_block_size,
        default_prompt_corpus=default_prompt_corpus,
    )


class TestInjectTraceKwargs:
    """Verify the trace branch of ``_build_loader_kwargs``."""

    def test_raises_when_no_tokenizer(self, aimo_config: UserConfig) -> None:
        """Trace public datasets MUST have a tokenizer for prompt synthesis."""
        composer = PublicDatasetComposer(aimo_config, tokenizer=None)
        assert composer.prompt_generator is None
        kwargs: dict = {}
        with pytest.raises(
            ValueError, match="Trace public datasets require a tokenizer"
        ):
            composer._inject_trace_kwargs(_trace_metadata(), kwargs)

    def test_coding_corpus_uses_coding_content_generator(
        self, aimo_config: UserConfig, mock_tokenizer_cls
    ) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        composer = PublicDatasetComposer(aimo_config, tokenizer)
        kwargs: dict = {}

        composer._inject_trace_kwargs(
            _trace_metadata(default_prompt_corpus=PromptCorpus.CODING), kwargs
        )

        assert isinstance(kwargs["prompt_generator"], CodingContentGenerator)

    def test_sonnet_corpus_uses_composer_prompt_generator(
        self, aimo_config: UserConfig, mock_tokenizer_cls
    ) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        composer = PublicDatasetComposer(aimo_config, tokenizer)
        kwargs: dict = {}

        composer._inject_trace_kwargs(
            _trace_metadata(default_prompt_corpus=PromptCorpus.SONNET), kwargs
        )

        # Sonnet path reuses the composer's own prompt_generator,
        # not a CodingContentGenerator.
        assert kwargs["prompt_generator"] is composer.prompt_generator
        assert not isinstance(kwargs["prompt_generator"], CodingContentGenerator)

    def test_user_prompt_corpus_overrides_metadata_default(
        self, aimo_config: UserConfig, mock_tokenizer_cls
    ) -> None:
        """A user-set --prompt-corpus must win over the loader default."""
        aimo_config.input.prompt.prompt_corpus = PromptCorpus.SONNET
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        composer = PublicDatasetComposer(aimo_config, tokenizer)
        kwargs: dict = {}

        composer._inject_trace_kwargs(
            _trace_metadata(default_prompt_corpus=PromptCorpus.CODING), kwargs
        )

        # User picked sonnet => composer prompt_generator, NOT coding.
        assert kwargs["prompt_generator"] is composer.prompt_generator
        assert not isinstance(kwargs["prompt_generator"], CodingContentGenerator)

    def test_default_block_size_injected_when_set(
        self, aimo_config: UserConfig, mock_tokenizer_cls
    ) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        composer = PublicDatasetComposer(aimo_config, tokenizer)
        kwargs: dict = {}

        composer._inject_trace_kwargs(_trace_metadata(default_block_size=64), kwargs)

        assert kwargs["default_block_size"] == 64

    def test_default_block_size_omitted_when_unset(
        self, aimo_config: UserConfig, mock_tokenizer_cls
    ) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        composer = PublicDatasetComposer(aimo_config, tokenizer)
        kwargs: dict = {}

        composer._inject_trace_kwargs(_trace_metadata(default_block_size=None), kwargs)

        assert "default_block_size" not in kwargs


class TestBuildLoaderKwargsTraceBranch:
    """Verify _build_loader_kwargs wires the trace branch end-to-end."""

    def test_non_trace_metadata_does_not_inject_trace_kwargs(
        self, aimo_config: UserConfig
    ) -> None:
        """Non-trace loaders (sharegpt, aimo style) must NOT receive
        ``prompt_generator`` or ``default_block_size`` kwargs."""
        composer = PublicDatasetComposer(aimo_config, tokenizer=None)
        with patch(
            "aiperf.dataset.composer.public.plugins.get_public_dataset_loader_metadata",
            return_value=PublicDatasetLoaderMetadata(
                hf_dataset_name="AI-MO/NuminaMath-TIR",
                hf_split="train",
                prompt_column="problem",
                is_trace=False,
            ),
        ):
            kwargs = composer._build_loader_kwargs(PublicDatasetType.AIMO)

        assert "prompt_generator" not in kwargs
        assert "default_block_size" not in kwargs

    def test_trace_metadata_injects_prompt_generator_and_block_size(
        self, aimo_config: UserConfig, mock_tokenizer_cls
    ) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        composer = PublicDatasetComposer(aimo_config, tokenizer)
        with patch(
            "aiperf.dataset.composer.public.plugins.get_public_dataset_loader_metadata",
            return_value=_trace_metadata(),
        ):
            kwargs = composer._build_loader_kwargs(PublicDatasetType.AIMO)

        assert "prompt_generator" in kwargs
        assert isinstance(kwargs["prompt_generator"], CodingContentGenerator)
        assert kwargs["default_block_size"] == 64

    def test_trace_metadata_without_tokenizer_raises(
        self, aimo_config: UserConfig
    ) -> None:
        composer = PublicDatasetComposer(aimo_config, tokenizer=None)
        with (
            patch(
                "aiperf.dataset.composer.public.plugins.get_public_dataset_loader_metadata",
                return_value=_trace_metadata(),
            ),
            pytest.raises(
                ValueError, match="Trace public datasets require a tokenizer"
            ),
        ):
            composer._build_loader_kwargs(PublicDatasetType.AIMO)


class TestHFWekaRepoOverride:
    """Verify the Weka-only HuggingFace repo override."""

    def test_weka_hf_requires_hf_weka_repo(
        self, aimo_config: UserConfig, mock_tokenizer_cls
    ) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        aimo_config.input.public_dataset = PublicDatasetType.WEKA_HF
        aimo_config.input.hf_weka_repo = None
        composer = PublicDatasetComposer(aimo_config, tokenizer)

        with pytest.raises(
            ValueError,
            match="--public-dataset weka_hf requires --hf-weka-repo",
        ):
            composer._build_loader_kwargs(PublicDatasetType.WEKA_HF)

    @pytest.mark.parametrize("hf_weka_repo", ["", "   "])
    def test_weka_hf_rejects_blank_hf_weka_repo(
        self, aimo_config: UserConfig, mock_tokenizer_cls, hf_weka_repo: str
    ) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        aimo_config.input.public_dataset = PublicDatasetType.WEKA_HF
        aimo_config.input.hf_weka_repo = hf_weka_repo
        composer = PublicDatasetComposer(aimo_config, tokenizer)

        with pytest.raises(
            ValueError,
            match="--hf-weka-repo must be a non-empty HuggingFace dataset repo",
        ):
            composer._build_loader_kwargs(PublicDatasetType.WEKA_HF)

    def test_weka_hf_uses_hf_weka_repo_as_dataset_name(
        self, aimo_config: UserConfig, mock_tokenizer_cls
    ) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        aimo_config.input.public_dataset = PublicDatasetType.WEKA_HF
        aimo_config.input.hf_weka_repo = "semianalysisai/cc-traces-weka-new"
        composer = PublicDatasetComposer(aimo_config, tokenizer)

        kwargs = composer._build_loader_kwargs(PublicDatasetType.WEKA_HF)

        assert kwargs["hf_dataset_name"] == "semianalysisai/cc-traces-weka-new"
        assert kwargs["hf_split"] == "train"
        assert kwargs["default_block_size"] == 64
        assert "prompt_generator" in kwargs

    def test_hf_weka_repo_rejected_for_non_weka_hf_public_dataset(
        self, aimo_config: UserConfig
    ) -> None:
        aimo_config.input.public_dataset = PublicDatasetType.AIMO
        aimo_config.input.hf_weka_repo = "semianalysisai/cc-traces-weka-new"
        composer = PublicDatasetComposer(aimo_config, tokenizer=None)

        with pytest.raises(
            ValueError,
            match="--hf-weka-repo can only be used with --public-dataset weka_hf",
        ):
            composer._build_loader_kwargs(PublicDatasetType.AIMO)

    def test_hf_weka_repo_rejected_for_pinned_weka_alias(
        self, aimo_config: UserConfig, mock_tokenizer_cls
    ) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        aimo_config.input.public_dataset = (
            PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_NO_SUBAGENTS
        )
        aimo_config.input.hf_weka_repo = (
            "semianalysisai/cc-traces-weka-with-subagents-new"
        )
        composer = PublicDatasetComposer(aimo_config, tokenizer)

        with pytest.raises(
            ValueError,
            match="--hf-weka-repo can only be used with --public-dataset weka_hf",
        ):
            composer._build_loader_kwargs(
                PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_NO_SUBAGENTS
            )

    def test_pinned_weka_alias_keeps_metadata_dataset_name(
        self, aimo_config: UserConfig, mock_tokenizer_cls
    ) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        aimo_config.input.public_dataset = (
            PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS
        )
        aimo_config.input.hf_weka_repo = None
        composer = PublicDatasetComposer(aimo_config, tokenizer)

        kwargs = composer._build_loader_kwargs(
            PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS
        )

        assert (
            kwargs["hf_dataset_name"]
            == "semianalysisai/cc-traces-weka-with-subagents-051926"
        )
