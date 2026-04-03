# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, mock_open, patch

import pytest

from aiperf.common.config import SynthesisConfig
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.composer.custom import CustomDatasetComposer
from aiperf.dataset.loader import (
    MooncakeTraceDatasetLoader,
    MultiTurnDatasetLoader,
    RandomPoolDatasetLoader,
    SingleTurnDatasetLoader,
)
from aiperf.plugin.enums import CustomDatasetType, DatasetSamplingStrategy


class TestInitialization:
    """Test class for CustomDatasetComposer basic initialization."""

    def test_initialization(self, custom_config, mock_tokenizer):
        """Test that CustomDatasetComposer can be instantiated with valid config."""
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)

        assert composer is not None
        assert isinstance(composer, CustomDatasetComposer)

    def test_config_storage(self, custom_config, mock_tokenizer):
        """Test that the config is properly stored."""
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)

        input_config = composer.config.input
        assert input_config is custom_config.input
        assert input_config.file == "test_data.jsonl"
        assert input_config.custom_dataset_type == CustomDatasetType.SINGLE_TURN


MOCK_TRACE_CONTENT = """{"timestamp": 0, "input_length": 655, "output_length": 52, "hash_ids": [46, 47]}
{"timestamp": 10535, "input_length": 672, "output_length": 52, "hash_ids": [46, 47]}
{"timestamp": 27482, "input_length": 655, "output_length": 52, "hash_ids": [46, 47]}
"""


class TestCoreFunctionality:
    """Test class for CustomDatasetComposer core functionality."""

    @pytest.mark.parametrize(
        "dataset_type,expected_instance",
        [
            (CustomDatasetType.SINGLE_TURN, SingleTurnDatasetLoader),
            (CustomDatasetType.MULTI_TURN, MultiTurnDatasetLoader),
            (CustomDatasetType.RANDOM_POOL, RandomPoolDatasetLoader),
            (CustomDatasetType.MOONCAKE_TRACE, MooncakeTraceDatasetLoader),
        ],
    )
    def test_create_loader_instance_dataset_types(
        self, custom_config, dataset_type, expected_instance, mock_tokenizer
    ):
        """Test _create_loader_instance with different dataset types."""
        custom_config.input.custom_dataset_type = dataset_type
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)
        composer._create_loader_instance(dataset_type)
        assert isinstance(composer.loader, expected_instance)

    @patch("aiperf.dataset.loader.base_trace_loader.parallel_decode")
    @patch("aiperf.dataset.composer.custom.check_file_exists")
    @patch("builtins.open", mock_open(read_data=MOCK_TRACE_CONTENT))
    def test_create_dataset_trace(
        self, mock_check_file, mock_parallel_decode, trace_config, mock_tokenizer
    ):
        """Test that create_dataset returns correct type."""
        mock_parallel_decode.return_value = ["decoded 1", "decoded 2", "decoded 3"]
        composer = CustomDatasetComposer(trace_config, mock_tokenizer)
        conversations = composer.create_dataset()

        assert len(conversations) == 3
        assert all(isinstance(c, Conversation) for c in conversations)
        assert all(isinstance(turn, Turn) for c in conversations for turn in c.turns)
        assert all(len(turn.texts) == 1 for c in conversations for turn in c.turns)

    @patch("aiperf.dataset.loader.base_trace_loader.parallel_decode")
    @patch("aiperf.dataset.composer.custom.check_file_exists")
    @patch("builtins.open", mock_open(read_data=MOCK_TRACE_CONTENT))
    def test_max_tokens_config(
        self, mock_check_file, mock_parallel_decode, trace_config, mock_tokenizer
    ):
        mock_parallel_decode.return_value = ["decoded 1", "decoded 2", "decoded 3"]
        trace_config.input.prompt.output_tokens.mean = 120
        trace_config.input.prompt.output_tokens.stddev = 8.0

        composer = CustomDatasetComposer(trace_config, mock_tokenizer)
        conversations = composer.create_dataset()

        assert len(conversations) > 0
        # With global RNG, verify max_tokens is set to a positive integer
        # around the mean of 120
        for conversation in conversations:
            for turn in conversation.turns:
                assert turn.max_tokens is not None
                assert turn.max_tokens > 0
                assert isinstance(turn.max_tokens, int)
                # Should be roughly around the mean of 120 (within 3 stddev)
                assert 96 < turn.max_tokens < 144

    @patch("aiperf.dataset.loader.base_trace_loader.parallel_decode")
    @patch("aiperf.dataset.composer.custom.check_file_exists")
    @patch("builtins.open", mock_open(read_data=MOCK_TRACE_CONTENT))
    @patch("pathlib.Path.iterdir", return_value=[])
    def test_max_tokens_mooncake(
        self,
        mock_iterdir,
        mock_check_file,
        mock_parallel_decode,
        custom_config,
        mock_tokenizer,
    ):
        """Test that max_tokens can be set from the custom file"""
        mock_parallel_decode.return_value = ["decoded 1", "decoded 2", "decoded 3"]
        mock_check_file.return_value = None
        custom_config.input.custom_dataset_type = CustomDatasetType.MOONCAKE_TRACE

        composer = CustomDatasetComposer(custom_config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conversation in conversations:
            for turn in conversation.turns:
                assert turn.max_tokens == 52


class TestErrorHandling:
    """Test class for CustomDatasetComposer error handling scenarios."""

    @patch("aiperf.dataset.composer.custom.check_file_exists")
    @patch("aiperf.dataset.composer.custom.plugins.get_class")
    def test_create_dataset_empty_result(
        self, mock_get_class, mock_check_file, custom_config, mock_tokenizer
    ):
        """Test create_dataset when loader returns empty data."""
        mock_check_file.return_value = None
        mock_loader = Mock()
        mock_loader.load_dataset.return_value = {}
        mock_loader.convert_to_conversations.return_value = []
        # Create a mock class that has get_preferred_sampling_strategy and can be instantiated
        mock_loader_class = Mock()
        mock_loader_class.return_value = mock_loader
        mock_loader_class.get_preferred_sampling_strategy.return_value = (
            DatasetSamplingStrategy.SEQUENTIAL
        )
        mock_get_class.return_value = mock_loader_class

        composer = CustomDatasetComposer(custom_config, mock_tokenizer)
        result = composer.create_dataset()

        assert isinstance(result, list)
        assert len(result) == 0


class TestSamplingStrategy:
    """Test class for CustomDatasetComposer sampling strategy configuration."""

    @pytest.mark.parametrize(
        "dataset_type,expected_strategy",
        [
            (CustomDatasetType.SINGLE_TURN, DatasetSamplingStrategy.SEQUENTIAL),
            (CustomDatasetType.MULTI_TURN, DatasetSamplingStrategy.SEQUENTIAL),
            (CustomDatasetType.RANDOM_POOL, DatasetSamplingStrategy.SHUFFLE),
            (CustomDatasetType.MOONCAKE_TRACE, DatasetSamplingStrategy.SEQUENTIAL),
        ],
    )
    def test_set_sampling_strategy_when_none(
        self, custom_config, mock_tokenizer, dataset_type, expected_strategy
    ):
        """Test that _set_sampling_strategy sets the correct strategy when None."""
        custom_config.input.dataset_sampling_strategy = None
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)

        composer._set_sampling_strategy(dataset_type)

        assert composer.config.input.dataset_sampling_strategy == expected_strategy

    @pytest.mark.parametrize(
        "dataset_type",
        [
            CustomDatasetType.SINGLE_TURN,
            CustomDatasetType.MULTI_TURN,
            CustomDatasetType.RANDOM_POOL,
            CustomDatasetType.MOONCAKE_TRACE,
        ],
    )
    def test_set_sampling_strategy_does_not_override(
        self, custom_config, mock_tokenizer, dataset_type
    ):
        """Test that _set_sampling_strategy does not override explicitly set strategy."""
        explicit_strategy = DatasetSamplingStrategy.SHUFFLE
        custom_config.input.dataset_sampling_strategy = explicit_strategy
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)

        composer._set_sampling_strategy(dataset_type)

        assert composer.config.input.dataset_sampling_strategy == explicit_strategy


class TestSynthesisValidation:
    """Test class for synthesis configuration validation."""

    @pytest.mark.parametrize(
        "dataset_type",
        [
            CustomDatasetType.MOONCAKE_TRACE,
            CustomDatasetType.BAILIAN_TRACE,
        ],
    )
    def test_synthesis_allowed_with_trace_datasets(
        self, trace_config, mock_tokenizer, dataset_type
    ):
        """Test that synthesis options are allowed with trace dataset types."""
        trace_config.input.synthesis = SynthesisConfig(speedup_ratio=2.0)
        composer = CustomDatasetComposer(trace_config, mock_tokenizer)

        # Should not raise
        composer._validate_synthesis_config(dataset_type)

    @pytest.mark.parametrize(
        "dataset_type",
        [
            CustomDatasetType.SINGLE_TURN,
            CustomDatasetType.MULTI_TURN,
            CustomDatasetType.RANDOM_POOL,
        ],
    )
    def test_synthesis_raises_error_with_non_trace_types(
        self, custom_config, mock_tokenizer, dataset_type
    ):
        """Test that synthesis options raise error with non-trace dataset types."""
        custom_config.input.synthesis = SynthesisConfig(speedup_ratio=2.0)
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)

        with pytest.raises(ValueError) as exc:
            composer._validate_synthesis_config(dataset_type)

        assert "only supported with trace datasets" in str(exc.value)
        assert dataset_type.value in str(exc.value)

    @pytest.mark.parametrize(
        "synthesis_config",
        [
            SynthesisConfig(speedup_ratio=2.0),
            SynthesisConfig(prefix_len_multiplier=2.0),
            SynthesisConfig(prefix_root_multiplier=2),
            SynthesisConfig(prompt_len_multiplier=2.0),
        ],
    )
    def test_various_synthesis_options_raise_error(
        self, custom_config, mock_tokenizer, synthesis_config
    ):
        """Test that various synthesis options all trigger validation error."""
        custom_config.input.synthesis = synthesis_config
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)

        with pytest.raises(ValueError) as exc:
            composer._validate_synthesis_config(CustomDatasetType.SINGLE_TURN)

        assert "only supported with trace datasets" in str(exc.value)

    def test_default_synthesis_allowed_with_any_type(
        self, custom_config, mock_tokenizer
    ):
        """Test that default synthesis config (no changes) is allowed with any type."""
        custom_config.input.synthesis = SynthesisConfig()  # All defaults
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)

        # Should not raise for any type
        for dataset_type in CustomDatasetType:
            composer._validate_synthesis_config(dataset_type)

    def test_max_isl_alone_allowed_with_any_type(self, custom_config, mock_tokenizer):
        """Test that max_isl alone doesn't trigger synthesis validation.

        max_isl is a filter, not a synthesis transformation.
        """
        custom_config.input.synthesis = SynthesisConfig(max_isl=4096)
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)

        # Should not raise - max_isl doesn't trigger should_synthesize()
        composer._validate_synthesis_config(CustomDatasetType.SINGLE_TURN)


class TestPreformatPayloads:
    """Test class for CustomDatasetComposer._preformat_payloads."""

    def test_preformat_sets_raw_payload_on_turns(self, custom_config, mock_tokenizer):
        """Test that _preformat_payloads populates raw_payload on each turn."""
        conversations = [
            Conversation(
                session_id="s1",
                turns=[
                    Turn(role="user", texts=[Text(contents=["hi"])]),
                ],
            ),
            Conversation(
                session_id="s2",
                turns=[
                    Turn(role="user", texts=[Text(contents=["bye"])]),
                ],
            ),
        ]

        composer = CustomDatasetComposer(custom_config, mock_tokenizer)

        with patch(
            "aiperf.dataset.composer.custom.format_conversation_payloads"
        ) as mock_fmt:
            mock_fmt.return_value = iter(
                [
                    (
                        "s1",
                        0,
                        {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    ),
                    (
                        "s2",
                        0,
                        {
                            "model": "m",
                            "messages": [{"role": "user", "content": "bye"}],
                        },
                    ),
                ]
            )
            composer._preformat_payloads(conversations)

        assert conversations[0].turns[0].raw_payload == {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }
        assert conversations[1].turns[0].raw_payload == {
            "model": "m",
            "messages": [{"role": "user", "content": "bye"}],
        }

    def test_preformat_skips_on_not_implemented(self, custom_config, mock_tokenizer):
        """Test that _preformat_payloads gracefully skips when endpoint raises NotImplementedError."""
        conversations = [
            Conversation(
                session_id="s1",
                turns=[
                    Turn(role="user", texts=[Text(contents=["hi"])]),
                ],
            ),
        ]

        composer = CustomDatasetComposer(custom_config, mock_tokenizer)

        with patch(
            "aiperf.dataset.composer.custom.format_conversation_payloads"
        ) as mock_fmt:
            mock_fmt.return_value = Mock(__iter__=Mock(side_effect=NotImplementedError))
            composer._preformat_payloads(conversations)

        assert conversations[0].turns[0].raw_payload is None

    def test_preformat_multi_turn_conversation(self, custom_config, mock_tokenizer):
        """Test that _preformat_payloads handles multi-turn conversations correctly."""
        conversations = [
            Conversation(
                session_id="s1",
                turns=[
                    Turn(role="user", texts=[Text(contents=["hello"])]),
                    Turn(role="user", texts=[Text(contents=["world"])]),
                ],
            ),
        ]

        composer = CustomDatasetComposer(custom_config, mock_tokenizer)

        with patch(
            "aiperf.dataset.composer.custom.format_conversation_payloads"
        ) as mock_fmt:
            mock_fmt.return_value = iter(
                [
                    ("s1", 0, {"turn": 0}),
                    ("s1", 1, {"turn": 1}),
                ]
            )
            composer._preformat_payloads(conversations)

        assert conversations[0].turns[0].raw_payload == {"turn": 0}
        assert conversations[0].turns[1].raw_payload == {"turn": 1}

    @pytest.mark.parametrize(
        "dataset_type",
        [CustomDatasetType.SINGLE_TURN, CustomDatasetType.RANDOM_POOL],
    )
    @patch("aiperf.dataset.composer.custom.check_file_exists")
    @patch("aiperf.dataset.composer.custom.plugins.get_class")
    def test_create_dataset_calls_preformat_for_single_turn_types(
        self,
        mock_get_class,
        mock_check_file,
        custom_config,
        mock_tokenizer,
        dataset_type,
    ):
        """Test that create_dataset calls _preformat_payloads for single-turn dataset types."""
        mock_check_file.return_value = None
        mock_loader = Mock()
        mock_loader.load_dataset.return_value = {}
        mock_loader.convert_to_conversations.return_value = [
            Conversation(
                session_id="s1",
                turns=[Turn(role="user", texts=[Text(contents=["hi"])])],
            ),
        ]
        mock_loader_class = Mock()
        mock_loader_class.return_value = mock_loader
        mock_loader_class.get_preferred_sampling_strategy.return_value = (
            DatasetSamplingStrategy.SEQUENTIAL
        )
        mock_get_class.return_value = mock_loader_class

        custom_config.input.custom_dataset_type = dataset_type
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)

        with patch.object(composer, "_preformat_payloads") as mock_preformat:
            composer.create_dataset()
            mock_preformat.assert_called_once()

    @patch("aiperf.dataset.composer.custom.check_file_exists")
    @patch("aiperf.dataset.composer.custom.plugins.get_class")
    def test_create_dataset_skips_preformat_for_multi_turn(
        self, mock_get_class, mock_check_file, custom_config, mock_tokenizer
    ):
        """Test that create_dataset does NOT call _preformat_payloads for multi-turn datasets."""
        mock_check_file.return_value = None
        mock_loader = Mock()
        mock_loader.load_dataset.return_value = {}
        mock_loader.convert_to_conversations.return_value = [
            Conversation(
                session_id="s1",
                turns=[Turn(role="user", texts=[Text(contents=["hi"])])],
            ),
        ]
        mock_loader_class = Mock()
        mock_loader_class.return_value = mock_loader
        mock_loader_class.get_preferred_sampling_strategy.return_value = (
            DatasetSamplingStrategy.SEQUENTIAL
        )
        mock_get_class.return_value = mock_loader_class

        custom_config.input.custom_dataset_type = CustomDatasetType.MULTI_TURN
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)

        with patch.object(composer, "_preformat_payloads") as mock_preformat:
            composer.create_dataset()
            mock_preformat.assert_not_called()
