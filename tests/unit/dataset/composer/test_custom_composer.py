# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, mock_open, patch

import pytest

from aiperf.common.models import Conversation, Turn
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.composer.custom import CustomDatasetComposer
from aiperf.dataset.loader import (
    MooncakeTraceDatasetLoader,
    MultiTurnDatasetLoader,
    RandomPoolDatasetLoader,
    SingleTurnDatasetLoader,
)
from aiperf.plugin.enums import CustomDatasetType, DatasetSamplingStrategy
from tests.unit.dataset.composer.conftest import make_run


class TestInitialization:
    """Test class for CustomDatasetComposer basic initialization."""

    def test_initialization(self, custom_config, mock_tokenizer):
        """Test that CustomDatasetComposer can be instantiated with valid config."""
        composer = CustomDatasetComposer(
            run=make_run(custom_config), tokenizer=mock_tokenizer
        )

        assert composer is not None
        assert isinstance(composer, CustomDatasetComposer)


MOCK_SINGLE_TURN_CONTENT = """{"text": "Write a haiku.", "output_length": 50}
{"text": "Explain quantum computing.", "output_length": 500}
{"text": "Summarize machine learning."}
"""

MOCK_DAG_MODEL_CONTENT = """{"session_id": "root", "turns": [{"messages": [{"role": "user", "content": "Use a child model."}], "model": "child-model"}]}
"""

MOCK_MULTI_TURN_CONTENT = """{"session_id": "s1", "turns": [{"text": "Summarize.", "output_length": 100}, {"text": "Key points only."}, {"text": "Expand on point 2.", "output_length": 300}]}
"""

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
        custom_config.custom_dataset_type = dataset_type
        composer = CustomDatasetComposer(
            run=make_run(custom_config), tokenizer=mock_tokenizer
        )
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
        composer = CustomDatasetComposer(
            run=make_run(trace_config), tokenizer=mock_tokenizer
        )
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
        # Per-line `output_length` on each trace record always wins over any
        # global `--osl` (FileDataset.osl) fallback. The trace fixture sets
        # output_length=52 which is what this test asserts.

        composer = CustomDatasetComposer(
            run=make_run(trace_config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        assert len(conversations) > 0
        # Per-line output_length (52) sourced from trace data
        for conversation in conversations:
            for turn in conversation.turns:
                assert turn.max_tokens == 52

    @patch("aiperf.dataset.composer.custom.check_file_exists")
    @patch("builtins.open", mock_open(read_data=MOCK_SINGLE_TURN_CONTENT))
    def test_single_turn_output_length_precedence(
        self, mock_check_file, custom_config, mock_tokenizer
    ):
        """Test that per-line output_length takes precedence over global --osl in single_turn."""
        custom_config.prompt_output_tokens_mean = 200
        custom_config.prompt_output_tokens_stddev = 0.0

        composer = CustomDatasetComposer(
            run=make_run(custom_config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        assert len(conversations) == 3
        # First two lines have output_length, third falls back to global --osl (200).
        assert conversations[0].turns[0].max_tokens == 50
        assert conversations[1].turns[0].max_tokens == 500
        assert conversations[2].turns[0].max_tokens == 200

    @pytest.mark.skip(
        reason="Test uses v1 CLIConfig.input.file/.custom_dataset_type pattern "
        "which doesn't exist in v2 (datasets are a list, not a nested input "
        "section). Port pending."
    )
    @patch("aiperf.dataset.composer.custom.check_file_exists")
    def test_dag_turn_model_precedence(
        self, mock_check_file, custom_config, mock_tokenizer, tmp_path
    ):
        path = tmp_path / "data.dag.jsonl"
        path.write_text(MOCK_DAG_MODEL_CONTENT, encoding="utf-8")
        custom_config.input.file = str(path)
        custom_config.input.custom_dataset_type = CustomDatasetType.DAG_JSONL
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)
        conversations = composer.create_dataset()

        assert conversations[0].turns[0].model == "child-model"

    @patch("aiperf.dataset.composer.custom.check_file_exists")
    @patch("builtins.open", mock_open(read_data=MOCK_MULTI_TURN_CONTENT))
    def test_multi_turn_output_length_precedence(
        self, mock_check_file, custom_config, mock_tokenizer
    ):
        """Test per-turn output_length precedence over global --osl in multi_turn."""
        custom_config.custom_dataset_type = CustomDatasetType.MULTI_TURN
        custom_config.prompt_output_tokens_mean = 200
        custom_config.prompt_output_tokens_stddev = 0.0

        composer = CustomDatasetComposer(
            run=make_run(custom_config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        assert len(conversations) == 1
        turns = conversations[0].turns
        assert len(turns) == 3
        # Turn 1 and 3 have output_length, turn 2 falls back to global --osl (200).
        assert turns[0].max_tokens == 100
        assert turns[1].max_tokens == 200
        assert turns[2].max_tokens == 300

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
        custom_config.custom_dataset_type = CustomDatasetType.MOONCAKE_TRACE

        composer = CustomDatasetComposer(
            run=make_run(custom_config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        for conversation in conversations:
            for turn in conversation.turns:
                assert turn.max_tokens == 52


class TestErrorHandling:
    """Test class for CustomDatasetComposer error handling scenarios."""

    @patch("aiperf.dataset.composer.custom.check_file_exists")
    @patch("aiperf.dataset.composer.custom.plugins.get_class")
    @patch.object(CustomDatasetComposer, "_infer_dataset_type")
    def test_create_dataset_empty_result(
        self,
        mock_infer,
        mock_get_class,
        mock_check_file,
        custom_config,
        mock_tokenizer,
    ):
        """Test create_dataset when loader returns empty data."""
        mock_check_file.return_value = None
        mock_infer.return_value = CustomDatasetType.SINGLE_TURN
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

        composer = CustomDatasetComposer(
            run=make_run(custom_config), tokenizer=mock_tokenizer
        )
        result = composer.create_dataset()

        assert isinstance(result, list)
        assert len(result) == 0


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
        trace_config.synthesis_speedup_ratio = 2.0
        composer = CustomDatasetComposer(
            run=make_run(trace_config), tokenizer=mock_tokenizer
        )

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
        custom_config.synthesis_speedup_ratio = 2.0
        composer = CustomDatasetComposer(
            run=make_run(custom_config), tokenizer=mock_tokenizer
        )

        with pytest.raises(ValueError) as exc:
            composer._validate_synthesis_config(dataset_type)

        assert "only supported with trace datasets" in str(exc.value)
        assert dataset_type.value in str(exc.value)

    @pytest.mark.parametrize(
        "synthesis_kwargs",
        [
            {"synthesis_speedup_ratio": 2.0},
            {"synthesis_prefix_len_multiplier": 2.0},
            {"synthesis_prefix_root_multiplier": 2},
            {"synthesis_prompt_len_multiplier": 2.0},
            {"synthesis_output_len_multiplier": 2.0},
            {"synthesis_max_isl": 4096},
            {"synthesis_max_osl": 4096},
        ],
    )
    def test_various_synthesis_options_raise_error(
        self, custom_config, mock_tokenizer, synthesis_kwargs
    ):
        """Test that various synthesis options all trigger validation error."""
        for k, v in synthesis_kwargs.items():
            setattr(custom_config, k, v)
        composer = CustomDatasetComposer(
            run=make_run(custom_config), tokenizer=mock_tokenizer
        )

        with pytest.raises(ValueError) as exc:
            composer._validate_synthesis_config(CustomDatasetType.SINGLE_TURN)

        assert "only supported with trace datasets" in str(exc.value)

    def test_default_synthesis_allowed_with_any_type(
        self, custom_config, mock_tokenizer
    ):
        """Test that default synthesis config (no changes) is allowed with any type."""
        # No synthesis_* fields set: behaves as default identity synthesis.
        composer = CustomDatasetComposer(
            run=make_run(custom_config), tokenizer=mock_tokenizer
        )

        # Should not raise for any type
        for dataset_type in CustomDatasetType:
            composer._validate_synthesis_config(dataset_type)

    @pytest.mark.parametrize(
        ("synthesis_field", "value"),
        [
            ("synthesis_max_isl", 4096),
            ("synthesis_max_osl", 4096),
        ],
    )
    @pytest.mark.parametrize(
        "dataset_type",
        [
            CustomDatasetType.MOONCAKE_TRACE,
            CustomDatasetType.BAILIAN_TRACE,
        ],
    )
    def test_trace_only_caps_allowed_with_trace_datasets(
        self, custom_config, mock_tokenizer, synthesis_field, value, dataset_type
    ):
        """Test that max_isl/max_osl are allowed with trace dataset types."""
        setattr(custom_config, synthesis_field, value)
        composer = CustomDatasetComposer(
            run=make_run(custom_config), tokenizer=mock_tokenizer
        )

        composer._validate_synthesis_config(dataset_type)


class TestExplicitCustomDatasetType:
    """Regression tests for `--custom-dataset-type random-pool` routing.

    The user-facing flag converts to ``FileDataset.format`` in v2; the composer
    must honor that explicit choice rather than re-inferring from the file.
    A bare JSONL like ``{"text": "...", "max_tokens": ...}`` is structurally
    valid for both single_turn and random_pool, but only random_pool gives the
    user random-with-replacement sampling. Inference (the previous default)
    silently picked single_turn, dropping the user's choice.
    """

    def test_dag_jsonl_routes_to_dag_loader(self, tmp_path, mock_tokenizer):
        from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader

        jsonl = tmp_path / "dag.jsonl"
        jsonl.write_text(
            '{"session_id": "root", "turns": [{"messages": [{"role": "user", "content": "hi"}]}]}\n'
        )

        cli_config = CLIConfig(
            model_names=["test-model"],
            input_file=str(jsonl),
            custom_dataset_type=CustomDatasetType.DAG_JSONL,
        )
        composer = CustomDatasetComposer(
            run=make_run(cli_config), tokenizer=mock_tokenizer
        )

        assert composer._explicit_format() == CustomDatasetType.DAG_JSONL

        composer.create_dataset()
        assert isinstance(composer.loader, DagJsonlLoader)

    def test_random_pool_routes_to_random_pool_loader(self, tmp_path, mock_tokenizer):
        """`--custom-dataset-type random-pool` must produce a RandomPoolDatasetLoader."""
        from aiperf.config.flags.cli_config import CLIConfig

        jsonl = tmp_path / "rp.jsonl"
        jsonl.write_text(
            '{"text": "hello world", "max_tokens": 16}\n'
            '{"text": "another prompt", "max_tokens": 16}\n'
        )

        cli_config = CLIConfig(
            model_names=["test-model"],
            input_file=str(jsonl),
            custom_dataset_type=CustomDatasetType.RANDOM_POOL,
            conversation_num_dataset_entries=10,
        )
        composer = CustomDatasetComposer(
            run=make_run(cli_config), tokenizer=mock_tokenizer
        )

        # The composer should resolve RANDOM_POOL from FileDataset.format,
        # not silently fall back to SingleTurn via structural inference.
        assert composer._explicit_format() == CustomDatasetType.RANDOM_POOL

        conversations = composer.create_dataset()
        assert isinstance(composer.loader, RandomPoolDatasetLoader)
        assert len(conversations) == 10

    def test_random_pool_samples_with_replacement(self, tmp_path, mock_tokenizer):
        """A 2-prompt pool sampled into 200 conversations should hit each prompt
        many times — silent fallback to single_turn would give exactly 2."""

        jsonl = tmp_path / "rp.jsonl"
        jsonl.write_text(
            '{"text": "alpha", "max_tokens": 8}\n{"text": "bravo", "max_tokens": 8}\n'
        )

        cli_config = CLIConfig(
            model_names=["test-model"],
            input_file=str(jsonl),
            custom_dataset_type=CustomDatasetType.RANDOM_POOL,
            conversation_num_dataset_entries=200,
        )
        composer = CustomDatasetComposer(
            run=make_run(cli_config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        # 200 conversations from a 2-prompt pool — both prompts must appear
        # often (binomial 200, p=0.5; deterministic via auto-fixture seed).
        prompts = [
            turn.texts[0].contents[0] for conv in conversations for turn in conv.turns
        ]
        counts: dict[str, int] = {}
        for p in prompts:
            counts[p] = counts.get(p, 0) + 1
        assert len(prompts) == 200
        assert set(counts) == {"alpha", "bravo"}
        # Each prompt should land between 50 and 150 (well inside the 99.999%
        # band for binomial(200, 0.5); this is a sampling-with-replacement
        # signature, NOT the sequential single_turn fallback which would
        # produce exactly 2 unique prompts in 2 conversations).
        assert 50 < counts["alpha"] < 150
        assert 50 < counts["bravo"] < 150

    def test_explicit_format_default_does_not_short_circuit(
        self, tmp_path, mock_tokenizer
    ):
        """When `--custom-dataset-type` was NOT supplied, ``_explicit_format``
        returns None so structural inference still runs.

        ``FileDataset.format`` defaults to ``SINGLE_TURN``; we must only
        short-circuit when the user explicitly chose a format.
        """

        jsonl = tmp_path / "no_type.jsonl"
        jsonl.write_text('{"text": "hi", "max_tokens": 4}\n')

        cli_config = CLIConfig(
            model_names=["test-model"],
            input_file=str(jsonl),
        )
        composer = CustomDatasetComposer(
            run=make_run(cli_config), tokenizer=mock_tokenizer
        )
        assert composer._explicit_format() is None
