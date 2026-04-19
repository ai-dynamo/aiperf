# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models import Turn
from aiperf.common.models.sequence_distribution import (
    SequenceLengthDistribution,
)
from aiperf.config import AIPerfConfig
from aiperf.dataset.composer.base import BaseDatasetComposer
from tests.unit.dataset.composer.conftest import _make_run

_BASE = dict(
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
)


class ConcreteBaseComposer(BaseDatasetComposer):
    """Concrete test implementation of BaseDatasetComposer."""

    def create_dataset(self):
        """Required abstract method implementation."""
        return []


class TestBaseDatasetComposer:
    """Test class for BaseDatasetComposer functionality."""

    @pytest.fixture
    def base_config(self):
        """Create a basic AIPerfConfig for testing."""
        return _make_run(
            AIPerfConfig(
                models={
                    "items": [
                        {"name": "test-model-1"},
                        {"name": "test-model-2"},
                    ],
                    "strategy": ModelSelectionStrategy.ROUND_ROBIN,
                },
                **_BASE,
                datasets={
                    "default": {
                        "type": "synthetic",
                        "entries": 1,
                        "prompts": {
                            "isl": {"mean": 100, "stddev": 10},
                            "osl": {"mean": 50, "stddev": 5},
                        },
                    }
                },
            )
        )

    @pytest.fixture
    def sequence_dist_config(self):
        """Create configuration with sequence distribution."""
        return _make_run(
            AIPerfConfig(
                models=["test-model"],
                **_BASE,
                datasets={
                    "default": {
                        "type": "synthetic",
                        "entries": 1,
                        "prompts": {
                            "isl": {"mean": 100, "stddev": 10},
                            "osl": {"mean": 50, "stddev": 5},
                            "sequence_distribution": [
                                {"isl": 100, "osl": 25, "probability": 50},
                                {"isl": 200, "osl": 50, "probability": 50},
                            ],
                        },
                    }
                },
            )
        )

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        return MagicMock()

    def test_initialization_with_sequence_distribution(
        self, sequence_dist_config, mock_tokenizer
    ):
        """Test initialization with sequence distribution."""
        composer = ConcreteBaseComposer(sequence_dist_config, mock_tokenizer)

        assert composer._seq_distribution is not None
        assert isinstance(composer._seq_distribution, SequenceLengthDistribution)
        assert len(composer._seq_distribution.pairs) == 2
        assert len(composer._turn_sequence_cache) == 0

    def test_model_selection_round_robin(self, base_config, mock_tokenizer):
        """Test round robin model selection."""
        composer = ConcreteBaseComposer(base_config, mock_tokenizer)

        assert composer._select_model_name() == "test-model-1"
        assert composer._select_model_name() == "test-model-2"
        assert composer._select_model_name() == "test-model-1"

    def test_model_selection_random(self, base_config, mock_tokenizer):
        """Test random model selection."""
        base_config.cfg.models.strategy = ModelSelectionStrategy.RANDOM
        composer = ConcreteBaseComposer(base_config, mock_tokenizer)

        result = composer._select_model_name()
        assert result in ["test-model-1", "test-model-2"]

    def test_model_selection_invalid_strategy(self, base_config, mock_tokenizer):
        """Test invalid model selection strategy raises error."""
        base_config.cfg.models.strategy = "INVALID"
        composer = ConcreteBaseComposer(base_config, mock_tokenizer)

        with pytest.raises(ValueError, match="Invalid model selection strategy"):
            composer._select_model_name()

    def test_get_turn_sequence_lengths_with_distribution(
        self, sequence_dist_config, mock_tokenizer
    ):
        """Test getting sequence lengths with distribution."""
        composer = ConcreteBaseComposer(sequence_dist_config, mock_tokenizer)

        turn_id = 12345

        with patch.object(composer._seq_distribution, "sample") as mock_sample:
            mock_sample.return_value = (150, 75)

            result = composer._get_turn_sequence_lengths(turn_id)
            assert result == (150, 75)
            mock_sample.assert_called_once()

            result2 = composer._get_turn_sequence_lengths(turn_id)
            assert result2 == (150, 75)
            mock_sample.assert_called_once()

        assert turn_id in composer._turn_sequence_cache
        assert composer._turn_sequence_cache[turn_id] == (150, 75)

    def test_get_turn_sequence_lengths_without_distribution(
        self, base_config, mock_tokenizer
    ):
        """Test getting sequence lengths without distribution (fallback).

        With stddev > 0, values are sampled from normal distribution using seed 42.
        """
        composer = ConcreteBaseComposer(base_config, mock_tokenizer)

        turn_id = 12345
        isl, osl = composer._get_turn_sequence_lengths(turn_id)

        # Sampled from normal(mean=100, stddev=10) and normal(mean=50, stddev=5)
        assert 50 <= isl <= 150, f"ISL {isl} outside expected range"
        assert 20 <= osl <= 80, f"OSL {osl} outside expected range"
        assert turn_id in composer._turn_sequence_cache

    def test_clear_turn_cache(self, sequence_dist_config, mock_tokenizer):
        """Test clearing turn cache."""
        composer = ConcreteBaseComposer(sequence_dist_config, mock_tokenizer)

        composer._turn_sequence_cache[123] = (100, 50)
        composer._turn_sequence_cache[456] = (200, 100)

        composer._clear_turn_cache(123)
        assert 123 not in composer._turn_sequence_cache
        assert 456 in composer._turn_sequence_cache

        composer._clear_turn_cache(999)

    def test_set_max_tokens_with_distribution(
        self, sequence_dist_config, mock_tokenizer
    ):
        """Test setting max_tokens using sequence distribution."""
        composer = ConcreteBaseComposer(sequence_dist_config, mock_tokenizer)
        turn = Turn()

        turn_id = id(turn)
        composer._turn_sequence_cache[turn_id] = (150, 75)

        composer._set_max_tokens(turn)
        assert turn.max_tokens == 75

    def test_set_max_tokens_without_distribution(self, base_config, mock_tokenizer):
        """Test setting max_tokens using legacy behavior."""
        composer = ConcreteBaseComposer(base_config, mock_tokenizer)
        turn = Turn()

        composer._set_max_tokens(turn)

        assert turn.max_tokens is not None
        assert turn.max_tokens > 0
        assert isinstance(turn.max_tokens, int)
        assert 30 < turn.max_tokens < 70

    def test_set_max_tokens_without_distribution_none_osl(self, mock_tokenizer):
        """Test setting max_tokens when osl is None."""
        config = AIPerfConfig(
            models=["test-model"],
            **_BASE,
            datasets={
                "default": {
                    "type": "synthetic",
                    "entries": 1,
                    "prompts": {"isl": 128},
                }
            },
        )
        composer = ConcreteBaseComposer(_make_run(config), mock_tokenizer)
        turn = Turn()

        composer._set_max_tokens(turn)

        # When no OSL is configured, max_tokens should be None
        assert turn.max_tokens is None

    def test_set_max_tokens_preserves_existing_value(self, base_config, mock_tokenizer):
        """Test that per-line max_tokens is not overwritten by global --osl config."""
        composer = ConcreteBaseComposer(base_config, mock_tokenizer)
        turn = Turn(max_tokens=42)

        composer._set_max_tokens(turn)

        assert turn.max_tokens == 42

    def test_set_max_tokens_preserves_existing_with_distribution(
        self, sequence_dist_config, mock_tokenizer
    ):
        """Test that per-line max_tokens is not overwritten by sequence distribution."""
        composer = ConcreteBaseComposer(sequence_dist_config, mock_tokenizer)
        turn = Turn(max_tokens=42)

        turn_id = id(turn)
        composer._turn_sequence_cache[turn_id] = (150, 75)

        composer._set_max_tokens(turn)

        assert turn.max_tokens == 42

    def test_finalize_turn(self, sequence_dist_config, mock_tokenizer):
        """Test turn finalization."""
        composer = ConcreteBaseComposer(sequence_dist_config, mock_tokenizer)
        turn = Turn()
        turn_id = id(turn)

        composer._turn_sequence_cache[turn_id] = (150, 75)

        composer._finalize_turn(turn)

        assert turn.model == "test-model"
        assert turn.max_tokens == 75
        assert turn_id not in composer._turn_sequence_cache
