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


# ============================================================================
# ISL-sampling regression tests
#
# Background: A refactor introduced sample_int() inside
# _get_turn_sequence_lengths while the prompt generator ALSO sampled using
# (mean, stddev), causing a double-sample that inflated effective stddev
# to stddev * sqrt(2). These tests lock the flow down end-to-end:
#
#   1. Turn-level sample matches the configured distribution's shape
#      (empirical stats for Normal, expected modes for Multimodal, etc.).
#   2. Per-turn samples are independent (different turn_id -> new sample).
#   3. ISL and OSL for the same turn are cached together (stable pair).
#   4. Cache is keyed by turn_id.
#
# The matching end-to-end assertion (prompt generator receives stddev=0
# and mean=<turn-level sample>) lives in test_synthetic_composer.py.
# ============================================================================


def _prompts_config(**prompts_overrides):
    dataset = {"type": "synthetic", "entries": 1, "prompts": prompts_overrides}
    return _make_run(
        AIPerfConfig(**_BASE, models=["test-model"], datasets={"default": dataset})
    )


class TestIslSamplingAtTurnLevel:
    """Regression tests for _get_turn_sequence_lengths across distribution types."""

    @pytest.fixture
    def mock_tokenizer(self):
        return MagicMock()

    def _sample_isls(self, composer, n):
        """Pull ISL for n distinct turns, using stable integer ids."""
        isls = []
        for i in range(n):
            turn_id = 1_000_000 + i
            isl, _osl = composer._get_turn_sequence_lengths(turn_id)
            isls.append(isl)
        assert len(composer._turn_sequence_cache) == n
        return isls

    def test_normal_isl_empirical_stats_match_config(self, mock_tokenizer):
        """N(mean=500, stddev=50): empirical stddev must be ~50, not 50*sqrt(2)≈70.7.

        Regression for the double-sampling bug. If the bug returns,
        either the turn-level sample is missing (empirical stddev ≈ 0) or
        the caller re-samples downstream (empirical stddev ≈ 70.7).
        """
        import statistics

        run = _prompts_config(isl={"mean": 500, "stddev": 50}, osl=64)
        composer = ConcreteBaseComposer(run, mock_tokenizer)

        isls = self._sample_isls(composer, n=500)

        empirical_mean = statistics.fmean(isls)
        empirical_stddev = statistics.stdev(isls)

        # Tolerances sized for N=500 with configured stddev=50:
        #   stderr(mean) ≈ 50/sqrt(500) ≈ 2.24 -> allow ±10
        #   stderr(stddev) ≈ 50/sqrt(2*500) ≈ 1.58 -> allow ±8 (must exclude 70.7)
        assert 490 <= empirical_mean <= 510, (
            f"mean drifted: {empirical_mean:.2f} (configured 500)"
        )
        assert 42 <= empirical_stddev <= 58, (
            f"stddev drifted: {empirical_stddev:.2f} "
            f"(configured 50; double-sample would give ~70.7)"
        )

    def test_normal_zero_stddev_is_deterministic(self, mock_tokenizer):
        """Normal with stddev=0 must return the mean literally (no noise)."""
        run = _prompts_config(isl={"mean": 256, "stddev": 0}, osl=64)
        composer = ConcreteBaseComposer(run, mock_tokenizer)

        isls = self._sample_isls(composer, n=50)
        assert all(v == 256 for v in isls), f"expected all==256, got {set(isls)}"

    def test_fixed_isl_scalar_is_constant(self, mock_tokenizer):
        """Fixed scalar ISL must return the literal value every turn."""
        run = _prompts_config(isl=128, osl=64)
        composer = ConcreteBaseComposer(run, mock_tokenizer)

        isls = self._sample_isls(composer, n=30)
        assert all(v == 128 for v in isls)

    def test_lognormal_isl_preserves_skew(self, mock_tokenizer):
        """LogNormal (mean > median) must produce a right-skewed empirical sample.

        If the turn-level sample were dropped (Option-A style regression),
        the samples would collapse to the literal mean and median == mean.
        """
        import statistics

        run = _prompts_config(isl={"mean": 1000, "median": 400}, osl=64)
        composer = ConcreteBaseComposer(run, mock_tokenizer)

        isls = self._sample_isls(composer, n=500)

        empirical_median = statistics.median(isls)
        # Right-skewed: median should be well below mean (1000).
        assert empirical_median < 800, (
            f"LogNormal median {empirical_median} not skewed below mean=1000 "
            f"(is turn-level sampling still happening?)"
        )
        # Variance must be non-trivial (would be 0 if collapsed to mean).
        assert statistics.stdev(isls) > 50

    def test_multimodal_isl_hits_both_peaks(self, mock_tokenizer):
        """Multimodal must actually sample from multiple peaks."""
        run = _prompts_config(
            isl={
                "peaks": [
                    {"mean": 100, "stddev": 5, "weight": 50},
                    {"mean": 2000, "stddev": 50, "weight": 50},
                ]
            },
            osl=64,
        )
        composer = ConcreteBaseComposer(run, mock_tokenizer)

        isls = self._sample_isls(composer, n=200)

        low_peak = sum(1 for v in isls if v < 500)
        high_peak = sum(1 for v in isls if v > 1500)
        middle = sum(1 for v in isls if 500 <= v <= 1500)

        assert low_peak > 50, f"low peak underpopulated: {low_peak}"
        assert high_peak > 50, f"high peak underpopulated: {high_peak}"
        assert middle < 20, f"{middle} samples landed in gap — distribution shape lost"

    def test_empirical_isl_respects_discrete_values(self, mock_tokenizer):
        """Empirical must only return the configured discrete values."""
        allowed = {128, 512, 2048}
        run = _prompts_config(
            isl={
                "points": [
                    {"value": 128, "weight": 1},
                    {"value": 512, "weight": 1},
                    {"value": 2048, "weight": 1},
                ]
            },
            osl=64,
        )
        composer = ConcreteBaseComposer(run, mock_tokenizer)

        isls = self._sample_isls(composer, n=200)
        unseen = allowed - set(isls)
        out_of_set = set(isls) - allowed

        assert not out_of_set, f"unexpected ISL values: {out_of_set}"
        assert not unseen, f"never sampled these values: {unseen}"

    def test_same_turn_id_returns_cached_pair(self, mock_tokenizer):
        """Calling _get_turn_sequence_lengths twice with same turn_id hits cache."""
        run = _prompts_config(
            isl={"mean": 500, "stddev": 50}, osl={"mean": 100, "stddev": 20}
        )
        composer = ConcreteBaseComposer(run, mock_tokenizer)

        first = composer._get_turn_sequence_lengths(42)
        second = composer._get_turn_sequence_lengths(42)
        assert first == second
        assert composer._turn_sequence_cache[42] == first

    def test_different_turn_ids_sample_independently(self, mock_tokenizer):
        """Different turn_ids must draw fresh samples."""
        run = _prompts_config(isl={"mean": 500, "stddev": 100}, osl=64)
        composer = ConcreteBaseComposer(run, mock_tokenizer)

        samples = {composer._get_turn_sequence_lengths(i)[0] for i in range(100)}
        assert len(samples) > 50, f"only {len(samples)} unique ISLs — sampling stuck?"

    def test_isl_osl_sampled_from_independent_distributions(self, mock_tokenizer):
        """ISL and OSL are sampled independently in the non-joint branch."""
        run = _prompts_config(
            isl={"mean": 500, "stddev": 50},
            osl={"mean": 100, "stddev": 10},
        )
        composer = ConcreteBaseComposer(run, mock_tokenizer)

        pairs = [composer._get_turn_sequence_lengths(i) for i in range(100)]
        isls = [p[0] for p in pairs]
        osls = [p[1] for p in pairs]

        assert min(isls) != max(isls)
        assert min(osls) != max(osls)
        # ISL mean 500, OSL mean 100, 8σ apart — ranges must not overlap.
        assert min(isls) > max(osls), (
            f"ISL ({min(isls)}..{max(isls)}) and OSL ({min(osls)}..{max(osls)}) ranges overlap"
        )

    def test_missing_isl_config_falls_back_to_default(self, mock_tokenizer):
        """When prompts has no isl field, default ISL=128 is used."""
        run = _prompts_config(osl=64)
        composer = ConcreteBaseComposer(run, mock_tokenizer)

        isl, _ = composer._get_turn_sequence_lengths(1)
        assert isl == 128

    def test_missing_osl_config_returns_none(self, mock_tokenizer):
        """When prompts has no osl field, OSL is None (propagates to max_tokens=None)."""
        run = _prompts_config(isl=256)
        composer = ConcreteBaseComposer(run, mock_tokenizer)

        _, osl = composer._get_turn_sequence_lengths(1)
        assert osl is None
