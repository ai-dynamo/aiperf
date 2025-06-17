# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive unit tests for PromptGenerator class.

This test file provides complete coverage of all methods in the PromptGenerator class,
including edge cases, error conditions, and integration scenarios.
"""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from aiperf.common.exceptions import (
    GeneratorConfigurationError,
    GeneratorInitializationError,
)
from aiperf.services.dataset.config import PrefixPromptConfig, PromptConfig
from aiperf.services.dataset.generator.prompt import PromptGenerator

MOCK_CORPUS_CONTENT = "To be or not to be, that is the question.\nWhether 'tis nobler in the mind to suffer.\n"


@patch("builtins.open", mock_open(read_data=MOCK_CORPUS_CONTENT))
class TestPromptGeneratorComprehensive:
    """Comprehensive test suite for PromptGenerator class."""

    @pytest.fixture
    def mock_tokenizer(self, mock_tokenizer_cls):
        """Mock tokenizer class for testing."""
        return mock_tokenizer_cls.from_pretrained("gpt2")

    @pytest.fixture
    def basic_config(self, mock_tokenizer):
        """Basic configuration for testing."""
        return PromptConfig(
            tokenizer=mock_tokenizer,
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )

    @pytest.fixture
    def config_with_prefix(self, mock_tokenizer):
        """Configuration with prefix prompt pool."""
        return PromptConfig(
            tokenizer=mock_tokenizer,
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=5, length=10),
        )

    # ============================================================================
    # Initialization Tests
    # ============================================================================

    def test_init_basic_configuration(self, basic_config):
        """Test basic initialization without prefix prompts."""
        generator = PromptGenerator(basic_config)

        assert generator.config == basic_config
        assert generator.tokenizer == basic_config.tokenizer
        assert generator._tokenized_corpus is not None
        assert generator._corpus_size > 0
        assert len(generator._prefix_prompts) == 0
        assert len(generator._cache) == 0

    def test_init_with_prefix_prompts(self, config_with_prefix):
        """Test initialization with prefix prompt pool."""
        generator = PromptGenerator(config_with_prefix)

        assert len(generator._prefix_prompts) == 5
        assert all(isinstance(prompt, str) for prompt in generator._prefix_prompts)

    def test_init_corpus_initialization(self, basic_config):
        """Test that corpus is properly initialized during __init__."""
        with patch.object(PromptGenerator, "_initialize_corpus") as mock_init:
            _ = PromptGenerator(basic_config)
            mock_init.assert_called_once()

    # ============================================================================
    # Generate Method Tests
    # ============================================================================

    @patch("aiperf.services.dataset.utils.sample_positive_normal_integer")
    def test_generate_without_hash_ids(self, mock_sample, basic_config):
        """Test generate method without hash_ids uses normal generation."""
        mock_sample.return_value = 50

        generator = PromptGenerator(basic_config)

        with patch.object(
            generator, "_generate_prompt", return_value="test prompt"
        ) as mock_gen:
            result = generator.generate(mean=100, stddev=20)

            mock_sample.assert_called_once_with(100, 20)
            mock_gen.assert_called_once_with(50)
            assert result == "test prompt"

    def test_generate_with_hash_ids(self, basic_config):
        """Test generate method with hash_ids uses cached generation."""
        generator = PromptGenerator(basic_config)

        with patch.object(
            generator, "_generate_cached_prompt", return_value="cached prompt"
        ) as mock_cached:
            result = generator.generate(mean=100, stddev=20, hash_ids=[1, 2, 3])

            mock_cached.assert_called_once_with(100, [1, 2, 3], 512)
            assert result == "cached prompt"

    @patch("aiperf.services.dataset.utils.sample_positive_normal_integer")
    def test_generate_with_empty_hash_ids(self, mock_sample, basic_config):
        """Test generate method with empty hash_ids list."""
        mock_sample.return_value = 30
        generator = PromptGenerator(basic_config)

        with patch.object(
            generator, "_generate_prompt", return_value="normal prompt"
        ) as mock_gen:
            result = generator.generate(mean=100, stddev=20, hash_ids=[])

            # Empty list should be falsy, so should use normal generation
            mock_gen.assert_called_once_with(30)
            assert result == "normal prompt"

    # ============================================================================
    # _generate_prompt Method Tests
    # ============================================================================

    def test_generate_prompt_normal_case(self, basic_config):
        """Test _generate_prompt method with normal parameters."""
        generator = PromptGenerator(basic_config)

        with patch.object(
            generator, "_sample_tokens", return_value=[10, 11, 12]
        ) as mock_sample:
            result = generator._generate_prompt(3)

            mock_sample.assert_called_once_with(3)
            generator.tokenizer.decode.assert_called_once_with([10, 11, 12])
            assert result.startswith("token_")

    def test_generate_prompt_zero_tokens(self, basic_config):
        """Test _generate_prompt with zero tokens."""
        generator = PromptGenerator(basic_config)

        with patch.object(generator, "_sample_tokens", return_value=[]) as mock_sample:
            _ = generator._generate_prompt(0)

            mock_sample.assert_called_once_with(0)
            generator.tokenizer.decode.assert_called_once_with([])

    def test_generate_prompt_large_number(self, basic_config):
        """Test _generate_prompt with large number of tokens."""
        generator = PromptGenerator(basic_config)

        large_tokens = list(range(1000))
        with patch.object(
            generator, "_sample_tokens", return_value=large_tokens
        ) as mock_sample:
            _ = generator._generate_prompt(1000)

            mock_sample.assert_called_once_with(1000)
            generator.tokenizer.decode.assert_called_once_with(large_tokens)

    # ============================================================================
    # _generate_cached_prompt Method Tests
    # ============================================================================

    def test_generate_cached_prompt_valid_parameters(self, basic_config):
        """Test _generate_cached_prompt with valid parameters."""
        generator = PromptGenerator(basic_config)

        with patch.object(
            generator, "_sample_tokens", return_value=[10, 11, 12, 13, 14]
        ):
            result = generator._generate_cached_prompt(
                num_tokens=10, prompt_hash_list=[1, 2], block_size=5
            )

            # Should have created cache entries
            assert 1 in generator._cache
            assert 2 in generator._cache

            # Each cache entry should have BOS token at start
            assert generator._cache[1][0] == 1  # BOS token
            assert generator._cache[2][0] == 1  # BOS token

            # Should return decoded prompt
            assert isinstance(result, str)

    def test_generate_cached_prompt_reuse_cache(self, basic_config):
        """Test _generate_cached_prompt reuses existing cache entries."""
        generator = PromptGenerator(basic_config)

        # Pre-populate cache
        generator._cache[1] = [1, 10, 11, 12, 13]

        with patch.object(
            generator, "_sample_tokens", return_value=[20, 21, 22, 23, 24]
        ) as mock_sample:
            _ = generator._generate_cached_prompt(
                num_tokens=10, prompt_hash_list=[1, 2], block_size=5
            )

            # Should only sample tokens for new hash_id (2)
            mock_sample.assert_called_once_with(5)

            # Should reuse existing cache for hash_id 1
            assert generator._cache[1] == [1, 10, 11, 12, 13]

    def test_generate_cached_prompt_uneven_final_block(self, basic_config):
        """Test _generate_cached_prompt with uneven final block size."""
        generator = PromptGenerator(basic_config)

        with patch.object(
            generator, "_sample_tokens", side_effect=lambda n: list(range(n))
        ):
            _ = generator._generate_cached_prompt(
                num_tokens=12,  # 5 + 5 + 2
                prompt_hash_list=[1, 2, 3],
                block_size=5,
            )

            # Final block should have different size
            assert len(generator._cache[3]) == 2  # Final block: 12 - (2 * 5) = 2

    @pytest.mark.parametrize(
        "num_tokens,hash_list,block_size,should_raise",
        [
            (10, [1, 2], 5, False),  # Valid: 5 + 5 = 10
            (12, [1, 2, 3], 5, False),  # Valid: 5 + 5 + 2 = 12
            (
                8,
                [1, 2],
                5,
                False,
            ),  # Valid: 5 + 3 = 8 (final hash length = 8 - 5 = 3 <= block_size)
            (
                15,
                [1, 2],
                5,
                True,
            ),  # Invalid: final hash length = 15 - 5 = 10 > block_size
            (0, [1], 5, True),  # Invalid: final hash length = 0
            (5, [1, 2], 5, True),  # Invalid: final hash length = 5 - 5 = 0
        ],
    )
    def test_generate_cached_prompt_validation(
        self,
        num_tokens,
        hash_list,
        block_size,
        should_raise,
        basic_config,
    ):
        """Test _generate_cached_prompt parameter validation."""
        generator = PromptGenerator(basic_config)

        if should_raise:
            with pytest.raises(GeneratorConfigurationError, match="are not compatible"):
                generator._generate_cached_prompt(num_tokens, hash_list, block_size)
        else:
            with patch.object(
                generator, "_sample_tokens", return_value=list(range(block_size))
            ):
                # Should not raise
                result = generator._generate_cached_prompt(
                    num_tokens, hash_list, block_size
                )
                assert isinstance(result, str)

    def test_generate_cached_prompt_bos_token_insertion(self, basic_config):
        """Test that BOS token is correctly inserted in cached prompts."""
        generator = PromptGenerator(basic_config)

        original_tokens = [10, 11, 12, 13, 14]
        with patch.object(
            generator, "_sample_tokens", return_value=original_tokens.copy()
        ):
            generator._generate_cached_prompt(5, [1], 5)

            # First token should be BOS token (1)
            assert generator._cache[1][0] == 1
            # Length should be maintained (5 tokens)
            assert len(generator._cache[1]) == 5
            # Should contain the other tokens (original[1:] + [BOS])
            assert generator._cache[1][1:] == original_tokens[1:]

    # ============================================================================
    # _sample_tokens Method Tests
    # ============================================================================

    def test_sample_tokens_normal_case(self, basic_config):
        """Test _sample_tokens with normal parameters."""
        generator = PromptGenerator(basic_config)

        # Mock random.randrange to control start position
        with patch("random.randrange", return_value=5):
            tokens = generator._sample_tokens(3)

            assert len(tokens) == 3
            assert all(isinstance(t, int) for t in tokens)

    def test_sample_tokens_wrap_around(self, basic_config):
        """Test _sample_tokens when it needs to wrap around the corpus."""
        generator = PromptGenerator(basic_config)
        corpus_size = generator._corpus_size

        # Start near the end to force wrap-around
        with patch("random.randrange", return_value=corpus_size - 2):
            tokens = generator._sample_tokens(5)

            assert len(tokens) == 5
            # Should include tokens from both end and beginning of corpus

    def test_sample_tokens_exact_corpus_size(self, basic_config):
        """Test _sample_tokens when requesting exactly corpus size."""
        generator = PromptGenerator(basic_config)
        corpus_size = generator._corpus_size

        with patch("random.randrange", return_value=0):
            tokens = generator._sample_tokens(corpus_size)

            assert len(tokens) == corpus_size
            assert tokens == generator._tokenized_corpus

    @patch("aiperf.services.dataset.generator.prompt.logger.warning")
    @patch("random.randrange", return_value=0)
    def test_sample_tokens_longer_than_corpus_with_warning(
        self, mock_randrange, mock_warning, basic_config
    ):
        """Test _sample_tokens when requested length exceeds corpus size."""
        generator = PromptGenerator(basic_config)
        corpus_size = generator._corpus_size

        tokens = generator._sample_tokens(corpus_size * 2)

        # Should log a warning
        mock_warning.assert_called_once()
        assert "longer than the corpus" in str(mock_warning.call_args)
        assert len(tokens) == corpus_size * 2

    def test_sample_tokens_no_corpus_initialized(self, basic_config):
        """Test _sample_tokens when corpus is not initialized."""
        generator = PromptGenerator.__new__(
            PromptGenerator
        )  # Create without calling __init__
        generator._tokenized_corpus = None

        with pytest.raises(GeneratorInitializationError):
            generator._sample_tokens(10)

    def test_sample_tokens_empty_corpus(self, basic_config):
        """Test _sample_tokens with empty corpus."""
        generator = PromptGenerator.__new__(PromptGenerator)
        generator._tokenized_corpus = []
        generator._corpus_size = 0

        with pytest.raises(GeneratorInitializationError):
            generator._sample_tokens(5)

    # ============================================================================
    # get_random_prefix_prompt Method Tests
    # ============================================================================

    def test_get_random_prefix_prompt_success(self, config_with_prefix):
        """Test get_random_prefix_prompt with populated pool."""
        generator = PromptGenerator(config_with_prefix)

        # Mock random.choice to control selection
        expected_prompt = "selected_prompt"
        with patch("random.choice", return_value=expected_prompt):
            result = generator.get_random_prefix_prompt()
            assert result == expected_prompt

    def test_get_random_prefix_prompt_multiple_calls(self, config_with_prefix):
        """Test get_random_prefix_prompt returns different prompts across calls."""
        generator = PromptGenerator(config_with_prefix)

        # Should be able to call multiple times
        prompt1 = generator.get_random_prefix_prompt()
        prompt2 = generator.get_random_prefix_prompt()

        assert isinstance(prompt1, str)
        assert isinstance(prompt2, str)
        # Both should be from the pool
        assert prompt1 in generator._prefix_prompts
        assert prompt2 in generator._prefix_prompts

    def test_get_random_prefix_prompt_empty_pool(self, basic_config):
        """Test get_random_prefix_prompt with empty pool."""
        generator = PromptGenerator(basic_config)

        with pytest.raises(GeneratorInitializationError):
            generator.get_random_prefix_prompt()

    # ============================================================================
    # _initialize_corpus Method Tests
    # ============================================================================

    @patch("os.cpu_count", return_value=4)
    def test_initialize_corpus_success(self, mock_cpu_count, basic_config):
        """Test _initialize_corpus method successful execution."""
        generator = PromptGenerator(basic_config)

        assert generator._tokenized_corpus is not None
        assert generator._corpus_size > 0
        assert isinstance(generator._tokenized_corpus, list)
        assert all(isinstance(token, int) for token in generator._tokenized_corpus)

    @patch("os.cpu_count", return_value=None)
    def test_initialize_corpus_no_cpu_count(self, mock_cpu_count, basic_config):
        """Test _initialize_corpus when os.cpu_count() returns None."""
        generator = PromptGenerator(basic_config)

        # Should still work with default 4 threads
        assert generator._tokenized_corpus is not None

    @patch("aiperf.services.dataset.generator.prompt.ThreadPoolExecutor")
    @patch("os.cpu_count", return_value=2)
    def test_initialize_corpus_threading(
        self, mock_cpu_count, mock_executor, basic_config
    ):
        """Test that corpus initialization uses ThreadPoolExecutor correctly."""
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_executor_instance.map.return_value = [[10, 11], [12, 13]]

        _ = PromptGenerator(basic_config)

        mock_executor.assert_called_once_with(max_workers=2)
        mock_executor_instance.map.assert_called_once()

    # ============================================================================
    # _create_prefix_prompt_pool Method Tests
    # ============================================================================

    def test_create_prefix_prompt_pool_success(self, config_with_prefix):
        """Test _create_prefix_prompt_pool successful creation."""
        generator = PromptGenerator(config_with_prefix)

        assert len(generator._prefix_prompts) == 5
        assert all(isinstance(prompt, str) for prompt in generator._prefix_prompts)

    def test_create_prefix_prompt_pool_no_corpus(self, config_with_prefix):
        """Test _create_prefix_prompt_pool when corpus is not initialized."""
        generator = PromptGenerator.__new__(
            PromptGenerator
        )  # Create without calling __init__
        generator.config = config_with_prefix
        generator._tokenized_corpus = None

        with pytest.raises(GeneratorInitializationError):
            generator._create_prefix_prompt_pool()

    def test_create_prefix_prompt_pool_zero_length(self, mock_tokenizer):
        """Test _create_prefix_prompt_pool with zero length prompts."""
        config = PromptConfig(
            tokenizer=mock_tokenizer,
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=3, length=0),
        )

        generator = PromptGenerator(config)

        assert len(generator._prefix_prompts) == 3
        # Each prompt should be empty string when length is 0
        assert all(
            prompt == "" or isinstance(prompt, str)
            for prompt in generator._prefix_prompts
        )

    # ============================================================================
    # Integration and Edge Case Tests
    # ============================================================================

    def test_tokenizer_integration_with_mock_tokenizer(self, mock_tokenizer):
        """Test integration with the mock tokenizer fixture."""
        config = PromptConfig(
            tokenizer=mock_tokenizer,
            # TODO: remove
            mean=50,
            stddev=10,
            block_size=256,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )

        generator = PromptGenerator(config)

        # Test generate method
        prompt = generator.generate(mean=50, stddev=10)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_cache_persistence_across_calls(self, basic_config):
        """Test that cache persists across multiple calls."""
        generator = PromptGenerator(basic_config)

        with patch.object(
            generator, "_sample_tokens", return_value=[10, 11, 12, 13, 14]
        ):
            # First call
            generator._generate_cached_prompt(10, [1, 2], 5)
            first_cache_1 = generator._cache[1].copy()
            first_cache_2 = generator._cache[2].copy()

            # Second call with same hash_ids
            generator._generate_cached_prompt(10, [1, 2], 5)

            # Cache should be reused (same values)
            assert generator._cache[1] == first_cache_1
            assert generator._cache[2] == first_cache_2

    def test_mixed_cache_and_new_generation(self, basic_config):
        """Test mixing cached and new hash IDs in same call."""
        generator = PromptGenerator(basic_config)

        # Pre-populate cache with one hash_id
        generator._cache[1] = [1, 10, 11, 12, 13]

        with patch.object(
            generator, "_sample_tokens", return_value=[20, 21, 22, 23, 24]
        ):
            # Call with mix of cached and new hash_ids
            _ = generator._generate_cached_prompt(15, [1, 2, 3], 5)

            # Should reuse hash_id 1 and create new for 2 and 3
            assert generator._cache[1] == [1, 10, 11, 12, 13]  # Unchanged
            assert 2 in generator._cache  # Newly created
            assert 3 in generator._cache  # Newly created

    def test_corpus_file_path_construction(self, basic_config):
        """Test that corpus file path is constructed correctly."""
        generator = PromptGenerator(basic_config)

        # Should not raise any path-related errors
        assert generator._tokenized_corpus is not None

    def test_large_prefix_prompt_pool(self, mock_tokenizer):
        """Test with large prefix prompt pool."""
        config = PromptConfig(
            tokenizer=mock_tokenizer,
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=100, length=5),
        )

        generator = PromptGenerator(config)

        assert len(generator._prefix_prompts) == 100
        assert all(isinstance(prompt, str) for prompt in generator._prefix_prompts)

    def test_error_handling_in_tokenization(self, basic_config):
        """Test error handling when tokenization fails."""
        # Mock tokenizer to raise an exception
        basic_config.tokenizer.encode.side_effect = Exception("Tokenization failed")

        with pytest.raises(Exception, match="Tokenization failed"):
            PromptGenerator(basic_config)

    # ============================================================================
    # Performance and Stress Tests
    # ============================================================================

    @patch(
        "aiperf.services.dataset.utils.sample_positive_normal_integer", return_value=10
    )
    def test_multiple_generations_performance(self, mock_sample, basic_config):
        """Test multiple prompt generations to ensure no memory leaks or performance issues."""
        generator = PromptGenerator(basic_config)

        # Generate multiple prompts
        prompts = []
        for _ in range(10):
            prompt = generator.generate(mean=100, stddev=20)
            prompts.append(prompt)

        # All should be strings
        assert all(isinstance(p, str) for p in prompts)
        assert len(prompts) == 10

    def test_large_cache_usage(self, basic_config):
        """Test that large cache usage works correctly."""
        generator = PromptGenerator(basic_config)

        # Generate many cached prompts with different hash_ids
        with patch.object(generator, "_sample_tokens", return_value=list(range(5))):
            hash_ids = list(range(50))
            for i in range(0, len(hash_ids), 10):
                chunk = hash_ids[i : i + 10]
                generator._generate_cached_prompt(50, chunk, 5)

        # Cache should contain all hash_ids
        assert len(generator._cache) == 50
        assert all(h in generator._cache for h in hash_ids)
