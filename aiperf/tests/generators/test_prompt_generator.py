# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive unit tests for PromptGenerator class.

This test file provides complete coverage of all methods in the PromptGenerator class,
including edge cases, error conditions, and integration scenarios.
"""

from unittest.mock import mock_open, patch

import pytest

from aiperf.common.config import PrefixPromptConfig, PromptConfig
from aiperf.common.exceptions import (
    ConfigurationError,
    InvalidStateError,
    NotInitializedError,
)
from aiperf.services.dataset.generator.prompt import PromptGenerator

MOCK_CORPUS_CONTENT = "To be or not to be, that is the question.\nWhether 'tis nobler in the mind to suffer.\n"


@patch("builtins.open", mock_open(read_data=MOCK_CORPUS_CONTENT))
@pytest.mark.asyncio
class TestPromptGeneratorComprehensive:
    """Comprehensive test suite for PromptGenerator class."""

    @pytest.fixture
    def mock_tokenizer(self, mock_tokenizer_cls):
        """Mock tokenizer class for testing."""
        return mock_tokenizer_cls.from_pretrained("gpt2")

    @pytest.fixture
    def basic_config(self, mock_tokenizer):
        """Basic configuration for testing."""
        config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )
        return mock_tokenizer, config

    @pytest.fixture
    def prefix_config(self, mock_tokenizer):
        """Configuration with prefix prompt pool."""
        config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=5, length=10),
        )
        return mock_tokenizer, config

    @pytest.fixture
    async def initialized_generator(self, basic_config):
        """Initialized PromptGenerator for testing."""
        tokenizer, config = basic_config
        generator = PromptGenerator(config, tokenizer)
        await generator.initialize()
        return generator

    # ============================================================================
    # Initialization Tests
    # ============================================================================

    async def test_init_basic_configuration(self, basic_config):
        """Test basic initialization without prefix prompts."""
        tokenizer, config = basic_config
        generator = PromptGenerator(config, tokenizer)

        assert generator.config == config
        assert generator.tokenizer == tokenizer
        assert generator._tokenized_corpus is None
        assert generator._corpus_size == 0
        assert len(generator._prefix_prompts) == 0
        assert len(generator._cache) == 0
        assert not generator.data_initialized.is_set()

    async def test_initialize_with_prefix_prompts(self, prefix_config):
        """Test initialization of both text corpus and prefix prompt pool."""
        tokenizer, config = prefix_config
        generator = PromptGenerator(config, tokenizer)
        await generator.initialize()

        assert generator.data_initialized.is_set()
        assert generator._tokenized_corpus is not None
        assert generator._corpus_size > 0
        assert len(generator._prefix_prompts) == 5
        assert all(isinstance(prompt, str) for prompt in generator._prefix_prompts)

    @pytest.mark.parametrize(
        "length, pool_size, enabled", [(0, 5, False), (10, 0, False), (10, 5, True)]
    )
    async def test_initialize_prefix_prompt_enabled(
        self, prefix_config, length, pool_size, enabled
    ):
        """Test initialize with zero length prefix prompts."""
        tokenizer, config = prefix_config
        config.prefix_prompt.length = length
        config.prefix_prompt.pool_size = pool_size

        generator = PromptGenerator(config, tokenizer)
        await generator.initialize()

        assert generator.prefix_prompt_enabled is enabled

        expected_pool_size = pool_size if enabled else 0
        assert len(generator._prefix_prompts) == expected_pool_size

    # ============================================================================
    # Generate Method Tests
    # ============================================================================

    @pytest.mark.parametrize("mean", [10, 20, 123])
    async def test_generate_various_lengths(self, mean, initialized_generator):
        """Test generate method without hash_ids uses normal generation."""
        result = await initialized_generator.generate(mean=mean, stddev=0)
        assert result.startswith("token_")
        assert len(result.split(" ")) == mean

    async def test_generate_with_hash_ids(self, initialized_generator):
        """Test generate method with hash_ids uses cached generation."""
        assert len(initialized_generator._cache) == 0

        initialized_generator.config.input_tokens.block_size = 10  # fix block size
        result = await initialized_generator.generate(
            mean=30, stddev=0, hash_ids=[1, 2, 3]
        )

        assert len(initialized_generator._cache) == 3
        assert len(result.split(" ")) == 30

    async def test_generate_with_empty_hash_ids(self, initialized_generator):
        """Test generate method with empty hash_ids list."""
        result = await initialized_generator.generate(mean=123, stddev=0, hash_ids=[])
        assert len(initialized_generator._cache) == 0  # no cache is created
        assert len(result.split(" ")) == 123

    # ============================================================================
    # _generate_cached_prompt Method Tests
    # ============================================================================

    async def test_generate_cached_prompt_valid_parameters(self, basic_config):
        """Test _generate_cached_prompt with valid parameters."""
        tokenizer, config = basic_config
        generator = PromptGenerator(config, tokenizer)

        with patch.object(
            generator, "_sample_tokens", return_value=[10, 11, 12, 13, 14]
        ):
            result = generator._generate_cached_prompt(
                num_tokens=10, hash_ids=[1, 2], block_size=5
            )

            # Should have created cache entries
            assert 1 in generator._cache
            assert 2 in generator._cache

            # Each cache entry should have BOS token at start
            assert generator._cache[1][0] == 1  # BOS token
            assert generator._cache[2][0] == 1  # BOS token

            # Should return decoded prompt
            assert isinstance(result, str)

    async def test_generate_cached_prompt_reuse_cache(self, basic_config):
        """Test _generate_cached_prompt reuses existing cache entries."""
        tokenizer, config = basic_config
        generator = PromptGenerator(config, tokenizer)

        # Pre-populate cache
        generator._cache[1] = [1, 10, 11, 12, 13]

        with patch.object(
            generator, "_sample_tokens", return_value=[20, 21, 22, 23, 24]
        ) as mock_sample:
            _ = generator._generate_cached_prompt(
                num_tokens=10, hash_ids=[1, 2], block_size=5
            )

            # Should only sample tokens for new hash_id (2)
            mock_sample.assert_called_once_with(5)

            # Should reuse existing cache for hash_id 1
            assert generator._cache[1] == [1, 10, 11, 12, 13]

    async def test_generate_cached_prompt_uneven_final_block(self, basic_config):
        """Test _generate_cached_prompt with uneven final block size."""
        tokenizer, config = basic_config
        generator = PromptGenerator(config, tokenizer)

        with patch.object(
            generator, "_sample_tokens", side_effect=lambda n: list(range(n))
        ):
            _ = generator._generate_cached_prompt(
                num_tokens=12,  # 5 + 5 + 2
                hash_ids=[1, 2, 3],
                block_size=5,
            )

            # Final block should have different size
            assert len(generator._cache[3]) == 2  # Final block: 12 - (2 * 5) = 2

    @pytest.mark.parametrize(
        "num_tokens, hash_ids, block_size, should_raise",
        [
            # Failing cases
            (10, [1, 2, 3], 5, True),  # final_block_size = 0 (should fail)
            (5, [1, 2, 3], 5, True),  # final_block_size = -5 (should fail)
            (20, [1, 2], 5, True),  # final_block_size = 15 > block_size (should fail)
            (0, [1], 5, True),  # final_block_size = 0 (should fail)
            (10, [1, 2, 3], 0, True),  # block_size = 0 (should fail)
            (10, [1, 2, 3], -1, True),  # negative block_size (should fail)
            # Passing cases
            (10, [1, 2], 5, False),  # final_block_size == block_size
            (10, [1], 15, False),  # final_block_size < block_size
            (6, [1, 2], 5, False),  # final_block_size < block_size
            (5, [1], 5, False),  # final_block_size == block_size
            (3, [1], 5, False),  # final_block_size < block_size
            (12, [1, 2, 3], 5, False),  # final_block_size < block_size
        ],
    )
    async def test_generate_cached_prompt_configuration_errors(
        self, num_tokens, hash_ids, block_size, should_raise, initialized_generator
    ):
        """Test GeneratorConfigurationErrors for both passing and failing cases."""
        if should_raise:
            with pytest.raises(ConfigurationError) as exc_info:
                initialized_generator._generate_cached_prompt(
                    num_tokens=num_tokens, hash_ids=hash_ids, block_size=block_size
                )

            # Verify error message contains expected information
            error_message = str(exc_info.value)
            assert "are not compatible" in error_message
            assert f"Input length: {num_tokens}" in error_message
            assert f"Hash IDs: {hash_ids}" in error_message
            assert f"Block size: {block_size}" in error_message
        else:
            _ = initialized_generator._generate_cached_prompt(
                num_tokens=num_tokens, hash_ids=hash_ids, block_size=block_size
            )

    async def test_generate_cached_prompt_bos_token_insertion(
        self, initialized_generator
    ):
        """Test that BOS token is correctly inserted in cached prompts."""
        original_tokens = [10, 11, 12, 13, 14]
        with patch.object(
            initialized_generator, "_sample_tokens", return_value=original_tokens.copy()
        ):
            initialized_generator._generate_cached_prompt(5, [1], 5)

            # First token should be BOS token (1)
            assert initialized_generator._cache[1][0] == 1
            # Length should be maintained (5 tokens)
            assert len(initialized_generator._cache[1]) == 5
            # Should contain the other tokens (original[1:] + [BOS])
            assert initialized_generator._cache[1][1:] == original_tokens[1:]

    async def test_cache_reuse_across_calls(self, initialized_generator):
        """Test that cache is reused across multiple calls."""
        with patch.object(
            initialized_generator, "_sample_tokens", return_value=[10, 11, 12, 13, 14]
        ):
            # First call
            initialized_generator._generate_cached_prompt(10, [1, 2], 5)
            first_cache_1 = initialized_generator._cache[1].copy()
            first_cache_2 = initialized_generator._cache[2].copy()

            # Second call with same hash_ids
            initialized_generator._generate_cached_prompt(10, [1, 2], 5)

            # Cache should be reused (same values)
            assert initialized_generator._cache[1] == first_cache_1
            assert initialized_generator._cache[2] == first_cache_2

    async def test_mixed_cache_and_new_generation(self, initialized_generator):
        """Test mixing cached and new hash IDs in same call."""
        # Pre-populate cache with one hash_id
        initialized_generator._cache[1] = [1, 10, 11, 12, 13]

        with patch.object(
            initialized_generator, "_sample_tokens", return_value=[20, 21, 22, 23, 24]
        ):
            # Call with mix of cached and new hash_ids
            _ = initialized_generator._generate_cached_prompt(15, [1, 2, 3], 5)

            # Should reuse hash_id 1 and create new for 2 and 3
            assert initialized_generator._cache[1] == [1, 10, 11, 12, 13]  # Unchanged
            assert 2 in initialized_generator._cache  # Newly created
            assert 3 in initialized_generator._cache  # Newly created

    async def test_large_cache_usage(self, initialized_generator):
        """Test that large cache usage works correctly."""
        # Generate many cached prompts with different hash_ids
        block_size = 5
        with patch.object(
            initialized_generator,
            "_sample_tokens",
            return_value=list(range(block_size)),
        ):
            hash_ids = list(range(50))
            for i in range(0, len(hash_ids), 10):
                chunk = hash_ids[i : i + 10]
                initialized_generator._generate_cached_prompt(50, chunk, block_size)

        # Cache should contain all hash_ids
        assert len(initialized_generator._cache) == len(hash_ids)
        assert all(h in initialized_generator._cache for h in hash_ids)
        assert all(len(initialized_generator._cache[h]) == block_size for h in hash_ids)

    # ============================================================================
    # _sample_tokens Method Tests
    # ============================================================================

    @pytest.mark.parametrize("num_tokens", [3, 10, 100])
    async def test_sample_tokens(self, num_tokens, initialized_generator):
        """Test _sample_tokens with normal parameters."""
        tokens = initialized_generator._sample_tokens(num_tokens)

        assert len(tokens) == num_tokens

    async def test_sample_tokens_wrap_around(self, initialized_generator):
        """Test _sample_tokens when it needs to wrap around the corpus."""
        corpus_size = initialized_generator._corpus_size

        # Start near the end to force wrap-around
        with patch("random.randrange", return_value=corpus_size - 2):
            tokens = initialized_generator._sample_tokens(5)
            expected_tokens = (
                initialized_generator._tokenized_corpus[corpus_size - 2 : corpus_size]
                + initialized_generator._tokenized_corpus[:3]
            )
            assert len(tokens) == 5
            assert tokens == expected_tokens

    async def test_sample_tokens_exact_corpus_size(self, initialized_generator):
        """Test _sample_tokens when requesting exactly corpus size."""
        corpus_size = initialized_generator._corpus_size

        with patch("random.randrange", return_value=0):
            tokens = initialized_generator._sample_tokens(corpus_size)

            assert len(tokens) == corpus_size
            assert tokens == initialized_generator._tokenized_corpus

    @patch("aiperf.services.dataset.generator.prompt.logger.warning")
    @patch("random.randrange", return_value=0)
    async def test_sample_tokens_longer_than_corpus_with_warning(
        self, mock_randrange, mock_warning, initialized_generator
    ):
        """Test _sample_tokens when requested length exceeds corpus size."""
        corpus_size = initialized_generator._corpus_size

        tokens = initialized_generator._sample_tokens(corpus_size * 2)

        # Should log a warning
        mock_warning.assert_called_once()
        assert "longer than the corpus" in str(mock_warning.call_args)
        assert len(tokens) == corpus_size * 2

    async def test_sample_tokens_empty_corpus(self, initialized_generator):
        """Test _sample_tokens with empty corpus."""
        initialized_generator._tokenized_corpus = []
        initialized_generator._corpus_size = 0

        with pytest.raises(NotInitializedError):
            initialized_generator._sample_tokens(5)

    # ============================================================================
    # sample_prefix_prompt Method Tests
    # ============================================================================

    async def test_sample_prefix_prompt(self, prefix_config):
        """Test sample_prefix_prompt with populated pool."""
        tokenizer, config = prefix_config
        generator = PromptGenerator(config, tokenizer)
        await generator.initialize()

        assert len(generator._prefix_prompts) == 5

        # Should not raise an error
        _ = await generator.sample_prefix_prompt()

    async def test_sample_prefix_prompt_empty_pool(self, initialized_generator):
        """Test sample_prefix_prompt with empty pool."""
        with pytest.raises(InvalidStateError):
            await initialized_generator.sample_prefix_prompt()

    # ============================================================================
    # _initialize_corpus Method Tests
    # ============================================================================

    @patch("os.cpu_count", return_value=4)
    async def test_initialize_corpus_success(self, mock_cpu_count, basic_config):
        """Test _initialize_corpus method successful execution."""
        tokenizer, config = basic_config
        generator = PromptGenerator(config, tokenizer)

        assert generator._tokenized_corpus is None
        assert generator._corpus_size == 0

        await generator.initialize()

        assert generator._tokenized_corpus is not None
        assert generator._corpus_size > 0

    # ============================================================================
    # _create_prefix_prompt_pool Method Tests
    # ============================================================================

    async def test_create_prefix_prompt_pool_no_corpus(self, prefix_config):
        """Test _create_prefix_prompt_pool when corpus is not initialized."""
        tokenizer, config = prefix_config
        generator = PromptGenerator(config, tokenizer)
        generator._tokenized_corpus = None

        with pytest.raises(NotInitializedError):
            generator._create_prefix_prompt_pool()
