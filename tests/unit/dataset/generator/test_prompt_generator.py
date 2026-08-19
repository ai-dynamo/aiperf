# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive unit tests for PromptGenerator class.

This test file provides complete coverage of all methods in the PromptGenerator class,
including edge cases, error conditions, and integration scenarios.
"""

from unittest.mock import mock_open, patch

import pytest

from aiperf.common.enums import PromptCorpus
from aiperf.common.exceptions import (
    ConfigurationError,
    InvalidStateError,
    NotInitializedError,
)
from aiperf.config.dataset.content import PrefixPromptConfig, PromptConfig
from aiperf.dataset.generator.prompt import PromptGenerator

MOCK_CORPUS_CONTENT = "To be or not to be, that is the question.\nWhether 'tis nobler in the mind to suffer.\n"


def _make_generator(
    tokenizer,
    *,
    prompts: PromptConfig | None = None,
    prefix_prompts: PrefixPromptConfig | None = None,
) -> PromptGenerator:
    """Construct a PromptGenerator with the v2 keyword-only signature."""
    return PromptGenerator(
        prompts=prompts,
        prefix_prompts=prefix_prompts,
        tokenizer=tokenizer,
    )


@patch("builtins.open", mock_open(read_data=MOCK_CORPUS_CONTENT))
class TestPromptGeneratorComprehensive:
    """Comprehensive test suite for PromptGenerator class."""

    @pytest.fixture
    def mock_tokenizer(self, mock_tokenizer_cls):
        """Mock tokenizer class for testing."""
        return mock_tokenizer_cls.from_pretrained("gpt2")

    @pytest.fixture
    def basic_config(self, mock_tokenizer):
        """Basic configuration for testing (no prefix prompt pool)."""
        prompts = PromptConfig(block_size=512)
        prefix_prompts = PrefixPromptConfig(pool_size=None, length=None)
        return mock_tokenizer, prompts, prefix_prompts

    @pytest.fixture
    def prefix_config(self, mock_tokenizer):
        """Configuration with prefix prompt pool."""
        prompts = PromptConfig(block_size=512)
        prefix_prompts = PrefixPromptConfig(pool_size=5, length=10)
        return mock_tokenizer, prompts, prefix_prompts

    # ============================================================================
    # Initialization Tests
    # ============================================================================

    def test_init_basic_configuration(self, basic_config):
        """Test basic initialization without prefix prompts."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        assert generator.prompts == prompts
        assert generator.prefix_prompts == prefix_prompts
        assert generator.tokenizer == tokenizer
        assert generator._tokenized_corpus is not None
        assert generator._corpus_size > 0
        assert len(generator._prefix_prompts) == 0
        assert len(generator._cache) == 0

    def test_init_with_prefix_prompts(self, prefix_config):
        """Test initialization with prefix prompt pool."""
        tokenizer, prompts, prefix_prompts = prefix_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        assert len(generator._prefix_prompts) == 5
        assert all(isinstance(prompt, str) for prompt in generator._prefix_prompts)

    def test_init_corpus_initialization(self, basic_config):
        """Test that corpus is properly initialized during __init__."""
        with patch.object(PromptGenerator, "_initialize_corpus") as mock_init:
            tokenizer, prompts, prefix_prompts = basic_config
            _ = _make_generator(
                tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
            )
            mock_init.assert_called_once()

    # ============================================================================
    # Generate Method Tests
    # ============================================================================

    def test_generate_without_hash_ids(self, basic_config):
        """Test generate method without hash_ids uses normal generation."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        # Test that generate without hash_ids returns a string
        result = generator.generate(mean=100, stddev=20)

        assert isinstance(result, str)
        assert len(result) > 0
        # Verify it contains tokens from the corpus
        assert " " in result or len(result.split()) > 0

    def test_generate_with_hash_ids(self, basic_config):
        """Test generate method with hash_ids uses cached generation."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        with patch.object(
            generator, "_generate_cached_prompt", return_value="cached prompt"
        ) as mock_cached:
            result = generator.generate(mean=100, stddev=20, hash_ids=[1, 2, 3])

            mock_cached.assert_called_once_with(100, [1, 2, 3], 512)
            assert result == "cached prompt"

    def test_generate_with_empty_hash_ids(self, basic_config):
        """Test generate method with empty hash_ids list."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        # Empty list should be falsy, so should use normal generation
        result = generator.generate(mean=100, stddev=20, hash_ids=[])

        # Verify it returns a string with tokens
        assert isinstance(result, str)
        assert len(result) > 0

    # ============================================================================
    # generate_prompt Method Tests
    # ============================================================================

    def testgenerate_prompt_normal_case(self, basic_config):
        """Test generate_prompt method with normal parameters."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        result = generator.generate_prompt(3)
        assert result.startswith("token_")

    def testgenerate_prompt_zero_tokens(self, basic_config):
        """generate_prompt raises on num_tokens <= 0 rather than silently returning empty."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        with pytest.raises(ValueError, match="num_tokens must be > 0"):
            generator.generate_prompt(0)

    def testgenerate_prompt_large_number(self, basic_config):
        """Test generate_prompt with large number of tokens."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        generator.generate_prompt(1000)

    def test_generate_prompt_retries_when_encode_drifts_long(self, basic_config):
        """generate_prompt trims and retries when re-encode returns too many tokens."""
        tokenizer, prompts, prefix_prompts = basic_config

        # First encode call returns 2 extra tokens; subsequent calls are accurate.
        real_encode = tokenizer._mock_encode
        call_count = {"n": 0}

        def drifting_encode(text, **kwargs):
            call_count["n"] += 1
            ids = real_encode(text, **kwargs)
            if call_count["n"] == 1:
                return ids + [998, 999]
            return ids

        tokenizer.encode = drifting_encode
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        result = generator.generate_prompt(3)
        assert isinstance(result, str)
        assert call_count["n"] >= 2

    def test_generate_prompt_accepts_mismatch_after_budget(self, basic_config):
        """generate_prompt accepts result after exhausting _max_retries."""
        tokenizer, prompts, prefix_prompts = basic_config

        # encode always returns one extra token — convergence is impossible.
        real_encode = tokenizer._mock_encode

        def always_drifting_encode(text, **kwargs):
            return real_encode(text, **kwargs) + [999]

        tokenizer.encode = always_drifting_encode
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        result = generator.generate_prompt(3, _max_retries=3)
        assert isinstance(result, str)

    # ============================================================================
    # _generate_cached_prompt Method Tests
    # ============================================================================

    def test_generate_cached_prompt_valid_parameters(self, basic_config):
        """Test _generate_cached_prompt with valid parameters."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

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

    def test_generate_cached_prompt_reuse_cache(self, basic_config):
        """Test _generate_cached_prompt reuses existing cache entries."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        # Pre-populate cache
        generator._cache[1] = [1, 10, 11, 12, 13]

        _ = generator._generate_cached_prompt(
            num_tokens=10, hash_ids=[1, 2], block_size=5
        )

        # Should reuse existing cache for hash_id 1
        assert generator._cache[1] == [1, 10, 11, 12, 13]

    def test_generate_cached_prompt_uneven_final_block(self, basic_config):
        """Test _generate_cached_prompt with uneven final block size."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        _ = generator._generate_cached_prompt(
            num_tokens=12,  # 5 + 5 + 2
            hash_ids=[1, 2, 3],
            block_size=5,
        )

        # Final block should have different size
        assert len(generator._cache[3]) == 2  # Final block: 12 - (2 * 5) = 2

    def test_generate_cached_prompt_same_hash_two_sizes_raises(self, basic_config):
        """A hash_id materialized at one size then requested at another is a
        corrupt-trace signal and must hard-error, not silently resize."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        # First materialize hash_id 7 as a full 5-token block.
        generator._generate_cached_prompt(num_tokens=5, hash_ids=[7], block_size=5)
        # Now hash_id 7 is the final partial (3 tokens) of a 8-token prompt.
        with pytest.raises(ConfigurationError, match="single fixed block size"):
            generator._generate_cached_prompt(
                num_tokens=8, hash_ids=[1, 7], block_size=5
            )

    @pytest.mark.parametrize(
        "num_tokens, hash_ids, block_size, should_raise",
        [
            # Failing cases
            (10, [1, 2, 3], 5, True),  # final_block_size = 0 (should fail)
            (5, [1, 2, 3], 5, True),  # final_block_size = -5 (should fail)
            # Prefix-only layout: M*block_size=10 < num_tokens=20, the un-hashed
            # remainder is a 10-token fresh tail. Valid since the
            # ``_build_token_sequence`` rewrite (real captured traces list only
            # the cached prefix in hash_ids).
            (20, [1, 2], 5, False),
            (0, [1], 5, True),  # num_tokens = 0 (should fail)
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
    def test_generate_cached_prompt_configuration_errors(
        self, num_tokens, hash_ids, block_size, should_raise, basic_config
    ):
        """Test GeneratorConfigurationErrors for both passing and failing cases."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        if should_raise:
            with pytest.raises(ConfigurationError) as exc_info:
                generator._generate_cached_prompt(
                    num_tokens=num_tokens, hash_ids=hash_ids, block_size=block_size
                )

            # Verify error message contains expected information
            error_message = str(exc_info.value)
            assert "are not compatible" in error_message
            assert f"Input length: {num_tokens}" in error_message
            assert f"Hash IDs: {hash_ids}" in error_message
            assert f"Block size: {block_size}" in error_message
        else:
            _ = generator._generate_cached_prompt(
                num_tokens=num_tokens, hash_ids=hash_ids, block_size=block_size
            )

    def test_generate_cached_prompt_bos_token_insertion(self, basic_config):
        """Test that BOS token is correctly inserted in cached prompts."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        cache_size = 5
        generator._generate_cached_prompt(
            num_tokens=cache_size, hash_ids=[1], block_size=5
        )

        assert len(generator._cache[1]) == cache_size
        assert generator._cache[1][0] in [
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        ]

    def test_cache_reuse_across_calls(self, basic_config):
        """Test that cache is reused across multiple calls."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

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
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        # Pre-populate cache with one hash_id
        generator._cache[1] = [1, 10, 11, 12, 13]

        # Call with mix of cached and new hash_ids
        _ = generator._generate_cached_prompt(15, [1, 2, 3], 5)

        # Should reuse hash_id 1 and create new for 2 and 3
        assert generator._cache[1] == [1, 10, 11, 12, 13]  # Unchanged
        assert 2 in generator._cache  # Newly created
        assert 3 in generator._cache  # Newly created

    def test_large_cache_usage(self, basic_config):
        """Test that large cache usage works correctly."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        # Generate many cached prompts with different hash_ids
        block_size = 5
        hash_ids = list(range(50))
        for i in range(0, len(hash_ids), 10):
            chunk = hash_ids[i : i + 10]
            generator._generate_cached_prompt(50, chunk, block_size)

        # Cache should contain all hash_ids
        assert len(generator._cache) == len(hash_ids)
        assert all(h in generator._cache for h in hash_ids)
        assert all(len(generator._cache[h]) == block_size for h in hash_ids)

    # ============================================================================
    # _sample_tokens Method Tests
    # ============================================================================

    def test_sample_tokens_normal_case(self, basic_config):
        """Test _sample_tokens with normal parameters."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        with patch.object(generator._corpus_rng, "randrange", return_value=5):
            tokens = generator._sample_tokens(3)

            assert len(tokens) == 3
            assert all(isinstance(t, int) for t in tokens)

    def test_sample_tokens_wrap_around(self, basic_config):
        """Test _sample_tokens when it needs to wrap around the corpus."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )
        corpus_size = generator._corpus_size

        # Start near the end to force wrap-around
        with patch.object(
            generator._corpus_rng, "randrange", return_value=corpus_size - 2
        ):
            tokens = generator._sample_tokens(5)
            expected_tokens = (
                generator._tokenized_corpus[corpus_size - 2 : corpus_size]
                + generator._tokenized_corpus[:3]
            )
            assert len(tokens) == 5
            assert tokens == expected_tokens

    def test_sample_tokens_exact_corpus_size(self, basic_config):
        """Test _sample_tokens when requesting exactly corpus size."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )
        corpus_size = generator._corpus_size

        with patch.object(generator._corpus_rng, "randrange", return_value=0):
            tokens = generator._sample_tokens(corpus_size)

            assert len(tokens) == corpus_size
            assert tokens == generator._tokenized_corpus

    @patch("aiperf.common.mixins.aiperf_logger_mixin.AIPerfLoggerMixin.warning")
    def test_sample_tokens_longer_than_corpus_with_warning(
        self, mock_warning, basic_config
    ):
        """Test _sample_tokens when requested length exceeds corpus size."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )
        corpus_size = generator._corpus_size

        with patch.object(generator._corpus_rng, "randrange", return_value=0):
            tokens = generator._sample_tokens(corpus_size * 2)

        # Should log a warning
        mock_warning.assert_called_once()
        assert "longer than the corpus" in str(mock_warning.call_args)
        assert len(tokens) == corpus_size * 2

    def test_sample_tokens_empty_corpus(self, basic_config):
        """Test _sample_tokens with empty corpus."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )
        generator._tokenized_corpus = []
        generator._corpus_size = 0

        with pytest.raises(NotInitializedError):
            generator._sample_tokens(5)

    # ============================================================================
    # get_random_prefix_prompt Method Tests
    # ============================================================================

    def test_get_random_prefix_prompt_success(self, prefix_config):
        """Test get_random_prefix_prompt with populated pool."""
        tokenizer, prompts, prefix_prompts = prefix_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        # Test that it returns one of the prefix prompts from the pool
        result = generator.get_random_prefix_prompt()
        assert isinstance(result, str)
        assert len(result) > 0
        # Verify it's from the prefix prompts pool
        assert result in generator._prefix_prompts

    def test_get_random_prefix_prompt_multiple_calls(self, prefix_config):
        """Test get_random_prefix_prompt returns different prompts across calls."""
        tokenizer, prompts, prefix_prompts = prefix_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

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
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        with pytest.raises(InvalidStateError):
            generator.get_random_prefix_prompt()

    # ============================================================================
    # _initialize_corpus Method Tests
    # ============================================================================

    @patch("os.cpu_count", return_value=4)
    def test_initialize_corpus_success(self, mock_cpu_count, basic_config):
        """Test _initialize_corpus method successful execution."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        assert generator._tokenized_corpus is not None
        assert generator._corpus_size > 0
        assert isinstance(generator._tokenized_corpus, list)
        assert all(isinstance(token, int) for token in generator._tokenized_corpus)

    # ============================================================================
    # _create_prefix_prompt_pool Method Tests
    # ============================================================================

    def test_create_prefix_prompt_pool_success(self, prefix_config):
        """Test _create_prefix_prompt_pool successful creation."""
        tokenizer, prompts, prefix_prompts = prefix_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        assert len(generator._prefix_prompts) == 5
        assert all(isinstance(prompt, str) for prompt in generator._prefix_prompts)

    def test_create_prefix_prompt_pool_no_corpus(self, prefix_config):
        """Test _create_prefix_prompt_pool when corpus is not initialized."""
        tokenizer, prompts, prefix_prompts = prefix_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )
        generator._tokenized_corpus = None

        with pytest.raises(NotInitializedError):
            generator._create_prefix_prompt_pool()

    def test_create_prefix_prompt_pool_zero_length(self, mock_tokenizer):
        """Test _create_prefix_prompt_pool with zero length prompts.

        v2 PrefixPromptConfig requires length >= 1, so we mutate the value
        post-init via Pydantic's allow-attribute-assignment behavior; if the
        config rejects 0 we test the equivalent code path where length is
        treated as falsy and pool entries are empty strings.
        """
        prompts = PromptConfig(block_size=512)
        prefix_prompts = PrefixPromptConfig(pool_size=5, length=1)
        generator = _make_generator(
            mock_tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        # Force length to 0 and rebuild the pool to mirror legacy behavior.
        generator.prefix_prompts = PrefixPromptConfig.model_construct(
            pool_size=5, length=0
        )
        generator._prefix_prompts = []
        generator._create_prefix_prompt_pool()

        assert len(generator._prefix_prompts) == 5
        assert all(prompt == "" for prompt in generator._prefix_prompts)

    # ============================================================================
    # Shared System Prompt Tests
    # ============================================================================

    def test_generate_shared_system_prompt_success(self, mock_tokenizer):
        """Test _generate_shared_system_prompt generates prompt successfully."""
        prompts = PromptConfig(block_size=512)
        prefix_prompts = PrefixPromptConfig(shared_system_length=50)
        generator = _make_generator(
            mock_tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        assert generator._shared_system_prompt is not None
        assert isinstance(generator._shared_system_prompt, str)
        assert len(generator._shared_system_prompt) > 0

    def test_generate_shared_system_prompt_none_when_not_configured(
        self, mock_tokenizer
    ):
        """Test _generate_shared_system_prompt does nothing when not configured."""
        prompts = PromptConfig(block_size=512)
        prefix_prompts = PrefixPromptConfig(shared_system_length=None)
        generator = _make_generator(
            mock_tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        assert generator._shared_system_prompt is None

    def test_get_shared_system_prompt_success(self, mock_tokenizer):
        """Test get_shared_system_prompt returns the prompt."""
        prompts = PromptConfig(block_size=512)
        prefix_prompts = PrefixPromptConfig(shared_system_length=50)
        generator = _make_generator(
            mock_tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        result = generator.get_shared_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == generator._shared_system_prompt

    def test_get_shared_system_prompt_not_initialized(self, mock_tokenizer):
        """Test get_shared_system_prompt raises error when not initialized."""
        prompts = PromptConfig(block_size=512)
        prefix_prompts = PrefixPromptConfig(shared_system_length=None)
        generator = _make_generator(
            mock_tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        with pytest.raises(InvalidStateError) as exc_info:
            generator.get_shared_system_prompt()

        assert "not initialized" in str(exc_info.value)
        assert "shared-system-prompt-length" in str(exc_info.value)

    # ============================================================================
    # User Context Prompt Tests
    # ============================================================================

    def test_generate_user_context_prompt_first_session(self, mock_tokenizer):
        """Test generate_user_context_prompt for first session."""
        prompts = PromptConfig(block_size=512)
        prefix_prompts = PrefixPromptConfig(user_context_length=30)
        generator = _make_generator(
            mock_tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        result = generator.generate_user_context_prompt(0)
        assert isinstance(result, str)
        assert len(result) > 0
        assert len(generator._user_context_prompts) == 1

    def test_generate_user_context_prompt_multiple_sessions(self, mock_tokenizer):
        """Test generate_user_context_prompt generates unique prompts."""
        prompts = PromptConfig(block_size=512)
        prefix_prompts = PrefixPromptConfig(user_context_length=30)
        generator = _make_generator(
            mock_tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        prompt0 = generator.generate_user_context_prompt(0)
        prompt1 = generator.generate_user_context_prompt(1)
        prompt2 = generator.generate_user_context_prompt(2)

        assert len(generator._user_context_prompts) == 3
        assert prompt0 == generator._user_context_prompts[0]
        assert prompt1 == generator._user_context_prompts[1]
        assert prompt2 == generator._user_context_prompts[2]

    def test_generate_user_context_prompt_caching(self, mock_tokenizer):
        """Test generate_user_context_prompt returns cached prompt."""
        prompts = PromptConfig(block_size=512)
        prefix_prompts = PrefixPromptConfig(user_context_length=30)
        generator = _make_generator(
            mock_tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        # Generate prompt for session 0
        prompt0_first = generator.generate_user_context_prompt(0)

        # Request same session again - should return cached
        prompt0_second = generator.generate_user_context_prompt(0)

        assert prompt0_first == prompt0_second
        assert len(generator._user_context_prompts) == 1

    def test_generate_user_context_prompt_non_sequential_access(self, mock_tokenizer):
        """Test generate_user_context_prompt with non-sequential session indices."""
        prompts = PromptConfig(block_size=512)
        prefix_prompts = PrefixPromptConfig(user_context_length=30)
        generator = _make_generator(
            mock_tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        # Request session 5 directly (should generate 0-5)
        prompt5 = generator.generate_user_context_prompt(5)

        assert len(generator._user_context_prompts) == 6
        assert prompt5 == generator._user_context_prompts[5]

    def test_generate_user_context_prompt_not_configured(self, mock_tokenizer):
        """Test generate_user_context_prompt raises error when not configured."""
        prompts = PromptConfig(block_size=512)
        prefix_prompts = PrefixPromptConfig(user_context_length=None)
        generator = _make_generator(
            mock_tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        with pytest.raises(InvalidStateError) as exc_info:
            generator.generate_user_context_prompt(0)

        assert "not configured" in str(exc_info.value)
        assert "user-context-prompt-length" in str(exc_info.value)

    def test_generate_user_context_prompt_corpus_not_initialized(self, mock_tokenizer):
        """Test generate_user_context_prompt when corpus not initialized."""
        prompts = PromptConfig(block_size=512)
        prefix_prompts = PrefixPromptConfig(user_context_length=30)
        generator = _make_generator(
            mock_tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )
        generator._tokenized_corpus = None

        with pytest.raises(NotInitializedError) as exc_info:
            generator.generate_user_context_prompt(0)

        assert "corpus" in str(exc_info.value).lower()

    # ============================================================================
    # Decoded String Cache Tests
    # ============================================================================

    def test_decoded_cache_initialized_empty(self, basic_config):
        """Test that decoded cache is initialized as empty dict."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        assert hasattr(generator, "_decoded_cache")
        assert isinstance(generator._decoded_cache, dict)
        assert len(generator._decoded_cache) == 0

    # NOTE: ``_generate_cached_prompt`` no longer reads/writes ``_decoded_cache``.
    # The hash-id reseed path (WekaTraceLoader + the trace loaders) scopes block
    # content per-(trace_id, hash_id) and clears ``_cache`` between trace files,
    # so a cross-trace decoded-cache hit would serve stale bytes and break
    # byte-exact trace replay. The former decoded-cache population tests
    # (populated_on_first_call / hit_on_repeated_call / miss_* / key_structure)
    # are removed: the decode is now always fresh. ``_decoded_cache`` remains a
    # declared attribute (other call sites still reference it).

    # ============================================================================
    # _build_token_sequence Method Tests
    # ============================================================================

    def test_build_token_sequence_returns_tokens(self, basic_config):
        """Test that _build_token_sequence returns a list of token IDs."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        tokens = generator._build_token_sequence(10, [1, 2], 5)

        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) == 10

    def test_build_token_sequence_populates_cache(self, basic_config):
        """Test that _build_token_sequence populates the token block cache."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        _ = generator._build_token_sequence(10, [1, 2], 5)

        # Token block cache should be populated
        assert 1 in generator._cache
        assert 2 in generator._cache

    def test_build_token_sequence_does_not_populate_decoded_cache(self, basic_config):
        """Test that _build_token_sequence does NOT populate decoded cache."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        _ = generator._build_token_sequence(10, [1, 2], 5)

        # Decoded cache should remain empty
        assert len(generator._decoded_cache) == 0

    def test_build_token_sequence_same_validation_as_generate_cached(
        self, basic_config
    ):
        """Test that _build_token_sequence has same validation as _generate_cached_prompt."""
        tokenizer, prompts, prefix_prompts = basic_config
        generator = _make_generator(
            tokenizer, prompts=prompts, prefix_prompts=prefix_prompts
        )

        # This should raise same error as _generate_cached_prompt
        with pytest.raises(ConfigurationError):
            generator._build_token_sequence(10, [1, 2, 3], 5)  # final_block_size = 0


class TestPromptGeneratorRandomCorpus:
    """Tests for PromptCorpus.RANDOM generate_prompt top-up behavior."""

    @pytest.fixture
    def random_tokenizer(self, mock_tokenizer_cls):
        tok = mock_tokenizer_cls.from_pretrained("gpt2")
        tok._tokenizer.vocab_size = 100
        return tok

    @pytest.fixture
    def random_generator(self, random_tokenizer):
        return PromptGenerator(
            prompts=None,
            prefix_prompts=None,
            tokenizer=random_tokenizer,
            corpus=PromptCorpus.RANDOM,
        )

    def test_generate_prompt_random_retries_when_encode_is_short(
        self, random_tokenizer, random_generator
    ):
        """RANDOM generate_prompt re-enters the loop when BPE re-encode yields fewer tokens."""
        real_encode = random_tokenizer._mock_encode
        call_count = {"n": 0}

        def shrink_first(text, **kwargs):
            call_count["n"] += 1
            ids = real_encode(text, **kwargs)
            if call_count["n"] == 1:
                return ids[:-1]
            return ids

        random_tokenizer.encode = shrink_first

        result = random_generator.generate_prompt(3)

        assert isinstance(result, str)
        assert call_count["n"] >= 2

    def test_generate_prompt_random_converges_when_exact_on_first_try(
        self, random_tokenizer, random_generator
    ):
        """RANDOM generate_prompt exits after the first iteration when re-encode is exact."""
        real_encode = random_tokenizer._mock_encode
        call_count = {"n": 0}

        def counting_encode(text, **kwargs):
            call_count["n"] += 1
            return real_encode(text, **kwargs)

        random_tokenizer.encode = counting_encode

        random_generator.generate_prompt(3)

        assert call_count["n"] == 1

    def test_generate_prompt_random_trims_when_encode_is_long(
        self, random_tokenizer, random_generator
    ):
        """RANDOM generate_prompt trims and retries when BPE re-encode yields extra tokens."""
        real_encode = random_tokenizer._mock_encode
        call_count = {"n": 0}

        def grow_first(text, **kwargs):
            call_count["n"] += 1
            ids = real_encode(text, **kwargs)
            if call_count["n"] == 1:
                return ids + [998]
            return ids

        random_tokenizer.encode = grow_first

        result = random_generator.generate_prompt(3)

        assert isinstance(result, str)
        assert call_count["n"] >= 2

    def test_random_corpus_prefix_pool_does_not_raise(self, random_tokenizer):
        """RANDOM corpus can generate a prefix prompt pool without a text corpus."""
        from aiperf.config.dataset.content import PrefixPromptConfig

        generator = PromptGenerator(
            prompts=None,
            prefix_prompts=PrefixPromptConfig(pool_size=3, length=5),
            tokenizer=random_tokenizer,
            corpus=PromptCorpus.RANDOM,
        )
        assert len(generator._prefix_prompts) == 3
        assert all(isinstance(p, str) for p in generator._prefix_prompts)

    def test_random_corpus_shared_system_prompt_does_not_raise(self, random_tokenizer):
        """RANDOM corpus can generate a shared system prompt without a text corpus."""
        from aiperf.config.dataset.content import PrefixPromptConfig

        generator = PromptGenerator(
            prompts=None,
            prefix_prompts=PrefixPromptConfig(shared_system_length=10),
            tokenizer=random_tokenizer,
            corpus=PromptCorpus.RANDOM,
        )
        assert generator._shared_system_prompt is not None
        assert isinstance(generator._shared_system_prompt, str)

    def test_random_corpus_excludes_special_tokens_from_sampling_pool(
        self, mock_tokenizer_cls
    ):
        """_allowed_tokens must not contain any special token IDs."""
        from aiperf.common.enums import RandomCorpusStyle

        tok = mock_tokenizer_cls.from_pretrained("gpt2")
        tok._tokenizer.vocab_size = 100
        tok._tokenizer.all_special_ids = [1, 2]

        generator = PromptGenerator(
            prompts=None,
            prefix_prompts=None,
            tokenizer=tok,
            corpus=PromptCorpus.RANDOM,
            corpus_style=RandomCorpusStyle.VLLM,
        )

        allowed = set(generator._allowed_tokens)
        assert 1 not in allowed
        assert 2 not in allowed
        assert len(generator._allowed_tokens) == 98

    def test_random_corpus_sglang_uses_full_vocab(self, mock_tokenizer_cls):
        """SGLANG style uses all_token_ids (full vocab_size range, no special-token exclusion)."""
        from aiperf.common.enums import RandomCorpusStyle

        tok = mock_tokenizer_cls.from_pretrained("gpt2")
        tok._tokenizer.vocab_size = 100
        tok._tokenizer.all_special_ids = [1, 2]

        generator = PromptGenerator(
            prompts=None,
            prefix_prompts=None,
            tokenizer=tok,
            corpus=PromptCorpus.RANDOM,
            corpus_style=RandomCorpusStyle.SGLANG,
        )

        # SGLANG does not exclude special tokens — full range(vocab_size)
        assert len(generator._allowed_tokens) == 100

    def test_offset_cache_exhaustion_falls_back_to_live_draw(self, random_generator):
        """Preseed sizes the offset cache per conversation but it is read once
        per prompt, so multi-turn / batched / prefix-pool runs outrun it.

        Exhaustion must degrade to a live `_corpus_rng` draw rather than raising
        `IndexError` out of the dataset composer.
        """
        import numpy as np

        random_generator.preseed(2, np.random.default_rng(42))
        assert len(random_generator._offset_cache) == 2

        for _ in range(2):
            random_generator._sample_tokens(4)
        assert random_generator._offset_idx == 2

        # Third read is past the end of the cache.
        tokens = random_generator._sample_tokens(4)

        assert len(tokens) == 4
        assert all(t in random_generator._allowed_tokens for t in tokens)

    def test_offset_cache_exhaustion_warns_once(self, random_generator, caplog):
        """The fallback is silent-by-default degradation otherwise, so it warns
        exactly once naming the cause rather than per prompt."""
        import logging

        import numpy as np

        random_generator.preseed(1, np.random.default_rng(42))
        random_generator._sample_tokens(4)

        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                random_generator._sample_tokens(4)

        exhausted = [
            r for r in caplog.records if "offset cache exhausted" in r.getMessage()
        ]
        assert len(exhausted) == 1
        assert "consumed once per prompt" in exhausted[0].getMessage()

    def test_offset_cache_within_bounds_is_unaffected(self, random_generator):
        """Reads inside the cache must still come from the preseed stream."""
        import numpy as np

        random_generator.preseed(4, np.random.default_rng(42))
        expected = list(random_generator._offset_cache)

        random_generator._sample_tokens(4)
        random_generator._sample_tokens(4)

        assert random_generator._offset_idx == 2
        assert random_generator._offset_cache == expected
