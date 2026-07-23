# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CodingContentGenerator."""

import pytest

from aiperf.common import random_generator as rng
from aiperf.common.exceptions import ConfigurationError, NotInitializedError
from aiperf.config.dataset.content import PromptConfig
from aiperf.dataset.generator.coding_content import (
    _FILE_PATHS,
    _LANG_FILE_PATHS,
    _TOOL_POOL_BLOCK_COUNTS,
    CodingContentGenerator,
)


class TestCodingContentGeneratorInit:
    @pytest.fixture
    def config(self):
        return PromptConfig(block_size=512)

    @pytest.fixture
    def generator(self, config, mock_tokenizer_cls):
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_pools_built(self, generator):
        assert generator._text_pool is None
        assert len(generator._tool_pool) > 0

    def test_text_pool_lazy_build(self, generator):
        assert generator._text_pool is None
        pool = generator._ensure_text_pool()
        assert len(pool) > 0
        assert generator._text_pool is pool

    def test_tokenized_corpus_aliases_tool_pool(self, generator):
        assert generator._tokenized_corpus is generator._tool_pool

    def test_hash_id_corpus_rng_exists(self, generator):
        assert generator._hash_id_corpus_rng is not None

    def test_pool_scale(self, mock_tokenizer_cls):
        config = PromptConfig(block_size=512)
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        gen_default = CodingContentGenerator(config, tokenizer)
        gen_2x = CodingContentGenerator(
            config, tokenizer, pool_tokens_target=20_000_000
        )
        assert gen_2x._pool_scale == pytest.approx(2.0)
        assert gen_default._pool_scale == pytest.approx(1.0)


class TestGenerate:
    @pytest.fixture
    def config(self):
        return PromptConfig(block_size=512)

    @pytest.fixture
    def generator(self, config, mock_tokenizer_cls):
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_generate_without_hash_ids(self, generator):
        result = generator.generate(mean=100, stddev=20)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_hash_ids(self, generator):
        result = generator.generate(mean=100, hash_ids=[1, 2], block_size=50)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_missing_mean_raises_value_error(self, generator):
        with pytest.raises(ValueError, match="mean must be provided"):
            generator.generate(hash_ids=[1, 2])

    def test_generate_empty_hash_ids_uses_normal_path(self, generator):
        result = generator.generate(mean=50, stddev=10, hash_ids=[])
        assert isinstance(result, str)


class TestGeneratePrompt:
    @pytest.fixture
    def generator(self, mock_tokenizer_cls):
        config = PromptConfig(block_size=512)
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_returns_decoded_string(self, generator):
        result = generator.generate_prompt(50)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_zero_tokens(self, generator):
        result = generator.generate_prompt(0)
        assert result == ""


class TestBuildTokenSequence:
    @pytest.fixture
    def generator(self, mock_tokenizer_cls):
        config = PromptConfig(block_size=512)
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_correct_total_length(self, generator):
        tokens = generator._build_token_sequence(100, [1, 2], 50)
        assert len(tokens) == 100

    def test_caches_per_hash_id(self, generator):
        generator._build_token_sequence(100, [10, 20], 50)
        assert 10 in generator._cache
        assert 20 in generator._cache

    def test_reuses_cache(self, generator):
        generator._build_token_sequence(100, [10, 20], 50)
        cached_10 = generator._cache[10]
        generator._build_token_sequence(100, [10, 30], 50)
        assert generator._cache[10] is cached_10

    def test_incompatible_params_raise_configuration_error(self, generator):
        with pytest.raises(ConfigurationError):
            generator._build_token_sequence(10, [1, 2, 3, 4, 5], 50)

    def test_deterministic_per_hash_id(self, mock_tokenizer_cls):
        config = PromptConfig(block_size=512)
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        gen1 = CodingContentGenerator(config, tokenizer)
        gen2 = CodingContentGenerator(config, tokenizer)

        gen1._build_token_sequence(100, [42, 99], 50)
        gen2._build_token_sequence(100, [99, 42], 50)

        assert gen1._cache[42] == gen2._cache[42]
        assert gen1._cache[99] == gen2._cache[99]

    def test_uses_hash_id_corpus_rng(self, generator):
        generator._build_token_sequence(100, [7, 8], 50)
        assert 7 in generator._cache

    def test_prefix_only_pads_tail_instead_of_raising(self, generator):
        # input_length > len(hash_ids) * block_size: real captured traces
        # (e.g. bailian block_size=16) list only the cached prefix; the
        # un-hashed tail must be padded with sampled tokens, not raise.
        tokens = generator._build_token_sequence(100, [1, 2], 16)
        assert len(tokens) == 100

    def test_exact_tile_full_blocks(self, generator):
        tokens = generator._build_token_sequence(48, [1, 2, 3], 16)
        assert len(tokens) == 48

    def test_non_positive_num_tokens_raises(self, generator):
        with pytest.raises(ConfigurationError):
            generator._build_token_sequence(0, [1, 2], 16)


class TestSampleTokens:
    @pytest.fixture
    def generator(self, mock_tokenizer_cls):
        config = PromptConfig(block_size=512)
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_empty_pool_raises(self, generator):
        with pytest.raises(NotInitializedError):
            generator._sample_tokens(10, [])

    def test_wraps_around_pool_boundary(self, generator):
        pool = [1, 2, 3, 4, 5]
        tokens = generator._sample_tokens(7, pool)
        assert len(tokens) == 7


class TestTemplateSmoke:
    """Smoke tests for all template generators."""

    @pytest.fixture
    def generator(self, mock_tokenizer_cls):
        config = PromptConfig(block_size=512)
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    @pytest.mark.parametrize("gen_name", list(_TOOL_POOL_BLOCK_COUNTS.keys()))
    def test_tool_pool_generators(self, generator, gen_name):
        gen_fn = getattr(generator, gen_name)
        result = gen_fn()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_gen_user_prompt(self, generator):
        result = generator._gen_user_prompt()
        assert isinstance(result, str)
        assert len(result) > 0


class TestMLTemplates:
    """Tests for ML-specific generators produce realistic ML content."""

    @pytest.fixture
    def generator(self, mock_tokenizer_cls):
        config = PromptConfig(block_size=512)
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_ml_training_contains_torch(self, generator):
        result = generator._gen_ml_training_code()
        assert "torch" in result

    def test_ml_inference_contains_generate(self, generator):
        result = generator._gen_ml_inference_code()
        assert "generate" in result
        assert "torch" in result

    def test_ml_config_contains_model_path(self, generator):
        result = generator._gen_ml_config()
        assert "model_name_or_path" in result

    def test_ml_training_log_contains_loss(self, generator):
        result = generator._gen_ml_training_log()
        assert "loss" in result

    def test_cuda_error_contains_cuda(self, generator):
        result = generator._gen_cuda_error()
        assert "CUDA" in result or "cuda" in result

    def test_sql_query_contains_sql_keywords(self, generator):
        result = generator._gen_sql_query()
        text_upper = result.upper()
        has_sql = any(
            kw in text_upper for kw in ("SELECT", "INSERT", "CREATE", "ALTER")
        )
        assert has_sql


class TestFilePool:
    @pytest.fixture
    def generator(self, mock_tokenizer_cls):
        config = PromptConfig(block_size=512)
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_language_specific_paths(self, generator):
        for lang in ("python", "go", "rust", "typescript"):
            pool = generator._file_pool(lang)
            assert pool is _LANG_FILE_PATHS[lang]

    def test_generic_paths(self, generator):
        assert generator._file_pool(None) is _FILE_PATHS
        assert generator._file_pool("unknown") is _FILE_PATHS


class TestCodingConversation:
    @pytest.fixture
    def generator(self, mock_tokenizer_cls):
        config = PromptConfig(block_size=512)
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_coding_conversation_has_role_markers(self, generator):
        result = generator._gen_coding_conversation()
        assert "[User]" in result
        assert "[Assistant]" in result

    def test_coding_conversation_has_tool_calls(self, generator):
        result = generator._gen_coding_conversation()
        assert "<tool_name>" in result

    @pytest.mark.parametrize(
        "pattern_name",
        [
            "_gen_conv_bugfix",
            "_gen_conv_review",
            "_gen_conv_feature",
            "_gen_conv_debug",
            "_gen_conv_qa",
            "_gen_conv_refactor",
            "_gen_conv_perf",
            "_gen_conv_cicd",
            "_gen_conv_ml_debug",
            "_gen_conv_test_write",
            "_gen_conv_migration",
            "_gen_conv_deploy",
            "_gen_conv_security",
            "_gen_conv_distributed",
            "_gen_conv_observability",
            "_gen_conv_db_optimize",
            "_gen_conv_architecture_review",
            "_gen_conv_incident_response",
        ],
    )
    def test_coding_conversation_patterns_all_produce_output(
        self, generator, pattern_name
    ):
        gen_fn = getattr(generator, pattern_name)
        result = gen_fn()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "[User]" in result
        assert "[Assistant]" in result

    @pytest.mark.parametrize(
        "pattern_name",
        ["_gen_conv_architecture_review", "_gen_conv_incident_response"],
    )
    def test_coding_conversation_deep_patterns_have_long_turns(
        self, generator, pattern_name
    ):
        gen_fn = getattr(generator, pattern_name)
        result = gen_fn()
        assert len(result) > 2000


class TestSeedDeterminism:
    """Whole-corpus determinism guarantees driven by the global RNG seed.

    Per-hash-id determinism is covered by `test_deterministic_per_hash_id`.
    These tests cover the broader contract: building a generator twice under
    the same global seed yields byte-identical pools and outputs, while a
    different seed produces different pools (so any stray non-derived RNG
    inside the generator would be caught here).
    """

    @pytest.fixture
    def config(self):
        return PromptConfig(block_size=512)

    def _build_at_seed(self, config, mock_tokenizer_cls, seed: int):
        rng.reset()
        rng.init(seed)
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_same_seed_pools_identical(self, config, mock_tokenizer_cls):
        gen_a = self._build_at_seed(config, mock_tokenizer_cls, 42)
        text_a = list(gen_a._ensure_text_pool())
        tool_a = list(gen_a._tool_pool)

        gen_b = self._build_at_seed(config, mock_tokenizer_cls, 42)
        text_b = list(gen_b._ensure_text_pool())
        tool_b = list(gen_b._tool_pool)

        assert text_a == text_b
        assert tool_a == tool_b

    def test_different_seeds_pools_differ(self, config, mock_tokenizer_cls):
        gen_42 = self._build_at_seed(config, mock_tokenizer_cls, 42)
        gen_99 = self._build_at_seed(config, mock_tokenizer_cls, 99)

        assert list(gen_42._tool_pool) != list(gen_99._tool_pool)
        assert list(gen_42._ensure_text_pool()) != list(gen_99._ensure_text_pool())

    def test_same_seed_hash_id_output_identical(self, config, mock_tokenizer_cls):
        gen_a = self._build_at_seed(config, mock_tokenizer_cls, 42)
        out_a = gen_a.generate(mean=180, hash_ids=[1, 2, 3], block_size=64)

        gen_b = self._build_at_seed(config, mock_tokenizer_cls, 42)
        out_b = gen_b.generate(mean=180, hash_ids=[1, 2, 3], block_size=64)

        assert out_a == out_b

    def test_different_seeds_hash_id_output_differs(self, config, mock_tokenizer_cls):
        gen_42 = self._build_at_seed(config, mock_tokenizer_cls, 42)
        gen_99 = self._build_at_seed(config, mock_tokenizer_cls, 99)

        out_42 = gen_42.generate(mean=180, hash_ids=[1, 2, 3], block_size=64)
        out_99 = gen_99.generate(mean=180, hash_ids=[1, 2, 3], block_size=64)

        assert out_42 != out_99

    def test_same_seed_generate_prompt_identical(self, config, mock_tokenizer_cls):
        gen_a = self._build_at_seed(config, mock_tokenizer_cls, 42)
        prompt_a = gen_a.generate_prompt(150)

        gen_b = self._build_at_seed(config, mock_tokenizer_cls, 42)
        prompt_b = gen_b.generate_prompt(150)

        assert prompt_a == prompt_b


class TestCodingContentGeneratorPrefixApis:
    """Prefix/shared-system/user-context must work on coding (same surface as sonnet)."""

    @pytest.fixture
    def tokenizer(self, mock_tokenizer_cls):
        return mock_tokenizer_cls.from_pretrained("gpt2")

    def test_prefix_pool_and_random_sample(self, tokenizer):
        from aiperf.config.dataset.content import PrefixPromptConfig

        gen = CodingContentGenerator(
            PromptConfig(),
            tokenizer,
            prefix_prompts=PrefixPromptConfig(pool_size=3, length=8),
        )
        assert len(gen._prefix_prompts) == 3
        sample = gen.get_random_prefix_prompt()
        assert isinstance(sample, str)
        assert sample in gen._prefix_prompts

    def test_shared_system_prompt(self, tokenizer):
        from aiperf.config.dataset.content import PrefixPromptConfig

        gen = CodingContentGenerator(
            PromptConfig(),
            tokenizer,
            prefix_prompts=PrefixPromptConfig(shared_system_length=12),
        )
        prompt = gen.get_shared_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_user_context_prompt_on_demand(self, tokenizer):
        from aiperf.config.dataset.content import PrefixPromptConfig

        gen = CodingContentGenerator(
            PromptConfig(),
            tokenizer,
            prefix_prompts=PrefixPromptConfig(user_context_length=10),
        )
        a = gen.generate_user_context_prompt(0)
        b = gen.generate_user_context_prompt(1)
        assert isinstance(a, str)
        assert isinstance(b, str)
        assert a == gen.generate_user_context_prompt(0)
