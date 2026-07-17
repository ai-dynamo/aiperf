# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coding content generator for realistic coding trace replay.

Generates structurally plausible coding content (code, bash output, JSON,
errors, git diffs, CI output, configs, markdown, test output, user prompts)
using template-based generation with random identifiers.

Unlike PromptGenerator which uses Shakespeare as its corpus, this generator
builds two token pools from structural templates:
- text_pool: user prompts (natural language coding requests)
- tool_pool: mixed technical content (code, errors, diffs, configs, etc.)

Generation uses window slicing from pre-built token pools, same as PromptGenerator.
"""

from __future__ import annotations

from aiperf.common import random_generator as rng
from aiperf.common.exceptions import ConfigurationError, NotInitializedError
from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
from aiperf.common.tokenizer import Tokenizer
from aiperf.config import PromptConfig
from aiperf.config.dataset.defaults import InputTokensDefaults
from aiperf.dataset.generator._coding_cicd_docs import _CicdDocsMixin
from aiperf.dataset.generator._coding_conversations import _ConversationsMixin
from aiperf.dataset.generator._coding_conversations_advanced import (
    _ConversationsAdvancedMixin,
)
from aiperf.dataset.generator._coding_errors_diff import _ErrorsDiffMixin
from aiperf.dataset.generator._coding_go import _GoMixin
from aiperf.dataset.generator._coding_json import _JsonMixin
from aiperf.dataset.generator._coding_ml import _MlMixin
from aiperf.dataset.generator._coding_prompts_conv import _PromptsConvMixin
from aiperf.dataset.generator._coding_python import _PythonMixin
from aiperf.dataset.generator._coding_rust import _RustMixin
from aiperf.dataset.generator._coding_sql import _SqlMixin
from aiperf.dataset.generator._coding_text import (
    _BASELINE_POOL_TOKENS,
    _TEXT_POOL_BLOCKS,
    _TOOL_POOL_BLOCK_COUNTS,
)
from aiperf.dataset.generator._coding_tool import _ToolMixin
from aiperf.dataset.generator._coding_tool_long import _ToolLongMixin
from aiperf.dataset.generator._coding_typescript import _TypeScriptMixin
from aiperf.dataset.generator.base import BaseGenerator
from aiperf.dataset.generator.prompt import sample_tokens_from_corpus


class CodingContentGenerator(
    _PythonMixin,
    _GoMixin,
    _RustMixin,
    _TypeScriptMixin,
    _ToolMixin,
    _JsonMixin,
    _ErrorsDiffMixin,
    _CicdDocsMixin,
    _MlMixin,
    _SqlMixin,
    _PromptsConvMixin,
    _ToolLongMixin,
    _ConversationsMixin,
    _ConversationsAdvancedMixin,
    BaseGenerator,
):
    """Generator for structurally plausible coding content.

    Builds two pre-tokenized pools from template-based content:
    - text_pool: natural language coding requests (built lazily on first use)
    - tool_pool: mixed technical content - code, errors, diffs, etc.
      (~500K tokens, built eagerly at init)

    Exposes the PromptGenerator-compatible interface (``generate`` /
    ``generate_prompt`` / ``calculate_num_tokens``); prompt sampling draws
    from the tool pool.
    """

    def __init__(
        self,
        config: PromptConfig,
        tokenizer: Tokenizer,
        pool_tokens_target: int | None = None,
        **kwargs,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self._pool_scale = max(
            1.0, (pool_tokens_target or _BASELINE_POOL_TOKENS) / _BASELINE_POOL_TOKENS
        )

        self._template_rng = rng.derive("dataset.coding_content.template")
        self._corpus_rng = rng.derive("dataset.coding_content.corpus")
        self._length_rng = rng.derive("dataset.coding_content.length")

        # Hash-ID-based RNG for deterministic per-hash_id generation in
        # _build_token_sequence; also read directly by the graph adapters'
        # CorpusContentSynthesizer (graph/adapters/shared/content.py), which
        # reseeds it per (trace_id, hash_id) for byte-deterministic blocks.
        self._hash_id_corpus_rng = HashIdRandomGenerator.from_base_rng(self._corpus_rng)

        super().__init__(config=config, tokenizer=tokenizer, **kwargs)

        self._text_pool: list[int] | None = None
        self._tool_pool: list[int] = []
        self._cache: dict[int, list[int]] = {}
        self._decoded_cache: dict[tuple[tuple[int, ...], int, int], str] = {}
        # No stable terminator probe for the coding corpus; the empty list
        # means "append no terminator". Nothing consumes this attribute on
        # this branch.
        self._bpe_stable_terminator_tokens: list[int] = []

        self._build_tool_pool()

        # PromptGenerator-parity surface read by the graph adapters'
        # CorpusContentSynthesizer (corpus hoisting / shared-memory attach).
        self._tokenized_corpus = self._tool_pool
        self._corpus_size = len(self._tool_pool)

    def generate(
        self,
        mean: int | None = None,
        stddev: int | None = None,
        hash_ids: list[int] | None = None,
        block_size: int | None = None,
    ) -> str:
        if hash_ids:
            if mean is None:
                raise ValueError("mean must be provided when hash_ids is set.")
            # config.block_size defaults to None; fall back to the same
            # default PromptGenerator.generate uses so hash_ids work with a
            # default config instead of raising TypeError.
            bs = block_size or self.config.block_size or InputTokensDefaults.BLOCK_SIZE
            return self._generate_cached_prompt(mean, hash_ids, bs)
        num_tokens = self.calculate_num_tokens(mean, stddev)
        return self.generate_prompt(num_tokens)

    def generate_prompt(self, num_tokens: int) -> str:
        tokens = self._sample_tokens(num_tokens, self._tool_pool)
        return self.tokenizer.decode(tokens)

    def calculate_num_tokens(
        self,
        mean: int | None = None,
        stddev: int | None = None,
    ) -> int:
        return self._length_rng.sample_positive_normal_integer(mean, stddev)

    def _ensure_text_pool(self) -> list[int]:
        if self._text_pool is None:
            self._build_text_pool()
        assert self._text_pool is not None
        return self._text_pool

    def _build_text_pool(self) -> None:
        blocks: list[str] = []
        for _ in range(int(_TEXT_POOL_BLOCKS * self._pool_scale)):
            blocks.append(self._gen_user_prompt())
        text = "\n\n".join(blocks)
        self._text_pool = self.tokenizer.encode(text)
        pool = self._text_pool
        self.debug(
            lambda: f"Built text pool with {len(pool)} tokens from {len(blocks)} blocks"
        )

    def _build_tool_pool(self) -> None:
        blocks: list[str] = []
        for gen_name, count in _TOOL_POOL_BLOCK_COUNTS.items():
            gen_fn = getattr(self, gen_name)
            for _ in range(int(count * self._pool_scale)):
                blocks.append(gen_fn())
        self._template_rng.shuffle(blocks)
        text = "\n\n".join(blocks)
        self._tool_pool = self.tokenizer.encode(text)
        self.debug(
            lambda: (
                f"Built tool pool with {len(self._tool_pool)} tokens "
                f"from {len(blocks)} blocks"
            )
        )

    def _sample_tokens(self, num_tokens: int, pool: list[int]) -> list[int]:
        if not pool:
            raise NotInitializedError("Token pool is not initialized.")
        pool_size = len(pool)
        if num_tokens <= 0:
            return []
        start_idx = self._corpus_rng.randrange(pool_size)
        end_idx = start_idx + num_tokens
        tokens = pool[start_idx:end_idx]
        if end_idx > pool_size:
            tokens += pool[: end_idx - pool_size]
        return tokens

    def _generate_cached_prompt(
        self,
        num_tokens: int,
        hash_ids: list[int],
        block_size: int,
    ) -> str:
        cache_key = (tuple(hash_ids), num_tokens, block_size)
        if cache_key in self._decoded_cache:
            return self._decoded_cache[cache_key]

        final_prompt = self._build_token_sequence(num_tokens, hash_ids, block_size)
        decoded = self.tokenizer.decode(final_prompt, skip_special_tokens=False)
        self._decoded_cache[cache_key] = decoded
        return decoded

    def _build_token_sequence(
        self,
        num_tokens: int,
        hash_ids: list[int],
        block_size: int,
    ) -> list[int]:
        final_prompt: list[int] = []
        current_block_size = block_size

        final_block_size = num_tokens - ((len(hash_ids) - 1) * block_size)
        if final_block_size <= 0 or block_size < final_block_size:
            raise ConfigurationError(
                f"Input length: {num_tokens}, Hash IDs: {hash_ids}, Block size: {block_size} "
                f"are not compatible. The final hash block size: {final_block_size} must be "
                f"greater than 0 and less than or equal to {block_size}."
            )

        for index, hash_id in enumerate(hash_ids):
            if index == len(hash_ids) - 1:
                current_block_size = final_block_size

            if hash_id not in self._cache:
                self._hash_id_corpus_rng.reseed_for_hash_id(hash_id)
                self._cache[hash_id] = sample_tokens_from_corpus(
                    self._tool_pool,
                    current_block_size,
                    self._hash_id_corpus_rng,
                    self.tokenizer.block_separation_token_id,
                )

            final_prompt.extend(self._cache[hash_id])

        return final_prompt
