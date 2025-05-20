#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
import os
import pathlib
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from aiperf.common.exceptions import (
    SyntheticDataConfigurationException,
    SyntheticDataInitializationException,
)
from aiperf.common.tokenizer import Tokenizer

logger = logging.getLogger("SyntheticPromptGenerator")

DEFAULT_CORPUS_FILE = "assets/shakespeare.txt"


class SyntheticPromptGenerator:
    _tokenized_corpus = None
    _corpus_length = 0
    _prefix_prompts: list[str] = []
    _cache: dict[int, list[int]] = {}

    @classmethod
    def create_synthetic_prompt(
        cls,
        tokenizer: Tokenizer,
        prompt_tokens_mean: int = 550,
        prompt_tokens_stddev: int = 250,
        hash_ids: Optional[list[int]] = None,
        block_size: int = 512,
    ) -> str:
        """
        Generate a synthetic prompt with a specific number of tokens.

        Args:
            tokenizer: Tokenizer instance.
            prompt_tokens_mean: Mean number of tokens in the prompt.
            prompt_tokens_stddev: Standard deviation for the number of tokens in the prompt.

        Returns:
            A synthetic prompt as a string.
        """
        if cls._tokenized_corpus is None:
            cls._initialize_corpus(tokenizer)

        if hash_ids:
            return cls._generate_prompt_with_token_reuse(
                tokenizer, prompt_tokens_mean, hash_ids, block_size
            )

        num_prompt_tokens = max(
            0, int(random.gauss(prompt_tokens_mean, prompt_tokens_stddev))
        )

        return cls._generate_prompt(tokenizer, num_prompt_tokens)

    @classmethod
    def _initialize_corpus(
        cls, tokenizer: Tokenizer, corpus_file: str = DEFAULT_CORPUS_FILE
    ) -> None:
        """
        Load and tokenize the corpus once, storing it for reuse.

        Args:
            tokenizer: Tokenizer for tokenizing the corpus.
        """
        corpus_path = pathlib.Path(__file__).parent / corpus_file

        with open(corpus_path) as f:
            lines = f.readlines()

        def tokenize_chunk(chunk):
            cleaned_text = " ".join(line.strip() for line in chunk if line.strip())
            tokens = tokenizer.encode(cleaned_text)
            return tokens

        num_threads = os.cpu_count()
        if num_threads is None:
            num_threads = 4
        chunk_size = len(lines) // num_threads
        chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            tokenized_chunks = list(executor.map(tokenize_chunk, chunks))

        cls._tokenized_corpus = [token for chunk in tokenized_chunks for token in chunk]
        cls._corpus_length = len(cls._tokenized_corpus)

    @classmethod
    def _generate_prompt_tokens(cls, num_tokens: int) -> list[int]:
        """
        Generate a prompt containing exactly `num_tokens` using the preloaded tokenized corpus.

        Args:
            num_tokens: Number of tokens required in the prompt.

        Returns:
            A synthetic prompt of tokens.

        Raises:
            ValueError: If the tokenized corpus is not initialized
        """
        if not cls._tokenized_corpus:
            raise SyntheticDataInitializationException(
                "Tokenized corpus is not initialized."
            )
        if num_tokens > cls._corpus_length:
            logger.warning(
                f"Requested prompt length {num_tokens} is longer than the corpus. "
                f"Returning a prompt of length {cls._corpus_length}."
            )

        start_idx = random.randrange(cls._corpus_length)

        end_idx = start_idx + num_tokens
        prompt_tokens = cls._tokenized_corpus[start_idx:end_idx]
        if end_idx > cls._corpus_length:
            prompt_tokens += cls._tokenized_corpus[: end_idx - cls._corpus_length]

        return prompt_tokens

    @classmethod
    def _generate_prompt(cls, tokenizer: Tokenizer, num_tokens: int) -> str:
        """
        Generate a prompt containing exactly `num_tokens` using the preloaded tokenized corpus.

        Args:
            tokenizer: Tokenizer instance.
            num_tokens: Number of tokens required in the prompt.

        Returns:
            A synthetic prompt as a string.

        Raises:
            ValueError: If the tokenized corpus is not initialized
        """

        return tokenizer.decode(cls._generate_prompt_tokens(num_tokens))

    @classmethod
    def _generate_prompt_with_token_reuse(
        cls,
        tokenizer: Tokenizer,
        num_tokens: int,
        prompt_hash_list: list[int],
        block_size: int,
    ) -> str:
        """
        Generate a prompt containing exactly `num_tokens` by reusing previously generated prompts
        stored in `_cache`. Each hash index in `prompt_hash_list` corresponds to a block of
        `block_size` tokens. If a hash index is found in `_cache`, its stored prompt is reused.
        Otherwise, a new prompt is generated using `_generate_prompt()` and stored in `_cache`.

        Args:
            tokenizer : Tokenizer
                The tokenizer used to generate prompts.
            num_tokens : int
                The number of tokens required in the prompt.
            prompt_hash_list : list[int]
                A list of hash indices used for token reuse.
            block_size : int
                The number of tokens allocated per hash block (default 512).

        Returns:
            str: A synthetic prompt as a string.
        """
        final_prompt: list[int] = []
        size_to_use = block_size
        last_hash_length = num_tokens - ((len(prompt_hash_list) - 1) * block_size)
        if last_hash_length <= 0 or block_size < last_hash_length:
            raise SyntheticDataConfigurationException(
                f"Input_length: {num_tokens}, Hash_ids: {prompt_hash_list}, Block_size: {block_size} "
                f"are not compatible. The final hash id length: {last_hash_length} must be greater "
                f"than 0 and less than or equal to {block_size}."
            )
        for index, hash_index in enumerate(prompt_hash_list):
            if index == len(prompt_hash_list) - 1:
                size_to_use = num_tokens - (index * block_size)
            if hash_index not in cls._cache:
                # To ensure that the prompt doesn't merge chunks, we pop the last token
                # and insert the bos token at the beginning. Length is maintained and
                # the prompt generates the expected number of tokens.
                prompt_tokens = cls._generate_prompt_tokens(size_to_use)
                prompt_tokens.pop(0)
                prompt_tokens.insert(0, tokenizer.bos_token_id())
                cls._cache[hash_index] = prompt_tokens
            final_prompt.extend(cls._cache[hash_index])
        prompt = tokenizer.decode(final_prompt, skip_special_tokens=False)

        return prompt

    @classmethod
    def create_prefix_prompts_pool(
        cls, tokenizer: Tokenizer, num_prompts: int, prompt_length: int
    ) -> None:
        """
        Generate a pool of prefix prompts.

        Args:
            tokenizer: Tokenizer instance.
            num_prompts: Number of prefix prompts to generate.
            prompt_length: Number of tokens per prefix prompt.
        """
        if cls._tokenized_corpus is None:
            cls._initialize_corpus(tokenizer)

        cls._prefix_prompts = [
            cls._generate_prompt(tokenizer, prompt_length) for _ in range(num_prompts)
        ]

    @classmethod
    def get_random_prefix_prompt(cls) -> str:
        """
        Fetch a random prefix prompt from the pool.

        Returns:
            A random prefix prompt.
        """
        return random.choice(cls._prefix_prompts)
