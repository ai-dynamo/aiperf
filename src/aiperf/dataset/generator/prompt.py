# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aiperf.common import random_generator as rng
from aiperf.common.exceptions import (
    ConfigurationError,
    InvalidStateError,
    NotInitializedError,
)
from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.dataset.content import PrefixPromptConfig, PromptConfig
from aiperf.config.dataset.defaults import InputTokensDefaults
from aiperf.dataset.generator.base import BaseGenerator

DEFAULT_CORPUS_FILE = "assets/shakespeare.txt"


def sample_tokens_from_corpus(
    corpus,
    num_tokens: int,
    rng_to_use,
    sep_token: int | None = None,
) -> list[int]:
    """Sample ``num_tokens`` tokens from ``corpus`` using ``rng_to_use``.

    Mirrors :meth:`PromptGenerator._sample_tokens` but takes the RNG explicitly,
    so callers (worker processes, the hash-id reseed path) can drive sampling
    without sharing PromptGenerator state. Wraps the slice if it overflows the
    corpus end so the result is always exactly ``num_tokens`` long.

    Args:
        corpus: Token corpus as a sequence of token IDs.
        num_tokens: Number of tokens to sample.
        rng_to_use: Random generator with a ``randrange(n)`` method.
        sep_token: Optional block-separation token prepended at index 0
            (consumes one slot of ``num_tokens``).

    Returns:
        List of sampled token IDs of length ``num_tokens``.
    """
    corpus_len = len(corpus)
    tokens: list[int] = []

    if sep_token is not None:
        tokens.append(sep_token)
        num_tokens -= 1

    start = rng_to_use.randrange(corpus_len)
    end = start + num_tokens

    if end <= corpus_len:
        tokens.extend(corpus[start:end])
    else:
        tokens.extend(corpus[start:])
        tokens.extend(corpus[: end - corpus_len])

    return tokens


class PromptGenerator(BaseGenerator):
    """A class for generating synthetic prompts from a text corpus.

    This class loads a text corpus (e.g., Shakespearean text), tokenizes it,
    and uses the tokenized corpus to generate synthetic prompts of specified
    lengths. It supports generating prompts with a target number of tokens
    (with optional randomization around a mean and standard deviation) and
    can reuse previously generated token blocks to optimize generation for
    certain use cases. It also allows for the creation of a pool of prefix
    prompts that can be randomly selected.
    """

    def __init__(
        self,
        *,
        prompts: PromptConfig | None,
        prefix_prompts: PrefixPromptConfig | None,
        tokenizer: Tokenizer,
        **kwargs,
    ):
        self.prompts = prompts
        self.prefix_prompts = prefix_prompts
        self.tokenizer = tokenizer
        self._tokenized_corpus = None
        self._corpus_size = 0
        self._prefix_prompts: list[str] = []

        # Conversation context prompts
        self._shared_system_prompt: str | None = None
        self._user_context_prompts: list[str] = []

        # Separate RNGs for independent concerns
        self._length_rng = rng.derive("dataset.prompt.length")
        self._corpus_rng = rng.derive("dataset.prompt.corpus")
        self._prefix_rng = rng.derive("dataset.prompt.prefix")

        # Hash-ID-based RNG for deterministic per-(trace_id, hash_id) sampling.
        # Used by the WekaTraceLoader so cross-process replay scopes block
        # content to a single trace_id, eliminating cross-trace cache collisions.
        self._hash_id_corpus_rng = HashIdRandomGenerator.from_base_rng(self._corpus_rng)

        super().__init__(tokenizer=tokenizer, **kwargs)

        # Cached prompts: block ID -> list of tokens
        self._cache: dict[int, list[int]] = {}

        # Decoded string cache: (hash_ids tuple, num_tokens, block_size) -> decoded string
        # This avoids redundant tokenizer.decode() calls for repeated hash_id combinations
        self._decoded_cache: dict[tuple[tuple[int, ...], int, int], str] = {}

        # TODO: move this under initialize() method
        # Initialize corpus if not already done
        if self._tokenized_corpus is None:
            self._initialize_corpus()

        # Probe the tokenizer for a BPE-stable terminator we can append to
        # every reconstructed segment so aiperf's join-with-" " ISL formula
        # equals the sum of per-segment token counts (eliminates segment-join
        # BPE re-merge drift). See spec §6.2.
        self._bpe_stable_terminator_tokens: list[int] = (
            self._determine_bpe_stable_terminator()
        )

        # Initialize prefix prompts pool if the pool size > 0
        pool_size = (
            self.prefix_prompts.pool_size if self.prefix_prompts is not None else None
        ) or 0
        if pool_size > 0:
            self._create_prefix_prompt_pool()

        # Initialize shared context prompts if configured
        shared_system_length = (
            self.prefix_prompts.shared_system_length
            if self.prefix_prompts is not None
            else None
        )
        if shared_system_length is not None:
            self._generate_shared_system_prompt()
        # Note: User context prompts are generated on-demand in generate_user_context_prompt()

    def _initialize_corpus(self) -> None:
        """Load and tokenize the corpus once, storing it for reuse.

        Uses character-based chunking for reproducibility across different machines.
        The chunk size is fixed (not CPU-dependent) to ensure the same tokenization
        boundaries regardless of hardware, which guarantees identical prompts with
        the same random seed across all environments.

        Thread Safety Note:
            This method uses parallel tokenization for performance. Most tokenizers
            (including Hugging Face transformers) are thread-safe and deterministic.
            Thread count doesn't affect reproducibility since chunks have deterministic
            boundaries based on character count.
        """
        corpus_path = Path(__file__).parent / DEFAULT_CORPUS_FILE

        with open(corpus_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Pre-filter empty lines for efficiency
        non_empty_lines = [line.strip() for line in lines if line.strip()]

        def tokenize_chunk(chunk):
            """Tokenize a chunk of pre-cleaned lines."""
            text = " ".join(chunk)
            tokens = self.tokenizer.encode(text)
            return tokens

        # Character-based chunking: Fixed chunk size ensures reproducibility
        # across machines with different CPU counts. Creates ~486 chunks for
        # optimal thread utilization (~294 lines per chunk).
        MAX_CHARS_PER_CHUNK = 10_000

        # Build chunks based on character count (deterministic chunking)
        chunks = []
        buffer = []
        char_count = 0

        for line in non_empty_lines:
            buffer.append(line)
            char_count += len(line)

            if char_count >= MAX_CHARS_PER_CHUNK:
                chunks.append(buffer)
                buffer = []
                char_count = 0

        # Add remaining lines as final chunk
        if buffer:
            chunks.append(buffer)

        # Multi-threaded tokenization: thread count doesn't affect reproducibility
        # since chunks are character-based (deterministic boundaries)
        num_threads = min(os.cpu_count() or 4, 8)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            tokenized_chunks = list(executor.map(tokenize_chunk, chunks))

        self._tokenized_corpus = [
            token for chunk in tokenized_chunks for token in chunk
        ]
        self._corpus_size = len(self._tokenized_corpus)
        self.debug(
            lambda: f"Initialized corpus with {self._corpus_size} tokens "
            f"from {len(chunks)} chunks using {num_threads} thread(s)"
        )

    def _determine_bpe_stable_terminator(self) -> list[int]:
        """Probe the tokenizer for a short token sequence that, when appended
        to arbitrary content and followed by ``" " + arbitrary content``, does
        NOT cause BPE re-merging across the seam.

        Tries candidates in order and returns the first that passes the
        join-byte-exact probe:

        1. ``"\\n\\n"`` (typically a singleton in modern tokenizers)
        2. ``"\\n"`` (fallback)
        3. ``" "`` (last resort — degenerate)

        Returns an empty list if no candidate passes; segment synthesis then
        falls back to no terminator and segment-join drift is unfixed.
        """
        if not self._tokenized_corpus:
            return []
        corpus_size = self._corpus_size
        if corpus_size < 264:
            return []

        a = self.tokenizer.decode(self._tokenized_corpus[100:164])
        b = self.tokenizer.decode(self._tokenized_corpus[200:264])

        a_tokens = self.tokenizer.encode(a, add_special_tokens=False)
        b_with_lead_space = self.tokenizer.encode(" " + b, add_special_tokens=False)

        for cand_text in ("\n\n", "\n", " "):
            cand_tokens = self.tokenizer.encode(cand_text, add_special_tokens=False)
            if len(cand_tokens) == 0:
                continue
            joined = a + cand_text + " " + b
            joined_tokens = self.tokenizer.encode(joined, add_special_tokens=False)
            expected_total = len(a_tokens) + len(cand_tokens) + len(b_with_lead_space)
            if len(joined_tokens) == expected_total:
                self.debug(
                    lambda ct=cand_text, t=cand_tokens: (
                        f"BPE-stable terminator chosen: {ct!r} -> tokens={t}"
                    )
                )
                return list(cand_tokens)

        self.debug("No BPE-stable terminator found; segment-join drift will be unfixed")
        return []

    def _create_prefix_prompt_pool(self) -> None:
        """Generate a pool of prefix prompts to sample from."""
        if self._tokenized_corpus is None:
            raise NotInitializedError("Tokenized corpus is not initialized.")

        if self.prefix_prompts is None:
            return

        length = self.prefix_prompts.length or 0
        pool_size = self.prefix_prompts.pool_size or 0
        self._prefix_prompts = [self.generate_prompt(length) for _ in range(pool_size)]
        self.debug(
            lambda: f"Initialized prefix prompts pool with {len(self._prefix_prompts)} prompts"
        )

    def generate(
        self,
        mean: int | None = None,
        stddev: int | None = None,
        hash_ids: list[int] | None = None,
    ) -> str:
        """Generate a synthetic prompt with the configuration parameters.
        Serves as a wrapper around other internal methods to provide a unified interface.

        Args:
            mean: The mean of the normal distribution.
            stddev: The standard deviation of the normal distribution.
            hash_ids: A list of hash indices used for token reuse.

        Returns:
            A synthetic prompt as a string.
        """
        if hash_ids:
            if mean is None:
                raise ValueError("mean must be provided when hash_ids is set.")
            block_size = (
                self.prompts.block_size if self.prompts is not None else None
            ) or InputTokensDefaults.BLOCK_SIZE
            return self._generate_cached_prompt(mean, hash_ids, block_size)

        num_tokens = self.calculate_num_tokens(mean, stddev)
        return self.generate_prompt(num_tokens)

    def calculate_num_tokens(
        self,
        mean: int | None = None,
        stddev: int | None = None,
    ) -> int:
        """Calculate the number of tokens for a prompt based on a normal distribution.

        Args:
            mean: The mean of the normal distribution.
            stddev: The standard deviation of the normal distribution.
        """

        return self._length_rng.sample_positive_normal_integer(mean, stddev)

    def generate_prompt(self, num_tokens: int) -> str:
        """Generate a prompt containing exactly `num_tokens` number of tokens.

        Args:
            num_tokens: Number of tokens required in the prompt.

        Returns:
            A synthetic prompt as a string.
        """
        return self.tokenizer.decode(self._sample_tokens(num_tokens))

    def _generate_cached_prompt(
        self,
        num_tokens: int,
        hash_ids: list[int],
        block_size: int,
    ) -> str:
        """
        Generate a prompt containing exactly `num_tokens` by reusing previously generated prompts
        stored in `_cache`. Each hash index in `hash_ids` corresponds to a block of
        `block_size` tokens. If a hash index is found in `_cache`, its stored prompt is reused.
        Otherwise, a new prompt is generated using `generate_prompt()` and stored in `_cache`.

        Args:
            num_tokens: The number of tokens required in the prompt.
            hash_ids: A list of hash IDs to use for token reuse.
            block_size: The number of tokens allocated per hash block.

        Returns:
            str: A synthetic prompt as a string.

        Raises:
            ConfigurationError: If the input parameters are not compatible.
        """
        final_prompt = self._build_token_sequence(num_tokens, hash_ids, block_size)
        return self.tokenizer.decode(final_prompt)

    def _build_token_sequence(
        self,
        num_tokens: int,
        hash_ids: list[int],
        block_size: int,
    ) -> list[int]:
        """
        Build a token sequence without decoding. Used for batch parallel decode.

        Each hash index in `hash_ids` corresponds to a block of `block_size` tokens.
        If a hash index is found in `_cache`, its stored tokens are reused.
        Otherwise, new tokens are sampled and stored in `_cache`.

        Three layouts are supported, matching how upstream trace formats record
        cache structure:

        - **Exact tile** (``len(hash_ids) * block_size == num_tokens``): every
          hash is a full block.
        - **Last block partial** (``(M-1)*block_size < num_tokens < M*block_size``):
          synthetic prompts authored by AIPerf — the final hash maps to a
          partial block of size ``num_tokens - (M-1)*block_size``.
        - **Prefix only** (``M*block_size < num_tokens``): real
          captured traces (e.g. weka kv-cache-tester) where ``hash_ids`` lists
          only the cached prefix and the un-hashed tail represents fresh
          tokens. The tail is padded with sampled (uncached) tokens.

        Args:
            num_tokens: The number of tokens required in the prompt.
            hash_ids: A list of hash IDs covering the cached prefix.
            block_size: The number of tokens allocated per hash block.

        Returns:
            list[int]: A list of token IDs of length ``num_tokens``.

        Raises:
            ConfigurationError: If ``num_tokens <= 0`` or ``block_size <= 0``,
                or if hash_ids overshoots and the implied partial block size is
                outside ``(0, block_size]``.
        """
        if num_tokens <= 0 or block_size <= 0:
            raise ConfigurationError(
                f"Input length: {num_tokens}, Hash IDs: {hash_ids}, "
                f"Block size: {block_size} are not compatible. num_tokens "
                f"and block_size must both be greater than 0."
            )

        m = len(hash_ids)
        total_hashed = m * block_size
        final_prompt: list[int] = []

        if not hash_ids:
            return self._sample_tokens(num_tokens)

        if total_hashed > num_tokens:
            # Synthetic-prompt path: last hash is a partial block.
            final_block_size = num_tokens - ((m - 1) * block_size)
            if final_block_size <= 0 or final_block_size > block_size:
                raise ConfigurationError(
                    f"Input length: {num_tokens}, Hash IDs: {hash_ids}, "
                    f"Block size: {block_size} are not compatible. The final "
                    f"hash block size: {final_block_size} must be greater than "
                    f"0 and less than or equal to {block_size}."
                )
        else:
            # Exact-tile or prefix-only path: every hash is a full block.
            final_block_size = block_size

        for index, hash_id in enumerate(hash_ids):
            current_block_size = final_block_size if index == m - 1 else block_size
            cached = self._cache.get(hash_id)
            if cached is None:
                # Reseed per-(trace_id, hash_id) so the same hash_id in a
                # different trace file (different trace_id scope) produces
                # different tokens. Trace loaders set the trace_id once per
                # file in BaseTraceDatasetLoader.load_dataset and clear
                # ``self._cache`` between files.
                self._hash_id_corpus_rng.reseed_for_hash_id(hash_id)
                cached = sample_tokens_from_corpus(
                    self._tokenized_corpus,
                    current_block_size,
                    self._hash_id_corpus_rng,
                    self.tokenizer.block_separation_token_id,
                )
                self._cache[hash_id] = cached
            elif len(cached) != current_block_size:
                # A hash_id identifies a fixed block of content, so it can only
                # ever have one size. The same id at two sizes means a corrupt
                # trace or a block_size that disagrees with the recorded blocks.
                raise ConfigurationError(
                    f"hash_id {hash_id} requested at {current_block_size} tokens "
                    f"but was already materialized at {len(cached)} tokens. A "
                    f"hash_id must map to a single fixed block size; inconsistent "
                    f"sizes indicate a corrupt trace or a --isl-block-size that "
                    f"disagrees with the recorded blocks."
                )

            final_prompt.extend(cached)

        # Prefix-only: pad the un-hashed tail with sampled (uncached) tokens.
        tail = num_tokens - len(final_prompt)
        if tail > 0:
            final_prompt.extend(self._sample_tokens(tail))

        return final_prompt

    def _sample_tokens(self, num_tokens: int) -> list[int]:
        """Generate a list of token IDs containing exactly `num_tokens` number of tokens
        using the preloaded tokenized corpus.

        Args:
            num_tokens: Number of tokens required in the prompt.

        Returns:
            A list of token IDs.

        Raises:
            NotInitializedError: If the tokenized corpus is not initialized
        """
        if not self._tokenized_corpus:
            raise NotInitializedError("Tokenized corpus is not initialized.")
        if num_tokens > self._corpus_size:
            self.warning(
                f"Requested prompt length {num_tokens} is longer than the corpus "
                f"({self._corpus_size} tokens). The corpus will wrap around, so "
                f"the prompt repeats corpus content but still has {num_tokens} tokens."
            )

        start_idx = self._corpus_rng.randrange(self._corpus_size)

        end_idx = start_idx + num_tokens
        prompt_tokens = self._tokenized_corpus[start_idx:end_idx]
        # Wrap as many times as needed: a single wrap is short whenever
        # num_tokens exceeds twice the corpus size.
        while len(prompt_tokens) < num_tokens:
            remaining = num_tokens - len(prompt_tokens)
            prompt_tokens += self._tokenized_corpus[:remaining]

        self.trace(lambda: f"Sampled {len(prompt_tokens)} tokens from corpus")
        return prompt_tokens

    def get_random_prefix_prompt(self) -> str:
        """
        Fetch a random prefix prompt from the pool.

        Returns:
            A random prefix prompt.

        Raises:
            InvalidStateError: If the prefix prompts pool is empty.
        """
        if not self._prefix_prompts:
            raise InvalidStateError(
                "Attempted to sample a prefix prompt but the prefix prompts pool is empty. "
                "Please ensure that the prefix prompts pool is initialized."
            )
        return self._prefix_rng.choice(self._prefix_prompts)

    def _generate_shared_system_prompt(self) -> None:
        """Generate the shared system prompt.

        This prompt is generated once and is identical across all sessions.
        It appears as a system message in turn 0 of every conversation.
        """
        if self._tokenized_corpus is None:
            raise NotInitializedError("Tokenized corpus is not initialized.")

        length = (
            self.prefix_prompts.shared_system_length
            if self.prefix_prompts is not None
            else None
        )
        if length is None:
            return

        self._shared_system_prompt = self.generate_prompt(length)
        self.debug(lambda: f"Generated shared system prompt with {length} tokens")

    def get_shared_system_prompt(self) -> str:
        """Get the shared system prompt.

        Returns:
            The shared system prompt string.

        Raises:
            InvalidStateError: If shared system prompt is not initialized.
        """
        if self._shared_system_prompt is None:
            raise InvalidStateError(
                "Shared system prompt is not initialized. "
                "Ensure --shared-system-prompt-length is specified."
            )
        return self._shared_system_prompt

    def generate_user_context_prompt(self, session_index: int) -> str:
        """Generate unique user context for given session index.

        Generates prompts on-demand as needed. Each session_index gets a unique prompt.
        This allows benchmarks to run with any number of sessions without pre-allocating.

        Args:
            session_index: Sequential index of the session (0, 1, 2, ...).

        Returns:
            Unique user context prompt for this session.

        Raises:
            NotInitializedError: If tokenized corpus is not initialized.
            InvalidStateError: If user context prompt length is not configured.
        """
        if self._tokenized_corpus is None:
            raise NotInitializedError("Tokenized corpus is not initialized.")

        length = (
            self.prefix_prompts.user_context_length
            if self.prefix_prompts is not None
            else None
        )
        if length is None:
            raise InvalidStateError(
                "User context prompt length is not configured. "
                "Ensure --user-context-prompt-length is specified."
            )

        # Generate new prompts on-demand as needed
        while session_index >= len(self._user_context_prompts):
            new_prompt = self.generate_prompt(length)
            self._user_context_prompts.append(new_prompt)
            self.debug(
                lambda: f"Generated user context prompt #{len(self._user_context_prompts) - 1} "
                f"for session {len(self._user_context_prompts) - 1}"
            )

        return self._user_context_prompts[session_index]
