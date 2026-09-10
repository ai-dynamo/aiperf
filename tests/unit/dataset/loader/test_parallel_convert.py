# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Determinism + cross-process consistency for parallel_convert workers."""

from __future__ import annotations

from multiprocessing import shared_memory
from unittest.mock import mock_open, patch

import numpy as np
import pytest

from aiperf.common.exceptions import ConfigurationError
from aiperf.config.dataset.content import (
    PrefixPromptConfig,
    PromptConfig,
)
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader import parallel_convert as pc

MOCK_CORPUS_CONTENT = " ".join([f"word{i}" for i in range(1024)]) + "\n"


@pytest.fixture
def real_prompt_generator(mock_tokenizer_cls):
    tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
    prompts = PromptConfig(block_size=4)
    prefix_prompts = PrefixPromptConfig(pool_size=None, length=None)
    with patch("builtins.open", mock_open(read_data=MOCK_CORPUS_CONTENT)):
        return PromptGenerator(
            prompts=prompts, prefix_prompts=prefix_prompts, tokenizer=tokenizer
        )


def _drive_worker_inproc(
    pg: PromptGenerator,
    sessions: list[tuple[str, list[dict]]],
    trace_id: str,
    block_size: int,
) -> list:
    """Run ``_init_worker`` + ``_process_batch`` in-process (no Pool), restoring global ``_worker_state`` afterward to avoid cross-test leakage."""
    corpus = pg._tokenized_corpus
    corpus_len = len(corpus)
    shm = shared_memory.SharedMemory(
        create=True, size=corpus_len * np.dtype(np.int32).itemsize
    )
    np.ndarray((corpus_len,), dtype=np.int32, buffer=shm.buf)[:] = corpus

    args = pc._WorkerInitArgs(
        shm_name=shm.name,
        corpus_len=corpus_len,
        tokenizer_name="gpt2",
        base_seed=pg._hash_id_corpus_rng.seed,
        block_size=block_size,
        sep_token=pg.tokenizer.block_separation_token_id,
        trace_id=trace_id,
    )

    saved_state = pc._worker_state
    try:
        # Reuse the mock tokenizer instead of loading a real one.
        with patch(
            "aiperf.dataset.loader.parallel_convert.Tokenizer.from_pretrained",
            return_value=pg.tokenizer,
        ):
            pc._init_worker(args)
        results = pc._process_batch(sessions)
    finally:
        pc._worker_state = saved_state
        shm.close()
        shm.unlink()
    return results


def test_parallel_convert_matches_in_process(real_prompt_generator):
    """In-process ``_build_token_sequence`` output equals the worker ``_process_batch`` path byte-for-byte over the same inputs."""
    pg = real_prompt_generator
    trace_id = "abcdef0123456789"
    block_size = 4

    pg._hash_id_corpus_rng.set_trace_id(trace_id)
    pg._cache.clear()

    # Mixed layout: exact-tile (8/4) and last-partial (6 = 4 + 2).
    traces = [
        {
            "hash_ids": [11, 22],
            "input_length": 8,
            "output_length": 4,
            "timestamp": 1.0,
            "delay": None,
        },
        {
            "hash_ids": [33, 44],
            "input_length": 6,
            "output_length": 4,
            "timestamp": 2.0,
            "delay": None,
        },
    ]

    in_process_prompts: list[str] = []
    joined_block_prompts: list[str] = []
    for tr in traces:
        tokens = pg._build_token_sequence(
            tr["input_length"], tr["hash_ids"], block_size
        )
        in_process_prompts.append(
            pg.tokenizer.decode(tokens, skip_special_tokens=False)
        )
        joined_block_prompts.append(
            "".join(
                pg.tokenizer.decode(pg._cache[hid], skip_special_tokens=False)
                for hid in tr["hash_ids"]
            )
        )

    assert in_process_prompts[0] != joined_block_prompts[0], (
        "oracle is degenerate: decode(full sequence) accidentally equals "
        "joined per-block strings"
    )

    # Reset PG state so the worker sees a fresh trace_id scope.
    pg._cache.clear()
    pg._hash_id_corpus_rng.set_trace_id(trace_id)

    worker_results = _drive_worker_inproc(
        pg,
        sessions=[("s1", traces)],
        trace_id=trace_id,
        block_size=block_size,
    )

    assert len(worker_results) == 1
    sid, turns = worker_results[0]
    assert sid == "s1"
    assert len(turns) == len(traces)
    worker_prompts = [t[2] for t in turns]

    assert worker_prompts == in_process_prompts, (
        "parallel_convert worker path must match in-process path byte-for-byte: "
        f"{worker_prompts!r} vs {in_process_prompts!r}"
    )


def test_parallel_convert_distinct_across_trace_ids(real_prompt_generator):
    """Worker path yields different content for the same hash_ids under two trace_ids."""
    pg = real_prompt_generator
    block_size = 4
    sessions = [
        (
            "s1",
            [
                {
                    "hash_ids": [101, 202],
                    "input_length": 8,
                    "output_length": 4,
                    "timestamp": 1.0,
                    "delay": None,
                },
            ],
        )
    ]

    out_a = _drive_worker_inproc(pg, sessions, "trace_alpha_id_aaaa", block_size)
    out_b = _drive_worker_inproc(pg, sessions, "trace_beta_id_bbbb", block_size)

    prompt_a = out_a[0][1][0][2]
    prompt_b = out_b[0][1][0][2]
    assert prompt_a != prompt_b


def test_parallel_convert_hash_ids_overshoot_raises(real_prompt_generator):
    """Worker path raises ConfigurationError on hash_id overshoot, matching serial ``_build_token_sequence`` rather than truncating."""
    pg = real_prompt_generator
    block_size = 4
    traces = [
        {
            "hash_ids": [11, 22],
            "input_length": 4,
            "output_length": 4,
            "timestamp": 1.0,
            "delay": None,
        }
    ]

    pg._hash_id_corpus_rng.set_trace_id("overshoot_trace_id")
    pg._cache.clear()
    with pytest.raises(ConfigurationError):
        pg._build_token_sequence(
            traces[0]["input_length"], traces[0]["hash_ids"], block_size
        )

    with pytest.raises(ConfigurationError):
        _drive_worker_inproc(pg, [("s1", traces)], "overshoot_trace_id", block_size)


def test_parallel_convert_prefix_only_token_count(real_prompt_generator):
    """A prefix-only row (hashed prefix < input_length) yields a full-length prompt with an un-hashed tail."""
    pg = real_prompt_generator
    block_size = 4
    input_length = 6
    traces = [
        {
            "hash_ids": [101],
            "input_length": input_length,
            "output_length": 4,
            "timestamp": 1.0,
            "delay": None,
        }
    ]

    results = _drive_worker_inproc(
        pg, [("s1", traces)], "prefix_only_trace_id", block_size
    )
    prompt = results[0][1][0][2]
    token_count = len(pg.tokenizer.encode(prompt, add_special_tokens=False))
    assert token_count == input_length


def test_parallel_convert_prefix_tail_matches_cold_and_warm_cache(
    real_prompt_generator,
):
    """Unhashed tail must not change when the final hash_id is already cached.

    ``get_block_tokens`` used to skip ``reseed_for_hash_id`` on a hit, so
    ``sample_tokens`` for the tail saw leftover worker RNG. Cold vs warm
    cache must produce the same prompt for identical ``hash_ids`` +
    ``input_length``.
    """
    pg = real_prompt_generator
    block_size = 4
    hid = 101
    prefix_tail = {
        "hash_ids": [hid],
        "input_length": 6,
        "output_length": 4,
        "timestamp": 2.0,
        "delay": None,
    }
    exact_block = {
        "hash_ids": [hid],
        "input_length": 4,
        "output_length": 4,
        "timestamp": 1.0,
        "delay": None,
    }
    trace_id = "warm_cache_tail_trace"

    cold = _drive_worker_inproc(pg, [("s1", [prefix_tail])], trace_id, block_size)
    warm = _drive_worker_inproc(
        pg, [("s1", [exact_block, prefix_tail])], trace_id, block_size
    )
    duplicate = _drive_worker_inproc(
        pg, [("s1", [prefix_tail, dict(prefix_tail)])], trace_id, block_size
    )

    cold_prompt = cold[0][1][0][2]
    warm_prompt = warm[0][1][1][2]
    first_dup, second_dup = duplicate[0][1][0][2], duplicate[0][1][1][2]

    assert cold_prompt == warm_prompt
    assert first_dup == second_dup == cold_prompt
