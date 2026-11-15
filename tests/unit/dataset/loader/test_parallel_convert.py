# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Determinism + cross-process consistency for parallel_convert workers."""

from __future__ import annotations

from collections.abc import Callable
from multiprocessing import shared_memory
from unittest.mock import mock_open, patch

import numpy as np
import pytest
from pytest import param

from aiperf.common.enums import PromptCorpus
from aiperf.common.exceptions import ConfigurationError
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.dataset.content import (
    PrefixPromptConfig,
    PromptConfig,
)
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader import parallel_convert as pc
from tests.harness.sentencepiece_tokenizer import (
    SENTENCEPIECE_CORPUS,
    SENTENCEPIECE_WHITESPACE_CORPUS,
    make_sentencepiece_tokenizer,
    make_sentencepiece_whitespace_tokenizer,
)

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
    pg: PromptGenerator | CodingContentGenerator,
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


@pytest.mark.parametrize(
    "corpus",
    [param(PromptCorpus.SONNET, id="sonnet"), param(PromptCorpus.CODING, id="coding")],
)  # fmt: skip
def test_build_token_sequence_one_token_blocks_rejected_and_partials_supported(
    corpus: PromptCorpus,
) -> None:
    tokenizer = Tokenizer.from_pretrained("builtin")
    config = PromptConfig(block_size=8)
    generator = (
        CodingContentGenerator(config, tokenizer)
        if corpus == PromptCorpus.CODING
        else PromptGenerator(prompts=config, prefix_prompts=None, tokenizer=tokenizer)
    )
    trace_id = "block-size-boundary"
    generator._hash_id_corpus_rng.set_trace_id(trace_id)
    trace = {"hash_ids": [101, 102], "input_length": 2}
    with pytest.raises(ConfigurationError, match="block_size >= 2"):
        generator._build_token_sequence(2, trace["hash_ids"], 1)
    with pytest.raises(ConfigurationError, match="block_size >= 2"):
        _drive_worker_inproc(generator, [("s1", [trace])], trace_id, 1)

    trace = {"hash_ids": [101, 102, 103], "input_length": 17}
    tokens = generator._build_token_sequence(17, trace["hash_ids"], 8)
    parallel = _drive_worker_inproc(generator, [("s1", [trace])], trace_id, 8)
    _, turns = parallel[0]
    _, _, parallel_prompt, _ = turns[0]
    assert len(tokens) == 17
    assert len(generator._cache[103]) == 1
    assert parallel_prompt == tokenizer.decode(tokens)


@pytest.mark.parametrize(
    "make_tokenizer, toy_corpus",
    [
        param(make_sentencepiece_tokenizer, SENTENCEPIECE_CORPUS, id="single-spaces"),
        param(make_sentencepiece_whitespace_tokenizer, SENTENCEPIECE_WHITESPACE_CORPUS, id="whitespace-runs"),
    ],
)  # fmt: skip
@pytest.mark.parametrize(
    "corpus",
    [param(PromptCorpus.SONNET, id="sonnet"), param(PromptCorpus.CODING, id="coding")],
)  # fmt: skip
def test_hash_blocks_sentencepiece_reencode_to_requested_length(
    corpus: PromptCorpus,
    make_tokenizer: Callable[[], Tokenizer],
    toy_corpus: str,
) -> None:
    """NVBug 7067483: in-process and worker hash-block prompts keep their token count."""
    tokenizer = make_tokenizer()
    config = PromptConfig(block_size=18)
    if corpus == PromptCorpus.CODING:
        generator = CodingContentGenerator(config, tokenizer)
        pool = tokenizer.encode(toy_corpus)
        generator._tool_pool = generator._tokenized_corpus = pool
        generator._corpus_size = len(pool)
    else:
        with patch("builtins.open", mock_open(read_data=toy_corpus)):
            generator = PromptGenerator(
                prompts=config, prefix_prompts=None, tokenizer=tokenizer
            )
    trace_id = "sentencepiece-drift"
    generator._hash_id_corpus_rng.set_trace_id(trace_id)
    in_process = {
        n: [
            len(tokenizer.encode(generator._generate_cached_prompt(n, ids, 18)))
            for ids in ([3 * i, 3 * i + 1, 3 * i + 2] for i in range(50))
        ]
        for n in (54, 80)
    }
    generator._cache.clear()
    traces = [
        {"hash_ids": [1000 + 3 * i, 1001 + 3 * i, 1002 + 3 * i], "input_length": 45}
        for i in range(50)
    ]

    [(_, turns)] = _drive_worker_inproc(generator, [("s1", traces)], trace_id, 18)

    assert in_process == {54: [54] * 50, 80: [80] * 50}
    assert [len(tokenizer.encode(prompt)) for _, _, prompt, _ in turns] == [45] * 50


def test_process_batch_literal_text_with_one_token_blocks_preserved() -> None:
    tokenizer = Tokenizer.from_pretrained("builtin")
    generator = PromptGenerator(
        prompts=PromptConfig(block_size=1),
        prefix_prompts=None,
        tokenizer=tokenizer,
    )
    trace = {
        "hash_ids": [101, 102],
        "input_length": 2,
        "text_input": "recorded prompt",
    }

    results = _drive_worker_inproc(generator, [("s1", [trace])], "literal-text", 1)

    assert results == [("s1", [(None, None, "recorded prompt", None)])]


def test_parallel_convert_matches_in_process(real_prompt_generator):
    """In-process ``_build_token_sequence`` output equals the worker ``_process_batch`` path byte-for-byte over the same inputs."""
    pg = real_prompt_generator
    trace_id = "abcdef0123456789"
    block_size = 4

    pg._hash_id_corpus_rng.set_trace_id(trace_id)
    pg._cache.clear()

    # Mixed layout: one exact-tile (8/4) and one last-partial (6 = 4 + 2).
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
    for tr in traces:
        tokens = pg._build_token_sequence(
            tr["input_length"], tr["hash_ids"], block_size
        )
        in_process_prompts.append(
            pg.tokenizer.decode(tokens, skip_special_tokens=False)
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
