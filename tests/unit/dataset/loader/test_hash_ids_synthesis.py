# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
from unittest.mock import MagicMock, patch

from aiperf.dataset.loader.hash_ids_synthesis import (
    HashIdsPromptRequest,
    HashIdsPromptSynthesisMixin,
)


def test_mixin_decodes_via_parallel_decode_for_hash_id_requests():
    """Non-empty hash_ids requests build a token sequence then go through"""
    pg = MagicMock()
    pg.tokenizer.resolved_name = "test-tok"
    pg._build_token_sequence.return_value = [10, 20, 30]

    class _Loader(HashIdsPromptSynthesisMixin):
        pass

    loader = _Loader()
    loader.prompt_generator = pg
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    requests = [HashIdsPromptRequest(key="a", hash_ids=[1, 2], input_length=10)]
    with patch(
        "aiperf.dataset.loader.hash_ids_synthesis.parallel_decode",
        return_value=["decoded-prompt"],
    ) as mock_decode:
        result = loader.synthesize_prompts_from_hash_ids(requests)

    assert result == {"a": "decoded-prompt"}
    mock_decode.assert_called_once()
    pg._build_token_sequence.assert_called_once_with(10, [1, 2], 64)


def test_mixin_falls_back_to_generator_for_empty_hash_ids():
    pg = MagicMock()
    pg.generate.return_value = "synth"
    pg.tokenizer.resolved_name = "test-tok"

    class _Loader(HashIdsPromptSynthesisMixin):
        pass

    loader = _Loader()
    loader.prompt_generator = pg
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    requests = [HashIdsPromptRequest(key="a", hash_ids=[], input_length=20)]
    result = loader.synthesize_prompts_from_hash_ids(requests)
    assert result == {"a": "synth"}
    pg.generate.assert_called_once_with(mean=20, stddev=0, hash_ids=[])


class _Loader(HashIdsPromptSynthesisMixin):
    pass


def _make_mixin_with_corpus():
    """Build a mixin instance with a 1000-token mock corpus + a stub tokenizer"""
    pg = MagicMock()
    pg._tokenized_corpus = list(range(10000, 11000))
    pg._corpus_size = 1000
    pg.tokenizer.decode.side_effect = lambda toks: "|".join(str(t) for t in toks)

    loader = _Loader()
    loader.prompt_generator = pg
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    return loader


def test_sample_partial_tail_deterministic_within_process():
    loader = _make_mixin_with_corpus()
    a = loader.sample_partial_tail(20, "trace_t1:turn_3:partial_tail")
    b = loader.sample_partial_tail(20, "trace_t1:turn_3:partial_tail")
    assert a == b


def test_sample_partial_tail_differs_by_seed():
    loader = _make_mixin_with_corpus()
    a = loader.sample_partial_tail(20, "seed_a")
    b = loader.sample_partial_tail(20, "seed_b")
    assert a != b


def test_sample_partial_tail_zero_tokens_returns_empty():
    loader = _make_mixin_with_corpus()
    assert loader.sample_partial_tail(0, "any") == ""


def test_sample_partial_tail_uses_sha256_keyed_offset_not_python_hash():
    """sha256 is stable across processes (PYTHONHASHSEED-independent); Python's"""
    loader = _make_mixin_with_corpus()
    seed = "deterministic_seed_test"
    digest = hashlib.sha256(seed.encode()).digest()
    expected_offset = int.from_bytes(digest[:8], "big") % max(
        loader.prompt_generator._corpus_size - 20, 1
    )
    expected_tokens = loader.prompt_generator._tokenized_corpus[
        expected_offset : expected_offset + 20
    ]
    expected = "|".join(str(t) for t in expected_tokens)

    actual = loader.sample_partial_tail(20, seed)
    assert actual == expected


def test_sample_partial_tail_handles_corpus_smaller_than_request():
    loader = _make_mixin_with_corpus()
    a = loader.sample_partial_tail(2000, "seed_x")
    b = loader.sample_partial_tail(2000, "seed_x")
    assert a == b
    assert a != ""


def test_sample_partial_tail_tokens_deterministic_within_process():
    loader = _make_mixin_with_corpus()
    a = loader.sample_partial_tail_tokens(20, "trace_t1:turn_3:partial_tail")
    b = loader.sample_partial_tail_tokens(20, "trace_t1:turn_3:partial_tail")
    assert a == b
    assert len(a) == 20


def test_sample_partial_tail_tokens_zero_returns_empty_list():
    loader = _make_mixin_with_corpus()
    assert loader.sample_partial_tail_tokens(0, "any") == []


def test_sample_partial_tail_tokens_matches_text_variant():
    """The text variant must equal ``decode(token_variant)`` — the two helpers"""
    loader = _make_mixin_with_corpus()
    seed = "trace_t1:turn_3:partial_tail"
    tokens = loader.sample_partial_tail_tokens(20, seed)
    text = loader.sample_partial_tail(20, seed)
    assert text == loader.prompt_generator.tokenizer.decode(tokens)


def test_sample_partial_tail_tokens_uses_sha256_keyed_offset():
    loader = _make_mixin_with_corpus()
    seed = "deterministic_seed_test"
    digest = hashlib.sha256(seed.encode()).digest()
    expected_offset = int.from_bytes(digest[:8], "big") % max(
        loader.prompt_generator._corpus_size - 20, 1
    )
    expected = list(
        loader.prompt_generator._tokenized_corpus[
            expected_offset : expected_offset + 20
        ]
    )
    actual = loader.sample_partial_tail_tokens(20, seed)
    assert actual == expected
