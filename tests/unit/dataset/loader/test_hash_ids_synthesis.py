# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
from unittest.mock import MagicMock, mock_open, patch

from aiperf.config.dataset.content import PrefixPromptConfig, PromptConfig
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.hash_ids_synthesis import (
    HashIdsPromptRequest,
    HashIdsPromptSynthesisMixin,
)


def test_mixin_decodes_via_parallel_decode_for_hash_id_requests():
    """Non-empty hash_ids requests build a token sequence then decode that full sequence."""
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
    assert mock_decode.call_args.args[0] == [[10, 20, 30]]
    pg._build_token_sequence.assert_called_once_with(10, [1, 2], 64)


def test_mixin_decodes_identical_full_sequences_once():
    """Duplicate assembled token sequences share a single ``parallel_decode`` call."""
    pg = MagicMock()
    pg.tokenizer.resolved_name = "test-tok"
    pg._build_token_sequence.side_effect = lambda n, hids, bs: {
        (1, 2): [10, 11, 20],
        (1, 3): [10, 11, 30, 31],
    }[tuple(hids)]

    class _Loader(HashIdsPromptSynthesisMixin):
        pass

    loader = _Loader()
    loader.prompt_generator = pg
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    requests = [
        HashIdsPromptRequest(key="a", hash_ids=[1, 2], input_length=3),
        HashIdsPromptRequest(key="b", hash_ids=[1, 3], input_length=4),
        HashIdsPromptRequest(key="c", hash_ids=[1, 2], input_length=3),
    ]

    def fake_decode(seqs, *args, **kwargs):
        return ["|".join(str(t) for t in seq) for seq in seqs]

    with patch(
        "aiperf.dataset.loader.hash_ids_synthesis.parallel_decode",
        side_effect=fake_decode,
    ) as mock_decode:
        result = loader.synthesize_prompts_from_hash_ids(requests)

    assert mock_decode.call_count == 1
    decoded_seqs = mock_decode.call_args.args[0]
    assert len(decoded_seqs) == 2
    assert {tuple(s) for s in decoded_seqs} == {(10, 11, 20), (10, 11, 30, 31)}
    assert result["a"] == "10|11|20"
    assert result["b"] == "10|11|30|31"
    assert result["c"] == result["a"]


def test_mixin_oracle_decodes_full_sequence_for_exact_partial_and_prefix_tail(
    mock_tokenizer_cls,
):
    """Replay prompts equal ``decode(assembled_tokens)``, not joined per-block strings.

    The mock tokenizer inserts spaces between ids, so ``decode(a)+decode(b)``
    differs from ``decode(a+b)`` at block boundaries. That is the segment-join
    drift the production path must not ship.
    """
    tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
    corpus = " ".join([f"word{i}" for i in range(1024)]) + "\n"
    with patch("builtins.open", mock_open(read_data=corpus)):
        pg = PromptGenerator(
            prompts=PromptConfig(block_size=4),
            prefix_prompts=PrefixPromptConfig(pool_size=None, length=None),
            tokenizer=tokenizer,
        )

    class _Loader(HashIdsPromptSynthesisMixin):
        pass

    loader = _Loader()
    loader.prompt_generator = pg
    loader._tokenizer_name = "gpt2"
    loader._trust_remote_code = False
    loader._tokenizer_revision = "main"
    loader._block_size = 4

    cases = [
        ("exact", [11, 22], 8),
        ("partial", [33, 44], 6),
        ("prefix_tail", [55], 6),
    ]
    requests = [
        HashIdsPromptRequest(key=key, hash_ids=hids, input_length=n)
        for key, hids, n in cases
    ]

    pg._cache.clear()
    assembled: dict[tuple[tuple[int, ...], int], list[int]] = {}
    orig_build = pg._build_token_sequence

    def capturing_build(n, hids, bs):
        tokens = orig_build(n, hids, bs)
        assembled[(tuple(hids), n)] = list(tokens)
        return tokens

    pg._build_token_sequence = capturing_build

    def decode_full_sequences(seqs, *args, **kwargs):
        return [
            pg.tokenizer.decode(list(seq), skip_special_tokens=False) for seq in seqs
        ]

    with patch(
        "aiperf.dataset.loader.hash_ids_synthesis.parallel_decode",
        side_effect=decode_full_sequences,
    ) as mock_decode:
        result = loader.synthesize_prompts_from_hash_ids(requests)

    expected = {
        key: pg.tokenizer.decode(assembled[(tuple(hids), n)], skip_special_tokens=False)
        for key, hids, n in cases
    }
    joined_blocks = {
        key: "".join(
            pg.tokenizer.decode(pg._cache[hid], skip_special_tokens=False)
            for hid in hids
        )
        for key, hids, n in cases
    }

    assert expected["exact"] != joined_blocks["exact"], (
        "oracle is degenerate: mock decode(full) accidentally equals joined blocks"
    )
    assert expected["prefix_tail"] != joined_blocks["prefix_tail"]
    assert result == expected
    passed = [tuple(seq) for seq in mock_decode.call_args.args[0]]
    assert set(passed) == {tuple(assembled[(tuple(hids), n)]) for _, hids, n in cases}


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
    """Build a mixin with a 1000-token mock corpus and a stub tokenizer whose ``.decode`` returns a deterministic slice-keyed string."""
    pg = MagicMock()
    pg._tokenized_corpus = list(range(10000, 11000))  # 1000 tokens
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
    """The partial-tail offset comes from a cross-process-stable sha256 digest, not Python's builtin ``hash()``."""
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
    # Requesting more tokens than the corpus has must stay deterministic and
    # non-empty; the truncate-or-wrap policy is intentionally unspecified.
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
    """The text variant equals ``decode(token_variant)`` since both share the same offset and corpus slice."""
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
