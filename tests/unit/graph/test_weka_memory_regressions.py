# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import numpy as np

from aiperf.common.environment import Environment


def test_hf_rows_are_streamed_lazily(monkeypatch) -> None:
    from aiperf.dataset.graph.adapters.weka import trace as weka_trace

    calls: list[dict[str, Any]] = []

    def load_dataset(*_args, **kwargs):  # noqa: ANN001
        calls.append(dict(kwargs))
        return iter([{"id": "row-0"}, {"id": "row-1"}])

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=load_dataset),
    )

    rows = weka_trace._load_hf_rows("org/weka-corpus", split="train", revision="rev")
    assert calls == []

    assert next(rows) == {"id": "row-0"}
    assert calls == [{"split": "train", "streaming": True, "revision": "rev"}]


def test_parse_items_streams_decoded_results_into_merge(monkeypatch) -> None:
    from aiperf.dataset.graph.adapters.weka import trace_parallel as parallel

    events: list[str] = []

    def fake_run_pool_streaming(
        _worker_fn: Any,
        _work_items: Any,
        **_kwargs: Any,
    ) -> Iterator[bytes]:
        events.append("pool_start")
        yield b"one"
        events.append("after_one")
        yield b"two"

    def fake_decode(blob: bytes) -> str:
        decoded = blob.decode()
        events.append(f"decode:{decoded}")
        return decoded

    def fake_merge(per_row: Any) -> str:
        assert not isinstance(per_row, list)
        events.append(f"merge_arg:{type(per_row).__name__}")
        iterator = iter(per_row)
        assert next(iterator) == "one"
        assert events == ["merge_arg:generator", "pool_start", "decode:one"]
        assert next(iterator) == "two"
        assert events == [
            "merge_arg:generator",
            "pool_start",
            "decode:one",
            "after_one",
            "decode:two",
        ]
        return "merged"

    monkeypatch.setattr(parallel, "_run_pool_streaming", fake_run_pool_streaming)
    monkeypatch.setattr(parallel, "decode_parsed_graph_msgpack", fake_decode)
    monkeypatch.setattr(parallel, "merge_parsed_graphs", fake_merge)

    assert (
        parallel.parse_items(
            [parallel._WorkItem(source="src#0", path=None, row={"id": "row-0"})],
            source_label="org/weka-corpus",
            threshold=0,
            workers=1,
            parse_kwargs={},
        )
        == "merged"
    )


def test_prefetch_multiplier_field_default_ratchet() -> None:
    """The Weka prefetch-window multiplier default must not regress below 16
    (window 256 at the auto 16 workers -- covers the rows remaining behind the
    heaviest full-corpus trace so fast workers do not stall head-of-line).

    Assert the declared FIELD default on the settings model, NOT the runtime
    ``Environment.DATASET`` value: the latter is env-overridable
    (``AIPERF_DATASET_WEKA_GRAPH_PARALLEL_PREFETCH_MULTIPLIER``), so the
    per-leg ABAB overrides used to measure this change would otherwise make the
    ratchet lie."""
    field = type(Environment.DATASET).model_fields[
        "WEKA_GRAPH_PARALLEL_PREFETCH_MULTIPLIER"
    ]
    assert field.default >= 16


def test_weka_worker_auto_matches_agentx(monkeypatch) -> None:
    from aiperf.dataset.graph.adapters.weka import trace_parallel as parallel

    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(Environment.DATASET, "WEKA_GRAPH_PARALLEL_WORKERS", 0)
    monkeypatch.setattr(Environment.DATASET, "WEKA_GRAPH_PARALLEL_AUTO_MAX_WORKERS", 16)

    assert parallel._get_workers(item_count=393, override=None) == 16
    assert parallel._get_workers(item_count=4, override=None) == 4
    assert parallel._get_workers(item_count=393, override=2) == 2
    assert parallel._get_workers(item_count=393, override=0) == 16

    monkeypatch.setattr(Environment.DATASET, "WEKA_GRAPH_PARALLEL_WORKERS", 3)
    assert parallel._get_workers(item_count=393, override=None) == 3


def test_numpy_shared_corpus_wraparound_matches_list_corpus() -> None:
    from aiperf.dataset.graph.adapters.shared.content import (
        CorpusContentSynthesizer,
    )

    class _Generator:
        def __init__(self, corpus) -> None:  # noqa: ANN001
            self._tokenized_corpus = corpus
            self._corpus_size = len(corpus)
            self._cache: dict[int, list[int]] = {}

    class _HashRng:
        def reseed_for_hash_id(self, _hash_id: int) -> None:
            return

        def randrange(self, _upper: int) -> int:
            return 8

    def decode(corpus) -> list[int]:  # noqa: ANN001
        synth = object.__new__(CorpusContentSynthesizer)
        synth._pg = _Generator(corpus)
        synth._hash_id_corpus_rng = _HashRng()
        synth._block_size = 5
        return CorpusContentSynthesizer._decode_block_tokens(synth, [123])

    expected = [8, 9, 0, 1, 2]
    assert decode(list(range(10))) == expected
    assert decode(np.asarray(list(range(10)), dtype=np.int32)) == expected


def _offset_synth(corpus, *, block_size: int, seed: int = 98765):
    """A CorpusContentSynthesizer bound to a tiny stub corpus + a REAL
    HashIdRandomGenerator (so offsets vary per hash id), mirroring
    ``test_numpy_shared_corpus_wraparound_matches_list_corpus`` but exercising
    the real reseed/randrange path both decoders share."""
    import types

    from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
    from aiperf.dataset.graph.adapters.shared.content import (
        CorpusContentSynthesizer,
    )

    synth = object.__new__(CorpusContentSynthesizer)
    synth._pg = types.SimpleNamespace(
        _tokenized_corpus=corpus, _corpus_size=len(corpus), _cache={}
    )
    synth._hash_id_corpus_rng = HashIdRandomGenerator(seed, _internal=True)
    synth._block_size = block_size
    synth._offset_cache_corpus_size = None
    return synth


def test_offset_cache_decode_matches_list_cache_across_backings() -> None:
    """The offset-cached decode is a byte-identical, memory-lean twin of the
    list-cache ``_decode_block_tokens`` on the MISS and the REPEAT (hit) path, for
    both a list and an ``np.int32`` corpus backing, wraparound blocks included.

    Both decoders issue the same ``reseed_for_hash_id(h)`` + ``randrange`` pair on
    a miss (a full reseed -> identical ``start``) and touch no RNG on a repeat, so
    the token streams must match exactly; only the cache SHAPE differs
    (``int`` offset vs decoded ``list``). The tiny corpus forces frequent
    wraparound, so BOTH branches of the offset decode (the extend-from-slice
    fast path and the verbatim two-step wraparound path) are pinned."""
    bs = 16
    base = list(range(40))  # tiny corpus forces frequent wraparound at bs=16
    # ~500 hash ids: positive u64-ish + negative virtual ids, deterministic.
    hash_ids = [((i * 2654435761) % (2**63)) + 1 for i in range(300)]
    hash_ids += [-(i + 1) for i in range(200)]

    for corpus in (base, np.asarray(base, dtype=np.int32)):
        s_list = _offset_synth(corpus, block_size=bs)
        s_off = _offset_synth(corpus, block_size=bs)
        dict_cache: dict[int, list[int]] = {}
        off_cache: dict[int, int] = {}

        first_list = s_list._decode_block_tokens(
            hash_ids, block_size=bs, cache=dict_cache
        )
        first_off = s_off._decode_block_tokens_offset_cached(
            hash_ids, block_size=bs, offset_cache=off_cache
        )
        assert first_off == first_list

        # Repeat decode: both caches hit; each must reproduce its first decode.
        repeat_list = s_list._decode_block_tokens(
            hash_ids, block_size=bs, cache=dict_cache
        )
        repeat_off = s_off._decode_block_tokens_offset_cached(
            hash_ids, block_size=bs, offset_cache=off_cache
        )
        assert repeat_list == first_list
        assert repeat_off == first_off == repeat_list

        # Every covered block is exactly block_size (wraparound blocks included).
        assert len(first_off) == len(hash_ids) * bs
        # Guard the guard: BOTH offset-decode branches were actually exercised
        # (some offsets wrap past the corpus end, some do not).
        assert any(v > len(base) - bs for v in off_cache.values())
        assert any(v <= len(base) - bs for v in off_cache.values())
        # Cache shape: plain int offsets keyed by the (int) hash id.
        assert all(type(v) is int for v in off_cache.values())


def test_offset_cache_fails_loud_on_corpus_size_change() -> None:
    """A rebind that changes ``_corpus_size`` under a populated offset cache fails
    loud: the cached offsets would otherwise silently reproduce wrong bytes (the
    corpus-immutability contract on ``_decode_block_tokens_offset_cached``)."""
    import pytest

    synth = _offset_synth(list(range(40)), block_size=16, seed=123)
    off: dict[int, int] = {}
    synth._decode_block_tokens_offset_cached([1, 2, 3], block_size=16, offset_cache=off)

    synth._pg._tokenized_corpus = list(range(80))
    synth._pg._corpus_size = 80
    with pytest.raises(RuntimeError, match="corpus size changed"):
        synth._decode_block_tokens_offset_cached(
            [1, 2, 3], block_size=16, offset_cache=off
        )
