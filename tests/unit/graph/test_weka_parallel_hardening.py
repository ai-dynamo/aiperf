# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weka parallel-parse hardening.

* Bounded waits: ``_bounded_ordered_pool_map`` waits on each pool result with a bounded
  ``get(timeout=...)`` -- a raw ``multiprocessing.Pool`` never completes the
  in-flight task of a killed worker (OOM / SIGKILL), so an unbounded ``get()``
  presents as a silent indefinite CLI hang. Expiry must raise a clear
  ``RuntimeError`` naming that cause; ordering is preserved.

* Shared-memory attach: a pool worker attaches the parent-built shared-memory corpus WITHOUT
  first paying a private corpus build -- ``get_or_build_synthesizer(...,
  shared_corpus=...)`` constructs the generator directly on the supplied
  array, and decoded content stays byte-identical to a self-built synthesizer.
"""

from __future__ import annotations

import multiprocessing
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import pytest

from aiperf.common.environment import Environment
from aiperf.dataset.graph.adapters.shared.content import (
    _WORKER_SYNTH_CACHE,
    CorpusContentSynthesizer,
    get_or_build_synthesizer,
)
from aiperf.dataset.graph.adapters.weka import trace_parallel as parallel


@pytest.fixture(autouse=True)
def _fresh_synth_cache():
    """Isolate the process-level synthesizer cache from other tests."""
    CorpusContentSynthesizer.reset_worker_cache()
    yield
    CorpusContentSynthesizer.reset_worker_cache()


# --- W5: bounded pool result wait -------------------------------------------


class _ReadyResult:
    """AsyncResult stub that completes immediately."""

    def __init__(self, value: bytes) -> None:
        self._value = value

    def get(self, timeout: float | None = None) -> bytes:
        assert timeout is not None, "pool result wait must be bounded"
        return self._value


class _DeadWorkerResult:
    """AsyncResult stub for a task whose worker was killed: never completes."""

    def get(self, timeout: float | None = None) -> bytes:
        assert timeout is not None, "pool result wait must be bounded"
        raise multiprocessing.TimeoutError


class _FakePool:
    """apply_async returns the pre-scripted result for each submitted item."""

    def __init__(self, results: list[Any]) -> None:
        self._results = list(results)

    def apply_async(self, fn, args):  # noqa: ANN001, ARG002
        return self._results.pop(0)


def test_bounded_pool_map_preserves_input_order() -> None:
    results = [_ReadyResult(b"a"), _ReadyResult(b"b"), _ReadyResult(b"c")]
    out = list(
        parallel._bounded_ordered_pool_map(
            _FakePool(results), lambda item: item, ["a", "b", "c"], window_size=2
        )
    )
    assert out == [b"a", b"b", b"c"]


def test_bounded_pool_map_raises_runtime_error_on_dead_worker(monkeypatch) -> None:
    """A killed worker's never-completing result raises instead of hanging."""
    monkeypatch.setattr(
        Environment.DATASET, "WEKA_GRAPH_PARALLEL_ITEM_TIMEOUT_SECONDS", 0.01
    )
    results = [_ReadyResult(b"a"), _DeadWorkerResult()]
    stream = parallel._bounded_ordered_pool_map(
        _FakePool(results), lambda item: item, ["a", "b"], window_size=2
    )
    assert next(stream) == b"a"
    with pytest.raises(RuntimeError, match="killed"):
        next(stream)


# --- W6: shared corpus without private rebuild ------------------------------


def test_shared_corpus_synthesizer_skips_private_build_and_matches_bytes(
    monkeypatch,
    fake_tokenizer: None,  # noqa: ARG001
) -> None:
    """Constructing on a shared corpus pays NO private build; bytes identical."""
    from aiperf.dataset.generator.coding_content import CodingContentGenerator

    builds: list[int] = []
    original_build = CodingContentGenerator._build_tool_pool

    def counting_build(self) -> None:  # noqa: ANN001
        builds.append(1)
        return original_build(self)

    monkeypatch.setattr(CodingContentGenerator, "_build_tool_pool", counting_build)

    # Parent role: pays the corpus build exactly once.
    parent = CorpusContentSynthesizer("tok-w6", prompt_corpus="coding", root_seed=0)
    assert builds == [1]

    corpus = np.asarray(parent.corpus_tokens(), dtype=np.int32)

    # Worker role: constructed ON the shared array -- no second build.
    worker = CorpusContentSynthesizer(
        "tok-w6", prompt_corpus="coding", root_seed=0, shared_corpus=corpus
    )
    assert builds == [1], "shared-corpus construction must not rebuild the corpus"
    assert worker._pg._corpus_size == len(corpus)

    # Determinism is sacred: decode + partial tails byte-identical.
    parent_cache: dict[int, list[int]] = {}
    worker_cache: dict[int, list[int]] = {}
    parent_tokens = parent._decode_block_tokens(
        [1, 2, 3], cache=parent_cache, trace_id="t"
    )
    worker_tokens = worker._decode_block_tokens(
        [1, 2, 3], cache=worker_cache, trace_id="t"
    )
    assert [int(t) for t in worker_tokens] == [int(t) for t in parent_tokens]
    assert [int(t) for t in worker._sample_partial_tail_tokens(7, "n0:response")] == [
        int(t) for t in parent._sample_partial_tail_tokens(7, "n0:response")
    ]


def test_init_worker_attaches_shared_corpus_without_rebuild(
    monkeypatch,
    fake_tokenizer: None,  # noqa: ARG001
) -> None:
    """``_init_worker`` warms the cache on the shm corpus with zero build calls."""
    from aiperf.dataset.generator.coding_content import CodingContentGenerator

    # Keep the test process's signal handlers and HF env untouched.
    monkeypatch.setattr(parallel, "_install_hard_exit_on_sigterm", lambda: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def forbidden_build(self) -> None:  # noqa: ANN001
        raise AssertionError("worker init must not build a private corpus")

    monkeypatch.setattr(CodingContentGenerator, "_build_tool_pool", forbidden_build)

    corpus = np.arange(512, dtype=np.int32)
    shm = shared_memory.SharedMemory(create=True, size=corpus.nbytes)
    try:
        np.ndarray(corpus.shape, dtype=np.int32, buffer=shm.buf)[:] = corpus
        args = parallel._WorkerInitArgs(
            tokenizer_name="tok-init",
            prompt_corpus="coding",
            root_seed=3,
            shm_name=shm.name,
            corpus_len=512,
        )

        parallel._init_worker(args)

        # _init_worker is fail-soft, so an empty cache means the private-build
        # path fired (and was swallowed); the cache MUST be warm on the shm array.
        synth = _WORKER_SYNTH_CACHE[("tok-init", "coding", 3)]
        assert synth._pg._corpus_size == 512
        assert [int(t) for t in synth._pg._tokenized_corpus[:4]] == [0, 1, 2, 3]
    finally:
        parallel._worker_corpus_shm = None
        shm.close()
        shm.unlink()


def test_get_or_build_rebinds_cached_synthesizer_to_shared_corpus(
    fake_tokenizer: None,  # noqa: ARG001
) -> None:
    """A cache HIT with a shared corpus rebinds instead of keeping the private one."""
    first = get_or_build_synthesizer("tok-hit", prompt_corpus="coding", root_seed=0)
    replacement = np.arange(64, dtype=np.int32)
    second = get_or_build_synthesizer(
        "tok-hit", prompt_corpus="coding", root_seed=0, shared_corpus=replacement
    )
    assert second is first
    assert second._pg._corpus_size == 64
