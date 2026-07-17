# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GraphIRReplayStrategy honors ``--dataset-sampling-strategy``.

Every cross-trace draw in the lane fan-out / recycle loop routes through
``GraphIRReplayStrategy._draw_index(x, total)``, which remaps the historical
monotonic ``x % total`` draw to a strategy-selected trace index:

* ``sequential`` (or ``None``) -> ``x % total`` byte-for-byte (the historical
  cursor-with-wrap draw; a golden sequence pins this).
* ``shuffle`` -> a seeded per-pass permutation: each pass of ``total`` draws
  covers every index exactly once (without replacement), deterministic under a
  fixed ``t_star_random_seed`` and sensitive to it.
* ``random`` -> coerced to ``shuffle`` semantics in this single-pass-per-cycle
  context (no dup/omission within a pass); random == shuffle here.

These construct the REAL strategy over REAL weka traces (no mocks) and exercise
the real ``_draw_index`` / ``_resolve_pass0_lanes`` path. Deselected by default;
run with ``-m component_integration``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest

pytestmark = [pytest.mark.component_integration]

_FIX_DIR = Path(__file__).parents[2] / "unit" / "graph" / "fixtures"
_MIN = _FIX_DIR / "weka_min.json"


class _Issuer:
    """Minimal CreditIssuer stub: the sampling path never issues a credit."""

    def issue_graph_credit(self, *a: Any, **k: Any) -> bool:
        return True

    def mark_graph_sending_complete(self) -> None: ...

    def graph_all_returned(self) -> bool:
        return True

    def set_graph_all_returned_event(self) -> None: ...


def _parsed_with_n_traces(n: int):
    """Real weka parse whose corpus is ``n`` distinct traces (id-cloned).

    The single ``weka_min`` trace is cloned into ``n`` distinct trace ids so the
    draw abstraction sees a corpus of size ``n`` to permute over.
    """
    from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace

    base = from_weka_trace(str(_MIN))
    t0 = base.traces[0]
    traces = [msgspec.structs.replace(t0, id=f"{t0.id}#{i}") for i in range(n)]
    return msgspec.structs.replace(base, traces=list(traces))


def _make_strategy(parsed, *, sampling: Any, seed: int = 42):
    from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

    return GraphIRReplayStrategy(
        credit_issuer=_Issuer(),
        parsed_graph=parsed,
        register_observer=lambda obs: None,
        start_min_ratio=0.0,
        start_max_ratio=0.0,
        t_star_random_seed=seed,
        dataset_sampling_strategy=sampling,
        max_concurrent_traces=8,
    )


# --------------------------------------------------------------------------- #
# (a) SEQUENTIAL / None: byte-identical to the historical ``x % total`` draw.
# --------------------------------------------------------------------------- #


def test_sequential_draw_index_is_byte_identical_golden():
    """``sequential`` maps ``x -> x % total`` exactly (pinned golden sequence)."""
    n = 5
    parsed = _parsed_with_n_traces(n)
    from aiperf.plugin.enums import DatasetSamplingStrategy

    strategy = _make_strategy(parsed, sampling=DatasetSamplingStrategy.SEQUENTIAL)

    drawn = [strategy._draw_index(x, n) for x in range(3 * n)]
    # Golden: pure sequential-with-wrap over 3 full passes.
    assert drawn == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert drawn == [x % n for x in range(3 * n)]


def test_none_sampling_draw_index_matches_sequential():
    """An unset (``None``) strategy is the historical sequential draw."""
    n = 4
    parsed = _parsed_with_n_traces(n)
    strategy = _make_strategy(parsed, sampling=None)

    drawn = [strategy._draw_index(x, n) for x in range(2 * n)]
    assert drawn == [x % n for x in range(2 * n)]


def test_sequential_pass0_trace_ids_are_corpus_order():
    """Real ``_resolve_pass0_lanes`` under ``sequential`` yields ``traces[i]``."""
    n = 5
    parsed = _parsed_with_n_traces(n)
    from aiperf.plugin.enums import DatasetSamplingStrategy

    strategy = _make_strategy(parsed, sampling=DatasetSamplingStrategy.SEQUENTIAL)
    traces = list(parsed.traces)

    pass0, cursor = strategy._resolve_pass0_lanes(traces, n)
    assert [t.id for t in pass0] == [t.id for t in traces]
    assert cursor == n

    # Draw-counter -> trace id over two passes is byte-identical to traces[x % N].
    drawn_ids = [traces[strategy._draw_index(x, n)].id for x in range(2 * n)]
    assert drawn_ids == [traces[x % n].id for x in range(2 * n)]


# --------------------------------------------------------------------------- #
# (b) SHUFFLE (seeded): without-replacement permutation, deterministic + salted.
# --------------------------------------------------------------------------- #


def test_shuffle_single_pass_is_permutation_without_replacement():
    """A single ``shuffle`` pass covers all N indices exactly once (no dup/omit)."""
    n = 8
    parsed = _parsed_with_n_traces(n)
    from aiperf.plugin.enums import DatasetSamplingStrategy

    strategy = _make_strategy(parsed, sampling=DatasetSamplingStrategy.SHUFFLE)

    perm = [strategy._draw_index(x, n) for x in range(n)]
    assert sorted(perm) == list(range(n))  # permutation: every index exactly once
    # Cheap + consistent: repeated draws within the same pass are stable.
    assert [strategy._draw_index(x, n) for x in range(n)] == perm


def test_shuffle_second_pass_is_a_fresh_permutation():
    """Pass 1 (indices N..2N-1) is its own without-replacement permutation."""
    n = 8
    parsed = _parsed_with_n_traces(n)
    from aiperf.plugin.enums import DatasetSamplingStrategy

    strategy = _make_strategy(parsed, sampling=DatasetSamplingStrategy.SHUFFLE)

    pass1 = [strategy._draw_index(x, n) for x in range(n, 2 * n)]
    assert sorted(pass1) == list(range(n))


def test_shuffle_is_deterministic_for_same_seed():
    """Two constructions with the same seed produce the SAME permutation."""
    n = 8
    from aiperf.plugin.enums import DatasetSamplingStrategy

    a = _make_strategy(
        _parsed_with_n_traces(n),
        sampling=DatasetSamplingStrategy.SHUFFLE,
        seed=42,
    )
    b = _make_strategy(
        _parsed_with_n_traces(n),
        sampling=DatasetSamplingStrategy.SHUFFLE,
        seed=42,
    )
    assert [a._draw_index(x, n) for x in range(n)] == [
        b._draw_index(x, n) for x in range(n)
    ]


def test_shuffle_differs_for_different_seed():
    """A different seed yields a different permutation order."""
    n = 8
    from aiperf.plugin.enums import DatasetSamplingStrategy

    a = _make_strategy(
        _parsed_with_n_traces(n),
        sampling=DatasetSamplingStrategy.SHUFFLE,
        seed=42,
    )
    b = _make_strategy(
        _parsed_with_n_traces(n),
        sampling=DatasetSamplingStrategy.SHUFFLE,
        seed=1234,
    )
    perm_a = [a._draw_index(x, n) for x in range(n)]
    perm_b = [b._draw_index(x, n) for x in range(n)]
    # Both are valid permutations, but seeded differently -> different order.
    assert sorted(perm_a) == list(range(n))
    assert sorted(perm_b) == list(range(n))
    assert perm_a != perm_b


# --------------------------------------------------------------------------- #
# (c) RANDOM: single-pass -> coerced to without-replacement (== shuffle).
# --------------------------------------------------------------------------- #


def test_random_single_pass_is_without_replacement():
    """``random`` in single-pass mode covers all N indices exactly once."""
    n = 8
    parsed = _parsed_with_n_traces(n)
    from aiperf.plugin.enums import DatasetSamplingStrategy

    strategy = _make_strategy(parsed, sampling=DatasetSamplingStrategy.RANDOM)

    perm = [strategy._draw_index(x, n) for x in range(n)]
    assert sorted(perm) == list(range(n))  # no dup, no omission


def test_random_equals_shuffle_for_same_seed():
    """``random`` is coerced to ``shuffle`` semantics: identical order per seed."""
    n = 8
    from aiperf.plugin.enums import DatasetSamplingStrategy

    rnd = _make_strategy(
        _parsed_with_n_traces(n),
        sampling=DatasetSamplingStrategy.RANDOM,
        seed=42,
    )
    shf = _make_strategy(
        _parsed_with_n_traces(n),
        sampling=DatasetSamplingStrategy.SHUFFLE,
        seed=42,
    )
    assert [rnd._draw_index(x, n) for x in range(n)] == [
        shf._draw_index(x, n) for x in range(n)
    ]
