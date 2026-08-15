# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sizes the four dynamo build tiers on a synthetic corpus and gates each memory optimization by calibrated ratio."""

from __future__ import annotations

import gc
import inspect
import math
import os
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pytest

from aiperf.common.tokenizer import BUILTIN_TOKENIZER_NAME
from tests.harness.dynamo_synth_corpus import (
    SyntheticCorpusShape,
    synthetic_corpus_shape,
    write_synthetic_dynamo_capture,
)

# POSIX-only ru_maxrss RSS accounting; the whole module is a POSIX isolate.
resource = pytest.importorskip("resource")

# The measurement tests are ``@pytest.mark.slow`` (deselected by default via
# ``pyproject.toml`` addopts ``not slow``), so their cost is opt-in:
#   uv run pytest -m slow tests/unit/dataset/graph/adapters/test_dynamo_corpus_scale_memory.py
#
# The four build tiers, and where each is allocated in the parse:
#   hash-id ints + lists  dynamo/{trace_reader,trie_lowering,trace}.py -- every turn
#                         re-lists its full prefix (~G*(T+1)/2 slots/node); read-time
#                         interning shares one int per UNIQUE value, so only list
#                         pointers/headers still scale with slot count.
#   decode cache          adapters/shared/content.py -- one cached int offset per
#                         unique hash id, blocks re-sliced from the corpus on demand.
#   resolution automaton  segment_trie/trie_content.py -- one flat state per unique block
#                         position; a TRANSIENT freed before emission, hence measured in
#                         its own phase isolate rather than the full-parse snapshot.
#   content pool          segment_trie/pool.py -- one content-addressed Segment per unique
#                         message segment (collapses to zero on the direct-store route).
#
# Each tier is extrapolated linearly to the confirmed real-capture scale (~1M nodes) so a
# per-node regression is caught at test scale (MBs) before it becomes tens of GB. Exact
# bytes vary by interpreter build, so every assertion gates a calibrated RATIO, never
# absolute bytes -- the load-bearing conclusions are tier ORDERING and extrapolated
# MAGNITUDE. CAVEAT on the @1M column: the extrapolation holds the synthetic chain shape
# fixed (H/N ~= 512 re-listed slots/node); real captures with deeper chains grow the
# hash-id tier SUPERLINEARLY in chain depth, so @1M magnitudes are chain-shape-conditional
# and pending real-capture re-validation via the ``AIPERF_TEST_DYNAMO_SCALE_NODES`` lane.
# The irreducible residue is the ~4.1 GB @1M hash-slot LIST-pointer term, which interning
# cannot shrink (every re-listed slot holds one pointer regardless of value sharing).

# --- calibrated default scale ----------------------------------------------
# S x T x G chosen so N ~= 6k nodes with a realistic chain shape (G unique
# blocks/turn, T turns => H/N = G*(T+1)/2 ~= 512 hash slots/node, matching real
# corpus captures) and the whole slow module runs in well under ~120 s.
_DEFAULT_SESSIONS = 150
_TURNS_PER_SESSION = 40
_NEW_BLOCKS_PER_TURN = 25
_BLOCK_SIZE = 16
_CONTENT_SEED = 1234
_EXTRAPOLATION_NODES = 1_000_000

# ``@pytest.mark.slow`` is deselected by default; opt in with ``-m slow``. The
# env override lets a developer size a manual corpus-scale run.
_SCALE_ENV = "AIPERF_TEST_DYNAMO_SCALE_NODES"
# nframes=1 keeps tracemalloc's per-allocation cost low (the top frame is the
# real allocation site, which is all ``statistics("filename")`` groups on) while
# the parse stays measurable in seconds rather than minutes.
_TRACE_NFRAMES = 1
# Measured total peak must stay within this multiple of the analytic per-tier
# model computed from the generator parameters (a calibrated RATIO, never an
# absolute-bytes gate): it catches a gross per-node regression without being
# fragile to interpreter-version object-layout drift.
_BUDGET_RATIO = 1.5
# Read-time interning shaves the recorded hash-id-int tier from one
# distinct int per re-listed slot to one per UNIQUE value. This calibrated RATIO
# gates that win permanently (following the ``_FLAT_BUDGET_RATIO`` precedent --
# calibrated ratio, never absolute bytes): the measured hash-id tier must stay at
# most this fraction of the distinct-int cost (``h`` slots x (int + pointer)).
# Post-intern measures ~0.32; a regression back to distinct ints per slot
# measures ~1.0, so 0.55 catches the regression with interpreter-drift headroom.
_HASH_INTERN_RATIO = 0.55
# Pool interning shaves the sid-string ADDRESSING tier (every node's
# prompt_segment_ids re-lists its whole chain) from one fresh hexdigest per slot to
# one canonical object per UNIQUE segment. This calibrated RATIO gates that win
# permanently (never absolute bytes): the retained ``segment_id`` hexdigest bytes
# must stay at most this fraction of ``n_unique_segments * sys.getsizeof(sid)``.
# Post-intern measures ~1.0-1.1x (the floor itself); a 246k-duplicate regression
# overshoots ~10x, so 2.0 catches it with headroom for the getsizeof-vs-tracemalloc
# base mismatch (see ``_SID_STR``).
_SID_INTERN_RATIO = 2.0
# A 32-hex-char ``segment_id`` (blake2b digest_size=16 -> 32 hex chars). getsizeof
# reports ~81 B for this compact-ASCII str; tracemalloc's per-allocation accounting
# observes ~73 B/str, so the ratio-gate denominator (getsizeof) runs ~11% HIGHER
# than the measured numerator base -- the ``_SID_INTERN_RATIO`` headroom absorbs
# that unit mismatch so it is not silently load-bearing.
_SID_STR = sys.getsizeof("0" * 32)

_T = TypeVar("_T")

# --- CPython object-layout sizes for the analytic model --------------------
# Measured at import (getsizeof) or documented structural constants -- per-object
# sizes scaled by param-derived element counts, NOT hardcoded byte totals.
_PTR = 8
_INT64 = sys.getsizeof(1 << 63)  # a u64-valued int object (above the small-int cache)
_LIST_HDR = sys.getsizeof([])  # empty list container header
_LIST_BLOCK = sys.getsizeof([0] * _BLOCK_SIZE)  # one decoded block's token list
# Amortized combined-dict slot (index array + key/value/hash entry at the ~2/3
# load factor CPython dicts grow to); covers the decode cache and pool dicts.
_DICT_ENTRY = 110
# Per unique content block: a slots ``Segment`` plus its ``_by_id`` dict slot.
_SEGMENT_ENTRY = 150
# Per node, the persistent non-hash overhead: the ``AgentTraceRecord`` pydantic
# object graph, the ``TrieNode``/``TrieRequest``, the assembled ``LlmNode`` and
# its metadata dict.
_NODE_FIXED = 3600


def _resolve_scale() -> tuple[int, int, int]:
    """``(sessions, turns, blocks_per_turn)``; ``AIPERF_TEST_DYNAMO_SCALE_NODES`` sizes sessions to hit an override node count."""
    turns, g = _TURNS_PER_SESSION, _NEW_BLOCKS_PER_TURN
    override = os.environ.get(_SCALE_ENV)
    if override:
        sessions = max(1, math.ceil(int(override) / turns))
    else:
        sessions = _DEFAULT_SESSIONS
    return sessions, turns, g


def _analytic_model_bytes(shape: SyntheticCorpusShape) -> int:
    """Sum the tiers coexisting at the emission peak from generator counts, excluding the transient resolution trie."""
    n, h, u = shape.n_nodes, shape.n_hash_slots, shape.n_unique_blocks
    # Read-time interning canonicalizes the recorded hash ints in
    # ``_collect_records``, so the int OBJECTS scale with UNIQUE values (``u``),
    # not total slots (``h``); only the per-slot list POINTERS (``h * _PTR``) and
    # the per-node list headers (``n * _LIST_HDR``) still scale with ``h``/``n``.
    hash_tier = u * _INT64 + h * _PTR + n * _LIST_HDR
    # Deliberately over-modeled: this term sizes a full per-hash list cache
    # (one ``_LIST_BLOCK`` of block_size ints per unique
    # hash), while the actual dynamo decode cache is one int OFFSET per unique hash
    # (``_decode_block_tokens_offset_cached``), so the true structural size is
    # ~``u * (_INT64 + _DICT_ENTRY)``. It is kept as a CONSERVATIVE per-tier
    # ceiling: shrinking it to the offset shape drops the model ~22 MB and leaves
    # the measured ~140 MB peak only ~3% under the 1.5x budget -- fragile to
    # interpreter object-layout drift. The interning-specific regression is gated
    # sharply by the ``_HASH_INTERN_RATIO`` hash-tier assertion instead, so this
    # looseness does not weaken the interning proof.
    decode_tier = u * (_LIST_BLOCK + _DICT_ENTRY)
    content_tier = u * _SEGMENT_ENTRY
    # Canonical sid-string addressing tier. Post-intern every node's
    # prompt_segment_ids shares one str object per UNIQUE segment (``S*(2T-1)+N``),
    # NOT per unique block (``u``) -- the floor scales with segments, not hashes.
    sid_tier = shape.n_unique_segments * _SID_STR
    fixed = n * _NODE_FIXED
    return hash_tier + decode_tier + content_tier + sid_tier + fixed


def _attribute_by_filename(snapshot: tracemalloc.Snapshot) -> dict[str, int]:
    """Map absolute source filename -> summed allocated bytes at the snapshot."""
    by: dict[str, int] = {}
    for stat in snapshot.statistics("filename"):
        fn = stat.traceback[0].filename
        by[fn] = by.get(fn, 0) + stat.size
    return by


def _tier_bytes(by_file: dict[str, int], *needles: str) -> int:
    """Sum bytes for every source file whose path contains any of ``needles``."""
    return sum(sz for fn, sz in by_file.items() if any(n in fn for n in needles))


def _segment_id_line_range() -> tuple[str, int, int]:
    """``(filename, first_lineno, last_lineno)`` of ``pool.segment_id``, resolved dynamically via ``inspect``."""
    # pool.py has THREE id-minting hexdigest sites (segment_id, text_segment_id,
    # raw_segment_id); the dynamo route mints only via segment_id, so the range is
    # scoped to that function alone. Resolving it live means a docstring-only edit to
    # the frozen pool.py cannot silently mis-target the addressing-tier split.
    from aiperf.dataset.graph.segment_trie.pool import segment_id

    lines, start = inspect.getsourcelines(segment_id)
    return segment_id.__code__.co_filename, start, start + len(lines) - 1


def _addressing_tier_bytes(snapshot: tracemalloc.Snapshot) -> int:
    """Bytes retained by the ``segment_id`` hexdigest line -- the sid-string tier, split out by lineno."""
    # statistics("lineno") groups by (filename, lineno), so summing only segment_id's
    # source range isolates the sid strings prompt_segment_ids retains, excluding the
    # sibling id mints and the resident Segment graph sharing the pool.py FILENAME tier.
    fname, lo, hi = _segment_id_line_range()
    total = 0
    for stat in snapshot.statistics("lineno"):
        frame = stat.traceback[0]
        if frame.filename == fname and lo <= frame.lineno <= hi:
            total += stat.size
    return total


def _extrapolate_gb(byte_count: int, n_nodes: int) -> float:
    """Linear per-node extrapolation of ``byte_count`` to ``_EXTRAPOLATION_NODES``."""
    return byte_count / n_nodes * _EXTRAPOLATION_NODES / 1e9


def _fmt_tier(label: str, byte_count: int, n_nodes: int) -> str:
    return (
        f"  {label:<34} {byte_count / 1e6:8.1f} MB  "
        f"{byte_count / n_nodes:8.0f} B/node  "
        f"-> {_extrapolate_gb(byte_count, n_nodes):6.1f} GB @ {_EXTRAPOLATION_NODES:,} nodes"
    )


def _force_serial_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the dynamo build to the in-process serial path.

    These are PARENT-PROCESS tracemalloc measurements. At the default corpus
    scale the build crosses DYNAMO_GRAPH_PARALLEL_THRESHOLD and runs in forked
    workers, where neither the parent's tracemalloc nor a parent-installed spy
    can observe anything -- the measurement silently becomes vacuous and the
    tier assertions gate nothing. Forcing serial is what makes the numbers mean
    what the assertions claim, not merely a way to keep the spy firing.
    """
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_WORKERS", 1)


def _prewarm_synthesizer() -> int:
    """Build the shared corpus synthesizer OUTSIDE any traced region and return the resolved content seed."""
    from aiperf.dataset.graph.adapters.shared.content import (
        CorpusContentSynthesizer,
        get_or_build_synthesizer,
        resolve_effective_root_seed,
    )

    CorpusContentSynthesizer.reset_worker_cache()
    seed = resolve_effective_root_seed(_CONTENT_SEED)
    get_or_build_synthesizer(
        BUILTIN_TOKENIZER_NAME, prompt_corpus="coding", root_seed=seed
    )
    return seed


def _traced_build(build: Callable[[], _T]) -> tuple[_T, int, tracemalloc.Snapshot]:
    """Run ``build()`` under ``tracemalloc`` with GC off; return ``(result, peak bytes, snapshot)``."""
    # The result stays referenced through the snapshot AND the peak read, so the structure
    # under test is fully attributed rather than partially collected. GC is disabled so a
    # collection pass mid-build cannot perturb the peak.
    gc.collect()
    gc.disable()
    try:
        tracemalloc.start(_TRACE_NFRAMES)
        tracemalloc.reset_peak()
        built = build()
        snapshot = tracemalloc.take_snapshot()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        gc.enable()
    return built, peak, snapshot


@pytest.fixture(scope="module")
def capture_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Write the synthetic capture once for the module's slow tests."""
    sessions, turns, g = _resolve_scale()
    path = tmp_path_factory.mktemp("dynamo_corpus") / "synth.jsonl"
    write_synthetic_dynamo_capture(
        path,
        sessions=sessions,
        turns_per_session=turns,
        new_blocks_per_turn=g,
        block_size=_BLOCK_SIZE,
        seed=1,
    )
    return str(path)


@pytest.fixture(scope="module")
def prebuilt_nodes(capture_path: str) -> tuple[list, int]:
    """Parse chains -> ``TrieNode``s ONCE (untraced) so each phase isolate attributes only its structure under test."""
    from aiperf.common.environment import Environment
    from aiperf.dataset.graph.adapters.dynamo.trace import _collect_chains
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import dynamo_trie_nodes

    chains = _collect_chains(
        capture_path, None, max_depth=Environment.DYNAMO.MAX_SUBAGENT_DEPTH
    )
    nodes, block_size, _ = dynamo_trie_nodes(chains, release_replay=True)
    return nodes, block_size


def test_synthetic_capture_parses_through_from_dynamo_trace(tmp_path: Path) -> None:
    """Fast guard (not slow-marked): a tiny synthetic capture parses cleanly end-to-end through ``from_dynamo_trace``."""
    # Pins the generator contract on every default unit run: the block-alignment gate
    # (_assert_block_aligned) and the covered-count ISL gate both accept the generated
    # input_length / hash-count relationship, and every node lowers to a prompt=[] trie
    # LlmNode addressed by the segment pool. Protects the slow measurements from silent
    # generator drift.
    from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
    from aiperf.dataset.graph.models import LlmNode

    path = tmp_path / "tiny.jsonl"
    write_synthetic_dynamo_capture(
        path,
        sessions=3,
        turns_per_session=4,
        new_blocks_per_turn=2,
        block_size=_BLOCK_SIZE,
        seed=7,
    )
    parsed = from_dynamo_trace(
        path, content_root_seed=_CONTENT_SEED, content_tokenizer="builtin"
    )
    assert parsed.segment_pool is not None and parsed.segment_pool.by_id
    # Dynamo emits one GraphRecord per independent session-tree (multi-graph), so
    # gather LlmNodes across every per-tree graph, not just the first (graph).
    graphs = list(parsed.graphs.values()) or [parsed.graph]
    llm_nodes = [n for g in graphs for n in g.nodes.values() if isinstance(n, LlmNode)]
    assert len(llm_nodes) == 3 * 4
    assert all(n.prompt == [] for n in llm_nodes)
    assert all(n.metadata["trie"]["prompt_segment_ids"] for n in llm_nodes), (
        "each node must address its prompt through the segment pool"
    )


@pytest.mark.slow
def test_full_parse_tier_measurement(
    capture_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Size the persistent build tiers on a REAL full ``from_dynamo_trace`` parse under ``tracemalloc``."""
    # The spy on stamp_theoretical_prefix_cache (its last call before the parse returns)
    # snapshots the peak window while the recorded hash lists, decode cache, content pool
    # and assembled nodes are ALL still live; a snapshot after the return would see every
    # transient already freed.
    sessions, turns, g = _resolve_scale()
    shape = synthetic_corpus_shape(
        sessions=sessions, turns_per_session=turns, new_blocks_per_turn=g
    )
    n_nodes = shape.n_nodes

    _prewarm_synthesizer()

    from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
    from aiperf.dataset.graph.segment_trie import prefix_cache

    captured: dict[str, tracemalloc.Snapshot] = {}
    orig_stamp = prefix_cache.stamp_theoretical_prefix_cache

    def _spy(*args, **kwargs) -> None:
        # *args/**kwargs, not a fixed arity: the production signature is
        # (llm_nodes, trie_nodes, block_size, on_counts=None) and the call site
        # passes three positional plus one keyword. A two-parameter spy raises
        # TypeError inside the parse, which surfaces only when the build is
        # serial (see _force_serial_build below).
        orig_stamp(*args, **kwargs)
        captured["peak_window"] = tracemalloc.take_snapshot()

    monkeypatch.setattr(prefix_cache, "stamp_theoretical_prefix_cache", _spy)
    _force_serial_build(monkeypatch)

    gc.collect()
    tracemalloc.start(_TRACE_NFRAMES)
    t0 = time.perf_counter()
    parsed = from_dynamo_trace(
        capture_path,
        content_root_seed=_CONTENT_SEED,
        content_tokenizer="builtin",
        release_replay=True,
    )
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert parsed.segment_pool is not None
    assert "peak_window" in captured, "spy never fired -- parse path changed?"
    by_file = _attribute_by_filename(captured["peak_window"])

    hash_id_bytes = _tier_bytes(
        by_file,
        "dynamo/trace_reader.py",
        "dynamo/trie_lowering.py",
        "dynamo/trace.py",
    )
    decode_bytes = _tier_bytes(by_file, "adapters/shared/content.py")
    content_bytes = _tier_bytes(by_file, "segment_trie/pool.py")
    trie_bytes = _tier_bytes(by_file, "segment_trie/trie_content.py")

    table = "\n".join(
        [
            f"[dynamo corpus-scale memory] full parse: N={n_nodes:,} nodes "
            f"H={shape.n_hash_slots:,} slots U={shape.n_unique_blocks:,} unique "
            f"blocks in {elapsed:.1f}s (nframes={_TRACE_NFRAMES})",
            _fmt_tier("hash-id ints+lists (trace/lower)", hash_id_bytes, n_nodes),
            _fmt_tier("decode cache (shared/content)", decode_bytes, n_nodes),
            _fmt_tier("content pool (segment_trie/pool)", content_bytes, n_nodes),
            _fmt_tier("trie_content (build artifacts)", trie_bytes, n_nodes),
            f"  {'MEASURED TOTAL PEAK':<34} {peak / 1e6:8.1f} MB  "
            f"{peak / n_nodes:8.0f} B/node  "
            f"-> {_extrapolate_gb(peak, n_nodes):6.1f} GB @ {_EXTRAPOLATION_NODES:,} nodes",
        ]
    )
    print("\n" + table)

    # Each named tier must attribute non-zero (a zero means the allocation site
    # moved and the measurement is silently mis-attributing).
    assert hash_id_bytes > 0, f"hash-id tier attributed 0 bytes\n{table}"
    assert decode_bytes > 0, f"decode-cache tier attributed 0 bytes\n{table}"
    assert content_bytes > 0, f"content-pool tier attributed 0 bytes\n{table}"
    assert trie_bytes > 0, f"trie_content tier attributed 0 bytes\n{table}"

    # Calibrated-ratio budget (never absolute bytes): measured peak within
    # 1.5x the analytic per-tier model derived from the generator parameters.
    model = _analytic_model_bytes(shape)
    assert peak <= _BUDGET_RATIO * model, (
        f"measured peak {peak / 1e6:.1f} MB exceeds "
        f"{_BUDGET_RATIO} x analytic model {model / 1e6:.1f} MB "
        f"(= {_BUDGET_RATIO * model / 1e6:.1f} MB) -- a per-node allocation "
        f"regression, not interpreter drift\n{table}"
    )

    # Read-time interning gate: the recorded hash-id tier must stay
    # well under the distinct-int cost (one int per re-listed slot). Measures
    # ~0.32 post-intern; a regression to distinct ints per slot measures ~1.0.
    hash_intern_budget = _HASH_INTERN_RATIO * shape.n_hash_slots * (_INT64 + _PTR)
    assert hash_id_bytes <= hash_intern_budget, (
        f"hash-id tier {hash_id_bytes / 1e6:.1f} MB exceeds {_HASH_INTERN_RATIO} x "
        f"the distinct-int cost {hash_intern_budget / 1e6:.1f} MB -- read-time "
        f"interning in _collect_records regressed. CAVEAT: this tier sums the "
        f"trace_reader / trie_lowering / trace needles, which also capture "
        f"record-model allocations, so a future record-model growth can fire this "
        f"assertion without an interning regression\n{table}"
    )


def _measure_pool_tier_at_peak(
    capture_path: str,
    real_stamp: Callable[[object, object], None],
    *,
    direct_store: object | None,
) -> tuple[int, int, int, tracemalloc.Snapshot]:
    """Full ``from_dynamo_trace``; return ``(pool.py bytes, total peak, resident segment count, peak snapshot)``."""
    # real_stamp must be the TRUE original (captured before any patch) so repeated calls
    # never wrap a prior spy. Resident count is len(segment_pool.by_id): 0 on the direct
    # write-through route, non-zero on the eager route.
    from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
    from aiperf.dataset.graph.segment_trie import prefix_cache

    _prewarm_synthesizer()
    captured: dict[str, tracemalloc.Snapshot] = {}

    def _spy(*args, **kwargs) -> None:
        # See the sibling spy: forward everything, the production signature has
        # four parameters.
        real_stamp(*args, **kwargs)
        captured["peak_window"] = tracemalloc.take_snapshot()

    prev = prefix_cache.stamp_theoretical_prefix_cache
    prefix_cache.stamp_theoretical_prefix_cache = _spy
    try:
        gc.collect()
        tracemalloc.start(_TRACE_NFRAMES)
        parsed = from_dynamo_trace(
            capture_path,
            content_root_seed=_CONTENT_SEED,
            content_tokenizer="builtin",
            release_replay=True,
            direct_store=direct_store,  # type: ignore[arg-type]
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        prefix_cache.stamp_theoretical_prefix_cache = prev

    assert parsed.segment_pool is not None
    assert "peak_window" in captured, "spy never fired -- parse path changed?"
    snapshot = captured["peak_window"]
    pool_bytes = _tier_bytes(_attribute_by_filename(snapshot), "segment_trie/pool.py")
    return pool_bytes, peak, len(parsed.segment_pool.by_id), snapshot


@pytest.mark.slow
def test_direct_route_content_tier_collapses(
    capture_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct write-through route: the resident ``Segment`` graph in ``_by_id`` collapses to EXACTLY zero."""
    # Task-6 gate evidence ("Task-2 re-run shows parse-window content tier approx 0").
    # Threading a live GraphSegmentUnifiedBackingStore as direct_store makes
    # StoreBackedSegmentPool.add write each segment THROUGH to the store's on-disk
    # content.blob at parse time, so the pool holds nothing and stops pinning content
    # strings (the total peak drops too).
    #
    # NUANCE (informational, not tightly asserted): the filename-attributed pool.py tier
    # does NOT drop to ~0 on the direct route, because it still carries the segment_id
    # hexdigest strings every node's metadata["trie"]["prompt_segment_ids"] retains on BOTH
    # routes -- per-node ADDRESSING cost, not the content pool. So the eager-minus-direct
    # delta isolates the removed Segment object graph, while the addressing tier stays at
    # the interned floor gated by _SID_INTERN_RATIO.
    from aiperf.dataset.graph.segment_trie import prefix_cache
    from aiperf.dataset.graph_segment_unified_store import (
        GraphSegmentUnifiedBackingStore,
    )

    _force_serial_build(monkeypatch)

    sessions, turns, g = _resolve_scale()
    shape = synthetic_corpus_shape(
        sessions=sessions, turns_per_session=turns, new_blocks_per_turn=g
    )
    n_nodes = shape.n_nodes

    # Capture the TRUE original ONCE so neither leg wraps the other's spy.
    real_stamp = prefix_cache.stamp_theoretical_prefix_cache

    eager_pool_bytes, eager_peak, eager_segs, eager_snap = _measure_pool_tier_at_peak(
        capture_path, real_stamp, direct_store=None
    )

    store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id="direct-content-tier"
    )
    try:
        (
            direct_pool_bytes,
            direct_peak,
            direct_segs,
            direct_snap,
        ) = _measure_pool_tier_at_peak(capture_path, real_stamp, direct_store=store)
    finally:
        store.abort()  # close the write handle + unlink the partial store files

    # Addressing tier: the segment_id hexdigest sid strings every node's
    # prompt_segment_ids retains, split out of the pool.py FILENAME tier via a
    # statistics("lineno") filter scoped to segment_id (see _addressing_tier_bytes).
    # Interning collapses ~246k re-listed duplicates to one canonical
    # per unique segment on BOTH routes.
    eager_addr_bytes = _addressing_tier_bytes(eager_snap)
    direct_addr_bytes = _addressing_tier_bytes(direct_snap)
    # The direct route's canonical _sids pointer list allocates in store_backed_pool.py
    # (a distinct FILENAME tier under nframes=1), NOT pool.py -- report it separately.
    direct_sids_bytes = _tier_bytes(
        _attribute_by_filename(direct_snap), "adapters/dynamo/store_backed_pool.py"
    )

    # eager_pool_bytes and direct_pool_bytes share the identical segment_id
    # hex-string allocation (byte-identical parse), so their delta isolates the
    # resident Segment object graph the write-through removes.
    resident_graph = eager_pool_bytes - direct_pool_bytes
    print(
        "\n[dynamo corpus-scale memory] parse-window content pool, "
        f"N={n_nodes:,} nodes U={shape.n_unique_blocks:,} unique blocks "
        f"seg={shape.n_unique_segments:,} unique segments:"
    )
    print(
        f"  resident Segment objects (by_id): EAGER={eager_segs:,} -> DIRECT={direct_segs}"
    )
    print(_fmt_tier("pool.py filename tier EAGER", eager_pool_bytes, n_nodes))
    print(_fmt_tier("pool.py filename tier DIRECT", direct_pool_bytes, n_nodes))
    print(_fmt_tier("addressing tier (segment_id) EAGER", eager_addr_bytes, n_nodes))
    print(_fmt_tier("addressing tier (segment_id) DIRECT", direct_addr_bytes, n_nodes))
    print(
        _fmt_tier("_sids list (store_backed_pool) DIRECT", direct_sids_bytes, n_nodes)
    )
    print(
        f"  resident Segment-graph removed by write-through: {resident_graph / 1e6:.2f} MB; "
        f"total peak {eager_peak / 1e6:.1f} MB (eager) -> {direct_peak / 1e6:.1f} MB (direct)"
    )

    # The eager route must actually build a content pool (else the test is vacuous).
    assert eager_segs > 0, "eager route interned 0 segments -- parse path changed?"
    # Cross-check the harness formula against the measured resident segment count so
    # the analytic sid tier + the ratio-gate denominator use a validated count.
    assert eager_segs == shape.n_unique_segments, (
        f"measured unique segments {eager_segs:,} != harness formula "
        f"{shape.n_unique_segments:,} (S*(2T-1)+N) -- generator shape drifted"
    )
    # Sid-interning gate: the retained sid-string addressing tier must stay
    # within _SID_INTERN_RATIO of the interned floor (one canonical str per unique
    # segment). A regression to a fresh hexdigest per re-listed slot overshoots ~10x.
    # Denominator base is sys.getsizeof (~81 B) vs the tracemalloc-observed ~73 B/str
    # in the numerator (see _SID_STR) -- the 2.0x headroom absorbs that unit mismatch.
    sid_intern_budget = _SID_INTERN_RATIO * shape.n_unique_segments * _SID_STR
    assert eager_addr_bytes <= sid_intern_budget, (
        f"addressing tier {eager_addr_bytes / 1e6:.2f} MB exceeds {_SID_INTERN_RATIO}x "
        f"the interned floor {sid_intern_budget / 1e6:.2f} MB "
        f"({shape.n_unique_segments:,} unique segments x {_SID_STR} B) -- "
        "pool interning regressed to a fresh sid per re-listed prompt_segment_ids slot"
    )
    # THE gate assertion: on the direct route the pool holds NOTHING.
    assert direct_segs == 0, (
        f"direct route pool retained {direct_segs} segments; the write-through shim "
        "must intern every segment into the store and hold none in _by_id"
    )
    # The resident Segment graph is a real, removed allocation, and freeing it (plus
    # its pinned content strings) lowers the measured peak.
    assert resident_graph > 0, (
        f"eager pool.py tier {eager_pool_bytes / 1e6:.2f} MB was not larger than the "
        f"direct route's {direct_pool_bytes / 1e6:.2f} MB -- the resident Segment "
        "graph should attribute to pool.py on the eager route only"
    )
    assert direct_peak < eager_peak, (
        f"direct-route peak {direct_peak / 1e6:.1f} MB was not below eager "
        f"{eager_peak / 1e6:.1f} MB -- the write-through should free content earlier"
    )


@pytest.mark.slow
def test_phase_isolate_resolution_trie(prebuilt_nodes: tuple[list, int]) -> None:
    """Compositional cross-check: size the resolution-automaton transient in isolation via the real ``_insert_flat``."""
    # The flat int-state automaton (transitions dict + parallel terminal/passer lists,
    # state 0 = root) is freed inside build_segment_trie before emission, so it is invisible to
    # the full-parse snapshot; holding the structures alive at snapshot time attributes it
    # to trie_content.py.
    from aiperf.dataset.graph.segment_trie import trie_content
    from aiperf.dataset.graph.segment_trie.trie_content import TrieNode

    nodes, _ = prebuilt_nodes
    n_nodes = len(nodes)

    def build_automaton() -> tuple[
        dict[tuple[int, int], int], list[TrieNode | None], list[TrieNode | None]
    ]:
        transitions: dict[tuple[int, int], int] = {}
        terminal: list[TrieNode | None] = [None]
        passer: list[TrieNode | None] = [None]
        for node in nodes:
            if node.request.hash_ids:
                trie_content._insert_flat(
                    transitions, terminal, passer, node.request.hash_ids, node
                )
        return transitions, terminal, passer

    built, peak, snap = _traced_build(build_automaton)

    trie_bytes = _tier_bytes(
        _attribute_by_filename(snap), "segment_trie/trie_content.py"
    )
    line = _fmt_tier("resolution automaton (isolate)", trie_bytes, n_nodes)
    print(
        f"\n[dynamo corpus-scale memory] resolution-automaton isolate "
        f"(peak {peak / 1e6:.1f} MB):\n{line}"
    )
    assert trie_bytes > 0, "resolution automaton attributed 0 bytes to trie_content.py"
    # Keep the structures reachable until after the snapshot is attributed.
    del built


@pytest.mark.slow
def test_phase_isolate_decode_cache(prebuilt_nodes: tuple[list, int]) -> None:
    """Compositional cross-check: drive the decoder over every unique hash id once, the full steady-state cache."""
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        dynamo_recon_callbacks,
    )

    nodes, block_size = prebuilt_nodes
    n_nodes = len(nodes)
    seed = _prewarm_synthesizer()
    unique_hashes = sorted({h for node in nodes for h in node.request.hash_ids})

    callbacks = dynamo_recon_callbacks(
        BUILTIN_TOKENIZER_NAME, "coding", seed, block_size=block_size, trace_scope="t"
    )

    decoded, peak, snap = _traced_build(
        lambda: callbacks.decode_block_tokens(unique_hashes)
    )

    assert len(decoded) == len(unique_hashes) * block_size
    decode_bytes = _tier_bytes(
        _attribute_by_filename(snap), "adapters/shared/content.py"
    )
    line = _fmt_tier("decode cache (isolate)", decode_bytes, n_nodes)
    print(
        f"\n[dynamo corpus-scale memory] decode-cache isolate over "
        f"{len(unique_hashes):,} unique hashes (peak {peak / 1e6:.1f} MB):\n{line}"
    )
    assert decode_bytes > 0, "decode cache attributed 0 bytes to shared/content.py"


@pytest.mark.slow
def test_peak_rss_logged(capture_path: str) -> None:
    """Log peak process RSS across a full parse; NEVER assert on it (allocator high-water retention makes it fragile)."""
    from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace

    _prewarm_synthesizer()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    parsed = from_dynamo_trace(
        capture_path,
        content_root_seed=_CONTENT_SEED,
        content_tokenizer="builtin",
        release_replay=True,
    )
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert parsed.segment_pool is not None
    # ru_maxrss is KiB on Linux, bytes on macOS; report both interpretations.
    print(
        f"\n[dynamo corpus-scale memory] peak RSS after full parse: "
        f"maxrss={after} (delta {after - before}); "
        f"~{after * 1024 / 1e9:.2f} GB if KiB (Linux)"
    )


# --- flat-automaton memory budget (NOT slow-marked; ~1-2 s) ------------------

# Chain-heavy position count for the resolution-automaton budget. A pure chain
# (one owner, strictly increasing ids) is the worst case for the deleted
# node-object trie -- every interior node held a one-entry ``children`` dict
# (the 224 B killer) -- and the best case for the flat automaton, which is
# exactly the shape the resolution trie takes on deep dynamo prefix chains.
_FLAT_BUDGET_POSITIONS = 200_000
# The flat automaton must use at most this fraction of the deleted trie's peak.
# Measured 0.51-0.59 across dict-resize phases on CPython
# 3.12; 0.80 leaves interpreter-drift headroom while still proving the win.
# (Distinct name from ``_BUDGET_RATIO`` above: that 1.5x gate belongs to the
# slow full-parse tier measurement and is read at call time -- rebinding the
# module global here would silently tighten it.)
_FLAT_BUDGET_RATIO = 0.80


def _measure_peak(build: Callable[[], object]) -> int:
    """Return tracemalloc peak bytes for ``build()`` with the result held alive through the peak read."""
    # The peak therefore reflects the fully-built structure plus its construction
    # transients (dict/list overallocation), not a half-collected one.
    built, peak, _ = _traced_build(build)
    del built
    return peak


class _ReplicaTrieNode:
    """~10-line in-test replica of the DELETED ``trie_content._PrefixTrieNode``, ``__slots__`` layout included."""

    # __slots__ reproduces the exact per-node memory layout (children dict +
    # terminal/passer) so the budget ratio compares the flat automaton against the real
    # thing it replaced, not an approximation.
    __slots__ = ("children", "terminal", "passer")

    def __init__(self) -> None:
        self.children: dict = {}
        self.terminal = None
        self.passer = None


def test_flat_automaton_memory_budget() -> None:
    """The flat int automaton peaks at <= 80% of the deleted node-object trie's peak, measured in-process."""
    # Calibrated-RATIO budget (the section-6 requirement that replaced the plan's rejected
    # 220 B absolute gate): an absolute-bytes assert is dict-resize-phase- and
    # CPython-version-fragile, so both structures are measured at the same position count
    # and only the ratio is asserted. Memory half of the frozen-code rewrite's proof --
    # the differential property test proves the output is byte-identical, this proves the
    # representation is actually leaner.
    from aiperf.dataset.graph.segment_trie import trie_content
    from aiperf.dataset.graph.segment_trie.trie_content import (
        TrieNode,
        TrieRequest,
    )

    positions = list(range(_FLAT_BUDGET_POSITIONS))
    owner = TrieNode(
        node_id="owner",
        request=TrieRequest(
            hash_ids=positions,
            input_length=1,
            output_length=1,
            t=0.0,
            api_time=0.0,
        ),
        order=0,
    )

    def build_replica() -> _ReplicaTrieNode:
        root = _ReplicaTrieNode()
        cur = root
        for h in positions:
            child = cur.children.get(h)
            if child is None:
                child = _ReplicaTrieNode()
                cur.children[h] = child
            cur = child
            if cur.passer is None:
                cur.passer = owner
        cur.terminal = owner
        return root

    def build_flat() -> tuple[
        dict[tuple[int, int], int], list[TrieNode | None], list[TrieNode | None]
    ]:
        transitions: dict[tuple[int, int], int] = {}
        terminal: list[TrieNode | None] = [None]
        passer: list[TrieNode | None] = [None]
        trie_content._insert_flat(transitions, terminal, passer, positions, owner)
        return transitions, terminal, passer

    replica_peak = _measure_peak(build_replica)
    flat_peak = _measure_peak(build_flat)

    ratio = flat_peak / replica_peak
    print(
        f"\n[flat-automaton budget @ {_FLAT_BUDGET_POSITIONS:,} positions] "
        f"replica(node-object trie)={replica_peak / 1e6:.1f} MB, "
        f"flat(automaton)={flat_peak / 1e6:.1f} MB, "
        f"ratio={ratio:.3f} (must be <= {_FLAT_BUDGET_RATIO})"
    )
    assert flat_peak <= _FLAT_BUDGET_RATIO * replica_peak, (
        f"flat automaton peak {flat_peak:,} B is not <= {_FLAT_BUDGET_RATIO} x the "
        f"deleted node-object trie peak {replica_peak:,} B (ratio {ratio:.3f}); "
        "the flat-automaton representation shave regressed -- investigate before landing."
    )
