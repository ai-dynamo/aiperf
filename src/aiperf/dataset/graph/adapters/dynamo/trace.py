# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter: convert Dynamo agent-trace files to the flat segment trie ParsedGraph.

One trace file (or segmented prefix / directory) carries ``request_end``
records for one or more ``agent_context.session_id`` trajectories. Sessions are
grouped into SESSION-TREES (a root session plus every descendant linked by
``agent_context.parent_session_id``; see :func:`group_chains_into_trees`) and
each tree is lowered INDEPENDENTLY via the reusable per-tree seam
(:func:`_build_graph_from_chains`) into its OWN single-graph ``ParsedGraph``
(``LlmNode``s ``{session_id}:{k}``, weka-identical graph shape). Those per-tree
``ParsedGraph``s are merged via
:func:`~aiperf.dataset.graph.merge.merge_parsed_graphs` (the SAME helper the
weka multi-item path uses) into ONE MULTI-GRAPH workload: one ``TraceRecord``
per tree, keyed by its root session id into ``graphs[root_id]`` with
``graph_ref=root_id``. Building per tree drops CROSS-PARENT interval-order edges
(between independent trees) by construction while preserving every WITHIN-tree
edge (parent<->subagent + intra-session); the shared build order (hash-prefix
content parents, frozen block tags, covered-count ISL gate, interval-order
finished-before edges) is owned by
:func:`~aiperf.dataset.graph.segment_trie.trie_content.build_segment_trie` and runs
over each tree's node set. Prompt content is content-addressed into a
:class:`~aiperf.dataset.graph.segment_trie.pool.SegmentPool` at parse
time -- there is no separate channel-replay build pass. The per-tree seam is
process-local so a later increment can fan trees to worker processes; the
segment pool / block size / content seed are all resolved ONCE and pinned into
every tree so the union is byte-identical to a single global build over
disjoint-content trees.

Hash source (see
:func:`~aiperf.dataset.graph.adapters.dynamo.trie_lowering.dynamo_trie_nodes`):
the recorded ``request.replay.input_sequence_hashes`` at the recorded
``trace_block_size`` whenever a record carries replay metadata (the
recorded-when-present rule the weka path follows). Current dynamo emits
replay on EVERY ``request_end`` (it skips the record entirely when the KV
block size is unavailable), so records without replay metadata only occur in
older captures or hand-authored traces; those fall back to
per-session VIRTUAL negative ids sized to ``input_tokens`` (block size
:data:`~aiperf.dataset.graph.adapters.dynamo.trie_lowering.DEFAULT_VIRTUAL_BLOCK_SIZE`)
and tag the trace ``virtual-hash-fallback``; consecutive turns of a session
still share a content prefix.

Schema source (dynamo repo): lib/llm/src/request_trace/types.rs
    and lib/llm/src/protocols/common/extensions.rs (AgentContext)
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Container, Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import msgspec
import orjson

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.environment import Environment
from aiperf.dataset.graph.adapters.dynamo.trace_reader import (
    DYNAMO_TRACE_SCHEMA_V1,
    AgentTraceRecord,
    DynamoTraceAdapterError,
    EmptyDynamoTraceError,
    _dir_segment_sort_key,
    iter_session_records,
    record_identity,
    resolve_parent,
    unwrap_sink_envelope,
)
from aiperf.dataset.graph.adapters.shared.selection import SelectionStats
from aiperf.dataset.graph.merge import merge_parsed_graphs
from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    ProvenanceSpec,
    TraceRecord,
)
from aiperf.dataset.graph.parse_context import (
    GraphParseContext,
    publish_ctx_tokenizer_env,
)

if TYPE_CHECKING:
    from aiperf.dataset.graph.segment_trie.trie_content import TrieAnalysis
    from aiperf.dataset.graph_segment_unified_store import (
        GraphSegmentUnifiedBackingStore,
    )

_DEFAULT_TAG = "from-dynamo-trace"
_SOURCE = "dynamo_trace"
_logger = AIPerfLogger(__name__)

# Kept as the private compatibility seam used by the trace-report command.
_record_identity = record_identity


def format_dropped_tail_rollup(
    nodes: int, tokens: int, trees: int, recorded_tokens: int
) -> str:
    """The one operator-facing line for partial block tails the trie dropped.

    Shared so the in-process build and the parallel parent emit IDENTICAL text
    for the same corpus -- the counts reach the parent by different routes
    (returned directly vs summed across worker frames), and a reader comparing
    a serial run to a parallel one must not see two wordings for one fact.

    Reports the drop three ways because the raw token count alone does not say
    whether it matters: a per-turn mean (is each prompt slightly or badly
    short?) and a share of the recorded total, ``dropped / (dropped + kept)``,
    where ``recorded_tokens`` is the summed recorded ``input_length``. A large
    absolute count can still be a rounding artifact on a huge corpus, and a
    small one can be most of a short-prompt capture.
    """
    per_turn = tokens / nodes if nodes else 0.0
    pct = (100.0 * tokens / recorded_tokens) if recorded_tokens else 0.0
    return (
        f"Dynamo load: dropped the partial block tail of {nodes:,} turn(s) "
        f"across {trees:,} tree(s) ({tokens:,} tokens total, {per_turn:,.1f}/turn, "
        f"{pct:.2f}% of the {recorded_tokens:,} recorded input tokens); prompts "
        f"are block-exact and shorter than the recorded input_length by that "
        f"amount"
    )


def assert_ctx_knobs_supported(ctx: GraphParseContext | None) -> None:
    """Refuse ctx knobs the Dynamo trace schema cannot express.

    Called from BOTH dynamo build entries, because they are separate code
    paths: ``DynamoTraceAdapter.parse`` (the registry dispatch) and the
    store builder's streaming route, which calls
    ``stream_dynamo_trace_segment_payloads`` directly and forwards a
    hand-picked subset of ctx fields. Living only in ``parse`` meant the
    streaming route -- the default for dynamo -- silently ignored both flags
    instead of failing loud, which is the whole point of a gate.

    Raising is the intended contract: neither knob has any effect on this
    adapter, so accepting one produces a run whose pacing differs from what
    the operator asked for, with nothing in the output saying so.
    """
    if ctx is None:
        return
    if ctx.use_think_time_only:
        raise DynamoTraceAdapterError(
            "dynamo_trace cannot honor --use-think-time-only: the Dynamo "
            "trace schema does not record per-request think_time"
        )
    if ctx.delay_cap_seconds is not None:
        raise DynamoTraceAdapterError(
            "dynamo_trace cannot honor --inter-turn-delay-cap-seconds: "
            "use --trace-idle-gap-cap-seconds for the graph's recorded "
            "timeline"
        )
    if ctx.replay_only_knobs:
        named = ", ".join(ctx.replay_only_knobs)
        raise DynamoTraceAdapterError(
            f"dynamo_trace cannot honor {named}: "
            "these flags are read only by the linear trace-replay loaders "
            "(baseten_trace / AGENTIC_REPLAY) and have no effect on a graph "
            "replay, which paces from the recorded graph timeline. Drop them, "
            "or pin a non-graph loader with --custom-dataset-type to run this "
            "input through the linear pipeline."
        )


def _collect_chains(
    path: str | Path,
    session_id_filter: str | None,
    *,
    max_depth: int,
    duplicate_out: list[int] | None = None,
    skipped_out: list[int] | None = None,
) -> dict[str, _Chain]:
    """Read trace records, sort per session, and reduce to non-empty chains."""
    started = perf_counter()
    _logger.info(f"Dynamo load: reading trace records from {path}")
    by_session, parent_link, skipped_no_context, duplicates = _collect_records(
        path, session_id_filter
    )
    if duplicate_out is not None:
        duplicate_out.append(duplicates)
    if skipped_out is not None:
        skipped_out.append(skipped_no_context)
    if not by_session:
        if skipped_no_context:
            raise EmptyDynamoTraceError(
                f"{path}: {skipped_no_context:,} non-request records had no "
                f"agent_context and could not be assigned a session"
            )
        raise EmptyDynamoTraceError(f"{path}: no trace records found")

    for sid in by_session:
        by_session[sid].sort(key=lambda r: r.event_time_unix_ms)

    chains: dict[str, _Chain] = {}
    for sid, recs in by_session.items():
        chain = _records_to_chain(
            recs, session_id=sid, parent_session_id=parent_link.get(sid)
        )
        if chain.turns:
            chains[sid] = chain

    if not chains:
        raise EmptyDynamoTraceError(
            f"{path}: no request_end events found across {len(by_session)} sessions"
        )

    _guard_chain_forest(chains, parent_link, max_depth=max_depth)
    _logger.info(
        f"Dynamo load: record scan complete in {perf_counter() - started:.2f}s "
        f"({len(chains):,} sessions, "
        f"{sum(len(chain.turns) for chain in chains.values()):,} request_end events, "
        f"{skipped_no_context:,} skipped, {duplicates:,} duplicates)"
    )
    return chains


def from_dynamo_trace(
    path: str | Path,
    *,
    tag: str = _DEFAULT_TAG,
    session_id_filter: str | None = None,
    idle_gap_cap_seconds: float | None = None,
    content_root_seed: int | None = None,
    content_tokenizer: str | None = None,
    prompt_corpus: str = "coding",
    release_replay: bool = False,
    direct_store: GraphSegmentUnifiedBackingStore | None = None,
    num_dataset_entries: int | None = None,
    max_context_length: int | None = None,
    max_isl: int | None = None,
    max_osl: int | None = None,
    streaming: bool | None = None,
    ignore_trace_delays: bool = False,
    selection_out: list[SelectionStats] | None = None,
    analysis_out: list[TrieAnalysis] | None = None,
) -> ParsedGraph:
    """Parse a Dynamo agent-trace file/dir/prefix into the flat segment trie.

    Every session's turns lower to one ``LlmNode`` per ``request_end``
    (``{session_id}:{k}``, the recorded session id verbatim + 0-based turn)
    in a SINGLE flat graph. Sessions are first grouped
    into session-trees (a root plus its ``parent_session_id`` descendants) and
    each tree is lowered independently, so parent and subagent sessions of ONE
    tree coexist with edges whose concurrency is EMERGENT from the recorded
    intervals (a child overlapping its parent gets no edge between them;
    disjoint intervals get a finished-before edge), while INDEPENDENT trees
    never gain a cross-parent edge -- their interval order is derived only over
    their own nodes. Prompt content is
    reconstructed deterministically from ``(content_tokenizer,
    prompt_corpus, content_root_seed)`` and content-addressed into the
    returned ``ParsedGraph.segment_pool``; each node's ISL is the block-
    aligned COVERED count of its hashes (never history + full-reconstruction
    double counting). The eager route interns via an
    :class:`~aiperf.dataset.graph.adapters.dynamo.store_backed_pool.InterningSegmentPool`
    so equal ``prompt_segment_ids`` across turns/sessions share one canonical
    str object (values byte-identical to a plain ``SegmentPool``).

    Args:
        path: Path to a ``.jsonl`` file, ``.jsonl.gz`` file, segmented prefix,
            or a directory containing one of the above.
        tag: Base tag added to EVERY emitted trace (each session-tree is its own
            single-root trace, so there is no ``"multi-root"`` tag).
        session_id_filter: When set, restrict to records whose
            ``agent_context.session_id`` matches.
        idle_gap_cap_seconds: Per-tree idle-gap cap (seconds) on the recorded
            timeline that stamps node ``arrival_offset_us`` and the edge delays
            (``delay_after_predecessor_us`` on the binding predecessor,
            ``min_start_delay_us`` on START edges) via ``ActiveIdleWarp``, so
            recorded end-to-start gaps replay with over-long idle gaps
            compressed to the cap. The run path resolves this from the dataset's
        ``trace_idle_gap_cap_seconds`` (``--trace-idle-gap-cap-seconds``);
            left unset here (``None``) no per-trace compression is applied and
            raw recorded gaps replay unchanged.
        content_root_seed: Seed pinning the deterministic content synthesis;
            ``None`` resolves via the ambient bootstrap root seed, else fresh
            per-run OS entropy.
        content_tokenizer: Tokenizer for content synthesis (``None`` selects
            the builtin deterministic tokenizer).
        prompt_corpus: Corpus the content synthesizer samples from.
        release_replay: When ``True``, free each record's ``request.replay``
            hash lists inside :func:`dynamo_trie_nodes` once they are copied
            into the segment trie (a build-time RAM adjunct). SAFE ONLY because this
            function lowers freshly-read ``chains`` exactly once per call;
            defaults ``False`` so a caller that re-lowers the same in-memory
            records never silently degrades to the virtual-hash fallback. The
            production store build (:meth:`DynamoTraceAdapter.parse`) opts in.
        direct_store: When set, the build plane's live
            :class:`~aiperf.dataset.graph_segment_unified_store.GraphSegmentUnifiedBackingStore`.
            The returned ``ParsedGraph.segment_pool`` becomes a
            :class:`~aiperf.dataset.graph.adapters.dynamo.store_backed_pool.StoreBackedSegmentPool`
            whose ``add()`` write-throughs intern each segment STRAIGHT INTO the
            store at parse time (no second RAM pool copy) -- the Stage B direct
            write-through route. ``None`` (every direct caller / tooling) keeps
            the eager ``SegmentPool``. Threaded only via the build-plane seam
            (``GraphStoreBuilder`` -> ``parse_graph_workload`` kwargs passthrough),
            never through the format-agnostic ``GraphParseContext``.
        num_dataset_entries: ``--num-dataset-entries`` cap on the number of
            session-TREES built (the graph-plane fix for ai-dynamo/aiperf#1106).
            ``None`` (default) builds every eligible tree. A tree is a root
            session plus its subagent descendants -- the selection unit, so a
            capped load never splits a tree.
        max_context_length: ``--max-context-length`` per-tree ceiling on peak
            context (max ``input_length + output_tokens`` over the tree's
            records, via :func:`dynamo_tree_peak_context`). Over-limit trees are
            rejected BEFORE the build; the first ``num_dataset_entries`` eligible
            trees (arrival-ordered) are then kept (filter THEN cap). ``None`` applies
            no context filter. Selection applies on BOTH the serial and
            fused-parallel build paths.
        selection_out: Optional sink; when provided AND a knob is set, the single
            :class:`SelectionStats` for the filter-then-cap scan is appended for
            the caller's report. Untouched when both knobs are ``None``.
        analysis_out: Optional sink for one :class:`TrieAnalysis` per built
            session-tree. Supplying it opts out of the fused process-parallel
            path so the analysis objects remain in the caller.

    Returns:
        A MULTI-GRAPH ``ParsedGraph`` (via :func:`merge_parsed_graphs`): one
        ``TraceRecord`` per session-tree, each keyed by its root session id into
        ``graphs[root_id]`` and selected by ``graph_ref=root_id``; traces are
        id-sorted and the content-addressed ``segment_pool`` is the union of the
        per-tree pools. ``graph`` is the FIRST tree (lex-min root, the default
        single-graph slot), NOT the whole-capture union. Each tree's graph holds its sessions'
        ``LlmNode``s + WITHIN-tree interval-order edges (cross-parent edges
        absent by construction).
    """
    load_started = perf_counter()
    _logger.info(f"Dynamo load: starting path={path}")
    max_depth = Environment.DYNAMO.MAX_SUBAGENT_DEPTH

    # Same seed ladder as the weka routes (explicit seed -> ambient bootstrap
    # root seed -> per-run OS entropy): dynamo shares the weka content
    # synthesizer, so an unresolved None here would mean ambient-RNG content
    # in-process but offline-default content in tooling -- the divergence the
    # weka unification removed. Resolved ONCE here and pinned into every tree
    # build so all trees share the same seed -> byte-identical pools.
    from aiperf.dataset.graph.adapters.shared.content import (
        resolve_effective_root_seed,
    )

    content_root_seed = resolve_effective_root_seed(content_root_seed)

    # Ignoring recorded delays with no cap authored means compress everything;
    # the fused-parallel build applies the same rule so both warp identically.
    if ignore_trace_delays and idle_gap_cap_seconds is None:
        idle_gap_cap_seconds = 0.0
    idle_gap_cap = idle_gap_cap_seconds

    # Fused read+build parallel path: a cheap grouping scan (session ids +
    # parent links only, NO ``input_sequence_hashes`` parse) decides tree
    # membership in the parent, then raw record lines are shuffled to per-batch
    # temp files and READ+BUILT inside worker processes, so the giant recorded
    # hash arrays never cross a process boundary -- only the compact built graph
    # returns. Gated to the pure tooling / direct-caller route (no live
    # write-through store, which cannot cross processes) and a whole-corpus build
    # (a single-session filter is one tree -> never parallel). Returns None below
    # the tree-count threshold / for a single tree, and the serial
    # read-then-build path below runs instead.
    if (
        direct_store is None
        and session_id_filter is None
        and max_isl is None
        and max_osl is None
        and analysis_out is None
    ):
        from aiperf.dataset.graph.adapters.dynamo import trace_parallel

        fused = trace_parallel.maybe_build_fused_parallel(
            path,
            content_root_seed=content_root_seed,
            idle_gap_cap_seconds=idle_gap_cap,
            content_tokenizer=content_tokenizer,
            prompt_corpus=prompt_corpus,
            release_replay=release_replay,
            max_depth=max_depth,
            max_osl=max_osl,
            streaming=streaming,
            num_dataset_entries=num_dataset_entries,
            max_context_length=max_context_length,
            selection_out=selection_out,
        )
        if fused is not None:
            _logger.info(
                f"Dynamo load: fused parallel build complete in "
                f"{perf_counter() - load_started:.2f}s ({len(fused):,} trees)"
            )
            return _finalize_parsed_graph(fused, tag=tag)
        _logger.info(
            f"Dynamo load: using serial read/build path after parallel dispatch "
            f"check ({perf_counter() - load_started:.2f}s)"
        )

    chains = _collect_chains(path, session_id_filter, max_depth=max_depth)
    _logger.info(f"Dynamo load: grouping {len(chains):,} sessions into session trees")
    if (
        num_dataset_entries is not None
        or max_context_length is not None
        or max_isl is not None
    ):
        chains = _select_chains_filter_then_cap(
            chains,
            path=path,
            num_dataset_entries=num_dataset_entries,
            max_context_length=max_context_length,
            max_isl=max_isl,
            selection_out=selection_out,
        )
    per_tree = _build_trees_flat(
        chains,
        content_root_seed=content_root_seed,
        idle_gap_cap_seconds=idle_gap_cap,
        content_tokenizer=content_tokenizer,
        prompt_corpus=prompt_corpus,
        release_replay=release_replay,
        direct_store=direct_store,
        max_osl=max_osl,
        streaming=streaming,
        analysis_out=analysis_out,
    )
    _logger.info(
        f"Dynamo load: graph build complete in {perf_counter() - load_started:.2f}s "
        f"({len(per_tree):,} trees); merging output"
    )
    result = _finalize_parsed_graph(per_tree, tag=tag)
    _logger.info(
        f"Dynamo load: finished in {perf_counter() - load_started:.2f}s "
        f"({len(result.graph.nodes):,} nodes, {len(result.graph.edges):,} edges)"
    )
    return result


def analyze_dynamo_chains_trie(chains: dict[str, _Chain]) -> list[TrieAnalysis]:
    """Collect trie analysis from already-loaded Dynamo chains."""
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        DEFAULT_VIRTUAL_BLOCK_SIZE,
        _resolve_block_size,
        dynamo_trie_nodes,
    )
    from aiperf.dataset.graph.segment_trie.prefix_cache import (
        compute_shared_prefix_cache_counts,
    )
    from aiperf.dataset.graph.segment_trie.trie_content import (
        TrieAnalysis,
        resolve_content_parents,
        validate_block_counts,
    )

    by_block_size: dict[int, dict[str, _Chain]] = {}
    for sid, chain in chains.items():
        block_size = _resolve_block_size({sid: chain}) or DEFAULT_VIRTUAL_BLOCK_SIZE
        by_block_size.setdefault(block_size, {})[sid] = chain

    analyses: list[TrieAnalysis] = []
    for block_size, block_chains in by_block_size.items():
        nodes, block_size, _tags = dynamo_trie_nodes(
            block_chains, block_size=block_size
        )
        analysis = TrieAnalysis(block_size=block_size)
        validate_block_counts(nodes, block_size)
        resolve_content_parents(nodes, analysis)
        for node in nodes:
            analysis.record_content_parent(node, block_size)
        analysis.finish_hash_counts()
        for hit_blocks, total_blocks in compute_shared_prefix_cache_counts(
            nodes, block_size
        ).values():
            analysis.record_cache_count(hit_blocks, total_blocks)
        analyses.append(analysis)
    return analyses


def _select_chains_filter_then_cap(
    chains: dict[str, _Chain],
    *,
    path: str | Path,
    num_dataset_entries: int | None,
    max_context_length: int | None,
    max_isl: int | None,
    selection_out: list[SelectionStats] | None,
) -> dict[str, _Chain]:
    """Filter-then-cap the collected chains at the session-TREE granularity.

    The SERIAL-path selection seam (the fused path selects in the parent scan):
    groups the already-collected chains into session-trees (arrival-ordered), screens
    each by :func:`dynamo_tree_peak_context` against ``max_context_length``, and
    keeps the first ``num_dataset_entries`` eligible trees. Reuses that helper
    directly over the flattened per-tree ``request_end`` records, so the selected
    set matches the fused path's hash-free scan selection. Returns the union of
    the kept trees' chains; raises :class:`EmptyDynamoTraceError` when every tree
    is filtered out (a ceiling below the whole capture is a user error, not an
    empty graph).
    """
    from aiperf.dataset.graph.adapters.shared.peak_context import (
        dynamo_tree_peak_context,
        dynamo_tree_peak_input,
    )
    from aiperf.dataset.graph.adapters.shared.selection import (
        filter_then_cap,
        log_selection_summary,
    )

    parent_link = {
        sid: c.parent_session_id
        for sid, c in chains.items()
        if c.parent_session_id is not None
    }
    trees = order_trees_by_recorded_start(group_chains_into_trees(chains, parent_link))

    # ``--max-isl`` rejects BEFORE the filter-then-cap generator yields, so those
    # trees never reach ``SelectionStats`` (its counters advance per yielded
    # item). Counted here so the summary and the empty-corpus error can name the
    # knob that actually did the rejecting instead of blaming
    # ``max_context_length`` for a corpus ``max_isl`` emptied.
    isl_rejected = 0

    def _candidates() -> Iterable[tuple[dict[str, _Chain], int]]:
        nonlocal isl_rejected
        for tree in trees:
            records = [turn.record for chain in tree.values() for turn in chain.turns]
            if max_isl is not None and dynamo_tree_peak_input(records) > max_isl:
                isl_rejected += 1
                continue
            yield tree, dynamo_tree_peak_context(records)

    kept_trees, stats = filter_then_cap(
        _candidates(),
        num_dataset_entries=num_dataset_entries,
        max_context_length=max_context_length,
    )
    # Parent-side finalize point for the SERIAL path (and the fused-decline
    # fallback): log the summary once here. The fused build path logs its own
    # when it actually fans out; the two paths are mutually exclusive per build.
    log_selection_summary(
        stats,
        source=str(path),
        num_dataset_entries=num_dataset_entries,
        max_context_length=max_context_length,
    )
    if selection_out is not None:
        selection_out.append(stats)
    if isl_rejected:
        _logger.info(
            f"Dynamo load: {isl_rejected:,} of {len(trees):,} session-tree(s) "
            f"rejected by max_isl={max_isl} before context screening"
        )
    if not kept_trees:
        rejected_by = f"max_context_length={max_context_length}"
        if isl_rejected:
            # Naming only max_context_length here sent operators tuning a knob
            # that had rejected nothing; max_isl may have emptied the corpus
            # on its own, in which case scanned is 0.
            rejected_by = (
                f"max_isl={max_isl} ({isl_rejected:,} tree(s)) / {rejected_by} "
                f"({stats.scanned} tree(s) screened)"
            )
        raise EmptyDynamoTraceError(
            f"{path}: all {len(trees):,} session-trees were rejected by "
            f"{rejected_by}; nothing to build"
        )
    selected: dict[str, _Chain] = {}
    for tree in kept_trees:
        selected.update(tree)
    return selected


def _finalize_parsed_graph(per_tree: list[ParsedGraph], *, tag: str) -> ParsedGraph:
    """Fold the base ``tag`` into each per-tree trace, then merge into ONE workload.

    Both build paths hand this the list of per-tree SINGLE-graph
    ``ParsedGraph``s (each = one root ``TraceRecord`` with ``graph_ref=None``,
    that tree's ``GraphRecord``, that tree's pool). This stamps the base ``tag``
    onto each trace (each tree is its own single-root trace, so there is no
    ``"multi-root"`` tag) and returns :func:`merge_parsed_graphs`, which -- the
    SAME helper the weka multi-item path uses -- keys each trace's graph under
    ``graphs[trace.id]``, sets ``graph_ref=trace.id``, guards duplicate ids,
    id-sorts the traces, unions the content-addressed pools, and keeps ``graph``
    as the FIRST tree (the default single-graph slot).
    """
    tagged = (
        msgspec.structs.replace(
            pg,
            traces=[
                msgspec.structs.replace(trace, tags=sorted({tag, *trace.tags}))
                for trace in pg.traces
            ],
        )
        for pg in per_tree
    )
    return merge_parsed_graphs(tagged)


# --- session-tree grouping + per-tree build ---------------------------------


def tree_recorded_start_ms(tree: dict[str, _Chain]) -> int:
    """Earliest recorded event across every turn of every session in ``tree``.

    The tree's arrival instant -- when this conversation first appears in the
    capture. Returns ``sys.maxsize`` for a tree with no turns at all so it sorts
    last rather than pretending to be the earliest thing recorded.
    """
    return min(
        (
            turn.record.event_time_unix_ms
            for chain in tree.values()
            for turn in chain.turns
        ),
        default=sys.maxsize,
    )


def order_trees_by_recorded_start(
    trees: list[dict[str, _Chain]],
) -> list[dict[str, _Chain]]:
    """Order session-trees by arrival time, keeping root-id order for ties.

    The selection cap (``--num-dataset-entries``, which ``--num-conversations``
    also lands as) keeps the FIRST N eligible trees in this order, so this is
    what makes a bounded load a contiguous slice of the recorded TIMELINE rather
    than of the alphabet. ``group_chains_into_trees`` returns root-id-sorted
    trees and Python's sort is stable, so equal arrival instants keep exactly the
    previous deterministic ordering.

    Applied at the SELECTION seam only -- ``group_chains_into_trees`` keeps its
    root-id contract for every other caller, so the downstream union stays
    byte-stable.
    """
    return sorted(trees, key=tree_recorded_start_ms)


def group_chains_into_trees(
    chains: dict[str, _Chain],
    parent_link: dict[str, str],
) -> list[dict[str, _Chain]]:
    """Partition chains into session-trees by ``parent_session_id`` linkage.

    A tree is a root session (one whose parent is absent, self, or external to
    ``chains``) plus every descendant reachable through ``parent_link``. Each
    session is unioned to its in-set root by walking ``parent_link`` to a
    fixpoint; a session with no parent and no children is its own singleton
    tree. Independent trees never share a dict, so lowering each one on its own
    node set drops cross-parent interval-order edges by construction while
    keeping every within-tree edge.

    ``parent_link`` maps ``session_id -> parent_session_id`` (the caller derives
    it from each chain's ``parent_session_id``); a link whose parent is not in
    ``chains`` marks the child as a forest root. The walk carries a per-session
    ``seen`` guard so a ``parent_link`` cycle (already rejected upstream by
    :func:`_guard_chain_forest`) can never spin here -- it halts at the repeat.

    Deterministic: trees are returned sorted by root session id and each tree's
    dict is insertion-ordered by session id, so the downstream union is
    byte-stable across runs.
    """
    root_of = root_of_sessions(chains, parent_link)
    trees: dict[str, dict[str, _Chain]] = defaultdict(dict)
    for sid in sorted(chains):
        trees[root_of[sid]][sid] = chains[sid]
    return [trees[root] for root in sorted(trees)]


def root_of_sessions(
    session_ids: Container[str] | Iterable[str],
    parent_link: dict[str, str],
) -> dict[str, str]:
    """Map each session id to its in-set tree root by walking ``parent_link``.

    ``session_ids`` need only support membership (``in``) and iteration -- a
    ``chains`` dict (keyed by session id) or a plain ``set`` both work, so the
    parent's cheap grouping scan (which only knows session ids, never the built
    chains) and :func:`group_chains_into_trees` share ONE walk and therefore
    ONE grouping decision. The walk stops at the first parent absent from the
    set (a forest root) and carries a per-session ``seen`` guard so a
    ``parent_link`` cycle halts at the repeat rather than spinning.
    """
    root_of: dict[str, str] = {}
    for sid in session_ids:
        seen: set[str] = {sid}
        cur = sid
        while True:
            parent = parent_link.get(cur)
            if parent is None or parent not in session_ids or parent in seen:
                break
            seen.add(parent)
            cur = parent
        root_of[sid] = cur
    return root_of


def _build_trees_flat(
    chains: dict[str, _Chain],
    *,
    content_root_seed: int,
    idle_gap_cap_seconds: float | None,
    content_tokenizer: str | None,
    prompt_corpus: str,
    release_replay: bool,
    direct_store: GraphSegmentUnifiedBackingStore | None,
    max_osl: int | None,
    streaming: bool | None = None,
    analysis_out: list[TrieAnalysis] | None = None,
) -> list[ParsedGraph]:
    """Group ``chains`` into session-trees and build one ``ParsedGraph`` per tree.

    Returns the per-tree single-graph ``ParsedGraph`` list (the caller merges
    them via :func:`merge_parsed_graphs`). The block size is resolved ONCE
    across the WHOLE capture (fail-loud on a mix, exactly as the single global
    build did) and pinned into every tree; the content seed is likewise pinned
    by the caller, so every tree build is byte-identical to a single global
    build over disjoint-content trees.

    This is the SERIAL build over already-collected ``chains`` -- the path taken
    when the fused read+build parallel dispatch in :func:`from_dynamo_trace`
    declines (below the tree-count threshold, a single-session filter, or a live
    ``direct_store`` write-through sink that cannot cross process boundaries).
    The parallel decision lives in :func:`from_dynamo_trace` (before the
    expensive full read), not here, because the fused path must AVOID collecting
    the hash-bearing chains in the parent in the first place.
    """
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        DEFAULT_VIRTUAL_BLOCK_SIZE,
        _resolve_block_size,
    )

    # One block size across the whole capture (fail-loud on a mix); pinned into
    # every tree so a replay-free tree never diverges to the virtual default.
    block_size = _resolve_block_size(chains) or DEFAULT_VIRTUAL_BLOCK_SIZE
    parent_link = {
        sid: c.parent_session_id
        for sid, c in chains.items()
        if c.parent_session_id is not None
    }
    trees = group_chains_into_trees(chains, parent_link)
    _logger.info(
        f"Dynamo load: session-tree grouping complete "
        f"({len(trees):,} trees, block_size={block_size})"
    )

    return _build_trees_sequential(
        trees,
        block_size=block_size,
        content_root_seed=content_root_seed,
        idle_gap_cap_seconds=idle_gap_cap_seconds,
        content_tokenizer=content_tokenizer,
        prompt_corpus=prompt_corpus,
        release_replay=release_replay,
        direct_store=direct_store,
        max_osl=max_osl,
        streaming=streaming,
        analysis_out=analysis_out,
    )


def _build_trees_sequential(
    trees: list[dict[str, _Chain]],
    *,
    block_size: int,
    content_root_seed: int,
    idle_gap_cap_seconds: float | None,
    content_tokenizer: str | None,
    prompt_corpus: str,
    release_replay: bool,
    direct_store: GraphSegmentUnifiedBackingStore | None,
    max_osl: int | None,
    streaming: bool | None = None,
    analysis_out: list[TrieAnalysis] | None = None,
    tails_out: list[tuple[int, int, int, int]] | None = None,
) -> list[ParsedGraph]:
    """Build each already-grouped session-tree into its OWN single-graph ParsedGraph.

    The in-process build loop, called BOTH by the serial dispatch of
    :func:`_build_trees_flat` and by every pool worker (which receives its
    contiguous BATCH of trees as ``trees``). Each returned ``ParsedGraph`` holds
    one tree's ``GraphRecord``, one ``TraceRecord`` (id = the tree's root session
    id, ``graph_ref=None``) carrying that tree's extra tags, and that tree's own
    pool. The caller folds the base tag and merges the list via
    :func:`merge_parsed_graphs` -- the SAME helper the weka multi-item path uses,
    so ``.graphs[root_id]`` / ``graph_ref`` / id-sorted traces / content-pool
    union are all byte-identical to weka. ``block_size`` is pinned by the caller
    (resolved ONCE across the whole capture in the parent), so a replay-free tree
    in a batch never diverges to the virtual default while a sibling batch
    carries a recorded size.

    Pool strategy: with ``direct_store`` set the trees SHARE one
    :class:`~aiperf.dataset.graph.adapters.dynamo.store_backed_pool.StoreBackedSegmentPool`
    (the store is a single write-through sink whose ``by_id`` stays empty), so the
    merged pool is the empty union the interned drain no-ops over. Otherwise each
    tree builds into its OWN
    :class:`~aiperf.dataset.graph.adapters.dynamo.store_backed_pool.InterningSegmentPool`
    and its per-tree ``ParsedGraph`` carries a plain ``SegmentPool`` over that
    tree's ``by_id``; :func:`merge_parsed_graphs` unions them by content-addressed
    id (identical ids carry identical content under the pinned seed). This
    per-tree-ParsedGraph shape is the seam the parallel path fans to worker
    processes -- each worker returns its batch's LIST of per-tree ParsedGraph
    blobs.
    """
    from aiperf.dataset.graph.adapters.dynamo import store_backed_pool
    from aiperf.dataset.graph.segment_trie.pool import SegmentPool

    shared_pool: SegmentPool | None = (
        store_backed_pool.StoreBackedSegmentPool(direct_store)
        if direct_store is not None
        else None
    )

    per_tree: list[ParsedGraph] = []
    dropped_tails: list[tuple[int, int]] = []
    for tree in trees:
        # Shared write-through store, or a fresh interning pool per tree that
        # this tree's ParsedGraph carries (the process-parallel seam: a worker
        # returns its batch's per-tree ParsedGraph blobs, merged in the parent).
        pool = (
            shared_pool
            if shared_pool is not None
            else store_backed_pool.InterningSegmentPool()
        )
        analysis = None
        if analysis_out is not None:
            from aiperf.dataset.graph.segment_trie.trie_content import TrieAnalysis

            analysis = TrieAnalysis()
        graph, extra_tags = _build_graph_from_chains(
            tree,
            pool=pool,
            content_root_seed=content_root_seed,
            block_size=block_size,
            idle_gap_cap_seconds=idle_gap_cap_seconds,
            content_tokenizer=content_tokenizer,
            prompt_corpus=prompt_corpus,
            release_replay=release_replay,
            max_osl=max_osl,
            streaming=streaming,
            analysis=analysis,
            dropped_tails_out=dropped_tails,
        )
        if analysis is not None:
            analysis_out.append(analysis)
        tree_pool: SegmentPool = (
            shared_pool if shared_pool is not None else SegmentPool(_by_id=pool.by_id)
        )
        per_tree.append(
            ParsedGraph(
                graph=graph,
                traces=[
                    TraceRecord(id=_tree_root_id(tree), tags=sorted(set(extra_tags)))
                ],
                segment_pool=tree_pool,
            )
        )
    # ONE rollup for everything built here, never one per tree: the trie emits
    # whole blocks only, so a corpus of non-aligned prompts is short by its
    # tails and that must be visible in a single operator-facing line.
    #
    # Pool workers pass ``tails_out`` and log NOTHING: a worker's root logger
    # has no handler configured (the forkserver preload carries no logging
    # setup), so a worker-side _logger.info is discarded and the operator
    # silently gets shorter prompts than the recording. The worker ships its
    # counts back instead and the PARENT emits the one line.
    tail_nodes = sum(n for n, _, _ in dropped_tails)
    tail_tokens = sum(t for _, t, _ in dropped_tails)
    recorded = sum(r for _, _, r in dropped_tails)
    if tails_out is not None:
        tails_out.append((tail_nodes, tail_tokens, len(trees), recorded))
    elif tail_nodes:
        _logger.info(
            format_dropped_tail_rollup(tail_nodes, tail_tokens, len(trees), recorded)
        )
    return per_tree


def _tree_root_id(tree: dict[str, _Chain]) -> str:
    """The tree's root session id = its lex-min session whose parent is out-of-tree.

    A well-formed session-tree has exactly ONE such session (every non-root
    session's parent is in the tree by construction of :func:`root_of_sessions`);
    the ``sorted(...)[0]`` is a defensive tiebreak and the ``min(tree)`` fallback
    covers the degenerate all-have-parents case (a cycle, already rejected
    upstream by :func:`_guard_chain_forest`). This is the SAME id the
    whole-capture root sort used as the single trace id before the multi-graph
    split, so a single-tree capture keeps its historical trace id.
    """
    roots = sorted(sid for sid, c in tree.items() if c.parent_session_id not in tree)
    return roots[0] if roots else min(tree)


def _build_graph_from_chains(
    chains: dict[str, _Chain],
    *,
    pool: Any,
    content_root_seed: int,
    block_size: int,
    idle_gap_cap_seconds: float | None,
    content_tokenizer: str | None,
    prompt_corpus: str,
    release_replay: bool,
    max_osl: int | None,
    streaming: bool | None = None,
    analysis: TrieAnalysis | None = None,
    dropped_tails_out: list[tuple[int, int, int]] | None = None,
) -> tuple[GraphRecord, list[str]]:
    """Lower ONE set of chains (a single session-tree) into a flat ``GraphRecord``.

    The reusable per-tree seam -- everything between ``dynamo_trie_nodes`` and
    ``assemble_trie_graph`` -- so a later increment can fan each tree's chains
    to a worker process that calls exactly this. Passing the WHOLE chain set
    reproduces the pre-tree single global build (the differential oracle the
    unit tests pin the tree-scoped union against). ``pool`` is caller-owned so
    the orchestrator can share one write-through store across trees or union
    per-tree pools; ``content_root_seed`` and ``block_size`` are resolved ONCE
    by the caller and pinned here so every tree build is byte-identical.

    Heavy imports stay lazy: the content synthesizer / trie core are only needed
    on a real parse, and ``DynamoTraceAdapter.can_load`` sniffing must not pull
    the corpus machinery onto the detection path.
    """
    from aiperf.common.tokenizer import BUILTIN_TOKENIZER_NAME
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        build_dynamo_llm_node,
        dynamo_recon_callbacks,
        dynamo_trie_nodes,
    )
    from aiperf.dataset.graph.segment_trie.trie_content import (
        TrieNode,
        assemble_trie_graph,
        build_segment_trie,
    )

    nodes, bs, extra_tags = dynamo_trie_nodes(
        chains,
        release_replay=release_replay,
        block_size=block_size,
        max_osl=max_osl,
    )
    callbacks = dynamo_recon_callbacks(
        content_tokenizer or BUILTIN_TOKENIZER_NAME,
        prompt_corpus,
        content_root_seed,
        block_size=bs,
        trace_scope=_tree_root_id(chains),
    )
    result = build_segment_trie(
        nodes,
        block_size=bs,
        callbacks=callbacks,
        pool=pool,
        idle_gap_cap_seconds=idle_gap_cap_seconds,
        small_prompt_fallback=True,
        analysis=analysis,
    )
    if dropped_tails_out is not None:
        # Per-tree counts for the caller's ONE corpus-level rollup; this seam
        # runs once per session-tree, so it must never log on its own.
        dropped_tails_out.append(
            (
                result.dropped_tail_nodes,
                result.dropped_tail_tokens,
                result.recorded_input_tokens,
            )
        )

    # build_dynamo_llm_node reads node.start (= warped_start), which is only
    # stamped by build_segment_trie's warp pass -- assembly must run AFTER the build.
    def build_node(node: TrieNode) -> LlmNode:
        return build_dynamo_llm_node(
            node,
            build=result.builds[node.node_id],
            streaming=streaming,
        )

    graph = assemble_trie_graph(
        nodes,
        result,
        build_node=build_node,
        provenance=ProvenanceSpec(source=_SOURCE, tool="aiperf-dynamo-trie/1"),
        analysis=analysis,
        block_size=bs,
    )
    return graph, extra_tags


_DYNAMO_TRACE_EVENT_TYPES = frozenset(
    {"request_end", "tool_start", "tool_end", "tool_error"}
)

# ``(GraphParseContext attribute, from_dynamo_trace kwarg)`` for every knob
# forwarded verbatim when set. Only ``run_streaming`` is renamed on the wire.
_CTX_FORWARD_KNOBS: tuple[tuple[str, str], ...] = (
    ("content_root_seed", "content_root_seed"),
    ("content_tokenizer", "content_tokenizer"),
    ("prompt_corpus", "prompt_corpus"),
    ("num_dataset_entries", "num_dataset_entries"),
    ("max_context_length", "max_context_length"),
    ("max_isl", "max_isl"),
    ("max_osl", "max_osl"),
    ("run_streaming", "streaming"),
    # Skip-None is correct here: ``None`` means no per-trace compression on both
    # the ctx and the entry, so omitting it yields the same replay as forwarding.
    ("idle_gap_cap_seconds", "idle_gap_cap_seconds"),
)


class DynamoTraceAdapter:
    """Dynamo agent-trace v1 workload adapter (JSONL/JSONL.gz files, segmented dirs)."""

    @classmethod
    def can_load(cls, path: Path) -> bool:
        if path.is_dir():
            # Mirror discover_segments: any *.jsonl / *.jsonl.gz content
            # (segmented or plain), sniffed on the first file in reader order.
            segs = sorted(
                (
                    c
                    for c in path.iterdir()
                    if c.is_file() and c.name.endswith((".jsonl", ".jsonl.gz"))
                ),
                key=_dir_segment_sort_key,
            )
            if not segs:
                return False
            return cls._first_record_matches(segs[0])
        if path.suffix.lower() in (".gz", ".jsonl"):
            return cls._first_record_matches(path)
        return False

    @classmethod
    def _first_record_matches(cls, path: Path) -> bool:
        """Sniff the first non-empty line; unreadable/corrupt bytes mean "not ours".

        Detection must never crash on a candidate file: a truncated final gzip
        member raises EOFError, corrupt deflate data raises zlib.error, and a
        non-gzip file behind a .gz name raises BadGzipFile (an OSError) -- all
        of these are "this adapter does not claim the file", not errors.
        """
        import gzip
        import zlib

        is_gz = path.name.lower().endswith(".gz")
        try:
            with gzip.open(path, "rb") if is_gz else path.open("rb") as f:
                for raw in f:
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    rec = orjson.loads(stripped)
                    return isinstance(rec, dict) and _is_dynamo_trace_record(rec)
        except (OSError, EOFError, zlib.error, orjson.JSONDecodeError):
            return False
        return False

    @classmethod
    def parse(
        cls,
        path: Path,
        ctx: GraphParseContext | None = None,
        *,
        direct_store: GraphSegmentUnifiedBackingStore | None = None,
    ) -> ParsedGraph:
        """Convert ``path`` into a :class:`ParsedGraph` via :func:`from_dynamo_trace`.

        ``ctx`` carries the run-derived knobs (seed / tokenizer / corpus /
        idle-gap cap / selection), each forwarded ONLY when set so a partial
        ctx never clobbers the entry's ``prompt_corpus="coding"`` default with
        ``None``. ``idle_gap_cap_seconds`` follows the same rule as every
        other knob: ``None`` means no per-trace compression on both the ctx and
        the entry, so recorded delays replay unchanged rather than being
        silently zeroed. Generation is likewise always
        pinned to the recorded ``output_tokens`` (weka parity; see
        ``build_dynamo_llm_node``) — there is no behavior env knob left on
        this entry. The run tokenizer
        trust/revision publish to the loader-preload env before any callbacks
        are built (:func:`publish_ctx_tokenizer_env`).

        This is the production store-build parse entry (the registry dispatches
        here for ``dynamo_trace``), and it lowers freshly-read chains exactly
        once, so it opts into ``release_replay=True`` to free the recorded
        replay hash lists during the build. The dynamo-only knobs are set HERE,
        at the adapter's own entry, rather than threaded through the
        format-agnostic ``GraphParseContext`` / ``parse_graph`` dispatch.

        ``direct_store`` is an optional live unified store to write segments
        through at parse time, reaching this entry via the ``parse_graph`` ->
        ``_parse_via_adapter`` ``**adapter_kwargs`` passthrough. It is a
        supported-but-UNWIRED capability: no production caller supplies it (the
        ``GraphStoreBuilder`` calls ``parse_graph_workload(run, path)`` with no
        adapter kwargs, and the parallel path passes ``direct_store=None``
        because the sink cannot cross a process boundary), so only tests
        exercise it.
        A keyword-only param -- like ``release_replay`` it is a dynamo-only knob
        the uniform ``parse(path, ctx)`` registry call never supplies -- forwarded
        ONLY when set so the ``direct_store=None`` protocol-default entry is
        byte-identical to before (the same forward-only-when-set rule the ctx
        knobs follow).

        ``ctx.num_dataset_entries`` / ``ctx.max_context_length`` forward the
        filter-then-cap session-tree selection (ai-dynamo/aiperf#1106) when set;
        this is the production dynamo build entry, so the run honors them on both
        the serial and fused-parallel build paths.
        """
        assert_ctx_knobs_supported(ctx)

        publish_ctx_tokenizer_env(ctx)
        kwargs: dict[str, Any] = {
            "release_replay": True,
        }
        if direct_store is not None:
            kwargs["direct_store"] = direct_store
        if ctx is not None:
            kwargs.update(cls._forward_ctx_kwargs(ctx))
        return from_dynamo_trace(path, **kwargs)

    @staticmethod
    def _forward_ctx_kwargs(ctx: GraphParseContext) -> dict[str, Any]:
        """Map the set-only ``ctx`` knobs onto ``from_dynamo_trace`` kwargs.

        Split out of :meth:`parse` so the entry point stays under the project's
        C901 gate rather than carrying a baseline suppression.

        Each knob forwards ONLY when set, so a partial ctx never clobbers an
        entry default (notably ``prompt_corpus="coding"``) with ``None``.
        ``idle_gap_cap_seconds`` needs no special case: ``None`` means no
        per-trace compression on both sides, so the table's skip-None rule
        yields the same replay as forwarding it.
        """
        kwargs: dict[str, Any] = {
            wire: value
            for attr, wire in _CTX_FORWARD_KNOBS
            if (value := getattr(ctx, attr)) is not None
        }
        if ctx.ignore_trace_delays:
            kwargs["ignore_trace_delays"] = True
        return kwargs


def _is_dynamo_trace_record(rec: dict) -> bool:
    """Sniff a raw record for the current ``dynamo.request.trace.v1`` schema.

    Dynamo's file sinks wrap every line in a ``{"timestamp", "event"}``
    envelope (see ``trace_reader.unwrap_sink_envelope``), so unwrap before
    matching; bare records (fixtures, older captures) match directly.
    ``agent_context`` is optional in the current schema (absent on replay-only
    records), so a missing/``None`` context still identifies a dynamo record.
    """
    rec = unwrap_sink_envelope(rec)
    ac = rec.get("agent_context")
    return (
        rec.get("schema") == DYNAMO_TRACE_SCHEMA_V1
        and rec.get("event_type") in _DYNAMO_TRACE_EVENT_TYPES
        and (isinstance(ac, dict) or ac is None)
    )


# --- internal data shapes -------------------------------------------------


class _Turn:
    """One assistant turn = one ``request_end`` record."""

    __slots__ = ("record",)

    def __init__(self, record: AgentTraceRecord) -> None:
        self.record = record


class _Chain:
    """Per-session chain: ordered turns + session-level identity."""

    __slots__ = (
        "session_id",
        "parent_session_id",
        "turns",
    )

    def __init__(
        self,
        session_id: str,
        *,
        parent_session_id: str | None,
        turns: list[_Turn],
    ) -> None:
        self.session_id = session_id
        self.parent_session_id = parent_session_id
        self.turns = turns


# --- record collection ----------------------------------------------------


def _intern_replay_hashes(record: AgentTraceRecord, table: dict[int, int]) -> None:
    """Rewrite the record's replay hash list in place with canonical int objects.

    Values never change (dict membership is by ``==``); only object identity is
    shared, so every re-listed occurrence of a hash value across the whole
    capture points at ONE ``int`` object instead of a fresh ~36 B ``orjson``
    allocation. Every downstream consumer of ``input_sequence_hashes`` /
    ``TrieRequest.hash_ids`` is value-semantic (dict/set/list/compare), so
    sharing objects is transparent -- store bytes, sidecar, and envelope are all
    unchanged (the golden digest and three-way parity gate that).

    ``hashes[:] =`` mutates the SAME list object inside the (validated) pydantic
    model, so no assignment-revalidation runs and the pre-intern duplicate
    objects drop one record at a time (O(one record) transient), never the whole
    capture at once.

    Negative-id footnote (performance, not correctness): recorded hashes are
    validated non-negative in
    :meth:`~aiperf.dataset.graph.adapters.dynamo.trace_reader.AgentReplayMetrics._reject_negative_hashes`,
    so negatives never enter this table; the virtual negative ids are minted
    later, at lowering, and never reach this code. Even hypothetically,
    ``hash(-1) == hash(-2)`` only costs an extra bucket probe -- dict membership
    is decided by ``==``, so equal hashes for unequal values can never alias.
    """
    req = record.request
    replay = req.replay if req is not None else None
    if replay is None:
        return
    hashes = replay.input_sequence_hashes
    sd = table.setdefault
    hashes[:] = [sd(h, h) for h in hashes]


@dataclass(slots=True)
class _SkipCounter:
    """Tallies records the shared fold skips, as an `iter_session_records` hook."""

    n: int = 0
    """Number of records seen."""
    duplicates: int = 0
    """Number of duplicate request records skipped."""

    def count(self, record: AgentTraceRecord) -> None:
        """Increment; the record itself is not needed, only the tally."""
        self.n += 1

    def count_duplicate(self, record: AgentTraceRecord) -> None:
        """Count a duplicate record skipped by the shared reader."""
        self.duplicates += 1


def _collect_records(
    path: str | Path,
    session_id_filter: str | None,
) -> tuple[dict[str, list[AgentTraceRecord]], dict[str, str], int, int]:
    """Group records by session; also build the parent map and a skip counter.

    Returns ``(by_session, parent_link, skipped_no_context, duplicates)`` where
    ``skipped_no_context`` counts non-request records dropped for carrying no
    ``agent_context`` and no synthetic session identity.

    This is the SINGLE point where every dynamo record is materialized before
    lowering: the streaming reader is fully drained into ``by_session`` here, so
    all ``H`` recorded ``input_sequence_hashes`` slots are simultaneously live
    (the "record-window plateau") before a single ``TrieRequest`` exists. Read-
    time interning of each record's replay hashes (via :func:`_intern_replay_hashes`,
    below) is therefore the ONLY interception that caps that plateau -- and it
    also carries the canonical objects on into lowering, since the ``list()``
    copy in ``dynamo_trie_nodes`` preserves element identity.

    ``intern`` is a PER-PARSE ``dict[int, int]`` local: born before the read loop,
    dead when this function returns (before lowering), never module/global state
    and never crossing parses -- re-parses get a fresh table, exactly like the
    lowering virtual-id counter. Its resident footprint is one entry per unique
    hash value (~5.2 MB @6k, ~0.9 GB @1M); the transient dict-resize high-water
    during growth is ~1.5x that resident size and is dropped at return, safely
    below the downstream content-loop peak, so it never sets the parse's
    high-water mark.
    """
    by_session: dict[str, list[AgentTraceRecord]] = defaultdict(list)
    parent_link: dict[str, str] = {}
    skipped = _SkipCounter()
    intern: dict[int, int] = {}
    for ctx, record in iter_session_records(
        path,
        session_id=session_id_filter,
        on_no_context=skipped.count,
        on_duplicate=skipped.count_duplicate,
        synthesize_contextless_requests=True,
    ):
        _intern_replay_hashes(record, intern)
        by_session[ctx.session_id].append(record)
        # AgentContext is stamped per request from headers, so records of one
        # session may disagree (parent header only on later calls). First
        # non-self parent wins. This map is the SINGLE parent authority: the
        # forest guard walks it and _records_to_chain receives the chain's
        # parent from it, so the two can never diverge.
        parent = resolve_parent(ctx)
        if parent is not None and ctx.session_id not in parent_link:
            parent_link[ctx.session_id] = parent
    return by_session, parent_link, skipped.n, skipped.duplicates


def _records_to_chain(
    recs: list[AgentTraceRecord],
    *,
    session_id: str,
    parent_session_id: str | None,
) -> _Chain:
    """Group time-ordered records into _Turn objects pivoted on request_end.

    ``parent_session_id`` comes from the caller's ``parent_link`` map (first
    non-self parent across ALL of the session's records), so a session whose
    earliest records lack the parent header -- e.g. a harness tool event
    before any request_end -- still links to its parent.
    """
    # tool_start / tool_end / tool_error records are recognized (see
    # _DYNAMO_TRACE_EVENT_TYPES) but not lowered: tool time is already implicit
    # in the recorded end-to-start gaps the replay honors, and no consumer
    # reads a per-node tool breakdown.
    turns: list[_Turn] = [
        _Turn(record=rec) for rec in recs if rec.event_type == "request_end"
    ]
    return _Chain(
        session_id=session_id,
        parent_session_id=parent_session_id,
        turns=turns,
    )


# --- cycle / depth guard --------------------------------------------------


def _guard_chain_forest(
    chains: dict[str, _Chain],
    parent_link: dict[str, str],
    *,
    max_depth: int,
) -> None:
    """Raise on parent_link cycles or chains exceeding ``max_depth``.

    A cycle is any path A -> B -> ... -> A through ``parent_link``. Depth is
    counted from the chain's root inclusive; a root is depth 1, its child is 2.
    """
    for pid in chains:
        seen: list[str] = []
        cur: str | None = pid
        while cur is not None:
            if cur in seen:
                cycle = " -> ".join(seen + [cur])
                raise DynamoTraceAdapterError(f"parent_link cycle detected: {cycle}")
            seen.append(cur)
            if len(seen) > max_depth:
                path = " -> ".join(seen)
                raise DynamoTraceAdapterError(
                    f"parent_link depth exceeds AIPERF_DYNAMO_MAX_SUBAGENT_DEPTH"
                    f"={max_depth}: {path}"
                )
            parent = parent_link.get(cur)
            # Stop walking when parent isn't itself in the chain set
            # (it's effectively the root for this forest).
            if parent is None or parent not in chains:
                break
            cur = parent


__all__ = [
    "analyze_dynamo_chains_trie",
    "DynamoTraceAdapter",
    "DynamoTraceAdapterError",
    "EmptyDynamoTraceError",
    "from_dynamo_trace",
    "group_chains_into_trees",
    "root_of_sessions",
]
