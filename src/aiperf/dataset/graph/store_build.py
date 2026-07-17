# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph unified-store build, extracted from the DatasetManager.

:class:`GraphStoreBuilder` owns the ONE store-build pipeline for every graph
workload (weka / dynamo / native / dag_jsonl): parse or stream the workload,
drain it into the unified segment store (content pool + per-node manifests)
rooted where the worker reads, write the mandatory content-free graph_meta
sidecar, and hand back the graph facet plus both locations as a
:class:`GraphStoreBuildResult`. The DatasetManager is the only production
caller; it broadcasts the result's locations in the graph-typed dataset
notification and never rebuilds.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.common.environment import Environment
from aiperf.common.exceptions import DatasetError
from aiperf.common.mixins.aiperf_logger_mixin import AIPerfLoggerMixin
from aiperf.common.models.dataset_models import GraphDatasetMetadata
from aiperf.dataset.graph.workload_detect import publish_graph_loader_tokenizer_env

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.dataset.graph.models import ParsedGraph
    from aiperf.dataset.graph.segment_ir.store_builder import TraceSegmentPayload
    from aiperf.dataset.graph_segment_unified_store import (
        GraphSegmentUnifiedBackingStore,
        GraphStoreBuildStats,
    )

__all__ = [
    "GraphStoreBuildResult",
    "GraphStoreBuilder",
]


def _format_store_build_stats(stats: GraphStoreBuildStats | None) -> str:
    """Format a store build snapshot as a compact ``key=value`` log suffix.

    Module-level (never a method) so the ``_build_graph_store_streaming*`` build
    methods -- which existing tests invoke unbound with stub selves -- can call
    it without a ``self.`` attribute that those stubs lack. Total over ``None``
    (``build_stats`` is ``None`` until finalize) so a pre-finalize store logs a
    marker instead of crashing the build-success line.
    """
    if stats is None:
        return "build_stats=unavailable"
    rss = "n/a" if stats.peak_rss_mib is None else f"{stats.peak_rss_mib:.1f}"
    return (
        f"segments={stats.segment_count} "
        f"content_bytes={stats.content_bytes} "
        f"node_manifests={stats.node_manifest_count} "
        f"manifest_bytes={stats.manifest_bytes} "
        f"traces={stats.trace_count} "
        f"peak_rss_mib={rss}"
    )


@dataclass(slots=True)
class GraphStoreBuildResult:
    """Everything the DatasetManager needs from one graph store build."""

    facet: GraphDatasetMetadata
    """Trace universe + per-node prefix-cache map for the broadcast."""

    sidecar_path: Path
    """Where the mandatory graph_meta sidecar was written."""

    base_path: Path
    """Store root the worker opens (derived ONCE here)."""


class GraphStoreBuilder(AIPerfLoggerMixin):
    """Builds the unified graph segment store + sidecar for one run.

    One instance per build: :meth:`build` streams or parses the workload into
    the unified store rooted at the SAME location the worker reads from
    (``Environment.DATASET.MMAP_BASE_PATH`` or the system temp dir, plus the
    run ``benchmark_id``), writes the mandatory graph_meta sidecar, and
    returns the :class:`GraphStoreBuildResult` the DatasetManager broadcasts.
    """

    def __init__(self, run: BenchmarkRun, **kwargs) -> None:
        super().__init__(**kwargs)
        self.run = run
        self._sidecar_path: Path | None = None

    async def build(self, graph_path: Path) -> GraphStoreBuildResult:
        """Build the graph store mmap for a graph workload; return facet + locations.

        Serves every graph workload (weka / dynamo / native / dag_jsonl)
        through the ONE streaming store build
        (:meth:`_build_graph_store_streaming`), which parses into a
        ``ParsedGraph`` (threading ``run.random_seed`` so the synthesized
        corpus / node ordinals stay deterministic for the run), serializes
        every node's request envelope into the graph store mmap, and recovers
        the graph facet (:class:`GraphDatasetMetadata`: the trace universe
        plus the per-node theoretical prefix-cache map). The worker
        materializes the real per-node payloads from the graph store.

        The sidecar is mandatory for graph runs (the TimingManager only
        ingests the sidecar from the broadcast path; it never re-parses), so a
        drain that completed without recording its written path is a hard
        build failure, not a degraded result.
        """
        publish_graph_loader_tokenizer_env(self.run)
        self._prestart_loader_forkserver()

        from aiperf.dataset.graph.workload_detect import resolve_graph_workload

        base_path = Environment.DATASET.MMAP_BASE_PATH or Path(tempfile.gettempdir())

        ref = resolve_graph_workload(self.run)
        fmt = ref.format if ref is not None else None
        # ONE store-build pipeline for every graph workload: weka streams
        # worker-pool payloads through the trie drain; every other format
        # parses once in-process and drains that single parse through the
        # eager interned builder. Both drains write their own sidecar.
        catalog, prefix_source = await self._build_graph_store_streaming(
            graph_path, base_path, fmt
        )

        node_count = sum(len(nodes) for nodes in catalog.values())
        self.info(
            f"graph unified store built: {len(catalog)} traces, "
            f"{node_count} node manifests at {base_path} (benchmark_id="
            f"{self.run.benchmark_id})"
        )

        # The weka payload drain drops each per-trace ParsedGraph, but the
        # returned prefix source (merged structural graph on the weka drain, or
        # the full parse in-process) preserves the native
        # ``theoretical_prefix_cache_*`` node fields, so the per-node
        # prefix-cache map is recovered from it. Trace universe + prefix-cache
        # map are first-class on the graph facet -- no stub conversations.
        prefix_cache_by_trace = await asyncio.to_thread(
            self._build_graph_prefix_cache_by_trace, prefix_source
        )
        if self._sidecar_path is None:
            raise DatasetError(
                "graph store build completed without recording a graph_meta "
                "sidecar path; the sidecar is mandatory for graph runs"
            )
        return GraphStoreBuildResult(
            facet=GraphDatasetMetadata(
                trace_ids=list(catalog),
                prefix_cache_by_trace=prefix_cache_by_trace,
            ),
            sidecar_path=self._sidecar_path,
            base_path=base_path,
        )

    def _prestart_loader_forkserver(self) -> None:
        """Start the trace-loader forkserver before the build is offloaded.

        ``_eagerly_start_forkserver`` dup2-redirects the PROCESS-WIDE stdio fds
        around the helper spawn. The graph store builds run in
        ``asyncio.to_thread`` workers, so a lazily started helper (first pool
        open inside the offloaded parse) would race that swap against the
        event loop's live logging -- console lines emitted during the window
        silently go to ``/dev/null``. Starting here, on the loop at a
        known-quiet point (right after the env publish, before any parse
        thread), makes the pool's later ``get_loader_mp_context`` call a
        cached no-op (the context is built once and reused). Threads the SAME
        preload tokenizer the pool path resolves
        (``_loader_pool_context`` <- ``resolve_graph_content_tokenizer``) so
        the helper preloads the right tokenizer.
        """
        from aiperf.common.tokenizer import BUILTIN_TOKENIZER_NAME
        from aiperf.dataset._mp_context import get_loader_mp_context
        from aiperf.timing.config import resolve_graph_content_tokenizer

        # Same fallback the pool applies to its non-scheduling tokenizer_name.
        get_loader_mp_context(
            preload_tokenizer=resolve_graph_content_tokenizer(self.run)
            or BUILTIN_TOKENIZER_NAME
        )

    async def _build_graph_store_streaming(
        self,
        graph_path: Path,
        base_path: Path,
        fmt: str | None,
    ) -> tuple[dict[str, dict[str, int]], ParsedGraph]:
        """Build the unified store for EVERY graph workload; two drains.

        The one store-build pipeline, chosen by ``fmt``:

        * ``weka_trace`` (worker-pool payload stream): resolves the SAME
          run-derived content knobs as the in-process
          :func:`parse_graph_workload` -- ONE resolution,
          :func:`~aiperf.dataset.graph.workload_detect.resolve_graph_parse_context`,
          whose fields (seed, tokenizer, corpus, ``max_osl``, idle-gap cap)
          spread verbatim into the stream entry -- so build-plane topology,
          catalog, and node ordinals stay deterministic for the run (and
          content bytes additionally match when an explicit ``--random-seed``
          is set), then streams worker-pool-built trie payloads
          (:func:`stream_weka_trace_segment_payloads` emits
          :class:`TraceSegmentPayload`s) into the unified store via
          :meth:`_build_graph_store_streaming_trie`. Each worker serializes its
          trace's envelopes before returning, so the parent does not decode a
          full ``ParsedGraph`` only to re-serialize the same real content.
          ``idle_gap_cap_seconds`` forwards AS-IS: the resolver always yields
          a resolved value (never ``UNSET``), and an explicit ``None`` means
          warping DISABLED and must arrive as ``None``.
        * Every other format (dynamo / native / dag_jsonl / undetected --
          ``fmt=None`` fails inside ``parse_graph_workload``'s own detection):
          parse ONCE in-process (whole-graph lowering) and drain that single
          parse through the eager :meth:`_build_interned_unified_store` -- the
          in-process interned drain. In-process there is no worker pool to fan
          out to, so the payload round trip the weka path needs is pure
          overhead; the interned drain also persists dynamic-slot envelopes
          (native ``@channel`` assembly items/capture, dag_jsonl live-reply
          lineage) that the streaming payload envelope cannot carry.
          ``dynamo_trace`` additionally takes the DIRECT write-through route:
          the store is constructed BEFORE the parse and passed
          as ``direct_store`` so ``build_trie_ir``'s ``pool.add`` interns each
          segment straight into the store during the parse (no second RAM pool
          copy); the interned drain then no-ops over the empty returned pool.
          Byte-identical to the eager drain by construction -- both intern in
          ``build_trie_ir``'s content-loop first-occurrence order.

        Both drains build the SAME on-disk unified store (content pool +
        per-node manifests; the byte-parity suites prove it) and each writes
        its own mandatory content-free graph_meta sidecar; no caller sidecar
        pass exists.

        Returns ``(catalog, prefix_source)`` where the second element is the
        caller's prefix-cache source: the content-free merged structural graph
        (weka payload drain) or the full parse (in-process interned drain).
        """
        if fmt == "weka_trace":
            from aiperf.dataset.graph.adapters.weka.trace import (
                stream_weka_trace_segment_payloads,
            )
            from aiperf.dataset.graph.workload_detect import (
                resolve_graph_parse_context,
            )

            ctx = resolve_graph_parse_context(self.run)
            payloads = stream_weka_trace_segment_payloads(
                str(graph_path),
                idle_gap_cap_seconds=ctx.idle_gap_cap_seconds,
                content_root_seed=ctx.content_root_seed,
                content_tokenizer=ctx.content_tokenizer,
                prompt_corpus=ctx.prompt_corpus,
                max_osl=ctx.max_osl,
                num_dataset_entries=ctx.num_dataset_entries,
                max_context_length=ctx.max_context_length,
            )
            return await self._build_graph_store_streaming_trie(payloads, base_path)

        # Every non-weka format: the adapter's parse returns ONE ParsedGraph
        # at this layer (dynamo fans out per session-tree INSIDE its own parse
        # and lowers each tree independently; dag_jsonl expands whole trees),
        # so there is a single parsed result to drain. Parse once off-loop,
        # then drain it through the eager interned builder: in-process there
        # is no worker pool to fan out to, so the payload round trip is pure
        # overhead, and the interned drain is the only one that persists
        # dynamic-slot envelopes (native @channel assembly items/capture,
        # dag_jsonl live-reply lineage).
        from aiperf.dataset.graph.workload_detect import parse_graph_workload
        from aiperf.dataset.graph_segment_unified_store import (
            GraphSegmentUnifiedBackingStore,
        )

        pool_missing_msg = (
            f"graph workload {graph_path} parsed without a segment_pool; "
            "every graph parse lowers onto the unified segment store, so "
            "a pool-less parse is a lowering bug"
        )
        if fmt == "dynamo_trace":
            # Dynamo direct write-through route: construct the
            # store BEFORE the off-loop parse and thread it as ``direct_store``
            # so build_trie_ir's ``pool.add`` interns each segment STRAIGHT INTO
            # the store during the parse (no second RAM pool copy). The store is
            # live before the parse, so the abort()+rmtree cleanup MUST cover the
            # parse too: content.blob spills incrementally, so a mid-parse
            # DynamoISLMismatchError (or any failure) must leave no partial file.
            unified_store = GraphSegmentUnifiedBackingStore(
                base_path=base_path,
                benchmark_id=self.run.benchmark_id,
            )
            try:
                parsed = await asyncio.to_thread(
                    parse_graph_workload,
                    self.run,
                    graph_path,
                    direct_store=unified_store,
                )
                if parsed.segment_pool is None:
                    raise ValueError(pool_missing_msg)
                catalog = await self._build_interned_unified_store(
                    parsed, unified_store
                )
            except BaseException:
                unified_store.abort()
                shutil.rmtree(unified_store.data_dir, ignore_errors=True)
                raise
        else:
            parsed = await asyncio.to_thread(parse_graph_workload, self.run, graph_path)
            if parsed.segment_pool is None:
                raise ValueError(pool_missing_msg)

            unified_store = GraphSegmentUnifiedBackingStore(
                base_path=base_path,
                benchmark_id=self.run.benchmark_id,
            )
            # The store spills content.blob incrementally, so a drain that raises
            # before finalize would leave a partial blob on disk; abort() + rmtree
            # remove it so a later store open never trips on the half-written file.
            try:
                catalog = await self._build_interned_unified_store(
                    parsed, unified_store
                )
            except BaseException:
                unified_store.abort()
                shutil.rmtree(unified_store.data_dir, ignore_errors=True)
                raise
        self.info(
            f"GRAPH_SEGMENT UNIFIED store built (in-process interned drain): "
            f"{len(catalog)} traces at {base_path} "
            f"(benchmark_id={self.run.benchmark_id}) "
            f"{_format_store_build_stats(unified_store.build_stats)}"
        )
        await asyncio.to_thread(self._write_graph_sidecar, parsed, catalog, base_path)
        return catalog, parsed

    async def _build_graph_store_streaming_trie(
        self,
        payloads: Iterable[TraceSegmentPayload],
        base_path: Path,
    ) -> tuple[dict[str, dict[str, int]], ParsedGraph]:
        """Drain the weka worker-pool payload STREAM into the ONE unified store.

        The weka drain: :meth:`_build_graph_store_streaming` routes only
        ``weka_trace`` here (every other format takes the in-process
        interned drain). Each pool worker serializes its trace's trie payloads,
        and this method drains that stream into the interned unified store
        (content pool + per-node manifests) via
        :func:`build_unified_trie_store_from_payloads`, so the corpus-scale
        path builds the SAME unified store the in-process interned drain does
        (the worker opens the unified reader either way). Streaming
        ``put_segment`` dedup on the content-addressed id bounds RAM. Each
        streamed row's content-free structural graph is collected and merged
        ONCE (:meth:`_merge_structural_graphs`); the merged graph feeds both
        the mandatory :meth:`_write_graph_sidecar` (the TimingManager loads the
        sidecar instead of re-parsing the whole corpus) and the returned
        prefix-cache source. Returns ``(catalog, merged_structural)``.
        """
        from aiperf.dataset.graph.segment_ir.store_builder import (
            build_unified_trie_store_from_payloads,
        )
        from aiperf.dataset.graph_segment_unified_store import (
            GraphSegmentUnifiedBackingStore,
        )

        structural_sink: list[bytes] = []
        unified = GraphSegmentUnifiedBackingStore(
            base_path=base_path,
            benchmark_id=self.run.benchmark_id,
        )

        def _drain_stream() -> dict[str, dict[str, int]]:
            # The payload iterator blocks in multiprocessing result.get() per
            # trace, so the whole drain runs in a worker thread (its own loop
            # covers the store's async finalize) and this service's event loop
            # keeps serving heartbeats during corpus-scale builds.
            return asyncio.run(
                build_unified_trie_store_from_payloads(
                    payloads, unified, structural_sink=structural_sink
                )
            )

        # As with the interned drain, a mid-stream failure leaves a partially
        # spilled content.blob; abort() + rmtree remove the store so no
        # half-written file survives for a later open.
        try:
            catalog = await asyncio.to_thread(_drain_stream)
        except BaseException:
            unified.abort()
            shutil.rmtree(unified.data_dir, ignore_errors=True)
            raise
        self.info(
            f"GRAPH_SEGMENT UNIFIED store built (streaming): "
            f"{len(catalog)} traces at {base_path} "
            f"(benchmark_id={self.run.benchmark_id}) "
            f"{_format_store_build_stats(unified.build_stats)}"
        )
        # Structural decode/merge + catalog cross-check + write are pure
        # CPU/sync-IO on corpus-scale inputs; off-loop like the drain above.
        merged = await asyncio.to_thread(self._merge_structural_graphs, structural_sink)
        await asyncio.to_thread(self._write_graph_sidecar, merged, catalog, base_path)
        return catalog, merged

    async def _build_interned_unified_store(
        self,
        parsed: ParsedGraph,
        unified: GraphSegmentUnifiedBackingStore,
    ) -> dict[str, dict[str, int]]:
        """Build the interned unified segment-trie store from a whole-graph parse.

        Drains the parse's content-addressed pool into ``unified`` and writes the
        per-node interned manifests via the node-typed
        :func:`build_unified_trie_store_interned`. The in-process drain for
        EVERY non-weka format inside :meth:`_build_graph_store_streaming`, and
        the only drain that persists dynamic-slot (assembly items/capture)
        envelopes.
        """
        from aiperf.dataset.graph.segment_ir.store_builder import (
            build_unified_trie_store_interned,
        )

        def _build() -> dict[str, dict[str, int]]:
            # The interned build is a zero-yield CPU drain (orjson-encode +
            # pool copy of ALL synthesized content) until its trailing
            # ``store.finalize()`` await, so it runs in a worker thread whose
            # own loop covers that finalize and this service's event loop
            # keeps serving heartbeats during corpus-scale builds.
            return asyncio.run(build_unified_trie_store_interned(parsed, unified))

        return await asyncio.to_thread(_build)

    def _merge_structural_graphs(self, structural_sink: list[bytes]) -> ParsedGraph:
        """Merge the weka payload drain's per-trace structural graphs; hard fail on any gap.

        Only the weka payload drain (:meth:`_build_graph_store_streaming_trie`)
        collects a per-trace structural stream to merge here; the in-process
        interned drain keeps the whole parse and never calls this. The merged
        graph is content-free but preserves node metadata, so it feeds BOTH the
        mandatory graph_meta sidecar and the per-node prefix-cache map. A
        missing or unmergeable structural stream is a build failure, not a
        degradation.
        """
        from aiperf.dataset.graph.codecs import decode_parsed_graph_msgpack
        from aiperf.dataset.graph.merge import merge_parsed_graphs

        if not structural_sink:
            raise DatasetError(
                "graph store build produced no structural graphs: the "
                "streaming drain yielded zero traces, so there is nothing to "
                "benchmark and no graph_meta sidecar can be written"
            )
        try:
            return merge_parsed_graphs(
                decode_parsed_graph_msgpack(b) for b in structural_sink
            )
        except Exception as e:
            raise DatasetError(
                f"streaming structural graphs failed to merge: {e!r}; the "
                "graph_meta sidecar is mandatory for graph runs"
            ) from e

    def _write_graph_sidecar(
        self,
        parsed: ParsedGraph,
        catalog: dict[str, dict[str, int]],
        base_path: Path,
    ) -> None:
        """Write the mandatory content-free graph_meta sidecar for this build.

        Every graph route MUST land this file: the TimingManager only ingests
        the sidecar, from the exact path recorded here for the graph-typed
        dataset broadcast (it never re-parses), so a skipped or failed write
        would leave the run unschedulable. A structural catalog that diverges
        from the store's build catalog means the sidecar would describe a
        DIFFERENT topology than the envelopes the worker reads -- hard fail.
        """
        from aiperf.dataset.graph.codecs import GRAPH_META_SCHEMA_VERSION
        from aiperf.dataset.graph.graph_meta_sidecar import (
            catalogs_match,
            write_graph_meta_sidecar,
        )

        if not catalogs_match(parsed, catalog):
            raise DatasetError(
                "graph_meta sidecar catalog mismatch: the structural graph's "
                "node ordinals diverged from the unified store's build "
                "catalog; the TimingManager cannot schedule this run"
            )
        out = write_graph_meta_sidecar(
            parsed,
            base_path=base_path,
            benchmark_id=self.run.benchmark_id,
            source_fingerprint={},
            schema_version=GRAPH_META_SCHEMA_VERSION,
        )
        self._sidecar_path = out
        self.info(f"graph_meta sidecar written: {out}")

    @staticmethod
    def _build_graph_prefix_cache_by_trace(
        parsed: ParsedGraph,
    ) -> dict[str, dict[str, list[int]]]:
        """Per-trace ``{node_id: [hit, total]}`` from a parsed trie graph.

        Resolves each trace's own graph (single-graph and heterogeneous
        multi-graph corpora), then reads the prefix-cache counts the shared
        trie build stamped on each ``LlmNode``'s native
        ``theoretical_prefix_cache_hit_blocks`` / ``_total_blocks`` fields.
        Empty when nothing is stamped (a non-trie graph or hash-id-free
        requests).
        """
        from aiperf.dataset.graph.models import resolve_trace_graph
        from aiperf.dataset.graph.segment_ir.prefix_cache import (
            extract_prefix_cache_by_node,
        )

        out: dict[str, dict[str, list[int]]] = {}
        for trace in parsed.traces:
            trace_graph = resolve_trace_graph(parsed, trace)
            by_node = extract_prefix_cache_by_node(trace_graph)
            if by_node:
                out[trace.id] = by_node
        return out
