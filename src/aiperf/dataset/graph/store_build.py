# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph unified-store build, extracted from the DatasetManager.

:class:`GraphStoreBuilder` owns the ONE store-build pipeline for every graph
workload (dynamo): parse or stream the workload,
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
    from aiperf.dataset.graph.segment_trie.store_builder import TraceSegmentPayload
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
        f"segments={stats.segment_count:,} "
        f"content_bytes={stats.content_bytes:,} "
        f"node_manifests={stats.node_manifest_count:,} "
        f"manifest_bytes={stats.manifest_bytes:,} "
        f"traces={stats.trace_count:,} "
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


def resolve_graph_store_base_path() -> Path:
    """The root both graph artifact dirs are created under.

    Shared with the DatasetManager, which claims and locks those dirs BEFORE
    handing the build to :class:`GraphStoreBuilder`; deriving the root twice
    would let the claim and the build drift onto different roots.
    """
    return Environment.DATASET.MMAP_BASE_PATH or Path(tempfile.gettempdir())


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

        Serves every graph workload (dynamo)
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

        base_path = resolve_graph_store_base_path()

        ref = resolve_graph_workload(self.run)
        fmt = ref.format if ref is not None else None
        # ONE store-build pipeline for every graph workload: parse once and
        # drain that parse through the eager interned builder. Dynamo's parser
        # may fan out internally before this parent-side drain.
        catalog, prefix_source = await self._build_graph_store_streaming(
            graph_path, base_path, fmt
        )

        node_count = sum(len(nodes) for nodes in catalog.values())
        self.info(
            f"graph unified store built: {len(catalog):,} traces, "
            f"{node_count:,} node manifests at {base_path} (benchmark_id="
            f"{self.run.benchmark_id})"
        )

        # The payload-stream drain drops each per-trace ParsedGraph, but the
        # returned prefix source (merged structural graph on that drain, or
        # the full parse in-process) preserves the
        # ``theoretical_prefix_cache_*`` node fields, so the per-node
        # prefix-cache map is recovered from it. Trace universe + prefix-cache
        # map are first-class on the graph facet -- no stub conversations.
        prefix_cache_by_trace = self._build_graph_prefix_cache_by_trace(prefix_source)
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

        * Dynamo uses a bounded worker-payload stream when its lowering options
          permit it. The parent drains those payloads directly into the
          unified store, so the corpus-sized ``ParsedGraph`` list never exists
          in the parent. A ``max_isl`` request falls back to the eager route
          because that option changes lowering semantics.
        * Undetected formats parse ONCE off-loop (whole-graph lowering) and
          drain that parse through the eager
          :meth:`_build_interned_unified_store`. The interned drain is also the
          only one that persists dynamic-slot envelopes (assembly
          items/capture) that the streaming payload envelope cannot carry.

        Both drains (this in-process one and the payload-stream drain
        :meth:`_build_graph_store_streaming_trie`, kept for worker-pool
        producers) build the SAME on-disk unified store (content pool +
        per-node manifests; the byte-parity suites prove it) and each writes
        its own mandatory content-free graph_meta sidecar; no caller sidecar
        pass exists.

        Returns ``(catalog, prefix_source)`` where the second element is the
        caller's prefix-cache source: the full parse (in-process interned
        drain) or the merged structural graph (payload-stream drain).
        """
        if fmt == "dynamo_trace":
            from aiperf.dataset.graph.adapters.dynamo.trace import (
                assert_ctx_knobs_supported,
            )
            from aiperf.dataset.graph.adapters.dynamo.trace_parallel import (
                stream_dynamo_trace_segment_payloads,
            )
            from aiperf.dataset.graph.parser import GraphParseError
            from aiperf.dataset.graph.workload_detect import (
                resolve_graph_parse_context,
            )

            ctx = resolve_graph_parse_context(self.run)
            # This route never calls DynamoTraceAdapter.parse, so the adapter's
            # own ctx gate must be applied here too or the flags it refuses are
            # silently ignored on the DEFAULT dynamo path.
            assert_ctx_knobs_supported(ctx)
            if ctx.max_isl is None:
                idle_gap_cap_seconds = ctx.idle_gap_cap_seconds
                # Same rule the parallel builder applies: ignoring recorded
                # delays with no cap authored means compress everything.
                if ctx.ignore_trace_delays and idle_gap_cap_seconds is None:
                    idle_gap_cap_seconds = 0.0
                self.info(
                    "Dynamo hybrid load: streaming worker payloads into "
                    "parent-side unified store"
                )
                payloads = stream_dynamo_trace_segment_payloads(
                    graph_path,
                    content_root_seed=ctx.content_root_seed,
                    idle_gap_cap_seconds=idle_gap_cap_seconds,
                    content_tokenizer=ctx.content_tokenizer,
                    prompt_corpus=ctx.prompt_corpus or "coding",
                    release_replay=True,
                    max_depth=Environment.DYNAMO.MAX_SUBAGENT_DEPTH,
                    num_dataset_entries=ctx.num_dataset_entries,
                    max_context_length=ctx.max_context_length,
                    max_isl=ctx.max_isl,
                    max_osl=ctx.max_osl,
                    streaming=ctx.run_streaming,
                    ignore_trace_delays=ctx.ignore_trace_delays,
                )
                try:
                    return await self._build_graph_store_streaming_trie(
                        payloads, base_path
                    )
                except GraphParseError:
                    raise
                except ValueError as exc:
                    raise GraphParseError(f"{graph_path}: {exc}") from exc

        # Every non-streaming format: the adapter's parse returns ONE
        # ParsedGraph at this layer, so there is a single parsed result to
        # drain. Dynamo also uses this fallback when max_isl is requested,
        # because that selection changes lowering semantics.
        from aiperf.dataset.graph.workload_detect import parse_graph_workload
        from aiperf.dataset.graph_segment_unified_store import (
            GraphSegmentUnifiedBackingStore,
        )

        pool_missing_msg = (
            f"graph workload {graph_path} parsed without a segment_pool; "
            "every graph parse lowers onto the unified segment store, so "
            "a pool-less parse is a lowering bug"
        )
        parsed = await asyncio.to_thread(parse_graph_workload, self.run, graph_path)
        if parsed.segment_pool is None:
            raise ValueError(pool_missing_msg)

        unified_store = GraphSegmentUnifiedBackingStore(
            base_path=base_path,
            benchmark_id=self.run.benchmark_id,
        )
        # The store is parent-owned: Dynamo workers build without a live store,
        # then this process drains the completed graph into it. If the drain
        # raises before finalize, remove the partial blob and manifests.
        try:
            catalog = await self._build_interned_unified_store(parsed, unified_store)
        except BaseException:
            unified_store.abort()
            shutil.rmtree(unified_store.data_dir, ignore_errors=True)
            raise
        self.info(
            f"GRAPH_SEGMENT UNIFIED store built (parent-side interned drain): "
            f"{len(catalog):,} traces at {base_path} "
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
        """Drain a worker-pool payload STREAM into the ONE unified store.

        The payload-stream drain, for worker-pool producers: taken by
        ``dynamo_trace`` workloads with no ``max_isl`` request; every other
        format takes the in-process interned drain. Each pool worker serializes its trace's
        trie payloads,
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
        from aiperf.dataset.graph.segment_trie.store_builder import (
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

        # As with the interned drain, a mid-stream failure leaves a partially
        # spilled content.blob; abort() + rmtree remove the store so no
        # half-written file survives for a later open.
        try:
            self.info("Dynamo hybrid load: unified-store payload drain started")

            def _drain() -> dict[str, dict[str, int]]:
                # The payload iterator and store writes are synchronous. Run
                # the async wrapper in the worker thread so iterator
                # advancement cannot freeze the DatasetManager event loop.
                return asyncio.run(
                    build_unified_trie_store_from_payloads(
                        payloads, unified, structural_sink=structural_sink
                    )
                )

            catalog = await asyncio.to_thread(_drain)
        except BaseException:
            unified.abort()
            shutil.rmtree(unified.data_dir, ignore_errors=True)
            raise
        self.info(
            f"GRAPH_SEGMENT UNIFIED store built (streaming): "
            f"{len(catalog):,} traces at {base_path} "
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
        """Build the interned unified segment trie store from a whole-graph parse.

        Drains the parse's content-addressed pool into ``unified`` and writes the
        per-node interned manifests via the node-typed
        :func:`build_unified_trie_store_interned`. The in-process drain for
        every format inside :meth:`_build_graph_store_streaming` except the
        dynamo payload-stream route, and
        the only drain that persists dynamic-slot (assembly items/capture)
        envelopes.
        """
        from aiperf.dataset.graph.segment_trie.store_builder import (
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
        """Merge the payload drain's per-trace structural graphs; hard fail on any gap.

        Only the payload-stream drain (:meth:`_build_graph_store_streaming_trie`)
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
        self._advise_virtual_hash_fallback(parsed)

    def _advise_virtual_hash_fallback(self, parsed: ParsedGraph) -> None:
        """Surface traces whose KV hints were synthesized, not recorded.

        A turn with no recorded replay is lowered onto per-session VIRTUAL
        negative block ids and tagged ``virtual-hash-fallback``
        (``dynamo.trie_lowering``). Those turns still replay, but their prefix
        reuse is synthetic: cross-session sharing is whatever the virtual
        chain happens to produce, not what the capture observed. Until now the
        tag was written to ``TraceRecord.tags`` and read by nothing at run
        time -- only ``aiperf dynamo trace-report`` surfaced it -- so a run on
        a partly-unrecorded corpus looked identical to a fully recorded one.
        Every graph route lands the sidecar, so this is the one chokepoint
        that sees every build.
        """
        from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
            VIRTUAL_HASH_FALLBACK_TAG,
        )

        tagged = sum(1 for t in parsed.traces if VIRTUAL_HASH_FALLBACK_TAG in t.tags)
        if not tagged:
            return
        total = len(parsed.traces)
        self.notice(
            f"{tagged:,} of {total:,} trace(s) fall back to virtual KV block hashes: "
            "those turns carried no recorded replay hashes, so their prefix "
            "reuse is synthesized per session rather than observed. Cache-hit "
            "metrics for them reflect the synthetic chain, not the capture. Run "
            "'aiperf dynamo trace-report <path>' for the per-record breakdown."
        )

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
        from aiperf.dataset.graph.segment_trie.prefix_cache import (
            extract_prefix_cache_by_node,
        )

        out: dict[str, dict[str, list[int]]] = {}
        for trace in parsed.traces:
            trace_graph = resolve_trace_graph(parsed, trace)
            by_node = extract_prefix_cache_by_node(trace_graph)
            if by_node:
                out[trace.id] = by_node
        return out
