# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph-workload detection + parsing seam for the dataset plane.

Graph traces are NOT a ``custom_dataset_loader`` plugin -- they are ingested via
the ``graph_adapter`` plugin registry (``weka_trace`` / ``dynamo_trace`` / ...)
into a :class:`~aiperf.dataset.graph.models.ParsedGraph`. Auto-detection
walks that registry (excluding ``native``, which is explicit-``--graph-format`` only) and
parse dispatch is registry-driven too -- :func:`parse_graph_workload` resolves
every run-derived parse knob into ONE :class:`GraphParseContext`
(:func:`resolve_graph_parse_context`) and every format goes through the generic
:func:`~aiperf.dataset.graph.parser.parse_graph`, whose adapters map the ctx
fields they consume via the uniform ``parse(path, ctx)`` protocol.

The **DatasetManager** (build plane) is the ONLY caller that parses: it builds
the ``ParsedGraph``, serializes the per-node envelopes into the graph store
mmap, and writes + broadcasts the content-free graph_meta sidecar. The
**TimingManager** (schedule plane) still calls :func:`resolve_graph_workload`
(detection only, from the run config alone) to decide whether the input is a
graph workload and to flip the profiling phase's timing mode to ``GRAPH_IR``,
but it ingests the broadcast sidecar rather than parsing. Deriving both the
build-time and the dispatch-time node ordinals (catalog) from that single
parse is what keeps the worker reading the right envelope.

Detection runs AT MOST ONCE per process: :func:`resolve_graph_workload` is the
single accessor every consumer calls; it reads the
``run.resolved.graph_workload`` memo (populated eagerly by the config resolver
chain in single-run mode) or derives-and-memoizes on first access for runs
that never pass the chain (``aiperf service`` processes, test-built runs).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.dataset.graph.parse_context import GraphParseContext

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun, GraphWorkloadRef
    from aiperf.dataset.graph.models import ParsedGraph

__all__ = [
    "GraphEndpointUnsupportedError",
    "is_graph_workload_path",
    "parse_graph_workload",
    "publish_graph_loader_tokenizer_env",
    "resolve_graph_parse_context",
    "resolve_graph_workload",
    "validate_graph_endpoint_type",
]


class GraphEndpointUnsupportedError(ValueError):
    """The run targets an endpoint type the graph workload dispatch cannot serve.

    The graph dispatch path materializes a chat-completions body
    (``{"messages": [...], "max_completion_tokens": N, "stream": bool}``) and
    sends it verbatim, bypassing the endpoint's ``format_payload`` reshaping.
    Pointing that body at a non-chat endpoint (``completions`` expects
    ``prompt``; ``embeddings`` expects ``input``; etc.) makes every request fail
    with a server-side 422 and no actionable up-front error. This is raised at
    configure time so the run rejects cleanly before any request is dispatched.
    """


def validate_graph_endpoint_type(run: BenchmarkRun) -> None:
    """Reject non-chat endpoint types for a graph workload run.

    The graph dispatch path materializes a chat-completions body
    (``{"messages": [...], "max_completion_tokens": N, "stream": bool}``) and
    sends it verbatim, bypassing the endpoint's ``format_payload`` reshaping.
    Only an endpoint whose path is the chat-completions path can serve that
    body; any other type (``completions`` expects ``prompt``, ``embeddings``
    expects ``input``, ...) would fail every request with a server-side 422.
    Chat-compatibility is keyed on the endpoint metadata's ``endpoint_path``
    ending in ``/chat/completions`` so any future chat-completions endpoint
    plugin passes without an allowlist edit. The DatasetManager runs this gate
    at configure time, before the store is built, so the failure is a clean
    configure-time rejection.

    Raises:
        GraphEndpointUnsupportedError: When the run's endpoint type is not a
            chat-completions endpoint.
    """
    from aiperf.plugin import plugins

    endpoint_type = run.cfg.endpoint.type
    endpoint_path = plugins.get_endpoint_metadata(endpoint_type).endpoint_path
    if endpoint_path is None or not endpoint_path.endswith("/chat/completions"):
        raise GraphEndpointUnsupportedError(
            f"graph workload: --endpoint-type '{endpoint_type}' is not "
            f"supported (endpoint_path={endpoint_path!r}). The graph "
            f"replay path emits a chat-completions body and only supports a "
            f"chat-completions endpoint (e.g. --endpoint-type chat). Re-run "
            f"with a chat endpoint, or convert the trace for the target "
            f"endpoint shape."
        )


# ``native`` is the explicit ``--graph-format`` / ``parse_graph`` fallback whose
# ``can_load`` matches any ``.yaml``/``.yml``/``.jsonl``; excluding it here keeps
# plain conversation datasets on the linear pipeline instead of hijacking them
# into graph mode. ``dag_jsonl`` is excluded so legacy
# ``--custom-dataset-type dag_jsonl`` runs stay on the linear pipeline -- the
# graph adapter is opt-in via ``--graph-format dag_jsonl`` only (its
# ``can_load`` stays implemented + tested for a future autodetect flip).
_AUTODETECT_EXCLUDED = frozenset({"native", "dag_jsonl"})


def _detect_graph_workload_format(path: Path) -> str | None:
    """Return the graph-adapter format name auto-detected for ``path``, else None.

    Walks the ``graph_adapter`` plugin registry, skipping the explicit-only
    adapters in :data:`_AUTODETECT_EXCLUDED` (``native`` and ``dag_jsonl``), and
    returns the highest-``detection_priority`` adapter whose ``can_load`` matches.
    Returns ``None`` when no trace adapter recognizes the file -- the caller then
    keeps the linear (non-graph) pipeline.
    """
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType
    from aiperf.plugin.schema.schemas import GraphAdapterMetadata

    matches: list[tuple[int, str]] = []
    for entry, cls in plugins.iter_all(PluginType.GRAPH_ADAPTER):
        if entry.name in _AUTODETECT_EXCLUDED:
            continue
        if cls.can_load(path):
            meta = entry.get_typed_metadata(GraphAdapterMetadata)
            matches.append((meta.detection_priority, entry.name))
    return max(matches)[1] if matches else None


def _graph_format_override(run: BenchmarkRun) -> str | None:
    """Return the explicit ``--graph-format`` override for ``run``, or None.

    Reads ``FileDataset.graph_format``. When set, the input is treated as a graph
    workload of exactly this adapter regardless of auto-detection (this is the
    only path that can select ``native``, which auto-detection excludes).
    """
    try:
        dataset = run.cfg.get_default_dataset()
    except (IndexError, AttributeError):
        return None
    fmt = getattr(dataset, "graph_format", None)
    return str(fmt) if fmt is not None else None


def resolve_graph_workload(run: BenchmarkRun) -> GraphWorkloadRef | None:
    """Graph workload for this run, detected AT MOST ONCE per process.

    Reads the memoized resolution when present (populated by the config
    resolver chain in single-run mode, or by a prior call); otherwise runs
    override-or-autodetect ONCE and memoizes onto ``run.resolved``. Detection
    failures degrade to None exactly like the per-consumer sniffs this
    accessor replaced (``can_load`` returns False on unreadable inputs), so
    ``aiperf service`` processes and test-built runs behave identically to the
    status quo -- minus the repeated registry walks and file I/O.
    """
    # STRICT ``is True``: a truthy check would auto-truthify on MagicMock runs
    # (the repo's documented MagicMock-path-drift trap).
    if run.resolved.graph_workload_resolved is True:
        return run.resolved.graph_workload
    ref = _derive_graph_workload(run)
    # Two threads racing the first call can both derive; the double-detect is
    # benign (idempotent same-value memoization, GIL-atomic assignments), so
    # no lock is taken.
    run.resolved.graph_workload = ref
    run.resolved.graph_workload_resolved = True
    return ref


def _derive_graph_workload(run: BenchmarkRun) -> GraphWorkloadRef | None:
    """Run override-or-autodetect once; None when the input is not a graph.

    Reads the default dataset's ``path``. When ``--graph-format`` is set
    (:func:`_graph_format_override`), the input is FORCED to be a graph
    workload of that adapter -- including ``native``, which auto-detection
    excludes. Otherwise asks the ``graph_adapter`` registry (via
    :func:`_detect_graph_workload_format`) whether any trace adapter
    recognizes the file. Returns ``None`` for any non-file dataset (synthetic
    / public / inline), a file no adapter recognizes, AND any derivation
    failure: the swallow consumers used to wrap around detection lives HERE
    now, so callers read the accessor's result bare.
    """
    from aiperf.config.resolution.plan import GraphWorkloadRef

    try:
        try:
            dataset = run.cfg.get_default_dataset()
        except (IndexError, AttributeError):
            return None
        raw_path = getattr(dataset, "path", None)
        if raw_path is None:
            return None
        path = Path(raw_path)
        fmt = _graph_format_override(run)
        if fmt is None:
            fmt = _detect_graph_workload_format(path)
        if fmt is None:
            return None
        return GraphWorkloadRef(path=path, format=fmt)
    except Exception:
        return None


def is_graph_workload_path(path: Path) -> bool:
    """True when `path` is a graph workload recognized by a trace adapter.

    Path-level companion to :func:`resolve_graph_workload` (which takes a
    run). Uses the SAME registry detection (`_detect_graph_workload_format`,
    which excludes `native`), so callers skip exactly what the graph pipeline
    will claim.
    """
    return _detect_graph_workload_format(path) is not None


def publish_graph_loader_tokenizer_env(run: BenchmarkRun) -> None:
    """Publish the run's tokenizer trust/revision for graph content synthesis.

    ``CorpusContentSynthesizer._build_generator`` (and the forkserver preload)
    reads ``AIPERF_LOADER_PRELOAD_TRUST_REMOTE_CODE`` / ``_REVISION`` at
    tokenizer-load time -- the parallel loader threads only the tokenizer NAME.
    :func:`parse_graph_workload` is the one seam every graph parse goes through,
    so it must publish the run's tokenizer trust/revision triple itself: a
    direct caller (tooling, tests) has no DatasetManager configure step to do
    it, and the forkserver helper snapshots the env once at spawn. Idempotent:
    re-publishing the same run-derived values is a no-op.
    """
    from aiperf.dataset._mp_context import configure_loader_tokenizer_env

    tokenizer = run.cfg.tokenizer
    configure_loader_tokenizer_env(
        trust_remote_code=tokenizer.trust_remote_code,
        revision=tokenizer.revision,
    )


def resolve_graph_parse_context(run: BenchmarkRun) -> GraphParseContext:
    """Resolve EVERY run-derived graph parse knob into one :class:`GraphParseContext`.

    The ONE knob spelling for the run -> parse seam: every field is populated
    with the run's RESOLVED value verbatim, so a registry-dispatched
    ``adapter.parse(path, ctx)`` is byte-identical to the run's own parse for
    every format. Each adapter maps only the ctx fields it consumes and
    ignores the rest (see :class:`GraphParseContext`), so resolving dag knobs
    for a weka run (and vice versa) is harmless -- every resolver here is a
    pure function of the run config, safe to evaluate for every format.

    ``idle_gap_cap_seconds`` carries :func:`_resolve_graph_idle_gap_cap`'s
    result AS-IS: 60.0 default / explicit float / ``None`` for the user's
    explicit ``synthesis.idle_gap_cap_seconds: null`` (warping DISABLED). It
    is never ``UNSET`` -- a run always has a resolved answer -- and the weka
    and dynamo adapters forward the tri-state verbatim.

    Lazy imports: ``timing.config`` avoids the ``timing`` <-> graph loader
    import cycle; the rest keep this module light at import time.
    """
    from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
    from aiperf.dataset.loader.dag_jsonl import _resolve_delay_cap_seconds
    from aiperf.timing.config import (
        resolve_graph_content_seed,
        resolve_graph_content_tokenizer,
    )

    tokenizer = run.cfg.tokenizer
    endpoint_info = ModelEndpointInfo.from_run(run)
    return GraphParseContext(
        content_root_seed=resolve_graph_content_seed(run),
        content_tokenizer=resolve_graph_content_tokenizer(run),
        tokenizer_trust_remote_code=tokenizer.trust_remote_code,
        tokenizer_revision=tokenizer.revision,
        prompt_corpus=_resolve_graph_corpus(run),
        max_osl=_resolve_graph_max_osl(run),
        num_dataset_entries=_resolve_graph_num_entries(run),
        max_context_length=_resolve_graph_max_context(run),
        idle_gap_cap_seconds=_resolve_graph_idle_gap_cap(run),
        trajectory_start_max_ratio=run.cfg.trajectory_start_max_ratio or 0.0,
        default_model=endpoint_info.primary_model_name,
        run_streaming=endpoint_info.endpoint.streaming,
        delay_cap_seconds=_resolve_delay_cap_seconds(run),
        endpoint_extra=endpoint_info.endpoint.extra,
    )


def parse_graph_workload(
    run: BenchmarkRun, path: Path | str, **adapter_kwargs: object
) -> ParsedGraph:
    """Parse a graph workload into a ``ParsedGraph`` for ``run``, via the registry.

    ONE dispatch for every format: resolve the run's parse knobs into a
    :class:`GraphParseContext` (:func:`resolve_graph_parse_context`) and hand
    it to :func:`~aiperf.dataset.graph.parser.parse_graph`, which routes to the
    selected adapter's uniform ``parse(path, ctx)``. Each adapter maps only the
    ctx fields it consumes (weka: seed / tokenizer / corpus / max_osl /
    idle-gap cap / selection caps; dynamo: the same minus max_osl;
    dag_jsonl: the four dispatch knobs;
    native: none), so a parse is fully determined by the run config + file for
    every format. The DatasetManager is the ONLY production caller; the
    TimingManager ingests the graph_meta sidecar this build writes and
    broadcasts, and never parses.

    ``**adapter_kwargs`` is the build-plane's seam for ADAPTER-SPECIFIC live
    objects the run-derived :class:`GraphParseContext` cannot carry -- currently
    only the dynamo direct route's ``direct_store`` (the ``GraphStoreBuilder``
    constructs the unified store BEFORE the parse and passes it here so
    ``build_trie_ir``'s ``pool.add`` write-throughs intern straight into the
    store). It is forwarded verbatim to :func:`parse_graph`, which fails loud
    (``TypeError`` for an unknown-to-the-adapter kwarg, ``GraphParseError`` for
    any kwarg with ``fmt == "native"``); a caller that passes no adapter kwargs
    is byte-identical to the pre-seam call.

    Adapter failures surface uniformly as
    :class:`~aiperf.dataset.graph.parser.GraphParseError` (the registry seam
    re-wraps adapter ``ValueError`` subclasses, message text preserved), so
    callers need exactly one except class.

    The t*/dynamic-slot gate runs uniformly on the parsed result;
    all-explicit-zero arrival offsets (the shape dag_jsonl lowering stamps and
    guards at its own parse seam) take the carve-out documented in
    :func:`_gate_dynamic_slots_vs_tstar`.
    """
    publish_graph_loader_tokenizer_env(run)
    p = Path(path)
    ref = resolve_graph_workload(run)
    # The ref's format describes the RUN'S OWN dataset input; parsing a
    # divergent path with that format would silently mis-route a future
    # caller, so pin the congruence here.
    if ref is not None and p != ref.path:
        raise ValueError(
            f"parse_graph_workload path {p} diverges from the run's resolved "
            f"graph workload {ref.path}; pass the run's own dataset path"
        )
    fmt = ref.format if ref is not None else None

    from aiperf.dataset.graph.parser import parse_graph

    ctx = resolve_graph_parse_context(run)
    parsed = parse_graph(p, format=fmt, ctx=ctx, **adapter_kwargs)
    _gate_dynamic_slots_vs_tstar(parsed, ctx.trajectory_start_max_ratio)
    return parsed


def _gate_dynamic_slots_vs_tstar(
    parsed: ParsedGraph, trajectory_start_max_ratio: float
) -> None:
    """Reject dynamic slots + an engaged t* snapshot window (unsupported by design).

    The t* chop partitions nodes into warmup/profiled by recorded offsets; a
    slot producer chopped into warmup would leave its consumer's pool value
    undefined. The gate runs on the one build-plane parse seam (the
    TimingManager ingests the sidecar of a build that already passed it). The
    window is off by default; it engages only via
    ``--scenario inferencex-agentx-mvp`` or explicit
    ``--trajectory-start-min/max-ratio`` flags, hence the message names both
    exits (drop the scenario or the flags) -- an explicit
    ``--trajectory-start-max-ratio 0`` would collide with a scenario-applied
    window and raise ``ScenarioLockError`` instead.

    Resolves :func:`graph_carries_assembly_slots` -- this t*-gate is the
    predicate's sole production caller -- so
    there is ONE definition of "carries dynamic slots"; the
    predicate's union of ``assembly`` and ``capture`` is equivalent to an
    assembly-only check at graph level (capture is only ever stamped on a
    producer referenced by some assembly program). Lazy import matches this
    module's local-import style and avoids the graph loader import cycle.
    """
    from aiperf.dataset.graph.segment_ir.store_builder import (
        graph_carries_assembly_slots,
    )

    if not graph_carries_assembly_slots(parsed):
        return
    if trajectory_start_max_ratio <= 0.0:
        return
    # EXPLICIT-ZERO carve-out: skip the t*-rejection iff EVERY node's
    # ``arrival_offset_us`` is explicitly ``0`` -- int zero; ``None`` (the
    # un-stamped default on natively authored nodes) does NOT qualify and
    # keeps gating exactly as before. The skip is load-bearing on an
    # INVARIANT, not on cross-plane determinism: dag_jsonl lowering stamps
    # ``arrival_offset_us=0`` on EVERY node, so such a trace's recorded
    # duration is 0 and any sampled t* = ratio * 0 = 0. At t*=0 the snapshot
    # chop drops nothing (no node's offset is < 0), so the chop -- and the
    # "slot producer chopped into warmup leaves its consumer's pool value
    # undefined" hazard this gate exists to reject -- is a structural no-op;
    # rejecting would ONLY ever false-positive on the scenario-applied
    # ``trajectory_start_max_ratio=1.0`` window. ``DagJsonlGraphAdapter.parse`` guards
    # the all-zero invariant at its own seam
    # (``_assert_dag_zero_arrival_offsets``); if dag ever emits recorded
    # offsets, that guard raises AND this carve-out stops matching, so the
    # gate re-engages by construction. Contract delta (c), intentional: a
    # NATIVE graph AUTHORED with all-zero offsets now passes where it used to
    # reject -- the t*-degeneracy invariant is identical there, so the old
    # rejection was the same false positive.
    if all(
        node.arrival_offset_us == 0
        for record in (parsed.graph, *parsed.graphs.values())
        for node in record.nodes.values()
    ):
        return
    raise ValueError(
        "graph workload carries dynamic content slots (prompt refs to "
        "LlmNode-written channels); the t* snapshot window is not "
        "supported with dynamic slots. The window was engaged by "
        "--scenario inferencex-agentx-mvp or explicit "
        "--trajectory-start-min/max-ratio flags; drop the scenario (or the "
        "flags) to run a full native replay."
    )


def _synthesis_attr(run: BenchmarkRun, name: str, default: Any) -> Any:
    """Read an attribute off the run's default-dataset synthesis config.

    Tolerates an absent default dataset or an absent ``synthesis`` block: both
    yield ``default``. A ``getattr`` default (not ``or``) is returned so a
    present-but-``None`` attribute is preserved by the caller, not coalesced.
    """
    dataset = run.cfg.get_default_dataset()
    synthesis = getattr(dataset, "synthesis", None) if dataset else None
    return getattr(synthesis, name, default) if synthesis else default


def _resolve_graph_max_osl(run: BenchmarkRun) -> int | None:
    """Resolve the ``--synthesis-max-osl`` cap for the run, or ``None``.

    Reads ``synthesis.max_osl`` off the run's default dataset (where
    ``base_trace_loader`` reads it for the linear path). Resolving from the run
    config alone keeps the cap identical for every parse of the run
    (in-process build or spawn-started pool worker). ``None`` (flag unset)
    leaves the recorded ``out`` uncapped.
    """
    return _synthesis_attr(run, "max_osl", None)


def _resolve_graph_num_entries(run: BenchmarkRun) -> int | None:
    """Resolve the explicit ``entries`` cap on the run's default dataset, or ``None``.

    Reads ``entries`` off the run's default dataset, gated on
    ``"entries" in ds.model_fields_set`` so ONLY a user-set value is a cap: a
    dataset class's own default (``FileDataset.entries=None``, or the linear
    ``SyntheticDataset.entries=100``) is not a graph-plane selection ceiling and
    resolves to ``None`` (use all eligible traces). Resolving from the run
    config alone keeps the cap identical for every parse of the run
    (in-process build or spawn-started pool worker) -- and, unlike
    ``DatasetResolver._resolve_one``,
    this seam is never skipped by the weka-HF ``org/name`` or local-graph
    early-returns.
    """
    dataset = run.cfg.get_default_dataset()
    if not dataset or "entries" not in getattr(dataset, "model_fields_set", set()):
        return None
    return getattr(dataset, "entries", None)


def _resolve_graph_max_context(run: BenchmarkRun) -> int | None:
    """Resolve the ``--max-context-length`` per-trace context cap, or ``None``.

    Reads ``synthesis.max_context_length`` off the run's default dataset,
    mirroring :func:`_resolve_graph_max_osl`. Resolving from the run config
    alone keeps the cap identical for every parse of the run. ``None`` (flag
    unset) applies no context-length filter.
    """
    return _synthesis_attr(run, "max_context_length", None)


def _resolve_graph_corpus(run: BenchmarkRun) -> str:
    """Resolve the weka content corpus (`--prompt-corpus`) for the run.

    Reads ``synthesis.corpus`` off the run's default dataset (where
    ``--prompt-corpus`` lands via the dataset converter), defaulting to
    ``"coding"`` -- the corpus the recorded weka workloads were captured against.
    Resolving from the run config alone keeps the corpus identical for every
    parse of the run. ``"sonnet"`` selects the Shakespeare
    pool. The trailing ``or "coding"`` also coalesces a present-but-empty/``None``
    corpus to the default (a getattr default alone would not).
    """
    return _synthesis_attr(run, "corpus", None) or "coding"


# Default per-trace idle-gap cap when no synthesis config is present (60s, the
# value the recorded weka workloads were captured/replayed against).
_GRAPH_IDLE_GAP_CAP_DEFAULT = 60.0


def _resolve_graph_idle_gap_cap(run: BenchmarkRun) -> float | None:
    """Resolve the per-trace idle-gap cap (`--synthesis-idle-gap-cap`) for the run.

    Reads ``synthesis.idle_gap_cap_seconds`` off the run's default dataset
    (default 60.0 when the field is unset, ``None`` when explicitly set to null to
    disable warping). Falls back to :data:`_GRAPH_IDLE_GAP_CAP_DEFAULT` when no
    synthesis config is present. Resolving from the run config alone keeps the
    cap identical for every parse of the run.
    """
    return _synthesis_attr(run, "idle_gap_cap_seconds", _GRAPH_IDLE_GAP_CAP_DEFAULT)
