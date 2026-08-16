# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph-workload detection + parsing seam for the dataset plane.

Graph traces are NOT a ``custom_dataset_loader`` plugin -- they are ingested via
the ``graph_adapter`` plugin registry (``dynamo_trace`` / ...)
into a :class:`~aiperf.dataset.graph.models.ParsedGraph`. Auto-detection
walks that registry and
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
graph workload and to flip the profiling phase's timing mode to ``AGENT_GRAPH``,
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
from typing import TYPE_CHECKING

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.dataset.graph.parse_context import GraphParseContext

_logger = AIPerfLogger(__name__)

if TYPE_CHECKING:
    from aiperf.config.dataset import FileDataset, PublicDataset
    from aiperf.config.dataset.trace import SynthesisConfig
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


# Opt-out hook for registry adapters that must be reachable only via an
# explicit ``--graph-format`` / ``parse_graph`` call. Deliberately EMPTY: every
# adapter currently in the ``graph_adapter`` registry participates in
# auto-detection. Formats that are not auto-detected (``dag_jsonl``) are
# excluded because they are not registry entries at all, not because they
# are listed here.
_AUTODETECT_EXCLUDED: frozenset[str] = frozenset()


def _detect_graph_workload_format(path: Path) -> str | None:
    """Return the graph-adapter format name auto-detected for ``path``, else None.

    Walks the ``graph_adapter`` plugin registry, skipping the explicit-only
    adapters in :data:`_AUTODETECT_EXCLUDED`, and returns the
    highest-``detection_priority`` adapter whose ``can_load`` matches.
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
        # Isolate each adapter's sniff (mirrors ``parser.detect_format``): one
        # crashing adapter must not abort detection for all the others.
        try:
            recognized = cls.can_load(path)
        except Exception as e:  # noqa: BLE001 - isolate third-party adapters
            _logger.debug(
                f"graph adapter {entry.name!r} can_load({path}) raised {e!r}; "
                "treating as does-not-claim"
            )
            continue
        if recognized:
            meta = entry.get_typed_metadata(GraphAdapterMetadata)
            matches.append((meta.detection_priority, entry.name))
    return max(matches)[1] if matches else None


def _default_dataset(run: BenchmarkRun) -> object | None:
    """Return the run's default dataset, or None when it cannot be resolved.

    The two narrowing helpers below both start here, so the swallow of a
    malformed/absent config lives in ONE place.
    """
    try:
        return run.cfg.get_default_dataset()
    except (IndexError, AttributeError):
        return None


def _file_dataset(run: BenchmarkRun) -> FileDataset | None:
    """Narrow the run's default dataset to ``FileDataset``, else None.

    ``graph_format`` and ``path`` are declared ONLY on ``FileDataset``, so
    narrowing once lets callers read them directly instead of probing a
    ``DatasetConfig`` union member that may not carry them. Lazy import matches
    this module's local-import style and avoids the graph loader import cycle.
    """
    from aiperf.config.dataset import as_file_dataset

    return as_file_dataset(_default_dataset(run))


def _trace_replay_dataset(run: BenchmarkRun) -> FileDataset | PublicDataset | None:
    """Narrow the run's default dataset to the members carrying trace-replay knobs.

    ``synthesis``, ``ignore_trace_delays``, ``use_think_time_only``, and
    ``trace_idle_gap_cap_seconds`` are declared on ``FileDataset`` AND
    ``PublicDataset`` but NOT ``SyntheticDataset``. A synthetic (or absent)
    default dataset yields None, which every caller treats as "knob unset".
    """
    from aiperf.config.dataset import as_trace_replay_dataset

    return as_trace_replay_dataset(_default_dataset(run))


def _graph_format_override(run: BenchmarkRun) -> str | None:
    """Return the explicit ``--graph-format`` override for ``run``, or None.

    Reads ``FileDataset.graph_format``. When set, the input is treated as a graph
    workload of exactly this adapter regardless of auto-detection.
    """
    dataset = _file_dataset(run)
    if dataset is None:
        return None
    fmt = dataset.graph_format
    return str(fmt) if fmt is not None else None


def _has_explicit_custom_format(run: BenchmarkRun) -> bool:
    """True when the default file dataset explicitly selects a custom loader.

    ``format`` is None unless the author named a custom loader, so the VALUE
    carries the provenance. It survives the sweep orchestrator's ``model_dump``
    -> subprocess ``model_validate`` round-trip, which ``model_fields_set``
    does not: there every dumped key returns marked "set", so a defaulted
    ``format`` would read as an explicit custom-loader selection.
    """
    dataset = _file_dataset(run)
    if dataset is None:
        return False
    return dataset.format is not None


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

    Reads the default dataset's ``path``. An explicit custom dataset format
    selects the custom-loader path and bypasses graph auto-detection. When
    ``--graph-format`` is set (:func:`_graph_format_override`), the input is
    forced to be a graph workload of that adapter. Otherwise asks the ``graph_adapter``
    registry (via :func:`_detect_graph_workload_format`) whether any trace
    adapter recognizes the file. Returns ``None`` for any non-file dataset
    (synthetic / public / inline), a file no adapter recognizes, AND any
    derivation failure: the swallow consumers used to wrap around detection
    lives HERE now, so callers read the accessor's result bare.
    """
    from aiperf.config.resolution.plan import GraphWorkloadRef

    try:
        dataset = _file_dataset(run)
        if dataset is None:
            return None
        fmt = _graph_format_override(run)
        if fmt is None and _has_explicit_custom_format(run):
            return None
        if dataset.path is None:
            return None
        path = Path(dataset.path)
        if fmt is None:
            fmt = _detect_graph_workload_format(path)
        if fmt is None:
            return None
        return GraphWorkloadRef(path=path, format=fmt)
    except Exception:
        # ``--graph-format`` is the user ASSERTING this is a graph workload:
        # silently degrading to the linear pipeline there would be a wrong run
        # with no diagnostic, so let the failure surface.
        if _graph_format_override(run) is not None:
            raise
        return None


def is_graph_workload_path(path: Path) -> bool:
    """True when `path` is a graph workload recognized by a trace adapter.

    Path-level companion to :func:`resolve_graph_workload` (which takes a
    run). Uses the SAME registry detection (`_detect_graph_workload_format`),
    so callers skip exactly what the graph pipeline will claim.
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
    result AS-IS: an explicit float caps idle gaps at that value, and ``None``
    means no per-trace compression.

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
    dataset = _trace_replay_dataset(run)
    synthesis = dataset.synthesis if dataset is not None else None
    endpoint_info = ModelEndpointInfo.from_run(run)
    return GraphParseContext(
        content_root_seed=resolve_graph_content_seed(run),
        content_tokenizer=resolve_graph_content_tokenizer(run),
        tokenizer_trust_remote_code=tokenizer.trust_remote_code,
        tokenizer_revision=tokenizer.revision,
        prompt_corpus=_resolve_graph_corpus(run),
        max_osl=_resolve_graph_max_osl(run),
        max_isl=synthesis.max_isl if synthesis else None,
        num_dataset_entries=_resolve_graph_num_entries(run),
        max_context_length=_resolve_graph_max_context(run),
        idle_gap_cap_seconds=_resolve_graph_idle_gap_cap(run),
        trajectory_start_max_ratio=_resolve_graph_tstar_max_ratio(run),
        default_model=endpoint_info.primary_model_name,
        run_streaming=endpoint_info.endpoint.streaming,
        delay_cap_seconds=_resolve_delay_cap_seconds(run),
        ignore_trace_delays=bool(dataset is not None and dataset.ignore_trace_delays),
        use_think_time_only=bool(dataset is not None and dataset.use_think_time_only),
        endpoint_extra=endpoint_info.endpoint.extra,
        open_loop_replay=_resolve_open_loop_replay(run),
        execute_tools=_resolve_execute_tools(run),
        use_family_sampling=_resolve_use_family_sampling(run),
        emit_warmup=_resolve_emit_warmup(run),
        replay_only_knobs=_resolve_replay_only_knobs(run),
    )


def parse_graph_workload(
    run: BenchmarkRun, path: Path | str, **adapter_kwargs: object
) -> ParsedGraph:
    """Parse a graph workload into a ``ParsedGraph`` for ``run``, via the registry.

    ONE dispatch for every format: resolve the run's parse knobs into a
    :class:`GraphParseContext` (:func:`resolve_graph_parse_context`) and hand
    it to :func:`~aiperf.dataset.graph.parser.parse_graph`, which routes to the
    selected adapter's uniform ``parse(path, ctx)``. Each adapter maps only the
    ctx fields it consumes (dynamo: seed / tokenizer / corpus / idle-gap cap /
    selection caps), so a parse is fully determined by the run
    config + file for every format. The DatasetManager is the ONLY production caller; the
    TimingManager ingests the graph_meta sidecar this build writes and
    broadcasts, and never parses.

    ``**adapter_kwargs`` is the build-plane's seam for ADAPTER-SPECIFIC live
    objects the run-derived :class:`GraphParseContext` cannot carry -- currently
    only the dynamo adapter's ``direct_store`` write-through sink. NO production
    caller passes it: the ``GraphStoreBuilder`` always calls
    ``parse_graph_workload(run, path)`` with no adapter kwargs, so the seam is a
    supported-but-unwired adapter capability exercised only by tests. It is
    forwarded verbatim to :func:`parse_graph`, which fails loud
    (``TypeError`` for an unknown-to-the-adapter kwarg); a caller that passes
    no adapter kwargs
    is byte-identical to the pre-seam call.

    Adapter failures surface uniformly as
    :class:`~aiperf.dataset.graph.parser.GraphParseError` (the registry seam
    re-wraps adapter ``ValueError`` subclasses, message text preserved), so
    callers need exactly one except class.

    The t*/dynamic-slot gate runs uniformly on the parsed result;
    all-explicit-zero arrival offsets take the carve-out documented in
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
    window is off by default and, on a graph workload, engages ONLY via the
    explicit ``--trajectory-start-min/max-ratio`` flags -- hence the message
    names just those.

    ``--scenario inferencex-agentx-mvp`` also carries window defaults, but it
    cannot reach this gate: the graph resolver returns before stamping
    ``resolved.dataset_types`` for a graph workload
    (``config/dataset/resolver.py``), so ``scenario.validator._detect_loader``
    yields ``None``, which is never in the scenario's ``require_loader`` tuple
    -- and a ``require_loader`` violation is unbypassable even under
    ``--unsafe-override``. The scenario is rejected before any graph parse.

    Resolves :func:`graph_carries_assembly_slots` -- this t*-gate is the
    predicate's sole production caller -- so
    there is ONE definition of "carries dynamic slots"; the
    predicate's union of ``assembly`` and ``capture`` is equivalent to an
    assembly-only check at graph level (capture is only ever stamped on a
    producer referenced by some assembly program). Lazy import matches this
    module's local-import style and avoids the graph loader import cycle.
    """
    from aiperf.dataset.graph.segment_trie.store_builder import (
        graph_carries_assembly_slots,
    )

    if not graph_carries_assembly_slots(parsed):
        return
    if trajectory_start_max_ratio <= 0.0:
        return
    # EXPLICIT-ZERO carve-out: skip the t*-rejection iff EVERY node's
    # ``arrival_offset_us`` is explicitly ``0`` -- int zero; ``None`` (the
    # un-stamped default) does NOT qualify and keeps gating exactly as
    # before. The skip is load-bearing on an INVARIANT, not on cross-plane
    # determinism: a lowering that stamps
    # ``arrival_offset_us=0`` on EVERY node yields a trace whose recorded
    # duration is 0 and any sampled t* = ratio * 0 = 0. At t*=0 the snapshot
    # chop drops nothing (no node's offset is < 0), so the chop -- and the
    # "slot producer chopped into warmup leaves its consumer's pool value
    # undefined" hazard this gate exists to reject -- is a structural no-op;
    # rejecting would ONLY ever false-positive on the scenario-applied
    # ``trajectory_start_max_ratio=1.0`` window. The carve-out keys on the
    # parsed offsets themselves, so a lowering that starts emitting recorded
    # offsets stops matching and the gate re-engages by construction --
    # nothing has to be kept in sync by hand. Contract delta (c),
    # intentional: any graph with all-zero offsets now passes where it used
    # to reject -- the t*-degeneracy invariant is identical there, so the old
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
        "supported with dynamic slots. The window was engaged by explicit "
        "--trajectory-start-min/max-ratio flags; drop them to run a full "
        "recorded replay."
    )


def _synthesis(run: BenchmarkRun) -> SynthesisConfig | None:
    """Return the run's default-dataset synthesis config, or None.

    Tolerates a synthetic/absent default dataset or an absent ``synthesis``
    block: both yield None. Callers read the field they want directly, so a
    present-but-``None`` field stays distinguishable from an absent block.
    """
    dataset = _trace_replay_dataset(run)
    return dataset.synthesis if dataset is not None else None


def _resolve_graph_max_osl(run: BenchmarkRun) -> int | None:
    """Resolve the ``--synthesis-max-osl`` cap for the run, or ``None``.

    Reads ``synthesis.max_osl`` off the run's default dataset (where
    ``base_trace_loader`` reads it for the linear path). Resolving from the run
    config alone keeps the cap identical for every parse of the run
    (in-process build or spawn-started pool worker). ``None`` (flag unset)
    leaves the recorded ``out`` uncapped.
    """
    synthesis = _synthesis(run)
    return synthesis.max_osl if synthesis is not None else None


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
    this seam is never skipped by the HF ``org/name`` or local-graph
    early-returns.
    """
    dataset = run.cfg.get_default_dataset()
    if not dataset or "entries" not in dataset.model_fields_set:
        return None
    return dataset.entries


def _resolve_graph_max_context(run: BenchmarkRun) -> int | None:
    """Resolve the ``--max-context-length`` per-trace context cap, or ``None``.

    Reads ``synthesis.max_context_length`` off the run's default dataset,
    mirroring :func:`_resolve_graph_max_osl`. Resolving from the run config
    alone keeps the cap identical for every parse of the run. ``None`` (flag
    unset) applies no context-length filter.
    """
    synthesis = _synthesis(run)
    return synthesis.max_context_length if synthesis is not None else None


def _resolve_graph_corpus(run: BenchmarkRun) -> str:
    """Resolve the graph content corpus (`--prompt-corpus`) for the run.

    Reads ``synthesis.corpus`` off the run's default dataset (where
    ``--prompt-corpus`` lands via the dataset converter), defaulting to
    ``"coding"`` -- the corpus the recorded agentic workloads were captured against.
    Resolving from the run config alone keeps the corpus identical for every
    parse of the run. ``"sonnet"`` selects the Shakespeare
    pool. The trailing ``or "coding"`` also coalesces a present-but-empty/``None``
    corpus to the default (a getattr default alone would not).
    """
    synthesis = _synthesis(run)
    corpus = synthesis.corpus if synthesis is not None else None
    return corpus or "coding"


def _resolve_graph_idle_gap_cap(run: BenchmarkRun) -> float | None:
    """Resolve the shared per-trace idle-gap cap for the graph run.

    Graph/Dynamo uses the same ``trace_idle_gap_cap_seconds`` field and CLI flag
    as Weka/AgentX, and the same two-state meaning: a float caps idle gaps at
    that value, and unset (``None``) means NO per-trace compression. There is no
    built-in default to substitute -- an earlier one made "unset" and
    "explicitly null" two different states, which then had to be told apart via
    ``model_fields_set``, a distinction that does not survive the sweep
    orchestrator's dump/validate boundary.
    """
    dataset = _trace_replay_dataset(run)
    return dataset.trace_idle_gap_cap_seconds if dataset is not None else None


def _resolve_open_loop_replay(run: BenchmarkRun) -> bool | None:
    """Resolve the run's EFFECTIVE open-loop replay setting, or None if absent.

    Reported as a value, not through :func:`_resolve_replay_only_knobs`: that
    tuple is keyed on value-differs-from-default and ``open_loop_replay``
    defaults to True, so the default-on case -- the one a tool-execution run
    actually hits -- would never appear there.

    ``open_loop_strict`` implies open-loop pacing (it is an open-loop-only
    modifier, enforced by ``FileDataset``'s own validator), so it is folded in
    here rather than left for each consumer to remember.
    """
    dataset = _trace_replay_dataset(run)
    if dataset is None:
        return None
    return bool(
        getattr(dataset, "open_loop_replay", False)
        or getattr(dataset, "open_loop_strict", False)
    )


def _resolve_execute_tools(run: BenchmarkRun) -> bool | None:
    """Resolve the run's ``--graph-execute-tools`` setting, or None if absent.

    ``graph_execute_tools`` is declared on ``FileDataset`` only (like
    ``graph_format``), so a synthetic/public/absent default dataset yields
    ``None`` -- "nothing told the adapter", which leaves the adapter default
    (off) in force. Reported as a value rather than through
    :func:`_resolve_replay_only_knobs` because the adapter needs the SETTING,
    not the name of a flag it cannot honor.
    """
    dataset = _file_dataset(run)
    if dataset is None:
        return None
    return bool(dataset.graph_execute_tools)


def _resolve_use_family_sampling(run: BenchmarkRun) -> bool:
    dataset = _file_dataset(run)
    if dataset is None:
        return True
    return bool(getattr(dataset, "graph_use_family_sampling", True))


def _resolve_emit_warmup(run: BenchmarkRun) -> bool:
    dataset = _file_dataset(run)
    if dataset is None:
        return False
    return bool(getattr(dataset, "graph_emit_warmup", False))


def _resolve_replay_only_knobs(run: BenchmarkRun) -> tuple[str, ...]:
    """Name the flags this run set that only the linear replay loaders consume.

    Each entry is a CLI flag whose backing field is declared on ``FileDataset``
    (or the profiling phase) and read by ``baseten_trace`` / AGENTIC_REPLAY, but
    by nothing on the graph path -- verified by there being no reader under
    ``dataset/graph`` or ``timing`` outside ``agentic_replay``. Reported by
    NAME so a refusing adapter can quote the flag the operator typed.

    Keyed on value-differs-from-default, never presence: ``--force-min-tokens``
    defaults to ``True``, so a presence test would flag every run, and
    ``model_fields_set`` does not survive the sweep orchestrator's
    dump/validate boundary.
    """
    found: list[str] = []
    dataset = _trace_replay_dataset(run)
    if dataset is not None:
        for field, flag, default in (
            ("trace_session_sample_ratio", "--trace-session-sample-ratio", None),
            ("max_idle_gap_cap_seconds", "--max-idle-gap-cap-seconds", None),
            ("omit_kv_hints", "--omit-kv-hints", False),
            ("force_min_tokens", "--no-force-min-tokens", True),
        ):
            if getattr(dataset, field, default) != default:
                found.append(flag)
    # Same first-profiling-phase read as _resolve_graph_tstar_max_ratio, so both
    # gates see the phase the dispatch plane actually runs.
    phases = run.cfg.get_profiling_phases()
    if phases and phases[0].system_idle_gap_cap_seconds is not None:
        found.append("--system-idle-gap-cap-seconds")
    return tuple(found)


def _resolve_graph_tstar_max_ratio(run: BenchmarkRun) -> float:
    """Resolve the t* snapshot-window ceiling (`--trajectory-start-max-ratio`).

    On this branch the ratio lives PER PROFILING PHASE
    (``CreditPhaseConfig.trajectory_start_max_ratio``), not on
    ``BenchmarkConfig``, so read the first profiling phase -- the SAME phase
    ``TimingConfig`` reads for its ``graph_fields``, through the SAME resolver
    (:func:`aiperf.config.phases.resolve_graph_tstar_window`), so the build-side
    parse gate (:func:`_gate_dynamic_slots_vs_tstar`) and the dispatch-side
    strategy agree on whether the window is engaged. Both resolve an unset
    (``None``) ratio to ``0.0``, leaving the window OFF and this gate a no-op;
    AGENTIC_REPLAY resolves the same unset state to the full trace instead, so
    reading the raw field here would arm the gate while dispatch replays cold.
    """
    from aiperf.config.phases import resolve_graph_tstar_window

    phases = run.cfg.get_profiling_phases()
    if not phases:
        return 0.0
    return resolve_graph_tstar_window(phases[0])[1]
