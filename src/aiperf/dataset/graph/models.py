# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph schema models - canonical frozen msgspec structs for graph/trace records.

These are the canonical typed graph model types. Trusted (typed) construction goes through the
struct constructors / `codecs.py`; loose / foreign dict input is
normalized to these types by the adapters. Every producer (currently the
dynamo trace adapter) emits flat `LlmNode` + `StaticEdge` graphs.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

import msgspec

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.dataset.graph.segment_trie.pool import SegmentPool

_logger = AIPerfLogger(__name__)

# Reserved graph sentinels: ``START`` is the virtual entry (a valid edge source
# only), ``END`` the virtual exit (a valid edge target only); neither is ever a
# real declared node id.
START_NODE_ID = "START"
END_NODE_ID = "END"


class ChannelType(str, Enum):
    TEXT = "text"
    MESSAGES = "messages"


class ReducerName(str, Enum):
    OVERWRITE = "overwrite"
    ADD_MESSAGES = "add_messages"


class ChannelSpec(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """A state-channel declaration."""

    type: ChannelType = ChannelType.TEXT  # Channel value type.
    # How concurrent writes to the channel are merged.
    reducer: ReducerName = ReducerName.OVERWRITE


class ChannelRequirement(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """AND-fan-in input requirement on a node.

    A node fires only when every requirement in its `inputs` list is satisfied.
    `count: int` requires exactly that many writes to the channel; `count: "all"`
    means "all statically declared producers" (computed at Scheduler init).

    Not ``omit_defaults`` so ``count`` always round-trips in the serialized
    requirement shape (``{"channel": str, "count": int|"all"}``).
    """

    channel: str  # Input channel name.
    # Required arrival count. 'all' resolves to the static topology count of
    # declared producers.
    count: Annotated[int, msgspec.Meta(ge=1)] | Literal["all"] = 1


class StaticEdge(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="edge_type",
    tag="static",
    forbid_unknown_fields=True,
):
    """Unconditional edge: `{source: <id>, target: <id>}`."""

    source: str  # Source node id (or 'START').
    target: str  # Destination node id (or 'END').
    # Idle / scheduling delay after the predecessor finishes, in microseconds.
    # Edge delay represents wall-clock idle time between the predecessor's
    # completion and the successor becoming ready. Not used for node execution
    # time. None means no extra edge delay.
    delay_after_predecessor_us: Annotated[float, msgspec.Meta(ge=0)] | None = None
    # Minimum wait time on the successor after all predecessors are satisfied,
    # in microseconds. Stacks with `delay_after_predecessor_us`; the runtime
    # takes the later of the two when both are set. None means no minimum.
    min_start_delay_us: Annotated[float, msgspec.Meta(ge=0)] | None = None
    # Idle / scheduling delay after the predecessor DISPATCHES (its firing
    # gate clears and it proceeds to its endpoint call), in microseconds. The
    # successor does NOT wait for the predecessor's completion: the runtime
    # schedules it at predecessor dispatch and gates it at dispatch + delay.
    # Mutually exclusive with `delay_after_predecessor_us`.
    # None means this edge is not start-anchored.
    delay_after_predecessor_start_us: Annotated[float, msgspec.Meta(ge=0)] | None = None
    # Idle delay after the predecessor's OBSERVED FIRST TOKEN, in microseconds.
    # Refines a start-anchored edge for children recorded post-TTFT: when the
    # runtime observes the predecessor's first token it gates the successor at
    # first_token_wall + this delay; when the predecessor terminates without one
    # the gate falls back to dispatch + delay_after_predecessor_start_us. Only
    # valid alongside delay_after_predecessor_start_us.
    delay_after_predecessor_first_token_us: (
        Annotated[float, msgspec.Meta(ge=0)] | None
    ) = None


class NodeKind(str, Enum):
    LLM = "llm"
    TOOL = "tool"


class ExpectedTokens(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Engine-validation predictions."""

    # Predicted input tokens served from cache (maps to OTel
    # `gen_ai.usage.cache_read.input_tokens`).
    cache_read_tokens: int | None = None
    # Predicted input tokens written to cache on this request (maps to OTel
    # `gen_ai.usage.cache_creation.input_tokens`).
    cache_creation_tokens: int | None = None
    input_tokens: int | None = None  # Predicted total input tokens.
    output_tokens: int | None = None  # Predicted output tokens.


class _BaseNode(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=True,
):
    """Common fields for graph nodes."""

    # Free-form node metadata carried through lowering; not exported to spans
    # or records. Reserved keys: `trie` (segment_trie envelope) and `dispatch`
    # (store_builder overrides).
    metadata: dict[str, Any] = {}
    # Minimum wait time (in microseconds) after all predecessors are satisfied,
    # before this node may start. The runtime takes the max of this and any
    # incoming edge-level
    # `min_start_delay_us` / `delay_after_predecessor_us` when computing the
    # firing gate (see `TraceExecutor._apply_firing_delay`). None means no
    # node-level minimum (edge-level gates still apply).
    min_start_delay_us: Annotated[float, msgspec.Meta(ge=0)] | None = None
    # Recorded wall-clock offset of this node from the trace's arrival, in
    # microseconds. Populated by adapters that carry per-node timestamps; used
    # by snapshot-at-t* replay to partition nodes into warmup vs
    # profiled.
    arrival_offset_us: Annotated[int, msgspec.Meta(ge=0)] | None = None
    # Absolute source timestamp for timestamp-driven replay. Adapters that do
    # not carry a wall-clock request start leave this unset.
    recorded_start_unix_ms: Annotated[int, msgspec.Meta(ge=0)] | None = None
    # AND-fan-in input requirements. Empty list = OR-fan-in successor-walk
    # fireability (default). Non-empty list = node fires only when every
    # requirement is satisfied.
    inputs: list[ChannelRequirement] = []

    @property
    def node_type(self) -> NodeKind:
        # msgspec stores the tag in __struct_config__, not as an attribute.
        return NodeKind(type(self).__struct_config__.tag)

    @property
    def write_channels(self) -> list[str]:
        """Channels this node writes to its parent graph.

        Each concrete node kind overrides this. Authoritative for any consumer
        that needs to know what a node produces (visualizers,
        explain/export, etc.) -- instead of pattern-matching on `output` at
        every callsite.
        """
        return []


class LlmNode(
    _BaseNode,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=True,
    tag_field="node_type",
    tag=NodeKind.LLM.value,
):
    """LLM node."""

    # Prompt grammar items. Empty on the trie route (content lives in the
    # SegmentPool, reached via `metadata["trie"]["prompt_segment_ids"]`).
    prompt: list[Any]
    output: str  # Channel name to capture the model response into.
    streaming: bool = True  # Whether to dispatch streaming.
    # Model name dispatched for this call, named like `Turn.model` (the
    # recorded model for replay adapters). Folded into the wire body by the
    # envelope builder (`store_builder._trie_envelope`); None falls through to
    # the run's --model at materialize.
    model: str | None = None
    # Generation cap for this call, named/typed like `Turn.max_tokens`. The
    # worker maps it to the endpoint's wire token field (max_completion_tokens,
    # or max_tokens under --use-legacy-max-tokens); the envelope builder folds
    # it into the wire body (`store_builder._trie_envelope`). A hand-authored
    # positional `extra_body` entry wins over the fold. None leaves generation
    # uncapped.
    max_tokens: Annotated[int, msgspec.Meta(ge=1)] | None = None
    # OpenAI-compatible tool definitions for this call, named like
    # `Turn.raw_tools`. Folded into the wire body as `tools` by the envelope
    # builder. dag_jsonl resolves lineage inheritance (the one per-turn field
    # that walks history) BEFORE stamping, so the field always carries the
    # effective definitions for this node.
    raw_tools: list[dict[str, Any]] | None = None
    # Per-call HTTP headers, named like `Turn.extra_headers`. Attached to the
    # request HEADERS by the worker, never the body (dynamo stamps its
    # x-dynamo-* session-identity headers here; the worker uniquifies them per
    # replay instance).
    extra_headers: dict[str, str] | None = None
    expected: ExpectedTokens | None = None  # Engine-prediction comparators.
    # Non-native per-call request-body fields (temperature, top_p, seed,
    # vendor tunables), named like `Turn.extra_body` and merged into the top
    # level of the dispatched body. Model / stream / token cap / tools ride
    # the native fields above, not here. None falls through to global CLI
    # defaults.
    extra_body: dict[str, Any] | None = None
    # Leading hash-id blocks that would hit an infinite per-trace prefix
    # cache, named like `Turn.theoretical_prefix_cache_hit_blocks`. Stamped by
    # the shared trie build (`segment_trie.prefix_cache`); None when the node's
    # request carries no hash blocks.
    theoretical_prefix_cache_hit_blocks: Annotated[int, msgspec.Meta(ge=0)] | None = (
        None
    )
    # Total hash-id blocks considered for the theoretical prefix-cache count,
    # named like `Turn.theoretical_prefix_cache_total_blocks`. Pairs with
    # `theoretical_prefix_cache_hit_blocks`.
    theoretical_prefix_cache_total_blocks: Annotated[int, msgspec.Meta(ge=0)] | None = (
        None
    )

    @property
    def write_channels(self) -> list[str]:
        return [self.output]


class ToolNode(
    _BaseNode,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=True,
    tag_field="node_type",
    tag=NodeKind.TOOL.value,
):
    """One recorded agent tool step, executed for real in the trace's sandbox.

    Carries the recorded commands rather than reading them from a producer's
    reply: Mode B prompts are predetermined, so the trajectory is known at parse
    time. A live-response-driven variant would instead declare an `inputs`
    requirement on the producing node's output channel.
    """

    # Recorded shell commands for this step, executed in order. A single
    # recorded step may batch several tool calls; they run sequentially, as the
    # capture did.
    commands: list[str]
    output: str  # Channel name the observation is written to.
    # Per-command wall-clock ceiling. None inherits the sandbox default.
    timeout_s: Annotated[float, msgspec.Meta(gt=0)] | None = None

    @property
    def write_channels(self) -> list[str]:
        return [self.output]


class ProvenanceSpec(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Where this benchmark file came from."""

    source: str = "hand-authored"  # Origin tag.
    tool: str = "manual"  # Generating tool plus version, or 'manual'.
    captured_at: str | None = None  # ISO-8601 capture timestamp.
    notes: str | None = None  # Free-form notes.
    # Catch-all for adapter-specific provenance keys; folded in at the codec
    # boundary like the other catch-alls.
    extra: dict[str, Any] = {}


NodeUnion = LlmNode | ToolNode


class ToolSandboxSpec(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Per-trace sandbox configuration for tool execution.

    Carried on :class:`TraceRecord` so each trace can declare the environment
    its recorded commands expect.  All fields are optional and fall back to
    the run-level ``--graph-tool-image`` when unset.
    """

    container: str | None = None  # Docker image reference for this trace's sandbox.
    cwd: str | None = None  # Working directory used for each tool command.
    interpreter: tuple[str, ...] | None = None  # Shell argv used to run each command.


class GraphRecord(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=True,
):
    """Topology record."""

    version: str = "2.0"  # Schema major version.
    provenance: ProvenanceSpec = msgspec.field(
        default_factory=ProvenanceSpec
    )  # Origin metadata.
    system: str | None = None  # Optional system prompt for linear chat.
    state: dict[str, ChannelSpec] = {}  # Explicit channel declarations.
    nodes: dict[str, NodeUnion] = {}  # Node id to node spec.
    edges: list[StaticEdge] = []  # Edge list.


class TraceRecord(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=True,
):
    """Per-trace data."""

    id: str  # Stable trace identifier.
    # Opaque provenance labels. Round-tripped through the codec/sidecar; must
    # not carry routing semantics -- they are authorable surface for external
    # tooling. The one legitimate runtime producer is VIRTUAL_HASH_FALLBACK_TAG
    # (dynamo trie lowering), which marks traces whose prompt hash fell back to
    # a virtual segment rather than a real one.
    tags: list[str] = []
    # Selects this trace's top-level graph from `ParsedGraph.graphs`. None (the
    # default) means the trace runs against the single `ParsedGraph.graph`. A
    # non-None value names a key in `ParsedGraph.graphs`. Resolution always goes
    # through `resolve_trace_graph`.
    graph_ref: str | None = None
    # Linear-chat shorthand: equivalent to init.messages.
    messages: list[dict[str, Any]] | None = None
    # Initial channel values at t=0.
    initial_state: dict[str, Any] = {}
    # Node id to {channel: value}. Authorable surface for hand-written graph
    # files (not included in this release); adapters never populate it, and
    # the structural sidecar strip clears it.
    replay_outputs: dict[str, dict[str, Any]] = {}
    # Per-trace sandbox override. When set, the trace's tool nodes run inside
    # the declared container rather than the run-level --graph-tool-image.
    # Adapters that record per-task Docker images (e.g. SWEBench) populate
    # this so each trace runs in its task-specific environment without
    # requiring a global flag. Falls back to the global when None.
    tool_sandbox: ToolSandboxSpec | None = None


class ParsedGraph(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """In-memory representation of a parsed graph workload file. Not yet auto-derived."""

    # Default / single-workload topology (defaults if absent in file). For
    # single-graph workloads this is THE graph every trace runs against and
    # `graphs` is empty. For multi-graph workloads it is the first source's graph
    # as the default slot; per-trace resolution goes through
    # `graphs[trace.graph_ref]` via `resolve_trace_graph`.
    graph: GraphRecord = msgspec.field(default_factory=GraphRecord)
    # Per-trace top-level graphs for multi-graph workloads, keyed by the key each
    # `TraceRecord.graph_ref` names (the trace id, for per-trace lowering).
    # Empty for single-graph workloads.
    # Resolve a trace's graph with `resolve_trace_graph`.
    graphs: dict[str, GraphRecord] = {}
    traces: list[TraceRecord] = []  # Profiled trace records, in file order.
    # Corpus-supplied WARMUP traces: complete, dispatchable graphs that run
    # under CreditPhase.WARMUP before the profiling phase to prime the server
    # KV cache (e.g. Agent Trace Replay's per-recording "Reply with exactly: ok" call).
    # DISJOINT from `traces` (never profiled, never counted toward the session
    # budget) but resolved through the same `graphs` map and materialized from
    # the same `segment_pool`. Empty for every corpus that emits no warmup;
    # under omit_defaults=True this keeps the msgpack encoding of every
    # pre-existing corpus byte-identical.
    warmup_traces: list[TraceRecord] = []
    # The content-addressed ``segment_trie.pool.SegmentPool`` whose entries every
    # ``LlmNode``'s ``metadata["trie"]["prompt_segment_ids"]`` path indexes.
    # Set by every live producer (the dynamo trie build); the build plane
    # drains it into a
    # ``GraphSegmentUnifiedBackingStore`` so the worker can materialize each
    # node's prompt. Typed ``SegmentPool | None`` (msgspec handles the plain
    # dataclass) so the msgpack codec round-trips the real pool across worker
    # processes instead of decoding it to a bare ``dict``.
    segment_pool: SegmentPool | None = None

    @property
    def all_traces(self) -> list[TraceRecord]:
        """Every trace whose topology and segments must exist in the built store.

        ``traces`` alone is the PROFILED corpus; warmup traces dispatch real
        requests and need catalog ordinals, node manifests, and interned segments
        exactly like a profiled trace. Returns ``traces`` itself (identity, no
        copy) in the overwhelmingly common warmup-free case so the hot build
        path allocates nothing.
        """
        return self.traces + self.warmup_traces if self.warmup_traces else self.traces


def graph_recorded_start_ms(graph: GraphRecord) -> int | None:
    """Earliest preserved source timestamp in one graph, or ``None`` if unstamped.

    The single definition of "when was this trace recorded". ``recorded_start_unix_ms``
    is stamped per node by the trie lowering, so a graph's start is its
    earliest node. ``None`` means the graph carries no timestamps at all --
    a hand-authored corpus paced purely by its authored edge delays.
    """
    starts = [
        int(node.recorded_start_unix_ms)
        for node in graph.nodes.values()
        if node.recorded_start_unix_ms is not None
    ]
    return min(starts) if starts else None


def trace_recorded_start_ms(parsed: ParsedGraph, trace: TraceRecord) -> int | None:
    """Earliest preserved source timestamp for one trace's OWN resolved graph."""
    return graph_recorded_start_ms(resolve_trace_graph(parsed, trace))


def resolve_trace_graph(parsed: ParsedGraph, trace: TraceRecord) -> GraphRecord:
    """Return the top-level :class:`GraphRecord` a trace runs against.

    For single-graph workloads (``trace.graph_ref is None``) this is the
    shared ``parsed.graph`` -- the byte-unchanged single-file path. For multi-graph workloads (per-trace lowering)
    ``trace.graph_ref`` names a key in ``parsed.graphs`` so each trace resolves
    to its own distinct topology. Falls back to ``parsed.graph`` when the ref
    is set but missing from ``parsed.graphs`` (defensive: tolerate a single bad
    ref at runtime rather than crash the whole workload), warning so the
    substitution is not silent -- the wrong topology mis-attributes every
    per-trace metric derived from it.
    """
    ref = trace.graph_ref
    if ref is None:
        return parsed.graph
    graph = parsed.graphs.get(ref)
    if graph is None:
        _logger.warning(
            lambda: f"trace graph_ref {ref!r} is not in parsed.graphs "
            f"({len(parsed.graphs)} entries); falling back to the default graph"
        )
        return parsed.graph
    return graph
