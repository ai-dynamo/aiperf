# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Empirical-proof tool: validate a trie ``--export-level raw`` run vs the REAL trace.

This is the Task-7 acceptance gate for the weka segment-trie IR rewrite. It does
NOT touch the live run path; it reads the deterministic offline artifacts a
profiling run produced (``profile_export_raw.jsonl``) plus the ORIGINAL recorded
Weka trace file, rebuilds the trie graph + segment pool from that trace exactly
as the build plane does, and proves three independent fidelity criteria:

1. :func:`content_byte_exact_vs_v04` (criterion 1) -- per matching trace, the
   ours-vs-v0.4 raw payloads are byte-identical after stripping the
   ``[rid:<12hex>]`` first-user cache-bust marker.
2. :func:`content_vs_real_trace` (criterion 2) -- each exported profiling record's
   reconstructed prompt ``messages`` equal what the recorded trace prescribes for
   that request (the trie node's root->tip segment path, recomputed from the trace
   via ``build_trie_graph`` + ``SegmentPool.materialize``).
3. :func:`causality_timing_vs_real_trace` (criterion 3 -- THE REQUIRED PROOF) --
   from the raw export ALONE, reconstruct each dispatched request's causal
   predecessor (the trie ``StaticEdge`` dependency, recovered from the per-record
   node id + the rebuilt trie graph) and its relative dispatch offset, and assert
   (a) the causal ORDER matches the recorded trace's causal order (a request never
   dispatched before its recorded predecessor) and (b) the relative inter-request
   offsets match the recorded ``t`` / ``api_time`` / ``think_time`` structure
   within a tolerance.

Node-identity recovery (the linchpin): the raw export carries no ``node_id``
field, but the worker folds the node id into ``x_request_id`` as the
4th ``|``-delimited field --
``{conversation_id}|{instance_hash}|{trace_id}|{node_id}|{phase_variant}`` (see
``aiperf.graph.credit_dispatch_adapter._mint``). We split it back out; the
``conversation_id`` base (the part before ``#``) selects the trace.

Run offline against any pair of artifacts; the live run that produces them is
driven by ``tests/component_integration/graph/test_weka_trace_fidelity.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

# --- report ---------------------------------------------------------------


@dataclass
class Mismatch:
    """One failing comparison: which record/node and why."""

    where: str
    detail: str


@dataclass
class Report:
    """Structured pass/fail result of one fidelity check.

    ``checked`` counts every comparison ATTEMPTED (passes and failures alike);
    ``passes`` counts the subset that produced zero mismatches. A single checked
    record can append more than one mismatch (e.g. a causal-order violation AND a
    relative-timing violation), so pass counts must be read from ``passes``, never
    derived as ``checked - len(mismatches)``.

    The timing criterion additionally splits its two distinct assertion kinds:
    ``order_checks`` / ``order_failures`` count per-edge causal-ORDER comparisons
    (a node never dispatched before a dispatched recorded predecessor), while
    ``timing_checks`` / ``timing_failures`` count relative-TIMING gate comparisons
    (observed dispatch vs the warped expected gate). ``exact_edges`` /
    ``idle_capped_edges`` further break the passing timing gates into those whose
    warped expected delay equals the raw recorded gap (``exact`` -- the export
    reproduces the REAL recorded think-time exactly) and those whose recorded gap
    exceeded the idle-gap cap and was warped down to it (``idle_capped`` --
    bounded by the documented faithful cap). Their sum need not equal ``checked``:
    anchors and predecessor-less roots are checked but carry no inter-request
    edge to classify, and a fan-in gate won by a delay-0.0 AND-join edge carries
    no think-time to classify.
    """

    name: str
    checked: int = 0
    passes: int = 0
    order_checks: int = 0
    order_failures: int = 0
    timing_checks: int = 0
    timing_failures: int = 0
    exact_edges: int = 0
    idle_capped_edges: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches and self.checked > 0

    def fail(self, where: str, detail: str) -> None:
        self.mismatches.append(Mismatch(where=where, detail=detail))

    def render(self) -> str:
        head = (
            f"{self.name}: {'PASS' if self.passed else 'FAIL'} "
            f"(checked={self.checked}, passed={self.passes}, "
            f"mismatches={len(self.mismatches)})"
        )
        lines = [head]
        if self.order_checks or self.timing_checks:
            lines.append(
                f"  causal-order: {self.order_checks - self.order_failures}/"
                f"{self.order_checks}  relative-timing: "
                f"{self.timing_checks - self.timing_failures}/{self.timing_checks}"
            )
        if self.exact_edges or self.idle_capped_edges:
            lines.append(
                f"  edges: exact={self.exact_edges} "
                f"idle_capped={self.idle_capped_edges}"
            )
        lines.extend(f"  note: {n}" for n in self.notes)
        for m in self.mismatches:
            lines.append(f"  MISMATCH @ {m.where}: {m.detail}")
        return "\n".join(lines)


# --- raw export record ----------------------------------------------------

# ``[rid:<12hex>]\n\n`` per ``timing/strategies/cache_bust.py`` (FIRST_TURN_PREFIX).
_RID_MARKER_RE = re.compile(r"^\[rid:[0-9a-f]{12}\]\n\n")

_PROFILING = "profiling"


@dataclass
class _ExportRecord:
    """The fidelity-relevant projection of one raw-export JSONL line.

    ``start_perf_ns`` (``perf_counter`` request start) and
    ``first_response_perf_ns`` (the first response's ``perf_ns``, same clock)
    recover a streaming request's OBSERVED time-to-first-token as their DIFFERENCE
    -- a duration valid to add to that record's wall-clock ``request_start_ns``. A
    NON-streaming record still carries one ``TextResponse`` whose ``perf_ns`` is
    the COMPLETION time, so its ``observed_ttft_ns`` is the FULL request duration,
    NOT ``None``; the pair is ``None`` only when a response is truly absent (an
    errored request that streamed nothing, or a zero-latency-style export with no
    responses). That duration is CONSULTED only for post-TTFT anchor parents --
    which stream by construction -- so a non-streaming record's full-duration
    value never actually anchors; the timing proof falls back to the dispatch
    anchor whenever the first token is unrecoverable.
    """

    conversation_id: str
    node_id: str | None
    phase: str
    request_start_ns: int | None
    credit_issued_ns: int | None
    messages: list[dict[str, str]]
    start_perf_ns: int | None = None
    first_response_perf_ns: int | None = None

    @property
    def trace_base(self) -> str:
        """The recorded trace id: the conversation_id stripped of its ``#inst`` tail."""
        return self.conversation_id.split("#", 1)[0]

    @property
    def observed_ttft_ns(self) -> int | None:
        """Observed request-start-to-first-token DURATION ns, or ``None``.

        ``first_response_perf_ns - start_perf_ns`` (both ``perf_counter``): a
        duration valid to add to the wall-clock ``request_start_ns``. ``None`` when
        either perf timestamp is missing or the difference is negative (a clock
        anomaly we refuse to trust rather than silently mis-anchor).
        """
        if self.start_perf_ns is None or self.first_response_perf_ns is None:
            return None
        dur = self.first_response_perf_ns - self.start_perf_ns
        return dur if dur >= 0 else None


def _node_id_from_request_id(x_request_id: str | None) -> str | None:
    """Recover the dispatched node id from the export's ``x_request_id``.

    Worker-minted graph shape: ``{node_id}::{uuid4().hex}`` where the node id
    is the legacy-shaped ``{scope}:{turn}`` coordinate. Returns ``None`` for
    any id without the ``::`` nonce separator (non-graph records).
    """
    if not x_request_id or "::" not in x_request_id:
        return None
    return x_request_id.rsplit("::", 1)[0]


def _norm_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Project wire messages to the comparable ``{role, content}`` shape."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def load_raw_export(raw_jsonl: Path) -> list[_ExportRecord]:
    """Parse a ``profile_export_raw.jsonl`` into fidelity projections.

    Each line is one dispatched request. ``metadata`` carries
    ``conversation_id`` / ``x_request_id`` / ``benchmark_phase`` /
    ``request_start_ns`` / ``credit_issued_ns``; top-level ``start_perf_ns`` +
    ``responses[*].perf_ns`` recover the observed first token (see
    :attr:`_ExportRecord.observed_ttft_ns`); ``payload.messages`` is the wire
    prompt. Blank lines are skipped.
    """
    records: list[_ExportRecord] = []
    for line in Path(raw_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = orjson.loads(line)
        meta = obj.get("metadata", {}) or {}
        payload = obj.get("payload", {}) or {}
        responses = obj.get("responses") or []
        first_response_perf_ns = (
            responses[0].get("perf_ns")
            if responses and isinstance(responses[0], dict)
            else None
        )
        records.append(
            _ExportRecord(
                conversation_id=str(meta.get("conversation_id") or ""),
                node_id=_node_id_from_request_id(meta.get("x_request_id")),
                phase=str(meta.get("benchmark_phase") or ""),
                request_start_ns=meta.get("request_start_ns"),
                credit_issued_ns=meta.get("credit_issued_ns"),
                messages=_norm_messages(payload.get("messages", []) or []),
                start_perf_ns=obj.get("start_perf_ns"),
                first_response_perf_ns=first_response_perf_ns,
            )
        )
    return records


# --- recorded-trace model (rebuilt trie + recorded timing) ----------------


@dataclass
class _RecordedNode:
    """The recorded facts for one trie node: prompt + timing + causal predecessor.

    Timing carries BOTH clocks: ``start_s`` / ``end_s`` are on the (possibly
    idle-gap-warped) clock the rebuilt trie placed the node on -- the SAME clock
    the capped RUN dispatched against -- while ``raw_start_s`` / ``raw_end_s`` are
    the unwarped recorded ``request.t`` / ``t + api_time``. ``pred_delay_us`` maps
    each recorded predecessor (trie ``StaticEdge`` source) to that edge's warped
    ``delay_after_predecessor_us`` -- the EXPECTED end-to-start delay the proof
    compares against. Comparing the warped expected against the raw recorded gap
    classifies each edge as exact (gap <= cap) or idle-capped (gap > cap).
    """

    node_id: str
    messages: list[dict[str, str]]
    start_s: float  # warped request start (== raw request.t when uncapped)
    end_s: float  # warped completion (warped start + raw api_time)
    raw_start_s: float  # unwarped recorded request.t
    raw_end_s: float  # unwarped recorded t + api_time
    predecessors: list[str]  # trie StaticEdge sources (excluding START)
    pred_delay_us: dict[str, float]  # warped StaticEdge delay per predecessor
    rooted_at_start: bool  # True iff its only recorded edge is from START
    # The ``StaticEdge(source="START").min_start_delay_us`` the trie builder
    # stamped on this node -- its ABSOLUTE warped arrival offset from the instance
    # run-origin. The executor fires a START-rooted node at
    # ``anchor_wall + min_start_delay_us`` (``_compute_firing_gate_us`` with
    # ``absolute_start_offsets=True``), so the proof anchors START-roots to this
    # offset rather than to the anchor's recorded END. ``None`` for a node with a
    # non-START predecessor edge.
    min_start_delay_us: float | None = None
    # (parent node id, warped start-to-start delay us D, first-token delay us D' |
    # None) when this node's ONLY recorded edge is start-anchored (mid-flight spawn
    # / chain overlap). The timing proof compares OBSERVED child_dispatch against
    # the parent's dispatch + D, and the order check requires parent dispatch
    # BEFORE child. When D' is not None the node was recorded POST-TTFT: the proof
    # prefers ``parent_first_token + D'`` whenever the parent's observed first token
    # is recoverable from the export, falling back to dispatch + D (LOUDLY) when it
    # is not. D' is None for a pre-TTFT child or a non-streaming parent.
    start_anchor: tuple[str, float, float | None] | None = None


@dataclass
class _RecordedTrace:
    """Every trie node of one recorded trace, keyed by node id."""

    trace_id: str
    nodes: dict[str, _RecordedNode]


def build_recorded_trace(
    trace_file: Path,
    idle_gap_cap_seconds: float | None = 60.0,
    *,
    tokenizer_name: str = "builtin",
    prompt_corpus: str = "coding",
    root_seed: int | None = None,
) -> _RecordedTrace:
    """Rebuild the trie graph + segment pool from a recorded Weka trace file.

    Reproduces the build-plane realization (``build_trie_graph``) so each node's
    EXPECTED prompt is ``pool.materialize(prompt_segment_ids)`` and its recorded
    causal predecessors are the trie ``StaticEdge`` sources into it.

    ``tokenizer_name`` / ``prompt_corpus`` / ``root_seed`` are the content knobs
    the run under proof was built with; the defaults mirror a bare live run
    (builtin tokenizer, ``"coding"`` corpus, no ``--random-seed``). A run built
    with different knobs (e.g. ``--tokenizer gpt2`` or an explicit seed) must
    pass the same values here or every content comparison spuriously fails.

    ``idle_gap_cap_seconds`` is passed straight through to ``build_trie_graph`` so
    the rebuilt trie's ``StaticEdge.delay_after_predecessor_us`` and the node
    arrival offsets sit on the SAME idle-gap-warped clock a capped RUN dispatched
    against (60.0s default == the ``inferencex-agentx-mvp`` scenario cap). The
    expected end-to-start delay the timing proof compares is that WARPED edge
    delay, captured per predecessor in ``_RecordedNode.pred_delay_us``. The RAW
    recorded ``t`` / ``api_time`` is kept alongside (``raw_start_s`` /
    ``raw_end_s``) so the proof can classify each edge exact-vs-idle-capped and
    report it transparently. Pass ``None`` to disable the warp (raw timeline).

    Imports are local so the (heavy) aiperf graph stack stays off the import path
    for callers that only need the pure :class:`Report` helpers.
    """
    from aiperf.dataset.graph.adapters.weka.trace_models import WekaTrace
    from aiperf.dataset.graph.adapters.weka.trie_build import (
        _flatten_requests,
        build_trie_graph,
    )
    from aiperf.dataset.graph.models import LlmNode, StaticEdge

    raw = orjson.loads(Path(trace_file).read_bytes())
    trace = WekaTrace.model_validate(raw)
    parsed, pool = build_trie_graph(
        trace,
        tokenizer_name=tokenizer_name,
        prompt_corpus=prompt_corpus,
        root_seed=root_seed,
        idle_gap_cap_seconds=idle_gap_cap_seconds,
    )
    graph = parsed.graph

    # RAW (unwarped) per-node timing straight from the recorded request.t /
    # api_time. ``_flatten_requests`` here does NOT apply the warp (warped_start
    # stays at its 0.0 default), so we read the raw request fields directly rather
    # than the warped ``.start`` / ``.end`` properties.
    flat = _flatten_requests(trace.requests, root_scope=trace.id)
    raw_timing: dict[str, tuple[float, float]] = {
        n.node_id: (n.request.t, n.request.t + (n.request.api_time or 0.0))
        for n in flat
    }

    # WARPED per-node start from the rebuilt trie's arrival offset (the clock the
    # capped run used). Warped end = warped start + RAW api_time (api_time is the
    # node's own processing, not an inter-request idle gap, so it is not warped --
    # mirrors ``_Node.end``).
    raw_api_time: dict[str, float] = {
        n.node_id: (n.request.api_time or 0.0) for n in flat
    }

    # Causal predecessors + the WARPED edge delay per predecessor edge. A node
    # whose only edge is from START is a recorded root (its recorded arrival is
    # its own warped t).
    preds: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    pred_delay_us: dict[str, dict[str, float]] = {nid: {} for nid in graph.nodes}
    start_rooted: dict[str, bool] = {nid: False for nid in graph.nodes}
    start_min_delay_us: dict[str, float] = {}
    start_anchor: dict[str, tuple[str, float, float | None]] = {}
    for edge in graph.edges:
        if not isinstance(edge, StaticEdge):
            continue
        tgt = edge.target
        if tgt not in preds:
            continue
        if edge.source == "START":
            start_rooted[tgt] = True
            start_min_delay_us[tgt] = edge.min_start_delay_us or 0.0
        elif edge.delay_after_predecessor_start_us is not None:
            # Start-anchored: gates off the predecessor's DISPATCH (dispatch-to-
            # dispatch), NOT its completion, so it does NOT model an end-to-start
            # wait. Keep it out of preds / pred_delay_us (those drive the
            # end-to-start gate) and record the (parent, D, D' | None) separately.
            start_anchor[tgt] = (
                edge.source,
                edge.delay_after_predecessor_start_us,
                edge.delay_after_predecessor_first_token_us,
            )
        else:
            preds[tgt].append(edge.source)
            pred_delay_us[tgt][edge.source] = edge.delay_after_predecessor_us or 0.0

    nodes: dict[str, _RecordedNode] = {}
    for nid, node in graph.nodes.items():
        if not isinstance(node, LlmNode):
            continue
        path = node.metadata["trie"]["prompt_segment_ids"]
        warped_start_s = (node.arrival_offset_us or 0) / 1e6
        warped_end_s = warped_start_s + raw_api_time.get(nid, 0.0)
        raw_start_s, raw_end_s = raw_timing.get(nid, (0.0, 0.0))
        nodes[nid] = _RecordedNode(
            node_id=nid,
            messages=pool.materialize(path),
            start_s=warped_start_s,
            end_s=warped_end_s,
            raw_start_s=raw_start_s,
            raw_end_s=raw_end_s,
            predecessors=preds.get(nid, []),
            pred_delay_us=pred_delay_us.get(nid, {}),
            rooted_at_start=start_rooted.get(nid, False) and not preds.get(nid),
            min_start_delay_us=(
                start_min_delay_us.get(nid)
                if start_rooted.get(nid, False) and not preds.get(nid)
                else None
            ),
            start_anchor=start_anchor.get(nid),
        )
    return _RecordedTrace(trace_id=trace.id, nodes=nodes)


# --- criterion 2: content vs real trace -----------------------------------


def content_vs_real_trace(
    raw_jsonl: Path | None,
    trace_file: Path | None,
    *,
    recorded: _RecordedTrace | None = None,
    records: list[_ExportRecord] | None = None,
    tokenizer_name: str = "builtin",
    prompt_corpus: str = "coding",
    root_seed: int | None = None,
) -> Report:
    """Each profiling export record's prompt == the recorded trace's prescribed content.

    Maps every profiling record to its recorded trie node (by the node id folded
    into ``x_request_id``) and asserts the exported ``payload.messages`` equal
    ``pool.materialize(node.prompt_segment_ids)`` recomputed from the trace.

    The exported first-user message may carry a ``[rid:...]`` cache-bust prefix
    (when the run used ``--cache-bust first_turn_prefix``); it is stripped before
    comparison so the check proves CONTENT fidelity independent of the bust marker.

    The corpus driver supplies a pre-built ``recorded`` / ``records`` so each
    recorded trace is built once and both criteria run against it; the single-file
    entrypoint builds them from ``trace_file`` / ``raw_jsonl``. Content fidelity is
    cap-INDEPENDENT (prompt content doesn't depend on timing).
    """
    report = Report(name="content_vs_real_trace")
    if recorded is None:
        assert trace_file is not None
        recorded = build_recorded_trace(
            trace_file,
            tokenizer_name=tokenizer_name,
            prompt_corpus=prompt_corpus,
            root_seed=root_seed,
        )
    if records is None:
        assert raw_jsonl is not None
        records = load_raw_export(raw_jsonl)

    profiling = [
        r
        for r in records
        if r.phase == _PROFILING and r.trace_base == recorded.trace_id
    ]
    if not profiling:
        report.fail("export", "no profiling records found in raw export")
        return report

    for i, rec in enumerate(profiling):
        where = f"profiling[{i}] node={rec.node_id} conv={rec.conversation_id}"
        report.checked += 1
        node = recorded.nodes.get(rec.node_id or "")
        if node is None:
            report.fail(where, f"node id {rec.node_id!r} not in rebuilt trie graph")
            continue
        got = _strip_rid_marker(rec.messages)
        if got != node.messages:
            report.fail(where, _first_message_diff(node.messages, got))
            continue
        report.passes += 1
    return report


def _strip_rid_marker(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return a copy with the ``[rid:...]`` prefix stripped from the first user msg."""
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "user":
            m["content"] = _RID_MARKER_RE.sub("", m["content"], count=1)
            break
    return out


def _first_message_diff(
    expected: list[dict[str, str]], got: list[dict[str, str]]
) -> str:
    """A compact description of the first differing message (for failure detail)."""
    if len(expected) != len(got):
        return f"message count {len(got)} != expected {len(expected)}"
    for idx, (e, g) in enumerate(zip(expected, got, strict=False)):
        if e != g:
            return (
                f"msg[{idx}] role {g.get('role')!r}/{e.get('role')!r}; "
                f"content {g.get('content', '')[:60]!r} != "
                f"{e.get('content', '')[:60]!r}"
            )
    return "messages differ (no positional diff found)"


# --- criterion 3: causality + timing vs real trace ------------------------

# Timing tolerance for the recorded-relative-offset comparison. Wall-clock is
# absolute-different (a fresh run-origin), but the RECORDED RELATIVE gap between a
# node and its causal predecessor must survive the replay. We allow the larger of
# an absolute floor (scheduling jitter, ZMQ transit, the gap-warp rounding) and a
# relative fraction of the recorded gap (longer recorded gaps accrue
# proportionally more replay slack).
_TIMING_ABS_TOLERANCE_S = 0.75
_TIMING_REL_TOLERANCE = 0.15

# Sub-second slack distinguishing a warped (idle-capped) edge from an exact one:
# float us->s round-trip on the warped delay introduces tiny noise, so treat a
# warped expected within this of the raw recorded gap as "exact" (gap <= cap).
_CLASSIFY_EPSILON_S = 1e-3


def causality_timing_vs_real_trace(
    raw_jsonl: Path | None,
    trace_file: Path | None,
    *,
    recorded: _RecordedTrace | None = None,
    records: list[_ExportRecord] | None = None,
    idle_gap_cap_seconds: float | None = 60.0,
    tokenizer_name: str = "builtin",
    prompt_corpus: str = "coding",
    root_seed: int | None = None,
) -> Report:
    """Reconstruct dispatch causality + relative timing from the export ALONE.

    For every profiling record we recover its node id (from ``x_request_id``)
    and look up the recorded trie causal predecessors (``StaticEdge`` sources). We
    then assert, against the OBSERVED ``request_start_ns`` of each dispatched
    record (warmup + profiling, since a profiling node's predecessor may have been
    chopped into warmup by t*):

    (a) Causal ORDER: for each recorded predecessor that was ALSO dispatched, the
        record's observed ``request_start_ns`` is >= the predecessor's observed
        ``request_start_ns`` -- a request never dispatched before its recorded
        predecessor (the closed-loop waits-for edge is honored on the wire).

    (b) Relative TIMING: the WARPED end-to-start gap between the node and its
        (latest-completing) recorded predecessor -- the rebuilt trie's
        ``StaticEdge.delay_after_predecessor_us`` on the idle-gap-CAPPED clock,
        captured in ``_RecordedNode.pred_delay_us`` -- matches the OBSERVED
        start-to-start gap within :data:`_TIMING_ABS_TOLERANCE_S` (abs) or
        :data:`_TIMING_REL_TOLERANCE` (relative to the warped gap).

        Cap-awareness (principled, NOT a fudge): a faithful replay applies the
        agentx idle-gap cap, so for any recorded end-to-start gap > cap the RUN
        dispatched after only ``cap`` seconds, not the raw recorded think-time.
        The EXPECTED delay must therefore be the WARPED edge delay, not the raw
        recorded gap. Each checked edge is CLASSIFIED by comparing the warped
        expected against the raw recorded gap:

        * "exact" when warped == raw (gap <= cap) -- the export reproduces the
          REAL recorded think-time exactly.
        * "idle-capped" when warped < raw (gap > cap) -- bounded by the documented
          faithful cap (warped expected == cap).

        Tolerances are NOT weakened and no edge is skipped to force a pass.

        Why END-to-start, not start-to-start: the executor parks each node for its
        ``delay_after_predecessor_us`` AFTER the predecessor RETURNS, then
        dispatches. The proof run drives the in-repo mock at ``--ttft 0 --itl 0``,
        so the predecessor returns ~instantly after IT dispatched; the observed
        dispatch-clock gap therefore collapses onto the (warped) end-to-start edge
        delay (the recorded ``api_time`` is the predecessor's own processing, which
        the zero-latency mock does not reproduce).

        For a recorded node with NO dispatched predecessor (a t*-chop survivor
        re-rooted at START, or a true root), the reference gap is measured
        (warped) start-to-start from the trace's earliest dispatched node (the run
        anchor) and compared to the observed gap from that anchor's dispatch.

        START-ANCHORED node (a mid-flight spawn / chain overlap whose only edge is
        ``StaticEdge.delay_after_predecessor_start_us`` D): the runtime schedules it
        at its parent's DISPATCH and gates it at dispatch + D, so the reference is
        DISPATCH-to-dispatch -- observed ``request_start_ns(child) -
        request_start_ns(parent)`` must equal the warped start-anchor delay (same
        tolerance), and the parent must dispatch before the child. These edges are
        counted in ``exact_edges`` (they reproduce the recorded start-to-start gap
        exactly; the idle-gap cap warps the delay itself, not the anchor kind).

        POST-TTFT start-anchored node (its edge also carries
        ``delay_after_predecessor_first_token_us`` D'): the runtime re-anchors it
        onto the parent's OBSERVED first token, so the proof compares the child
        against ``parent_first_token + D'`` whenever that first token is recoverable
        from the export (``responses[0].perf_ns - start_perf_ns`` added to the
        parent's dispatch wall clock; post-TTFT anchor parents STREAM by
        construction). The runtime falls back to dispatch + D -- and the proof does
        too, emitting a LOUD ``FALLBACK`` note (the edge is still CHECKED, never
        silently skipped) -- only when that first token is truly unrecoverable: an
        errored parent that streamed nothing, or a zero-latency export with no
        responses. A non-streaming record is not a fallback cause here -- its lone
        ``TextResponse`` still yields a duration, and it never sources a post-TTFT
        anchor edge anyway.

    Absolute wall-clock differs run-to-run; only the recorded RELATIVE timing +
    causal order are asserted. The corpus driver supplies a pre-built ``recorded``
    / ``records`` (built ONCE with the cap); the single-file entrypoint builds
    them from ``trace_file`` / ``raw_jsonl`` with ``idle_gap_cap_seconds``.
    """
    report = Report(name="causality_timing_vs_real_trace")
    if recorded is None:
        assert trace_file is not None
        recorded = build_recorded_trace(
            trace_file,
            idle_gap_cap_seconds=idle_gap_cap_seconds,
            tokenizer_name=tokenizer_name,
            prompt_corpus=prompt_corpus,
            root_seed=root_seed,
        )
    if records is None:
        assert raw_jsonl is not None
        records = load_raw_export(raw_jsonl)

    def _earliest_by_node(phase: str | None) -> dict[str, int]:
        """Earliest observed ``request_start_ns`` per node id, optionally one phase."""
        out: dict[str, int] = {}
        for rec in records:
            if rec.trace_base != recorded.trace_id or rec.node_id is None:
                continue
            if rec.request_start_ns is None:
                continue
            if phase is not None and rec.phase != phase:
                continue
            prev = out.get(rec.node_id)
            if prev is None or rec.request_start_ns < prev:
                out[rec.node_id] = rec.request_start_ns
        return out

    # CAUSAL-ORDER uses the cross-phase earliest dispatch (a profiling node's
    # recorded predecessor may have been chopped into warmup by t*). RELATIVE
    # TIMING must stay phase-consistent: a profiling node's recorded predecessor,
    # when itself dispatched in profiling, replays the SAME re-rooted t*-relative
    # schedule, so we compare profiling-vs-profiling timestamps (warmup primes a
    # node ~immediately and would alias the edge delay).
    any_phase_ns = _earliest_by_node(None)
    profiling_ns = _earliest_by_node(_PROFILING)

    def _first_token_duration_by_node() -> dict[str, int]:
        """Observed first-token DURATION ns for each node's EARLIEST profiling record.

        Pairs the same earliest-dispatch record ``profiling_ns`` picks, so a node's
        recovered first token is ``profiling_ns[node] + this[node]`` on the wall
        clock. Absent only for a record with NO responses (an errored request that
        streamed nothing, or a zero-latency export); a non-streaming record's lone
        ``TextResponse`` still yields its full duration here, but this map is
        consulted only for post-TTFT anchor parents (which stream by construction),
        so that value never anchors -- when absent the timing proof falls back to
        dispatch + D.
        """
        best_start: dict[str, int] = {}
        out: dict[str, int] = {}
        for rec in records:
            if rec.trace_base != recorded.trace_id or rec.node_id is None:
                continue
            if rec.phase != _PROFILING or rec.request_start_ns is None:
                continue
            prev = best_start.get(rec.node_id)
            if prev is not None and rec.request_start_ns >= prev:
                continue
            best_start[rec.node_id] = rec.request_start_ns
            dur = rec.observed_ttft_ns
            if dur is None:
                out.pop(rec.node_id, None)
            else:
                out[rec.node_id] = dur
        return out

    profiling_ft_ns = _first_token_duration_by_node()

    if not any_phase_ns:
        report.fail("export", "no dispatched trie nodes recovered from raw export")
        return report

    # Instance run-origin for START-rooted timing. The executor fires a
    # START-rooted node at ``anchor_wall + min_start_delay_us``
    # (``_compute_firing_gate_us``, ``absolute_start_offsets=True``), where
    # ``anchor_wall`` is the SHARED instance run-start pinned once on the top
    # executor's run. We recover that origin from the profiled START-root with the
    # SMALLEST ``min_start_delay_us`` (typically the recorded root, offset 0):
    # ``origin_ns = obs_ns(root) - root.min_start_delay_us``. Every other
    # START-root must then dispatch at ``origin_ns + its own min_start_delay_us``,
    # NOT relative to the root's END (independent roots share one t* origin and do
    # NOT chain off the root's processing time).
    if not profiling_ns:
        report.fail("export", "no profiling records for this trace in raw export")
        return report
    start_roots = [
        nid
        for nid in profiling_ns
        if nid in recorded.nodes and recorded.nodes[nid].min_start_delay_us is not None
    ]
    if start_roots:
        # Prefer the profiled START-root with the SMALLEST min_start_delay (the
        # recorded root, offset 0) to recover the shared instance run-origin.
        anchor_id = min(
            start_roots,
            key=lambda nid: recorded.nodes[nid].min_start_delay_us or 0.0,
        )
        origin_ns: float | None = (
            profiling_ns[anchor_id]
            - (recorded.nodes[anchor_id].min_start_delay_us or 0.0) * 1e3
        )
        # Independent START-roots must all imply the SAME origin within tolerance;
        # a disagreement means a root fired off-schedule (a real fidelity signal),
        # so surface it as a note (each root is also asserted per-node below).
        for nid in start_roots:
            implied = (
                profiling_ns[nid]
                - (recorded.nodes[nid].min_start_delay_us or 0.0) * 1e3
            )
            drift_s = abs(implied - origin_ns) / 1e9
            if drift_s > _TIMING_ABS_TOLERANCE_S:
                report.notes.append(
                    f"START-root {nid} implies origin drift {drift_s:.3f}s vs "
                    f"anchor {anchor_id} (checked per-node below)"
                )
    else:
        # No profiled START-root (the profiled subset is a mid-chain slice whose
        # roots were never dispatched in profiling): there is no run-origin to
        # anchor START-roots against. Fall back to the earliest-RECORDED profiled
        # node as a bare relative-timing anchor that carries NO constraint (it is
        # the origin of the comparable slice). START-root expecteds are then
        # unreachable, so every profiled node is timed off its profiled preds.
        anchor_id = min(
            profiling_ns,
            key=lambda nid: (
                recorded.nodes[nid].start_s if nid in recorded.nodes else float("inf")
            ),
        )
        origin_ns = None
    report.notes.append(
        f"trace={recorded.trace_id} dispatched_nodes={sorted(any_phase_ns)} "
        f"profiled={sorted(profiling_ns)} anchor={anchor_id}"
    )

    profiling = [
        r
        for r in records
        if r.phase == _PROFILING
        and r.node_id is not None
        and r.trace_base == recorded.trace_id
    ]

    def _check_record(rec: _ExportRecord, where: str) -> None:
        """Run every order/timing assertion for one profiling record.

        Appends mismatches (possibly several -- causal-order and relative-timing
        are DISTINCT assertion kinds, tallied separately on the report) and
        returns; the caller owns the record-level ``checked``/``passes`` tally.
        """
        nid = rec.node_id
        node = recorded.nodes.get(nid or "")
        if node is None:
            report.fail(where, f"node id {nid!r} not in rebuilt trie graph")
            return
        if rec.request_start_ns is None:
            report.fail(where, "record has no request_start_ns")
            return
        obs_s = rec.request_start_ns / 1e9

        # Start-anchored node: its ONE recorded edge gates off the parent's
        # DISPATCH (dispatch-to-dispatch), not its completion. Compare the
        # OBSERVED start-to-start gap to the warped start-anchor delay, and require
        # the parent to have dispatched before this child. (Start-anchored edges
        # are kept out of predecessors/pred_delay_us, so this branch is the only
        # place they are timed.)
        if node.start_anchor is not None:
            parent, delay_us, first_token_delay_us = node.start_anchor
            # (a) Causal order: never dispatched before the recorded parent (same
            # tolerance the end-to-start order check uses).
            parent_any_ns = any_phase_ns.get(parent)
            if parent_any_ns is not None:
                report.order_checks += 1
                if obs_s + _TIMING_ABS_TOLERANCE_S < parent_any_ns / 1e9:
                    report.order_failures += 1
                    report.fail(
                        where,
                        f"dispatched {parent_any_ns / 1e9 - obs_s:.3f}s BEFORE "
                        f"recorded start-anchor parent {parent} "
                        f"(causal-order violation)",
                    )
            # (b) Relative timing. If the parent was never dispatched in profiling
            # there is no gate to compare against; the order check above still
            # guards it.
            parent_prof_ns = profiling_ns.get(parent)
            if parent_prof_ns is None:
                return
            # POST-TTFT node (D' present): the runtime re-anchors it onto the
            # parent's OBSERVED first token, so the proof must too -- expected ==
            # parent_first_token + D' whenever that first token is recoverable from
            # the export. Post-TTFT anchor parents STREAM by construction; the first
            # token is unrecoverable only when the parent has NO responses (an
            # errored parent that streamed nothing, or a zero-latency export), in
            # which case the runtime falls back to dispatch + D and so does the
            # proof, LOUDLY: the edge is still CHECKED, never silently skipped. A
            # pre-TTFT node (D' is None) always gates dispatch-to-dispatch on D.
            parent_ft_dur_ns = (
                profiling_ft_ns.get(parent)
                if first_token_delay_us is not None
                else None
            )
            if parent_ft_dur_ns is not None:
                expected_gap_s = first_token_delay_us / 1e6
                expected_s = (parent_prof_ns + parent_ft_dur_ns) / 1e9 + expected_gap_s
                ref = f"first_token({parent})+D'"
            else:
                expected_gap_s = delay_us / 1e6
                expected_s = parent_prof_ns / 1e9 + expected_gap_s
                if first_token_delay_us is not None:
                    report.notes.append(
                        f"FALLBACK first-token edge {parent}->{nid}: parent observed "
                        f"TTFT unrecoverable from raw export; compared child dispatch "
                        f"vs parent dispatch + D ({expected_gap_s:.3f}s)"
                    )
                    ref = f"dispatch({parent})+D [first-token fallback]"
                else:
                    ref = f"dispatch({parent})+D"
            report.timing_checks += 1
            tol = max(
                _TIMING_ABS_TOLERANCE_S, _TIMING_REL_TOLERANCE * abs(expected_gap_s)
            )
            if abs(obs_s - expected_s) > tol:
                report.timing_failures += 1
                report.fail(
                    where,
                    f"start-anchor offset vs {ref}: observed dispatch "
                    f"{obs_s - expected_s:+.3f}s off expected delay "
                    f"{expected_gap_s:.3f}s (tol {tol:.3f}s)",
                )
                return
            report.exact_edges += 1
            return

        # Causal order considers any-phase dispatch; relative timing stays within
        # the profiling phase (phase-consistent re-rooted schedule).
        order_preds = [p for p in node.predecessors if p in any_phase_ns]
        timing_preds = [p for p in node.predecessors if p in profiling_ns]

        # (a) Causal order: never dispatched before any dispatched recorded pred.
        for pred in order_preds:
            report.order_checks += 1
            pred_obs_s = any_phase_ns[pred] / 1e9
            if obs_s + _TIMING_ABS_TOLERANCE_S < pred_obs_s:
                report.order_failures += 1
                report.fail(
                    where,
                    f"dispatched {pred_obs_s - obs_s:.3f}s BEFORE recorded "
                    f"predecessor {pred} (causal-order violation)",
                )

        # (b) Relative timing: compute the EXPECTED observed dispatch the way the
        # executor computes its firing gate -- the AND-fan-in MAX over incoming
        # edges -- using OBSERVED predecessor dispatch times:
        #
        #   expected_ns = max over profiled preds p of
        #                 ( profiling_ns[p] + warped_delay_us[p] * 1e3 )
        #
        # and for a START-rooted node (no profiled predecessor):
        #
        #   expected_ns = origin_ns + min_start_delay_us * 1e3
        #
        # The BINDING predecessor is the argmax (the gate the executor actually
        # waited on), which may NOT be the recorded latest-completing pred when a
        # different edge's warped delay (e.g. a 60s-capped one) pushes its gate
        # later. ``expected_gap_s`` (the binding edge's warped delay, or the
        # START-root offset) sets the relative-tolerance magnitude exactly as
        # before. ``raw_gap_s`` is the unwarped recorded gap of the binding edge,
        # kept only to CLASSIFY exact-vs-idle-capped (NOT the comparison target).
        classify = False
        if timing_preds:
            gates_ns = {
                p: profiling_ns[p] + node.pred_delay_us.get(p, 0.0) * 1e3
                for p in timing_preds
            }
            cause = max(gates_ns, key=lambda p: gates_ns[p])
            expected_ns = gates_ns[cause]
            expected_gap_s = node.pred_delay_us.get(cause, 0.0) / 1e6
            raw_gap_s = node.raw_start_s - recorded.nodes[cause].raw_end_s
            ref = f"pred={cause}"
            # Classify exact-vs-idle-capped only when the argmax edge carries a
            # nonzero warped delay: a NON-BINDING AND-join edge has delay 0.0 by
            # construction (build_interval_edges), so when it wins the argmax it
            # carries no think-time to compare -- its positive raw end-to-start
            # gap would misread as "idle-capped" although no gap exceeded the
            # cap. (A capped binding edge always has warped delay == cap > 0, so
            # no idle-capped edge is ever skipped by this guard.)
            classify = expected_gap_s > 0.0
        elif node.min_start_delay_us is not None and origin_ns is not None:
            expected_ns = origin_ns + node.min_start_delay_us * 1e3
            expected_gap_s = node.min_start_delay_us / 1e6
            raw_gap_s = expected_gap_s
            ref = f"origin={anchor_id}"
        else:
            # A node with recorded non-START preds but NONE dispatched in profiling
            # (its preds were chopped into warmup by t*): it has no profiling-phase
            # gate to compare against. The causal-order check above still guards it;
            # it stays checked without a relative-timing assertion.
            return

        report.timing_checks += 1
        expected_s = expected_ns / 1e9
        tol = max(_TIMING_ABS_TOLERANCE_S, _TIMING_REL_TOLERANCE * abs(expected_gap_s))
        if abs(obs_s - expected_s) > tol:
            report.timing_failures += 1
            report.fail(
                where,
                f"relative offset vs {ref}: observed dispatch "
                f"{obs_s - expected_s:+.3f}s off warped-expected (binding warped "
                f"delay {expected_gap_s:.3f}s, raw {raw_gap_s:.3f}s, tol {tol:.3f}s)",
            )
            return
        if classify:
            # warped < raw => the recorded gap exceeded the cap and was compressed
            # to it (idle-capped); warped == raw => sub-cap gap reproduced exactly.
            if expected_gap_s + _CLASSIFY_EPSILON_S < raw_gap_s:
                report.idle_capped_edges += 1
            else:
                report.exact_edges += 1

    for i, rec in enumerate(profiling):
        report.checked += 1
        mismatches_before = len(report.mismatches)
        _check_record(rec, f"profiling[{i}] node={rec.node_id}")
        if len(report.mismatches) == mismatches_before:
            report.passes += 1
    return report


# --- criterion 1: byte-exact vs v0.4 --------------------------------------


def content_byte_exact_vs_v04(ours_raw: Path, v04_raw: Path) -> Report:
    """Per matching trace, ours-vs-v0.4 raw payloads are byte-identical (rid-stripped).

    Matches records across the two raw exports by ``(conversation_id base,
    node_id)`` -- the stable per-trace-per-node identity -- and asserts the
    rid-stripped ``payload.messages`` are byte-identical. Only profiling records
    are compared (warmup dispatch counts can legitimately differ between the two
    builds). A trace/node present in one export but not the other is reported as a
    coverage mismatch, so a vacuous (empty-overlap) pass cannot slip through.
    """
    report = Report(name="content_byte_exact_vs_v04")
    ours = _index_by_trace_node(load_raw_export(ours_raw))
    v04 = _index_by_trace_node(load_raw_export(v04_raw))

    common = sorted(set(ours) & set(v04))
    only_ours = sorted(set(ours) - set(v04))
    only_v04 = sorted(set(v04) - set(ours))
    for key in only_ours:
        report.fail(f"{key}", "present in OURS export but missing from v0.4 export")
    for key in only_v04:
        report.fail(f"{key}", "present in v0.4 export but missing from OURS export")

    for key in common:
        report.checked += 1
        ours_msgs = _strip_rid_marker(ours[key])
        v04_msgs = _strip_rid_marker(v04[key])
        if ours_msgs != v04_msgs:
            report.fail(str(key), _first_message_diff(v04_msgs, ours_msgs))
            continue
        report.passes += 1
    return report


def _index_by_trace_node(
    records: list[_ExportRecord],
) -> dict[tuple[str, str | None], list[dict[str, str]]]:
    """Index profiling records by ``(trace_base, node_id)`` -> messages.

    Last write wins on a duplicate key (a recycled node re-fire is byte-identical
    on content, so the choice is immaterial for the byte-exact comparison).
    """
    out: dict[tuple[str, str | None], list[dict[str, str]]] = {}
    for rec in records:
        if rec.phase != _PROFILING:
            continue
        out[(rec.trace_base, rec.node_id)] = rec.messages
    return out


# --- corpus-scale driver --------------------------------------------------


@dataclass
class _TraceResult:
    """Per-trace fidelity outcome the corpus driver aggregates + prints.

    ``dispatched_records`` counts the raw-export PROFILING records mapped to
    this trace -- the records both criteria actually assert on. Zero means the
    bounded run never profiled the trace (either never dispatched it at all, or
    only warmup-primed it; ``warmup_records`` distinguishes the two) and is
    reported as coverage (``SKIP``), never as a failure -- while a nonzero count
    with a failing report is a REAL fidelity failure the exit code must surface.
    """

    trace_id: str
    content: Report
    timing: Report
    dispatched_nodes: int
    total_nodes: int
    dispatched_records: int = 0
    warmup_records: int = 0

    @property
    def coverage(self) -> float:
        """Fraction of recorded trie nodes this run actually profiled.

        A bounded run (``--request-count`` / a short window) may never reach the
        deepest recorded turns, and graph auto-warmup may prime a node without
        ever profiling it; those nodes are reported COVERAGE, not failures (the
        proof only asserts the profiled subset).
        """
        return self.dispatched_nodes / self.total_nodes if self.total_nodes else 0.0

    @property
    def passed(self) -> bool:
        return self.content.passed and self.timing.passed


def prove_corpus(
    raw_jsonl: Path,
    trace_dir: Path,
    idle_gap_cap_seconds: float | None = 60.0,
    *,
    tokenizer_name: str = "builtin",
    prompt_corpus: str = "coding",
    root_seed: int | None = None,
) -> int:
    """Prove a whole corpus: one raw export vs every ``*.json`` trace in a dir.

    The raw export interleaves ALL traces; we map each record to its trace by
    ``_ExportRecord.trace_base == recorded.trace_id`` and run BOTH
    :func:`content_vs_real_trace` and :func:`causality_timing_vs_real_trace`
    restricted to that trace's records. Each recorded trace is built ONCE (with
    the cap + content knobs -- see :func:`build_recorded_trace`) and both
    criteria share it.

    Prints a per-trace + TOTAL summary (records dispatched, content/causal-order/
    relative-timing pass counts, exact-vs-idle-capped edge counts, node coverage).

    Exit contract: returns nonzero iff ANY trace with PROFILING records fails
    either criterion (including unresolvable node ids and records missing
    ``request_start_ns`` -- a printed MISMATCH is never a passing exit), OR the
    proof was VACUOUS (no trace files, or no export record checked against any
    trace): a proof that checked nothing must not pass. Traces without profiling
    records -- never dispatched at all, or only warmup-primed (graph auto-warmup
    bursts priming credits before profiling) -- are reported as coverage
    (``SKIP``), NOT failures; an ALL-warmup export still fails loudly through
    the VACUOUS gate.
    """
    records = load_raw_export(raw_jsonl)
    by_trace: dict[str, list[_ExportRecord]] = {}
    for rec in records:
        by_trace.setdefault(rec.trace_base, []).append(rec)

    trace_files = sorted(Path(trace_dir).glob("*.json"))
    results: list[_TraceResult] = []
    for trace_file in trace_files:
        recorded = build_recorded_trace(
            trace_file,
            idle_gap_cap_seconds=idle_gap_cap_seconds,
            tokenizer_name=tokenizer_name,
            prompt_corpus=prompt_corpus,
            root_seed=root_seed,
        )
        trace_records = by_trace.get(recorded.trace_id, [])
        profiling_records = [r for r in trace_records if r.phase == _PROFILING]
        if not profiling_records:
            # No PROFILING records for this trace: either the bounded run never
            # dispatched it at all, or auto-warmup only primed it. Both criteria
            # hard-fail on zero profiling records, so running them here would
            # mislabel coverage as a fidelity failure -- report SKIP instead
            # (the corpus-level VACUOUS gate still fails an all-warmup export).
            results.append(
                _TraceResult(
                    trace_id=recorded.trace_id,
                    content=Report(name="content_vs_real_trace"),
                    timing=Report(name="causality_timing_vs_real_trace"),
                    dispatched_nodes=0,
                    total_nodes=len(recorded.nodes),
                    dispatched_records=0,
                    warmup_records=len(trace_records),
                )
            )
            continue
        content = content_vs_real_trace(
            None, None, recorded=recorded, records=trace_records
        )
        timing = causality_timing_vs_real_trace(
            None, None, recorded=recorded, records=trace_records
        )
        dispatched = {
            r.node_id
            for r in profiling_records
            if r.node_id is not None and r.node_id in recorded.nodes
        }
        results.append(
            _TraceResult(
                trace_id=recorded.trace_id,
                content=content,
                timing=timing,
                dispatched_nodes=len(dispatched),
                total_nodes=len(recorded.nodes),
                dispatched_records=len(profiling_records),
                warmup_records=len(trace_records) - len(profiling_records),
            )
        )

    _print_corpus_summary(results, idle_gap_cap_seconds)

    total_checked = sum(r.content.checked + r.timing.checked for r in results)
    if not results or total_checked == 0:
        print(
            "VACUOUS: nothing checked (no recorded traces, or no raw-export "
            "record maps to any trace) -- a proof that checked nothing FAILS"
        )
        return 1
    failed = [r.trace_id for r in results if r.dispatched_records > 0 and not r.passed]
    if failed:
        print(f"corpus proof: FAIL ({len(failed)} trace(s): {', '.join(failed)})")
        return 1
    print("corpus proof: PASS")
    return 0


def _print_corpus_summary(
    results: list[_TraceResult], idle_gap_cap_seconds: float | None
) -> None:
    """Print the per-trace + TOTAL corpus fidelity table.

    ``content`` counts profiling records whose prompt matched; ``order`` counts
    per-edge causal-order comparisons; ``timing`` counts relative-timing gate
    comparisons -- two DIFFERENT assertion kinds, so they get separate columns
    rather than one mixed pass/checked pair. All pass counts come from the
    reports' dedicated counters (never derived by subtracting mismatch counts,
    which can exceed one per checked record).
    """
    cap = "off" if idle_gap_cap_seconds is None else f"{idle_gap_cap_seconds:g}s"
    print(f"weka trace-fidelity corpus proof (idle-gap cap = {cap})")
    print(
        f"{'trace':<28} {'disp/tot':>9} {'cov':>5} "
        f"{'content':>14} {'order':>12} {'timing':>12} {'edges(ex/cap)':>14}"
    )
    tot_disp = tot_total = 0
    tot_c_chk = tot_c_pass = 0
    tot_o_chk = tot_o_pass = tot_t_chk = tot_t_pass = 0
    tot_exact = tot_capped = 0
    for r in results:
        o_pass = r.timing.order_checks - r.timing.order_failures
        t_pass = r.timing.timing_checks - r.timing.timing_failures
        if r.dispatched_records == 0:
            status = (
                "SKIP (warmup-primed, not profiled)"
                if r.warmup_records
                else "SKIP (not dispatched)"
            )
        elif r.passed:
            status = "OK"
        else:
            status = "FAIL"
        print(
            f"{r.trace_id:<28} "
            f"{r.dispatched_nodes:>4}/{r.total_nodes:<4} "
            f"{r.coverage * 100:>4.0f}% "
            f"{r.content.passes:>6}/{r.content.checked:<6} "
            f"{o_pass:>5}/{r.timing.order_checks:<5} "
            f"{t_pass:>5}/{r.timing.timing_checks:<5} "
            f"{r.timing.exact_edges:>6}/{r.timing.idle_capped_edges:<6} "
            f"{status}"
        )
        for m in (*r.content.mismatches, *r.timing.mismatches):
            print(f"    MISMATCH @ {m.where}: {m.detail}")
        for n in r.timing.notes:
            if n.startswith("FALLBACK"):
                print(f"    {n}")
        tot_disp += r.dispatched_nodes
        tot_total += r.total_nodes
        tot_c_chk += r.content.checked
        tot_c_pass += r.content.passes
        tot_o_chk += r.timing.order_checks
        tot_o_pass += o_pass
        tot_t_chk += r.timing.timing_checks
        tot_t_pass += t_pass
        tot_exact += r.timing.exact_edges
        tot_capped += r.timing.idle_capped_edges
    cov = (tot_disp / tot_total * 100) if tot_total else 0.0
    print(
        f"{'TOTAL':<28} {tot_disp:>4}/{tot_total:<4} {cov:>4.0f}% "
        f"{tot_c_pass:>6}/{tot_c_chk:<6} {tot_o_pass:>5}/{tot_o_chk:<5} "
        f"{tot_t_pass:>5}/{tot_t_chk:<5} {tot_exact:>6}/{tot_capped:<6}"
    )
    print(
        f"edges: {tot_exact} exact (reproduce real recorded think-time), "
        f"{tot_capped} idle-capped (bounded by the {cap} cap)"
    )


def _main(argv: list[str] | None = None) -> int:
    """CLI: ``--raw <export.jsonl> --trace-dir <dir> [--idle-gap-cap-seconds 60]
    [--tokenizer builtin] [--corpus coding] [--seed N]``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="raw export JSONL")
    parser.add_argument(
        "--trace-dir", type=Path, required=True, help="dir of *.json recorded traces"
    )
    parser.add_argument(
        "--idle-gap-cap-seconds",
        type=float,
        default=60.0,
        help="agentx idle-gap cap applied to the rebuilt trie (default 60; "
        "pass a negative value to disable the warp)",
    )
    parser.add_argument(
        "--tokenizer",
        default="builtin",
        help="content tokenizer the proved run was built with "
        "(default 'builtin', the bare live-run default)",
    )
    parser.add_argument(
        "--corpus",
        default="coding",
        help="prompt corpus the proved run was built with "
        "(default 'coding', the live-run default)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="content root seed the proved run was built with "
        "(the run's --random-seed; default none, the live-run default)",
    )
    args = parser.parse_args(argv)
    cap = args.idle_gap_cap_seconds if args.idle_gap_cap_seconds >= 0 else None
    return prove_corpus(
        args.raw,
        args.trace_dir,
        idle_gap_cap_seconds=cap,
        tokenizer_name=args.tokenizer,
        prompt_corpus=args.corpus,
        root_seed=args.seed,
    )


__all__ = [
    "Mismatch",
    "Report",
    "build_recorded_trace",
    "causality_timing_vs_real_trace",
    "content_byte_exact_vs_v04",
    "content_vs_real_trace",
    "load_raw_export",
    "prove_corpus",
]


if __name__ == "__main__":
    raise SystemExit(_main())
