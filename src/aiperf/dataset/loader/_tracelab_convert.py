# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-python TraceLab -> Weka-trace conversion.

TraceLab publishes one JSONL row per LLM round. The Weka trace schema wants one
object per *session*, carrying a per-request KV-cache block id list. This module
groups rounds into sessions, synthesizes those block ids from TraceLab's
engine-reported prefix decomposition, and reconstructs subagent nesting.

Apart from one file-opening helper, nothing here imports aiperf: the output is
a plain ``dict`` matching the Weka trace schema, which
:mod:`aiperf.dataset.loader.tracelab_trace` validates into a ``WekaTrace``. Keeping the conversion free of framework types makes it directly
unit-testable and keeps the loader itself a thin adapter.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "DEFAULT_MIN_SPAWN_MS",
    "HashIdMinter",
    "JoinStats",
    "Spawn",
    "build_join_index",
    "build_trace",
    "group_children_by_parent",
    "order_rounds",
    "safe_trace_id",
    "session_span",
    "synthesize_hash_ids",
]

INPUT_EVENTS = frozenset({"user_message", "tool_result"})
OUTPUT_EVENTS = frozenset({"text", "reasoning", "tool_call"})

# Claude Code spawns a subagent through a single blocking tool call. Codex
# instead uses an async spawn_agent/wait_agent/close_agent lifecycle; see
# :func:`build_join_index`.
CLAUDE_SPAWN_TOOLS = frozenset({"Agent", "Task"})

# A subagent round-trip is long. Short spawning-tool calls are overwhelmingly
# no-op or error returns, and admitting them widens the containment window
# enough to start capturing unrelated concurrent sessions.
DEFAULT_MIN_SPAWN_MS = 10000

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

# Recorded in the trace ``totals`` so a reconstructed trace names its origin.
SOURCE_TAG = "tracelab"


def parse_ts(value: str) -> float:
    """ISO-8601 -> POSIX seconds. Naive timestamps are assumed UTC."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def round_timing(row: dict[str, Any]) -> tuple[float | None, float | None]:
    """``(submission time, api_time)`` in POSIX seconds / seconds.

    Submission time is the LATEST input event in the round: for a round fed by
    several parallel tool results, the request cannot have been sent before the
    last one landed. ``api_time`` is the span from submission to the last
    model-emitted event.

    TraceLab records no server-reported latency and no TTFT, so ``api_time`` is
    a derived proxy with no ground truth to check it against, not a recorded
    value.
    """
    events = row.get("timing_events") or []
    ins = [
        parse_ts(e["timestamp"])
        for e in events
        if e.get("event_type") in INPUT_EVENTS and e.get("timestamp")
    ]
    outs = [
        parse_ts(e["timestamp"])
        for e in events
        if e.get("event_type") in OUTPUT_EVENTS and e.get("timestamp")
    ]
    if not ins and not outs:
        return None, None
    submitted = max(ins) if ins else min(outs)
    if not outs:
        return submitted, None
    return submitted, max(max(outs) - submitted, 0.0)


def session_span(rows: list[dict[str, Any]]) -> tuple[float, float] | None:
    """Wall-clock extent of a whole session, across round events and tool calls."""
    stamps: list[float] = []
    for row in rows:
        for event in row.get("timing_events") or []:
            if event.get("timestamp"):
                stamps.append(parse_ts(event["timestamp"]))
        for tool in row.get("tools") or []:
            for key in ("emitted_at", "result_at"):
                if tool.get(key):
                    stamps.append(parse_ts(tool[key]))
    if not stamps:
        return None
    return min(stamps), max(stamps)


class HashIdMinter:
    """Block id allocator. Ids are unique positive ints within one trace.

    ``hash_id_scope: "local"`` is one namespace per trace, NOT per conversation:
    a subagent shares its parent trace's scope, so a block id reused across
    parent and child decodes to the same tokens and reproduces the real
    cross-agent shared prefix. One minter therefore spans a parent and every
    subagent nested inside it.
    """

    __slots__ = ("_next",)

    def __init__(self) -> None:
        self._next = 1

    def take(self, count: int) -> list[int]:
        if count <= 0:
            return []
        ids = list(range(self._next, self._next + count))
        self._next += count
        return ids


def synthesize_hash_ids(
    rows: list[dict[str, Any]], block_size: int, minter: HashIdMinter
) -> list[list[int]]:
    """Rebuild a per-round KV-cache block chain from TraceLab's prefix split.

    TraceLab records ``input_tokens_total = prefix_tokens + newly_append_tokens``,
    an engine-reported split of how much of this round's input was already
    resident. That becomes block ids by keeping the first
    ``prefix_tokens // block_size`` ids from the previous round and minting
    fresh ids for the rest of the input.

    ``hash_ids`` describes the request's ENTIRE input, not just its cached part,
    so the list is FLOOR(``input_tokens_total`` / ``block_size``) long: only
    whole blocks are hashed and ``input_length % block_size`` is carried as an
    unhashed partial tail. A ceil-length list is silently tolerated and wrong
    twice over, since the loader drops the trailing id as content while the
    cache-hit metric still counts it, and the extra id perturbs agent-chain
    detection. The cache hit is recovered downstream by taking the longest
    common prefix against the preceding request.

    Degrades correctly through context compaction: when the agent compacts, the
    engine reports a smaller ``prefix_tokens``, the reused span shrinks, and
    fresh ids are minted.

    The minter is passed in rather than created here so that a parent and its
    nested subagents share one id namespace (see :class:`HashIdMinter`).
    """
    prev: list[int] = []
    out: list[list[int]] = []
    for row in rows:
        total = max(int(row.get("input_tokens_total") or 0), 0)
        prefix = max(int(row.get("prefix_tokens") or 0), 0)
        # Never claim more prefix than the input actually holds.
        prefix = min(prefix, total)
        n_blocks = total // block_size
        n_reuse = min(prefix // block_size, len(prev), n_blocks)
        ids = prev[:n_reuse] + minter.take(n_blocks - n_reuse)
        out.append(ids)
        prev = ids
    return out


def order_rounds(
    rows: list[dict[str, Any]],
) -> list[tuple[float, float | None, dict[str, Any]]]:
    """Rows -> ``[(submitted, api_time, row)]`` in replay order; undated dropped.

    Ordered by SUBMISSION TIME first, with TraceLab's own ``round_index`` as the
    tie-break. Sorting by ``round_index`` alone is wrong in two ways that both
    bite:

    * A spawn turn must strictly precede its join turn, but spawn position is
      read by ARRAY INDEX while the join is placed by TIMESTAMP. Where the
      corpus's ``round_index`` disagrees with its timestamps the join lands
      ahead of the spawn, orchestrator validation raises, and the ENTIRE trace
      is discarded rather than degrading.
    * The KV-cache prefix chain evolves in wall-clock order, not in
      ``round_index`` order, so time ordering is also the more faithful basis
      for the hash-id reuse chain.

    Remaining ties fall back to file order, which python's stable sort
    preserves. Duplicate ``(session_id, round_index)`` pairs exist in the
    corpus and must not crash it.
    """
    timed: list[tuple[float, float | None, dict[str, Any]]] = []
    for row in rows:
        submitted, api = round_timing(row)
        if submitted is None:
            continue
        timed.append((submitted, api, row))
    timed.sort(key=lambda item: (item[0], item[2].get("round_index") or 0))
    return timed


@dataclass(frozen=True, kw_only=True, slots=True)
class Spawn:
    """One recovered parent -> child link."""

    parent_sid: str
    child_sid: str
    start: float
    end: float
    duration_ms: int
    kind: str


class JoinStats:
    """Counters describing one subagent-join pass, for logging."""

    __slots__ = (
        "ambiguous",
        "grandchildren",
        "matched",
        "matched_claude",
        "matched_codex",
        "windows",
        "windows_claude",
        "windows_codex",
        "windows_matched",
    )

    def __init__(self) -> None:
        self.windows = 0
        self.windows_claude = 0
        self.windows_codex = 0
        self.windows_matched = 0
        self.matched = 0
        self.matched_claude = 0
        self.matched_codex = 0
        self.ambiguous = 0
        self.grandchildren = 0

    def summary(self) -> str:
        barren = self.windows - self.windows_matched
        pct = f" ({barren / self.windows * 100:.1f}%)" if self.windows else ""
        return (
            f"{self.windows} spawn windows "
            f"({self.windows_claude} claude, {self.windows_codex} codex), "
            f"{self.matched} children matched "
            f"({self.matched_claude} claude, {self.matched_codex} codex), "
            f"{self.ambiguous} ambiguous (tightest window kept), "
            f"{barren} windows matched no child{pct}, "
            f"{self.grandchildren} grandchildren kept as standalone traces"
        )


def build_join_index(
    sessions: dict[str, list[dict[str, Any]]],
    min_spawn_ms: int = DEFAULT_MIN_SPAWN_MS,
    enable_codex: bool = True,
) -> tuple[dict[str, Spawn], JoinStats]:
    """Recover parent -> child subagent links. Returns ``(child_sid -> Spawn, stats)``.

    TraceLab carries NO explicit parent link: a subagent round is its own
    top-level ``session_id``, ``session_id`` is a flat ``<provider>:<uuid>``,
    ``trace_key`` is a row id, and no cross-reference to another session appears
    anywhere in the row.

    The link is recovered by TIMING CONTAINMENT. For a spawning tool call
    lasting at least ``min_spawn_ms``, a different session with the same
    ``user`` and ``project`` whose ENTIRE span falls inside
    ``[emitted_at, result_at]`` is taken to be that call's subagent. Where a
    child fits inside several candidate windows the TIGHTEST window wins, and
    the ambiguity is counted.

    This is a CONTAINMENT rate, not an accuracy: the corpus has no ground truth
    for the join, so the counters describe how often the rule fired, never
    whether an individual match is right.

    Two provider lifecycles, and they are not equally strong:

    * Claude (``Agent`` / ``Task``) blocks for the whole subagent run, so the
      tool call's own window IS the subagent's extent. This join is precise and
      attributes a child to a specific call.
    * Codex uses an async ``spawn_agent`` -> ``wait_agent`` -> ``close_agent``
      lifecycle. ``spawn_agent`` returns in milliseconds with a handle, and that
      handle lived in the tool ARGUMENTS, which the released corpus strips. A
      spawn cannot therefore be paired to its own wait, and the only usable
      window is the session-level ``[earliest spawn_agent, latest wait_agent
      result]``. That is COARSER: it cannot attribute a child to a specific
      spawn, and a session that fans out several agents collapses them all into
      one window. Counted separately, and separately disableable, for that
      reason.
    """
    span: dict[str, tuple[float, float]] = {}
    ident: dict[str, tuple[Any, Any]] = {}
    for sid, rows in sessions.items():
        extent = session_span(rows)
        if extent is None:
            continue
        span[sid] = extent
        ident[sid] = (rows[0].get("user"), rows[0].get("project"))

    # (start, end, parent_sid, duration_ms, kind)
    windows: list[tuple[float, float, str, int, str]] = []
    for sid, rows in sessions.items():
        if rows[0].get("provider") == "codex":
            if not enable_codex:
                continue
            window = _codex_window(rows)
            if window is not None:
                start, end = window
                windows.append((start, end, sid, int((end - start) * 1000), "codex"))
            continue
        windows.extend(_claude_windows(sid, rows, min_spawn_ms))

    by_ident: dict[tuple[Any, Any], list[tuple[float, float, str, int, str]]] = (
        defaultdict(list)
    )
    for win in windows:
        by_ident[ident.get(win[2], (None, None))].append(win)

    stats = JoinStats()
    stats.windows = len(windows)
    stats.windows_claude = sum(1 for w in windows if w[4] == "claude")
    stats.windows_codex = sum(1 for w in windows if w[4] == "codex")
    return _match_children(span, ident, by_ident, stats), stats


def _match_children(
    span: dict[str, tuple[float, float]],
    ident: dict[str, tuple[Any, Any]],
    by_ident: dict[tuple[Any, Any], list[tuple[float, float, str, int, str]]],
    stats: JoinStats,
) -> dict[str, Spawn]:
    """Assign each session to the tightest spawn window that contains it."""
    links: dict[str, Spawn] = {}
    hit_windows: set[tuple[float, float, str]] = set()
    for csid, (cstart, cend) in span.items():
        hits = [
            (wend - wstart, wstart, wend, psid, dur, kind)
            for (wstart, wend, psid, dur, kind) in by_ident.get(ident[csid], ())
            if psid != csid and cstart >= wstart and cend <= wend
        ]
        if not hits:
            continue
        hits.sort(key=lambda h: (h[0], h[1], h[3]))
        if len(hits) > 1:
            stats.ambiguous += 1
        _, wstart, wend, psid, dur, kind = hits[0]
        links[csid] = Spawn(
            parent_sid=psid,
            child_sid=csid,
            start=wstart,
            end=wend,
            duration_ms=dur,
            kind=kind,
        )
        hit_windows.add((wstart, wend, psid))
        stats.matched += 1
        if kind == "claude":
            stats.matched_claude += 1
        else:
            stats.matched_codex += 1
    stats.windows_matched = len(hit_windows)
    return links


def _codex_window(rows: list[dict[str, Any]]) -> tuple[float, float] | None:
    """Session-level ``[earliest spawn_agent, latest wait_agent result]``."""
    spawns: list[float] = []
    waits: list[float] = []
    for row in rows:
        for tool in row.get("tools") or []:
            name = tool.get("tool_name")
            if name == "spawn_agent" and tool.get("emitted_at"):
                spawns.append(parse_ts(tool["emitted_at"]))
            elif name == "wait_agent" and tool.get("result_at"):
                waits.append(parse_ts(tool["result_at"]))
    if not spawns or not waits:
        return None
    start, end = min(spawns), max(waits)
    return (start, end) if end > start else None


def _claude_windows(
    sid: str, rows: list[dict[str, Any]], min_spawn_ms: int
) -> Iterator[tuple[float, float, str, int, str]]:
    """One window per blocking spawning tool call of sufficient duration."""
    for row in rows:
        for tool in row.get("tools") or []:
            if tool.get("tool_name") not in CLAUDE_SPAWN_TOOLS:
                continue
            latency = tool.get("tool_wall_latency_ms")
            if latency is None or latency < min_spawn_ms:
                continue
            if not tool.get("emitted_at") or not tool.get("result_at"):
                continue
            yield (
                parse_ts(tool["emitted_at"]),
                parse_ts(tool["result_at"]),
                sid,
                int(latency),
                "claude",
            )


def subagent_type_for(models: list[str]) -> str:
    """Best available stand-in for the child's role.

    The real subagent type is NOT RECOVERABLE: it lived in the spawning tool
    call's arguments, and the released corpus strips tool arguments. Nothing in
    the row records which agent type was requested.

    ``WekaSubagentEntry.subagent_type`` is required by the schema but is not
    read during reconstruction, so emitting the child's model costs nothing at
    replay and is at least true, where inventing a role name would read as
    something that had been recovered.
    """
    return models[0] if models else "unknown"


def build_requests(
    timed: list[tuple[float, float | None, dict[str, Any]]],
    hash_chains: list[list[int]],
    t0: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Rounds -> Weka ``type: "n"`` requests, timestamped relative to ``t0``."""
    requests: list[dict[str, Any]] = []
    models: list[str] = []
    prev_end: float | None = None
    for (submitted, api, row), hash_ids in zip(timed, hash_chains, strict=True):
        model = row.get("model") or "unknown"
        if model not in models:
            models.append(model)

        t = submitted - t0
        # think_time is the client-side idle gap before this request: the span
        # from the previous request COMPLETING to this one being sent. For this
        # corpus that gap is the human reading and typing, plus the local tool
        # execution.
        think = None if prev_end is None else max(t - prev_end, 0.0)

        out_tokens = int(row.get("output_tokens") or 0)
        reasoning = row.get("reasoning_output_tokens")
        if reasoning:
            # There is no separate reasoning-token field in the trace schema;
            # reasoning is part of what the server has to generate, so it folds
            # into output_length.
            out_tokens += int(reasoning)

        input_types = (
            ["tool_result"]
            if row.get("first_input_event_type") == "tool_result"
            else ["text"]
        )
        # A round that emitted a tool_call ended because the model wanted a tool.
        emitted_tool = any(
            e.get("event_type") == "tool_call" for e in (row.get("timing_events") or [])
        )

        requests.append(
            {
                "t": round(t, 6),
                "type": "n",
                "model": model,
                "in": max(int(row.get("input_tokens_total") or 0), 1),
                "out": max(out_tokens, 1),
                "hash_ids": hash_ids,
                "input_types": input_types,
                "output_types": ["text"],
                "stop": "tool_use" if emitted_tool else "end_turn",
                "api_time": None if api is None else round(api, 6),
                "think_time": None if think is None else round(think, 6),
            }
        )
        prev_end = t + (api or 0.0)
    return requests, models


def build_subagent_entry(
    spawn: Spawn,
    *,
    child_rows: list[dict[str, Any]],
    block_size: int,
    minter: HashIdMinter,
    t0: float,
) -> dict[str, Any] | None:
    """One matched child session -> a ``type: "subagent"`` entry, or None."""
    timed = order_rounds(child_rows)
    if not timed:
        return None

    hash_chains = synthesize_hash_ids([item[2] for item in timed], block_size, minter)
    inner, models = build_requests(timed, hash_chains, t0)

    entry_t = spawn.start - t0
    # An inner timestamp EARLIER than the spawn marker is read downstream as
    # subagent-RELATIVE and rewritten to entry.t + req.t. That heuristic is
    # per-request and silent, so one inner request slipping below entry_t would
    # be flung far into the future while its siblings stayed put.
    #
    # The containment join makes that unreachable: a child only matches when
    # its whole span falls inside the spawn window, and the span mins over
    # timing events AND tool stamps while a request time is drawn from timing
    # events alone, so every request is at or after entry_t already. This is a
    # guard on that precondition, not a live transform. It stays because the
    # failure it prevents is silent and whole-trace, while the check is one
    # comparison per request, and any future join rule that is not
    # containment-based would need it.
    for req in inner:
        if req["t"] < entry_t:
            req["t"] = round(entry_t, 6)
    # think_time on the child's first request would be a gap measured from
    # nothing.
    inner[0]["think_time"] = None

    child_end = max(item[0] + (item[1] or 0.0) for item in timed)
    return {
        "t": round(entry_t, 6),
        "type": "subagent",
        # Unique within the trace: duplicate agent_ids among retained subagents
        # are rejected at load, and each child maps to exactly one parent.
        "agent_id": safe_trace_id(spawn.child_sid),
        "subagent_type": subagent_type_for(models),
        "duration_ms": spawn.duration_ms,
        "total_tokens": sum(r["in"] + r["out"] for r in inner),
        "tool_use_count": sum(len(row.get("tools") or []) for row in child_rows),
        "status": "completed" if child_end <= spawn.end else "incomplete",
        "requests": inner,
        "models": models,
        "tool_tokens": 0,
        "system_tokens": 0,
    }


def build_trace(
    session_id: str,
    rows: list[dict[str, Any]],
    block_size: int,
    children: dict[str, tuple[Spawn, list[dict[str, Any]]]] | None = None,
    *,
    placed_sids: set[str] | None = None,
) -> dict[str, Any] | None:
    """One TraceLab session (plus any recovered subagents) -> one trace dict."""
    timed = order_rounds(rows)
    if not timed:
        return None

    t0 = timed[0][0]
    # One namespace for the parent and every nested child (hash_id_scope local).
    minter = HashIdMinter()
    hash_chains = synthesize_hash_ids([item[2] for item in timed], block_size, minter)
    requests, models = build_requests(timed, hash_chains, t0)

    n_subagents = 0
    if children:
        requests, n_subagents = _merge_subagent_entries(
            requests=requests,
            models=models,
            children=children,
            block_size=block_size,
            minter=minter,
            t0=t0,
            placed_sids=placed_sids,
        )

    normals = [r for r in requests if r["type"] == "n"]
    return {
        "id": session_id,
        "models": models,
        "block_size": block_size,
        "hash_id_scope": "local",
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": requests,
        "totals": {
            "rounds": len(normals),
            "subagents": n_subagents,
            "input_tokens": sum(r["in"] for r in normals),
            "output_tokens": sum(r["out"] for r in normals),
            "source": SOURCE_TAG,
        },
    }


def _merge_subagent_entries(
    *,
    requests: list[dict[str, Any]],
    models: list[str],
    children: dict[str, tuple[Spawn, list[dict[str, Any]]]],
    block_size: int,
    minter: HashIdMinter,
    t0: float,
    placed_sids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Interleave subagent markers into a parent's request array.

    Each marker goes after the LAST parent request whose t is at or before the
    spawn. Anchoring to the emitting round's array position instead looks more
    direct but is wrong: spawn position is read by ARRAY INDEX while the join is
    placed by TIMESTAMP, so any disagreement between the two makes the join land
    ahead of the spawn, orchestrator validation raises, and the entire trace is
    discarded rather than degrading. Ordering by time keeps the request array
    monotonic in t, which is the condition that reconciles them.

    This satisfies the other placement rule for free: a subagent needs some
    top-level request at a lower array index or it is silently dropped. There is
    always at least one, because the spawning round was submitted before its own
    tool call was emitted.

    ``models`` is extended in place with any model only the children used.
    """
    req_times = [r["t"] for r in requests]
    pending: dict[int, list[dict[str, Any]]] = defaultdict(list)
    n_subagents = 0
    for csid, (spawn, child_rows) in children.items():
        entry = build_subagent_entry(
            spawn,
            child_rows=child_rows,
            block_size=block_size,
            minter=minter,
            t0=t0,
        )
        if entry is None:
            continue
        anchor = _anchor_index(req_times, entry["t"])
        if anchor is None:
            continue
        pending[anchor].append(entry)
        if placed_sids is not None:
            placed_sids.add(csid)
        for model in entry["models"]:
            if model not in models:
                models.append(model)
        n_subagents += 1

    if not pending:
        return requests, n_subagents

    merged: list[dict[str, Any]] = []
    for idx, req in enumerate(requests):
        merged.append(req)
        merged.extend(sorted(pending.get(idx, ()), key=lambda e: e["t"]))
    return merged, n_subagents


def _anchor_index(req_times: list[float], entry_t: float) -> int | None:
    """Index of the last request at or before ``entry_t``, or None."""
    anchor = None
    for i, rt in enumerate(req_times):
        if rt <= entry_t:
            anchor = i
    return anchor


def group_children_by_parent(
    sessions: dict[str, list[dict[str, Any]]],
    links: dict[str, Spawn],
    stats: JoinStats,
) -> dict[str, dict[str, tuple[Spawn, list[dict[str, Any]]]]]:
    """Bucket recovered children under their parents.

    The schema nests exactly ONE level: a subagent entry's inner requests are
    plain calls and cannot themselves carry a subagent marker. So a child is
    only nested when its parent is itself a root. A GRANDCHILD (whose parent is
    someone else's child) is kept as its own independent trace rather than being
    dropped, since losing it silently would discard exactly the deepest agentic
    structure this pass exists to recover.
    """
    out: dict[str, dict[str, tuple[Spawn, list[dict[str, Any]]]]] = defaultdict(dict)
    for csid, spawn in links.items():
        if spawn.parent_sid not in sessions or csid not in sessions:
            continue
        if spawn.parent_sid in links:
            stats.grandchildren += 1
            continue
        out[spawn.parent_sid][csid] = (spawn, sessions[csid])
    return out


def safe_trace_id(session_id: str) -> str:
    """Session id -> an identifier safe to use as a trace or agent id."""
    return _SAFE.sub("_", session_id)[:150]
