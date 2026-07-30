# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-trace parallel reconstruction for WekaTraceLoader.

Each Weka trace (one parent + zero or more subagent children) is a
self-contained reconstruction unit: one trace-scoped cache and
HashIdRandomGenerator shared by the parent and all its children
(``hash_id_scope: "local"`` = one namespace per trace file), plus
scope-keyed partial-tail seeds. The byte-exact
LCP-driven reconstruction in
:class:`aiperf.dataset.loader.weka_synth_buf.ConversationReconstructor`
carries cross-turn state, but never cross-trace state.

Output is byte-identical to the in-process serial path; tests in
``test_weka_trace_parallel.py`` assert this against the serial loader.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import os
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import NotRequired, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset._mp_context import get_loader_mp_context
from aiperf.dataset.loader._delay_cap import DelayCapTracker

_JOIN_EPSILON_SECONDS = 1e-6


def _api_time_ms(api_time: float | None) -> float | None:
    """Per-turn server-processing duration in ms (mirrors weka_trace._api_time_ms).

    Kept local to avoid a circular import (weka_trace imports this module)."""
    if api_time is None or not math.isfinite(api_time):
        return None
    return max(0.0, api_time) * 1000.0


def _end_to_start_delay_ms(
    start_to_start_ms: float | None, prev_api_seconds: float | None
) -> float | None:
    """Idle inter-turn delay ``t_k - (t_{k-1} + api_{k-1})`` (mirrors
    weka_trace._end_to_start_delay_ms). Kept local to avoid a circular import."""
    if start_to_start_ms is None:
        return None
    api_ms = (
        prev_api_seconds * 1000.0
        if prev_api_seconds is not None and math.isfinite(prev_api_seconds)
        else 0.0
    )
    return max(0.0, start_to_start_ms - api_ms)


class _WekaParentTurnDict(TypedDict):
    """One reconstructed turn (parent or child) shipped from worker -> orchestrator."""

    timestamp: float | None
    delay: float | None
    api_time_ms: float | None
    source_trace_id: str | None
    source_outer_idx: int | None
    source_inner_idx: int | None
    source_kind: str | None
    model: str
    max_tokens: int
    prompt: str
    raw_messages: list[dict[str, str]]
    reset_context: bool
    theoretical_prefix_cache_hit_blocks: int
    theoretical_prefix_cache_total_blocks: int
    input_kind: str | None


class _WekaBranchDict(TypedDict):
    """Subagent SPAWN branch metadata for one tiered join bucket."""

    branch_id: str
    child_session_ids: list[str]
    is_background: bool
    preceding_turn: int
    following_turn: int | None
    start_timestamp: float | None


class _WekaChildDict(TypedDict):
    """One reconstructed subagent conversation."""

    session_id: str
    parent_conversation_id: str
    turns: list[_WekaParentTurnDict]
    is_root: bool
    agent_depth: int


class _WekaProcessTaskResult(TypedDict):
    """Per-trace reconstruction output from `_process_task`."""

    trace_id: str
    parent_turns: list[_WekaParentTurnDict]
    branches: list[_WekaBranchDict]
    children: list[_WekaChildDict]
    dropped_agent_ids: list[str]
    capped_count: int
    max_observed_ms: float
    non_finite_count: int


class _WekaNormalRequestPayload(TypedDict):
    """Wire-format dict for one normal/streaming request, parent or child."""

    hash_ids: list[int]
    input_length: int
    output_length: int
    model: str
    t: float
    think_time: float | None
    api_time: NotRequired[float | None]
    # Turn classification computed in the orchestrator (one source of truth in
    # weka_trace._classify_turn_input); workers copy it into the turn dict.
    input_kind: NotRequired[str | None]
    source_trace_id: NotRequired[str | None]
    source_outer_idx: NotRequired[int | None]
    source_inner_idx: NotRequired[int | None]
    source_kind: NotRequired[str | None]
    # Only present in parent normals (not in child requests):
    capped_output_length: NotRequired[int]
    # Theoretical prefix-cache values precomputed parent-side from the
    # per-trace shared seen-set (one hash namespace per trace file).
    theoretical_hit_blocks: int
    theoretical_total_blocks: int


class _WekaSubagentMarkerPayload(TypedDict):
    """Wire-format dict for one subagent marker (in parent.subagents).

    Chain detection happens in the parent process (parity with the serial
    path): ``child_session_ids`` enumerates the per-chain child SIDs the
    worker must register on the SPAWN branch (single-chain subagents emit
    one ``::sa:{agent_id}`` SID; detected spawned chains add ``:fa:000`` /
    ``:aux:000`` / ... siblings -- agents and one-shot sidecars). ``sa_end_seconds`` is the subagent's recorded
    end time, used by the worker to select the first later parent turn that
    should join this child.
    """

    agent_id: str
    tool_tokens: int
    system_tokens: int
    child_session_ids: list[str]
    sa_end_seconds: float
    t: float


class _WekaFlatChainMarkerPayload(TypedDict):
    """Branch-anchoring info for one detected flat chain (spec §5.3)."""

    session_id: str
    chain_index: int
    first_outer_idx: int
    end_seconds: float
    t: float


class _WekaParentPayload(TypedDict):
    """Per-trace parent payload shipped to a worker."""

    normals: list[tuple[int, _WekaNormalRequestPayload]]
    subagents: list[tuple[int, _WekaSubagentMarkerPayload]]
    tool_tokens: int
    system_tokens: int
    flat_markers: NotRequired[list[_WekaFlatChainMarkerPayload]]


class _WekaChildPayload(TypedDict):
    """Per-subagent child payload shipped to a worker.

    ``tool_tokens`` / ``system_tokens`` are the chain's gated turn-0
    attribution values (the main chain keeps the entry's declared counts;
    spawned chains carry them only when their first request provably starts
    with the same declared-prefix blocks -- see
    ``weka_trace._expand_subagent_to_child_plans``), not necessarily the raw
    entry-declared counts. Child request ``t`` values are in root-trace
    coordinates.
    """

    session_id: str
    parent_trace_id: str
    subagent_index: int
    agent_id: str
    tool_tokens: int
    system_tokens: int
    requests: list[_WekaNormalRequestPayload]


_DecodeBlocksFn: TypeAlias = Callable[[list[int]], list[int]]
_SamplePartialTailFn: TypeAlias = Callable[[int, str], list[int]]
_DecodeTokensFn: TypeAlias = Callable[[list[int]], str]


@dataclass(slots=True)
class _WekaWorkerInitArgs:
    """Static args passed to each Pool worker via initargs."""

    shm_name: str
    corpus_len: int
    tokenizer_name: str
    base_seed: int
    block_size: int
    bpe_stable_terminator_tokens: list[int]
    trust_remote_code: bool = False
    revision: str = "main"


@dataclass(slots=True)
class _WekaTraceTask:
    """Per-trace payload shipped to a worker.

    Holds the parsed parent trace plus its subagent children so the worker
    can run reconstruction without touching any PromptGenerator state from
    the main process. Prompts are synthesized inside the worker via the same
    hash-id-seeded RNG and sha256-keyed partial-tail primitives the LCP
    reconstructor uses, so no parent-side ``parallel_decode`` phase is
    needed.

    ``model_map`` rewrites the trace's per-request ``model`` field to the
    run's configured ``endpoint.model_names``. Built per-trace in the parent
    process so workers don't need ``UserConfig``.

    ``block_size`` is per-trace (real Weka captures declare their own
    ``block_size`` per file; the parent process resolves
    user-override > trace-declared > 64 before shipping the task here).
    """

    trace_id: str
    parent: _WekaParentPayload
    children: list[_WekaChildPayload]
    cap_seconds: float | None
    ignore_delays: bool
    think_time_only: bool
    model_map: dict[str, str]
    block_size: int
    tool_shaped_messages: bool = False


@dataclass(slots=True)
class _WekaWorkerState:
    tokenizer: Tokenizer
    corpus: np.ndarray
    corpus_size: int
    shm: shared_memory.SharedMemory
    base_seed: int
    block_size: int
    bpe_stable_terminator_tokens: list[int]


_worker_state: _WekaWorkerState | None = None


def _init_worker(args: _WekaWorkerInitArgs) -> None:
    """Worker init: attach corpus shared memory + load tokenizer from cache."""
    global _worker_state

    from aiperf.dataset.loader.parallel_convert import _install_hard_exit_on_sigterm

    _install_hard_exit_on_sigterm()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    shm = shared_memory.SharedMemory(name=args.shm_name)
    corpus = np.ndarray((args.corpus_len,), dtype=np.int32, buffer=shm.buf)

    from aiperf.dataset._tokenizer_preload import get_preloaded

    tokenizer = get_preloaded(
        args.tokenizer_name,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    if tokenizer is None:
        tokenizer = Tokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            resolve_alias=False,
        )

    _worker_state = _WekaWorkerState(
        tokenizer=tokenizer,
        corpus=corpus,
        corpus_size=int(corpus.shape[0]),
        shm=shm,
        base_seed=args.base_seed,
        block_size=args.block_size,
        bpe_stable_terminator_tokens=list(args.bpe_stable_terminator_tokens),
    )


def _make_scope_helpers(
    scope: str,
    block_size: int,
) -> tuple[_DecodeBlocksFn, _SamplePartialTailFn, _DecodeTokensFn]:
    """Return (decode_block_tokens, sample_partial_tail_tokens, decode_tokens_to_text)
    bound to a fresh per-scope cache + RNG.

    ``block_size`` is per-trace (the parent process resolves
    user-override > trace-declared > 64 before shipping the task to the
    worker; see ``WekaTraceLoader._block_size_for_trace``). The closure
    captures it so multiple traces processed by the same worker can use
    different block sizes.
    """
    assert _worker_state is not None
    state = _worker_state
    bs = block_size
    corpus = state.corpus
    corpus_size = state.corpus_size

    rng = HashIdRandomGenerator(state.base_seed, _internal=True)
    rng.set_trace_id(scope)
    cache: dict[int, list[int]] = {}

    def decode_block_tokens(hash_ids: list[int]) -> list[int]:
        out: list[int] = []
        for h in hash_ids:
            cached = cache.get(h)
            if cached is None:
                rng.reseed_for_hash_id(h)
                start = rng.randrange(corpus_size)
                end = start + bs
                if end <= corpus_size:
                    cached = list(corpus[start:end])
                else:
                    cached = list(corpus[start:end]) + list(corpus[: end - corpus_size])
                cache[h] = cached
            out.extend(cached)
        return out

    def sample_partial_tail_tokens(n_tokens: int, seed: str) -> list[int]:
        if n_tokens <= 0:
            return []
        digest = hashlib.sha256(seed.encode()).digest()
        offset = int.from_bytes(digest[:8], "big") % max(corpus_size - n_tokens, 1)
        return list(corpus[offset : offset + n_tokens])

    def decode_tokens_to_text(tokens: list[int]) -> str:
        return state.tokenizer.decode(tokens)

    return decode_block_tokens, sample_partial_tail_tokens, decode_tokens_to_text


def _process_task(task: _WekaTraceTask) -> _WekaProcessTaskResult:
    """Reconstruct one parent trace + its subagent children.

    We return a dict (not Conversation) because Pydantic model unpickling
    is more expensive than dict unpickling and the parent-side wire-up is
    trivial.
    """
    assert _worker_state is not None
    from aiperf.dataset.loader.weka_synth_buf import (
        ConversationReconstructor,
        compute_asst_block_caps,
    )

    state = _worker_state
    bs = task.block_size
    cap_seconds = task.cap_seconds
    delay_tracker = DelayCapTracker(cap_seconds=cap_seconds)

    parent = task.parent
    parent_decode, parent_partial, parent_decode_text = _make_scope_helpers(
        task.trace_id, bs
    )

    parent_recon = ConversationReconstructor(
        block_size=bs,
        decode_block_tokens=parent_decode,
        sample_partial_tail_tokens=parent_partial,
        decode_tokens_to_text=parent_decode_text,
        bpe_stable_terminator_tokens=state.bpe_stable_terminator_tokens,
        tool_shaped_messages=task.tool_shaped_messages,
    )

    parent_turns: list[_WekaParentTurnDict] = []
    outer_to_turn_pos: dict[int, int] = {}
    normals: list[tuple[int, _WekaNormalRequestPayload]] = parent["normals"]
    parent_asst_block_caps = compute_asst_block_caps(
        [(r["hash_ids"], r["input_length"]) for _, r in normals],
        bs,
    )
    for k, (outer_idx, req) in enumerate(normals):
        seed = f"{task.trace_id}:turn_{k}:partial_tail"
        is_tool_result = req.get("input_kind") == "tool_result"
        if k == 0:
            parent_recon.init_turn_0(
                hash_ids=req["hash_ids"],
                in_tokens=req["input_length"],
                tool_tokens=parent["tool_tokens"],
                system_tokens=parent["system_tokens"],
                seed=seed,
                is_tool_result=is_tool_result,
            )
        else:
            prev_req = normals[k - 1][1]
            parent_recon.advance_turn(
                prev_hash_ids=prev_req["hash_ids"],
                prev_in_tokens=prev_req["input_length"],
                prev_out_tokens=prev_req["output_length"],
                curr_hash_ids=req["hash_ids"],
                curr_in_tokens=req["input_length"],
                seed=seed,
                is_tool_result=is_tool_result,
                max_asst_blocks=parent_asst_block_caps[k],
            )

        t_ms = req["t"] * 1000.0
        if k == 0:
            delay_ms = None
        elif task.think_time_only and req.get("think_time") is not None:
            delay_ms = req["think_time"] * 1000.0
        else:
            prev_req = normals[k - 1][1]
            delay_ms = _end_to_start_delay_ms(
                t_ms - prev_req["t"] * 1000.0, prev_req.get("api_time")
            )
        if delay_ms is not None:
            delay_ms = delay_tracker.clamp(delay_ms)
            # Floor at 0, mirroring the serial parent loop in
            # weka_trace._reconstruct_serial: a negative inter-turn delay (corrupt
            # think_time, or a non-monotonic main-chain timestamp gap) would tell
            # the load generator to dispatch a request in the past. Omitting it
            # here made the parallel parent path emit a negative Turn.delay where
            # the serial path emits 0.0, breaking the module's byte-identical
            # serial/parallel contract. (Child / flat-chain paths intentionally
            # do not floor in either path, so they stay in parity untouched.)
            # Clamp maps non-finite delays to None — skip the floor.
            if delay_ms is not None:
                delay_ms = max(delay_ms, 0.0)

        parent_delta = parent_recon.turn_delta()
        theoretical_hit_blocks = req["theoretical_hit_blocks"]
        theoretical_total_blocks = req["theoretical_total_blocks"]
        parent_turns.append(
            {
                "timestamp": None if task.ignore_delays else t_ms,
                "delay": None if task.ignore_delays else delay_ms,
                "api_time_ms": None
                if task.ignore_delays
                else _api_time_ms(req.get("api_time")),
                "source_trace_id": task.trace_id,
                "source_outer_idx": outer_idx,
                "source_inner_idx": None,
                "source_kind": "weka_main",
                "model": task.model_map.get(req["model"], req["model"]),
                "max_tokens": req["capped_output_length"],
                "raw_messages": parent_delta.delta_messages,
                "reset_context": parent_delta.reset_context,
                "theoretical_prefix_cache_hit_blocks": theoretical_hit_blocks,
                "theoretical_prefix_cache_total_blocks": theoretical_total_blocks,
                "input_kind": req.get("input_kind"),
            }
        )
        outer_to_turn_pos[outer_idx] = len(parent_turns) - 1

    # Subagent grouping: spawning parent turn plus computed join turn. This
    # mirrors the serial loader and preserves tiered joins for mixed-duration
    # sibling subagents.
    #
    # Examples:
    #   parent[0] t=0
    #   subagent A ends t=6
    #   subagent B ends t=12.5
    #   subagent C ends t=24
    #   parent[1] t=6
    #   parent[2] t=20
    #
    #   A joins parent[1] because parent[1].t >= A.end.
    #   B joins parent[2] because parent[1].t < B.end <= parent[2].t.
    #   C is background because no later parent turn reaches C.end.
    #
    # Additional examples:
    #   Shared join group:
    #     parent[0] t=0
    #     subagent A ends t=4
    #     subagent B ends t=5
    #     parent[1] t=6
    #     => A and B share group (parent[0], parent[1]); parent[1]
    #        waits for both.
    #
    #   Tiered siblings:
    #     parent[0] t=0
    #     subagent A ends t=4
    #     subagent B ends t=9
    #     parent[1] t=6
    #     parent[2] t=12
    #     => A gates parent[1]; B keeps running through parent[1] and
    #        gates parent[2].
    #
    #   No spawning parent:
    #     subagent A marker t=1 appears before the first retained
    #     parent turn
    #     parent[0] t=5
    #     => A is dropped because no parent turn can spawn it.
    #
    #   Equality joins:
    #     parent[0] t=0
    #     subagent A ends t=10
    #     parent[1] t=10
    #     => A joins parent[1] within _JOIN_EPSILON_SECONDS.
    groups: dict[tuple[int, int | None], list[_WekaSubagentMarkerPayload]] = (
        defaultdict(list)
    )
    group_order: list[tuple[int, int | None]] = []
    outer_to_t: dict[int, float] = {oi: req["t"] for oi, req in normals}
    dropped_agent_ids: set[str] = set()
    dropped_subagent_indices: set[int] = set()
    for subagent_index, (sa_outer_idx, sa_entry) in enumerate(parent["subagents"]):
        preceding = max(
            (pos for oi, pos in outer_to_turn_pos.items() if oi < sa_outer_idx),
            default=None,
        )
        if preceding is None:
            dropped_agent_ids.add(sa_entry["agent_id"])
            dropped_subagent_indices.add(subagent_index)
            continue

        join_turn: int | None = None
        for oi, pos in sorted(outer_to_turn_pos.items()):
            if oi <= sa_outer_idx:
                continue
            sa_end_seconds = sa_entry["sa_end_seconds"]
            if outer_to_t[oi] + _JOIN_EPSILON_SECONDS >= sa_end_seconds:
                join_turn = pos
                break

        key = (preceding, join_turn)
        if key not in groups:
            group_order.append(key)
        groups[key].append(sa_entry)

    branches: list[_WekaBranchDict] = []
    for preceding, join_turn in group_order:
        entries = groups[(preceding, join_turn)]
        child_sids: list[str] = []
        for e in entries:
            child_sids.extend(e["child_session_ids"])
        is_background = join_turn is None
        branches.append(
            {
                "branch_id": f"{task.trace_id}:spawn:{entries[0]['agent_id']}",
                "child_session_ids": child_sids,
                "is_background": is_background,
                "preceding_turn": preceding,
                "following_turn": join_turn,
                "start_timestamp": min(e["t"] for e in entries) * 1000.0,
            }
        )

    # Detected flat chains: SPAWN off the last main turn preceding the
    # chain's first request (turn 0 fallback — never drop real load),
    # SPAWN_JOIN on the first later main turn at/after the chain's end.
    # Mirrors the serial path's grouping exactly.
    flat_groups: dict[tuple[int, int | None], list[_WekaFlatChainMarkerPayload]] = (
        defaultdict(list)
    )
    flat_group_order: list[tuple[int, int | None]] = []
    for marker in parent.get("flat_markers", []):
        first_outer = marker["first_outer_idx"]
        preceding = max(
            (pos for oi, pos in outer_to_turn_pos.items() if oi < first_outer),
            default=0,
        )
        fp_end = marker["end_seconds"]
        join_turn = None
        for oi, pos in sorted(outer_to_turn_pos.items()):
            if oi <= first_outer:
                continue
            if outer_to_t[oi] + _JOIN_EPSILON_SECONDS >= fp_end:
                join_turn = pos
                break
        key = (preceding, join_turn)
        if key not in flat_groups:
            flat_group_order.append(key)
        flat_groups[key].append(marker)

    for preceding, join_turn in flat_group_order:
        markers = flat_groups[(preceding, join_turn)]
        branches.append(
            {
                "branch_id": (f"{task.trace_id}:flatspawn:{markers[0]['chain_index']}"),
                "child_session_ids": [m["session_id"] for m in markers],
                "is_background": join_turn is None,
                "preceding_turn": preceding,
                "following_turn": join_turn,
                "start_timestamp": min(m["t"] for m in markers) * 1000.0,
            }
        )

    children_out: list[_WekaChildDict] = []
    for cp in task.children:
        if cp["subagent_index"] in dropped_subagent_indices:
            continue

        # Subagents share the parent trace's ``hash_id_scope: "local"``
        # namespace (see _reconstruct_serial): scope on parent_trace_id, not
        # the child session_id, so shared blocks decode identically.
        child_decode, child_partial, child_decode_text = _make_scope_helpers(
            cp["parent_trace_id"], bs
        )
        child_recon = ConversationReconstructor(
            block_size=bs,
            decode_block_tokens=child_decode,
            sample_partial_tail_tokens=child_partial,
            decode_tokens_to_text=child_decode_text,
            bpe_stable_terminator_tokens=state.bpe_stable_terminator_tokens,
            tool_shaped_messages=task.tool_shaped_messages,
        )

        child_turns: list[_WekaParentTurnDict] = []
        creqs: list[_WekaNormalRequestPayload] = cp["requests"]
        child_asst_block_caps = compute_asst_block_caps(
            [(r["hash_ids"], r["input_length"]) for r in creqs],
            bs,
        )
        for k, creq in enumerate(creqs):
            seed = f"{cp['session_id']}:turn_{k}:partial_tail"
            is_tool_result = creq.get("input_kind") == "tool_result"
            if k == 0:
                child_recon.init_turn_0(
                    hash_ids=creq["hash_ids"],
                    in_tokens=creq["input_length"],
                    tool_tokens=cp["tool_tokens"],
                    system_tokens=cp["system_tokens"],
                    seed=seed,
                    is_tool_result=is_tool_result,
                )
            else:
                prev_creq = creqs[k - 1]
                child_recon.advance_turn(
                    prev_hash_ids=prev_creq["hash_ids"],
                    prev_in_tokens=prev_creq["input_length"],
                    prev_out_tokens=prev_creq["output_length"],
                    curr_hash_ids=creq["hash_ids"],
                    curr_in_tokens=creq["input_length"],
                    seed=seed,
                    is_tool_result=is_tool_result,
                    max_asst_blocks=child_asst_block_caps[k],
                )
            t_ms = creq["t"] * 1000.0
            if k == 0:
                child_delay_ms = None
            elif task.think_time_only and creq.get("think_time") is not None:
                child_delay_ms = creq["think_time"] * 1000.0
            else:
                prev_creq = creqs[k - 1]
                child_delay_ms = _end_to_start_delay_ms(
                    t_ms - prev_creq["t"] * 1000.0, prev_creq.get("api_time")
                )
            if child_delay_ms is not None:
                child_delay_ms = delay_tracker.clamp(child_delay_ms)

            child_delta = child_recon.turn_delta()
            theoretical_hit_blocks = creq["theoretical_hit_blocks"]
            theoretical_total_blocks = creq["theoretical_total_blocks"]
            child_turns.append(
                {
                    "timestamp": None if task.ignore_delays else t_ms,
                    "delay": None if task.ignore_delays else child_delay_ms,
                    "api_time_ms": None
                    if task.ignore_delays
                    else _api_time_ms(creq.get("api_time")),
                    "source_trace_id": creq.get(
                        "source_trace_id", cp["parent_trace_id"]
                    ),
                    "source_outer_idx": creq.get("source_outer_idx"),
                    "source_inner_idx": creq.get("source_inner_idx"),
                    "source_kind": creq.get("source_kind", "weka_subagent"),
                    "model": task.model_map.get(creq["model"], creq["model"]),
                    # Flat-chain children carry capped_output_length (their
                    # rows were top-level and honor --max-osl); subagent
                    # children keep the recorded output_length. Clamp 0→1
                    # so Turn.max_tokens (ge=1) accepts aborted captures.
                    "max_tokens": max(
                        1,
                        creq.get("capped_output_length", creq["output_length"]),
                    ),
                    "raw_messages": child_delta.delta_messages,
                    "reset_context": child_delta.reset_context,
                    "theoretical_prefix_cache_hit_blocks": theoretical_hit_blocks,
                    "theoretical_prefix_cache_total_blocks": theoretical_total_blocks,
                    "input_kind": creq.get("input_kind"),
                }
            )
        children_out.append(
            {
                "session_id": cp["session_id"],
                "parent_conversation_id": cp["parent_trace_id"],
                "turns": child_turns,
                "is_root": False,
                "agent_depth": 1,
            }
        )

    return {
        "trace_id": task.trace_id,
        "parent_turns": parent_turns,
        "branches": branches,
        "children": children_out,
        "dropped_agent_ids": list(dropped_agent_ids),
        "capped_count": delay_tracker.capped_count,
        "max_observed_ms": delay_tracker.max_observed_ms,
        "non_finite_count": delay_tracker.non_finite_count,
    }


def _drive_reconstruction_pool(
    pool: mp.pool.Pool, tasks: list[_WekaTraceTask]
) -> list[_WekaProcessTaskResult]:
    """Run ``_process_task`` across the pool with periodic progress logs.

    ``chunksize=1`` for proper work-stealing on the heavy-tail corpus (max
    trace ~29x median tokenize cost). Submission order is preserved so the
    result stream stays byte-identical to the serial path (parity tests in
    ``tests/integration/dataset/test_weka_parallel_heavy.py``).
    """
    from aiperf.common.aiperf_logger import AIPerfLogger as _ALogger

    log = _ALogger(__name__)
    n_tasks = len(tasks)
    log_every = max(1, n_tasks // 10)
    results: list[_WekaProcessTaskResult] = []
    t_start = time.monotonic()
    for i, res in enumerate(pool.imap(_process_task, tasks, chunksize=1), 1):
        results.append(res)
        if i == n_tasks or i % log_every == 0:
            elapsed = time.monotonic() - t_start
            rate = i / elapsed if elapsed > 0 else 0.0
            pct = 100.0 * i / n_tasks
            log.info(
                f"WekaTraceLoader: reconstructed "
                f"{i}/{n_tasks} ({pct:.0f}%) "
                f"in {elapsed:.1f}s ({rate:.1f} traces/s)"
            )
    return results


def run_parallel_weka_reconstruction(
    tasks: list[_WekaTraceTask],
    *,
    tokenizer_name: str,
    corpus: NDArray[np.int32] | list[int],
    base_seed: int,
    block_size: int,
    bpe_stable_terminator_tokens: list[int],
    trust_remote_code: bool = False,
    revision: str = "main",
    num_workers: int,
) -> list[_WekaProcessTaskResult]:
    """Run :func:`_process_task` for every task across ``num_workers`` processes.

    Returns reconstruction-result dicts in the same order as ``tasks``.
    """
    from aiperf.dataset.loader.parallel_convert import (
        _POOL_JOIN_TIMEOUT_S,
        _ensure_valid_stdio_fds,
        _set_daemon,
        _shutdown_pool,
    )

    _ensure_valid_stdio_fds()

    corpus_len = len(corpus)
    corpus_arr = np.ascontiguousarray(corpus, dtype=np.int32)
    shm = shared_memory.SharedMemory(
        create=True, size=corpus_len * np.dtype(np.int32).itemsize
    )
    try:
        np.ndarray((corpus_len,), dtype=np.int32, buffer=shm.buf)[:] = corpus_arr

        init_args = _WekaWorkerInitArgs(
            shm_name=shm.name,
            corpus_len=corpus_len,
            tokenizer_name=tokenizer_name,
            base_seed=base_seed,
            block_size=block_size,
            bpe_stable_terminator_tokens=bpe_stable_terminator_tokens,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )

        was_daemon = mp.current_process().daemon
        try:
            if was_daemon:
                _set_daemon(False)
            ctx = get_loader_mp_context(
                preload_tokenizer=tokenizer_name,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )
            pool = ctx.Pool(num_workers, _init_worker, (init_args,))
            try:
                results = _drive_reconstruction_pool(pool, tasks)
            finally:
                # See ``_shutdown_pool`` for why ``terminate()`` would wedge
                # on weka workers' rayon-threaded HF tokenizer.
                _shutdown_pool(pool, timeout_s=_POOL_JOIN_TIMEOUT_S)
        finally:
            if was_daemon:
                _set_daemon(True)
        return results
    finally:
        shm.close()
        with suppress(FileNotFoundError):
            shm.unlink()
