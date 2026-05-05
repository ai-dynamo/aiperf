# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-trace parallel reconstruction for WekaTraceLoader.

Each Weka trace (one parent + zero or more subagent children) is a
self-contained reconstruction unit: scope-keyed cache, scope-keyed
HashIdRandomGenerator, scope-keyed partial-tail seed. The byte-exact
LCP-driven reconstruction in
:class:`aiperf.dataset.loader.weka_synth_buf.ConversationReconstructor`
carries cross-turn state, but never cross-trace state.

Output is byte-identical to the in-process serial path; tests in
``test_weka_trace_parallel.py`` assert this against the serial loader.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired

from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset._mp_context import get_loader_mp_context
from aiperf.dataset.loader._delay_cap import DelayCapTracker


class _WekaParentTurnDict(TypedDict):
    """One reconstructed turn (parent or child) shipped from worker -> orchestrator."""

    timestamp: float | None
    delay: float | None
    model: str
    max_tokens: int
    prompt: str
    raw_messages: list[dict[str, str]]
    reset_context: bool


class _WekaBranchDict(TypedDict):
    """Subagent SPAWN branch metadata for one (preceding, following) anchor pair."""

    branch_id: str
    child_session_ids: list[str]
    is_background: bool
    preceding_turn: int
    following_turn: int | None


class _WekaChildDict(TypedDict):
    """One reconstructed subagent conversation."""

    session_id: str
    turns: list[_WekaParentTurnDict]


class _WekaProcessTaskResult(TypedDict):
    """Per-trace reconstruction output from `_process_task`."""

    trace_id: str
    parent_turns: list[_WekaParentTurnDict]
    branches: list[_WekaBranchDict]
    children: list[_WekaChildDict]
    dropped_agent_ids: list[str]
    capped_count: int
    max_observed_ms: float


class _WekaNormalRequestPayload(TypedDict):
    """Wire-format dict for one normal/streaming request, parent or child."""

    hash_ids: list[int]
    input_length: int
    output_length: int
    model: str
    t: float
    think_time: float | None
    # Only present in parent normals (not in child requests):
    capped_output_length: NotRequired[int]


class _WekaSubagentMarkerPayload(TypedDict):
    """Wire-format dict for one subagent marker (in parent.subagents)."""

    agent_id: str
    tool_tokens: int
    system_tokens: int


class _WekaParentPayload(TypedDict):
    """Per-trace parent payload shipped to a worker."""

    normals: list[tuple[int, _WekaNormalRequestPayload]]
    subagents: list[tuple[int, _WekaSubagentMarkerPayload]]
    tool_tokens: int
    system_tokens: int


class _WekaChildPayload(TypedDict):
    """Per-subagent child payload shipped to a worker."""

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
    """

    trace_id: str
    parent: _WekaParentPayload
    children: list[_WekaChildPayload]
    cap_seconds: float | None
    ignore_delays: bool
    think_time_only: bool
    model_map: dict[str, str]
    emit_assistant_segments: bool = True


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
) -> tuple[_DecodeBlocksFn, _SamplePartialTailFn, _DecodeTokensFn]:
    """Return (decode_block_tokens, sample_partial_tail_tokens, decode_tokens_to_text)
    bound to a fresh per-scope cache + RNG.
    """
    assert _worker_state is not None
    state = _worker_state
    bs = state.block_size
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
    )

    state = _worker_state
    bs = state.block_size
    cap_seconds = task.cap_seconds
    delay_tracker = DelayCapTracker(cap_seconds=cap_seconds)

    parent = task.parent
    parent_decode, parent_partial, parent_decode_text = _make_scope_helpers(
        task.trace_id
    )

    parent_recon = ConversationReconstructor(
        block_size=bs,
        decode_block_tokens=parent_decode,
        sample_partial_tail_tokens=parent_partial,
        decode_tokens_to_text=parent_decode_text,
        bpe_stable_terminator_tokens=state.bpe_stable_terminator_tokens,
        emit_assistant_segments=task.emit_assistant_segments,
    )

    parent_turns: list[_WekaParentTurnDict] = []
    outer_to_turn_pos: dict[int, int] = {}
    normals: list[tuple[int, _WekaNormalRequestPayload]] = parent["normals"]
    for k, (outer_idx, req) in enumerate(normals):
        seed = f"{task.trace_id}:turn_{k}:partial_tail"
        if k == 0:
            parent_recon.init_turn_0(
                hash_ids=req["hash_ids"],
                in_tokens=req["input_length"],
                tool_tokens=parent["tool_tokens"],
                system_tokens=parent["system_tokens"],
                seed=seed,
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
            )

        t_ms = req["t"] * 1000.0
        if k == 0:
            delay_ms: float | None = None
        elif task.think_time_only and req.get("think_time") is not None:
            delay_ms = req["think_time"] * 1000.0
        else:
            delay_ms = t_ms - normals[k - 1][1]["t"] * 1000.0
        if delay_ms is not None:
            delay_ms = delay_tracker.clamp(delay_ms)

        parent_delta = parent_recon.turn_delta()
        parent_turns.append(
            {
                "timestamp": None if task.ignore_delays else t_ms,
                "delay": None if task.ignore_delays else delay_ms,
                "model": task.model_map.get(req["model"], req["model"]),
                "max_tokens": req["capped_output_length"],
                "raw_messages": parent_delta.delta_messages,
                "reset_context": parent_delta.reset_context,
            }
        )
        outer_to_turn_pos[outer_idx] = len(parent_turns) - 1

    # Subagent grouping: anchor pair (preceding parent turn pos, following parent turn pos).
    groups: dict[tuple[int | None, int | None], list[_WekaSubagentMarkerPayload]] = (
        defaultdict(list)
    )
    group_order: list[tuple[int | None, int | None]] = []
    dropped_agent_ids: set[str] = set()
    for sa_outer_idx, sa_entry in parent["subagents"]:
        preceding = max(
            (pos for oi, pos in outer_to_turn_pos.items() if oi < sa_outer_idx),
            default=None,
        )
        following = min(
            (pos for oi, pos in outer_to_turn_pos.items() if oi > sa_outer_idx),
            default=None,
        )
        if preceding is None:
            dropped_agent_ids.add(sa_entry["agent_id"])
            continue
        key = (preceding, following)
        if key not in groups:
            group_order.append(key)
        groups[key].append(sa_entry)

    branches: list[_WekaBranchDict] = []
    for preceding, following in group_order:
        entries = groups[(preceding, following)]
        child_sids = [f"{task.trace_id}::sa:{e['agent_id']}" for e in entries]
        branch_id = f"{task.trace_id}:spawn:{entries[0]['agent_id']}"
        branches.append(
            {
                "branch_id": branch_id,
                "child_session_ids": child_sids,
                "is_background": following is None,
                "preceding_turn": preceding,
                "following_turn": following,
            }
        )

    children_out: list[_WekaChildDict] = []
    for cp in task.children:
        if cp["agent_id"] in dropped_agent_ids:
            continue

        child_decode, child_partial, child_decode_text = _make_scope_helpers(
            cp["session_id"]
        )
        child_recon = ConversationReconstructor(
            block_size=bs,
            decode_block_tokens=child_decode,
            sample_partial_tail_tokens=child_partial,
            decode_tokens_to_text=child_decode_text,
            bpe_stable_terminator_tokens=state.bpe_stable_terminator_tokens,
            emit_assistant_segments=task.emit_assistant_segments,
        )

        child_turns: list[_WekaParentTurnDict] = []
        creqs: list[_WekaNormalRequestPayload] = cp["requests"]
        for k, creq in enumerate(creqs):
            seed = f"{cp['session_id']}:turn_{k}:partial_tail"
            if k == 0:
                child_recon.init_turn_0(
                    hash_ids=creq["hash_ids"],
                    in_tokens=creq["input_length"],
                    tool_tokens=cp["tool_tokens"],
                    system_tokens=cp["system_tokens"],
                    seed=seed,
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
                )
            t_ms = creq["t"] * 1000.0
            if k == 0:
                child_delay_ms: float | None = None
            elif task.think_time_only and creq.get("think_time") is not None:
                child_delay_ms = creq["think_time"] * 1000.0
            else:
                child_delay_ms = t_ms - creqs[k - 1]["t"] * 1000.0
            if child_delay_ms is not None:
                child_delay_ms = delay_tracker.clamp(child_delay_ms)

            child_delta = child_recon.turn_delta()
            child_turns.append(
                {
                    "timestamp": None if task.ignore_delays else t_ms,
                    "delay": None if task.ignore_delays else child_delay_ms,
                    "model": task.model_map.get(creq["model"], creq["model"]),
                    "max_tokens": creq["output_length"],
                    "raw_messages": child_delta.delta_messages,
                    "reset_context": child_delta.reset_context,
                }
            )
        children_out.append(
            {
                "session_id": cp["session_id"],
                "turns": child_turns,
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
    }


def _drive_reconstruction_pool(
    pool, tasks: list[_WekaTraceTask]
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
        shm.unlink()
