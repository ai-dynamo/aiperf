# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parallel reconstruction parity + structural tests for WekaTraceLoader.

Drives :func:`weka_parallel_convert._process_task` in-process (no real Pool)
so xdist-safe.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aiperf.dataset.loader import weka_parallel_convert as wpc
from aiperf.dataset.loader.weka_trace import WekaTraceLoader

FIXTURES = Path(__file__).parents[3] / "fixtures" / "weka_traces"


def _mk_user_config(model_names=None):
    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = None
    uc.input.fixed_schedule_end_offset = None
    uc.input.ignore_trace_delays = False
    uc.input.use_think_time_only = False
    uc.input.use_end_to_start_delays = False
    uc.loadgen.inter_turn_delay_cap_seconds = None
    uc.loadgen.trace_idle_gap_cap_seconds = None
    uc.input.synthesis.max_isl = None
    uc.input.synthesis.max_osl = None
    uc.input.max_context_length = None
    uc.input.synthesis.should_synthesize.return_value = False
    uc.input.prompt.input_tokens.block_size = None
    uc.tokenizer.trust_remote_code = False
    uc.tokenizer.revision = None
    uc.tokenizer.name = "test-tok"
    uc.endpoint.model_names = model_names or [
        "claude-opus-4-5-20251101",
        "claude-haiku-4-5-20251001",
    ]
    return uc


def _stub_loader(loader: WekaTraceLoader) -> None:
    """Wire a deterministic stubbed PromptGenerator the serial reconstructor needs.

    Mirrors the fixtures in test_weka_trace.py so the serial run is byte-exact
    reproducible without a real tokenizer.
    """
    from tests.unit.dataset.loader.conftest import stub_hash_id_corpus_rng

    pg = MagicMock()
    pg._cache = {}
    pg._sample_tokens.side_effect = lambda n: [0] * n
    pg._tokenized_corpus = list(range(10000, 11000))
    pg._corpus_size = 1000
    pg._bpe_stable_terminator_tokens = []
    stub_hash_id_corpus_rng(pg)
    pg.tokenizer.decode.side_effect = lambda toks: f"<dec:{len(toks)}>"
    pg._hash_id_corpus_rng.seed = 12345
    loader.prompt_generator = pg
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64


def _drive_parallel_inproc(
    loader: WekaTraceLoader, parent_plans, child_plans, data
) -> list:
    """Run :func:`_reconstruct_parallel` but with the worker pool replaced by
    in-process execution of :func:`_process_task`.

    Initializes :data:`weka_parallel_convert._worker_state` once with a real
    HashIdRandomGenerator (matching the serial path's seed) over a corpus
    matching the stubbed ``pg._tokenized_corpus``. Restores prior worker
    state at end so other tests aren't affected.
    """
    from multiprocessing import shared_memory

    pg = loader.prompt_generator
    corpus_arr = np.array(pg._tokenized_corpus, dtype=np.int32)
    corpus_len = len(corpus_arr)
    shm = shared_memory.SharedMemory(
        create=True, size=corpus_len * np.dtype(np.int32).itemsize
    )
    np.ndarray((corpus_len,), dtype=np.int32, buffer=shm.buf)[:] = corpus_arr

    saved_state = wpc._worker_state
    try:
        with patch(
            "aiperf.dataset.loader.weka_parallel_convert.Tokenizer.from_pretrained",
            return_value=pg.tokenizer,
        ):
            args = wpc._WekaWorkerInitArgs(
                shm_name=shm.name,
                corpus_len=corpus_len,
                tokenizer_name="test-tok",
                base_seed=pg._hash_id_corpus_rng.seed,
                block_size=loader._block_size,
                bpe_stable_terminator_tokens=[],
            )
            wpc._init_worker(args)

        # Build tasks via the same helper code _reconstruct_parallel uses,
        # then call _process_task on each.
        ignore_delays = loader.user_config.input.ignore_trace_delays
        think_time_only = loader.user_config.input.use_think_time_only
        cap_seconds = loader.user_config.loadgen.inter_turn_delay_cap_seconds

        from collections import defaultdict

        metric_values_by_trace = loader._build_shared_metric_values(
            parent_plans, child_plans
        )

        children_by_trace = defaultdict(list)
        sids_by_subagent: dict[tuple[str, int], list[str]] = defaultdict(list)
        for cp in child_plans:
            child_metric_values = metric_values_by_trace[cp.parent_trace_id]
            requests_dicts = [
                {
                    "hash_ids": list(creq.hash_ids),
                    "input_length": creq.input_length,
                    "output_length": creq.output_length,
                    "model": creq.model,
                    "t": creq.t,
                    "think_time": getattr(creq, "think_time", None),
                    # Dropped children have no pre-pass entry; they are
                    # skipped by _process_task so the fallback is unused.
                    "theoretical_hit_blocks": child_metric_values.get(
                        (cp.session_id, k), (0, 0)
                    )[0],
                    "theoretical_total_blocks": child_metric_values.get(
                        (cp.session_id, k), (0, len(creq.hash_ids))
                    )[1],
                }
                for k, creq in enumerate(cp.requests)
            ]
            children_by_trace[cp.parent_trace_id].append(
                {
                    "session_id": cp.session_id,
                    "parent_trace_id": cp.parent_trace_id,
                    "subagent_index": cp.subagent_index,
                    "agent_id": cp.entry.agent_id,
                    "tool_tokens": cp.init_tool_tokens,
                    "system_tokens": cp.init_system_tokens,
                    "requests": requests_dicts,
                }
            )
            sids_by_subagent[(cp.parent_trace_id, cp.subagent_index)].append(
                cp.session_id
            )

        results = []
        for plan in parent_plans:
            trace = data[plan.trace_id][0]
            normals_dicts = [
                (
                    outer_idx,
                    {
                        "hash_ids": list(req.hash_ids),
                        "input_length": req.input_length,
                        "output_length": req.output_length,
                        "model": req.model,
                        "t": req.t,
                        "think_time": getattr(req, "think_time", None),
                        "capped_output_length": loader._cap_output(req),
                        "theoretical_hit_blocks": metric_values_by_trace[plan.trace_id][
                            (plan.trace_id, k)
                        ][0],
                        "theoretical_total_blocks": metric_values_by_trace[
                            plan.trace_id
                        ][(plan.trace_id, k)][1],
                    },
                )
                for k, (outer_idx, req) in enumerate(plan.normals)
            ]
            subagents_dicts = []
            for sa_index, (outer_idx, sa) in enumerate(plan.subagents):
                child_sids = sids_by_subagent.get((plan.trace_id, sa_index), [])
                if sa.duration_ms is not None:
                    sa_end = sa.t + sa.duration_ms / 1000.0
                elif sa.requests:
                    sa_end = max(ir.t + (ir.api_time or 0.0) for ir in sa.requests)
                else:
                    sa_end = sa.t
                subagents_dicts.append(
                    (
                        outer_idx,
                        {
                            "agent_id": sa.agent_id,
                            "tool_tokens": sa.tool_tokens,
                            "system_tokens": sa.system_tokens,
                            "child_session_ids": child_sids,
                            "sa_end_seconds": sa_end,
                            "t": sa.t,
                        },
                    )
                )
            task = wpc._WekaTraceTask(
                trace_id=plan.trace_id,
                parent={
                    "normals": normals_dicts,
                    "subagents": subagents_dicts,
                    "tool_tokens": trace.tool_tokens,
                    "system_tokens": trace.system_tokens,
                },
                children=children_by_trace.get(plan.trace_id, []),
                cap_seconds=cap_seconds,
                ignore_delays=ignore_delays,
                think_time_only=think_time_only,
                model_map=loader._build_model_map(trace),
                block_size=loader._block_size_for_trace(trace),
            )
            results.append(wpc._process_task(task))
        return results
    finally:
        wpc._worker_state = saved_state
        shm.close()
        shm.unlink()


def _make_stub_pg_with_real_rng(corpus_size: int = 1000):
    """A pg whose RNG is real (so serial + parallel both reseed identically)."""
    from aiperf.common.hash_id_random_generator import HashIdRandomGenerator

    pg = MagicMock()
    pg._cache = {}
    pg._tokenized_corpus = list(range(10000, 10000 + corpus_size))
    pg._corpus_size = corpus_size
    pg._bpe_stable_terminator_tokens = []
    pg._hash_id_corpus_rng = HashIdRandomGenerator(12345, _internal=True)
    pg.tokenizer.decode.side_effect = lambda toks: f"<dec:{len(toks)}>"
    return pg


def _build_plans(loader: WekaTraceLoader, data: dict) -> tuple:
    """Re-derive parent_plans/child_plans/dropped_per_trace the way
    convert_to_conversations does, since both serial and parallel helpers
    consume them as inputs."""
    from dataclasses import dataclass

    from aiperf.dataset.loader.weka_trace import _expand_subagent_to_child_plans
    from aiperf.dataset.loader.weka_trace_models import (
        WekaNormalRequest,
        WekaStreamingRequest,
    )

    @dataclass
    class _ParentPlan:
        trace_id: str
        normals: list
        subagents: list
        block_size: int

    parent_plans: list = []
    child_plans: list = []

    for trace_id, wekas in data.items():
        trace = wekas[0]
        trace_bs = loader._block_size_for_trace(trace)
        normals = []
        subagents = []
        for idx, req in enumerate(trace.requests):
            if isinstance(req, WekaNormalRequest | WekaStreamingRequest):
                if not loader._request_passes_filters(req):
                    continue
                normals.append((idx, req))
            else:
                sa_index = len(subagents)
                subagents.append((idx, req))
                child_plans.extend(
                    _expand_subagent_to_child_plans(trace_id, sa_index, req, trace_bs)
                )
        parent_plans.append(_ParentPlan(trace_id, normals, subagents, trace_bs))

    return parent_plans, child_plans, {}


def _stub_loader_real_rng(loader: WekaTraceLoader) -> None:
    """Like _stub_loader but with a real HashIdRandomGenerator instance.

    The serial path uses ``loader.prompt_generator._hash_id_corpus_rng`` to
    pick block content via ``set_trace_id`` + ``reseed_for_hash_id``. The
    parallel path also uses a fresh real RNG seeded from
    ``pg._hash_id_corpus_rng.seed``. Both must end up at byte-identical
    outputs when run with the same trace_id scope.
    """
    pg = _make_stub_pg_with_real_rng(corpus_size=1000)
    loader.prompt_generator = pg
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64


def test_parallel_byte_equivalence_simple_fixture(tmp_path):
    """Parallel raw_messages == serial raw_messages on the simple fixture."""
    # Load + parse once per loader since the file traversal is non-pure.
    serial_loader = WekaTraceLoader(
        filename=str(FIXTURES / "simple.json"), user_config=_mk_user_config()
    )
    _stub_loader_real_rng(serial_loader)
    data = serial_loader.load_dataset()

    parent_plans, child_plans, dropped_per_trace = _build_plans(serial_loader, data)

    from aiperf.common.enums import (
        ConversationBranchMode,
        ConversationContextMode,
        PrerequisiteKind,
    )
    from aiperf.common.models import (
        Conversation,
        ConversationBranchInfo,
        Turn,
        TurnPrerequisite,
    )

    serial_convs = serial_loader._reconstruct_serial(
        parent_plans=parent_plans,
        child_plans=child_plans,
        data=data,
        dropped_per_trace=dropped_per_trace,
        ignore_delays=False,
        think_time_only=False,
        cap_seconds=None,
        t_start=0.0,
        model_map_per_trace={
            tid: serial_loader._build_model_map(wekas[0]) for tid, wekas in data.items()
        },
        trace_idle_timing_by_trace={},
        metric_values_by_trace=serial_loader._build_shared_metric_values(
            parent_plans, child_plans
        ),
    )

    # Parallel path: drive _process_task in-process to get reconstruction
    # results, then assemble Conversations the same way _reconstruct_parallel does.
    parallel_results = _drive_parallel_inproc(
        serial_loader, parent_plans, child_plans, data
    )

    # Reassemble into Conversation list (mirroring _reconstruct_parallel tail).
    parallel_convs = []
    for result in parallel_results:
        trace_id = result["trace_id"]
        parent_conv = Conversation(
            session_id=trace_id,
            context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES,
        )
        for t in result["parent_turns"]:
            parent_conv.turns.append(
                Turn(
                    timestamp=t["timestamp"],
                    delay=t["delay"],
                    model=t["model"],
                    max_tokens=t["max_tokens"],
                    raw_messages=t["raw_messages"],
                    reset_context=t["reset_context"],
                )
            )
        for branch in result["branches"]:
            parent_conv.branches.append(
                ConversationBranchInfo(
                    branch_id=branch["branch_id"],
                    child_conversation_ids=branch["child_session_ids"],
                    mode=ConversationBranchMode.SPAWN,
                    is_background=branch["is_background"],
                )
            )
            parent_conv.turns[branch["preceding_turn"]].branch_ids.append(
                branch["branch_id"]
            )
            if branch["following_turn"] is not None:
                parent_conv.turns[branch["following_turn"]].prerequisites.append(
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN,
                        branch_id=branch["branch_id"],
                    )
                )
        parallel_convs.append(parent_conv)
        for child in result["children"]:
            child_conv = Conversation(
                session_id=child["session_id"],
                context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES,
            )
            for t in child["turns"]:
                child_conv.turns.append(
                    Turn(
                        timestamp=t["timestamp"],
                        delay=t["delay"],
                        model=t["model"],
                        max_tokens=t["max_tokens"],
                        raw_messages=t["raw_messages"],
                        reset_context=t["reset_context"],
                    )
                )
            parallel_convs.append(child_conv)

    assert len(serial_convs) == len(parallel_convs)
    for sc, pc in zip(serial_convs, parallel_convs, strict=True):
        assert sc.session_id == pc.session_id
        assert len(sc.turns) == len(pc.turns)
        for k, (st, pt) in enumerate(zip(sc.turns, pc.turns, strict=True)):
            assert st.timestamp == pt.timestamp, (
                f"{sc.session_id} turn {k}: timestamp drift"
            )
            assert st.delay == pt.delay, f"{sc.session_id} turn {k}: delay drift"
            assert st.max_tokens == pt.max_tokens
            assert st.model == pt.model
            assert st.raw_messages == pt.raw_messages, (
                f"{sc.session_id} turn {k}: raw_messages drift\n"
                f"  serial:   {st.raw_messages!r}\n"
                f"  parallel: {pt.raw_messages!r}"
            )


def test_parallel_byte_equivalence_with_subagent(tmp_path):
    """Parallel path matches serial on a fixture with a subagent."""
    serial_loader = WekaTraceLoader(
        filename=str(FIXTURES / "one_subagent.json"), user_config=_mk_user_config()
    )
    _stub_loader_real_rng(serial_loader)
    data = serial_loader.load_dataset()

    parent_plans, child_plans, dropped_per_trace = _build_plans(serial_loader, data)

    serial_convs = serial_loader._reconstruct_serial(
        parent_plans=parent_plans,
        child_plans=child_plans,
        data=data,
        dropped_per_trace=dropped_per_trace,
        ignore_delays=False,
        think_time_only=False,
        cap_seconds=None,
        t_start=0.0,
        model_map_per_trace={
            tid: serial_loader._build_model_map(wekas[0]) for tid, wekas in data.items()
        },
        trace_idle_timing_by_trace={},
        metric_values_by_trace=serial_loader._build_shared_metric_values(
            parent_plans, child_plans
        ),
    )
    parallel_results = _drive_parallel_inproc(
        serial_loader, parent_plans, child_plans, data
    )

    # Quick sanity: subagent fixture has parent + child conversation = 2 results,
    # parallel results contains 1 parent result with 1 child embedded.
    serial_session_ids = {c.session_id for c in serial_convs}
    parallel_session_ids = set()
    for r in parallel_results:
        parallel_session_ids.add(r["trace_id"])
        for ch in r["children"]:
            parallel_session_ids.add(ch["session_id"])
    assert serial_session_ids == parallel_session_ids

    # Parent raw_messages parity
    serial_by_sid = {c.session_id: c for c in serial_convs}
    for result in parallel_results:
        sc = serial_by_sid[result["trace_id"]]
        for k, t in enumerate(result["parent_turns"]):
            assert sc.turns[k].raw_messages == t["raw_messages"], (
                f"{result['trace_id']} turn {k}: parent raw_messages drift"
            )
        for child in result["children"]:
            csc = serial_by_sid[child["session_id"]]
            for k, t in enumerate(child["turns"]):
                assert csc.turns[k].raw_messages == t["raw_messages"], (
                    f"{child['session_id']} turn {k}: child raw_messages drift"
                )


def test_parallel_threshold_falls_back_to_serial(monkeypatch):
    """N < threshold -> serial path (no Pool spawn).

    We verify by setting the threshold above the trace count and asserting
    weka_parallel_convert.run_parallel_weka_reconstruction is never called.
    """
    from aiperf.common import environment as env_mod

    serial_loader = WekaTraceLoader(
        filename=str(FIXTURES / "simple.json"), user_config=_mk_user_config()
    )
    _stub_loader(serial_loader)

    monkeypatch.setattr(env_mod.Environment.DATASET, "WEKA_PARALLEL_THRESHOLD", 100)

    called = {"hit": False}

    def boom(*a, **kw):
        called["hit"] = True
        raise AssertionError("parallel path should not run when N < threshold")

    monkeypatch.setattr(
        "aiperf.dataset.loader.weka_parallel_convert.run_parallel_weka_reconstruction",
        boom,
    )

    data = serial_loader.load_dataset()
    convs = serial_loader.convert_to_conversations(data)
    assert convs, "expected at least one conversation from serial path"
    assert not called["hit"]


def test_parallel_workers_one_disables_parallel(monkeypatch):
    """WEKA_PARALLEL_WORKERS=1 forces the serial path."""
    from aiperf.common import environment as env_mod

    serial_loader = WekaTraceLoader(
        filename=str(FIXTURES / "simple.json"), user_config=_mk_user_config()
    )
    _stub_loader(serial_loader)

    monkeypatch.setattr(env_mod.Environment.DATASET, "WEKA_PARALLEL_THRESHOLD", 1)
    monkeypatch.setattr(env_mod.Environment.DATASET, "WEKA_PARALLEL_WORKERS", 1)

    called = {"hit": False}

    def boom(*a, **kw):
        called["hit"] = True
        raise AssertionError("parallel path should not run when WORKERS=1")

    monkeypatch.setattr(
        "aiperf.dataset.loader.weka_parallel_convert.run_parallel_weka_reconstruction",
        boom,
    )

    data = serial_loader.load_dataset()
    convs = serial_loader.convert_to_conversations(data)
    assert convs
    assert not called["hit"]


def test_worker_scope_helpers_deterministic_per_trace_id(tmp_path):
    """Helpers in two scopes produce different content for the same hash_id."""
    from multiprocessing import shared_memory

    corpus = list(range(10000, 11000))
    corpus_arr = np.array(corpus, dtype=np.int32)
    shm = shared_memory.SharedMemory(create=True, size=corpus_arr.nbytes)
    np.ndarray((len(corpus),), dtype=np.int32, buffer=shm.buf)[:] = corpus_arr
    saved_state = wpc._worker_state
    try:
        with patch(
            "aiperf.dataset.loader.weka_parallel_convert.Tokenizer.from_pretrained",
            return_value=MagicMock(decode=lambda toks: f"<dec:{len(toks)}>"),
        ):
            args = wpc._WekaWorkerInitArgs(
                shm_name=shm.name,
                corpus_len=len(corpus),
                tokenizer_name="test-tok",
                base_seed=99,
                block_size=64,
                bpe_stable_terminator_tokens=[],
            )
            wpc._init_worker(args)

        decode_a, _, _ = wpc._make_scope_helpers("scope-a", 64)
        decode_b, _, _ = wpc._make_scope_helpers("scope-b", 64)
        toks_a = decode_a([42])
        toks_b = decode_b([42])
        assert toks_a != toks_b, (
            "different scopes must produce different content for the same hash_id"
        )

        # Determinism: re-running with same scope yields identical content.
        decode_a2, _, _ = wpc._make_scope_helpers("scope-a", 64)
        toks_a2 = decode_a2([42])
        assert toks_a == toks_a2
    finally:
        wpc._worker_state = saved_state
        shm.close()
        shm.unlink()


def test_directory_with_multiple_traces_parallel_path_byte_exact(tmp_path):
    """Multi-trace directory: parallel reconstruction matches serial across files."""
    src_files = ["simple.json", "one_subagent.json", "terminal_subagent.json"]
    traces_dir = tmp_path / "weka"
    traces_dir.mkdir()
    for name in src_files:
        shutil.copy(FIXTURES / name, traces_dir / name)

    serial_loader = WekaTraceLoader(
        filename=str(traces_dir), user_config=_mk_user_config()
    )
    _stub_loader_real_rng(serial_loader)
    data = serial_loader.load_dataset()

    parent_plans, child_plans, dropped_per_trace = _build_plans(serial_loader, data)

    serial_convs = serial_loader._reconstruct_serial(
        parent_plans=parent_plans,
        child_plans=child_plans,
        data=data,
        dropped_per_trace=dropped_per_trace,
        ignore_delays=False,
        think_time_only=False,
        cap_seconds=None,
        t_start=0.0,
        model_map_per_trace={
            tid: serial_loader._build_model_map(wekas[0]) for tid, wekas in data.items()
        },
        trace_idle_timing_by_trace={},
        metric_values_by_trace=serial_loader._build_shared_metric_values(
            parent_plans, child_plans
        ),
    )

    parallel_results = _drive_parallel_inproc(
        serial_loader, parent_plans, child_plans, data
    )

    serial_by_sid = {c.session_id: c for c in serial_convs}

    for result in parallel_results:
        sc = serial_by_sid[result["trace_id"]]
        for k, t in enumerate(result["parent_turns"]):
            assert sc.turns[k].raw_messages == t["raw_messages"], (
                f"{result['trace_id']} turn {k} parent raw_messages drift"
            )
        for child in result["children"]:
            csc = serial_by_sid[child["session_id"]]
            for k, t in enumerate(child["turns"]):
                assert csc.turns[k].raw_messages == t["raw_messages"], (
                    f"{child['session_id']} turn {k} child raw_messages drift"
                )


@pytest.mark.parametrize("n_traces", [1, 3])
def test_parallel_path_handles_small_trace_counts(tmp_path, n_traces):
    """Parallel path executes cleanly for N=1 and N=3 traces."""
    src_files = ["simple.json", "one_subagent.json", "terminal_subagent.json"][
        :n_traces
    ]
    traces_dir = tmp_path / "weka"
    traces_dir.mkdir()
    for name in src_files:
        shutil.copy(FIXTURES / name, traces_dir / name)

    loader = WekaTraceLoader(filename=str(traces_dir), user_config=_mk_user_config())
    _stub_loader_real_rng(loader)
    data = loader.load_dataset()
    parent_plans, child_plans, _ = _build_plans(loader, data)

    parallel_results = _drive_parallel_inproc(loader, parent_plans, child_plans, data)
    assert len(parallel_results) == n_traces
    for r in parallel_results:
        assert r["parent_turns"], f"{r['trace_id']}: empty parent_turns"


def test_fanout_split_parallel_byte_identical_to_serial(monkeypatch):
    """Flat-chain splitting must be byte-identical across both paths.

    Runs the FULL convert_to_conversations twice — serial (workers=1) and
    parallel (threshold=1, the pool replaced by an in-process map over
    _process_task) — so the real task builder and assembly are exercised.
    """
    from multiprocessing import shared_memory

    import aiperf.common.environment as env_mod

    fanout = FIXTURES.parent / "weka_traces_fanout" / "fanout.json"

    def _serial_convs():
        monkeypatch.setattr(env_mod.Environment.DATASET, "WEKA_PARALLEL_WORKERS", 1)
        loader = WekaTraceLoader(filename=str(fanout), user_config=_mk_user_config())
        _stub_loader_real_rng(loader)
        return loader.convert_to_conversations(loader.load_dataset())

    def _parallel_convs():
        monkeypatch.setattr(env_mod.Environment.DATASET, "WEKA_PARALLEL_WORKERS", 2)
        monkeypatch.setattr(env_mod.Environment.DATASET, "WEKA_PARALLEL_THRESHOLD", 1)
        loader = WekaTraceLoader(filename=str(fanout), user_config=_mk_user_config())
        _stub_loader_real_rng(loader)
        pg = loader.prompt_generator

        corpus_arr = np.array(pg._tokenized_corpus, dtype=np.int32)
        shm = shared_memory.SharedMemory(
            create=True, size=len(corpus_arr) * np.dtype(np.int32).itemsize
        )
        np.ndarray((len(corpus_arr),), dtype=np.int32, buffer=shm.buf)[:] = corpus_arr
        saved_state = wpc._worker_state
        try:
            with patch(
                "aiperf.dataset.loader.weka_parallel_convert.Tokenizer.from_pretrained",
                return_value=pg.tokenizer,
            ):
                wpc._init_worker(
                    wpc._WekaWorkerInitArgs(
                        shm_name=shm.name,
                        corpus_len=len(corpus_arr),
                        tokenizer_name="test-tok",
                        base_seed=pg._hash_id_corpus_rng.seed,
                        block_size=loader._block_size,
                        bpe_stable_terminator_tokens=[],
                    )
                )

            def _inproc_pool(tasks, **_kwargs):
                return [wpc._process_task(t) for t in tasks]

            monkeypatch.setattr(wpc, "run_parallel_weka_reconstruction", _inproc_pool)
            return loader.convert_to_conversations(loader.load_dataset())
        finally:
            wpc._worker_state = saved_state
            shm.close()
            shm.unlink()

    serial_convs = _serial_convs()
    parallel_convs = _parallel_convs()

    assert [c.session_id for c in serial_convs] == [
        c.session_id for c in parallel_convs
    ]
    for sc, pc in zip(serial_convs, parallel_convs, strict=True):
        assert sc.is_root == pc.is_root, sc.session_id
        assert sc.agent_depth == pc.agent_depth, sc.session_id
        assert sc.parent_conversation_id == pc.parent_conversation_id, sc.session_id
        s_branches = [
            (b.branch_id, b.child_conversation_ids, b.is_background)
            for b in sc.branches
        ]
        p_branches = [
            (b.branch_id, b.child_conversation_ids, b.is_background)
            for b in pc.branches
        ]
        assert s_branches == p_branches, sc.session_id
        for k, (st, pt) in enumerate(zip(sc.turns, pc.turns, strict=True)):
            assert st.timestamp == pt.timestamp, f"{sc.session_id} turn {k}"
            assert st.delay == pt.delay, f"{sc.session_id} turn {k}"
            assert st.max_tokens == pt.max_tokens, f"{sc.session_id} turn {k}"
            assert st.model == pt.model, f"{sc.session_id} turn {k}"
            assert st.branch_ids == pt.branch_ids, f"{sc.session_id} turn {k}"
            assert [p.branch_id for p in st.prerequisites] == [
                p.branch_id for p in pt.prerequisites
            ], f"{sc.session_id} turn {k}"
            assert st.reset_context == pt.reset_context, f"{sc.session_id} turn {k}"
            assert (
                st.theoretical_prefix_cache_hit_blocks
                == pt.theoretical_prefix_cache_hit_blocks
            ), f"{sc.session_id} turn {k}"
            assert st.raw_messages == pt.raw_messages, (
                f"{sc.session_id} turn {k}: raw_messages drift\n"
                f"  serial:   {st.raw_messages!r}\n"
                f"  parallel: {pt.raw_messages!r}"
            )
