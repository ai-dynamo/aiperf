# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial serial-vs-parallel byte-parity tests for flattened-agent splitting.

Spec: ~/.aiperf/docs/superpowers/specs/2026-06-10-weka-flattened-agent-lcp-detection-design.md
(goal 5: "Deterministic, byte-identical across the serial and parallel
reconstruction paths").

Each test crafts a hostile trace shape (idle-gap warp mid-chain, mixed
split/no-split directories, subagent + flat-chain interleaving, compaction
seams inside the main chain, nonce-poisoned hashes, max-osl caps, cross-model
worker batches, effective-prefix overrides, epsilon-exact join boundaries),
runs the FULL ``convert_to_conversations`` through both reconstruction paths
(the worker pool replaced by an in-process map over ``_process_task``, per the
pattern in test_weka_trace_parallel.test_fanout_split_parallel_byte_identical_to_serial),
and compares the outputs field by field.
"""

from __future__ import annotations

from multiprocessing import shared_memory
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import orjson
import pytest

import aiperf.common.environment as env_mod
from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
from aiperf.common.models import Conversation
from aiperf.dataset.loader import weka_parallel_convert as wpc
from aiperf.dataset.loader.weka_trace import WekaTraceLoader

# ---------------------------------------------------------------------------
# Config / loader helpers (conventions copied from test_weka_trace_parallel.py)
# ---------------------------------------------------------------------------


def _mk_user_config(
    *,
    model_names: list[str] | None = None,
    idle_gap_cap_seconds: float | None = None,
    think_time_only: bool = False,
    end_to_start_delays: bool = False,
    ignore_delays: bool = False,
    max_osl: int | None = None,
    inter_turn_delay_cap_seconds: float | None = None,
):
    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = None
    uc.input.fixed_schedule_end_offset = None
    uc.input.ignore_trace_delays = ignore_delays
    uc.input.use_think_time_only = think_time_only
    uc.input.use_end_to_start_delays = end_to_start_delays
    uc.loadgen.inter_turn_delay_cap_seconds = inter_turn_delay_cap_seconds
    uc.loadgen.trace_idle_gap_cap_seconds = idle_gap_cap_seconds
    uc.input.synthesis.max_isl = None
    uc.input.synthesis.max_osl = max_osl
    uc.input.max_context_length = None
    uc.input.synthesis.should_synthesize.return_value = False
    uc.input.prompt.input_tokens.block_size = None
    uc.tokenizer.trust_remote_code = False
    uc.tokenizer.revision = None
    uc.tokenizer.name = "test-tok"
    uc.endpoint.model_names = (
        model_names if model_names is not None else ["served-a", "served-b"]
    )
    return uc


def _stub_loader_real_rng(loader: WekaTraceLoader) -> None:
    """Stub pg with a REAL HashIdRandomGenerator so both paths reseed identically."""
    pg = MagicMock()
    pg._cache = {}
    pg._tokenized_corpus = list(range(10000, 11000))
    pg._corpus_size = 1000
    pg._bpe_stable_terminator_tokens = []
    pg._hash_id_corpus_rng = HashIdRandomGenerator(12345, _internal=True)
    pg.tokenizer.decode.side_effect = lambda toks: f"<dec:{len(toks)}>"
    loader.prompt_generator = pg
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64


# ---------------------------------------------------------------------------
# Trace builders
# ---------------------------------------------------------------------------


def _nreq(
    t: float,
    hash_ids: list[int],
    *,
    model: str = "m",
    api_time: float | None = 1.0,
    out: int = 10,
    think_time: float | None = None,
    in_tokens: int | None = None,
    typ: str = "n",
    ttft: float | None = None,
) -> dict[str, Any]:
    req: dict[str, Any] = {
        "t": t,
        "type": typ,
        "model": model,
        "in": in_tokens if in_tokens is not None else len(hash_ids) * 64,
        "out": out,
        "hash_ids": hash_ids,
    }
    if api_time is not None:
        req["api_time"] = api_time
    if think_time is not None:
        req["think_time"] = think_time
    if ttft is not None:
        req["ttft"] = ttft
    return req


def _trace(
    trace_id: str,
    requests: list[dict[str, Any]],
    *,
    models: list[str] | None = None,
    tool_tokens: int = 0,
    system_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "id": trace_id,
        "models": models or ["m"],
        "block_size": 64,
        "hash_id_scope": "local",
        "tool_tokens": tool_tokens,
        "system_tokens": system_tokens,
        "requests": requests,
    }


def _fanout_requests(offset: int = 0, **kw: Any) -> list[dict[str, Any]]:
    """The known-good fan-out shape (mirrors fixtures/weka_traces_fanout):
    main chain 3 turns, worker fa:000 2 turns, worker fa:001 1 turn."""
    o = offset
    return [
        _nreq(0.0, [o + 1, o + 2, o + 3], api_time=1.0, **kw),
        _nreq(2.0, [o + 1, o + 2, o + 50, o + 51], api_time=6.0, **kw),
        _nreq(2.5, [o + 1, o + 2, o + 60, o + 61], api_time=4.0, **kw),
        _nreq(9.0, [o + 1, o + 2, o + 3, o + 4, o + 5], api_time=1.0, **kw),
        _nreq(8.5, [o + 1, o + 2, o + 50, o + 51, o + 52], api_time=1.0, **kw),
        _nreq(12.0, [o + 1, o + 2, o + 3, o + 4, o + 5, o + 6], api_time=1.0, **kw),
    ]


def _write_traces(tmp_path: Path, traces: list[dict[str, Any]]) -> Path:
    d = tmp_path / "traces"
    d.mkdir(exist_ok=True)
    for trace in traces:
        (d / f"{trace['id']}.json").write_bytes(orjson.dumps(trace))
    return d


# ---------------------------------------------------------------------------
# Serial / in-proc-parallel runners + field-by-field comparator
# ---------------------------------------------------------------------------


def _convert_serial(path: Path, uc, monkeypatch) -> list[Conversation]:
    monkeypatch.setattr(env_mod.Environment.DATASET, "WEKA_PARALLEL_WORKERS", 1)
    loader = WekaTraceLoader(filename=str(path), user_config=uc)
    _stub_loader_real_rng(loader)
    return loader.convert_to_conversations(loader.load_dataset())


def _convert_parallel(path: Path, uc, monkeypatch) -> list[Conversation]:
    """Full convert_to_conversations through the parallel path, with the pool
    replaced by an in-process map over _process_task (real task builder and
    result assembly are exercised)."""
    monkeypatch.setattr(env_mod.Environment.DATASET, "WEKA_PARALLEL_WORKERS", 2)
    monkeypatch.setattr(env_mod.Environment.DATASET, "WEKA_PARALLEL_THRESHOLD", 1)
    loader = WekaTraceLoader(filename=str(path), user_config=uc)
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


def _assert_parity(serial: list[Conversation], parallel: list[Conversation]) -> None:
    """Field-by-field equality of full convert_to_conversations output."""
    assert [c.session_id for c in serial] == [c.session_id for c in parallel], (
        "conversation order drift between serial and parallel paths"
    )
    for sc, pc in zip(serial, parallel, strict=True):
        sid = sc.session_id
        assert sc.is_root == pc.is_root, sid
        assert sc.agent_depth == pc.agent_depth, sid
        assert sc.parent_conversation_id == pc.parent_conversation_id, sid
        assert sc.context_mode == pc.context_mode, sid
        s_branches = [
            (
                b.branch_id,
                b.child_conversation_ids,
                b.mode,
                b.is_background,
                b.start_timestamp_ms,
            )
            for b in sc.branches
        ]
        p_branches = [
            (
                b.branch_id,
                b.child_conversation_ids,
                b.mode,
                b.is_background,
                b.start_timestamp_ms,
            )
            for b in pc.branches
        ]
        assert s_branches == p_branches, f"{sid}: branch drift"
        assert len(sc.turns) == len(pc.turns), sid
        for k, (st, pt) in enumerate(zip(sc.turns, pc.turns, strict=True)):
            ctx = f"{sid} turn {k}"
            assert st.timestamp == pt.timestamp, ctx
            assert st.delay == pt.delay, ctx
            assert st.source_trace_id == pt.source_trace_id, ctx
            assert st.source_outer_idx == pt.source_outer_idx, ctx
            assert st.source_inner_idx == pt.source_inner_idx, ctx
            assert st.source_kind == pt.source_kind, ctx
            assert st.model == pt.model, ctx
            assert st.max_tokens == pt.max_tokens, ctx
            assert st.branch_ids == pt.branch_ids, ctx
            assert [(p.kind, p.branch_id) for p in st.prerequisites] == [
                (p.kind, p.branch_id) for p in pt.prerequisites
            ], ctx
            assert st.reset_context == pt.reset_context, ctx
            assert (
                st.theoretical_prefix_cache_hit_blocks
                == pt.theoretical_prefix_cache_hit_blocks
            ), ctx
            assert (
                st.theoretical_prefix_cache_total_blocks
                == pt.theoretical_prefix_cache_total_blocks
            ), ctx
            assert st.raw_messages == pt.raw_messages, (
                f"{ctx}: raw_messages drift\n"
                f"  serial:   {st.raw_messages!r}\n"
                f"  parallel: {pt.raw_messages!r}"
            )


def _run_both(
    tmp_path: Path,
    monkeypatch,
    traces: list[dict[str, Any]],
    **uc_kwargs: Any,
) -> tuple[list[Conversation], list[Conversation]]:
    path = _write_traces(tmp_path, traces)
    serial = _convert_serial(path, _mk_user_config(**uc_kwargs), monkeypatch)
    parallel = _convert_parallel(path, _mk_user_config(**uc_kwargs), monkeypatch)
    _assert_parity(serial, parallel)
    return serial, parallel


def _by_sid(convs: list[Conversation]) -> dict[str, Conversation]:
    return {c.session_id: c for c in convs}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_convert_fanout_idle_gap_warp_parallel_byte_identical(tmp_path, monkeypatch):
    """Spec 5.6: idle-gap warp redistributes the same start set after the split.

    Three compressible gaps, one INSIDE the worker chain (2.5 -> 8.5 across
    chains, 9 -> 200 on main, 200 -> 210 reaching a worker-chain request whose
    api_time is unrecorded/None). Warped timestamps and per-chain delays must
    be byte-identical across paths.
    """
    reqs = _fanout_requests()
    reqs[5]["t"] = 200.0  # main turn 3 after a 191s idle gap
    reqs.append(
        # third worker-chain request crossing two gaps; api_time absent (None)
        _nreq(210.0, [1, 2, 50, 51, 52, 53], api_time=None)
    )
    serial, _parallel = _run_both(
        tmp_path,
        monkeypatch,
        [_trace("trace_warp", reqs)],
        idle_gap_cap_seconds=5.0,
    )

    convs = _by_sid(serial)
    root = convs["trace_warp"]
    # Gaps: [2.5, 8.5] excess 1, [9, 200] excess 186, [200, 210] excess 5.
    assert [t.timestamp for t in root.turns] == pytest.approx([0.0, 8000.0, 13000.0])
    assert root.turns[1].delay == pytest.approx(8000.0)
    assert root.turns[2].delay == pytest.approx(5000.0)
    w0 = convs["trace_warp::fa:000"]
    assert [t.timestamp for t in w0.turns] == pytest.approx([2000.0, 7500.0, 18000.0])
    assert w0.turns[1].delay == pytest.approx(5500.0)
    assert w0.turns[2].delay == pytest.approx(10500.0)


def test_convert_nonmonotonic_parent_delay_floored_parallel_byte_identical(
    tmp_path, monkeypatch
):
    """A non-monotonic parent timestamp yields a negative raw inter-turn delay.

    The serial parent loop floors it to 0.0; the parallel parent loop must too,
    or the module's byte-identical serial/parallel contract breaks (and a
    negative Turn.delay would tell the load generator to dispatch in the past).
    All three turns share a growing prefix, so they stay one main chain (no flat
    split) and exercise the parent path; trace-file order is preserved (the
    loader does not re-sort parent normals by t).
    """
    reqs = [
        _nreq(0.0, [1, 2, 3]),
        _nreq(5.0, [1, 2, 3, 4]),
        _nreq(3.0, [1, 2, 3, 4, 5]),  # t=3 < prev t=5 -> raw delay -2000 ms
    ]
    serial, _parallel = _run_both(
        tmp_path, monkeypatch, [_trace("trace_nonmono", reqs)]
    )
    # _run_both already asserts serial/parallel byte-parity on every field
    # including delay; an unfloored parallel path (-2000.0) fails there against
    # the serial path's 0.0.
    root = _by_sid(serial)["trace_nonmono"]
    assert len(root.turns) == 3
    assert root.turns[0].delay is None
    assert root.turns[1].delay == pytest.approx(5000.0)
    assert root.turns[2].delay == pytest.approx(0.0)  # floored, not -2000


def test_convert_mixed_split_directory_ordering_parallel_byte_identical(
    tmp_path, monkeypatch
):
    """Spec goal 5 + 5.2: directory where some traces split and others do not.

    Conversation ordering across traces (roots in trace order, then each
    trace's children grouped) must match across paths. trace_b carries a
    negative api_time (interval end clamps to zero); trace_c carries a
    streaming-type worker row and a partial-tail turn 0 (in % block_size != 0).
    """
    reqs_c = _fanout_requests(offset=200)
    reqs_c[2]["in"] = 4 * 64 + 17  # partial tail on fa:001 turn 0
    reqs_c[4] = _nreq(8.5, [201, 202, 250, 251, 252], api_time=1.0, typ="s", ttft=0.4)
    traces = [
        _trace("trace_a", _fanout_requests()),
        _trace(
            "trace_b",
            [
                _nreq(0.0, [1, 2, 3], api_time=1.0),
                _nreq(2.0, [1, 2, 3, 4], api_time=-5.0),  # negative end clamp
                _nreq(3.0, [1, 2, 3, 4, 5], api_time=1.0),
            ],
        ),
        _trace("trace_c", reqs_c),
    ]
    serial, _parallel = _run_both(tmp_path, monkeypatch, traces)

    assert [c.session_id for c in serial] == [
        "trace_a",
        "trace_b",
        "trace_c",
        "trace_a::fa:000",
        "trace_a::fa:001",
        "trace_c::fa:000",
        "trace_c::fa:001",
    ]
    # Invariant: every retained request appears in exactly one conversation
    # exactly once (6 + 3 + 6 requests -> 15 turns).
    assert sum(len(c.turns) for c in serial) == 15
    convs = _by_sid(serial)
    assert [
        (t.source_trace_id, t.source_outer_idx, t.source_inner_idx, t.source_kind)
        for t in convs["trace_a"].turns
    ] == [
        ("trace_a", 0, None, "weka_main"),
        ("trace_a", 3, None, "weka_main"),
        ("trace_a", 5, None, "weka_main"),
    ]
    assert [
        (t.source_trace_id, t.source_outer_idx, t.source_inner_idx, t.source_kind)
        for t in convs["trace_a::fa:000"].turns
    ] == [
        ("trace_a", 1, None, "weka_flat"),
        ("trace_a", 4, None, "weka_flat"),
    ]
    assert [
        (t.source_trace_id, t.source_outer_idx, t.source_inner_idx, t.source_kind)
        for t in convs["trace_a::fa:001"].turns
    ] == [("trace_a", 2, None, "weka_flat")]


@pytest.mark.parametrize("idle_gap_cap", [None, 4.0])
def test_convert_split_trace_with_subagent_children_order_parallel_byte_identical(
    tmp_path, monkeypatch, idle_gap_cap
):
    """Spec 5.3: type:"subagent" handling coexists with flat-chain splitting.

    Children must emit subagent children first, then flat chains, in both
    paths; subagent SPAWN/JOIN anchors against main-chain turns only. Run
    with and without the idle-gap warp (warp pulls subagent child starts into
    the shared per-trace timeline alongside flat-chain requests).
    """
    requests = [
        _nreq(0.0, [1, 2, 3], api_time=1.0),
        {
            "t": 1.5,
            "type": "subagent",
            "agent_id": "agent_x",
            "subagent_type": "Explore",
            "duration_ms": 2000,
            "total_tokens": 100,
            "tool_use_count": 1,
            "status": "completed",
            "requests": [
                _nreq(1.6, [300, 301], api_time=0.4, out=5),
            ],
            "models": ["m"],
            "tool_tokens": 0,
            "system_tokens": 0,
        },
        _nreq(2.0, [1, 2, 50, 51], api_time=6.0),
        _nreq(2.5, [1, 2, 60, 61], api_time=4.0),
        _nreq(9.0, [1, 2, 3, 4, 5], api_time=1.0),
        _nreq(8.5, [1, 2, 50, 51, 52], api_time=1.0),
        _nreq(12.0, [1, 2, 3, 4, 5, 6], api_time=1.0),
    ]
    serial, _parallel = _run_both(
        tmp_path,
        monkeypatch,
        [_trace("trace_mix", requests)],
        idle_gap_cap_seconds=idle_gap_cap,
    )

    assert [c.session_id for c in serial] == [
        "trace_mix",
        "trace_mix::sa:agent_x",
        "trace_mix::fa:000",
        "trace_mix::fa:001",
    ]
    root = serial[0]
    branch_ids = {b.branch_id for b in root.branches}
    assert "trace_mix:spawn:agent_x" in branch_ids
    assert any(":flatspawn:" in b for b in branch_ids)
    # All three spawn off main turn 0; subagent + fa:001 join main turn 1,
    # fa:000 (ends t=9.5) joins main turn 2.
    assert len(root.turns[0].branch_ids) == 3
    t1_joins = {p.branch_id for p in root.turns[1].prerequisites}
    assert "trace_mix:spawn:agent_x" in t1_joins
    assert len(root.turns[2].prerequisites) == 1
    child = _by_sid(serial)["trace_mix::sa:agent_x"]
    assert [
        (t.source_trace_id, t.source_outer_idx, t.source_inner_idx, t.source_kind)
        for t in child.turns
    ] == [("trace_mix", 1, 0, "weka_subagent")]


def test_convert_think_time_only_with_delay_cap_parallel_byte_identical(
    tmp_path, monkeypatch
):
    """Spec 5.6: per-chain delays honor think_time_only and the delay cap.

    Worker-chain request carries think_time=0.25; one main request carries
    think_time=0.0 (boundary: present-but-zero must be used, not fall back);
    another has no think_time (falls back to per-chain t-delta, then clamps
    at the 2s cap). Both paths must agree on every delay.
    """
    reqs = _fanout_requests()
    reqs[3]["think_time"] = 0.0  # main turn 1: explicit zero
    reqs[4]["think_time"] = 0.25  # fa:000 turn 1
    serial, _parallel = _run_both(
        tmp_path,
        monkeypatch,
        [_trace("trace_tt", reqs)],
        think_time_only=True,
        inter_turn_delay_cap_seconds=2.0,
    )

    convs = _by_sid(serial)
    root = convs["trace_tt"]
    assert root.turns[0].delay is None
    assert root.turns[1].delay == pytest.approx(0.0)
    # No think_time on main turn 2 -> falls back to 12.0-9.0=3s, capped to 2s.
    assert root.turns[2].delay == pytest.approx(2000.0)
    w0 = convs["trace_tt::fa:000"]
    assert w0.turns[0].delay is None
    assert w0.turns[1].delay == pytest.approx(250.0)


def test_convert_ignore_delays_nulls_all_timing_parallel_byte_identical(
    tmp_path, monkeypatch
):
    """ignore_trace_delays must null timestamp AND delay on every turn of
    every conversation (root and worker chains) identically in both paths."""
    serial, parallel = _run_both(
        tmp_path,
        monkeypatch,
        [_trace("trace_nd", _fanout_requests())],
        ignore_delays=True,
    )
    for convs in (serial, parallel):
        for c in convs:
            for k, t in enumerate(c.turns):
                assert t.timestamp is None, f"{c.session_id} turn {k}"
                assert t.delay is None, f"{c.session_id} turn {k}"


def test_convert_main_chain_compaction_seam_reset_context_parallel_byte_identical(
    tmp_path, monkeypatch
):
    """Spec 4 phase 2 + 5: a compaction seam spliced INTO the main chain of a
    split trace must reproduce reset_context=True at the seam turn in both
    paths.

    M2's tail dies (compaction to [1,2,9] elected as seam); the in-flight
    sibling fork [1,2,3,40] (temporal veto) stays a spawn, forcing the trace
    to split so the seam rides the flat-chain code path. Turn 0 carries one
    unshared user block past the observed prefix so the all-prefix turn-0
    quirk (see the xfail test below) does not mask the seam semantics.
    """
    requests = [
        _nreq(0.0, [1, 2, 3, 7], api_time=1.0),
        _nreq(2.0, [1, 2, 3, 7, 8], api_time=1.0),
        _nreq(2.5, [1, 2, 3, 40], api_time=10.0),  # overlaps M2 -> spawn
        _nreq(4.0, [1, 2, 9], api_time=1.0),  # compaction seam -> main
    ]
    serial, _parallel = _run_both(
        tmp_path, monkeypatch, [_trace("trace_seam", requests)]
    )

    assert [c.session_id for c in serial] == ["trace_seam", "trace_seam::fa:000"]
    root = serial[0]
    assert len(root.turns) == 3
    assert root.turns[1].reset_context is False  # pure growth: no reset
    assert root.turns[2].reset_context is True  # the compaction seam
    # Worker never joins (ends t=12.5, no later main turn) -> background.
    assert root.branches[0].is_background is True
    # 0/0 declared -> no fabricated system role: turn 0 is one user message
    # carrying all 4 blocks (the shared prefix lives inside the user content).
    assert root.turns[0].raw_messages == [{"role": "user", "content": "<dec:256>"}]


def test_convert_all_prefix_turn0_pure_growth_has_no_reset(tmp_path, monkeypatch):
    """Spec 4/11 case 1: pure context growth is a chain extension and must
    not flag reset_context. Here the observed group prefix (3 blocks) covers
    the main chain's ENTIRE first request, making turn 0 all-system."""
    requests = [
        _nreq(0.0, [1, 2, 3], api_time=1.0),
        _nreq(2.0, [1, 2, 3, 4, 5], api_time=1.0),
        _nreq(2.5, [1, 2, 3, 40], api_time=10.0),  # in-flight fork -> spawn
    ]
    path = _write_traces(tmp_path, [_trace("trace_allpfx", requests)])
    serial = _convert_serial(path, _mk_user_config(), monkeypatch)

    root = _by_sid(serial)["trace_allpfx"]
    assert len(root.turns) == 2
    # Turn 1 only grows the context ([1,2,3] -> [1,2,3,4,5]).
    assert root.turns[1].reset_context is False


def _poisoned_requests() -> list[dict[str, Any]]:
    """Nonce-poisoned shape (spec 8): chained block hashes make LCP=0 between
    ALL requests -> every request founds a zero-depth chain."""
    return [
        _nreq(2.0 * i, [9000 + 10 * i, 9001 + 10 * i], api_time=0.5, out=8)
        for i in range(9)
    ]


def test_convert_poisoned_trace_alongside_healthy_parallel_byte_identical(
    tmp_path, monkeypatch
):
    """A nonce-poisoned trace and a healthy fan-out trace in one directory:
    whatever the split decision is, it must be the SAME decision with the
    same bytes in both paths."""
    traces = [
        _trace("trace_heal", _fanout_requests()),
        _trace("trace_poison", _poisoned_requests()),
    ]
    serial, _parallel = _run_both(tmp_path, monkeypatch, traces)

    # Invariant regardless of the (missing) poisoned guard: all 15 retained
    # requests appear exactly once.
    assert sum(len(c.turns) for c in serial) == 15
    sids = [c.session_id for c in serial]
    assert sids[0] == "trace_heal" and sids[1] == "trace_poison"
    assert len(sids) == len(set(sids))


def test_convert_disjoint_batch_splits_serial(tmp_path, monkeypatch):
    """With the nonce-poison guard removed, a fully-disjoint trace splits into
    independent per-agent chains (root + spawned), retaining every request,
    instead of collapsing to one linear conversation."""
    path = _write_traces(tmp_path, [_trace("trace_poison", _poisoned_requests())])
    serial = _convert_serial(path, _mk_user_config(), monkeypatch)
    sids = [c.session_id for c in serial]
    assert sids[0] == "trace_poison" and len(serial) > 1
    assert sum(len(c.turns) for c in serial) == 9


def test_convert_max_osl_cross_model_batch_rewrite_parallel_byte_identical(
    tmp_path, monkeypatch
):
    """Spec 3/5.4 + 5: cross-model disjoint-namespace worker batch (the Haiku
    workers under an Opus main shape) with --max-osl capping and model
    rewriting to endpoint.model_names.

    max_tokens of flat-chain children must honor max-osl in BOTH paths (the
    parallel path ships capped_output_length in the child payload); with 0/0
    declared, both the worker group and the singleton main group keep their
    shared prefixes inside user content (no fabricated system role).
    """
    requests = [
        _nreq(0.0, [1, 2, 3], model="opus", api_time=1.0, out=10),
        _nreq(1.5, [100, 101, 110], model="haiku", api_time=3.0, out=10),
        _nreq(1.6, [100, 101, 120], model="haiku", api_time=3.0, out=10),
        _nreq(6.0, [1, 2, 3, 4], model="opus", api_time=1.0, out=10),
    ]
    serial, _parallel = _run_both(
        tmp_path,
        monkeypatch,
        [_trace("trace_xm", requests, models=["opus", "haiku"])],
        model_names=["served-main", "served-worker"],
        max_osl=5,
    )

    convs = _by_sid(serial)
    assert set(convs) == {"trace_xm", "trace_xm::fa:000", "trace_xm::fa:001"}
    root = convs["trace_xm"]
    assert all(t.model == "served-main" for t in root.turns)
    assert all(t.max_tokens == 5 for t in root.turns)
    for wid in ("trace_xm::fa:000", "trace_xm::fa:001"):
        w = convs[wid]
        assert all(t.model == "served-worker" for t in w.turns), wid
        assert all(t.max_tokens == 5 for t in w.turns), wid
        # 0/0 declared -> worker turn 0 is one user message with all 3 blocks.
        assert w.turns[0].raw_messages[0]["role"] == "user", wid
        assert w.turns[0].raw_messages[0]["content"] == "<dec:192>", wid
    # Main group is a singleton with 0/0 declared -> all-user turn 0.
    assert root.turns[0].raw_messages[0]["role"] == "user"


def test_convert_zero_declared_fanout_all_user_parallel_byte_identical(
    tmp_path, monkeypatch
):
    """The system role is never fabricated: a 0/0-declared fan-out keeps the
    observed namespace-group prefix inside the user content for the parent
    AND every worker chain, byte-identical across paths (no turn-0 override
    keys ship in the parallel payload)."""
    serial, parallel = _run_both(
        tmp_path, monkeypatch, [_trace("trace_obs", _fanout_requests())]
    )
    for convs in (serial, parallel):
        by_sid = _by_sid(convs)
        for sid, total in (
            ("trace_obs", 192),
            ("trace_obs::fa:000", 256),
            ("trace_obs::fa:001", 256),
        ):
            msgs = by_sid[sid].turns[0].raw_messages
            assert [m["role"] for m in msgs] == ["user"], sid
            assert msgs[0]["content"] == f"<dec:{total}>", sid


def test_convert_declared_prefix_wins_main_observed_for_workers_parallel_byte_identical(
    tmp_path, monkeypatch
):
    """Declared tool/system tokens (3 blocks) emit the root's system segment
    (recorded truth). Worker chains do NOT share the declared-prefix blocks
    ([1,2,50..] vs [1,2,3..] diverge at block 2), so their turn 0 is honest
    user content -- the system role is never fabricated for them."""
    requests = [
        _nreq(0.0, [1, 2, 3, 4], api_time=1.0),  # 4 blocks: covers declared 3
        _nreq(2.0, [1, 2, 50, 51], api_time=6.0),
        _nreq(2.5, [1, 2, 60, 61], api_time=4.0),
        _nreq(9.0, [1, 2, 3, 4, 5], api_time=1.0),
        _nreq(8.5, [1, 2, 50, 51, 52], api_time=1.0),
        _nreq(12.0, [1, 2, 3, 4, 5, 6], api_time=1.0),
    ]
    serial, _parallel = _run_both(
        tmp_path,
        monkeypatch,
        [_trace("trace_decl", requests, tool_tokens=192, system_tokens=0)],
    )

    convs = _by_sid(serial)
    root_msgs = convs["trace_decl"].turns[0].raw_messages
    assert root_msgs[0]["role"] == "system"
    assert root_msgs[0]["content"] == "<dec:192>"  # declared 3 blocks wins
    for wid in ("trace_decl::fa:000", "trace_decl::fa:001"):
        w_msgs = convs[wid].turns[0].raw_messages
        assert [m["role"] for m in w_msgs] == ["user"], wid
        assert w_msgs[0]["content"] == "<dec:256>", wid  # all 4 blocks, no system


def test_convert_empty_hash_first_request_split_parallel_byte_identical(
    tmp_path, monkeypatch
):
    """Spec 8 edge: an empty-hash request must stay on the main chain and
    never found a chain or witness a fork — even as the FIRST retained
    request. The first hash-bearing request after leading empty rows IS the
    main agent (it must not be exiled to a worker chain). Both paths must
    agree byte-for-byte, every request must appear exactly once, and
    orchestrator-v1 validation (run inside convert) must pass.
    """
    requests = [
        _nreq(0.0, [], api_time=0.5, in_tokens=100),  # empty hash turn 0
        _nreq(1.0, [1, 2, 3], api_time=1.0),
        _nreq(2.0, [1, 2, 40], api_time=6.0),
        _nreq(9.0, [1, 2, 3, 4], api_time=1.0),
    ]
    serial, _parallel = _run_both(tmp_path, monkeypatch, [_trace("trace_eh", requests)])

    assert sum(len(c.turns) for c in serial) == 4
    convs = _by_sid(serial)
    root = convs["trace_eh"]
    # Main = empty row + the chained [1,2,3] -> [1,2,3,4] growth; the
    # in-flight sibling [1,2,40] (overlaps the live main) is the one spawn.
    assert len(root.turns) == 3
    assert root.turns[0].theoretical_prefix_cache_total_blocks == 0
    assert set(convs) == {"trace_eh", "trace_eh::fa:000"}
    assert len(convs["trace_eh::fa:000"].turns) == 1
    assert len(root.branches) == 1
    # The worker ends at t=8; main turn at t=9 is at/after it -> gated join.
    assert root.branches[0].is_background is False
    assert root.branches[0].child_conversation_ids == ["trace_eh::fa:000"]


def test_convert_exact_equality_join_boundary_parallel_byte_identical(
    tmp_path, monkeypatch
):
    """Spec 5.3 join rule boundary: chain end EXACTLY equals the next main
    turn's timestamp (t + eps >= end with eps=1e-6 must admit equality), so
    the chain joins rather than running background — identically in both
    paths (serial computes the join from plan objects, the worker from
    shipped flat markers)."""
    requests = [
        _nreq(0.0, [1, 2, 3], api_time=1.0),
        _nreq(1.5, [1, 2, 70], api_time=2.5),  # ends exactly at 4.0
        _nreq(4.0, [1, 2, 3, 4], api_time=1.0),
    ]
    serial, _parallel = _run_both(
        tmp_path, monkeypatch, [_trace("trace_eps", requests)]
    )

    assert [c.session_id for c in serial] == ["trace_eps", "trace_eps::fa:000"]
    root = serial[0]
    assert len(root.branches) == 1
    assert root.branches[0].is_background is False
    assert root.turns[0].branch_ids == [root.branches[0].branch_id]
    assert [p.branch_id for p in root.turns[1].prerequisites] == [
        root.branches[0].branch_id
    ]
