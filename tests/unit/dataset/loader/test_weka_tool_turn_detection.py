# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool-turn detection for weka trace replay: classify each turn's ``input_kind`` from own ``input_types`` then previous-request ``stop`` fallback."""

from __future__ import annotations

import json
from multiprocessing import shared_memory
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pytest import param

from aiperf.common.enums import TurnInputKind
from aiperf.common.models import Turn
from aiperf.dataset.loader import weka_parallel_convert as wpc
from aiperf.dataset.loader.weka_trace import WekaTraceLoader, _classify_turn_input
from aiperf.dataset.loader.weka_trace_models import WekaNormalRequest
from tests.unit.dataset.loader._shared_helpers import _mk_user_config, _stub_loader

_MODEL = "claude-opus-4-5-20251101"
_HAIKU = "claude-haiku-4-5-20251001"


def _req(
    *,
    t: float = 0.0,
    hash_ids: list[int],
    input_types: list[str] | None = None,
    stop: str | None = None,
    out: int = 10,
    model: str = _MODEL,
) -> dict:
    d = {
        "t": t,
        "type": "n",
        "model": model,
        "in": len(hash_ids) * 64,
        "out": out,
        "hash_ids": hash_ids,
        "api_time": 1.0,
        "think_time": 0.0,
    }
    if input_types is not None:
        d["input_types"] = input_types
    if stop is not None:
        d["stop"] = stop
    return d


def _model_req(input_types: list[str] | None, stop: str | None) -> WekaNormalRequest:
    return WekaNormalRequest.model_validate(
        _req(hash_ids=[1], input_types=input_types, stop=stop)
    )


def _trace(trace_id: str, requests: list[dict]) -> dict:
    return {
        "id": trace_id,
        "models": [_MODEL, _HAIKU],
        "block_size": 64,
        "hash_id_scope": "local",
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": requests,
    }


def _write_trace(tmp_path: Path, trace: dict) -> str:
    p = tmp_path / f"{trace['id']}.json"
    p.write_text(json.dumps(trace))
    return str(p)


# _classify_turn_input


@pytest.mark.parametrize(
    ("input_types", "prev_stop", "expected"),
    [
        param(["tool_result"], None, TurnInputKind.TOOL_RESULT, id="own_tool_result"),
        param(
            ["text", "tool_result"],
            "end_turn",
            TurnInputKind.TOOL_RESULT,
            id="own_mixed_tool_result_wins_over_prev_stop",
        ),
        param(
            ["text"],
            "tool_use",
            TurnInputKind.USER_INPUT,
            id="own_text_wins_over_prev_tool_use",
        ),
        param(["image", "text"], None, TurnInputKind.USER_INPUT, id="own_image_text"),
        param(None, "tool_use", TurnInputKind.TOOL_RESULT, id="fallback_prev_tool_use"),
        param(None, "end_turn", TurnInputKind.USER_INPUT, id="fallback_prev_end_turn"),
        param(None, "", None, id="prev_stop_empty_is_no_signal"),
        param(None, None, None, id="no_signal_at_all"),
    ],
)  # fmt: skip
def test_classify_turn_input(
    input_types: list[str] | None,
    prev_stop: str | None,
    expected: TurnInputKind | None,
):
    req = _model_req(input_types, None)
    prev = _model_req(None, prev_stop) if prev_stop is not None else None
    assert _classify_turn_input(req, prev) == expected


def test_classify_turn_input_first_turn_no_prev_with_own_types():
    assert (
        _classify_turn_input(_model_req(["text"], None), None)
        == TurnInputKind.USER_INPUT
    )


# Turn model plumbing


def test_turn_input_kind_defaults_to_none():
    assert Turn().input_kind is None


def test_turn_input_kind_projects_into_metadata():
    turn = Turn(input_kind=TurnInputKind.TOOL_RESULT)
    assert turn.metadata().input_kind == TurnInputKind.TOOL_RESULT


# Serial reconstruction


def _scope_trace(trace_id: str) -> dict:
    """Main chain exercising own-signal + fallback, plus one subagent child."""
    sub = {
        "t": 2.0,
        "type": "subagent",
        "agent_id": "agent_001",
        "subagent_type": "Explore",
        "duration_ms": 2000,
        "total_tokens": 100,
        "tool_use_count": 1,
        "status": "completed",
        "requests": [
            _req(
                t=0.0,
                hash_ids=[50, 51],
                input_types=["text"],
                stop="tool_use",
                model=_HAIKU,
            ),
            _req(t=1.0, hash_ids=[50, 51, 52], model=_HAIKU),  # no own signal
        ],
        "models": [_HAIKU],
        "tool_tokens": 0,
        "system_tokens": 0,
    }
    return _trace(
        trace_id,
        [
            _req(t=0.0, hash_ids=[1, 2], input_types=["text"], stop="tool_use"),
            _req(
                t=1.0, hash_ids=[1, 2, 3], input_types=["tool_result"], stop="tool_use"
            ),
            sub,
            _req(
                t=5.0, hash_ids=[1, 2, 3, 4], stop="end_turn"
            ),  # fallback: prev tool_use
            _req(t=6.0, hash_ids=[1, 2, 3, 4, 5]),  # fallback: prev end_turn
        ],
    )


def test_convert_to_conversations_sets_input_kind_serial(tmp_path):
    loader = WekaTraceLoader(
        filename=_write_trace(tmp_path, _scope_trace("trace_tk")),
        run=_mk_user_config(),
    )
    _stub_loader(loader)
    convs = {
        c.session_id: c for c in loader.convert_to_conversations(loader.load_dataset())
    }

    main = convs["trace_tk"]
    assert [t.input_kind for t in main.turns] == [
        TurnInputKind.USER_INPUT,  # own ["text"]
        TurnInputKind.TOOL_RESULT,  # own ["tool_result"]
        TurnInputKind.TOOL_RESULT,  # fallback: prev stop tool_use
        TurnInputKind.USER_INPUT,  # fallback: prev stop end_turn
    ]
    child = convs["trace_tk::sa:agent_001"]
    assert [t.input_kind for t in child.turns] == [
        TurnInputKind.USER_INPUT,  # own ["text"]
        TurnInputKind.TOOL_RESULT,  # fallback: prev stop tool_use
    ]


def test_convert_to_conversations_legacy_trace_input_kind_none(tmp_path):
    trace = _trace(
        "trace_legacy",
        [
            _req(t=0.0, hash_ids=[1, 2]),
            _req(t=1.0, hash_ids=[1, 2, 3]),
        ],
    )
    loader = WekaTraceLoader(
        filename=_write_trace(tmp_path, trace), run=_mk_user_config()
    )
    _stub_loader(loader)
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert all(t.input_kind is None for c in convs for t in c.turns)


# Parallel parity: the REAL _reconstruct_parallel (pool run in-process) must
# emit the same input_kind per turn as the serial path.


def _run_pool_inproc(tasks, *, corpus, base_seed, block_size, **_kwargs):
    corpus_arr = np.array(corpus, dtype=np.int32)
    shm = shared_memory.SharedMemory(create=True, size=corpus_arr.nbytes)
    np.ndarray(corpus_arr.shape, dtype=np.int32, buffer=shm.buf)[:] = corpus_arr
    saved = wpc._worker_state
    try:
        wpc._init_worker(
            wpc._WekaWorkerInitArgs(
                shm_name=shm.name,
                corpus_len=len(corpus_arr),
                tokenizer_name="test-tok",
                base_seed=base_seed,
                block_size=block_size,
                bpe_stable_terminator_tokens=[],
            )
        )
        return [wpc._process_task(t) for t in tasks]
    finally:
        wpc._worker_state = saved
        shm.close()
        shm.unlink()


def test_parallel_reconstruction_input_kind_matches_serial(tmp_path):
    loader = WekaTraceLoader(
        filename=_write_trace(tmp_path, _scope_trace("trace_tkp")),
        run=_mk_user_config(),
    )
    _stub_loader(loader)
    data = loader.load_dataset()
    plans = loader._build_reconstruction_plans(data)
    parent_plans, child_plans = plans.parent_plans, plans.child_plans
    model_maps = {tid: loader._build_model_map(wekas[0]) for tid, wekas in data.items()}
    common = dict(
        parent_plans=parent_plans,
        child_plans=child_plans,
        data=data,
        ignore_delays=False,
        think_time_only=False,
        cap_seconds=None,
        t_start=0.0,
        model_map_per_trace=model_maps,
        metric_values_by_trace=loader._build_shared_metric_values(
            parent_plans, child_plans, plans.flat_plans
        ),
        flat_plans=plans.flat_plans,
    )
    serial = loader._reconstruct_serial(dropped_per_trace={}, **common)

    with (
        patch(
            "aiperf.dataset.loader.weka_parallel_convert.run_parallel_weka_reconstruction",
            side_effect=lambda tasks, **kw: _run_pool_inproc(tasks, **kw),
        ),
        patch(
            "aiperf.dataset.loader.weka_parallel_convert.Tokenizer.from_pretrained",
            return_value=loader.prompt_generator.tokenizer,
        ),
    ):
        parallel = loader._reconstruct_parallel(configured_workers=1, **common)

    s = {c.session_id: c for c in serial}
    p = {c.session_id: c for c in parallel}
    assert set(s) == set(p)
    for sid in s:
        s_kinds = [t.input_kind for t in s[sid].turns]
        p_kinds = [t.input_kind for t in p[sid].turns]
        assert s_kinds == p_kinds, f"{sid}: serial {s_kinds} != parallel {p_kinds}"
        assert any(k is not None for k in s_kinds), f"{sid}: no classification at all"
