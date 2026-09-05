# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WekaTraceLoader rewrites each request's ``model`` to ``endpoint.model_names`` via a per-trace deterministic mapping (always-on, no flag)."""

from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.dataset.loader.weka_trace_models import (
    WekaTrace,
)
from tests.unit.dataset.loader._shared_helpers import _make_loader, _write_trace


def _mk_user_config(*, max_isl=None, model_names=("primary",), **overrides):
    from tests.unit.dataset.loader.conftest import make_weka_run

    return make_weka_run(
        model_names=list(model_names),
        tokenizer_name="t",
        max_isl=max_isl,
        **overrides,
    )


def _trace(trace_id, requests, models=("m",)):
    return {
        "id": trace_id,
        "models": list(models),
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": requests,
    }


def _normal(t=0.0, model="m", in_=10, out=1):
    return {"t": t, "type": "n", "model": model, "in": in_, "out": out}


def _subagent(t, agent_id, inner_requests, models=("m",)):
    return {
        "t": t,
        "type": "subagent",
        "agent_id": agent_id,
        "subagent_type": "Explore",
        "status": "completed",
        "requests": inner_requests,
        "models": list(models),
    }


# Unit tests for _build_model_map


def _make_trace_obj(requests_dicts, trace_id="tr"):
    return WekaTrace.model_validate(
        {
            "id": trace_id,
            "models": ["m"],
            "block_size": 64,
            "hash_id_scope": "local",
            "requests": requests_dicts,
        }
    )


def _bare_loader(*, model_names):
    """Minimal loader for _build_model_map: sets ``_configured_model_names`` directly and skips the real constructor (no I/O paths)."""
    loader = WekaTraceLoader.__new__(WekaTraceLoader)
    loader._configured_model_names = list(model_names)
    return loader


def test_build_model_map_single_model_single_configured():
    loader = _bare_loader(model_names=("M0",))
    trace = _make_trace_obj([_normal(model="m")])
    assert loader._build_model_map(trace) == {"m": "M0"}


def test_build_model_map_single_model_multi_configured_uses_only_main_slot():
    loader = _bare_loader(model_names=("M0", "M1", "M2"))
    trace = _make_trace_obj([_normal(model="m"), _normal(t=1.0, model="m")])
    assert loader._build_model_map(trace) == {"m": "M0"}


def test_build_model_map_main_plus_subagent_two_configured():
    loader = _bare_loader(model_names=("M0", "M1"))
    trace = _make_trace_obj(
        [
            _normal(t=0.0, model="parent-m"),
            _subagent(
                t=1.0,
                agent_id="a1",
                inner_requests=[_normal(model="sa-m")],
                models=("sa-m",),
            ),
            _normal(t=2.0, model="parent-m"),
        ]
    )
    assert loader._build_model_map(trace) == {"parent-m": "M0", "sa-m": "M1"}


def test_build_model_map_more_distinct_than_configured_modulo_wrap():
    loader = _bare_loader(model_names=("M0", "M1"))
    trace = _make_trace_obj(
        [
            _normal(t=0.0, model="A"),
            _subagent(
                t=1.0,
                agent_id="s1",
                inner_requests=[_normal(model="B")],
                models=("B",),
            ),
            _subagent(
                t=2.0,
                agent_id="s2",
                inner_requests=[_normal(model="C")],
                models=("C",),
            ),
            _subagent(
                t=3.0,
                agent_id="s3",
                inner_requests=[_normal(model="D")],
                models=("D",),
            ),
            _subagent(
                t=4.0,
                agent_id="s4",
                inner_requests=[_normal(model="E")],
                models=("E",),
            ),
        ]
    )
    # A→M0, B→M1, C→M0, D→M1, E→M0
    assert loader._build_model_map(trace) == {
        "A": "M0",
        "B": "M1",
        "C": "M0",
        "D": "M1",
        "E": "M0",
    }


def test_build_model_map_first_appearance_order_in_outer_list():
    """Main is the first parent normal's model regardless of subagent position, then walk-order picks up other distinct models."""
    loader = _bare_loader(model_names=("M0", "M1", "M2"))
    trace = _make_trace_obj(
        [
            _normal(t=0.0, model="A"),  # main
            _subagent(
                t=1.0,
                agent_id="s",
                inner_requests=[_normal(model="B")],
                models=("B",),
            ),
            _normal(t=2.0, model="C"),
        ]
    )
    # main=A, then walk: A(seen), B(new→M1), C(new→M2)
    assert loader._build_model_map(trace) == {"A": "M0", "B": "M1", "C": "M2"}


def test_build_model_map_only_subagents_no_parent_normals():
    """Parent-less trace: first subagent's first request defines main."""
    loader = _bare_loader(model_names=("M0", "M1"))
    trace = _make_trace_obj(
        [
            _subagent(
                t=0.0,
                agent_id="s",
                inner_requests=[
                    _normal(model="sa-main"),
                    _normal(t=1.0, model="sa-other"),
                ],
                models=("sa-main", "sa-other"),
            ),
        ]
    )
    assert loader._build_model_map(trace) == {"sa-main": "M0", "sa-other": "M1"}


def test_build_model_map_empty_model_names_returns_empty():
    loader = _bare_loader(model_names=())
    trace = _make_trace_obj([_normal(model="m")])
    assert loader._build_model_map(trace) == {}


def test_build_model_map_empty_trace_returns_empty():
    loader = _bare_loader(model_names=("M0",))
    trace = _make_trace_obj([])
    assert loader._build_model_map(trace) == {}


# End-to-end loader tests (serial path; parallel path is forced off in conftest)


def test_loader_rewrites_parent_turn_model_to_configured_model_zero(
    tmp_path, monkeypatch
):
    uc = _mk_user_config(model_names=("CONFIGURED",))
    path = _write_trace(
        tmp_path, _trace("tr", [_normal(model="trace-m")], models=("trace-m",))
    )
    loader = _make_loader(path, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "tr")
    assert all(t.model == "CONFIGURED" for t in parent.turns)


def test_loader_rewrites_subagent_turn_model_to_configured_slot_one(
    tmp_path, monkeypatch
):
    uc = _mk_user_config(model_names=("PARENT", "SA"))
    requests = [
        _normal(t=0.0, model="parent-m"),
        _subagent(
            t=1.0,
            agent_id="a1",
            inner_requests=[_normal(model="sa-m")],
            models=("sa-m",),
        ),
        _normal(t=2.0, model="parent-m"),
    ]
    path = _write_trace(tmp_path, _trace("tr", requests, models=("parent-m",)))
    loader = _make_loader(path, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "tr")
    assert all(t.model == "PARENT" for t in parent.turns)
    child = next(c for c in convs if c.session_id == "tr::sa:a1")
    assert all(t.model == "SA" for t in child.turns)


def test_loader_no_longer_raises_on_unmatched_trace_model(tmp_path, monkeypatch):
    """Regression: the old _validate_models would have rejected this run."""
    uc = _mk_user_config(model_names=("ANYTHING",))
    path = _write_trace(
        tmp_path,
        _trace(
            "tr",
            [_normal(model="completely-unrelated")],
            models=("completely-unrelated",),
        ),
    )
    loader = _make_loader(path, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())  # no raise
    parent = next(c for c in convs if c.session_id == "tr")
    assert all(t.model == "ANYTHING" for t in parent.turns)


def test_loader_case_mismatch_still_rewrites(tmp_path, monkeypatch):
    """Trace's case-mismatched model name still gets rewritten, no error."""
    uc = _mk_user_config(model_names=("modela",))
    path = _write_trace(
        tmp_path, _trace("tr", [_normal(model="ModelA")], models=("ModelA",))
    )
    loader = _make_loader(path, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "tr")
    assert all(t.model == "modela" for t in parent.turns)


def test_loader_empty_model_names_preserves_trace_model(tmp_path, monkeypatch):
    """With empty endpoint.model_names, mapping is empty → trace value passes through."""
    # A real v2 config cannot carry an empty models list (ModelsAdvanced.items
    # has min_length=1), so build the loader normally and clear the resolved
    # _configured_model_names directly -- the empty-config behavior under test.
    uc = _mk_user_config(model_names=("placeholder",))
    path = _write_trace(tmp_path, _trace("tr", [_normal(model="trace-m")]))
    loader = _make_loader(path, uc, monkeypatch)
    loader._configured_model_names = []
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "tr")
    assert all(t.model == "trace-m" for t in parent.turns)


def test_loader_unicode_model_name_rewritten_correctly(tmp_path, monkeypatch):
    name_in = "trace-模型"
    name_out = "configured-模型"
    uc = _mk_user_config(model_names=(name_out,))
    path = _write_trace(
        tmp_path, _trace("tr", [_normal(model=name_in)], models=(name_in,))
    )
    loader = _make_loader(path, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "tr")
    assert all(t.model == name_out for t in parent.turns)


def test_loader_modulo_wrap_collapses_to_single_configured(tmp_path, monkeypatch):
    """3 distinct trace models, 1 configured → all collapse to it."""
    uc = _mk_user_config(model_names=("ONLY",))
    requests = [
        _normal(t=0.0, model="A"),
        _subagent(
            t=1.0,
            agent_id="s1",
            inner_requests=[_normal(model="B")],
            models=("B",),
        ),
        _subagent(
            t=2.0,
            agent_id="s2",
            inner_requests=[_normal(model="C")],
            models=("C",),
        ),
    ]
    path = _write_trace(tmp_path, _trace("tr", requests, models=("A",)))
    loader = _make_loader(path, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    for c in convs:
        for t in c.turns:
            assert t.model == "ONLY"
