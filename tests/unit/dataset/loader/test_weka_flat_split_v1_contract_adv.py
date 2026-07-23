# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the flattened-agent-split EMITTED-DATASET CONTRACT."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.environment import Environment
from aiperf.dataset.loader.weka_agent_chains import detect_agent_chains
from aiperf.dataset.loader.weka_trace import WekaTraceLoader

from tests.unit.dataset.loader.test_weka_trace import (
    _mk_user_config,
    _stub_prompt_generator_for_reconstructor,
)


def _row(
    *,
    t: float,
    hash_ids: list[int],
    in_len: int | None = None,
    out_len: int = 10,
    api_time: float = 1.0,
    model: str = "m",
    rtype: str = "n",
) -> dict:
    """One top-level request row in Weka JSON shape (in/out aliases)."""
    row: dict = {
        "t": t,
        "type": rtype,
        "model": model,
        "in": in_len if in_len is not None else max(len(hash_ids), 1) * 64,
        "out": out_len,
        "hash_ids": hash_ids,
        "api_time": api_time,
    }
    return row


def _write_trace(
    path: Path,
    *,
    trace_id: str,
    requests: list[dict],
    block_size: int = 64,
    tool_tokens: int = 0,
    system_tokens: int = 0,
    models: list[str] | None = None,
) -> Path:
    blob = {
        "id": trace_id,
        "models": models or ["m"],
        "block_size": block_size,
        "hash_id_scope": "local",
        "tool_tokens": tool_tokens,
        "system_tokens": system_tokens,
        "requests": requests,
    }
    path.write_text(json.dumps(blob))
    return path


def _loader_for(path: Path, uc=None) -> WekaTraceLoader:
    loader = WekaTraceLoader(
        filename=str(path), user_config=uc if uc is not None else _mk_user_config()
    )
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    return loader


def _convs_by_sid(loader: WekaTraceLoader) -> dict:
    return {
        c.session_id: c for c in loader.convert_to_conversations(loader.load_dataset())
    }


def _retained_top_level_count(requests: list[dict], uc=None) -> int:
    """Count rows a fresh loader's filters would retain (all here, no filters)."""
    return sum(1 for r in requests if r["type"] in ("n", "s"))


def test_convert_to_conversations_hostile_fanout_passes_orchestrator_v1(tmp_path):
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[100, 101], model="hk", rtype="s"),
        _row(t=2.0, hash_ids=[200, 201], model="hk"),
        _row(t=3.0, hash_ids=[1, 2, 3, 4, 5]),
        _row(t=4.0, hash_ids=[100, 101, 102], model="hk", rtype="s"),
        _row(t=6.0, hash_ids=[1, 2, 3, 4, 5, 6]),
        _row(t=7.0, hash_ids=[200, 201, 202], model="hk", api_time=999.0),
    ]
    p = _write_trace(tmp_path / "h.json", trace_id="hostile", requests=reqs)
    loader = _loader_for(p)
    convs = _convs_by_sid(loader)
    assert "hostile" in convs
    assert any(sid.startswith("hostile::fa:") for sid in convs)


def test_branch_child_ids_reference_only_emitted_conversations(tmp_path):
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[1, 2, 50], model="hk"),
        _row(t=2.0, hash_ids=[1, 2, 3, 4]),
        _row(t=3.0, hash_ids=[1, 2, 50, 51], model="hk"),
        _row(t=5.0, hash_ids=[1, 2, 3, 4, 5]),
    ]
    p = _write_trace(tmp_path / "c.json", trace_id="cref", requests=reqs)
    loader = _loader_for(p)
    convs = _convs_by_sid(loader)
    emitted = set(convs)
    for conv in convs.values():
        for branch in conv.branches:
            for child_id in branch.child_conversation_ids:
                assert child_id in emitted, (
                    f"branch {branch.branch_id} references unemitted child {child_id}"
                )


def test_no_duplicate_branch_id_per_turn_many_adjacent_groups(tmp_path):
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=0.10, hash_ids=[300, 301], model="w0", api_time=0.1),
        _row(t=0.30, hash_ids=[400, 401], model="w1", api_time=0.1),
        _row(t=0.50, hash_ids=[500, 501], model="w2", api_time=0.1),
        _row(t=5.0, hash_ids=[1, 2, 3, 4]),
    ]
    p = _write_trace(tmp_path / "d.json", trace_id="dups", requests=reqs)
    loader = _loader_for(p)
    convs = _convs_by_sid(loader)
    root = convs["dups"]
    for idx, turn in enumerate(root.turns):
        assert len(turn.branch_ids) == len(set(turn.branch_ids)), (
            f"turn {idx} has a duplicate branch_id: {turn.branch_ids}"
        )


def test_spawn_join_prereq_always_references_strictly_earlier_branch(tmp_path):
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=2.0, hash_ids=[1, 2, 3, 4]),
        _row(t=4.0, hash_ids=[1, 2, 3, 4, 5]),
        _row(t=6.0, hash_ids=[1, 2, 3, 4, 5, 6]),
        _row(t=8.0, hash_ids=[1, 2, 3, 4, 5, 6, 7]),
        _row(t=9.0, hash_ids=[1, 2, 3, 4, 5, 6, 7, 99], model="hk"),
        _row(t=10.0, hash_ids=[1, 2, 3, 4, 5, 6, 7, 99, 100], model="hk"),
    ]
    p = _write_trace(tmp_path / "g.json", trace_id="gate", requests=reqs)
    loader = _loader_for(p)
    convs = _convs_by_sid(loader)
    root = convs["gate"]
    n_prereqs = sum(len(t.prerequisites) for t in root.turns)
    flat_branches = [b for b in root.branches if ":flatspawn:" in b.branch_id]
    assert flat_branches, "expected the worker to be emitted as a flat branch"
    assert all(b.is_background for b in flat_branches), (
        "a chain that forks off the last main turn must be background"
    )
    assert n_prereqs == 0


def _hostile_traces() -> list[tuple[str, list[dict]]]:
    cases: list[tuple[str, list[dict]]] = []
    a = [
        _row(t=0.0, hash_ids=[1, 2]),
        _row(t=1.0, hash_ids=[1, 2, 10], model="w1"),
        _row(t=1.5, hash_ids=[1, 2, 20], model="w2"),
        _row(t=2.0, hash_ids=[1, 2, 3]),
        _row(t=3.0, hash_ids=[1, 2, 10, 11], model="w1"),
        _row(t=4.0, hash_ids=[1, 2, 3, 4]),
    ]
    cases.append(("ix", list(reversed(a))))
    b = [
        _row(t=0.0, hash_ids=[1, 2, 3, 4, 5], api_time=1.0),
        _row(t=2.0, hash_ids=[1, 2], api_time=1.0),
        _row(t=4.0, hash_ids=[1, 2, 6]),
        _row(t=5.0, hash_ids=[1, 2, 90], model="hk"),
    ]
    cases.append(("seamspawn", b))
    c = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[]),
        _row(t=2.0, hash_ids=[500], model="solo"),
        _row(t=3.0, hash_ids=[1, 2, 3, 4]),
    ]
    cases.append(("singleton", c))
    d = [
        _row(t=0.0, hash_ids=[1, 2, 3], rtype="s"),
        _row(t=1.0, hash_ids=[1, 2, 77], model="hk", rtype="s"),
        _row(t=2.0, hash_ids=[1, 2, 3, 4], rtype="s"),
        _row(t=3.0, hash_ids=[1, 2, 77, 78], model="hk", rtype="s"),
    ]
    cases.append(("allstream", d))
    return cases


@pytest.mark.parametrize(
    "name,reqs", _hostile_traces(), ids=[c[0] for c in _hostile_traces()]
)
def test_emitted_metadata_validates_and_gates_are_well_ordered(name, reqs, tmp_path):
    from aiperf.common.models import DatasetMetadata
    from aiperf.common.validators.orchestrator_v1 import validate_for_orchestrator_v1

    p = _write_trace(tmp_path / f"{name}.json", trace_id=name, requests=reqs)
    loader = _loader_for(p)
    convs = loader.convert_to_conversations(loader.load_dataset())
    metadata = DatasetMetadata(
        conversations=[c.to_metadata() for c in convs],
        sampling_strategy=loader.get_preferred_sampling_strategy(),
    )
    validate_for_orchestrator_v1(metadata)

    for conv in convs:
        decl_turn: dict[str, int] = {}
        for idx, turn in enumerate(conv.turns):
            for bid in turn.branch_ids:
                decl_turn.setdefault(bid, idx)
        for idx, turn in enumerate(conv.turns):
            for prereq in turn.prerequisites:
                assert prereq.branch_id is not None
                d = decl_turn.get(prereq.branch_id)
                assert d is not None and d < idx, (
                    f"{conv.session_id} turn {idx}: SPAWN_JOIN references branch "
                    f"{prereq.branch_id} declared on turn {d} (not strictly earlier)"
                )


def test_conservation_every_retained_request_in_exactly_one_turn(tmp_path):
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[1, 2, 40], model="w1"),
        _row(t=1.5, hash_ids=[1, 2, 50], model="w2"),
        _row(t=2.0, hash_ids=[1, 2, 3, 4]),
        _row(t=3.0, hash_ids=[1, 2, 40, 41], model="w1"),
        _row(t=3.5, hash_ids=[]),
        _row(t=4.0, hash_ids=[1, 2, 3, 4, 5]),
        _row(t=5.0, hash_ids=[1, 2, 50, 51], model="w2"),
    ]
    retained = _retained_top_level_count(reqs)
    p = _write_trace(tmp_path / "cons.json", trace_id="cons", requests=reqs)
    loader = _loader_for(p)
    convs = _convs_by_sid(loader)
    root_and_flat = [
        c for sid, c in convs.items() if sid == "cons" or sid.startswith("cons::fa:")
    ]
    total_turns = sum(len(c.turns) for c in root_and_flat)
    assert total_turns == retained, (
        f"turn conservation: emitted {total_turns} turns over root+flat for "
        f"{retained} retained top-level requests"
    )


def test_conservation_matches_detection_partition(tmp_path):
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[1, 2, 40], model="w1"),
        _row(t=2.0, hash_ids=[1, 2, 3, 4]),
        _row(t=3.0, hash_ids=[1, 2, 40, 41], model="w1"),
        _row(t=4.0, hash_ids=[1, 2, 3, 4, 5]),
    ]
    p = _write_trace(tmp_path / "part.json", trace_id="part", requests=reqs)
    loader = _loader_for(p)
    convs = _convs_by_sid(loader)

    indexed = list(enumerate(_normals_from_rows(reqs)))
    result = detect_agent_chains(indexed)
    live = [i for i, c in enumerate(result.chains) if c.spliced_into is None]
    detector_counts = sorted(len(result.chains[i].requests) for i in live)

    emitted_counts = sorted(
        len(c.turns)
        for sid, c in convs.items()
        if sid == "part" or sid.startswith("part::fa:")
    )
    assert emitted_counts == detector_counts


def _normals_from_rows(rows: list[dict]):
    """Build WekaNormalRequest/WekaStreamingRequest objects in file order."""
    from aiperf.dataset.loader.weka_trace_models import (
        WekaNormalRequest,
        WekaStreamingRequest,
    )

    out = []
    for r in rows:
        cls = WekaStreamingRequest if r["type"] == "s" else WekaNormalRequest
        out.append(cls.model_validate(r))
    return out


def test_per_chain_timestamps_increasing_and_delays_are_intra_chain_diffs(tmp_path):
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[1, 2, 80], model="hk"),
        _row(t=4.0, hash_ids=[1, 2, 3, 4]),
        _row(t=7.0, hash_ids=[1, 2, 80, 81], model="hk"),
        _row(t=10.0, hash_ids=[1, 2, 3, 4, 5]),
    ]
    p = _write_trace(tmp_path / "tdelay.json", trace_id="td", requests=reqs)
    loader = _loader_for(p)
    convs = _convs_by_sid(loader)
    for sid, conv in convs.items():
        if not (sid == "td" or sid.startswith("td::fa:")):
            continue
        ts = [t.timestamp for t in conv.turns]
        assert all(b > a for a, b in zip(ts, ts[1:], strict=False)), (sid, ts)
        assert conv.turns[0].delay is None, sid
        for k in range(1, len(conv.turns)):
            assert conv.turns[k].delay == pytest.approx(ts[k] - ts[k - 1]), (sid, k)


def test_flat_chain_children_are_leaf_non_root_depth_one(tmp_path):
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[1, 2, 90], model="hk"),
        _row(t=2.0, hash_ids=[1, 2, 3, 4]),
        _row(t=3.0, hash_ids=[1, 2, 90, 91], model="hk"),
        _row(t=5.0, hash_ids=[1, 2, 3, 4, 5]),
    ]
    p = _write_trace(tmp_path / "leaf.json", trace_id="leaf", requests=reqs)
    loader = _loader_for(p)
    convs = _convs_by_sid(loader)
    flat = {sid: c for sid, c in convs.items() if sid.startswith("leaf::fa:")}
    assert flat, "expected at least one flat-chain child"
    for sid, c in flat.items():
        assert c.is_root is False, sid
        assert c.agent_depth == 1, sid
        assert c.parent_conversation_id == "leaf", sid
        assert c.branches == [], f"{sid}: flat child must not declare branches"
        for turn in c.turns:
            assert turn.branch_ids == [], sid
            assert turn.prerequisites == [], sid


def test_flat_session_ids_are_zero_padded_and_dense(tmp_path):
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[10, 11], model="w0"),
        _row(t=2.0, hash_ids=[20, 21], model="w1"),
        _row(t=3.0, hash_ids=[30, 31], model="w2"),
        _row(t=4.0, hash_ids=[1, 2, 3, 4]),
    ]
    p = _write_trace(tmp_path / "ids.json", trace_id="ids", requests=reqs)
    loader = _loader_for(p)
    convs = _convs_by_sid(loader)
    fa = sorted(sid for sid in convs if sid.startswith("ids::fa:"))
    suffixes = [sid.rsplit(":", 1)[-1] for sid in fa]
    assert suffixes == [f"{i:03d}" for i in range(len(suffixes))], suffixes
    assert len(suffixes) == 3, suffixes


def test_small_fresh_context_singletons_emit_aux_session_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(Environment.DATASET, "WEKA_AUX_MAX_REQUESTS", 1)
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[10, 11], model="w0"),
        _row(t=2.0, hash_ids=[20, 21], model="w1"),
        _row(t=3.0, hash_ids=[30, 31], model="w2"),
        _row(t=4.0, hash_ids=[1, 2, 3, 4]),
    ]
    p = _write_trace(tmp_path / "sc.json", trace_id="sc", requests=reqs)
    convs = _convs_by_sid(_loader_for(p))
    aux = sorted(sid for sid in convs if sid.startswith("sc::aux:"))
    assert [sid.rsplit(":", 1)[-1] for sid in aux] == ["000", "001", "002"], sorted(
        convs
    )
    assert not any(sid.startswith("sc::fa:") for sid in convs), sorted(convs)


def test_cross_model_large_singleton_emits_aux_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(Environment.DATASET, "WEKA_AUX_MAX_REQUESTS", 1)
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[50], in_len=200_000, model="hk"),
        _row(t=2.0, hash_ids=[1, 2, 3, 4]),
    ]
    p = _write_trace(tmp_path / "xm.json", trace_id="xm", requests=reqs)
    convs = _convs_by_sid(_loader_for(p))
    assert "xm::aux:000" in convs, sorted(convs)
    assert not any(sid.startswith("xm::fa:") for sid in convs), sorted(convs)


def test_same_model_large_reduction_emits_aux_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(Environment.DATASET, "WEKA_AUX_REDUCTION_OSL_MAX", 4000)
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[50], in_len=41_280, out_len=330),
        _row(t=2.0, hash_ids=[1, 2, 3, 4]),
    ]
    p = _write_trace(tmp_path / "rd.json", trace_id="rd", requests=reqs)
    convs = _convs_by_sid(_loader_for(p))
    assert "rd::aux:000" in convs, sorted(convs)
    assert not any(sid.startswith("rd::fa:") for sid in convs), sorted(convs)


def test_shared_spawn_fanout_emits_worker_group_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(Environment.DATASET, "WEKA_WORKER_GROUP_MIN", 3)
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3, 4, 5, 6, 7, 8], api_time=100.0),
        _row(t=1.0, hash_ids=[1, 90], api_time=100.0),
        _row(t=2.0, hash_ids=[1, 91], api_time=100.0),
        _row(t=3.0, hash_ids=[1, 92], api_time=100.0),
    ]
    p = _write_trace(tmp_path / "wg.json", trace_id="wg", requests=reqs)
    convs = _convs_by_sid(_loader_for(p))
    wg = sorted(sid for sid in convs if sid.startswith("wg::wg:"))
    assert [sid.rsplit(":", 1)[-1] for sid in wg] == [
        "000_000",
        "000_001",
        "000_002",
    ], sorted(convs)
    assert not any(sid.startswith("wg::fa:") for sid in convs), sorted(convs)


def _enable_production_classification(monkeypatch) -> None:
    """Restore aux / reduction / worker-group classification to their real"""
    ds = Environment.DATASET
    fields = type(ds).model_fields
    for name in (
        "WEKA_AUX_MAX_REQUESTS",
        "WEKA_AUX_REDUCTION_OSL_MAX",
        "WEKA_WORKER_GROUP_MIN",
    ):
        monkeypatch.setattr(ds, name, fields[name].default)


def test_production_defaults_agent_and_cross_model_sidecar_coexist(
    tmp_path, monkeypatch
):
    """At the shipped defaults, one trace with a multi-request same-model agent"""
    _enable_production_classification(monkeypatch)
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[50], model="hk"),
        _row(t=2.0, hash_ids=[80, 81]),
        _row(t=3.0, hash_ids=[80, 81, 82]),
        _row(t=4.0, hash_ids=[1, 2, 3, 4]),
    ]
    p = _write_trace(tmp_path / "prod_mix.json", trace_id="prod_mix", requests=reqs)
    convs = _convs_by_sid(_loader_for(p))
    sids = sorted(convs)
    assert any(s.startswith("prod_mix::aux:") for s in sids), sids
    assert any(s.startswith("prod_mix::fa:") for s in sids), sids


def test_production_defaults_shared_spawn_fanout_is_worker_group(tmp_path, monkeypatch):
    """At the shipped defaults, workers that fork from the still-open main"""
    _enable_production_classification(monkeypatch)
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3, 4, 5, 6, 7, 8], api_time=100.0),
        _row(t=1.0, hash_ids=[1, 90], in_len=20000, out_len=5000, api_time=100.0),
        _row(t=2.0, hash_ids=[1, 91], in_len=20000, out_len=5000, api_time=100.0),
        _row(t=3.0, hash_ids=[1, 92], in_len=20000, out_len=5000, api_time=100.0),
    ]
    p = _write_trace(tmp_path / "prod_wg.json", trace_id="prod_wg", requests=reqs)
    convs = _convs_by_sid(_loader_for(p))
    wg = sorted(s for s in convs if s.startswith("prod_wg::wg:"))
    assert len(wg) == 3, sorted(convs)
    assert not any(s.startswith("prod_wg::aux:") for s in convs), sorted(convs)
    assert not any(s.startswith("prod_wg::fa:") for s in convs), sorted(convs)


def test_two_identical_loads_byte_stable_structure(tmp_path):
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[1, 2, 60], model="hk"),
        _row(t=1.5, hash_ids=[1, 2, 70], model="hk2"),
        _row(t=2.0, hash_ids=[1, 2, 3, 4]),
        _row(t=3.0, hash_ids=[1, 2, 60, 61], model="hk"),
        _row(t=5.0, hash_ids=[1, 2, 3, 4, 5]),
    ]
    p = _write_trace(tmp_path / "det.json", trace_id="det", requests=reqs)

    def _snapshot():
        loader = _loader_for(p)
        convs = loader.convert_to_conversations(loader.load_dataset())
        return [
            (
                c.session_id,
                c.is_root,
                c.agent_depth,
                c.parent_conversation_id,
                tuple(
                    (b.branch_id, tuple(b.child_conversation_ids), b.is_background)
                    for b in c.branches
                ),
                tuple(
                    (t.timestamp, t.delay, tuple(t.branch_ids), t.reset_context)
                    for t in c.turns
                ),
            )
            for c in convs
        ]

    assert _snapshot() == _snapshot()


def test_streaming_rows_split_identically_to_normal_rows(tmp_path):
    base = [
        (0.0, [1, 2, 3], "m"),
        (1.0, [1, 2, 88], "hk"),
        (2.0, [1, 2, 3, 4], "m"),
        (3.0, [1, 2, 88, 89], "hk"),
        (5.0, [1, 2, 3, 4, 5], "m"),
    ]

    def _counts(stream_idxs: set[int]) -> list[int]:
        reqs = [
            _row(
                t=t,
                hash_ids=h,
                model=mdl,
                rtype="s" if i in stream_idxs else "n",
            )
            for i, (t, h, mdl) in enumerate(base)
        ]
        p = _write_trace(
            tmp_path / f"mix_{'-'.join(map(str, sorted(stream_idxs)))}.json",
            trace_id=f"mix{len(stream_idxs)}",
            requests=reqs,
        )
        loader = _loader_for(p)
        convs = _convs_by_sid(loader)
        tid = f"mix{len(stream_idxs)}"
        return sorted(
            len(c.turns)
            for sid, c in convs.items()
            if sid == tid or sid.startswith(f"{tid}::fa:")
        )

    all_normal = _counts(set())
    worker_stream = _counts({1, 3})
    all_stream = _counts({0, 1, 2, 3, 4})
    assert all_normal == worker_stream == all_stream


def test_env_off_emits_single_root_with_all_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(Environment.DATASET, "WEKA_SPLIT_FLATTENED_AGENTS", False)
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[1, 2, 60], model="hk"),
        _row(t=2.0, hash_ids=[1, 2, 3, 4]),
        _row(t=3.0, hash_ids=[1, 2, 60, 61], model="hk"),
    ]
    retained = _retained_top_level_count(reqs)
    p = _write_trace(tmp_path / "off.json", trace_id="off", requests=reqs)
    loader = _loader_for(p)
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert [c.session_id for c in convs] == ["off"]
    assert len(convs[0].turns) == retained
    assert convs[0].branches == []


def test_theoretical_total_blocks_equals_hash_len_per_request(tmp_path):
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[1, 2, 44], model="hk"),
        _row(t=2.0, hash_ids=[1, 2, 3, 4]),
        _row(t=3.0, hash_ids=[1, 2, 44, 45], model="hk"),
        _row(t=5.0, hash_ids=[1, 2, 3, 4, 5]),
    ]
    expected_total = sum(len(r["hash_ids"]) for r in reqs)
    p = _write_trace(tmp_path / "blk.json", trace_id="blk", requests=reqs)
    loader = _loader_for(p)
    convs = _convs_by_sid(loader)
    emitted_total = 0
    for sid, c in convs.items():
        if not (sid == "blk" or sid.startswith("blk::fa:")):
            continue
        for turn in c.turns:
            assert turn.theoretical_prefix_cache_total_blocks is not None, sid
            assert (
                turn.theoretical_prefix_cache_hit_blocks
                <= turn.theoretical_prefix_cache_total_blocks
            ), sid
            emitted_total += turn.theoretical_prefix_cache_total_blocks
    assert emitted_total == expected_total


def test_no_emitted_delay_is_negative(tmp_path):
    reqs = [
        _row(t=5.0, hash_ids=[1, 2, 3, 4, 5]),
        _row(t=3.0, hash_ids=[1, 2, 60, 61], model="hk"),
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[1, 2, 60], model="hk"),
        _row(t=2.0, hash_ids=[1, 2, 3, 4]),
    ]
    p = _write_trace(tmp_path / "neg.json", trace_id="neg", requests=reqs)
    loader = _loader_for(p)
    convs = _convs_by_sid(loader)
    for sid, c in convs.items():
        for k, turn in enumerate(c.turns):
            if turn.delay is not None:
                assert turn.delay >= 0.0, f"{sid} turn {k} delay {turn.delay} < 0"


def test_flat_branches_are_spawn_mode_with_min_start_timestamp(tmp_path):
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.25, hash_ids=[1, 2, 70], model="hk"),
        _row(t=2.0, hash_ids=[1, 2, 3, 4]),
        _row(t=3.0, hash_ids=[1, 2, 70, 71], model="hk"),
        _row(t=5.0, hash_ids=[1, 2, 3, 4, 5]),
    ]
    p = _write_trace(tmp_path / "ts.json", trace_id="ts", requests=reqs)
    loader = _loader_for(p)
    convs = _convs_by_sid(loader)
    root = convs["ts"]
    flat = [b for b in root.branches if ":flatspawn:" in b.branch_id]
    assert flat, "expected a flat branch"
    for b in flat:
        assert b.mode == ConversationBranchMode.SPAWN
        assert b.start_timestamp_ms == pytest.approx(1.25 * 1000.0)


def test_exact_equal_timestamps_tie_break_by_outer_index(tmp_path):
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[1, 2, 3, 4]),
        _row(t=1.0, hash_ids=[900, 901], model="zz"),
        _row(t=2.0, hash_ids=[1, 2, 3, 4, 5]),
        _row(t=3.0, hash_ids=[900, 901, 902], model="zz"),
    ]
    retained = _retained_top_level_count(reqs)
    p = _write_trace(tmp_path / "tie.json", trace_id="tie", requests=reqs)
    loader = _loader_for(p)
    convs = _convs_by_sid(loader)
    total = sum(
        len(c.turns)
        for sid, c in convs.items()
        if sid == "tie" or sid.startswith("tie::fa:")
    )
    assert total == retained


def test_subagent_and_flat_chain_branch_ids_disjoint(tmp_path):
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        {
            "t": 0.5,
            "type": "subagent",
            "agent_id": "agent_001",
            "subagent_type": "Explore",
            "duration_ms": 1000,
            "total_tokens": 100,
            "tool_use_count": 1,
            "status": "completed",
            "models": ["m"],
            "tool_tokens": 0,
            "system_tokens": 0,
            "requests": [
                {
                    "t": 0.6,
                    "type": "n",
                    "model": "m",
                    "in": 128,
                    "out": 10,
                    "hash_ids": [1, 2, 7],
                    "api_time": 0.2,
                }
            ],
        },
        _row(t=1.0, hash_ids=[1, 2, 33], model="hk"),
        _row(t=2.0, hash_ids=[1, 2, 3, 4]),
        _row(t=3.0, hash_ids=[1, 2, 33, 34], model="hk"),
        _row(t=5.0, hash_ids=[1, 2, 3, 4, 5]),
    ]
    p = _write_trace(tmp_path / "mix.json", trace_id="mix", requests=reqs)
    loader = _loader_for(p)
    convs = _convs_by_sid(loader)
    root = convs["mix"]
    all_branch_ids = [b.branch_id for b in root.branches]
    assert len(all_branch_ids) == len(set(all_branch_ids)), all_branch_ids
    sa_ids = [b for b in all_branch_ids if ":spawn:" in b and ":flatspawn:" not in b]
    flat_ids = [b for b in all_branch_ids if ":flatspawn:" in b]
    assert sa_ids and flat_ids
    assert set(sa_ids).isdisjoint(set(flat_ids))
    emitted = set(convs)
    for b in root.branches:
        for cid in b.child_conversation_ids:
            assert cid in emitted, cid


def test_flat_child_observed_prefix_never_exceeds_first_request(tmp_path):
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=1.0, hash_ids=[50, 51, 52, 53], model="hk"),
        _row(t=1.5, hash_ids=[50, 51], model="hk"),
        _row(t=2.0, hash_ids=[1, 2, 3, 4]),
        _row(t=3.0, hash_ids=[50, 51, 52, 53, 54], model="hk"),
        _row(t=4.0, hash_ids=[50, 51, 60], model="hk"),
        _row(t=6.0, hash_ids=[1, 2, 3, 4, 5]),
    ]
    p = _write_trace(tmp_path / "pfx.json", trace_id="pfx", requests=reqs)
    loader = _loader_for(p)
    convs = _convs_by_sid(loader)
    flat = {sid: c for sid, c in convs.items() if sid.startswith("pfx::fa:")}
    assert len(flat) == 2
    for sid, c in flat.items():
        t0 = c.turns[0]
        roles = [m["role"] for m in t0.raw_messages]
        assert roles[0] in ("system", "user"), (sid, roles)
        if roles[0] == "system":
            assert t0.theoretical_prefix_cache_total_blocks is not None


def test_collapsed_group_branch_uses_min_chain_start(tmp_path):
    reqs = [
        _row(t=0.0, hash_ids=[1, 2, 3]),
        _row(t=0.30, hash_ids=[700, 701], model="wa", api_time=0.05),
        _row(t=0.10, hash_ids=[800, 801], model="wb", api_time=0.05),
        _row(t=5.0, hash_ids=[1, 2, 3, 4]),
    ]
    p = _write_trace(tmp_path / "grp.json", trace_id="grp", requests=reqs)
    loader = _loader_for(p)
    convs = _convs_by_sid(loader)
    root = convs["grp"]
    flat = [b for b in root.branches if ":flatspawn:" in b.branch_id]
    assert len(flat) == 1, "two co-grouped workers must collapse into one branch"
    b = flat[0]
    assert len(b.child_conversation_ids) == 2
    assert b.start_timestamp_ms == pytest.approx(0.10 * 1000.0), (
        "branch start must be the group-wide minimum chain start"
    )
