# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pathological / adversarial probes for the Weka trace loaders."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.dataset.loader.weka_trace import (
    WekaTraceLoader,
    _canonicalize_weka_nested_timestamps,
    _expand_subagent_to_child_plans,
    _IdleGapTimeWarp,
    _sa_end_seconds,
)
from aiperf.dataset.loader.weka_trace_models import (
    WekaNormalRequest,
    WekaSubagentEntry,
)

FIXTURES = Path(__file__).parents[3] / "fixtures" / "weka_traces"


# Shared harness (mirrors test_weka_trace.py / *_filters_adversarial.py)


def _mk_user_config(**overrides):
    from tests.unit.dataset.loader.conftest import make_weka_run

    overrides.setdefault(
        "model_names",
        ["claude-opus-4-5-20251101", "claude-haiku-4-5-20251001"],
    )
    return make_weka_run(**overrides)


def _stub_pg(loader) -> None:
    from tests.unit.dataset.loader.conftest import stub_hash_id_corpus_rng

    loader.prompt_generator = MagicMock()
    loader.prompt_generator._cache = {}
    loader.prompt_generator._sample_tokens.side_effect = lambda n: [0] * n
    loader.prompt_generator._tokenized_corpus = list(range(10000, 11000))
    loader.prompt_generator._corpus_size = 1000
    stub_hash_id_corpus_rng(loader.prompt_generator)
    loader.prompt_generator.tokenizer.decode.side_effect = lambda toks: (
        f"<dec:{len(toks)}>"
    )
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64


def _make_loader(path, uc):
    loader = WekaTraceLoader(filename=str(path), run=uc)
    _stub_pg(loader)
    return loader


def _normal(
    t: float,
    hash_ids: list[int],
    *,
    in_tokens: int = 64,
    out_tokens: int = 10,
    api_time: float = 1.0,
    think_time: float = 0.0,
    model: str = "claude-opus-4-5-20251101",
) -> dict:
    return {
        "t": t,
        "type": "n",
        "model": model,
        "in": in_tokens,
        "out": out_tokens,
        "hash_ids": hash_ids,
        "input_types": ["text"],
        "output_types": ["text"],
        "stop": "end_turn",
        "api_time": api_time,
        "think_time": think_time,
    }


def _subagent(
    t: float,
    agent_id: str,
    *,
    duration_ms: int | None = 1000,
    inner_hash_ids: list[int] | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    inner_hash_ids = inner_hash_ids if inner_hash_ids is not None else [8]
    return {
        "t": t,
        "type": "subagent",
        "agent_id": agent_id,
        "subagent_type": "Explore",
        "duration_ms": duration_ms,
        "total_tokens": 10,
        "tool_use_count": 1,
        "status": "completed",
        "requests": [_normal(t, inner_hash_ids, model=model)],
        "models": [model],
        "tool_tokens": 0,
        "system_tokens": 0,
    }


def _base_trace(requests: list[dict], trace_id: str = "trace") -> dict:
    return {
        "id": trace_id,
        "models": ["claude-opus-4-5-20251101", "claude-haiku-4-5-20251001"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": requests,
    }


def _make_subagent_entry(**overrides) -> WekaSubagentEntry:
    base = {
        "t": 10.0,
        "type": "subagent",
        "agent_id": "a",
        "subagent_type": "Explore",
        "duration_ms": None,
        "total_tokens": None,
        "tool_use_count": None,
        "status": "completed",
        "requests": [],
        "models": ["m"],
        "tool_tokens": 0,
        "system_tokens": 0,
    }
    base.update(overrides)
    return WekaSubagentEntry.model_validate(base)


def _inner_request(**overrides) -> WekaNormalRequest:
    base = {
        "t": 0.0,
        "type": "n",
        "model": "m",
        "in": 10,
        "out": 5,
        "hash_ids": [1],
        "api_time": 1.0,
    }
    base.update(overrides)
    return WekaNormalRequest.model_validate(base)


# Regression: idle-gap-mapped subagent spawn time (fixed)


def test_idle_gap_branch_start_timestamp_uses_mapped_time_not_raw(tmp_path):
    """SPAWN ``start_timestamp_ms`` must live on the same mapped timeline as every other turn, never exceeding the maximum mapped turn timestamp."""
    trace = _base_trace(
        [
            _normal(0.0, [1]),
            _normal(1000.0, [1, 2]),  # 1000s start-gap -> compressed
            _subagent(1005.0, "a", inner_hash_ids=[8]),
            _normal(1006.0, [1, 2, 3]),
        ],
        trace_id="idle_branch",
    )
    path = tmp_path / "t.json"
    path.write_text(json.dumps(trace))
    uc = _mk_user_config(trace_idle_gap_cap_seconds=60.0)
    loader = _make_loader(path, uc)

    convs = loader.convert_to_conversations(loader.load_dataset())
    root = next(c for c in convs if c.session_id == "idle_branch")
    child = next(c for c in convs if c.session_id == "idle_branch::sa:a")

    max_mapped_ms = max(t.timestamp for t in root.turns)
    branch = root.branches[0]
    # The branch entered the timeline when the subagent spawned; on the mapped
    # timeline that is the child's first-request timestamp, never ~940s later.
    assert branch.start_timestamp_ms <= max_mapped_ms
    assert branch.start_timestamp_ms == child.turns[0].timestamp


def test_parallel_subagent_payload_carries_mapped_spawn_time(tmp_path):
    """The parallel marker payload must carry the mapped spawn time via ``effective_t``, not revert to raw seconds when an idle-gap warp shifts the timeline."""
    trace = _base_trace(
        [
            _normal(0.0, [1]),
            _normal(1000.0, [1, 2]),
            _subagent(1005.0, "a", inner_hash_ids=[8]),
            _normal(1006.0, [1, 2, 3]),
        ],
        trace_id="idle_parallel",
    )
    path = tmp_path / "t.json"
    path.write_text(json.dumps(trace))
    uc = _mk_user_config(trace_idle_gap_cap_seconds=60.0)
    loader = _make_loader(path, uc)

    data = loader.load_dataset()
    plans = loader._build_reconstruction_plans(data)
    parent_plans, child_plans = plans.parent_plans, plans.child_plans
    timing = loader._build_trace_idle_timing_by_trace(parent_plans, child_plans)
    metric_values = loader._build_shared_metric_values(
        parent_plans, child_plans, plans.flat_plans
    )
    tasks = loader._build_parallel_reconstruction_tasks(
        parent_plans=parent_plans,
        child_plans=child_plans,
        data=data,
        ignore_delays=False,
        think_time_only=False,
        cap_seconds=None,
        model_map_per_trace={"idle_parallel": {}},
        trace_idle_timing_by_trace=timing,
        metric_values_by_trace=metric_values,
    )
    _, marker = tasks[0].parent["subagents"][0]
    # The mapped end time is plumbed through; the mapped spawn time must be too.
    assert "effective_t" in marker
    assert marker["effective_t"] != marker["t"]


def test_sa_end_seconds_negative_duration_not_before_spawn():
    """A subagent's recorded end time can never precede its own spawn, even with a corrupt negative ``duration_ms``."""
    entry = _make_subagent_entry(t=10.0, duration_ms=-5000)
    assert _sa_end_seconds(entry) >= entry.t


def test_sa_end_seconds_nan_inner_api_time_is_finite():
    """A NaN inner ``api_time`` must not poison the duration_ms=None fallback subagent end time into NaN."""
    entry = _make_subagent_entry(
        t=10.0,
        duration_ms=None,
        requests=[_inner_request(t=20.0, api_time=float("nan"))],
    )
    end = _sa_end_seconds(entry)
    assert end == end  # not NaN  # noqa: PLR0124


def test_think_time_only_negative_think_time_not_negative_delay(tmp_path):
    """A recorded negative ``think_time`` must not become a negative inter-turn delay in ``--use-think-time-only`` mode."""
    trace = _base_trace(
        [
            _normal(0.0, [1], think_time=0.0),
            _normal(10.0, [1, 2], think_time=-3.0),
        ],
        trace_id="neg_tt",
    )
    path = tmp_path / "t.json"
    path.write_text(json.dumps(trace))
    uc = _mk_user_config(use_think_time_only=True)
    loader = _make_loader(path, uc)

    convs = loader.convert_to_conversations(loader.load_dataset())
    assert convs[0].turns[1].delay >= 0.0


# PASSING CHARACTERIZATIONS (surprising but intended / not invariant-breaking)


def test_idle_gap_exactly_equal_to_cap_is_not_compressed():
    """A request-start gap exactly equal to the cap is left untouched (``_IdleGapTimeWarp`` compresses only strict ``gap_seconds > cap_seconds``)."""
    warp = _IdleGapTimeWarp([0.0, 60.0], cap_seconds=60.0)
    assert warp.map(60.0) == 60.0
    # One microsecond over the cap does get compressed back to the boundary.
    warp_over = _IdleGapTimeWarp([0.0, 60.001], cap_seconds=60.0)
    assert warp_over.map(60.001) == pytest.approx(60.0)


def test_idle_gap_collapsed_tail_event_maps_to_cap_boundary():
    """A non-request event inside a collapsed gap tail is pinned to ``raw_start + cap`` so a join cannot wait past the next shifted request."""
    warp = _IdleGapTimeWarp([0.0, 20.0, 220.0], cap_seconds=60.0)
    assert warp.map(80.0) == pytest.approx(80.0)  # at the boundary
    assert warp.map(150.0) == pytest.approx(80.0)  # deep in the collapsed tail
    assert warp.map(220.0) == pytest.approx(80.0)  # the gap end
    assert warp.map(300.0) == pytest.approx(160.0)  # after: shifted left by excess


def test_nested_chain_nan_api_time_treated_as_zero_duration():
    """A NaN inner ``api_time`` is clamped to zero duration in chain detection, so a same-context continuation still extends the chain."""
    entry = _make_subagent_entry(
        t=0.0,
        requests=[
            _inner_request(t=0.0, api_time=float("nan"), hash_ids=[1]).model_dump(
                by_alias=True
            ),
            _inner_request(t=100.0, api_time=1.0, hash_ids=[1, 2]).model_dump(
                by_alias=True
            ),
        ],
    )
    plans = _expand_subagent_to_child_plans("tr", 0, 0, entry, 64)
    assert [p.session_id for p in plans] == ["tr::sa:a"]
    assert [r.hash_ids for r in plans[0].requests] == [[1], [1, 2]]


def test_nested_chain_infinite_api_time_does_not_block_extension():
    """A +inf inner ``api_time`` must not permanently occupy a chain tail and explode an N-request subagent into N one-turn children."""
    entry = _make_subagent_entry(
        t=0.0,
        requests=[
            _inner_request(t=0.0, api_time=float("inf"), hash_ids=[1]).model_dump(
                by_alias=True
            ),
            _inner_request(t=10.0, api_time=1.0, hash_ids=[1, 2]).model_dump(
                by_alias=True
            ),
            _inner_request(t=20.0, api_time=1.0, hash_ids=[1, 2, 3]).model_dump(
                by_alias=True
            ),
        ],
    )
    plans = _expand_subagent_to_child_plans("tr", 0, 0, entry, 64)
    assert [p.session_id for p in plans] == ["tr::sa:a"]
    assert len(plans[0].requests) == 3


def test_nested_chain_equal_t_disjoint_requests_split_deterministically():
    """Equal-``t`` context-disjoint inner requests split deterministically via stable ``(t, index)`` order (pinned for reproducibility)."""
    entry = _make_subagent_entry(
        t=5.0,
        requests=[
            _inner_request(t=5.0, api_time=None, hash_ids=[3]).model_dump(
                by_alias=True
            ),
            _inner_request(t=5.0, api_time=None, hash_ids=[1]).model_dump(
                by_alias=True
            ),
            _inner_request(t=5.0, api_time=None, hash_ids=[2]).model_dump(
                by_alias=True
            ),
        ],
    )
    plans = _expand_subagent_to_child_plans("tr", 0, 0, entry, 64)
    assert [p.session_id for p in plans] == ["tr::sa:a", "tr::sa:a:fa:000"]
    assert [r.hash_ids[0] for r in plans[0].requests] == [3, 1]
    assert [r.hash_ids[0] for r in plans[1].requests] == [2]


def test_public_loader_absolute_rejects_request_before_marker_with_source(tmp_path):
    trace = _base_trace(
        [_normal(0.0, [1]), _subagent(100.0, "a"), _normal(200.0, [1, 2])],
        trace_id="bad_absolute",
    )
    trace["requests"][1]["requests"][0]["t"] = 10.0
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config(weka_nested_timestamp_basis="absolute"))
    with pytest.raises(
        DatasetLoaderError,
        match=(
            r"Trace 'bad_absolute' in .*bad\.json, subagent 'a': inner "
            r"request\[0\] timestamp 10\.0 precedes its marker timestamp 100\.0"
        ),
    ):
        loader.convert_to_conversations(loader.load_dataset())


def test_absolute_timestamp_within_epsilon_clamps_to_marker(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="aiperf.dataset.loader.weka_trace")
    trace = _base_trace(
        [_normal(0.0, [1]), _subagent(100.0, "a"), _normal(200.0, [1, 2])],
        trace_id="jitter",
    )
    trace["requests"][1]["requests"][0]["t"] = 99.9999995
    path = tmp_path / "jitter.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config(weka_nested_timestamp_basis="absolute"))

    conversations = loader.convert_to_conversations(loader.load_dataset())
    child = next(c for c in conversations if c.session_id == "jitter::sa:a")
    assert child.turns[0].timestamp == pytest.approx(100_000.0)
    assert "basis 'absolute' (basis was explicitly configured)" in caplog.text
    assert "clamped 1 sub-microsecond pre-marker timestamp" in caplog.text


@pytest.mark.parametrize("basis", ["auto", "relative"])
def test_relative_nested_timestamps_canonicalize_for_whole_corpus(
    tmp_path, caplog, basis
):
    caplog.set_level(logging.INFO, logger="aiperf.dataset.loader.weka_trace")
    trace = _base_trace(
        [_normal(0.0, [1]), _subagent(100.0, "a"), _normal(200.0, [1, 2])],
        trace_id="raw_weka",
    )
    trace["requests"][1]["requests"] = [
        _normal(0.0, [8], model="claude-haiku-4-5-20251001"),
        _normal(5.0, [8, 9], model="claude-haiku-4-5-20251001"),
    ]
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config(weka_nested_timestamp_basis=basis))

    conversations = loader.convert_to_conversations(loader.load_dataset())
    child = next(c for c in conversations if c.session_id == "raw_weka::sa:a")
    assert [turn.timestamp for turn in child.turns] == pytest.approx(
        [100_000.0, 105_000.0]
    )
    expected_decision = (
        "auto heuristic found 2 nested request(s)"
        if basis == "auto"
        else "basis was explicitly configured"
    )
    assert f"basis 'relative' ({expected_decision}" in caplog.text
    if basis == "auto":
        assert "heuristic is not proof of the producer format" in caplog.text


def test_preflighted_relative_corpus_survives_copied_and_reordered_containers(tmp_path):
    trace = _base_trace(
        [_normal(0.0, [1]), _subagent(100.0, "a"), _normal(200.0, [1, 2])],
        trace_id="repeat",
    )
    trace["requests"][1]["requests"][0]["t"] = 0.0
    path = tmp_path / "repeat.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config(weka_nested_timestamp_basis="relative"))
    normalized, _ = loader.preflight_nested_timestamps(loader.load_dataset())
    copied = {
        trace_id: list(reversed(traces)) for trace_id, traces in normalized.items()
    }
    deep_copied = {
        trace_id: [trace.model_copy(deep=True) for trace in traces]
        for trace_id, traces in reversed(list(normalized.items()))
    }
    copied_again, repeated_resolution = loader.preflight_nested_timestamps(copied)

    first = loader.convert_to_conversations(normalized)
    second = loader.convert_to_conversations(copied)
    third = loader.convert_to_conversations(deep_copied)
    first_child = next(c for c in first if c.session_id == "repeat::sa:a")
    second_child = next(c for c in second if c.session_id == "repeat::sa:a")
    third_child = next(c for c in third if c.session_id == "repeat::sa:a")
    assert repeated_resolution.basis == "relative"
    assert copied_again["repeat"][0].requests[1].requests[0].t == 100.0
    assert second_child.turns[0].timestamp == first_child.turns[0].timestamp
    assert third_child.turns[0].timestamp == first_child.turns[0].timestamp


def test_auto_uses_any_early_child_as_corpus_wide_relative_heuristic(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="aiperf.dataset.loader.weka_trace")
    absolute = _base_trace(
        [_normal(0.0, [1]), _subagent(10.0, "abs")], trace_id="absolute"
    )
    relative = _base_trace(
        [_normal(0.0, [1]), _subagent(10.0, "rel")], trace_id="relative"
    )
    relative["requests"][1]["requests"][0]["t"] = 0.0
    (tmp_path / "a.json").write_text(json.dumps(absolute))
    (tmp_path / "r.json").write_text(json.dumps(relative))
    loader = _make_loader(tmp_path, _mk_user_config(weka_nested_timestamp_basis="auto"))

    conversations = loader.convert_to_conversations(loader.load_dataset())
    absolute_child = next(
        c for c in conversations if c.session_id == "absolute::sa:abs"
    )
    relative_child = next(
        c for c in conversations if c.session_id == "relative::sa:rel"
    )
    assert absolute_child.turns[0].timestamp == pytest.approx(20_000.0)
    assert relative_child.turns[0].timestamp == pytest.approx(10_000.0)
    assert "selected relative for the whole corpus" in caplog.text
    assert "heuristic is not proof" in caplog.text
    assert "cannot reliably detect mixed producer conventions" in caplog.text


def test_auto_without_early_child_selects_absolute_for_ambiguous_values(
    tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger="aiperf.dataset.loader.weka_trace")
    trace = _base_trace(
        [_normal(0.0, [1]), _subagent(10.0, "ambiguous")], trace_id="ambiguous"
    )
    trace["requests"][1]["requests"][0]["t"] = 20.0
    path = tmp_path / "ambiguous.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config(weka_nested_timestamp_basis="auto"))

    conversations = loader.convert_to_conversations(loader.load_dataset())
    child = next(c for c in conversations if c.session_id == "ambiguous::sa:ambiguous")
    assert child.turns[0].timestamp == pytest.approx(20_000.0)
    assert "selected absolute for the whole corpus" in caplog.text


def test_auto_preserves_delayed_absolute_children(tmp_path):
    trace = _base_trace(
        [_normal(0.0, [1]), _subagent(10.0, "absolute")], trace_id="absolute"
    )
    trace["requests"][1]["requests"] = [
        _normal(10.2, [8], model="claude-haiku-4-5-20251001"),
        _normal(11.0, [8, 9], model="claude-haiku-4-5-20251001"),
    ]
    path = tmp_path / "absolute.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config(weka_nested_timestamp_basis="auto"))

    normalized, resolution = loader.preflight_nested_timestamps(loader.load_dataset())
    nested = normalized["absolute"][0].requests[1].requests
    assert resolution.basis == "absolute"
    assert [request.t for request in nested] == [10.2, 11.0]


def test_auto_inspects_later_children_not_only_first_anchor(tmp_path):
    trace = _base_trace([_normal(0.0, [1]), _subagent(10.0, "tail")], trace_id="tail")
    trace["requests"][1]["requests"] = [
        _normal(10.0, [8], model="claude-haiku-4-5-20251001"),
        _normal(5.0, [8, 9], model="claude-haiku-4-5-20251001"),
    ]
    path = tmp_path / "tail.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config(weka_nested_timestamp_basis="auto"))

    normalized, resolution = loader.preflight_nested_timestamps(loader.load_dataset())
    nested = normalized["tail"][0].requests[1].requests
    assert resolution.basis == "relative"
    assert [request.t for request in nested] == [20.0, 15.0]


def test_auto_delayed_relative_start_uses_early_child_evidence(tmp_path):
    trace = _base_trace(
        [_normal(0.0, [1]), _subagent(100.0, "delayed")], trace_id="delayed"
    )
    trace["requests"][1]["requests"][0]["t"] = 50.0
    path = tmp_path / "delayed.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config(weka_nested_timestamp_basis="auto"))

    normalized, resolution = loader.preflight_nested_timestamps(loader.load_dataset())
    assert resolution.basis == "relative"
    assert normalized["delayed"][0].requests[1].requests[0].t == 150.0


def test_explicit_relative_overrides_coincidental_absolute_looking_anchor(
    tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger="aiperf.dataset.loader.weka_trace")
    trace = _base_trace(
        [_normal(0.0, [1]), _subagent(10.0, "coincidental")],
        trace_id="coincidental",
    )
    path = tmp_path / "coincidental.json"
    path.write_text(json.dumps(trace))
    auto_loader = _make_loader(
        path, _mk_user_config(weka_nested_timestamp_basis="auto")
    )
    relative_loader = _make_loader(
        path, _mk_user_config(weka_nested_timestamp_basis="relative")
    )

    auto_data, auto_resolution = auto_loader.preflight_nested_timestamps(
        auto_loader.load_dataset()
    )
    relative_conversations = relative_loader.convert_to_conversations(
        relative_loader.load_dataset()
    )
    relative_child = next(
        c
        for c in relative_conversations
        if c.session_id == "coincidental::sa:coincidental"
    )
    assert auto_resolution.basis == "absolute"
    assert auto_data["coincidental"][0].requests[1].requests[0].t == 10.0
    assert relative_child.turns[0].timestamp == pytest.approx(20_000.0)
    assert "conflict with absolute-looking anchor evidence" in caplog.text


def test_auto_zero_marker_is_equivalent_and_no_subagent_is_not_applicable(
    tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger="aiperf.dataset.loader.weka_trace")
    zero = _base_trace([_normal(0.0, [1]), _subagent(0.0, "zero")], trace_id="zero")
    zero_path = tmp_path / "zero.json"
    zero_path.write_text(json.dumps(zero))
    zero_loader = _make_loader(
        zero_path, _mk_user_config(weka_nested_timestamp_basis="auto")
    )
    zero_loader.convert_to_conversations(zero_loader.load_dataset())
    assert "relative and absolute interpretations are equivalent" in caplog.text

    caplog.clear()
    plain = _base_trace([_normal(0.0, [1])], trace_id="plain")
    plain_path = tmp_path / "plain.json"
    plain_path.write_text(json.dumps(plain))
    plain_loader = _make_loader(
        plain_path, _mk_user_config(weka_nested_timestamp_basis="auto")
    )
    plain_loader.convert_to_conversations(plain_loader.load_dataset())
    assert "timestamp basis is not applicable" in caplog.text


def test_constructor_rejects_invalid_timestamp_basis(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(_base_trace([_normal(0.0, [1])], trace_id="trace")))
    with pytest.raises(ValueError, match="nested_timestamp_basis must be"):
        WekaTraceLoader(
            filename=str(path),
            run=_mk_user_config(),
            nested_timestamp_basis="sometimes",
        )
    with pytest.raises(ValueError, match="configured_basis must be"):
        _canonicalize_weka_nested_timestamps(
            {}, configured_basis="sometimes", source_by_trace={}
        )


def test_preflight_rejects_mixed_raw_and_canonical_objects(tmp_path):
    first = _base_trace([_normal(0.0, [1])], trace_id="first")
    second = _base_trace([_normal(0.0, [2])], trace_id="second")
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first))
    second_path.write_text(json.dumps(second))
    loader = _make_loader(
        first_path, _mk_user_config(weka_nested_timestamp_basis="auto")
    )
    canonical, _ = loader.preflight_nested_timestamps(loader.load_dataset())
    raw_second = WekaTraceLoader(
        filename=str(second_path), run=_mk_user_config()
    ).load_dataset()
    mixed = {**canonical, **raw_second}

    with pytest.raises(DatasetLoaderError, match="mixes raw and already-canonicalized"):
        loader.preflight_nested_timestamps(mixed)


@pytest.mark.parametrize("invalid", ["NaN", "Infinity", -1.0, 1e308])
def test_corpus_preflight_rejects_invalid_nested_timestamp(tmp_path, invalid):
    trace = _base_trace([_normal(0.0, [1]), _subagent(10.0, "bad")], trace_id="invalid")
    trace["requests"][1]["requests"][0]["t"] = invalid
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config(weka_nested_timestamp_basis="absolute"))

    with pytest.raises(DatasetLoaderError, match=r"must be finite, non-negative"):
        loader.convert_to_conversations(loader.load_dataset())


def test_relative_canonical_timestamp_rejects_millisecond_overflow(tmp_path):
    trace = _base_trace(
        [_normal(0.0, [1]), _subagent(1e305, "overflow")], trace_id="overflow"
    )
    trace["requests"][1]["requests"][0]["t"] = 1e305
    path = tmp_path / "overflow.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config(weka_nested_timestamp_basis="relative"))

    with pytest.raises(
        DatasetLoaderError,
        match=r"canonical inner request\[0\].*representable in milliseconds",
    ):
        loader.convert_to_conversations(loader.load_dataset())


def test_spawned_chain_inherits_declared_prefix_only_when_proven():
    """Spawned-chain turn-0 tool/system attribution requires hash proof; only a fork whose first request matches the declared-prefix blocks inherits them."""

    def entry_with(worker_hash: list[int]) -> WekaSubagentEntry:
        # Main thread = r0 + r2 (r2 extends r0's prefix, anchoring the main
        # chain so the preamble rule cannot peel r0); the worker chain forks
        # off at t=1 with ``worker_hash``.
        return _make_subagent_entry(
            t=0.0,
            tool_tokens=128,
            system_tokens=64,
            requests=[
                _inner_request(
                    t=0.0, api_time=0.5, hash_ids=[1, 2, 3, 4], **{"in": 256}
                ).model_dump(by_alias=True),
                _inner_request(
                    t=1.0, api_time=100.0, hash_ids=worker_hash, **{"in": 256}
                ).model_dump(by_alias=True),
                _inner_request(
                    t=2.0, api_time=0.5, hash_ids=[1, 2, 3, 4, 5], **{"in": 320}
                ).model_dump(by_alias=True),
            ],
        )

    # declared_blocks = ceil((128 + 64) / 64) = 3; [1, 2, 3] matches the main
    # chain's first request, so the fork provably carries the declared prefix.
    proven = _expand_subagent_to_child_plans("tr", 0, 0, entry_with([1, 2, 3, 9]), 64)
    assert [p.session_id for p in proven] == ["tr::sa:a", "tr::sa:a:fa:000"]
    assert (proven[0].init_tool_tokens, proven[0].init_system_tokens) == (128, 64)
    assert (proven[1].init_tool_tokens, proven[1].init_system_tokens) == (128, 64)

    unproven = _expand_subagent_to_child_plans(
        "tr", 0, 0, entry_with([7, 8, 9, 10]), 64
    )
    assert [p.session_id for p in unproven] == ["tr::sa:a", "tr::sa:a:fa:000"]
    assert (unproven[0].init_tool_tokens, unproven[0].init_system_tokens) == (128, 64)
    assert (unproven[1].init_tool_tokens, unproven[1].init_system_tokens) == (0, 0)


def test_split_chains_disabled_emits_one_sequential_child():
    """``split_chains=False`` skips nested detection, emitting one child with every inner request in time order."""
    entry = _make_subagent_entry(
        t=0.0,
        requests=[
            _inner_request(t=0.0, api_time=100.0, hash_ids=[1]).model_dump(
                by_alias=True
            ),
            _inner_request(t=1.0, api_time=100.0, hash_ids=[50]).model_dump(
                by_alias=True
            ),
        ],
    )
    plans = _expand_subagent_to_child_plans("tr", 0, 0, entry, 64, split_chains=False)
    assert [p.session_id for p in plans] == ["tr::sa:a"]
    assert [r.hash_ids for r in plans[0].requests] == [[1], [50]]


def test_absolute_inner_timestamps_emit_root_timeline_child_turns(tmp_path):
    sa = {
        "t": 10.0,
        "type": "subagent",
        "agent_id": "a",
        "subagent_type": "Explore",
        "duration_ms": 7000,
        "total_tokens": 10,
        "tool_use_count": 1,
        "status": "completed",
        "requests": [
            _inner_request(t=10.0, api_time=1.0, hash_ids=[8]).model_dump(
                by_alias=True
            ),
            _inner_request(t=15.0, api_time=1.0, hash_ids=[8, 9]).model_dump(
                by_alias=True
            ),
        ],
        "models": ["claude-haiku-4-5-20251001"],
        "tool_tokens": 0,
        "system_tokens": 0,
    }
    trace = _base_trace(
        [_normal(0.0, [1]), sa, _normal(40.0, [1, 2])],
        trace_id="rel_inner",
    )
    path = tmp_path / "t.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config())
    convs = loader.convert_to_conversations(loader.load_dataset())

    child = next(c for c in convs if c.session_id == "rel_inner::sa:a")
    assert child.turns[0].timestamp == pytest.approx(10_000.0)
    assert child.turns[1].timestamp == pytest.approx(15_000.0)
    # end-to-start idle gap: 5.0s start-to-start minus 1.0s prev api_time.
    assert child.turns[1].delay == pytest.approx(4_000.0)


def test_duplicate_subagent_agent_id_in_one_trace_raises(tmp_path):
    """Two retained subagent entries sharing an ``agent_id`` are rejected, since session and branch ids are derived from ``agent_id``."""
    trace = _base_trace(
        [
            _normal(0.0, [1]),
            _subagent(1.0, "a"),
            _normal(5.0, [1, 2]),
            _subagent(6.0, "a"),
            _normal(10.0, [1, 2, 3]),
        ],
        trace_id="dup_agent",
    )
    path = tmp_path / "t.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config())
    with pytest.raises(DatasetLoaderError, match="duplicate subagent agent_id"):
        loader.convert_to_conversations(loader.load_dataset())


def test_duplicate_hash_ids_in_request_inflate_theoretical_hit_to_full(tmp_path):
    """Duplicate hash-ids within one request drive theoretical hits to 100% (no per-request de-dup) while the hit<=total invariant still holds."""
    trace = _base_trace(
        [
            _normal(0.0, [1, 2], in_tokens=128),
            _normal(1.0, [1, 2, 1, 2], in_tokens=256),
        ],
        trace_id="dup_hash",
    )
    path = tmp_path / "t.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config())

    convs = loader.convert_to_conversations(loader.load_dataset())
    turn1 = convs[0].turns[1]
    assert turn1.theoretical_prefix_cache_hit_blocks == 4
    assert turn1.theoretical_prefix_cache_total_blocks == 4
    assert (
        turn1.theoretical_prefix_cache_hit_blocks
        <= turn1.theoretical_prefix_cache_total_blocks
    )


def test_empty_requests_trace_reconstructs_empty_conversation(tmp_path):
    """A trace with zero requests yields a single empty Conversation, no crash."""
    trace = _base_trace([], trace_id="empty_trace")
    path = tmp_path / "t.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config())

    convs = loader.convert_to_conversations(loader.load_dataset())
    assert len(convs) == 1
    assert convs[0].session_id == "empty_trace"
    assert convs[0].turns == []
    assert convs[0].branches == []


def test_zero_duration_subagent_joins_first_following_turn(tmp_path):
    """A duration_ms=0 subagent (end == spawn) joins the first later parent turn, producing a SPAWN_JOIN prereq rather than a background branch."""
    trace = _base_trace(
        [
            _normal(0.0, [1]),
            _subagent(1.0, "a", duration_ms=0),
            _normal(2.0, [1, 2]),
        ],
        trace_id="zero_dur",
    )
    path = tmp_path / "t.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config())

    convs = loader.convert_to_conversations(loader.load_dataset())
    root = next(c for c in convs if c.session_id == "zero_dur")
    assert root.branches[0].is_background is False
    assert len(root.turns[1].prerequisites) == 1
