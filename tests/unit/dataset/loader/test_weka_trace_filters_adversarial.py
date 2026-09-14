# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial filter-boundary tests for WekaTraceLoader."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from tests.unit.dataset.loader._shared_helpers import _write_trace

FIXTURES = Path(__file__).parents[3] / "fixtures" / "weka_traces"


def _mk_user_config(*, max_isl=None, max_osl=None, **overrides):
    from tests.unit.dataset.loader.conftest import make_weka_run

    overrides.setdefault(
        "model_names",
        ["claude-opus-4-5-20251101", "claude-haiku-4-5-20251001", "m"],
    )
    # Fixed-schedule start/end offsets are NOT threaded through the v2 config:
    # the FixedSchedulePhase fields are ``int >= 0`` with a ``start <= end``
    # validator, so this suite's adversarial offsets (negative, inverted,
    # fractional) are unrepresentable in a real config. They are applied to the
    # loader's resolved attributes post-construction instead (see _make_loader),
    # which is exactly what those instance fields exist for and exercises the
    # same _request_passes_filters comparison the tests pin.
    return make_weka_run(
        tokenizer_name="t",
        max_isl=max_isl,
        max_osl=max_osl,
        **overrides,
    )


def _make_loader(filename, uc, monkeypatch, *, start=None, end=None):
    loader = WekaTraceLoader(filename=str(filename), run=uc)
    if start is not None:
        loader._fixed_schedule_start_offset = start
    if end is not None:
        loader._fixed_schedule_end_offset = end
    monkeypatch.setattr(
        loader,
        "synthesize_prompts_from_hash_ids",
        lambda rs: {r.key: f"p-{r.key}" for r in rs},
    )
    loader.prompt_generator = MagicMock()
    loader.prompt_generator._cache = {}
    loader.prompt_generator._sample_tokens.side_effect = lambda n: [0] * n
    loader.prompt_generator._tokenized_corpus = list(range(10000, 11000))
    loader.prompt_generator._corpus_size = 1000
    from tests.unit.dataset.loader.conftest import stub_hash_id_corpus_rng

    stub_hash_id_corpus_rng(loader.prompt_generator)
    loader.prompt_generator.tokenizer.decode.side_effect = lambda toks: (
        f"<dec:{len(toks)}>"
    )
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    return loader


def _normal(t, in_tokens, out_tokens, hash_ids, model="m"):
    """Build one WekaNormalRequest dict with required fields."""
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
        "api_time": 1.0,
        "think_time": 0.0,
    }


def _subagent(t, agent_id, inner_requests, model="m"):
    """Build one WekaSubagentEntry dict."""
    return {
        "t": t,
        "type": "subagent",
        "agent_id": agent_id,
        "subagent_type": "Explore",
        "duration_ms": 1000,
        "total_tokens": 100,
        "tool_use_count": 1,
        "status": "completed",
        "requests": inner_requests,
        "models": [model],
        "tool_tokens": 10,
        "system_tokens": 5,
    }


def _base_trace(requests, trace_id="t", model="m"):
    return {
        "id": trace_id,
        "models": [model],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": requests,
    }


# Boundary equality: filter is strict > / <


def test_max_isl_equals_input_length_keeps_request(tmp_path, monkeypatch):
    """`max_isl == input_length` is NOT filtered (strict `>` comparison)."""
    data = _base_trace([_normal(0.0, 100, 10, [1, 2])])
    path = _write_trace(tmp_path, data)
    uc = _mk_user_config(max_isl=100)
    loader = _make_loader(path, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert len(convs[0].turns) == 1


def test_max_isl_one_less_than_input_length_drops_request(tmp_path, monkeypatch):
    """`max_isl == input_length - 1` filters the request out."""
    data = _base_trace([_normal(0.0, 100, 10, [1, 2])])
    path = _write_trace(tmp_path, data)
    uc = _mk_user_config(max_isl=99)
    loader = _make_loader(path, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert len(convs[0].turns) == 0


def test_max_context_length_drops_trace_with_oversized_subagent(tmp_path, monkeypatch):
    """The context cap applies to subagent branches, not just parent turns."""
    good = _base_trace([_normal(0.0, 100, 10, [1, 2])], trace_id="good")
    bad = _base_trace(
        [
            _normal(0.0, 100, 10, [1, 2]),
            _subagent(
                1.0,
                "agent_oversized",
                [_normal(1.0, 2000, 10, [3, 4], model="m")],
            ),
        ],
        trace_id="bad",
    )
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    _write_trace(traces_dir, good, name="good.json")
    _write_trace(traces_dir, bad, name="bad.json")

    uc = _mk_user_config(max_context_length=1000)
    loader = _make_loader(traces_dir, uc, monkeypatch)

    convs = loader.convert_to_conversations(loader.load_dataset())
    assert [conv.session_id for conv in convs] == ["good"]


def test_max_context_length_includes_requested_output(tmp_path, monkeypatch):
    """The context cap must account for prompt tokens plus max_tokens."""
    good = _base_trace([_normal(0.0, 900, 99, [1, 2])], trace_id="good")
    bad = _base_trace([_normal(0.0, 900, 101, [1, 2])], trace_id="bad")
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    _write_trace(traces_dir, good, name="good.json")
    _write_trace(traces_dir, bad, name="bad.json")

    uc = _mk_user_config(max_context_length=1000)
    loader = _make_loader(traces_dir, uc, monkeypatch)

    convs = loader.convert_to_conversations(loader.load_dataset())
    assert [conv.session_id for conv in convs] == ["good"]


def test_max_context_keep_drop_uses_uncapped_subagent_output(tmp_path, monkeypatch):
    """The max-context keep/drop peak must use the subagent child's uncapped output, since --max-osl caps parent/flat output but not subagent-child output."""
    # max_context=5000, max_osl=2000.
    #  - parent in=100,out=10 -> capped peak 110 (well under).
    #  - subagent inner in=1000,out=10000:
    #      buggy (capped)   = 1000 + min(10000, 2000) = 3000 <= 5000 -> KEPT
    #      fixed (uncapped) = 1000 + 10000            = 11000 > 5000 -> DROPPED
    good = _base_trace([_normal(0.0, 100, 10, [1, 2])], trace_id="good")
    bad = _base_trace(
        [
            _normal(0.0, 100, 10, [1, 2]),
            _subagent(1.0, "agent_big_out", [_normal(1.0, 1000, 10000, [3, 4])]),
        ],
        trace_id="bad",
    )
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    _write_trace(traces_dir, good, name="good.json")
    _write_trace(traces_dir, bad, name="bad.json")

    uc = _mk_user_config(max_osl=2000, max_context_length=5000)
    loader = _make_loader(traces_dir, uc, monkeypatch)

    convs = loader.convert_to_conversations(loader.load_dataset())
    assert [conv.session_id for conv in convs] == ["good"]


@pytest.mark.xfail(
    reason="v2 divergence: Turn.max_tokens is now ge=1 (main #889/#942 hardening) "
    "and SynthesisConfig.max_osl is ge=1, so max_osl=0 -> max_tokens=0 is "
    "unrepresentable in v2. The v1-only 'cap to zero' behavior cannot be "
    "reproduced without relaxing main's model constraint.",
    raises=Exception,
    strict=True,
)
def test_max_osl_zero_caps_all_outputs_to_zero(monkeypatch):
    """`max_osl=0` caps every turn's max_tokens to zero (not falsy-skipped)."""
    uc = _mk_user_config()
    loader = _make_loader(FIXTURES / "simple.json", uc, monkeypatch)
    loader._max_osl = 0
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert len(convs[0].turns) == 2
    for turn in convs[0].turns:
        assert turn.max_tokens == 0


def test_max_osl_greater_than_output_preserves_output(monkeypatch):
    """`max_osl > output_length` leaves max_tokens at the original value."""
    uc = _mk_user_config(max_osl=1000)
    loader = _make_loader(FIXTURES / "simple.json", uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert convs[0].turns[0].max_tokens == 30
    assert convs[0].turns[1].max_tokens == 40


def test_schedule_start_offset_equal_to_request_timestamp_keeps(tmp_path, monkeypatch):
    """`start == req.t` is kept (strict `req.t < start`); trace `t` is seconds and the start offset is ms, so t=5.0s equals a 5000ms offset."""
    data = _base_trace([_normal(5.0, 50, 10, [1])])
    path = _write_trace(tmp_path, data)
    uc = _mk_user_config()
    loader = _make_loader(path, uc, monkeypatch, start=5000.0)
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert len(convs[0].turns) == 1


def test_schedule_end_offset_equal_to_request_timestamp_keeps(tmp_path, monkeypatch):
    """`end == req.t` is KEPT (filter compares `req.t > end`, strict)."""
    data = _base_trace([_normal(5.0, 50, 10, [1])])
    path = _write_trace(tmp_path, data)
    uc = _mk_user_config()
    loader = _make_loader(path, uc, monkeypatch, end=5000.0)
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert len(convs[0].turns) == 1


def test_schedule_start_greater_than_end_filters_all(monkeypatch):
    """Inverted range (start > end) filters every request."""
    uc = _mk_user_config()
    loader = _make_loader(
        FIXTURES / "simple.json", uc, monkeypatch, start=10_000.0, end=5_000.0
    )
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert len(convs[0].turns) == 0


def test_schedule_start_zero_honors_is_none_check(monkeypatch):
    """`start=0.0` is not falsy-skipped (the `is None` guard keeps it active), so both t>=0.0 requests in simple.json survive."""
    uc = _mk_user_config()
    loader = _make_loader(FIXTURES / "simple.json", uc, monkeypatch, start=0.0)
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert len(convs[0].turns) == 2


def test_schedule_negative_start_offset_accepted(tmp_path, monkeypatch):
    """A negative `start` offset (t=0.0 > -1.0) keeps the request."""
    data = _base_trace([_normal(0.0, 50, 10, [1])])
    path = _write_trace(tmp_path, data)
    uc = _mk_user_config()
    loader = _make_loader(path, uc, monkeypatch, start=-1.0)
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert len(convs[0].turns) == 1


def test_schedule_negative_end_offset_filters_everything(monkeypatch):
    """A negative `end` offset filters all requests (all t>=0 > -1)."""
    uc = _mk_user_config()
    loader = _make_loader(FIXTURES / "simple.json", uc, monkeypatch, end=-1.0)
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert len(convs[0].turns) == 0


# Filter + subagent interaction


def test_filter_kills_following_turn_subagent_becomes_background(monkeypatch):
    """Filtering the following parent turn turns the subagent into a background branch (is_background=True, no SPAWN_JOIN prereq)."""
    uc = _mk_user_config(max_isl=250)
    loader = _make_loader(FIXTURES / "one_subagent.json", uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "trace_sa")
    assert len(parent.turns) == 1
    assert len(parent.branches) == 1
    branch = parent.branches[0]
    assert branch.is_background is True
    assert parent.turns[0].prerequisites == []
    assert parent.turns[0].branch_ids == [branch.branch_id]


def test_filter_kills_middle_parent_subagent_reanchors(tmp_path, monkeypatch):
    """Filtering a middle parent re-anchors the subagent's preceding turn to the earlier surviving parent; a following turn remains so the branch is not background."""
    data = _base_trace(
        [
            _normal(0.0, 50, 10, [1]),
            _normal(1.0, 500, 10, [1, 2]),
            _subagent(2.0, "a1", [_normal(0.0, 30, 5, [100])]),
            # Chains onto the surviving [1] prefix so detection keeps the
            # post-filter parent as one conversation.
            _normal(4.0, 50, 10, [1, 3]),
        ],
        trace_id="tmid",
    )
    path = _write_trace(tmp_path, data)
    uc = _mk_user_config(max_isl=100)
    loader = _make_loader(path, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "tmid")
    assert len(parent.turns) == 2
    assert len(parent.branches) == 1
    branch = parent.branches[0]
    assert branch.is_background is False
    # Branch anchored to re-targeted preceding (turn 0) and prereq on turn 1.
    assert parent.turns[0].branch_ids == [branch.branch_id]
    assert len(parent.turns[1].prerequisites) == 1
    assert parent.turns[1].prerequisites[0].branch_id == branch.branch_id


def test_subagent_inner_not_filtered_by_max_isl(tmp_path, monkeypatch):
    """`max_isl` applies only to top-level requests; subagent inner requests pass through regardless of their input_length."""
    # Inner request has in=500 with bs=64 -> floor(500/64)=7 hash blocks; the
    # reconstructor asserts on this corpus invariant so we tile 7 ids here.
    data = _base_trace(
        [
            _normal(0.0, 50, 10, [1]),
            _subagent(
                1.0, "a1", [_normal(0.0, 500, 10, [100, 101, 102, 103, 104, 105, 106])]
            ),
            _normal(2.0, 50, 10, [2]),
        ],
        trace_id="tinner",
    )
    path = _write_trace(tmp_path, data)
    uc = _mk_user_config(max_isl=50)
    loader = _make_loader(path, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    child = next(c for c in convs if c.session_id == "tinner::sa:a1")
    assert len(child.turns) == 1


def test_subagent_inner_max_tokens_not_capped_by_max_osl(tmp_path, monkeypatch):
    """`max_osl` only caps top-level turns; subagent inner turns keep their original output_length as max_tokens."""
    data = _base_trace(
        [
            _normal(0.0, 50, 10, [1]),
            _subagent(1.0, "a1", [_normal(0.0, 30, 50, [100])]),
            _normal(2.0, 50, 10, [2]),
        ],
        trace_id="tosl",
    )
    path = _write_trace(tmp_path, data)
    uc = _mk_user_config(max_osl=1)
    loader = _make_loader(path, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    child = next(c for c in convs if c.session_id == "tosl::sa:a1")
    assert child.turns[0].max_tokens == 50


def test_orphan_child_pruned_when_all_parents_filtered(tmp_path, monkeypatch):
    """When max_isl filters every parent turn, both the branch and the orphaned child conversation are dropped."""
    uc = _mk_user_config(max_isl=50)  # filters both in=200 and in=400
    loader = _make_loader(FIXTURES / "one_subagent.json", uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    session_ids = {c.session_id for c in convs}
    assert session_ids == {"trace_sa"}, f"unexpected conversations: {session_ids}"
    parent = convs[0]
    assert parent.turns == []
    assert parent.branches == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
