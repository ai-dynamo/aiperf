# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
from aiperf.dataset.loader.weka_trace import WekaTraceLoader

FIXTURES = Path(__file__).parents[3] / "fixtures" / "weka_traces"


def _mk_user_config(**overrides):
    from tests.unit.dataset.loader.conftest import make_weka_run

    overrides.setdefault(
        "model_names", ["claude-opus-4-5-20251101", "claude-haiku-4-5-20251001"]
    )
    return make_weka_run(**overrides)


def _stub_prompt_generator_for_reconstructor(loader) -> None:
    """Wire a MagicMock prompt_generator with the cache, corpus, tokenizer, and hash-id RNG surface the reconstructor needs."""
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


def test_can_load_single_weka_file():
    assert WekaTraceLoader.can_load(filename=FIXTURES / "simple.json") is True


def test_can_load_detects_directory():
    assert WekaTraceLoader.can_load(filename=FIXTURES) is True


def test_can_load_rejects_non_weka_json(tmp_path: Path):
    p = tmp_path / "x.json"
    p.write_text('{"not": "weka"}')
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_rejects_non_json_file(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("not json")
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_rejects_empty_directory(tmp_path: Path):
    assert WekaTraceLoader.can_load(filename=tmp_path) is False


def test_load_dataset_single_file_yields_one_trace():
    loader = WekaTraceLoader(
        filename=str(FIXTURES / "simple.json"), run=_mk_user_config()
    )
    data = loader.load_dataset()
    assert set(data.keys()) == {"trace_simple"}
    assert len(data["trace_simple"]) == 1  # one WekaTrace object


def test_load_dataset_directory_yields_one_per_file():
    loader = WekaTraceLoader(filename=str(FIXTURES), run=_mk_user_config())
    data = loader.load_dataset()
    # simple.json, one_subagent.json, terminal_subagent.json, multi_model.json
    assert "trace_simple" in data
    assert "trace_sa" in data
    assert "trace_term" in data


def test_load_dataset_rejects_extra_fields_with_filename(tmp_path):
    import shutil

    good = FIXTURES / "simple.json"
    bad = FIXTURES.parent / "weka_traces_invalid" / "bad_extra_field.json"
    d = tmp_path / "traces"
    d.mkdir()
    shutil.copy(good, d)
    shutil.copy(bad, d)
    loader = WekaTraceLoader(filename=str(d), run=_mk_user_config())
    with pytest.raises(ValueError, match="bad_extra_field.json"):
        loader.load_dataset()


def test_convert_to_conversations_builds_one_conversation_per_normal_request(
    monkeypatch,
):
    uc = _mk_user_config()
    loader = WekaTraceLoader(filename=str(FIXTURES / "simple.json"), run=uc)

    # Required attributes set by __init__ (we bypass the real PromptGenerator wiring).
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    data = loader.load_dataset()
    convs = loader.convert_to_conversations(data)
    assert len(convs) == 1
    conv = convs[0]
    assert conv.session_id == "trace_simple"
    assert len(conv.turns) == 2
    assert conv.turns[0].model == "claude-opus-4-5-20251101"
    assert conv.turns[0].max_tokens == 30
    # Trace `t` is in seconds; Turn.timestamp/delay contract is milliseconds.
    assert conv.turns[0].timestamp == 0.0
    assert conv.turns[1].timestamp == 5000.0
    # end-to-start idle gap: 5.0s start-to-start minus 1.0s prev api_time.
    assert conv.turns[1].delay == pytest.approx(4000.0)
    # weka loader populates only ``Turn.raw_messages`` (the multi-message chat
    # form consumed by ChatEndpoint.build_messages). ``Turn.texts`` is left
    # at its default empty list — a separate full-prompt decode previously
    # populated it but no consumer reads it for chat-shape traces, so the
    # decode was removed.
    assert conv.turns[0].texts == []
    # Weka now emits delta-encoded turns. Turn 0 carries the full initial
    # state (system + user). Turn 1 may either be a strict append (just
    # asst + user_k) or a full re-emit (reset_context=True) if the LCP
    # truncate disturbed an emitted segment — both forms are valid; we
    # assert on the accumulated wire shape instead.
    turn_0_roles = [m["role"] for m in conv.turns[0].raw_messages]
    assert "user" in turn_0_roles
    assert "assistant" not in turn_0_roles
    assert conv.turns[0].reset_context is False
    turn_1_roles = [m["role"] for m in conv.turns[1].raw_messages]
    assert "user" in turn_1_roles
    # If turn 1 was a strict append, system stays in turn 0 only; if it
    # was a reset, turn 1 carries the full state including system. Either
    # is correct under DELTAS_WITH_RESPONSES semantics.
    if conv.turns[1].reset_context:
        assert "system" in turn_1_roles
    else:
        assert "system" not in turn_1_roles
    # Accumulated state across both turns (mimicking what
    # BaseEndpoint.build_messages produces at request time): the first
    # non-system message is ALWAYS user. simple.json's tool+system prefix
    # covers every full block of turn 0, so the turn-1 LCP boundary strip
    # deletes turn 0's user tail — a context loss; the context-loss rule
    # resumes the conversation at a user turn (no fabricated assistant).
    accumulated: list[dict] = []
    for t in conv.turns:
        if t.reset_context:
            accumulated = list(t.raw_messages)
        else:
            accumulated.extend(t.raw_messages)
    accumulated_roles = [m["role"] for m in accumulated]
    assert "system" in accumulated_roles
    assert "user" in accumulated_roles
    non_system = [r for r in accumulated_roles if r != "system"]
    assert non_system and non_system[0] == "user", accumulated_roles


def test_convert_to_conversations_emits_alternating_roles(monkeypatch):
    """Turn 1+ keeps the assistant segment between surviving user content and new user_k content (symmetric attribution, spec 4.4.1) when a user turn survives."""
    import orjson

    uc = _mk_user_config()
    loader = WekaTraceLoader(filename=str(FIXTURES / "simple.json"), run=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    convs = loader.convert_to_conversations(loader.load_dataset())
    conv = convs[0]

    # Turn 0: just system / user (no asst).
    turn_0_roles = [m["role"] for m in conv.turns[0].raw_messages]
    assert "assistant" not in turn_0_roles

    # simple.json turn 1: context loss (the boundary strip deletes turn 0's
    # tail-only user segment) -> resume at a user turn, no assistant.
    turn_1_roles = [m["role"] for m in conv.turns[1].raw_messages]
    assert conv.turns[1].reset_context is True
    assert turn_1_roles == ["system", "user"]

    # Alternation case: turn 0's user segment owns a full block (in=448 =
    # 7 blocks > 3 prefix blocks), so it survives the turn-1 truncation and
    # the assistant segment is attributed before the new user_k content.
    trace = {
        "id": "trace_alt",
        "models": ["claude-opus-4-5-20251101"],
        "block_size": 64,
        "hash_id_scope": "local",
        "tool_tokens": 100,
        "system_tokens": 50,
        "requests": [
            {
                "t": 0.0,
                "type": "n",
                "model": "claude-opus-4-5-20251101",
                "in": 448,
                "out": 30,
                "hash_ids": [1, 2, 3, 4, 5, 6, 7],
                "api_time": 1.0,
            },
            {
                "t": 5.0,
                "type": "n",
                "model": "claude-opus-4-5-20251101",
                "in": 576,
                "out": 40,
                "hash_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "api_time": 1.2,
            },
        ],
    }
    from aiperf.dataset.loader.weka_trace_models import WekaTrace

    loader2 = WekaTraceLoader(filename=None, run=_mk_user_config())
    _stub_prompt_generator_for_reconstructor(loader2)
    convs2 = loader2.convert_to_conversations(
        {"trace_alt": [WekaTrace.model_validate(orjson.loads(orjson.dumps(trace)))]}
    )
    alt = convs2[0]
    turn_1_roles = [m["role"] for m in alt.turns[1].raw_messages]
    assert "assistant" in turn_1_roles
    asst_idx = turn_1_roles.index("assistant")
    user_indices = [i for i, r in enumerate(turn_1_roles) if r == "user"]
    assert max(user_indices) > asst_idx, (
        f"asst should precede the new user_k segment; got roles={turn_1_roles}"
    )


def test_subagent_produces_child_conversation_and_branch_plus_prereq(monkeypatch):
    from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind

    uc = _mk_user_config()
    loader = WekaTraceLoader(filename=str(FIXTURES / "one_subagent.json"), run=uc)

    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    convs = loader.convert_to_conversations(loader.load_dataset())
    # Parent + one subagent = 2 conversations.
    assert {c.session_id for c in convs} == {"trace_sa", "trace_sa::sa:agent_001"}
    parent = next(c for c in convs if c.session_id == "trace_sa")
    child = next(c for c in convs if c.session_id == "trace_sa::sa:agent_001")

    # Parent root turn declares one SPAWN branch.
    assert len(parent.branches) == 1
    branch = parent.branches[0]
    assert branch.mode == ConversationBranchMode.SPAWN
    assert branch.child_conversation_ids == ["trace_sa::sa:agent_001"]
    assert branch.start_timestamp_ms == pytest.approx(2000.0)
    assert parent.turns[0].branch_ids == [branch.branch_id]

    # Parent's next turn carries a SPAWN_JOIN prereq referencing the branch.
    assert len(parent.turns[1].prerequisites) == 1
    p = parent.turns[1].prerequisites[0]
    assert p.kind == PrerequisiteKind.SPAWN_JOIN
    assert p.branch_id == branch.branch_id

    # Child conversation has one inner turn.
    assert child.is_root is False
    assert child.agent_depth == 1
    assert child.parent_conversation_id == "trace_sa"
    assert len(child.turns) == 1
    assert child.turns[0].model == "claude-haiku-4-5-20251001"


def test_terminal_subagent_becomes_background_branch_no_prereq(monkeypatch):
    from aiperf.common.enums import ConversationBranchMode

    uc = _mk_user_config()
    loader = WekaTraceLoader(filename=str(FIXTURES / "terminal_subagent.json"), run=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "trace_term")
    assert len(parent.branches) == 1
    branch = parent.branches[0]
    assert branch.is_background is True
    assert branch.mode == ConversationBranchMode.SPAWN
    # Only one parent turn exists -> no prereq anywhere.
    assert all(not t.prerequisites for t in parent.turns)


def test_weka_zero_request_subagent_branch_targets_empty_child(tmp_path):
    model = "claude-opus-4-5-20251101"
    child_model = "claude-haiku-4-5-20251001"
    trace = {
        "id": "zero_child",
        "models": [model, child_model],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {
                "t": 0.0,
                "type": "n",
                "model": model,
                "in": 64,
                "out": 10,
                "hash_ids": [1],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "end_turn",
                "api_time": 1.0,
                "think_time": 0.0,
            },
            {
                "t": 1.0,
                "type": "subagent",
                "agent_id": "empty",
                "subagent_type": "Explore",
                "duration_ms": 100,
                "total_tokens": 0,
                "tool_use_count": 0,
                "status": "completed",
                "requests": [],
                "models": [child_model],
                "tool_tokens": 0,
                "system_tokens": 0,
            },
            {
                "t": 2.0,
                "type": "n",
                "model": model,
                "in": 128,
                "out": 10,
                "hash_ids": [1, 2],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "end_turn",
                "api_time": 1.0,
                "think_time": 0.0,
            },
        ],
    }
    path = tmp_path / "zero_child.json"
    path.write_text(json.dumps(trace))
    loader = WekaTraceLoader(filename=str(path), run=_mk_user_config())
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    conversations = loader.convert_to_conversations(loader.load_dataset())

    root = next(c for c in conversations if c.session_id == "zero_child")
    child = next(c for c in conversations if c.session_id == "zero_child::sa:empty")
    assert root.branches[0].child_conversation_ids == ["zero_child::sa:empty"]
    assert child.turns == []
    assert child.is_root is False
    assert child.agent_depth == 1


def test_weka_parallel_child_conversation_metadata_is_non_root(monkeypatch):
    import aiperf.dataset.loader.weka_parallel_convert as parallel_convert

    loader = WekaTraceLoader(
        filename=str(FIXTURES / "one_subagent.json"), run=_mk_user_config()
    )
    _stub_prompt_generator_for_reconstructor(loader)
    data = loader.load_dataset()
    plans = loader._build_reconstruction_plans(data)
    parent_plans, child_plans = plans.parent_plans, plans.child_plans

    def fake_run_parallel_weka_reconstruction(*args, **kwargs):
        return [
            {
                "trace_id": "trace_sa",
                "parent_turns": [],
                "branches": [],
                "children": [
                    {
                        "session_id": "trace_sa::sa:agent_001",
                        "turns": [],
                        "is_root": False,
                        "agent_depth": 1,
                    }
                ],
                "dropped_agent_ids": [],
                "capped_count": 0,
                "max_observed_ms": 0.0,
            }
        ]

    monkeypatch.setattr(
        parallel_convert,
        "run_parallel_weka_reconstruction",
        fake_run_parallel_weka_reconstruction,
    )

    conversations = loader._reconstruct_parallel(
        parent_plans=parent_plans,
        child_plans=child_plans,
        data=data,
        ignore_delays=False,
        think_time_only=False,
        cap_seconds=None,
        configured_workers=1,
        t_start=0.0,
        model_map_per_trace={"trace_sa": {}},
        metric_values_by_trace=loader._build_shared_metric_values(
            parent_plans, child_plans
        ),
    )

    child = next(c for c in conversations if c.session_id == "trace_sa::sa:agent_001")
    assert child.is_root is False
    assert child.agent_depth == 1


def test_duplicate_agent_id_orphan_does_not_drop_later_valid_subagent(tmp_path):
    model = "claude-opus-4-5-20251101"
    child_model = "claude-haiku-4-5-20251001"

    def normal(t, hash_ids):
        return {
            "t": t,
            "type": "n",
            "model": model,
            "in": 100,
            "out": 20,
            "hash_ids": hash_ids,
            "input_types": ["text"],
            "output_types": ["text"],
            "stop": "end_turn",
            "api_time": 1.0,
            "think_time": 0.0,
        }

    def subagent(t, hash_ids):
        return {
            "t": t,
            "type": "subagent",
            "agent_id": "dup_agent",
            "subagent_type": "Explore",
            "duration_ms": 100,
            "total_tokens": 10,
            "tool_use_count": 1,
            "status": "completed",
            "requests": [
                {
                    "t": t,
                    "type": "n",
                    "model": child_model,
                    "in": 10,
                    "out": 5,
                    "hash_ids": hash_ids,
                    "input_types": ["text"],
                    "output_types": ["text"],
                    "stop": "end_turn",
                    "api_time": 0.1,
                    "think_time": 0.0,
                }
            ],
            "models": [child_model],
            "tool_tokens": 1,
            "system_tokens": 1,
        }

    trace = {
        "id": "trace_dup",
        "models": [model, child_model],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            subagent(0.5, [10]),
            normal(1.0, [1, 2]),
            subagent(2.0, [11]),
            normal(3.0, [1, 2, 3]),
        ],
    }
    path = tmp_path / "dup.json"
    path.write_text(json.dumps(trace))

    loader = WekaTraceLoader(filename=str(path), run=_mk_user_config())
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    convs = loader.convert_to_conversations(loader.load_dataset())

    assert {c.session_id for c in convs} == {
        "trace_dup",
        "trace_dup::sa:dup_agent",
    }
    parent = next(c for c in convs if c.session_id == "trace_dup")
    assert parent.branches[0].child_conversation_ids == ["trace_dup::sa:dup_agent"]


def test_mixed_duration_subagents_emit_tiered_join_branches(tmp_path, monkeypatch):
    from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind

    model = "claude-opus-4-5-20251101"
    child_model = "claude-haiku-4-5-20251001"

    def normal(t, input_length, output_length, hash_ids):
        return {
            "t": t,
            "type": "n",
            "model": model,
            "in": input_length,
            "out": output_length,
            "hash_ids": hash_ids,
            "input_types": ["text"],
            "output_types": ["text"],
            "stop": "end_turn",
            "api_time": 1.0,
            "think_time": 0.0,
        }

    def subagent(agent_id, t, duration_ms, hash_ids):
        return {
            "t": t,
            "type": "subagent",
            "agent_id": agent_id,
            "subagent_type": "Explore",
            "duration_ms": duration_ms,
            "total_tokens": 100,
            "tool_use_count": 1,
            "status": "completed",
            "requests": [
                {
                    "t": t,
                    "type": "n",
                    "model": child_model,
                    "in": 100,
                    "out": 20,
                    "hash_ids": hash_ids,
                    "input_types": ["text"],
                    "output_types": ["text"],
                    "stop": "end_turn",
                    "api_time": 0.5,
                    "think_time": 0.0,
                }
            ],
            "models": [child_model],
            "tool_tokens": 20,
            "system_tokens": 10,
        }

    trace = {
        "id": "trace_tiered",
        "models": [model, child_model],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            normal(0.0, 200, 30, [1, 2, 3]),
            subagent("agent_a", 1.0, 5000, [10, 11]),  # ends at t=6, joins turn 1
            subagent("agent_b", 1.5, 11000, [12, 13]),  # ends at t=12.5, joins turn 2
            subagent("agent_c", 1.6, 24000, [14, 15]),  # ends after all main turns
            normal(6.0, 250, 40, [1, 2, 3, 4]),
            normal(20.0, 300, 50, [1, 2, 3, 4, 5]),
        ],
    }
    path = tmp_path / "tiered.json"
    path.write_text(json.dumps(trace))

    uc = _mk_user_config()
    loader = WekaTraceLoader(filename=str(path), run=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "trace_tiered")

    assert len(parent.branches) == 3
    branches_by_child = {
        tuple(branch.child_conversation_ids): branch for branch in parent.branches
    }
    branch_a = branches_by_child[("trace_tiered::sa:agent_a",)]
    branch_b = branches_by_child[("trace_tiered::sa:agent_b",)]
    branch_c = branches_by_child[("trace_tiered::sa:agent_c",)]

    assert all(
        branch.mode == ConversationBranchMode.SPAWN for branch in parent.branches
    )
    assert parent.turns[0].branch_ids == [
        branch_a.branch_id,
        branch_b.branch_id,
        branch_c.branch_id,
    ]

    assert branch_a.is_background is False
    assert len(parent.turns[1].prerequisites) == 1
    prereq_a = parent.turns[1].prerequisites[0]
    assert prereq_a.kind == PrerequisiteKind.SPAWN_JOIN
    assert prereq_a.branch_id == branch_a.branch_id

    assert branch_b.is_background is False
    assert len(parent.turns[2].prerequisites) == 1
    prereq_b = parent.turns[2].prerequisites[0]
    assert prereq_b.kind == PrerequisiteKind.SPAWN_JOIN
    assert prereq_b.branch_id == branch_b.branch_id

    assert branch_c.is_background is True
    assert branch_c.branch_id not in {
        prereq.branch_id for turn in parent.turns for prereq in turn.prerequisites
    }


def test_filters_requests_exceeding_max_isl(monkeypatch):
    uc = _mk_user_config(max_isl=210)  # simple.json has in=200 and in=250
    loader = WekaTraceLoader(filename=str(FIXTURES / "simple.json"), run=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    convs = loader.convert_to_conversations(loader.load_dataset())
    conv = convs[0]
    assert len(conv.turns) == 1
    assert conv.turns[0].timestamp == 0.0


def test_caps_max_osl(monkeypatch):
    uc = _mk_user_config(max_osl=25)
    loader = WekaTraceLoader(filename=str(FIXTURES / "simple.json"), run=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    convs = loader.convert_to_conversations(loader.load_dataset())
    for t in convs[0].turns:
        assert t.max_tokens <= 25


def test_trace_model_rewritten_to_configured_model_zero(monkeypatch):
    """Trace's per-request model is unconditionally rewritten to model_names[0]."""
    uc = _mk_user_config(model_names=["override-model"])
    loader = WekaTraceLoader(filename=str(FIXTURES / "simple.json"), run=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    convs = loader.convert_to_conversations(loader.load_dataset())
    for c in convs:
        for t in c.turns:
            assert t.model == "override-model"


def test_orphaned_subagent_is_dropped_when_preceding_turn_filtered(monkeypatch):
    # Raise the bar so BOTH parent turns in one_subagent.json get filtered (in=200, in=400).
    uc = _mk_user_config(max_isl=50)
    loader = WekaTraceLoader(filename=str(FIXTURES / "one_subagent.json"), run=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "trace_sa")
    # No parent turns remain -> subagent branch also dropped.
    assert parent.branches == []


def _real_pg():
    """Build a PromptGenerator-shape mock with a real HashIdRandomGenerator and only the cache/rng/corpus surface ``_decode_block_tokens`` touches."""
    from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
    from aiperf.common.random_generator import RandomGenerator

    pg = MagicMock()
    base_rng = RandomGenerator(0, _internal=True)
    pg._hash_id_corpus_rng = HashIdRandomGenerator.from_base_rng(base_rng)
    pg._cache = {}
    pg._tokenized_corpus = list(range(10000, 11000))
    pg._corpus_size = 1000
    return pg


def _real_loader_with_pg(pg):
    uc = _mk_user_config()
    loader = WekaTraceLoader(filename=str(FIXTURES / "two_turns.json"), run=uc)
    loader.prompt_generator = pg
    loader._block_size = 64
    return loader


def test_decode_block_tokens_distinct_across_scopes():
    """Same hash_id under different trace scopes produces different tokens, so local-scope traces see real cache misses rather than cross-trace hits."""
    pg = _real_pg()
    loader = _real_loader_with_pg(pg)

    pg._cache.clear()
    pg._hash_id_corpus_rng.set_trace_id("trace_alpha")
    a = loader._decode_block_tokens([1])

    pg._cache.clear()
    pg._hash_id_corpus_rng.set_trace_id("trace_beta")
    b = loader._decode_block_tokens([1])

    assert a != b
    assert len(a) == 64 and len(b) == 64


def test_decode_block_tokens_deterministic_within_scope():
    """Same (scope, hash_id) called twice after cache clear and reseed is byte-identical, as cross-process reproducibility requires."""
    pg = _real_pg()
    loader = _real_loader_with_pg(pg)

    pg._cache.clear()
    pg._hash_id_corpus_rng.set_trace_id("trace_alpha")
    a1 = loader._decode_block_tokens([7])

    pg._cache.clear()
    pg._hash_id_corpus_rng.set_trace_id("trace_alpha")
    a2 = loader._decode_block_tokens([7])

    assert a1 == a2


def test_decode_block_tokens_deterministic_across_loaders():
    """Two freshly built loaders with the same seed produce identical bytes for the same (scope, hash_id)."""
    pg1 = _real_pg()
    loader1 = _real_loader_with_pg(pg1)
    pg1._hash_id_corpus_rng.set_trace_id("trace_x")
    a = loader1._decode_block_tokens([3, 5, 11])

    pg2 = _real_pg()
    loader2 = _real_loader_with_pg(pg2)
    pg2._hash_id_corpus_rng.set_trace_id("trace_x")
    b = loader2._decode_block_tokens([3, 5, 11])

    assert a == b


def test_ignore_trace_delays_nulls_timestamp_and_delay(monkeypatch):
    """With ``ignore_trace_delays=True``, parent and child turns have ``timestamp`` and ``delay`` None so timing modes dispatch back-to-back."""
    uc = _mk_user_config(ignore_trace_delays=True)
    loader = WekaTraceLoader(filename=str(FIXTURES / "one_subagent.json"), run=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    convs = loader.convert_to_conversations(loader.load_dataset())
    assert len(convs) >= 2  # parent + at least one subagent child
    for conv in convs:
        for turn in conv.turns:
            assert turn.timestamp is None
            assert turn.delay is None


def test_use_think_time_only_emits_recorded_think_time_as_delay(monkeypatch, tmp_path):
    """With ``use_think_time_only=True``, ``Turn.delay`` equals recorded ``think_time * 1000`` (falling back to the full delta when think_time is None)."""
    import orjson

    trace = {
        "id": "trace_tt",
        "models": ["claude-opus-4-5-20251101"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {
                "t": 0.0,
                "type": "n",
                "model": "claude-opus-4-5-20251101",
                "in": 100,
                "out": 10,
                "hash_ids": [1, 2],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "end_turn",
                "api_time": 5.5,
                "think_time": 0.0,
            },
            {
                "t": 12.0,
                "type": "n",
                "model": "claude-opus-4-5-20251101",
                "in": 200,
                "out": 20,
                # Chains onto [1, 2] so detection keeps one conversation.
                "hash_ids": [1, 2, 3],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "end_turn",
                "api_time": 4.0,
                "think_time": 7.0,
            },
            {
                "t": 25.0,
                "type": "n",
                "model": "claude-opus-4-5-20251101",
                "in": 300,
                "out": 30,
                # Chains onto [1, 2, 3] so detection keeps one conversation.
                "hash_ids": [1, 2, 3, 4],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "end_turn",
                "api_time": 3.0,
                "think_time": None,  # forces fallback to full delta
            },
        ],
    }
    f = tmp_path / "trace_tt.json"
    f.write_bytes(orjson.dumps(trace))

    uc = _mk_user_config(use_think_time_only=True)
    loader = WekaTraceLoader(filename=str(f), run=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    convs = loader.convert_to_conversations(loader.load_dataset())
    turns = convs[0].turns
    assert len(turns) == 3
    assert turns[0].delay is None  # first turn always
    assert (
        turns[1].delay == 7000.0
    )  # think_time=7.0s -> 7000ms (NOT 12000ms full delta)
    # think_time=None -> falls back to end-to-start: (25-12)s start-to-start
    # minus 4.0s prev api_time = 9000ms.
    assert turns[2].delay == 9000.0


def _wire_real_scope_rng(loader, *, block_size: int, seed: int = 1234) -> None:
    """Wire a MagicMock prompt_generator backed by the real scope-sensitive RNG, with a token-reflecting ``tokenizer.decode`` so raw_messages track decoded tokens."""
    pg = MagicMock()
    pg._cache = {}
    pg._tokenized_corpus = list(range(4096))
    pg._corpus_size = 4096
    pg._hash_id_corpus_rng = HashIdRandomGenerator(seed, _internal=True)
    pg.tokenizer.decode.side_effect = lambda toks: "|".join(str(t) for t in toks)
    loader.prompt_generator = pg
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = block_size


def _write_trace(tmp_path: Path, trace: dict) -> str:
    p = tmp_path / f"{trace['id']}.json"
    p.write_text(json.dumps(trace))
    return str(p)


def _normal_req(
    *, t: float, in_tokens: int, hash_ids: list[int], stop: str = "end_turn"
):
    return {
        "t": t,
        "type": "n",
        "model": "m",
        "in": in_tokens,
        "out": 10,
        "hash_ids": hash_ids,
        "input_types": ["text"],
        "output_types": ["text"],
        "stop": stop,
        "api_time": 1.0,
        "think_time": 0.0,
    }


def _subagent(*, agent_id: str, t: float, in_tokens: int, hash_ids: list[int]):
    return {
        "t": t,
        "type": "subagent",
        "agent_id": agent_id,
        "subagent_type": "Explore",
        "duration_ms": 1000,
        "total_tokens": 100,
        "tool_use_count": 1,
        "status": "completed",
        "requests": [_normal_req(t=0.0, in_tokens=in_tokens, hash_ids=hash_ids)],
        "models": ["m"],
        "tool_tokens": 0,
        "system_tokens": 0,
    }


def test_convert_to_conversations_subagent_inherits_parent_hash_id_scope(tmp_path):
    """A hash_id shared by a parent request and a subagent inner request decodes identically, since the subagent shares the parent trace's scope."""
    bs = 16
    shared = [100, 101, 102]
    trace = {
        "id": "trace_scope",
        "models": ["m"],
        "block_size": bs,
        "hash_id_scope": "local",
        "requests": [
            _normal_req(t=0.0, in_tokens=bs * len(shared), hash_ids=shared),
            _subagent(
                agent_id="agent_001",
                t=2.0,
                in_tokens=bs * len(shared),
                hash_ids=shared,
            ),
        ],
    }
    loader = WekaTraceLoader(
        filename=_write_trace(tmp_path, trace), run=_mk_user_config()
    )
    _wire_real_scope_rng(loader, block_size=bs)

    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "trace_scope")
    child = next(c for c in convs if c.parent_conversation_id == "trace_scope")

    # in == n*bs and tool/system == 0, so turn-0 content is PURELY the decoded
    # shared blocks (no partial tail, no system segment). Equal iff same scope.
    assert child.turns[0].raw_messages == parent.turns[0].raw_messages


def test_convert_to_conversations_sibling_subagents_share_hash_id_scope(tmp_path):
    """Two sibling subagents referencing the same hash_id blocks decode them identically, since both share the parent trace's scope."""
    bs = 16
    shared = [200, 201]
    trace = {
        "id": "trace_sib",
        "models": ["m"],
        "block_size": bs,
        "hash_id_scope": "local",
        "requests": [
            _normal_req(t=0.0, in_tokens=bs * 3, hash_ids=[1, 2, 3], stop="tool_use"),
            _subagent(
                agent_id="agent_001", t=2.0, in_tokens=bs * len(shared), hash_ids=shared
            ),
            _subagent(
                agent_id="agent_002", t=3.0, in_tokens=bs * len(shared), hash_ids=shared
            ),
        ],
    }
    loader = WekaTraceLoader(
        filename=_write_trace(tmp_path, trace), run=_mk_user_config()
    )
    _wire_real_scope_rng(loader, block_size=bs)

    convs = loader.convert_to_conversations(loader.load_dataset())
    children = [c for c in convs if c.parent_conversation_id == "trace_sib"]
    assert len(children) == 2
    sib_a, sib_b = children
    assert sib_a.turns[0].raw_messages == sib_b.turns[0].raw_messages


def test_subagent_child_shares_trace_decode_scope():
    """Same hash_id decodes to identical tokens in parent and child (local scope = one namespace per trace file)."""
    uc = _mk_user_config()
    loader = WekaTraceLoader(filename=str(FIXTURES / "one_subagent.json"), run=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    scopes_used: list[str] = []
    orig = loader.prompt_generator._hash_id_corpus_rng.set_trace_id

    def _spy(scope: str):
        scopes_used.append(scope)
        return orig(scope)

    loader.prompt_generator._hash_id_corpus_rng.set_trace_id = _spy
    loader.convert_to_conversations(loader.load_dataset())
    assert scopes_used and all(s == "trace_sa" for s in scopes_used), scopes_used


def test_theoretical_metric_values_unchanged_for_disjoint_namespaces():
    """With no parent/child hash overlap, the shared seen-set reproduces the legacy per-conversation values exactly."""
    uc = _mk_user_config()
    loader = WekaTraceLoader(filename=str(FIXTURES / "one_subagent.json"), run=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    convs = {
        c.session_id: c for c in loader.convert_to_conversations(loader.load_dataset())
    }
    root = convs["trace_sa"]
    child = convs["trace_sa::sa:agent_001"]
    assert root.turns[0].theoretical_prefix_cache_hit_blocks == 0
    assert root.turns[0].theoretical_prefix_cache_total_blocks == 3
    assert root.turns[1].theoretical_prefix_cache_hit_blocks == 3
    assert root.turns[1].theoretical_prefix_cache_total_blocks == 5
    assert child.turns[0].theoretical_prefix_cache_hit_blocks == 0
    assert child.turns[0].theoretical_prefix_cache_total_blocks == 2


def test_theoretical_metric_shares_seen_set_with_subagent_children():
    """A hash block first sent by the parent counts as a hit when the subagent child later sends it (shared per-trace seen-set)."""
    from aiperf.dataset.loader.weka_trace_models import WekaTrace

    trace = WekaTrace.model_validate(
        {
            "id": "trace_shared",
            "models": ["m"],
            "block_size": 64,
            "hash_id_scope": "local",
            "requests": [
                {
                    "t": 0.0,
                    "type": "n",
                    "model": "m",
                    "in": 192,
                    "out": 30,
                    "hash_ids": [1, 2, 3],
                    "api_time": 1.0,
                },
                {
                    "t": 2.0,
                    "type": "subagent",
                    "agent_id": "agent_001",
                    "subagent_type": "Explore",
                    "duration_ms": 3000,
                    "total_tokens": 500,
                    "tool_use_count": 1,
                    "status": "completed",
                    "requests": [
                        {
                            "t": 2.5,
                            "type": "n",
                            "model": "m",
                            "in": 192,
                            "out": 50,
                            # Child re-sends the parent's [1, 2] prefix.
                            "hash_ids": [1, 2, 99],
                            "api_time": 0.5,
                        }
                    ],
                    "models": ["m"],
                    "tool_tokens": 64,
                    "system_tokens": 0,
                },
                {
                    "t": 6.0,
                    "type": "n",
                    "model": "m",
                    "in": 320,
                    "out": 40,
                    "hash_ids": [1, 2, 3, 4, 99],
                    "api_time": 1.5,
                },
            ],
        }
    )
    uc = _mk_user_config()
    loader = WekaTraceLoader(filename=None, run=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    convs = {
        c.session_id: c
        for c in loader.convert_to_conversations({"trace_shared": [trace]})
    }
    child = convs["trace_shared::sa:agent_001"]
    # Child turn 0 at t=2.5 re-sends [1, 2] already seen from the parent.
    assert child.turns[0].theoretical_prefix_cache_hit_blocks == 2
    assert child.turns[0].theoretical_prefix_cache_total_blocks == 3
    root = convs["trace_shared"]
    # Root turn 1: [1,2,3,4,99] -> 1,2,3 seen from itself; 4 novel stops the
    # leading run even though 99 was seen from the child.
    assert root.turns[1].theoretical_prefix_cache_hit_blocks == 3


FANOUT_FIXTURES = Path(__file__).parents[3] / "fixtures" / "weka_traces_fanout"


def _fanout_loader():
    uc = _mk_user_config()
    loader = WekaTraceLoader(filename=str(FANOUT_FIXTURES / "fanout.json"), run=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    return loader


def test_flattened_fanout_splits_into_three_conversations():
    loader = _fanout_loader()
    convs = {
        c.session_id: c for c in loader.convert_to_conversations(loader.load_dataset())
    }
    assert set(convs) == {
        "trace_fanout",
        "trace_fanout::fa:000",
        "trace_fanout::fa:001",
    }
    root = convs["trace_fanout"]
    assert len(root.turns) == 3
    w0 = convs["trace_fanout::fa:000"]
    assert len(w0.turns) == 2
    assert w0.agent_depth == 1
    assert w0.is_root is False
    assert w0.parent_conversation_id == "trace_fanout"
    assert len(convs["trace_fanout::fa:001"].turns) == 1


def test_flattened_fanout_branch_anchoring_and_joins():
    loader = _fanout_loader()
    convs = {
        c.session_id: c for c in loader.convert_to_conversations(loader.load_dataset())
    }
    root = convs["trace_fanout"]
    # Both workers spawn off turn 0 (last main turn before their first req).
    spawn_branch_ids = root.turns[0].branch_ids
    assert len(spawn_branch_ids) == 2
    branches = {b.branch_id: b for b in root.branches}
    # Worker 1 (ends t=6.5) gates main turn 1 (t=9); worker 0 (ends t=9.5)
    # gates main turn 2 (t=12) -> different join turns -> separate branches.
    join_prereqs_t1 = [p.branch_id for p in root.turns[1].prerequisites]
    join_prereqs_t2 = [p.branch_id for p in root.turns[2].prerequisites]
    assert len(join_prereqs_t1) == 1 and len(join_prereqs_t2) == 1
    assert branches[join_prereqs_t1[0]].child_conversation_ids == [
        "trace_fanout::fa:001"
    ]
    assert branches[join_prereqs_t2[0]].child_conversation_ids == [
        "trace_fanout::fa:000"
    ]


def test_flattened_fanout_per_chain_delays():
    loader = _fanout_loader()
    convs = {
        c.session_id: c for c in loader.convert_to_conversations(loader.load_dataset())
    }
    w0 = convs["trace_fanout::fa:000"]
    assert w0.turns[0].delay is None
    # end-to-start within the worker chain: (8.5-2.0)s minus 6.0s prev api_time.
    assert w0.turns[1].delay == pytest.approx(500.0)
    root = convs["trace_fanout"]
    # Main delays computed within the main chain only, end-to-start:
    # (9.0-0.0)s minus 1.0s api, (12.0-9.0)s minus 1.0s api.
    assert root.turns[1].delay == pytest.approx(8000.0)
    assert root.turns[2].delay == pytest.approx(2000.0)


def test_flattened_fanout_shares_decode_scope_with_root():
    """Worker chains decode under the trace scope, never their own."""
    loader = _fanout_loader()
    scopes_used: list[str] = []
    orig = loader.prompt_generator._hash_id_corpus_rng.set_trace_id

    def _spy(scope: str):
        scopes_used.append(scope)
        return orig(scope)

    loader.prompt_generator._hash_id_corpus_rng.set_trace_id = _spy
    loader.convert_to_conversations(loader.load_dataset())
    assert scopes_used and all(s == "trace_fanout" for s in scopes_used)


def test_split_disabled_restores_legacy_single_stream(monkeypatch):
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.DATASET, "WEKA_SPLIT_FLATTENED_AGENTS", False)
    loader = _fanout_loader()
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert [c.session_id for c in convs] == ["trace_fanout"]
    assert len(convs[0].turns) == 6


def test_flattened_fanout_logs_detection_summary(caplog):
    import logging

    loader = _fanout_loader()
    # The per-trace "detected N agents" summary is emitted at DEBUG; the
    # split-count summary stays at INFO. Capture at DEBUG to see both.
    with caplog.at_level(logging.DEBUG, logger="aiperf.dataset.loader.weka_trace"):
        loader.convert_to_conversations(loader.load_dataset())
    text = caplog.text
    assert "detected 3 agents" in text
    assert "split 1 trace(s) into 2 extra agent chain(s)" in text


def test_flattened_fanout_zero_declared_emits_no_fabricated_system_role():
    """The system role comes only from declared header counts, so a fixture with tool_tokens=0/system_tokens=0 fabricates no system role."""
    loader = _fanout_loader()
    convs = {
        c.session_id: c for c in loader.convert_to_conversations(loader.load_dataset())
    }
    for sid, conv in convs.items():
        for k, turn in enumerate(conv.turns):
            roles = [m["role"] for m in turn.raw_messages]
            assert "system" not in roles, (sid, k, roles)
    # Turn 0 of every conversation is one user message with the full input.
    assert [m["role"] for m in convs["trace_fanout"].turns[0].raw_messages] == ["user"]
    assert convs["trace_fanout"].turns[0].raw_messages[0]["content"] == "<dec:192>"
    for wid, total in (("trace_fanout::fa:000", 256), ("trace_fanout::fa:001", 256)):
        msgs = convs[wid].turns[0].raw_messages
        assert [m["role"] for m in msgs] == ["user"], wid
        assert msgs[0]["content"] == f"<dec:{total}>", wid
