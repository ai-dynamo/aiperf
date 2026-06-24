# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial orchestrator-v1 integration tests for WekaTraceLoader."""

from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.common.validators.orchestrator_v1 import validate_for_orchestrator_v1
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.plugin.enums import DatasetSamplingStrategy

FIXTURES = Path(__file__).parents[2] / "fixtures" / "weka_traces"

pytestmark = pytest.mark.component_integration


@pytest.fixture(autouse=True)
def _legacy_single_stream(monkeypatch):
    """Pin the legacy collapsed single-stream loader shape.

    These tests stress orchestrator-v1 validation of collapsed subagent /
    mixed top-level shapes. Their ``_normal``/``_streaming`` builders mint a
    unique single-block hash per request, which flattened-agent detection
    would (correctly) classify as one disjoint chain per request — a
    different shape than the one under test.
    """
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.DATASET, "WEKA_SPLIT_FLATTENED_AGENTS", False)


def _mk_user_config(
    *,
    max_isl=None,
    max_osl=None,
    start=None,
    end=None,
    model_names=("claude-opus-4-5-20251101", "claude-haiku-4-5-20251001", "m"),
):
    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = start
    uc.input.fixed_schedule_end_offset = end
    uc.input.ignore_trace_delays = False
    uc.input.use_think_time_only = False
    uc.input.use_end_to_start_delays = False
    uc.input.synthesis.max_isl = max_isl
    uc.input.synthesis.max_osl = max_osl
    uc.input.max_context_length = None
    uc.input.synthesis.should_synthesize.return_value = False
    uc.input.prompt.input_tokens.block_size = None
    uc.tokenizer.trust_remote_code = False
    uc.tokenizer.revision = None
    uc.tokenizer.name = "t"
    uc.endpoint.model_names = list(model_names)
    uc.loadgen.inter_turn_delay_cap_seconds = None
    return uc


def _make_loader(filename, uc, monkeypatch):
    loader = WekaTraceLoader(filename=str(filename), user_config=uc)
    monkeypatch.setattr(
        loader,
        "synthesize_prompts_from_hash_ids",
        lambda rs: {r.key: f"p-{r.key}" for r in rs},
    )
    pg = MagicMock()
    # sample_partial_tail_tokens reads _corpus_size as an int and slices
    # _tokenized_corpus; give the mock real values so the partial-tail path
    # doesn't trip MagicMock arithmetic.
    pg._corpus_size = 10000
    pg._tokenized_corpus = list(range(10000))
    # _decode_tokens_to_text routes through prompt_generator.tokenizer.decode;
    # return a real str so Pydantic Text validation accepts the prompt.
    pg.tokenizer.decode = lambda tokens: f"decoded-{len(tokens)}"
    loader.prompt_generator = pg
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    return loader


def _write_trace(tmp_path, data, name="t.json"):
    p = tmp_path / name
    p.write_bytes(orjson.dumps(data))
    return p


def _normal(t=0.0, model="m", in_=10, out=1):
    return {
        "t": t,
        "type": "n",
        "model": model,
        "in": in_,
        "out": out,
        "hash_ids": [int(t * 1000) + in_],
        "input_types": ["text"],
        "output_types": ["text"],
        "stop": "end_turn",
    }


def _streaming(t=0.0, model="m", in_=10, out=1):
    return {
        "t": t,
        "type": "s",
        "model": model,
        "in": in_,
        "out": out,
        "hash_ids": [int(t * 1000) + in_ + 7],
        "input_types": ["text"],
        "output_types": ["text"],
        "stop": "end_turn",
    }


def _subagent(
    agent_id,
    *,
    t=1.0,
    inner_model="m",
    inner=(("n", 0.0, 10, 1),),
    models=("m",),
):
    inner_reqs = []
    for _ty, it, ins, outs in inner:
        inner_reqs.append(
            {
                "t": it,
                "type": "n",
                "model": inner_model,
                "in": ins,
                "out": outs,
                "hash_ids": [int(it * 1000) + ins + 99],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "end_turn",
            }
        )
    return {
        "t": t,
        "type": "subagent",
        "agent_id": agent_id,
        "subagent_type": "Explore",
        "duration_ms": 1,
        "total_tokens": 0,
        "tool_use_count": 0,
        "status": "completed",
        "requests": inner_reqs,
        "models": list(models),
    }


def _build_trace(trace_id, requests, models=("m",)):
    return {
        "id": trace_id,
        "models": list(models),
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": requests,
    }


def _to_metadata(convs):
    return DatasetMetadata(
        conversations=[c.to_metadata() for c in convs],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def test_multi_subagent_collapsed_branch_passes_v1(tmp_path, monkeypatch):
    """Three adjacent subagents between two parents collapse to one branch and pass v1."""
    trace = _build_trace(
        "trace_multi",
        [
            _normal(t=0.0, in_=50),
            _subagent("a1", t=1.0),
            _subagent("a2", t=1.1),
            _subagent("a3", t=1.2),
            _normal(t=2.0, in_=60),
        ],
    )
    path = _write_trace(tmp_path, trace, name="multi.json")
    uc = _mk_user_config()
    loader = _make_loader(path, uc, monkeypatch)

    convs = loader.convert_to_conversations(loader.load_dataset())
    md = _to_metadata(convs)
    validate_for_orchestrator_v1(md)

    parent = next(c for c in md.conversations if c.conversation_id == "trace_multi")
    assert len(parent.branches) == 1
    assert sorted(parent.branches[0].child_conversation_ids) == [
        "trace_multi::sa:a1",
        "trace_multi::sa:a2",
        "trace_multi::sa:a3",
    ]
    assert len(parent.turns[1].prerequisites) == 1


def test_terminal_background_branch_passes_v1(monkeypatch):
    """Terminal subagent becomes a background branch (is_background=True) and passes v1."""
    uc = _mk_user_config()
    loader = _make_loader(FIXTURES / "terminal_subagent.json", uc, monkeypatch)

    convs = loader.convert_to_conversations(loader.load_dataset())
    md = _to_metadata(convs)
    validate_for_orchestrator_v1(md)

    parent = next(c for c in md.conversations if c.conversation_id == "trace_term")
    assert len(parent.branches) == 1
    assert parent.branches[0].is_background is True
    # Background branches must not be referenced by any prereq.
    for turn in parent.turns:
        for prereq in turn.prerequisites:
            assert prereq.branch_id != parent.branches[0].branch_id


def test_mixed_streaming_and_normal_top_level_passes_v1(tmp_path, monkeypatch):
    """Alternating normal+streaming top-level requests round-trip and pass v1."""
    trace = _build_trace(
        "trace_mixed",
        [
            _normal(t=0.0, in_=10, out=2),
            _streaming(t=1.0, in_=20, out=3),
            _normal(t=2.0, in_=30, out=4),
            _streaming(t=3.0, in_=40, out=5),
        ],
    )
    path = _write_trace(tmp_path, trace, name="mixed.json")
    uc = _mk_user_config()
    loader = _make_loader(path, uc, monkeypatch)

    convs = loader.convert_to_conversations(loader.load_dataset())
    md = _to_metadata(convs)
    validate_for_orchestrator_v1(md)

    parent = next(c for c in md.conversations if c.conversation_id == "trace_mixed")
    assert len(parent.turns) == 4
    assert len(parent.branches) == 0


def test_orphan_child_pruning_prevents_v1_failure(monkeypatch):
    """max_isl filters both parents; post-fix the orphan child is pruned so only the
    (0-turn) parent conversation remains and v1 validates cleanly."""
    uc = _mk_user_config(max_isl=50)
    loader = _make_loader(FIXTURES / "one_subagent.json", uc, monkeypatch)

    convs = loader.convert_to_conversations(loader.load_dataset())
    md = _to_metadata(convs)
    validate_for_orchestrator_v1(md)

    assert len(md.conversations) == 1
    parent = md.conversations[0]
    assert parent.conversation_id == "trace_sa"
    assert parent.turns == []
    assert parent.branches == []


def test_subagent_at_index_zero_dropped_path_passes_v1(tmp_path, monkeypatch):
    """Subagent at outer index 0 (no preceding normal parent turn) is dropped;
    child is pruned; remaining normal becomes the sole parent turn; v1 passes."""
    trace = _build_trace(
        "trace_sa0",
        [
            _subagent("a1", t=0.0),
            _normal(t=1.0, in_=10),
        ],
    )
    path = _write_trace(tmp_path, trace, name="sa0.json")
    uc = _mk_user_config()
    loader = _make_loader(path, uc, monkeypatch)

    convs = loader.convert_to_conversations(loader.load_dataset())
    md = _to_metadata(convs)
    validate_for_orchestrator_v1(md)

    # Only the parent survives; its child was pruned because its branch was dropped.
    assert len(md.conversations) == 1
    parent = md.conversations[0]
    assert parent.conversation_id == "trace_sa0"
    assert len(parent.turns) == 1
    assert parent.branches == []


def test_fully_filtered_trace_passes_v1(tmp_path, monkeypatch):
    """All normal requests filtered by max_isl — parent has 0 turns, no branches,
    no children. v1 passes trivially (no prereqs to check)."""
    trace = _build_trace(
        "trace_empty",
        [_normal(t=0.0, in_=100, out=1)],
    )
    path = _write_trace(tmp_path, trace, name="empty.json")
    uc = _mk_user_config(max_isl=50)
    loader = _make_loader(path, uc, monkeypatch)

    convs = loader.convert_to_conversations(loader.load_dataset())
    md = _to_metadata(convs)
    validate_for_orchestrator_v1(md)

    assert len(md.conversations) == 1
    parent = md.conversations[0]
    assert parent.conversation_id == "trace_empty"
    assert parent.turns == []
    assert parent.branches == []


def test_hundred_subagents_collapsed_passes_v1(tmp_path, monkeypatch):
    """Parent + 100 adjacent subagents + parent → one collapsed branch with 100
    children — passes v1."""
    requests = [_normal(t=0.0, in_=10)]
    requests.extend(_subagent(f"a{i:03d}", t=1.0 + 0.001 * i) for i in range(100))
    requests.append(_normal(t=200.0, in_=20))
    trace = _build_trace("trace_many", requests)
    path = _write_trace(tmp_path, trace, name="many.json")
    uc = _mk_user_config()
    loader = _make_loader(path, uc, monkeypatch)

    convs = loader.convert_to_conversations(loader.load_dataset())
    md = _to_metadata(convs)
    validate_for_orchestrator_v1(md)

    parent = next(c for c in md.conversations if c.conversation_id == "trace_many")
    assert len(parent.branches) == 1
    assert len(parent.branches[0].child_conversation_ids) == 100
    assert len(parent.turns[1].prerequisites) == 1
    # And all 100 child conversations are present.
    assert (
        sum(
            1
            for c in md.conversations
            if c.conversation_id.startswith("trace_many::sa:")
        )
        == 100
    )


def test_manually_malformed_prereq_branch_id_rejected_by_v1():
    """A hand-built metadata with a SPAWN_JOIN prereq pointing at a nonexistent
    branch is rejected by v1 with the 'does not reference a prior branch' message."""
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="c",
                turns=[
                    TurnMetadata(
                        timestamp_ms=0.0,
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN,
                                branch_id="nonexistent",
                            )
                        ],
                    )
                ],
                branches=[],
            )
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    with pytest.raises(NotImplementedError, match="does not reference a prior branch"):
        validate_for_orchestrator_v1(md)


def test_manually_malformed_branch_child_reference_rejected_by_v1():
    """v1 now verifies that ConversationBranchInfo.child_conversation_ids resolve to
    existing ConversationMetadata.conversation_id entries in the same DatasetMetadata.
    A dangling child reference is rejected with NotImplementedError."""
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="c",
                turns=[
                    TurnMetadata(
                        timestamp_ms=0.0,
                        branch_ids=["b1"],
                    ),
                    TurnMetadata(
                        timestamp_ms=1.0,
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN,
                                branch_id="b1",
                            )
                        ],
                    ),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="b1",
                        child_conversation_ids=["does_not_exist"],
                        mode=ConversationBranchMode.SPAWN,
                        is_background=False,
                    )
                ],
            )
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    with pytest.raises(
        NotImplementedError, match="does not reference an existing conversation"
    ):
        validate_for_orchestrator_v1(md)


def test_dataset_metadata_json_roundtrip_preserves_prereqs_and_branches(monkeypatch):
    """DatasetMetadata survives JSON round-trip; re-parsed metadata still validates
    and retains conversation count, branch count, and prereq branch_ids."""
    uc = _mk_user_config()
    loader = _make_loader(FIXTURES / "one_subagent.json", uc, monkeypatch)

    convs = loader.convert_to_conversations(loader.load_dataset())
    md = _to_metadata(convs)
    blob = md.model_dump_json()
    restored = DatasetMetadata.model_validate_json(blob)
    validate_for_orchestrator_v1(restored)

    assert len(restored.conversations) == len(md.conversations)

    orig_parent = next(c for c in md.conversations if c.conversation_id == "trace_sa")
    new_parent = next(
        c for c in restored.conversations if c.conversation_id == "trace_sa"
    )
    assert len(new_parent.branches) == len(orig_parent.branches)
    assert [b.branch_id for b in new_parent.branches] == [
        b.branch_id for b in orig_parent.branches
    ]
    assert [b.child_conversation_ids for b in new_parent.branches] == [
        b.child_conversation_ids for b in orig_parent.branches
    ]
    orig_prereq_ids = [p.branch_id for t in orig_parent.turns for p in t.prerequisites]
    new_prereq_ids = [p.branch_id for t in new_parent.turns for p in t.prerequisites]
    assert orig_prereq_ids == new_prereq_ids
    assert orig_prereq_ids  # sanity: at least one prereq exists
