# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from aiperf.common.models import DatasetMetadata
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.plugin.enums import DatasetSamplingStrategy
from tests.unit.dataset.loader.test_weka_trace import (
    _mk_user_config,
    _stub_prompt_generator_for_reconstructor,
)

FIXTURE = (
    Path(__file__).parents[3]
    / "fixtures"
    / "weka_traces_overlap_groups"
    / "abc_join_d.json"
)


def _load():
    loader = WekaTraceLoader(filename=str(FIXTURE), user_config=_mk_user_config())
    _stub_prompt_generator_for_reconstructor(loader)
    return loader.convert_to_conversations(loader.load_dataset())


def test_loader_encodes_abc_parallel_frontier_before_d() -> None:
    conversations = _load()
    root = next(c for c in conversations if c.is_root)
    children = [c for c in conversations if not c.is_root]

    assert len(root.turns) == 2
    assert len(children) == 2
    assert all(len(child.turns) == 1 for child in children)
    assert {conversation.replay_scope_id for conversation in conversations} == {
        "abc_join_d"
    }
    assert all(child.turns[0].replay_predecessors == [] for child in children)
    assert {
        (ref.conversation_id, ref.turn_index)
        for ref in root.turns[1].replay_predecessors
    } == {(child.session_id, 0) for child in children}


def test_loader_keeps_join_prerequisite_on_d() -> None:
    root = next(c for c in _load() if c.is_root)

    assert root.turns[0].branch_ids
    assert {
        prerequisite.branch_id for prerequisite in root.turns[1].prerequisites
    } == set(root.turns[0].branch_ids)


def test_overlap_frontier_survives_cached_metadata_round_trip() -> None:
    conversations = _load()
    metadata = DatasetMetadata(
        conversations=[conversation.metadata() for conversation in conversations],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )

    restored = DatasetMetadata.model_validate_json(metadata.model_dump_json())
    root = next(
        conversation for conversation in restored.conversations if conversation.is_root
    )
    children = [
        conversation
        for conversation in restored.conversations
        if not conversation.is_root
    ]

    assert root.replay_scope_id == "abc_join_d"
    assert {
        (reference.conversation_id, reference.turn_index)
        for reference in root.turns[1].replay_predecessors
    } == {(child.conversation_id, 0) for child in children}
