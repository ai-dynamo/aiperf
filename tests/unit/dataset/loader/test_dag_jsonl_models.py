# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aiperf.dataset.loader.dag_jsonl_models import (
    DagConversation,
    DagSpawn,
    DagTurn,
)


class TestDagSpawn:
    def test_minimal_construction(self):
        s = DagSpawn(children=["child-session-1"])
        assert s.children == ["child-session-1"]
        assert s.join_at is None

    def test_round_trip(self):
        s = DagSpawn(children=["t1"], join_at=3)
        dumped = s.model_dump()
        rebuilt = DagSpawn.model_validate(dumped)
        assert rebuilt == s

    def test_children_is_required(self):
        with pytest.raises(ValidationError):
            DagSpawn()  # type: ignore[call-arg]

    def test_children_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            DagSpawn(children=[])


class TestDagTurnMaxTokens:
    def test_max_tokens_int_accepted(self):
        turn = DagTurn(messages=[{"role": "user", "content": "x"}], max_tokens=128)
        assert turn.max_tokens == 128

    def test_max_tokens_bool_rejected(self):
        # bool is an int subclass in Python; without an explicit guard
        # Pydantic accepts ``True`` -> 1 silently. This rejects both forms.
        with pytest.raises(ValidationError, match="must be an integer, not a boolean"):
            DagTurn.model_validate(
                {"messages": [{"role": "user", "content": "x"}], "max_tokens": True}
            )
        with pytest.raises(ValidationError, match="must be an integer, not a boolean"):
            DagTurn.model_validate(
                {"messages": [{"role": "user", "content": "x"}], "max_tokens": False}
            )


class TestDagConversationOrchestrator:
    def test_orchestrator_conversation_loads_with_spawns(self):
        conv = DagConversation(
            session_id="start",
            turns=[],
            orchestrator=True,
            spawns=["fan-out-a", "fan-out-b"],
        )
        assert conv.orchestrator is True
        assert conv.turns == []
        assert conv.spawns == ["fan-out-a", "fan-out-b"]

    def test_orchestrator_requires_spawns(self):
        with pytest.raises(ValueError, match="spawns"):
            DagConversation(session_id="s", turns=[], orchestrator=True, spawns=[])

    def test_orchestrator_rejects_non_empty_turns(self):
        with pytest.raises(ValueError, match="orchestrator"):
            DagConversation(
                session_id="s",
                orchestrator=True,
                spawns=["c"],
                turns=[{"messages": [{"role": "user", "content": "hi"}]}],
            )

    def test_orchestrator_rejects_pre_session_spawns(self):
        with pytest.raises(ValueError, match="pre_session_spawns"):
            DagConversation(
                session_id="s",
                turns=[],
                orchestrator=True,
                spawns=["c"],
                pre_session_spawns=["c"],
            )

    def test_empty_turns_without_orchestrator_rejected(self):
        with pytest.raises(ValueError):
            DagConversation(session_id="s", turns=[])
