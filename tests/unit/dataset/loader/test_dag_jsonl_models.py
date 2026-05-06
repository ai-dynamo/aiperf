# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aiperf.dataset.loader.dag_jsonl_models import DagSpawn, DagTurn


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
