# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.models.dataset_models import TurnMetadata
from aiperf.credit.structs import Credit, TurnToSend


class TestCreditDagFields:
    def _base_credit_kwargs(self) -> dict:
        return {
            "id": 0,
            "phase": CreditPhase.PROFILING,
            "conversation_id": "c",
            "x_correlation_id": "x",
            "turn_index": 0,
            "num_turns": 1,
            "issued_at_ns": 0,
        }

    def test_credit_default_dag_fields(self):
        c = Credit(**self._base_credit_kwargs())
        assert c.agent_depth == 0
        assert c.parent_correlation_id is None
        assert c.has_forks is False
        assert c.branch_mode is ConversationBranchMode.FORK

    def test_credit_explicit_dag_fields(self):
        kwargs = self._base_credit_kwargs()
        kwargs["id"] = 1
        kwargs["x_correlation_id"] = "child-1"
        c = Credit(
            **kwargs,
            agent_depth=2,
            parent_correlation_id="root-corr",
            has_forks=True,
            branch_mode=ConversationBranchMode.SPAWN,
        )
        assert c.agent_depth == 2
        assert c.parent_correlation_id == "root-corr"
        assert c.has_forks is True
        assert c.branch_mode is ConversationBranchMode.SPAWN

    def test_turn_to_send_dag_fields_propagate_from_credit(self):
        kwargs = self._base_credit_kwargs()
        kwargs["id"] = 2
        kwargs["x_correlation_id"] = "child"
        kwargs["num_turns"] = 3
        c = Credit(
            **kwargs,
            agent_depth=1,
            parent_correlation_id="root",
            has_forks=False,
            branch_mode=ConversationBranchMode.FORK,
        )
        tts = TurnToSend.from_previous_credit(c)
        assert tts.agent_depth == 1
        assert tts.parent_correlation_id == "root"
        assert tts.branch_mode is ConversationBranchMode.FORK
        assert tts.has_forks is False

    def test_turn_to_send_has_forks_from_next_meta(self):
        kwargs = self._base_credit_kwargs()
        kwargs["id"] = 3
        kwargs["num_turns"] = 3
        c = Credit(**kwargs)
        meta = TurnMetadata(has_forks=True)
        tts = TurnToSend.from_previous_credit(c, next_meta=meta)
        assert tts.has_forks is True

    def test_turn_to_send_default_dag_fields(self):
        tts = TurnToSend(
            conversation_id="c",
            x_correlation_id="x",
            turn_index=0,
            num_turns=1,
        )
        assert tts.agent_depth == 0
        assert tts.parent_correlation_id is None
        assert tts.has_forks is False
        assert tts.branch_mode is ConversationBranchMode.FORK
