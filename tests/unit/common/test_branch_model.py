# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def test_conversation_branch_info_has_no_join_turn_index_field():
    from aiperf.common.models import ConversationBranchInfo

    fields = ConversationBranchInfo.model_fields
    assert "join_turn_index" not in fields, (
        "join_turn_index removed in dag3; use TurnPrerequisite(kind=SPAWN_JOIN, ...) "
        "on the gated turn instead"
    )
