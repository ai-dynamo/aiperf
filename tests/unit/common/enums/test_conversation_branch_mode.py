# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import ConversationBranchMode


class TestConversationBranchMode:
    def test_members_present(self):
        assert ConversationBranchMode.FORK == "fork"
        assert ConversationBranchMode.SPAWN == "spawn"

    def test_string_round_trip(self):
        assert ConversationBranchMode("fork") is ConversationBranchMode.FORK
        assert ConversationBranchMode("FORK") is ConversationBranchMode.FORK
        assert str(ConversationBranchMode.SPAWN) == "spawn"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("fork", ConversationBranchMode.FORK),
            ("FORK", ConversationBranchMode.FORK),
            ("Fork", ConversationBranchMode.FORK),
            ("spawn", ConversationBranchMode.SPAWN),
            ("SPAWN", ConversationBranchMode.SPAWN),
        ],
    )
    def test_case_insensitive(self, raw: str, expected: ConversationBranchMode):
        assert ConversationBranchMode(raw) is expected
