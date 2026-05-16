# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import msgspec
import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.models.base_models import _msgspec_dec_hook, _msgspec_enc_hook
from aiperf.common.models.branch import ConversationBranchInfo


class TestConversationBranchInfoDefaults:
    def test_fork_default_dispatch_post(self):
        b = ConversationBranchInfo(
            branch_id="root:0",
            child_conversation_ids=["c1"],
            mode=ConversationBranchMode.FORK,
        )
        assert b.dispatch_timing == "post"
        assert b.mode is ConversationBranchMode.FORK

    def test_spawn_default_dispatch_post(self):
        b = ConversationBranchInfo(
            branch_id="root:0",
            child_conversation_ids=["c1"],
            mode=ConversationBranchMode.SPAWN,
        )
        assert b.dispatch_timing == "post"

    def test_spawn_can_set_pre(self):
        b = ConversationBranchInfo(
            branch_id="root:0",
            child_conversation_ids=["c1"],
            mode=ConversationBranchMode.SPAWN,
            dispatch_timing="pre",
        )
        assert b.dispatch_timing == "pre"


class TestConversationBranchInfoValidator:
    def test_fork_rejects_pre(self):
        # __post_init__ raises ValueError on FORK+pre.
        with pytest.raises(ValueError) as exc_info:
            ConversationBranchInfo(
                branch_id="root:0",
                child_conversation_ids=["c1"],
                mode=ConversationBranchMode.FORK,
                dispatch_timing="pre",
            )
        assert "SPAWN" in str(exc_info.value) or "spawn" in str(exc_info.value)

    def test_invalid_dispatch_value(self):
        # msgspec does NOT validate Literal types on direct construction; only
        # on convert/decode. Round through convert to exercise the validator.
        with pytest.raises(msgspec.ValidationError):
            msgspec.convert(
                {
                    "branch_id": "root:0",
                    "child_conversation_ids": ["c1"],
                    "mode": "spawn",
                    "dispatch_timing": "bogus",
                },
                ConversationBranchInfo,
                dec_hook=_msgspec_dec_hook,
            )


class TestConversationBranchInfoSerialization:
    def test_round_trip(self):
        b = ConversationBranchInfo(
            branch_id="root:0",
            child_conversation_ids=["c1", "c2"],
            mode=ConversationBranchMode.SPAWN,
            dispatch_timing="pre",
        )
        dumped = msgspec.to_builtins(b, enc_hook=_msgspec_enc_hook)
        restored = msgspec.convert(
            dumped, ConversationBranchInfo, dec_hook=_msgspec_dec_hook
        )
        assert restored == b
