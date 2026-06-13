# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Dynamo nvext.session_control helpers."""

from aiperf.workers.dynamo_session_control import (
    build_session_control,
    merge_session_control,
)


class TestBuildSessionControl:
    """Policy: which lifecycle action each turn emits."""

    def test_non_final_turn_binds_with_timeout(self):
        """Non-final turns re-bind (idempotent, refreshes the router TTL)."""
        sc = build_session_control(
            session_id="conv-1", is_final_turn=False, timeout_seconds=300
        )
        assert sc == {"session_id": "conv-1", "action": "bind", "timeout": 300}

    def test_final_turn_closes_without_timeout(self):
        """Final turn closes the session; timeout is irrelevant to close."""
        sc = build_session_control(
            session_id="conv-1", is_final_turn=True, timeout_seconds=300
        )
        assert sc == {"session_id": "conv-1", "action": "close"}

    def test_session_id_and_timeout_are_passed_through(self):
        """session_id is the routing key; timeout flows through on bind."""
        sc = build_session_control(
            session_id="x-corr-abc", is_final_turn=False, timeout_seconds=42
        )
        assert sc["session_id"] == "x-corr-abc"
        assert sc["timeout"] == 42


class TestMergeSessionControl:
    """Mechanism: overlay session_control under nvext without mutating input."""

    def test_adds_nvext_session_control_to_bare_payload(self):
        payload = {"messages": [], "model": "m"}
        sc = {"session_id": "c1", "action": "bind", "timeout": 300}

        merged = merge_session_control(payload, sc)

        assert merged["nvext"]["session_control"] == sc
        assert merged["messages"] == []
        assert merged["model"] == "m"

    def test_preserves_existing_nvext_and_session_control_fields(self):
        payload = {"nvext": {"trace": "keep", "session_control": {"existing": "keep"}}}
        sc = {"session_id": "c1", "action": "bind", "timeout": 300}

        merged = merge_session_control(payload, sc)

        assert merged["nvext"] == {
            "trace": "keep",
            "session_control": {
                "existing": "keep",
                "session_id": "c1",
                "action": "bind",
                "timeout": 300,
            },
        }

    def test_does_not_mutate_input_payload_or_nested_dicts(self):
        """Safe to call on a cached Turn.raw_payload / shared extra_body."""
        nested_sc = {"existing": "keep"}
        nvext = {"session_control": nested_sc}
        payload = {"nvext": nvext}
        sc = {"session_id": "c1", "action": "close"}

        merged = merge_session_control(payload, sc)

        # Inputs are left pristine at every level.
        assert payload == {"nvext": {"session_control": {"existing": "keep"}}}
        assert nvext == {"session_control": {"existing": "keep"}}
        assert nested_sc == {"existing": "keep"}
        assert merged is not payload
        # The returned copy carries the overlay.
        assert merged["nvext"]["session_control"] == {
            "existing": "keep",
            "session_id": "c1",
            "action": "close",
        }
