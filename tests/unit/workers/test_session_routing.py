# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aiperf.workers.session_routing import (
    DynamoHeadersRouting,
    DynamoNvextOptions,
    DynamoNvextRouting,
    RoutingContext,
    SessionIdHeaderOptions,
    SessionIdHeaderRouting,
    SessionRoutingBase,
    SmgRoutingKeyRouting,
)


def _ctx(**overrides) -> RoutingContext:
    defaults = dict(
        x_correlation_id="corr-1",
        parent_correlation_id=None,
        root_correlation_id="corr-1",
        is_final_turn=False,
        is_parent_final=None,
        is_tree_final=False,
    )
    defaults.update(overrides)
    return RoutingContext(**defaults)


class TestDynamoHeadersRouting:
    def test_root_emits_session_header_only(self):
        plugin = DynamoHeadersRouting(DynamoHeadersRouting.Options())
        assert plugin.headers(_ctx()) == {"X-Dynamo-Session-ID": "corr-1"}
        assert plugin.mutates_body is False

    def test_child_emits_parent_header(self):
        plugin = DynamoHeadersRouting(DynamoHeadersRouting.Options())
        headers = plugin.headers(_ctx(parent_correlation_id="parent-1"))
        assert headers == {
            "X-Dynamo-Session-ID": "corr-1",
            "X-Dynamo-Parent-Session-ID": "parent-1",
        }

    def test_body_untouched(self):
        plugin = DynamoHeadersRouting(DynamoHeadersRouting.Options())
        payload = {"messages": []}
        assert plugin.transform_body(payload, _ctx()) is payload


class TestDynamoNvextRouting:
    def test_non_final_turn_binds_with_timeout(self):
        plugin = DynamoNvextRouting(DynamoNvextOptions(timeout_seconds=123))
        merged = plugin.transform_body({"messages": []}, _ctx())
        assert merged["nvext"]["session_control"] == {
            "session_id": "corr-1",
            "action": "bind",
            "timeout": 123,
        }
        assert plugin.mutates_body is True

    def test_final_turn_closes_without_timeout(self):
        plugin = DynamoNvextRouting(DynamoNvextOptions())
        merged = plugin.transform_body({}, _ctx(is_final_turn=True))
        assert merged["nvext"]["session_control"] == {
            "session_id": "corr-1",
            "action": "close",
        }

    def test_never_mutates_input_payload(self):
        nested_sc = {"existing": "keep"}
        nvext = {"trace": "keep", "session_control": nested_sc}
        payload = {"nvext": nvext}
        plugin = DynamoNvextRouting(DynamoNvextOptions())
        merged = plugin.transform_body(payload, _ctx())
        assert payload == {
            "nvext": {"trace": "keep", "session_control": {"existing": "keep"}}
        }
        assert nvext == {"trace": "keep", "session_control": {"existing": "keep"}}
        assert merged is not payload
        assert merged["nvext"]["session_control"]["existing"] == "keep"

    def test_options_default_and_bounds(self):
        assert DynamoNvextOptions().timeout_seconds == 300
        with pytest.raises(ValidationError):
            DynamoNvextOptions(timeout_seconds=0)

    def test_options_reject_unknown_keys(self):
        with pytest.raises(ValidationError):
            DynamoNvextOptions(timeout_secs=5)

    def test_typed_options_access(self):
        plugin = DynamoNvextRouting(DynamoNvextOptions(timeout_seconds=42))
        assert plugin.options.timeout_seconds == 42

    def test_legacy_open_contract_opens_once_then_bare_session_id(self):
        """contract=open (released Dynamo v1.2.x, SessionAction open/close
        only): 'open' (+timeout) on the FIRST request this worker sends for
        the session -- open is NOT idempotent -- then bare session_id."""
        plugin = DynamoNvextRouting(
            DynamoNvextOptions(contract="open", timeout_seconds=77)
        )
        first = plugin.transform_body({}, _ctx())
        assert first["nvext"]["session_control"] == {
            "session_id": "corr-1",
            "action": "open",
            "timeout": 77,
        }
        second = plugin.transform_body({}, _ctx())
        assert second["nvext"]["session_control"] == {"session_id": "corr-1"}
        # A DIFFERENT session opens independently.
        other = plugin.transform_body({}, _ctx(x_correlation_id="corr-2"))
        assert other["nvext"]["session_control"]["action"] == "open"

    def test_legacy_open_contract_final_closes_and_releases_marker(self):
        plugin = DynamoNvextRouting(DynamoNvextOptions(contract="open"))
        plugin.transform_body({}, _ctx())  # opens
        final = plugin.transform_body({}, _ctx(is_final_turn=True))
        assert final["nvext"]["session_control"] == {
            "session_id": "corr-1",
            "action": "close",
        }
        # Marker released: a recycled session with the same corr re-opens.
        reopened = plugin.transform_body({}, _ctx())
        assert reopened["nvext"]["session_control"]["action"] == "open"

    def test_legacy_open_marker_released_by_on_session_end(self):
        """Cancellation paths never send the final turn; on_session_end is
        the terminal hook that must release the per-worker opened marker
        (idempotent) so the set cannot grow unboundedly."""
        plugin = DynamoNvextRouting(DynamoNvextOptions(contract="open"))
        plugin.transform_body({}, _ctx())
        plugin.on_session_end("corr-1")
        plugin.on_session_end("corr-1")  # idempotent
        reopened = plugin.transform_body({}, _ctx())
        assert reopened["nvext"]["session_control"]["action"] == "open"

    def test_bind_contract_keeps_no_per_session_state(self):
        plugin = DynamoNvextRouting(DynamoNvextOptions())
        plugin.transform_body({}, _ctx())
        assert plugin._opened == set()

    def test_options_reject_unknown_contract(self):
        with pytest.raises(ValidationError):
            DynamoNvextOptions(contract="handshake")


class TestSmgRoutingKeyRouting:
    def test_emits_routing_key(self):
        plugin = SmgRoutingKeyRouting(SmgRoutingKeyRouting.Options())
        assert plugin.headers(_ctx()) == {"X-SMG-Routing-Key": "corr-1"}

    def test_rejects_any_opt(self):
        with pytest.raises(ValidationError):
            SmgRoutingKeyRouting.Options(anything="x")


class TestSessionIdHeaderRouting:
    def test_default_header_name(self):
        plugin = SessionIdHeaderRouting(SessionIdHeaderOptions())
        assert plugin.headers(_ctx()) == {"X-Session-ID": "corr-1"}

    def test_custom_header_name(self):
        plugin = SessionIdHeaderRouting(
            SessionIdHeaderOptions(header_name="X-Affinity")
        )
        assert plugin.headers(_ctx()) == {"X-Affinity": "corr-1"}


class TestBaseDefaults:
    def test_on_session_end_default_noop_and_idempotent(self):
        class Passthrough(SessionRoutingBase):
            pass

        plugin = Passthrough(Passthrough.Options())
        plugin.on_session_end("corr-1")
        plugin.on_session_end("corr-1")

    def test_stateful_open_once_lifecycle_expressible(self):
        """The legacy-nvext shape: open-once instance state, bounded by on_session_end."""

        class OpenOnce(SessionRoutingBase):
            mutates_body = True

            def __init__(self, options):
                super().__init__(options)
                self.opened: set[str] = set()

            def transform_body(self, payload, ctx):
                merged = dict(payload)
                if ctx.x_correlation_id not in self.opened:
                    self.opened.add(ctx.x_correlation_id)
                    merged["action"] = "open"
                return merged

            def on_session_end(self, x_correlation_id):
                self.opened.discard(x_correlation_id)

        plugin = OpenOnce(OpenOnce.Options())
        assert plugin.transform_body({}, _ctx())["action"] == "open"
        assert "action" not in plugin.transform_body({}, _ctx())
        plugin.on_session_end("corr-1")
        plugin.on_session_end("corr-1")  # idempotent
        assert plugin.opened == set()
