# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differential tests: the chat fast path and slow path must AGREE on malformed
``choices``.

``_fast_parse_response`` guards the ``choices`` shape (non-list, or a first
entry that isn't a dict) and returns ``_FAST_PARSE_FALLBACK``; the slow path
``extract_chat_response_data`` must degrade the same inputs to ``None`` rather
than crashing the parser. Both are "no data" outcomes, so for every malformed
body neither path may raise and both must decline to produce response data.
Regression guard for the slow-path ``AttributeError``/``KeyError`` on
``choices[0].get(...)``.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.models.record_models import TextResponseData
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_config,
    create_endpoint_with_mock_transport,
    create_mock_response,
)

# Malformed ``choices`` shapes: each previously crashed the slow path at
# ``choices[0].get(data_key)`` while the fast path already degraded gracefully.
MALFORMED_BODIES = [
    param({"object": "chat.completion.chunk", "choices": [None]}, id="first_is_none"),
    param({"object": "chat.completion.chunk", "choices": ["x"]}, id="first_is_str"),
    param({"object": "chat.completion.chunk", "choices": [5]}, id="first_is_int"),
    param(
        {"object": "chat.completion.chunk", "choices": {"delta": {"content": "hi"}}},
        id="dict_not_list",
    ),
    param({"object": "chat.completion.chunk", "choices": "oops"}, id="str_not_list"),
    param({"object": "chat.completion.chunk", "choices": []}, id="empty_list"),
    param({"object": "chat.completion.chunk"}, id="missing_choices"),
    # Non-streaming sibling: first choice is a dict but ``message`` is a truthy
    # non-dict, which crashed ``data.get(...)`` after the choice-shape guard.
    param(
        {"object": "chat.completion", "choices": [{"message": "hello"}]},
        id="data_not_dict",
    ),
]  # fmt: skip


class TestChatMalformedChoicesAgreement:
    """Fast path and slow path must degrade identically on malformed choices."""

    @pytest.fixture
    def endpoint(self):
        cfg = create_config(EndpointType.CHAT)
        return create_endpoint_with_mock_transport(ChatEndpoint, cfg)

    def _fast_is_no_data(self, endpoint, fast: object) -> bool:
        """Fast-path "no data" sentinel: either explicit None or the fallback."""
        return fast is None or fast is endpoint._FAST_PARSE_FALLBACK

    @pytest.mark.parametrize("json_obj", MALFORMED_BODIES)
    def test_slow_path_degrades_to_none(self, endpoint, json_obj):
        """The slow path must return None (never raise) on malformed choices."""
        assert endpoint.extract_chat_response_data(json_obj) is None

    @pytest.mark.parametrize("json_obj", MALFORMED_BODIES)
    def test_fast_path_degrades_without_raising(self, endpoint, json_obj):
        """The fast path must return a no-data sentinel (never raise)."""
        fast = endpoint._fast_parse_response(json_obj, 123456789)
        assert self._fast_is_no_data(endpoint, fast)

    @pytest.mark.parametrize("json_obj", MALFORMED_BODIES)
    def test_fast_and_slow_agree(self, endpoint, json_obj):
        """Differential: both paths decline to produce data for the same body."""
        fast = endpoint._fast_parse_response(json_obj, 123456789)
        slow = endpoint.extract_chat_response_data(json_obj)
        assert self._fast_is_no_data(endpoint, fast)
        assert slow is None

    @pytest.mark.parametrize("json_obj", MALFORMED_BODIES)
    def test_parse_response_end_to_end_returns_none(self, endpoint, json_obj):
        """End-to-end contract (worker/inference-result reachability): a malformed
        body parses to None instead of raising through ``parse_response``."""
        mock_response = create_mock_response(123456789, json_obj)
        assert endpoint.parse_response(mock_response) is None

    def test_build_assistant_turn_survives_malformed_choices(self, endpoint):
        """The replay-capture path shares the choices-shape assumption; a
        malformed choices entry must not crash ``build_assistant_turn``."""
        from aiperf.common.models import RequestRecord

        record = RequestRecord()
        record.responses = [
            create_mock_response(
                1, {"object": "chat.completion.chunk", "choices": [None]}
            ),
            create_mock_response(
                2, {"object": "chat.completion.chunk", "choices": "oops"}
            ),
            create_mock_response(3, {"object": "chat.completion.chunk", "choices": []}),
        ]
        # Must not raise; with no valid tool_calls it falls back to base behaviour.
        assert endpoint.build_assistant_turn(record) is None

    @pytest.mark.parametrize(
        "json_obj, expected_text",
        [
            param(
                {
                    "object": "chat.completion",
                    "choices": [{"message": {"content": "Hello, world"}}],
                },
                "Hello, world",
                id="non_streaming_message",
            ),
            param(
                {
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": "Hello"}}],
                },
                "Hello",
                id="streaming_delta",
            ),
        ],
    )  # fmt: skip
    def test_happy_path_still_extracts(self, endpoint, json_obj, expected_text):
        """No regression: well-formed bodies still extract text on both paths."""
        slow = endpoint.extract_chat_response_data(json_obj)
        assert isinstance(slow, TextResponseData)
        assert slow.text == expected_text

        fast = endpoint._fast_parse_response(json_obj, 123456789)
        assert fast is not endpoint._FAST_PARSE_FALLBACK
        assert fast is not None
        assert isinstance(fast.data, TextResponseData)
        assert fast.data.text == expected_text
