# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.enums import CreditPhase
from aiperf.common.models import Conversation, ModelEndpointInfo, Text, Turn
from aiperf.dataset.payload_formatting import format_conversation_payloads


@pytest.fixture
def model_endpoint():
    config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
    return ModelEndpointInfo.from_user_config(config)


@pytest.fixture
def conversations():
    return [
        Conversation(
            session_id="s1",
            turns=[
                Turn(role="user", texts=[Text(contents=["hello"])]),
                Turn(role="user", texts=[Text(contents=["world"])]),
            ],
        ),
        Conversation(
            session_id="s2",
            turns=[
                Turn(role="user", texts=[Text(contents=["foo"])]),
            ],
        ),
    ]


class TestFormatConversationPayloads:
    def test_yields_payload_per_turn(self, conversations, model_endpoint):
        mock_endpoint = MagicMock()
        mock_endpoint.format_payload.side_effect = [
            {"p": 1},
            {"p": 2},
            {"p": 3},
        ]
        mock_endpoint.get_endpoint_headers.return_value = {}
        mock_endpoint.get_endpoint_params.return_value = {}

        with patch(
            "aiperf.dataset.payload_formatting.plugins.get_class",
            return_value=lambda **kwargs: mock_endpoint,
        ):
            results = list(format_conversation_payloads(conversations, model_endpoint))

        assert len(results) == 3
        assert results[0] == ("s1", 0, {"p": 1})
        assert results[1] == ("s1", 1, {"p": 2})
        assert results[2] == ("s2", 0, {"p": 3})

    def test_request_info_fields(self, conversations, model_endpoint):
        mock_endpoint = MagicMock()
        mock_endpoint.format_payload.return_value = {"payload": "test"}
        mock_endpoint.get_endpoint_headers.return_value = {"h": "v"}
        mock_endpoint.get_endpoint_params.return_value = {"p": "v"}

        captured_infos = []

        def capture_format(request_info):
            captured_infos.append(request_info)
            return {"payload": "test"}

        mock_endpoint.format_payload.side_effect = capture_format

        with patch(
            "aiperf.dataset.payload_formatting.plugins.get_class",
            return_value=lambda **kwargs: mock_endpoint,
        ):
            list(format_conversation_payloads(conversations[:1], model_endpoint))

        assert len(captured_infos) == 2
        info = captured_infos[0]
        assert info.conversation_id == "s1"
        assert info.turn_index == 0
        assert info.credit_phase == CreditPhase.PROFILING
        assert info.endpoint_headers == {"h": "v"}
        assert info.endpoint_params == {"p": "v"}
        assert len(info.turns) == 1

    def test_empty_conversations(self, model_endpoint):
        mock_endpoint = MagicMock()
        with patch(
            "aiperf.dataset.payload_formatting.plugins.get_class",
            return_value=lambda **kwargs: mock_endpoint,
        ):
            results = list(format_conversation_payloads([], model_endpoint))

        assert results == []
        mock_endpoint.format_payload.assert_not_called()

    def test_propagates_not_implemented(self, conversations, model_endpoint):
        mock_endpoint = MagicMock()
        mock_endpoint.format_payload.side_effect = NotImplementedError
        mock_endpoint.get_endpoint_headers.return_value = {}
        mock_endpoint.get_endpoint_params.return_value = {}

        with (
            patch(
                "aiperf.dataset.payload_formatting.plugins.get_class",
                return_value=lambda **kwargs: mock_endpoint,
            ),
            pytest.raises(NotImplementedError),
        ):
            list(format_conversation_payloads(conversations, model_endpoint))
