# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import Conversation, ModelEndpointInfo, Text, Turn
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.payload_formatting import format_conversation_payloads
from tests.unit.conftest import make_run_from_cli


@pytest.fixture
def model_endpoint():
    run = make_run_from_cli(CLIConfig(model_names=["test-model"]))
    return ModelEndpointInfo.from_run(run)


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

    def test_raw_payload_bypasses_format_payload(self, model_endpoint):
        raw1 = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        raw2 = {"model": "m", "messages": [{"role": "user", "content": "bye"}]}
        conversations = [
            Conversation(
                session_id="s1",
                turns=[
                    Turn(role="user", raw_payload=raw1),
                    Turn(role="user", raw_payload=raw2),
                ],
            ),
        ]

        mock_endpoint = MagicMock()
        mock_endpoint.get_endpoint_headers.return_value = {}
        mock_endpoint.get_endpoint_params.return_value = {}

        with patch(
            "aiperf.dataset.payload_formatting.plugins.get_class",
            return_value=lambda **kwargs: mock_endpoint,
        ):
            results = list(format_conversation_payloads(conversations, model_endpoint))

        assert results == [("s1", 0, raw1), ("s1", 1, raw2)]
        mock_endpoint.format_payload.assert_not_called()

    def test_mixed_raw_and_normal_turns(self, model_endpoint):
        raw_payload = {"model": "m", "messages": [{"role": "user", "content": "raw"}]}
        conversations = [
            Conversation(
                session_id="s1",
                turns=[
                    Turn(role="user", raw_payload=raw_payload),
                    Turn(role="user", texts=[Text(contents=["normal"])]),
                ],
            ),
        ]

        mock_endpoint = MagicMock()
        mock_endpoint.format_payload.return_value = {"formatted": True}
        mock_endpoint.get_endpoint_headers.return_value = {}
        mock_endpoint.get_endpoint_params.return_value = {}

        with patch(
            "aiperf.dataset.payload_formatting.plugins.get_class",
            return_value=lambda **kwargs: mock_endpoint,
        ):
            results = list(format_conversation_payloads(conversations, model_endpoint))

        assert results[0] == ("s1", 0, raw_payload)
        assert results[1] == ("s1", 1, {"formatted": True})
        assert mock_endpoint.format_payload.call_count == 1

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

    def test_propagates_system_and_user_context(self, model_endpoint):
        """Conversation-level system_message / user_context_message must
        flow into the synthesised ``RequestInfo`` so ``format_payload``
        inlines them into the preformatted wire bytes. Regression for
        the bug where composer-injected context was silently dropped at
        preformat time and the server never saw it.
        """
        convs = [
            Conversation(
                session_id="s1",
                system_message="shared system prompt",
                user_context_message="per-session context",
                turns=[Turn(role="user", texts=[Text(contents=["q"])])],
            ),
            Conversation(
                session_id="s2",
                turns=[Turn(role="user", texts=[Text(contents=["q"])])],
            ),
        ]

        captured_infos = []

        def capture_format(request_info):
            captured_infos.append(request_info)
            return {"payload": "test"}

        mock_endpoint = MagicMock()
        mock_endpoint.format_payload.side_effect = capture_format
        mock_endpoint.get_endpoint_headers.return_value = {}
        mock_endpoint.get_endpoint_params.return_value = {}

        with patch(
            "aiperf.dataset.payload_formatting.plugins.get_class",
            return_value=lambda **kwargs: mock_endpoint,
        ):
            list(format_conversation_payloads(convs, model_endpoint))

        assert len(captured_infos) == 2
        assert captured_infos[0].system_message == "shared system prompt"
        assert captured_infos[0].user_context_message == "per-session context"
        assert captured_infos[1].system_message is None
        assert captured_infos[1].user_context_message is None
