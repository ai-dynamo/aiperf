# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import _wrap_run, create_config, create_request_info


class TestCompletionsEndpoint:
    """Test CompletionsEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test BenchmarkConfig."""
        return create_config(
            EndpointType.CHAT,
            base_url="http://localhost:8000",
            path="/v1/chat/completions",
            api_key="test-api-key",
        )

    def test_format_payload_basic(self, model_endpoint, sample_conversations):
        endpoint = CompletionsEndpoint(run=_wrap_run(model_endpoint))
        # Use the first turn from the sample_conversations fixture
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        request_info = create_request_info(config=model_endpoint, turns=turns)
        payload = endpoint.format_payload(request_info)
        expected_payload = {
            "prompt": ["Hello, world!"],
            "model": "test-model",
            "stream": False,
        }
        assert payload == expected_payload

    def test_format_payload_with_extra_options(self, sample_conversations):
        cfg = create_config(
            EndpointType.CHAT,
            base_url="http://localhost:8000",
            path="/v1/chat/completions",
            api_key="test-api-key",
            streaming=True,
            extra={"ignore_eos": True},
        )
        endpoint = CompletionsEndpoint(run=_wrap_run(cfg))
        # Use the first turn from the sample_conversations fixture
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        turns[0].max_tokens = 50
        request_info = create_request_info(config=cfg, turns=turns)
        payload = endpoint.format_payload(request_info)
        expected_payload = {
            "prompt": ["Hello, world!"],
            "model": "test-model",
            "stream": True,
            "max_tokens": 50,
            "ignore_eos": True,
        }
        assert payload == expected_payload

    @pytest.mark.parametrize(
        "streaming,use_server_token_count,user_extra,expected_stream_options",
        [
            # Auto-add when both flags enabled
            (True, True, None, {"include_usage": True}),
            # Don't add when not streaming
            (False, True, None, None),
            # Don't add when flag disabled
            (True, False, None, None),
            # Don't add when neither enabled
            (False, False, None, None),
            # Preserve user's include_usage=False
            (True, True, {"stream_options": {"include_usage": False}}, {"include_usage": False}),
            # Merge with user's other options
            (True, True, {"stream_options": {"continuous_updates": True}}, {"continuous_updates": True, "include_usage": True}),
        ],
    )  # fmt: skip
    def test_stream_options_auto_configuration(
        self,
        sample_conversations,
        streaming,
        use_server_token_count,
        user_extra,
        expected_stream_options,
    ):
        """Verify stream_options.include_usage is automatically configured based on flags and user settings."""
        cfg = create_config(
            EndpointType.CHAT,
            base_url="http://localhost:8000",
            path="/v1/chat/completions",
            api_key="test-api-key",
            streaming=streaming,
            use_server_token_count=use_server_token_count,
            extra=user_extra or {},
        )
        endpoint = CompletionsEndpoint(run=_wrap_run(cfg))
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        request_info = create_request_info(turns=turns, config=cfg)
        payload = endpoint.format_payload(request_info)

        if expected_stream_options is None:
            assert "stream_options" not in payload
        else:
            assert "stream_options" in payload
            assert payload["stream_options"] == expected_stream_options
