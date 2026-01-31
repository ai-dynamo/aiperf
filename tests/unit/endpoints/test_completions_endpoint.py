# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models import Text, Turn
from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)


class TestCompletionsEndpoint:
    """Comprehensive tests for CompletionsEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for completions."""
        return create_model_endpoint(
            EndpointType.COMPLETIONS, model_name="completion-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        """Create a CompletionsEndpoint instance."""
        return create_endpoint_with_mock_transport(CompletionsEndpoint, model_endpoint)

    def test_format_payload_single_prompt(self, endpoint, model_endpoint):
        """Test single prompt formatting."""
        turn = Turn(
            texts=[Text(contents=["Once upon a time"])],
            model="completion-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "completion-model"
        assert payload["stream"] is False
        assert payload["prompt"] == ["Once upon a time"]

    def test_format_payload_multiple_prompts(self, endpoint, model_endpoint):
        """Test multiple prompts are all included."""
        turn = Turn(
            texts=[
                Text(contents=["Prompt 1", "Prompt 2"]),
                Text(contents=["Prompt 3"]),
            ],
            model="completion-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["prompt"]) == 3
        assert payload["prompt"] == ["Prompt 1", "Prompt 2", "Prompt 3"]

    def test_format_payload_filters_empty_prompts(self, endpoint, model_endpoint):
        """Test that empty strings are filtered from prompts."""
        turn = Turn(
            texts=[
                Text(contents=["Valid", "", "Also valid", ""]),
            ],
            model="completion-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["prompt"]) == 2
        assert payload["prompt"] == ["Valid", "Also valid"]

    @pytest.mark.parametrize(
        "max_tokens,should_be_in_payload",
        [
            (200, True),
            (None, False),
            (1000, True),
        ],
    )
    def test_format_payload_max_tokens(
        self, endpoint, model_endpoint, max_tokens, should_be_in_payload
    ):
        """Test max_tokens handling in payload."""
        turn = Turn(
            texts=[Text(contents=["Generate"])],
            model="completion-model",
            max_tokens=max_tokens,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        if should_be_in_payload:
            assert payload["max_tokens"] == max_tokens
        else:
            assert "max_tokens" not in payload

    def test_format_payload_streaming_enabled(self):
        """Test streaming flag from endpoint config."""
        streaming_endpoint = create_model_endpoint(
            EndpointType.COMPLETIONS, model_name="completion-model", streaming=True
        )
        endpoint = create_endpoint_with_mock_transport(
            CompletionsEndpoint, streaming_endpoint
        )

        turn = Turn(texts=[Text(contents=["Test"])], model="completion-model")
        request_info = create_request_info(
            model_endpoint=streaming_endpoint, turns=[turn]
        )

        payload = endpoint.format_payload(request_info)

        assert payload["stream"] is True

    def test_format_payload_extra_parameters(self):
        """Test extra parameters are merged into payload."""
        extra_params = [
            ("temperature", 0.8),
            ("top_p", 0.95),
            ("frequency_penalty", 0.5),
        ]
        model_endpoint = create_model_endpoint(
            EndpointType.COMPLETIONS, model_name="completion-model", extra=extra_params
        )
        endpoint = create_endpoint_with_mock_transport(
            CompletionsEndpoint, model_endpoint
        )

        turn = Turn(texts=[Text(contents=["Test"])], model="completion-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["temperature"] == 0.8
        assert payload["top_p"] == 0.95
        assert payload["frequency_penalty"] == 0.5

    def test_format_payload_empty_texts_list(self, endpoint, model_endpoint):
        """Test behavior with empty texts list."""
        turn = Turn(
            texts=[],  # No texts
            model="completion-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["prompt"] == []

    def test_format_payload_uses_first_turn_only(self, endpoint, model_endpoint):
        """Test that only request_info.turns[0] is used (hardcoded)."""
        turn1 = Turn(texts=[Text(contents=["First"])], model="model1")
        turn2 = Turn(texts=[Text(contents=["Second"])], model="model2")

        request_info = create_request_info(
            model_endpoint=model_endpoint, turn_index=0, turns=[turn1, turn2]
        )

        # Should raise ValueError because completions endpoint only supports one turn
        with pytest.raises(
            ValueError, match="Completions endpoint only supports one turn"
        ):
            endpoint.format_payload(request_info)
