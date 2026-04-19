# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for ResponsesEndpoint after config-v3 / model_endpoint port."""

import pytest

from aiperf.common.models import Text, Turn
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_config,
    create_endpoint_with_mock_transport,
    create_request_info,
)


class TestResponsesEndpointSmoke:
    """Verify ResponsesEndpoint works against k8s's request_info.config pattern."""

    @pytest.fixture
    def model_endpoint(self):
        return create_config(EndpointType.CHAT, model_name="responses-model")

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(ResponsesEndpoint, model_endpoint)

    def test_format_payload_single_turn(self, endpoint, model_endpoint):
        turn = Turn(
            texts=[Text(contents=["Hello"])],
            model="responses-model",
        )
        request_info = create_request_info(config=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "responses-model"
        assert payload["stream"] is False
        assert "input" in payload

    def test_format_payload_requires_turns(self, endpoint, model_endpoint):
        request_info = create_request_info(config=model_endpoint, turns=[])
        with pytest.raises(ValueError, match="at least one turn"):
            endpoint.format_payload(request_info)
