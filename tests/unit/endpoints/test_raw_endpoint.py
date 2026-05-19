# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models import Turn
from aiperf.endpoints.raw_endpoint import RawEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)


@pytest.fixture
def raw_endpoint():
    ep_info = create_model_endpoint(EndpointType.RAW)
    return create_endpoint_with_mock_transport(RawEndpoint, ep_info)


class TestRawEndpointFormatPayload:
    def test_returns_raw_payload_from_last_turn(self, raw_endpoint):
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "Qwen/Qwen3-0.6B",
            "max_tokens": 16,
        }
        turn = Turn(role="user", raw_payload=payload)
        ep_info = create_model_endpoint(EndpointType.RAW)
        request_info = create_request_info(model_endpoint=ep_info, turns=[turn])
        assert raw_endpoint.format_payload(request_info) == payload

    def test_no_raw_payload_raises(self, raw_endpoint):
        turn = Turn(role="user")
        ep_info = create_model_endpoint(EndpointType.RAW)
        request_info = create_request_info(model_endpoint=ep_info, turns=[turn])
        with pytest.raises(NotImplementedError) as exc_info:
            raw_endpoint.format_payload(request_info)
        assert "raw_payload" in str(exc_info.value)

    def test_no_turns_raises(self, raw_endpoint):
        ep_info = create_model_endpoint(EndpointType.RAW)
        request_info = create_request_info(model_endpoint=ep_info, turns=[])
        with pytest.raises(NotImplementedError):
            raw_endpoint.format_payload(request_info)
