# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-request extra payload merging in ChatEndpoint.format_payload."""

from typing import Any

import pytest
from pytest import param

from aiperf.common.models import Text, Turn
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)


def _make_endpoint(
    global_extra: list[tuple[str, Any]] | None = None,
) -> tuple[ChatEndpoint, ModelEndpointInfo]:
    model_endpoint = create_model_endpoint(
        EndpointType.CHAT,
        extra=global_extra or [],
    )
    endpoint = create_endpoint_with_mock_transport(ChatEndpoint, model_endpoint)
    return endpoint, model_endpoint


class TestPerRequestExtraMerge:
    """Payload merge tests for per-request extra vs global --extra-inputs."""

    def test_per_request_extra_merged_into_payload(self):
        """Per-request extra fields appear in the formatted payload."""
        endpoint, model_endpoint = _make_endpoint()
        turn = Turn(
            texts=[Text(contents=["hello"])],
            extra_body={"nvext": {"priority": 1}},
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["nvext"] == {"priority": 1}

    def test_global_extra_merged_into_payload(self):
        """Global --extra-inputs fields appear in the formatted payload."""
        endpoint, model_endpoint = _make_endpoint(global_extra=[("temperature", 0.7)])
        turn = Turn(texts=[Text(contents=["hello"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["temperature"] == 0.7

    def test_per_request_extra_overrides_global_same_key(self):
        """Per-request extra overrides global extra for the same top-level key."""
        endpoint, model_endpoint = _make_endpoint(
            global_extra=[("nvext", {"priority": 1})]
        )
        turn = Turn(
            texts=[Text(contents=["hello"])],
            extra_body={"nvext": {"priority": 99}},
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["nvext"] == {"priority": 99}

    def test_none_extra_does_not_break_payload(self):
        """Turn with extra_body=None produces a valid payload without extra keys."""
        endpoint, model_endpoint = _make_endpoint()
        turn = Turn(texts=[Text(contents=["hello"])], extra_body=None)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "messages" in payload
        assert "model" in payload

    @pytest.mark.parametrize(
        "global_extra,per_request_extra,expected_key,expected_value",
        [
            param(
                [],
                {"routing": "fast"},
                "routing",
                "fast",
                id="per_request_only",
            ),
            param(
                [("routing", "slow")],
                None,
                "routing",
                "slow",
                id="global_only",
            ),
            param(
                [("routing", "slow")],
                {"routing": "fast"},
                "routing",
                "fast",
                id="per_request_overrides_global",
            ),
            param(
                [("shared", "global_val"), ("global_only", "x")],
                {"shared": "per_req_val", "per_req_only": "y"},
                "shared",
                "per_req_val",
                id="mixed_keys_override",
            ),
        ],
    )  # fmt: skip
    def test_merge_priority(
        self,
        global_extra,
        per_request_extra,
        expected_key,
        expected_value,
    ):
        """Per-request extra has higher priority than global extra."""
        endpoint, model_endpoint = _make_endpoint(global_extra=global_extra)
        turn = Turn(texts=[Text(contents=["hello"])], extra_body=per_request_extra)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload[expected_key] == expected_value

    def test_both_global_and_per_request_disjoint_keys_present(self):
        """Both global and per-request extra keys appear in payload when they differ."""
        endpoint, model_endpoint = _make_endpoint(
            global_extra=[("global_key", "from_global")]
        )
        turn = Turn(
            texts=[Text(contents=["hello"])],
            extra_body={"per_req_key": "from_per_req"},
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["global_key"] == "from_global"
        assert payload["per_req_key"] == "from_per_req"
