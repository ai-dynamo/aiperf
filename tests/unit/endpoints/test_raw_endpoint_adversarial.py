# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial coverage for RawEndpoint.format_payload and JMESPathResponseMixin.

Pins current behavior at edge inputs:
- format_payload accepts/refuses raw_payload variants and always uses the last turn
- JMESPath compile is robust to non-string and falsy response_field values
  (TypeError is caught alongside JMESPathError)
- parse_response handles empty/invalid bodies and falls back to auto-detect
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.models import Turn
from aiperf.common.models.record_models import TextResponseData
from aiperf.endpoints.raw_endpoint import RawEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


@pytest.fixture
def raw_model_endpoint():
    return create_model_endpoint(EndpointType.RAW)


@pytest.fixture
def raw_endpoint(raw_model_endpoint):
    return create_endpoint_with_mock_transport(RawEndpoint, raw_model_endpoint)


class TestFormatPayloadEdges:
    def test_format_payload_empty_dict_raw_payload_returns_empty_dict(
        self, raw_endpoint, raw_model_endpoint
    ):
        """Empty dict raw_payload is accepted (not None) and returned verbatim."""
        request_info = create_request_info(
            model_endpoint=raw_model_endpoint,
            turns=[Turn(role="user", raw_payload={})],
        )
        assert raw_endpoint.format_payload(request_info) == {}

    def test_format_payload_none_raw_payload_raises_not_implemented(
        self, raw_endpoint, raw_model_endpoint
    ):
        """Explicit None raw_payload triggers NotImplementedError, not silent return."""
        request_info = create_request_info(
            model_endpoint=raw_model_endpoint,
            turns=[Turn(role="user", raw_payload=None)],
        )
        with pytest.raises(NotImplementedError, match="no raw_payload set"):
            raw_endpoint.format_payload(request_info)

    def test_format_payload_no_turns_raises(self, raw_endpoint, raw_model_endpoint):
        """Empty turns list cannot satisfy the raw-payload contract."""
        request_info = create_request_info(
            model_endpoint=raw_model_endpoint,
            turns=[],
        )
        with pytest.raises(NotImplementedError, match="no raw_payload set"):
            raw_endpoint.format_payload(request_info)

    def test_format_payload_uses_last_turn_not_first(
        self, raw_endpoint, raw_model_endpoint
    ):
        """When multiple turns are present, format_payload returns the last one."""
        first = {"marker": "first", "messages": [{"role": "user", "content": "a"}]}
        last = {"marker": "last", "messages": [{"role": "user", "content": "z"}]}
        request_info = create_request_info(
            model_endpoint=raw_model_endpoint,
            turns=[
                Turn(role="user", raw_payload=first),
                Turn(role="assistant", raw_payload=last),
            ],
        )
        result = raw_endpoint.format_payload(request_info)
        assert result == last
        assert result["marker"] == "last"
        assert result != first

    def test_format_payload_raw_payload_with_nested_structure_preserved_verbatim(
        self, raw_endpoint, raw_model_endpoint
    ):
        """Nested dicts/lists/unicode survive Pydantic round-trip with deep equality.

        Note: Turn is a Pydantic model that copies dict inputs, so identity is
        not preserved -- but every key, value, and unicode character must match.
        """
        payload = {
            "model": "llama-3",
            "messages": [
                {"role": "system", "content": "你好, world"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "emoji rocket"},
                        {"type": "image", "url": "data:image/png;base64,AAAA"},
                    ],
                },
            ],
            "metadata": {"nested": {"deep": [1, 2, [3, 4, {"x": None}]]}},
            "stream": True,
        }
        request_info = create_request_info(
            model_endpoint=raw_model_endpoint,
            turns=[Turn(role="user", raw_payload=payload)],
        )
        result = raw_endpoint.format_payload(request_info)
        assert result == payload
        assert result["messages"][0]["content"] == "你好, world"
        assert result["metadata"]["nested"]["deep"][2][2]["x"] is None


class TestJMESPathCompileEdges:
    @pytest.mark.parametrize(
        "response_field",
        [
            param(123, id="non_string_typeerror_caught"),
            param(None, id="none_skips_compile"),
            param("", id="empty_string_falsy_skips_compile"),
        ],
    )  # fmt: skip
    def test_jmespath_compile_edge_response_field_installs_no_parser(
        self, response_field
    ):
        """Non-string/None/empty response_field never installs a compiled parser.

        Non-string values raise TypeError inside jmespath, which the mixin must
        catch alongside JMESPathError; None and empty string are falsy so the
        compile step is skipped entirely. All leave ``_compiled_jmespath`` None.
        """
        model_endpoint = create_model_endpoint(
            EndpointType.RAW,
            extra=[("response_field", response_field)],
        )
        endpoint = create_endpoint_with_mock_transport(RawEndpoint, model_endpoint)
        assert endpoint._compiled_jmespath is None

    def test_jmespath_compile_none_response_field_uses_auto_detect(self):
        """With no compiled parser, parse_response falls back to auto-detect."""
        model_endpoint = create_model_endpoint(
            EndpointType.RAW,
            extra=[("response_field", None)],
        )
        endpoint = create_endpoint_with_mock_transport(RawEndpoint, model_endpoint)
        assert endpoint._compiled_jmespath is None

        json_data = {"choices": [{"text": "auto"}]}
        parsed = endpoint.parse_response(create_mock_response(json_data=json_data))
        assert parsed is not None
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "auto"


class TestParseResponseEdges:
    def test_parse_response_empty_string_returns_none_or_empty(self, raw_endpoint):
        """Empty body (no JSON, empty text) returns None."""
        parsed = raw_endpoint.parse_response(
            create_mock_response(json_data=None, text="")
        )
        assert parsed is None

    def test_parse_response_invalid_json_falls_back_to_text(self, raw_endpoint):
        """When get_json() returns None but get_text() yields raw text, return text."""
        parsed = raw_endpoint.parse_response(
            create_mock_response(json_data=None, text="not-json: <<<garbage>>>")
        )
        assert parsed is not None
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "not-json: <<<garbage>>>"

    def test_parse_response_valid_json_no_response_field_uses_auto_detect(
        self, raw_endpoint
    ):
        """With no JMESPath query, auto_detect_and_extract handles known shapes."""
        assert raw_endpoint._compiled_jmespath is None
        json_data = {"choices": [{"text": "hi"}]}
        parsed = raw_endpoint.parse_response(create_mock_response(json_data=json_data))
        assert parsed is not None
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "hi"
