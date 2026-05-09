# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage tests for genai_semconv error classification, attribute builders, and host extraction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest import param

from aiperf.post_processors.strategies.genai_semconv import (
    _build_duration_attributes,
    _build_token_usage_attributes,
    _classify_error_type,
    _extract_host,
    convert_metric_value,
    translate,
)


class TestClassifyErrorType:
    """Cover all branches of _classify_error_type."""

    def test_none_error_returns_none(self) -> None:
        assert _classify_error_type(None) is None

    @pytest.mark.parametrize(
        "code,expected",
        [
            param(500, "http_5xx", id="500"),
            param(502, "http_5xx", id="502"),
            param(599, "http_5xx", id="599"),
            param(400, "http_4xx", id="400"),
            param(404, "http_4xx", id="404"),
            param(429, "http_4xx", id="429"),
            param(499, "http_4xx", id="499"),
        ],
    )  # fmt: skip
    def test_http_status_codes(self, code: int, expected: str) -> None:
        error = SimpleNamespace(code=code, type=None, cause_chain=[], message="")
        assert _classify_error_type(error) == expected

    @pytest.mark.parametrize(
        "error_type,expected",
        [
            param("timeout", "timeout", id="timeout"),
            param("asyncio.TimeoutError", "timeout", id="asyncio-timeout"),
            param("TimeoutError", "timeout", id="timeout-error"),
            param("cancelled", "cancelled", id="cancelled"),
        ],
    )  # fmt: skip
    def test_error_type_mapping(self, error_type: str, expected: str) -> None:
        error = SimpleNamespace(code=None, type=error_type, cause_chain=[], message="")
        assert _classify_error_type(error) == expected

    def test_cause_chain_timeout(self) -> None:
        error = SimpleNamespace(
            code=None, type=None, cause_chain=["Connection Timeout reached"], message=""
        )
        assert _classify_error_type(error) == "timeout"

    def test_cause_chain_cancelled(self) -> None:
        error = SimpleNamespace(
            code=None,
            type=None,
            cause_chain=["Request was Cancelled by user"],
            message="",
        )
        assert _classify_error_type(error) == "cancelled"

    def test_json_parse_error_in_message(self) -> None:
        error = SimpleNamespace(
            code=None, type=None, cause_chain=[], message="JSON parse error at line 5"
        )
        assert _classify_error_type(error) == "parse_error"

    def test_json_decode_error_in_message(self) -> None:
        error = SimpleNamespace(
            code=None,
            type=None,
            cause_chain=[],
            message="Failed to json decode response",
        )
        assert _classify_error_type(error) == "parse_error"

    def test_unknown_error_returns_other(self) -> None:
        error = SimpleNamespace(
            code=None,
            type="unknown_type",
            cause_chain=[],
            message="something went wrong",
        )
        assert _classify_error_type(error) == "_OTHER"

    def test_error_with_no_attributes(self) -> None:
        error = object()
        assert _classify_error_type(error) == "_OTHER"

    @pytest.mark.parametrize(
        "non_int_code",
        [
            param("500", id="string-digit"),
            param("timeout", id="string-word"),
            param(5.5, id="float"),
            param([500], id="list"),
            param({"status": 500}, id="dict"),
        ],
    )  # fmt: skip
    def test_non_integer_code_falls_through(self, non_int_code: Any) -> None:
        """A non-int `.code` must not raise TypeError; the classifier should
        fall through to the other branches (here: returns '_OTHER' because
        no other field matches).
        """
        error = SimpleNamespace(
            code=non_int_code, type=None, cause_chain=[], message=""
        )
        # Must not raise; classifier should ignore the non-int code and
        # return the fallback.
        assert _classify_error_type(error) == "_OTHER"

    def test_bool_code_is_not_treated_as_int(self) -> None:
        """bool is a subclass of int but is never a meaningful HTTP status."""
        error = SimpleNamespace(code=True, type=None, cause_chain=[], message="")
        assert _classify_error_type(error) == "_OTHER"


class TestBuildTokenUsageAttributes:
    """Cover _build_token_usage_attributes."""

    def test_builds_input_token_attributes(self) -> None:
        record = MagicMock()
        record.error = None
        user_config = MagicMock()
        user_config.endpoint.type = "chat"
        user_config.endpoint.model_names = ["gpt-4"]
        user_config.endpoint.urls = ["http://api.openai.com/v1"]
        user_config.gen_ai_provider = None

        attrs = _build_token_usage_attributes(record, user_config, token_type="input")
        assert attrs["gen_ai.token.type"] == "input"
        assert attrs["gen_ai.operation.name"] == "chat"
        assert attrs["gen_ai.request.model"] == "gpt-4"

    def test_builds_output_token_attributes(self) -> None:
        record = MagicMock()
        record.error = None
        user_config = MagicMock()
        user_config.endpoint.type = "completions"
        user_config.endpoint.model_names = ["llama-3"]
        user_config.endpoint.urls = ["http://localhost:8000"]
        user_config.gen_ai_provider = None

        attrs = _build_token_usage_attributes(record, user_config, token_type="output")
        assert attrs["gen_ai.token.type"] == "output"
        assert attrs["gen_ai.operation.name"] == "text_completion"


class TestBuildDurationAttributes:
    """Cover _build_duration_attributes with errors."""

    def test_error_type_attached_when_present(self) -> None:
        record = MagicMock()
        record.error = SimpleNamespace(code=503, type=None, cause_chain=[], message="")
        user_config = MagicMock()
        user_config.endpoint.type = "chat"
        user_config.endpoint.model_names = ["model"]
        user_config.endpoint.urls = ["http://localhost"]
        user_config.gen_ai_provider = None

        attrs = _build_duration_attributes(record, user_config)
        assert attrs["error.type"] == "http_5xx"

    def test_no_error_type_when_no_error(self) -> None:
        record = MagicMock()
        record.error = None
        user_config = MagicMock()
        user_config.endpoint.type = "chat"
        user_config.endpoint.model_names = ["model"]
        user_config.endpoint.urls = ["http://localhost"]
        user_config.gen_ai_provider = None

        attrs = _build_duration_attributes(record, user_config)
        assert "error.type" not in attrs


class TestExtractHost:
    """Cover _extract_host edge cases."""

    def test_empty_string_returns_none(self) -> None:
        assert _extract_host("") is None

    def test_whitespace_returns_none(self) -> None:
        assert _extract_host("   ") is None

    def test_bare_host(self) -> None:
        assert _extract_host("api.openai.com") == "api.openai.com"

    def test_host_with_port(self) -> None:
        assert _extract_host("localhost:4318") == "localhost"

    def test_full_url(self) -> None:
        assert _extract_host("https://api.anthropic.com/v1") == "api.anthropic.com"

    def test_url_with_port(self) -> None:
        assert _extract_host("http://collector:4318/v1/metrics") == "collector"

    def test_uppercase_normalized(self) -> None:
        assert _extract_host("API.OpenAI.COM") == "api.openai.com"


class TestConvertMetricValue:
    """Cover convert_metric_value public helper."""

    def test_ns_to_seconds(self) -> None:
        result = convert_metric_value("request_latency", 1_000_000_000.0)
        assert abs(result - 1.0) < 1e-12

    def test_unknown_metric_identity(self) -> None:
        result = convert_metric_value("unknown_metric", 42.0)
        assert result == 42.0

    def test_token_count_identity(self) -> None:
        result = convert_metric_value("input_token_count", 100.0)
        assert result == 100.0


class TestTranslateTokenUsage:
    """Cover translate() with token usage metrics."""

    def test_input_token_count_translates(self) -> None:
        record = MagicMock()
        record.error = None
        user_config = MagicMock()
        user_config.endpoint.type = "chat"
        user_config.endpoint.model_names = ["model"]
        user_config.endpoint.urls = ["http://localhost"]
        user_config.gen_ai_provider = None

        emission = translate("input_token_count", 50.0, record, user_config=user_config)
        assert emission is not None
        assert emission.spec_metric_name == "gen_ai.client.token.usage"
        assert emission.attributes["gen_ai.token.type"] == "input"
        assert emission.value == 50.0

    def test_output_token_count_translates(self) -> None:
        record = MagicMock()
        record.error = None
        user_config = MagicMock()
        user_config.endpoint.type = "chat"
        user_config.endpoint.model_names = ["model"]
        user_config.endpoint.urls = ["http://localhost"]
        user_config.gen_ai_provider = None

        emission = translate(
            "output_token_count", 200.0, record, user_config=user_config
        )
        assert emission is not None
        assert emission.spec_metric_name == "gen_ai.client.token.usage"
        assert emission.attributes["gen_ai.token.type"] == "output"

    def test_unknown_metric_returns_none(self) -> None:
        record = MagicMock()
        user_config = MagicMock()
        assert (
            translate("some_random_metric", 1.0, record, user_config=user_config)
            is None
        )
