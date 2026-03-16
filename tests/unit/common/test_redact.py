# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for centralized API key / credential redaction."""

from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from pytest import param

from aiperf.common.config import EndpointConfig
from aiperf.common.config.input_config import InputConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.models import AioHttpTraceData
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.models.model_endpoint_info import EndpointInfo
from aiperf.common.redact import (
    _SENSITIVE_HEADER_NAMES,
    REDACTED_VALUE,
    redact_cli_command,
    redact_headers,
    redact_string,
)
from aiperf.transports.aiohttp_trace import create_aiohttp_trace_config

# =============================================================================
# redact_headers
# =============================================================================


class TestRedactHeaders:
    """Tests for redact_headers()."""

    def test_none_returns_none(self):
        assert redact_headers(None) is None

    def test_empty_dict_returns_empty_dict(self):
        assert redact_headers({}) == {}

    def test_returns_new_dict(self):
        """Redaction must not mutate the original headers dict."""
        original = {"Authorization": "Bearer secret", "Accept": "text/plain"}
        result = redact_headers(original)
        assert original["Authorization"] == "Bearer secret"
        assert result["Authorization"] == REDACTED_VALUE

    @pytest.mark.parametrize(
        "header_name",
        sorted(_SENSITIVE_HEADER_NAMES),
        ids=sorted(_SENSITIVE_HEADER_NAMES),
    )
    def test_sensitive_header_redacted(self, header_name):
        """Every header name in _SENSITIVE_HEADER_NAMES is redacted."""
        result = redact_headers({header_name: "some-secret-value"})
        assert result[header_name] == REDACTED_VALUE

    @pytest.mark.parametrize(
        "header_name, value",
        [
            param("authorization", "Bearer token", id="authorization-lower"),
            param("AUTHORIZATION", "Bearer token2", id="AUTHORIZATION-upper"),
            param("x-api-key", "key1", id="x-api-key-lower"),
            param("X-Api-Key", "key2", id="X-Api-Key-mixed"),
        ],
    )
    def test_case_insensitive_matching(self, header_name, value):
        result = redact_headers({header_name: value})
        assert result[header_name] == REDACTED_VALUE

    @pytest.mark.parametrize(
        "header_name, value",
        [
            param("Content-Type", "application/json", id="content-type"),
            param("Accept", "text/event-stream", id="accept"),
            param("X-Request-ID", "abc-123", id="x-request-id"),
            param("User-Agent", "aiperf/1.0", id="user-agent"),
        ],
    )
    def test_non_sensitive_headers_unchanged(self, header_name, value):
        result = redact_headers({header_name: value})
        assert result[header_name] == value

    def test_mixed_sensitive_and_non_sensitive(self):
        headers = {
            "Authorization": "Bearer sk-1234",
            "X-API-Key": "nvapi-abc",
            "Content-Type": "application/json",
            "X-Request-ID": "req-001",
            "User-Agent": "aiperf/1.0",
        }
        result = redact_headers(headers)
        assert result["Authorization"] == REDACTED_VALUE
        assert result["X-API-Key"] == REDACTED_VALUE
        assert result["Content-Type"] == "application/json"
        assert result["X-Request-ID"] == "req-001"
        assert result["User-Agent"] == "aiperf/1.0"


# =============================================================================
# redact_string
# =============================================================================

_REDACT_STRING_CASES = [
    # Bearer token patterns
    param(
        "Authorization: Bearer sk-secret-key",
        ["sk-secret-key"],
        id="bearer-plain-text",
    ),
    param(
        "authorization: bearer MY_TOKEN",
        ["MY_TOKEN"],
        id="bearer-case-insensitive",
    ),
    param(
        '"Authorization":"Bearer sk-secret-json-key"',
        ["sk-secret-json-key"],
        id="bearer-json-serialized",
    ),
    # Query-string style key=value
    param(
        "api_key=supersecret&other=value",
        ["supersecret"],
        id="api-key-equals",
    ),
    param("api-key=my-secret", ["my-secret"], id="api-hyphen-key-equals"),
    param("token=abc123", ["abc123"], id="token-equals"),
    param("secret=xyzzy", ["xyzzy"], id="secret-equals"),
    # X-API-Key header
    param(
        "X-API-Key: nvapi-my-secret-key",
        ["nvapi-my-secret-key"],
        id="x-api-key-plain",
    ),
    param(
        '"X-API-Key":"nvapi-json-secret"',
        ["nvapi-json-secret"],
        id="x-api-key-json",
    ),
    # ZMQ trace message
    param(
        'b\'{"endpoint_headers":{"Authorization":"Bearer sk-zmq-leak-123",'
        '"Content-Type":"application/json"}}\'',
        ["sk-zmq-leak-123"],
        id="zmq-trace-message",
    ),
    # Multiple patterns in one string
    param(
        "Authorization: Bearer tok123, api_key=secret456, X-API-Key: key789",
        ["tok123", "secret456", "key789"],
        id="multiple-patterns",
    ),
]

_REDACT_STRING_PRESERVE_CASES = [
    param("Content-Type: application/json", id="content-type-unchanged"),
    param("", id="empty-string"),
    param("Normal log message with no secrets", id="plain-text"),
]


class TestRedactString:
    """Tests for redact_string()."""

    @pytest.mark.parametrize("input_str, secrets", _REDACT_STRING_CASES)
    def test_secret_redacted(self, input_str, secrets):
        result = redact_string(input_str)
        for secret in secrets:
            assert secret not in result, f"Secret {secret!r} leaked in: {result}"
        assert REDACTED_VALUE in result

    @pytest.mark.parametrize("input_str", _REDACT_STRING_PRESERVE_CASES)
    def test_non_sensitive_unchanged(self, input_str):
        assert redact_string(input_str) == input_str

    def test_api_key_equals_preserves_other_params(self):
        result = redact_string("api_key=supersecret&other=value")
        assert "other=value" in result

    def test_zmq_trace_preserves_non_sensitive_headers(self):
        s = (
            'b\'{"endpoint_headers":{"Authorization":"Bearer sk-zmq-leak-123",'
            '"Content-Type":"application/json"}}\''
        )
        result = redact_string(s)
        assert "application/json" in result


# =============================================================================
# redact_cli_command
# =============================================================================

_MUST_REDACT_CASES = [
    # --api-key forms
    param("aiperf --api-key 'sk-12345'", ["sk-12345"], id="api-key-quoted"),
    param("aiperf --api-key sk-12345", ["sk-12345"], id="api-key-unquoted"),
    param("aiperf --api-key='sk-12345'", ["sk-12345"], id="api-key-equals-quoted"),
    param("aiperf --api-key=sk-12345", ["sk-12345"], id="api-key-equals-unquoted"),
    param(
        "aiperf --api-key 'sk-proj-abc_123-XYZ.456'",
        ["sk-proj-abc_123-XYZ.456"],
        id="api-key-special-chars",
    ),
    # Quoted sensitive headers
    param(
        "aiperf --header 'Authorization:Bearer sk-abc'",
        ["sk-abc"],
        id="header-bearer-colon",
    ),
    param(
        "aiperf --header 'Authorization: Bearer sk-abc'",
        ["sk-abc"],
        id="header-bearer-colon-space",
    ),
    param(
        "aiperf --header 'Authorization Bearer sk-abc'",
        ["sk-abc"],
        id="header-bearer-space",
    ),
    param(
        "aiperf --header 'Authorization:Basic dXNlcjpwYXNz'",
        ["dXNlcjpwYXNz"],
        id="header-basic-auth",
    ),
    param(
        "aiperf --header 'X-API-Key:nvapi-secret'",
        ["nvapi-secret"],
        id="header-x-api-key",
    ),
    param(
        "aiperf --header 'X-API-Key: nvapi-secret'",
        ["nvapi-secret"],
        id="header-x-api-key-space",
    ),
    param(
        "aiperf --header 'API-Key:my-secret'", ["my-secret"], id="header-api-key-no-x"
    ),
    param(
        "aiperf --header 'Proxy-Authorization:Bearer proxy-tok'",
        ["proxy-tok"],
        id="header-proxy-auth",
    ),
    param("aiperf -H 'Authorization:Bearer sk-abc'", ["sk-abc"], id="H-shorthand"),
    # Case variations
    param(
        "aiperf --header 'AUTHORIZATION:Bearer sk-abc'",
        ["sk-abc"],
        id="header-uppercase",
    ),
    param(
        "aiperf --header 'authorization:Bearer sk-abc'",
        ["sk-abc"],
        id="header-lowercase",
    ),
    param(
        "aiperf --header 'x-api-key:nvapi-secret'",
        ["nvapi-secret"],
        id="header-x-api-key-lower",
    ),
    param(
        "aiperf --header 'api-key:my-secret'", ["my-secret"], id="header-api-key-lower"
    ),
    param(
        "aiperf --header 'proxy-authorization:Bearer tok'",
        ["tok"],
        id="header-proxy-auth-lower",
    ),
    # Unquoted forms
    param(
        "aiperf --header Authorization:Bearer sk-abc",
        ["sk-abc"],
        id="header-unquoted-bearer",
    ),
    param(
        "aiperf -H X-API-Key:nvapi-secret", ["nvapi-secret"], id="H-unquoted-x-api-key"
    ),
    param(
        "aiperf --header API-Key:my-secret", ["my-secret"], id="header-unquoted-api-key"
    ),
    param(
        "aiperf --header Proxy-Authorization:Bearer tok",
        ["tok"],
        id="header-unquoted-proxy-auth",
    ),
    param(
        "aiperf --header Authorization:Bearer sk-abc --url http://host",
        ["sk-abc"],
        id="header-unquoted-bearer-trailing-flag",
    ),
    # Edge cases
    param(
        "aiperf --header 'Authorization:Bearer sk-abc=123=456'",
        ["sk-abc=123=456"],
        id="header-bearer-with-equals",
    ),
    param(
        "aiperf --header 'Authorization:Bearer http://token-server/abc'",
        ["http://token-server/abc"],
        id="header-bearer-url-like-value",
    ),
    param(
        "aiperf --api-key 'sk-1' --header 'Authorization:Bearer sk-2' -H 'X-API-Key:nvapi-3'",
        ["sk-1", "sk-2", "nvapi-3"],
        id="multiple-secrets",
    ),
    # Cloud provider headers
    param(
        "aiperf --header 'Ocp-Apim-Subscription-Key:abc-sub-key-123'",
        ["abc-sub-key-123"],
        id="header-azure-apim-subscription-key",
    ),
    param(
        "aiperf --header 'X-Goog-Api-Key:AIzaSy-google-key'",
        ["AIzaSy-google-key"],
        id="header-google-api-key",
    ),
    param(
        "aiperf --header 'X-Functions-Key:azure-func-key-xyz'",
        ["azure-func-key-xyz"],
        id="header-azure-functions-key",
    ),
    param(
        "aiperf --header 'Aeg-Sas-Key:sas-token-abc123'",
        ["sas-token-abc123"],
        id="header-azure-event-grid-sas",
    ),
    param(
        "aiperf --header 'X-Amz-Security-Token:FwoGZX-aws-temp-token'",
        ["FwoGZX-aws-temp-token"],
        id="header-aws-security-token",
    ),
    param(
        "aiperf --header 'ocp-apim-subscription-key:lowercase-key'",
        ["lowercase-key"],
        id="header-azure-apim-lowercase",
    ),
]


class TestRedactCliCommandSecrets:
    """Verify secrets are redacted from CLI command strings."""

    @pytest.mark.parametrize("cmd, secrets", _MUST_REDACT_CASES)
    def test_secret_redacted(self, cmd, secrets):
        result = redact_cli_command(cmd)
        for secret in secrets:
            assert secret not in result, f"Secret {secret!r} leaked in: {result}"
        assert REDACTED_VALUE in result


_MUST_KEEP_CASES = [
    # Normal flags and values
    param("aiperf --model 'gpt-4'", ["gpt-4"], id="model-name"),
    param("aiperf --url 'http://localhost:8000'", ["http://localhost:8000"], id="url"),
    param("aiperf --endpoint-type 'chat'", ["chat"], id="endpoint-type"),
    param("aiperf --concurrency 10", ["10"], id="concurrency"),
    param("aiperf --streaming", ["--streaming"], id="boolean-flag"),
    # Non-sensitive headers
    param(
        "aiperf --header 'Content-Type:application/json'",
        ["Content-Type:application/json"],
        id="header-content-type",
    ),
    param(
        "aiperf --header 'Accept:text/event-stream'",
        ["Accept:text/event-stream"],
        id="header-accept",
    ),
    param(
        "aiperf --header 'X-Custom-Tracking:trace-abc-123'",
        ["X-Custom-Tracking:trace-abc-123"],
        id="header-custom",
    ),
    param(
        "aiperf --header 'X-Request-ID:req-001'",
        ["X-Request-ID:req-001"],
        id="header-request-id",
    ),
    param(
        "aiperf -H 'User-Agent:aiperf/1.0'",
        ["User-Agent:aiperf/1.0"],
        id="header-user-agent",
    ),
    param(
        "aiperf --header 'Cache-Control:no-cache'",
        ["Cache-Control:no-cache"],
        id="header-cache-control",
    ),
    # Headers that look similar but aren't in _SENSITIVE_HEADER_NAMES
    param(
        "aiperf --header 'X-Authorization:Bearer tok'",
        ["Bearer tok"],
        id="x-authorization-not-sensitive",
    ),
    param(
        "aiperf --header 'Auth-Token:my-token'",
        ["my-token"],
        id="auth-token-not-sensitive",
    ),
    param(
        "aiperf --header 'X-API-Version:2024-01'",
        ["X-API-Version:2024-01"],
        id="x-api-version-not-sensitive",
    ),
    # Partial matches in non-header contexts
    param(
        "aiperf --model 'authorization-test-model'",
        ["authorization-test-model"],
        id="model-with-auth-in-name",
    ),
    param(
        "aiperf --url 'http://host/api-key-manager/v1'",
        ["api-key-manager"],
        id="url-with-api-key-in-path",
    ),
    param(
        "aiperf --custom-endpoint '/v1/authorization/check'",
        ["/v1/authorization/check"],
        id="endpoint-with-auth",
    ),
    param(
        "aiperf --extra-inputs 'token_count:100'",
        ["token_count:100"],
        id="extra-input-with-token-word",
    ),
]


class TestRedactCliCommandPreservesNonSecrets:
    """Verify non-secret values are NOT redacted (no over-redaction)."""

    @pytest.mark.parametrize("cmd, must_keep", _MUST_KEEP_CASES)
    def test_value_preserved(self, cmd, must_keep):
        result = redact_cli_command(cmd)
        for value in must_keep:
            assert value in result, f"Value {value!r} was over-redacted in: {result}"


_INTERLEAVED_CASES = [
    param(
        "aiperf --header 'Authorization:Bearer sk-abc' --header 'X-Custom:keep-me'",
        ["sk-abc"],
        ["X-Custom:keep-me"],
        id="sensitive-then-non-sensitive",
    ),
    param(
        "aiperf --header 'X-Custom:keep-me' --header 'Authorization:Bearer sk-abc'",
        ["sk-abc"],
        ["X-Custom:keep-me"],
        id="non-sensitive-then-sensitive",
    ),
    param(
        "aiperf -H 'Authorization:Bearer sk-abc' -H 'X-API-Key:nvapi-secret'",
        ["sk-abc", "nvapi-secret"],
        [],
        id="two-sensitive-back-to-back",
    ),
    param(
        "aiperf --header 'Accept:text/json' --header 'Authorization:Bearer sk-abc' --header 'X-Trace:trace-123'",
        ["sk-abc"],
        ["text/json", "trace-123"],
        id="sensitive-sandwiched",
    ),
    param(
        "aiperf --api-key 'sk-secret' --header 'X-Custom:keep-me'",
        ["sk-secret"],
        ["X-Custom:keep-me"],
        id="api-key-then-non-sensitive-header",
    ),
    param(
        "aiperf --header 'X-Custom:keep-me' --api-key 'sk-secret'",
        ["sk-secret"],
        ["X-Custom:keep-me"],
        id="non-sensitive-header-then-api-key",
    ),
    param(
        "aiperf --header 'Content-Type:application/json' --api-key 'sk-secret' --header 'Accept:text/plain'",
        ["sk-secret"],
        ["application/json", "text/plain"],
        id="api-key-between-non-sensitive",
    ),
    param(
        "aiperf --api-key 'sk-secret' --header 'Authorization:Bearer sk-other' --header 'X-Trace:trace-456'",
        ["sk-secret", "sk-other"],
        ["trace-456"],
        id="api-key-and-auth-then-non-sensitive",
    ),
    param(
        "aiperf -H 'Authorization:Bearer t1' -H 'X-API-Key:t2' -H 'API-Key:t3' -H 'Proxy-Authorization:Bearer t4'",
        ["t1", "t2", "t3", "t4"],
        [],
        id="all-four-sensitive-header-types",
    ),
    param(
        "aiperf -H 'Accept:k1' -H 'Authorization:Bearer s1' -H 'X-Trace:k2' -H 'X-API-Key:s2' -H 'Content-Type:k3'",
        ["s1", "s2"],
        ["k1", "k2", "k3"],
        id="interleaved-sensitive-and-non-sensitive",
    ),
    param(
        "aiperf --api-key 'sk-secret' --extra-inputs 'temperature:0.7' --extra-inputs 'top_p:0.9'",
        ["sk-secret"],
        ["temperature:0.7", "top_p:0.9"],
        id="api-key-adjacent-to-extra-inputs",
    ),
    param(
        "aiperf --api-key 'sk-secret' --model 'gpt-4' --header 'Authorization:Bearer sk-other'",
        ["sk-secret", "sk-other"],
        ["gpt-4"],
        id="model-sandwiched-between-secrets",
    ),
    param(
        (
            "aiperf 'profile' --model 'gpt-4' --url 'http://localhost:8000' "
            "--api-key 'sk-real-key' --header 'Authorization:Bearer sk-real-key' "
            "--header 'X-Custom:my-trace' --extra-inputs 'temperature:0.7' "
            "--endpoint-type 'chat' --streaming --concurrency 5"
        ),
        ["sk-real-key"],
        [
            "gpt-4",
            "http://localhost:8000",
            "my-trace",
            "temperature:0.7",
            "chat",
            "--streaming",
        ],
        id="full-realistic-command",
    ),
]


class TestRedactCliCommandInterleaved:
    """Verify correct behavior when sensitive and non-sensitive args are adjacent."""

    @pytest.mark.parametrize("cmd, secrets, must_keep", _INTERLEAVED_CASES)
    def test_interleaved(self, cmd, secrets, must_keep):
        result = redact_cli_command(cmd)
        for secret in secrets:
            assert secret not in result, f"Secret {secret!r} leaked in: {result}"
        for value in must_keep:
            assert value in result, f"Value {value!r} over-redacted in: {result}"


# =============================================================================
# EndpointConfig api_key protection
# =============================================================================


class TestEndpointConfigApiKeyProtected:
    """Verify api_key is hidden from repr and redacted in serialization."""

    def test_api_key_not_in_repr(self):
        config = EndpointConfig(model_names=["gpt2"], api_key="sk-secret")
        assert "sk-secret" not in repr(config)

    def test_api_key_still_accessible_as_attribute(self):
        config = EndpointConfig(model_names=["gpt2"], api_key="sk-secret")
        assert config.api_key == "sk-secret"

    def test_api_key_redacted_in_model_dump(self):
        config = EndpointConfig(model_names=["gpt2"], api_key="sk-secret")
        assert config.model_dump()["api_key"] == REDACTED_VALUE

    def test_api_key_redacted_in_json(self):
        config = EndpointConfig(model_names=["gpt2"], api_key="sk-secret")
        json_str = config.model_dump_json()
        assert "sk-secret" not in json_str
        assert REDACTED_VALUE in json_str

    def test_api_key_preserved_with_include_secrets_context(self):
        config = EndpointConfig(model_names=["gpt2"], api_key="sk-secret")
        dumped = config.model_dump(context={"include_secrets": True})
        assert dumped["api_key"] == "sk-secret"

    def test_api_key_none_not_redacted(self):
        config = EndpointConfig(model_names=["gpt2"])
        assert config.model_dump()["api_key"] is None


# =============================================================================
# EndpointInfo api_key protection
# =============================================================================


class TestEndpointInfoApiKeyExcluded:
    """Verify api_key is excluded from serialization on EndpointInfo."""

    def test_api_key_excluded_from_model_dump(self):
        info = EndpointInfo(api_key="nvapi-secret")
        assert "api_key" not in info.model_dump()

    def test_api_key_excluded_from_json(self):
        info = EndpointInfo(api_key="nvapi-secret")
        assert "nvapi-secret" not in info.model_dump_json()

    def test_api_key_not_in_repr(self):
        info = EndpointInfo(api_key="nvapi-secret")
        assert "nvapi-secret" not in repr(info)

    def test_api_key_still_accessible(self):
        info = EndpointInfo(api_key="nvapi-secret")
        assert info.api_key == "nvapi-secret"


# =============================================================================
# InputConfig headers redaction
# =============================================================================


class TestInputConfigHeadersRedaction:
    """Verify sensitive headers passed via --header are redacted in serialization."""

    @pytest.mark.parametrize(
        "headers, expected",
        [
            param(
                [
                    ("Authorization", "Bearer sk-secret-123"),
                    ("Content-Type", "application/json"),
                ],
                [
                    ("Authorization", REDACTED_VALUE),
                    ("Content-Type", "application/json"),
                ],
                id="authorization-redacted-content-type-kept",
            ),
            param(
                [("X-API-Key", "nvapi-my-secret")],
                [("X-API-Key", REDACTED_VALUE)],
                id="x-api-key-redacted",
            ),
            param(
                [("X-Custom-Header", "my-value"), ("Accept", "text/event-stream")],
                [("X-Custom-Header", "my-value"), ("Accept", "text/event-stream")],
                id="non-sensitive-unchanged",
            ),
        ],
    )
    def test_headers_redacted_in_dump(self, headers, expected):
        config = InputConfig(headers=headers)
        assert config.model_dump()["headers"] == expected

    def test_headers_preserved_with_include_secrets_context(self):
        config = InputConfig(headers=[("Authorization", "Bearer sk-secret")])
        dumped = config.model_dump(context={"include_secrets": True})
        assert dumped["headers"] == [("Authorization", "Bearer sk-secret")]

    def test_headers_still_accessible_as_attribute(self):
        config = InputConfig(headers=[("Authorization", "Bearer sk-secret")])
        assert config.headers == [("Authorization", "Bearer sk-secret")]


# =============================================================================
# CLI command redaction (via UserConfig)
# =============================================================================


class TestCliCommandRedaction:
    """Verify --api-key and sensitive --header values are redacted in cli_command."""

    def test_api_key_redacted_in_cli_command(self):
        with patch(
            "sys.argv",
            [
                "aiperf",
                "profile",
                "--model",
                "gpt2",
                "--api-key",
                "sk-12345",
                "--url",
                "http://localhost:8000",
            ],
        ):
            config = UserConfig(endpoint={"model_names": ["gpt2"]}, cli_command=None)
            assert "sk-12345" not in config.cli_command
            assert f"--api-key '{REDACTED_VALUE}'" in config.cli_command

    @pytest.mark.parametrize(
        "flag, header_value, secret",
        [
            param(
                "--header",
                "Authorization:Bearer sk-abc123",
                "sk-abc123",
                id="header-authorization",
            ),
            param("-H", "X-API-Key:nvapi-secret", "nvapi-secret", id="H-x-api-key"),
            param(
                "--header",
                "Ocp-Apim-Subscription-Key:azure-sub-key",
                "azure-sub-key",
                id="header-azure-apim",
            ),
        ],
    )
    def test_sensitive_header_redacted_in_cli_command(self, flag, header_value, secret):
        with patch(
            "sys.argv", ["aiperf", "profile", "--model", "gpt2", flag, header_value]
        ):
            config = UserConfig(endpoint={"model_names": ["gpt2"]}, cli_command=None)
            assert secret not in config.cli_command

    def test_non_sensitive_args_preserved_in_cli_command(self):
        with patch(
            "sys.argv",
            ["aiperf", "profile", "--model", "gpt2", "--url", "http://localhost:8000"],
        ):
            config = UserConfig(endpoint={"model_names": ["gpt2"]}, cli_command=None)
            assert "http://localhost:8000" in config.cli_command
            assert "gpt2" in config.cli_command


# =============================================================================
# ErrorDetails safe repr
# =============================================================================


class TestErrorDetailsSafeRepr:
    """Verify ErrorDetails._safe_repr uses centralized redaction."""

    @pytest.mark.parametrize(
        "message, secret",
        [
            param(
                "Failed with Authorization: Bearer sk-12345",
                "sk-12345",
                id="bearer-token",
            ),
            param(
                "Connection failed: api_key=supersecret",
                "supersecret",
                id="api-key-equals",
            ),
            param(
                "Headers: X-API-Key: my-key-value",
                "my-key-value",
                id="x-api-key-header",
            ),
        ],
    )
    def test_secret_redacted_in_exception(self, message, secret):
        exc = Exception(message)
        assert secret not in ErrorDetails.from_exception(exc).message


# =============================================================================
# aiohttp trace header redaction
# =============================================================================


class TestAioHttpTraceRedaction:
    """Verify that aiohttp trace captures redacted headers."""

    @pytest.mark.asyncio
    async def test_request_headers_redacted_in_trace(self):
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        callbacks = trace_config.on_request_headers_sent
        assert len(callbacks) == 1

        session = MagicMock(spec=aiohttp.ClientSession)
        params = MagicMock(spec=aiohttp.TraceRequestHeadersSentParams)
        params.headers = {
            "Authorization": "Bearer sk-secret-token-123",
            "Content-Type": "application/json",
            "X-API-Key": "nvapi-my-key",
        }

        await callbacks[0](session, MagicMock(), params)

        assert trace_data.request_headers is not None
        assert trace_data.request_headers["Authorization"] == REDACTED_VALUE
        assert trace_data.request_headers["X-API-Key"] == REDACTED_VALUE
        assert trace_data.request_headers["Content-Type"] == "application/json"


# =============================================================================
# Log filter redaction
# =============================================================================
