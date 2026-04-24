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
    # ── Bearer token: plain text ──
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
        "AUTHORIZATION: BEARER UPPER_TOKEN",
        ["UPPER_TOKEN"],
        id="bearer-all-upper",
    ),
    param(
        "Authorization:Bearer no-space-after-colon",
        ["no-space-after-colon"],
        id="bearer-no-space-after-colon",
    ),
    # ── Bearer token: JSON-serialized ──
    param(
        '"Authorization":"Bearer sk-secret-json-key"',
        ["sk-secret-json-key"],
        id="bearer-json-serialized",
    ),
    param(
        '"authorization":"bearer json-lower"',
        ["json-lower"],
        id="bearer-json-lowercase",
    ),
    # ── Bearer token: Python repr ──
    param(
        "'Authorization': 'Bearer sk-repr-key'",
        ["sk-repr-key"],
        id="bearer-python-repr",
    ),
    # ── Basic auth ──
    param(
        "Authorization: Basic dXNlcjpwYXNz",
        ["dXNlcjpwYXNz"],
        id="auth-basic-plain",
    ),
    param(
        '"Authorization":"Basic dXNlcjpwYXNzSlNPTg=="',
        ["dXNlcjpwYXNzSlNPTg=="],
        id="auth-basic-json",
    ),
    param(
        "authorization: basic lower-basic-token",
        ["lower-basic-token"],
        id="auth-basic-lowercase",
    ),
    # ── Proxy-Authorization: Bearer ──
    param(
        "Proxy-Authorization: Bearer proxy-secret-key",
        ["proxy-secret-key"],
        id="proxy-auth-bearer-plain",
    ),
    param(
        '"Proxy-Authorization":"Bearer proxy-json-key"',
        ["proxy-json-key"],
        id="proxy-auth-bearer-json",
    ),
    param(
        "'Proxy-Authorization': 'Bearer proxy-repr-key'",
        ["proxy-repr-key"],
        id="proxy-auth-bearer-repr",
    ),
    # ── Proxy-Authorization: Basic ──
    param(
        "proxy-authorization: basic dXNlcjpwYXNz",
        ["dXNlcjpwYXNz"],
        id="proxy-auth-basic-plain",
    ),
    param(
        '"proxy-authorization":"basic proxy-basic-json"',
        ["proxy-basic-json"],
        id="proxy-auth-basic-json",
    ),
    # ── SigV4 Authorization (multi-token) ──
    param(
        "Authorization: AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, SignedHeaders=host;x-amz-date, Signature=fe5f80f77d5fa3beca038a248ff027d0445342fe2855dfe1e8aa344f27c0d3bb",
        [
            "AKIAIOSFODNN7EXAMPLE",
            "fe5f80f77d5fa3beca038a248ff027d0445342fe2855dfe1e8aa344f27c0d3bb",
        ],
        id="sigv4-plain-text",
    ),
    param(
        '"Authorization":"AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, SignedHeaders=host;x-amz-date, Signature=abcdef1234567890"',
        ["AKIAIOSFODNN7EXAMPLE", "abcdef1234567890"],
        id="sigv4-json-serialized",
    ),
    param(
        "'Authorization': 'AWS4-HMAC-SHA256 Credential=ASIAEXAMPLE/20260318/us-west-2/bedrock/aws4_request, Signature=deadbeef'",
        ["ASIAEXAMPLE", "deadbeef"],
        id="sigv4-python-repr",
    ),
    # ── Authorization: opaque token (no scheme keyword) ──
    param(
        "Authorization: nvapi-opaque-token-no-bearer",
        ["nvapi-opaque-token-no-bearer"],
        id="auth-opaque-no-scheme",
    ),
    param(
        '"Authorization":"raw-opaque-token"',
        ["raw-opaque-token"],
        id="auth-opaque-json",
    ),
    # ── x-api-key: plain, JSON, repr ──
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
    param(
        "'x-api-key': 'repr-api-key'",
        ["repr-api-key"],
        id="x-api-key-repr",
    ),
    param(
        "x-api-key: lowercase-key",
        ["lowercase-key"],
        id="x-api-key-lowercase",
    ),
    # ── api-key (Azure OpenAI): plain, JSON ──
    param(
        "api-key: azure-openai-key-123",
        ["azure-openai-key-123"],
        id="api-key-header-plain",
    ),
    param(
        '"api-key":"azure-json-key"',
        ["azure-json-key"],
        id="api-key-header-json",
    ),
    param(
        "API-Key: Mixed-Case-Azure-Key",
        ["Mixed-Case-Azure-Key"],
        id="api-key-header-mixed-case",
    ),
    # ── ocp-apim-subscription-key: plain, JSON ──
    param(
        "Ocp-Apim-Subscription-Key: sub-key-abc",
        ["sub-key-abc"],
        id="azure-apim-plain",
    ),
    param(
        '"ocp-apim-subscription-key":"sub-json-key"',
        ["sub-json-key"],
        id="azure-apim-json",
    ),
    # ── x-goog-api-key: plain, JSON ──
    param(
        "X-Goog-Api-Key: AIzaSy-google-key",
        ["AIzaSy-google-key"],
        id="google-api-key-plain",
    ),
    param(
        '"x-goog-api-key":"google-json-key"',
        ["google-json-key"],
        id="google-api-key-json",
    ),
    # ── x-functions-key: plain, JSON ──
    param(
        "X-Functions-Key: azure-func-key-xyz",
        ["azure-func-key-xyz"],
        id="azure-functions-key-plain",
    ),
    param(
        '"x-functions-key":"func-json-key"',
        ["func-json-key"],
        id="azure-functions-key-json",
    ),
    # ── aeg-sas-key: plain, JSON ──
    param(
        "Aeg-Sas-Key: sas-token-abc123",
        ["sas-token-abc123"],
        id="azure-event-grid-plain",
    ),
    param(
        '"aeg-sas-key":"sas-json-key"',
        ["sas-json-key"],
        id="azure-event-grid-json",
    ),
    # ── x-amz-security-token: plain, JSON ──
    param(
        "X-Amz-Security-Token: FwoGZX-aws-token",
        ["FwoGZX-aws-token"],
        id="aws-security-token-plain",
    ),
    param(
        '"x-amz-security-token":"FwoGZX-aws-json-token"',
        ["FwoGZX-aws-json-token"],
        id="aws-security-token-json",
    ),
    # ── Query-string style key=value ──
    param(
        "api_key=supersecret&other=value",
        ["supersecret"],
        id="api-key-equals",
    ),
    param("api-key=my-secret", ["my-secret"], id="api-hyphen-key-equals"),
    param("api key=space-key", ["space-key"], id="api-space-key-equals"),
    param("token=abc123", ["abc123"], id="token-equals"),
    param("secret=xyzzy", ["xyzzy"], id="secret-equals"),
    param("TOKEN=UPPER_TOKEN", ["UPPER_TOKEN"], id="token-equals-upper"),
    param("SECRET=UPPER_SECRET", ["UPPER_SECRET"], id="secret-equals-upper"),
    param("Api_Key=mixed_case", ["mixed_case"], id="api-key-equals-mixed-case"),
    # ── ZMQ trace messages ──
    param(
        'b\'{"endpoint_headers":{"Authorization":"Bearer sk-zmq-leak-123",'
        '"Content-Type":"application/json"}}\'',
        ["sk-zmq-leak-123"],
        id="zmq-trace-bearer",
    ),
    param(
        'b\'{"endpoint_headers":{"X-API-Key":"nvapi-zmq-key",'
        '"Content-Type":"application/json"}}\'',
        ["nvapi-zmq-key"],
        id="zmq-trace-x-api-key",
    ),
    param(
        'b\'{"endpoint_headers":{"Authorization":"AWS4-HMAC-SHA256 Credential=AKIA123/date/region/svc/aws4_request, Signature=abc123"}}\'',
        ["AKIA123", "abc123"],
        id="zmq-trace-sigv4",
    ),
    # ── Exception-style messages ──
    param(
        "ClientError: 401 Unauthorized, headers={'Authorization': 'Bearer sk-leaked'}",
        ["sk-leaked"],
        id="exception-with-bearer",
    ),
    param(
        "ConnectionError: host=api.openai.com, api-key: sk-conn-err-key",
        ["sk-conn-err-key"],
        id="exception-with-api-key-header",
    ),
    param(
        "aiohttp.ClientResponseError: 403, headers: {api-key: forbidden-key-123}",
        ["forbidden-key-123"],
        id="exception-with-api-key-in-braces",
    ),
    # ── Multiple patterns in one string ──
    param(
        "Authorization: Bearer tok123, api_key=secret456, X-API-Key: key789",
        ["tok123", "secret456", "key789"],
        id="multiple-patterns-mixed",
    ),
    param(
        "Proxy-Authorization: Basic proxy-basic, api-key: azure-key-leak, secret=s3cr3t",
        ["proxy-basic", "azure-key-leak", "s3cr3t"],
        id="multiple-patterns-proxy-plus-headers",
    ),
    param(
        '"Authorization":"Bearer j1","X-API-Key":"j2","api-key":"j3"',
        ["j1", "j2", "j3"],
        id="multiple-patterns-all-json",
    ),
]

_REDACT_STRING_PRESERVE_CASES = [
    param("Content-Type: application/json", id="content-type-unchanged"),
    param("", id="empty-string"),
    param("Normal log message with no secrets", id="plain-text"),
    param("X-Request-ID: abc-123-def", id="x-request-id-unchanged"),
    param("User-Agent: aiperf/1.0", id="user-agent-unchanged"),
    param("Accept: text/event-stream", id="accept-unchanged"),
    param("Cache-Control: no-cache", id="cache-control-unchanged"),
    param(
        "model=gpt-4&temperature=0.7&top_p=0.9",
        id="query-params-no-sensitive-keys",
    ),
    param("X-Custom-Header: keep-me", id="custom-header-unchanged"),
    param("200 OK", id="status-line"),
    param(
        '{"model":"gpt-4","temperature":0.7}',
        id="json-payload-no-secrets",
    ),
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

    def test_sigv4_json_preserves_surrounding_fields(self):
        s = (
            '{"Authorization":"AWS4-HMAC-SHA256 Credential=AKIA/date/region/svc/req, '
            'Signature=abc","Content-Type":"application/json","model":"gpt-4"}'
        )
        result = redact_string(s)
        assert "AKIA" not in result
        assert "abc" not in result
        assert "application/json" in result
        assert "gpt-4" in result

    def test_proxy_auth_json_preserves_other_headers(self):
        s = '"Proxy-Authorization":"Basic secret123","Accept":"text/plain"'
        result = redact_string(s)
        assert "secret123" not in result
        assert "text/plain" in result

    def test_multiple_sensitive_headers_in_json_object(self):
        s = (
            '{"X-API-Key":"nvapi-key1","api-key":"azure-key2",'
            '"Authorization":"Bearer tok3","Content-Type":"application/json"}'
        )
        result = redact_string(s)
        assert "nvapi-key1" not in result
        assert "azure-key2" not in result
        assert "tok3" not in result
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
    # Double-quoted headers: auth schemes
    param(
        'aiperf --header "Authorization:Bearer sk-abc"',
        ["sk-abc"],
        id="dq-bearer",
    ),
    param(
        'aiperf --header "Authorization: Bearer sk-with-space"',
        ["sk-with-space"],
        id="dq-bearer-space",
    ),
    param(
        'aiperf --header "Authorization:Basic dXNlcjpwYXNz"',
        ["dXNlcjpwYXNz"],
        id="dq-basic",
    ),
    param(
        'aiperf --header "Proxy-Authorization:Bearer proxy-tok"',
        ["proxy-tok"],
        id="dq-proxy-auth-bearer",
    ),
    param(
        'aiperf --header "proxy-authorization:basic proxy-basic-tok"',
        ["proxy-basic-tok"],
        id="dq-proxy-auth-basic-lower",
    ),
    # Double-quoted headers: -H shorthand
    param(
        'aiperf -H "Authorization:Bearer sk-H-dq"',
        ["sk-H-dq"],
        id="dq-H-bearer",
    ),
    param(
        'aiperf -H "X-API-Key:nvapi-secret"',
        ["nvapi-secret"],
        id="dq-H-x-api-key",
    ),
    # Double-quoted headers: all cloud provider headers
    param(
        'aiperf --header "API-Key:azure-dq-key"',
        ["azure-dq-key"],
        id="dq-api-key",
    ),
    param(
        'aiperf --header "Ocp-Apim-Subscription-Key:azure-dq-sub"',
        ["azure-dq-sub"],
        id="dq-azure-apim",
    ),
    param(
        'aiperf --header "X-Goog-Api-Key:google-dq-key"',
        ["google-dq-key"],
        id="dq-google-api-key",
    ),
    param(
        'aiperf --header "X-Functions-Key:azure-dq-func"',
        ["azure-dq-func"],
        id="dq-azure-functions",
    ),
    param(
        'aiperf --header "Aeg-Sas-Key:azure-dq-sas"',
        ["azure-dq-sas"],
        id="dq-azure-event-grid",
    ),
    param(
        'aiperf --header "X-Amz-Security-Token:aws-dq-token"',
        ["aws-dq-token"],
        id="dq-aws-security-token",
    ),
    # Double-quoted headers: case variations
    param(
        'aiperf --header "authorization:bearer sk-dq-lower"',
        ["sk-dq-lower"],
        id="dq-bearer-all-lower",
    ),
    param(
        'aiperf --header "AUTHORIZATION:BEARER SK-DQ-UPPER"',
        ["SK-DQ-UPPER"],
        id="dq-bearer-all-upper",
    ),
    param(
        'aiperf --header "x-api-key:dq-lower-x-api"',
        ["dq-lower-x-api"],
        id="dq-x-api-key-lower",
    ),
    # Mixed single and double quotes in one command
    param(
        """aiperf --header 'Authorization:Bearer sq-tok' --header "X-API-Key:dq-key" """,
        ["sq-tok", "dq-key"],
        id="mixed-single-double-quotes",
    ),
    param(
        """aiperf --api-key 'sk-1' -H "Authorization:Bearer dq-tok2" --header 'X-API-Key:sq-key3'""",
        ["sk-1", "dq-tok2", "sq-key3"],
        id="api-key-plus-mixed-quotes",
    ),
    # Cloud provider headers (single-quoted)
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
    # Double-quoted non-sensitive headers preserved
    param(
        'aiperf --header "Content-Type:application/json"',
        ["Content-Type:application/json"],
        id="dq-header-content-type",
    ),
    param(
        'aiperf --header "Accept:text/event-stream"',
        ["Accept:text/event-stream"],
        id="dq-header-accept",
    ),
    param(
        'aiperf -H "X-Custom:my-value"',
        ["X-Custom:my-value"],
        id="dq-header-custom",
    ),
    param(
        'aiperf --header "X-Request-ID:req-dq-001"',
        ["X-Request-ID:req-dq-001"],
        id="dq-header-request-id",
    ),
    # Double-quoted look-alikes preserved
    param(
        'aiperf --header "X-Authorization:Bearer tok"',
        ["Bearer tok"],
        id="dq-x-authorization-not-sensitive",
    ),
    param(
        'aiperf --header "X-API-Version:2024-01"',
        ["X-API-Version:2024-01"],
        id="dq-x-api-version-not-sensitive",
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
    # Double-quoted interleaved
    param(
        'aiperf --header "Authorization:Bearer dq-s1" --header "X-Custom:dq-k1"',
        ["dq-s1"],
        ["dq-k1"],
        id="dq-sensitive-then-non-sensitive",
    ),
    param(
        'aiperf --header "X-Custom:dq-k1" --header "Authorization:Bearer dq-s1"',
        ["dq-s1"],
        ["dq-k1"],
        id="dq-non-sensitive-then-sensitive",
    ),
    param(
        'aiperf -H "Authorization:Bearer dq1" -H "X-API-Key:dq2" -H "API-Key:dq3"',
        ["dq1", "dq2", "dq3"],
        [],
        id="dq-three-sensitive-back-to-back",
    ),
    param(
        'aiperf --header "Accept:dq-keep" --header "Authorization:Bearer dq-hide" --header "X-Trace:dq-keep2"',
        ["dq-hide"],
        ["dq-keep", "dq-keep2"],
        id="dq-sensitive-sandwiched",
    ),
    # Mixed single-quoted and double-quoted interleaved
    param(
        """aiperf --header 'Authorization:Bearer sq-tok' --header "X-Custom:dq-keep" --header "X-API-Key:dq-secret" """,
        ["sq-tok", "dq-secret"],
        ["dq-keep"],
        id="mixed-sq-dq-interleaved",
    ),
    param(
        """aiperf -H "Authorization:Bearer dq1" -H 'X-Custom:sq-keep' -H "X-Goog-Api-Key:dq2" -H 'Content-Type:sq-ct'""",
        ["dq1", "dq2"],
        ["sq-keep", "sq-ct"],
        id="mixed-alternating-sq-dq",
    ),
    # All sensitive header types in double-quoted form
    param(
        (
            'aiperf -H "Authorization:Bearer dq-t1" '
            '-H "X-API-Key:dq-t2" '
            '-H "API-Key:dq-t3" '
            '-H "Proxy-Authorization:Bearer dq-t4" '
            '-H "Ocp-Apim-Subscription-Key:dq-t5" '
            '-H "X-Goog-Api-Key:dq-t6" '
            '-H "X-Functions-Key:dq-t7" '
            '-H "Aeg-Sas-Key:dq-t8" '
            '-H "X-Amz-Security-Token:dq-t9"'
        ),
        [
            "dq-t1",
            "dq-t2",
            "dq-t3",
            "dq-t4",
            "dq-t5",
            "dq-t6",
            "dq-t7",
            "dq-t8",
            "dq-t9",
        ],
        [],
        id="dq-all-nine-sensitive-header-types",
    ),
    # Full realistic command with double-quoted headers
    param(
        (
            'aiperf profile --model "gpt-4" --url "http://localhost:8000" '
            '--api-key "sk-dq-real" --header "Authorization:Bearer sk-dq-real" '
            '--header "X-Custom:my-trace" --extra-inputs "temperature:0.7" '
            "--endpoint-type chat --streaming --concurrency 5"
        ),
        ["sk-dq-real"],
        [
            "gpt-4",
            "http://localhost:8000",
            "my-trace",
            "temperature:0.7",
        ],
        id="full-realistic-double-quoted",
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
            param(
                "401 Unauthorized: Authorization: Basic dXNlcjpwYXNz",
                "dXNlcjpwYXNz",
                id="basic-auth-in-error",
            ),
            param(
                "Proxy-Authorization: Bearer proxy-err-tok",
                "proxy-err-tok",
                id="proxy-auth-in-error",
            ),
            param(
                "SigV4 failure: Authorization: AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/date/region/svc/req, Signature=deadbeef",
                "AKIAEXAMPLE",
                id="sigv4-credential-in-error",
            ),
            param(
                "Azure error: api-key: azure-err-key",
                "azure-err-key",
                id="api-key-header-in-error",
            ),
            param(
                "Google error: X-Goog-Api-Key: goog-err-key",
                "goog-err-key",
                id="google-api-key-in-error",
            ),
            param(
                "AWS error: X-Amz-Security-Token: FwoGZX-err-token",
                "FwoGZX-err-token",
                id="aws-token-in-error",
            ),
            param(
                "secret=leaked-in-traceback&debug=true",
                "leaked-in-traceback",
                id="secret-query-param-in-error",
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
