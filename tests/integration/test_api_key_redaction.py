# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for API key redaction with real HTTP transport.

These tests run a full benchmark against a real mock server over HTTP
with --extra-verbose (TRACE level logging) enabled, verifying that API
keys are redacted in all exported artifacts AND log files, including
HTTP trace data, while the server still receives the real credentials.

Using --extra-verbose is critical because TRACE-level logging captures
formatted payloads, request headers, and ZMQ messages — all potential
leak vectors.
"""

import platform

import orjson
import pytest

from aiperf.common.redact import REDACTED_VALUE
from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults

API_KEY = "sk-integration-secret-REDACT-12345"

# Common CLI flags used by all tests: extra-verbose enables TRACE logging
_COMMON_FLAGS = f"""\
    --endpoint-type chat \
    --streaming \
    --request-count 5 \
    --concurrency {defaults.concurrency} \
    --workers-max {defaults.workers_max} \
    --extra-verbose"""


def _assert_api_key_not_in_logs(temp_output_dir):
    """Scan all log files in the artifact directory for the API key."""
    for log_file in temp_output_dir.rglob("**/*.log"):
        content = log_file.read_text(errors="replace")
        assert API_KEY not in content, (
            f"API key leaked into log file: {log_file.relative_to(temp_output_dir)}"
        )


def _assert_api_key_not_in_any_artifact(temp_output_dir):
    """Scan every text file in the artifact directory for the API key."""
    text_extensions = {".json", ".jsonl", ".csv", ".log", ".yaml", ".yml", ".txt"}
    for file_path in temp_output_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix in text_extensions:
            content = file_path.read_text(errors="replace")
            assert API_KEY not in content, (
                f"API key leaked into artifact: {file_path.relative_to(temp_output_dir)}"
            )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Integration tests are flaky on macOS in Github Actions.",
)
class TestApiKeyRedactionRawExportHTTP:
    """Verify API keys are redacted in --export-level raw output over real HTTP
    with TRACE-level logging enabled."""

    async def test_api_key_redacted_in_raw_records_http(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer, temp_output_dir
    ):
        """Raw records from real HTTP transport must not contain the API key."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                {_COMMON_FLAGS} \
                --api-key {API_KEY} \
                --export-level raw
            """
        )

        assert result.raw_records is not None
        assert len(result.raw_records) == 5

        for record in result.raw_records:
            assert record.request_headers is not None
            headers_str = orjson.dumps(record.request_headers).decode()
            assert API_KEY not in headers_str, (
                f"API key leaked into raw record headers: {record.request_headers}"
            )
            assert record.request_headers.get("Authorization") == REDACTED_VALUE

        _assert_api_key_not_in_logs(temp_output_dir)

    async def test_raw_file_and_logs_do_not_contain_api_key_http(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer, temp_output_dir
    ):
        """Scan the entire raw JSONL file and all logs for the API key."""
        await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                {_COMMON_FLAGS} \
                --api-key {API_KEY} \
                --export-level raw
            """
        )

        raw_file = next(temp_output_dir.glob("**/*profile_export_raw.jsonl"), None)
        assert raw_file is not None
        assert API_KEY not in raw_file.read_text(), (
            "Real API key found in raw records JSONL file"
        )

        _assert_api_key_not_in_logs(temp_output_dir)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Integration tests are flaky on macOS in Github Actions.",
)
class TestApiKeyRedactionHttpTrace:
    """Verify API keys are redacted in --export-http-trace output with TRACE logging.

    This is critical because:
    1. HTTP trace captures the actual request headers sent over the wire
    2. TRACE-level logging dumps formatted payloads and internal state
    """

    async def test_api_key_redacted_in_http_trace(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer, temp_output_dir
    ):
        """JSONL records with --export-http-trace must not leak API keys in trace_data."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                {_COMMON_FLAGS} \
                --api-key {API_KEY} \
                --export-http-trace
            """
        )

        assert result.jsonl is not None
        assert len(result.jsonl) == 5

        for record in result.jsonl:
            if (
                record.trace_data is not None
                and record.trace_data.request_headers is not None
            ):
                headers_str = orjson.dumps(record.trace_data.request_headers).decode()
                assert API_KEY not in headers_str, (
                    f"API key leaked into trace request_headers: "
                    f"{record.trace_data.request_headers}"
                )
                assert (
                    record.trace_data.request_headers.get("Authorization")
                    == REDACTED_VALUE
                )

        _assert_api_key_not_in_logs(temp_output_dir)

    async def test_jsonl_file_and_logs_do_not_contain_api_key_with_trace(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer, temp_output_dir
    ):
        """Scan the entire JSONL file and all logs for the API key."""
        await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                {_COMMON_FLAGS} \
                --api-key {API_KEY} \
                --export-http-trace
            """
        )

        jsonl_file = next(temp_output_dir.glob("**/*profile_export.jsonl"), None)
        assert jsonl_file is not None
        assert API_KEY not in jsonl_file.read_text(), (
            "API key found in JSONL file with --export-http-trace enabled"
        )

        _assert_api_key_not_in_logs(temp_output_dir)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Integration tests are flaky on macOS in Github Actions.",
)
class TestApiKeyRedactionRawAndTraceCombo:
    """Verify redaction with both --export-level raw and --export-http-trace
    at TRACE log level."""

    async def test_combined_raw_and_trace_redaction(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer, temp_output_dir
    ):
        """Both raw records and trace data must be redacted simultaneously."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                {_COMMON_FLAGS} \
                --api-key {API_KEY} \
                --export-level raw \
                --export-http-trace
            """
        )

        # Check raw records
        assert result.raw_records is not None
        for record in result.raw_records:
            if record.request_headers:
                assert API_KEY not in orjson.dumps(record.request_headers).decode()

        # Check JSONL trace data
        assert result.jsonl is not None
        for record in result.jsonl:
            if record.trace_data and record.trace_data.request_headers:
                assert (
                    API_KEY
                    not in orjson.dumps(record.trace_data.request_headers).decode()
                )

        # Full file scan including logs
        _assert_api_key_not_in_any_artifact(temp_output_dir)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Integration tests are flaky on macOS in Github Actions.",
)
class TestApiKeyRedactionAllArtifacts:
    """Comprehensive scan: no artifact file on disk contains the API key,
    even at TRACE log level which logs formatted payloads and headers."""

    async def test_no_artifact_contains_api_key(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer, temp_output_dir
    ):
        """Scan every text file in the artifact directory for the API key."""
        await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                {_COMMON_FLAGS} \
                --api-key {API_KEY} \
                --export-level raw \
                --export-http-trace
            """
        )

        _assert_api_key_not_in_any_artifact(temp_output_dir)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Integration tests are flaky on macOS in Github Actions.",
)
class TestApiKeyStillFunctionalHTTP:
    """Verify the benchmark succeeds over real HTTP with TRACE logging
    (key reaches the server, redaction doesn't break functionality)."""

    async def test_benchmark_succeeds_with_api_key_http(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer, temp_output_dir
    ):
        """The mock server accepts requests with the API key and benchmark succeeds."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                {_COMMON_FLAGS} \
                --api-key {API_KEY}
            """
        )

        assert result.request_count == 5
        assert result.json is not None
        assert result.json.request_latency is not None

        _assert_api_key_not_in_logs(temp_output_dir)

    async def test_non_sensitive_headers_preserved_http(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer, temp_output_dir
    ):
        """Non-sensitive custom headers must appear unredacted in raw records."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                {_COMMON_FLAGS} \
                --header "X-Custom-Tracking:trace-abc-123" \
                --export-level raw
            """
        )

        assert result.raw_records is not None
        for record in result.raw_records:
            assert record.request_headers is not None
            assert record.request_headers.get("X-Custom-Tracking") == "trace-abc-123"
