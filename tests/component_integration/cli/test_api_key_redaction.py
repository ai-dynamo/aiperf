# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration tests for API key redaction in exported artifacts.

These tests run a full benchmark end-to-end (in-process via FakeTransport)
and verify that API keys are redacted in all exported files, while still
being used for actual requests.
"""

import orjson
import pytest

from aiperf.common.redact import REDACTED_VALUE
from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI

API_KEY = "sk-test-secret-key-REDACT-ME-12345"
CUSTOM_HEADER_KEY = "nvapi-custom-secret-999"

_BASE_CMD = f"""\
    aiperf profile \
        --model {defaults.model} \
        --endpoint-type chat \
        --streaming \
        --request-count 5 \
        --concurrency {defaults.concurrency} \
        --ui {defaults.ui}"""


@pytest.mark.component_integration
class TestApiKeyRedactionRawExport:
    """Verify API keys are redacted in --export-level raw output."""

    def test_api_key_redacted_in_raw_records(self, cli: AIPerfCLI):
        """Raw record export must not contain the real API key in request_headers."""
        result = cli.run_sync(f"{_BASE_CMD} --api-key {API_KEY} --export-level raw")

        assert result.raw_records is not None
        assert len(result.raw_records) == 5

        for record in result.raw_records:
            assert record.request_headers is not None
            headers_str = orjson.dumps(record.request_headers).decode()
            assert API_KEY not in headers_str, (
                f"Real API key leaked into raw record headers: {record.request_headers}"
            )
            if "Authorization" in record.request_headers:
                assert record.request_headers["Authorization"] == REDACTED_VALUE

    def test_custom_auth_header_redacted_in_raw_records(self, cli: AIPerfCLI):
        """API keys passed via --header Authorization:... must also be redacted."""
        result = cli.run_sync(
            f'{_BASE_CMD} --header "Authorization:Bearer {CUSTOM_HEADER_KEY}" --export-level raw'
        )

        assert result.raw_records is not None
        assert len(result.raw_records) == 5

        for record in result.raw_records:
            assert record.request_headers is not None
            headers_str = orjson.dumps(record.request_headers).decode()
            assert CUSTOM_HEADER_KEY not in headers_str, (
                f"Custom auth key leaked into raw record headers: {record.request_headers}"
            )

    def test_x_api_key_header_redacted_in_raw_records(self, cli: AIPerfCLI):
        """X-API-Key passed via --header must be redacted."""
        result = cli.run_sync(
            f'{_BASE_CMD} --header "X-API-Key:{CUSTOM_HEADER_KEY}" --export-level raw'
        )

        assert result.raw_records is not None
        for record in result.raw_records:
            assert record.request_headers is not None
            headers_str = orjson.dumps(record.request_headers).decode()
            assert CUSTOM_HEADER_KEY not in headers_str

    def test_raw_record_file_does_not_contain_api_key(
        self, cli: AIPerfCLI, temp_output_dir
    ):
        """Scan the entire raw JSONL file for the API key string."""
        cli.run_sync(f"{_BASE_CMD} --api-key {API_KEY} --export-level raw")

        raw_file = next(temp_output_dir.glob("**/*profile_export_raw.jsonl"), None)
        assert raw_file is not None, "Raw records file should exist"
        assert API_KEY not in raw_file.read_text(), (
            "Real API key found in raw records JSONL file"
        )


@pytest.mark.component_integration
class TestApiKeyRedactionAllArtifacts:
    """Verify API keys are not present in any exported artifact file."""

    def test_no_artifact_contains_api_key(self, cli: AIPerfCLI, temp_output_dir):
        """Scan JSON, JSONL, inputs, and log files for the API key."""
        cli.run_sync(f"{_BASE_CMD} --api-key {API_KEY}")

        text_extensions = {".json", ".jsonl", ".csv", ".log", ".yaml", ".yml", ".txt"}
        for file_path in temp_output_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in text_extensions:
                content = file_path.read_text(errors="replace")
                assert API_KEY not in content, (
                    f"API key leaked into: {file_path.relative_to(temp_output_dir)}"
                )


@pytest.mark.component_integration
class TestApiKeyRedactionNonSensitiveHeadersPreserved:
    """Verify non-sensitive headers still appear correctly in exports."""

    def test_non_sensitive_headers_present_in_raw_records(self, cli: AIPerfCLI):
        """Custom non-sensitive headers must still appear in raw records."""
        result = cli.run_sync(
            f'{_BASE_CMD} --header "X-Custom-Tracking:trace-abc-123" --export-level raw'
        )

        assert result.raw_records is not None
        assert len(result.raw_records) > 0

        for record in result.raw_records:
            assert record.request_headers is not None
            assert record.request_headers.get("X-Custom-Tracking") == "trace-abc-123"


@pytest.mark.component_integration
class TestApiKeyRedactionStillFunctional:
    """Verify the benchmark still succeeds with an API key (requests are not broken)."""

    def test_benchmark_succeeds_with_api_key(self, cli: AIPerfCLI):
        """A full benchmark with --api-key must succeed (key reaches FakeTransport)."""
        result = cli.run_sync(f"{_BASE_CMD} --api-key {API_KEY}")

        assert result.request_count == 5
        assert result.json is not None
        assert result.json.request_latency is not None
