# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for header redaction at JSON-export serialization time.

These guard the on-disk artifacts (profile_export_aiperf.json,
server_metrics_export.json) against leaking Authorization / X-API-Key /
api-key etc. via the structured EndpointConfig.headers field. The cli_command
*string* path is covered separately in tests/unit/common/test_redact.py.
"""

import orjson
import pytest
from pytest import param

from aiperf.common.models.model_endpoint_info import EndpointInfo
from aiperf.common.redact import REDACTED_VALUE
from aiperf.config.endpoint import EndpointConfig
from aiperf.plugin.enums import EndpointType


class TestEndpointConfigHeadersRedaction:
    """EndpointConfig.headers must be redacted in JSON dumps (both modes)."""

    @pytest.mark.parametrize(
        "header_name,header_value",
        [
            param("Authorization", "Bearer secret-token-123", id="authorization-bearer"),
            param("authorization", "Basic dXNlcjpwYXNz", id="authorization-basic-lowercase"),
            param("Proxy-Authorization", "Bearer proxy-secret", id="proxy-authorization"),
            param("X-API-Key", "another-secret", id="x-api-key"),
            param("api-key", "azure-secret-789", id="api-key-azure"),
            param("Ocp-Apim-Subscription-Key", "apim-secret", id="ocp-apim"),
            param("X-Goog-Api-Key", "google-secret", id="x-goog-api-key"),
            param("X-Functions-Key", "functions-secret", id="x-functions-key"),
            param("Aeg-Sas-Key", "eventgrid-secret", id="aeg-sas-key"),
            param("X-Amz-Security-Token", "aws-sts-secret", id="x-amz-security-token"),
        ],
    )  # fmt: skip
    def test_sensitive_header_redacted_in_model_dump_json(
        self, header_name: str, header_value: str
    ) -> None:
        cfg = EndpointConfig(
            type=EndpointType.CHAT,
            urls=["http://localhost:8000/v1/chat/completions"],
            headers={header_name: header_value, "Accept": "application/json"},
        )

        dumped = orjson.loads(cfg.model_dump_json())

        assert header_value not in cfg.model_dump_json()
        assert dumped["headers"][header_name] == REDACTED_VALUE
        assert dumped["headers"]["Accept"] == "application/json"

    def test_non_sensitive_headers_preserved(self) -> None:
        cfg = EndpointConfig(
            type=EndpointType.CHAT,
            urls=["http://localhost:8000/v1/chat/completions"],
            headers={
                "Accept": "application/json",
                "User-Agent": "aiperf/test",
                "X-Request-ID": "abc123",
            },
        )

        dumped = orjson.loads(cfg.model_dump_json())

        assert dumped["headers"] == {
            "Accept": "application/json",
            "User-Agent": "aiperf/test",
            "X-Request-ID": "abc123",
        }

    def test_redaction_keeps_header_keys_visible(self) -> None:
        """Tests should still see WHICH header was passed, just not the secret."""
        cfg = EndpointConfig(
            type=EndpointType.CHAT,
            urls=["http://localhost:8000/v1/chat/completions"],
            headers={
                "Authorization": "Bearer secret-token-123",
                "X-API-Key": "another-secret",
            },
        )

        dumped = orjson.loads(cfg.model_dump_json())

        assert "Authorization" in dumped["headers"]
        assert "X-API-Key" in dumped["headers"]
        assert dumped["headers"]["Authorization"] == REDACTED_VALUE
        assert dumped["headers"]["X-API-Key"] == REDACTED_VALUE

    def test_model_dump_mode_json_redacts(self) -> None:
        """Server-metrics exporter calls model_dump(mode='json'); must also redact."""
        cfg = EndpointConfig(
            type=EndpointType.CHAT,
            urls=["http://localhost:8000/v1/chat/completions"],
            headers={"Authorization": "Bearer secret-token-123"},
        )

        dumped = cfg.model_dump(mode="json")

        assert dumped["headers"]["Authorization"] == REDACTED_VALUE

    def test_model_dump_python_mode_does_not_redact(self) -> None:
        """In-memory python dumps must keep real values for runtime use."""
        cfg = EndpointConfig(
            type=EndpointType.CHAT,
            urls=["http://localhost:8000/v1/chat/completions"],
            headers={"Authorization": "Bearer secret-token-123"},
        )

        dumped = cfg.model_dump()

        assert dumped["headers"]["Authorization"] == "Bearer secret-token-123"

    def test_in_memory_field_unchanged(self) -> None:
        """Redaction is serialization-only; the model attribute keeps the real value."""
        cfg = EndpointConfig(
            type=EndpointType.CHAT,
            urls=["http://localhost:8000/v1/chat/completions"],
            headers={"Authorization": "Bearer secret-token-123"},
        )

        assert cfg.headers == {"Authorization": "Bearer secret-token-123"}


class TestEndpointInfoHeadersRedaction:
    """Defense-in-depth: EndpointInfo.headers (tuple form) is also redacted."""

    def test_sensitive_header_tuples_redacted_in_json(self) -> None:
        info = EndpointInfo(
            type=EndpointType.CHAT,
            base_urls=["http://localhost:8000"],
            headers=[
                ("Authorization", "Bearer secret-token-123"),
                ("X-API-Key", "another-secret"),
                ("Accept", "application/json"),
            ],
        )

        dumped_json = info.model_dump_json()

        assert "secret-token-123" not in dumped_json
        assert "another-secret" not in dumped_json

        dumped = orjson.loads(dumped_json)
        # tuples round-trip as 2-element lists in JSON
        headers_map = {k: v for k, v in dumped["headers"]}
        assert headers_map["Authorization"] == REDACTED_VALUE
        assert headers_map["X-API-Key"] == REDACTED_VALUE
        assert headers_map["Accept"] == "application/json"

    def test_python_mode_keeps_real_values(self) -> None:
        info = EndpointInfo(
            type=EndpointType.CHAT,
            base_urls=["http://localhost:8000"],
            headers=[("Authorization", "Bearer secret-token-123")],
        )

        # Plain model_dump (python mode) must keep real values for runtime use.
        dumped = info.model_dump()
        assert dumped["headers"] == [("Authorization", "Bearer secret-token-123")]
