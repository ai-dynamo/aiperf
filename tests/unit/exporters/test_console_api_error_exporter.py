# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from aiperf.exporters.console_api_error_exporter import (
    ConsoleApiErrorExporter,
    DynamoSessionControlDetector,
    MaxCompletionTokensDetector,
)
from aiperf.exporters.exporter_config import ExporterConfig


class MockErrorDetails:
    def __init__(
        self, code=400, type="Bad Request", message="", cause=None, details=None
    ):
        self.code = code
        self.type = type
        self.message = message
        self.cause = cause
        self.details = details


class MockErrorDetailsCount:
    def __init__(self, error_details, count):
        self.error_details = error_details
        self.count = count


def make_summary(err):
    return [MockErrorDetailsCount(err, 1)]


@pytest.fixture
def basic_error_payload():
    """Minimal TRT-style forbidden-field error payload."""
    return json.dumps(
        {
            "message": (
                "[{'type': 'extra_forbidden','loc': ('body','max_completion_tokens'),"
                "'msg': 'Extra inputs are not permitted'}]"
            )
        }
    )


class TestConsoleApiErrorExporter:
    """Unit tests for the API error insight detector and console exporter."""

    def test_detector_detects_max_completion_tokens_error(self, basic_error_payload):
        """Detector should return an ErrorInsight for unsupported max_completion_tokens."""
        err = MockErrorDetails(message=basic_error_payload)
        summary = make_summary(err)

        insight = MaxCompletionTokensDetector.detect(summary)

        assert insight is not None
        assert "max_completion_tokens" in insight.problem
        assert "max_tokens" in insight.problem
        assert any("max_completion_tokens" in c for c in insight.causes)

    def test_detector_returns_none_for_unrelated_error(self):
        err = MockErrorDetails(message='{"message": "context_length_exceeded"}')
        summary = make_summary(err)

        assert MaxCompletionTokensDetector.detect(summary) is None

    def test_detector_returns_none_when_no_errors(self):
        assert MaxCompletionTokensDetector.detect(None) is None
        assert MaxCompletionTokensDetector.detect([]) is None

    @pytest.mark.asyncio
    async def test_exporter_prints_panel_for_detected_error(self, basic_error_payload):
        """Exporter should print a Rich panel when an insight is returned."""
        mock_console = MagicMock(spec=Console)

        err = MockErrorDetails(message=basic_error_payload)
        summary = make_summary(err)

        exporter_config = MagicMock(spec=ExporterConfig)
        exporter_config.results = MagicMock()
        exporter_config.results.error_summary = summary

        exporter = ConsoleApiErrorExporter(exporter_config)

        await exporter.export(mock_console)

        assert mock_console.print.call_count >= 2

        _, args, _ = mock_console.print.mock_calls[1]
        panel = args[0]

        assert hasattr(panel, "renderable")
        panel_text = str(panel.renderable)
        panel_title = str(panel.title)

        assert "Unsupported Parameter: max_completion_tokens" in panel_title
        assert "The backend rejected 'max_completion_tokens'" in panel_text
        assert "This backend only supports 'max_tokens'." in panel_text
        assert "--use-legacy-max-tokens" in panel_text

    @pytest.mark.asyncio
    async def test_exporter_skips_when_no_insight(self):
        mock_console = MagicMock(spec=Console)

        exporter_config = MagicMock(spec=ExporterConfig)
        exporter_config.results = MagicMock()
        exporter_config.results.error_summary = []

        exporter = ConsoleApiErrorExporter(exporter_config)

        await exporter.export(mock_console)

        assert mock_console.print.call_count == 0


class TestDynamoSessionControlDetector:
    """Unit tests for the Dynamo 'bind' session_control rejection detector."""

    def test_detects_raw_serde_unknown_variant_error(self):
        """serde rejects the unknown enum value in a raw (non-JSON) body."""
        err = MockErrorDetails(
            message="unknown variant `bind`, expected `open` or `close`"
        )
        insight = DynamoSessionControlDetector.detect(make_summary(err))

        assert insight is not None
        assert "bind" in insight.problem
        assert any("--use-legacy-dynamo-session-control" in f for f in insight.fixes)

    def test_detects_json_wrapped_unknown_variant_error(self):
        """The serde message is often wrapped in a JSON error envelope."""
        body = json.dumps(
            {
                "message": (
                    "Failed to deserialize the JSON body: nvext.session_control."
                    "action: unknown variant `bind`, expected `open` or `close`"
                )
            }
        )
        err = MockErrorDetails(message=body)
        insight = DynamoSessionControlDetector.detect(make_summary(err))

        assert insight is not None
        assert any("v1.3.0-dev" in f for f in insight.fixes)

    def test_returns_none_for_unrelated_error(self):
        err = MockErrorDetails(message='{"message": "context_length_exceeded"}')
        assert DynamoSessionControlDetector.detect(make_summary(err)) is None

    def test_returns_none_for_unknown_variant_without_bind(self):
        """A different unknown-variant error must not trip this detector."""
        err = MockErrorDetails(
            message="unknown variant `frobnicate`, expected `open` or `close`"
        )
        assert DynamoSessionControlDetector.detect(make_summary(err)) is None

    def test_returns_none_when_no_errors(self):
        assert DynamoSessionControlDetector.detect(None) is None
        assert DynamoSessionControlDetector.detect([]) is None

    @pytest.mark.asyncio
    async def test_exporter_prints_panel_for_bind_rejection(self):
        mock_console = MagicMock(spec=Console)
        err = MockErrorDetails(
            message="unknown variant `bind`, expected `open` or `close`"
        )

        exporter_config = MagicMock(spec=ExporterConfig)
        exporter_config.results = MagicMock()
        exporter_config.results.error_summary = make_summary(err)

        exporter = ConsoleApiErrorExporter(exporter_config)
        await exporter.export(mock_console)

        assert mock_console.print.call_count >= 2
        _, args, _ = mock_console.print.mock_calls[1]
        panel = args[0]
        assert "session_control action: bind" in str(panel.title)
        assert "--use-legacy-dynamo-session-control" in str(panel.renderable)
