# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
from unittest.mock import MagicMock

import pytest
from pytest import param
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
    """Unit tests for the Dynamo session_control 'bind' rejection detector."""

    @pytest.mark.parametrize(
        "message",
        [
            param(
                json.dumps(
                    {
                        "message": "Failed to deserialize the JSON body into the target type: "
                        "nvext.session_control.action: unknown variant `bind`, "
                        "expected `open` or `close` at line 1 column 100"
                    }
                ),
                id="json_wrapped_backend_message",
            ),
            param(
                "unknown variant `bind`, expected `open` or `close` at line 1 column 100",
                id="raw_non_json_message",
            ),
            param(
                json.dumps("unknown variant `bind`, expected `open` or `close`"),
                id="json_non_dict_falls_back_to_raw",
            ),
        ],
    )  # fmt: skip
    def test_detect_unknown_bind_variant_returns_insight(self, message):
        """serde-style 'unknown variant `bind`' errors should map to the Dynamo insight."""
        summary = make_summary(MockErrorDetails(message=message))

        insight = DynamoSessionControlDetector.detect(summary)

        assert insight is not None
        assert "bind" in insight.title
        assert "session_control" in insight.problem
        assert any("--use-dynamo-conv-aware-routing" in c for c in insight.causes)
        assert any("--use-legacy-dynamo-session-control" in f for f in insight.fixes)

    @pytest.mark.parametrize(
        "message",
        [
            param('{"message": "context_length_exceeded"}', id="unrelated_json_error"),
            param(
                "unknown variant `frobnicate`, expected `open` or `close`",
                id="unknown_variant_without_bind",
            ),
            param(
                '{"message": "session bind failed: worker unavailable"}',
                id="bind_without_unknown_variant",
            ),
            param("", id="empty_message"),
        ],
    )  # fmt: skip
    def test_detect_unrelated_error_returns_none(self, message):
        summary = make_summary(MockErrorDetails(message=message))

        assert DynamoSessionControlDetector.detect(summary) is None

    def test_detect_no_errors_returns_none(self):
        assert DynamoSessionControlDetector.detect(None) is None
        assert DynamoSessionControlDetector.detect([]) is None

    def test_detect_item_without_error_details_skipped_returns_none(self):
        summary = [MockErrorDetailsCount(None, 1)]

        assert DynamoSessionControlDetector.detect(summary) is None

    def test_detect_none_message_returns_none(self):
        summary = make_summary(MockErrorDetails(message=None))

        assert DynamoSessionControlDetector.detect(summary) is None

    @pytest.mark.asyncio
    async def test_exporter_prints_panel_for_bind_rejection(self):
        """The registered detector should surface a Rich panel via the exporter."""
        mock_console = MagicMock(spec=Console)
        err = MockErrorDetails(
            message=json.dumps(
                {"message": "unknown variant `bind`, expected `open` or `close`"}
            )
        )
        exporter_config = MagicMock(spec=ExporterConfig)
        exporter_config.results = MagicMock()
        exporter_config.results.error_summary = make_summary(err)

        exporter = ConsoleApiErrorExporter(exporter_config)

        await exporter.export(mock_console)

        assert mock_console.print.call_count >= 2
        _, args, _ = mock_console.print.mock_calls[1]
        panel = args[0]
        assert "Unsupported Dynamo session_control action: bind" in str(panel.title)
        panel_text = str(panel.renderable)
        assert "--use-legacy-dynamo-session-control" in panel_text
