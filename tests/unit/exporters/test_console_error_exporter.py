# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`ConsoleErrorExporter`.

The exporter renders the aggregated Code/Type/Message/Count table that is the
primary diagnostic for a run in which requests failed.
"""

import asyncio

import pytest
from rich.console import Console

from aiperf.common.models import ErrorDetails, ErrorDetailsCount, ProfileResults
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.exporters.console_error_exporter import ConsoleErrorExporter
from tests.unit.exporters.conftest import make_exporter_config


def make_results(error_summary=None, **kwargs) -> ProfileResults:
    """Build a minimal ProfileResults carrying the given error summary."""
    defaults = dict(
        records=[],
        completed=0,
        start_ns=0,
        end_ns=1,
        successful_request_count=0,
        error_request_count=sum(e.count for e in error_summary or []),
        error_summary=error_summary or [],
    )
    defaults.update(kwargs)
    return ProfileResults(**defaults)


def make_error(code=404, type="Not Found", message="Not Found", count=1):
    return ErrorDetailsCount(
        error_details=ErrorDetails(code=code, type=type, message=message),
        count=count,
    )


def render(results) -> str:
    """Run the exporter against a recording console and return the text."""
    console = Console(record=True, width=200)
    exporter = ConsoleErrorExporter(
        make_exporter_config(
            results=results,
            cli_config=CLIConfig(model_names=["test_model"]),
        )
    )
    asyncio.run(exporter.export(console))
    return console.export_text()


class TestConsoleErrorExporter:
    """Rendering behaviour of the error summary table."""

    def test_renders_all_four_columns(self):
        """The table exposes Code, Type, Message and Count headers."""
        out = render(make_results([make_error()]))

        for header in ("Code", "Type", "Message", "Count"):
            assert header in out, f"missing column header: {header}"

    def test_renders_http_status_code(self):
        """The HTTP status is the single most useful field and must appear."""
        out = render(make_results([make_error(code=404)]))

        assert "404" in out

    def test_renders_type_and_message(self):
        out = render(
            make_results(
                [make_error(code=401, type="Unauthorized", message="bad token")]
            )
        )

        assert "401" in out
        assert "Unauthorized" in out
        assert "bad token" in out

    def test_renders_nothing_when_summary_empty(self):
        """No errors means no table, not an empty table."""
        out = render(make_results([]))

        assert "Error Summary" not in out
        assert out.strip() == ""

    def test_renders_one_row_per_distinct_error(self):
        out = render(
            make_results(
                [
                    make_error(code=404, type="Not Found", message="no such path"),
                    make_error(code=500, type="Server Error", message="boom"),
                ]
            )
        )

        assert "404" in out
        assert "500" in out
        assert "no such path" in out
        assert "boom" in out

    def test_count_uses_thousands_separator(self):
        out = render(make_results([make_error(count=1234)]))

        assert "1,234" in out

    @pytest.mark.parametrize("missing", ["code", "type"])
    def test_missing_fields_render_as_na(self, missing):
        """A missing code or type degrades to N/A rather than crashing."""
        kwargs = {"code": 404, "type": "Not Found"}
        kwargs[missing] = None
        out = render(make_results([make_error(**kwargs)]))

        assert "N/A" in out


class TestMarkupSafety:
    """Server-controlled text must never be parsed as Rich console markup.

    ``ErrorDetails.type`` and ``.message`` carry the response reason and body
    verbatim, so a server can put arbitrary bracketed text in them.
    """

    def test_stray_closing_tag_does_not_raise(self):
        """A closing tag with no opener used to raise MarkupError."""
        out = render(
            make_results(
                [make_error(message='{"error":"unexpected closing tag [/INST]"}')]
            )
        )

        assert "[/INST]" in out

    def test_opening_tag_is_not_swallowed(self):
        """An opening tag used to render as an empty Message cell."""
        out = render(make_results([make_error(message="[not a tag]")]))

        assert "[not a tag]" in out

    def test_markup_in_type_is_literal(self):
        out = render(make_results([make_error(type="[/oops]", message="body")]))

        assert "[/oops]" in out

    def test_style_markup_is_not_interpreted(self):
        """A server sending style tags must not colour our output."""
        out = render(make_results([make_error(message="[red]not really red[/red]")]))

        assert "[red]not really red[/red]" in out

    def test_na_placeholders_still_render(self):
        """The intentional [dim]N/A[/dim] placeholders must keep working."""
        out = render(make_results([make_error(code=None, type=None)]))

        assert "N/A" in out
        assert "[dim]" not in out
