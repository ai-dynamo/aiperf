# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for plugins CLI command."""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from pytest import param
from rich.console import Console

from aiperf.cli_commands.plugins_cli import (
    _hint,
    _title,
    plugins_cli_command,
    run_validate,
    show_categories_overview,
    show_category_types,
    show_packages_detailed,
    show_type_details,
)
from aiperf.plugin.types import PackageInfo, PluginEntry, TypeNotFoundError

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def mock_console() -> Generator[MagicMock, None, None]:
    """Mock the console for output capture."""
    with patch("aiperf.cli_commands.plugins_cli.console") as mock:
        yield mock


@pytest.fixture
def capture_console() -> Generator[tuple[Console, StringIO], None, None]:
    """Create a console that captures output to a StringIO buffer."""
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=120)
    with patch("aiperf.cli_commands.plugins_cli.console", console):
        yield console, buffer


@pytest.fixture
def mock_plugins() -> Generator[MagicMock, None, None]:
    """Mock the plugins module for testing."""
    with patch("aiperf.cli_commands.plugins_cli.plugins") as mock:
        yield mock


@pytest.fixture
def mock_validate_all() -> Generator[MagicMock, None, None]:
    """Mock plugins.validate_all for testing."""
    with patch("aiperf.cli_commands.plugins_cli.plugins.validate_all") as mock:
        yield mock


# =============================================================================
# Shared Helpers
# =============================================================================


def make_plugin_entry(
    name: str = "test_type",
    category: str = "test_category",
    package: str = "aiperf",
    class_path: str = "aiperf.test:TestClass",
    description: str = "Test description",
    priority: int = 0,
) -> PluginEntry:
    """Create a PluginEntry for testing."""
    return PluginEntry(
        name=name,
        category=category,
        package=package,
        class_path=class_path,
        description=description,
        priority=priority,
        metadata={},
    )


def make_package_info(
    name: str = "aiperf",
    version: str = "1.0.0",
    description: str = "Test package",
) -> PackageInfo:
    """Create a PackageInfo for testing."""
    return PackageInfo(name=name, version=version, description=description)


def setup_mock_plugins(
    mock: MagicMock,
    *,
    packages: list[str] | None = None,
    package_metadata: list[PackageInfo] | None = None,
    categories: list[str] | None = None,
    entries: list[PluginEntry] | None = None,
    category_metadata: dict[str, Any] | None = None,
) -> None:
    """Configure mock plugins module with common patterns."""
    if packages is not None:
        mock.list_packages.return_value = packages
    if package_metadata is not None:
        mock.get_package_metadata.side_effect = package_metadata
    if categories is not None:
        mock.list_categories.return_value = categories
    if entries is not None:
        mock.list_entries.return_value = entries
        mock.iter_entries.return_value = iter(entries)
    mock.get_category_metadata.return_value = category_metadata


def get_console_output(capture_console: tuple[Console, StringIO]) -> str:
    """Extract output from capture_console fixture."""
    _, buffer = capture_console
    return buffer.getvalue()


def assert_console_contains(mock_console: MagicMock, *texts: str) -> None:
    """Assert that console.print was called with strings containing all given texts."""
    calls = [str(c) for c in mock_console.print.call_args_list]
    for text in texts:
        assert any(text in c for c in calls), f"Expected '{text}' in console output"


def assert_console_not_contains(mock_console: MagicMock, *texts: str) -> None:
    """Assert that console.print was NOT called with strings containing given texts."""
    calls = [str(c) for c in mock_console.print.call_args_list]
    for text in texts:
        assert not any(text in c for c in calls), (
            f"Unexpected '{text}' in console output"
        )


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestTitle:
    """Tests for _title() helper function."""

    @pytest.mark.parametrize(
        ("input_str", "expected"),
        [
            param("endpoint", "Endpoint", id="single word"),
            param("data_exporter", "Data Exporter", id="snake_case"),
            param("api_handler", "API Handler", id="api acronym"),
            param("ui", "UI", id="ui acronym"),
            param("gpu_telemetry", "GPU Telemetry", id="gpu acronym"),
            param("http_transport", "HTTP Transport", id="http acronym"),
            param("csv_exporter", "CSV Exporter", id="csv acronym"),
            param("json_parser", "JSON Parser", id="json acronym"),
            param("hf_model", "HF Model", id="hf acronym"),
            param("tei_embeddings", "TEI Embeddings", id="tei acronym"),
            param("zmq_proxy", "ZMQ Proxy", id="zmq acronym"),
            param("cpu_metrics", "CPU Metrics", id="cpu acronym"),
        ],
    )  # fmt: skip
    def test_title_formatting(self, input_str: str, expected: str) -> None:
        """Test that _title formats category names correctly."""
        assert _title(input_str) == expected


class TestHint:
    """Tests for _hint() helper function."""

    def test_hint_prints_dim_message(self, mock_console: MagicMock) -> None:
        """Test that _hint prints a dim styled message."""
        _hint("Test hint message")
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "[dim]" in call_args
        assert "Test hint message" in call_args


# =============================================================================
# show_packages_detailed Tests
# =============================================================================


class TestShowPackagesDetailed:
    """Tests for show_packages_detailed() function."""

    def test_no_packages_shows_warning(
        self, mock_console: MagicMock, mock_plugins: MagicMock
    ) -> None:
        """Test that no packages shows yellow warning."""
        setup_mock_plugins(mock_plugins, packages=[])
        show_packages_detailed()

        call_args = str(mock_console.print.call_args_list[0])
        assert "No packages found" in call_args

    def test_shows_package_table(
        self, capture_console: tuple[Console, StringIO], mock_plugins: MagicMock
    ) -> None:
        """Test that packages are displayed in a table."""
        setup_mock_plugins(
            mock_plugins,
            packages=["aiperf", "my-plugin"],
            package_metadata=[
                make_package_info("aiperf", "1.0.0", "Core package"),
                make_package_info("my-plugin", "2.0.0", "Custom plugin"),
            ],
            entries=[
                make_plugin_entry(package="aiperf"),
                make_plugin_entry(package="aiperf"),
                make_plugin_entry(package="my-plugin"),
            ],
        )

        show_packages_detailed()

        output = get_console_output(capture_console)
        assert "Installed Packages" in output
        assert "aiperf" in output
        assert "my-plugin" in output

    def test_shows_plugin_counts(
        self, capture_console: tuple[Console, StringIO], mock_plugins: MagicMock
    ) -> None:
        """Test that plugin counts per package are shown."""
        setup_mock_plugins(
            mock_plugins,
            packages=["aiperf"],
            package_metadata=[make_package_info()],
            entries=[make_plugin_entry(package="aiperf") for _ in range(3)],
        )

        show_packages_detailed()

        output = get_console_output(capture_console)
        assert "3" in output

    def test_shows_hint_for_all_flag(
        self, mock_console: MagicMock, mock_plugins: MagicMock
    ) -> None:
        """Test that hint for --all flag is shown."""
        setup_mock_plugins(
            mock_plugins,
            packages=["aiperf"],
            package_metadata=[make_package_info()],
            entries=[],
        )

        show_packages_detailed()

        assert_console_contains(mock_console, "--all")


# =============================================================================
# show_categories_overview Tests
# =============================================================================


class TestShowCategoriesOverview:
    """Tests for show_categories_overview() function."""

    def test_no_categories_shows_warning(
        self, mock_console: MagicMock, mock_plugins: MagicMock
    ) -> None:
        """Test that no categories shows yellow warning."""
        setup_mock_plugins(mock_plugins, categories=[])
        show_categories_overview()

        call_args = str(mock_console.print.call_args_list[0])
        assert "No categories found" in call_args

    def test_shows_categories_table(
        self, capture_console: tuple[Console, StringIO], mock_plugins: MagicMock
    ) -> None:
        """Test that categories are displayed in a table."""
        mock_plugins.list_categories.return_value = ["endpoint", "transport"]
        mock_plugins.list_entries.side_effect = [
            [make_plugin_entry(name="chat"), make_plugin_entry(name="completions")],
            [make_plugin_entry(name="http")],
        ]

        show_categories_overview()

        output = get_console_output(capture_console)
        for expected in [
            "Plugin Categories",
            "Endpoint",
            "Transport",
            "chat",
            "completions",
            "http",
        ]:
            assert expected in output

    def test_shows_hint_for_category(
        self, mock_console: MagicMock, mock_plugins: MagicMock
    ) -> None:
        """Test that hint for category usage is shown."""
        setup_mock_plugins(
            mock_plugins, categories=["endpoint"], entries=[make_plugin_entry()]
        )
        show_categories_overview()

        assert_console_contains(mock_console, "aiperf plugins <category>")


# =============================================================================
# show_category_types Tests
# =============================================================================


class TestShowCategoryTypes:
    """Tests for show_category_types() function."""

    def test_unknown_category_shows_warning(
        self, mock_console: MagicMock, mock_plugins: MagicMock
    ) -> None:
        """Test that unknown category shows yellow warning."""
        setup_mock_plugins(
            mock_plugins, entries=[], categories=["endpoint", "transport"]
        )
        show_category_types("unknown_category")

        call_args = str(mock_console.print.call_args_list[0])
        assert "Unknown category: unknown_category" in call_args

    def test_shows_available_categories_hint(
        self, mock_console: MagicMock, mock_plugins: MagicMock
    ) -> None:
        """Test that available categories are shown as hint."""
        setup_mock_plugins(
            mock_plugins, entries=[], categories=["endpoint", "transport"]
        )
        show_category_types("unknown")

        assert_console_contains(mock_console, "endpoint", "transport")

    def test_shows_types_table(
        self, capture_console: tuple[Console, StringIO], mock_plugins: MagicMock
    ) -> None:
        """Test that types are displayed in a table."""
        setup_mock_plugins(
            mock_plugins,
            entries=[
                make_plugin_entry(name="chat", description="Chat endpoint"),
                make_plugin_entry(
                    name="completions", description="Completions endpoint"
                ),
            ],
            category_metadata=None,
        )

        show_category_types("endpoint")

        output = get_console_output(capture_console)
        for expected in ["Endpoint Types", "chat", "completions", "Chat endpoint"]:
            assert expected in output

    def test_shows_category_description(
        self, capture_console: tuple[Console, StringIO], mock_plugins: MagicMock
    ) -> None:
        """Test that category description is shown if available."""
        setup_mock_plugins(
            mock_plugins,
            entries=[make_plugin_entry()],
            category_metadata={"description": "API endpoints for LLM inference"},
        )

        show_category_types("endpoint")

        output = get_console_output(capture_console)
        assert "API endpoints for LLM inference" in output

    @pytest.mark.parametrize(
        "description",
        [param("", id="empty"), param("   \n\t  ", id="whitespace only")],
    )
    def test_handles_empty_description(
        self,
        capture_console: tuple[Console, StringIO],
        mock_plugins: MagicMock,
        description: str,
    ) -> None:
        """Test that empty/whitespace description is handled gracefully."""
        setup_mock_plugins(
            mock_plugins,
            entries=[make_plugin_entry(name="test", description=description)],
            category_metadata=None,
        )

        show_category_types("endpoint")

        output = get_console_output(capture_console)
        assert "test" in output

    def test_shows_hint_for_type_details(
        self, mock_console: MagicMock, mock_plugins: MagicMock
    ) -> None:
        """Test that hint for type details is shown."""
        setup_mock_plugins(
            mock_plugins, entries=[make_plugin_entry()], category_metadata=None
        )
        show_category_types("endpoint")

        assert_console_contains(mock_console, "aiperf plugins endpoint <type>")


# =============================================================================
# show_type_details Tests
# =============================================================================


class TestShowTypeDetails:
    """Tests for show_type_details() function."""

    @pytest.mark.parametrize(
        "exception",
        [
            param(KeyError("Not found"), id="KeyError"),
            param(TypeNotFoundError("endpoint", "nonexistent", ["chat", "completions"]), id="TypeNotFoundError"),
        ],
    )  # fmt: skip
    def test_type_not_found_shows_error(
        self, mock_console: MagicMock, mock_plugins: MagicMock, exception: Exception
    ) -> None:
        """Test that not found type shows red error."""
        mock_plugins.get_entry.side_effect = exception
        mock_plugins.list_entries.return_value = [
            make_plugin_entry(name="chat"),
            make_plugin_entry(name="completions"),
        ]

        show_type_details("endpoint", "nonexistent")

        call_args = str(mock_console.print.call_args_list[0])
        assert "Not found: endpoint:nonexistent" in call_args

    def test_shows_type_panel(
        self, capture_console: tuple[Console, StringIO], mock_plugins: MagicMock
    ) -> None:
        """Test that type details are shown in a panel."""
        mock_plugins.get_entry.return_value = make_plugin_entry(
            name="chat",
            category="endpoint",
            package="aiperf",
            class_path="aiperf.endpoints:ChatEndpoint",
            description="OpenAI-compatible chat completions endpoint",
        )

        show_type_details("endpoint", "chat")

        output = get_console_output(capture_console)
        for expected in [
            "endpoint:chat",
            "chat",
            "aiperf",
            "aiperf.endpoints:ChatEndpoint",
            "OpenAI-compatible",
        ]:
            assert expected in output

    def test_handles_no_description(
        self, capture_console: tuple[Console, StringIO], mock_plugins: MagicMock
    ) -> None:
        """Test that missing description shows dim message."""
        mock_plugins.get_entry.return_value = make_plugin_entry(description="")
        show_type_details("endpoint", "chat")

        output = get_console_output(capture_console)
        assert "No description" in output

    def test_empty_entries_hides_hint(
        self, mock_console: MagicMock, mock_plugins: MagicMock
    ) -> None:
        """Test that empty entries list doesn't show hint."""
        mock_plugins.get_entry.side_effect = KeyError("Not found")
        mock_plugins.list_entries.return_value = []

        show_type_details("endpoint", "nonexistent")

        assert_console_contains(mock_console, "Not found")
        assert_console_not_contains(mock_console, "Available:")


# =============================================================================
# run_validate Tests
# =============================================================================


class TestRunValidate:
    """Tests for run_validate() function."""

    @pytest.mark.parametrize(
        ("registry_errors", "expected_strings"),
        [
            param({}, ["All checks passed", "✓"], id="all pass"),
            param(
                {"endpoint": [("broken_type", "Module not found")]},
                ["Class paths", "broken_type", "Module not found", "Validation failed"],
                id="registry errors",
            ),
        ],
    )  # fmt: skip
    def test_validate_output(
        self,
        capture_console: tuple[Console, StringIO],
        mock_validate_all: MagicMock,
        registry_errors: dict,
        expected_strings: list[str],
    ) -> None:
        """Test validation output for various scenarios."""
        mock_validate_all.return_value = registry_errors

        run_validate()

        output = get_console_output(capture_console)
        for expected in expected_strings:
            assert expected in output


# =============================================================================
# CLI Command Routing Tests
# =============================================================================


class TestPluginsCliCommand:
    """Tests for plugins_cli_command() routing."""

    @pytest.mark.parametrize(
        ("category", "name", "all_plugins", "validate", "expected_call"),
        [
            param(None, None, False, False, "show_packages_detailed", id="default"),
            param(None, None, True, False, "show_categories_overview", id="--all flag"),
            param(None, None, False, True, "run_validate", id="--validate flag"),
            param(None, None, True, True, "run_validate", id="validate precedence"),
        ],
    )  # fmt: skip
    def test_command_routing_no_category(
        self,
        mock_console: MagicMock,
        category: None,
        name: str | None,
        all_plugins: bool,
        validate: bool,
        expected_call: str,
    ) -> None:
        """Test CLI command routing without category."""
        with patch(f"aiperf.cli_commands.plugins_cli.{expected_call}") as mock_fn:
            plugins_cli_command(
                category=category, name=name, all_plugins=all_plugins, validate=validate
            )
            mock_fn.assert_called_once()

    def test_category_only_shows_types(self, mock_console: MagicMock) -> None:
        """Test that category without name shows types."""
        with patch("aiperf.cli_commands.plugins_cli.show_category_types") as mock_show:
            mock_category = MagicMock()
            plugins_cli_command(
                category=mock_category, name=None, all_plugins=False, validate=False
            )
            mock_show.assert_called_once_with(mock_category)

    def test_category_and_name_shows_details(self, mock_console: MagicMock) -> None:
        """Test that category with name shows type details."""
        with patch("aiperf.cli_commands.plugins_cli.show_type_details") as mock_show:
            mock_category = MagicMock()
            plugins_cli_command(
                category=mock_category, name="chat", all_plugins=False, validate=False
            )
            mock_show.assert_called_once_with(mock_category, "chat")


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.parametrize(
        ("version", "description"),
        [
            param("", "Test", id="empty version"),
            param("1.0.0", "", id="empty description"),
        ],
    )
    def test_package_with_empty_fields(
        self,
        capture_console: tuple[Console, StringIO],
        mock_plugins: MagicMock,
        version: str,
        description: str,
    ) -> None:
        """Test packages with empty fields are handled gracefully."""
        setup_mock_plugins(
            mock_plugins,
            packages=["test-pkg"],
            package_metadata=[make_package_info("test-pkg", version, description)],
            entries=[],
        )

        show_packages_detailed()

        output = get_console_output(capture_console)
        assert "test-pkg" in output

    def test_category_metadata_strips_whitespace(
        self, capture_console: tuple[Console, StringIO], mock_plugins: MagicMock
    ) -> None:
        """Test category metadata description is stripped."""
        setup_mock_plugins(
            mock_plugins,
            entries=[make_plugin_entry()],
            category_metadata={"description": "  API endpoints  \n"},
        )

        show_category_types("endpoint")

        output = get_console_output(capture_console)
        assert "API endpoints" in output
