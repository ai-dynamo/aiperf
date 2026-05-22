# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared template-CLI helpers.

Covers the logic that is non-obvious at a reading: the singular/plural form
detection in ``build_overrides`` (each template uses either ``model`` /
``models`` and ``endpoint.url`` / ``endpoint.urls``, and the override must
land on the form actually present), and the ``cmd=`` parameter that lets
``aiperf config init`` and ``aiperf kube init`` share these helpers while
printing their own hints.
"""

from __future__ import annotations

import pytest

from aiperf.config._cli_runner_templates import (
    build_overrides,
    handle_list,
    handle_search,
    print_template_table,
)


class TestBuildOverrides:
    """build_overrides must match the singular/plural form declared in the template."""

    def test_no_overrides_returns_empty(self) -> None:
        assert build_overrides("model: x\n", None, None) == {}

    def test_model_override_picks_singular_when_template_uses_singular(self) -> None:
        content = "model: placeholder\n"
        assert build_overrides(content, "my-llama", None) == {"model": "my-llama"}

    def test_model_override_picks_plural_when_template_uses_plural(self) -> None:
        content = "models:\n  - placeholder\n"
        assert build_overrides(content, "my-llama", None) == {"models": ["my-llama"]}

    def test_model_override_defaults_to_plural_when_neither_form_present(self) -> None:
        """Empty/unrelated templates get the canonical plural form."""
        content = "phases:\n  type: concurrency\n"
        assert build_overrides(content, "my-llama", None) == {"models": ["my-llama"]}

    def test_url_override_picks_singular_when_template_uses_singular(self) -> None:
        content = "endpoint:\n  url: http://placeholder:8000\n"
        assert build_overrides(content, None, "http://svc:8000") == {
            "endpoint": {"url": "http://svc:8000"}
        }

    def test_url_override_picks_plural_when_template_uses_plural(self) -> None:
        content = "endpoint:\n  urls:\n    - http://placeholder:8000\n"
        assert build_overrides(content, None, "http://svc:8000") == {
            "endpoint": {"urls": ["http://svc:8000"]}
        }

    def test_url_override_defaults_to_plural_when_endpoint_missing(self) -> None:
        """No `endpoint:` key in the template → canonical plural form."""
        content = "model: x\n"
        assert build_overrides(content, None, "http://svc:8000") == {
            "endpoint": {"urls": ["http://svc:8000"]}
        }

    def test_combined_overrides_use_each_templates_form_independently(self) -> None:
        """Singular `model` + plural `endpoint.urls` both get matched correctly."""
        content = "model: placeholder\nendpoint:\n  urls:\n    - http://x\n"
        assert build_overrides(content, "my-llama", "http://svc:8000") == {
            "model": "my-llama",
            "endpoint": {"urls": ["http://svc:8000"]},
        }

    def test_empty_template_content_uses_canonical_defaults(self) -> None:
        """yaml.safe_load on empty string returns None; helper must not crash."""
        assert build_overrides("", "my-llama", "http://svc:8000") == {
            "models": ["my-llama"],
            "endpoint": {"urls": ["http://svc:8000"]},
        }


class TestHandleSearch:
    """handle_search prints matching templates and routes empty results through cmd."""

    def test_matching_query_prints_template_rows(self, capsys) -> None:
        handle_search("goodput", verbose=False, cmd="aiperf config init")

        out = capsys.readouterr().out
        assert "goodput_slo" in out

    def test_no_match_hints_at_the_cmd_that_was_passed(self, capsys) -> None:
        handle_search(
            "zzz_definitely_no_such_template",
            verbose=False,
            cmd="aiperf kube init",
        )

        out = capsys.readouterr().out
        assert "No templates match 'zzz_definitely_no_such_template'." in out
        assert "Run 'aiperf kube init --list' to see all templates." in out

    def test_default_cmd_is_config_init(self, capsys) -> None:
        """Callers that omit cmd get the `aiperf config init` hint."""
        handle_search("zzz_nomatch", verbose=False)

        assert "aiperf config init --list" in capsys.readouterr().out


class TestHandleList:
    """handle_list prints category headers and a trailing cmd-specific hint."""

    def test_prints_category_headers_and_cmd_specific_hint(self, capsys) -> None:
        handle_list(category=None, verbose=False, cmd="aiperf kube init")

        out = capsys.readouterr().out
        assert "Getting Started" in out
        assert "minimal" in out
        assert "Use 'aiperf kube init --template <name>' to generate a template." in out

    def test_category_filter_narrows_results(self, capsys) -> None:
        handle_list(category="Load Testing", verbose=False, cmd="aiperf config init")

        out = capsys.readouterr().out
        assert "Load Testing" in out
        assert "goodput_slo" in out
        # "Getting Started" templates like 'minimal' should not appear
        assert "minimal" not in out

    def test_unknown_category_prints_empty_hint_without_table(self, capsys) -> None:
        handle_list(
            category="zzz_no_such_category",
            verbose=False,
            cmd="aiperf config init",
        )

        out = capsys.readouterr().out
        assert "No templates in category 'zzz_no_such_category'." in out
        # No trailing "Use '...' to generate" line when there's nothing to generate from.
        assert "to generate a template" not in out


class TestPrintTemplateTable:
    """print_template_table renders grouped, sorted Rich tables."""

    @pytest.fixture(autouse=True)
    def _widen_console(self, monkeypatch) -> None:
        """Prevent Rich from truncating verbose column headers at 80 cols."""
        monkeypatch.setenv("COLUMNS", "200")

    def test_verbose_adds_tags_and_difficulty_columns(self, capsys) -> None:
        from aiperf.config.templates import list_templates

        templates = list_templates()
        print_template_table(templates, verbose=True)

        out = capsys.readouterr().out
        # Non-verbose headers always present
        assert "Name" in out
        assert "Title" in out
        assert "Description" in out
        # Verbose-only headers
        assert "Tags" in out
        assert "Difficulty" in out
        # Tag values land in the verbose Tags column
        assert "quick-start" in out

    def test_non_verbose_omits_tags_and_difficulty(self, capsys) -> None:
        from aiperf.config.templates import list_templates

        templates = list_templates()
        print_template_table(templates, verbose=False)

        out = capsys.readouterr().out
        assert "Difficulty" not in out
        # Tag values only appear via the verbose Tags column.
        assert "quick-start" not in out

    def test_category_groups_rendered_in_category_order(self, capsys) -> None:
        """Output order must match CATEGORY_ORDER, not template insertion order."""
        from aiperf.config.templates import CATEGORY_ORDER, list_templates

        print_template_table(list_templates(), verbose=False)

        out = capsys.readouterr().out
        present = [c for c in CATEGORY_ORDER if c in out]
        indices = [out.index(c) for c in present]
        assert indices == sorted(indices), (
            f"categories emitted out of order: {present} at {indices}"
        )

    def test_empty_template_list_produces_no_output(self, capsys) -> None:
        print_template_table([], verbose=False)
        assert capsys.readouterr().out == ""
