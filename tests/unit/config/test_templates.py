# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the config template registry."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.config.templates import (
    CATEGORY_ORDER,
    apply_overrides,
    get_template,
    list_templates,
    load_template_content,
    parse_template_meta,
    search_templates,
    strip_spdx_header,
)


class TestParseTemplateMeta:
    """Tests for parsing # @template comment blocks."""

    def test_parses_all_fields(self, tmp_path: Path) -> None:
        p = tmp_path / "test.yaml"
        p.write_text(
            "# @template\n"
            "# title: Test Title\n"
            "# description: A test template.\n"
            "# category: Getting Started\n"
            "# tags: foo, bar\n"
            "# difficulty: intermediate\n"
            "# features: feat1, feat2\n"
            "model: x\n"
        )
        info = parse_template_meta(p)
        assert info.name == "test"
        assert info.title == "Test Title"
        assert info.category == "Getting Started"
        assert info.tags == ("foo", "bar")
        assert info.difficulty == "intermediate"
        assert info.features == ("feat1", "feat2")

    def test_optional_fields_default(self, tmp_path: Path) -> None:
        p = tmp_path / "test.yaml"
        p.write_text(
            "# @template\n# title: T\n# description: D\n# category: Advanced\nmodel: x\n"
        )
        info = parse_template_meta(p)
        assert info.tags == ()
        assert info.difficulty == "beginner"
        assert info.features == ()

    @pytest.mark.parametrize(
        "content, match",
        [
            param("model: x\n", "missing", id="no-sentinel"),
            param("# @template\n# title: T\n# description: D\nmodel: x\n", "category", id="missing-field"),
            param("# @template\n# title: T\n# description: D\n# category: Bogus\n", "Bogus", id="bad-category"),
            param("# @template\n# title: T\n# description: D\n# category: Advanced\n# difficulty: expert\n", "expert", id="bad-difficulty"),
        ],
    )  # fmt: skip
    def test_invalid_raises(self, tmp_path: Path, content: str, match: str) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(content)
        with pytest.raises((ValueError, Exception), match=match):
            parse_template_meta(p)


class TestRegistry:
    def test_discovers_all_templates(self) -> None:
        assert len(list_templates()) == 19

    def test_unique_names(self) -> None:
        names = [t.name for t in list_templates()]
        assert len(names) == len(set(names))

    def test_category_order_preserved(self) -> None:
        cats = []
        for t in list_templates():
            if not cats or cats[-1] != t.category:
                cats.append(t.category)
        for i, cat in enumerate(cats):
            assert cat == CATEGORY_ORDER[i]


class TestGetTemplate:
    def test_valid(self) -> None:
        assert get_template("minimal").title == "Minimal Configuration"

    def test_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown template"):
            get_template("nonexistent")


class TestListTemplates:
    @pytest.mark.parametrize(
        "category, expected_min",
        [
            param("Getting Started", 2, id="getting-started"),
            param("Load Testing", 3, id="load-testing"),
            param("Sweep", 2, id="sweep"),
        ],
    )  # fmt: skip
    def test_filter_by_category(self, category: str, expected_min: int) -> None:
        assert len(list_templates(category=category)) >= expected_min

    def test_filter_by_tag(self) -> None:
        assert len(list_templates(tag="sweep")) >= 2

    def test_no_match(self) -> None:
        assert list_templates(category="Nonexistent") == []


class TestSearch:
    def test_name_ranks_first(self) -> None:
        assert search_templates("latency")[0].name == "latency_test"

    def test_no_match(self) -> None:
        assert search_templates("xyznonexistent") == []


class TestStripSpdxHeader:
    def test_strips_spdx_only(self) -> None:
        content = (
            "# SPDX-FileCopyrightText: X\n"
            "# SPDX-License-Identifier: Y\n"
            "# @template\n"
            "# title: Foo\n"
            "model: x\n"
        )
        result = strip_spdx_header(content)
        assert "SPDX" not in result
        assert result.startswith("# @template")
        assert "model: x" in result

    def test_no_spdx_passthrough(self) -> None:
        content = "# @template\nmodel: x\n"
        assert strip_spdx_header(content) == content


class TestApplyOverrides:
    def test_no_overrides_unchanged(self) -> None:
        content = load_template_content("minimal")
        assert apply_overrides(content, {}) == content

    def test_model_override_singular(self) -> None:
        content = load_template_content("minimal")  # uses model: (singular)
        result = apply_overrides(content, {"model": "new/model"})
        assert "new/model" in result
        assert "meta-llama" not in result

    def test_model_override_plural(self) -> None:
        content = load_template_content("goodput_slo")  # uses models: (plural)
        result = apply_overrides(content, {"models": ["new/model"]})
        assert "new/model" in result

    def test_url_override(self) -> None:
        content = load_template_content("goodput_slo")  # uses urls: (plural)
        result = apply_overrides(content, {"endpoint": {"urls": ["http://test:8000"]}})
        assert "http://test:8000" in result

    def test_preserves_comment_header(self) -> None:
        content = load_template_content("minimal")
        result = apply_overrides(content, {"models": ["test"]})
        assert "# @template" in result
        assert "# Minimal Configuration" in result

    def test_arbitrary_field(self) -> None:
        content = load_template_content("minimal")
        result = apply_overrides(content, {"random_seed": 99})
        assert "random_seed: 99" in result
