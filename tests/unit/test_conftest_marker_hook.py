# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pytest import param

from tests.conftest import (
    _PATH_MARKER_MAP,
    pytest_collection_modifyitems,
    pytest_configure,
)


def _make_config(
    markexpr: str,
    args: list[str],
    rootdir: Path = Path("/project"),
) -> MagicMock:
    config = MagicMock(
        spec=["option", "invocation_params", "args", "hook", "_aiperf_enabled_markers"]
    )
    config.option = SimpleNamespace(markexpr=markexpr)
    config.invocation_params = SimpleNamespace(args=args, dir=rootdir)
    config.args = list(args)
    config.hook = MagicMock()
    return config


class _FakeItem:
    def __init__(
        self, name: str, markers: list[str], path: str = "tests/unit/test_example.py"
    ) -> None:
        self.name = name
        self.path = Path(path)
        self._markers = markers

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        self._markers.append(marker.name)

    def iter_markers(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=marker) for marker in self._markers]


_DEFAULT_MARKEXPR = (
    "not fern and not integration and not component_integration and not server_unit"
)


class TestPytestConfigureEarlyReturns:
    def test_no_markexpr_is_noop(self) -> None:
        config = _make_config(markexpr="", args=["tests/integration"])
        pytest_configure(config)
        assert config.option.markexpr == ""

    def test_none_markexpr_is_noop(self) -> None:
        config = _make_config(markexpr="", args=["tests/integration"])
        config.option.markexpr = None
        pytest_configure(config)
        assert config.option.markexpr is None

    def test_unrelated_path_is_noop(self) -> None:
        config = _make_config(markexpr=_DEFAULT_MARKEXPR, args=["tests/unit/common"])
        pytest_configure(config)
        assert config.option.markexpr == _DEFAULT_MARKEXPR


class TestPytestConfigureMarkerPathDefaults:
    @pytest.mark.parametrize(
        "markexpr, expected_path, expected_enabled",
        [
            param("integration", "tests/integration", {"integration"}, id="integration"),
            param(
                "integration and not slow",
                "tests/integration",
                {"integration"},
                id="integration-complex",
            ),
            param(
                "component_integration",
                "tests/component_integration",
                {"component_integration"},
                id="component-integration",
            ),
            param(
                "component_integration and not stress",
                "tests/component_integration",
                {"component_integration"},
                id="component-integration-complex",
            ),
            param(
                "integration or component_integration",
                ["tests/integration", "tests/component_integration"],
                {"integration", "component_integration"},
                id="both-integration-suites",
            ),
        ],
    )  # fmt: skip
    def test_simple_marker_expression_sets_default_path(
        self, markexpr: str, expected_path: str | list[str], expected_enabled: set[str]
    ) -> None:
        config = _make_config(markexpr=markexpr, args=[])
        pytest_configure(config)
        expected_args = (
            expected_path if isinstance(expected_path, list) else [expected_path]
        )
        assert config.args == expected_args
        assert config._aiperf_enabled_markers == expected_enabled

    def test_simple_marker_expression_replaces_tests_root_default(self) -> None:
        config = _make_config(markexpr="integration", args=["tests"])
        pytest_configure(config)
        assert config.args == ["tests/integration"]
        assert config._aiperf_enabled_markers == {"integration"}

    @pytest.mark.parametrize(
        "markexpr",
        [
            param("server_unit", id="server-unit"),
            param("fern", id="fern"),
            param("not integration", id="negative-expression"),
            param("not (integration or component_integration)", id="grouped-negative-expression"),
            param("unknown_marker", id="unknown-marker"),
        ],
    )  # fmt: skip
    def test_complex_or_unknown_marker_expression_uses_unit_default(
        self, markexpr: str
    ) -> None:
        config = _make_config(markexpr=markexpr, args=[])
        pytest_configure(config)
        assert config.args == ["tests/unit"]
        assert config._aiperf_enabled_markers == set()


class TestPytestConfigureDefaultPaths:
    @pytest.mark.parametrize(
        "args",
        [
            param([], id="no-args"),
            param(["tests"], id="tests"),
            param(["tests/"], id="tests-slash"),
        ],
    )  # fmt: skip
    def test_broad_invocations_collect_unit_tests(self, args: list[str]) -> None:
        config = _make_config(markexpr=_DEFAULT_MARKEXPR, args=args)
        pytest_configure(config)
        assert config.args == ["tests/unit"]
        assert config.option.markexpr == _DEFAULT_MARKEXPR

    def test_absolute_tests_path_collects_unit_tests(self, tmp_path: Path) -> None:
        rootdir = tmp_path
        config = _make_config(
            markexpr=_DEFAULT_MARKEXPR,
            args=[str(rootdir / "tests")],
            rootdir=rootdir,
        )
        pytest_configure(config)
        assert config.args == ["tests/unit"]
        assert config.option.markexpr == _DEFAULT_MARKEXPR

    @pytest.mark.parametrize(
        "args",
        [
            param(["tests/integration"], id="integration"),
            param(["tests/unit/server"], id="server-unit"),
            param(["tests/unit/test_foo.py::test_bar"], id="node-id"),
        ],
    )  # fmt: skip
    def test_specific_targets_are_unchanged(self, args: list[str]) -> None:
        config = _make_config(markexpr=_DEFAULT_MARKEXPR, args=args)
        pytest_configure(config)
        assert config.args == args


class TestPytestConfigureMarkerOptIn:
    @pytest.mark.parametrize(
        "args, expected_enabled",
        [
            param(["tests/integration"], {"integration"}, id="integration"),
            param(
                ["tests/component_integration"],
                {"component_integration"},
                id="component-integration",
            ),
            param(["tests/unit/server"], {"server_unit"}, id="server-unit"),
            param(["tests/unit/fern"], {"fern"}, id="fern"),
        ],
    )  # fmt: skip
    def test_path_enables_expected_markers(
        self, args: list[str], expected_enabled: set[str]
    ) -> None:
        config = _make_config(markexpr="", args=args)
        pytest_configure(config)
        assert config._aiperf_enabled_markers == expected_enabled

    def test_specific_file_under_prefix_enables_marker(self) -> None:
        config = _make_config(
            markexpr="",
            args=["tests/component_integration/timing/test_dag_join_end_to_end.py"],
        )
        pytest_configure(config)
        assert config._aiperf_enabled_markers == {"component_integration"}

    def test_path_prefix_requires_directory_boundary(self) -> None:
        config = _make_config(markexpr="", args=["tests/integration_extra/test_foo.py"])
        pytest_configure(config)
        assert config._aiperf_enabled_markers == set()

    def test_node_id_stripped_for_matching(self) -> None:
        config = _make_config(
            markexpr="",
            args=["tests/integration/test_foo.py::TestClass::test_method"],
        )
        pytest_configure(config)
        assert config._aiperf_enabled_markers == {"integration"}

    def test_user_markexpr_is_not_modified_for_explicit_path(self) -> None:
        config = _make_config(markexpr="not integration", args=["tests/integration"])
        pytest_configure(config)
        assert config.option.markexpr == "not integration"


class TestPytestConfigureAbsolutePaths:
    def test_absolute_path_resolved_to_relative(self, tmp_path: Path) -> None:
        rootdir = tmp_path
        abs_path = str(rootdir / "tests" / "integration" / "test_x.py")
        config = _make_config(markexpr="", args=[abs_path], rootdir=rootdir)
        pytest_configure(config)
        assert config._aiperf_enabled_markers == {"integration"}


class TestPytestCollectionModifyItems:
    def test_integration_path_item_is_auto_marked(self) -> None:
        config = _make_config(markexpr="", args=["tests/integration"])
        pytest_configure(config)
        integration = _FakeItem(
            "integration", [], path="tests/integration/test_example.py"
        )
        items = [integration]

        pytest_collection_modifyitems(config, items)

        assert items == [integration]
        assert [marker.name for marker in integration.iter_markers()] == ["integration"]
        config.hook.pytest_deselected.assert_not_called()

    def test_absolute_path_item_is_auto_marked(self) -> None:
        rootdir = Path("/project")
        config = _make_config(
            markexpr="", args=[str(rootdir / "tests" / "integration")], rootdir=rootdir
        )
        pytest_configure(config)
        integration = _FakeItem(
            "integration", [], path=str(rootdir / "tests/integration/test_example.py")
        )
        items = [integration]

        pytest_collection_modifyitems(config, items)

        assert items == [integration]
        assert [marker.name for marker in integration.iter_markers()] == ["integration"]
        config.hook.pytest_deselected.assert_not_called()

    def test_component_integration_path_item_is_auto_marked(self) -> None:
        config = _make_config(markexpr="", args=["tests/component_integration"])
        pytest_configure(config)
        component = _FakeItem(
            "component", [], path="tests/component_integration/test_example.py"
        )
        items = [component]

        pytest_collection_modifyitems(config, items)

        assert items == [component]
        assert [marker.name for marker in component.iter_markers()] == [
            "component_integration"
        ]
        config.hook.pytest_deselected.assert_not_called()

    @pytest.mark.parametrize(
        "path, marker",
        [
            param("tests/unit/server/test_example.py", "server_unit", id="server-unit"),
            param("tests/unit/fern/test_example.py", "fern", id="fern"),
        ],
    )  # fmt: skip
    def test_unit_subsuite_path_item_is_auto_marked(
        self, path: str, marker: str
    ) -> None:
        config = _make_config(markexpr="", args=[path])
        pytest_configure(config)
        item = _FakeItem(marker, [], path=path)
        items = [item]

        pytest_collection_modifyitems(config, items)

        assert items == [item]
        assert [mark.name for mark in item.iter_markers()] == [marker]
        config.hook.pytest_deselected.assert_not_called()

    @pytest.mark.parametrize(
        "path",
        [
            param("tests/integration_extra/test_example.py", id="integration-extra"),
            param("tests/component_integration_extra/test_example.py", id="component-extra"),
            param("tests/unit/server_extra/test_example.py", id="server-extra"),
            param("tests/unit/fern_extra/test_example.py", id="fern-extra"),
        ],
    )  # fmt: skip
    def test_auto_marking_requires_directory_boundary(self, path: str) -> None:
        config = _make_config(markexpr="", args=[path])
        pytest_configure(config)
        item = _FakeItem("near_prefix", [], path=path)
        items = [item]

        pytest_collection_modifyitems(config, items)

        assert items == [item]
        assert [mark.name for mark in item.iter_markers()] == []
        config.hook.pytest_deselected.assert_not_called()

    def test_default_excluded_markers_are_deselected(self) -> None:
        config = _make_config(markexpr="", args=["tests/unit"])
        pytest_configure(config)
        unit = _FakeItem("unit", [])
        server = _FakeItem("server", ["server_unit"])
        performance = _FakeItem("performance", ["performance"])
        items = [unit, server, performance]

        pytest_collection_modifyitems(config, items)

        assert items == [unit]
        config.hook.pytest_deselected.assert_called_once_with(
            items=[server, performance]
        )

    def test_explicit_path_enabled_marker_is_not_deselected(self) -> None:
        config = _make_config(markexpr="", args=["tests/unit/server"])
        pytest_configure(config)
        server = _FakeItem("server", ["server_unit"])
        performance = _FakeItem("performance", ["performance"])
        items = [server, performance]

        pytest_collection_modifyitems(config, items)

        assert items == [server]
        config.hook.pytest_deselected.assert_called_once_with(items=[performance])

    def test_markexpr_still_deselects_unmentioned_default_markers(self) -> None:
        config = _make_config(markexpr="integration", args=[])
        pytest_configure(config)
        integration = _FakeItem("integration", ["integration"])
        slow_integration = _FakeItem("slow_integration", ["integration", "slow"])
        stress_integration = _FakeItem("stress_integration", ["integration", "stress"])
        items = [integration, slow_integration, stress_integration]

        pytest_collection_modifyitems(config, items)

        assert items == [integration]
        config.hook.pytest_deselected.assert_called_once_with(
            items=[slow_integration, stress_integration]
        )

    def test_positive_markexpr_marker_opts_into_default_exclusion(self) -> None:
        config = _make_config(markexpr="integration and stress", args=[])
        pytest_configure(config)
        integration = _FakeItem("integration", ["integration"])
        slow_integration = _FakeItem("slow_integration", ["integration", "slow"])
        stress_integration = _FakeItem("stress_integration", ["integration", "stress"])
        items = [integration, slow_integration, stress_integration]

        pytest_collection_modifyitems(config, items)

        assert items == [integration, stress_integration]
        config.hook.pytest_deselected.assert_called_once_with(items=[slow_integration])

    def test_grouped_negative_markexpr_does_not_opt_in(self) -> None:
        config = _make_config(
            markexpr="not (integration or component_integration)", args=[]
        )
        pytest_configure(config)
        unit = _FakeItem("unit", [])
        integration = _FakeItem("integration", ["integration"])
        component = _FakeItem("component", ["component_integration"])
        items = [unit, integration, component]

        pytest_collection_modifyitems(config, items)

        assert items == [unit]
        config.hook.pytest_deselected.assert_called_once_with(
            items=[integration, component]
        )

    def test_negative_markexpr_marker_does_not_opt_in(self) -> None:
        config = _make_config(markexpr="integration and not stress", args=[])
        pytest_configure(config)
        integration = _FakeItem("integration", ["integration"])
        stress_integration = _FakeItem("stress_integration", ["integration", "stress"])
        items = [integration, stress_integration]

        pytest_collection_modifyitems(config, items)

        assert items == [integration]
        config.hook.pytest_deselected.assert_called_once_with(
            items=[stress_integration]
        )


class TestPathMarkerMapConsistency:
    def test_all_entries_start_with_tests(self) -> None:
        for path_prefix, _ in _PATH_MARKER_MAP:
            assert path_prefix.startswith("tests/")

    def test_all_markers_are_nonempty_strings(self) -> None:
        for _, markers in _PATH_MARKER_MAP:
            assert markers
            for marker in markers:
                assert isinstance(marker, str) and marker
