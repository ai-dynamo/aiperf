# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pytest import param

from tests.conftest import _PATH_MARKER_MAP, pytest_configure


def _make_config(
    markexpr: str,
    args: list[str],
    rootdir: Path = Path("/project"),
) -> MagicMock:
    """Build a minimal mock pytest.Config for pytest_configure."""
    config = MagicMock(spec=["option", "invocation_params"])
    config.option = SimpleNamespace(markexpr=markexpr)
    config.invocation_params = SimpleNamespace(args=args, dir=rootdir)
    return config


_DEFAULT_MARKEXPR = (
    "not k8s and not gpu and not vllm and not dynamo"
    " and not integration and not component_integration"
)


class TestPytestConfigureEarlyReturns:
    def test_no_markexpr_is_noop(self) -> None:
        config = _make_config(markexpr="", args=["tests/kubernetes"])
        pytest_configure(config)
        assert config.option.markexpr == ""

    def test_none_markexpr_is_noop(self) -> None:
        config = _make_config(markexpr="", args=["tests/kubernetes"])
        config.option.markexpr = None
        pytest_configure(config)
        assert config.option.markexpr is None

    def test_no_args_is_noop(self) -> None:
        config = _make_config(markexpr=_DEFAULT_MARKEXPR, args=[])
        pytest_configure(config)
        assert config.option.markexpr == _DEFAULT_MARKEXPR

    def test_unrelated_path_is_noop(self) -> None:
        config = _make_config(markexpr=_DEFAULT_MARKEXPR, args=["tests/unit/common"])
        pytest_configure(config)
        assert config.option.markexpr == _DEFAULT_MARKEXPR


class TestPytestConfigureMarkerStripping:
    @pytest.mark.parametrize(
        "args, expected_stripped",
        [
            param(
                ["tests/kubernetes"],
                {"k8s", "gpu", "vllm", "dynamo"},
                id="kubernetes-root-enables-all-children",
            ),
            param(
                ["tests/kubernetes/gpu"],
                {"k8s", "gpu", "vllm", "dynamo"},
                id="gpu-enables-k8s-and-children",
            ),
            param(
                ["tests/kubernetes/gpu/vllm"],
                {"k8s", "gpu", "vllm"},
                id="vllm-enables-parents",
            ),
            param(
                ["tests/kubernetes/gpu/dynamo"],
                {"k8s", "gpu", "dynamo"},
                id="dynamo-enables-parents",
            ),
            param(
                ["tests/integration"],
                {"integration"},
                id="integration-only",
            ),
            param(
                ["tests/component_integration"],
                {"component_integration"},
                id="component-integration-only",
            ),
        ],
    )  # fmt: skip
    def test_path_strips_expected_markers(
        self, args: list[str], expected_stripped: set[str]
    ) -> None:
        config = _make_config(markexpr=_DEFAULT_MARKEXPR, args=args)
        pytest_configure(config)

        remaining = config.option.markexpr
        for marker in expected_stripped:
            assert f"not {marker}" not in remaining

        # Markers NOT in expected_stripped should still be present
        all_markers = {
            "k8s",
            "gpu",
            "vllm",
            "dynamo",
            "integration",
            "component_integration",
        }
        for marker in all_markers - expected_stripped:
            assert f"not {marker}" in remaining

    def test_specific_file_under_prefix(self) -> None:
        config = _make_config(
            markexpr=_DEFAULT_MARKEXPR,
            args=["tests/kubernetes/gpu/vllm/test_foo.py"],
        )
        pytest_configure(config)
        assert "not k8s" not in config.option.markexpr
        assert "not gpu" not in config.option.markexpr
        assert "not vllm" not in config.option.markexpr

    def test_node_id_stripped_for_matching(self) -> None:
        config = _make_config(
            markexpr=_DEFAULT_MARKEXPR,
            args=["tests/integration/test_foo.py::TestClass::test_method"],
        )
        pytest_configure(config)
        assert "not integration" not in config.option.markexpr

    def test_all_markers_stripped_produces_empty_expr(self) -> None:
        markexpr = "not k8s and not gpu"
        config = _make_config(markexpr=markexpr, args=["tests/kubernetes"])
        pytest_configure(config)
        assert config.option.markexpr == ""

    def test_single_marker_stripped(self) -> None:
        config = _make_config(markexpr="not integration", args=["tests/integration"])
        pytest_configure(config)
        assert config.option.markexpr == ""

    def test_preserves_non_excluded_clauses(self) -> None:
        markexpr = "not k8s and not slow and not gpu"
        config = _make_config(markexpr=markexpr, args=["tests/kubernetes/gpu"])
        pytest_configure(config)
        assert config.option.markexpr == "not slow"


class TestPytestConfigureAbsolutePaths:
    def test_absolute_path_resolved_to_relative(self, tmp_path: Path) -> None:
        rootdir = tmp_path
        abs_path = str(rootdir / "tests" / "kubernetes" / "gpu" / "test_x.py")
        config = _make_config(
            markexpr=_DEFAULT_MARKEXPR, args=[abs_path], rootdir=rootdir
        )
        pytest_configure(config)
        assert "not k8s" not in config.option.markexpr
        assert "not gpu" not in config.option.markexpr


class TestPathMarkerMapConsistency:
    def test_all_entries_start_with_tests(self) -> None:
        for path_prefix, _ in _PATH_MARKER_MAP:
            assert path_prefix.startswith("tests/")

    def test_all_markers_are_nonempty_strings(self) -> None:
        for _, markers in _PATH_MARKER_MAP:
            assert markers
            for m in markers:
                assert isinstance(m, str) and m
