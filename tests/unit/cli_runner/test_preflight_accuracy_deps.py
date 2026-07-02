# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight for accuracy grader optional-dependency checks.

A grader backed by an optional package (lighteval) used to raise at
instantiation inside the daemon record-processor: the crash wasn't propagated,
so the user got a raw multiprocessing traceback and the main process hung. The
preflight moves that check into the main process, before any service spawns, so
a missing dependency is a clean ``ConfigurationError`` with a non-zero exit.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aiperf.cli_runner._preflight import _preflight_accuracy_grader_deps
from aiperf.config.loader.errors import ConfigurationError


def _plan(accuracy: object) -> SimpleNamespace:
    return SimpleNamespace(configs=[SimpleNamespace(accuracy=accuracy)])


def _acc(enabled: bool, benchmark: str = "math_500", grader: str | None = None):
    return SimpleNamespace(enabled=enabled, benchmark=benchmark, grader=grader)


class TestPreflightAccuracyGraderDeps:
    def test_missing_dep_raises_configuration_error(self, monkeypatch) -> None:
        """A grader whose check_available raises must surface as a clean
        ConfigurationError (not a daemon crash / hang)."""

        class _UnavailableGrader:
            @classmethod
            def check_available(cls) -> None:
                raise RuntimeError("lighteval is not installed; ... 'aiperf[accuracy]'")

        monkeypatch.setattr(
            "aiperf.plugin.plugins.get_class",
            lambda _type, _name: _UnavailableGrader,
        )
        with pytest.raises(ConfigurationError, match=r"aiperf\[accuracy\]"):
            _preflight_accuracy_grader_deps(
                _plan(_acc(enabled=True, grader="lighteval_latex"))
            )

    def test_available_grader_does_not_raise(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "aiperf.plugin.plugins.get_class",
            lambda _type, _name: MagicMock(check_available=lambda: None),
        )
        _preflight_accuracy_grader_deps(_plan(_acc(enabled=True, grader="exact_match")))

    def test_skips_when_accuracy_disabled(self, monkeypatch) -> None:
        """Non-accuracy runs must not touch the grader registry at all."""
        called = False

        def _boom(*_a, **_k):
            nonlocal called
            called = True
            raise AssertionError("get_class should not be called")

        monkeypatch.setattr("aiperf.plugin.plugins.get_class", _boom)
        _preflight_accuracy_grader_deps(_plan(_acc(enabled=False)))
        _preflight_accuracy_grader_deps(_plan(None))
        assert called is False

    def test_resolves_default_grader_from_benchmark_metadata(self, monkeypatch) -> None:
        """When grader is unset, the default_grader from benchmark metadata is
        resolved and checked."""
        seen: dict[str, str] = {}

        def _get_metadata(_type, name):
            return {"default_grader": "lighteval_latex"}

        def _get_class(_type, name):
            seen["grader"] = name
            return MagicMock(check_available=lambda: None)

        monkeypatch.setattr("aiperf.plugin.plugins.get_metadata", _get_metadata)
        monkeypatch.setattr("aiperf.plugin.plugins.get_class", _get_class)
        _preflight_accuracy_grader_deps(_plan(_acc(enabled=True, grader=None)))
        assert seen["grader"] == "lighteval_latex"


class TestGraderCheckAvailable:
    """The grader classes report missing optional deps via check_available."""

    def test_code_execution_check_available(self, monkeypatch) -> None:
        import aiperf.accuracy.graders.code_execution as ce

        monkeypatch.setattr(ce, "_HAS_LIGHTEVAL_LCB", False)
        with pytest.raises(RuntimeError, match="lighteval is not installed"):
            ce.CodeExecutionGrader.check_available()

        monkeypatch.setattr(ce, "_HAS_LIGHTEVAL_LCB", True)
        ce.CodeExecutionGrader.check_available()  # no raise

    def test_lighteval_grader_check_available(self, monkeypatch) -> None:
        import aiperf.accuracy.graders.lighteval_grader as le

        monkeypatch.setattr(le, "_HAS_LIGHTEVAL", False)
        with pytest.raises(RuntimeError, match="lighteval is not installed"):
            le._LightevalBaseGrader.check_available()

        monkeypatch.setattr(le, "_HAS_LIGHTEVAL", True)
        le._LightevalBaseGrader.check_available()  # no raise

    def test_base_grader_check_available_is_noop(self) -> None:
        """Graders with no optional deps are always available."""
        from aiperf.accuracy.graders.exact_match import ExactMatchGrader

        ExactMatchGrader.check_available()  # no raise
