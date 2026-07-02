# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight for accuracy benchmark/grader optional-dependency checks.

The grader (record-processor daemon) and benchmark loader (dataset-manager)
both raise at instantiation when their optional package (lighteval / deepeval)
is missing. The grader crash isn't propagated, so the user got a raw
multiprocessing traceback and a hung run. The preflight moves both checks into
the main process, before any service spawns, so a missing dependency is a clean
``ConfigurationError`` with a non-zero exit.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aiperf.cli_runner._preflight import _preflight_accuracy_deps
from aiperf.config.loader.errors import ConfigurationError
from aiperf.plugin.enums import PluginType


def _plan(accuracy: object) -> SimpleNamespace:
    return SimpleNamespace(configs=[SimpleNamespace(accuracy=accuracy)])


def _acc(enabled: bool, benchmark: str = "math_500", grader: str | None = None):
    return SimpleNamespace(enabled=enabled, benchmark=benchmark, grader=grader)


def _patch_registry(monkeypatch, *, benchmark_cls, grader_cls, default_grader="g"):
    """Stub the plugin registry: metadata carries default_grader; get_class
    returns benchmark_cls for ACCURACY_BENCHMARK and grader_cls for ACCURACY_GRADER."""
    monkeypatch.setattr(
        "aiperf.plugin.plugins.get_metadata",
        lambda _type, _name: {"default_grader": default_grader},
    )

    def _get_class(plugin_type, _name):
        return (
            benchmark_cls
            if plugin_type == PluginType.ACCURACY_BENCHMARK
            else grader_cls
        )

    monkeypatch.setattr("aiperf.plugin.plugins.get_class", _get_class)


class _Available:
    @classmethod
    def check_available(cls) -> None:
        return None


class TestPreflightAccuracyDeps:
    def test_missing_grader_dep_raises_configuration_error(self, monkeypatch) -> None:
        """A grader whose check_available raises must surface as a clean
        ConfigurationError (not a daemon crash / hang)."""

        class _UnavailableGrader:
            @classmethod
            def check_available(cls) -> None:
                raise RuntimeError("lighteval is not installed; ... 'aiperf[accuracy]'")

        _patch_registry(
            monkeypatch, benchmark_cls=_Available, grader_cls=_UnavailableGrader
        )
        with pytest.raises(ConfigurationError, match=r"aiperf\[accuracy\]"):
            _preflight_accuracy_deps(
                _plan(_acc(enabled=True, grader="lighteval_latex"))
            )

    def test_missing_benchmark_loader_dep_raises_configuration_error(
        self, monkeypatch
    ) -> None:
        """A benchmark loader whose check_available raises (e.g. HellaSwag/BigBench
        without deepeval) must also surface cleanly — its default grader is the
        dep-free exact_match, so only the loader check catches it."""

        class _UnavailableBenchmark:
            @classmethod
            def check_available(cls) -> None:
                raise RuntimeError("deepeval is not installed; ... 'aiperf[accuracy]'")

        _patch_registry(
            monkeypatch, benchmark_cls=_UnavailableBenchmark, grader_cls=_Available
        )
        with pytest.raises(ConfigurationError, match=r"deepeval"):
            _preflight_accuracy_deps(_plan(_acc(enabled=True, benchmark="hellaswag")))

    def test_available_deps_do_not_raise(self, monkeypatch) -> None:
        _patch_registry(monkeypatch, benchmark_cls=_Available, grader_cls=_Available)
        _preflight_accuracy_deps(_plan(_acc(enabled=True, grader="exact_match")))

    def test_skips_when_accuracy_disabled(self, monkeypatch) -> None:
        """Non-accuracy runs must not touch the plugin registry at all."""

        def _boom(*_a, **_k):
            raise AssertionError("registry should not be touched")

        monkeypatch.setattr("aiperf.plugin.plugins.get_class", _boom)
        monkeypatch.setattr("aiperf.plugin.plugins.get_metadata", _boom)
        _preflight_accuracy_deps(_plan(_acc(enabled=False)))
        _preflight_accuracy_deps(_plan(None))

    def test_plugin_without_check_available_is_allowed(self, monkeypatch) -> None:
        """A custom benchmark/grader satisfying only its protocol (no
        check_available) must pass preflight, not raise AttributeError."""

        class _ProtocolOnly:
            pass

        _patch_registry(
            monkeypatch, benchmark_cls=_ProtocolOnly, grader_cls=_ProtocolOnly
        )
        _preflight_accuracy_deps(_plan(_acc(enabled=True, grader="custom")))

    @pytest.mark.parametrize(
        "exc_type", [KeyError, ValueError, ImportError, AttributeError]
    )
    def test_lookup_errors_become_configuration_error(
        self, monkeypatch, exc_type
    ) -> None:
        """Unknown/malformed names (TypeNotFoundError/KeyError/ValueError),
        broken external plugin modules (ImportError), and a module that imports
        but is missing its configured class (AttributeError, from
        PluginEntry.load) must all convert to ConfigurationError, not leak a
        raw traceback."""
        monkeypatch.setattr(
            "aiperf.plugin.plugins.get_metadata",
            lambda _type, _name: {"default_grader": "g"},
        )

        def _raise(*_a, **_k):
            raise exc_type("boom")

        monkeypatch.setattr("aiperf.plugin.plugins.get_class", _raise)
        with pytest.raises(ConfigurationError):
            _preflight_accuracy_deps(_plan(_acc(enabled=True, grader="nope")))

    def test_resolves_default_grader_from_benchmark_metadata(self, monkeypatch) -> None:
        """When grader is unset, the default_grader from benchmark metadata is
        resolved and checked."""
        grader_names: list[str] = []

        def _get_class(plugin_type, name):
            if plugin_type == PluginType.ACCURACY_GRADER:
                grader_names.append(name)
            return _Available

        monkeypatch.setattr(
            "aiperf.plugin.plugins.get_metadata",
            lambda _type, _name: {"default_grader": "lighteval_latex"},
        )
        monkeypatch.setattr("aiperf.plugin.plugins.get_class", _get_class)
        _preflight_accuracy_deps(_plan(_acc(enabled=True, grader=None)))
        assert grader_names == ["lighteval_latex"]

    def test_real_accuracy_config_attribute_contract(self) -> None:
        """Pin the attribute contract against the real AccuracyConfig (not a
        SimpleNamespace mock): if .enabled/.benchmark/.grader are renamed, this
        fails loudly here instead of silently no-opping. GSM8K's benchmark and
        default grader are dependency-free, so a real preflight passes."""
        from aiperf.config.accuracy import AccuracyConfig
        from aiperf.plugin.enums import AccuracyBenchmarkType

        acc = AccuracyConfig(benchmark=AccuracyBenchmarkType.GSM8K)
        # Real registry + real config, no optional deps needed → no raise.
        _preflight_accuracy_deps(_plan(acc))


class TestCheckAvailable:
    """Benchmark loaders and graders report missing optional deps via check_available."""

    def test_code_execution_grader(self, monkeypatch) -> None:
        import aiperf.accuracy.graders.code_execution as ce

        monkeypatch.setattr(ce, "_HAS_LIGHTEVAL_LCB", False)
        with pytest.raises(RuntimeError, match="lighteval is not installed"):
            ce.CodeExecutionGrader.check_available()
        monkeypatch.setattr(ce, "_HAS_LIGHTEVAL_LCB", True)
        ce.CodeExecutionGrader.check_available()

    def test_lighteval_grader(self, monkeypatch) -> None:
        import aiperf.accuracy.graders.lighteval_grader as le

        monkeypatch.setattr(le, "_HAS_LIGHTEVAL", False)
        with pytest.raises(RuntimeError, match="lighteval is not installed"):
            le._LightevalBaseGrader.check_available()
        monkeypatch.setattr(le, "_HAS_LIGHTEVAL", True)
        le._LightevalBaseGrader.check_available()

    def test_hellaswag_benchmark(self, monkeypatch) -> None:
        import aiperf.accuracy.benchmarks.hellaswag as hs

        monkeypatch.setattr(hs, "_HAS_DEEPEVAL", False)
        with pytest.raises(RuntimeError, match="deepeval is not installed"):
            hs.HellaSwagBenchmark.check_available()
        monkeypatch.setattr(hs, "_HAS_DEEPEVAL", True)
        hs.HellaSwagBenchmark.check_available()

    def test_bigbench_benchmark(self, monkeypatch) -> None:
        import aiperf.accuracy.benchmarks.bigbench as bb

        monkeypatch.setattr(bb, "_HAS_DEEPEVAL", False)
        with pytest.raises(RuntimeError, match="deepeval is not installed"):
            bb.BigBenchBenchmark.check_available()
        monkeypatch.setattr(bb, "_HAS_DEEPEVAL", True)
        bb.BigBenchBenchmark.check_available()

    def test_base_grader_check_available_is_noop(self) -> None:
        """Graders/benchmarks with no optional deps are always available."""
        from aiperf.accuracy.graders.exact_match import ExactMatchGrader

        ExactMatchGrader.check_available()
