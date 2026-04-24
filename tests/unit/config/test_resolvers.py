# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for config resolver chain and individual resolvers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aiperf.config import BenchmarkConfig
from aiperf.config.artifacts import GpuTelemetryConfig
from aiperf.config.benchmark import BenchmarkRun
from aiperf.config.models import TokenizerConfig
from aiperf.config.resolvers import (
    ArtifactDirResolver,
    CommConfigResolver,
    ConfigResolver,
    ConfigResolverChain,
    DatasetResolver,
    GpuMetricsResolver,
    TimingResolver,
    TokenizerResolver,
    build_default_resolver_chain,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_run(config: object, *, artifact_dir: Path | None = None) -> BenchmarkRun:
    """Build a minimal BenchmarkRun wrapping a config."""
    return BenchmarkRun(
        benchmark_id="test-run",
        cfg=config,
        artifact_dir=artifact_dir or Path("/tmp/test-artifacts"),
    )


@pytest.fixture()
def minimal_config():
    """Minimal BenchmarkConfig with synthetic dataset and concurrency phase."""
    from aiperf.config import BenchmarkConfig

    return BenchmarkConfig(
        models=["test-model"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets={"main": {"type": "synthetic", "entries": 10, "prompts": {"isl": 32}}},
        phases={"default": {"type": "concurrency", "duration": 60, "concurrency": 1}},
    )


@pytest.fixture()
def run_with_config(minimal_config, tmp_path):
    return _make_run(minimal_config, artifact_dir=tmp_path / "artifacts")


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestConfigResolverProtocol:
    def test_all_resolvers_satisfy_protocol(self):
        resolvers = [
            ArtifactDirResolver(),
            TokenizerResolver(),
            GpuMetricsResolver(),
            DatasetResolver(),
            TimingResolver(),
        ]
        for r in resolvers:
            assert isinstance(r, ConfigResolver)

    def test_custom_resolver_satisfies_protocol(self):
        class MyResolver:
            def resolve(self, run: BenchmarkRun) -> None:
                pass

        assert isinstance(MyResolver(), ConfigResolver)


# ---------------------------------------------------------------------------
# ConfigResolverChain
# ---------------------------------------------------------------------------


class TestConfigResolverChain:
    def test_resolve_all_calls_resolvers_in_order(self, run_with_config):
        call_order: list[str] = []

        class RecordingResolver:
            def __init__(self, name: str) -> None:
                self._name = name

            def resolve(self, run: BenchmarkRun) -> None:
                call_order.append(self._name)

        chain = ConfigResolverChain(
            [
                RecordingResolver("first"),
                RecordingResolver("second"),
                RecordingResolver("third"),
            ]
        )
        chain.resolve_all(run_with_config)
        assert call_order == ["first", "second", "third"]

    def test_empty_chain_is_noop(self, run_with_config):
        chain = ConfigResolverChain([])
        chain.resolve_all(run_with_config)

    def test_resolver_exception_propagates(self, run_with_config):
        class FailingResolver:
            def resolve(self, run: BenchmarkRun) -> None:
                raise ValueError("boom")

        chain = ConfigResolverChain([FailingResolver()])
        with pytest.raises(ValueError, match="boom"):
            chain.resolve_all(run_with_config)


# ---------------------------------------------------------------------------
# ArtifactDirResolver
# ---------------------------------------------------------------------------


class TestArtifactDirResolver:
    def test_creates_directory(self, minimal_config, tmp_path):
        target = tmp_path / "nested" / "artifacts"
        run = _make_run(minimal_config, artifact_dir=target)

        ArtifactDirResolver().resolve(run)

        assert target.exists()
        assert target.is_dir()
        assert run.resolved.artifact_dir_created is True

    def test_resolves_to_absolute_path(self, minimal_config, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run = _make_run(minimal_config, artifact_dir=Path("relative/dir"))

        ArtifactDirResolver().resolve(run)

        assert run.artifact_dir.is_absolute()
        assert run.resolved.artifact_dir_created is True

    def test_idempotent_on_existing_dir(self, minimal_config, tmp_path):
        target = tmp_path / "artifacts"
        target.mkdir()
        run = _make_run(minimal_config, artifact_dir=target)

        ArtifactDirResolver().resolve(run)

        assert run.resolved.artifact_dir_created is True


# ---------------------------------------------------------------------------
# TokenizerResolver
# ---------------------------------------------------------------------------


class TestTokenizerResolver:
    def test_skips_when_no_tokenizer(self, run_with_config):
        run_with_config.cfg = run_with_config.cfg.model_copy(update={"tokenizer": None})

        TokenizerResolver().resolve(run_with_config)

        assert run_with_config.resolved.tokenizer_names is None

    def test_calls_validator_when_tokenizer_set(self, minimal_config, tmp_path):
        config = minimal_config.model_copy(
            update={"tokenizer": TokenizerConfig(name="test-tok")}
        )
        run = _make_run(config, artifact_dir=tmp_path)

        with patch(
            "aiperf.common.tokenizer_validator.validate_tokenizer_early",
            return_value={"test-model": "resolved-tok"},
        ) as mock_validate:
            TokenizerResolver().resolve(run)

        assert run.resolved.tokenizer_names == {"test-model": "resolved-tok"}
        mock_validate.assert_called_once()


# ---------------------------------------------------------------------------
# GpuMetricsResolver
# ---------------------------------------------------------------------------


class TestGpuMetricsResolver:
    def test_skips_when_no_metrics_file(self, run_with_config):
        GpuMetricsResolver().resolve(run_with_config)
        assert run_with_config.resolved.gpu_custom_metrics is None

    def test_validates_csv_when_configured(self, minimal_config, tmp_path):
        csv_file = tmp_path / "metrics.csv"
        csv_file.write_text("# header\n")

        config = minimal_config.model_copy(
            update={"gpu_telemetry": GpuTelemetryConfig(metrics_file=csv_file)}
        )
        run = _make_run(config, artifact_dir=tmp_path)

        mock_instance = MagicMock()
        mock_instance.build_custom_metrics_from_csv.return_value = (
            [("GPU Power", "gpu_power", "W")],
            {"DCGM_FI_DEV_POWER": "gpu_power"},
        )
        with patch(
            "aiperf.gpu_telemetry.metrics_config.MetricsConfigLoader",
            return_value=mock_instance,
        ):
            GpuMetricsResolver().resolve(run)

        assert run.resolved.gpu_custom_metrics == [("GPU Power", "gpu_power", "W")]
        assert run.resolved.gpu_dcgm_mappings == {"DCGM_FI_DEV_POWER": "gpu_power"}
        mock_instance.build_custom_metrics_from_csv.assert_called_once_with(csv_file)

    def test_propagates_error(self, minimal_config, tmp_path):
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("")

        config = minimal_config.model_copy(
            update={"gpu_telemetry": GpuTelemetryConfig(metrics_file=csv_file)}
        )
        run = _make_run(config, artifact_dir=tmp_path)

        mock_instance = MagicMock()
        mock_instance.build_custom_metrics_from_csv.side_effect = ValueError("bad csv")
        with (
            patch(
                "aiperf.gpu_telemetry.metrics_config.MetricsConfigLoader",
                return_value=mock_instance,
            ),
            pytest.raises(ValueError, match="bad csv"),
        ):
            GpuMetricsResolver().resolve(run)


# ---------------------------------------------------------------------------
# DatasetResolver
# ---------------------------------------------------------------------------


class TestDatasetResolver:
    def test_skips_synthetic_datasets(self, run_with_config):
        DatasetResolver().resolve(run_with_config)
        assert run_with_config.resolved.dataset_file_paths is None

    def test_resolves_file_dataset_paths(self, tmp_path):
        dataset_file = tmp_path / "data.jsonl"
        dataset_file.write_text('{"prompt": "hello"}\n')

        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets={"main": {"type": "file", "path": str(dataset_file)}},
            phases={
                "default": {"type": "concurrency", "requests": 10, "concurrency": 1}
            },
        )
        run = _make_run(config, artifact_dir=tmp_path / "out")

        DatasetResolver().resolve(run)

        assert run.resolved.dataset_file_paths is not None
        assert "main" in run.resolved.dataset_file_paths
        assert run.resolved.dataset_file_paths["main"].is_absolute()

    def test_raises_on_missing_file(self, tmp_path):
        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets={"main": {"type": "file", "path": "/nonexistent/data.jsonl"}},
            phases={
                "default": {"type": "concurrency", "requests": 10, "concurrency": 1}
            },
        )
        run = _make_run(config, artifact_dir=tmp_path / "out")

        with pytest.raises(FileNotFoundError, match="Dataset 'main' file not found"):
            DatasetResolver().resolve(run)

    def test_mixed_datasets_only_resolves_file_based(self, tmp_path):
        dataset_file = tmp_path / "data.jsonl"
        dataset_file.write_text('{"prompt": "hello"}\n')

        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets={
                "synth": {"type": "synthetic", "entries": 10, "prompts": {"isl": 32}},
                "real": {"type": "file", "path": str(dataset_file)},
            },
            phases={
                "default": {"type": "concurrency", "requests": 10, "concurrency": 1}
            },
        )
        run = _make_run(config, artifact_dir=tmp_path / "out")

        DatasetResolver().resolve(run)

        assert run.resolved.dataset_file_paths is not None
        assert "real" in run.resolved.dataset_file_paths
        assert "synth" not in run.resolved.dataset_file_paths


# ---------------------------------------------------------------------------
# TimingResolver
# ---------------------------------------------------------------------------


class TestTimingResolver:
    def test_sums_durations(self, tmp_path):
        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets={
                "main": {"type": "synthetic", "entries": 10, "prompts": {"isl": 32}}
            },
            phases={
                "warmup": {
                    "type": "concurrency",
                    "duration": 30,
                    "concurrency": 1,
                    "exclude_from_results": True,
                },
                "main": {"type": "concurrency", "duration": 120, "concurrency": 4},
            },
        )
        run = _make_run(config, artifact_dir=tmp_path)

        TimingResolver().resolve(run)

        assert run.resolved.total_expected_duration == 150.0

    def test_includes_grace_periods(self, tmp_path):
        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets={
                "main": {"type": "synthetic", "entries": 10, "prompts": {"isl": 32}}
            },
            phases={
                "main": {
                    "type": "concurrency",
                    "duration": 60,
                    "grace_period": 10,
                    "concurrency": 1,
                },
            },
        )
        run = _make_run(config, artifact_dir=tmp_path)

        TimingResolver().resolve(run)

        assert run.resolved.total_expected_duration == 70.0

    def test_none_when_phase_lacks_duration(self, tmp_path):
        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets={
                "main": {"type": "synthetic", "entries": 10, "prompts": {"isl": 32}}
            },
            phases={
                "warmup": {
                    "type": "concurrency",
                    "duration": 30,
                    "concurrency": 1,
                    "exclude_from_results": True,
                },
                "main": {"type": "concurrency", "requests": 100, "concurrency": 4},
            },
        )
        run = _make_run(config, artifact_dir=tmp_path)

        TimingResolver().resolve(run)

        assert run.resolved.total_expected_duration is None

    def test_single_phase_with_duration(self, run_with_config):
        TimingResolver().resolve(run_with_config)
        assert run_with_config.resolved.total_expected_duration == 60.0


# ---------------------------------------------------------------------------
# build_default_resolver_chain
# ---------------------------------------------------------------------------


class TestBuildDefaultResolverChain:
    def test_returns_chain_with_all_resolvers(self):
        chain = build_default_resolver_chain()
        assert isinstance(chain, ConfigResolverChain)
        assert len(chain._resolvers) == 6

    def test_resolver_order(self):
        chain = build_default_resolver_chain()
        types = [type(r) for r in chain._resolvers]
        assert types == [
            ArtifactDirResolver,
            TokenizerResolver,
            GpuMetricsResolver,
            CommConfigResolver,
            DatasetResolver,
            TimingResolver,
        ]

    def test_full_chain_integration(self, run_with_config):
        """Run the full chain on a simple config - no errors."""
        chain = build_default_resolver_chain()
        chain.resolve_all(run_with_config)

        assert run_with_config.resolved.artifact_dir_created is True
        assert run_with_config.resolved.total_expected_duration == 60.0
