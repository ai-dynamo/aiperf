# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for resolver edge cases not covered by test_resolvers.py.

Focuses on:
- Resolver ordering and chain failure semantics
- ArtifactDirResolver symlinks, permissions, deep nesting, relative paths
- TokenizerResolver exception propagation and resolved state
- DatasetResolver multi-file, symlinks, error messages, empty datasets
- TimingResolver empty load, zero duration, multiple grace periods
- Full chain integration with file datasets and idempotency
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.config.resolution.resolvers import (
    ArtifactDirResolver,
    ConfigResolverChain,
    DatasetResolver,
    TimingResolver,
    TokenizerResolver,
    build_default_resolver_chain,
)
from aiperf.config.tokenizer import TokenizerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(config: object, *, artifact_dir: Path | None = None) -> BenchmarkRun:
    """Build a minimal BenchmarkRun wrapping a config."""
    return BenchmarkRun(
        benchmark_id="test-run",
        cfg=config,
        artifact_dir=artifact_dir or Path("/tmp/test-artifacts"),
    )


def _make_config(**overrides):
    """Build a minimal BenchmarkConfig with optional overrides."""
    from aiperf.config import BenchmarkConfig

    defaults = {
        "models": ["test-model"],
        "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
        "datasets": [
            {
                "name": "profiling",
                "type": "synthetic",
                "entries": 10,
                "prompts": {"isl": 32},
            }
        ],
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "duration": 60,
                "concurrency": 1,
            }
        ],
    }
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_config():
    """Minimal BenchmarkConfig with synthetic dataset and concurrency phase."""
    return _make_config()


@pytest.fixture()
def run_with_config(minimal_config, tmp_path):
    return _make_run(minimal_config, artifact_dir=tmp_path / "artifacts")


# ============================================================
# Resolver Ordering
# ============================================================


class TestResolverOrdering:
    """Verify chain ordering guarantees and failure semantics."""

    def test_artifact_dir_resolved_before_dataset_resolver(
        self, run_with_config
    ) -> None:
        """Default chain runs ArtifactDirResolver before DatasetResolver."""
        call_order: list[str] = []

        class RecordingResolver:
            def __init__(self, name: str) -> None:
                self._name = name

            def resolve(self, run: BenchmarkRun) -> None:
                call_order.append(self._name)

        chain = build_default_resolver_chain()
        # Replace each resolver with a recording wrapper that tracks order
        wrapped = []
        for resolver in chain._resolvers:
            name = type(resolver).__name__
            recorder = RecordingResolver(name)
            wrapped.append(recorder)
        chain._resolvers = wrapped
        chain.resolve_all(run_with_config)

        artifact_idx = call_order.index("ArtifactDirResolver")
        dataset_idx = call_order.index("DatasetResolver")
        assert artifact_idx < dataset_idx

    def test_resolver_failure_stops_chain(self, run_with_config) -> None:
        """When resolver 2 raises, resolvers 3+ are never called."""
        calls: list[str] = []

        class Recorder:
            def __init__(self, name: str) -> None:
                self._name = name

            def resolve(self, run: BenchmarkRun) -> None:
                calls.append(self._name)

        class Exploder:
            def resolve(self, run: BenchmarkRun) -> None:
                calls.append("exploder")
                raise RuntimeError("resolver 2 failed")

        chain = ConfigResolverChain([Recorder("first"), Exploder(), Recorder("third")])
        with pytest.raises(RuntimeError, match="resolver 2 failed"):
            chain.resolve_all(run_with_config)

        assert calls == ["first", "exploder"]

    def test_resolver_failure_preserves_prior_state(
        self, minimal_config, tmp_path
    ) -> None:
        """Resolver 1 succeeds and sets state, resolver 2 fails; prior state retained."""
        run = _make_run(minimal_config, artifact_dir=tmp_path / "arts")

        class SuccessResolver:
            def resolve(self, run: BenchmarkRun) -> None:
                run.artifact_dir.mkdir(parents=True, exist_ok=True)
                run.resolved.artifact_dir_created = True

        class FailResolver:
            def resolve(self, run: BenchmarkRun) -> None:
                raise ValueError("boom")

        chain = ConfigResolverChain([SuccessResolver(), FailResolver()])
        with pytest.raises(ValueError, match="boom"):
            chain.resolve_all(run)

        assert run.resolved.artifact_dir_created is True


# ============================================================
# ArtifactDirResolver Edge Cases
# ============================================================


class TestArtifactDirResolverEdgeCases:
    """Boundary conditions for directory creation and path resolution."""

    def test_symlink_in_path_resolved(self, minimal_config, tmp_path) -> None:
        """Symlinked directory is resolved to its real path."""
        real_dir = tmp_path / "real_artifacts"
        real_dir.mkdir()
        link = tmp_path / "link_to_artifacts"
        link.symlink_to(real_dir)

        run = _make_run(minimal_config, artifact_dir=link / "output")
        ArtifactDirResolver().resolve(run)

        assert run.artifact_dir.is_absolute()
        # .resolve() follows symlinks
        assert "real_artifacts" in str(run.artifact_dir)
        assert run.resolved.artifact_dir_created is True

    def test_permission_error_propagates(self, minimal_config, tmp_path) -> None:
        """PermissionError from mkdir propagates to caller."""
        run = _make_run(minimal_config, artifact_dir=tmp_path / "no_perms" / "sub")

        with (
            patch.object(Path, "mkdir", side_effect=PermissionError("denied")),
            pytest.raises(PermissionError, match="denied"),
        ):
            ArtifactDirResolver().resolve(run)

    def test_deeply_nested_creation(self, minimal_config, tmp_path) -> None:
        """Deeply nested path (a/b/c/d/e/f) is fully created."""
        deep = tmp_path / "a" / "b" / "c" / "d" / "e" / "f"
        run = _make_run(minimal_config, artifact_dir=deep)

        ArtifactDirResolver().resolve(run)

        assert deep.exists()
        assert deep.is_dir()
        assert run.resolved.artifact_dir_created is True

    def test_current_dir_relative_path(
        self, minimal_config, tmp_path, monkeypatch
    ) -> None:
        """'./artifacts' is resolved to absolute path under cwd."""
        monkeypatch.chdir(tmp_path)
        run = _make_run(minimal_config, artifact_dir=Path("./artifacts"))

        ArtifactDirResolver().resolve(run)

        assert run.artifact_dir.is_absolute()
        assert str(run.artifact_dir).startswith(str(tmp_path))
        assert run.resolved.artifact_dir_created is True


# ============================================================
# TokenizerResolver Edge Cases
# ============================================================


class TestTokenizerResolverEdgeCases:
    """Exception propagation and resolved state for tokenizer validation."""

    def test_validator_exception_propagates(self, tmp_path) -> None:
        """Exception from validate_tokenizer_early propagates to caller."""

        config = _make_config(tokenizer=TokenizerConfig(name="bad-tok"))
        run = _make_run(config, artifact_dir=tmp_path)

        with (
            patch(
                "aiperf.common.tokenizer_validator.validate_tokenizer_early",
                side_effect=RuntimeError("tokenizer not found"),
            ),
            pytest.raises(RuntimeError, match="tokenizer not found"),
        ):
            TokenizerResolver().resolve(run)

    def test_result_stored_in_resolved(self, tmp_path) -> None:
        """Validated tokenizer names are stored in run.resolved."""

        config = _make_config(tokenizer=TokenizerConfig(name="tok-a"))
        run = _make_run(config, artifact_dir=tmp_path)

        expected = {"model-a": "tok-a", "model-b": "tok-b"}
        with patch(
            "aiperf.common.tokenizer_validator.validate_tokenizer_early",
            return_value=expected,
        ):
            TokenizerResolver().resolve(run)

        assert run.resolved.tokenizer_names == expected


# ============================================================
# DatasetResolver Edge Cases
# ============================================================


class TestDatasetResolverEdgeCases:
    """Multi-file, symlinks, error messages, and empty dataset handling."""

    def test_multiple_file_datasets_all_resolved(self, tmp_path) -> None:
        """File dataset is resolved to an absolute path and indexed by name."""
        f = tmp_path / "train.jsonl"
        f.write_text('{"prompt": "hello"}\n')

        config = _make_config(
            datasets=[
                {"name": "train", "type": "file", "path": str(f)},
            ],
            phases=[
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
        )
        run = _make_run(config, artifact_dir=tmp_path / "out")

        DatasetResolver().resolve(run)

        assert run.resolved.dataset_file_paths is not None
        assert set(run.resolved.dataset_file_paths.keys()) == {"train"}
        for path in run.resolved.dataset_file_paths.values():
            assert path.is_absolute()

    def test_symlink_dataset_file_resolved(self, tmp_path) -> None:
        """Symlinked dataset file is resolved to its real path."""
        real_file = tmp_path / "real_data.jsonl"
        real_file.write_text('{"prompt": "hello"}\n')
        link = tmp_path / "link_data.jsonl"
        link.symlink_to(real_file)

        config = _make_config(
            datasets=[{"name": "profiling", "type": "file", "path": str(link)}],
            phases=[
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
        )
        run = _make_run(config, artifact_dir=tmp_path / "out")

        DatasetResolver().resolve(run)

        resolved_path = run.resolved.dataset_file_paths["profiling"]
        assert "real_data.jsonl" in str(resolved_path)

    def test_error_message_includes_dataset_name(self, tmp_path) -> None:
        """FileNotFoundError for missing file includes the dataset key name."""
        config = _make_config(
            datasets=[
                {
                    "name": "my_special_ds",
                    "type": "file",
                    "path": "/nonexistent/data.jsonl",
                }
            ],
            phases=[
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
        )
        run = _make_run(config, artifact_dir=tmp_path / "out")

        with pytest.raises(FileNotFoundError, match="my_special_ds"):
            DatasetResolver().resolve(run)

    def test_all_synthetic_datasets_noop(self, tmp_path) -> None:
        """Config with a synthetic dataset leaves dataset_file_paths as None."""
        config = _make_config(
            datasets=[
                {
                    "name": "synth",
                    "type": "synthetic",
                    "entries": 5,
                    "prompts": {"isl": 32},
                },
            ],
        )
        run = _make_run(config, artifact_dir=tmp_path / "out")

        DatasetResolver().resolve(run)

        assert run.resolved.dataset_file_paths is None


# ============================================================
# TimingResolver Edge Cases
# ============================================================


class TestTimingResolverEdgeCases:
    """Empty loads, zero durations, excluded phases, and multiple grace periods."""

    def test_empty_load_dict_returns_zero(self, tmp_path) -> None:
        """No phases in load dict yields total_expected_duration=0.0."""

        # BenchmarkConfig requires at least one phase; build the run manually
        # with a config that has an empty load dict by using model_copy
        config = _make_config()
        run = _make_run(config, artifact_dir=tmp_path)

        # Override load to be empty after construction
        object.__setattr__(run.cfg, "phases", {})
        TimingResolver().resolve(run)

        assert run.resolved.total_expected_duration == 0.0

    def test_excluded_phases_included_in_total(self, tmp_path) -> None:
        """Phases with exclude_from_results=True still contribute to total duration."""
        config = _make_config(
            phases=[
                {
                    "name": "warmup",
                    "type": "concurrency",
                    "duration": 30,
                    "concurrency": 1,
                    "exclude_from_results": True,
                },
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 60,
                    "concurrency": 2,
                },
            ],
        )
        run = _make_run(config, artifact_dir=tmp_path)

        TimingResolver().resolve(run)

        assert run.resolved.total_expected_duration == 90.0

    def test_zero_duration_counted(self, tmp_path) -> None:
        """Phase with very small duration adds to total without being skipped."""
        config = _make_config(
            phases=[
                {
                    "name": "warmup",
                    "type": "concurrency",
                    "duration": 0.001,
                    "concurrency": 1,
                    "exclude_from_results": True,
                },
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 60,
                    "concurrency": 1,
                },
            ],
        )
        run = _make_run(config, artifact_dir=tmp_path)

        TimingResolver().resolve(run)

        assert run.resolved.total_expected_duration == pytest.approx(60.001)

    def test_multiple_grace_periods_summed(self, tmp_path) -> None:
        """Grace periods from multiple phases are all added to total."""
        config = _make_config(
            phases=[
                {
                    "name": "warmup",
                    "type": "concurrency",
                    "duration": 30,
                    "grace_period": 5,
                    "concurrency": 1,
                    "exclude_from_results": True,
                },
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 60,
                    "grace_period": 10,
                    "concurrency": 2,
                },
            ],
        )
        run = _make_run(config, artifact_dir=tmp_path)

        TimingResolver().resolve(run)

        # 30+5 + 60+10 = 105
        assert run.resolved.total_expected_duration == 105.0

    def test_none_duration_short_circuits(self, tmp_path) -> None:
        """If any phase lacks duration, total is None even if others have durations."""
        config = _make_config(
            phases=[
                {
                    "name": "warmup",
                    "type": "concurrency",
                    "duration": 60,
                    "concurrency": 1,
                    "exclude_from_results": True,
                },
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 100,
                    "concurrency": 2,
                },
            ],
        )
        run = _make_run(config, artifact_dir=tmp_path)

        TimingResolver().resolve(run)

        assert run.resolved.total_expected_duration is None


# ============================================================
# Resolver Chain Integration
# ============================================================


class TestResolverChainIntegration:
    """Full chain tests with real filesystem state."""

    def test_full_chain_with_file_dataset(self, tmp_path) -> None:
        """Full chain with a real file dataset populates dataset_file_paths."""
        dataset_file = tmp_path / "data.jsonl"
        dataset_file.write_text('{"prompt": "hello"}\n')

        config = _make_config(
            datasets=[{"name": "profiling", "type": "file", "path": str(dataset_file)}],
            phases=[
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
        )
        run = _make_run(config, artifact_dir=tmp_path / "artifacts")

        chain = build_default_resolver_chain()
        chain.resolve_all(run)

        assert run.resolved.artifact_dir_created is True
        assert run.resolved.dataset_file_paths is not None
        assert "profiling" in run.resolved.dataset_file_paths
        assert run.resolved.dataset_file_paths["profiling"].is_absolute()

    def test_full_chain_populates_all_resolved_fields(self, tmp_path) -> None:
        """Full chain with config triggering all resolvers populates all fields."""
        dataset_file = tmp_path / "data.jsonl"
        dataset_file.write_text('{"prompt": "hello"}\n')

        config = _make_config(
            tokenizer=TokenizerConfig(name="test-tok"),
            datasets=[{"name": "profiling", "type": "file", "path": str(dataset_file)}],
            phases=[
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 60,
                    "grace_period": 10,
                    "concurrency": 1,
                }
            ],
        )
        run = _make_run(config, artifact_dir=tmp_path / "artifacts")

        with patch(
            "aiperf.common.tokenizer_validator.validate_tokenizer_early",
            return_value={"test-model": "resolved-tok"},
        ):
            chain = build_default_resolver_chain()
            chain.resolve_all(run)

        assert run.resolved.artifact_dir_created is True
        assert run.resolved.tokenizer_names == {"test-model": "resolved-tok"}
        assert run.resolved.dataset_file_paths is not None
        assert "profiling" in run.resolved.dataset_file_paths
        assert run.resolved.total_expected_duration == 70.0
        # gpu_custom_metrics stays None (no metrics_file configured)
        assert run.resolved.gpu_custom_metrics is None

    def test_chain_idempotent(self, tmp_path) -> None:
        """Running the full chain twice produces no errors and same state."""
        config = _make_config()
        run = _make_run(config, artifact_dir=tmp_path / "artifacts")

        chain = build_default_resolver_chain()
        chain.resolve_all(run)

        first_artifact_dir = run.artifact_dir
        first_duration = run.resolved.total_expected_duration

        chain.resolve_all(run)

        assert run.artifact_dir == first_artifact_dir
        assert run.resolved.total_expected_duration == first_duration
        assert run.resolved.artifact_dir_created is True
