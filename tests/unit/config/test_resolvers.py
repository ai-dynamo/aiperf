# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for config resolver chain and individual resolvers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from aiperf.config import BenchmarkConfig
from aiperf.config.gpu_telemetry import GpuTelemetryConfig
from aiperf.config.resolution.plan import BenchmarkRun, ResolvedConfig
from aiperf.config.resolution.resolvers import (
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
from aiperf.config.tokenizer import TokenizerConfig

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
        datasets=[
            {
                "name": "profiling",
                "type": "synthetic",
                "entries": 10,
                "prompts": {"isl": 32},
            }
        ],
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "duration": 60,
                "concurrency": 1,
            }
        ],
    )


@pytest.fixture()
def run_with_config(minimal_config, tmp_path):
    return _make_run(minimal_config, artifact_dir=tmp_path / "artifacts")


# ---------------------------------------------------------------------------
# Resolved runtime state
# ---------------------------------------------------------------------------


def test_resolved_config_rejects_negative_dataset_root_count() -> None:
    with pytest.raises(ValidationError, match="dataset_root_count"):
        ResolvedConfig(dataset_root_count={"profiling": -1})


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

    def test_resolve_for_probe_skips_user_files_materialization(self, tmp_path):
        """Probe runs must NOT materialize user_files (they re-run per variation).

        ``cli_runner._estimate_and_log_duration`` clones the user's first config
        into a probe run only to estimate duration. After Task 6 the resolver
        also wrote user_files; that produced a stray artifact tree before the
        actual benchmark and could bake in template values (e.g. ``epoch``)
        that don't match the per-variation runs.
        """
        from aiperf.config.loader import load_config_from_string

        yaml_str = """
            benchmark:
              artifacts:
                user_files:
                  - path: input_config.json
                    format: json
                    content:
                      note: "literal value"
              models:
                - test/model
              endpoint:
                type: chat
                urls: ["http://localhost:8000"]
              datasets:
                - name: default
                  type: synthetic
                  entries: 10
                  prompts:
                    isl: 32
                    osl: 16
              phases:
                - name: profiling
                  type: concurrency
                  requests: 10
                  concurrency: 1
        """
        # Probe run: target dir gets created, but user_files MUST NOT exist.
        probe_dir = tmp_path / "probe"
        probe_cfg = load_config_from_string(yaml_str)
        probe_cfg.benchmark.artifacts.dir = probe_dir
        probe_run = _make_run(probe_cfg.benchmark, artifact_dir=probe_dir)
        ArtifactDirResolver().resolve(probe_run, for_probe=True)
        assert probe_dir.is_dir()
        assert not (probe_dir / "input_config.json").exists()

        # Real per-variation run (default for_probe=False) DOES materialize.
        real_dir = tmp_path / "real"
        real_cfg = load_config_from_string(yaml_str)
        real_cfg.benchmark.artifacts.dir = real_dir
        real_run = _make_run(real_cfg.benchmark, artifact_dir=real_dir)
        ArtifactDirResolver().resolve(real_run)
        assert (real_dir / "input_config.json").exists()

    def test_user_centric_default_artifact_name_uses_current_fields(self, tmp_path):
        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets=[
                {
                    "name": "main",
                    "type": "synthetic",
                    "entries": 10,
                    "turns": {"mean": 2},
                    "prompts": {"isl": 32, "osl": 16},
                }
            ],
            phases=[
                {
                    "name": "profiling",
                    "type": "user_centric",
                    "requests": 2,
                    "users": 2,
                    "rate": 1.0,
                }
            ],
        )
        run = _make_run(config, artifact_dir=tmp_path / "artifacts")

        ArtifactDirResolver().resolve(run)

        assert "user_centric-users2-qps1.0" in run.artifact_dir.name


# ---------------------------------------------------------------------------
# TokenizerResolver
# ---------------------------------------------------------------------------


class TestTokenizerResolver:
    def test_runs_validator_even_when_tokenizer_unset(self, run_with_config):
        """Resolver always invokes the validator so fake-model detection fires
        even when the user passed no `--tokenizer*` flags (and v1 left
        ``cfg.benchmark.tokenizer`` as None).
        """
        run_with_config.cfg = run_with_config.cfg.model_copy(update={"tokenizer": None})

        with patch(
            "aiperf.common.tokenizer_validator.validate_tokenizer_early",
            return_value=None,
        ) as mock_validate:
            TokenizerResolver().resolve(run_with_config)

        mock_validate.assert_called_once()
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
        run = _make_run(config, artifact_dir=tmp_path / "out")

        DatasetResolver().resolve(run)

        assert run.resolved.dataset_file_paths is not None
        assert "profiling" in run.resolved.dataset_file_paths
        assert run.resolved.dataset_file_paths["profiling"].is_absolute()

    def test_raises_on_missing_file(self, tmp_path):
        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets=[
                {"name": "main", "type": "file", "path": "/nonexistent/data.jsonl"}
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

        with pytest.raises(FileNotFoundError, match="Dataset 'main' file not found"):
            DatasetResolver().resolve(run)

    def test_format_map_includes_dag_jsonl(self):
        from aiperf.plugin.enums import CustomDatasetType

        assert (
            DatasetResolver._build_format_map()["dag_jsonl"]
            == CustomDatasetType.DAG_JSONL
        )

    def test_counts_dag_roots_for_dag_jsonl(self, tmp_path):
        """Forking datasets (dag_jsonl) should populate dataset_root_count.

        File: s1 (root), s2 (child of s1.forks), s3 (child of s2.spawns),
        s4 (root, referenced by no one). Roots = {s1, s4} -> count=2.
        Non-roots = {s2, s3}.
        """
        import json

        dataset_file = tmp_path / "dag.jsonl"
        lines = [
            {"session_id": "s1", "turns": [{"forks": ["s2"]}]},
            {
                "session_id": "s2",
                "turns": [{"spawns": [{"children": ["s3"]}]}],
            },
            {"session_id": "s3", "turns": []},
            {"session_id": "s4", "turns": []},
        ]
        dataset_file.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets=[
                {
                    "name": "profiling",
                    "type": "file",
                    "path": str(dataset_file),
                    "format": "dag_jsonl",
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

        DatasetResolver().resolve(run)

        assert run.resolved.dataset_is_forking == {"profiling": True}
        assert run.resolved.dataset_root_count == {"profiling": 2}

    def test_non_forking_dataset_marks_is_forking_false(self, tmp_path):
        """Non-DAG file datasets get is_forking=False, root_count unset."""
        dataset_file = tmp_path / "data.jsonl"
        dataset_file.write_text('{"prompt": "hello"}\n')

        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
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
        run = _make_run(config, artifact_dir=tmp_path / "out")

        DatasetResolver().resolve(run)

        assert run.resolved.dataset_is_forking == {"profiling": False}
        assert run.resolved.dataset_root_count is None

    def test_burst_gpt_csv_reports_timing_data(self, tmp_path):
        """BurstGPT CSVs always carry a ``Timestamp`` column; the resolver
        must report ``has_timing=True`` so ``--fixed-schedule`` is accepted.

        Regression: the JSONL-only first-record probe in ``_check_timing_data``
        could not parse a CSV header and silently returned False, blocking
        the BurstGPT tutorial's ``aiperf profile … --fixed-schedule`` run.
        """
        dataset_file = tmp_path / "burst_gpt.csv"
        dataset_file.write_text(
            "Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
            "0.123,gpt-4,512,128,640,chat\n"
            "0.456,gpt-4,300,80,380,chat\n"
        )

        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets=[
                {
                    "name": "main",
                    "type": "file",
                    "path": str(dataset_file),
                    "format": "burst_gpt_trace",
                }
            ],
            phases=[
                {
                    "name": "profiling",
                    "type": "fixed_schedule",
                }
            ],
        )
        run = _make_run(config, artifact_dir=tmp_path / "out")

        DatasetResolver().resolve(run)

        assert run.resolved.dataset_has_timing_data == {"main": True}
        # Pair with TimingResolver to confirm fixed_schedule validation passes.
        TimingResolver().resolve(run)


# ---------------------------------------------------------------------------
# TimingResolver
# ---------------------------------------------------------------------------


class TestTimingResolver:
    def test_sums_durations(self, tmp_path):
        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets=[
                {
                    "name": "main",
                    "type": "synthetic",
                    "entries": 10,
                    "prompts": {"isl": 32},
                }
            ],
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
                    "duration": 120,
                    "concurrency": 4,
                },
            ],
        )
        run = _make_run(config, artifact_dir=tmp_path)

        TimingResolver().resolve(run)

        assert run.resolved.total_expected_duration == 150.0

    def test_includes_grace_periods(self, tmp_path):
        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets=[
                {
                    "name": "main",
                    "type": "synthetic",
                    "entries": 10,
                    "prompts": {"isl": 32},
                }
            ],
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
        run = _make_run(config, artifact_dir=tmp_path)

        TimingResolver().resolve(run)

        assert run.resolved.total_expected_duration == 70.0

    def test_none_when_phase_lacks_duration(self, tmp_path):
        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets=[
                {
                    "name": "main",
                    "type": "synthetic",
                    "entries": 10,
                    "prompts": {"isl": 32},
                }
            ],
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
                    "requests": 100,
                    "concurrency": 4,
                },
            ],
        )
        run = _make_run(config, artifact_dir=tmp_path)

        TimingResolver().resolve(run)

        assert run.resolved.total_expected_duration is None

    def test_fixed_schedule_without_timing_data_rejects_before_runtime(self, tmp_path):
        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets=[
                {
                    "name": "main",
                    "type": "file",
                    "format": "single_turn",
                    "records": [{"text": "hello"}],
                }
            ],
            phases=[
                {
                    "name": "profiling",
                    "type": "fixed_schedule",
                }
            ],
        )
        run = _make_run(config, artifact_dir=tmp_path)
        run.resolved.dataset_has_timing_data = {"main": False}

        with pytest.raises(ValueError, match="uses fixed_schedule which requires"):
            TimingResolver().resolve(run)

    def test_fixed_schedule_without_timing_metadata_rejects_before_runtime(
        self, tmp_path
    ):
        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets=[
                {
                    "name": "main",
                    "type": "synthetic",
                    "entries": 10,
                    "prompts": {"isl": 32},
                }
            ],
            phases=[
                {
                    "name": "profiling",
                    "type": "fixed_schedule",
                }
            ],
        )
        run = _make_run(config, artifact_dir=tmp_path)

        with pytest.raises(ValueError, match="could not verify timing data"):
            TimingResolver().resolve(run)

    def test_fixed_schedule_accepts_timed_public_dataset(self, tmp_path):
        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets=[
                {
                    "name": "main",
                    "type": "public",
                    "dataset": "exgentic",
                    "entries": 1,
                }
            ],
            phases=[{"name": "profiling", "type": "fixed_schedule"}],
        )
        run = _make_run(config, artifact_dir=tmp_path)

        DatasetResolver().resolve(run)
        TimingResolver().resolve(run)

        assert run.resolved.dataset_has_timing_data == {"main": True}

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


# ---------------------------------------------------------------------------
# _derive_run_meta — operator vs local layout detection
# ---------------------------------------------------------------------------


class TestDeriveRunMeta:
    """Cover the EPOCH_RE-gated branch in ``_derive_run_meta``.

    Operator layout is ``<base>/<ns>/<name>/<epoch>``; an epoch-shaped leaf
    (matched by ``aiperf.operator.results_layout.EPOCH_RE``) means the parent
    is the AIPerfJob name. A non-epoch leaf is treated as a local-CLI run.
    """

    @pytest.fixture
    def _clear_namespace_env(self, monkeypatch):
        monkeypatch.delenv("AIPERF_NAMESPACE", raising=False)

    def test_operator_layout(self, _clear_namespace_env):
        """Epoch-shaped leaf -> operator layout: epoch=leaf, job_name=parent."""
        from aiperf.config.resolution.resolvers import _derive_run_meta

        meta = _derive_run_meta(Path("/artifacts/myns/myjob/1714000000"))
        assert meta.epoch == "1714000000"
        assert meta.job_name == "myjob"
        assert meta.namespace == ""

    def test_local_layout(self, _clear_namespace_env):
        """Non-epoch leaf -> local layout: epoch=wall-clock, job_name=leaf."""
        from aiperf.config.resolution.resolvers import _derive_run_meta

        meta = _derive_run_meta(Path("/tmp/llama-3-8b-bench"))
        assert meta.epoch.isdigit()
        assert meta.job_name == "llama-3-8b-bench"
        assert meta.namespace == ""

    def test_local_path_with_short_digit_basename_not_treated_as_operator(
        self, _clear_namespace_env
    ):
        """A local path like /tmp/bench/42 must NOT be misread as operator layout."""
        from aiperf.config.resolution.resolvers import _derive_run_meta

        # 42 is too short to match EPOCH_RE (^\d{9,11}$|^legacy$).
        meta = _derive_run_meta(Path("/tmp/bench/42"))
        assert meta.job_name == "42"  # leaf used as job_name, NOT parent.
        # epoch is wall-clock seconds, not "42".
        assert meta.epoch.isdigit() and len(meta.epoch) >= 9

    def test_legacy_epoch_match(self, _clear_namespace_env):
        """The literal 'legacy' is a valid EPOCH_RE match (sentinel run dir)."""
        from aiperf.config.resolution.resolvers import _derive_run_meta

        meta = _derive_run_meta(Path("/artifacts/myns/myjob/legacy"))
        assert meta.epoch == "legacy"
        assert meta.job_name == "myjob"

    def test_namespace_from_env(self, monkeypatch):
        from aiperf.config.resolution.resolvers import _derive_run_meta

        monkeypatch.setenv("AIPERF_NAMESPACE", "production")
        meta = _derive_run_meta(Path("/tmp/bench"))
        assert meta.namespace == "production"

    def test_empty_namespace_env_returns_empty(self, monkeypatch):
        from aiperf.config.resolution.resolvers import _derive_run_meta

        monkeypatch.setenv("AIPERF_NAMESPACE", "")
        meta = _derive_run_meta(Path("/tmp/bench"))
        assert meta.namespace == ""
