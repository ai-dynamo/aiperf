# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Weights & Biases post-run data exporter."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.common.redact import REDACTED_VALUE
from aiperf.config import (
    ArtifactsConfig,
    BenchmarkConfig,
    EndpointConfig,
    WandbConfig,
)
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.wandb_data_exporter import WandbDataExporter
from aiperf.plugin.enums import EndpointType


def _make_cfg(
    tmp_path: Path,
    *,
    project: str | None = "aiperf-dev",
    entity: str | None = "coreweave1",
    run_name: str | None = None,
    tags: list[str] | None = None,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        model="test-model",
        endpoint=EndpointConfig(
            urls=["http://localhost:8000"],
            type=EndpointType.CHAT,
        ),
        dataset={"type": "synthetic"},
        profiling={"type": "concurrency", "requests": 1, "concurrency": 1},
        artifacts=ArtifactsConfig(dir=tmp_path),
        wandb=WandbConfig(
            project=project,
            entity=entity,
            run_name=run_name,
            tags=tags,
        ),
    )


@pytest.fixture
def sample_results() -> ProfileResults:
    return ProfileResults(
        records=[
            MetricResult(
                tag="request_throughput",
                header="Request Throughput",
                unit="req/s",
                avg=42.5,
                count=10,
                sum=425.0,
            ),
            MetricResult(
                tag="time_to_first_token",
                header="Time to First Token",
                unit="ms",
                avg=None,
            ),
        ],
        total_expected=12,
        completed=10,
        start_ns=0,
        end_ns=1,
        was_cancelled=False,
        error_summary=[],
    )


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install a fake wandb module into sys.modules and return call state."""
    state: dict[str, Any] = {
        "init_kwargs": None,
        "logged": [],
        "artifact_files": [],
        "logged_artifacts": [],
        "finished": False,
    }

    class FakeArtifact:
        # `type` shadows the builtin on purpose: the exporter calls
        # wandb.Artifact(name=..., type=...) by keyword, mirroring wandb's API.
        def __init__(self, name: str, type: str) -> None:
            self.name = name
            self.type = type

        def add_file(self, path: str, name: str | None = None) -> None:
            state["artifact_files"].append((path, name))

    class FakeRun:
        id = "run-123"
        url = "https://wandb.ai/fake/run-123"

        def log(self, payload: dict[str, Any]) -> None:
            state["logged"].append(payload)

        def log_artifact(self, artifact: FakeArtifact) -> None:
            state["logged_artifacts"].append(artifact)

        def finish(self) -> None:
            state["finished"] = True

    class FakeTable:
        def __init__(self, columns: list[str], data: list[list[Any]]) -> None:
            self.columns = columns
            self.data = data

    def fake_init(**kwargs: Any) -> FakeRun:
        state["init_kwargs"] = kwargs
        return FakeRun()

    fake_module = types.ModuleType("wandb")
    fake_module.init = fake_init  # type: ignore[attr-defined]
    fake_module.Table = FakeTable  # type: ignore[attr-defined]
    fake_module.Artifact = FakeArtifact  # type: ignore[attr-defined]
    state["table_cls"] = FakeTable
    monkeypatch.setitem(sys.modules, "wandb", fake_module)
    return state


def _make_exporter(
    cfg: BenchmarkConfig,
    results: ProfileResults | None,
    run: Any | None = None,
) -> WandbDataExporter:
    return WandbDataExporter(
        ExporterConfig(
            results=results,
            cfg=cfg,
            telemetry_results=None,
            run=run,
        )
    )


class TestWandbDataExporter:
    def test_disabled_without_project(
        self, tmp_path: Path, sample_results: ProfileResults
    ) -> None:
        cfg = _make_cfg(tmp_path, project=None, entity=None)
        with pytest.raises(DataExporterDisabled):
            _make_exporter(cfg, sample_results)

    def test_disabled_without_results(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        with pytest.raises(DataExporterDisabled):
            _make_exporter(cfg, results=None)

    @pytest.mark.asyncio
    async def test_export_logs_console_style_table_and_artifacts(
        self,
        tmp_path: Path,
        sample_results: ProfileResults,
        fake_wandb: dict[str, Any],
    ) -> None:
        """The run carries the full resolved config, tags, exactly one logged
        payload (the console-style table), and the artifact-dir file bundle.
        """
        (tmp_path / "profile_export_aiperf.json").write_text("{}", encoding="utf-8")
        (tmp_path / "profile_export_aiperf.csv").write_text("a,b", encoding="utf-8")

        cfg = _make_cfg(tmp_path, run_name="my-run", tags=["aa-repro"])
        exporter = _make_exporter(cfg, sample_results)
        await asyncio.to_thread(exporter._export_sync)

        init_kwargs = fake_wandb["init_kwargs"]
        assert init_kwargs["entity"] == "coreweave1"
        assert init_kwargs["project"] == "aiperf-dev"
        assert init_kwargs["name"] == "my-run"
        assert "aa-repro" in init_kwargs["tags"]
        config = init_kwargs["config"]
        assert config["models"]["items"][0]["name"] == "test-model"
        assert config["phases"][0]["concurrency"] == 1
        assert config["endpoint"]["type"] == "chat"
        assert config["wandb"]["project"] == "aiperf-dev"

        [logged] = fake_wandb["logged"]
        assert set(logged) == {"summary_metrics"}  # the table is the only payload
        table = logged["summary_metrics"]
        assert isinstance(table, fake_wandb["table_cls"])
        assert table.columns == ["Metric", "avg", "min", "max", "p99", "p90", "p50", "std"]  # fmt: skip
        by_label = {row[0]: row for row in table.data}
        throughput_row = next(
            row for label, row in by_label.items() if "req/s" in label.lower()
        )
        assert throughput_row[1] == 42.5  # avg column
        ttft_row = next(
            row for label, row in by_label.items() if "ttft" in label.lower()
        )
        assert ttft_row[1] is None  # missing stats stay None

        uploaded = {name for _, name in fake_wandb["artifact_files"]}
        assert uploaded == {"profile_export_aiperf.json", "profile_export_aiperf.csv"}
        [artifact] = fake_wandb["logged_artifacts"]
        assert artifact.type == "aiperf-run"
        assert fake_wandb["finished"] is True

    @pytest.mark.asyncio
    async def test_export_finishes_run_on_failure(
        self,
        tmp_path: Path,
        sample_results: ProfileResults,
        fake_wandb: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failure after wandb.init must still finish the run so it does
        not linger in the 'running' state in the wandb UI.
        """
        exporter = _make_exporter(_make_cfg(tmp_path), sample_results)
        monkeypatch.setattr(exporter, "_build_metric_table_rows", lambda: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            await asyncio.to_thread(exporter._export_sync)
        assert fake_wandb["finished"] is True
        assert fake_wandb["logged_artifacts"] == []

    @pytest.mark.asyncio
    async def test_export_non_finite_stats_become_none(
        self, tmp_path: Path, fake_wandb: dict[str, Any]
    ) -> None:
        """NaN/Inf stats are scrubbed to None per the repo's NaN/Inf
        discipline for values crossing a serialization boundary.
        """
        results = ProfileResults(
            records=[
                MetricResult(
                    tag="request_throughput",
                    header="Request Throughput",
                    unit="req/s",
                    avg=float("nan"),
                    max=float("inf"),
                    min=1.0,
                ),
            ],
            total_expected=1,
            completed=1,
            start_ns=0,
            end_ns=1,
            was_cancelled=False,
            error_summary=[],
        )
        exporter = _make_exporter(_make_cfg(tmp_path), results)
        await asyncio.to_thread(exporter._export_sync)

        [logged] = fake_wandb["logged"]
        [row] = logged["summary_metrics"].data
        by_col = dict(zip(logged["summary_metrics"].columns, row, strict=True))
        assert by_col["avg"] is None  # nan scrubbed
        assert by_col["max"] is None  # inf scrubbed
        assert by_col["min"] == 1.0

    @pytest.mark.asyncio
    async def test_export_includes_redacted_cli_command(
        self,
        tmp_path: Path,
        sample_results: ProfileResults,
        fake_wandb: dict[str, Any],
    ) -> None:
        run = types.SimpleNamespace(
            benchmark_id="abc123def456",
            cli_command="aiperf profile --api-key secret --model m",
        )
        exporter = _make_exporter(_make_cfg(tmp_path), sample_results, run=run)
        await asyncio.to_thread(exporter._export_sync)

        cli_command = fake_wandb["init_kwargs"]["config"]["aiperf.cli_command"]
        assert "secret" not in cli_command
        assert cli_command.startswith("aiperf profile")

    @pytest.mark.asyncio
    async def test_export_config_payload_redacts_api_key(
        self,
        tmp_path: Path,
        sample_results: ProfileResults,
        fake_wandb: dict[str, Any],
    ) -> None:
        """The uploaded config relies on the model-layer json serializers for
        secret redaction; lock that in so a future change can't silently leak.
        """
        cfg = _make_cfg(tmp_path)
        cfg.endpoint.api_key = "super-secret-key"
        exporter = _make_exporter(cfg, sample_results)
        await asyncio.to_thread(exporter._export_sync)

        config = fake_wandb["init_kwargs"]["config"]
        assert "super-secret-key" not in str(config)
        assert config["endpoint"]["api_key"] == REDACTED_VALUE

    def test_collect_artifact_files_skips_wandb_state_dir(
        self, tmp_path: Path, sample_results: ProfileResults
    ) -> None:
        """wandb.init(dir=...) writes its own run dir under <artifact_dir>/wandb/
        before collection runs; the recursive globs must not re-upload it.
        """
        (tmp_path / "plots").mkdir()
        (tmp_path / "plots" / "ttft.png").write_bytes(b"png")
        wandb_media = tmp_path / "wandb" / "run-20260612_010101-abc" / "files" / "media"
        wandb_media.mkdir(parents=True)
        (wandb_media / "table.png").write_bytes(b"png")
        (wandb_media.parent / "config.json").write_text("{}", encoding="utf-8")

        exporter = _make_exporter(_make_cfg(tmp_path), sample_results)
        collected = exporter._collect_artifact_files()

        names = {f.relative_to(tmp_path).as_posix() for f in collected}
        assert names == {"plots/ttft.png"}

    def test_default_run_name_uses_benchmark_id(
        self, tmp_path: Path, sample_results: ProfileResults
    ) -> None:
        run = types.SimpleNamespace(benchmark_id="abcdef1234567890", cli_command=None)
        exporter = _make_exporter(_make_cfg(tmp_path), sample_results, run=run)
        assert exporter._derive_default_run_name() == "aiperf-abcdef12"

    @pytest.mark.asyncio
    async def test_export_subprocess_terminates_on_timeout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        sample_results: ProfileResults,
    ) -> None:
        """``export()`` must terminate a hung subprocess on timeout.

        Simulates a wedged W&B upload by stubbing the subprocess entrypoint
        with a sleeper at module scope (spawn can't pickle local closures).
        With a 1s timeout the outer ``export()`` returns quickly (not 30s
        default) and the worker subprocess is no longer alive.
        """
        import time as time_module

        from aiperf.common import environment as env_module

        monkeypatch.setattr(
            "aiperf.exporters.wandb_export_subprocess.run_export_in_subprocess",
            _hang_forever_subprocess_entry,
        )
        monkeypatch.setattr(env_module.Environment.WANDB, "EXPORT_TIMEOUT_SECONDS", 1.0)

        exporter = _make_exporter(_make_cfg(tmp_path), sample_results)

        start = time_module.monotonic()
        await exporter.export()
        elapsed = time_module.monotonic() - start

        # Must return within a small multiple of the 1s timeout, not 30s.
        assert elapsed < 10.0, f"export() did not honor timeout: elapsed={elapsed:.1f}s"


def _hang_forever_subprocess_entry(exporter_config: Any, result_queue: Any) -> None:
    """Module-level subprocess stub (must be picklable for spawn context)."""
    import time as _time

    _time.sleep(30)
    result_queue.put(None)
