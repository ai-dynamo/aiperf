# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

import orjson
import pytest
from pytest import param

from aiperf.common.environment import Environment
from aiperf.common.models.record_models import MetricRecordInfo
from aiperf.common.results_markers import (
    ResultFileUnavailable,
    collect_ready_artifacts,
    remove_committed_request_checkpoint,
    resolve_result_file,
    write_ready_marker,
)
from aiperf.config import BenchmarkRun
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import ServiceRunType
from aiperf.post_processors.record_export_jsonl_writer import RecordExportJSONLWriter
from tests.unit.conftest import make_run_from_cli
from tests.unit.post_processors.conftest import (
    aiperf_lifecycle,
    create_metric_records_data,
)


@pytest.fixture
def checkpoint_run(tmp_path: Path) -> BenchmarkRun:
    """Build a Kubernetes run with per-request export enabled."""
    run = make_run_from_cli(CLIConfig(model_names=["test-model"]))
    run.cfg.artifacts.dir = tmp_path
    run.cfg.runtime.service_run_type = ServiceRunType.KUBERNETES
    return run


@pytest.mark.parametrize("prefix", [None, "custom_profile"])
@pytest.mark.asyncio
async def test_request_metrics_recoverable_before_final_export(
    checkpoint_run: BenchmarkRun,
    prefix: str | None,
) -> None:
    """Flushed records are fetchable before final export is ready."""
    checkpoint_run.cfg.artifacts.prefix = prefix
    writer = RecordExportJSONLWriter(service_id="records-manager", run=checkpoint_run)
    checkpoint = writer.output_file.parent / "checkpoints" / writer.output_file.name

    async with aiperf_lifecycle(writer):
        record = create_metric_records_data(
            x_request_id="request-0",
            results=[{"request_latency": 2_000_000, "output_token_count": 10}],
        )
        await writer.process_record(record)
        await writer.flush_buffer()

        # No finalization or readiness marker: these are the files the recovery
        # fetch can download if a sibling dies during final aggregation.
        for recursive in (False, True):
            artifacts = collect_ready_artifacts(
                writer.output_file.parent, recursive=recursive
            )
            assert [name for name, _ in artifacts] == [f"checkpoints/{checkpoint.name}"]
        recovered = resolve_result_file(
            writer.output_file.parent, f"checkpoints/{checkpoint.name}"
        )
        records = [
            MetricRecordInfo.model_validate(orjson.loads(line))
            for line in recovered.read_bytes().splitlines()
        ]
        assert len(records) == 1
        recovered_record = records[0]
        assert recovered_record.metadata.x_request_id == "request-0"
        assert recovered_record.metrics["request_latency"].value == 2.0
        assert recovered_record.metrics["request_latency"].unit == "ms"
        assert recovered_record.metrics["output_token_count"].value == 10
        assert checkpoint.samefile(writer.output_file)
        with pytest.raises(ResultFileUnavailable):
            resolve_result_file(writer.output_file.parent, writer.output_file.name)


@pytest.mark.parametrize(
    "service_run_type,interval",
    [
        param(ServiceRunType.MULTIPROCESSING, 30, id="local-run"),
        param(ServiceRunType.KUBERNETES, 0, id="checkpoints-disabled"),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_request_checkpoint_respects_run_settings(
    checkpoint_run: BenchmarkRun,
    monkeypatch: pytest.MonkeyPatch,
    service_run_type: ServiceRunType,
    interval: float,
) -> None:
    """Local runs and explicitly disabled checkpoints create no extra artifact."""
    checkpoint_run.cfg.runtime.service_run_type = service_run_type
    monkeypatch.setattr(Environment.RECORD, "CHECKPOINT_INTERVAL", interval)
    writer = RecordExportJSONLWriter(service_id="records-manager", run=checkpoint_run)
    async with aiperf_lifecycle(writer):
        await writer.process_record(
            create_metric_records_data(results=[{"request_latency": 1_000_000}])
        )
        await writer.flush_buffer()
        assert not (writer.output_file.parent / "checkpoints").exists()


@pytest.mark.asyncio
async def test_request_checkpoint_replaces_previous_run(
    checkpoint_run: BenchmarkRun,
) -> None:
    """Reusing an artifact path must expose the new file rather than a stale inode."""
    output = checkpoint_run.cfg.artifacts.profile_export_jsonl_file
    checkpoint = output.parent / "checkpoints" / output.name
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'{"previous_run": true}\n')
    writer = RecordExportJSONLWriter(service_id="records-manager", run=checkpoint_run)
    async with aiperf_lifecycle(writer):
        await writer.process_record(
            create_metric_records_data(
                x_request_id="new-run", results=[{"request_latency": 1_000_000}]
            )
        )
        await writer.flush_buffer()
        assert checkpoint.samefile(output)
        assert b"previous_run" not in checkpoint.read_bytes()
        assert b"new-run" in checkpoint.read_bytes()


@pytest.mark.parametrize("interval", [0, 30])
def test_previous_checkpoint_removed_before_startup(
    checkpoint_run: BenchmarkRun,
    monkeypatch: pytest.MonkeyPatch,
    interval: int,
) -> None:
    """A run that never reaches on_start cannot expose its predecessor's data."""
    monkeypatch.setattr(Environment.RECORD, "CHECKPOINT_INTERVAL", interval)
    checkpoint_run.cfg.artifacts.prefix = "reused_profile"
    output = checkpoint_run.cfg.artifacts.profile_export_jsonl_file
    output.write_bytes(b'{"previous_run": true}\n')
    checkpoint = output.parent / "checkpoints" / output.name
    checkpoint.parent.mkdir()
    checkpoint.hardlink_to(output)

    RecordExportJSONLWriter(service_id="records-manager", run=checkpoint_run)

    assert not checkpoint.exists()
    assert collect_ready_artifacts(output.parent, recursive=True) == []


def test_previous_checkpoint_removed_before_output_reset_failure(
    checkpoint_run: BenchmarkRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure resetting the output must not leave a stale recovery alias behind."""
    output = checkpoint_run.cfg.artifacts.profile_export_jsonl_file
    output.write_bytes(b'{"previous_run": true}\n')
    checkpoint = output.parent / "checkpoints" / output.name
    checkpoint.parent.mkdir()
    checkpoint.hardlink_to(output)
    unlink = Path.unlink

    def fail_output_reset(path: Path, *, missing_ok: bool = False) -> None:
        if path == output:
            raise OSError("simulated output reset failure")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_output_reset)
    with pytest.raises(OSError, match="output reset failure"):
        RecordExportJSONLWriter(service_id="records-manager", run=checkpoint_run)
    assert not checkpoint.exists()
    assert output.read_bytes() == b'{"previous_run": true}\n'


@pytest.mark.asyncio
async def test_checkpoint_link_failure_does_not_abort_export(
    checkpoint_run: BenchmarkRun,
) -> None:
    """A filesystem without hard links can still export ordinary request metrics."""
    writer = RecordExportJSONLWriter(service_id="records-manager", run=checkpoint_run)
    with (
        patch.object(Path, "hardlink_to", side_effect=OSError("links unavailable")),
        patch.object(writer, "warning") as warning,
    ):
        async with aiperf_lifecycle(writer):
            await writer.process_record(
                create_metric_records_data(results=[{"request_latency": 1_000_000}])
            )
        warning.assert_called_once()
    assert writer.output_file.stat().st_size > 0


@pytest.mark.asyncio
async def test_empty_request_checkpoint_removed_on_shutdown(
    checkpoint_run: BenchmarkRun,
) -> None:
    """An empty run must not leave a misleading recovery artifact."""
    writer = RecordExportJSONLWriter(service_id="records-manager", run=checkpoint_run)
    checkpoint = writer.output_file.parent / "checkpoints" / writer.output_file.name
    async with aiperf_lifecycle(writer):
        assert checkpoint.exists()
    assert not checkpoint.exists()


@pytest.mark.parametrize("live_alias", [True, False])
def test_committed_checkpoint_cleanup_preserves_exports(
    checkpoint_run: BenchmarkRun, live_alias: bool
) -> None:
    """Completed directory copies must have one normal request artifact."""
    output = checkpoint_run.cfg.artifacts.profile_export_jsonl_file
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b'{"request_latency": 42}\n')
    checkpoint = output.parent / "checkpoints" / output.name
    checkpoint.parent.mkdir()
    if live_alias:
        checkpoint.hardlink_to(output)
    else:
        checkpoint.write_bytes(b'{"independent_checkpoint": true}\n')
    write_ready_marker(output.parent)
    remove_committed_request_checkpoint(output)
    assert output.read_bytes() == b'{"request_latency": 42}\n'
    assert checkpoint.exists() is not live_alias
