# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for MLflow metadata round-trip including parent_run_id.

Validates that the final mlflow_export.json written to disk matches what is
uploaded to MLflow, and that parent_run_id is correctly propagated through
the export pipeline.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Generator
from pathlib import Path
from typing import Any

import orjson
import pytest
from pytest import param

from aiperf.common.config import EndpointConfig, OutputConfig, ServiceConfig, UserConfig
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.mlflow_data_exporter import MLflowDataExporter
from aiperf.plugin.enums import EndpointType


def _write_artifact(path: Path, content: str = "test") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _install_fake_mlflow_with_parent_tracking() -> dict[str, Any]:
    """Install fake mlflow that tracks parent_run_id in start_run calls."""
    state: dict[str, Any] = {
        "start_run_calls": [],
        "artifacts_uploaded": {},
        "run_id_counter": 0,
        # run_id -> pre-existing run_name (simulates a live-streaming run that
        # MLflow already named before the deferred exporter reuses it).
        "live_run_names": {},
    }

    class FakeMetric:
        def __init__(self, key: str, value: float, timestamp: int, step: int) -> None:
            pass

    class FakeParam:
        def __init__(self, key: str, value: str) -> None:
            pass

    class FakeRunTag:
        def __init__(self, key: str, value: str) -> None:
            pass

    class FakeMlflowClient:
        def log_batch(self, **kwargs: Any) -> None:
            pass

    class FakeRunContext:
        def __init__(self, run_id: str, run_name: str | None) -> None:
            self._run_id = run_id
            self._run_name = run_name

        def __enter__(self) -> Any:
            info = types.SimpleNamespace(run_id=self._run_id, run_name=self._run_name)
            return types.SimpleNamespace(info=info)

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

    mlflow_module = types.ModuleType("mlflow")
    entities_module = types.ModuleType("mlflow.entities")
    tracking_module = types.ModuleType("mlflow.tracking")

    def set_tracking_uri(uri: str) -> None:
        pass

    def set_experiment(name: str) -> None:
        pass

    def start_run(**kwargs: Any) -> FakeRunContext:
        state["start_run_calls"].append(kwargs)
        run_id = kwargs.get("run_id") or f"new-run-{state['run_id_counter']}"
        state["run_id_counter"] += 1
        # Mimic MLflow: reusing by run_id returns the stored run_name (if the
        # test registered one); a fresh start_run echoes back the supplied name
        # or None when the caller did not pass one.
        if kwargs.get("run_id") is not None:
            run_name = state.get("live_run_names", {}).get(kwargs["run_id"])
        else:
            run_name = kwargs.get("run_name")
        return FakeRunContext(run_id, run_name)

    def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
        state["artifacts_uploaded"][local_path] = Path(local_path).read_bytes()

    mlflow_module.set_tracking_uri = set_tracking_uri  # type: ignore[attr-defined]
    mlflow_module.set_experiment = set_experiment  # type: ignore[attr-defined]
    mlflow_module.start_run = start_run  # type: ignore[attr-defined]
    mlflow_module.log_artifact = log_artifact  # type: ignore[attr-defined]
    mlflow_module.entities = entities_module  # type: ignore[attr-defined]
    mlflow_module.tracking = tracking_module  # type: ignore[attr-defined]

    entities_module.Metric = FakeMetric  # type: ignore[attr-defined]
    entities_module.Param = FakeParam  # type: ignore[attr-defined]
    entities_module.RunTag = FakeRunTag  # type: ignore[attr-defined]
    tracking_module.MlflowClient = FakeMlflowClient  # type: ignore[attr-defined]

    sys.modules["mlflow"] = mlflow_module
    sys.modules["mlflow.entities"] = entities_module
    sys.modules["mlflow.tracking"] = tracking_module
    return state


def _cleanup_fake_mlflow() -> None:
    """Remove fake mlflow modules from sys.modules."""
    sys.modules.pop("mlflow", None)
    sys.modules.pop("mlflow.entities", None)
    sys.modules.pop("mlflow.tracking", None)


@pytest.fixture(autouse=True)
def _restore_mlflow_modules() -> Generator[None, None, None]:
    """Ensure fake mlflow modules are cleaned up after each test."""
    yield
    _cleanup_fake_mlflow()


@pytest.fixture
def sample_results() -> ProfileResults:
    return ProfileResults(
        records=[
            MetricResult(
                tag="request_throughput",
                header="Request Throughput",
                unit="req/s",
                avg=42.5,
            ),
        ],
        total_expected=10,
        completed=10,
        start_ns=0,
        end_ns=1,
        was_cancelled=False,
        error_summary=[],
    )


@pytest.mark.parametrize(
    "parent_run_id, expect_parent_in_metadata",
    [
        param("parent-run-abc123", "parent-run-abc123", id="with-parent-run-id"),
        param(None, None, id="without-parent-run-id"),
    ],
)  # fmt: skip
class TestMLflowMetadataRoundtripParentRunId:
    """Verify parent_run_id propagation through the MLflow export pipeline."""

    def test_new_run_parent_run_id_roundtrip(
        self,
        tmp_path: Path,
        sample_results: ProfileResults,
        parent_run_id: str | None,
        expect_parent_in_metadata: str | None,
    ) -> None:
        """New-run path: parent_run_id from CLI appears in final metadata."""
        _write_artifact(tmp_path / "profile_export_aiperf.json")
        # Pre-create mlflow_export.json so it's found by _collect_artifact_files
        # and uploaded alongside other artifacts.
        (tmp_path / "mlflow_export.json").write_bytes(orjson.dumps({}))

        user_config = UserConfig(
            endpoint=EndpointConfig(
                type=EndpointType.CHAT,
                model_names=["test-model"],
                urls=["http://localhost:8000"],
            ),
            output=OutputConfig(artifact_directory=tmp_path),
            mlflow_tracking_uri="http://mlflow:5000",
            mlflow_experiment="roundtrip-test",
            mlflow_parent_run_id=parent_run_id,
        )

        state = _install_fake_mlflow_with_parent_tracking()
        exporter_config = ExporterConfig(
            results=sample_results,
            user_config=user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )
        exporter = MLflowDataExporter(exporter_config)
        exporter._export_sync()

        # Verify the start_run call included parent_run_id when set
        start_call = state["start_run_calls"][0]
        if parent_run_id:
            assert start_call.get("parent_run_id") == parent_run_id
        else:
            assert "parent_run_id" not in start_call

        # Verify the final metadata on disk has the correct parent_run_id
        final_metadata = orjson.loads((tmp_path / "mlflow_export.json").read_bytes())
        assert final_metadata["parent_run_id"] == expect_parent_in_metadata

        # Verify the uploaded metadata matches the on-disk metadata (byte equality)
        metadata_path = str(tmp_path / "mlflow_export.json")
        assert metadata_path in state["artifacts_uploaded"]
        uploaded_bytes = state["artifacts_uploaded"][metadata_path]
        disk_bytes = (tmp_path / "mlflow_export.json").read_bytes()
        assert uploaded_bytes == disk_bytes

    def test_reuse_run_parent_run_id_from_metadata(
        self,
        tmp_path: Path,
        sample_results: ProfileResults,
        parent_run_id: str | None,
        expect_parent_in_metadata: str | None,
    ) -> None:
        """Reuse path: parent_run_id from live metadata is preserved, CLI is ignored."""
        _write_artifact(tmp_path / "profile_export_aiperf.json")

        benchmark_id = "bench-roundtrip-001"
        live_parent = "live-parent-xyz"
        live_metadata = {
            "tracking_uri": "http://mlflow:5000",
            "experiment": "roundtrip-test",
            "run_id": "live-run-999",
            "run_name": "live-run",
            "benchmark_id": benchmark_id,
            "parent_run_id": live_parent,
            "live_streaming": True,
        }
        (tmp_path / "mlflow_export.json").write_bytes(orjson.dumps(live_metadata))

        user_config = UserConfig(
            endpoint=EndpointConfig(
                type=EndpointType.CHAT,
                model_names=["test-model"],
                urls=["http://localhost:8000"],
            ),
            output=OutputConfig(artifact_directory=tmp_path),
            mlflow_tracking_uri="http://mlflow:5000",
            mlflow_experiment="roundtrip-test",
            mlflow_parent_run_id=parent_run_id,
            benchmark_id=benchmark_id,
        )

        state = _install_fake_mlflow_with_parent_tracking()
        exporter_config = ExporterConfig(
            results=sample_results,
            user_config=user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )
        exporter = MLflowDataExporter(exporter_config)
        exporter._export_sync()

        # On reuse, start_run uses run_id (not parent_run_id)
        start_call = state["start_run_calls"][0]
        assert start_call.get("run_id") == "live-run-999"
        assert "parent_run_id" not in start_call

        # Final metadata preserves the live parent_run_id, ignoring CLI value
        final_metadata = orjson.loads((tmp_path / "mlflow_export.json").read_bytes())
        assert final_metadata["parent_run_id"] == live_parent
        assert final_metadata["reused_live_run"] is True


class TestMLflowMetadataByteEqualityRoundtrip:
    """Verify byte-equality of uploaded vs local mlflow_export.json with multiple artifacts."""

    def test_metadata_byte_equality_with_multiple_artifacts(
        self,
        tmp_path: Path,
        sample_results: ProfileResults,
    ) -> None:
        """Uploaded mlflow_export.json bytes equal on-disk bytes after export completes."""
        _write_artifact(tmp_path / "profile_export_aiperf.json", "profile-data")
        _write_artifact(tmp_path / "summary.csv", "col1,col2\n1,2")
        _write_artifact(tmp_path / "plots" / "throughput.png", "fake-png")
        _write_artifact(tmp_path / "timeslices.jsonl", '{"ts":1}')

        user_config = UserConfig(
            endpoint=EndpointConfig(
                type=EndpointType.CHAT,
                model_names=["test-model"],
                urls=["http://localhost:8000"],
            ),
            output=OutputConfig(artifact_directory=tmp_path),
            mlflow_tracking_uri="http://mlflow:5000",
            mlflow_experiment="byte-equality-test",
        )

        state = _install_fake_mlflow_with_parent_tracking()
        exporter_config = ExporterConfig(
            results=sample_results,
            user_config=user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )
        exporter = MLflowDataExporter(exporter_config)
        exporter._export_sync()

        metadata_path = str(tmp_path / "mlflow_export.json")

        # mlflow_export.json was uploaded
        assert metadata_path in state["artifacts_uploaded"]

        # Byte-equality: uploaded bytes == final disk bytes
        uploaded_bytes = state["artifacts_uploaded"][metadata_path]
        disk_bytes = (tmp_path / "mlflow_export.json").read_bytes()
        assert uploaded_bytes == disk_bytes

        # Verify metadata content includes all expected artifacts + mlflow_export.json
        metadata = orjson.loads(disk_bytes)
        assert "mlflow_export.json" in metadata["uploaded_artifacts"]
        assert "profile_export_aiperf.json" in metadata["uploaded_artifacts"]
        assert "summary.csv" in metadata["uploaded_artifacts"]
        assert "plots/throughput.png" in metadata["uploaded_artifacts"]
        assert "timeslices.jsonl" in metadata["uploaded_artifacts"]

    def test_metadata_byte_equality_no_extra_artifacts(
        self,
        tmp_path: Path,
        sample_results: ProfileResults,
    ) -> None:
        """Even with no matching artifact files, mlflow_export.json is still uploaded."""
        user_config = UserConfig(
            endpoint=EndpointConfig(
                type=EndpointType.CHAT,
                model_names=["test-model"],
                urls=["http://localhost:8000"],
            ),
            output=OutputConfig(artifact_directory=tmp_path),
            mlflow_tracking_uri="http://mlflow:5000",
            mlflow_experiment="empty-artifacts-test",
            mlflow_artifact_globs=["nonexistent_pattern_*.xyz"],
        )

        state = _install_fake_mlflow_with_parent_tracking()
        exporter_config = ExporterConfig(
            results=sample_results,
            user_config=user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )
        exporter = MLflowDataExporter(exporter_config)
        exporter._export_sync()

        metadata_path = str(tmp_path / "mlflow_export.json")
        assert metadata_path in state["artifacts_uploaded"]

        uploaded_bytes = state["artifacts_uploaded"][metadata_path]
        disk_bytes = (tmp_path / "mlflow_export.json").read_bytes()
        assert uploaded_bytes == disk_bytes

        metadata = orjson.loads(disk_bytes)
        assert metadata["uploaded_artifacts"] == ["mlflow_export.json"]


class TestMLflowTrackingUriRedactedInMetadata:
    """Verify tracking URI userinfo is redacted before persistence + upload."""

    def test_credentialed_tracking_uri_redacted_in_on_disk_metadata(
        self,
        tmp_path: Path,
        sample_results: ProfileResults,
    ) -> None:
        """Regression: mlflow_export.json is a run artifact, so credentials in
        --mlflow-tracking-uri must never be written verbatim (they would then
        round-trip through the uploaded artifact)."""
        _write_artifact(tmp_path / "profile_export_aiperf.json")

        user_config = UserConfig(
            endpoint=EndpointConfig(
                type=EndpointType.CHAT,
                model_names=["test-model"],
                urls=["http://localhost:8000"],
            ),
            output=OutputConfig(artifact_directory=tmp_path),
            mlflow_tracking_uri="postgresql://dbuser:s3cret@db:5432/mlflow",
            mlflow_experiment="redaction-test",
        )

        state = _install_fake_mlflow_with_parent_tracking()
        exporter_config = ExporterConfig(
            results=sample_results,
            user_config=user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )
        exporter = MLflowDataExporter(exporter_config)
        exporter._export_sync()

        metadata_path = tmp_path / "mlflow_export.json"
        final_metadata = orjson.loads(metadata_path.read_bytes())
        assert final_metadata["tracking_uri"] == (
            "postgresql://<redacted>@db:5432/mlflow"
        )
        assert "s3cret" not in metadata_path.read_text(encoding="utf-8")
        assert "dbuser" not in metadata_path.read_text(encoding="utf-8")

        # MLflow client calls still receive the real URI (needs to authenticate).
        metadata_bytes = metadata_path.read_bytes()
        uploaded = state["artifacts_uploaded"][str(metadata_path)]
        assert uploaded == metadata_bytes  # byte-equality preserved

    def test_reuse_check_accepts_redacted_on_disk_with_credentialed_in_memory(
        self,
        tmp_path: Path,
        sample_results: ProfileResults,
    ) -> None:
        """Reuse still works: live metadata stores redacted URI, in-memory URI has creds."""
        benchmark_id = "bench-redact-001"
        live_metadata = {
            "tracking_uri": "postgresql://<redacted>@db:5432/mlflow",
            "experiment": "redaction-test",
            "run_id": "live-run-redacted",
            "run_name": "live-run",
            "benchmark_id": benchmark_id,
            "live_streaming": True,
        }
        (tmp_path / "mlflow_export.json").write_bytes(orjson.dumps(live_metadata))
        _write_artifact(tmp_path / "profile_export_aiperf.json")

        user_config = UserConfig(
            endpoint=EndpointConfig(
                type=EndpointType.CHAT,
                model_names=["test-model"],
                urls=["http://localhost:8000"],
            ),
            output=OutputConfig(artifact_directory=tmp_path),
            mlflow_tracking_uri="postgresql://dbuser:s3cret@db:5432/mlflow",
            mlflow_experiment="redaction-test",
            benchmark_id=benchmark_id,
        )

        state = _install_fake_mlflow_with_parent_tracking()
        exporter_config = ExporterConfig(
            results=sample_results,
            user_config=user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )
        exporter = MLflowDataExporter(exporter_config)
        exporter._export_sync()

        # Reuse path: start_run was invoked with run_id from live metadata.
        start_call = state["start_run_calls"][0]
        assert start_call.get("run_id") == "live-run-redacted"

        final_metadata = orjson.loads((tmp_path / "mlflow_export.json").read_bytes())
        assert final_metadata["reused_live_run"] is True
        assert final_metadata["tracking_uri"] == (
            "postgresql://<redacted>@db:5432/mlflow"
        )
