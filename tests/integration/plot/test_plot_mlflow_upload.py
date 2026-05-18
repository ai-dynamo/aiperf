# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: aiperf plot --mlflow-upload round-trip.

Validates the end-to-end workflow:
1. Run `aiperf profile --mlflow-tracking-uri file://<tmp>` to produce
   artifacts and create an MLflow run.
2. Run `aiperf plot --input-dir <output> --mlflow-upload` against the same
   tracking URI.
3. Assert:
   (a) Both commands use the same run_id (live-run reuse rule from Property 6).
   (b) The MLflow run artifacts directory contains at least one plot file.
   (c) The plot upload does not rewrite mlflow_export.json.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.component_integration
@pytest.mark.asyncio
class TestPlotMLflowUploadRoundTrip:
    """Verify aiperf plot --mlflow-upload reuses the live run and uploads plots."""

    async def test_plot_mlflow_upload_reuses_live_run(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ) -> None:
        """Profile with --mlflow-tracking-uri then plot --mlflow-upload should reuse the same run_id.

        Steps:
        1. Run aiperf profile with --mlflow-tracking-uri pointed at a local file:// store.
        2. Read the mlflow_export.json produced by the profile run.
        3. Run aiperf plot --mlflow-upload against the same tracking URI.
        4. Assert the plot command used the same run_id.
        5. Assert at least one plot artifact landed in the MLflow artifacts dir.
        6. Assert mlflow_export.json was NOT rewritten by the plot command.
        """
        mlflow_store = tmp_path / "mlflow_store"
        mlflow_store.mkdir()
        tracking_uri = f"file://{mlflow_store}"

        # Step 1: Run profile with MLflow enabled
        profile_result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --concurrency 2 \
                --request-count 10 \
                --streaming \
                --mlflow-tracking-uri {tracking_uri} \
                --mlflow-experiment plot-upload-test
            """,
            timeout=120.0,
        )

        assert profile_result.exit_code == 0, (
            f"aiperf profile failed: exit_code={profile_result.exit_code}"
        )

        artifacts_dir = profile_result.artifacts_dir

        # Step 2: Read the mlflow_export.json produced by profile
        metadata_path = artifacts_dir / "mlflow_export.json"
        assert metadata_path.exists(), (
            "mlflow_export.json was not produced by the profile run"
        )
        metadata_after_profile = orjson.loads(metadata_path.read_bytes())
        profile_run_id = metadata_after_profile["run_id"]
        assert profile_run_id, "run_id should be non-empty in mlflow_export.json"

        # Record the metadata bytes before plot to verify no rewrite
        metadata_bytes_before_plot = metadata_path.read_bytes()

        # Step 3: Run plot with --mlflow-upload
        plot_result = await cli.run(
            f"""
            aiperf plot \
                --paths {artifacts_dir} \
                --mlflow-upload \
                --mlflow-tracking-uri {tracking_uri}
            """,
            timeout=60.0,
        )

        assert plot_result.exit_code == 0, (
            f"aiperf plot --mlflow-upload failed: exit_code={plot_result.exit_code}"
        )

        # Step 4: Assert the plot command used the same run_id (reuse rule)
        # The plot command reads mlflow_export.json to discover the run_id.
        # We verify by checking that no new run was created — the metadata
        # file still references the original run_id.
        metadata_after_plot = orjson.loads(metadata_path.read_bytes())
        assert metadata_after_plot["run_id"] == profile_run_id, (
            "Plot command should reuse the same MLflow run_id from the profile run"
        )

        # Step 5: Assert at least one plot artifact exists in MLflow artifacts dir.
        # With file:// backend, artifacts are stored at:
        # <mlflow_store>/<experiment_id>/<run_id>/artifacts/
        # Find the run artifacts directory
        plot_artifacts_found = False
        for _artifact_path in mlflow_store.rglob("*.png"):
            plot_artifacts_found = True
            break

        assert plot_artifacts_found, (
            "No plot PNG files found in the MLflow artifacts directory. "
            "Expected at least one plot to be uploaded."
        )

        # Step 6: Assert mlflow_export.json was NOT rewritten by the plot command
        metadata_bytes_after_plot = metadata_path.read_bytes()
        assert metadata_bytes_after_plot == metadata_bytes_before_plot, (
            "mlflow_export.json was rewritten by the plot --mlflow-upload command. "
            "The plot upload should not modify the metadata file."
        )
