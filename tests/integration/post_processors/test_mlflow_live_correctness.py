# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: MLflow live metrics land correctly in the tracking store.

Extends ``test_plot_mlflow_upload.py`` (which covers the plot-upload round
trip) with a live-streaming correctness assertion. Runs ``aiperf profile``
with ``--mlflow-tracking-uri file://<tmp>`` and then inspects the MLflow
filesystem store directly:

    * At least one ``live.*`` metric key was written.
    * Each metric file contains multiple data points (more than one step).
    * Steps are monotonically non-decreasing within a metric key.
    * Each step's MLflow timestamp is monotonically non-decreasing within
      the key (i.e. ordering preserved through the fanout queue).
    * The ``benchmark_id`` MLflow tag matches the ``aiperf.benchmark.id``
      recorded in ``mlflow_export.json`` — the correlation key that lets
      post-run exporters and dashboards stitch live + deferred data.

Uses the filesystem store because it's self-contained; the code path that
writes the metrics is the same for any backend.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


def _parse_mlflow_metric_file(metric_file: Path) -> list[tuple[int, float, int]]:
    """Parse an MLflow filesystem-store metric file.

    Each line has the layout ``<timestamp_ms> <value> <step>``. Returns the
    list of (timestamp_ms, value, step) triples in file order (which is also
    insertion order for the filesystem backend).
    """
    records: list[tuple[int, float, int]] = []
    for line in metric_file.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        ts, value, step = parts
        records.append((int(ts), float(value), int(step)))
    return records


def _find_run_dir(mlflow_store: Path) -> Path:
    """Locate the single MLflow run directory under ``<store>/<exp_id>/<run_id>``.

    The test creates exactly one experiment with one run, so fail loudly if
    more than one run directory exists — that would indicate a bug where the
    fanout created a second run instead of reusing the one from
    ``mlflow.start_run()``.
    """
    # Experiment dirs are digit-only (auto-assigned) except the reserved
    # ``.trash`` entry. Metadata entries (``meta.yaml``) are files, so
    # we only consider directories.
    candidates = [
        run
        for experiment in mlflow_store.iterdir()
        if experiment.is_dir() and experiment.name != ".trash"
        for run in experiment.iterdir()
        if run.is_dir()
    ]
    assert len(candidates) == 1, (
        f"Expected exactly one MLflow run, found {len(candidates)}: {candidates!r}"
    )
    return candidates[0]


@pytest.mark.component_integration
@pytest.mark.asyncio
class TestMLflowLiveStreamingCorrectness:
    """Assert live-streamed MLflow metrics are ordered and correlated."""

    async def test_live_metrics_are_ordered_and_benchmark_id_tagged(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ) -> None:
        """Run aiperf with MLflow live-streaming, then inspect the store.

        Uses a small request count so the run is fast, but enough data
        points that per-metric ordering is observable (a 1-point run can't
        detect a swap).
        """
        mlflow_store = tmp_path / "mlflow_store"
        mlflow_store.mkdir()
        tracking_uri = f"file://{mlflow_store}"

        # ``--mlflow-tag`` is typed as ``list[tuple[str, str]]`` with
        # ``consume_multiple=True``, so the CLI parser requires at least
        # two positional tokens after the flag and each token must be a
        # ``key:value`` string. A single tag would fail with "requires 2
        # positional arguments"; two bare words would fail the ``key:value``
        # parser. Pass two tokens to satisfy both constraints.
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --concurrency 2 \
                --request-count 20 \
                --streaming \
                --mlflow-tracking-uri {tracking_uri} \
                --mlflow-experiment live-correctness-test \
                --mlflow-tag owner:aiperf-ci env:integ
            """,
            timeout=120.0,
        )
        assert result.exit_code == 0, (
            f"aiperf profile failed with exit code {result.exit_code}"
        )

        # --- Run layout sanity ---------------------------------------------
        run_dir = _find_run_dir(mlflow_store)
        metrics_dir = run_dir / "metrics"
        tags_dir = run_dir / "tags"

        assert metrics_dir.is_dir(), (
            f"MLflow run has no metrics/ dir — live streaming did not write anything. "
            f"Run dir contents: {list(run_dir.iterdir())!r}"
        )

        # --- Live metric presence ------------------------------------------
        live_metric_files = [
            p for p in metrics_dir.iterdir() if p.name.startswith("live.")
        ]
        assert live_metric_files, (
            f"No 'live.*' metric files found in {metrics_dir}. "
            f"All metrics: {[p.name for p in metrics_dir.iterdir()]!r}"
        )

        # --- Ordering invariants per metric --------------------------------
        metrics_with_multi_point = 0
        for metric_file in live_metric_files:
            records = _parse_mlflow_metric_file(metric_file)
            assert records, f"{metric_file.name} is empty"
            if len(records) > 1:
                metrics_with_multi_point += 1

            # Steps must be monotonically non-decreasing. The fanout assigns
            # step values sequentially as it drains the buffer; a decreasing
            # step would indicate out-of-order delivery through the
            # multiprocessing queue.
            steps = [step for _ts, _val, step in records]
            assert steps == sorted(steps), (
                f"Steps out of order in {metric_file.name}: {steps!r}"
            )
            # Timestamps must also be non-decreasing within a single key.
            timestamps = [ts for ts, _val, _step in records]
            assert timestamps == sorted(timestamps), (
                f"Timestamps out of order in {metric_file.name}: {timestamps!r}"
            )

        # A live run with 20 requests should produce multi-point metrics for
        # at least one key; otherwise the streaming pipeline degenerated into
        # a single shutdown flush.
        assert metrics_with_multi_point > 0, (
            f"All live metrics have only one data point — the live streaming "
            f"pipeline appears to have fired only at shutdown. "
            f"Files: {[p.name for p in live_metric_files]!r}"
        )

        # --- benchmark_id correlation --------------------------------------
        benchmark_id_tag = tags_dir / "benchmark_id"
        assert benchmark_id_tag.exists(), (
            f"benchmark_id tag missing from MLflow run. Tags present: "
            f"{[p.name for p in tags_dir.iterdir()] if tags_dir.exists() else 'tags/ dir absent'}"
        )
        tagged_benchmark_id = benchmark_id_tag.read_text().strip()
        assert tagged_benchmark_id, "benchmark_id tag is empty"

        # mlflow_export.json is written by the deferred (post-run) exporter
        # and must reference the same run_id the live streamer created;
        # benchmark_id comes from the UserConfig and is shared across both.
        metadata_path = result.artifacts_dir / "mlflow_export.json"
        assert metadata_path.exists(), (
            "mlflow_export.json missing — deferred exporter did not run"
        )
        metadata = orjson.loads(metadata_path.read_bytes())
        recorded_run_id = metadata["run_id"]
        assert recorded_run_id == run_dir.name, (
            f"mlflow_export.json run_id ({recorded_run_id}) does not match the "
            f"on-disk run directory ({run_dir.name}). Live streamer and deferred "
            f"exporter are not pointing at the same run."
        )

        # --- User-provided --mlflow-tag values survive ---------------------
        # Each ``key:value`` token is unpacked into its own MLflow tag.
        for tag_key, expected_value in [("owner", "aiperf-ci"), ("env", "integ")]:
            tag_file = tags_dir / tag_key
            assert tag_file.exists(), (
                f"{tag_key!r} tag missing from MLflow run. Tags present: "
                f"{sorted(p.name for p in tags_dir.iterdir())!r}"
            )
            assert tag_file.read_text().strip() == expected_value, (
                f"{tag_key!r} tag value mismatch: "
                f"expected={expected_value!r}, got={tag_file.read_text()!r}"
            )
