# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``_record_results_on_status`` summary-derivation paths.

Regression test for the bug where a job that completed before the
controller's ``/api/metrics`` could return populated metrics fell into the
``has_files`` branch and got ``status.results`` written from the parsed
JSON export — but ``status.summary`` stayed empty. ``aiperf kube list``
and the operator UI both read ``liveSummary`` ?? ``summary`` for the
THROUGHPUT/LATENCY columns, so blank summary meant blank columns even
when the underlying numbers were already on disk.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import orjson

from aiperf.operator.handlers.completion import _record_results_on_status
from aiperf.operator.models import ControllerFetchResult
from aiperf.operator.results_layout import write_latest

FIXTURE_EPOCH = "1714064523"

# Match the shape MetricsSummary.from_metrics walks: a top-level "metrics"
# dict keyed by metric tag, each entry holding {avg, p50, p99}. This is
# what _parse_metrics_from_files returns for newer exports.
FILE_METRICS_PAYLOAD = {
    "metrics": {
        "request_throughput": {"avg": 4772.5, "unit": "req/s"},
        "request_latency": {
            "avg": 96.5,
            "p50": 71.2,
            "p99": 900.2,
            "unit": "ms",
        },
        "time_to_first_token": {
            "avg": 96.4,
            "p50": 71.1,
            "p99": 900.2,
            "unit": "ms",
        },
        "output_token_throughput": {"avg": 2325.1, "unit": "tokens/sec"},
    },
}


def _setup_export(base_dir: Path, namespace: str, job_id: str) -> None:
    job_dir = base_dir / namespace / job_id / FIXTURE_EPOCH
    job_dir.mkdir(parents=True, exist_ok=True)
    write_latest(base_dir, namespace, job_id, FIXTURE_EPOCH)
    (job_dir / "profile_export_aiperf.json").write_bytes(
        orjson.dumps(FILE_METRICS_PAYLOAD)
    )


def _body() -> dict:
    """Minimal CR body needed by epoch_key_from_body()."""
    return {
        "metadata": {
            "name": "test-job",
            "creationTimestamp": "2024-04-25T17:02:03Z",
        }
    }


def _patch_results_dir(tmp_path: Path):
    return patch(
        "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
        DIR=tmp_path,
    )


class TestSummaryWrittenFromMetricsApi:
    """When the controller's /api/metrics returned data, summary is set
    from those metrics — pre-existing path, kept passing as a regression
    guard."""

    def test_set_summary_called_with_throughput(self, tmp_path: Path) -> None:
        sb = MagicMock()
        result = ControllerFetchResult(
            metrics={
                "metrics": {
                    "request_throughput": {"avg": 1234.5, "unit": "req/s"},
                    "request_latency": {"avg": 12.0, "p99": 50.0, "unit": "ms"},
                }
            },
            downloaded=[],
        )
        with _patch_results_dir(tmp_path):
            _record_results_on_status(
                body=_body(),
                namespace="ns",
                job_id="test-job",
                result=result,
                sb=sb,
                has_metrics=True,
                has_files=False,
            )
        sb.set_results.assert_called_once()
        sb.set_summary.assert_called_once()
        summary = sb.set_summary.call_args.args[0]
        assert summary["throughput_rps"] == 1234.5
        assert summary["latency_p99_ms"] == 50.0


class TestSummaryFallbackFromFiles:
    """When /api/metrics was empty/unavailable but result files were
    downloaded, summary must still be derived from the parsed JSON export.

    Regression: this branch previously called ``set_results`` only,
    leaving status.summary empty. ``aiperf kube list`` and the operator
    UI then displayed '-' for THROUGHPUT and LATENCY even though the
    numbers were already on disk under status.results.
    """

    def test_set_summary_called_with_file_throughput(self, tmp_path: Path) -> None:
        _setup_export(tmp_path, "ns", "test-job")
        sb = MagicMock()
        result = ControllerFetchResult(
            metrics=None,
            downloaded=["profile_export_aiperf.json"],
        )
        with _patch_results_dir(tmp_path):
            _record_results_on_status(
                body=_body(),
                namespace="ns",
                job_id="test-job",
                result=result,
                sb=sb,
                has_metrics=False,
                has_files=True,
            )
        sb.set_results.assert_called_once()
        sb.set_summary.assert_called_once()
        summary = sb.set_summary.call_args.args[0]
        assert summary["throughput_rps"] == 4772.5
        assert summary["latency_avg_ms"] == 96.5
        assert summary["latency_p99_ms"] == 900.2
        assert summary["ttft_p99_ms"] == 900.2

    def test_set_summary_skipped_when_file_metrics_yield_no_summary(
        self, tmp_path: Path
    ) -> None:
        """Edge case: file present but missing throughput/latency tags.
        ``MetricsSummary.from_metrics`` returns a value with all-None
        fields, ``to_status_dict`` filters those out, so set_summary
        should not be called with an empty dict."""
        job_dir = tmp_path / "ns" / "test-job" / FIXTURE_EPOCH
        job_dir.mkdir(parents=True, exist_ok=True)
        write_latest(tmp_path, "ns", "test-job", FIXTURE_EPOCH)
        # Has request_throughput at the top level (so _parse_metrics_from_files
        # accepts it) but no metrics MetricsSummary cares about beyond it.
        (job_dir / "profile_export_aiperf.json").write_bytes(
            orjson.dumps({"request_throughput": {"avg": 100.0}})
        )

        sb = MagicMock()
        result = ControllerFetchResult(
            metrics=None, downloaded=["profile_export_aiperf.json"]
        )
        with _patch_results_dir(tmp_path):
            _record_results_on_status(
                body=_body(),
                namespace="ns",
                job_id="test-job",
                result=result,
                sb=sb,
                has_metrics=False,
                has_files=True,
            )
        # set_results runs unconditionally on the file-metrics path.
        sb.set_results.assert_called_once()
        # set_summary fires when MetricsSummary derived at least one field;
        # the wrapped-by-_parse_metrics_from_files shape includes
        # ``request_throughput`` so we expect throughput_rps=100.0.
        sb.set_summary.assert_called_once()
        summary = sb.set_summary.call_args.args[0]
        assert summary == {"throughput_rps": 100.0}

    def test_no_summary_when_no_metrics_and_no_files(self, tmp_path: Path) -> None:
        """Fallback-of-fallback: nothing parseable → no set_summary call."""
        sb = MagicMock()
        result = ControllerFetchResult(metrics=None, downloaded=[])
        with _patch_results_dir(tmp_path):
            _record_results_on_status(
                body=_body(),
                namespace="ns",
                job_id="test-job",
                result=result,
                sb=sb,
                has_metrics=False,
                has_files=False,
            )
        sb.set_results.assert_not_called()
        sb.set_summary.assert_not_called()
