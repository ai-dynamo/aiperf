# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for _parse_metrics_from_files in completion handler.

Focuses on:
- Parsing from .json files
- Parsing from .json.zst files
- Missing files, empty files, corrupted JSON
- Validation of metric structure
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import orjson
import pytest
import zstandard
from pytest import param

from aiperf.operator.handlers.completion import _parse_metrics_from_files
from aiperf.operator.results_layout import write_latest

# ============================================================
# Helpers
# ============================================================

# Fixed epoch for deterministic on-disk layout in unit tests
# (matches metadata.creationTimestamp=2024-04-25T17:02:03Z).
FIXTURE_EPOCH = "1714064523"


def _setup_result_file(
    base_dir: Path,
    namespace: str,
    job_id: str,
    data: dict | bytes | None = None,
    *,
    compress: bool = False,
    filename: str = "profile_export_aiperf.json",
) -> None:
    """Write a result file in the expected directory structure."""
    job_dir = base_dir / namespace / job_id / FIXTURE_EPOCH
    job_dir.mkdir(parents=True, exist_ok=True)
    write_latest(base_dir, namespace, job_id, FIXTURE_EPOCH)

    if data is None:
        return

    content = orjson.dumps(data) if isinstance(data, dict) else data

    if compress:
        cctx = zstandard.ZstdCompressor()
        (job_dir / (filename + ".zst")).write_bytes(cctx.compress(content))
    else:
        (job_dir / filename).write_bytes(content)


VALID_METRICS = {
    "request_throughput": {"avg": 100.0, "p50": 90.0, "p99": 150.0, "unit": "req/s"},
    "request_latency": {"avg": 50.0, "p50": 45.0, "p99": 120.0, "unit": "ms"},
}


def _patch_results_dir(tmp_path: Path):
    """Patch OperatorEnvironment.RESULTS.DIR to use tmp_path."""
    return patch(
        "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
        DIR=tmp_path,
    )


# ============================================================
# Happy Path
# ============================================================


class TestParseMetricsHappyPath:
    """Verify successful metric parsing from result files."""

    def test_parse_from_json_file(self, tmp_path: Path) -> None:
        _setup_result_file(tmp_path, "ns", "job-1", VALID_METRICS)

        with _patch_results_dir(tmp_path):
            result = _parse_metrics_from_files(
                ["profile_export_aiperf.json"], "ns", "job-1", epoch=FIXTURE_EPOCH
            )

        assert result is not None
        assert result["request_throughput"]["avg"] == 100.0

    def test_parse_from_zst_file(self, tmp_path: Path) -> None:
        _setup_result_file(tmp_path, "ns", "job-1", VALID_METRICS, compress=True)

        with _patch_results_dir(tmp_path):
            result = _parse_metrics_from_files(
                ["profile_export_aiperf.json.zst"], "ns", "job-1", epoch=FIXTURE_EPOCH
            )

        assert result is not None
        assert result["request_throughput"]["avg"] == 100.0

    def test_prefers_zst_over_json(self, tmp_path: Path) -> None:
        # Write both variants with different data
        _setup_result_file(tmp_path, "ns", "job-1", VALID_METRICS)
        zst_metrics = {
            **VALID_METRICS,
            "request_throughput": {**VALID_METRICS["request_throughput"], "avg": 999.0},
        }
        _setup_result_file(tmp_path, "ns", "job-1", zst_metrics, compress=True)

        with _patch_results_dir(tmp_path):
            result = _parse_metrics_from_files(
                ["profile_export_aiperf.json", "profile_export_aiperf.json.zst"],
                "ns",
                "job-1",
                epoch=FIXTURE_EPOCH,
            )

        assert result is not None
        assert result["request_throughput"]["avg"] == 999.0


# ============================================================
# Missing / Empty Files
# ============================================================


class TestParseMetricsMissingFiles:
    """Verify handling of missing or empty files."""

    def test_no_summary_file_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "ns" / "job-1" / FIXTURE_EPOCH).mkdir(parents=True)
        write_latest(tmp_path, "ns", "job-1", FIXTURE_EPOCH)

        with _patch_results_dir(tmp_path):
            result = _parse_metrics_from_files(
                ["other.json"], "ns", "job-1", epoch=FIXTURE_EPOCH
            )

        assert result is None

    def test_empty_downloaded_list_returns_none(self, tmp_path: Path) -> None:
        with _patch_results_dir(tmp_path):
            result = _parse_metrics_from_files([], "ns", "job-1", epoch=FIXTURE_EPOCH)

        assert result is None

    def test_dir_not_exists_returns_none(self, tmp_path: Path) -> None:
        with _patch_results_dir(tmp_path):
            result = _parse_metrics_from_files(
                ["profile_export_aiperf.json"],
                "ns",
                "nonexistent",
                epoch=FIXTURE_EPOCH,
            )

        assert result is None


# ============================================================
# Corrupted / Invalid Data
# ============================================================


class TestParseMetricsCorruptedData:
    """Verify handling of corrupted or invalid data."""

    def test_invalid_json_returns_none(self, tmp_path: Path) -> None:
        _setup_result_file(tmp_path, "ns", "job-1", b"not valid json")

        with _patch_results_dir(tmp_path):
            result = _parse_metrics_from_files(
                ["profile_export_aiperf.json"], "ns", "job-1", epoch=FIXTURE_EPOCH
            )

        assert result is None

    def test_json_without_request_throughput_returns_none(self, tmp_path: Path) -> None:
        _setup_result_file(tmp_path, "ns", "job-1", {"other_metric": {"avg": 1.0}})

        with _patch_results_dir(tmp_path):
            result = _parse_metrics_from_files(
                ["profile_export_aiperf.json"], "ns", "job-1", epoch=FIXTURE_EPOCH
            )

        assert result is None

    def test_json_array_not_dict_returns_none(self, tmp_path: Path) -> None:
        _setup_result_file(tmp_path, "ns", "job-1", b"[1, 2, 3]")

        with _patch_results_dir(tmp_path):
            result = _parse_metrics_from_files(
                ["profile_export_aiperf.json"], "ns", "job-1", epoch=FIXTURE_EPOCH
            )

        assert result is None

    def test_corrupted_zst_file_returns_none(self, tmp_path: Path) -> None:
        job_dir = tmp_path / "ns" / "job-1" / FIXTURE_EPOCH
        job_dir.mkdir(parents=True)
        write_latest(tmp_path, "ns", "job-1", FIXTURE_EPOCH)
        (job_dir / "profile_export_aiperf.json.zst").write_bytes(b"not zstd data")

        with _patch_results_dir(tmp_path):
            result = _parse_metrics_from_files(
                ["profile_export_aiperf.json.zst"], "ns", "job-1", epoch=FIXTURE_EPOCH
            )

        assert result is None

    def test_empty_json_file_returns_none(self, tmp_path: Path) -> None:
        _setup_result_file(tmp_path, "ns", "job-1", b"")

        with _patch_results_dir(tmp_path):
            result = _parse_metrics_from_files(
                ["profile_export_aiperf.json"], "ns", "job-1", epoch=FIXTURE_EPOCH
            )

        assert result is None

    def test_non_json_zst_candidate_does_not_block_valid_json(
        self, tmp_path: Path
    ) -> None:
        """A non-JSON .zst sibling (e.g. profile_export.jsonl.zst, which decompresses
        to JSONL not JSON) must not abort the candidate scan before the valid
        ``profile_export_aiperf.json`` is reached. The candidate sort puts .zst
        first, so a per-loop bail-out skips the only file we actually want.
        """
        cctx = zstandard.ZstdCompressor()
        job_dir = tmp_path / "ns" / "job-1" / FIXTURE_EPOCH
        job_dir.mkdir(parents=True)
        write_latest(tmp_path, "ns", "job-1", FIXTURE_EPOCH)
        # Mimic the real on-disk layout: jsonl.zst decompresses fine but is
        # newline-delimited JSON, so orjson.loads raises JSONDecodeError.
        jsonl_payload = b'{"a": 1}\n{"b": 2}\n'
        (job_dir / "profile_export.jsonl.zst").write_bytes(cctx.compress(jsonl_payload))
        (job_dir / "profile_export_aiperf.json").write_bytes(
            orjson.dumps(VALID_METRICS)
        )

        with _patch_results_dir(tmp_path):
            result = _parse_metrics_from_files(
                ["profile_export.jsonl.zst", "profile_export_aiperf.json"],
                "ns",
                "job-1",
                epoch=FIXTURE_EPOCH,
            )

        assert result is not None
        assert result["request_throughput"]["avg"] == 100.0

    @pytest.mark.parametrize(
        "bad_first_writer,bad_kind",
        [
            param(
                lambda p: p.write_bytes(b"\x00\x01\x02 not zstd"),
                "ZstdError",
                id="corrupt-zst-bytes",
            ),
            param(
                lambda p: p.write_bytes(b'{"truncated":'),
                "ValueError-or-JSONDecode",
                id="malformed-json-truncated",
            ),
            param(
                # Directory at the candidate path -> open() raises IsADirectoryError (OSError)
                lambda p: p.mkdir(),
                "OSError-isdir",
                id="path-is-directory",
            ),
        ],
    )  # fmt: skip
    def test_bad_first_candidate_does_not_block_valid_second(
        self, tmp_path: Path, bad_first_writer, bad_kind: str
    ) -> None:
        """All four exception types in the per-file except clause
        ``(OSError, ValueError, orjson.JSONDecodeError, zstandard.ZstdError)``
        must be skipped over so a valid ``profile_export_aiperf.json`` later in
        the candidate list still resolves. Parametrized over the three
        non-JSONDecode shapes (the JSONDecode path is covered by
        ``test_non_json_zst_candidate_does_not_block_valid_json`` above).
        """
        job_dir = tmp_path / "ns" / "job-bad-first" / FIXTURE_EPOCH
        job_dir.mkdir(parents=True)
        write_latest(tmp_path, "ns", "job-bad-first", FIXTURE_EPOCH)
        # First candidate path is bad in some way; the writer lambda decides how.
        bad_first_writer(job_dir / "profile_export.jsonl.zst")
        # Valid second candidate.
        (job_dir / "profile_export_aiperf.json").write_bytes(
            orjson.dumps(VALID_METRICS)
        )

        with _patch_results_dir(tmp_path):
            result = _parse_metrics_from_files(
                ["profile_export.jsonl.zst", "profile_export_aiperf.json"],
                "ns",
                "job-bad-first",
                epoch=FIXTURE_EPOCH,
            )

        assert result is not None, f"valid second candidate dropped after {bad_kind}"
        assert result["request_throughput"]["avg"] == 100.0
