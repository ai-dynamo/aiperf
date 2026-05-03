# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RecordExportJSONLWriter.

NOTE: The previous test_record_export_results_processor.py covered
RecordExportResultsProcessor (k8s, BenchmarkRun-based). The class was
renamed to RecordExportJSONLWriter as part of the metrics-accumulator
synthesis. Detailed behavioral coverage is pending re-port.
"""

from aiperf.post_processors.record_export_jsonl_writer import RecordExportJSONLWriter


def test_record_export_jsonl_writer_class_importable() -> None:
    """Smoke test: the renamed class is importable."""
    assert RecordExportJSONLWriter is not None
