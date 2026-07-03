# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for ``RecordExportJSONLWriter``.

Detailed behavioral coverage (initialization, process_record, file format,
HTTP trace, lifecycle) is pending.
"""

from aiperf.post_processors.record_export_jsonl_writer import RecordExportJSONLWriter


def test_record_export_jsonl_writer_class_importable() -> None:
    """The renamed class is importable under its new path."""
    assert RecordExportJSONLWriter is not None


def test_record_export_jsonl_writer_dual_dispatch_alias() -> None:
    """``process_result`` aliases ``process_record`` so the writer can be
    dispatched as either a legacy ``results_processor`` or a
    ``stream_exporter``.
    """
    assert (
        RecordExportJSONLWriter.process_result is RecordExportJSONLWriter.process_record
    )
