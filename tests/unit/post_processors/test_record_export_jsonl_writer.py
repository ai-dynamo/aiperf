# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for ``RecordExportJSONLWriter``."""

from aiperf.post_processors.record_export_jsonl_writer import RecordExportJSONLWriter


def test_record_export_jsonl_writer_class_importable() -> None:
    """The stream-exporter class is importable under its current path."""
    assert RecordExportJSONLWriter is not None


def test_record_export_jsonl_writer_uses_stream_exporter_entrypoint() -> None:
    assert callable(RecordExportJSONLWriter.process_record)
    assert not hasattr(RecordExportJSONLWriter, "process_result")
