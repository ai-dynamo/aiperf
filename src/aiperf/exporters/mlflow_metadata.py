# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared schema for mlflow_export.json metadata files.

This TypedDict defines the wire format for the metadata file written by both
the live-streaming fanout process and the post-run MLflowDataExporter. It is
consumed by:

- ``MLflowDataExporter._load_existing_metadata`` (post-run reuse detection)
- ``MLflowDataExporter._resolve_live_streaming_run_id`` (live-run identity)
- ``cli_runner._load_mlflow_metadata`` (plot upload target resolution)
"""

from __future__ import annotations

from typing import TypedDict


class MLflowExportMetadata(TypedDict, total=False):
    """Schema for mlflow_export.json — written atomically by the exporter/fanout."""

    tracking_uri: str
    experiment: str
    run_id: str
    run_name: str | None
    benchmark_id: str | None
    parent_run_id: str | None
    live_streaming: bool
    reused_live_run: bool
    metric_keys: list[str]
    param_keys: list[str]
    tag_keys: list[str]
    uploaded_artifacts: list[str]
    exported_at_ns: int
    stream_started_at_ns: int
