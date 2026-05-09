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
from urllib.parse import urlparse, urlunparse


def normalize_mlflow_uri(uri: str | None) -> str:
    """Normalize an MLflow tracking URI for equality comparison.

    Lowercases only the scheme and host (per RFC 3986 §3.1 and §3.2.2) and
    strips a trailing slash from the path. Path / query / fragment keep their
    original case — on case-sensitive filesystems, ``file:///tmp/MLRuns`` and
    ``file:///tmp/mlruns`` point at different directories and must not
    compare equal.

    Non-URI inputs and bare paths fall back to ``strip().rstrip("/")``.
    """
    if not uri:
        return ""
    stripped = uri.strip()
    parsed = urlparse(stripped)
    scheme = parsed.scheme.lower()
    if not scheme:
        return stripped.rstrip("/")
    if not parsed.netloc:
        # ``file:///path`` and ``sqlite:///path`` have empty netloc but a
        # case-sensitive path. Still normalize the scheme so callers that
        # pass ``FILE:///tmp/mlruns`` vs ``file:///tmp/mlruns`` compare equal.
        _, _, rest = stripped.partition(":")
        return f"{scheme}:{rest}".rstrip("/")
    host = (parsed.hostname or "").lower()
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo = f"{userinfo}:{parsed.password}"
        userinfo = f"{userinfo}@"
    netloc = f"{userinfo}{host}"
    path = parsed.path.rstrip("/")
    return urlunparse(
        (scheme, netloc, path, parsed.params, parsed.query, parsed.fragment)
    )


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
