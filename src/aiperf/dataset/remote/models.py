# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data passed between resolvers, the service, and the preflight."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, Field

from aiperf.common.models import AIPerfBaseModel
from aiperf.config.dataset.remote_ref import RemoteRef, coerce_remote_ref
from aiperf.dataset.remote.manifest import AIPerfManifest

UNVERSIONED = "unversioned"
"""Version of an HTTP object with no ETag/Last-Modified; never served from cache across runs."""

_RemoteRefField = Annotated[RemoteRef, BeforeValidator(coerce_remote_ref)]


class RemoteFile(AIPerfBaseModel):
    """One downloaded file and its digest."""

    path: str = Field(
        description="Path relative to the repo/bucket root, or the URL file name."
    )
    sha256: str = Field(description="Lowercase hex sha256 of the downloaded bytes.")
    size_bytes: int = Field(ge=0, description="Downloaded size in bytes.")


class ProbeResult(AIPerfBaseModel):
    """Metadata-only view of a remote source, pinned to one exact version."""

    ref: _RemoteRefField = Field(description="The reference with resolved_version set.")
    pinned_ref: str = Field(
        description="Redacted reference to this version. It reproduces the version "
        "exactly only for an HF commit or an S3 VersionId; an HTTP URL or an S3 "
        "reference pinned by ETag or listing reads the current version when reused."
    )
    data_relpath: str | None = Field(
        description="Data file or directory relative to the repo/bucket root; None for http(s)."
    )
    data_is_dir: bool = Field(description="True when the data is a directory of files.")
    filename: str = Field(description="Base name of the data file or directory.")
    manifest: AIPerfManifest | None = Field(description="Parsed manifest, if any.")
    manifest_sha256: str | None = Field(description="sha256 of the raw manifest bytes.")
    needs_auth: bool = Field(
        description="True when reading this source requires credentials."
    )
    data_files: list[str] | None = Field(
        default=None,
        description="Repo or bucket paths of the parquet shards to read, in order, "
        "when the data is a folder whose data files are all parquet; None for a "
        "single file or a non-parquet folder.",
    )


class ResolvedSource(AIPerfBaseModel):
    """A probed source whose bytes are present in a local cache."""

    probe: ProbeResult = Field(description="The probe this download satisfied.")
    local_path: Path = Field(description="Local file or directory holding the data.")
    files: list[RemoteFile] = Field(
        description="Every downloaded data file with its digest."
    )
