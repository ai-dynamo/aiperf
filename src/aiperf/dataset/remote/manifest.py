# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``aiperf.json`` manifest a publisher ships with a trace, and its checks."""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

import orjson
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import ConfigDict, Field, ValidationError

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.models import AIPerfBaseModel
from aiperf.dataset.remote.errors import RemoteErrorKind, RemoteSourceError, RemoteStage

_logger = AIPerfLogger(__name__)

SUPPORTED_MANIFEST_VERSION = 1
ROOT_MANIFEST_NAME = "aiperf.json"
SIDECAR_SUFFIX = ".aiperf.json"
_SCHEMA_HINT = "See docs/tutorials/publishing-traces.md for the aiperf.json schema"


class ManifestRequires(AIPerfBaseModel):
    """Conditions a run must meet to replay the trace."""

    model_config = ConfigDict(extra="allow")

    aiperf: str | None = Field(
        default=None,
        description="PEP 440 version specifier the installed AIPerf must satisfy, e.g. `>=0.14`.",
    )
    endpoint_types: list[str] | None = Field(
        default=None, description="Allowed `--endpoint-type` values."
    )
    max_context_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Peak recorded input+output tokens of any request in the trace.",
    )


class AIPerfManifest(AIPerfBaseModel):
    """Publisher-declared metadata for a remote dataset."""

    model_config = ConfigDict(extra="allow")

    aiperf_manifest_version: int = Field(
        ge=1, strict=True, description="Manifest schema version."
    )
    format: str | None = Field(
        default=None, description="Custom dataset loader name, e.g. `mooncake_trace`."
    )
    path: str | None = Field(
        default=None,
        description="Data file or subdirectory, relative to the repo/prefix root.",
    )
    sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Lowercase hex sha256 of the single data file.",
    )
    block_size: int | None = Field(
        default=None, ge=1, description="Hash block size the trace was recorded with."
    )
    requires: ManifestRequires = Field(
        default_factory=ManifestRequires, description="Compatibility requirements."
    )
    description: str | None = Field(default=None, description="Free-text description.")
    license: str | None = Field(default=None, description="SPDX license identifier.")
    tokenizer: str | None = Field(
        default=None,
        description="Tokenizer the trace was recorded with (informational).",
    )


def _error(
    ref_display: str, kind: RemoteErrorKind, what: str, hint: str
) -> RemoteSourceError:
    return RemoteSourceError(
        ref=ref_display, kind=kind, stage=RemoteStage.ACCESS, what=what, hint=hint
    )


def _invalid(ref_display: str, what: str) -> RemoteSourceError:
    return _error(ref_display, RemoteErrorKind.MANIFEST_INVALID, what, _SCHEMA_HINT)


def _check_contained(path: str, ref_display: str) -> None:
    pure = PurePosixPath(path)
    windows = PureWindowsPath(path)
    if (
        not path.strip()
        or path.strip() == "."
        or "\\" in path
        or pure.is_absolute()
        or windows.drive
        or windows.anchor
        or ".." in pure.parts
    ):
        raise _invalid(
            ref_display,
            f"its manifest `path` '{path}' must be a relative path inside the repo or prefix",
        )


def _check_shape(
    manifest: AIPerfManifest, ref_display: str, kind: Literal["root", "sidecar"]
) -> None:
    if kind == "root":
        if manifest.format is None:
            raise _invalid(ref_display, "its aiperf.json does not declare `format`")
        if manifest.path is None:
            raise _invalid(ref_display, "its aiperf.json does not declare `path`")
    elif manifest.path is not None:
        raise _invalid(ref_display, "a sidecar manifest must not set `path`")
    if manifest.path is not None:
        _check_contained(manifest.path, ref_display)
    for extra in (manifest.model_extra or {}, manifest.requires.model_extra or {}):
        for key in extra:
            _logger.warning(f"{ref_display}: ignoring unknown manifest field '{key}'")


def parse_manifest(
    raw: bytes, *, ref_display: str, kind: Literal["root", "sidecar"]
) -> AIPerfManifest:
    """Parse and validate a manifest.

    Raises:
        RemoteSourceError: ``MANIFEST_INVALID`` for malformed content, ``INCOMPATIBLE``
            for a manifest version newer than this AIPerf supports.
    """
    try:
        data = orjson.loads(raw)
    except orjson.JSONDecodeError as e:
        raise _invalid(
            ref_display, f"its {kind} manifest is not valid JSON ({e})"
        ) from e
    if not isinstance(data, dict):
        raise _invalid(ref_display, f"its {kind} manifest is not a JSON object")
    version = data.get("aiperf_manifest_version")
    if (
        isinstance(version, int)
        and not isinstance(version, bool)
        and version > SUPPORTED_MANIFEST_VERSION
    ):
        raise _error(
            ref_display,
            RemoteErrorKind.INCOMPATIBLE,
            f"its manifest uses aiperf_manifest_version {version}, newer than this "
            f"AIPerf supports ({SUPPORTED_MANIFEST_VERSION})",
            "Upgrade AIPerf",
        )
    try:
        manifest = AIPerfManifest.model_validate(data)
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(part) for part in first["loc"])
        raise _invalid(
            ref_display, f"its manifest field '{loc}' is invalid: {first['msg']}"
        ) from e
    _check_shape(manifest, ref_display, kind)
    return manifest


def check_compatibility(
    manifest: AIPerfManifest,
    *,
    ref_display: str,
    endpoint_type: str,
    max_context_length: int | None,
    aiperf_version: str,
) -> None:
    """Check the manifest's requirements against this run.

    Raises:
        RemoteSourceError: ``INCOMPATIBLE`` when the loader, AIPerf version, or
            endpoint type does not satisfy the manifest. A context-length shortfall
            only logs a warning.
    """
    from aiperf.plugin.enums import CustomDatasetType

    if manifest.format is not None:
        try:
            CustomDatasetType(manifest.format)
        except ValueError:
            raise _error(
                ref_display,
                RemoteErrorKind.INCOMPATIBLE,
                f"it needs dataset format '{manifest.format}', which this AIPerf does not have",
                "Upgrade AIPerf",
            ) from None
    requires = manifest.requires
    if requires.aiperf:
        try:
            spec = SpecifierSet(requires.aiperf)
        except InvalidSpecifier as e:
            raise _invalid(
                ref_display,
                f"requires.aiperf '{requires.aiperf}' is not a valid version specifier",
            ) from e
        try:
            installed = Version(aiperf_version)
        except InvalidVersion:
            _logger.warning(
                f"{ref_display}: cannot check requires.aiperf '{requires.aiperf}' because "
                f"the installed AIPerf version is '{aiperf_version}'"
            )
        else:
            if not spec.contains(installed, prereleases=True):
                raise _error(
                    ref_display,
                    RemoteErrorKind.INCOMPATIBLE,
                    f"it requires AIPerf {requires.aiperf}, but {aiperf_version} is installed",
                    "Upgrade AIPerf",
                )
    if requires.endpoint_types and endpoint_type not in requires.endpoint_types:
        raise _error(
            ref_display,
            RemoteErrorKind.INCOMPATIBLE,
            f"it supports --endpoint-type {', '.join(requires.endpoint_types)} only, not {endpoint_type}",
            "Pass a supported --endpoint-type",
        )
    if (
        requires.max_context_tokens is not None
        and max_context_length is not None
        and requires.max_context_tokens > max_context_length
    ):
        _logger.warning(
            f"{ref_display}: the trace peaks at {requires.max_context_tokens} tokens, above "
            f"--max-context-length {max_context_length}; longer requests or traces will be "
            "filtered or rejected"
        )
