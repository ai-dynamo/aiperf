# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build hook for the platform-specific native runner companion wheel."""

from __future__ import annotations

import os
import re
from hmac import compare_digest
from pathlib import Path
from typing import Any

import orjson
from blake3 import blake3
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging.tags import platform_tags

_BINARY_NAMES = ("aiperf-runner", "aiperf-runner.exe")
_PLATFORM_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.]*$")
_LOWER_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_LOWER_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_DEPENDENCY_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_DISTRIBUTION_ID_DOMAIN = b"aiperf-runner-distribution-v1\0"
_NATIVE_MAGICS = (
    b"\x7fELF",
    b"MZ",
    b"\xfe\xed\xfa\xce",
    b"\xfe\xed\xfa\xcf",
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
)


class CustomBuildHook(BuildHookInterface):
    """Place the prebuilt native image directly in the wheel scripts scheme."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Validate the binary and assign an interpreter-independent platform tag."""
        del version
        input_directory = Path(self.root) / "bin"
        binary = _select_native_binary(input_directory)
        manifest = input_directory / "runner-build.json"
        _validate_build_manifest(manifest, binary)
        platform_tag = os.environ.get("AIPERF_RUNNER_WHEEL_PLATFORM_TAG")
        if platform_tag is None:
            platform_tag = next(platform_tags())
        if _PLATFORM_TAG.fullmatch(platform_tag) is None:
            raise ValueError(
                "AIPERF_RUNNER_WHEEL_PLATFORM_TAG must contain only letters, "
                "digits, underscores, and periods"
            )

        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{platform_tag}"
        build_data["shared_scripts"][str(binary)] = binary.name
        build_data["extra_metadata"][str(manifest)] = "runner-build.json"


def _select_native_binary(directory: Path) -> Path:
    candidates = [
        directory / name for name in _BINARY_NAMES if (directory / name).is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "runner companion build requires exactly one prebuilt "
            f"bin/aiperf-runner[.exe]; found {len(candidates)}"
        )
    binary = candidates[0]
    with binary.open("rb", buffering=0) as image:
        prefix = image.read(4)
    if not any(prefix.startswith(magic) for magic in _NATIVE_MAGICS):
        raise RuntimeError(
            f"runner companion input {binary} is not a native ELF, Mach-O, or PE image; "
            "Python and shell launchers are forbidden"
        )
    if os.name != "nt" and not os.access(binary, os.X_OK):
        raise RuntimeError(f"runner companion input {binary} is not executable")
    return binary


def _validate_build_manifest(manifest_path: Path, binary: Path) -> None:
    if not manifest_path.is_file():
        raise RuntimeError(
            "runner companion build requires bin/runner-build.json from the "
            "trusted native build job"
        )
    try:
        manifest = orjson.loads(manifest_path.read_bytes())
    except orjson.JSONDecodeError as error:
        raise RuntimeError(f"invalid runner build manifest: {error}") from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "distribution_id",
        "source_revision",
        "cargo_lock_sha256",
        "features",
        "dependency_revisions",
    }:
        raise RuntimeError(
            "runner build manifest requires exactly schema_version, distribution_id, "
            "source_revision, cargo_lock_sha256, features, and dependency_revisions"
        )
    if manifest["schema_version"] != 2:
        raise RuntimeError("runner build manifest schema_version must be 2")
    if (
        not isinstance(manifest["source_revision"], str)
        or _LOWER_HEX_40.fullmatch(manifest["source_revision"]) is None
    ):
        raise RuntimeError(
            "runner build manifest source_revision must be 40 lowercase hex digits"
        )
    lock_digest = manifest["cargo_lock_sha256"]
    if (
        not isinstance(lock_digest, str)
        or not lock_digest.startswith("sha256:")
        or _LOWER_HEX_64.fullmatch(lock_digest.removeprefix("sha256:")) is None
    ):
        raise RuntimeError(
            "runner build manifest cargo_lock_sha256 must be sha256: plus 64 lowercase hex digits"
        )
    features = manifest["features"]
    if (
        not isinstance(features, list)
        or not all(isinstance(feature, str) and feature for feature in features)
        or features != sorted(set(features))
    ):
        raise RuntimeError(
            "runner build manifest features must be sorted unique non-empty strings"
        )
    dependency_revisions = manifest["dependency_revisions"]
    if not isinstance(dependency_revisions, dict) or list(
        dependency_revisions
    ) != sorted(dependency_revisions):
        raise RuntimeError(
            "runner build manifest dependency_revisions must be a key-sorted object"
        )
    for dependency, revision in dependency_revisions.items():
        if (
            not isinstance(dependency, str)
            or _DEPENDENCY_NAME.fullmatch(dependency) is None
            or not isinstance(revision, str)
            or _LOWER_HEX_40.fullmatch(revision) is None
        ):
            raise RuntimeError(
                "runner build manifest dependency revisions require lowercase names "
                "and 40-digit lowercase hexadecimal revisions"
            )
    expected = manifest["distribution_id"]
    if not isinstance(expected, str) or not expected.startswith("blake3:"):
        raise RuntimeError(
            "runner build manifest distribution_id must use the blake3: prefix"
        )
    digest = blake3()
    digest.update(_DISTRIBUTION_ID_DOMAIN)
    with binary.open("rb", buffering=0) as image:
        while chunk := image.read(1024 * 1024):
            digest.update(chunk)
    actual = f"blake3:{digest.hexdigest()}"
    if not compare_digest(expected, actual):
        raise RuntimeError(
            "runner build manifest distribution_id does not match the exact staged binary"
        )
