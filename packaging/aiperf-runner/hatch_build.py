# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build hook for the platform-specific native runner companion wheel."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
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
_PROVIDER_ROOTS_SCHEMA = "aiperf-stock-evaluator-roots-v1"
_PROVIDER_ROOTS_REGISTRY = "evaluator-roots-v1.json"
_PROVIDER_ROOTS_SOURCE = "evaluator-roots"
_PROVIDER_ROOTS_WHEEL_TARGET = "_aiperf_runner/evaluator-roots"
_PROVIDER_ROOT_SPECS = (
    ("cpython_3_12_10_linux_x86_64", "python_runtime", "runtime"),
    ("nvidia_nemo_evaluator_0_4_locked", "python_environment", "nemo"),
    (
        "groq_openbench_0_5_3_inspect_0_3_141_locked",
        "python_environment",
        "openbench",
    ),
    ("system_linux_x86_64", "system", "system"),
)
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
        provider_roots = Path(self.root) / _PROVIDER_ROOTS_SOURCE
        if _supports_stock_evaluators(platform_tag):
            _validate_provider_roots(provider_roots)
            build_data.setdefault("force_include", {})[str(provider_roots)] = (
                _PROVIDER_ROOTS_WHEEL_TARGET
            )
        elif provider_roots.exists():
            raise RuntimeError(
                "stock evaluator roots are supported only by Linux x86_64 "
                "runner companions"
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


def _supports_stock_evaluators(platform_tag: str) -> bool:
    normalized = platform_tag.lower()
    return (
        re.fullmatch(
            r"(?:linux|manylinux(?:1|2010|2014)|manylinux_[0-9]+_[0-9]+|"
            r"musllinux_[0-9]+_[0-9]+)_x86_64",
            normalized,
        )
        is not None
    )


def _validate_provider_roots(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(
            "Linux x86_64 runner companion build requires staged evaluator roots"
        )
    registry_path = root / _PROVIDER_ROOTS_REGISTRY
    try:
        registry_bytes = registry_path.read_bytes()
        registry = json.loads(registry_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("stock evaluator root registry is unreadable") from error
    if _canonical_registry(registry) != registry_bytes:
        raise RuntimeError("stock evaluator root registry is not canonical JSON")
    roots = _validate_registry_value(registry)
    physical = _physical_files(root)
    for entry in roots:
        prefix = f"{entry['path']}/"
        members = {
            relative.removeprefix(prefix): _file_sha256(path)
            for relative, path in physical.items()
            if relative.startswith(prefix)
        }
        if len(members) != entry["file_count"]:
            raise RuntimeError(
                f"stock evaluator root {entry['id']!r} file count drifted"
            )
        if _tree_sha256(members) != entry["tree_sha256"]:
            raise RuntimeError(
                f"stock evaluator root {entry['id']!r} content tree drifted"
            )


def _canonical_registry(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _validate_registry_value(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {
        "platform",
        "roots",
        "schema_version",
    }:
        raise RuntimeError("stock evaluator root registry has unknown fields")
    if value["schema_version"] != _PROVIDER_ROOTS_SCHEMA:
        raise RuntimeError("stock evaluator root registry schema drifted")
    if value["platform"] != "linux-x86_64":
        raise RuntimeError("stock evaluator root registry platform drifted")
    roots = value["roots"]
    if not isinstance(roots, list) or len(roots) != len(_PROVIDER_ROOT_SPECS):
        raise RuntimeError("stock evaluator root registry is incomplete")
    result: list[dict[str, Any]] = []
    for entry, (expected_id, expected_kind, expected_path) in zip(
        roots, _PROVIDER_ROOT_SPECS, strict=True
    ):
        if not isinstance(entry, dict) or set(entry) != {
            "file_count",
            "id",
            "kind",
            "path",
            "tree_sha256",
        }:
            raise RuntimeError("stock evaluator root entry has unknown fields")
        if (
            entry["id"] != expected_id
            or entry["kind"] != expected_kind
            or entry["path"] != expected_path
            or not isinstance(entry["file_count"], int)
            or isinstance(entry["file_count"], bool)
            or entry["file_count"] <= 0
            or not _is_sha256(entry["tree_sha256"])
        ):
            raise RuntimeError("stock evaluator root entry drifted")
        result.append(entry)
    return result


def _physical_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            if path.is_symlink():
                raise RuntimeError(
                    "stock evaluator payload contains a symlink directory"
                )
            continue
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise RuntimeError("stock evaluator payload contains a special file")
        relative = path.relative_to(root).as_posix()
        if relative in files:
            raise RuntimeError("stock evaluator payload contains a duplicate path")
        files[relative] = path
    expected_top_level = {
        _PROVIDER_ROOTS_REGISTRY,
        *(path for _, _, path in _PROVIDER_ROOT_SPECS),
    }
    if {relative.split("/", 1)[0] for relative in files} != expected_top_level:
        raise RuntimeError("stock evaluator payload root set drifted")
    return files


def _tree_sha256(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, content_sha256 in sorted(files.items()):
        encoded = relative.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(content_sha256))
    return f"sha256:{digest.hexdigest()}"


def _file_sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("stock evaluator payload contains a special file")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )
