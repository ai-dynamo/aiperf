#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create and verify immutable native runner release inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from hmac import compare_digest
from pathlib import Path
from typing import Any, Literal

from blake3 import blake3

RunnerProfile = Literal["online", "offline"]

_DISTRIBUTION_ID_DOMAIN = b"aiperf-runner-distribution-v1\0"
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_FIELDS = {
    "schema_version",
    "distribution_id",
    "source_revision",
    "cargo_lock_sha256",
    "features",
}
_OFFLINE_PAIRS = {
    ("dynamo_offline", "graph"),
    ("dynamo_offline", "scheduled"),
}
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


def create_manifest(
    *,
    binary: Path,
    cargo_lock: Path,
    source_revision: str,
    features: list[str],
) -> dict[str, Any]:
    """Create a manifest bound to one exact executable and Cargo lock."""
    _validate_native_binary(binary)
    _validate_source_revision(source_revision)
    normalized_features = _validate_features(features)
    capabilities = _load_capabilities(binary)
    distribution_id = _distribution_id(binary)
    if capabilities.get("distribution_id") != distribution_id:
        raise RuntimeError(
            "runner capabilities distribution_id disagrees with its exact image bytes"
        )
    return {
        "schema_version": 1,
        "distribution_id": distribution_id,
        "source_revision": source_revision,
        "cargo_lock_sha256": _cargo_lock_digest(cargo_lock),
        "features": normalized_features,
    }


def verify_release_input(
    *,
    binary: Path,
    manifest_path: Path,
    cargo_lock: Path,
    source_revision: str,
    profile: RunnerProfile,
) -> dict[str, Any]:
    """Verify immutable source, lock, feature, capability, and image identity."""
    _validate_native_binary(binary)
    _validate_source_revision(source_revision)
    manifest = _load_manifest(manifest_path)
    if manifest["source_revision"] != source_revision:
        raise RuntimeError(
            "runner release input source_revision does not match the selected source"
        )
    if manifest["cargo_lock_sha256"] != _cargo_lock_digest(cargo_lock):
        raise RuntimeError(
            "runner release input Cargo.lock digest does not match the selected source"
        )
    distribution_id = _distribution_id(binary)
    if not compare_digest(manifest["distribution_id"], distribution_id):
        raise RuntimeError(
            "runner release input distribution_id does not match the exact binary"
        )
    capabilities = _load_capabilities(binary)
    if capabilities.get("distribution_id") != distribution_id:
        raise RuntimeError(
            "runner capabilities distribution_id disagrees with the immutable manifest"
        )
    _verify_profile(profile, manifest["features"], capabilities)
    return {
        "profile": profile,
        "distribution_id": distribution_id,
        "source_revision": source_revision,
        "cargo_lock_sha256": manifest["cargo_lock_sha256"],
        "features": manifest["features"],
        "supported_pairs": capabilities.get("supported_pairs", []),
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot read runner build manifest {path}: {error}"
        ) from error
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise RuntimeError(
            "runner build manifest requires exactly schema_version, distribution_id, "
            "source_revision, cargo_lock_sha256, and features"
        )
    if manifest["schema_version"] != 1:
        raise RuntimeError("runner build manifest schema_version must be 1")
    _validate_source_revision(manifest["source_revision"])
    distribution_id = manifest["distribution_id"]
    if (
        not isinstance(distribution_id, str)
        or not distribution_id.startswith("blake3:")
        or _HEX_64.fullmatch(distribution_id.removeprefix("blake3:")) is None
    ):
        raise RuntimeError(
            "runner build manifest distribution_id must be blake3: plus 64 lowercase hex digits"
        )
    lock_digest = manifest["cargo_lock_sha256"]
    if (
        not isinstance(lock_digest, str)
        or not lock_digest.startswith("sha256:")
        or _HEX_64.fullmatch(lock_digest.removeprefix("sha256:")) is None
    ):
        raise RuntimeError(
            "runner build manifest cargo_lock_sha256 must be sha256: plus 64 lowercase hex digits"
        )
    manifest["features"] = _validate_features(manifest["features"])
    return manifest


def _validate_native_binary(binary: Path) -> None:
    if not binary.is_file():
        raise RuntimeError(f"runner release input is not a file: {binary}")
    with binary.open("rb", buffering=0) as image:
        prefix = image.read(4)
    if not any(prefix.startswith(magic) for magic in _NATIVE_MAGICS):
        raise RuntimeError(f"runner release input is not a native executable: {binary}")
    if os.name != "nt" and not os.access(binary, os.X_OK):
        raise RuntimeError(f"runner release input is not executable: {binary}")


def _validate_source_revision(value: object) -> None:
    if not isinstance(value, str) or _HEX_40.fullmatch(value) is None:
        raise RuntimeError("source_revision must be 40 lowercase hexadecimal digits")


def _validate_features(features: object) -> list[str]:
    if (
        not isinstance(features, list)
        or not all(isinstance(feature, str) and feature for feature in features)
        or features != sorted(set(features))
    ):
        raise RuntimeError("features must be sorted unique non-empty strings")
    return list(features)


def _distribution_id(binary: Path) -> str:
    digest = blake3()
    digest.update(_DISTRIBUTION_ID_DOMAIN)
    with binary.open("rb", buffering=0) as image:
        while chunk := image.read(1024 * 1024):
            digest.update(chunk)
    return f"blake3:{digest.hexdigest()}"


def _cargo_lock_digest(cargo_lock: Path) -> str:
    try:
        with cargo_lock.open("rb", buffering=0) as lock_file:
            digest = hashlib.file_digest(lock_file, "sha256")
    except OSError as error:
        raise RuntimeError(f"cannot hash Cargo lock {cargo_lock}: {error}") from error
    return f"sha256:{digest.hexdigest()}"


def _load_capabilities(binary: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [os.fspath(binary), "--capabilities"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "runner release input capability probe failed with exit "
            f"{completed.returncode}: {completed.stderr.decode(errors='replace')[-2000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("runner capabilities must contain exactly one JSON line")
    try:
        capabilities = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"runner returned invalid capability JSON: {error}"
        ) from error
    if (
        not isinstance(capabilities, dict)
        or capabilities.get("event") != "runner_capabilities"
    ):
        raise RuntimeError("runner returned an invalid capability response")
    return capabilities


def _verify_profile(
    profile: RunnerProfile,
    features: list[str],
    capabilities: dict[str, Any],
) -> None:
    if profile not in ("online", "offline"):
        raise ValueError(f"unknown runner release profile: {profile!r}")
    pairs = {
        tuple(pair)
        for pair in capabilities.get("supported_pairs", [])
        if isinstance(pair, list)
        and len(pair) == 2
        and all(isinstance(value, str) for value in pair)
    }
    backend_ids = {
        descriptor.get("id")
        for descriptor in capabilities.get("backends", [])
        if isinstance(descriptor, dict)
    }
    if profile == "offline":
        if "dynamo-offline" not in features:
            raise RuntimeError(
                "offline runner manifest must include the dynamo-offline Cargo feature"
            )
        if not _OFFLINE_PAIRS.issubset(pairs) or "dynamo_offline" not in backend_ids:
            raise RuntimeError(
                "offline runner must advertise executable Dynamo scheduled and graph pairs"
            )
        return
    if any(feature.startswith("dynamo-") for feature in features):
        raise RuntimeError(
            "online runner manifest cannot include Dynamo Cargo features"
        )
    if any(backend == "dynamo_offline" for backend, _workload in pairs):
        raise RuntimeError("online runner unexpectedly advertises an offline pair")
    if "dynamo_offline" in backend_ids:
        raise RuntimeError("online runner unexpectedly advertises the offline backend")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create one runner-build.json")
    create.add_argument("--binary", type=Path, required=True)
    create.add_argument("--cargo-lock", type=Path, required=True)
    create.add_argument("--source-revision", required=True)
    create.add_argument("--feature", action="append", default=[])
    create.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="verify one immutable runner input")
    verify.add_argument("--binary", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--cargo-lock", type=Path, required=True)
    verify.add_argument("--source-revision", required=True)
    verify.add_argument("--profile", choices=("online", "offline"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the release-input create or verification operation."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "create":
        manifest = create_manifest(
            binary=arguments.binary,
            cargo_lock=arguments.cargo_lock,
            source_revision=arguments.source_revision,
            features=arguments.feature,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    result = verify_release_input(
        binary=arguments.binary,
        manifest_path=arguments.manifest,
        cargo_lock=arguments.cargo_lock,
        source_revision=arguments.source_revision,
        profile=arguments.profile,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
