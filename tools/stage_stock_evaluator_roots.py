#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage the exact stock-evaluator source roots for one runner deployment.

The canonical generator remains the only dependency/RECORD/source-closure
authority. This tool receives its already verified inventory, copies each
opened regular file into four isolated roots, and writes the strict registry
consumed by the runner companion build and Python deployment discovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from tools.generate_stock_evaluator_manifest import (
    DeploymentSourceFile,
    DeploymentSourceRoot,
    deployment_source_roots,
)

PROVIDER_ROOTS_SCHEMA = "aiperf-stock-evaluator-roots-v1"
PROVIDER_ROOTS_REGISTRY = "evaluator-roots-v1.json"
PROVIDER_ROOT_SPECS = (
    ("cpython_3_12_10_linux_x86_64", "python_runtime", "runtime"),
    ("nvidia_nemo_evaluator_0_4_locked", "python_environment", "nemo"),
    (
        "groq_openbench_0_5_3_inspect_0_3_141_locked",
        "python_environment",
        "openbench",
    ),
    ("system_linux_x86_64", "system", "system"),
)


def stage_stock_evaluator_roots(
    output: Path,
    *,
    nemo_root: Path,
    openbench_root: Path,
) -> None:
    """Atomically stage one complete verified four-root deployment payload."""
    roots = deployment_source_roots(
        nemo_root=nemo_root,
        openbench_root=openbench_root,
    )
    _stage_verified_roots(Path(output), roots)


def _stage_verified_roots(
    output: Path,
    roots: Sequence[DeploymentSourceRoot],
) -> None:
    _validate_root_inventory(roots)
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"evaluator root output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.staging-",
        dir=output.parent,
    ) as temporary:
        payload = Path(temporary) / "payload"
        payload.mkdir()
        registry_roots = []
        for root in roots:
            staged_root = payload / root.relative_path
            staged_root.mkdir()
            content_digests: dict[str, str] = {}
            for item in root.files:
                relative = _strict_relative(item.relative_path)
                logical = relative.as_posix()
                if logical in content_digests:
                    raise ValueError(
                        f"duplicate staged source path {root.id!r}/{logical!r}"
                    )
                target = staged_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                _copy_verified_source(item, target)
                content_digests[logical] = item.artifact_content_sha256
            registry_roots.append(
                {
                    "file_count": len(content_digests),
                    "id": root.id,
                    "kind": root.kind,
                    "path": root.relative_path,
                    "tree_sha256": _tree_sha256(content_digests),
                }
            )
        registry = {
            "platform": "linux-x86_64",
            "roots": registry_roots,
            "schema_version": PROVIDER_ROOTS_SCHEMA,
        }
        _write_regular_file(
            payload / PROVIDER_ROOTS_REGISTRY,
            _canonical_registry(registry),
            executable=False,
        )
        payload.rename(output)


def _validate_root_inventory(roots: Sequence[DeploymentSourceRoot]) -> None:
    observed = tuple((root.id, root.kind, root.relative_path) for root in roots)
    if observed != PROVIDER_ROOT_SPECS:
        raise ValueError("verified evaluator deployment root set drifted")
    if any(not root.files for root in roots):
        raise ValueError("verified evaluator deployment root cannot be empty")


def _copy_verified_source(item: DeploymentSourceFile, target: Path) -> None:
    source = Path(item.source)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise RuntimeError(f"cannot open verified evaluator source {source}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"evaluator source is not a regular file: {source}")
        executable = bool(metadata.st_mode & 0o111)
        if executable != item.executable:
            raise RuntimeError(f"evaluator source executable mode drifted: {source}")
        digest = hashlib.sha256()
        with (
            os.fdopen(descriptor, "rb", closefd=False) as stream,
            target.open("xb") as output,
        ):
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
            os.fchmod(output.fileno(), 0o755 if executable else 0o644)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)
    actual = digest.hexdigest()
    if actual != item.artifact_content_sha256:
        raise RuntimeError(f"evaluator source digest drifted: {source}")


def _write_regular_file(target: Path, content: bytes, *, executable: bool) -> None:
    with target.open("xb") as output:
        output.write(content)
        os.fchmod(output.fileno(), 0o755 if executable else 0o644)
        output.flush()
        os.fsync(output.fileno())


def _strict_relative(value: str) -> PurePosixPath:
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or "\0" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"invalid evaluator source relative path {value!r}")
    return PurePosixPath(value)


def _tree_sha256(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, content_sha256 in sorted(files.items()):
        encoded = relative.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(content_sha256))
    return f"sha256:{digest.hexdigest()}"


def _canonical_registry(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage exact stock evaluator roots for an aiperf-runner deployment"
    )
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--openbench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    stage_stock_evaluator_roots(
        args.output,
        nemo_root=args.nemo_root,
        openbench_root=args.openbench_root,
    )


if __name__ == "__main__":
    main()
