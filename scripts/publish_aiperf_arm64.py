#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build, push, and publish the local AIPerf arm64 image to NVCR."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path

REPOSITORY = "nvcr.io/nvidian/dynamo-dev/aiperf"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_FILES = [
    PROJECT_ROOT / "deploy/helm/aiperf-operator/values.yaml",
    PROJECT_ROOT / "dev/deploy/mock-10k-streaming.yaml",
    PROJECT_ROOT / "dev/deploy/mock-50k-streaming.yaml",
    PROJECT_ROOT / "dev/deploy/mock-100-streaming-debug.yaml",
    PROJECT_ROOT / "dev/deploy/mock-250k-benchmark.yaml",
    PROJECT_ROOT / "dev/deploy/mock-250k-streaming.yaml",
]


def build_tag(now: dt.datetime, git_sha: str) -> str:
    """Build the standard arm64 image tag."""
    return f"k8s-arm64-{now.strftime('%Y%m%d-%H%M%S')}-{git_sha[:9]}"


def build_push_command(image: str, dockerfile: str, build_context: Path) -> list[str]:
    """Build the docker buildx push command."""
    return [
        "docker",
        "buildx",
        "build",
        "--platform",
        "linux/arm64",
        "--push",
        "-t",
        image,
        "-f",
        dockerfile,
        str(build_context),
    ]


def rewrite_image_refs(
    file_paths: list[Path], repository: str, new_tag: str
) -> list[Path]:
    """Rewrite matching image references to the new tag."""
    pattern = re.compile(rf'({re.escape(repository)}:)([^"\'\s]+)')
    changed_files: list[Path] = []

    for file_path in file_paths:
        original = file_path.read_text(encoding="utf-8")
        updated, count = pattern.subn(rf"\1{new_tag}", original)
        if count > 0 and updated != original:
            file_path.write_text(updated, encoding="utf-8")
            changed_files.append(file_path)

    return changed_files


def docker_config_has_registry_auth(registry: str) -> bool:
    """Return True when Docker config contains a way to auth to the registry."""
    config_path = Path.home() / ".docker/config.json"
    if not config_path.exists():
        return False

    config = json.loads(config_path.read_text(encoding="utf-8"))
    auths = config.get("auths", {})
    if registry in auths:
        return True

    cred_helpers = config.get("credHelpers", {})
    if registry in cred_helpers:
        return True

    return bool(config.get("credsStore"))


def require_buildx() -> None:
    """Fail if docker buildx is unavailable."""
    result = subprocess.run(
        ["docker", "buildx", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit("docker buildx is required")


def git_short_sha() -> str:
    """Return the current short git SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "--short=9", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build and push the local AIPerf arm64 image to NVCR."
    )
    parser.add_argument("--tag", help="Override the generated image tag.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without building, pushing, or editing files.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Specific file to rewrite. May be passed multiple times.",
    )
    return parser.parse_args(argv)


def resolve_target_files(raw_paths: list[str]) -> list[Path]:
    """Resolve target files to rewrite."""
    target_files = (
        [
            path if path.is_absolute() else PROJECT_ROOT / path
            for path in (Path(raw_path) for raw_path in raw_paths)
        ]
        if raw_paths
        else DEFAULT_IMAGE_FILES
    )
    missing = [str(path) for path in target_files if not path.exists()]
    if missing:
        raise SystemExit(f"Missing image reference files: {', '.join(missing)}")
    return target_files


def run_command(command: list[str]) -> None:
    """Run a command from the project root."""
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    """Run the publish flow."""
    args = parse_args(argv)
    tag = args.tag or build_tag(dt.datetime.now(dt.UTC), git_short_sha())
    image = f"{REPOSITORY}:{tag}"
    target_files = resolve_target_files(args.file)
    command = build_push_command(
        image=image, dockerfile="Dockerfile", build_context=PROJECT_ROOT
    )

    print(f"Publishing image: {image}")
    print("Command:")
    print(" ".join(command))
    print("Files to update:")
    for file_path in target_files:
        print(f"- {file_path.relative_to(PROJECT_ROOT)}")

    if args.dry_run:
        return 0

    require_buildx()

    if not docker_config_has_registry_auth("nvcr.io"):
        raise SystemExit(
            "Docker is not configured for nvcr.io. Run: docker login nvcr.io"
        )

    run_command(command)
    changed_files = rewrite_image_refs(target_files, repository=REPOSITORY, new_tag=tag)

    print("Updated files:")
    for file_path in changed_files:
        print(f"- {file_path.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
