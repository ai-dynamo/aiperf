#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write a PEP 440 version into pyproject.toml and the root entry of uv.lock.

The nightly pipeline stamps a date-based version onto a checkout before
building. uv.lock records the root project's version too, so patching only
pyproject.toml leaves the lockfile stale: `uv sync --locked` rejects it and a
plain `uv run` rewrites it. Patching both keeps the tree lock-consistent.

The rewrite is textual so the lockfile keeps uv's formatting byte-for-byte
apart from the one version string. The root entry is identified by its
`source = { editable = "." }` marker, and the result is re-parsed with tomllib
(Python 3.11+) when available to confirm exactly that entry changed.

Usage:
    python3 tools/patch_version.py 0.13.0.dev20260910 [--pyproject P] [--lock L]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 on the runner; textual checks still apply.
    tomllib = None

# PEP 440: X.Y.Z[(a|b|rc)N][.postN][.devN][+local]; local segments are [a-zA-Z0-9] joined by dots.
_PEP440 = re.compile(
    r"^\d+\.\d+\.\d+((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?"
    r"(\+[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)?$"
)
_PROJECT_VERSION = re.compile(r'^(version\s*=\s*")[^"]*(")', re.MULTILINE)
_PROJECT_NAME = re.compile(r'^name\s*=\s*"([^"]+)"', re.MULTILINE)


class PatchError(Exception):
    """A version patch could not be applied or verified."""


def _root_package_pattern(name: str) -> re.Pattern[str]:
    """Match the uv.lock `[[package]]` block for the editable root project `name`."""
    return re.compile(
        r'(\[\[package\]\]\r?\nname = "'
        + re.escape(name)
        + r'"\r?\nversion = ")[^"]*("\r?\nsource = \{ editable = "\." \})'
    )


def _verify_toml(pyproject: Path, lock: Path, name: str, version: str) -> None:
    """Re-parse both files and confirm exactly the intended entries carry `version`."""
    if tomllib is None:
        return
    project = tomllib.loads(pyproject.read_text())["project"]
    if project.get("version") != version:
        raise PatchError(
            f"{pyproject}: project.version is not {version!r} after patching"
        )
    packages = tomllib.loads(lock.read_text()).get("package", [])
    roots = [
        p
        for p in packages
        if p.get("name") == name and p.get("source", {}).get("editable") == "."
    ]
    if len(roots) != 1 or roots[0].get("version") != version:
        raise PatchError(
            f"{lock}: root package {name!r} does not carry version {version!r}"
        )


def patch_version(version: str, pyproject: Path, lock: Path) -> str:
    """Write `version` into `pyproject` and the root package entry of `lock`.

    Returns the project name read from `pyproject`. Raises PatchError on an
    invalid version, a missing file, or an entry that cannot be located.
    """
    if not _PEP440.match(version):
        raise PatchError(f"Invalid PEP 440 version {version!r}")
    if not pyproject.is_file():
        raise PatchError(f"{pyproject} not found")
    if not lock.is_file():
        raise PatchError(
            f"{lock} not found; it is required to keep `uv sync --locked` satisfied"
        )

    text = pyproject.read_text()
    name_match = _PROJECT_NAME.search(text)
    if name_match is None:
        raise PatchError(f"Could not find 'name = \"...\"' in {pyproject}")
    name = name_match.group(1)
    patched, n = _PROJECT_VERSION.subn(rf"\g<1>{version}\g<2>", text, count=1)
    if n != 1:
        raise PatchError(f"Could not find 'version = \"...\"' in {pyproject}")

    lock_text = lock.read_text()
    lock_patched, n = _root_package_pattern(name).subn(
        rf"\g<1>{version}\g<2>", lock_text, count=1
    )
    if n != 1:
        raise PatchError(
            f"Could not find the editable root package entry for {name!r} in {lock}"
        )

    pyproject.write_text(patched)
    lock.write_text(lock_patched)
    _verify_toml(pyproject, lock, name, version)
    return name


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; prints GitHub Actions `::error::` annotations on failure."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("version", help="PEP 440 version to write")
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    args = parser.parse_args(argv)
    try:
        name = patch_version(args.version, args.pyproject, args.lock)
    except PatchError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    print(f'Patched {args.pyproject}: version = "{args.version}"')
    print(f'Patched {args.lock}: {name} version = "{args.version}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
