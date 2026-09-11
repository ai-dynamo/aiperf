#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write a PEP 440 version into pyproject.toml and the root entry of uv.lock.

The nightly pipeline stamps a date-based version onto a checkout before
building. uv.lock records the root project's version too, so patching only
pyproject.toml leaves the lockfile stale: `uv sync --locked` rejects it and a
plain `uv run` rewrites it. Patching both keeps the tree lock-consistent.

Both rewrites are textual so each file keeps its formatting and line endings
apart from the one version string. The root lock entry is identified by its
`source = { editable = "." }` marker. Both patched documents are re-parsed
with tomllib in memory to confirm the intended entries carry the new version
before either file is written, so a failure never leaves a half-patched tree.

Requires Python 3.11+ (tomllib).

Usage:
    python3 tools/patch_version.py 0.13.0.dev20260910 [--pyproject P] [--lock L]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
from pathlib import Path

# PEP 440: X.Y.Z[(a|b|rc)N][.postN][.devN][+local]; local segments are [a-zA-Z0-9] joined by dots.
_PEP440 = re.compile(
    r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?"
    r"(\+[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)?"
)
# The [project] table body: from its header to the next table header or EOF.
_PROJECT_TABLE = re.compile(
    r"^\[project\][^\S\r\n]*\r?\n(.*?)(?=^\[|\Z)", re.MULTILINE | re.DOTALL
)
_VERSION_LINE = re.compile(r'^(version\s*=\s*")[^"]*(")', re.MULTILINE)


class PatchError(Exception):
    """A version patch could not be applied or verified; nothing was written."""


def _root_package_pattern(name: str) -> re.Pattern[str]:
    """Match the uv.lock `[[package]]` block for the editable root project `name`."""
    return re.compile(
        r'(\[\[package\]\]\r?\nname = "'
        + re.escape(name)
        + r'"\r?\nversion = ")[^"]*("\r?\nsource = \{ editable = "\." \})'
    )


def _read(path: Path) -> str:
    """Read text verbatim; newline="" keeps CRLF so the rewrite preserves it."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _write(path: Path, text: str) -> None:
    """Write text verbatim via a same-directory temp file and atomic replace."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(tmp, path)


def _patch_pyproject(text: str, version: str) -> tuple[str, str]:
    """Return the patched pyproject text and the project name it declares."""
    try:
        project = tomllib.loads(text)["project"]
    except tomllib.TOMLDecodeError as exc:
        raise PatchError(f"pyproject.toml is not valid TOML: {exc}") from exc
    except KeyError:
        raise PatchError("pyproject.toml has no [project] table") from None
    name = project.get("name")
    if not isinstance(name, str) or not name:
        raise PatchError("[project] has no name")
    table = _PROJECT_TABLE.search(text)
    if table is None:
        raise PatchError("Could not locate the [project] table in pyproject.toml")
    body, n = _VERSION_LINE.subn(rf"\g<1>{version}\g<2>", table.group(1), count=1)
    if n != 1:
        raise PatchError('[project] has no version = "..." line')
    return text[: table.start(1)] + body + text[table.end(1) :], name


def _patch_lock(text: str, name: str, version: str) -> str:
    """Return the lock text with the editable root entry for `name` at `version`."""
    patched, n = _root_package_pattern(name).subn(
        rf"\g<1>{version}\g<2>", text, count=1
    )
    if n != 1:
        raise PatchError(
            f"Could not find exactly one editable root package entry for {name!r} in uv.lock"
        )
    return patched


def _verify(pyproject_text: str, lock_text: str, name: str, version: str) -> None:
    """Re-parse both patched documents and confirm the intended entries carry `version`."""
    try:
        project_version = tomllib.loads(pyproject_text)["project"].get("version")
        packages = tomllib.loads(lock_text).get("package", [])
    except tomllib.TOMLDecodeError as exc:
        raise PatchError(f"patched document is not valid TOML: {exc}") from exc
    if project_version != version:
        raise PatchError(f"[project] version is not {version!r} after patching")
    roots = [
        p
        for p in packages
        if p.get("name") == name and p.get("source", {}).get("editable") == "."
    ]
    if len(roots) != 1 or roots[0].get("version") != version:
        raise PatchError(f"uv.lock root package {name!r} does not carry {version!r}")


def patch_version(version: str, pyproject: Path, lock: Path) -> str:
    """Write `version` into `pyproject` and the root package entry of `lock`.

    Both documents are patched and verified in memory first; files are written
    only when both succeed. Returns the project name. Raises PatchError on an
    invalid version, a missing file, or an entry that cannot be located.
    """
    if not _PEP440.fullmatch(version):
        raise PatchError(f"Invalid PEP 440 version {version!r}")
    if not pyproject.is_file():
        raise PatchError(f"{pyproject} not found")
    if not lock.is_file():
        raise PatchError(
            f"{lock} not found; run `uv lock` to create it, and check that the "
            "repository is checked out before this step"
        )

    pyproject_new, name = _patch_pyproject(_read(pyproject), version)
    lock_new = _patch_lock(_read(lock), name, version)
    _verify(pyproject_new, lock_new, name, version)
    _write(pyproject, pyproject_new)
    _write(lock, lock_new)
    return name


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; failures print a GitHub Actions `::error::` annotation."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("version", help="PEP 440 version to write")
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    args = parser.parse_args(argv)
    try:
        name = patch_version(args.version, args.pyproject, args.lock)
    except (PatchError, OSError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    print(f'Patched {args.pyproject}: version = "{args.version}"')
    print(f'Patched {args.lock}: {name} version = "{args.version}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
