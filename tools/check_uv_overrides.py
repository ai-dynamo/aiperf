#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Report `[tool.uv] override-dependencies` entries that upstream no longer needs.

An override replaces every other package's requirement on the overridden
package, so a stale one silently forces a version upstream may have moved
past. For each override this resolves the locked version of the overridden
package and the locked packages that depend on it, then asks PyPI whether
each dependent's latest release already admits that version. When all of
them do, the override can be removed.

Every requirement line naming the package counts, whatever its environment
marker or extra. A single excluding line keeps the override, so the check can
hold an override longer than necessary but never reports one as removable
early.

Exit codes:
    0  every override is still needed (or there are none)
    1  at least one override can be removed
    2  PyPI could not be queried

Usage:
    python tools/check_uv_overrides.py
    python tools/check_uv_overrides.py --pyproject pyproject.toml --lock uv.lock
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPI_URL = "https://pypi.org/pypi/{name}/json"
_TIMEOUT_S = 30

EXIT_NEEDED = 0
EXIT_REMOVABLE = 1
EXIT_PYPI_ERROR = 2


class PyPIError(Exception):
    """A PyPI metadata request failed."""


@dataclass(frozen=True, slots=True)
class LatestRelease:
    """The parts of a project's latest PyPI release this check reads."""

    version: str
    """Version PyPI reports as latest (never a pre-release)."""
    requires_dist: list[str]
    """Raw `Requires-Dist` lines of that release."""


@dataclass(slots=True)
class DependentStatus:
    """Whether one dependent's latest release admits the overridden version."""

    name: str
    """Canonical name of the dependent package."""
    locked_version: str
    """Version of the dependent pinned in uv.lock."""
    latest_version: str
    """Latest version on PyPI."""
    excluding: list[str]
    """Requirement lines in the latest release that reject a locked version."""

    @property
    def admits(self) -> bool:
        return not self.excluding


@dataclass(slots=True)
class OverrideStatus:
    """Result for one override entry."""

    override: str
    """The override as written in pyproject.toml."""
    package: str
    """Canonical name of the overridden package."""
    locked_versions: list[str]
    """Versions of the overridden package in uv.lock."""
    dependents: list[DependentStatus] = field(default_factory=list)
    """One entry per locked package that depends on the overridden package."""

    @property
    def removable(self) -> bool:
        return all(d.admits for d in self.dependents)


def fetch_latest_release(name: str) -> LatestRelease:
    """Return the latest release metadata for `name` from the PyPI JSON API."""
    url = _PYPI_URL.format(name=name)
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:  # noqa: S310
            info = json.load(resp)["info"]
    except (OSError, ValueError, KeyError) as e:
        raise PyPIError(f"{url}: {e}") from e
    return LatestRelease(
        version=info["version"], requires_dist=info["requires_dist"] or []
    )


def _lock_dependency_names(package: dict) -> set[str]:
    deps = list(package.get("dependencies", []))
    for extra_deps in package.get("optional-dependencies", {}).values():
        deps.extend(extra_deps)
    return {canonicalize_name(d["name"]) for d in deps}


def _is_project_root(package: dict) -> bool:
    source = package.get("source", {})
    return "editable" in source or "virtual" in source


def check_overrides(
    pyproject: dict,
    lock: dict,
    fetch: Callable[[str], LatestRelease] = fetch_latest_release,
) -> list[OverrideStatus]:
    """Evaluate every `[tool.uv] override-dependencies` entry against PyPI."""
    overrides = pyproject.get("tool", {}).get("uv", {}).get("override-dependencies", [])
    packages = lock.get("package", [])
    results = []
    for override in overrides:
        target = canonicalize_name(Requirement(override).name)
        status = OverrideStatus(
            override=override,
            package=target,
            locked_versions=sorted(
                p["version"] for p in packages if canonicalize_name(p["name"]) == target
            ),
        )
        for pkg in packages:
            if _is_project_root(pkg) or target not in _lock_dependency_names(pkg):
                continue
            latest = fetch(pkg["name"])
            reqs = [Requirement(line) for line in latest.requires_dist]
            excluding = [
                str(req)
                for req in reqs
                if canonicalize_name(req.name) == target
                and not all(
                    req.specifier.contains(v, prereleases=True)
                    for v in status.locked_versions
                )
            ]
            status.dependents.append(
                DependentStatus(
                    name=canonicalize_name(pkg["name"]),
                    locked_version=pkg["version"],
                    latest_version=latest.version,
                    excluding=excluding,
                )
            )
        results.append(status)
    return results


def _describe(status: OverrideStatus) -> str:
    locked = ", ".join(status.locked_versions) or "not locked"
    lines = [f"override {status.override!r} ({status.package} locked at {locked}):"]
    for d in status.dependents:
        verdict = "admits it" if d.admits else f"excludes it via {d.excluding}"
        lines.append(
            f"  {d.name} locked {d.locked_version}, latest {d.latest_version}: {verdict}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report uv override-dependencies entries upstream no longer needs."
    )
    parser.add_argument("--pyproject", type=Path, default=_REPO_ROOT / "pyproject.toml")
    parser.add_argument("--lock", type=Path, default=_REPO_ROOT / "uv.lock")
    args = parser.parse_args(argv)

    pyproject = tomllib.loads(args.pyproject.read_text(encoding="utf-8"))
    lock = tomllib.loads(args.lock.read_text(encoding="utf-8"))
    try:
        results = check_overrides(pyproject, lock, fetch_latest_release)
    except PyPIError as e:
        print(f"::error::Could not query PyPI, override status unknown: {e}")
        return EXIT_PYPI_ERROR

    if not results:
        print("No [tool.uv] override-dependencies entries.")
        return EXIT_NEEDED

    removable = [r for r in results if r.removable]
    for r in results:
        print(_describe(r))
    for r in removable:
        print(
            f"::error::uv override {r.override!r} is no longer needed: the latest release of "
            f"every package depending on {r.package} admits the locked version. Raise those "
            "packages' minimums to the releases listed above, delete the entry from "
            "[tool.uv] override-dependencies in pyproject.toml, and run `uv lock`."
        )
    return EXIT_REMOVABLE if removable else EXIT_NEEDED


if __name__ == "__main__":
    sys.exit(main())
