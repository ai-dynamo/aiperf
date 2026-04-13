# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal version provider and computation utilities.

Not part of the public API. Used by commitizen (registered as a commitizen.provider
entry point) and by scripts/version.py for CI/Makefile version computation.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import tomlkit
from commitizen.providers.base_provider import VersionProvider

_REPO_ROOT = Path(__file__).parent.parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# Matches the PEP 440 forms this project produces:
#   0.8.0
#   0.8.0a20260403          alpha
#   0.8.0b1                 beta
#   0.8.0rc2                release candidate
#   0.8.0.dev5+gabc1234     dev (optionally + .dirty)
_PEP440_RE = re.compile(
    r"^(?P<base>\d+\.\d+\.\d+)"
    r"(?:"
    r"(?:\.dev(?P<devN>\d+)\+(?P<devSHA>[^.+]+)(?:\.(?P<dirty>dirty))?)?"
    r"|(?:a(?P<aN>\d+))?"
    r"|(?:b(?P<bN>\d+))?"
    r"|(?:rc(?P<rcN>\d+))?"
    r")$"
)

_CALVER_RE = re.compile(r"^\d{8}$")


def _run(cmd: list[str]) -> str:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        cwd=_REPO_ROOT,
    )
    return result.stdout.strip()


class AiperfVersionProvider(VersionProvider):
    """Commitizen version provider for AIPerf.

    Implements the commitizen VersionProvider protocol (get_version / set_version)
    and exposes additional methods for dev and nightly version computation used by
    scripts/version.py and CI workflows.
    """

    def get_version(self) -> str:
        doc = tomlkit.parse(_PYPROJECT.read_text())
        return str(doc["project"]["version"])

    def set_version(self, version: str) -> None:
        doc = tomlkit.parse(_PYPROJECT.read_text())

        # Collect version_files before mutating the document.
        version_files: list[str] = (
            doc.get("tool", {}).get("commitizen", {}).get("version_files", [])  # type: ignore[union-attr]
        )

        doc["project"]["version"] = version
        _PYPROJECT.write_text(tomlkit.dumps(doc))

        # Keep version_files in sync — same files commitizen would update on bump.
        for entry in version_files:
            path_str = entry.rsplit(":", 1)[0] if ":" in entry else entry
            file_path = _REPO_ROOT / path_str
            if not file_path.exists():
                continue
            if file_path.suffix == ".toml":
                file_doc = tomlkit.parse(file_path.read_text())
                if "project" in file_doc and "version" in file_doc["project"]:  # type: ignore[operator]
                    file_doc["project"]["version"] = version  # type: ignore[index]
                    file_path.write_text(tomlkit.dumps(file_doc))

    def dev_version(self, *, container: bool = False) -> str:
        """Compute a dev version string from git state.

        PEP 440:    0.7.0.devN+gSHA[.dirty]
        OCI semver: 0.7.0-dev.N.gSHA[.dirty]
        """
        base = self.get_version()
        distance, sha = self._git_describe()
        dirty = self._is_dirty()

        if container:
            ver = f"{base}-dev.{distance}.{sha}"
            if dirty:
                ver += ".dirty"
        else:
            ver = f"{base}.dev{distance}+{sha}"
            if dirty:
                ver += ".dirty"
        return ver

    def nightly_version(
        self, *, container: bool = False, date: str | None = None
    ) -> str:
        """Compute a nightly alpha version string using a calver date.

        PEP 440:    0.6.0a20260403
        OCI semver: 0.6.0-alpha.20260403

        Args:
            container: Emit OCI-safe semver instead of PEP 440.
            date: Override the date as YYYYMMDD. Defaults to today (UTC).
        """
        base = self.get_version()
        if date is not None:
            if not _CALVER_RE.match(date):
                raise ValueError(f"date must be YYYYMMDD (8 digits), got: {date!r}")
            calver = date
        else:
            calver = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        if container:
            return f"{base}-alpha.{calver}"
        return f"{base}a{calver}"

    @staticmethod
    def to_semver(pep440: str) -> str:
        """Convert a PEP 440 version string to an OCI-safe semver tag.

        '+' is replaced with '.' because OCI image tag names do not permit it.
        """
        m = _PEP440_RE.match(pep440.strip())
        if not m:
            return pep440.replace("+", ".")

        base = m.group("base")

        if m.group("devN") is not None:
            suffix = f"-dev.{m.group('devN')}.{m.group('devSHA')}"
            if m.group("dirty"):
                suffix += ".dirty"
            return base + suffix

        if m.group("aN") is not None:
            return f"{base}-alpha.{m.group('aN')}"

        if m.group("bN") is not None:
            return f"{base}-beta.{m.group('bN')}"

        if m.group("rcN") is not None:
            return f"{base}-rc.{m.group('rcN')}"

        return base

    @staticmethod
    def _git_describe() -> tuple[int, str]:
        """Return (commit_distance, short_sha) since the most recent v* tag."""
        try:
            raw = _run(["git", "describe", "--tags", "--long", "--match", "v*"])
            # e.g. v0.6.0-post.1-42-g1b517a0f — split from right to handle hyphens in tag names
            sha_part = raw.rsplit("-", 1)[-1]
            distance = int(raw.rsplit("-", 2)[-2])
            return distance, sha_part
        except subprocess.CalledProcessError:
            distance_str = _run(["git", "rev-list", "--count", "HEAD"])
            sha = "g" + _run(["git", "rev-parse", "--short", "HEAD"])
            return int(distance_str), sha

    @staticmethod
    def _is_dirty() -> bool:
        return bool(_run(["git", "status", "--porcelain"]))
