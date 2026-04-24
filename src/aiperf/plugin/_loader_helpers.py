# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Free-function helpers for `_PluginRegistryLoaderMixin`.

Split out to keep `_loader.py` under the ergonomics file-size limit.
"""

from __future__ import annotations

from importlib.metadata import Distribution
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.plugin.types import PackageInfo, PluginEntry

_logger = AIPerfLogger(__name__)

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable


def read_registry_file(registry_path: Path | str | Traversable) -> str:
    """Read registry YAML file content."""
    try:
        if hasattr(registry_path, "read_text"):
            # Traversable from importlib.resources
            return registry_path.read_text(encoding="utf-8")

        path = Path(registry_path) if isinstance(registry_path, str) else registry_path

        if not path.exists():
            raise FileNotFoundError(
                f"Registry file not found: {path.absolute()}\n"
                f"Please ensure the plugins.yaml file exists at this location.\n"
                f"Tip: Check your package installation or path configuration"
            )

        if not path.is_file():
            raise ValueError(
                f"Registry path is not a file: {path.absolute()}\n"
                f"Expected a YAML file, got a directory or special file"
            )

        return path.read_text(encoding="utf-8")

    except FileNotFoundError as e:
        raise RuntimeError(
            f"Built-in plugins.yaml not found at {registry_path}.\n"
            "This is a critical error - the package system cannot function without it.\n"
            "Tip: Reinstall the aiperf package or check your installation"
        ) from e
    except OSError as e:
        raise RuntimeError(
            f"Failed to read registry file: {registry_path}\n"
            f"Reason: {e}\n"
            f"Tip: Check file permissions and disk status"
        ) from e


def resolve_conflict(
    existing: PluginEntry,
    new: PluginEntry,
) -> tuple[PluginEntry, str]:
    """Resolve conflict between existing and new type. Returns (winner, reason)."""
    # Rule 1: Higher priority wins
    if new.priority > existing.priority:
        return new, f"priority {new.priority} > {existing.priority}"
    if new.priority < existing.priority:
        return existing, f"priority {existing.priority} > {new.priority}"

    # Rule 2: Equal priority - package beats built-in
    if not new.is_builtin and existing.is_builtin:
        return new, "package overrides built-in (equal priority)"
    if new.is_builtin and not existing.is_builtin:
        return existing, "package overrides built-in (equal priority)"

    # Rule 3: Both same type - first wins (warn)
    _logger.warning(
        f"Plugin conflict for {new.category}:{new.name}: {existing.package} vs {new.package} (priority={new.priority})"
    )

    return existing, "first registered wins (both same type)"


def load_package_metadata(
    package: str, *, dist: Distribution | None = None
) -> PackageInfo:
    """Load package metadata from distribution or installed package.

    If dist is provided, uses it directly. Otherwise falls back to looking up
    the package by name.
    """
    if dist is not None:
        pkg_metadata = dist.metadata
    else:
        try:
            import importlib.metadata

            pkg_metadata = importlib.metadata.metadata(package)
        except importlib.metadata.PackageNotFoundError:
            _logger.warning(f"Failed to load package metadata for {package}")
            return PackageInfo(name=package)

    # Parse author: PEP 621 uses Author-email with "Name <email>" format
    author = pkg_metadata.get("Author", "")
    if not author:
        author_email = pkg_metadata.get("Author-email", "")
        if author_email:
            if "<" in author_email:
                name_part = author_email[: author_email.index("<")].strip()
                if name_part.startswith('"'):
                    name_part = name_part[1:]
                if name_part.endswith('"'):
                    name_part = name_part[:-1]
                author = name_part.strip()
            else:
                author = author_email.split(",")[0].strip()

    return PackageInfo(
        name=package,
        version=pkg_metadata.get("Version", "unknown"),
        description=pkg_metadata.get("Summary", ""),
        author=author,
        license=pkg_metadata.get("License", ""),
        homepage=pkg_metadata.get("Home-page", ""),
    )
