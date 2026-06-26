# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mode detection for visualization.

This module provides functionality to detect whether the input represents
a single profiling run or multiple runs based on directory structure.
"""

from pathlib import Path

from aiperf.common.enums import CaseInsensitiveStrEnum
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.plot.constants import (
    PROFILE_EXPORT_AIPERF_AGGREGATE_JSON,
    PROFILE_EXPORT_AIPERF_JSON,
    PROFILE_EXPORT_JSONL,
    TRIAL_RUNS_SUBDIR,
)
from aiperf.plot.exceptions import ModeDetectionError


class VisualizationMode(CaseInsensitiveStrEnum):
    """Enumeration of visualization modes."""

    SINGLE_RUN = "single_run"
    MULTI_RUN = "multi_run"


class ModeDetector(AIPerfLoggerMixin):
    """
    Mode detection for visualization with logging support.

    This class provides mode detection functionality to determine whether
    input paths represent a single profiling run or multiple runs based on
    directory structure.
    """

    def __init__(self):
        super().__init__()

    def detect_mode(self, paths: list[Path]) -> tuple[VisualizationMode, list[Path]]:
        """
        Detect visualization mode based on input paths and return run directories.

        This function analyzes the provided paths to determine whether they
        represent a single profiling run or multiple runs by counting the total
        number of run directories found.

        Mode determination:
        - 1 run directory -> SINGLE_RUN
        - 2+ run directories -> MULTI_RUN

        Note: This function searches recursively for run directories, including
        nested runs. Duplicate paths (resolved to the same directory) are
        deduplicated.

        Args:
            paths: List of Path objects to analyze. Can be:
                - Single path to a run directory
                - Single path to a parent directory containing run subdirectories
                - Multiple paths to run directories or parent directories

        Returns:
            Tuple of (VisualizationMode, list of Path objects):
                - VisualizationMode.SINGLE_RUN if exactly 1 run directory is found
                - VisualizationMode.MULTI_RUN if 2 or more run directories are found
                - List of unique run directory paths (sorted)

        Raises:
            ModeDetectionError: If mode cannot be determined or paths are invalid.
        """
        if not paths:
            raise ModeDetectionError("No paths provided")

        run_dirs = self.find_run_directories(paths)

        if len(run_dirs) == 1:
            self.info("Detected SINGLE_RUN mode: 1 run directory found")
            return VisualizationMode.SINGLE_RUN, run_dirs
        else:
            self.info(f"Detected MULTI_RUN mode: {len(run_dirs)} run directories found")
            return VisualizationMode.MULTI_RUN, run_dirs

    def find_run_directories(self, paths: list[Path]) -> list[Path]:
        """
        Find all run directories from input paths.

        This function expands the input paths to a list of run directories:
        - If a path is a run directory, it's included directly
        - If a path is a parent directory, its run subdirectories are discovered recursively
        - Duplicate paths (resolved to the same directory) are deduplicated
        - Nested run directories are all included

        Args:
            paths: List of paths that may be run directories or parent directories.

        Returns:
            List of unique Path objects representing run directories, sorted by path.

        Raises:
            ModeDetectionError: If no valid run directories are found.
        """
        all_run_dirs = []
        seen_resolved = set()

        for path in paths:
            if not path.exists():
                raise ModeDetectionError(f"Path does not exist: {path}")

            if not path.is_dir():
                raise ModeDetectionError(f"Path is not a directory: {path}")

            run_dirs = self._find_all_run_directories_recursive(path)

            if not run_dirs:
                raise ModeDetectionError(
                    f"Path does not contain any valid run directories: {path}"
                )

            for run_dir in run_dirs:
                try:
                    resolved = run_dir.resolve(strict=True)
                except (OSError, RuntimeError) as e:
                    self.warning(
                        f"Cannot resolve run directory {run_dir}, skipping: {e}"
                    )
                    continue

                if resolved not in seen_resolved:
                    all_run_dirs.append(run_dir)
                    seen_resolved.add(resolved)
                else:
                    self.debug(f"Skipping duplicate run directory: {run_dir}")

        if not all_run_dirs:
            raise ModeDetectionError("No valid run directories found")

        # Sort for consistent ordering
        all_run_dirs.sort()

        self.info(f"Found {len(all_run_dirs)} unique run directories")
        return all_run_dirs

    def _is_run_directory(self, path: Path) -> bool:
        """
        Check if a path is a valid run directory.

        A valid run directory matches one of two layouts:

        - **Single-run / per-trial layout**: contains both
          ``profile_export.jsonl`` (per-request events) and
          ``profile_export_aiperf.json`` (per-run aggregate). Emitted at the
          artifact root, at every ``<cell>/`` for trials==1 sweeps, and at
          every ``profile_runs/trial_NNNN/``.
        - **Per-cell confidence-aggregate layout**: contains
          ``profile_export_aiperf_aggregate.json`` (no JSONL — aggregates
          carry no per-request events). Emitted by the sweep orchestrator at
          ``<base>/aggregate/<cell>/`` (REPEATED) or ``<base>/<cell>/aggregate/``
          (INDEPENDENT) for trials>1 sweeps. Redundant shadows that duplicate a
          sibling single-run cell (trials==1) are rejected so each cell is
          enrolled exactly once — see :meth:`_is_redundant_aggregate_shadow`.

        Matches both raw and ``.zst``-compressed marker files. Follows symlinks;
        a broken-symlink marker counts as "file not present".

        Args:
            path: Path to check.

        Returns:
            True if path is a valid run directory of either shape, False otherwise.
        """
        if not path.is_dir():
            return False

        try:
            if self._has_single_run_markers(path):
                return True

            if self._has_valid_aggregate_marker(path):
                return True
        except (PermissionError, OSError) as e:
            self.warning(f"Cannot check file status under {path}: {e}")
            return False

        return False

    def _has_marker(self, path: Path, name: str) -> bool:
        """Return True iff ``name`` (raw or ``.zst``) exists under ``path`` and
        is not a broken symlink."""
        raw = path / name
        zst = path / f"{name}.zst"
        if raw.is_symlink() and not raw.exists():
            self.warning(f"Directory {path} contains broken symlink for {raw}")
            return False
        if raw.exists():
            return True
        if zst.is_symlink() and not zst.exists():
            self.warning(f"Directory {path} contains broken symlink for {zst}")
            return False
        return zst.exists()

    def _has_single_run_markers(self, path: Path) -> bool:
        """Return True when ``path`` has the complete single-run marker pair."""
        return self._has_marker(path, PROFILE_EXPORT_JSONL) and self._has_marker(
            path, PROFILE_EXPORT_AIPERF_JSON
        )

    def _has_valid_aggregate_marker(self, path: Path) -> bool:
        """Return True when ``path`` carries a non-redundant aggregate marker."""
        return self._has_marker(
            path, PROFILE_EXPORT_AIPERF_AGGREGATE_JSON
        ) and not self._is_redundant_aggregate_shadow(path)

    def _is_redundant_aggregate_shadow(self, path: Path) -> bool:
        """Detect aggregate dirs that duplicate a single-run sibling cell.

        For trials==1 sweeps the orchestrator writes BOTH the canonical
        single-run cell and a per-cell confidence-aggregate shadow:
        - INDEPENDENT: ``<base>/<cell>/aggregate/`` shadows ``<base>/<cell>/``.
        - REPEATED: ``<base>/aggregate/<cell>/`` shadows ``<base>/<cell>/``.

        Without this check each cell would be enrolled twice (once per layout).
        Returns True iff the complete canonical single-run marker pair exists at
        either natural shadow location.
        """
        if path.name == "aggregate" and self._has_single_run_markers(path.parent):
            return True
        return path.parent.name == "aggregate" and self._has_single_run_markers(
            path.parent.parent / path.name
        )

    def _find_all_run_directories_recursive(
        self, path: Path, visited: set[Path] | None = None
    ) -> list[Path]:
        """
        Recursively find all run directories within a path, including nested ones.

        This function searches for all run directories, including those nested within
        other run directories. It protects against circular symlinks by tracking
        visited paths.

        Args:
            path: Directory path to search.
            visited: Set of already visited resolved paths (for circular symlink protection).

        Returns:
            List of all run directories found (may be empty).
        """
        if visited is None:
            visited = set()

        run_dirs = []

        if not path.is_dir():
            return run_dirs

        try:
            resolved_path = path.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            self.warning(f"Cannot resolve path {path}: {e}")
            return run_dirs

        if resolved_path in visited:
            self.debug(f"Skipping already visited path: {path}")
            return run_dirs

        visited.add(resolved_path)

        if self._is_run_directory(path):
            run_dirs.append(path)

        # The per-trial ``profile_runs/`` subtree is redundant ONLY when an
        # ``aggregate/`` sibling carries the canonical per-cell view (trials>1
        # REPEATED / INDEPENDENT sweeps). Skipping it there collapses the cells
        # to one aggregate per variation instead of one entry per trial run.
        # When there is no aggregate sibling (adaptive BO, recipe convergence
        # loops, non-sweep multi-trial runs) ``profile_runs/`` is the only place
        # data lives, so it is still traversed. Passing ``.../profile_runs/``
        # explicitly as the top-level path opts back into the per-trial view.
        try:
            has_aggregate_sibling = self._has_valid_aggregate_marker(path / "aggregate")
            for subdir in path.iterdir():
                if not subdir.is_dir():
                    continue
                if subdir.name == TRIAL_RUNS_SUBDIR and has_aggregate_sibling:
                    self.debug(
                        f"Skipping per-trial subtree {subdir} (sibling aggregate/ "
                        "carries the canonical per-cell view)"
                    )
                    continue
                run_dirs.extend(
                    self._find_all_run_directories_recursive(subdir, visited)
                )
        except PermissionError:
            self.warning(f"Permission denied accessing directory: {path}")
        except OSError as e:
            self.warning(f"Cannot read directory {path}: {e}")

        return run_dirs
