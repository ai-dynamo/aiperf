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

        A valid run directory matches one of:

        - **Single-run / per-trial layout**: contains both
          ``profile_export.jsonl`` (per-request events) and
          ``profile_export_aiperf.json`` (per-run aggregate). This is the
          shape the runner emits at the artifact root, at every
          ``<cell>/`` for trials==1 sweeps, and at every
          ``profile_runs/trial_NNNN/`` (or ``run_NNNN/``).

        - **Per-cell confidence-aggregate layout**: contains
          ``profile_export_aiperf_aggregate.json`` (no JSONL because
          aggregates have no per-request events). This is the shape the
          sweep orchestrator emits at ``<base>/aggregate/<cell>/``
          (REPEATED) or ``<base>/<cell>/aggregate/`` (INDEPENDENT) for
          trials>1 sweeps. ``DataLoader.load_run`` un-flattens the
          confidence shape into the single-run shape so downstream
          plotting code stays uniform.

          Aggregate dirs that REDUNDANTLY shadow a sibling single-run
          cell (the trials==1 case, where the orchestrator writes both
          ``<base>/<cell>/`` AND ``<base>/aggregate/<cell>/`` even though
          the per-cell aggregate is just a single-trial passthrough)
          are rejected here so each cell shows up exactly once. See
          :meth:`_is_redundant_aggregate_shadow` for the canonical-
          sibling probe.

        Note: This function follows symlinks. Broken symlinks for any of
        the canonical files are treated as "file not present."

        Args:
            path: Path to check.

        Returns:
            True if path is a valid run directory of either shape, False
            otherwise.
        """
        if not path.is_dir():
            return False

        try:
            jsonl_file = path / PROFILE_EXPORT_JSONL
            aiperf_json_file = path / PROFILE_EXPORT_AIPERF_JSON
            if self._file_present(path, jsonl_file) and self._file_present(
                path, aiperf_json_file
            ):
                return True

            aggregate_file = path / PROFILE_EXPORT_AIPERF_AGGREGATE_JSON
            if self._file_present(path, aggregate_file):
                return not self._is_redundant_aggregate_shadow(path)
        except (PermissionError, OSError) as e:
            self.warning(f"Cannot check file status under {path}: {e}")
            return False

        return False

    def _is_redundant_aggregate_shadow(self, path: Path) -> bool:
        """Detect aggregate dirs that duplicate a single-run sibling cell.

        For trials==1 sweeps the orchestrator writes BOTH the canonical
        single-run cell and a per-cell confidence-aggregate shadow:

        - REPEATED + trials==1: ``<base>/<cell>/`` (jsonl + json) AND
          ``<base>/aggregate/<cell>/`` (aggregate-only). Canonical is the
          sibling of the ``aggregate`` parent dir at the same name.
        - INDEPENDENT + trials==1: ``<base>/<cell>/`` (jsonl + json) AND
          ``<base>/<cell>/aggregate/`` (aggregate-only). Canonical is the
          parent dir.

        Without this check each cell would be enrolled twice — once via
        the single-run path, once via the aggregate-only path — and
        ``RunMetadata.variation_label`` would diverge between them
        (the aggregate JSON's stamped ``metadata.variation_label`` form
        vs. the path-walk ``concurrency_4`` form), producing duplicate
        ``experiment_groups`` entries for the same cell.

        Returns True iff the canonical single-run JSONL file exists at
        either of the two natural shadow locations.
        """
        # INDEPENDENT shadow: <base>/<cell>/aggregate/  → canonical at <base>/<cell>/
        if path.name == "aggregate":
            canonical = path.parent / PROFILE_EXPORT_JSONL
            if self._file_present(path, canonical):
                return True
        # REPEATED shadow: <base>/aggregate/<cell>/  → canonical at <base>/<cell>/
        if path.parent.name == "aggregate":
            canonical = path.parent.parent / path.name / PROFILE_EXPORT_JSONL
            if self._file_present(path, canonical):
                return True
        return False

    def _file_present(self, parent: Path, candidate: Path) -> bool:
        """Return True iff ``candidate`` exists and is not a broken symlink."""
        if candidate.is_symlink() and not candidate.exists():
            self.warning(f"Directory {parent} contains broken symlink for {candidate}")
            return False
        return candidate.exists()

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

        if not path.is_dir():
            return []

        try:
            resolved_path = path.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            self.warning(f"Cannot resolve path {path}: {e}")
            return []

        if resolved_path in visited:
            self.debug(f"Skipping already visited path: {path}")
            return []

        visited.add(resolved_path)

        run_dirs: list[Path] = []
        if self._is_run_directory(path):
            run_dirs.append(path)
        run_dirs.extend(self._recurse_into_subdirs(path, visited))
        return run_dirs

    def _recurse_into_subdirs(self, path: Path, visited: set[Path]) -> list[Path]:
        """Iterate ``path`` children, applying the conditional ``profile_runs`` skip.

        The per-trial ``profile_runs/`` subtree is only redundant when an
        ``aggregate/`` sibling carries the canonical per-cell view (the
        trials>1 REPEATED / INDEPENDENT cases). Without that condition,
        ``profile_runs/`` is the ONLY place benchmark data lives — adaptive
        BO at ``<base>/search_iter_NNNN/profile_runs/run_NNNN/``, recipes
        that wrap each cell in its own multi-run convergence loop, and
        non-sweep multi-trial runs without an aggregate. Skipping
        unconditionally would surface zero runs in those cases.

        Users who DO want per-trial plots even with an aggregate sibling
        can pass ``<...>/profile_runs/`` explicitly — when that's the
        top-level argument, the parent-sibling check never runs.
        """
        has_aggregate_sibling = (path / "aggregate").is_dir()
        run_dirs: list[Path] = []
        try:
            for subdir in path.iterdir():
                if not subdir.is_dir():
                    continue
                if subdir.name == TRIAL_RUNS_SUBDIR and has_aggregate_sibling:
                    self.debug(
                        f"Skipping per-trial subtree {subdir} during recursion "
                        f"(sibling aggregate/ at {path / 'aggregate'} carries "
                        "the canonical per-cell view); pass it explicitly to "
                        "opt back in to the per-trial view."
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
