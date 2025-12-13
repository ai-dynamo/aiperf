# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for mode detection functionality.
"""

from pathlib import Path

import orjson
import pytest

from aiperf.plot.core.mode_detector import (
    ModeDetector,
    VisualizationMode,
)
from aiperf.plot.exceptions import ModeDetectionError


class TestModeDetection:
    """Tests for detect_mode method."""

    def test_single_run_detection(
        self, mode_detector: ModeDetector, single_run_dir: Path
    ) -> None:
        """Test detection of single run mode."""
        mode, run_dirs = mode_detector.detect_mode([single_run_dir])
        assert mode == VisualizationMode.SINGLE_RUN
        assert len(run_dirs) == 1
        assert run_dirs[0] == single_run_dir

    def test_multiple_runs_explicit_paths(
        self, mode_detector: ModeDetector, multiple_run_dirs: list[Path]
    ) -> None:
        """Test detection of multi-run mode with explicit paths."""
        mode, run_dirs = mode_detector.detect_mode(multiple_run_dirs)
        assert mode == VisualizationMode.MULTI_RUN
        assert len(run_dirs) == 3
        assert set(run_dirs) >= set(multiple_run_dirs)

    def test_multiple_runs_parent_directory(
        self, mode_detector: ModeDetector, parent_dir_with_runs: Path
    ) -> None:
        """Test detection of multi-run mode from parent directory."""
        mode, run_dirs = mode_detector.detect_mode([parent_dir_with_runs])
        assert mode == VisualizationMode.MULTI_RUN
        assert len(run_dirs) == 3

    def test_empty_paths_raises_error(self, mode_detector: ModeDetector) -> None:
        """Test that empty path list raises error."""
        with pytest.raises(ModeDetectionError, match="No paths provided"):
            mode_detector.detect_mode([])

    def test_nonexistent_path_raises_error(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test that nonexistent path raises error."""
        fake_path = tmp_path / "does_not_exist"
        with pytest.raises(ModeDetectionError, match="Path does not exist"):
            mode_detector.detect_mode([fake_path])

    def test_non_directory_path_raises_error(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test that file path raises error."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")

        with pytest.raises(ModeDetectionError, match="Path is not a directory"):
            mode_detector.detect_mode([file_path])

    def test_invalid_run_directory_raises_error(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test that directory without required files raises error."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(
            ModeDetectionError,
            match="does not contain any valid run directories",
        ):
            mode_detector.detect_mode([empty_dir])

    def test_multiple_invalid_paths_raises_error(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test that multiple paths with invalid run raises error."""
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = tmp_path / "dir2"
        dir2.mkdir()

        with pytest.raises(
            ModeDetectionError, match="does not contain any valid run directories"
        ):
            mode_detector.detect_mode([dir1, dir2])


class TestFindRunDirectories:
    """Tests for find_run_directories method."""

    def test_find_single_run(
        self, mode_detector: ModeDetector, single_run_dir: Path
    ) -> None:
        """Test finding single run directory."""
        runs = mode_detector.find_run_directories([single_run_dir])
        assert len(runs) == 1
        assert runs[0] == single_run_dir

    def test_find_multiple_runs_explicit(
        self, mode_detector: ModeDetector, multiple_run_dirs: list[Path]
    ) -> None:
        """Test finding multiple run directories from explicit paths."""
        runs = mode_detector.find_run_directories(multiple_run_dirs)
        assert len(runs) == 3
        assert set(runs) >= set(multiple_run_dirs)

    def test_find_runs_from_parent(
        self, mode_detector: ModeDetector, parent_dir_with_runs: Path
    ) -> None:
        """Test finding run directories from parent directory."""
        runs = mode_detector.find_run_directories([parent_dir_with_runs])
        assert len(runs) == 3
        # Verify all runs are under the parent directory
        for run in runs:
            assert run.is_relative_to(parent_dir_with_runs)

    def test_find_runs_sorted(
        self, mode_detector: ModeDetector, parent_dir_with_runs: Path
    ) -> None:
        """Test that found runs are sorted."""
        runs = mode_detector.find_run_directories([parent_dir_with_runs])
        run_names = [r.name for r in runs]
        assert run_names == sorted(run_names)

    def test_nonexistent_path_raises_error(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test that nonexistent path raises error."""
        fake_path = tmp_path / "does_not_exist"
        with pytest.raises(ModeDetectionError, match="Path does not exist"):
            mode_detector.find_run_directories([fake_path])

    def test_non_directory_raises_error(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test that file path raises error."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")

        with pytest.raises(ModeDetectionError, match="Path is not a directory"):
            mode_detector.find_run_directories([file_path])

    def test_invalid_directory_raises_error(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test that directory without valid runs raises error."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(
            ModeDetectionError,
            match="does not contain any valid run directories",
        ):
            mode_detector.find_run_directories([empty_dir])

    def test_mixed_valid_and_invalid_paths(
        self, mode_detector: ModeDetector, single_run_dir: Path, tmp_path: Path
    ) -> None:
        """Test handling of mixed valid and invalid paths."""
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()

        # Should raise error on first invalid path
        with pytest.raises(ModeDetectionError):
            mode_detector.find_run_directories([single_run_dir, invalid_dir])


class TestIsRunDirectory:
    """Tests for _is_run_directory helper."""

    def test_valid_run_directory(
        self, mode_detector: ModeDetector, single_run_dir: Path
    ) -> None:
        """Test that valid run directory is detected."""

        assert mode_detector._is_run_directory(single_run_dir) is True

    def test_directory_without_required_file(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test that directory without profile_export.jsonl is not a run."""

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        assert mode_detector._is_run_directory(empty_dir) is False

    def test_file_path_returns_false(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test that file path returns False."""

        file_path = tmp_path / "test.txt"
        file_path.write_text("test")

        assert mode_detector._is_run_directory(file_path) is False


class TestDuplicatePaths:
    """Tests for duplicate path handling."""

    def test_same_path_twice(
        self, mode_detector: ModeDetector, single_run_dir: Path
    ) -> None:
        """Test same path specified multiple times."""
        mode, run_dirs_from_mode = mode_detector.detect_mode(
            [single_run_dir, single_run_dir]
        )
        # Should deduplicate to single run
        assert mode == VisualizationMode.SINGLE_RUN
        assert len(run_dirs_from_mode) == 1

        # Should deduplicate
        runs = mode_detector.find_run_directories([single_run_dir, single_run_dir])
        assert len(runs) == 1

    def test_resolved_path_duplicates(
        self, mode_detector: ModeDetector, single_run_dir: Path
    ) -> None:
        """Test paths that resolve to same directory."""
        # Create paths that look different but resolve to same location
        path1 = single_run_dir
        path2 = single_run_dir / ".." / single_run_dir.name

        runs = mode_detector.find_run_directories([path1, path2])
        assert len(runs) == 1

    def test_parent_and_child_paths_mixed(
        self, mode_detector: ModeDetector, parent_dir_with_runs: Path
    ) -> None:
        """Test parent directory + explicit child paths."""
        # Get first child run directory
        children = [d for d in parent_dir_with_runs.iterdir() if d.is_dir()]
        child = children[0]

        runs = mode_detector.find_run_directories([parent_dir_with_runs, child])
        # Should deduplicate - child appears once (3 total, not 4)
        assert len(runs) == 3


class TestNestedRunDirectories:
    """Tests for nested run directory handling."""

    def test_nested_run_directories(
        self, mode_detector: ModeDetector, nested_run_dirs: Path
    ) -> None:
        """Test run directory containing another run."""
        # Should find both outer and inner runs
        runs = mode_detector.find_run_directories([nested_run_dirs])
        assert len(runs) == 2

        # Should detect as multi-run (2 runs found)
        mode, run_dirs = mode_detector.detect_mode([nested_run_dirs])
        assert mode == VisualizationMode.MULTI_RUN
        assert len(run_dirs) == 2

    def test_nested_runs_counted_separately(
        self,
        mode_detector: ModeDetector,
        tmp_path: Path,
        sample_jsonl_data,
        sample_aggregated_data,
    ) -> None:
        """Test that nested runs are counted separately."""
        # Create outer run
        outer = tmp_path / "outer_run"
        outer.mkdir()
        (outer / "profile_export.jsonl").write_text('{"test": "outer"}\n')
        (outer / "profile_export_aiperf.json").write_text('{"test": "outer"}')

        # Create inner nested run
        inner = outer / "inner_run"
        inner.mkdir()
        (inner / "profile_export.jsonl").write_text('{"test": "inner"}\n')
        (inner / "profile_export_aiperf.json").write_text('{"test": "inner"}')

        # Add another standalone run at the outer level
        standalone = outer / "standalone_run"
        standalone.mkdir()
        jsonl_file = standalone / "profile_export.jsonl"
        with open(jsonl_file, "w") as f:
            for record in sample_jsonl_data:
                f.write(f"{orjson.dumps(record).decode('utf-8')}\n")
        json_file = standalone / "profile_export_aiperf.json"
        json_file.write_bytes(orjson.dumps(sample_aggregated_data))

        # Should find 3 runs total: outer, inner, standalone
        runs = mode_detector.find_run_directories([outer])
        assert len(runs) == 3

    def test_deeply_nested_runs(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test multiple levels of nesting."""
        # Create three levels of nesting
        level1 = tmp_path / "level1"
        level1.mkdir()
        (level1 / "profile_export.jsonl").write_text('{"test": "level1"}\n')
        (level1 / "profile_export_aiperf.json").write_text('{"test": "level1"}\n')

        level2 = level1 / "level2"
        level2.mkdir()
        (level2 / "profile_export.jsonl").write_text('{"test": "level2"}\n')
        (level2 / "profile_export_aiperf.json").write_text('{"test": "level2"}\n')

        level3 = level2 / "level3"
        level3.mkdir()
        (level3 / "profile_export.jsonl").write_text('{"test": "level3"}\n')
        (level3 / "profile_export_aiperf.json").write_text('{"test": "level3"}\n')

        # Should find all 3 levels
        runs = mode_detector.find_run_directories([tmp_path])
        assert len(runs) == 3


class TestFileContentEdgeCases:
    """Tests for file content edge cases."""

    def test_empty_profile_export_file(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test run with empty profile_export.jsonl."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "profile_export.jsonl").write_text("")
        (run_dir / "profile_export_aiperf.json").write_text("{}")

        # Mode detection treats as valid
        assert mode_detector._is_run_directory(run_dir)
        mode, run_dirs = mode_detector.detect_mode([run_dir])
        assert mode == VisualizationMode.SINGLE_RUN
        assert len(run_dirs) == 1

        # DataLoader will fail (tested elsewhere)

    def test_corrupted_profile_export_file(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test run with corrupted JSON."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "profile_export.jsonl").write_text("not valid json{{{")
        (run_dir / "profile_export_aiperf.json").write_text("{}")

        # Mode detection doesn't validate content
        assert mode_detector._is_run_directory(run_dir)


class TestVeryDeepNesting:
    """Tests for very deep nesting scenarios."""

    def test_very_deep_nesting_11_levels(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test handling of very deep nesting (11 levels)."""
        # Create 11 levels of nested run directories
        current_dir = tmp_path
        for i in range(11):
            level_dir = current_dir / f"level{i}"
            level_dir.mkdir()
            (level_dir / "profile_export.jsonl").write_text(
                f'{{"level": {i}, "test": "data"}}\n'
            )
            (level_dir / "profile_export_aiperf.json").write_text(
                f'{{"level": {i}, "test": "data"}}'
            )
            current_dir = level_dir

        # Should find all 11 levels
        runs = mode_detector.find_run_directories([tmp_path])
        assert len(runs) == 11

    def test_very_deep_nesting_15_levels(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test handling of extremely deep nesting (15 levels)."""
        # Create 15 levels of nested run directories
        current_dir = tmp_path
        for i in range(15):
            level_dir = current_dir / f"level{i}"
            level_dir.mkdir()
            (level_dir / "profile_export.jsonl").write_text(
                f'{{"level": {i}, "test": "data"}}\n'
            )
            (level_dir / "profile_export_aiperf.json").write_text(
                f'{{"level": {i}, "test": "data"}}'
            )
            current_dir = level_dir

        # Should find all 15 levels
        runs = mode_detector.find_run_directories([tmp_path])
        assert len(runs) == 15

    def test_mixed_depth_nesting(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test mixed depth nesting with runs at various levels."""
        # Create shallow run
        shallow = tmp_path / "shallow_run"
        shallow.mkdir()
        (shallow / "profile_export.jsonl").write_text('{"test": "shallow"}\n')
        (shallow / "profile_export_aiperf.json").write_text('{"test": "shallow"}')

        # Create medium depth run (5 levels deep)
        current_dir = tmp_path / "medium"
        current_dir.mkdir()
        for i in range(5):
            level_dir = current_dir / f"level{i}"
            level_dir.mkdir()
            current_dir = level_dir

        (current_dir / "profile_export.jsonl").write_text('{"test": "medium"}\n')
        (current_dir / "profile_export_aiperf.json").write_text('{"test": "medium"}')

        # Create deep run (12 levels deep)
        current_dir = tmp_path / "deep"
        current_dir.mkdir()
        for i in range(12):
            level_dir = current_dir / f"level{i}"
            level_dir.mkdir()
            current_dir = level_dir

        (current_dir / "profile_export.jsonl").write_text('{"test": "deep"}\n')
        (current_dir / "profile_export_aiperf.json").write_text('{"test": "deep"}')

        # Should find all 3 runs at different depths
        runs = mode_detector.find_run_directories([tmp_path])
        assert len(runs) == 3


class TestSymlinkEdgeCases:
    """Tests for additional symlink edge cases."""

    def test_broken_symlink_to_file(
        self, mode_detector: ModeDetector, tmp_path: Path
    ) -> None:
        """Test broken symlink pointing to a file (not directory)."""
        # Create a symlink to a non-existent file
        symlink = tmp_path / "broken_file_symlink"
        target = tmp_path / "nonexistent_file.txt"
        symlink.symlink_to(target)

        # Should handle gracefully - broken file symlinks should be ignored
        with pytest.raises(ModeDetectionError):
            mode_detector.find_run_directories([tmp_path])

    def test_symlink_chain_to_run(
        self,
        mode_detector: ModeDetector,
        tmp_path: Path,
        sample_jsonl_data,
        sample_aggregated_data,
    ) -> None:
        """Test chain of symlinks pointing to a run directory."""
        # Create actual run directory
        actual_run = tmp_path / "actual_run"
        actual_run.mkdir()
        jsonl_file = actual_run / "profile_export.jsonl"
        with open(jsonl_file, "w") as f:
            for record in sample_jsonl_data:
                f.write(f"{orjson.dumps(record).decode('utf-8')}\n")
        json_file = actual_run / "profile_export_aiperf.json"
        json_file.write_bytes(orjson.dumps(sample_aggregated_data))

        # Create chain of symlinks: symlink1 -> symlink2 -> actual_run
        symlink2 = tmp_path / "symlink2"
        symlink2.symlink_to(actual_run, target_is_directory=True)

        symlink1 = tmp_path / "symlink1"
        symlink1.symlink_to(symlink2, target_is_directory=True)

        # Should resolve the chain and find the run
        runs = mode_detector.find_run_directories([symlink1])
        assert len(runs) == 1
        # Should resolve to the actual directory
        assert runs[0].resolve() == actual_run.resolve()

    def test_symlink_to_parent_directory_containing_runs(
        self,
        mode_detector: ModeDetector,
        tmp_path: Path,
        sample_jsonl_data,
        sample_aggregated_data,
    ) -> None:
        """Test symlink pointing to parent directory containing multiple runs."""
        # Create parent directory with multiple runs
        parent = tmp_path / "parent"
        parent.mkdir()

        for i in range(3):
            run_dir = parent / f"run{i}"
            run_dir.mkdir()
            jsonl_file = run_dir / "profile_export.jsonl"
            with open(jsonl_file, "w") as f:
                for record in sample_jsonl_data:
                    f.write(f"{orjson.dumps(record).decode('utf-8')}\n")
            json_file = run_dir / "profile_export_aiperf.json"
            json_file.write_bytes(orjson.dumps(sample_aggregated_data))

        # Create symlink to parent
        parent_symlink = tmp_path / "parent_link"
        parent_symlink.symlink_to(parent, target_is_directory=True)

        # Should find all 3 runs through symlink
        runs = mode_detector.find_run_directories([parent_symlink])
        assert len(runs) == 3
