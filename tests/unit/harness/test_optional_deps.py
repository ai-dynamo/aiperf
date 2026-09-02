# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import patch

import orjson

from tests.harness.optional_deps import _test_files_needing_unavailable_deps


def test__test_files_needing_unavailable_deps_first_call_populates_cache(
    tmp_path: Path,
) -> None:
    """Cache JSON is written to _CACHE_DIR on the first call when deps are absent."""
    fake_dep = "totally_absent_dep_xyz"
    test_file = tmp_path / "test_fake.py"
    test_file.write_text(f"import {fake_dep}\n")

    cache_dir = tmp_path / ".pytest_cache"

    with (
        patch(
            "tests.harness.optional_deps.unavailable_gated_deps",
            return_value={fake_dep},
        ),
        patch("tests.harness.optional_deps._CACHE_DIR", cache_dir),
    ):
        result = _test_files_needing_unavailable_deps(tmp_path)

    assert result == [test_file]
    cache_file = cache_dir / "optional_deps_scan.json"
    assert cache_file.exists()
    data = orjson.loads(cache_file.read_bytes())
    key = f"{tmp_path}|{fake_dep}"
    assert str(test_file) in data[key]["files"]
    assert data[key]["total"] == 1


def test__test_files_needing_unavailable_deps_second_call_reads_from_cache(
    tmp_path: Path,
) -> None:
    """Second call with the same file tree returns the cached result without re-scanning."""
    fake_dep = "totally_absent_dep_xyz"
    test_file = tmp_path / "test_fake.py"
    test_file.write_text(f"import {fake_dep}\n")

    cache_dir = tmp_path / ".pytest_cache"

    with (
        patch(
            "tests.harness.optional_deps.unavailable_gated_deps",
            return_value={fake_dep},
        ),
        patch("tests.harness.optional_deps._CACHE_DIR", cache_dir),
    ):
        # First call populates cache.
        _test_files_needing_unavailable_deps(tmp_path)
        # Overwrite the file so a fresh AST scan would return [] (import gone),
        # but the file count is unchanged, so the cache should still be used.
        test_file.write_text("# no imports\n")
        result = _test_files_needing_unavailable_deps(tmp_path)

    assert result == [test_file]


def test__test_files_needing_unavailable_deps_new_file_invalidates_cache(
    tmp_path: Path,
) -> None:
    """Adding a test_*.py file invalidates the cache so the new file is not missed."""
    fake_dep = "totally_absent_dep_xyz"
    test_file = tmp_path / "test_fake.py"
    test_file.write_text(f"import {fake_dep}\n")

    cache_dir = tmp_path / ".pytest_cache"

    with (
        patch(
            "tests.harness.optional_deps.unavailable_gated_deps",
            return_value={fake_dep},
        ),
        patch("tests.harness.optional_deps._CACHE_DIR", cache_dir),
    ):
        # First call: caches [test_fake.py], total=1.
        _test_files_needing_unavailable_deps(tmp_path)
        # Add a second test file with the gated import.
        new_file = tmp_path / "test_new.py"
        new_file.write_text(f"import {fake_dep}\n")
        # Second call: total is now 2, so cache is stale → re-scan finds both.
        result = _test_files_needing_unavailable_deps(tmp_path)

    # Both files must be present — the rescan must not lose the original file.
    assert test_file in result
    assert new_file in result
    # Cache entry should be refreshed with total=2.
    key = f"{tmp_path}|{fake_dep}"
    data = orjson.loads((cache_dir / "optional_deps_scan.json").read_bytes())
    assert data[key]["total"] == 2


def test__test_files_needing_unavailable_deps_all_deps_present_skips_cache(
    tmp_path: Path,
) -> None:
    """Cache is never written when all gated deps are present on this platform."""
    cache_dir = tmp_path / ".pytest_cache"

    with (
        patch(
            "tests.harness.optional_deps.unavailable_gated_deps",
            return_value=set(),
        ),
        patch("tests.harness.optional_deps._CACHE_DIR", cache_dir),
    ):
        result = _test_files_needing_unavailable_deps(tmp_path)

    assert result == []
    assert not cache_dir.exists()
