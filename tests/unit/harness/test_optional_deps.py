# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import patch

from tests.harness.optional_deps import _test_files_needing_unavailable_deps


def test_cache_is_written_on_first_call(tmp_path):
    """A cache JSON is written to .pytest_cache on Windows-ARM (unavailable deps)."""
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
    data = json.loads(cache_file.read_text())
    assert str(test_file) in data[str(tmp_path)]


def test_cache_is_read_on_second_call(tmp_path):
    """Second call reads from cache without re-scanning the filesystem."""
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
        # Delete the test file so a fresh scan would return [].
        test_file.unlink()
        # Second call should still return the cached result.
        result = _test_files_needing_unavailable_deps(tmp_path)

    assert result == [test_file]


def test_no_cache_when_all_deps_present(tmp_path):
    """On platforms with all deps present the cache path is never written."""
    cache_dir = tmp_path / ".pytest_cache"

    with (
        patch("tests.harness.optional_deps.unavailable_gated_deps", return_value=set()),
        patch("tests.harness.optional_deps._CACHE_DIR", cache_dir),
    ):
        result = _test_files_needing_unavailable_deps(tmp_path)

    assert result == []
    assert not cache_dir.exists()
