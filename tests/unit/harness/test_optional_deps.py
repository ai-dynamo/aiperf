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
    assert str(test_file) in data[key]


def test__test_files_needing_unavailable_deps_second_call_reads_from_cache(
    tmp_path: Path,
) -> None:
    """Second call returns the cached result without re-scanning the filesystem."""
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
