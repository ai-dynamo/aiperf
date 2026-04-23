# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end kill test for the memory watchdog.

Unlike test_memory_watchdog.py (which swaps in a recorder for
_watchdog_kill_action to keep the test in-process), this test launches a
real pytest subprocess that allocates until the watchdog's actual
os._exit(137) path fires. The runaway allocation has an 8 GiB self-imposed
safety fuse so a broken watchdog cannot exhaust host memory.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.conftest import _WATCHDOG_SUPPORTED

pytestmark = pytest.mark.skipif(
    not _WATCHDOG_SUPPORTED,
    reason="memory watchdog requires Linux /proc/self/smaps_rollup or /proc/self/status",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.no_memory_limit
def test_runaway_allocation_is_killed_by_watchdog(tmp_path: Path) -> None:
    """Spawn a pytest subprocess whose test allocates until killed.

    The marker sets a 256 MiB cap. The inline test allocates 32 MiB chunks
    and writes into each page to force residency. Safety fuse: if the test
    ever reaches 8 GiB of allocations, it fails loudly instead of continuing
    to grow - that would indicate the watchdog never fired.
    """
    _RUNAWAY_TEST = textwrap.dedent(
        """
        import pytest


        @pytest.mark.memory_limit(mb=256)
        def test_runaway_inline():
            chunks: list[bytearray] = []
            CHUNK_MIB = 32
            SAFETY_FUSE_MIB = 8192
            for _ in range(SAFETY_FUSE_MIB // CHUNK_MIB):
                buf = bytearray(CHUNK_MIB * 1024 * 1024)
                for i in range(0, len(buf), 4096):
                    buf[i] = 1  # force page residency
                chunks.append(buf)
            raise AssertionError(
                "safety fuse tripped at 8 GiB: watchdog did not kill the worker"
            )
        """
    ).lstrip()

    # The watchdog only arms for nodeids starting with "tests/unit/" or
    # "tests/component_integration/". Put the test there relative to the
    # subprocess's rootdir (= tmp_path).
    unit_dir = tmp_path / "tests" / "unit"
    unit_dir.mkdir(parents=True)
    (unit_dir / "test_runaway_inline.py").write_text(_RUNAWAY_TEST)

    # Delegate conftest to the real one so the watchdog hooks load.
    conftest = tmp_path / "tests" / "conftest.py"
    conftest.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, "{_REPO_ROOT}")
            # Import installs the watchdog hookimpls and starts the thread.
            from tests.conftest import *  # noqa: F401,F403
            """
        ).lstrip()
    )

    # Minimal pyproject so the subprocess recognizes the markers.
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [tool.pytest.ini_options]
            testpaths = ["tests"]
            markers = [
              "memory_limit(mb=N): cap this test's memory usage at N MiB",
              "no_memory_limit: disable the per-test memory watchdog",
            ]
            """
        ).lstrip()
    )

    log_path = tmp_path / "watchdog.log"
    env = {**os.environ, "AIPERF_WATCHDOG_LOG_FILE": str(log_path)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_runaway_inline.py",
            "-p",
            "no:xdist",
            "-p",
            "no:cacheprovider",
            "-v",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )

    assert result.returncode == 137, (
        f"expected exit 137 (watchdog os._exit), got {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    # pytest's fd-level capture eats stderr writes across os._exit; the
    # watchdog's diagnostic is instead written to the log file whose path
    # we set via AIPERF_WATCHDOG_LOG_FILE. Read it to verify attribution.
    assert log_path.exists(), (
        f"watchdog log file missing at {log_path}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    log = log_path.read_text()
    assert "pytest memory watchdog tripped" in log, log
    assert "test_runaway_inline" in log, log
    assert "killing worker pid" in log, log
