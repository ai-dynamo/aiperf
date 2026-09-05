# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin that the loader package works on platforms without pyarrow.

pyarrow publishes no Windows-on-ARM wheel (apache/arrow#47195), so it is a
platform-conditional dependency. A fresh subprocess with a meta-path finder
that blocks pyarrow imports proves ``aiperf.dataset.loader`` stays importable
and the baseten_trace loader self-disables instead of poisoning the package.
"""

import subprocess
import sys

_NO_PYARROW_SCRIPT = """
import sys


class BlockPyarrow:
    def find_spec(self, name, path=None, target=None):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("pyarrow blocked to simulate Windows-on-ARM")
        return None


sys.meta_path.insert(0, BlockPyarrow())

import aiperf.dataset.loader  # must not eagerly import pyarrow

from aiperf.dataset.loader.baseten_trace import (
    BasetenTraceDatasetLoader,
    count_baseten_parquet_records_and_sessions,
)

assert BasetenTraceDatasetLoader.can_load(filename="x.parquet") is False
assert BasetenTraceDatasetLoader.can_load(filename="x.arrow") is False
assert count_baseten_parquet_records_and_sessions("x.parquet") == (0, 0)

try:
    BasetenTraceDatasetLoader()
except ValueError as exc:
    assert "requires pyarrow" in str(exc), f"unexpected message: {exc}"
else:
    raise AssertionError("BasetenTraceDatasetLoader() should raise without pyarrow")

print("no-pyarrow contract holds")
"""


def test_loader_package_self_disables_without_pyarrow() -> None:
    """The loader package imports, can_load auto-detection declines, and the
    baseten_trace constructor raises actionably when pyarrow is unavailable."""
    result = subprocess.run(
        [sys.executable, "-c", _NO_PYARROW_SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"subprocess failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "no-pyarrow contract holds" in result.stdout
