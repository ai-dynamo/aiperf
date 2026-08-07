# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detection of optional native dependencies that have no Windows-on-ARM build.

Some dependencies ship no ``win_arm64`` wheel (pyarrow, datasets, cryptography
via trustme) or bundle a native library with no ARM build (soundfile bundles
libsndfile). A test module that imports one of these at module top can't even
be collected on such a platform.

Rather than hand-maintain a list of such test files (which silently rots as new
tests are added), we **statically scan** each test module's top-level imports
with ``ast`` -- never importing it, so the soundfile/libsndfile load that would
crash is avoided -- and skip the ones that import a currently-unavailable dep.
On platforms where every gated dep is present (dev / Linux / Windows-x86) the
scan yields nothing, so those platforms are unaffected.

Consumed by:
- ``tests/unit/conftest.py`` and ``tests/component_integration/conftest.py`` --
  ``collect_ignore`` (skip whole modules before their imports crash collection).
- ``tests/unit/test_imports.py`` -- filters the all-modules import sweep, which
  imports test modules directly and would otherwise crash on the same imports.

The plot subtree is handled separately (``tests/unit/plot/conftest.py``): its
kaleido/plotly stack imports fine but hard-crashes at *render* time, so it is
gated on ``IS_WINDOWS_ARM`` rather than on a missing import.
"""

import ast
import importlib.util
import os
from pathlib import Path

import orjson

# Windows-on-ARM: native render/codec backends (kaleido's browser engine, etc.)
# have no working ARM build and hard-crash (access violation) rather than
# raising, so affected tests must be skipped by platform rather than probed.
from aiperf.common.constants import IS_WINDOWS_ARM  # noqa: F401

# .pytest_cache lives next to the repo root conftest.  All xdist workers share
# the same filesystem, so one worker populates the cache and the rest read it —
# no subprocess coordination needed.  The cache is only written on platforms
# where unavailable_gated_deps() is non-empty (currently Windows-ARM); on all
# other platforms _test_files_needing_unavailable_deps() short-circuits before
# touching this path.
_CACHE_DIR = Path(__file__).parents[2] / ".pytest_cache"
_CACHE_FILE_NAME = "optional_deps_scan.json"


def is_installed(module: str) -> bool:
    """Whether ``module`` is present, without importing it.

    Suitable for deps that are simply absent on a platform (ImportError),
    e.g. pyarrow/datasets/trustme on Windows-on-ARM.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def soundfile_usable() -> bool:
    """Whether ``soundfile`` can actually load on this platform.

    ``find_spec`` is insufficient: soundfile installs everywhere, but its
    bundled ``libsndfile`` has no Windows-on-ARM build, so the import raises
    OSError at native-library load time rather than ImportError.
    """
    try:
        import soundfile  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


HAS_PYARROW = is_installed("pyarrow")
HAS_DATASETS = is_installed("datasets")
HAS_TRUSTME = is_installed("trustme")
HAS_SOUNDFILE = soundfile_usable()


def unavailable_gated_deps() -> set[str]:
    """Top-level module names of gated native deps unusable on this platform.

    A test importing any of these at module top cannot be collected here.
    Empty on platforms where all are present.
    """
    unavailable: set[str] = set()
    if not HAS_PYARROW:
        unavailable.add("pyarrow")
    if not HAS_DATASETS:
        unavailable.add("datasets")
    if not HAS_TRUSTME:
        unavailable.add("trustme")
    if not HAS_SOUNDFILE:
        unavailable.add("soundfile")
    return unavailable


def _top_level_imports(path: Path) -> set[str]:
    """Top-level module names a file imports, via static AST parse (not executed).

    Only module-scope imports count: imports inside functions or
    ``if TYPE_CHECKING:`` blocks don't run at import time, so they can't crash
    collection and are intentionally ignored. Relative imports (first-party)
    are ignored too.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _test_files_needing_unavailable_deps(test_dir: Path) -> list[Path]:
    """Sorted test files under ``test_dir`` whose top-level imports need a dep
    that is unavailable on this platform. Empty when all gated deps are present.

    Results are cached in ``.pytest_cache/optional_deps_scan.json`` so that
    multiple xdist worker processes pay the rglob+AST cost at most once per CI
    job.  The cache key includes the resolved ``test_dir`` path and the sorted
    set of currently-unavailable deps; the entry also stores the total count of
    ``test_*.py`` files so that adding a new file (which would not appear in
    the cached skip list) invalidates the entry on the next call.
    """
    unavailable = unavailable_gated_deps()
    if not unavailable:
        return []

    cache_file = _CACHE_DIR / _CACHE_FILE_NAME
    # Include the sorted unavailable set so a dep becoming available (e.g. a
    # local install) also invalidates the cached skip list.
    key = f"{test_dir.resolve()}|{','.join(sorted(unavailable))}"

    # List all test files once: used for count-based cache validation on a hit
    # and for the AST scan on a miss.  Directory listing (no reads) is cheap
    # relative to 912 read+parse operations.
    all_test_files = sorted(test_dir.rglob("test_*.py"))
    total = len(all_test_files)

    # Try to read an existing cache entry.  A hit is only valid when the total
    # file count matches — any addition or removal of test_*.py files forces a
    # re-scan so newly gated imports are never silently missed.
    try:
        raw: dict[str, dict[str, object]] = orjson.loads(cache_file.read_bytes())
        entry = raw.get(key)
        if isinstance(entry, dict) and entry.get("total") == total:
            return [Path(p) for p in entry["files"]]  # type: ignore[arg-type]
        data = dict(raw)
    except (OSError, orjson.JSONDecodeError):
        data = {}

    # Cache miss: full scan (read + AST parse per file).
    found = [p for p in all_test_files if _top_level_imports(p) & unavailable]

    # Write result back (best-effort). Re-read first to merge any entries
    # another worker wrote while we were scanning, reducing lost-update
    # exposure. Use a per-process tmp path so concurrent workers don't stomp
    # each other's writes on Windows (where os.replace on a shared .tmp path
    # can raise PermissionError).
    data[key] = {"files": [str(p) for p in found], "total": total}
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            merged: dict[str, dict[str, object]] = orjson.loads(cache_file.read_bytes())
            merged.update(data)
            data = merged
        except (OSError, orjson.JSONDecodeError):
            pass
        tmp = _CACHE_DIR / f"optional_deps_scan.{os.getpid()}.tmp"
        tmp.write_bytes(orjson.dumps(data))
        os.replace(tmp, cache_file)
    except OSError:
        pass

    return found


def collect_ignore_for_unavailable_deps(test_dir: str | Path) -> list[str]:
    """``collect_ignore`` entries (relative to ``test_dir``) for test modules
    whose top-level imports need a dependency unavailable on this platform.

    New tests that add a gated top-level import self-gate with no edits here.
    """
    base = Path(test_dir)
    return [
        str(path.relative_to(base))
        for path in _test_files_needing_unavailable_deps(base)
    ]


def unsupported_test_module_names(
    test_dir: str | Path, package_prefix: str
) -> set[str]:
    """Dotted module names (e.g. ``tests.unit.x.test_y``) for the same files.

    ``test_imports.py`` imports test modules directly (bypassing
    ``collect_ignore``), so it must filter the same set out of its sweep.
    """
    base = Path(test_dir)
    return {
        package_prefix + "." + ".".join(path.relative_to(base).with_suffix("").parts)
        for path in _test_files_needing_unavailable_deps(base)
    }
