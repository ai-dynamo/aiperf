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
from pathlib import Path

# Windows-on-ARM: native render/codec backends (kaleido's browser engine, etc.)
# have no working ARM build and hard-crash (access violation) rather than
# raising, so affected tests must be skipped by platform rather than probed.
from aiperf.common.constants import IS_WINDOWS_ARM  # noqa: F401


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
    """
    unavailable = unavailable_gated_deps()
    if not unavailable:
        return []
    return [
        path
        for path in sorted(test_dir.rglob("test_*.py"))
        if _top_level_imports(path) & unavailable
    ]


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
