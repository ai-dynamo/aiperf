# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detection of optional native dependencies that have no Windows-on-ARM build.

Some dependencies ship no ``win_arm64`` wheel (pyarrow, datasets, cryptography
via trustme) or bundle a native library with no ARM build (soundfile bundles
libsndfile). Tests and source modules that hard-depend on these cannot run on
platforms where the dependency is unavailable.

This module is the single source of truth for *what* is unavailable and *which*
modules to skip. It is consumed by:
- ``tests/unit/conftest.py`` -- ``collect_ignore`` (skip whole test modules at
  collection, before their import chain crashes the collector).
- ``tests/unit/test_imports.py`` -- filters the all-modules import sweep so it
  does not attempt (and hard-crash on) modules requiring absent native libs.
"""

import importlib.util
import platform

from aiperf.common.constants import IS_WINDOWS

# Windows-on-ARM: native render/codec backends (kaleido's browser engine, etc.)
# have no working ARM build and hard-crash (access violation) rather than
# raising, so affected tests must be skipped by platform rather than probed.
IS_WINDOWS_ARM = IS_WINDOWS and platform.machine() == "ARM64"


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

# --- Unit-test modules that must be skipped when a native dep is absent -------
#
# Only test modules that DIRECTLY import the dep at module top are listed: src
# modules import these deps lazily (so importing a loader no longer requires
# them), and the import-sweep test (test_imports.py) treats marker-gated
# ImportErrors as expected skips. The plot subtree is handled separately in
# tests/unit/plot/conftest.py (its visualization stack hard-crashes on win-arm).
_PYARROW_TEST_RELPATHS = ("server_metrics/test_parquet_exporter.py",)
_TRUSTME_TEST_RELPATHS = ("transports/test_tcp_connector.py",)
# Accuracy benchmark tests import their (datasets-backed) benchmark module, and
# test_hf_image_feature_schemas imports ``datasets`` directly.
_DATASETS_TEST_RELPATHS = (
    "accuracy/test_lcb_codegeneration_benchmark.py",
    "accuracy/test_aime_benchmark.py",
    "accuracy/test_aime24_benchmark.py",
    "accuracy/test_aime25_benchmark.py",
    "accuracy/test_bigbench_benchmark.py",
    "accuracy/test_gpqa_diamond_benchmark.py",
    "accuracy/test_gsm8k_benchmark.py",
    "accuracy/test_math_500_benchmark.py",
    "dataset/loader/test_hf_image_feature_schemas.py",
)
# These import ``soundfile`` directly at module top.
_SOUNDFILE_TEST_RELPATHS = (
    "dataset/generator/test_audio_generator.py",
    "dataset/generator/test_video_generator.py",
    "dataset/loader/test_hf_asr_loader.py",
)


def unsupported_unit_test_relpaths() -> list[str]:
    """Unit-test modules (relative to tests/unit) to skip given absent deps.

    Each directly imports, at module top, a native dependency with no
    Windows-on-ARM build, so it crashes at collection there.
    """
    relpaths: list[str] = []
    if not HAS_PYARROW:
        relpaths += _PYARROW_TEST_RELPATHS
    if not HAS_TRUSTME:
        relpaths += _TRUSTME_TEST_RELPATHS
    if not HAS_DATASETS:
        relpaths += _DATASETS_TEST_RELPATHS
    if not HAS_SOUNDFILE:
        relpaths += _SOUNDFILE_TEST_RELPATHS
    return relpaths
