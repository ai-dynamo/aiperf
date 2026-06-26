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

# --- Canonical lists of modules gated on the deps above -----------------------
#
# Unit-test modules (paths relative to tests/unit) whose top-level import chain
# hard-depends on a native lib with no Windows-on-ARM build.
_PYARROW_TEST_RELPATHS = ("server_metrics/test_parquet_exporter.py",)
_TRUSTME_TEST_RELPATHS = ("transports/test_tcp_connector.py",)
# Accuracy benchmarks import the HF ``datasets`` library at module top.
_DATASETS_TEST_RELPATHS = (
    "accuracy/test_lcb_codegeneration_benchmark.py",
    "accuracy/test_aime_benchmark.py",
    "accuracy/test_aime24_benchmark.py",
    "accuracy/test_aime25_benchmark.py",
    "accuracy/test_bigbench_benchmark.py",
    "accuracy/test_gpqa_diamond_benchmark.py",
    "accuracy/test_gsm8k_benchmark.py",
    "accuracy/test_math_500_benchmark.py",
)
# HF/public loaders whose module chain (base_hf_dataset, hf_asr) eagerly imports
# both ``datasets`` and ``soundfile``.
_HF_LOADER_TEST_RELPATHS = (
    "dataset/loader/test_hf_image_feature_schemas.py",
    "dataset/loader/test_hf_asr_loader.py",
    "dataset/loader/test_hf_conversation_loader.py",
    "dataset/loader/test_hf_dataset_loader.py",
    "dataset/loader/test_mmvu_loader.py",
    "dataset/loader/test_mt_bench.py",
    "dataset/composer/test_public_composer.py",
)
_SOUNDFILE_TEST_RELPATHS = (
    "dataset/generator/test_audio_generator.py",
    "dataset/generator/test_video_generator.py",
)

# src/aiperf modules whose top-level import chain hard-depends on absent natives.
_DATASETS_AIPERF_MODULES = (
    "aiperf.accuracy.benchmarks.aime",
    "aiperf.accuracy.benchmarks.aime24",
    "aiperf.accuracy.benchmarks.aime25",
    "aiperf.accuracy.benchmarks.bigbench",
    "aiperf.accuracy.benchmarks.gpqa_diamond",
    "aiperf.accuracy.benchmarks.gsm8k",
    "aiperf.accuracy.benchmarks.hellaswag",
    "aiperf.accuracy.benchmarks.lcb_codegeneration",
    "aiperf.accuracy.benchmarks.math_500",
    "aiperf.accuracy.benchmarks.mmlu",
)
_HF_LOADER_AIPERF_MODULES = (
    "aiperf.dataset.loader.base_hf_dataset",
    "aiperf.dataset.loader.hf_asr",
    "aiperf.dataset.loader.hf_conversation",
    "aiperf.dataset.loader.hf_instruction_response",
    "aiperf.dataset.loader.mmvu",
    "aiperf.dataset.loader.mt_bench",
)


def unsupported_unit_test_relpaths() -> list[str]:
    """Test modules (relative to tests/unit) to skip given the absent deps."""
    relpaths: list[str] = []
    if not HAS_PYARROW:
        relpaths += _PYARROW_TEST_RELPATHS
    if not HAS_TRUSTME:
        relpaths += _TRUSTME_TEST_RELPATHS
    if not HAS_DATASETS:
        relpaths += _DATASETS_TEST_RELPATHS
    if not HAS_DATASETS or not HAS_SOUNDFILE:
        relpaths += _HF_LOADER_TEST_RELPATHS
    if not HAS_SOUNDFILE:
        relpaths += _SOUNDFILE_TEST_RELPATHS
    return relpaths


def unsupported_import_sweep_modules() -> set[str]:
    """Module paths the all-modules import sweep must skip given absent deps.

    Includes both src/aiperf modules and the ``tests.unit.*`` modules above
    (the sweep imports test modules directly, bypassing ``collect_ignore``).
    Attempting these imports risks a hard interpreter crash from loading a
    wrong-architecture native library, so they are filtered, not caught.
    """
    modules: set[str] = set()
    if not HAS_DATASETS:
        modules.update(_DATASETS_AIPERF_MODULES)
    if not HAS_DATASETS or not HAS_SOUNDFILE:
        modules.update(_HF_LOADER_AIPERF_MODULES)
    for relpath in unsupported_unit_test_relpaths():
        modules.add("tests.unit." + relpath[: -len(".py")].replace("/", "."))
    return modules
