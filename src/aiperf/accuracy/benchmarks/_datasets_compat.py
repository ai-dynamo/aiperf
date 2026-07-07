# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy ``load_dataset`` shim for accuracy benchmarks.

The HuggingFace ``datasets`` package has no ``win_arm64`` wheel, so importing a
benchmark module must not require it at import time (benchmark plugins are
discovered/loaded eagerly). Each benchmark imports ``load_dataset`` from here
instead of from ``datasets`` directly: importing this module is free, the real
``datasets`` import happens only when a dataset is actually loaded, and its
absence surfaces as an actionable ``ConfigurationError`` rather than a raw
``ImportError`` from the plugin loader.

Because benchmarks bind ``load_dataset`` as a module-level name, existing tests
that ``patch("aiperf.accuracy.benchmarks.<name>.load_dataset")`` keep working
unchanged.
"""

from typing import Any

from aiperf.common.exceptions import ConfigurationError


def load_dataset(*args: Any, **kwargs: Any) -> Any:
    """Call ``datasets.load_dataset`` lazily, with a clear win-arm error.

    Raises:
        ConfigurationError: if ``datasets`` cannot be imported (e.g. on
            Windows-on-ARM, which has no ``datasets``/``pyarrow`` wheel).
    """
    try:
        from datasets import load_dataset as _load_dataset
    except ImportError as e:
        raise ConfigurationError(
            "Accuracy benchmarks require the 'datasets' package, which has no "
            "prebuilt Windows-on-ARM wheel (it pulls pyarrow). Run accuracy "
            "benchmarks on Linux or WSL."
        ) from e
    return _load_dataset(*args, **kwargs)
