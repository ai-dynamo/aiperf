# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module loader for AIPerf.

This module is used to load all modules into the system to ensure everything is
registered and ready to be used. This is done to avoid the performance penalty of
importing all modules during CLI startup, while still ensuring that all
implementations are properly registered with their factories.
"""

import importlib
import time
from pathlib import Path

from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)


def _load_all_modules() -> None:
    """Import all top-level modules to trigger their registration decorators.

    This is called only when modules are actually needed, not during CLI startup.
    """
    for module in Path(__file__).parent.iterdir():
        if (
            module.is_dir()
            and not module.name.startswith("_")
            and not module.name.startswith(".")
            and (module / "__init__.py").exists()
        ):
            _logger.debug(f"Loading module: aiperf.{module.name}")
            importlib.import_module(f"aiperf.{module.name}")


_modules_loaded = False


def ensure_modules_loaded() -> None:
    """Ensure all modules are loaded exactly once."""
    global _modules_loaded
    if not _modules_loaded:
        start_time = time.perf_counter()
        _logger.debug("Loading all modules")
        _load_all_modules()
        _logger.debug(
            f"Modules loaded in {time.perf_counter() - start_time:.2f} seconds"
        )
        _modules_loaded = True
