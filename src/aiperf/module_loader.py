# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module loader for AIPerf.

This module is used to load all modules into the system to ensure everything is
registered and ready to be used. This is done to avoid the performance penalty of
importing all modules during CLI startup, while still ensuring that all
implementations are properly registered with their factories.
"""

import importlib
import threading
import time
from pathlib import Path

from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)


def _load_all_modules() -> None:
    """Import all top-level modules to trigger their registration decorators.

    This is called only when modules are actually needed, not during CLI startup.
    """
    for module in sorted(Path(__file__).parent.iterdir()):
        if (
            module.is_dir()
            and not module.name.startswith("_")
            and not module.name.startswith(".")
            # ignore plot module to avoid loading plot dependencies (slow)
            and module.name != "plot"
            and (module / "__init__.py").exists()
        ):
            # Recursively find all Python files in this module and its subdirectories
            for file in sorted(module.rglob("*.py")):
                # Skip __pycache__ and other hidden/private directories
                # Check each directory component (not the filename) for leading underscore/dot
                skip = False
                for i, part in enumerate(file.parts):
                    # Skip check for the last part (filename) and for parts before our module
                    if (
                        i >= len(Path(__file__).parent.parts)
                        and part != file.name
                        and (part.startswith("_") or part.startswith("."))
                    ):
                        skip = True
                        break
                if skip:
                    _logger.debug(f"Skipping private/hidden path: {file}")
                    continue

                # Build qualified module name from path
                relative_path = file.relative_to(Path(__file__).parent)
                parts = list(relative_path.parts[:-1])  # Remove filename
                if file.name != "__init__.py":
                    parts.append(file.stem)  # Add module name without .py extension

                qualified_name = (
                    "aiperf." + ".".join(parts) if parts else f"aiperf.{file.stem}"
                )

                _logger.debug(f"Loading module: {qualified_name}")
                try:
                    importlib.import_module(qualified_name)
                except ImportError:
                    _logger.exception(
                        f"Error loading AIPerf module: {qualified_name}. Ensure the file {file.resolve()} is a valid Python module"
                    )
                    raise


_modules_loaded = False
_modules_loaded_lock = threading.Lock()


def ensure_modules_loaded() -> None:
    """Ensure all modules are loaded exactly once."""
    global _modules_loaded
    with _modules_loaded_lock:
        if not _modules_loaded:
            start_time = time.perf_counter()
            _logger.debug("Loading all modules")
            _load_all_modules()
            _logger.debug(
                f"Modules loaded in {time.perf_counter() - start_time:.2f} seconds"
            )
            _modules_loaded = True
