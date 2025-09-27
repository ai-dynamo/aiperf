# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any

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
            try:
                importlib.import_module(f"aiperf.{module.name}")
            except ImportError:
                _logger.exception(
                    f"Error loading AIPerf module: aiperf.{module.name}. Ensure the folder {module.resolve()} is a valid Python package"
                )
                raise


_modules_loaded = False
_modules_loaded_lock = threading.Lock()
_plugin_discovery_lock = threading.Lock()
_discovered_plugins: set[tuple[str, str]] = set()


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


# Factory name to module mapping for targeted plugin discovery
_FACTORY_MODULE_MAPPING = {
    "ServiceFactory": ["controller", "workers", "records", "dataset", "timing"],
    "InferenceClientFactory": ["clients"],
    "RequestConverterFactory": ["clients"],
    "ResponseExtractorFactory": ["clients", "parsers"],
    "DataExporterFactory": ["exporters"],
    "ConsoleExporterFactory": ["exporters"],
    "ComposerFactory": ["dataset"],
    "CustomDatasetFactory": ["dataset"],
    "CommunicationClientFactory": ["zmq"],
    "CommunicationFactory": ["zmq"],
    "ZMQProxyFactory": ["zmq"],
    "AIPerfUIFactory": ["ui"],
    "ServiceManagerFactory": ["controller"],
    "RecordProcessorFactory": ["post_processors"],
    "ResultsProcessorFactory": ["post_processors"],
    "RequestRateGeneratorFactory": ["timing"],
    "OpenAIObjectParserFactory": ["parsers"],
}


def discover_and_load_plugin(factory_name: str, class_type: Any) -> None:
    """Discover and load a plugin for a specific factory and class type.

    Args:
        factory_name: The name of the factory (e.g., "ServiceFactory")
        class_type: The class type to discover a plugin for
    """
    class_type_str = str(class_type)
    plugin_key = (factory_name, class_type_str)

    # Check if we've already attempted to discover this plugin
    with _plugin_discovery_lock:
        if plugin_key in _discovered_plugins:
            _logger.debug(
                f"Plugin discovery for {class_type_str!r} in {factory_name} already attempted"
            )
            return
        _discovered_plugins.add(plugin_key)

    _logger.debug(f"Discovering plugin for {class_type_str!r} in {factory_name}")

    # Get the candidate modules for this factory
    candidate_modules = _FACTORY_MODULE_MAPPING.get(factory_name, [])

    if not candidate_modules:
        _logger.debug(f"No candidate modules found for factory {factory_name}")
        return

    # Try to load each candidate module
    for module_name in candidate_modules:
        try:
            _logger.debug(f"Attempting to load module: aiperf.{module_name}")
            importlib.import_module(f"aiperf.{module_name}")
            _logger.debug(f"Successfully loaded module: aiperf.{module_name}")
        except ImportError as e:
            _logger.debug(f"Failed to load module aiperf.{module_name}: {e}")
            continue
        except Exception as e:
            _logger.warning(f"Error loading module aiperf.{module_name}: {e}")
            continue
