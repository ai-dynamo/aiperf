# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module loader for AIPerf.

This module recursively scans Python files for factory registration decorators and provides
lazy loading of modules only when their registered classes are needed. This avoids
the performance penalty of importing all modules during CLI startup while ensuring
implementations are available when requested.
"""

import ast
import importlib
import threading
from pathlib import Path
from typing import Any

from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)


# Global registry for lazy registrations
_lazy_registrations: dict[
    str, dict[str, str]
] = {}  # factory -> {class_type -> module_path}
_registrations_loaded = False
_registrations_lock = threading.Lock()
_plugin_discovery_lock = threading.Lock()
_discovered_plugins: set[tuple[str, str]] = set()


def _parse_decorator_node(
    decorator: ast.expr, file_path: Path
) -> tuple[str, str, list[str]] | None:
    """Parse a decorator AST node to extract factory registration information.

    Returns:
        (factory_name, method_name, class_types) or None if not a factory registration
    """
    if not isinstance(decorator, ast.Call):
        return None

    # Check if it's a factory method call like FactoryName.register()
    if not isinstance(decorator.func, ast.Attribute):
        return None

    if not isinstance(decorator.func.value, ast.Name):
        return None

    factory_name = decorator.func.value.id
    method_name = decorator.func.attr

    # Only process factory registrations
    if not factory_name.endswith("Factory") or method_name not in [
        "register",
        "register_all",
    ]:
        return None

    # Extract class type arguments
    class_types = []
    for arg in decorator.args:
        if isinstance(arg, ast.Constant):
            class_types.append(repr(arg.value))
        elif isinstance(arg, ast.Attribute):
            # Handle enum values like ServiceType.WORKER
            if isinstance(arg.value, ast.Name):
                class_types.append(f"{arg.value.id}.{arg.attr}")
            else:
                class_types.append(ast.unparse(arg))
        else:
            # For other expressions, convert back to string
            try:
                class_types.append(ast.unparse(arg))
            except:  # noqa: E722
                class_types.append(str(arg))

    return factory_name, method_name, class_types


def _file_path_to_module_path(file_path: Path) -> str:
    """Convert a file path to a Python module path."""
    aiperf_root = Path(__file__).parent
    try:
        rel_path = file_path.relative_to(aiperf_root)
        # Convert path to module notation
        module_parts = list(rel_path.parts[:-1])  # Remove filename
        if rel_path.name != "__init__.py":
            module_parts.append(rel_path.stem)  # Add filename without .py
        return "aiperf." + ".".join(module_parts)
    except ValueError:
        # File is not under aiperf package
        return str(file_path)


def _scan_directory_for_registrations(directory: Path) -> None:
    """Recursively scan a directory for Python files with factory registrations."""
    global _lazy_registrations

    for file_path in directory.rglob("*.py"):
        # Skip __pycache__, hidden files, and module_loader.py itself
        if (
            "__pycache__" in file_path.parts
            or file_path.name.startswith(".")
            or file_path.name == "module_loader.py"
        ):
            continue

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse the file as AST to find decorator registrations
            tree = ast.parse(content)
            module_path = _file_path_to_module_path(file_path)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Look at decorators for this class
                    for decorator in node.decorator_list:
                        reg_info = _parse_decorator_node(decorator, file_path)
                        if reg_info:
                            factory_name, method_name, class_types = reg_info

                            # Initialize factory dict if needed
                            if factory_name not in _lazy_registrations:
                                _lazy_registrations[factory_name] = {}

                            # Store each class type -> module mapping
                            for class_type in class_types:
                                _lazy_registrations[factory_name][class_type] = (
                                    module_path
                                )

        except Exception as e:
            _logger.debug(f"Error scanning file {file_path}: {e}")
            continue


def _build_lazy_registration_index() -> None:
    """Build an index of all factory registrations without importing modules."""
    global _registrations_loaded

    with _registrations_lock:
        if _registrations_loaded:
            return

        _logger.debug("Building lazy registration index...")

        # Scan the entire aiperf package
        aiperf_root = Path(__file__).parent
        _scan_directory_for_registrations(aiperf_root)

        _registrations_loaded = True
        total_regs = sum(
            len(class_types) for class_types in _lazy_registrations.values()
        )
        _logger.debug(f"Lazy registration index built with {total_regs} registrations")


def discover_and_load_plugin(factory_name: str, class_type: Any) -> None:
    """Discover and load a plugin for a specific factory and class type."""
    class_type_str = str(class_type)
    plugin_key = (factory_name, class_type_str)

    # Check if we've already attempted to discover this plugin
    with _plugin_discovery_lock:
        if plugin_key in _discovered_plugins:
            return
        _discovered_plugins.add(plugin_key)

    # Ensure the lazy registration index is built
    _build_lazy_registration_index()

    # Look for lazy registrations for this factory
    factory_registrations = _lazy_registrations.get(factory_name, {})

    # Try different representations of the class type
    possible_types = [class_type_str]

    # Add full enum representation if it's an enum
    if hasattr(class_type, "__class__") and hasattr(class_type.__class__, "__name__"):  # noqa: SIM102
        if "." not in class_type_str:  # If it's just the enum value (e.g., "worker")
            enum_class_name = class_type.__class__.__name__
            full_enum_repr = f"{enum_class_name}.{class_type.name}"
            possible_types.append(full_enum_repr)

    # Try to find and load the module
    for possible_type in possible_types:
        if possible_type in factory_registrations:
            module_path = factory_registrations[possible_type]
            try:
                _logger.debug(f"Loading module {module_path} for {possible_type!r}")
                importlib.import_module(module_path)
                _logger.debug(f"Successfully loaded module: {module_path}")
                return
            except Exception as e:
                _logger.debug(f"Failed to load module {module_path}: {e}")
                continue


# Legacy function for backward compatibility
def ensure_modules_loaded() -> None:
    """Legacy function - now just ensures registration index is built."""
    _build_lazy_registration_index()
