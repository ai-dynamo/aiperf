# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module loader for AIPerf.

This module recursively scans Python files for registration headers and provides
lazy loading of modules only when their registered classes are needed. This avoids
the performance penalty of importing all modules during CLI startup while ensuring
implementations are available when requested.

Registration Header Format:
Files can include special comments that declare their factory registrations:
# AIPERF_REGISTER: ServiceFactory.register(ServiceType.WORKER)
# AIPERF_REGISTER: InferenceClientFactory.register_all(EndpointType.CHAT, EndpointType.COMPLETIONS)
"""

import ast
import importlib
import re
import threading
from pathlib import Path
from typing import Any

from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)


# Registration information storage
class RegistrationInfo:
    """Information about a factory registration found in a Python file."""

    def __init__(
        self,
        factory_name: str,
        method_name: str,
        class_types: list[str],
        module_path: str,
        file_path: Path,
        override_priority: int = 0,
    ):
        self.factory_name = factory_name
        self.method_name = method_name  # 'register' or 'register_all'
        self.class_types = class_types
        self.module_path = module_path
        self.file_path = file_path
        self.override_priority = override_priority


# Global registry for lazy registrations
_lazy_registrations: dict[str, dict[str, list[RegistrationInfo]]] = {}
_registrations_loaded = False
_registrations_lock = threading.Lock()
_plugin_discovery_lock = threading.Lock()
_discovered_plugins: set[tuple[str, str]] = set()


def _parse_registration_header(line: str) -> RegistrationInfo | None:
    """Parse a registration header comment to extract registration information.

    Expected formats:
    # AIPERF_REGISTER: ServiceFactory.register(ServiceType.WORKER)
    # AIPERF_REGISTER: InferenceClientFactory.register_all(EndpointType.CHAT, EndpointType.COMPLETIONS)
    # AIPERF_REGISTER: ServiceFactory.register(ServiceType.WORKER, override_priority=1)
    """
    # Match the registration header pattern
    match = re.match(r"#\s*AIPERF_REGISTER:\s*(\w+)\.(\w+)\((.*)\)", line.strip())
    if not match:
        return None

    factory_name = match.group(1)
    method_name = match.group(2)
    args_str = match.group(3).strip()

    if method_name not in ["register", "register_all"]:
        return None

    # Parse the arguments
    class_types = []
    override_priority = 0

    if args_str:
        # Simple parsing - split by comma and clean up
        parts = [part.strip() for part in args_str.split(",")]
        for part in parts:
            if part.startswith("override_priority="):
                try:
                    override_priority = int(part.split("=")[1])
                except (ValueError, IndexError):
                    pass
            elif part and not part.startswith("override_priority="):
                class_types.append(part)

    return RegistrationInfo(
        factory_name=factory_name,
        method_name=method_name,
        class_types=class_types,
        module_path="",  # Will be set later
        file_path=Path(),  # Will be set later
        override_priority=override_priority,
    )


def _parse_decorator_registration(
    file_content: str, file_path: Path
) -> list[RegistrationInfo]:
    """Parse Python file content to extract factory registrations from decorators.

    This looks for patterns like:
    @ServiceFactory.register(ServiceType.WORKER)
    @InferenceClientFactory.register_all(EndpointType.CHAT, EndpointType.COMPLETIONS)
    @ResponseExtractorFactory.register_all(
        EndpointType.CHAT,
        EndpointType.COMPLETIONS,
        EndpointType.EMBEDDINGS,
    )

    Only actual decorators are parsed, not examples in docstrings or comments.
    """
    registrations = []

    try:
        # Parse the file as AST to properly handle multi-line decorators
        # This automatically excludes decorators in docstrings and comments
        tree = ast.parse(file_content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Look at decorators for this class
                for decorator in node.decorator_list:
                    reg_info = _parse_decorator_node(decorator, file_path)
                    if reg_info:
                        registrations.append(reg_info)

    except SyntaxError as e:
        # If AST parsing fails, fall back to regex-based parsing
        _logger.debug(f"AST parsing failed for {file_path}, falling back to regex: {e}")
        registrations.extend(
            _parse_decorator_registration_regex(file_content, file_path)
        )
    except Exception as e:
        _logger.debug(f"Error parsing decorators in {file_path}: {e}")
        # Don't use regex fallback for general exceptions

    return registrations


def _parse_decorator_node(
    decorator: ast.expr, file_path: Path
) -> RegistrationInfo | None:
    """Parse a decorator AST node to extract factory registration information."""
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

    # Extract arguments
    class_types = []
    override_priority = 0

    # Process positional arguments
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
            except:
                class_types.append(str(arg))

    # Process keyword arguments
    for keyword in decorator.keywords:
        if keyword.arg == "override_priority":
            if isinstance(keyword.value, ast.Constant) and isinstance(
                keyword.value.value, int
            ):
                override_priority = keyword.value.value

    module_path = _file_path_to_module_path(file_path)

    return RegistrationInfo(
        factory_name=factory_name,
        method_name=method_name,
        class_types=class_types,
        module_path=module_path,
        file_path=file_path,
        override_priority=override_priority,
    )


def _parse_decorator_registration_regex(
    file_content: str, file_path: Path
) -> list[RegistrationInfo]:
    """Fallback regex-based parsing for decorator registrations."""
    registrations = []

    # Pattern to match factory registration decorators (handles multi-line with non-greedy matching)
    decorator_pattern = re.compile(
        r"@(\w+)\.(\w+)\((.*?)\)(?=\s*(?:@|\w|\s*class|\s*def|\s*$))",
        re.MULTILINE | re.DOTALL,
    )

    for match in decorator_pattern.finditer(file_content):
        factory_name = match.group(1)
        method_name = match.group(2)
        args_str = match.group(3).strip()

        # Only process factory registrations
        if not factory_name.endswith("Factory") or method_name not in [
            "register",
            "register_all",
        ]:
            continue

        # Parse arguments
        class_types = []
        override_priority = 0

        if args_str:
            # Clean up the arguments string - remove newlines and extra whitespace
            args_str = re.sub(r"\s+", " ", args_str)

            # Simple parsing - split by comma and clean up
            parts = [part.strip() for part in args_str.split(",")]
            for part in parts:
                if "override_priority=" in part:
                    try:
                        override_priority = int(part.split("=")[1])
                    except (ValueError, IndexError):
                        pass
                elif part and "override_priority=" not in part:
                    class_types.append(part)

        module_path = _file_path_to_module_path(file_path)

        registration = RegistrationInfo(
            factory_name=factory_name,
            method_name=method_name,
            class_types=class_types,
            module_path=module_path,
            file_path=file_path,
            override_priority=override_priority,
        )
        registrations.append(registration)

    return registrations


def _file_path_to_module_path(file_path: Path) -> str:
    """Convert a file path to a Python module path."""
    # Get relative path from aiperf package root
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


def _scan_directory_for_registrations(directory: Path) -> list[RegistrationInfo]:
    """Recursively scan a directory for Python files with factory registrations."""
    registrations = []

    for file_path in directory.rglob("*.py"):
        # Skip __pycache__ and other non-source directories
        if "__pycache__" in file_path.parts or file_path.name.startswith("."):
            continue

        # Skip the module_loader.py file itself to avoid picking up examples in docstrings
        if file_path.name == "module_loader.py":
            continue

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # First, check for registration headers
            for line_num, line in enumerate(content.splitlines(), 1):
                if "# AIPERF_REGISTER:" in line:
                    reg_info = _parse_registration_header(line)
                    if reg_info:
                        reg_info.module_path = _file_path_to_module_path(file_path)
                        reg_info.file_path = file_path
                        registrations.append(reg_info)

            # Then, parse decorator registrations
            decorator_registrations = _parse_decorator_registration(content, file_path)
            registrations.extend(decorator_registrations)

        except Exception as e:
            _logger.debug(f"Error scanning file {file_path}: {e}")
            continue

    return registrations


def _build_lazy_registration_index() -> None:
    """Build an index of all factory registrations without importing modules."""
    global _lazy_registrations, _registrations_loaded

    with _registrations_lock:
        if _registrations_loaded:
            return

        _logger.debug("Building lazy registration index...")

        # Scan the entire aiperf package
        aiperf_root = Path(__file__).parent
        all_registrations = _scan_directory_for_registrations(aiperf_root)

        # Organize by factory and class type
        _lazy_registrations.clear()
        for reg_info in all_registrations:
            if reg_info.factory_name not in _lazy_registrations:
                _lazy_registrations[reg_info.factory_name] = {}

            for class_type in reg_info.class_types:
                if class_type not in _lazy_registrations[reg_info.factory_name]:
                    _lazy_registrations[reg_info.factory_name][class_type] = []

                _lazy_registrations[reg_info.factory_name][class_type].append(reg_info)

        _registrations_loaded = True
        _logger.debug(
            f"Lazy registration index built with {len(all_registrations)} registrations"
        )


def ensure_registrations_indexed() -> None:
    """Ensure the lazy registration index is built."""
    _build_lazy_registration_index()


def get_lazy_registrations_for_factory(
    factory_name: str,
) -> dict[str, list[RegistrationInfo]]:
    """Get all lazy registrations for a specific factory."""
    ensure_registrations_indexed()
    return _lazy_registrations.get(factory_name, {})


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

    # Ensure the lazy registration index is built
    ensure_registrations_indexed()

    # Look for lazy registrations for this factory and class type
    factory_registrations = _lazy_registrations.get(factory_name, {})

    # Create a list of possible class type representations to match against
    possible_class_types = [class_type_str]

    # If it's an enum, also try the full enum representation
    if hasattr(class_type, "__class__") and hasattr(class_type.__class__, "__name__"):
        if "." not in class_type_str:  # If it's just the enum value (e.g., "worker")
            enum_class_name = class_type.__class__.__name__
            full_enum_repr = f"{enum_class_name}.{class_type.name}"
            possible_class_types.append(full_enum_repr)

    # Also try the repr() version which might be different
    try:
        repr_version = repr(class_type)
        if repr_version not in possible_class_types:
            possible_class_types.append(repr_version)
    except:
        pass

    _logger.debug(f"Trying class type variations: {possible_class_types}")

    # Try exact matches for all possible representations
    for possible_type in possible_class_types:
        if possible_type in factory_registrations:
            registrations = factory_registrations[possible_type]
            for reg_info in registrations:
                try:
                    _logger.debug(
                        f"Loading module {reg_info.module_path} for {possible_type!r}"
                    )
                    importlib.import_module(reg_info.module_path)
                    _logger.debug(f"Successfully loaded module: {reg_info.module_path}")
                    return  # Success, no need to try other registrations
                except ImportError as e:
                    _logger.debug(f"Failed to load module {reg_info.module_path}: {e}")
                    continue
                except Exception as e:
                    _logger.warning(f"Error loading module {reg_info.module_path}: {e}")
                    continue

    # If no exact match, try partial matches (for enum values)
    for registered_type, registrations in factory_registrations.items():
        # Check if any of our possible types match the registered type
        for possible_type in possible_class_types:
            if (
                possible_type in registered_type
                or registered_type in possible_type
                or _enum_values_match(possible_type, registered_type)
            ):
                for reg_info in registrations:
                    try:
                        _logger.debug(
                            f"Loading module {reg_info.module_path} for potential match {registered_type}"
                        )
                        importlib.import_module(reg_info.module_path)
                        _logger.debug(
                            f"Successfully loaded module: {reg_info.module_path}"
                        )
                        return  # Success
                    except ImportError as e:
                        _logger.debug(
                            f"Failed to load module {reg_info.module_path}: {e}"
                        )
                        continue
                    except Exception as e:
                        _logger.warning(
                            f"Error loading module {reg_info.module_path}: {e}"
                        )
                        continue
                break  # Found a match for this registered_type, no need to check other possible_types

    _logger.debug(
        f"No lazy registration found for {class_type_str!r} in {factory_name}"
    )


def _enum_values_match(type1: str, type2: str) -> bool:
    """Check if two enum type strings represent the same enum value."""
    # Handle cases like "ServiceType.WORKER" vs "worker"
    if "." in type1 and "." not in type2:
        enum_value = type1.split(".")[-1].lower()
        return enum_value == type2.lower()
    elif "." in type2 and "." not in type1:
        enum_value = type2.split(".")[-1].lower()
        return enum_value == type1.lower()
    return False


# Legacy function for backward compatibility
def ensure_modules_loaded() -> None:
    """Legacy function - now just ensures registration index is built."""
    ensure_registrations_indexed()
