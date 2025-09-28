# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blazing fast module loader for AIPerf.

Thread-safe singleton that automatically scans for @Factory.register decorators on first
instantiation, then loads modules on demand. The scan happens only once during singleton
creation, ensuring all subsequent operations are fast and don't need to check scan status.
"""

import ast
import importlib
import threading
from enum import Enum
from pathlib import Path


class ModuleRegistry:
    """Thread-safe singleton for lazy module loading.

    Automatically scans all Python files for @Factory.register decorators on first
    instantiation. All subsequent operations are guaranteed to have complete registry
    data available without additional scan checks or locks.
    """

    _instance = None
    _instance_lock = threading.Lock()
    _registrations: dict[
        str, dict[str, str]
    ] = {}  # factory -> {class_type -> module_path}
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # Automatically scan on first instantiation
                    cls._instance._scan_all()
        return cls._instance

    def load_plugin(self, factory_name: str, class_type: str) -> None:
        """Load a plugin module for the given factory and class type."""
        # Try different representations in order of preference
        type_keys_to_try = []

        # If it's an enum, prioritize the string representation
        if hasattr(class_type, "name") and isinstance(class_type, Enum):
            type_keys_to_try.extend(
                [
                    str(class_type),  # String representation (e.g., "none")
                    f"{class_type.__class__.__name__}.{class_type.name}",  # Full format (e.g., "AIPerfUIType.NONE")
                ]
            )
        else:
            # For string inputs, try as-is first, then check if it might be an enum format
            type_keys_to_try.append(str(class_type))

            # If the input looks like an enum format, also try it as-is
            if "." in str(class_type):
                type_keys_to_try.append(str(class_type))

        for type_key in type_keys_to_try:
            module_path = self._registrations.get(factory_name, {}).get(type_key)
            if module_path:
                importlib.import_module(module_path)
                return

    def get_available_types(self, factory_name: str) -> list[str]:
        """Get all available types for a factory without loading them.

        Args:
            factory_name: Name of the factory to get types for

        Returns:
            List of available type strings for the factory
        """
        return list(self._registrations.get(factory_name, {}).keys())

    def get_all_factories(self) -> list[str]:
        """Get all available factory names.

        Returns:
            List of all factory names that have registered implementations
        """
        return list(self._registrations.keys())

    def load_all_plugins(self, factory_name: str) -> None:
        """Load all available plugins for a factory.

        Args:
            factory_name: Name of the factory to load all plugins for
        """
        factory_registrations = self._registrations.get(factory_name, {})

        # Load all modules for this factory
        for module_path in factory_registrations.values():
            importlib.import_module(module_path)

    def _scan_all(self) -> None:
        """Scan all Python files for @Factory.register decorators.

        This method is only called during singleton instantiation, so no additional
        locking is needed as the instance lock already protects this operation.
        """
        if self._loaded:
            return

        aiperf_root = Path(__file__).parent
        for file_path in aiperf_root.rglob("*.py"):
            if "__pycache__" in str(file_path) or file_path.name == "module_loader.py":
                continue

            try:
                tree = ast.parse(file_path.read_text())
                module_path = f"aiperf.{'.'.join(file_path.relative_to(aiperf_root).with_suffix('').parts)}"

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for decorator in node.decorator_list:
                            self._parse_decorator(decorator, module_path)
            except Exception:
                continue

        self._loaded = True

    def _parse_decorator(self, decorator: ast.expr, module_path: str) -> None:
        """Parse @Factory.register() decorator and store registration."""
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Name)
            and decorator.func.value.id.endswith("Factory")
            and decorator.func.attr in ["register", "register_all"]
        ):
            factory_name = decorator.func.value.id

            for arg in decorator.args:
                if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
                    # Store the full enum format for backward compatibility
                    full_enum_format = f"{arg.value.id}.{arg.attr}"

                    if factory_name not in self._registrations:
                        self._registrations[factory_name] = {}

                    # Store the full enum format
                    self._registrations[factory_name][full_enum_format] = module_path

                    # Try to get the actual enum value and store its string representation
                    try:
                        # Dynamically import and get the enum value
                        enum_class_name = arg.value.id
                        enum_value_name = arg.attr

                        # Import the enum class from common.enums
                        import importlib

                        enums_module = importlib.import_module("aiperf.common.enums")

                        if hasattr(enums_module, enum_class_name):
                            enum_class = getattr(enums_module, enum_class_name)
                            if hasattr(enum_class, enum_value_name):
                                enum_value = getattr(enum_class, enum_value_name)
                                # Store the string representation as the primary key
                                string_representation = str(enum_value)
                                self._registrations[factory_name][
                                    string_representation
                                ] = module_path
                    except Exception:
                        # If we can't resolve the enum, just continue with the full format
                        pass
