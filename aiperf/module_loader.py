# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blazing fast module loader for AIPerf.

Simple singleton that scans for @Factory.register decorators and loads modules on demand.
"""

import ast
import importlib
import threading
from enum import Enum
from pathlib import Path


class ModuleRegistry:
    """Thread-safe singleton for lazy module loading."""

    _instance = None
    _instance_lock = threading.Lock()
    _registrations: dict[
        str, dict[str, str]
    ] = {}  # factory -> {class_type -> module_path}
    _loaded = False
    _scan_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def load_plugin(self, factory_name: str, class_type: str) -> None:
        """Load a plugin module for the given factory and class type."""
        if not self._loaded:
            self._scan_all()

        # Try both string representations
        for type_key in [
            str(class_type),
            f"{class_type.__class__.__name__}.{class_type.name}"
            if hasattr(class_type, "name") and isinstance(class_type, Enum)
            else str(class_type),
        ]:
            with self._scan_lock:
                module_path = self._registrations.get(factory_name, {}).get(type_key)
            if module_path:
                importlib.import_module(module_path)
                return

    def _scan_all(self) -> None:
        """Scan all Python files for @Factory.register decorators."""
        with self._scan_lock:
            if self._loaded:
                return

            aiperf_root = Path(__file__).parent
            for file_path in aiperf_root.rglob("*.py"):
                if (
                    "__pycache__" in str(file_path)
                    or file_path.name == "module_loader.py"
                ):
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
                    class_type = f"{arg.value.id}.{arg.attr}"
                    if factory_name not in self._registrations:
                        self._registrations[factory_name] = {}
                    self._registrations[factory_name][class_type] = module_path
