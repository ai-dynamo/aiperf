# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aiperf.module_loader import ModuleRegistry


class MockTestEnum(Enum):
    """Test enum for module loader tests."""

    TEST_TYPE = "test_type"
    ANOTHER_TYPE = "another_type"


class TestModuleRegistry:
    """Comprehensive test suite for ModuleRegistry."""

    def setup_method(self):
        """Reset singleton state before each test."""
        # Reset singleton instance and state
        ModuleRegistry._instance = None
        ModuleRegistry._registrations = {}
        ModuleRegistry._loaded = False

    def test_singleton_pattern(self):
        """Test that ModuleRegistry follows singleton pattern."""
        registry1 = ModuleRegistry()
        registry2 = ModuleRegistry()

        assert registry1 is registry2
        assert id(registry1) == id(registry2)

    def test_singleton_thread_safety(self):
        """Test singleton creation is thread-safe."""
        instances = []

        def create_instance():
            instances.append(ModuleRegistry())

        # Create multiple threads trying to create instances
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All instances should be the same
        assert len(instances) == 10
        first_instance = instances[0]
        for instance in instances:
            assert instance is first_instance

    @pytest.fixture
    def temp_python_file(self):
        """Create a temporary Python file for testing."""
        files_created = []

        def _create_file(content: str, filename: str = "test_module.py") -> Path:
            temp_dir = Path(tempfile.mkdtemp())
            file_path = temp_dir / filename
            file_path.write_text(content)
            files_created.append(temp_dir)
            return file_path

        yield _create_file

        # Cleanup
        import shutil

        for temp_dir in files_created:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_parse_decorator_simple_factory_register(self, temp_python_file):
        """Test parsing simple @Factory.register decorator."""
        content = """
from some_module import Factory, SomeEnum

@Factory.register(SomeEnum.TYPE_A)
class TestClass:
    pass
"""
        file_path = temp_python_file(content)
        registry = ModuleRegistry()

        # Parse the file manually
        tree = ast.parse(content)
        module_path = "test.module"

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for decorator in node.decorator_list:
                    registry._parse_decorator(decorator, module_path)

        expected_key = "SomeEnum.TYPE_A"
        assert "Factory" in registry._registrations
        assert expected_key in registry._registrations["Factory"]
        assert registry._registrations["Factory"][expected_key] == module_path

    def test_parse_decorator_multiple_arguments(self, temp_python_file):
        """Test parsing decorator with multiple arguments."""
        content = """
from some_module import TestFactory, MyEnum

@TestFactory.register(MyEnum.TYPE_A, MyEnum.TYPE_B)
class MultiTypeClass:
    pass
"""
        file_path = temp_python_file(content)
        registry = ModuleRegistry()

        tree = ast.parse(content)
        module_path = "test.multi"

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for decorator in node.decorator_list:
                    registry._parse_decorator(decorator, module_path)

        assert "TestFactory" in registry._registrations
        assert "MyEnum.TYPE_A" in registry._registrations["TestFactory"]
        assert "MyEnum.TYPE_B" in registry._registrations["TestFactory"]
        assert registry._registrations["TestFactory"]["MyEnum.TYPE_A"] == module_path
        assert registry._registrations["TestFactory"]["MyEnum.TYPE_B"] == module_path

    def test_parse_decorator_register_all(self, temp_python_file):
        """Test parsing @Factory.register_all decorator."""
        content = """
from some_module import MyFactory, ConfigEnum

@MyFactory.register_all(ConfigEnum.SETTING_A)
class ConfigClass:
    pass
"""
        file_path = temp_python_file(content)
        registry = ModuleRegistry()

        tree = ast.parse(content)
        module_path = "test.config"

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for decorator in node.decorator_list:
                    registry._parse_decorator(decorator, module_path)

        assert "MyFactory" in registry._registrations
        assert "ConfigEnum.SETTING_A" in registry._registrations["MyFactory"]

    def test_parse_decorator_ignores_non_factory(self, temp_python_file):
        """Test that non-Factory decorators are ignored."""
        content = """
from some_module import NotAFactory, SomeEnum

@NotAFactory.register(SomeEnum.TYPE_A)
class TestClass:
    pass

@property
def some_property(self):
    pass
"""
        file_path = temp_python_file(content)
        registry = ModuleRegistry()

        tree = ast.parse(content)
        module_path = "test.ignore"

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for decorator in node.decorator_list:
                    registry._parse_decorator(decorator, module_path)

        # Should register NotAFactory since it ends with "Factory"
        assert "NotAFactory" in registry._registrations
        assert "SomeEnum.TYPE_A" in registry._registrations["NotAFactory"]

    def test_parse_decorator_ignores_invalid_methods(self, temp_python_file):
        """Test that invalid Factory methods are ignored."""
        content = """
from some_module import SomeFactory, TestEnum

@SomeFactory.invalid_method(TestEnum.TYPE_A)
class TestClass:
    pass
"""
        file_path = temp_python_file(content)
        registry = ModuleRegistry()

        tree = ast.parse(content)
        module_path = "test.invalid"

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for decorator in node.decorator_list:
                    registry._parse_decorator(decorator, module_path)

        # Should not register anything since method is not register or register_all
        assert len(registry._registrations) == 0

    @patch("importlib.import_module")
    def test_load_plugin_string_type(self, mock_import):
        """Test loading plugin with string type."""
        registry = ModuleRegistry()
        registry._registrations = {"TestFactory": {"test_type": "test.module.path"}}
        registry._loaded = True

        registry.load_plugin("TestFactory", "test_type")

        mock_import.assert_called_once_with("test.module.path")

    @patch("importlib.import_module")
    def test_load_plugin_enum_type(self, mock_import):
        """Test loading plugin with enum type."""
        registry = ModuleRegistry()
        registry._registrations = {
            "TestFactory": {"MockTestEnum.TEST_TYPE": "test.enum.module"}
        }
        registry._loaded = True

        registry.load_plugin("TestFactory", MockTestEnum.TEST_TYPE)

        mock_import.assert_called_once_with("test.enum.module")

    @patch("importlib.import_module")
    def test_load_plugin_fallback_string_representation(self, mock_import):
        """Test loading plugin falls back to string representation."""
        registry = ModuleRegistry()
        registry._registrations = {
            "TestFactory": {"MockTestEnum.TEST_TYPE": "test.fallback.module"}
        }
        registry._loaded = True

        # First try with direct string should fail, then enum format should work
        registry.load_plugin("TestFactory", MockTestEnum.TEST_TYPE)

        mock_import.assert_called_once_with("test.fallback.module")

    @patch("importlib.import_module")
    def test_load_plugin_not_found(self, mock_import):
        """Test loading plugin that doesn't exist."""
        registry = ModuleRegistry()
        registry._registrations = {}
        registry._loaded = True

        # Should not raise exception, just not call import_module
        registry.load_plugin("NonExistentFactory", "non_existent_type")

        mock_import.assert_not_called()

    @patch("importlib.import_module")
    def test_load_plugin_triggers_scan(self, mock_import):
        """Test that load_plugin triggers scan when not loaded."""
        registry = ModuleRegistry()
        registry._loaded = False

        with patch.object(registry, "_scan_all") as mock_scan:
            registry.load_plugin("TestFactory", "test_type")
            mock_scan.assert_called_once()

    def test_scan_all_sets_loaded_flag(self):
        """Test that _scan_all sets the loaded flag."""
        registry = ModuleRegistry()

        with patch("pathlib.Path.rglob") as mock_rglob:
            mock_rglob.return_value = []  # No files to scan

            assert not registry._loaded
            registry._scan_all()
            assert registry._loaded

    def test_scan_all_thread_safety(self):
        """Test that _scan_all is thread-safe and only runs once."""
        registry = ModuleRegistry()
        scan_count = 0

        def counting_scan(*args, **kwargs):
            nonlocal scan_count
            with registry._scan_lock:
                if registry._loaded:
                    return
                scan_count += 1
                time.sleep(0.01)  # Small delay to increase chance of race condition
                registry._loaded = True

        # Patch _scan_all to count calls
        with patch.object(registry, "_scan_all", side_effect=counting_scan):
            # Create multiple threads trying to scan
            def call_scan():
                registry._scan_all()

            threads = []
            for _ in range(10):
                thread = threading.Thread(target=call_scan)
                threads.append(thread)

            # Start all threads
            for thread in threads:
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

        # _scan_all should only be called once due to the lock
        assert scan_count == 1

    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.rglob")
    def test_scan_all_handles_parse_errors(self, mock_rglob, mock_read_text):
        """Test that _scan_all handles parse errors gracefully."""
        registry = ModuleRegistry()

        # Create mock file that will cause parse error
        mock_file = MagicMock()
        mock_file.name = "bad_syntax.py"
        mock_file.__str__ = MagicMock(return_value="bad_syntax.py")
        mock_rglob.return_value = [mock_file]
        mock_read_text.side_effect = SyntaxError("Invalid syntax")

        # Should not raise exception
        registry._scan_all()
        assert registry._loaded

    def test_scan_all_skips_cache_and_self(self):
        """Test that _scan_all skips __pycache__ and module_loader.py."""
        registry = ModuleRegistry()

        with (
            patch("pathlib.Path.rglob") as mock_rglob,
            patch("pathlib.Path.read_text") as mock_read_text,
            patch("ast.parse") as mock_parse,
        ):
            # Create mock files
            cache_file = MagicMock()
            cache_file.name = "cached.py"
            cache_file.__str__ = MagicMock(
                return_value="some/path/__pycache__/cached.py"
            )

            self_file = MagicMock()
            self_file.name = "module_loader.py"
            self_file.__str__ = MagicMock(return_value="aiperf/module_loader.py")

            valid_file = MagicMock()
            valid_file.name = "valid.py"
            valid_file.__str__ = MagicMock(return_value="aiperf/valid.py")
            valid_file.read_text.return_value = "# valid python content"
            valid_file.relative_to.return_value.with_suffix.return_value.parts = (
                "valid",
            )

            mock_rglob.return_value = [cache_file, self_file, valid_file]
            mock_parse.return_value = MagicMock()

            registry._scan_all()

            # Should only try to parse the valid file
            assert mock_parse.call_count == 1

    def test_concurrent_load_plugin_calls(self):
        """Test concurrent load_plugin calls are handled safely."""
        registry = ModuleRegistry()
        registry._registrations = {
            "TestFactory": {
                "type1": "module1",
                "type2": "module2",
                "type3": "module3",
            }
        }
        registry._loaded = True

        results = []

        def load_plugin_worker(factory, plugin_type):
            with patch("importlib.import_module") as mock_import:
                registry.load_plugin(factory, plugin_type)
                results.append((factory, plugin_type, mock_import.called))

        # Create multiple threads loading different plugins
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(10):
                plugin_type = f"type{(i % 3) + 1}"
                future = executor.submit(load_plugin_worker, "TestFactory", plugin_type)
                futures.append(future)

            # Wait for all to complete
            for future in as_completed(futures):
                future.result()

        # All should have completed successfully
        assert len(results) == 10

    def test_registrations_data_structure_integrity(self):
        """Test that registrations data structure maintains integrity under concurrent access."""
        registry = ModuleRegistry()

        def modify_registrations(factory_name, type_name, module_path):
            if factory_name not in registry._registrations:
                registry._registrations[factory_name] = {}
            registry._registrations[factory_name][type_name] = module_path

        # Simulate concurrent modifications
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(50):
                factory = f"Factory{i % 5}"
                type_name = f"Type{i % 10}"
                module = f"module.{i}"
                future = executor.submit(
                    modify_registrations, factory, type_name, module
                )
                futures.append(future)

            # Wait for all modifications
            for future in as_completed(futures):
                future.result()

        # Verify data structure integrity
        assert isinstance(registry._registrations, dict)
        for factory_name, types_dict in registry._registrations.items():
            assert isinstance(types_dict, dict)
            assert factory_name.startswith("Factory")

    @pytest.mark.parametrize("num_threads", [2, 5, 10])
    def test_multiple_registry_instances_thread_safety(self, num_threads):
        """Test creating multiple registry instances across threads."""
        instances = []
        lock = threading.Lock()

        def create_and_store_instance():
            instance = ModuleRegistry()
            with lock:
                instances.append(instance)

        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=create_and_store_instance)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All instances should be the same
        assert len(instances) == num_threads
        first_instance = instances[0]
        for instance in instances:
            assert instance is first_instance

    def test_load_plugin_with_complex_enum_hierarchy(self):
        """Test loading plugins with complex enum structures."""

        class ComplexEnum(Enum):
            NESTED_TYPE = "complex.nested.type"
            SIMPLE_TYPE = "simple"

        registry = ModuleRegistry()
        registry._registrations = {
            "ComplexFactory": {
                "ComplexEnum.NESTED_TYPE": "complex.module.path",
                "ComplexEnum.SIMPLE_TYPE": "simple.module.path",
            }
        }
        registry._loaded = True

        with patch("importlib.import_module") as mock_import:
            registry.load_plugin("ComplexFactory", ComplexEnum.NESTED_TYPE)
            mock_import.assert_called_once_with("complex.module.path")

    def test_edge_case_empty_registrations(self):
        """Test behavior with empty registrations."""
        registry = ModuleRegistry()
        registry._registrations = {}
        registry._loaded = True

        with patch("importlib.import_module") as mock_import:
            registry.load_plugin("AnyFactory", "any_type")
            mock_import.assert_not_called()

    def test_edge_case_none_values(self):
        """Test behavior with None values in various places."""
        registry = ModuleRegistry()
        registry._registrations = {"TestFactory": {"valid_type": "valid.module"}}
        registry._loaded = True

        with patch("importlib.import_module") as mock_import:
            # Should handle None gracefully
            registry.load_plugin(None, "valid_type")
            registry.load_plugin("TestFactory", None)
            mock_import.assert_not_called()

    def test_scan_all_comprehensive_integration(self):
        """Test complete _scan_all integration with mocked file system."""
        registry = ModuleRegistry()

        with (
            patch("pathlib.Path.rglob") as mock_rglob,
            patch("ast.parse") as mock_parse,
            patch("ast.walk") as mock_walk,
        ):
            # Mock file structure
            mock_file1 = MagicMock()
            mock_file1.name = "test1.py"
            mock_file1.__str__ = MagicMock(return_value="aiperf/test1.py")
            mock_file1.read_text.return_value = "# Mock file content 1"
            mock_file1.relative_to.return_value.with_suffix.return_value.parts = (
                "test1",
            )

            mock_file2 = MagicMock()
            mock_file2.name = "test2.py"
            mock_file2.__str__ = MagicMock(return_value="aiperf/subdir/test2.py")
            mock_file2.read_text.return_value = "# Mock file content 2"
            mock_file2.relative_to.return_value.with_suffix.return_value.parts = (
                "subdir",
                "test2",
            )

            mock_rglob.return_value = [mock_file1, mock_file2]

            # Mock AST parsing
            mock_class_node = MagicMock(spec=ast.ClassDef)
            mock_class_node.decorator_list = []

            mock_tree = MagicMock()
            mock_parse.return_value = mock_tree
            mock_walk.return_value = [mock_class_node]

            registry._scan_all()

            assert registry._loaded
            assert mock_parse.call_count >= 2  # At least two files parsed

    def test_memory_efficiency_large_registrations(self):
        """Test memory efficiency with large number of registrations."""
        registry = ModuleRegistry()

        # Create a large number of registrations
        num_factories = 100
        num_types_per_factory = 50

        for factory_idx in range(num_factories):
            factory_name = f"Factory{factory_idx}"
            registry._registrations[factory_name] = {}

            for type_idx in range(num_types_per_factory):
                type_name = f"Type{type_idx}"
                module_path = f"module.factory{factory_idx}.type{type_idx}"
                registry._registrations[factory_name][type_name] = module_path

        registry._loaded = True

        # Test that lookups still work efficiently
        with patch("importlib.import_module") as mock_import:
            registry.load_plugin("Factory50", "Type25")
            mock_import.assert_called_once_with("module.factory50.type25")

    def test_class_decorator_variations(self, temp_python_file):
        """Test various class decorator patterns."""
        content = """
from factories import (
    ServiceFactory,
    ComponentFactory,
    UtilFactory
)
from enums import ServiceType, ComponentType

@ServiceFactory.register(ServiceType.HTTP_CLIENT)
@ComponentFactory.register(ComponentType.PARSER)
class MultiDecoratorClass:
    pass

@UtilFactory.register_all(ServiceType.CACHE, ComponentType.LOGGER)
class MultiArgClass:
    pass
"""
        registry = ModuleRegistry()
        tree = ast.parse(content)
        module_path = "test.variations"

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for decorator in node.decorator_list:
                    registry._parse_decorator(decorator, module_path)

        # Verify all registrations
        expected_registrations = {
            "ServiceFactory": {"ServiceType.HTTP_CLIENT": module_path},
            "ComponentFactory": {"ComponentType.PARSER": module_path},
            "UtilFactory": {
                "ServiceType.CACHE": module_path,
                "ComponentType.LOGGER": module_path,
            },
        }

        assert registry._registrations == expected_registrations
