# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import tempfile
import threading
import time
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aiperf.module_loader import ModuleRegistry


class MockTestEnum(Enum):
    """Test enum for module loader tests."""

    TEST_TYPE = "test_type"
    ANOTHER_TYPE = "another_type"


class ComplexTestEnum(Enum):
    """Complex test enum for advanced scenarios."""

    NESTED_TYPE = "complex.nested.type"
    SIMPLE_TYPE = "simple"


@pytest.fixture
def mock_import_module():
    """Mock importlib.import_module."""
    with patch("importlib.import_module") as mock:
        yield mock


@pytest.fixture
def mock_file_system():
    """Mock file system operations for scanning tests."""
    with (
        patch("pathlib.Path.rglob") as mock_rglob,
        patch("pathlib.Path.read_text") as mock_read_text,
        patch("ast.parse") as mock_parse,
        patch("ast.walk") as mock_walk,
    ):
        yield {
            "rglob": mock_rglob,
            "read_text": mock_read_text,
            "parse": mock_parse,
            "walk": mock_walk,
        }


@pytest.fixture
def temp_python_file():
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


@pytest.fixture
def loaded_registry():
    """Create a registry instance with pre-loaded state."""
    # Registry is automatically loaded on instantiation now
    registry = ModuleRegistry()
    return registry


@pytest.fixture
def sample_registrations():
    """Sample registration data for testing."""
    return {
        "TestFactory": {
            "test_type": "test.module.path",
            "MockTestEnum.TEST_TYPE": "test.enum.module",
        },
        "ComplexFactory": {
            "ComplexTestEnum.NESTED_TYPE": "complex.module.path",
            "ComplexTestEnum.SIMPLE_TYPE": "simple.module.path",
        },
    }


def create_mock_file(name: str, path: str, content: str | None = None) -> MagicMock:
    """Helper to create mock file objects for testing."""
    mock_file = MagicMock()
    mock_file.name = name
    mock_file.__str__ = MagicMock(return_value=path)
    if content:
        mock_file.read_text.return_value = content
    # Mock relative_to chain for module path generation
    mock_file.relative_to.return_value.with_suffix.return_value.parts = tuple(
        path.replace(".py", "").split("/")[-1:]
    )
    return mock_file


def parse_decorators_from_content(
    registry: ModuleRegistry, content: str, module_path: str
):
    """Helper to parse decorators from content and register them."""
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                registry._parse_decorator(decorator, module_path)


def run_concurrent_operations(operation_func, num_threads: int = 10, *args, **kwargs):
    """Helper to run operations concurrently and collect results."""
    results = []

    def worker():
        result = operation_func(*args, **kwargs)
        results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return results


class TestModuleRegistry:
    """Comprehensive test suite for ModuleRegistry."""

    def test_singleton_pattern(self):
        """Test that ModuleRegistry follows singleton pattern."""
        registry1 = ModuleRegistry()
        registry2 = ModuleRegistry()

        assert registry1 is registry2
        assert id(registry1) == id(registry2)

    def test_singleton_thread_safety(self):
        """Test singleton creation is thread-safe."""
        instances = run_concurrent_operations(ModuleRegistry, num_threads=10)

        # All instances should be the same
        assert len(instances) == 10
        first_instance = instances[0]
        for instance in instances:
            assert instance is first_instance

    @pytest.mark.parametrize(
        "content,module_path,expected_registrations",
        [
            # Simple factory register
            (
                """
from some_module import Factory, SomeEnum

@Factory.register(SomeEnum.TYPE_A)
class TestClass:
    pass
""",
                "test.module",
                {"Factory": {"SomeEnum.TYPE_A": "test.module"}},
            ),
            # Multiple arguments
            (
                """
from some_module import TestFactory, MyEnum

@TestFactory.register(MyEnum.TYPE_A, MyEnum.TYPE_B)
class MultiTypeClass:
    pass
""",
                "test.multi",
                {
                    "TestFactory": {
                        "MyEnum.TYPE_A": "test.multi",
                        "MyEnum.TYPE_B": "test.multi",
                    }
                },
            ),
            # Register all
            (
                """
from some_module import MyFactory, ConfigEnum

@MyFactory.register_all(ConfigEnum.SETTING_A)
class ConfigClass:
    pass
""",
                "test.config",
                {"MyFactory": {"ConfigEnum.SETTING_A": "test.config"}},
            ),
            # Non-Factory decorator (should still register if ends with Factory)
            (
                """
from some_module import NotAFactory, SomeEnum

@NotAFactory.register(SomeEnum.TYPE_A)
class TestClass:
    pass
""",
                "test.ignore",
                {"NotAFactory": {"SomeEnum.TYPE_A": "test.ignore"}},
            ),
            # Invalid method (should not register)
            (
                """
from some_module import SomeFactory, TestEnum

@SomeFactory.invalid_method(TestEnum.TYPE_A)
class TestClass:
    pass
""",
                "test.invalid",
                {},
            ),
        ],
    )
    def test_parse_decorator_variations(
        self, content, module_path, expected_registrations
    ):
        """Test parsing various decorator patterns."""
        registry = ModuleRegistry()
        # Clear any existing registrations from the real scan
        registry._registrations.clear()
        parse_decorators_from_content(registry, content, module_path)
        assert registry._registrations == expected_registrations

    @pytest.mark.parametrize(
        "factory_name,plugin_type,registrations,expected_module,should_import",
        [
            # String type
            (
                "TestFactory",
                "test_type",
                {"TestFactory": {"test_type": "test.module.path"}},
                "test.module.path",
                True,
            ),
            # Enum type
            (
                "TestFactory",
                MockTestEnum.TEST_TYPE,
                {"TestFactory": {"MockTestEnum.TEST_TYPE": "test.enum.module"}},
                "test.enum.module",
                True,
            ),
            # Not found
            (
                "NonExistentFactory",
                "non_existent_type",
                {},
                None,
                False,
            ),
            # Complex enum
            (
                "ComplexFactory",
                ComplexTestEnum.NESTED_TYPE,
                {
                    "ComplexFactory": {
                        "ComplexTestEnum.NESTED_TYPE": "complex.module.path"
                    }
                },
                "complex.module.path",
                True,
            ),
        ],
    )
    def test_load_plugin_variations(
        self,
        mock_import_module,
        loaded_registry,
        factory_name,
        plugin_type,
        registrations,
        expected_module,
        should_import,
    ):
        """Test loading plugins with various types and scenarios."""
        loaded_registry._registrations = registrations
        loaded_registry.load_plugin(factory_name, plugin_type)

        if should_import:
            mock_import_module.assert_called_once_with(expected_module)
        else:
            mock_import_module.assert_not_called()

    def test_automatic_scan_on_instantiation(self, mock_file_system):
        """Test that scan happens automatically on first instantiation."""
        # Reset singleton for this test
        ModuleRegistry._instance = None
        ModuleRegistry._loaded = False

        mock_file_system["rglob"].return_value = []

        # First instantiation should trigger scan
        registry = ModuleRegistry()

        # Scan should have happened automatically
        assert registry._loaded
        mock_file_system["rglob"].assert_called_once()

    def test_scan_all_sets_loaded_flag(self, mock_file_system):
        """Test that _scan_all sets the loaded flag."""
        # Reset singleton to test _scan_all directly
        ModuleRegistry._instance = None
        ModuleRegistry._loaded = False

        mock_file_system["rglob"].return_value = []  # No files to scan

        # Create instance (which triggers scan automatically)
        registry = ModuleRegistry()

        # Should be loaded after instantiation
        assert registry._loaded

    def test_singleton_instantiation_thread_safety(self):
        """Test that singleton instantiation is thread-safe and scan only runs once."""
        # Reset singleton for this test
        ModuleRegistry._instance = None
        ModuleRegistry._loaded = False

        scan_count = 0
        original_scan = ModuleRegistry._scan_all

        def counting_scan(self):
            nonlocal scan_count
            scan_count += 1
            time.sleep(0.01)  # Small delay to increase chance of race condition
            # Call original scan logic
            original_scan(self)

        # Patch _scan_all to count calls
        with patch.object(ModuleRegistry, "_scan_all", counting_scan):
            # Multiple threads trying to create instances
            instances = run_concurrent_operations(ModuleRegistry, num_threads=10)

        # All instances should be the same
        assert len(instances) == 10
        first_instance = instances[0]
        for instance in instances:
            assert instance is first_instance

        # _scan_all should only be called once due to singleton pattern
        assert scan_count == 1

    def test_scan_all_handles_parse_errors(self, mock_file_system):
        """Test that _scan_all handles parse errors gracefully."""
        # Reset singleton state and set up mocks before creating registry
        ModuleRegistry._instance = None
        ModuleRegistry._loaded = False

        # Create mock file that will cause parse error
        mock_file = create_mock_file("bad_syntax.py", "bad_syntax.py")
        mock_file_system["rglob"].return_value = [mock_file]
        mock_file.read_text.side_effect = SyntaxError("Invalid syntax")

        # Create registry - should not raise exception even with parse error
        registry = ModuleRegistry()
        assert registry._loaded

    def test_scan_all_skips_cache_and_self(self, mock_file_system):
        """Test that _scan_all skips __pycache__ and module_loader.py."""
        # Reset singleton state and set up mocks before creating registry
        ModuleRegistry._instance = None
        ModuleRegistry._loaded = False

        # Create mock files
        cache_file = create_mock_file("cached.py", "some/path/__pycache__/cached.py")
        self_file = create_mock_file("module_loader.py", "aiperf/module_loader.py")
        valid_file = create_mock_file(
            "valid.py", "aiperf/valid.py", "# valid python content"
        )

        mock_file_system["rglob"].return_value = [cache_file, self_file, valid_file]
        mock_file_system["parse"].return_value = MagicMock()

        # Create registry - this will automatically trigger _scan_all() with our mocks
        _ = ModuleRegistry()

        # Should only try to parse the valid file
        assert mock_file_system["parse"].call_count == 1

    def test_concurrent_load_plugin_calls(self, mock_import_module, loaded_registry):
        """Test concurrent load_plugin calls are handled safely."""
        loaded_registry._registrations = {
            "TestFactory": {
                "type1": "module1",
                "type2": "module2",
                "type3": "module3",
            }
        }

        def load_plugin_worker():
            # Use different plugin types in rotation
            import random

            plugin_type = f"type{random.randint(1, 3)}"
            loaded_registry.load_plugin("TestFactory", plugin_type)
            return True

        # Run concurrent operations
        results = run_concurrent_operations(load_plugin_worker, num_threads=10)

        # All should have completed successfully
        assert len(results) == 10
        assert all(results)

    def test_registrations_data_structure_integrity(self):
        """Test that registrations data structure maintains integrity under concurrent access."""
        registry = ModuleRegistry()

        def modify_registrations():
            import random

            i = random.randint(0, 49)
            factory = f"TestFactory{i % 5}"
            type_name = f"Type{i % 10}"
            module = f"module.{i}"

            if factory not in registry._registrations:
                registry._registrations[factory] = {}
            registry._registrations[factory][type_name] = module
            return True

        # Simulate concurrent modifications
        results = run_concurrent_operations(modify_registrations, num_threads=20)

        # Verify data structure integrity
        assert len(results) == 20
        assert isinstance(registry._registrations, dict)
        # Check that our test factories were created correctly
        test_factories = {
            k: v
            for k, v in registry._registrations.items()
            if k.startswith("TestFactory")
        }
        assert len(test_factories) > 0  # Should have at least some test factories
        for factory_name, types_dict in test_factories.items():
            assert isinstance(types_dict, dict)
            assert factory_name.startswith("TestFactory")

    @pytest.mark.parametrize("num_threads", [2, 5, 10])
    def test_multiple_registry_instances_thread_safety(self, num_threads):
        """Test creating multiple registry instances across threads."""
        instances = run_concurrent_operations(ModuleRegistry, num_threads=num_threads)

        # All instances should be the same
        assert len(instances) == num_threads
        first_instance = instances[0]
        for instance in instances:
            assert instance is first_instance

    @pytest.mark.parametrize(
        "factory_name,plugin_type,registrations",
        [
            # Empty registrations
            ("AnyFactory", "any_type", {}),
            # None factory name
            (None, "valid_type", {"TestFactory": {"valid_type": "valid.module"}}),
            # None plugin type
            ("TestFactory", None, {"TestFactory": {"valid_type": "valid.module"}}),
        ],
    )
    def test_edge_cases(
        self,
        mock_import_module,
        loaded_registry,
        factory_name,
        plugin_type,
        registrations,
    ):
        """Test edge cases that should not trigger imports."""
        loaded_registry._registrations = registrations
        loaded_registry.load_plugin(factory_name, plugin_type)
        mock_import_module.assert_not_called()

    def test_scan_all_comprehensive_integration(self, mock_file_system):
        """Test complete _scan_all integration with mocked file system."""
        # Reset singleton state and set up mocks before creating registry
        ModuleRegistry._instance = None
        ModuleRegistry._loaded = False

        # Mock file structure
        mock_file1 = create_mock_file(
            "test1.py", "aiperf/test1.py", "# Mock file content 1"
        )
        mock_file2 = create_mock_file(
            "test2.py", "aiperf/subdir/test2.py", "# Mock file content 2"
        )

        mock_file_system["rglob"].return_value = [mock_file1, mock_file2]

        # Mock AST parsing - need to provide all required AST node attributes
        mock_class_node = MagicMock(spec=ast.ClassDef)
        mock_class_node.decorator_list = []
        # Add required AST node attributes that pytest might access
        mock_class_node.lineno = 1
        mock_class_node.col_offset = 0
        mock_class_node.end_lineno = 2
        mock_class_node.end_col_offset = 0

        mock_tree = MagicMock()
        mock_file_system["parse"].return_value = mock_tree
        mock_file_system["walk"].return_value = [mock_class_node]

        # Now create registry - this will automatically trigger _scan_all() with our mocks
        registry = ModuleRegistry()

        assert registry._loaded
        assert mock_file_system["parse"].call_count >= 2  # At least two files parsed

    def test_memory_efficiency_large_registrations(
        self, mock_import_module, loaded_registry
    ):
        """Test memory efficiency with large number of registrations."""
        # Create a large number of registrations
        num_factories = 100
        num_types_per_factory = 50

        for factory_idx in range(num_factories):
            factory_name = f"Factory{factory_idx}"
            loaded_registry._registrations[factory_name] = {}

            for type_idx in range(num_types_per_factory):
                type_name = f"Type{type_idx}"
                module_path = f"module.factory{factory_idx}.type{type_idx}"
                loaded_registry._registrations[factory_name][type_name] = module_path

        # Test that lookups still work efficiently
        loaded_registry.load_plugin("Factory50", "Type25")
        mock_import_module.assert_called_once_with("module.factory50.type25")

    def test_class_decorator_variations(self):
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
        # Clear any existing registrations from the real scan
        registry._registrations.clear()
        module_path = "test.variations"
        parse_decorators_from_content(registry, content, module_path)

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

    def test_get_available_types(self, loaded_registry, sample_registrations):
        """Test getting available types for a factory."""
        loaded_registry._registrations = sample_registrations

        available_types = loaded_registry.get_available_types("TestFactory")
        expected_types = ["test_type", "MockTestEnum.TEST_TYPE"]

        assert set(available_types) == set(expected_types)

    def test_get_available_types_empty_factory(self, loaded_registry):
        """Test getting available types for non-existent factory."""
        available_types = loaded_registry.get_available_types("NonExistentFactory")
        assert available_types == []

    def test_get_all_factories(self, loaded_registry, sample_registrations):
        """Test getting all factory names."""
        loaded_registry._registrations = sample_registrations

        factories = loaded_registry.get_all_factories()
        expected_factories = ["TestFactory", "ComplexFactory"]

        assert set(factories) == set(expected_factories)

    def test_load_all_plugins(self, mock_import_module, loaded_registry):
        """Test loading all plugins for a factory."""
        loaded_registry._registrations = {
            "TestFactory": {
                "type1": "module1",
                "type2": "module2",
                "type3": "module3",
            }
        }

        loaded_registry.load_all_plugins("TestFactory")

        # Should import all three modules
        assert mock_import_module.call_count == 3
        expected_calls = ["module1", "module2", "module3"]
        actual_calls = [call.args[0] for call in mock_import_module.call_args_list]
        assert set(actual_calls) == set(expected_calls)

    def test_load_all_plugins_empty_factory(self, mock_import_module, loaded_registry):
        """Test loading all plugins for non-existent factory."""
        loaded_registry.load_all_plugins("NonExistentFactory")
        mock_import_module.assert_not_called()
