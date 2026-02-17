# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the plugin registry module."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pytest
from pytest import param

from aiperf.plugin.extensible_enums import ExtensibleStrEnum
from aiperf.plugin.plugins import _PluginRegistry
from aiperf.plugin.types import PackageInfo, PluginEntry, PluginError, TypeNotFoundError

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Test Dummy Classes
# =============================================================================


class DummyClass:
    """A dummy class for testing plugin registration."""


class AnotherDummyClass:
    """Another dummy class for testing."""


class HighPriorityClass:
    """A high priority dummy class."""


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def empty_registry() -> Generator[_PluginRegistry, None, None]:
    """Create a fresh registry with mocked discovery."""
    with patch.object(_PluginRegistry, "discover_plugins"):
        yield _PluginRegistry()


@pytest.fixture
def registry_with_types(empty_registry: _PluginRegistry) -> _PluginRegistry:
    """Registry with test types pre-registered."""
    empty_registry.register("test_category", "type_a", DummyClass, priority=0)
    empty_registry.register("test_category", "type_b", AnotherDummyClass, priority=5)
    empty_registry.register("other_category", "type_c", DummyClass, priority=0)
    return empty_registry


@pytest.fixture
def sample_manifest_yaml() -> str:
    """Sample plugins.yaml content for testing."""
    return """
schema_version: "1.0"

plugin:
  name: test-plugin
  version: "1.0.0"

test_category:
  test_type:
    class: tests.unit.plugin.test_plugins:DummyClass
    description: A test plugin type
    priority: 0
    metadata:
      key1: value1
      key2: 42

  another_type:
    class: tests.unit.plugin.test_plugins:AnotherDummyClass
    description: Another test type
    priority: 5
"""


@pytest.fixture
def manifest_path(tmp_path: Path, sample_manifest_yaml: str) -> Path:
    """Create a temporary manifest file."""
    manifest = tmp_path / "plugins.yaml"
    manifest.write_text(sample_manifest_yaml)
    return manifest


# =============================================================================
# Shared Helpers
# =============================================================================


def make_entry(
    category: str = "test",
    name: str = "test",
    package: str = "test",
    class_path: str = "tests.unit.plugin.test_plugins:DummyClass",
    priority: int = 0,
    metadata: dict[str, Any] | None = None,
    loaded_class: type | None = None,
) -> PluginEntry:
    """Create a PluginEntry with sensible defaults."""
    return PluginEntry(
        category=category,
        name=name,
        package=package,
        class_path=class_path,
        priority=priority,
        metadata=metadata or {},
        loaded_class=loaded_class,
    )


def make_mock_dist(
    version: str = "1.0.0",
    author: str | None = None,
    author_email: str | None = None,
    summary: str = "",
    license_: str = "",
    homepage: str = "",
) -> Mock:
    """Create a mock Distribution for testing package metadata."""
    mock_dist = Mock()
    metadata = {
        "Version": version,
        "Summary": summary,
        "License": license_,
        "Home-page": homepage,
    }
    if author:
        metadata["Author"] = author
    if author_email:
        metadata["Author-email"] = author_email
    mock_dist.metadata = metadata
    return mock_dist


@contextmanager
def temp_module(tmp_path: Path, module_name: str, content: str) -> Any:
    """Create a temporary Python module and add it to sys.path."""
    module_dir = tmp_path / module_name
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text(content)
    sys.path.insert(0, str(tmp_path))
    try:
        yield module_dir
    finally:
        sys.path.remove(str(tmp_path))


def write_manifest(tmp_path: Path, content: str) -> Path:
    """Write a manifest YAML file and return its path."""
    manifest = tmp_path / "plugins.yaml"
    manifest.write_text(content)
    return manifest


# =============================================================================
# Registry Initialization Tests
# =============================================================================


class TestRegistryInit:
    """Tests for registry initialization."""

    def test_init_creates_empty_structures(
        self, empty_registry: _PluginRegistry
    ) -> None:
        """Registry initializes with empty data structures."""
        assert empty_registry._types == {}
        assert empty_registry._type_entries_by_class_path == {}
        assert empty_registry._loaded_plugins == {}

    def test_init_calls_discover_plugins(self) -> None:
        """Registry calls discover_plugins on init."""
        with patch.object(_PluginRegistry, "discover_plugins") as mock_discover:
            _PluginRegistry()
            mock_discover.assert_called_once()


# =============================================================================
# Registration Tests
# =============================================================================


class TestRegister:
    """Tests for register() method."""

    def test_register_adds_type(self, empty_registry: _PluginRegistry) -> None:
        """register() adds type to registry."""
        empty_registry.register("category", "name", DummyClass)

        assert empty_registry.has_entry("category", "name")
        entry = empty_registry.get_entry("category", "name")
        assert entry.name == "name"
        assert entry.category == "category"

    @pytest.mark.parametrize(
        ("priority", "metadata"),
        [
            (0, None),
            (10, None),
            (0, {"key": "value"}),
            (5, {"key": "value", "count": 42}),
        ],
    )
    def test_register_with_options(
        self, empty_registry: _PluginRegistry, priority: int, metadata: dict | None
    ) -> None:
        """register() respects priority and metadata parameters."""
        empty_registry.register(
            "category", "name", DummyClass, priority=priority, metadata=metadata
        )
        entry = empty_registry.get_entry("category", "name")
        assert entry.priority == priority
        assert entry.metadata == (metadata or {})

    def test_register_sets_class_path(self, empty_registry: _PluginRegistry) -> None:
        """register() generates correct class_path."""
        empty_registry.register("category", "name", DummyClass)
        entry = empty_registry.get_entry("category", "name")
        assert entry.class_path == f"{DummyClass.__module__}:{DummyClass.__name__}"

    def test_register_preloads_class(self, empty_registry: _PluginRegistry) -> None:
        """register() caches loaded class."""
        empty_registry.register("category", "name", DummyClass)
        entry = empty_registry.get_entry("category", "name")
        assert entry.loaded_class is DummyClass


# =============================================================================
# Unregister Tests
# =============================================================================


class TestUnregister:
    """Tests for unregister() method."""

    def test_unregister_removes_entry(
        self, registry_with_types: _PluginRegistry
    ) -> None:
        """unregister() removes entry from registry."""
        assert registry_with_types.has_entry("test_category", "type_a")
        removed = registry_with_types.unregister("test_category", "type_a")
        assert removed is not None
        assert not registry_with_types.has_entry("test_category", "type_a")

    def test_unregister_returns_removed_entry(
        self, registry_with_types: _PluginRegistry
    ) -> None:
        """unregister() returns the removed entry."""
        removed = registry_with_types.unregister("test_category", "type_a")
        assert removed is not None
        assert removed.name == "type_a"
        assert removed.category == "test_category"

    def test_unregister_nonexistent_returns_none(
        self, registry_with_types: _PluginRegistry
    ) -> None:
        """unregister() returns None for nonexistent entries."""
        assert registry_with_types.unregister("test_category", "nonexistent") is None

    def test_unregister_with_restore(
        self, registry_with_types: _PluginRegistry
    ) -> None:
        """unregister() can restore a previous entry."""
        original = registry_with_types.get_entry("test_category", "type_a")
        registry_with_types.register(
            "test_category", "type_a", HighPriorityClass, priority=100
        )
        registry_with_types.unregister(
            "test_category", "type_a", restore_entry=original
        )
        restored = registry_with_types.get_entry("test_category", "type_a")
        assert restored.loaded_class is DummyClass


# =============================================================================
# Conflict Resolution Tests
# =============================================================================


class TestConflictResolution:
    """Tests for priority-based conflict resolution."""

    @pytest.mark.parametrize(
        ("first_cls", "first_priority", "second_cls", "second_priority", "expected_cls"),
        [
            param(DummyClass, 0, HighPriorityClass, 10, HighPriorityClass, id="higher priority wins"),
            param(HighPriorityClass, 10, DummyClass, 0, HighPriorityClass, id="lower priority loses"),
            param(DummyClass, 5, AnotherDummyClass, 5, DummyClass, id="equal priority first wins"),
        ],
    )  # fmt: skip
    def test_priority_resolution(
        self,
        empty_registry: _PluginRegistry,
        first_cls: type,
        first_priority: int,
        second_cls: type,
        second_priority: int,
        expected_cls: type,
    ) -> None:
        """Conflict resolution respects priority rules."""
        empty_registry.register("category", "name", first_cls, priority=first_priority)
        empty_registry.register(
            "category", "name", second_cls, priority=second_priority
        )
        entry = empty_registry.get_entry("category", "name")
        assert entry.loaded_class is expected_cls

    def test_package_beats_builtin_equal_priority(
        self, empty_registry: _PluginRegistry
    ) -> None:
        """External package beats builtin at equal priority."""
        builtin = make_entry(
            package="aiperf.builtins",
            class_path="aiperf.builtins:BuiltinClass",
            loaded_class=DummyClass,
        )
        external = make_entry(
            package="external.package",
            class_path="external.package:ExternalClass",
            loaded_class=AnotherDummyClass,
        )
        empty_registry.register_type(builtin)
        empty_registry.register_type(external)
        assert empty_registry.get_entry("test", "test").package == "external.package"

    def test_same_type_conflict_logs_warning(
        self, empty_registry: _PluginRegistry
    ) -> None:
        """Both same package type at equal priority uses first registered."""
        entry1 = make_entry(
            name="conflicting",
            package="external.package1",
            class_path="external.package1:Class1",
            priority=5,
            loaded_class=DummyClass,
        )
        entry2 = make_entry(
            name="conflicting",
            package="external.package2",
            class_path="external.package2:Class2",
            priority=5,
            loaded_class=AnotherDummyClass,
        )
        empty_registry.register_type(entry1)
        empty_registry.register_type(entry2)
        assert (
            empty_registry.get_entry("test", "conflicting").package
            == "external.package1"
        )


# =============================================================================
# Get Entry Tests
# =============================================================================


class TestGetEntry:
    """Tests for get_entry() and has_entry() methods."""

    def test_get_entry_returns_entry(
        self, registry_with_types: _PluginRegistry
    ) -> None:
        """get_entry() returns correct entry."""
        entry = registry_with_types.get_entry("test_category", "type_a")
        assert isinstance(entry, PluginEntry)
        assert entry.name == "type_a"
        assert entry.category == "test_category"

    def test_get_entry_unknown_category_raises(
        self, registry_with_types: _PluginRegistry
    ) -> None:
        """get_entry() raises KeyError for unknown category."""
        with pytest.raises(KeyError, match="Unknown plugin category"):
            registry_with_types.get_entry("nonexistent_category", "type_a")

    def test_get_entry_unknown_name_raises(
        self, registry_with_types: _PluginRegistry
    ) -> None:
        """get_entry() raises TypeNotFoundError for unknown name."""
        with pytest.raises(TypeNotFoundError) as exc_info:
            registry_with_types.get_entry("test_category", "nonexistent")
        assert exc_info.value.category == "test_category"
        assert exc_info.value.name == "nonexistent"
        assert "type_a" in exc_info.value.available

    @pytest.mark.parametrize(
        ("category", "name", "expected"),
        [
            ("test_category", "type_a", True),
            ("nonexistent", "type_a", False),
            ("test_category", "nonexistent", False),
        ],
    )
    def test_has_entry(
        self,
        registry_with_types: _PluginRegistry,
        category: str,
        name: str,
        expected: bool,
    ) -> None:
        """has_entry() returns correct boolean."""
        assert registry_with_types.has_entry(category, name) is expected

    @pytest.mark.parametrize(
        "lookup_name",
        ["type_a", "type-a", "TYPE_A", "TYPE-A", "Type_A", "Type-A"],
    )
    def test_get_entry_normalizes_dashes_underscores(
        self, registry_with_types: _PluginRegistry, lookup_name: str
    ) -> None:
        """get_entry() normalizes dashes/underscores and case."""
        entry = registry_with_types.get_entry("test_category", lookup_name)
        assert entry.name == "type_a"

    @pytest.mark.parametrize(
        "lookup_name",
        ["type_a", "type-a", "TYPE_A", "TYPE-A"],
    )
    def test_has_entry_normalizes_dashes_underscores(
        self, registry_with_types: _PluginRegistry, lookup_name: str
    ) -> None:
        """has_entry() normalizes dashes/underscores and case."""
        assert registry_with_types.has_entry("test_category", lookup_name) is True


# =============================================================================
# Get Class Tests
# =============================================================================


class TestGetClass:
    """Tests for get_class() method."""

    def test_get_class_by_name(self, registry_with_types: _PluginRegistry) -> None:
        """get_class() retrieves class by name."""
        assert registry_with_types.get_class("test_category", "type_a") is DummyClass

    def test_get_class_by_class_path(self, empty_registry: _PluginRegistry) -> None:
        """get_class() retrieves class by class_path."""
        empty_registry.register("test_category", "test_type", DummyClass)
        class_path = f"{DummyClass.__module__}:{DummyClass.__name__}"
        assert empty_registry.get_class("test_category", class_path) is DummyClass

    def test_get_class_unknown_raises(
        self, registry_with_types: _PluginRegistry
    ) -> None:
        """get_class() raises TypeNotFoundError for unknown name."""
        with pytest.raises(TypeNotFoundError):
            registry_with_types.get_class("test_category", "nonexistent")

    def test_get_class_path_category_mismatch_raises(
        self, empty_registry: _PluginRegistry
    ) -> None:
        """get_class() raises ValueError for category mismatch."""
        empty_registry.register("category_a", "type", DummyClass)
        class_path = f"{DummyClass.__module__}:{DummyClass.__name__}"
        with pytest.raises(ValueError, match="Category mismatch"):
            empty_registry.get_class("category_b", class_path)

    def test_get_class_unregistered_class_path_raises(
        self, empty_registry: _PluginRegistry
    ) -> None:
        """get_class() raises KeyError for unregistered class path."""
        with pytest.raises(KeyError, match="Class path not registered"):
            empty_registry.get_class("test", "some.module:SomeClass")


# =============================================================================
# List Methods Tests
# =============================================================================


class TestListMethods:
    """Tests for list_categories(), list_entries(), list_packages()."""

    def test_list_categories(self, registry_with_types: _PluginRegistry) -> None:
        """list_categories() returns sorted category names."""
        categories = registry_with_types.list_categories()
        assert isinstance(categories, list)
        assert "test_category" in categories
        assert "other_category" in categories
        assert categories == sorted(categories)

    def test_list_categories_exclude_internal(
        self, empty_registry: _PluginRegistry
    ) -> None:
        """list_categories(include_internal=False) excludes internal."""
        empty_registry.register("public_cat", "type", DummyClass)
        empty_registry.register("internal_cat", "type", AnotherDummyClass)
        empty_registry._category_metadata = {
            "public_cat": {"internal": False},
            "internal_cat": {"internal": True},
        }
        categories = empty_registry.list_categories(include_internal=False)
        assert "public_cat" in categories
        assert "internal_cat" not in categories

    def test_list_entries(self, registry_with_types: _PluginRegistry) -> None:
        """list_entries() returns entries for category."""
        entries = registry_with_types.list_entries("test_category")
        assert len(entries) == 2
        names = [e.name for e in entries]
        assert "type_a" in names
        assert "type_b" in names

    def test_list_entries_empty_category(
        self, registry_with_types: _PluginRegistry
    ) -> None:
        """list_entries() returns empty list for unknown category."""
        assert registry_with_types.list_entries("nonexistent") == []

    def test_list_packages_empty(self, empty_registry: _PluginRegistry) -> None:
        """list_packages() returns empty list initially."""
        assert empty_registry.list_packages() == []

    def test_list_packages_builtin_only(self, empty_registry: _PluginRegistry) -> None:
        """list_packages(builtin_only=True) filters non-builtin."""
        empty_registry._loaded_plugins = {
            "aiperf": PackageInfo(name="aiperf"),
            "external-plugin": PackageInfo(name="external-plugin"),
        }
        builtin = empty_registry.list_packages(builtin_only=True)
        assert "aiperf" in builtin
        assert "external-plugin" not in builtin


# =============================================================================
# Iteration Tests
# =============================================================================


class TestIteration:
    """Tests for iter_all() and iter_entries()."""

    @pytest.mark.parametrize(
        ("category", "expected_count"),
        [
            param("test_category", 2, id="test category"),
            param("other_category", 1, id="other category"),
            param("nonexistent", 0, id="nonexistent"),
            param(None, 3, id="all categories"),
        ],
    )
    def test_iter_entries(
        self,
        registry_with_types: _PluginRegistry,
        category: str | None,
        expected_count: int,
    ) -> None:
        """iter_entries() yields correct entries for category or all."""
        entries = list(registry_with_types.iter_entries(category))
        assert len(entries) == expected_count

    def test_iter_all_yields_entry_class_tuples(
        self, registry_with_types: _PluginRegistry
    ) -> None:
        """iter_all() yields (entry, class) tuples."""
        results = list(registry_with_types.iter_all("test_category"))
        assert len(results) == 2
        for entry, cls in results:
            assert isinstance(entry, PluginEntry)
            assert cls in (DummyClass, AnotherDummyClass)

    def test_iter_all_none_category(self, registry_with_types: _PluginRegistry) -> None:
        """iter_all(None) yields all entries across categories."""
        results = list(registry_with_types.iter_all(None))
        assert len(results) == 3
        categories = {entry.category for entry, _ in results}
        assert categories == {"test_category", "other_category"}


# =============================================================================
# Find Registered Name Tests
# =============================================================================


class TestFindRegisteredName:
    """Tests for find_registered_name() reverse lookup."""

    def test_find_by_loaded_class(self, registry_with_types: _PluginRegistry) -> None:
        """find_registered_name() finds name for loaded class."""
        registry_with_types.get_class("test_category", "type_a")  # Load class
        assert (
            registry_with_types.find_registered_name("test_category", DummyClass)
            == "type_a"
        )

    def test_find_by_class_path(self, registry_with_types: _PluginRegistry) -> None:
        """find_registered_name() finds name via class path."""
        assert (
            registry_with_types.find_registered_name("test_category", DummyClass)
            == "type_a"
        )

    @pytest.mark.parametrize(
        ("category", "cls"),
        [
            param(
                "test_category", type("Unregistered", (), {}), id="unregistered class"
            ),
            param("other_category", AnotherDummyClass, id="wrong category"),
            param("nonexistent_category", DummyClass, id="nonexistent category"),
        ],
    )
    def test_find_returns_none(
        self,
        registry_with_types: _PluginRegistry,
        category: str,
        cls: type,
    ) -> None:
        """find_registered_name() returns None when not found."""
        assert registry_with_types.find_registered_name(category, cls) is None

    def test_find_uses_class_to_name_cache(
        self, registry_with_types: _PluginRegistry
    ) -> None:
        """find_registered_name() uses _class_to_name cache for loaded classes."""
        # Load the class to populate _class_to_name
        cls = registry_with_types.get_class("test_category", "type_a")
        # Verify cache is populated
        assert cls in registry_with_types._class_to_name
        # Fast path should find it
        assert (
            registry_with_types.find_registered_name("test_category", cls) == "type_a"
        )

    def test_find_uses_cache(self, empty_registry: _PluginRegistry) -> None:
        """find_registered_name() caches results."""
        empty_registry.register("test_category", "type_name", DummyClass)
        empty_registry.find_registered_name("test_category", DummyClass)
        class_path = f"{DummyClass.__module__}:{DummyClass.__name__}"
        cache_key = ("test_category", class_path)
        assert cache_key in empty_registry._class_path_to_name


# =============================================================================
# Create Enum Tests
# =============================================================================


class TestCreateEnum:
    """Tests for create_enum() method."""

    def test_create_enum_basic(self, registry_with_types: _PluginRegistry) -> None:
        """create_enum() creates enum from registered types."""
        enum_cls = registry_with_types.create_enum(
            "test_category", "TestEnum", module=__name__
        )
        assert issubclass(enum_cls, ExtensibleStrEnum)
        assert hasattr(enum_cls, "TYPE_A")
        assert hasattr(enum_cls, "TYPE_B")

    def test_create_enum_values(self, registry_with_types: _PluginRegistry) -> None:
        """create_enum() creates correct enum values."""
        enum_cls = registry_with_types.create_enum(
            "test_category", "TestEnum", module=__name__
        )
        assert enum_cls.TYPE_A.value == "type_a"
        assert enum_cls.TYPE_B.value == "type_b"

    def test_create_enum_empty_category_raises(
        self, registry_with_types: _PluginRegistry
    ) -> None:
        """create_enum() raises KeyError for empty category."""
        with pytest.raises(KeyError, match="No types registered"):
            registry_with_types.create_enum("nonexistent", "TestEnum", module=__name__)

    def test_create_enum_stores_category(
        self, registry_with_types: _PluginRegistry
    ) -> None:
        """create_enum() stores plugin category on enum."""
        enum_cls = registry_with_types.create_enum(
            "test_category", "TestEnum", module=__name__
        )
        assert enum_cls._plugin_category_ == "test_category"


# =============================================================================
# Load Manifest Tests
# =============================================================================


class TestLoadManifest:
    """Tests for load_manifest() method."""

    def test_load_manifest_from_path(
        self, empty_registry: _PluginRegistry, manifest_path: Path
    ) -> None:
        """load_manifest() loads types from file path."""
        empty_registry.load_manifest(manifest_path, plugin_name="test-plugin")
        assert empty_registry.has_entry("test_category", "test_type")
        assert empty_registry.has_entry("test_category", "another_type")

    def test_load_manifest_registers_package(
        self, empty_registry: _PluginRegistry, manifest_path: Path
    ) -> None:
        """load_manifest() registers package in loaded_plugins."""
        empty_registry.load_manifest(manifest_path, plugin_name="test-plugin")
        assert "test-plugin" in empty_registry.list_packages()

    def test_load_manifest_file_not_found_raises(
        self, empty_registry: _PluginRegistry
    ) -> None:
        """load_manifest() raises for missing file."""
        with pytest.raises(RuntimeError, match="not found"):
            empty_registry.load_manifest(Path("/nonexistent/plugins.yaml"))

    def test_load_manifest_empty_yaml(
        self, empty_registry: _PluginRegistry, tmp_path: Path
    ) -> None:
        """load_manifest() handles empty YAML gracefully."""
        manifest = write_manifest(tmp_path, "")
        empty_registry.load_manifest(manifest)  # Should not raise

    def test_load_manifest_invalid_schema_raises(
        self, empty_registry: _PluginRegistry, tmp_path: Path
    ) -> None:
        """load_manifest() raises for invalid schema."""
        manifest = write_manifest(tmp_path, "schema_version: [invalid, list]")
        with pytest.raises(ValueError, match=r"Invalid plugins\.yaml schema"):
            empty_registry.load_manifest(manifest)

    def test_load_manifest_respects_priority(
        self, empty_registry: _PluginRegistry, manifest_path: Path
    ) -> None:
        """load_manifest() respects priority in manifest."""
        empty_registry.load_manifest(manifest_path, plugin_name="test-plugin")
        assert empty_registry.get_entry("test_category", "test_type").priority == 0
        assert empty_registry.get_entry("test_category", "another_type").priority == 5

    def test_load_manifest_with_metadata(
        self, empty_registry: _PluginRegistry, manifest_path: Path
    ) -> None:
        """load_manifest() preserves metadata from manifest."""
        empty_registry.load_manifest(manifest_path, plugin_name="test-plugin")
        entry = empty_registry.get_entry("test_category", "test_type")
        assert entry.metadata["key1"] == "value1"
        assert entry.metadata["key2"] == 42

    def test_load_manifest_invalid_resource_path(
        self, empty_registry: _PluginRegistry
    ) -> None:
        """load_manifest() raises for invalid resource path."""
        with pytest.raises(ValueError, match="Invalid registry path"):
            empty_registry.load_manifest("nonexistent.package:plugins.yaml")

    @pytest.mark.parametrize(
        ("yaml_content", "expected_package"),
        [
            param(
                'schema_version: "1.0"\nplugin:\n  name: yaml-defined\ntest_category:\n  t:\n    class: tests.unit.plugin.test_plugins:DummyClass',
                "yaml-defined",
                id="from plugin block",
            ),
            param(
                'schema_version: "1.0"\ntest_category:\n  t:\n    class: tests.unit.plugin.test_plugins:DummyClass',
                "unknown",
                id="fallback to unknown",
            ),
        ],
    )  # fmt: skip
    def test_load_manifest_package_name_fallback(
        self,
        empty_registry: _PluginRegistry,
        tmp_path: Path,
        yaml_content: str,
        expected_package: str,
    ) -> None:
        """load_manifest() uses plugin.name or falls back to 'unknown'."""
        manifest = write_manifest(tmp_path, yaml_content)
        empty_registry.load_manifest(manifest)  # No plugin_name arg
        entry = empty_registry.get_entry("test_category", "t")
        assert entry.package == expected_package

    @pytest.mark.parametrize(
        "yaml_content",
        [
            param('schema_version: "1.0"\ninvalid_category: "not a dict"', id="invalid category type"),
            param('schema_version: "1.0"\ntest_category:\n  invalid_type: "not a dict"', id="invalid type spec"),
            param('schema_version: "1.0"\ntest_category:\n  missing_class:\n    description: "No class"', id="missing class field"),
            param('schema_version: "99.0"\ntest_category:\n  test_type:\n    class: tests.unit.plugin.test_plugins:DummyClass', id="unsupported schema"),
        ],
    )  # fmt: skip
    def test_load_manifest_warnings(
        self,
        empty_registry: _PluginRegistry,
        tmp_path: Path,
        yaml_content: str,
    ) -> None:
        """load_manifest() handles various invalid content gracefully."""
        manifest = write_manifest(tmp_path, yaml_content)
        empty_registry.load_manifest(manifest, plugin_name="test")  # Should not raise


# =============================================================================
# Validate All Tests
# =============================================================================


class TestValidateAll:
    """Tests for validate_all() method."""

    def test_validate_all_valid_entries(
        self, registry_with_types: _PluginRegistry
    ) -> None:
        """validate_all() returns empty dict for valid entries."""
        assert registry_with_types.validate_all() == {}

    def test_validate_all_with_invalid_entry(
        self, empty_registry: _PluginRegistry
    ) -> None:
        """validate_all() reports invalid entries."""
        entry = make_entry(class_path="nonexistent.module:Class")
        empty_registry._types["test"] = {"invalid": entry}
        errors = empty_registry.validate_all()
        assert "test" in errors
        assert len(errors["test"]) == 1
        assert errors["test"][0][0] == "invalid"


# =============================================================================
# Reset Registry Tests
# =============================================================================


class TestResetRegistry:
    """Tests for reset_registry() method."""

    def test_reset_clears_all(self, registry_with_types: _PluginRegistry) -> None:
        """reset_registry() clears all registered types and caches."""
        registry_with_types.get_class("test_category", "type_a")  # Populate cache

        with patch.object(registry_with_types, "discover_plugins"):
            registry_with_types.reset_registry()

        assert registry_with_types._types == {}
        assert registry_with_types._type_entries_by_class_path == {}
        assert registry_with_types._class_path_to_name == {}

    def test_reset_calls_discover(self, registry_with_types: _PluginRegistry) -> None:
        """reset_registry() calls discover_plugins."""
        with patch.object(registry_with_types, "discover_plugins") as mock:
            registry_with_types.reset_registry()
            mock.assert_called_once()


# =============================================================================
# Category Metadata Tests
# =============================================================================


class TestCategoryMetadata:
    """Tests for category metadata methods."""

    @pytest.mark.parametrize(
        ("metadata", "expected"),
        [
            ({"internal": False}, False),
            ({"internal": True}, True),
            ({}, False),  # Missing key
        ],
    )
    def test_is_internal_category(
        self, empty_registry: _PluginRegistry, metadata: dict, expected: bool
    ) -> None:
        """is_internal_category() returns correct boolean."""
        empty_registry._category_metadata = {"test_category": metadata}
        assert empty_registry.is_internal_category("test_category") is expected

    def test_is_internal_category_unknown(
        self, empty_registry: _PluginRegistry
    ) -> None:
        """is_internal_category() returns False for unknown category."""
        empty_registry._category_metadata = {}
        assert empty_registry.is_internal_category("unknown") is False

    def test_get_category_metadata(self, empty_registry: _PluginRegistry) -> None:
        """get_category_metadata() returns category metadata."""
        empty_registry._category_metadata = {"category": {"key": "value"}}
        assert empty_registry.get_category_metadata("category") == {"key": "value"}

    def test_get_category_metadata_none(self, empty_registry: _PluginRegistry) -> None:
        """get_category_metadata() returns None for unknown."""
        empty_registry._category_metadata = {}
        assert empty_registry.get_category_metadata("unknown") is None

    def test_load_category_metadata_fallback(
        self, empty_registry: _PluginRegistry
    ) -> None:
        """_load_category_metadata() uses fallback when resources fail."""
        with patch("importlib.resources.files", side_effect=Exception("Not found")):
            empty_registry._category_metadata = None
            meta = empty_registry.get_category_metadata("some_category")
            assert meta is None or isinstance(meta, dict)

    def test_load_category_metadata_filters_schema_version(
        self, empty_registry: _PluginRegistry
    ) -> None:
        """_load_category_metadata() filters out schema_version."""
        yaml_content = 'schema_version: "1.0"\nendpoint:\n  internal: false'
        with patch("aiperf.plugin.plugins.importlib.resources.files") as mock_files:
            mock_files.return_value.__truediv__.return_value.read_text.return_value = (
                yaml_content
            )
            empty_registry._category_metadata = None
            empty_registry._load_category_metadata()
            assert "schema_version" not in empty_registry._category_metadata


# =============================================================================
# Package Metadata Tests
# =============================================================================


class TestPackageMetadata:
    """Tests for package metadata methods."""

    def test_get_package_metadata(self, empty_registry: _PluginRegistry) -> None:
        """get_package_metadata() returns package info."""
        empty_registry._loaded_plugins["test-pkg"] = PackageInfo(
            name="test-pkg", version="1.0.0", description="Test"
        )
        result = empty_registry.get_package_metadata("test-pkg")
        assert result.name == "test-pkg"
        assert result.version == "1.0.0"

    def test_get_package_metadata_not_found_raises(
        self, empty_registry: _PluginRegistry
    ) -> None:
        """get_package_metadata() raises KeyError for unknown package."""
        with pytest.raises(KeyError, match="Package not found"):
            empty_registry.get_package_metadata("unknown-pkg")

    @pytest.mark.parametrize(
        ("author", "author_email", "expected"),
        [
            ("Test Author", None, "Test Author"),
            (None, "Jane Doe <jane@example.com>", "Jane Doe"),
            (None, '"John Smith" <john@example.com>', "John Smith"),
        ],
    )
    def test_load_package_metadata_author_parsing(
        self,
        empty_registry: _PluginRegistry,
        author: str | None,
        author_email: str | None,
        expected: str,
    ) -> None:
        """_load_package_metadata() parses author correctly."""
        mock_dist = make_mock_dist(author=author, author_email=author_email)
        info = empty_registry._load_package_metadata("test-pkg", dist=mock_dist)
        assert info.author == expected

    def test_load_package_metadata_no_dist_fallback(
        self, empty_registry: _PluginRegistry
    ) -> None:
        """_load_package_metadata() falls back to importlib.metadata."""
        info = empty_registry._load_package_metadata("nonexistent-package-xyz")
        assert info.name == "nonexistent-package-xyz"
        assert info.version == "unknown"


# =============================================================================
# Plugin Discovery Tests
# =============================================================================


class TestPluginDiscovery:
    """Tests for discover_plugins edge cases."""

    @pytest.mark.parametrize(
        ("find_spec_return", "find_spec_effect"),
        [
            param(None, None, id="module not found"),
            param(None, Exception("Import failed"), id="import exception"),
        ],
    )
    def test_discover_plugins_error_handling(
        self,
        empty_registry: _PluginRegistry,
        find_spec_return: Any,
        find_spec_effect: Exception | None,
    ) -> None:
        """discover_plugins() handles errors gracefully."""
        mock_ep = Mock(name="test-plugin", value="some.module:plugins.yaml", dist=None)

        with (
            patch("importlib.metadata.entry_points", return_value=[mock_ep]),
            patch(
                "importlib.util.find_spec",
                return_value=find_spec_return,
                side_effect=find_spec_effect,
            ),
        ):
            empty_registry.discover_plugins()  # Should not raise

    def test_discover_plugins_skips_already_loaded(
        self, empty_registry: _PluginRegistry
    ) -> None:
        """discover_plugins() skips already-loaded plugins."""
        # Pre-populate loaded plugins
        empty_registry._loaded_plugins["already-loaded"] = PackageInfo(
            name="already-loaded"
        )
        mock_ep = Mock()
        mock_ep.name = "already-loaded"
        mock_ep.value = "some.module:plugins.yaml"

        with (
            patch("aiperf.plugin.plugins.entry_points", return_value=[mock_ep]),
            patch.object(empty_registry, "load_manifest") as mock_load,
        ):
            empty_registry.discover_plugins()
            mock_load.assert_not_called()


# =============================================================================
# Registry File Reading Tests
# =============================================================================


class TestRegistryFileReading:
    """Tests for _read_registry_file edge cases."""

    def test_read_traversable(self, empty_registry: _PluginRegistry) -> None:
        """_read_registry_file() handles Traversable objects."""
        mock = Mock()
        mock.read_text.return_value = "schema_version: '1.0'"
        content = empty_registry._read_registry_file(mock)
        assert content == "schema_version: '1.0'"
        mock.read_text.assert_called_once_with(encoding="utf-8")

    def test_read_string_path(
        self, empty_registry: _PluginRegistry, tmp_path: Path
    ) -> None:
        """_read_registry_file() handles string paths."""
        manifest = write_manifest(tmp_path, "schema_version: '1.0'")
        content = empty_registry._read_registry_file(str(manifest))
        assert "schema_version" in content

    @pytest.mark.parametrize(
        ("path_or_error", "match"),
        [
            param("/nonexistent/file.yaml", "not found", id="file not found"),
            param("directory", "Failed to read", id="directory path"),
        ],
    )
    def test_read_errors(
        self,
        empty_registry: _PluginRegistry,
        tmp_path: Path,
        path_or_error: str,
        match: str,
    ) -> None:
        """_read_registry_file() raises appropriate errors."""
        path = tmp_path if path_or_error == "directory" else path_or_error
        with pytest.raises(RuntimeError, match=match):
            empty_registry._read_registry_file(path)

    def test_read_os_error(
        self, empty_registry: _PluginRegistry, tmp_path: Path
    ) -> None:
        """_read_registry_file() handles OS errors."""
        manifest = write_manifest(tmp_path, "test")
        with (
            patch.object(Path, "read_text", side_effect=OSError("Permission denied")),
            pytest.raises(RuntimeError, match="Failed to read"),
        ):
            empty_registry._read_registry_file(manifest)


# =============================================================================
# PluginEntry Tests
# =============================================================================


class TestPluginEntry:
    """Tests for PluginEntry class."""

    @pytest.mark.parametrize(
        ("package", "expected"),
        [
            ("aiperf.endpoints", True),
            ("aiperf", True),
            ("external.package", False),
        ],
    )
    def test_is_builtin(self, package: str, expected: bool) -> None:
        """is_builtin returns correct boolean."""
        entry = make_entry(package=package)
        assert entry.is_builtin is expected

    def test_load_valid_class(self) -> None:
        """load() imports and returns class."""
        entry = make_entry()
        assert entry.load() is DummyClass

    def test_load_caches_class(self) -> None:
        """load() caches class after first call."""
        entry = make_entry()
        cls1 = entry.load()
        cls2 = entry.load()
        assert cls1 is cls2 is DummyClass

    @pytest.mark.parametrize(
        ("class_path", "error_type", "match"),
        [
            param("no_colon_in_path", ValueError, "Invalid class_path format", id="no colon"),
            param(":ClassName", ValueError, "Invalid class_path format", id="empty module"),
            param("module.path:", ValueError, "Invalid class_path format", id="empty class"),
            param("nonexistent.module:Class", ImportError, "Failed to import", id="module not found"),
            param("tests.unit.plugin.test_plugins:NonexistentClass", AttributeError, "not found", id="class not found"),
        ],
    )  # fmt: skip
    def test_load_errors(
        self, class_path: str, error_type: type[Exception], match: str
    ) -> None:
        """load() raises appropriate errors for invalid class paths."""
        entry = make_entry(class_path=class_path)
        with pytest.raises(error_type, match=match):
            entry.load()

    def test_from_type_spec(self) -> None:
        """from_type_spec() creates entry from PluginSpec."""
        from aiperf.plugin.schema import PluginSpec

        spec = PluginSpec(
            class_="module:Class",
            description="Test",
            priority=5,
            metadata={"key": "value"},
        )
        entry = PluginEntry.from_type_spec(spec, "pkg", "cat", "name")
        assert entry.category == "cat"
        assert entry.name == "name"
        assert entry.package == "pkg"
        assert entry.class_path == "module:Class"
        assert entry.priority == 5
        assert entry.metadata == {"key": "value"}

    def test_get_typed_metadata(self) -> None:
        """get_typed_metadata() validates and returns typed metadata."""
        from pydantic import BaseModel, Field

        class TestMeta(BaseModel):
            name: str = Field(...)
            count: int = Field(default=0)

        entry = make_entry(metadata={"name": "test", "count": 42})
        result = entry.get_typed_metadata(TestMeta)
        assert result.name == "test"
        assert result.count == 42


# =============================================================================
# PluginEntry Validation Tests
# =============================================================================


class TestPluginEntryValidation:
    """Tests for PluginEntry.validate() method."""

    def test_validate_valid_entry(self) -> None:
        """validate() returns (True, None) for valid entry."""
        entry = make_entry()
        valid, error = entry.validate()
        assert valid is True
        assert error is None

    @pytest.mark.parametrize(
        ("class_path", "match"),
        [
            ("invalid_format_no_colon", "Invalid class_path format"),
            ("nonexistent.module:Class", "Module not found"),
            ("some.deeply.nested.nonexistent.module:Class", "Module not found"),
        ],
    )
    def test_validate_errors(self, class_path: str, match: str) -> None:
        """validate() returns appropriate errors."""
        entry = make_entry(class_path=class_path)
        valid, error = entry.validate()
        assert valid is False
        assert match in error

    def test_validate_generic_exception(self) -> None:
        """validate() handles generic exceptions during spec lookup."""
        entry = make_entry(class_path="test.module:Class")
        with patch("importlib.util.find_spec", side_effect=ValueError("Test error")):
            valid, error = entry.validate()
            assert valid is False
            assert "Error checking module" in error

    def test_validate_with_check_class_valid(self) -> None:
        """validate(check_class=True) validates class existence."""
        entry = make_entry()
        valid, _error = entry.validate(check_class=True)
        assert valid is True

    def test_validate_with_check_class_syntax_error(self, tmp_path: Path) -> None:
        """validate(check_class=True) handles syntax errors."""
        with temp_module(tmp_path, "syntax_error_mod", "def broken(:\n    pass"):
            entry = make_entry(
                package="syntax_error_mod",
                class_path="syntax_error_mod:SomeClass",
            )
            valid, error = entry.validate(check_class=True)
            assert valid is False
            assert "Syntax error" in error

    def test_validate_with_check_class_missing(self, tmp_path: Path) -> None:
        """validate(check_class=True) detects missing class in module."""
        with temp_module(tmp_path, "empty_mod", "# No class here"):
            entry = make_entry(
                package="empty_mod",
                class_path="empty_mod:MissingClass",
            )
            valid, error = entry.validate(check_class=True)
            assert valid is False
            assert "MissingClass" in error

    @pytest.mark.parametrize(
        ("content", "class_name"),
        [
            param(
                "from collections import OrderedDict as AliasedClass",
                "AliasedClass",
                id="aliased import",
            ),
            param(
                "DynamicClass = type('DynamicClass', (), {})",
                "DynamicClass",
                id="dynamic type",
            ),
        ],
    )
    def test_validate_with_check_class_dynamic(
        self, tmp_path: Path, content: str, class_name: str
    ) -> None:
        """validate(check_class=True) handles imports and assignments."""
        mod_name = f"dynamic_mod_{class_name.lower()}"
        with temp_module(tmp_path, mod_name, content):
            entry = make_entry(
                package=mod_name,
                class_path=f"{mod_name}:{class_name}",
            )
            valid, _error = entry.validate(check_class=True)
            assert valid is True


# =============================================================================
# PackageInfo Tests
# =============================================================================


class TestPackageInfo:
    """Tests for PackageInfo class."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("aiperf", True),
            ("aiperf-plugins", True),
            ("my-custom-plugin", False),
        ],
    )
    def test_is_builtin(self, name: str, expected: bool) -> None:
        """is_builtin returns correct boolean for package names."""
        assert PackageInfo(name=name).is_builtin is expected

    def test_default_values(self) -> None:
        """PackageInfo has correct default values."""
        info = PackageInfo(name="test")
        assert info.version == "unknown"
        assert info.description == "unknown"
        assert info.author == "unknown"
        assert info.license == "unknown"
        assert info.homepage == "unknown"


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for plugin exceptions."""

    def test_plugin_error_is_exception(self) -> None:
        """PluginError is an Exception subclass."""
        assert issubclass(PluginError, Exception)
        error = PluginError("Test error")
        assert str(error) == "Test error"

    def test_type_not_found_error_message(self) -> None:
        """TypeNotFoundError has informative message."""
        error = TypeNotFoundError("category", "missing", ["type_a", "type_b"])
        assert error.category == "category"
        assert error.name == "missing"
        assert error.available == ["type_a", "type_b"]
        message = str(error)
        assert all(s in message for s in ["missing", "category", "type_a", "type_b"])


# =============================================================================
# Module-Level API Tests
# =============================================================================


class TestModuleLevelAPI:
    """Tests for module-level public API functions."""

    def test_module_exports_core_functions(self) -> None:
        """Module exports expected public API."""
        from aiperf.plugin import plugins

        expected_functions = [
            "get_class",
            "get_entry",
            "has_entry",  # Core lookup
            "iter_all",
            "iter_entries",  # Iteration
            "list_categories",
            "list_entries",
            "list_packages",  # Listing
            "create_enum",
            "register",
            "unregister",  # Utilities
        ]
        for func_name in expected_functions:
            assert callable(getattr(plugins, func_name))
