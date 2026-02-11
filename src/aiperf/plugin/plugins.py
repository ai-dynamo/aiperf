# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugin registry with lazy loading and priority-based conflict resolution.

Conflict resolution: higher priority wins; equal priority: external beats built-in.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterator
from importlib.metadata import Distribution, entry_points
from pathlib import Path

from typing import TYPE_CHECKING, Any, TypeAlias
from weakref import WeakKeyDictionary

from pydantic import ValidationError
from ruamel.yaml import YAML

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.plugin.constants import (
    DEFAULT_ENTRY_POINT_GROUP,
    SUPPORTED_SCHEMA_VERSIONS,
)
from aiperf.plugin.extensible_enums import ExtensibleStrEnum, _normalize_name
from aiperf.plugin.schema.schemas import (
    EndpointMetadata,
    PlotMetadata,
    PluginsManifest,
    PluginSpec,
    ServiceMetadata,
    TransportMetadata,
)
from aiperf.plugin.types import (
    PackageInfo,
    PluginEntry,
    TypeNotFoundError,
)

# Alias for category normalization - same as plugin name normalization
_normalize_category = _normalize_name

_logger = AIPerfLogger(__name__)
_yaml = YAML(typ="safe")

# Type alias to reduce repetition throughout the module
if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

    from aiperf.plugin.enums import PluginType, PluginTypeStr

    CategoryT: TypeAlias = PluginType | PluginTypeStr
else:
    CategoryT: TypeAlias = str


# ==============================================================================
# Registry Class
# ==============================================================================


class _PluginRegistry:
    """Plugin registry with discovery and lazy loading."""

    def __init__(self) -> None:
        _logger.debug("Initializing plugin registry")
        # Nested dict: category -> name -> PluginEntry
        self._types: dict[str, dict[str, PluginEntry]] = {}
        # Reverse lookup: class_path -> PluginEntry
        self._type_entries_by_class_path: dict[str, PluginEntry] = {}
        # Loaded plugin metadata: plugin_name -> metadata
        self._loaded_plugins: dict[str, PackageInfo] = {}
        # Reverse mapping from class to normalized name key (for find_registered_name)
        self._class_to_name: WeakKeyDictionary[type, str] = WeakKeyDictionary()
        # Cache for class_path -> name lookup (for find_registered_name slow path)
        self._class_path_to_name: dict[tuple[str, str], str] = {}
        # Category metadata cache (loaded lazily from categories.yaml)
        # Keys are normalized (lowercase, underscores)
        self._category_metadata: dict[str, dict] | None = None

        # Load the builtin registry manifest and discover plugins once on startup
        self.discover_plugins()

    def reset_registry(self) -> None:
        """Reset registry to empty state and reload built-in plugins.

        Intended for testing only. Clears all registered types and reloads
        the built-in registry manifest.
        """
        self._types.clear()
        self._type_entries_by_class_path.clear()
        self._loaded_plugins.clear()
        self._class_to_name.clear()
        self._class_path_to_name.clear()
        self._category_metadata = None
        self.discover_plugins()
        _logger.debug("Registry reset")

    def load_manifest(
        self,
        manifest_path: Path | str | Traversable,
        *,
        plugin_name: str | None = None,
        dist: Distribution | None = None,
    ) -> None:
        """Load plugin types from a YAML registry manifest.

        Parses the YAML file, validates the schema, and registers all types
        with priority-based conflict resolution.

        Args:
            manifest_path: Path to the manifest YAML file.
            plugin_name: Optional plugin name override.
            dist: Optional distribution for metadata lookup.

        Raises:
            FileNotFoundError: If the manifest file doesn't exist.
            ValueError: If the path is a directory or schema is invalid.
            RuntimeError: If the file cannot be read.
        """
        if isinstance(manifest_path, str) and ":" in manifest_path:
            package, _, path = manifest_path.rpartition(":")
            try:
                manifest_path = importlib.resources.files(package) / path
            except Exception as e:
                raise ValueError(
                    f"Invalid registry path: {manifest_path}\nReason: {e!r}"
                ) from e

        # Load YAML content
        yaml_content = self._read_registry_file(manifest_path)

        # Parse YAML
        raw_data = _yaml.load(yaml_content)

        if not raw_data:
            _logger.warning(f"Empty registry YAML: {manifest_path}")
            return

        # Validate and parse using Pydantic model
        try:
            plugins_file = PluginsManifest.model_validate(raw_data)
        except ValidationError as e:
            raise ValueError(
                f"Invalid plugins.yaml schema at {manifest_path}:\n{e}"
            ) from e

        # Check schema version
        if plugins_file.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            _logger.warning(
                f"Unknown schema version {plugins_file.schema_version}, "
                f"supported: {list(SUPPORTED_SCHEMA_VERSIONS)}"
            )

        # Get package name: prefer explicit arg, fallback to YAML plugin.name (for tests)
        if plugin_name:
            package_name = plugin_name
        elif plugin_meta := raw_data.get("plugin"):
            package_name = plugin_meta.get("name", "unknown")
        else:
            package_name = "unknown"

        _logger.info(
            f"Loading registry: {package_name} (schema={plugins_file.schema_version})"
        )

        # Register types from manifest (use model_extra for category data)
        self._register_types_from_manifest(package_name, plugins_file, dist=dist)

        # Count categories (fields in model_extra)
        category_count = (
            len(plugins_file.model_extra) if plugins_file.model_extra else 0
        )
        _logger.info(
            f"Loaded registry: {package_name} with {category_count} categories"
        )

    def discover_plugins(
        self, entry_point_group: str = DEFAULT_ENTRY_POINT_GROUP
    ) -> None:
        """Discover and load plugin registries via setuptools entry points."""
        _logger.debug(lambda: f"Discovering plugins in {entry_point_group}")

        # Discover entry points (Python 3.10+ API)
        eps = entry_points(group=entry_point_group)

        plugin_eps = list(eps)
        loaded_count = 0
        skipped_count = 0
        failed_plugins: list[tuple[str, str]] = []  # (name, error_message)

        for ep in plugin_eps:
            try:
                # Skip already-loaded plugins (e.g., builtin aiperf loaded in __init__)
                if ep.name in self._loaded_plugins:
                    _logger.debug(
                        lambda name=ep.name: f"Skipping already-loaded plugin: {name}"
                    )
                    skipped_count += 1
                    continue

                # Load entry point (should return path to plugins.yaml)
                module_name, _, filename = ep.value.rpartition(":")
                spec = importlib.util.find_spec(module_name)
                if not spec or not spec.submodule_search_locations:
                    failed_plugins.append(
                        (ep.name, f"Could not locate module: {module_name}")
                    )
                    continue
                registry_path = Path(spec.submodule_search_locations[0]) / filename

                _logger.info(f"Loading plugin: {ep.name}")

                # Load plugin registry (pass dist for metadata lookup)
                self.load_manifest(registry_path, plugin_name=ep.name, dist=ep.dist)
                loaded_count += 1

            except Exception as e:
                # Collect error for summary
                failed_plugins.append((ep.name, str(e)))
                _logger.debug(
                    lambda name=ep.name, err=e: f"Plugin load error: {name}: {err!r}"
                )

        # Log summary
        if failed_plugins:
            error_summary = "\n".join(
                f"  • {name}: {error}" for name, error in failed_plugins
            )
            _logger.warning(
                f"Plugin discovery: {loaded_count} loaded, {len(failed_plugins)} failed:\n{error_summary}"
            )
        else:
            _logger.info(
                f"Plugin discovery complete: {loaded_count} loaded, {skipped_count} skipped"
            )

    def get_class(self, category: CategoryT, name_or_class_path: str) -> type:
        """Get type class by name or fully qualified class path.

        Args:
            category: Plugin category (e.g., PluginType.ENDPOINT or "endpoint").
            name_or_class_path: Either a short type name (e.g., 'chat') or
                a fully qualified class path (e.g., 'aiperf.endpoints:ChatEndpoint').

        Returns:
            The plugin class (lazy-loaded, cached after first access).

        Raises:
            TypeNotFoundError: If the type name is not found in the category.
            KeyError: If the category or class path is not registered.
            ValueError: If using class path and category doesn't match.
        """
        # Check if it's a class path (contains ':')
        if ":" in name_or_class_path:
            return self._get_class_by_class_path(category, name_or_class_path)
        else:
            return self._get_class_by_name(category, name_or_class_path)

    def iter_all(
        self, category: CategoryT | None = None
    ) -> Iterator[tuple[PluginEntry, type]]:
        """Iterate over plugin entries with loaded classes.

        Args:
            category: Plugin category to iterate. If None, iterates all categories.

        Yields:
            Tuples of (PluginEntry, loaded_class) for each registered plugin.

        Note:
            This loads each plugin class. For metadata-only iteration without
            loading classes, use iter_entries() instead.
        """
        for entry in self.iter_entries(category):
            yield entry, entry.load()

    def iter_entries(self, category: CategoryT | None = None) -> Iterator[PluginEntry]:
        """Iterate over plugin entries without loading classes.

        Use this for inspection/enumeration when you don't need the actual classes.
        Much faster than iter_all() as it avoids importing plugin modules.

        Args:
            category: Plugin category to iterate. Supports dash/underscore normalized
                matching. If None, iterates all categories.

        Yields:
            PluginEntry for each registered plugin.
        """
        if category is not None:
            category = _normalize_category(category)
            if category not in self._types:
                return
            yield from self._types[category].values()
        else:
            for cat_entries in self._types.values():
                yield from cat_entries.values()

    def validate_all(
        self, check_class: bool = False
    ) -> dict[CategoryT, list[tuple[str, str]]]:
        """Validate all registered types without loading them.

        Checks that modules are importable (and optionally that classes exist)
        without actually executing any import statements.

        Args:
            check_class: If True, also verify class exists via AST parsing.

        Returns:
            Dict mapping category names to lists of (name, error_message) tuples.
            Empty dict means all types are valid.
        """
        errors: dict[CategoryT, list[tuple[str, str]]] = {}

        for category, types in self._types.items():
            for name, entry in types.items():
                valid, error = entry.validate(check_class=check_class)
                if not valid and error:
                    errors.setdefault(category, []).append((name, error))

        return errors

    def list_packages(self, builtin_only: bool = False) -> list[str]:
        """List all loaded plugin package names.

        Args:
            builtin_only: If True, only return built-in packages (aiperf core).

        Returns:
            List of package names that have been loaded into the registry.
        """
        if builtin_only:
            return [
                name for name, meta in self._loaded_plugins.items() if meta.is_builtin
            ]
        return list(self._loaded_plugins.keys())

    def get_package_metadata(self, package_name: str) -> PackageInfo:
        """Get metadata for a loaded plugin package.

        Args:
            package_name: Name of the loaded plugin package.

        Returns:
            PackageInfo with version, description, etc.

        Raises:
            KeyError: If package not found in loaded plugins.
        """
        if package_name not in self._loaded_plugins:
            available = "\n".join(
                f"  • {p}" for p in sorted(self._loaded_plugins.keys())
            )
            raise KeyError(
                f"Package not found: '{package_name}'\nLoaded packages:\n{available}"
            )
        return self._loaded_plugins[package_name]

    def get_category_metadata(self, category: CategoryT) -> dict | None:
        """Get metadata for a plugin category from categories.yaml.

        Args:
            category: Category name to get metadata for. Supports dash/underscore normalized matching.

        Returns:
            Category metadata dict or None if not found.
        """
        if self._category_metadata is None:
            self._load_category_metadata()

        return self._category_metadata.get(_normalize_category(category))

    def register_type(self, entry: PluginEntry) -> None:
        """Register a type entry with conflict resolution.

        Args:
            entry: PluginEntry to register. Must have category and name set.
        """
        self._resolve_conflict_and_register(entry)

    def register(
        self,
        category: CategoryT,
        name: str,
        cls: type,
        *,
        priority: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a class programmatically (for dynamic classes or test overrides).

        Useful for registering classes created at runtime or overriding built-in
        types in tests. Uses the same priority-based conflict resolution as YAML.

        Args:
            category: Plugin category to register under.
            name: Short name for the type.
            cls: The class to register.
            priority: Conflict resolution priority (higher wins). Default: 0.
            metadata: Optional metadata dict to associate with the plugin entry.
        """

        # Create a PluginEntry with the pre-loaded class
        entry = PluginEntry(
            category=category,
            name=name,
            package=cls.__module__,
            class_path=f"{cls.__module__}:{cls.__name__}",
            priority=priority,
            description=cls.__doc__ or "",
            metadata=metadata or {},
            loaded_class=cls,
        )

        # Register with conflict resolution
        self.register_type(entry)

        _logger.debug(
            lambda: (
                f"Registered dynamic type {category}:{name} -> {cls.__name__} (priority={priority})"
            )
        )

    def unregister(
        self,
        category: CategoryT,
        name: str,
        *,
        restore_entry: PluginEntry | None = None,
    ) -> PluginEntry | None:
        """Unregister a plugin entry (for testing only).

        Removes a plugin from the registry and optionally restores a previous entry.
        This method is intended for test cleanup and should not be used in production.

        Args:
            category: Plugin category.
            name: Plugin name to unregister.
            restore_entry: Optional PluginEntry to restore after removal.

        Returns:
            The removed PluginEntry, or None if not found.
        """
        category = _normalize_category(category)
        name = _normalize_name(name)

        # Get the current entry before removal
        current_entry = self._types.get(category, {}).get(name)
        if current_entry is None:
            return None

        # Clean up class_path caches
        self._type_entries_by_class_path.pop(current_entry.class_path, None)
        cache_key = (category, current_entry.class_path)
        self._class_path_to_name.pop(cache_key, None)

        # Remove or restore
        if restore_entry is not None:
            self._types[category][name] = restore_entry
            self._type_entries_by_class_path[restore_entry.class_path] = restore_entry
            _logger.debug(
                lambda: f"Restored {category}:{name} to {restore_entry.package}"
            )
        else:
            del self._types[category][name]
            _logger.debug(lambda: f"Unregistered {category}:{name}")

        return current_entry

    def list_categories(self, *, include_internal: bool = True) -> list[CategoryT]:
        """List all registered category names (sorted alphabetically).

        Args:
            include_internal: If False, exclude internal categories (default: True).

        Returns:
            Sorted list of category names (e.g., ['endpoint', 'transport', ...]).
        """
        categories = sorted(self._types.keys())
        if not include_internal:
            categories = [c for c in categories if not self.is_internal_category(c)]
        return categories

    def list_entries(self, category: CategoryT) -> list[PluginEntry]:
        """List all registered PluginEntry objects for a category.

        Args:
            category: Plugin category to list entries for. Supports dash/underscore normalized matching.

        Returns:
            List of PluginEntry objects with metadata (name, description, priority, etc.).
            Returns empty list if category doesn't exist.
        """
        category = _normalize_category(category)
        if category not in self._types:
            return []
        return list(self._types[category].values())

    def get_entry(self, category: CategoryT, name: str) -> PluginEntry:
        """Get a plugin entry by category and name.

        Args:
            category: Plugin category to search in. Supports dash/underscore
                normalized matching (e.g., 'timing-strategy' matches 'timing_strategy').
            name: Plugin name to find. Supports case-insensitive and dash/underscore
                normalized matching (e.g., 'my-plugin' matches 'my_plugin').

        Returns:
            PluginEntry for the requested plugin.

        Raises:
            TypeNotFoundError: If the plugin is not found.
        """
        category = _normalize_category(category)
        if category not in self._types:
            available = "\n".join(f"  • {c}" for c in sorted(self._types.keys()))
            raise KeyError(
                f"Unknown plugin category: '{category}'\n"
                f"Available categories:\n{available}"
            )

        name = _normalize_name(name)
        if name in self._types[category]:
            return self._types[category][name]

        # Get original names from entries for error message
        available = [entry.name for entry in self.iter_entries(category)]
        raise TypeNotFoundError(category, name, available)

    def has_entry(self, category: CategoryT, name: str) -> bool:
        """Check if a plugin entry exists without raising an exception.

        Args:
            category: Plugin category to search in. Supports dash/underscore
                normalized matching.
            name: Plugin name to find. Supports case-insensitive and dash/underscore
                normalized matching.

        Returns:
            True if the entry exists, False otherwise.
        """
        category = _normalize_category(category)
        if category not in self._types:
            return False
        return _normalize_name(name) in self._types[category]

    def is_internal_category(self, category: CategoryT) -> bool:
        """Check if a category is internal (not user-facing).

        Args:
            category: Category name to check.

        Returns:
            True if the category is marked as internal, False otherwise.
        """
        meta = self.get_category_metadata(category)
        if meta is None:
            return False
        return meta.get("internal", False)

    def create_enum(
        self, category: CategoryT, enum_name: str, *, module: str
    ) -> type[ExtensibleStrEnum]:
        """Create an ExtensibleStrEnum from registered types in a category.

        Dynamically generates an enum class with members for each registered type.
        Member names are UPPER_SNAKE_CASE, values are the original type names.

        Args:
            category: Plugin category to create enum from. Supports dash/underscore normalized matching.
            enum_name: Name for the generated enum class.
            module: Module name for the enum. Required for pickling since pickle
                looks up classes by module.name.

        Returns:
            A new ExtensibleStrEnum subclass.

        Raises:
            KeyError: If no types are registered for the category.
        """
        from aiperf.plugin.extensible_enums import create_enum as _create_enum

        # Get entries without loading the plugin classes to avoid circular imports
        category = _normalize_category(category)
        if category not in self._types or not self._types[category]:
            available = "\n".join(f"  • {c}" for c in self.list_categories())
            raise KeyError(
                f"No types registered for category '{category}'.\n"
                f"Available categories:\n{available}"
            )

        # Create members dict: UPPER_SNAKE_CASE name -> string value (using original names)
        members = {
            entry.name.replace("-", "_").upper(): entry.name
            for entry in self._types[category].values()
        }

        enum_cls = _create_enum(enum_name, members, module=module)

        # Store the category on the enum for reverse lookup (used by docs generation)
        enum_cls._plugin_category_ = category

        return enum_cls

    def _load_category_metadata(self) -> None:
        """Load category metadata from categories.yaml (lazy, cached)."""
        try:
            categories_path = (
                importlib.resources.files("aiperf.plugin") / "categories.yaml"
            )
            content = categories_path.read_text(encoding="utf-8")
        except Exception:
            # Fallback to relative path
            fallback = Path(__file__).parent / "categories.yaml"
            if not fallback.exists():
                _logger.warning("categories.yaml not found")
                self._category_metadata = {}
                return
            content = fallback.read_text(encoding="utf-8")

        data = _yaml.load(content) or {}

        # Store with normalized keys for O(1) lookup
        self._category_metadata = {
            _normalize_category(k): v
            for k, v in data.items()
            if k not in ("schema_version",) and isinstance(v, dict)
        }

    # --------------------------------------------------------------------------
    # Private: Class Path Operations
    # --------------------------------------------------------------------------

    def _get_class_by_class_path(self, category: CategoryT, class_path: str) -> type:
        """Get type by class path with category validation."""
        if class_path not in self._type_entries_by_class_path:
            raise KeyError(
                f"Class path not registered: '{class_path}'\n"
                f"Tip: Ensure the class path is registered in a plugins.yaml file"
            )

        lazy_type = self._type_entries_by_class_path[class_path]

        # Verify category matches (using normalized comparison)
        if _normalize_category(lazy_type.category) != _normalize_category(category):
            raise ValueError(
                f"Category mismatch: {class_path} is registered for category "
                f"'{lazy_type.category}', not '{category}'"
            )

        return self._load_entry(lazy_type)

    def _get_class_by_name(self, category: CategoryT, name: str) -> type:
        """Get type by short name."""
        return self._load_entry(self.get_entry(category, name))

    def _load_entry(self, entry: PluginEntry) -> type:
        """Load a PluginEntry and update the reverse class-to-name mapping."""
        cls = entry.load()
        # Store normalized name for O(1) lookup in find_registered_name
        self._class_to_name[cls] = _normalize_name(entry.name)
        return cls

    def find_registered_name(self, category: CategoryT, cls: type) -> str | None:
        """Reverse lookup: find the registered name for a class.

        Searches by class identity first (for loaded classes), then by class path
        (for classes not loaded via registry). The class_path lookup is cached.

        Args:
            category: Plugin category to search in. Supports dash/underscore normalized matching.
            cls: The class to find the registered name for.

        Returns:
            The registered type name (original form), or None if not found.
        """
        category = _normalize_category(category)
        if category not in self._types:
            return None

        # Fast path: check reverse mapping for already-loaded classes
        if cls in self._class_to_name:
            name = self._class_to_name[cls]  # already normalized
            # Verify it's in the requested category and get original name from entry
            if name in self._types[category]:
                return self._types[category][name].name

        # Medium path: check class_path cache
        target_class_path = f"{cls.__module__}:{cls.__name__}"
        cache_key = (category, target_class_path)
        if cache_key in self._class_path_to_name:
            return self._class_path_to_name[cache_key]

        # Slow path: search by class path for classes not loaded via registry
        for entry in self.iter_entries(category):
            if entry.class_path == target_class_path:
                # Cache the result for future lookups (original name)
                self._class_path_to_name[cache_key] = entry.name
                return entry.name

        return None

    # --------------------------------------------------------------------------
    # Private: Registry Loading
    # --------------------------------------------------------------------------

    def _read_registry_file(self, registry_path: Path | str | Traversable) -> str:
        """Read registry YAML file content."""
        try:
            if hasattr(registry_path, "read_text"):
                # Traversable from importlib.resources
                return registry_path.read_text(encoding="utf-8")
            else:
                # Regular Path (convert str to Path)
                path = (
                    Path(registry_path)
                    if isinstance(registry_path, str)
                    else registry_path
                )

                if not path.exists():
                    raise FileNotFoundError(
                        f"Registry file not found: {path.absolute()}\n"
                        f"Please ensure the plugins.yaml file exists at this location.\n"
                        f"Tip: Check your package installation or path configuration"
                    )

                if not path.is_file():
                    raise ValueError(
                        f"Registry path is not a file: {path.absolute()}\n"
                        f"Expected a YAML file, got a directory or special file"
                    )

                return path.read_text(encoding="utf-8")

        except FileNotFoundError as e:
            raise RuntimeError(
                f"Built-in plugins.yaml not found at {registry_path}.\n"
                "This is a critical error - the package system cannot function without it.\n"
                "Tip: Reinstall the aiperf package or check your installation"
            ) from e
        except OSError as e:
            # Handle permission errors, I/O errors, etc.
            raise RuntimeError(
                f"Failed to read registry file: {registry_path}\n"
                f"Reason: {e}\n"
                f"Tip: Check file permissions and disk status"
            ) from e

    def _register_types_from_manifest(
        self,
        package: str,
        plugins_file: PluginsManifest,
        *,
        dist: Distribution | None = None,
    ) -> None:
        """Register types from manifest with conflict resolution."""
        # Load package metadata from distribution or installed package
        package_metadata = self._load_package_metadata(package, dist=dist)
        self._loaded_plugins[package] = package_metadata

        # Process each category from model_extra (where Pydantic stores extra fields)
        categories = plugins_file.model_extra or {}
        for category_name, types_dict in categories.items():
            if not isinstance(types_dict, dict):
                _logger.warning(
                    f"Invalid category section type for {category_name}: {type(types_dict).__name__}"
                )
                continue

            # Register each type in this category
            for name, type_spec_data in types_dict.items():
                # Convert raw dict to PluginSpec (model_extra stores raw dicts)
                if isinstance(type_spec_data, dict):
                    try:
                        type_spec = PluginSpec.model_validate(type_spec_data)
                    except ValidationError as e:
                        _logger.warning(
                            f"Invalid type spec for {category_name}:{name}: {e}"
                        )
                        continue
                else:
                    _logger.warning(
                        f"Invalid type spec format for {category_name}:{name}: "
                        f"expected dict, got {type(type_spec_data).__name__}"
                    )
                    continue

                if not type_spec.class_:
                    _logger.warning(f"Missing 'class' field for {category_name}:{name}")
                    continue

                entry = PluginEntry.from_type_spec(
                    type_spec, package, category_name, name
                )
                self._resolve_conflict_and_register(entry)

    def _resolve_conflict_and_register(self, entry: PluginEntry) -> None:
        """Resolve conflicts and register type.

        Keys are stored normalized (lowercase, dashes->underscores) for O(1) lookup.
        Original names are preserved in entry.category/entry.name for display.
        """
        category = _normalize_category(entry.category)
        name = _normalize_name(entry.name)
        self._types.setdefault(category, {})
        existing = self._types[category].get(name)

        if existing is None:
            # No conflict - register directly
            self._types[category][name] = entry
            self._type_entries_by_class_path[entry.class_path] = entry
            _logger.debug(
                lambda e=entry: (
                    f"Registered {e.category}:{e.name} from {e.package} (priority={e.priority})"
                )
            )
            return

        # Conflict exists - resolve based on priority and type
        winner, reason = self._resolve_conflict(existing, entry)

        # Always register by class_path so ALL plugins remain accessible via fully-qualified path
        self._type_entries_by_class_path[entry.class_path] = entry

        if winner is entry:
            # New type wins - update the name-based lookup
            self._types[category][name] = entry
            _logger.info(
                f"Override registered {category}:{name}: {entry.package} beats {existing.package} ({reason})"
            )
        else:
            # Existing type wins - name lookup unchanged, but class_path still accessible
            _logger.debug(
                lambda ex=existing, e=entry, r=reason: (
                    f"Override rejected {e.category}:{e.name}: {ex.package} beats {e.package} ({r})"
                )
            )

    def _resolve_conflict(
        self,
        existing: PluginEntry,
        new: PluginEntry,
    ) -> tuple[PluginEntry, str]:
        """Resolve conflict between existing and new type. Returns (winner, reason)."""
        # Rule 1: Higher priority wins
        if new.priority > existing.priority:
            return new, f"priority {new.priority} > {existing.priority}"
        elif new.priority < existing.priority:
            return existing, f"priority {existing.priority} > {new.priority}"

        # Rule 2: Equal priority - package beats built-in
        if not new.is_builtin and existing.is_builtin:
            return new, "package overrides built-in (equal priority)"
        elif new.is_builtin and not existing.is_builtin:
            return existing, "package overrides built-in (equal priority)"

        # Rule 3: Both same type - first wins (warn)
        _logger.warning(
            f"Plugin conflict for {new.category}:{new.name}: {existing.package} vs {new.package} (priority={new.priority})"
        )

        return existing, "first registered wins (both same type)"

    def _load_package_metadata(
        self, package: str, *, dist: Distribution | None = None
    ) -> PackageInfo:
        """Load package metadata from distribution or installed package.

        If dist is provided, uses it directly. Otherwise falls back to looking up
        the package by name.
        """
        # Use distribution directly if provided (from entry point)
        if dist is not None:
            pkg_metadata = dist.metadata
        else:
            # Fallback: look up by package name
            try:
                import importlib.metadata

                pkg_metadata = importlib.metadata.metadata(package)
            except importlib.metadata.PackageNotFoundError:
                _logger.warning(f"Failed to load package metadata for {package}")
                return PackageInfo(name=package)

        # Parse author: PEP 621 uses Author-email with "Name <email>" format
        author = pkg_metadata.get("Author", "")
        if not author:
            author_email = pkg_metadata.get("Author-email", "")
            if author_email:
                # Extract name from "Name <email>" or '"Name" <email>' format
                if "<" in author_email:
                    # Get everything before the '<' and strip whitespace
                    name_part = author_email[: author_email.index("<")].strip()
                    # Remove surrounding quotes if present
                    if name_part.startswith('"'):
                        name_part = name_part[1:]
                    if name_part.endswith('"'):
                        name_part = name_part[:-1]
                    author = name_part.strip()
                else:
                    author = author_email.split(",")[0].strip()

        return PackageInfo(
            name=package,
            version=pkg_metadata.get("Version", "unknown"),
            description=pkg_metadata.get("Summary", ""),
            author=author,
            license=pkg_metadata.get("License", ""),
            homepage=pkg_metadata.get("Home-page", ""),
        )


# ==============================================================================
# Overloaded functions
# ==============================================================================
if TYPE_CHECKING:
    # <generated-imports>
    # fmt: off
    # ruff: noqa: I001
    from aiperf.common.protocols import CommunicationClientProtocol, CommunicationProtocol, ServiceProtocol
    from aiperf.controller.protocols import ServiceManagerProtocol
    from aiperf.dataset.composer.base import BaseDatasetComposer
    from aiperf.dataset.protocols import CustomDatasetLoaderProtocol, DatasetBackingStoreProtocol, DatasetClientStoreProtocol, DatasetSamplingStrategyProtocol
    from aiperf.endpoints.protocols import EndpointProtocol
    from aiperf.exporters.protocols import ConsoleExporterProtocol, DataExporterProtocol
    from aiperf.gpu_telemetry.protocols import GPUTelemetryCollectorProtocol
    from aiperf.plot.core.plot_type_handlers import PlotTypeHandlerProtocol
    from aiperf.plugin.enums import ArrivalPattern, CommClientType, CommunicationBackend, ComposerType, ConsoleExporterType, CustomDatasetType, DataExporterType, DatasetBackingStoreType, DatasetClientStoreType, DatasetSamplingStrategy, EndpointType, GPUTelemetryCollectorType, PlotType, PluginType, PluginTypeStr, RampType, RecordProcessorType, ResultsProcessorType, ServiceRunType, ServiceType, TimingMode, TransportType, UIType, URLSelectionStrategy, ZMQProxyType
    from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor
    from aiperf.post_processors.protocols import RecordProcessorProtocol
    from aiperf.timing.intervals import IntervalGeneratorProtocol
    from aiperf.timing.ramping import RampStrategyProtocol
    from aiperf.timing.strategies.core import TimingStrategyProtocol
    from aiperf.timing.url_samplers import URLSelectionStrategyProtocol
    from aiperf.transports.base_transports import TransportProtocol
    from aiperf.ui.protocols import AIPerfUIProtocol
    from aiperf.zmq.zmq_proxy_base import BaseZMQProxy
    from typing import Literal, overload
    # </generated-imports>
    # <generated-overloads>
    @overload
    def get_class(category: Literal[PluginType.TIMING_STRATEGY, "timing_strategy"], name_or_class_path: TimingMode | str) -> type[TimingStrategyProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.TIMING_STRATEGY, "timing_strategy"]) -> Iterator[tuple[PluginEntry, type[TimingStrategyProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.ARRIVAL_PATTERN, "arrival_pattern"], name_or_class_path: ArrivalPattern | str) -> type[IntervalGeneratorProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.ARRIVAL_PATTERN, "arrival_pattern"]) -> Iterator[tuple[PluginEntry, type[IntervalGeneratorProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.RAMP, "ramp"], name_or_class_path: RampType | str) -> type[RampStrategyProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.RAMP, "ramp"]) -> Iterator[tuple[PluginEntry, type[RampStrategyProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.DATASET_BACKING_STORE, "dataset_backing_store"], name_or_class_path: DatasetBackingStoreType | str) -> type[DatasetBackingStoreProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.DATASET_BACKING_STORE, "dataset_backing_store"]) -> Iterator[tuple[PluginEntry, type[DatasetBackingStoreProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.DATASET_CLIENT_STORE, "dataset_client_store"], name_or_class_path: DatasetClientStoreType | str) -> type[DatasetClientStoreProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.DATASET_CLIENT_STORE, "dataset_client_store"]) -> Iterator[tuple[PluginEntry, type[DatasetClientStoreProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.DATASET_SAMPLER, "dataset_sampler"], name_or_class_path: DatasetSamplingStrategy | str) -> type[DatasetSamplingStrategyProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.DATASET_SAMPLER, "dataset_sampler"]) -> Iterator[tuple[PluginEntry, type[DatasetSamplingStrategyProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.DATASET_COMPOSER, "dataset_composer"], name_or_class_path: ComposerType | str) -> type[BaseDatasetComposer]: ...
    @overload
    def iter_all(category: Literal[PluginType.DATASET_COMPOSER, "dataset_composer"]) -> Iterator[tuple[PluginEntry, type[BaseDatasetComposer]]]: ...
    @overload
    def get_class(category: Literal[PluginType.CUSTOM_DATASET_LOADER, "custom_dataset_loader"], name_or_class_path: CustomDatasetType | str) -> type[CustomDatasetLoaderProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.CUSTOM_DATASET_LOADER, "custom_dataset_loader"]) -> Iterator[tuple[PluginEntry, type[CustomDatasetLoaderProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.ENDPOINT, "endpoint"], name_or_class_path: EndpointType | str) -> type[EndpointProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.ENDPOINT, "endpoint"]) -> Iterator[tuple[PluginEntry, type[EndpointProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.TRANSPORT, "transport"], name_or_class_path: TransportType | str) -> type[TransportProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.TRANSPORT, "transport"]) -> Iterator[tuple[PluginEntry, type[TransportProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.RECORD_PROCESSOR, "record_processor"], name_or_class_path: RecordProcessorType | str) -> type[RecordProcessorProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.RECORD_PROCESSOR, "record_processor"]) -> Iterator[tuple[PluginEntry, type[RecordProcessorProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.RESULTS_PROCESSOR, "results_processor"], name_or_class_path: ResultsProcessorType | str) -> type[BaseMetricsProcessor]: ...
    @overload
    def iter_all(category: Literal[PluginType.RESULTS_PROCESSOR, "results_processor"]) -> Iterator[tuple[PluginEntry, type[BaseMetricsProcessor]]]: ...
    @overload
    def get_class(category: Literal[PluginType.DATA_EXPORTER, "data_exporter"], name_or_class_path: DataExporterType | str) -> type[DataExporterProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.DATA_EXPORTER, "data_exporter"]) -> Iterator[tuple[PluginEntry, type[DataExporterProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.CONSOLE_EXPORTER, "console_exporter"], name_or_class_path: ConsoleExporterType | str) -> type[ConsoleExporterProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.CONSOLE_EXPORTER, "console_exporter"]) -> Iterator[tuple[PluginEntry, type[ConsoleExporterProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.UI, "ui"], name_or_class_path: UIType | str) -> type[AIPerfUIProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.UI, "ui"]) -> Iterator[tuple[PluginEntry, type[AIPerfUIProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.URL_SELECTION_STRATEGY, "url_selection_strategy"], name_or_class_path: URLSelectionStrategy | str) -> type[URLSelectionStrategyProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.URL_SELECTION_STRATEGY, "url_selection_strategy"]) -> Iterator[tuple[PluginEntry, type[URLSelectionStrategyProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.SERVICE, "service"], name_or_class_path: ServiceType | str) -> type[ServiceProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.SERVICE, "service"]) -> Iterator[tuple[PluginEntry, type[ServiceProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.SERVICE_MANAGER, "service_manager"], name_or_class_path: ServiceRunType | str) -> type[ServiceManagerProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.SERVICE_MANAGER, "service_manager"]) -> Iterator[tuple[PluginEntry, type[ServiceManagerProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.COMMUNICATION, "communication"], name_or_class_path: CommunicationBackend | str) -> type[CommunicationProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.COMMUNICATION, "communication"]) -> Iterator[tuple[PluginEntry, type[CommunicationProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.COMMUNICATION_CLIENT, "communication_client"], name_or_class_path: CommClientType | str) -> type[CommunicationClientProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.COMMUNICATION_CLIENT, "communication_client"]) -> Iterator[tuple[PluginEntry, type[CommunicationClientProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.ZMQ_PROXY, "zmq_proxy"], name_or_class_path: ZMQProxyType | str) -> type[BaseZMQProxy]: ...
    @overload
    def iter_all(category: Literal[PluginType.ZMQ_PROXY, "zmq_proxy"]) -> Iterator[tuple[PluginEntry, type[BaseZMQProxy]]]: ...
    @overload
    def get_class(category: Literal[PluginType.PLOT, "plot"], name_or_class_path: PlotType | str) -> type[PlotTypeHandlerProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.PLOT, "plot"]) -> Iterator[tuple[PluginEntry, type[PlotTypeHandlerProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.GPU_TELEMETRY_COLLECTOR, "gpu_telemetry_collector"], name_or_class_path: GPUTelemetryCollectorType | str) -> type[GPUTelemetryCollectorProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.GPU_TELEMETRY_COLLECTOR, "gpu_telemetry_collector"]) -> Iterator[tuple[PluginEntry, type[GPUTelemetryCollectorProtocol]]]: ...
    @overload
    def get_class(category: PluginType | PluginTypeStr, name_or_class_path: str) -> type: ...
    # fmt: on
    # </generated-overloads>


# ==============================================================================
# Module-Level Singleton
# ==============================================================================
# This pattern follows the random_generator module design.
# Usage:
#   from aiperf.plugin import plugins
#   from aiperf.plugin.enums import PluginType
#   EndpointClass = plugins.get_class(PluginType.ENDPOINT, 'openai')
#   endpoint = EndpointClass(...)
# ==============================================================================

# Create singleton instance at module load
_registry = _PluginRegistry()

# ==============================================================================
# Public API: Module-Level Functions
# ==============================================================================

# Core lookup
get_class = _registry.get_class
get_entry = _registry.get_entry
has_entry = _registry.has_entry

# Iteration
iter_all = _registry.iter_all
iter_entries = _registry.iter_entries

# Listing
list_categories = _registry.list_categories
list_entries = _registry.list_entries
list_packages = _registry.list_packages

# Metadata
get_category_metadata = _registry.get_category_metadata
get_package_metadata = _registry.get_package_metadata

# Utilities
create_enum = _registry.create_enum
find_registered_name = _registry.find_registered_name
is_internal_category = _registry.is_internal_category
validate_all = _registry.validate_all

# Registration (for plugins and tests)
register = _registry.register
unregister = _registry.unregister
load_manifest = _registry.load_manifest
reset_registry = _registry.reset_registry


# ==============================================================================
# Metadata Helpers
# ==============================================================================


def get_metadata(category: CategoryT, name: str) -> dict[str, Any]:
    """Get raw metadata dict for a plugin.

    Args:
        category: Plugin category.
        name: Plugin name.

    Returns:
        Metadata dict from plugins.yaml.
    """
    return get_entry(category, name).metadata


def get_endpoint_metadata(name: str) -> EndpointMetadata:
    """Get typed metadata for an endpoint plugin.

    Args:
        name: Endpoint plugin name (e.g., 'chat', 'completions').

    Returns:
        Validated EndpointMetadata instance.
    """
    return get_entry("endpoint", name).get_typed_metadata(EndpointMetadata)


def get_transport_metadata(name: str) -> TransportMetadata:
    """Get typed metadata for a transport plugin.

    Args:
        name: Transport plugin name (e.g., 'http').

    Returns:
        Validated TransportMetadata instance.
    """
    return get_entry("transport", name).get_typed_metadata(TransportMetadata)


def get_plot_metadata(name: str) -> PlotMetadata:
    """Get typed metadata for a plot plugin.

    Args:
        name: Plot plugin name (e.g., 'scatter', 'histogram').

    Returns:
        Validated PlotMetadata instance.
    """
    return get_entry("plot", name).get_typed_metadata(PlotMetadata)


def get_service_metadata(name: str) -> ServiceMetadata:
    """Get typed metadata for a service plugin.

    Args:
        name: Service plugin name (e.g., 'worker', 'timing_manager').

    Returns:
        Validated ServiceMetadata instance.
    """
    return get_entry("service", name).get_typed_metadata(ServiceMetadata)


# Mapping of categories to their metadata classes (for categories with typed metadata)
_CATEGORY_METADATA_CLASSES: dict[str, type] = {
    "endpoint": EndpointMetadata,
    "transport": TransportMetadata,
    "plot": PlotMetadata,
    "service": ServiceMetadata,
}


def get_typed_metadata(category: CategoryT, name: str) -> Any:
    """Get typed metadata for any plugin that has a registered metadata class.

    This is a generic helper that automatically uses the correct metadata class
    based on the category. For categories without a registered metadata class,
    returns the raw metadata dict.

    Args:
        category: Plugin category (e.g., PluginType.ENDPOINT or "endpoint").
            Supports dash/underscore normalized matching.
        name: Plugin name within the category.

    Returns:
        Validated metadata instance if the category has a metadata class,
        otherwise the raw metadata dict.

    Example:
        >>> endpoint_meta = get_typed_metadata(PluginType.ENDPOINT, "chat")
        >>> print(endpoint_meta.streaming)  # Typed access
        True
    """
    category = _normalize_category(category)
    entry = get_entry(category, name)
    if metadata_cls := _CATEGORY_METADATA_CLASSES.get(category):
        return entry.get_typed_metadata(metadata_cls)

    # Fall back to raw metadata dict
    return entry.metadata
