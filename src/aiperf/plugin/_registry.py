# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugin registry core state and lookup.

The public facade lives in `plugins.py`; manifest-loading / discovery /
programmatic registration lives in `_loader.py` as a mixin to keep each
file under the ergonomics file-size limit.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, TypeAlias
from weakref import WeakKeyDictionary

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.plugin._loader import _PluginRegistryLoaderMixin
from aiperf.plugin.extensible_enums import _normalize_name
from aiperf.plugin.types import (
    PackageInfo,
    PluginEntry,
    TypeNotFoundError,
)

# Alias for category normalization - same as plugin name normalization
_normalize_category = _normalize_name

_logger = AIPerfLogger(__name__)

if TYPE_CHECKING:
    from aiperf.plugin.enums import PluginType, PluginTypeStr

    CategoryT: TypeAlias = PluginType | PluginTypeStr
else:
    CategoryT: TypeAlias = str


class _PluginRegistry(_PluginRegistryLoaderMixin):
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

    # --------------------------------------------------------------------------
    # Lookup: classes and entries
    # --------------------------------------------------------------------------

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
        if ":" in name_or_class_path:
            return self._get_class_by_class_path(category, name_or_class_path)
        return self._get_class_by_name(category, name_or_class_path)

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

    # --------------------------------------------------------------------------
    # Iteration and listing
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # Package / category metadata
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------

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
                self._class_path_to_name[cache_key] = entry.name
                return entry.name

        return None
