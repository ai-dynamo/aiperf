# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry loader mixin: manifest parsing, entry-point discovery, registration.

Split from `_registry.py` to keep each file under the ergonomics file-size limit.
The mixin relies on state owned by `_PluginRegistry` (`self._types`,
`self._loaded_plugins`, `self._type_entries_by_class_path`, `self._category_metadata`).
"""

from __future__ import annotations

import importlib
import importlib.util
from importlib.metadata import Distribution, entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic import ValidationError
from ruamel.yaml import YAML

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.plugin._loader_helpers import (
    load_package_metadata,
    read_registry_file,
    resolve_conflict,
)
from aiperf.plugin.constants import (
    DEFAULT_ENTRY_POINT_GROUP,
    SUPPORTED_SCHEMA_VERSIONS,
)
from aiperf.plugin.extensible_enums import ExtensibleStrEnum, _normalize_name
from aiperf.plugin.schema.schemas import (
    PluginsManifest,
    PluginSpec,
)
from aiperf.plugin.types import (
    PackageInfo,
    PluginEntry,
)

_normalize_category = _normalize_name

_logger = AIPerfLogger(__name__)
_yaml = YAML(typ="safe")

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

    from aiperf.plugin.enums import PluginType, PluginTypeStr

    CategoryT: TypeAlias = PluginType | PluginTypeStr
else:
    CategoryT: TypeAlias = str


class _PluginRegistryLoaderMixin:
    """Manifest loading, entry-point discovery, and programmatic registration.

    Attributes expected on `self` (populated by `_PluginRegistry.__init__`):
      _types, _type_entries_by_class_path, _loaded_plugins, _category_metadata.
    """

    # Type stubs so type-checkers see the shared state (populated by subclass).
    _types: dict[str, dict[str, PluginEntry]]
    _type_entries_by_class_path: dict[str, PluginEntry]
    _loaded_plugins: dict[str, PackageInfo]
    _category_metadata: dict[str, dict] | None

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

        yaml_content = read_registry_file(manifest_path)
        raw_data = _yaml.load(yaml_content)

        if not raw_data:
            _logger.warning(f"Empty registry YAML: {manifest_path}")
            return

        try:
            plugins_file = PluginsManifest.model_validate(raw_data)
        except ValidationError as e:
            raise ValueError(
                f"Invalid plugins.yaml schema at {manifest_path}:\n{e}"
            ) from e

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

        self._register_types_from_manifest(package_name, plugins_file, dist=dist)

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

        eps = entry_points(group=entry_point_group)
        plugin_eps = list(eps)
        loaded_count = 0
        skipped_count = 0
        failed_plugins: list[tuple[str, str]] = []  # (name, error_message)

        for ep in plugin_eps:
            try:
                if ep.name in self._loaded_plugins:
                    _logger.debug(
                        lambda name=ep.name: f"Skipping already-loaded plugin: {name}"
                    )
                    skipped_count += 1
                    continue

                module_name, _, filename = ep.value.rpartition(":")
                spec = importlib.util.find_spec(module_name)
                if not spec or not spec.submodule_search_locations:
                    failed_plugins.append(
                        (ep.name, f"Could not locate module: {module_name}")
                    )
                    continue
                registry_path = Path(spec.submodule_search_locations[0]) / filename

                _logger.info(f"Loading plugin: {ep.name}")

                self.load_manifest(registry_path, plugin_name=ep.name, dist=ep.dist)
                loaded_count += 1

            except Exception as e:  # noqa: BLE001 - failed plugin loads are collected into a summary and don't abort discovery
                failed_plugins.append((ep.name, str(e)))
                _logger.debug(
                    lambda name=ep.name, err=e: f"Plugin load error: {name}: {err!r}"
                )

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

        current_entry = self._types.get(category, {}).get(name)
        if current_entry is None:
            return None

        self._type_entries_by_class_path.pop(current_entry.class_path, None)
        cache_key = (category, current_entry.class_path)
        self._class_path_to_name.pop(cache_key, None)  # type: ignore[attr-defined]

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

        category = _normalize_category(category)
        if category not in self._types or not self._types[category]:
            available = "\n".join(f"  • {c}" for c in sorted(self._types.keys()))
            raise KeyError(
                f"No types registered for category '{category}'.\n"
                f"Available categories:\n{available}"
            )

        members = {
            entry.name.replace("-", "_").upper(): entry.name
            for entry in self._types[category].values()
        }

        enum_cls = _create_enum(enum_name, members, module=module)
        enum_cls._plugin_category_ = category
        return enum_cls

    def _load_category_metadata(self) -> None:
        """Load category metadata from categories.yaml (lazy, cached)."""
        try:
            categories_path = (
                importlib.resources.files("aiperf.plugin") / "categories.yaml"
            )
            content = categories_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001 - fallback to relative path on any resources/read error
            fallback = Path(__file__).parent / "categories.yaml"
            if not fallback.exists():
                _logger.warning("categories.yaml not found")
                self._category_metadata = {}
                return
            content = fallback.read_text(encoding="utf-8")

        data = _yaml.load(content) or {}

        self._category_metadata = {
            _normalize_category(k): v
            for k, v in data.items()
            if k not in ("schema_version",) and isinstance(v, dict)
        }

    def _register_types_from_manifest(
        self,
        package: str,
        plugins_file: PluginsManifest,
        *,
        dist: Distribution | None = None,
    ) -> None:
        """Register types from manifest with conflict resolution."""
        package_metadata = load_package_metadata(package, dist=dist)
        self._loaded_plugins[package] = package_metadata

        categories = plugins_file.model_extra or {}
        for category_name, types_dict in categories.items():
            if not isinstance(types_dict, dict):
                _logger.warning(
                    f"Invalid category section type for {category_name}: {type(types_dict).__name__}"
                )
                continue

            for name, type_spec_data in types_dict.items():
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
            self._types[category][name] = entry
            self._type_entries_by_class_path[entry.class_path] = entry
            _logger.debug(
                lambda e=entry: (
                    f"Registered {e.category}:{e.name} from {e.package} (priority={e.priority})"
                )
            )
            return

        winner, reason = resolve_conflict(existing, entry)

        # Always register by class_path so ALL plugins remain accessible via fully-qualified path
        self._type_entries_by_class_path[entry.class_path] = entry

        if winner is entry:
            self._types[category][name] = entry
            _logger.info(
                f"Override registered {category}:{name}: {entry.package} beats {existing.package} ({reason})"
            )
        else:
            _logger.debug(
                lambda ex=existing, e=entry, r=reason: (
                    f"Override rejected {e.category}:{e.name}: {ex.package} beats {e.package} ({r})"
                )
            )

    # Thin method wrappers kept for test compatibility (tests call these as methods).
    def _read_registry_file(self, registry_path: Path | str | Traversable) -> str:
        return read_registry_file(registry_path)

    def _load_package_metadata(
        self, package: str, *, dist: Distribution | None = None
    ) -> PackageInfo:
        return load_package_metadata(package, dist=dist)
