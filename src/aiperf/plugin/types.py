# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import importlib
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Self

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.plugin.schema.schemas import PluginSpec

MetadataT = TypeVar("MetadataT", bound=BaseModel)

_logger = AIPerfLogger(__name__)


# ==============================================================================
# Package Info
# ==============================================================================


class PackageInfo(BaseModel):
    """Metadata about the plugin package.

    This section identifies your plugin package and is displayed in
    plugin listings and error messages.
    """

    name: str = Field(
        description=(
            "Unique identifier for your plugin package. "
            "Use your Python package name, e.g., 'my-aiperf-plugins'."
        )
    )
    version: str = Field(
        default="unknown",
        description="Semantic version of your plugin package, e.g., '1.0.0' or '2.1.3-beta'.",
    )
    description: str = Field(
        default="unknown",
        description="One-line summary of what your plugin package provides.",
    )
    author: str = Field(
        default="unknown",
        description="Author name, team, or organization, e.g., 'NVIDIA' or 'Jane Doe <jane@example.com>'.",
    )
    license: str = Field(
        default="unknown",
        description="License of the plugin package, e.g., 'Apache-2.0' or 'MIT'.",
    )
    homepage: str = Field(
        default="unknown",
        description="Homepage of the plugin package, e.g., 'https://example.com'.",
    )

    @property
    def is_builtin(self) -> bool:
        """Whether this is a built-in plugin package.

        A built-in plugin package is one that is included in the AIPerf core distribution.
        """
        return self.name.startswith("aiperf")


# ==============================================================================
# Custom Exceptions
# ==============================================================================


class PluginError(Exception):
    """Base exception for plugin system errors."""


class TypeNotFoundError(PluginError):
    """Type not found in category. Includes available types in error message."""

    def __init__(self, category: str, name: str, available: list[str]) -> None:
        self.category = category
        self.name = name
        self.available = available

        available_str = "\n".join(f"  • {name}" for name in sorted(available))
        super().__init__(
            f"Type '{name}' not found for category '{category}'.\n"
            f"Available types:\n{available_str}"
        )


# ==============================================================================
# Implementation Classes
# ==============================================================================


class PluginEntry(BaseModel):
    """Lazy-loading plugin entry with metadata. Call load() to import the class."""

    model_config = ConfigDict(frozen=True)

    category: str = Field(..., description="Category identifier")
    name: str = Field(..., description="Type name")
    package: str = Field(..., description="Package providing this type")
    class_path: str = Field(
        ..., description="Fully qualified class path (module:Class)"
    )
    priority: int = Field(default=0, description="Conflict resolution priority")
    description: str = Field(default="", description="Human-readable description")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Plugin-specific metadata from plugins.yaml"
    )
    loaded_class: SkipJsonSchema[type | None] = Field(
        default=None, description="Cached class after loading"
    )

    @property
    def is_builtin(self) -> bool:
        """Whether this is a built-in type (aiperf core package)."""
        return self.package.startswith("aiperf")

    @classmethod
    def from_type_spec(
        cls, type_spec: PluginSpec, package: str, category: str, name: str
    ) -> Self:
        return cls(
            category=category,
            name=name,
            package=package,
            class_path=type_spec.class_,
            priority=type_spec.priority,
            description=type_spec.description,
            metadata=type_spec.metadata or {},
        )

    def load(self) -> type:
        """Import and return the class (cached after first call)."""
        # Return cached class if already loaded
        if self.loaded_class is not None:
            return self.loaded_class

        # Validate and parse class path using structural pattern matching
        module_path, _, class_name = self.class_path.rpartition(":")
        if not module_path or not class_name:
            raise ValueError(
                f"Invalid class_path format: {self.class_path}\n"
                f"Expected format: 'module.path:ClassName'\n"
                f"Example: 'aiperf.endpoints.openai:OpenAIEndpoint'"
            )

        # Import and cache the class
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            # Cache for future calls (idempotent, so safe under concurrent access)
            object.__setattr__(self, "loaded_class", cls)

            _logger.debug(
                lambda: f"Loaded {self.category}:{self.name} from {self.class_path}"
            )

            return cls

        except ImportError as e:
            # Raise enriched ImportError for backward compatibility
            raise ImportError(
                f"Failed to import module for {self.category}:{self.name} from '{self.class_path}'\n"
                f"Reason: {e!r}\n"
                f"Tip: Check that the module is installed and importable"
            ) from e
        except AttributeError as e:
            # Raise enriched AttributeError for backward compatibility
            raise AttributeError(
                f"Class '{class_name}' not found for {self.category}:{self.name} from '{self.class_path}'\n"
                f"Reason: {e!r}\n"
                f"Tip: Check that the class name is spelled correctly and exported from the module"
            ) from e

    def validate(self, check_class: bool = False) -> tuple[bool, str | None]:
        """Validate class is loadable without importing. Returns (is_valid, error_message)."""
        # Already loaded means it's valid
        if self.loaded_class is not None:
            return True, None

        parsed = _parse_class_path(self.class_path)
        if isinstance(parsed, str):
            return False, parsed
        module_path, class_name = parsed

        spec_result = _find_module_spec(module_path)
        if isinstance(spec_result, str):
            return False, spec_result
        spec = spec_result

        if check_class:
            ast_error = _verify_class_in_module_ast(spec, module_path, class_name)
            if ast_error is not None:
                return False, ast_error

        return True, None

    def get_typed_metadata(self, metadata_class: type[MetadataT]) -> MetadataT:
        """Return metadata validated and typed against the given Pydantic model.

        Args:
            metadata_class: Pydantic model class to validate metadata against.

        Returns:
            Validated metadata instance.

        Raises:
            ValidationError: If metadata doesn't match the schema.
        """
        return metadata_class.model_validate(self.metadata)


def _parse_class_path(class_path: str) -> tuple[str, str] | str:
    """Parse ``module:ClassName``. Returns (module, class) on success or an error string."""
    parts = class_path.split(":")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return f"Invalid class_path format: {class_path} (expected 'module:ClassName')"
    return parts[0], parts[1]


def _find_module_spec(module_path: str) -> Any:
    """Find module spec without importing. Returns the spec or an error string."""
    try:
        spec = importlib.util.find_spec(module_path)
    except ModuleNotFoundError as e:
        return f"Module not found: {module_path} ({e})"
    except Exception as e:  # noqa: BLE001 - importlib can raise arbitrary errors during spec discovery; report as validation failure
        return f"Error checking module {module_path}: {e}"
    if spec is None:
        return f"Module not found: {module_path}"
    return spec


def _verify_class_in_module_ast(
    spec: Any, module_path: str, class_name: str
) -> str | None:
    """Return an error string if AST parsing proves the class is absent; else None."""
    if spec.origin is None:
        return None
    try:
        source_path = Path(spec.origin)
        if source_path.suffix != ".py" or not source_path.exists():
            return None
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        if not _ast_defines_name(tree, class_name):
            return f"Class '{class_name}' not found in {module_path}"
    except SyntaxError as e:
        return f"Syntax error in {module_path}: {e}"
    except Exception as e:  # noqa: BLE001 - AST verification is advisory; any read/parse failure is logged and tolerated
        _logger.debug(lambda err=e: f"Could not verify class via AST: {err}")
    return None


def _ast_defines_name(tree: ast.Module, class_name: str) -> bool:
    """Return True if the AST module tree defines ``class_name`` at top level.

    Matches class definitions, ``from ... import`` aliases (original or aliased
    name), and module-level simple assignments. Used by ``PluginEntry.validate``
    to check class presence without executing the module.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return True
        if (
            isinstance(node, ast.ImportFrom)
            and node.names
            and any(
                alias.name == class_name or alias.asname == class_name
                for alias in node.names
            )
        ):
            return True
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == class_name
            for target in node.targets
        ):
            return True
    return False
