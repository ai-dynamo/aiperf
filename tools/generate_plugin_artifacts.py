#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate plugin system artifacts from YAML files.

Generates:
- JSON schemas (categories.schema.json, plugins.schema.json)
- Enum files (enums.py, enums.pyi)
- Type overloads in plugins.py

Usage:
    ./tools/generate_plugin_artifacts.py
    ./tools/generate_plugin_artifacts.py --schemas --enums --overloads
    ./tools/generate_plugin_artifacts.py --check
    ./tools/generate_plugin_artifacts.py --validate
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution: add repo root to path for 'tools' package imports
if __name__ == "__main__" and "tools" not in sys.modules:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import re
import time
from collections import defaultdict
from typing import Any

import yaml
from rich.traceback import Traceback

from tools._core import (
    EnumGenerationError,
    GeneratorError,
    OverloadGenerationError,
    SchemaGenerationError,
    YAMLLoadError,
    console,
    error_console,
    make_generated_header,
    print_error,
    print_generated,
    print_out_of_date,
    print_section,
    print_up_to_date,
    print_updated,
    print_warning,
    write_if_changed,
)

# =============================================================================
# Paths
# =============================================================================

PLUGIN_DIR = Path("src/aiperf/plugin")
SCHEMA_DIR = PLUGIN_DIR / "schema"
CATEGORIES_YAML = PLUGIN_DIR / "categories.yaml"
PLUGINS_YAML = PLUGIN_DIR / "plugins.yaml"
ENUMS_PY = PLUGIN_DIR / "enums.py"
ENUMS_PYI = PLUGIN_DIR / "enums.pyi"
PLUGINS_PY = PLUGIN_DIR / "plugins.py"

METADATA_KEYS = frozenset({"schema_version"})
ACRONYMS = frozenset(
    {"ui", "zmq", "gpu", "api", "cpu", "llm", "json", "yaml", "csv", "id"}
)

GENERATED_HEADER = (
    *make_generated_header("generate_plugin_artifacts"),
    "# fmt: off",
)

IMPORTS_START = "    # <generated-imports>"
IMPORTS_END = "    # </generated-imports>"
OVERLOADS_START = "    # <generated-overloads>"
OVERLOADS_END = "    # </generated-overloads>"

# =============================================================================
# Utilities
# =============================================================================


def load_yaml(path: Path, name: str) -> dict[str, Any]:
    """Load and validate a YAML file."""
    if not path.exists():
        raise YAMLLoadError(f"{name} not found", {"path": str(path)})
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise YAMLLoadError(f"Failed to parse {name}", {"error": str(e)}) from e
    if not isinstance(data, dict):
        raise YAMLLoadError(
            f"Invalid {name}: expected dict", {"type": type(data).__name__}
        )
    return {
        k: v for k, v in data.items() if k not in METADATA_KEYS and isinstance(v, dict)
    }


def load_categories() -> dict[str, dict[str, Any]]:
    return load_yaml(CATEGORIES_YAML, "categories.yaml")


def load_plugins() -> dict[str, dict[str, Any]]:
    return load_yaml(PLUGINS_YAML, "plugins.yaml")


def parse_class_path(path: str) -> tuple[str, str]:
    """Split 'module:Class' into (module, class)."""
    if ":" not in path:
        raise ValueError(f"Invalid class path: {path}")
    return path.rsplit(":", 1)


def to_pascal(s: str) -> str:
    return "".join(w.capitalize() for w in s.replace("-", "_").split("_"))


def to_enum_member(s: str) -> str:
    return s.upper().replace("-", "_")


def to_display(s: str) -> str:
    words = s.replace("-", "_").split("_")
    return " ".join(
        w.upper() if w.lower() in ACRONYMS else w.capitalize() for w in words
    )


# =============================================================================
# Schema Generation
# =============================================================================


def generate_schemas(check: bool = False) -> int:
    """Generate JSON schema files."""
    import copy
    import importlib

    categories = load_categories()

    try:
        from aiperf.plugin.schema.schemas import (
            CategoriesManifest,
            CategorySpec,
            PluginsManifest,
            PluginSpec,
        )
    except ImportError as e:
        raise SchemaGenerationError(
            "Failed to import schema models",
            {
                "error": str(e),
                "hint": "Run: uv pip install -e .",
            },
        ) from e

    # Categories schema
    cat_schema = CategoriesManifest.model_json_schema()
    cat_spec = CategorySpec.model_json_schema()
    if "$defs" in cat_spec:
        cat_schema.setdefault("$defs", {}).update(cat_spec.pop("$defs"))
    cat_schema["additionalProperties"] = cat_spec

    # Plugins schema
    plug_schema = PluginsManifest.model_json_schema()
    plug_spec = PluginSpec.model_json_schema()

    for name, spec in categories.items():
        entry = {
            "type": "object",
            "properties": copy.deepcopy(plug_spec.get("properties", {})),
            "required": list(plug_spec.get("required", [])),
            "title": f"{to_display(name)} Plugin",
        }
        if desc := spec.get("description", "").strip():
            entry["description"] = desc

        if meta_class := spec.get("metadata_class"):
            try:
                mod, cls = parse_class_path(meta_class)
                meta_schema = getattr(
                    importlib.import_module(mod), cls
                ).model_json_schema()
                plug_schema.setdefault("$defs", {}).update(meta_schema.pop("$defs", {}))
                entry["properties"]["metadata"] = meta_schema
            except Exception as e:
                print_warning(f"Could not load metadata schema for {name}: {e}")

        def_name = to_pascal(name) + "Plugin"
        plug_schema.setdefault("$defs", {})[def_name] = entry
        plug_schema["properties"][name] = {
            "title": f"{to_display(name)} Plugins",
            "type": "object",
            "description": desc,
            "additionalProperties": {"$ref": f"#/$defs/{def_name}"},
        }

    plug_schema["additionalProperties"] = False

    # Write
    changed = 0
    for filename, schema in [
        ("categories.schema.json", cat_schema),
        ("plugins.schema.json", plug_schema),
    ]:
        content = (
            json.dumps(
                {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "$id": filename,
                    **schema,
                },
                indent=2,
            )
            + "\n"
        )
        path = SCHEMA_DIR / filename
        existing = path.read_text() if path.exists() else None
        if existing != content:
            changed += 1
            if check:
                print_out_of_date(f"{path} is out of date")
            else:
                write_if_changed(path, content)
                print_generated(path)
        else:
            print_up_to_date(f"{path.name} is up-to-date")

    if check and changed:
        return -changed
    return changed


# =============================================================================
# Enum Generation
# =============================================================================


def generate_enums_py() -> str:
    """Generate enums.py content."""
    categories = load_categories()
    plugins = load_plugins()

    # Find categories with enums and plugins
    enum_data = []
    for name, spec in categories.items():
        if (enum_name := spec.get("enum")) and (cat_plugins := plugins.get(name, {})):
            enum_data.append((name, enum_name, sorted(cat_plugins.keys())))

    if not enum_data:
        raise EnumGenerationError("No categories with plugins found")

    # Build __all__
    names = ["PluginType", "PluginTypeStr"]
    for _, enum_name, _ in enum_data:
        names.extend([enum_name, f"{enum_name}Str"])

    lines = [
        *GENERATED_HEADER,
        '"""Plugin Type Enums - generated dynamically from the plugin registry."""',
        "",
        "from typing import TYPE_CHECKING, TypeAlias",
        "",
        "from aiperf.plugin import plugins",
        "from aiperf.plugin.extensible_enums import create_enum",
        "",
        "__all__ = [" + ", ".join(f'"{n}"' for n in sorted(names)) + "]",
        "",
        "# Plugin Protocol Categories",
        "if TYPE_CHECKING:",
        "    from aiperf.plugin.enums import PluginType, PluginTypeStr",
        "else:",
        "    _all_plugin_categories = plugins.list_categories()",
        '    PluginType = create_enum("PluginType", {',
        '        category.replace("-", "_").upper(): category',
        "        for category in _all_plugin_categories",
        "    })",
        "    PluginTypeStr: TypeAlias = str",
        "",
    ]

    for name, enum_name, plugin_names in enum_data:
        member = to_enum_member(name)
        examples = ", ".join(
            f"{enum_name}.{to_enum_member(n)}" for n in plugin_names[:3]
        )
        lines.extend(
            [
                f"{enum_name}Str: TypeAlias = str",
                f'{enum_name} = plugins.create_enum(PluginType.{member}, "{enum_name}")',
                f'"""Dynamic enum for {name.replace("_", " ")}. Example: {examples}"""',
                "",
            ]
        )

    return "\n".join(lines)


def generate_enums_pyi() -> str | None:
    """Generate enums.pyi type stub."""
    try:
        from aiperf.plugin import plugins

        runtime_cats = list(plugins.list_categories())
    except Exception as e:
        print_warning(f"Could not load plugin system for .pyi: {e}")
        return None

    categories = load_categories()
    all_cats = sorted(set(runtime_cats) | set(categories.keys()))

    lines = [
        *GENERATED_HEADER,
        '"""Type stubs for plugin enums."""',
        "",
        "from typing import Literal, TypeAlias",
        "",
        "from aiperf.plugin.extensible_enums import ExtensibleStrEnum",
        "",
        "class PluginType(ExtensibleStrEnum):",
        '    """Enum for all plugin categories."""',
        "",
    ]

    for cat in all_cats:
        spec = categories.get(cat, {})
        desc = spec.get("description", "").strip().split("\n")[0] if spec else ""
        lines.extend(
            [
                f'    {to_enum_member(cat)} = "{cat}"',
                f'    """{desc or f"{cat} plugins."}"""',
            ]
        )

    lines.extend(
        [
            "",
            "PluginTypeStr: TypeAlias = Literal[",
            "    " + ", ".join(f'"{c}"' for c in all_cats),
            "]",
            "",
        ]
    )

    yaml_plugins = load_plugins()
    for cat in all_cats:
        spec = categories.get(cat, {})
        if not (enum_name := spec.get("enum")):
            continue

        try:
            runtime_entries = list(plugins.list_entries(cat))
        except Exception:
            runtime_entries = []

        cat_yaml = yaml_plugins.get(cat, {})
        plugin_names = sorted(
            set(e.name for e in runtime_entries) | set(cat_yaml.keys())
        )
        if not plugin_names:
            continue

        lines.extend(
            [
                f"class {enum_name}(ExtensibleStrEnum):",
                f'    """Enum for {cat.replace("_", " ")} plugins."""',
                "",
            ]
        )

        for pname in plugin_names:
            lines.append(f'    {to_enum_member(pname)} = "{pname}"')
            if (yaml_spec := cat_yaml.get(pname)) and (
                desc := yaml_spec.get("description")
            ):
                lines.append(f'    """{desc.strip().split(chr(10))[0]}"""')

        literal_members = ", ".join(f'"{n}"' for n in plugin_names)
        lines.extend(
            ["", f"{enum_name}Str: TypeAlias = Literal[{literal_members}]", ""]
        )

    return "\n".join(lines) + "\n"


def generate_enums(check: bool = False) -> int:
    """Generate enum files."""
    changed = 0

    def check_or_write(path: Path, content: str) -> bool:
        """Compare content and optionally write. Returns True if changed."""
        existing = path.read_text() if path.exists() else None
        if existing == content:
            print_up_to_date(f"{path.name} is up-to-date")
            return False
        if check:
            print_out_of_date(f"{path} is out of date")
        else:
            write_if_changed(path, content)
            print_generated(path)
        return True

    content = generate_enums_py()
    if check_or_write(ENUMS_PY, content):
        changed += 1

    if (pyi := generate_enums_pyi()) and check_or_write(ENUMS_PYI, pyi):
        changed += 1

    if check and changed:
        return -changed
    return changed


# =============================================================================
# Overload Generation
# =============================================================================


def generate_overloads(check: bool = False) -> int:
    """Generate type overloads in plugins.py."""
    categories = load_categories()

    if not PLUGINS_PY.exists():
        raise OverloadGenerationError("plugins.py not found", {"path": str(PLUGINS_PY)})

    content = PLUGINS_PY.read_text()

    # Generate imports
    imports: dict[str, list[str]] = defaultdict(list)
    imports["typing"].extend(["Literal", "overload"])
    imports["aiperf.plugin.enums"].extend(["PluginType", "PluginTypeStr"])

    for spec in categories.values():
        if proto := spec.get("protocol"):
            mod, cls = parse_class_path(proto)
            imports[mod].append(cls)
        if enum := spec.get("enum"):
            imports["aiperf.plugin.enums"].append(enum)

    import_lines = ["    # fmt: off", "    # ruff: noqa: I001"]
    for mod in sorted(imports):
        import_lines.append(
            f"    from {mod} import {', '.join(sorted(set(imports[mod])))}"
        )

    # Generate overloads
    overload_lines = []
    for name, spec in categories.items():
        proto = spec.get("protocol", "")
        cls = parse_class_path(proto)[1] if proto and ":" in proto else ""
        ret = f"type[{cls}]" if cls else "type"
        member = to_enum_member(name)
        enum = spec.get("enum")
        name_type = f"{enum} | str" if enum else "str"
        overload_lines.extend(
            [
                "    @overload",
                f'    def get_class(category: Literal[PluginType.{member}, "{name}"], name_or_class_path: {name_type}) -> {ret}: ...',
                "    @overload",
                f'    def iter_all(category: Literal[PluginType.{member}, "{name}"]) -> Iterator[tuple[PluginEntry, {ret}]]: ...',
            ]
        )

    overload_lines.extend(
        [
            "    @overload",
            "    def get_class(category: PluginType | PluginTypeStr, name_or_class_path: str) -> type: ...",
            "    # fmt: on",
        ]
    )

    # Replace markers
    def replace(text: str, start: str, end: str, new: str) -> str:
        pattern = re.compile(
            rf"({re.escape(start)})\n(.*?)({re.escape(end)})", re.DOTALL
        )
        if not pattern.search(text):
            raise OverloadGenerationError(
                "Markers not found", {"start": start, "end": end}
            )
        return pattern.sub(rf"\1\n{new}\n\3", text)

    updated = replace(content, IMPORTS_START, IMPORTS_END, "\n".join(import_lines))
    updated = replace(
        updated, OVERLOADS_START, OVERLOADS_END, "\n".join(overload_lines)
    )

    if content == updated:
        print_up_to_date("Overloads are up-to-date")
        return 0

    if check:
        print_out_of_date("Overloads are out of date!")
        return -1

    PLUGINS_PY.write_text(updated)
    print_updated(PLUGINS_PY)
    return 1


# =============================================================================
# Validation
# =============================================================================


def validate_plugins(verbose: bool = False) -> tuple[int, int, int, float]:
    """Validate all plugin configurations."""
    import importlib
    from collections.abc import Callable

    start = time.perf_counter()
    categories = load_categories()
    plugins_data = load_plugins()
    cat_count = len(categories)
    plug_count = sum(len(p) for p in plugins_data.values())

    failed = 0
    max_errors = 5

    def run_check(name: str, check: Callable[[], list[str]], detail: str) -> None:
        nonlocal failed
        t = time.perf_counter()
        errors = check()
        ms = (time.perf_counter() - t) * 1000
        if errors:
            console.print(f"  [red]✗[/] {name} [dim]({ms:.0f}ms)[/]")
            for e in errors[:max_errors]:
                console.print(f"    [dim]•[/] [red]{e}[/]")
            if len(errors) > max_errors:
                console.print(f"    [dim]... and {len(errors) - max_errors} more[/]")
            failed += 1
        else:
            console.print(f"  [green]✓[/] {name} [dim]({detail}, {ms:.0f}ms)[/]")

    # Schema validation
    def check_cat_schema() -> list[str]:
        from pydantic import ValidationError

        from aiperf.plugin.schema.schemas import CategoriesManifest

        try:
            CategoriesManifest.model_validate(categories)
            return []
        except ValidationError as e:
            return [
                f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
                for err in e.errors()
            ]

    def check_plug_schema() -> list[str]:
        from pydantic import ValidationError

        from aiperf.plugin.schema.schemas import PluginsManifest

        try:
            PluginsManifest.model_validate(plugins_data)
            return []
        except ValidationError as e:
            return [
                f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
                for err in e.errors()
            ]

    def check_refs() -> list[str]:
        return [
            f"'{c}' not in categories.yaml" for c in plugins_data if c not in categories
        ]

    run_check("categories.yaml", check_cat_schema, f"{cat_count} categories")
    run_check("plugins.yaml", check_plug_schema, f"{plug_count} plugins")
    run_check("Category References", check_refs, f"{len(plugins_data)} categories")

    # Class validation
    proto_count = sum(1 for s in categories.values() if s.get("protocol"))
    meta_count = sum(1 for s in categories.values() if s.get("metadata_class"))

    def check_classes() -> list[str]:
        errors = []
        for name, spec in categories.items():
            for key in ("protocol", "metadata_class"):
                if path := spec.get(key):
                    try:
                        mod, cls = parse_class_path(path)
                        getattr(importlib.import_module(mod), cls)
                    except Exception as e:
                        errors.append(f"{name}.{key}: {e}")
        return errors

    run_check(
        "Category Classes",
        check_classes,
        f"{proto_count} protocols, {meta_count} metadata",
    )

    # Plugin class validation
    def check_plugins() -> list[str]:
        from aiperf.plugin import plugins

        errors = []
        for cat in plugins.list_categories():
            for entry in plugins.list_entries(cat):
                try:
                    entry.load()
                except Exception as e:
                    errors.append(f"{cat}.{entry.name}: {e}")
        return errors

    try:
        run_check("Plugin Classes", check_plugins, f"{plug_count} plugins")
    except Exception as e:
        console.print("  [red]✗[/] Plugin Classes")
        console.print(f"    [dim]•[/] [red]Failed to load plugin system: {e}[/]")
        if verbose:
            error_console.print(Traceback.from_exception(type(e), e, e.__traceback__))
        failed += 1

    return failed, cat_count, plug_count, time.perf_counter() - start


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate plugin system artifacts")
    parser.add_argument("--schemas", action="store_true", help="Generate JSON schemas")
    parser.add_argument("--enums", action="store_true", help="Generate enum files")
    parser.add_argument(
        "--overloads", action="store_true", help="Generate type overloads"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate only (no generation)"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check if up-to-date (exit 1 if not)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    args = parser.parse_args()

    if args.validate:
        print_section("Plugin Validation")
        failed, cats, plugs, elapsed = validate_plugins(args.verbose)
        console.print()
        if failed:
            console.print(
                f"[bold red]✗[/] {failed} validation(s) failed. [dim]({elapsed:.2f}s)[/]"
            )
            return 1
        console.print(
            f"[bold green]✓[/] Validated {cats} categories and {plugs} plugins. [dim]({elapsed:.2f}s)[/]"
        )
        return 0

    run_all = not (args.schemas or args.enums or args.overloads)
    generators = [
        ("Plugin Schemas", args.schemas, lambda: generate_schemas(args.check)),
        ("Plugin Enums", args.enums, lambda: generate_enums(args.check)),
        ("Plugin Overloads", args.overloads, lambda: generate_overloads(args.check)),
    ]

    start = time.perf_counter()
    total = 0
    errors = 0
    for name, flag, gen in generators:
        if run_all or flag:
            print_section(name)
            try:
                result = gen()
                if result < 0:
                    errors += 1
                else:
                    total += result
            except GeneratorError as e:
                print_error(e, args.verbose)
                errors += 1
            except Exception as e:
                print_error(e, verbose=True)
                errors += 1

    elapsed = time.perf_counter() - start
    console.print()
    if errors:
        console.print(
            f"[bold red]✗[/] {errors} error(s) occurred. [dim]({elapsed:.2f}s)[/]"
        )
        return 1
    if args.check and total:
        console.print(
            f"[bold yellow]{total}[/] file(s) would be updated. [dim]({elapsed:.2f}s)[/]"
        )
        console.print("Run without [cyan]--check[/] to apply.")
        return 1
    if total:
        console.print(
            f"[bold green]✓[/] Generated {total} plugin file(s). [dim]({elapsed:.2f}s)[/]"
        )
    else:
        console.print(
            f"[bold green]✓[/] All plugin files are up-to-date. [dim]({elapsed:.2f}s)[/]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
