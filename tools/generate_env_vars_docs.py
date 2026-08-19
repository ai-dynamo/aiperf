#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate environment variable documentation from Pydantic Settings classes.

Usage:
    ./tools/generate_env_vars_docs.py
    ./tools/generate_env_vars_docs.py --check
    ./tools/generate_env_vars_docs.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution: add repo root to path for 'tools' package imports
if __name__ == "__main__" and "tools" not in sys.modules:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import ast
import re
from dataclasses import dataclass

from tools._core import (
    CONSTRAINT_SYMBOLS,
    GeneratedFile,
    Generator,
    GeneratorResult,
    ParseError,
    main,
    md_frontmatter,
    normalize_text,
    print_step,
)

# =============================================================================
# Configuration
# =============================================================================

# Every module that declares private ``BaseSettings`` classes. Kubernetes and
# operator settings live in their own modules, so documenting only the common
# one silently omits their environment-variable surfaces while reporting
# "up-to-date".
ENV_FILES = (
    Path("src/aiperf/common/environment.py"),
    Path("src/aiperf/kubernetes/environment.py"),
    Path("src/aiperf/operator/environment.py"),
)
OUTPUT_FILE = Path("docs/environment-variables.md")

# =============================================================================
# Data Models
# =============================================================================


@dataclass
class Field:
    """A Pydantic field definition."""

    name: str
    default: str
    description: str
    constraints: list[str]


@dataclass
class Settings:
    """A Pydantic Settings class."""

    name: str
    docstring: str
    env_prefix: str
    fields: list[Field]


# =============================================================================
# AST Parsing
# =============================================================================


def _parse_field(node: ast.AnnAssign) -> Field | None:
    """Parse a Pydantic Field() definition."""
    if not isinstance(node.target, ast.Name):
        return None

    name = node.target.id
    default = "—"
    constraints = []
    description = "—"

    if (
        isinstance(node.annotation, ast.Subscript)
        and isinstance(node.annotation.value, ast.Name)
        and node.annotation.value.id == "Literal"
    ):
        values = (
            node.annotation.slice.elts
            if isinstance(node.annotation.slice, ast.Tuple)
            else [node.annotation.slice]
        )
        constraints.append(
            f"one of: {' / '.join(ast.unparse(value) for value in values)}"
        )

    if isinstance(node.value, ast.Call):
        func = node.value.func
        if isinstance(func, ast.Name) and func.id == "Field":
            if any(kw.arg == "default_factory" for kw in node.value.keywords):
                return None

            # First positional arg is default
            if node.value.args:
                default = ast.unparse(node.value.args[0])

            # Extract keyword args - extend shared symbols with verbose length format
            constraint_map = {
                **CONSTRAINT_SYMBOLS,
                "min_length": "min length:",
                "max_length": "max length:",
            }
            for kw in node.value.keywords:
                if kw.arg == "default":
                    default = ast.unparse(kw.value)
                elif kw.arg == "description" and isinstance(kw.value, ast.Constant):
                    description = normalize_text(kw.value.value)
                elif kw.arg in constraint_map:
                    constraints.append(
                        f"{constraint_map[kw.arg]} {ast.unparse(kw.value)}"
                    )
    elif node.value:
        default = ast.unparse(node.value)

    return Field(name, default, description, constraints)


def _declared_fields(node: ast.ClassDef) -> list[Field]:
    """Return documented fields declared directly on one class."""
    return [
        field
        for item in node.body
        if isinstance(item, ast.AnnAssign)
        and (field := _parse_field(item))
        and not field.name.startswith("_")
        and "default_factory" not in field.default
    ]


def _fields_with_bases(
    node: ast.ClassDef,
    classes: dict[str, ast.ClassDef],
    seen: set[str] | None = None,
) -> list[Field]:
    """Collect inherited field mixins before the class's own declarations."""
    seen = set() if seen is None else seen
    if node.name in seen:
        raise ParseError(f"cyclic settings inheritance involving {node.name}")
    seen.add(node.name)

    fields: dict[str, Field] = {}
    for base in node.bases:
        if not isinstance(base, ast.Name) or base.id not in classes:
            continue
        for field in _fields_with_bases(classes[base.id], classes, set(seen)):
            fields[field.name] = field
    for field in _declared_fields(node):
        fields[field.name] = field
    return list(fields.values())


def _parse_settings_class(
    node: ast.ClassDef, classes: dict[str, ast.ClassDef]
) -> Settings | None:
    """Parse a Pydantic BaseSettings class."""
    # Must inherit from BaseSettings
    if not any(isinstance(b, ast.Name) and b.id == "BaseSettings" for b in node.bases):
        return None

    docstring = normalize_text(ast.get_docstring(node) or "")
    env_prefix = "AIPERF_"

    # Extract env_prefix from model_config
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "model_config"
                    and isinstance(item.value, ast.Call)
                ):
                    for kw in item.value.keywords:
                        if kw.arg == "env_prefix" and isinstance(
                            kw.value, ast.Constant
                        ):
                            env_prefix = kw.value.value

    fields = _fields_with_bases(node, classes)

    return Settings(node.name, docstring, env_prefix, fields) if fields else None


def _parse_resource_settings_calls(tree: ast.AST) -> list[Settings]:
    """Parse ``_resource_settings("PREFIX_", cpu, memory)`` factory call sites.

    These per-container resource settings are built with ``type()`` at runtime,
    so no ``ClassDef`` exists for the AST walk to find -- but every call site
    passes string literals, so the prefix and both defaults are still static.
    Without this, ten containers' CPU/MEMORY knobs (including
    ``AIPERF_K8S_RECORDS_MANAGER_CPU``) stay undocumented.
    """
    settings: list[Settings] = []
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != "_resource_settings"
            or len(node.args) != 3
            or not all(isinstance(a, ast.Constant) for a in node.args)
        ):
            continue
        prefix, cpu, memory = (a.value for a in node.args)
        name = prefix.rstrip("_").replace("_", " ").title()
        settings.append(
            Settings(
                name=f"_{prefix.rstrip('_')}Settings",
                docstring=f"{name} container CPU and memory (Guaranteed QoS).",
                env_prefix=f"AIPERF_K8S_{prefix}",
                fields=[
                    Field("CPU", cpu, "CPU request and limit (Guaranteed QoS)", []),
                    Field(
                        "MEMORY",
                        memory,
                        "Memory request and limit (Guaranteed QoS)",
                        [],
                    ),
                ],
            )
        )
    return settings


def parse_settings_file(path: Path) -> list[Settings]:
    """Parse all private Settings and Environment classes from a Python file."""
    tree = ast.parse(path.read_text())
    settings = []
    classes = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name.startswith("_")
            and node.name.endswith(("Settings", "Environment"))
            and (parsed := _parse_settings_class(node, classes))
        ):
            settings.append(parsed)

    settings.extend(_parse_resource_settings_calls(tree))
    return settings


# =============================================================================
# Markdown Generation
# =============================================================================


def _format_default(default: str) -> str:
    """Format a default value for markdown."""
    if default == "—":
        return default
    if default.startswith("[") and default.endswith("]"):
        items = re.findall(r'"([^"]*)"', default)
        if items:
            formatted = ", ".join(f"`{i}`" for i in items[:3])
            return f"[{formatted}{', ...' if len(items) > 3 else ''}]"
    return f"`{default}`"


def generate_markdown(settings_list: list[Settings]) -> str:
    """Generate markdown documentation."""
    lines = [
        *md_frontmatter("Environment Variables"),
        "",
        "# Environment Variables",
        "",
        "AIPerf can be configured using environment variables with the `AIPERF_` prefix.",
        "All settings are organized into logical subsystems for better discoverability.",
        "",
        "**Pattern:** `AIPERF_{SUBSYSTEM}_{SETTING_NAME}`",
        "",
        "**Examples:**",
        "```bash",
        "export AIPERF_HTTP_CONNECTION_LIMIT=5000",
        "export AIPERF_WORKER_CPU_UTILIZATION_FACTOR=0.8",
        "export AIPERF_ZMQ_RCVTIMEO=600000",
        "```",
        "",
        "> [!WARNING]",
        "> Environment variable names, default values, and definitions are subject to change.",
        "> These settings may be modified, renamed, or removed in future releases.",
        "",
    ]

    # Sort by prefix, DEV last
    sorted_settings = sorted(
        settings_list,
        key=lambda s: s.env_prefix if s.env_prefix != "AIPERF_DEV_" else "ZZZZ",
    )

    for settings in sorted_settings:
        # Derive section heading from env_prefix (e.g. AIPERF_HTTP_ -> HTTP).
        # When env_prefix is the bare ``AIPERF_`` (no subsystem segment),
        # fall back to the class name with the leading underscore and
        # trailing ``Settings`` suffix stripped (e.g. ``_CLIRunnerSettings``
        # -> ``CLI RUNNER``). Otherwise the H2 heading would be empty.
        section = settings.env_prefix.replace("AIPERF_", "").replace("_", "")
        if not section:
            stripped = settings.name.lstrip("_")
            if stripped.endswith("Settings"):
                stripped = stripped[: -len("Settings")]
            # Insert a space before each interior capital so CamelCase
            # renders as space-separated (CLIRunner -> "CLI RUNNER").
            spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stripped)
            spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
            section = spaced.upper()
        lines.append(f"## {section}")
        lines.append("")
        if settings.docstring:
            lines.append(settings.docstring)
            lines.append("")

        lines.append("| Environment Variable | Default | Constraints | Description |")
        lines.append("|----------------------|---------|-------------|-------------|")

        for field in settings.fields:
            env_var = f"{settings.env_prefix}{field.name}"
            default = _format_default(field.default)
            constraints = ", ".join(field.constraints) or "—"
            desc = field.description.replace("|", "\\|")
            lines.append(f"| `{env_var}` | {default} | {constraints} | {desc} |")

        lines.append("")

    return "\n".join(line.rstrip() for line in lines)


# =============================================================================
# Generator
# =============================================================================


class EnvVarsDocsGenerator(Generator):
    """Generate environment variable documentation."""

    name = "Environment Variables Documentation"
    description = "Generate environment variable documentation for AIPerf"

    def generate(self) -> GeneratorResult:
        missing = [f for f in ENV_FILES if not f.exists()]
        if missing:
            raise ParseError(
                f"Source file not found: {missing[0]}",
                {
                    "hint": "Run from the project root directory",
                },
            )

        settings_list: list[Settings] = []
        for env_file in ENV_FILES:
            try:
                settings_list.extend(parse_settings_file(env_file))
            except SyntaxError as e:
                raise ParseError(
                    f"Failed to parse {env_file.name}",
                    {
                        "file": str(env_file),
                        "error": str(e),
                        "line": e.lineno,
                    },
                ) from e

        if not settings_list:
            raise ParseError(
                "No settings classes found",
                {"files": ", ".join(str(f) for f in ENV_FILES)},
            )

        total_fields = sum(len(s.fields) for s in settings_list)

        if self.verbose:
            print_step(
                f"Found {len(settings_list)} settings classes ({total_fields} fields)"
            )
            for s in settings_list:
                print_step(f"  {s.name}: {len(s.fields)} fields ({s.env_prefix}*)")

        return GeneratorResult(
            files=[GeneratedFile(OUTPUT_FILE, generate_markdown(settings_list))],
            summary=f"[bold]{len(settings_list)}[/] subsystems with [bold]{total_fields}[/] environment variables",
        )


if __name__ == "__main__":
    main(EnvVarsDocsGenerator)
