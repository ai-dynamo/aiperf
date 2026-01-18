#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate environment variable documentation for AIPerf."""

import ast
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FieldDefinition:
    """Represents a Pydantic field definition."""

    name: str
    default: str
    description: str
    constraints: list[str]


@dataclass
class SettingsClass:
    """Represents a settings class."""

    name: str
    docstring: str
    env_prefix: str
    fields: list[FieldDefinition]


def _normalize_text(text: str) -> str:
    """Normalize text by replacing newlines with spaces and stripping whitespace."""
    return " ".join(text.strip().split())


def _extract_field_call_args(
    node: ast.Call,
) -> tuple[str | None, list[str], str | None]:
    """Extract default, constraints, and description from a Field() call.

    Returns:
        Tuple of (default, constraints, description)
    """
    default = None
    constraints = []
    description = None

    # First positional argument is the default
    if node.args:
        default = ast.unparse(node.args[0])

    # Process keyword arguments
    for keyword in node.keywords:
        if keyword.arg == "default":
            default = ast.unparse(keyword.value)
        elif keyword.arg == "description":
            # Extract string literal
            if isinstance(keyword.value, ast.Constant):
                description = keyword.value.value
        elif keyword.arg in ["ge", "le", "gt", "lt", "min_length", "max_length"]:
            # Extract constraint
            constraint_val = ast.unparse(keyword.value)
            symbol_map = {
                "ge": "≥",
                "le": "≤",
                "gt": ">",
                "lt": "<",
                "min_length": "min length:",
                "max_length": "max length:",
            }
            symbol = symbol_map.get(keyword.arg, keyword.arg)
            constraints.append(f"{symbol} {constraint_val}")

    return default, constraints, description


def _parse_field_definition(node: ast.AnnAssign) -> FieldDefinition | None:
    """Parse a field definition from an annotated assignment."""
    if not isinstance(node.target, ast.Name):
        return None

    field_name = node.target.id
    default = "—"
    constraints = []
    description = "—"

    # Check if the value is a Field() call
    if isinstance(node.value, ast.Call):
        if isinstance(node.value.func, ast.Name) and node.value.func.id == "Field":
            default_val, field_constraints, field_desc = _extract_field_call_args(
                node.value
            )
            if default_val:
                default = default_val
            if field_constraints:
                constraints = field_constraints
            if field_desc:
                description = _normalize_text(field_desc)
    elif node.value:
        # Direct assignment without Field()
        default = ast.unparse(node.value)

    return FieldDefinition(
        name=field_name,
        default=default,
        description=description,
        constraints=constraints,
    )


def _extract_env_prefix(class_node: ast.ClassDef) -> str:
    """Extract the env_prefix from the model_config."""
    for item in class_node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "model_config"
                    and isinstance(item.value, ast.Call)
                ):
                    # Find env_prefix in the SettingsConfigDict call
                    for keyword in item.value.keywords:
                        if keyword.arg == "env_prefix" and isinstance(
                            keyword.value, ast.Constant
                        ):
                            return keyword.value.value
    return "AIPERF_"


def _parse_settings_class(class_node: ast.ClassDef) -> SettingsClass | None:
    """Parse a settings class definition."""
    # Skip non-settings classes (must inherit from BaseSettings)
    is_settings = False
    for base in class_node.bases:
        if isinstance(base, ast.Name) and base.id == "BaseSettings":
            is_settings = True
            break

    if not is_settings:
        return None

    # Extract docstring
    docstring = ast.get_docstring(class_node) or ""
    docstring = _normalize_text(docstring) if docstring else ""

    # Extract env_prefix
    env_prefix = _extract_env_prefix(class_node)

    # Extract fields
    fields = []
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign):
            field_def = _parse_field_definition(node)
            if (
                field_def
                and not field_def.name.startswith("_")
                and "default_factory" not in field_def.default
            ):
                # Skip private fields and nested settings fields (they use default_factory)
                fields.append(field_def)

    return SettingsClass(
        name=class_node.name,
        docstring=docstring,
        env_prefix=env_prefix,
        fields=fields,
    )


def _parse_environment_file(file_path: Path) -> list[SettingsClass]:
    """Parse the environment.py file and extract settings classes."""
    content = file_path.read_text()
    tree = ast.parse(content)

    settings_classes = []

    # Find all class definitions
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name.startswith("_")
            and node.name.endswith("Settings")
        ):
            # Parse the settings class
            settings_class = _parse_settings_class(node)
            if settings_class and settings_class.fields:
                settings_classes.append(settings_class)

    return settings_classes


def _format_default_value(default: str) -> str:
    """Format default value for display in markdown."""
    if default == "—":
        return default

    # Clean up some common patterns
    if default.startswith("[") and default.endswith("]"):
        # Format lists
        items = re.findall(r'"([^"]*)"', default)
        if items:
            if len(items) <= 3:
                return "[" + ", ".join(f"`{item}`" for item in items) + "]"
            else:
                return "[" + ", ".join(f"`{item}`" for item in items[:3]) + ", ...]"
        return f"`{default}`"

    # Wrap in backticks
    return f"`{default}`"


def _format_constraints(constraints: list[str]) -> str:
    """Format constraints for display in markdown."""
    if not constraints:
        return "—"
    return ", ".join(constraints)


def generate_markdown_docs(settings_classes: list[SettingsClass]) -> str:
    """Generate markdown documentation for environment variables."""
    lines = [
        "<!--",
        "SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "SPDX-License-Identifier: Apache-2.0",
        "-->",
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
        "> These settings may be modified, renamed, or removed in future releases. Always refer to the",
        "> documentation for your specific release version and test thoroughly when upgrading AIPerf.",
        "",
    ]
    # Sort by env_prefix for consistent output, except DEV which should be last (least important to end users)
    settings_classes = sorted(
        settings_classes,
        key=lambda x: x.env_prefix if x.env_prefix != "AIPERF_DEV_" else "Z" * 100,
    )

    for settings_class in settings_classes:
        # Extract section name from env_prefix (e.g., AIPERF_DATASET_ -> DATASET)
        section_name = settings_class.env_prefix.replace("AIPERF_", "").replace("_", "")

        # Add section header
        lines.append(f"## {section_name}")
        lines.append("")
        if settings_class.docstring:
            lines.append(settings_class.docstring)
            lines.append("")

        # Create table
        lines.append("| Environment Variable | Default | Constraints | Description |")
        lines.append("|----------------------|---------|-------------|-------------|")

        for field in settings_class.fields:
            env_var = f"{settings_class.env_prefix}{field.name}"
            default = _format_default_value(field.default)
            constraints = _format_constraints(field.constraints)
            description = field.description.replace("|", "\\|")  # Escape pipes

            lines.append(f"| `{env_var}` | {default} | {constraints} | {description} |")

        lines.append("")

    return "\n".join(line.rstrip() for line in lines)


def main():
    """Generate environment variable documentation."""
    env_file = Path("src/aiperf/common/environment.py")
    if not env_file.exists():
        print(f"Error: {env_file} not found")
        return

    settings_classes = _parse_environment_file(env_file)

    if not settings_classes:
        print("Warning: No settings classes found")
        return

    markdown_content = generate_markdown_docs(settings_classes)

    output_file = Path("docs/environment_variables.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(markdown_content)
    print(f"Documentation written to {output_file}")


if __name__ == "__main__":
    main()
