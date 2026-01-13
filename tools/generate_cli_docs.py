#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate CLI docs for AIPerf."""

import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from inspect import isclass
from io import StringIO
from pathlib import Path
from typing import Any, get_origin

from cyclopts.argument import Argument, ArgumentCollection
from cyclopts.bind import normalize_tokens
from cyclopts.field_info import FieldInfo
from cyclopts.help import InlineText
from rich.console import Console

# Add src to path so we can import aiperf.cli
sys.path.append("src")

from pydantic import BaseModel

from aiperf.cli import app


@dataclass
class ParameterInfo:
    """Information about a CLI parameter."""

    display_name: str
    long_options: str
    short: str
    description: str
    required: bool
    type_suffix: str
    default_value: str = ""
    choices: list[str] | None = None
    constraints: list[str] | None = None


def _normalize_text(text: str) -> str:
    """Normalize text by replacing newlines with spaces and stripping whitespace."""
    return " ".join(text.strip().split())


def _build_constraint_map(model_class: type[BaseModel]) -> dict[str, list[str]]:
    """Build a map of field names to their constraints from a Pydantic model.

    Args:
        model_class: A Pydantic BaseModel class

    Returns:
        Dictionary mapping field names to lists of constraint strings
    """
    constraint_map_result: dict[str, list[str]] = {}

    # Map Pydantic constraint attribute names to display symbols
    constraint_symbols = {
        "ge": "≥",
        "le": "≤",
        "gt": ">",
        "lt": "<",
        "min_length": "min length:",
        "max_length": "max length:",
    }

    # Iterate through model fields
    for field_name, field_info in model_class.model_fields.items():
        constraints = []

        # Check for numeric and length constraints
        for attr_name, symbol in constraint_symbols.items():
            if hasattr(field_info, attr_name):
                value = getattr(field_info, attr_name)
                if value is not None:
                    constraints.append(f"{symbol} {value}")

        # Also check constraints in metadata (Pydantic v2 Annotated pattern)
        if hasattr(field_info, "metadata") and field_info.metadata:
            for metadata_item in field_info.metadata:
                # Check for Pydantic constraint objects (Ge, Le, Gt, Lt, etc.)
                for attr_name, symbol in constraint_symbols.items():
                    if hasattr(metadata_item, attr_name):
                        value = getattr(metadata_item, attr_name)
                        if value is not None and f"{symbol} {value}" not in constraints:
                            constraints.append(f"{symbol} {value}")

        if constraints:
            constraint_map_result[field_name] = constraints

    # Recursively process nested BaseModel fields
    for _field_name, field_info in model_class.model_fields.items():
        annotation = field_info.annotation
        # Handle Optional types
        if hasattr(annotation, "__origin__"):
            args = getattr(annotation, "__args__", ())
            for arg in args:
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    nested_constraints = _build_constraint_map(arg)
                    # Merge nested constraints with flattened names
                    for nested_name, nested_vals in nested_constraints.items():
                        # Use the nested field name directly (cyclopts flattens them)
                        if nested_name not in constraint_map_result:
                            constraint_map_result[nested_name] = nested_vals
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            nested_constraints = _build_constraint_map(annotation)
            for nested_name, nested_vals in nested_constraints.items():
                if nested_name not in constraint_map_result:
                    constraint_map_result[nested_name] = nested_vals

    return constraint_map_result


def extract_plain_text(obj: Any) -> str:
    """Extract plain text from cyclopts objects."""
    if isinstance(obj, InlineText):
        console = Console(file=StringIO(), record=True, width=1000)
        console.print(obj)
        text = console.export_text(clear=False, styles=False)
        return _normalize_text(text.replace("\r", ""))
    return str(obj) if obj else ""


def get_type_suffix(hint: Any) -> str:
    """Get type suffix for parameter hints."""
    type_mapping: dict[type, str] = {
        bool: "",
        int: " <int>",
        float: " <float>",
        list: " <list>",
        tuple: " <list>",
        set: " <list>",
    }

    # Check direct type first, then origin type for generics
    lookup_type = hint if hint in type_mapping else get_origin(hint)
    return type_mapping.get(lookup_type, " <str>")


def _extract_display_name(arg: Argument) -> str:
    """Extract display name from argument, following cyclopts convention."""
    name = arg.names[0].lstrip("-").replace("-", " ").title()
    return f"{name} _(Required)_" if arg.required else name


def _extract_default_value(arg: Argument) -> str:
    """Extract default value from argument showing raw values."""
    if not arg.show_default:
        return ""

    default = arg.field_info.default
    if default is FieldInfo.empty or default is None:
        return ""

    # Handle callable show_default
    if callable(arg.show_default):
        return str(arg.show_default(default))

    # For all other cases, show the raw value
    return str(default)


def _extract_choices(arg: Argument) -> list[str] | None:
    """Extract choices from argument using only public APIs."""
    if arg.parameter.show_choices and isclass(arg.hint) and issubclass(arg.hint, Enum):
        return [f"`{choice}`" for choice in arg.hint.__members__.values()]
    return None


def _extract_constraints(
    arg: Argument, constraint_map: dict[str, list[str]]
) -> list[str] | None:
    """Extract constraints from Pydantic Field annotations using a pre-built constraint map.

    Args:
        arg: The cyclopts Argument to extract constraints for
        constraint_map: Pre-built mapping of field names to constraint strings

    Returns:
        List of constraint strings if any exist, None otherwise
    """
    # Get the field name (without leading dashes)
    field_name = arg.names[0].lstrip("-").replace("-", "_")

    # Look up constraints in the pre-built map
    return constraint_map.get(field_name)


def _split_argument_names(names: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Split argument names into short and long options."""
    long_opts = [name for name in names if name.startswith("--")]
    short_opts = [
        name for name in names if name not in long_opts and name.startswith("-")
    ]
    return short_opts, long_opts


def _create_parameter_info(
    arg: Argument, constraint_map: dict[str, list[str]]
) -> ParameterInfo:
    """Create ParameterInfo from cyclopts argument using clean property access."""
    short_opts, long_opts = _split_argument_names(arg.names)

    return ParameterInfo(
        display_name=_extract_display_name(arg),
        long_options=" --".join(long_opts),
        short=" ".join(short_opts),
        description=extract_plain_text(arg.parameter.help),
        required=arg.required,
        type_suffix=get_type_suffix(arg.hint),
        default_value=_extract_default_value(arg),
        choices=_extract_choices(arg),
        constraints=_extract_constraints(arg, constraint_map),
    )


def _extract_parameter_groups(
    argument_collection: ArgumentCollection, constraint_map: dict[str, list[str]]
) -> dict[str, list[ParameterInfo]]:
    """Extract parameter groups from argument collection."""
    groups: dict[str, list[ParameterInfo]] = defaultdict(list)

    for arg in argument_collection.filter_by(show=True):
        group_name = arg.parameter.group[0].name
        param_info = _create_parameter_info(arg, constraint_map)
        groups[group_name].append(param_info)

    return dict(groups)


def extract_command_info() -> list[tuple[str, str]]:
    """Extract available commands and their descriptions."""
    skip_commands = {"--help", "-h", "--version"}
    commands = []

    for name, command_obj in app._commands.items():
        if name in skip_commands:
            continue

        help_text = command_obj.help if hasattr(command_obj, "help") else ""
        if callable(help_text):
            help_text = help_text()
        if help_text:
            help_text = extract_plain_text(help_text).split("\n")[0].strip()

        commands.append((name, help_text))
    return commands


def extract_help_data(subcommand: str) -> dict[str, list[ParameterInfo]]:
    """Extract structured help data from the CLI."""
    from typing import get_type_hints

    tokens = normalize_tokens(subcommand)
    _, apps, _ = app.parse_commands(tokens)

    # Build constraint map from Pydantic models in the function signature
    constraint_map: dict[str, list[str]] = {}
    func = apps[-1].default_command
    if func:
        hints = get_type_hints(func, include_extras=False)
        for _param_name, hint_type in hints.items():
            # Check if the hint is a BaseModel or Optional[BaseModel]
            if isinstance(hint_type, type) and issubclass(hint_type, BaseModel):
                constraint_map.update(_build_constraint_map(hint_type))
            # Handle Union types (e.g., Optional[ServiceConfig])
            elif hasattr(hint_type, "__origin__"):
                args = getattr(hint_type, "__args__", ())
                for arg in args:
                    if isinstance(arg, type) and issubclass(arg, BaseModel):
                        constraint_map.update(_build_constraint_map(arg))

    argument_collection = apps[-1].assemble_argument_collection(parse_docstring=True)
    return _extract_parameter_groups(argument_collection, constraint_map)


def _format_parameter_header(param: ParameterInfo) -> str:
    """Format parameter header and extract aliases."""
    all_opts = []

    if param.short:
        all_opts.append(f"`{param.short}`")

    for option in param.long_options.split(" --"):
        if option := option.strip():
            if not option.startswith("--"):
                option = f"--{option.lower().replace(' ', '-')}"
            all_opts.append(f"`{option}`")

    if not all_opts:
        return ""

    primary = ", ".join(all_opts)
    type_display = f" `{param.type_suffix.strip()}`" if param.type_suffix else ""
    required_tag = " _(Required)_" if param.required else ""

    return f"#### {primary}{type_display}{required_tag}"


def _format_parameter_body(param: ParameterInfo) -> list[str]:
    """Format parameter description and metadata as markdown list."""
    lines = [f"{_normalize_text(param.description).rstrip('.')}."]

    if param.constraints:
        lines.append(f"<br>_Constraints: {', '.join(param.constraints)}_")

    if param.choices:
        lines.append(f"<br>_Choices: [{', '.join(param.choices)}]_")

    if param.default_value and param.default_value != "False":
        lines.append(f"<br>_Default: `{param.default_value}`_")

    lines.append("")
    return lines


def generate_markdown_docs(
    commands_data: dict[str, dict[str, list[ParameterInfo]]],
) -> str:
    """Generate markdown documentation from help data.

    Args:
        commands_data: Dictionary mapping command names to their parameter groups
    """
    lines = [
        "<!--",
        "SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "SPDX-License-Identifier: Apache-2.0",
        "-->",
        "",
        "# Command Line Options",
        "",
    ]

    # Add Commands section with links
    if commands_data:
        lines.append("## `aiperf` Commands")
        lines.append("")
        commands = extract_command_info()
        for name, description in commands:
            if name in commands_data:
                anchor = f"aiperf-{name}".lower().replace(" ", "-")
                lines.append(f"- [`{name}`](#{anchor}) - {description}")
        lines.append("")

    # Generate sections for each command
    for command_name, parameter_groups in commands_data.items():
        lines.append(f"## `aiperf {command_name}`")
        lines.append("")

        for group_name, parameters in parameter_groups.items():
            lines.append(f"## {group_name} Options")
            lines.append("")

            for param in parameters:
                if header := _format_parameter_header(param):
                    lines.append(header)
                    lines.append("")
                    lines.extend(_format_parameter_body(param))

    return "\n".join(line.rstrip() for line in lines)


def main():
    """Generate CLI documentation."""
    commands_data = {}

    for command_name, _ in extract_command_info():
        try:
            commands_data[command_name] = extract_help_data(command_name)
        except Exception as e:
            print(
                f"Warning: Could not extract help data for command '{command_name}': {e}"
            )

    markdown_content = generate_markdown_docs(commands_data)

    output_file = Path("docs/cli_options.md")
    output_file.write_text(markdown_content)
    print(f"Documentation written to {output_file}")


if __name__ == "__main__":
    main()
