#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate CLI docs for AIPerf."""

import ast
import inspect
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
    choice_descriptions: dict[str, str] | None = None  # Maps choice to its description
    constraints: list[str] | None = None


def _normalize_text(text: str) -> str:
    """Normalize text by replacing newlines with spaces and stripping whitespace."""
    return " ".join(text.strip().split())


def _extract_enum_member_docstrings(enum_class: type[Enum]) -> dict[str, str]:
    """Extract docstrings for enum members by parsing the source code.

    Args:
        enum_class: The enum class to extract docstrings from

    Returns:
        Dictionary mapping enum member names to their docstrings
    """
    try:
        source = inspect.getsource(enum_class)
        tree = ast.parse(source)

        # Find the class definition
        class_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == enum_class.__name__:
                class_def = node
                break

        if not class_def:
            return {}

        member_docs = {}
        i = 0
        while i < len(class_def.body):
            node = class_def.body[i]

            # Look for assignments (enum members)
            if isinstance(node, ast.Assign):  # noqa: SIM102
                # Get the member name
                if node.targets and isinstance(node.targets[0], ast.Name):
                    member_name = node.targets[0].id

                    # Check if the next node is a string (docstring)
                    if i + 1 < len(class_def.body):
                        next_node = class_def.body[i + 1]
                        if isinstance(next_node, ast.Expr) and isinstance(  # noqa: SIM102
                            next_node.value, ast.Constant
                        ):
                            if isinstance(next_node.value.value, str):
                                docstring = next_node.value.value.strip()
                                member_docs[member_name] = docstring
            i += 1

        return member_docs
    except (OSError, TypeError, SyntaxError):
        # Source not available or can't be parsed
        return {}


def _build_constraint_map(
    model_class: type[BaseModel], _visited: set[type] | None = None
) -> dict[str, list[str]]:
    """Build a map of field names to their constraints from a Pydantic model.

    Args:
        model_class: A Pydantic BaseModel class
        _visited: Set of already-visited model classes to prevent infinite recursion

    Returns:
        Dictionary mapping field names to lists of constraint strings
    """
    # Keep track of visited models to prevent infinite recursion
    if _visited is None:
        _visited = set()
    if model_class in _visited:
        return {}
    _visited.add(model_class)

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
                    nested_constraints = _build_constraint_map(arg, _visited)
                    # Merge nested constraints with flattened names
                    for nested_name, nested_vals in nested_constraints.items():
                        # Use the nested field name directly (cyclopts flattens them)
                        if nested_name not in constraint_map_result:
                            constraint_map_result[nested_name] = nested_vals
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            nested_constraints = _build_constraint_map(annotation, _visited)
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


def _extract_choices(arg: Argument) -> tuple[list[str] | None, dict[str, str] | None]:
    """Extract choices and their descriptions from argument.

    Returns:
        Tuple of (choices, choice_descriptions)
    """
    # Check if we should show choices
    if not arg.parameter.show_choices:
        return None, None

    # Direct enum type
    enum_class = None
    if isclass(arg.hint) and issubclass(arg.hint, Enum):
        enum_class = arg.hint
    # Handle list[EnumType] for multi-select parameters
    elif get_origin(arg.hint) in (list, tuple, set):
        args = getattr(arg.hint, "__args__", ())
        if args and isclass(args[0]) and issubclass(args[0], Enum):
            enum_class = args[0]

    if enum_class:
        choices = []
        descriptions = {}

        # Extract docstrings from source code
        member_docstrings = _extract_enum_member_docstrings(enum_class)

        for member_name, member_value in enum_class.__members__.items():
            choice_str = f"`{member_value}`"
            choices.append(choice_str)

            # Get docstring from parsed source code
            if member_name in member_docstrings:
                doc = _normalize_text(member_docstrings[member_name])
                if doc:
                    descriptions[choice_str] = doc

        return choices, descriptions if descriptions else None
    return None, None


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
    choices, choice_descriptions = _extract_choices(arg)

    return ParameterInfo(
        display_name=_extract_display_name(arg),
        long_options=" --".join(long_opts),
        short=" ".join(short_opts),
        description=extract_plain_text(arg.parameter.help),
        required=arg.required,
        type_suffix=get_type_suffix(arg.hint),
        default_value=_extract_default_value(arg),
        choices=choices,
        choice_descriptions=choice_descriptions,
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
            # Handle Union types (e.g., Optional[ServiceConfig], ServiceConfig | None)
            # Python 3.10+ union syntax (X | Y) creates types.UnionType which has __args__ but no __origin__
            elif hasattr(hint_type, "__args__"):
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

    # Detect boolean flags (no type suffix or explicitly bool)
    is_bool_flag = param.type_suffix == "" or param.type_suffix == " <bool>"
    has_negative = "--no-" in param.long_options

    # Add flag indicator for boolean flags
    if is_bool_flag and not has_negative:
        lines.append("<br>_Flag (no value required)_")

    if param.constraints:
        lines.append(f"<br>_Constraints: {', '.join(param.constraints)}_")

    if param.choices:
        # Check if we have descriptions for the choices
        if param.choice_descriptions:
            # Format choices with descriptions as a 3-column markdown table
            lines.append("")
            lines.append("**Choices:**")
            lines.append("")
            lines.append("| | | |")
            lines.append("|-------|:-------:|-------------|")
            for choice in param.choices:
                # Keep backticks from original choice formatting
                desc = param.choice_descriptions.get(choice, "")

                # Check if this is a default value (strip backticks for comparison)
                choice_value = choice.strip("`")
                is_default = False

                if param.default_value and param.default_value != "False":
                    # Handle list defaults (e.g., [ServerMetricsFormat.JSON, ServerMetricsFormat.CSV])
                    default_str = str(param.default_value)
                    if default_str.startswith("[") and default_str.endswith("]"):
                        # Parse list format, extract enum values
                        # e.g., "[ServerMetricsFormat.JSON, ServerMetricsFormat.CSV]"
                        import re

                        # Extract all enum values from the list
                        enum_values = re.findall(r"\.(\w+)", default_str)
                        # Also try matching lowercase enum strings (e.g., ['json', 'csv'])
                        quoted_values = re.findall(r"'([^']+)'", default_str)
                        all_values = enum_values + quoted_values
                        # Check if choice_value matches any extracted value (case-insensitive)
                        is_default = any(
                            choice_value.lower() == val.lower() for val in all_values
                        )
                    else:
                        # Single default value
                        is_default = choice_value == param.default_value

                # Add "_default_" text in middle column if this is the default
                default_marker = "_default_" if is_default else ""

                if desc:
                    lines.append(f"| {choice} | {default_marker} | {desc} |")
                else:
                    lines.append(f"| {choice} | {default_marker} | |")
        else:
            # No descriptions - show inline
            lines.append(f"<br>_Choices: [{', '.join(param.choices)}]_")

            # Add default on separate line if no descriptions
            if param.default_value and param.default_value != "False":
                lines.append(f"<br>_Default: `{param.default_value}`_")
    elif param.default_value and param.default_value != "False":
        # No choices, just default
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

    # Add Commands section with subsections TOC
    if commands_data:
        lines.append("## `aiperf` Commands")
        lines.append("")
        commands = extract_command_info()
        for name, description in commands:
            if name in commands_data:
                command_anchor = f"aiperf-{name}".lower().replace(" ", "-")
                lines.append(f"### [`{name}`](#{command_anchor})")
                lines.append("")
                lines.append(description)
                lines.append("")

                # Add horizontal TOC for this command's parameter groups
                parameter_groups = commands_data[name]
                skip_group_header = len(parameter_groups) == 1 and list(
                    parameter_groups.keys()
                )[0] in ("Parameters", "Options", "General")

                if not skip_group_header:
                    # Create horizontal bulleted list for parameter groups
                    group_links = []
                    for group_name in parameter_groups:
                        # Create anchor from group name
                        group_anchor = (
                            group_name.lower()
                            .replace(" ", "-")
                            .replace("(", "")
                            .replace(")", "")
                        )
                        group_links.append(f"[{group_name}](#{group_anchor})")
                    # Join with bullet separator
                    lines.append(" • ".join(group_links))
                    lines.append("")

    # Generate sections for each command
    for command_name, parameter_groups in commands_data.items():
        lines.append("<hr>")
        lines.append("")
        lines.append(f"## `aiperf {command_name}`")
        lines.append("")

        # Add command description if available
        command_obj = app._commands.get(command_name)
        if command_obj and hasattr(command_obj, "help"):
            help_text = command_obj.help
            if callable(help_text):
                help_text = help_text()
            if help_text:
                full_text = extract_plain_text(help_text)

                # Split by lines to properly handle Examples section
                text_lines = full_text.split("\n")

                # Extract description (everything before Examples: or Args:)
                description_lines = []
                examples_start_idx = None
                examples_end_idx = None

                for i, line in enumerate(text_lines):
                    if line.strip().lower().startswith("examples:"):
                        examples_start_idx = i
                        break
                    elif line.strip().lower().startswith("args:"):
                        break
                    else:
                        description_lines.append(line)

                # Add description (remove common indentation)
                description = "\n".join(description_lines).strip()
                if description:
                    # Split description into paragraphs and normalize indentation
                    paragraphs = description.split("\n\n")
                    for para in paragraphs:
                        if para.strip():
                            # Remove common leading whitespace from each line in paragraph
                            para_lines = para.split("\n")
                            # Find minimum indentation
                            min_indent = min(
                                (
                                    len(line) - len(line.lstrip())
                                    for line in para_lines
                                    if line.strip()
                                ),
                                default=0,
                            )
                            # Remove common indentation
                            normalized_lines = [
                                line[min_indent:] if len(line) > min_indent else line
                                for line in para_lines
                            ]
                            lines.append("\n".join(normalized_lines).strip())
                            lines.append("")

                # Extract examples if found
                if examples_start_idx is not None:
                    # Find where examples end (at Args: or end of text)
                    for i in range(examples_start_idx + 1, len(text_lines)):
                        if text_lines[i].strip().lower().startswith("args:"):
                            examples_end_idx = i
                            break

                    if examples_end_idx is None:
                        examples_end_idx = len(text_lines)

                    # Extract example lines (skip the "Examples:" header)
                    example_lines = text_lines[
                        examples_start_idx + 1 : examples_end_idx
                    ]

                    # Filter out empty lines at start and end, but keep internal structure
                    while example_lines and not example_lines[0].strip():
                        example_lines.pop(0)
                    while example_lines and not example_lines[-1].strip():
                        example_lines.pop()

                    if example_lines:
                        # Remove common indentation from examples
                        non_empty_lines = [
                            line for line in example_lines if line.strip()
                        ]
                        if non_empty_lines:
                            min_indent = min(
                                len(line) - len(line.lstrip())
                                for line in non_empty_lines
                            )
                            normalized_examples = [
                                line[min_indent:] if len(line) > min_indent else line
                                for line in example_lines
                            ]

                            lines.append("**Examples:**")
                            lines.append("")
                            lines.append("```bash")
                            lines.extend(normalized_examples)
                            lines.append("```")
                            lines.append("")

        # For single-group commands with generic group names, skip the group header
        skip_group_header = len(parameter_groups) == 1 and list(
            parameter_groups.keys()
        )[0] in ("Parameters", "Options", "General")

        for group_name, parameters in parameter_groups.items():
            # Use ### for group headers to create proper hierarchy
            if not skip_group_header:
                lines.append(f"### {group_name}")
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
