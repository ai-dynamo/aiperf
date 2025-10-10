#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate CLI docs for AIPerf."""

from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, get_origin

from cyclopts.argument import Argument, ArgumentCollection
from cyclopts.bind import normalize_tokens
from cyclopts.field_info import FieldInfo
from cyclopts.help import InlineText
from rich.console import Console

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


@dataclass
class HelpData:
    """Structured help data from CLI."""

    usage: str
    description: str
    parameter_groups: dict[str, list[ParameterInfo]]


def extract_plain_text(obj: Any) -> str:
    """Extract plain text from cyclopts objects."""
    if isinstance(obj, InlineText):
        console = Console(file=StringIO(), record=True, width=1000)
        console.print(obj)
        return console.export_text(clear=False, styles=False).replace("\n", "").strip()
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
    first_name = arg.names[0].lstrip("-").upper().replace("-", " ").title()
    if arg.required:
        first_name = f"{first_name} _(Required)_"
    return first_name


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
    if not arg.parameter.show_choices:
        return None

    # Handle enum types directly
    from enum import Enum
    from inspect import isclass

    if isclass(arg.hint) and issubclass(arg.hint, Enum):
        choices = list(arg.hint.__members__.values())
        return [f"`{choice}`" for choice in choices]

    return None


def _split_argument_names(names: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Split argument names into short and long options."""
    short_opts = [
        name for name in names if name.startswith("-") and not name.startswith("--")
    ]
    long_opts = [name for name in names if name.startswith("--")]
    return short_opts, long_opts


def _create_parameter_info(arg: Argument) -> ParameterInfo:
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
    )


def _extract_parameter_groups(
    argument_collection: ArgumentCollection,
) -> dict[str, list[ParameterInfo]]:
    """Extract parameter groups from argument collection."""
    groups: dict[str, list[ParameterInfo]] = defaultdict(list)

    for arg in argument_collection.filter_by(show=True):
        group_name = arg.parameter.group[0].name
        param_info = _create_parameter_info(arg)
        groups[group_name].append(param_info)

    return dict(groups)


def extract_help_data(subcommand: str) -> dict[str, list[ParameterInfo]]:
    """Extract structured help data from the CLI."""
    tokens = normalize_tokens(subcommand)
    _, apps, _ = app.parse_commands(tokens)

    argument_collection = apps[-1].assemble_argument_collection(
        apps=apps, parse_docstring=True
    )
    return _extract_parameter_groups(argument_collection)


def _format_parameter_options(param: ParameterInfo) -> list[str]:
    """Format parameter options for display."""
    options = []

    if param.short:
        options.append(f"{param.short}{param.type_suffix}")

    for option in param.long_options.split(" --"):
        option = option.strip()
        if option:
            if not option.startswith("--"):
                option = "--" + option.lower().replace(" ", "-")
            formatted = f"{option}{param.type_suffix}"
            if formatted not in options:
                options.append(formatted)

    if not options:
        return []

    combined = "<br>".join(f"`{opt}`" for opt in options)
    if param.display_name:
        return [f"#### {param.display_name}\n", f"{combined}\n"]
    else:
        return [f"#### {combined}\n"]


def _add_parameter_details(lines: list[str], param: ParameterInfo) -> None:
    """Add description with choices and default value inline."""
    description_parts = [param.description.strip().rstrip(".") + "."]

    if param.choices:
        choices_str = ", ".join(param.choices)
        description_parts.append(f"\n<br>**Choices:** {choices_str}.")

    if param.default_value and param.default_value != "False":
        description_parts.append(f"\n<br>**Default:** `{param.default_value}`.")

    full_description = " ".join(description_parts)
    lines.extend([full_description.strip() + "\n<hr>\n"])


def generate_markdown_docs(parameter_groups: dict[str, list[ParameterInfo]]) -> str:
    """Generate markdown documentation from help data."""
    lines = [
        "<!--",
        "# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "-->",
        "",
    ]

    lines.extend(["## Command Line Options\n"])

    for group_name, parameters in parameter_groups.items():
        lines.extend([f"### {group_name} Options\n"])

        for param in parameters:
            lines.extend(_format_parameter_options(param))
            _add_parameter_details(lines, param)

    return "\n".join([line.strip(" ") for line in lines]) + "\n"


def main():
    """Generate CLI documentation."""
    parameter_groups = extract_help_data("profile")
    markdown_content = generate_markdown_docs(parameter_groups)

    output_file = Path("docs/cli_options.md")
    output_file.write_text(markdown_content)
    print(f"Documentation written to {output_file}")


if __name__ == "__main__":
    main()
