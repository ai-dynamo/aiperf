#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate CLI docs for AIPerf."""

from io import StringIO
from pathlib import Path
from typing import Any

from cyclopts.bind import normalize_tokens
from cyclopts.help import format_doc, format_usage, resolve_help_format
from rich.console import Console

from aiperf.cli import app


def extract_plain_text_from_inline_text(description_obj) -> str:
    """Extract plain text specifically from cyclopts InlineText objects."""
    # Import here to check the type
    from cyclopts.help import InlineText

    # If it's an InlineText object, extract using Console
    if isinstance(description_obj, InlineText):
        string_io = StringIO()
        console = Console(file=string_io, record=True, width=1000)
        console.print(description_obj)
        return console.export_text(clear=False, styles=False).strip()

    # For other types, use the native cyclopts approach (str conversion)
    return str(description_obj) if description_obj else ""


def get_type_suffix_from_hint(hint) -> str:
    """Extract type suffix from Python type hint."""
    import typing
    from typing import get_args, get_origin

    if hint is None:
        return " <str>"

    # Handle basic types
    if hint == str:
        return " <str>"
    elif hint == int:
        return " <int>"
    elif hint == float:
        return " <float>"
    elif hint == bool:
        return ""  # Boolean flags don't take values

    # Handle typing constructs
    origin = get_origin(hint)
    if (
        origin is list
        or origin is list
        or origin is tuple
        or origin is tuple
        or origin is set
        or origin is set
    ):
        return " <list>"
    elif origin is typing.Union:
        # For Optional types (Union[X, None]), use the non-None type
        args = get_args(hint)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            return get_type_suffix_from_hint(non_none_args[0])

    # Default to string for unknown types
    return " <str>"


def extract_help_data(subcommand: str) -> dict[str, Any]:
    """Extract structured help data from the CLI."""
    tokens = normalize_tokens(subcommand)
    command_chain, apps, _ = app.parse_commands(tokens)
    executing_app = apps[-1]

    help_format = resolve_help_format(apps)

    # Extract usage
    usage = None
    if executing_app.usage is None:
        usage = format_usage(app, command_chain)
    elif executing_app.usage:
        usage = executing_app.usage

    # Extract description
    description = extract_plain_text_from_inline_text(
        format_doc(executing_app, help_format)
    )

    # Extract parameter groups by accessing argument collection directly
    parameter_groups = {}

    # Get the argument collection to access type information
    if executing_app.default_command:
        argument_collection = executing_app.assemble_argument_collection(
            apps=apps, parse_docstring=True
        )

        # Group arguments by their groups
        from collections import defaultdict

        groups = defaultdict(list)

        for argument in argument_collection.filter_by(show=True):
            # Group is a tuple of Group objects, take the first one
            group_name = (
                argument.parameter.group[0]._name
                if argument.parameter.group
                else "Parameters"
            )

            # Get type information from the hint
            type_suffix = get_type_suffix_from_hint(argument.hint)

            # Handle boolean flags specially
            if argument.hint == bool or (
                hasattr(argument, "field_info")
                and hasattr(argument.field_info, "default")
                and argument.field_info.default is False
            ):
                type_suffix = ""

            # Separate short and long options
            short_options = [
                name
                for name in argument.names
                if name.startswith("-") and not name.startswith("--")
            ]
            long_options = [name for name in argument.names if name.startswith("--")]

            # Build parameter info
            param_info = {
                "name": " --".join(long_options),  # Join long options with " --"
                "short": " ".join(short_options),
                "description": extract_plain_text_from_inline_text(
                    argument.parameter.help
                )
                if argument.parameter.help
                else "",
                "required": argument.required,
                "type_suffix": type_suffix,
            }

            groups[group_name].append(param_info)

        parameter_groups = dict(groups)

    return {
        "usage": usage,
        "description": description,
        "parameter_groups": parameter_groups,
    }


def generate_markdown_docs(help_data: dict[str, Any]) -> str:
    """Generate markdown documentation from help data."""
    md_content = []

    # Title and description
    md_content.append("# AIPerf CLI Reference")
    md_content.append("")
    md_content.append(
        "This document provides a comprehensive reference for all AIPerf CLI parameters."
    )
    md_content.append("")

    # Usage
    if help_data.get("usage"):
        md_content.append("## Usage")
        md_content.append("")
        md_content.append("```bash")
        md_content.append(str(help_data["usage"]).strip())
        md_content.append("```")
        md_content.append("")

    # Description
    if help_data.get("description"):
        md_content.append("## Description")
        md_content.append("")
        md_content.append(str(help_data["description"]).strip())
        md_content.append("")

    # Parameter groups
    if help_data.get("parameter_groups"):
        md_content.append("## Parameters")
        md_content.append("")

        for group_name, parameters in help_data["parameter_groups"].items():
            if not parameters:  # Skip empty groups
                continue

            md_content.append(f"### {group_name}")
            md_content.append("")

            for param in parameters:
                # Parse parameter name and aliases - the name field contains all options
                all_options = param["name"].split(" --")

                # Use the type suffix from the extracted data
                value_suffix = param.get("type_suffix", " <str>")

                # Collect unique options
                unique_options = []

                # Add short option if available
                if param["short"]:
                    unique_options.append(f"{param['short']}{value_suffix}")

                # Process all long options
                for option in all_options:
                    option = option.strip()
                    if option:
                        # Ensure it starts with --
                        if not option.startswith("--"):
                            option = "--" + option.lower().replace(" ", "-")
                        formatted_option = f"{option}{value_suffix}"
                        if formatted_option not in unique_options:
                            unique_options.append(formatted_option)

                # Combine all options on a single line
                if unique_options:
                    combined_options = " | ".join(f"`{opt}`" for opt in unique_options)
                    md_content.append(f"##### {combined_options}")

                md_content.append("")

                # Description (without "Description:" prefix)
                if param["description"]:
                    md_content.append(param["description"])
                    md_content.append("")

    return "\n".join(md_content)


def main():
    """Main function."""
    help_data = extract_help_data("profile")
    markdown_content = generate_markdown_docs(help_data)

    # Write to file
    output_file = Path("CLI_REFERENCE.md")
    output_file.write_text(markdown_content)
    print(f"Documentation written to {output_file}")

    # Also print to console for inspection
    print("\n" + "=" * 50)
    print("GENERATED MARKDOWN:")
    print("=" * 50)
    print(
        markdown_content[:2000] + "..."
        if len(markdown_content) > 2000
        else markdown_content
    )


if __name__ == "__main__":
    main()
