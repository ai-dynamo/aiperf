#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate CLI docs for AIPerf."""

from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from cyclopts.bind import normalize_tokens
from cyclopts.help import InlineText, format_doc, format_usage, resolve_help_format
from rich.console import Console

from aiperf.cli import app


@dataclass
class ParameterInfo:
    """Information about a CLI parameter."""

    name: str
    short: str
    description: str
    required: bool
    type_suffix: str


@dataclass
class HelpData:
    """Structured help data from CLI."""

    usage: str
    description: str
    parameter_groups: dict[str, list[ParameterInfo]]


def extract_plain_text(obj) -> str:
    """Extract plain text from cyclopts objects."""
    if isinstance(obj, InlineText):
        console = Console(file=StringIO(), record=True, width=1000)
        console.print(obj)
        return console.export_text(clear=False, styles=False).strip()
    return str(obj) if obj else ""


def get_type_suffix(hint) -> str:
    """Get type suffix for parameter hints."""
    type_mapping = {
        bool: "",
        int: " <int>",
        float: " <float>",
        list: " <list>",
        tuple: " <list>",
        set: " <list>",
    }

    # Check direct type first, then origin type for generics
    lookup_type = hint if hint in type_mapping else getattr(hint, "__origin__", None)
    return type_mapping.get(lookup_type, " <str>")


def extract_help_data(subcommand: str) -> HelpData:
    """Extract structured help data from the CLI."""
    tokens = normalize_tokens(subcommand)
    command_chain, apps, _ = app.parse_commands(tokens)
    executing_app = apps[-1]
    help_format = resolve_help_format(apps)

    # Extract usage and description
    usage = executing_app.usage or format_usage(app, command_chain)
    description = extract_plain_text(format_doc(executing_app, help_format))

    # Extract parameter groups
    parameter_groups = {}
    if executing_app.default_command:
        argument_collection = executing_app.assemble_argument_collection(
            apps=apps, parse_docstring=True
        )
        groups = defaultdict(list)

        for arg in argument_collection.filter_by(show=True):
            group_name = (
                arg.parameter.group[0]._name if arg.parameter.group else "Parameters"
            )
            type_suffix = get_type_suffix(arg.hint)

            short_opts = [
                name
                for name in arg.names
                if name.startswith("-") and not name.startswith("--")
            ]
            long_opts = [name for name in arg.names if name.startswith("--")]

            param_info = ParameterInfo(
                name=" --".join(long_opts),
                short=" ".join(short_opts),
                description=extract_plain_text(arg.parameter.help),
                required=arg.required,
                type_suffix=type_suffix,
            )
            groups[group_name].append(param_info)

        parameter_groups = dict(groups)

    return HelpData(
        usage=usage, description=description, parameter_groups=parameter_groups
    )


def generate_markdown_docs(help_data: HelpData) -> str:
    """Generate markdown documentation from help data."""
    lines = [
        "# AIPerf CLI Reference",
        "",
        "This document provides a comprehensive reference for all AIPerf CLI parameters.",
        "",
    ]

    # Usage section
    if help_data.usage:
        lines.extend(
            ["## Usage", "", "```bash", str(help_data.usage).strip(), "```", ""]
        )

    # Description section
    if help_data.description:
        lines.extend(["## Description", "", str(help_data.description).strip(), ""])

    # Parameters section
    if help_data.parameter_groups:
        lines.extend(["## Parameters", ""])

        for group_name, parameters in help_data.parameter_groups.items():
            if not parameters:
                continue

            lines.extend([f"### {group_name}", ""])

            for param in parameters:
                options = []
                value_suffix = param.type_suffix

                # Add short option
                if param.short:
                    options.append(f"{param.short}{value_suffix}")

                # Add long options
                for option in param.name.split(" --"):
                    option = option.strip()
                    if option:
                        if not option.startswith("--"):
                            option = "--" + option.lower().replace(" ", "-")
                        formatted = f"{option}{value_suffix}"
                        if formatted not in options:
                            options.append(formatted)

                # Format options
                if options:
                    combined = " | ".join(f"`{opt}`" for opt in options)
                    lines.extend([f"##### {combined}", ""])

                # Add description
                if param.description:
                    lines.extend([param.description, ""])

    return "\n".join(lines)


def main():
    """Generate CLI documentation."""
    help_data = extract_help_data("profile")
    markdown_content = generate_markdown_docs(help_data)

    output_file = Path("CLI_REFERENCE.md")
    output_file.write_text(markdown_content)
    print(f"Documentation written to {output_file}")

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
