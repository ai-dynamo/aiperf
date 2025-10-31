#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate the CLI options documentation from the aiperf profile --help output.

This script runs `aiperf profile --help` and formats the output into a markdown file.
It should be run from the repository root.

Usage:
    python tools/generate_cli_options_md.py [--check]

Options:
    --check     Check if the current cli_options.md matches the generated output.
                Returns exit code 1 if they differ, 0 if they match.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_help_output() -> str:
    """Run aiperf profile --help and return the output."""
    try:
        result = subprocess.run(
            ["aiperf", "profile", "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running aiperf profile --help: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(
            "Error: aiperf command not found. Make sure it's installed and in your PATH.",
            file=sys.stderr,
        )
        sys.exit(1)


def format_help_as_markdown(help_output: str) -> str:
    """Format the help output as markdown tables."""
    # Parse the help output into sections
    sections = parse_help_sections(help_output)

    markdown_lines = [
        "<!--",
        "SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "SPDX-License-Identifier: Apache-2.0",
        "-->",
        "",
        "# CLI Options",
        "Use these options to profile with AIPerf.",
        "",
    ]

    # Convert each section to a markdown table
    for section_name, options in sections.items():
        if not options:
            continue

        markdown_lines.append(f"## {section_name}")
        markdown_lines.append("")

        # Create table header
        markdown_lines.append("| Option | Description |")
        markdown_lines.append("|:-------|:-----------:|")

        # Add each option as a table row
        for option in options:
            option_col = format_option_column(option)
            desc_col = option["description"]
            markdown_lines.append(f"| {option_col} | {desc_col} |")

        markdown_lines.append("")

    return "\n".join(markdown_lines)


def parse_help_sections(help_output: str) -> dict:
    """Parse CLI help output into sections with options.

    Returns:
        Dict mapping section names to lists of option dicts
    """
    lines = help_output.split("\n")
    sections = {}
    current_section = None
    current_section_lines = []
    in_section = False

    for line in lines:
        # Check if this is a section header (╭─ ... ─╮)
        if line.strip().startswith("╭─"):
            # Save previous section if exists
            if current_section and current_section_lines:
                sections[current_section] = parse_section_options(current_section_lines)

            # Extract section name
            title_start = line.find("─ ") + 2
            title_end = line.rfind(" ─")
            if title_start > 1 and title_end > title_start:
                current_section = line[title_start:title_end].strip()
            else:
                current_section = "Options"

            current_section_lines = []
            in_section = True

        elif line.strip().startswith("╰─"):
            # End of section - save it
            if current_section and current_section_lines:
                sections[current_section] = parse_section_options(current_section_lines)
            current_section = None
            current_section_lines = []
            in_section = False

        elif in_section and line.strip():
            # Content line - strip box borders
            if len(line) > 4 and line.startswith("│") and line.endswith("│"):
                content = line[2:-2]  # Remove │ and surrounding spaces
                current_section_lines.append(content)

    # Add any remaining section (shouldn't happen with proper box format)
    if current_section and current_section_lines:
        sections[current_section] = parse_section_options(current_section_lines)

    return sections


def parse_section_options(lines: list[str]) -> list[dict]:
    """Parse option lines into structured option dictionaries.

    Returns:
        List of dicts with 'name', 'aliases', 'short', 'description', 'required'
    """
    options = []
    current_option = None

    for line in lines:
        # Detect new option based on indentation:
        # - Starts with * (required, no leading space)
        # - Starts with exactly 3 spaces (some options in Endpoint section)
        # - Starts with NO spaces and uppercase (options in Input, Output, etc.)
        # - More than 3 spaces or starts with many spaces = continuation
        is_new_option = False

        if not line or line.isspace():
            continue

        if line[0] == "*":
            # Required option (no leading space)
            is_new_option = True
        elif line.startswith("   ") and not line.startswith("    "):
            # Exactly 3 spaces = new option (Endpoint section style)
            is_new_option = True
        elif not line.startswith(" ") and line[0].isupper():
            # No leading space and uppercase = new option (Input/Output section style)
            is_new_option = True

        if is_new_option:
            # Save previous option
            if current_option:
                options.append(current_option)

            # Start new option
            current_option = parse_option_line(line.lstrip())
        elif current_option:
            # Continuation of description
            desc = line.strip()
            if desc:
                # Add space only if description already has content
                if current_option["description"]:
                    current_option["description"] += " " + desc
                else:
                    current_option["description"] = desc

    # Add the last option
    if current_option:
        options.append(current_option)

    return options


def parse_option_line(line: str) -> dict:
    """Parse a single option line into components.

    Returns:
        Dict with 'name', 'aliases', 'short', 'description', 'required'
    """
    import re

    option = {
        "name": "",
        "aliases": [],
        "short": "",
        "description": "",
        "required": False,
    }

    # Check if required (starts with *)
    if line.lstrip().startswith("*"):
        option["required"] = True
        line = line.lstrip()[1:].lstrip()  # Remove the *

    # Split on multiple spaces to separate option names from description
    parts = re.split(r"\s{2,}", line.strip())

    if not parts:
        return option

    # First part contains option names
    option_names = parts[0]

    # Extract option name and aliases
    # Pattern: OPTION-NAME --long-name --alias -s
    tokens = option_names.split()

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        if token.startswith("--"):
            # Long option
            if not option["name"]:
                option["name"] = token
            else:
                option["aliases"].append(token)
        elif token.startswith("-") and len(token) == 2:
            # Short option
            option["short"] = token
        elif (token.isupper() or (token and token[0].isupper())) and not option["name"]:
            # Environment variable style name
            option["name"] = token

    # Description is everything after the option names
    if len(parts) > 1:
        option["description"] = " ".join(parts[1:])

    return option


def format_option_column(option: dict) -> str:
    """Format the option column with name, aliases, and required marker.

    Returns:
        Formatted string for the option column
    """
    parts = []

    # Required marker
    if option["required"]:
        parts.append("**`*`**")

    # Main option name
    if option["name"]:
        name = option["name"]
        # Format as code
        if name.startswith("--") or name.startswith("-"):
            parts.append(f"`{name}`")
        else:
            parts.append(f"**{name}**")

    # Aliases
    for alias in option["aliases"]:
        parts.append(f"`{alias}`")

    # Short option
    if option["short"]:
        parts.append(f"`{option['short']}`")

    return "<br>".join(parts) if parts else ""


def stage_file(file_path: Path) -> None:
    """Stage a file using git add."""
    try:
        subprocess.run(
            ["git", "add", str(file_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"✓ Staged {file_path} for commit", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(
            f"Warning: Could not stage {file_path}: {e.stderr}",
            file=sys.stderr,
        )
    except FileNotFoundError:
        print(
            "Warning: git command not found, file not staged",
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLI options documentation from aiperf profile --help"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if the current cli_options.md matches the generated output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/cli_options.md"),
        help="Output file path (default: docs/cli_options.md)",
    )
    parser.add_argument(
        "--no-stage",
        action="store_true",
        help="Don't automatically stage the file with git add (default: auto-stage)",
    )
    args = parser.parse_args()

    # Get the help output
    print("Running aiperf profile --help...", file=sys.stderr)
    help_output = get_help_output()

    # Format as markdown
    print("Formatting output as markdown...", file=sys.stderr)
    markdown = format_help_as_markdown(help_output)

    if args.check:
        # Check mode: compare with existing file
        if not args.output.exists():
            print(
                f"Error: {args.output} does not exist. Run without --check to generate it.",
                file=sys.stderr,
            )
            sys.exit(1)

        current_content = args.output.read_text()
        if current_content.strip() == markdown.strip():
            print(f"✓ {args.output} is up to date!", file=sys.stderr)
            sys.exit(0)
        else:
            print(
                f"✗ {args.output} is out of sync with aiperf profile --help output!",
                file=sys.stderr,
            )
            print(
                "  Run 'make update-cli-docs' or 'python tools/generate_cli_options_md.py' to update it.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # Write mode: write to file and optionally stage it
        file_existed = args.output.exists()
        if file_existed:
            old_content = args.output.read_text()

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)

        # Check if content actually changed
        content_changed = not file_existed or old_content != markdown

        if content_changed:
            print(f"✓ Generated {args.output}", file=sys.stderr)

            # Auto-stage the file unless --no-stage is specified
            if not args.no_stage:
                stage_file(args.output)
        else:
            print(f"✓ {args.output} already up to date", file=sys.stderr)

        sys.exit(0)


if __name__ == "__main__":
    main()
