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
    """Format the help output as markdown with proper header and code blocks."""
    # Split the output into sections based on the ╭─ ... ─╮ headers
    lines = help_output.split("\n")
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

    current_section = []
    in_section = False

    for line in lines:
        # Check if this is a section header line (starts with ╭─)
        if line.strip().startswith("╭─"):
            # If we were in a previous section, add it
            if current_section:
                markdown_lines.append("```")
                markdown_lines.extend(current_section)
                markdown_lines.append("```")
                current_section = []
            # Start new section
            in_section = True
            current_section.append(line)
        elif line.strip().startswith("╰─"):
            # End of section
            current_section.append(line)
            markdown_lines.append("```")
            markdown_lines.extend(current_section)
            markdown_lines.append("```")
            current_section = []
            in_section = False
        elif in_section:
            current_section.append(line)

    # Add any remaining section
    if current_section:
        markdown_lines.append("```")
        markdown_lines.extend(current_section)
        markdown_lines.append("```")

    return "\n".join(markdown_lines)


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
