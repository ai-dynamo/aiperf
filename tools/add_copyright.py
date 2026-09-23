#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add or update NVIDIA copyright headers in source files.

Usage:
    ./tools/add_copyright.py file1.py file2.py
    ./tools/add_copyright.py --check file1.py    # Check only, don't modify
    ./tools/add_copyright.py --dry-run file1.py  # Show what would change
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

# Standalone-compatible imports (works with pre-commit without full tools module)
try:
    from tools._core import console, print_generated, print_up_to_date, print_warning
except ImportError:
    try:
        from rich.console import Console

        console = Console()

        def print_generated(path: Path | str) -> None:
            console.print(f"  [green]✓[/] {path}")

        def print_up_to_date(name: str) -> None:
            console.print(f"  [dim]•[/] {name} [dim](up-to-date)[/]")

        def print_warning(msg: str) -> None:
            console.print(f"  [yellow]⚠[/] {msg}")

    except ImportError:
        # Minimal fallback for pre-commit (no rich available)
        class _PlainConsole:
            @staticmethod
            def print(msg: str) -> None:
                # Strip rich markup
                import re

                print(re.sub(r"\[/?[^\]]+\]", "", msg))

        console = _PlainConsole()  # type: ignore[assignment]

        def print_generated(path: Path | str) -> None:
            print(f"  ✓ {path}")

        def print_up_to_date(name: str) -> None:
            print(f"  • {name} (up-to-date)")

        def print_warning(msg: str) -> None:
            print(f"  ⚠ {msg}", file=sys.stderr)

# =============================================================================
# Configuration
# =============================================================================

CURRENT_YEAR = str(datetime.now().year)
COPYRIGHT_FILE = Path(__file__).parent / "COPYRIGHT"

CANONICAL_NVIDIA_COPYRIGHT_PAT = re.compile(
    r"SPDX-FileCopyrightText: Copyright \(c\) "
    r"(?:(\d{4})-)?(\d{4}) NVIDIA CORPORATION & AFFILIATES\. All rights reserved\."
)
NVIDIA_COPYRIGHT_PAT = re.compile(
    r"SPDX-FileCopyrightText:[ \t]*Copyright(?:[ \t]+\(c\))?[ \t]+"
    r"(?:(\d{4})-)?(\d{4})[ \t]+NVIDIA CORPORATION"
    r"(?:[ \t]*&[ \t]*AFFILIATES)?"
    r"(?:\.[ \t]*(?:All rights reserved\.)?)?",
    re.IGNORECASE,
)
LICENSE_IDENTIFIER_PAT = re.compile(
    r"SPDX-License-Identifier:[ \t]*"
    r"(?P<expression>[A-Za-z0-9.+-]+"
    r"(?:[ \t]+(?:AND|OR|WITH)[ \t]+[A-Za-z0-9.+-]+)*)"
)
SPDX_COMMENT_AFFIX_PAT = re.compile(r"^[\s#/%.*<>{}!~-]*$")

# =============================================================================
# Copyright Utilities
# =============================================================================


def has_nvidia_copyright(content: str) -> bool:
    """Check if content has an NVIDIA copyright header."""
    return bool(NVIDIA_COPYRIGHT_PAT.search(content))


def was_modified_this_year(path: Path) -> bool:
    """Check if file was modified in the current year (via git).

    Returns True if:
    - File has uncommitted changes (staged or unstaged)
    - File's last commit was in the current year
    """
    try:
        # Check for uncommitted changes (staged or unstaged)
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode == 0 and status.stdout.strip():
            return True  # Has uncommitted changes

        # Check last commit year
        log = subprocess.run(
            ["git", "log", "-1", "--format=%cd", "--date=format:%Y", "--", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        return log.returncode == 0 and log.stdout.strip() == CURRENT_YEAR
    except OSError:
        # git not available, assume modified
        return True


def get_license_text() -> str:
    """Get the license text from COPYRIGHT file."""
    if not COPYRIGHT_FILE.exists():
        raise FileNotFoundError(f"COPYRIGHT file not found: {COPYRIGHT_FILE}")
    return COPYRIGHT_FILE.read_text().strip()


def update_copyright_year(
    content: str,
    disallow_range: bool = False,
    match: re.Match[str] | None = None,
) -> str:
    """Update NVIDIA copyright year in content.

    Updates the supplied match, or the first occurrence when no match is supplied.

    Args:
        content: File content to update
        disallow_range: If True, use single year instead of range

    Returns:
        Updated content (or original if no change needed)
    """
    match = match or NVIDIA_COPYRIGHT_PAT.search(content)
    if not match:
        return content

    min_year = match.group(1) or match.group(2)

    # Build new copyright text
    if min_year < CURRENT_YEAR and not disallow_range:
        year_part = f"{min_year}-{CURRENT_YEAR}"
    else:
        year_part = CURRENT_YEAR

    new_copyright = (
        "SPDX-FileCopyrightText: Copyright (c) "
        f"{year_part} NVIDIA CORPORATION & AFFILIATES. All rights reserved."
    )

    return content[: match.start()] + new_copyright + content[match.end() :]


# =============================================================================
# Header Insertion
# =============================================================================


def prefix_lines(content: str, prefix: str) -> str:
    """Add prefix to each line of content."""
    return prefix + f"\n{prefix}".join(content.splitlines())


def split_bom(content: str) -> tuple[str, str]:
    """Separate a UTF-8 BOM so it remains the first character in the file."""
    if content.startswith("\ufeff"):
        return "\ufeff", content[1:]
    return "", content


def insert_after_script_preamble(header: str, content: str) -> str:
    """Preserve a shebang and Python encoding cookie before the header."""
    bom, content = split_bom(content)
    lines = content.splitlines(keepends=True)
    line_index = 0
    if lines and lines[0].startswith("#!"):
        line_index = 1
    encoding_index = next(
        (
            index
            for index, line in enumerate(lines[:2])
            if re.match(r"^[ \t\f]*#.*?coding[:=][ \t]*[-_.a-zA-Z0-9]+", line)
        ),
        None,
    )
    if encoding_index is not None:
        line_index = max(line_index, encoding_index + 1)
    pos = sum(len(line) for line in lines[:line_index])
    return bom + content[:pos] + header + "\n" + content[pos:]


def prepend_header(header: str, content: str) -> str:
    """Insert header at the start of content."""
    bom, content = split_bom(content)
    return bom + header + "\n" + content


def insert_markdown_header(license_text: str, content: str) -> str:
    """Keep complete YAML frontmatter ahead of an HTML SPDX comment."""
    bom, content = split_bom(content)
    header = "<!--\n" + license_text + "\n-->"
    frontmatter = re.match(r"^---\n.*?^---(?:\n|$)", content, re.MULTILINE | re.DOTALL)
    if frontmatter is not None:
        pos = frontmatter.end()
        separator = "" if content[:pos].endswith("\n") else "\n"
        return bom + content[:pos] + separator + header + "\n" + content[pos:]
    return bom + header + "\n" + content


def insert_after_docker_directives(header: str, content: str) -> str:
    """Preserve leading Docker parser directives before the header."""
    bom, content = split_bom(content)
    lines = content.splitlines(keepends=True)
    line_index = 0
    while line_index < len(lines) and re.match(
        r"^#\s*(?:syntax|escape|check)\s*=", lines[line_index], re.IGNORECASE
    ):
        line_index += 1
    pos = sum(len(line) for line in lines[:line_index])
    return bom + content[:pos] + header + "\n" + content[pos:]


def insert_after_html_doctype(header: str, content: str) -> str:
    """Preserve a leading HTML doctype before the header."""
    bom, content = split_bom(content)
    match = re.match(r"(?i:<!doctype\s+html[^>]*>)\s*\n?", content)
    if match is None:
        return bom + header + "\n" + content
    return bom + content[: match.end()] + header + "\n" + content[match.end() :]


def insert_after_css_charset(header: str, content: str) -> str:
    """Preserve a leading CSS charset declaration before the header."""
    bom, content = split_bom(content)
    match = re.match(r"@charset\s+(['\"]).+?\1;\s*\n?", content, re.IGNORECASE)
    if match is None:
        return bom + header + "\n" + content
    return bom + content[: match.end()] + header + "\n" + content[match.end() :]


# =============================================================================
# File Type Handlers
# =============================================================================

# Maps path matcher -> (header_formatter, inserter)
FileHandler = tuple[Callable[[str], str], Callable[[str, str], str]]
FILE_HANDLERS: dict[Callable[[str], bool], FileHandler] = {}


def has_ext(exts: Sequence[str]) -> Callable[[str], bool]:
    """Match files by extension."""
    return lambda p: Path(p).suffix.lower() in exts


def basename_is(name: str) -> Callable[[str], bool]:
    """Match files by basename."""
    return lambda p: Path(p).name == name


def basename_starts_with(prefix: str) -> Callable[[str], bool]:
    return lambda p: Path(p).name.startswith(prefix)


def any_of(*funcs: Callable[[str], bool]) -> Callable[[str], bool]:
    """Match if any function matches."""
    return lambda p: any(f(p) for f in funcs)


def register(
    match: Callable[[str], bool],
    formatter: Callable[[str], str],
    inserter: Callable[[str, str], str] = prepend_header,
) -> None:
    """Register a file type handler."""
    FILE_HANDLERS[match] = (formatter, inserter)


# Register handlers for different file types
register(
    any_of(
        has_ext(
            [
                ".bash",
                ".pbtxt",
                ".py",
                ".pyi",
                ".sh",
                ".toml",
                ".tmpl",
                ".yaml",
                ".yml",
            ]
        ),
        basename_is(".dockerignore"),
        basename_is(".editorconfig"),
        basename_is(".gitignore"),
        basename_is(".helmignore"),
        basename_is("CMakeLists.txt"),
        basename_is("CODEOWNERS"),
        basename_is("Makefile"),
    ),
    lambda lic: prefix_lines(lic, "# "),
    insert_after_script_preamble,
)
register(
    any_of(basename_is("Dockerfile"), basename_starts_with("Dockerfile.")),
    lambda lic: prefix_lines(lic, "# "),
    insert_after_docker_directives,
)
register(
    has_ext(
        [
            ".c",
            ".cc",
            ".cpp",
            ".cu",
            ".cuh",
            ".h",
            ".hpp",
            ".js",
            ".mjs",
            ".proto",
            ".tsx",
        ]
    ),
    lambda lic: prefix_lines(lic, "// "),
    insert_after_script_preamble,
)
register(
    has_ext([".css"]),
    lambda lic: "/* " + lic.replace("\n", "\n   ") + " */",
    insert_after_css_charset,
)
register(has_ext([".mmd"]), lambda lic: prefix_lines(lic, "%% "))
register(has_ext([".tpl"]), lambda lic: "{{/*\n" + lic + "\n*/}}")
register(
    has_ext([".html"]),
    lambda lic: "<!--\n" + lic + "\n-->",
    insert_after_html_doctype,
)
register(
    has_ext([".md", ".mdc"]),
    lambda lic: lic,
    insert_markdown_header,
)
register(has_ext([".rst"]), lambda lic: prefix_lines(lic, ".. "))


def get_handler(path: str) -> FileHandler | None:
    """Get the handler for a file path."""
    for matcher, handler in FILE_HANDLERS.items():
        if matcher(path):
            return handler
    return None


def _line_header_span(content: str, copyright_match: re.Match[str]) -> tuple[int, int]:
    start = content.rfind("\n", 0, copyright_match.start()) + 1
    newline = content.find("\n", copyright_match.end())
    end = len(content) if newline < 0 else newline + 1
    next_newline = content.find("\n", end)
    next_end = len(content) if next_newline < 0 else next_newline + 1
    next_line = content[end:next_end]
    license_match = LICENSE_IDENTIFIER_PAT.search(next_line)
    if (
        license_match is not None
        and SPDX_COMMENT_AFFIX_PAT.fullmatch(next_line[: license_match.start()])
        and SPDX_COMMENT_AFFIX_PAT.fullmatch(next_line[license_match.end() :])
    ):
        end = next_end
    return start, end


def _spdx_header_span(content: str, copyright_match: re.Match[str]) -> tuple[int, int]:
    delimiters = (
        ("<!--", "-->"),
        ("{{/*", "*/}}"),
        ("/*", "*/"),
    )
    for opener, closer in delimiters:
        start = content.rfind(opener, 0, copyright_match.start())
        previous_close = content.rfind(closer, 0, copyright_match.start())
        close = content.find(closer, copyright_match.end())
        if start >= 0 and start > previous_close and close >= 0:
            end = close + len(closer)
            if end < len(content) and content[end] == "\n":
                end += 1
            return start, end
    return _line_header_span(content, copyright_match)


def _match_has_line_comment(content: str, match: re.Match[str], marker: str) -> bool:
    line_start = content.rfind("\n", 0, match.start()) + 1
    return content[line_start : match.start()].strip() == marker


def _match_is_inside_comment_block(
    content: str,
    match: re.Match[str],
    opener: str,
    closer: str,
) -> bool:
    open_position = content.rfind(opener, 0, match.start())
    close_position = content.rfind(closer, 0, match.start())
    return open_position > close_position and content.find(closer, match.end()) >= 0


def _match_is_in_header(path: Path, content: str, match: re.Match[str]) -> bool:
    _, content_without_bom = split_bom(content)
    bom_offset = len(content) - len(content_without_bom)
    line_index = content_without_bom.count("\n", 0, match.start() - bom_offset)
    scan_limit = 10
    lines = content_without_bom.splitlines()
    if path.suffix.lower() in {".md", ".mdc"} and lines[:1] == ["---"]:
        closing_index = next(
            (index for index, line in enumerate(lines[1:], start=1) if line == "---"),
            None,
        )
        if closing_index is not None:
            scan_limit += closing_index + 1
    return line_index < scan_limit


def _has_valid_match_syntax(
    path: Path,
    content: str,
    match: re.Match[str],
    rendered_header: str,
) -> bool:
    if rendered_header.startswith("// "):
        return _match_has_line_comment(content, match, "//") or (
            _match_is_inside_comment_block(content, match, "/*", "*/")
        )
    if rendered_header.startswith("/* "):
        return _match_is_inside_comment_block(content, match, "/*", "*/")
    if rendered_header.startswith("<!--"):
        return _match_is_inside_comment_block(content, match, "<!--", "-->") or (
            path.suffix.lower() in {".md", ".mdc"}
            and _match_has_line_comment(content, match, "#")
        )
    if rendered_header.startswith("{{/*"):
        return _match_is_inside_comment_block(content, match, "{{/*", "*/}}") or (
            _match_has_line_comment(content, match, "#")
        )

    rendered_line = next(
        line
        for line in rendered_header.splitlines()
        if "SPDX-FileCopyrightText:" in line
    )
    marker = rendered_line[: rendered_line.find("SPDX-FileCopyrightText:")].strip()
    return _match_has_line_comment(content, match, marker)


def _has_valid_copyright_syntax(
    path: Path,
    content: str,
    match: re.Match[str],
    rendered_header: str,
) -> bool:
    return _match_is_in_header(path, content, match) and _has_valid_match_syntax(
        path, content, match, rendered_header
    )


def _find_header_copyright(
    path: Path, content: str, rendered_header: str
) -> re.Match[str] | None:
    return next(
        (
            match
            for match in NVIDIA_COPYRIGHT_PAT.finditer(content)
            if _has_valid_copyright_syntax(path, content, match, rendered_header)
        ),
        None,
    )


def _has_valid_spdx_syntax(path: Path, header: str, rendered_header: str) -> bool:
    tags = ("SPDX-FileCopyrightText:", "SPDX-License-Identifier:")
    copyright_match = re.search(tags[0], header)
    license_match = re.search(tags[1], header)
    if copyright_match is None or license_match is None:
        return False
    copyright_line = header.count("\n", 0, copyright_match.start())
    license_line = header.count("\n", 0, license_match.start())
    return license_line == copyright_line + 1 and all(
        _has_valid_match_syntax(path, header, match, rendered_header)
        for match in (copyright_match, license_match)
    )


def _repair_spdx_license(
    path: Path,
    content: str,
    formatter: Callable[[str], str],
    inserter: Callable[[str, str], str],
) -> tuple[str, bool]:
    plain_header = (
        "SPDX-FileCopyrightText: Copyright (c) "
        f"{CURRENT_YEAR} NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "SPDX-License-Identifier: Apache-2.0"
    )
    rendered_header = inserter(formatter(plain_header), "").rstrip("\n")
    copyright_match = _find_header_copyright(path, content, rendered_header)
    if copyright_match is None:
        return content, False

    plain_header = copyright_match.group(0) + "\nSPDX-License-Identifier: Apache-2.0"
    rendered_header = inserter(formatter(plain_header), "").rstrip("\n")
    start, end = _spdx_header_span(content, copyright_match)
    existing_header = content[start:end]
    license_match = LICENSE_IDENTIFIER_PAT.search(existing_header)
    if (
        license_match is not None
        and license_match.group("expression") == "Apache-2.0"
        and _has_valid_spdx_syntax(path, existing_header, rendered_header)
    ):
        return content, False

    trailing_newline = "\n" if existing_header.endswith("\n") else ""
    return content[:start] + rendered_header + trailing_newline + content[end:], True


def _files_to_process(files: Sequence[str], args_env: str | None) -> list[str]:
    if args_env is None:
        return list(files)
    return [*files, *shlex.split(os.environ.get(args_env, ""))]


# =============================================================================
# Main Processing
# =============================================================================


def process_file(
    path: Path,
    license_text: str,
    *,
    check: bool = False,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Process a single file.

    Returns:
        (changed, status_message)
    """
    if not path.exists():
        return False, f"not found: {path}"

    handler = get_handler(str(path))
    if not handler:
        return False, f"no handler: {path}"

    content = path.read_text()
    formatter, inserter = handler

    example_header = inserter(formatter(license_text), "").rstrip("\n")
    legacy_match = _find_header_copyright(path, content, example_header)
    if legacy_match is not None:
        canonical_match = CANONICAL_NVIDIA_COPYRIGHT_PAT.fullmatch(
            legacy_match.group(0)
        )
        is_canonical = canonical_match is not None and (
            canonical_match.group(1) is None
            or int(canonical_match.group(1)) <= int(canonical_match.group(2))
        )
        updated_year = update_copyright_year(content, match=legacy_match)
        updated, repaired_license = _repair_spdx_license(
            path, updated_year, formatter, inserter
        )
        if content == updated:
            return False, "up-to-date"

        # Only update year if file was actually modified this year
        if not repaired_license and is_canonical and not was_modified_this_year(path):
            return False, "up-to-date (not modified this year)"

        if check:
            if repaired_license:
                return True, "needs SPDX header repair"
            return True, "needs year update"
        if dry_run:
            if repaired_license:
                return True, "would repair SPDX header"
            return True, f"would update year to {CURRENT_YEAR}"

        path.write_text(updated)
        if repaired_license:
            return True, "repaired SPDX header"
        return True, "updated year"

    # Add new copyright header
    header = formatter(license_text)
    updated = inserter(header, content)

    # Sanity check
    if _find_header_copyright(path, updated, example_header) is None:
        return False, "WARNING: No valid NVIDIA copyright after insertion"

    if check:
        return True, "needs copyright header"
    if dry_run:
        return True, "would add copyright header"

    path.write_text(updated)
    return True, "added copyright"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add or update NVIDIA copyright headers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="*", help="Files to process")
    parser.add_argument(
        "--check", action="store_true", help="Check only, exit 1 if changes needed"
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Show what would change"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all files")
    parser.add_argument("--args-env", help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        files = _files_to_process(args.files, args.args_env)
    except ValueError as error:
        parser.error(f"invalid argument string: {error}")

    if not files:
        parser.print_help()
        return 0

    try:
        license_text = get_license_text()
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/] {e}")
        return 1

    changed_count = 0
    error_count = 0

    for file_path in files:
        path = Path(file_path)
        changed, status = process_file(
            path, license_text, check=args.check, dry_run=args.dry_run
        )

        if status.startswith("WARNING"):
            print_warning(f"{path}: {status}")
            error_count += 1
        elif status.startswith("not found") or status.startswith("no handler"):
            print_warning(f"{status}")
            error_count += 1
        elif changed:
            if args.check or args.dry_run:
                console.print(f"  [yellow]![/] {path}: {status}")
            else:
                print_generated(path)
            changed_count += 1
        elif args.verbose:
            print_up_to_date(f"{path.name}")

    # Summary
    if args.check and changed_count:
        console.print(f"\n[yellow]{changed_count}[/] file(s) need updates.")
        return 1

    if changed_count and not args.check:
        action = "would update" if args.dry_run else "updated"
        console.print(f"\n[green]✓[/] {action} {changed_count} file(s)")

    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())
