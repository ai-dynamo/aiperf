# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate NVIDIA Apache-2.0 headers on every tracked first-party source file."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

EXEMPT_PATHS = {
    Path(".claude/skills"),
    Path("LICENSE"),
    Path("deploy/helm/aiperf-operator/LICENSE"),
    Path("src/aiperf/analysis/fzstd.umd.js"),
    Path("tools/COPYRIGHT"),
}
EXEMPT_PREFIXES = (
    Path("src/aiperf/api/static/vendor"),
    Path("src/aiperf/api/static-v2/vendor"),
    Path("src/aiperf/operator/ui/vendor"),
)
EXEMPT_SUFFIXES = {
    ".ipynb",
    ".jpg",
    ".json",
    ".jsonl",
    ".lock",
    ".png",
    ".svg",
    ".txt",
    ".wav",
    ".woff2",
    ".xlsx",
}
SOURCE_SUFFIXES = {
    ".bash",
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".html",
    ".js",
    ".md",
    ".mdc",
    ".mjs",
    ".mmd",
    ".proto",
    ".py",
    ".pyi",
    ".rst",
    ".sh",
    ".toml",
    ".tpl",
    ".tmpl",
    ".tsx",
    ".yaml",
    ".yml",
}
SOURCE_FILENAMES = {
    ".dockerignore",
    ".editorconfig",
    ".gitignore",
    ".helmignore",
    "CMakeLists.txt",
    "CODEOWNERS",
    "Dockerfile",
    "Makefile",
}
SOURCE_FILENAME_PREFIXES = ("Dockerfile.",)
HEADER_SCAN_LINES = 10
YEAR_PATTERN = r"(?P<start_year>\d{4})(?:-(?P<end_year>\d{4}))?"
COPYRIGHT_TEXT = (
    r"SPDX-FileCopyrightText: Copyright \(c\) "
    rf"{YEAR_PATTERN} NVIDIA CORPORATION & AFFILIATES\. All rights reserved\."
)
LICENSE_TEXT = r"SPDX-License-Identifier: Apache-2\.0"
COPYRIGHT_RE = re.compile(rf"^[^A-Za-z0-9]*{COPYRIGHT_TEXT}[^A-Za-z0-9]*$")
BASETEN_COPYRIGHT_RE = re.compile(
    rf"^[^A-Za-z0-9]*SPDX-FileCopyrightText: Copyright \(c\) {YEAR_PATTERN} "
    r"Baseten\.co, NVIDIA CORPORATION & AFFILIATES\. All rights reserved\."
    r"[^A-Za-z0-9]*$"
)
ANY_COPYRIGHT_RE = re.compile(r"^[^A-Za-z0-9]*SPDX-FileCopyrightText: .+[^A-Za-z0-9]*$")
LICENSE_RE = re.compile(rf"^[^A-Za-z0-9]*{LICENSE_TEXT}[^A-Za-z0-9]*$")
NVIDIA_ORG_RE = re.compile(r"\bNVIDIA CORPORATION\b", re.IGNORECASE)
C_STYLE_SUFFIXES = {
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
}


def is_under(path: Path, prefix: Path) -> bool:
    return path == prefix or prefix in path.parents


def is_exempt(root: Path, relative_path: Path) -> bool:
    path = root / relative_path
    return (
        relative_path in EXEMPT_PATHS
        or (
            not has_source_filename(relative_path)
            and relative_path.suffix.lower() in EXEMPT_SUFFIXES
        )
        or any(is_under(relative_path, prefix) for prefix in EXEMPT_PREFIXES)
        or path.is_symlink()
    )


def requires_header(path: Path) -> bool:
    return path.suffix.lower() in SOURCE_SUFFIXES or has_source_filename(path)


def has_source_filename(path: Path) -> bool:
    return path.name in SOURCE_FILENAMES or path.name.startswith(
        SOURCE_FILENAME_PREFIXES
    )


def has_ordered_year_range(match: re.Match[str]) -> bool:
    end_year = match.group("end_year")
    return end_year is None or int(match.group("start_year")) <= int(end_year)


def has_line_comment(line: str, marker: str) -> bool:
    tag_index = line.find("SPDX-")
    return tag_index >= 0 and line[:tag_index].strip() == marker


def is_inside_comment_block(
    lines: Sequence[str], line_index: int, opener: str, closer: str
) -> bool:
    text = "\n".join(lines)
    line_offset = sum(len(line) + 1 for line in lines[:line_index])
    tag_position = text.find("SPDX-", line_offset)
    if tag_position < 0:
        return False
    open_position = text.rfind(opener, 0, tag_position)
    close_position = text.rfind(closer, 0, tag_position)
    return open_position > close_position and text.find(closer, tag_position) >= 0


def has_valid_comment_syntax(
    path: Path, header: Sequence[str], line_indices: Iterable[int]
) -> bool:
    suffix = path.suffix.lower()
    for line_index in line_indices:
        line = header[line_index]
        if suffix in C_STYLE_SUFFIXES:
            valid = has_line_comment(line, "//") or is_inside_comment_block(
                header, line_index, "/*", "*/"
            )
        elif suffix == ".css":
            valid = is_inside_comment_block(header, line_index, "/*", "*/")
        elif suffix == ".html":
            valid = is_inside_comment_block(header, line_index, "<!--", "-->")
        elif suffix in {".md", ".mdc"}:
            valid = has_line_comment(line, "#") or is_inside_comment_block(
                header, line_index, "<!--", "-->"
            )
        elif suffix == ".mmd":
            valid = has_line_comment(line, "%%")
        elif suffix == ".rst":
            valid = has_line_comment(line, "..")
        elif suffix == ".tpl":
            valid = has_line_comment(line, "#") or is_inside_comment_block(
                header, line_index, "{{/*", "*/}}"
            )
        else:
            valid = has_line_comment(line, "#")
        if not valid:
            return False
    return True


def header_lines(path: Path, lines: Sequence[str]) -> Sequence[str]:
    scan_limit = HEADER_SCAN_LINES
    if path.suffix.lower() in {".md", ".mdc"} and lines[:1] == ["---"]:
        closing_index = next(
            (index for index, line in enumerate(lines[1:], start=1) if line == "---"),
            None,
        )
        if closing_index is not None:
            scan_limit += closing_index + 1
    return lines[:scan_limit]


def validate_file(root: Path, relative_path: Path) -> list[str]:
    if is_exempt(root, relative_path):
        return []
    if not requires_header(relative_path):
        return [f"{relative_path}: unsupported tracked file type"]

    path = root / relative_path
    try:
        contents = path.read_bytes()
    except OSError as error:
        return [f"{relative_path}: cannot read file: {error}"]
    if not contents.strip():
        return [f"{relative_path}: empty source file is missing an SPDX header"]
    if b"\x00" in contents:
        return [f"{relative_path}: binary files require an explicit policy exemption"]
    try:
        lines = contents.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError:
        return [f"{relative_path}: file is not valid UTF-8 text"]

    header = header_lines(relative_path, lines)
    copyright_index = next(
        (
            index
            for index, line in enumerate(header)
            if ANY_COPYRIGHT_RE.fullmatch(line)
        ),
        None,
    )
    if copyright_index is None:
        return [f"{relative_path}: malformed or missing copyright header"]

    license_index = copyright_index
    while license_index < len(header) and ANY_COPYRIGHT_RE.fullmatch(
        header[license_index]
    ):
        copyright_line = header[license_index]
        if NVIDIA_ORG_RE.search(copyright_line):
            match = COPYRIGHT_RE.fullmatch(copyright_line)
            if match is None:
                match = BASETEN_COPYRIGHT_RE.fullmatch(copyright_line)
            if match is None or not has_ordered_year_range(match):
                return [f"{relative_path}: malformed NVIDIA copyright header"]
        license_index += 1
    if license_index >= len(header) or not LICENSE_RE.fullmatch(header[license_index]):
        return [f"{relative_path}: malformed or missing Apache-2.0 header"]
    if not has_valid_comment_syntax(
        relative_path, header, range(copyright_index, license_index + 1)
    ):
        return [f"{relative_path}: SPDX header uses invalid comment syntax"]
    return []


def validation_results(
    root: Path, paths: Iterable[Path]
) -> list[tuple[Path, list[str]]]:
    return [
        (path, validate_file(root, path))
        for path in sorted(paths, key=lambda item: item.as_posix())
    ]


def validate_paths(root: Path, paths: Iterable[Path]) -> list[str]:
    return [
        violation
        for _, path_violations in validation_results(root, paths)
        for violation in path_violations
    ]


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [Path(path) for path in result.stdout.decode("utf-8").split("\0") if path]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root containing the tracked files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    try:
        paths = tracked_files(root)
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as error:
        print(f"unable to enumerate tracked files: {error}", file=sys.stderr)
        return 2

    results = validation_results(root, paths)
    violations = [
        violation for _, path_violations in results for violation in path_violations
    ]
    if violations:
        print("SPDX header violations:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        fixable_paths = [
            str(path)
            for path, path_violations in results
            if path_violations and not is_exempt(root, path) and requires_header(path)
        ]
        fix_command = shlex.join(["./tools/add_copyright.py", "--", *fixable_paths])
        print(
            f"\nAdd or repair supported headers with:\n  {fix_command}",
            file=sys.stderr,
        )
        return 1
    print(f"Validated SPDX policy for {len(paths)} tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
