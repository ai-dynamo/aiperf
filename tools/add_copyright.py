# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, Optional, Sequence

current_year = str(datetime.now().year)

ROOT_DIR = os.path.dirname(__file__)

LICENSE_PATH = os.path.join(ROOT_DIR, "COPYRIGHT")

COPYRIGHT_YEAR_PAT = re.compile(
    r"SPDX-FileCopyrightText: Copyright \(c\) (\d{4})?-?(\d{4}) NVIDIA CORPORATION"
)


def has_copyright(content: str) -> bool:
    return bool(COPYRIGHT_YEAR_PAT.search(content))


def update_copyright_year(
    path: str, content: Optional[str] = None, disallow_range: bool = False
) -> None:
    """
    Updates the copyright year in the provided file.
    If the copyright is not present in the file, this function has no effect.
    """
    if content is None:
        with open(path, "r") as f:
            content = f.read()

    match = COPYRIGHT_YEAR_PAT.search(content)
    assert match is not None, f"File {path} does not contain a valid copyright."
    min_year = match.groups()[0] or match.groups()[1]
    new_copyright = "SPDX-FileCopyrightText: Copyright (c) "
    if int(min_year) < int(current_year) and not disallow_range:
        new_copyright += f"{min_year}-{current_year}"
    else:
        new_copyright += f"{current_year}"
    new_copyright += " NVIDIA CORPORATION"

    updated_content = COPYRIGHT_YEAR_PAT.sub(new_copyright, content)

    if content != updated_content:
        with open(path, "w") as f:
            f.write(updated_content)


def update_and_get_license() -> str:
    """
    Updates the copyright year in the LICENSE file if necessary and then
    returns its contents.
    """
    # TODO: Check if this is right - if the license file needs to have a range,
    # we need to remove the range before returning the license text.
    #
    # License file should always have the current year.
    update_copyright_year(LICENSE_PATH, disallow_range=True)

    with open(LICENSE_PATH, "r") as license_file:
        return license_file.read()


LICENSE_TEXT = update_and_get_license()

#
# Header manipulation helpers
#


def prefix_lines(content: str, prefix: str) -> str:
    # NOTE: This could have been done via `textwrap.indent`, but we're not actually indenting,
    # so it seems semantically wrong to do that.
    return prefix + f"\n{prefix}".join(content.splitlines())


def insert_after(regex: str) -> Callable[[str, str], str]:
    """
    Builds a callback that will insert a provided header after
    the specified regular expression. If the expression is not
    found in the file contents, the header will be inserted at the
    beginning of the file.

    Args:
        regex: The regular expression to match.

    Returns:
        A callable that can be used as the `add_header` argument to `update_or_add_header`.
    """

    def add_header(header: str, content: str) -> str:
        match = re.match(regex, content)

        if match is None:
            return header + "\n" + content

        insertion_point = match.span()[-1]

        return content[:insertion_point] + f"{header}\n" + content[insertion_point:]

    return add_header


def update_or_add_header(
    path: str, header: str, add_header: Optional[Callable[[str, str], str]] = None
):
    """
    Updates in place or adds a new copyright header to the specified file.

    Args:
        path: The path of the file.
        header: The contents of the copyright header.
        add_header: A callback that receives the copyright header and file contents and
            controls how the contents of the file are updated. By default, the copyright
            header is prepended to the file.
    """
    with open(path, "r") as f:
        content = f.read()

    if has_copyright(content):
        update_copyright_year(path, content)
        return

    add_header = add_header or (lambda header, content: header + "\n" + content)

    content = add_header(header, content)

    # As a sanity check, make sure we didn't accidentally add the copyright header
    # twice, or add a new header when one was already present.
    if content.count("Copyright (c)") != 1:
        print(
            f"WARNING: Something went wrong while processing: {path}!\n"
            "Please check if the copyright header was included twice or wasn't added at all. "
        )

    with open(path, "w") as f:
        f.write(content)


# Each file type requires slightly different handling when inserting the copyright
# header. For example, for C++ files, the header must be prefixed with `//` and for
# shell scripts, it must be prefixed with `#` and must be inserted *after* the shebang.
#
# This mapping stores callables that return whether a handler wants to process a specified
# file based on the path along with callables that will accept the file path and update
# it with the copyright header.
FILE_TYPE_HANDLERS: Dict[Callable[[str], bool], Callable[[str], None]] = {}


#
# Path matching callables
# These allow registered functions to more easily specify what kinds of
# paths they should be applied to.
#
def has_ext(exts: Sequence[str]):
    def has_ext_impl(path: str):
        _, ext = os.path.splitext(path)
        return ext in exts

    return has_ext_impl


def basename_is(expected_path: str):
    return lambda path: os.path.basename(path) == expected_path


def path_contains(expected: str):
    return lambda path: expected in path


def any_of(*funcs: Callable[[str], bool]) -> Callable[[str], bool]:
    return lambda path: any(func(path) for func in funcs)


def is_hidden_file(filepath: str) -> bool:
    # Extract just the filename using Path.name
    filename = Path(filepath).name

    # Check if the filename starts with a period
    return filename.startswith(".")


#
# File handlers for different types of files.
# Many types of files require very similar handling - those are combined where possible.
#


def register(match: Callable[[str], bool]):
    def register_impl(func):
        FILE_TYPE_HANDLERS[match] = func
        return func

    return register_impl


@register(
    any_of(
        has_ext([".py", ".pyi", ".sh", ".bash", ".pbtxt"]),
        basename_is("CMakeLists.txt"),
        path_contains("Dockerfile"),
    )
)
def py_or_shell_like(path):
    update_or_add_header(
        path,
        prefix_lines(LICENSE_TEXT, "# "),
        # Insert the header *after* the shebang.
        # NOTE: This could break if there is a shebang-like pattern elsewhere in the file.
        # In that case, this could be edited to check only the first line of the file (after removing whitespace).
        insert_after(r"#!(.*)\n"),
    )


@register(has_ext([".cc", ".h"]))
def cpp(path):
    update_or_add_header(path, prefix_lines(LICENSE_TEXT, "// "))


@register(has_ext([".tpl"]))
def tpl(path):
    update_or_add_header(path, "{{/*\n" + prefix_lines(LICENSE_TEXT, "# ") + "\n*/}}")


@register(has_ext([".html", ".md"]))
def html_md(path):
    update_or_add_header(path, "<!--\n" + prefix_lines(LICENSE_TEXT, "# ") + "\n-->")


@register(has_ext([".rst"]))
def rst(path):
    update_or_add_header(path, prefix_lines(LICENSE_TEXT, ".. "))


SKIP_PATTERNS = [
    ".toml",
    ".yml",
    ".yaml",
    "LICENSE",
    "COPYRIGHT",
    "Makefile",
    "devcontainer.json",
]


def should_skip(path):
    return any(pattern in path for pattern in SKIP_PATTERNS)


def add_copyrights(paths):
    for path in paths:
        if is_hidden_file(path) or should_skip(path):
            continue
        for match, handler in FILE_TYPE_HANDLERS.items():
            if match(path):
                handler(path)
                break
        else:
            print(
                f"WARNING: No handler registered for file: {path}. Please add a new handler to {__file__}!"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adds copyright headers to source files"
    )
    parser.add_argument("files", nargs="*")

    args, _ = parser.parse_known_args()
    add_copyrights(args.files)
    return 0


if __name__ == "__main__":
    # sys.exit is important here to avoid the test-related imports below during normal execution.
    sys.exit(main())
