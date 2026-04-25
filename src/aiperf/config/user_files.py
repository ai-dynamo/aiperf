# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User-defined templated output files materialized into the run directory.

See docs/kubernetes/user-files.md for the user-facing reference.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from aiperf.common.exceptions import AIPerfError
from aiperf.config._base import BaseConfig

_FORBIDDEN_PATH_CHARS = frozenset(chr(c) for c in range(32)) | {"\x7f"}


class UserFileError(AIPerfError):
    """Raised when a user_files entry fails validation, render, or write."""


class UserFile(BaseConfig):
    """One user-declared output file rendered into the run directory before benchmark start.

    Path is relative to the run directory; subdirectories are allowed; absolute
    paths and any segment equal to '..' are rejected. Content is rendered with
    jinja2 against a documented context (variables: + system-injected names);
    see docs/kubernetes/user-files.md.
    """

    path: Annotated[
        str,
        Field(
            description=(
                "Output path relative to the run directory. Subdirectories allowed. "
                "Absolute paths and any segment equal to '..' are rejected."
            ),
        ),
    ]

    format: Annotated[
        Literal["json", "yaml", "text"] | None,
        Field(
            default=None,
            description=(
                "Serialization format. If omitted: 'text' when content is a string, "
                "'json' otherwise."
            ),
        ),
    ] = None

    content: Annotated[
        Any,
        Field(
            description=(
                "Templated value. Structured (dict/list/scalar) for json/yaml; "
                "string for text. Jinja2 expressions in any string leaf are "
                "rendered with the user_files context."
            ),
        ),
    ]

    @model_validator(mode="after")
    def _validate_path(self) -> UserFile:
        if not self.path:
            raise ValueError("user_files entry has empty path")
        if any(c in _FORBIDDEN_PATH_CHARS for c in self.path):
            raise ValueError(
                f"user_files path contains control characters: {self.path!r}"
            )
        p = PurePosixPath(self.path)
        if p.is_absolute():
            raise ValueError(f"user_files absolute path rejected: {self.path!r}")
        if any(part == ".." for part in p.parts):
            raise ValueError(f"user_files path '..' rejected: {self.path!r}")
        return self

    @model_validator(mode="after")
    def _resolve_format(self) -> UserFile:
        if self.format is None:
            self.format = "text" if isinstance(self.content, str) else "json"
        if self.format in {"json", "yaml"} and isinstance(self.content, str):
            raise ValueError(
                f"user_files path={self.path!r}: format={self.format!r} "
                "requires structured content (dict/list/scalar); got str. "
                "Wrap in a dict or set format: text."
            )
        if self.format == "text" and not isinstance(self.content, str):
            raise ValueError(
                f"user_files path={self.path!r}: format='text' requires string content; "
                f"got {type(self.content).__name__}."
            )
        return self
