# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User-facing errors for remote dataset sources."""

from __future__ import annotations

import re

from aiperf.common.enums import CaseInsensitiveStrEnum
from aiperf.common.redact import redact_url
from aiperf.config.loader.errors import ConfigurationError

_WHITESPACE_RUN = re.compile(r"(\s+)")


def _redact_text(text: str) -> str:
    """``redact_url`` for free text that may embed URLs.

    ``redact_url`` treats everything after a sensitive query value as part of that
    value, so applying it to a whole sentence would swallow the words after the
    URL. Redacting each whitespace-delimited token keeps the sentence readable.
    """
    return "".join(
        part if part.isspace() else redact_url(part)
        for part in _WHITESPACE_RUN.split(text)
    )


class RemoteErrorKind(CaseInsensitiveStrEnum):
    """Why a remote dataset source could not be used."""

    AUTH_MISSING = "auth_missing"
    AUTH_DENIED = "auth_denied"
    GATED = "gated"
    NOT_FOUND = "not_found"
    REVISION_NOT_FOUND = "revision_not_found"
    MANIFEST_INVALID = "manifest_invalid"
    INCOMPATIBLE = "incompatible"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    MISSING_DEPENDENCY = "missing_dependency"
    TRANSIENT = "transient"


class RemoteStage(CaseInsensitiveStrEnum):
    """Where in the resolution flow the failure happened."""

    PARSE = "parse"
    ACCESS = "access"
    MATERIALIZE = "materialize"
    POD = "pod"


class RemoteSourceError(ConfigurationError):
    """A remote dataset reference that cannot be used, with the fix in the message."""

    def __init__(
        self,
        *,
        ref: str,
        kind: RemoteErrorKind,
        stage: RemoteStage,
        what: str,
        hint: str,
    ) -> None:
        self.ref = redact_url(ref)
        self.kind = kind
        self.stage = stage
        # ``what`` and ``hint`` often embed text from a library exception that
        # quotes the request URL, so every part of the message is redacted.
        self.hint = _redact_text(hint)
        super().__init__(
            f"Cannot use {self.ref}: {_redact_text(what)}. {self.hint}".rstrip()
        )


def changed_after_pin(
    ref_display: str, stage: RemoteStage, detail: str
) -> RemoteSourceError:
    """The remote changed after AIPerf pinned its version.

    ``detail`` names what changed, e.g. "the file", "the object", "the repo".
    """
    return RemoteSourceError(
        ref=ref_display,
        kind=RemoteErrorKind.NOT_FOUND,
        stage=stage,
        what=f"{detail} changed after AIPerf pinned its version",
        hint="Run again to use the new version",
    )
