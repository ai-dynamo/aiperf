# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path-safety helpers for user-supplied filesystem paths.

Centralizes the CWE-22 path-injection hardening applied to anywhere AIPerf
interprets a user-supplied string as "either a filesystem path to read, or a
literal value." Today that pattern appears in template loading (CLI
``--extra-inputs payload_template=...`` and YAML/direct ``TemplateEndpoint``
inputs); future call sites with the same shape should reuse this helper rather
than reinventing the sanitizer chain.
"""

from __future__ import annotations

from pathlib import Path


def safe_read_template_path(ts: str) -> str | None:
    """Return file contents if ``ts`` safely resolves to a regular file, else ``None``.

    Sanitizer chain (in the order SAST tools walk it):
      1. ``Path(ts).expanduser()`` — also catches ``RuntimeError`` from
         unresolvable ``~user`` / unset ``HOME`` prefixes.
      2. Reject if any component (leaf or any parent) is a symlink. ``resolve()``
         alone is insufficient because it follows symlinked parent directories.
      3. ``Path.resolve(strict=True)`` — the canonical sanitizer that SAST
         engines (Snyk/CodeQL/Semgrep) recognize; raises on missing paths.
      4. Require ``is_file()`` on the resolved target (rejects directories,
         devices, fifos).
      5. ``read_text(encoding="utf-8")`` — explicit decode, no platform default.

    Returning ``None`` signals the caller to treat ``ts`` as a literal value
    (the existing "inline template body" fallback in both call sites).
    """
    try:
        path = Path(ts).expanduser()
    except (TypeError, ValueError, RuntimeError):
        return None
    try:
        for candidate in (path, *path.parents):
            if candidate.is_symlink():
                return None
    except OSError:
        return None
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not resolved.is_file():
        return None
    try:
        return resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
