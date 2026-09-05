# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mechanical invariant: CLI help text may only reference flags that exist.

`CLIConfig` field descriptions are the source of `docs/cli-options.md`, so a
description naming a flag that was renamed (or never existed) ships a wrong
instruction to users in two places at once and no existing check catches it.
This pins the reference graph closed.
"""

import re

from aiperf.config.cli_parameter import CLIParameter
from aiperf.config.flags.cli_config import CLIConfig

# Matches an inline-code flag reference such as `--random-corpus-style`.
_FLAG_REFERENCE = re.compile(r"`(--[a-z0-9][a-z0-9-]*)`")


def _as_tuple(value: str | tuple[str, ...] | bool | None) -> tuple[str, ...]:
    """Normalize a CLIParameter name/negative to a tuple of flag strings.

    cyclopts accepts a bare string and normalizes it to a tuple, and `negative`
    may additionally be a bool (auto-derive) or None (disabled).
    """
    if isinstance(value, str):
        return (value,)
    if isinstance(value, tuple):
        return value
    return ()


def _declared_flags() -> set[str]:
    """Every flag CLIConfig actually exposes, including negative forms."""
    declared: set[str] = set()
    for field in CLIConfig.model_fields.values():
        for meta in field.metadata:
            if not isinstance(meta, CLIParameter):
                continue
            names = _as_tuple(meta.name)
            declared.update(names)
            declared.update(_as_tuple(meta.negative))
            # `negative=True` means cyclopts derives `--no-<name>` itself.
            if meta.negative is True:
                declared.update(f"--no-{n.lstrip('-')}" for n in names)
    return declared


def _referenced_flags() -> dict[str, set[str]]:
    """Flags cited in field descriptions, mapped to the citing field names."""
    referenced: dict[str, set[str]] = {}
    for name, field in CLIConfig.model_fields.items():
        if not field.description:
            continue
        for flag in _FLAG_REFERENCE.findall(field.description):
            referenced.setdefault(flag, set()).add(name)
    return referenced


def test_help_text_references_only_existing_flags():
    """A description citing a nonexistent flag is a user-facing lie.

    This caught `--random-range-ratio-mode` in the `--random-range-ratio`
    description, which had never existed under that name (the flag is
    `--random-corpus-style`) and was mirrored into docs/cli-options.md twice.
    """
    declared = _declared_flags()
    dangling = {
        flag: sorted(fields)
        for flag, fields in _referenced_flags().items()
        if flag not in declared
    }

    assert not dangling, (
        "CLI help text references flags that do not exist:\n"
        + "\n".join(
            f"  {flag} <- cited by {fields}" for flag, fields in dangling.items()
        )
    )


def test_invariant_has_coverage():
    """Guards the test above against silently passing on an empty scan."""
    assert len(_declared_flags()) > 100
    assert len(_referenced_flags()) > 20
