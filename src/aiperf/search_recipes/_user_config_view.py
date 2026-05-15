# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural view of the user's CLI config for search recipes.

Search recipes need to read a few attributes off the user's CLI config when
expanding (today: ``endpoint.streaming``). Rather than coupling
``aiperf.search_recipes`` to any concrete CLI-input class, we declare the
read-only surface here as a ``Protocol``. The cyclopts CLI populates an
``aiperf.config.v1.UserConfig`` DTO directly from CLI flags, which satisfies
this Protocol structurally; future config-input shapes satisfy it without
any recipe-side changes.

When a recipe needs a NEW attribute off the user config, add it here (and to
the matching nested view if it lives on a sub-config). Don't speculate:
extend the surface only when a recipe actually reads it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class _EndpointView(Protocol):
    """Read-only structural view of the endpoint section.

    Only the attributes recipes read are declared. ``streaming`` is a plain
    ``bool``; recipes that compare with ``is False`` (rather than ``not``)
    let an unset / defaulted-True flag fall through cleanly.
    """

    streaming: bool


@runtime_checkable
class RecipeUserConfigView(Protocol):
    """Read-only structural view of the user's CLI config for recipes.

    Any object with a matching ``.endpoint.streaming`` attribute satisfies
    this Protocol structurally; recipes that type ``ctx.user_config`` as
    ``RecipeUserConfigView`` get IDE completion on the documented
    attributes without coupling to a specific CLI-input shape.

    Today only ``endpoint`` (and ``endpoint.streaming``) is read. Extend in
    lockstep with new recipe accesses -- see ``builtins.py``.
    """

    endpoint: _EndpointView | None
