# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural view of v1 ``UserConfig`` for search recipes.

The ``v1-import-leak`` rule (``tools/check_ergonomics.py::check_v1_import_leak``)
forbids ``from aiperf.config.v1 import ...`` outside ``cli_commands/**`` and
``config/v1/**`` so downstream code doesn't grow a dependency on the
CLI-input shim. Search recipes still need to read a few attributes off the
``UserConfig`` they're expanding under (today: ``endpoint.streaming``), so we
declare the read-only surface here as a ``Protocol``. A real ``UserConfig`` is
structurally compatible — recipes get IDE completion on the documented
attributes without an actual v1 import.

When a recipe needs a NEW attribute off ``UserConfig``, add it here (and to
the matching nested view if it lives on a sub-config). Don't speculate:
extend the surface only when a recipe actually reads it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class _EndpointView(Protocol):
    """Read-only structural view of ``aiperf.config.v1.EndpointConfig``.

    Only the attributes recipes read are declared. ``streaming`` matches the
    v1 type (plain ``bool``, defaults to ``True``); recipes that compare with
    ``is False`` work regardless because ``True is False`` and ``False is False``
    are both well-defined.
    """

    streaming: bool


@runtime_checkable
class RecipeUserConfigView(Protocol):
    """Read-only structural view of ``aiperf.config.v1.UserConfig`` for recipes.

    A real ``UserConfig`` satisfies this Protocol structurally; recipes that
    type ``ctx.user_config`` as ``RecipeUserConfigView`` get IDE completion
    on the documented attributes without importing v1.

    Today only ``endpoint`` (and ``endpoint.streaming``) is read. Extend in
    lockstep with new recipe accesses — see ``builtins.py``.
    """

    endpoint: _EndpointView | None
