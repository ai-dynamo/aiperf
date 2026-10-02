# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guards the handler-module names that tests patch for progress clients.

Handler modules acquire progress clients through the leased
``acquire_progress_client`` / ``release_progress_client`` pair. Tests isolate
themselves from the network by patching those names on the handler module.

A handler that keeps an unused ``get_or_create_progress_client`` import is
worse than one that drops it: patching a name the module never calls succeeds
silently, so the test builds a real ``ProgressClient``, the mock is never
touched, and the assertions pass against nothing. That is exactly how two
cancellation tests passed vacuously for the life of this branch, while their
siblings failed loudly with ``AttributeError``. Loud is the desired failure.
"""

from __future__ import annotations

import importlib

import pytest
from pytest import param

_HANDLER_MODULES = (
    "aiperf.operator.handlers.completion",
    "aiperf.operator.handlers.lifecycle",
    "aiperf.operator.handlers.monitor",
    "aiperf.operator.handlers._completion_fetch",
)


@pytest.mark.parametrize(
    "module_name",
    [param(name, id=name.rsplit(".", 1)[-1]) for name in _HANDLER_MODULES],
)  # fmt: skip
def test_handler_module_does_not_reexport_unleased_client_getter(
    module_name: str,
) -> None:
    """No handler may bind get_or_create_progress_client without calling it."""
    module = importlib.import_module(module_name)

    assert not hasattr(module, "get_or_create_progress_client"), (
        f"{module_name} binds get_or_create_progress_client. If it does not "
        "call it, drop the import: a patch aimed at the dead name succeeds "
        "and silently tests nothing."
    )


def test_leasing_handlers_expose_both_halves_of_the_lease() -> None:
    """Acquiring without releasing leaks a lease that blocks cache eviction."""
    for module_name in _HANDLER_MODULES:
        module = importlib.import_module(module_name)
        if not hasattr(module, "acquire_progress_client"):
            continue
        assert hasattr(module, "release_progress_client"), (
            f"{module_name} acquires a progress client but never releases one; "
            "a retained lease pins the entry against _evict_idle_clients."
        )
