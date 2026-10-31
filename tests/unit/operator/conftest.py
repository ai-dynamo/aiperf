# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for operator module tests.

Heavy lifting lives in ``tests.harness.operator``; this file exposes those
builders as pytest fixtures.
"""

import importlib
import sys
from typing import Any

import pytest

from aiperf.common.constants import IS_LINUX
from tests.harness.operator import (
    build_full_aiperfjob_spec,
    build_high_concurrency_spec,
    build_minimal_aiperfjob_spec,
    build_progress_response_running,
    build_progress_response_with_error,
    build_sample_body,
    build_sample_conditions_list,
)

pytestmark = pytest.mark.skipif(
    not IS_LINUX,
    reason="AIPerf operator runs only in Linux Kubernetes containers",
)

# =============================================================================
# Module-reload isolation
# =============================================================================


@pytest.fixture(autouse=True)
def _restore_reloaded_operator_modules():
    """Undo ``importlib.reload`` side effects that strand module singletons.

    Several operator tests call ``importlib.reload(aiperf.operator.environment)``
    (directly or via a fixture) to re-materialize ``OperatorEnvironment`` from a
    fresh set of ``AIPERF_*`` env vars. ``reload`` rebinds the module's
    ``OperatorEnvironment`` (and the nested ``RESULTS``/``SERVICE``/``DASHBOARD``
    settings) to brand-new instances *in place*. Production code reads the
    singleton via a function-local ``from aiperf.operator.environment import
    OperatorEnvironment`` at call time, so it picks up whatever the module
    currently holds — but sibling test files captured the *original* object at
    their own import time. After a reload, ``monkeypatch.setattr`` on the stale
    object no longer affects the value the handler reads (it falls back to the
    ``/data`` default), so a later test in the same process sees its patch
    silently ignored.

    This snapshots the live module objects before each test and rebinds them
    afterwards if a reload swapped them out, keeping every test file's
    module-level reference and the production read path pointed at one object.

    Restoring the ``environment`` module attribute alone is not enough. Most
    operator modules bind the singleton at *their* import time (``from
    aiperf.operator.environment import OperatorEnvironment``), so any module
    first imported — or itself reloaded — inside a test's reload window captures
    the post-reload instance. Rebinding only ``environment`` then orphans that
    instance: the module attribute points at the original, a consumer such as
    ``routers.mutating_auth`` points at the orphan, and a later
    ``monkeypatch.setattr`` on the object reachable from the module is invisible
    to the handler that reads the orphan. So sweep every ``aiperf`` module and
    repoint each captured reference at the original singleton.
    """
    from aiperf.operator import environment as env_mod

    original_module = env_mod
    original_singleton = env_mod.OperatorEnvironment
    yield
    current = importlib.import_module("aiperf.operator.environment")
    if current.OperatorEnvironment is original_singleton:
        return

    original_module.OperatorEnvironment = original_singleton
    for name, module in list(sys.modules.items()):
        if not name.startswith("aiperf.") or module is None:
            continue
        if getattr(module, "OperatorEnvironment", original_singleton) is not (
            original_singleton
        ):
            module.OperatorEnvironment = original_singleton


@pytest.fixture(autouse=True)
def _restore_reloaded_operator_main_registry_bindings():
    """Undo ``importlib.reload`` side effects that strand kopf handler refs.

    The timer-override test reloads ``aiperf.operator.main`` to re-materialize
    its kopf registrations against a patched environment. Reload re-executes
    the module, rebinding every function defined in it — ``owns_namespace``
    and the handler callables included — to brand-new objects *in place*. The
    global default registry is left untouched (registration is redirected to
    throwaway registries), but it keeps holding the *original* function
    objects, so a later test in the same process that asserts handler identity
    against the module — ``handler.fn is operator_main.cleanup_old_sweeps`` or
    ``handler.when is operator_main.owns_namespace`` — sees every handler as a
    stranger and fails purely by execution order.

    Importing ``main`` here — rather than when a test first asks for it —
    guarantees the snapshot holds the pre-reload bindings even when the
    reload-based test is the first one to run in the process. After each test,
    every module attribute whose original object the global registry still
    references is rebound to that original, keeping the module and the
    registry mutually consistent.
    """
    import kopf

    from aiperf.operator import main as operator_main

    registry = kopf.get_default_registry()
    referenced: set[int] = set()
    for sub_name in (
        "_activities",
        "_changing",
        "_indexing",
        "_spawning",
        "_watching",
        "_webhooks",
    ):
        sub_registry = getattr(registry, sub_name, None)
        get_all_handlers = getattr(sub_registry, "get_all_handlers", None)
        if get_all_handlers is None:
            continue
        for handler in get_all_handlers():  # noqa: SLF001
            referenced.add(id(handler.fn))
            when = getattr(handler, "when", None)
            if when is not None:
                referenced.add(id(when))
    original_bindings = {
        name: obj
        for name, obj in vars(operator_main).items()
        if callable(obj) and id(obj) in referenced
    }
    yield
    for name, original in original_bindings.items():
        if getattr(operator_main, name, None) is not original:
            setattr(operator_main, name, original)


# =============================================================================
# Kubernetes Resource Body Fixtures
# =============================================================================


@pytest.fixture
def sample_body() -> dict[str, Any]:
    """Create a sample Kubernetes resource body for event testing."""
    return build_sample_body()


# =============================================================================
# AIPerfJob Spec Fixtures (flat — no userConfig wrapper)
# =============================================================================


@pytest.fixture
def minimal_aiperfjob_spec() -> dict[str, Any]:
    """Create a minimal AIPerfJob spec for testing (nested benchmark format)."""
    return build_minimal_aiperfjob_spec()


@pytest.fixture
def full_aiperfjob_spec() -> dict[str, Any]:
    """Create a complete AIPerfJob spec with all options (nested benchmark format)."""
    return build_full_aiperfjob_spec()


@pytest.fixture
def aiperfjob_spec_high_concurrency() -> dict[str, Any]:
    """Create an AIPerfJob spec with high concurrency for worker scaling tests."""
    return build_high_concurrency_spec()


# =============================================================================
# Progress API Response Fixtures
# =============================================================================


@pytest.fixture
def progress_api_response_running() -> dict[str, Any]:
    """Create a progress API response for a running job."""
    return build_progress_response_running()


@pytest.fixture
def progress_api_response_with_error() -> dict[str, Any]:
    """Create a progress API response with an error."""
    return build_progress_response_with_error()


# =============================================================================
# Condition Fixtures
# =============================================================================


@pytest.fixture
def sample_conditions_list() -> list[dict[str, Any]]:
    """Create a sample conditions list from Kubernetes status."""
    return build_sample_conditions_list()
