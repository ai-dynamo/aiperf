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

from tests.harness.operator import (
    build_full_aiperfjob_spec,
    build_high_concurrency_spec,
    build_minimal_aiperfjob_spec,
    build_progress_response_running,
    build_progress_response_with_error,
    build_sample_body,
    build_sample_conditions_list,
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
