# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SIGUSR1 stack-dump diagnostic handler dumps every thread's traceback for hang debugging."""

from __future__ import annotations

import signal
from unittest.mock import MagicMock, patch

import pytest
from pytest import param

from aiperf.common import bootstrap


@pytest.mark.skipif(
    not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 not available on this platform"
)
def test_registers_sigusr1_handler() -> None:
    with patch(
        "aiperf.common.bootstrap.faulthandler.register", create=True
    ) as mock_register:
        bootstrap.register_sigusr1_faulthandler()
    mock_register.assert_called_once_with(signal.SIGUSR1, all_threads=True, chain=False)


def test_noop_without_sigusr1() -> None:
    """No SIGUSR1 (e.g. Windows): the handler must not be registered."""
    fake_signal = MagicMock(
        spec=[]
    )  # no attributes -> hasattr(..., "SIGUSR1") is False
    with (
        patch("aiperf.common.bootstrap.signal", fake_signal),
        patch(
            "aiperf.common.bootstrap.faulthandler.register", create=True
        ) as mock_register,
    ):
        bootstrap.register_sigusr1_faulthandler()
    mock_register.assert_not_called()


@pytest.mark.skipif(
    not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 not available on this platform"
)
@pytest.mark.parametrize(
    "exc_type",
    [
        param(ValueError, id="valueerror_stderr_without_fileno"),
        param(RuntimeError, id="runtimeerror"),
        param(AttributeError, id="attributeerror"),
    ],
)  # fmt: skip
def test_registration_errors_are_suppressed(exc_type: type[Exception]) -> None:
    """The best-effort helper swallows registration errors rather than crash startup."""
    with patch(
        "aiperf.common.bootstrap.faulthandler.register",
        side_effect=exc_type("sys.stderr has no fileno"),
        create=True,
    ):
        bootstrap.register_sigusr1_faulthandler()


@pytest.mark.skipif(
    not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 not available on this platform"
)
def test_real_registration_installs_handler() -> None:
    """End-to-end: after the call, faulthandler reports a handler for SIGUSR1."""
    import faulthandler

    faulthandler.unregister(signal.SIGUSR1)
    try:
        bootstrap.register_sigusr1_faulthandler()
        # unregister() returns True iff a handler was installed for the signal.
        assert faulthandler.unregister(signal.SIGUSR1) is True
    finally:
        faulthandler.unregister(signal.SIGUSR1)
