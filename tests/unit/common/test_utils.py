# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp

import orjson
import pytest

from aiperf.common.exceptions import AIPerfMultiError
from aiperf.common.utils import (
    _set_daemon,
    allow_daemon_children,
    call_all_functions,
    call_all_functions_self,
    load_json_str,
)


class TestAllowDaemonChildren:
    """``allow_daemon_children`` clears the current process's daemon flag so
    daemon services (every AIPerf service is spawned ``daemon=True``) can fan
    out to ProcessPoolExecutor / multiprocessing.Pool.
    """

    def _set_daemon(self, value: bool) -> None:
        try:
            mp.current_process().daemon = value
        except AssertionError:
            mp.current_process()._config["daemon"] = value

    def test_clears_daemon_flag_inside_context(self) -> None:
        original = mp.current_process().daemon
        try:
            self._set_daemon(True)
            with allow_daemon_children():
                assert mp.current_process().daemon is False
        finally:
            self._set_daemon(original)

    def test_restores_daemon_flag_on_exit(self) -> None:
        original = mp.current_process().daemon
        try:
            self._set_daemon(True)
            with allow_daemon_children():
                pass
            assert mp.current_process().daemon is True
        finally:
            self._set_daemon(original)

    def test_restores_daemon_flag_on_exception(self) -> None:
        original = mp.current_process().daemon
        try:
            self._set_daemon(True)
            with pytest.raises(RuntimeError), allow_daemon_children():
                raise RuntimeError("boom")
            assert mp.current_process().daemon is True
        finally:
            self._set_daemon(original)

    def test_noop_when_not_daemon(self) -> None:
        original = mp.current_process().daemon
        try:
            self._set_daemon(False)
            with allow_daemon_children():
                assert mp.current_process().daemon is False
            assert mp.current_process().daemon is False
        finally:
            self._set_daemon(original)


class TestSetDaemon:
    """``_set_daemon`` sets the flag via the public property, falling back to
    the internal ``_config`` dict when the property setter raises (Python
    asserts non-daemon parents)."""

    def test_uses_property(self) -> None:
        from unittest.mock import MagicMock, patch

        mock_proc = MagicMock()
        mock_proc.daemon = True
        with patch("aiperf.common.utils.mp.current_process", return_value=mock_proc):
            _set_daemon(False)
        assert mock_proc.daemon is False

    def test_falls_back_to_config_on_assertion_error(self) -> None:
        from unittest.mock import MagicMock, patch

        mock_proc = MagicMock()
        type(mock_proc).daemon = property(
            fget=lambda self: self._config.get("daemon"),
            fset=MagicMock(side_effect=AssertionError),
        )
        mock_proc._config = {"daemon": True}
        with patch("aiperf.common.utils.mp.current_process", return_value=mock_proc):
            _set_daemon(False)
        assert mock_proc._config["daemon"] is False


class TestLoadJsonStrErrors:
    """Tests that load_json_str raises the original error for both str and bytes."""

    @pytest.mark.parametrize(
        "json_str",
        [
            pytest.param("{not valid json}", id="invalid-str"),
            pytest.param("", id="empty-str"),
            pytest.param(b"", id="empty-bytes"),
            pytest.param(b"not json", id="invalid-bytes"),
            pytest.param('{"key": ', id="truncated"),
            pytest.param('{"key": 1,}', id="trailing-comma"),
        ],
    )  # fmt: skip
    def test_invalid_input_raises_decode_error(self, json_str: str | bytes) -> None:
        with pytest.raises(orjson.JSONDecodeError):
            load_json_str(json_str)

    def test_validation_func_error_propagates(self) -> None:
        def fail(_: object) -> None:
            raise ValueError("bad data")

        with pytest.raises(ValueError, match="bad data"):
            load_json_str('{"key": 1}', func=fail)


class TestCallAllFunctions:
    """Test call_all_functions and call_all_functions_self error handling."""

    @pytest.mark.asyncio
    async def test_call_all_functions_logs_and_raises_on_error(self) -> None:
        def bad_func() -> None:
            raise RuntimeError("boom")

        with pytest.raises(AIPerfMultiError):
            await call_all_functions([bad_func])

    @pytest.mark.asyncio
    async def test_call_all_functions_self_logs_and_raises_on_error(self) -> None:
        class Dummy:
            pass

        def bad_method(self_) -> None:
            raise RuntimeError("boom")

        with pytest.raises(AIPerfMultiError):
            await call_all_functions_self(Dummy(), [bad_method])
