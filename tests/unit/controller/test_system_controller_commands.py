# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`aiperf.controller.system_controller_commands`.

Focuses on:
- _encode_command_payload type dispatch (BaseModel / bytes / orjson fallback)
- _dispatch_control_command Ack/Ok/Err pathways and traceback capture
- _send_control_command UUID-stamped Command construction + router routing
- _send_control_command_to_all timeout / exception → ErrorDetails wrapping
- _send_control_command_to_all_fail_fast early-abort and remaining-task cancellation
- _force_exit shutdown semantics (calls os._exit, never returns)
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from pydantic import BaseModel, Field

from aiperf.common.control_structs import (
    Command,
    CommandAck,
    CommandErr,
    CommandOk,
)
from aiperf.common.hooks import AIPerfHook
from aiperf.common.models import ErrorDetails
from aiperf.controller.system_controller import SystemController
from aiperf.controller.system_controller_commands import SystemControllerCommandMixin

# ============================================================
# Helpers
# ============================================================


class _FakeHook:
    """Minimal stand-in for an aiperf Hook with resolve_params + async func."""

    def __init__(
        self,
        commands: list[str] | None,
        func: Any,
    ) -> None:
        self._commands = commands
        self.func = func

    def resolve_params(self, _self_obj: Any) -> list[str] | None:
        return self._commands


def _attach_hooks(controller: SystemController, hooks: list[_FakeHook]) -> None:
    """Force a controller's get_hooks(ON_COMMAND) to return our fake hooks."""
    controller.get_hooks = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda hook_type: hooks
        if hook_type == AIPerfHook.ON_COMMAND
        else []
    )


class _SamplePayload(BaseModel):
    """Pydantic model used to verify BaseModel encoding path."""

    name: str = Field(description="Sample field")
    count: int = Field(description="Sample count")


# ============================================================
# _encode_command_payload
# ============================================================


class TestEncodeCommandPayload:
    """Verify payload encoding dispatches by type."""

    def test_basemodel_uses_model_dump_json(self) -> None:
        result = SystemControllerCommandMixin._encode_command_payload(
            _SamplePayload(name="x", count=3)
        )
        assert isinstance(result, bytes)
        assert orjson.loads(result) == {"name": "x", "count": 3}

    def test_bytes_passthrough(self) -> None:
        raw = b"\x00\x01raw"
        assert SystemControllerCommandMixin._encode_command_payload(raw) is raw

    @pytest.mark.parametrize(
        "value,expected",
        [
            ({"a": 1}, b'{"a":1}'),
            ([1, 2, 3], b"[1,2,3]"),
            ("hello", b'"hello"'),
            (42, b"42"),
            (None, b"null"),
        ],
    )  # fmt: skip
    def test_other_values_orjson_dumped(self, value: Any, expected: bytes) -> None:
        assert SystemControllerCommandMixin._encode_command_payload(value) == expected


# ============================================================
# _dispatch_control_command
# ============================================================


class TestDispatchControlCommand:
    """Verify @on_command hook dispatch produces correct response struct."""

    async def test_no_handler_returns_ack(
        self, system_controller: SystemController
    ) -> None:
        _attach_hooks(system_controller, [])

        msg = Command(cid="c1", cmd="UNHANDLED", payload=b"")
        result = await system_controller._dispatch_control_command("svc-A", msg)

        assert isinstance(result, CommandAck)
        assert result.cid == "c1"
        assert result.sid == system_controller.service_id

    async def test_handler_returning_none_returns_ack(
        self, system_controller: SystemController
    ) -> None:
        async def _func(_message: Command) -> None:
            return None

        _attach_hooks(system_controller, [_FakeHook(["DO_THING"], _func)])

        msg = Command(cid="c2", cmd="DO_THING")
        result = await system_controller._dispatch_control_command("svc-B", msg)

        assert isinstance(result, CommandAck)
        assert result.cid == "c2"

    async def test_handler_returning_value_returns_ok_with_payload(
        self, system_controller: SystemController
    ) -> None:
        async def _func(_message: Command) -> dict[str, int]:
            return {"answer": 42}

        _attach_hooks(system_controller, [_FakeHook(["GET"], _func)])

        msg = Command(cid="c3", cmd="GET")
        result = await system_controller._dispatch_control_command("svc-C", msg)

        assert isinstance(result, CommandOk)
        assert result.cid == "c3"
        assert orjson.loads(result.payload) == {"answer": 42}

    async def test_handler_raising_returns_err_with_traceback(
        self, system_controller: SystemController
    ) -> None:
        async def _func(_message: Command) -> None:
            raise RuntimeError("boom")

        _attach_hooks(system_controller, [_FakeHook(["BREAK"], _func)])

        msg = Command(cid="c4", cmd="BREAK")
        result = await system_controller._dispatch_control_command("svc-D", msg)

        assert isinstance(result, CommandErr)
        assert result.cid == "c4"
        assert "boom" in result.error
        assert "RuntimeError" in result.traceback

    async def test_handler_cancelled_propagates(
        self, system_controller: SystemController
    ) -> None:
        async def _func(_message: Command) -> None:
            raise asyncio.CancelledError

        _attach_hooks(system_controller, [_FakeHook(["X"], _func)])

        msg = Command(cid="c5", cmd="X")
        with pytest.raises(asyncio.CancelledError):
            await system_controller._dispatch_control_command("svc", msg)

    async def test_skips_hooks_with_non_iterable_params(
        self, system_controller: SystemController
    ) -> None:
        """Hooks whose resolve_params returns None are not eligible — fall through to Ack."""

        async def _func(_message: Command) -> dict[str, int]:
            return {"unreachable": 1}

        _attach_hooks(system_controller, [_FakeHook(None, _func)])

        msg = Command(cid="c6", cmd="X")
        result = await system_controller._dispatch_control_command("svc", msg)

        assert isinstance(result, CommandAck)


# ============================================================
# _send_control_command
# ============================================================


class TestSendControlCommand:
    """Verify single-target command send + UUID stamping."""

    async def test_sends_command_with_uuid_cid_via_router(
        self, system_controller: SystemController
    ) -> None:
        ack = CommandAck(cid="UUID_FAKE", sid="target")
        system_controller.control_router = MagicMock()
        system_controller.control_router.request_to = AsyncMock(return_value=ack)

        with patch(
            "aiperf.controller.system_controller_commands.uuid.uuid4"
        ) as mock_uuid:
            mock_uuid.return_value.hex = "UUID_FAKE"
            result = await system_controller._send_control_command(
                "target", "PING", payload=b"data", timeout=2.5
            )

        assert result is ack
        sent_identity, sent_command, sent_timeout = (
            system_controller.control_router.request_to.await_args.args
        )
        assert sent_identity == "target"
        assert isinstance(sent_command, Command)
        assert sent_command.cid == "UUID_FAKE"
        assert sent_command.cmd == "PING"
        assert sent_command.payload == b"data"
        assert sent_timeout == 2.5

    async def test_uses_uuid4_hex_format_for_cid(
        self, system_controller: SystemController
    ) -> None:
        """Without monkeypatching, the cid should be a 32-char hex string."""
        captured: dict[str, Command] = {}

        async def _capture(
            identity: str, command: Command, _timeout: float
        ) -> CommandAck:
            captured["cmd"] = command
            return CommandAck(cid=command.cid, sid=identity)

        system_controller.control_router = MagicMock()
        system_controller.control_router.request_to = AsyncMock(side_effect=_capture)

        await system_controller._send_control_command("svc", "PING")

        cid = captured["cmd"].cid
        assert len(cid) == 32
        assert all(c in "0123456789abcdef" for c in cid)


# ============================================================
# _send_control_command_to_all (gather-all, no fail-fast)
# ============================================================


class TestSendControlCommandToAll:
    """Verify fan-out send waits for every response and wraps errors."""

    async def test_collects_all_successful_responses(
        self, system_controller: SystemController
    ) -> None:
        responses = {
            "a": CommandAck(cid="1", sid="a"),
            "b": CommandOk(cid="2", sid="b", payload=b"ok"),
            "c": CommandErr(cid="3", sid="c", error="nope"),
        }
        system_controller._send_control_command = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda sid, *_a, **_kw: responses[sid]
        )

        result = await system_controller._send_control_command_to_all(
            "CMD", ["a", "b", "c"]
        )

        assert result == [responses["a"], responses["b"], responses["c"]]

    async def test_timeouterror_becomes_error_details(
        self, system_controller: SystemController
    ) -> None:
        async def _send(sid: str, *_a: Any, **_kw: Any) -> Any:
            if sid == "slow":
                raise TimeoutError
            return CommandAck(cid="1", sid=sid)

        system_controller._send_control_command = AsyncMock(side_effect=_send)  # type: ignore[method-assign]

        result = await system_controller._send_control_command_to_all(
            "CMD", ["fast", "slow"]
        )

        assert isinstance(result[0], CommandAck)
        assert isinstance(result[1], ErrorDetails)
        assert result[1].type == "TimeoutError"
        assert "slow" in result[1].message

    async def test_generic_exception_becomes_error_details(
        self, system_controller: SystemController
    ) -> None:
        async def _send(sid: str, *_a: Any, **_kw: Any) -> Any:
            if sid == "broken":
                raise RuntimeError("connection refused")
            return CommandAck(cid="1", sid=sid)

        system_controller._send_control_command = AsyncMock(side_effect=_send)  # type: ignore[method-assign]

        result = await system_controller._send_control_command_to_all(
            "CMD", ["ok", "broken"]
        )

        assert isinstance(result[0], CommandAck)
        assert isinstance(result[1], ErrorDetails)
        assert "connection refused" in result[1].message

    async def test_cancelled_propagates(
        self, system_controller: SystemController
    ) -> None:
        async def _send(*_a: Any, **_kw: Any) -> Any:
            raise asyncio.CancelledError

        system_controller._send_control_command = AsyncMock(side_effect=_send)  # type: ignore[method-assign]

        with pytest.raises(asyncio.CancelledError):
            await system_controller._send_control_command_to_all("CMD", ["a"])

    async def test_empty_service_list_returns_empty(
        self, system_controller: SystemController
    ) -> None:
        result = await system_controller._send_control_command_to_all("CMD", [])
        assert result == []


# ============================================================
# _send_control_command_to_all_fail_fast
# ============================================================


class TestSendControlCommandToAllFailFast:
    """Verify fail-fast semantics: first CommandErr aborts remaining waits."""

    async def test_collects_all_when_no_errors(
        self, system_controller: SystemController
    ) -> None:
        async def _send(sid: str, *_a: Any, **_kw: Any) -> Any:
            return CommandAck(cid="1", sid=sid)

        system_controller._send_control_command = AsyncMock(side_effect=_send)  # type: ignore[method-assign]

        result = await system_controller._send_control_command_to_all_fail_fast(
            "CMD", ["a", "b", "c"]
        )

        assert len(result) == 3
        assert all(isinstance(r, CommandAck) for r in result)

    async def test_first_command_err_breaks_loop(
        self, system_controller: SystemController
    ) -> None:
        async def _send(sid: str, *_a: Any, **_kw: Any) -> Any:
            if sid == "a":
                return CommandErr(cid="1", sid="a", error="bad")
            await asyncio.sleep(
                60
            )  # would block forever, but tasks should be cancelled
            return CommandAck(cid="2", sid=sid)

        system_controller._send_control_command = AsyncMock(side_effect=_send)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            system_controller._send_control_command_to_all_fail_fast(
                "CMD", ["a", "b", "c"]
            ),
            timeout=5.0,
        )

        # Only the first (failing) response should be collected before break.
        assert len(result) == 1
        assert isinstance(result[0], CommandErr)

    async def test_timeout_breaks_with_timeout_error_details(
        self, system_controller: SystemController
    ) -> None:
        async def _send(*_a: Any, **_kw: Any) -> Any:
            raise TimeoutError

        system_controller._send_control_command = AsyncMock(side_effect=_send)  # type: ignore[method-assign]

        result = await system_controller._send_control_command_to_all_fail_fast(
            "CMD", ["a", "b"]
        )

        assert len(result) == 1
        assert isinstance(result[0], ErrorDetails)
        assert result[0].type == "TimeoutError"

    async def test_generic_exception_breaks_with_error_details(
        self, system_controller: SystemController
    ) -> None:
        async def _send(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("network down")

        system_controller._send_control_command = AsyncMock(side_effect=_send)  # type: ignore[method-assign]

        result = await system_controller._send_control_command_to_all_fail_fast(
            "CMD", ["a", "b"]
        )

        assert len(result) == 1
        assert isinstance(result[0], ErrorDetails)
        assert "network down" in result[0].message


# ============================================================
# _force_exit
# ============================================================


class TestForceExit:
    """Verify _force_exit flushes streams and calls os._exit."""

    def test_calls_os_exit_with_code(self) -> None:
        with (
            patch("aiperf.controller.system_controller_commands.os._exit") as m_exit,
            patch(
                "aiperf.controller.system_controller_commands.sys.stdout.flush"
            ) as m_stdout,
            patch(
                "aiperf.controller.system_controller_commands.sys.stderr.flush"
            ) as m_stderr,
        ):
            SystemControllerCommandMixin._force_exit(7)

        m_stdout.assert_called_once()
        m_stderr.assert_called_once()
        m_exit.assert_called_once_with(7)
