# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-command dispatch and send helpers for the SystemController."""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
import uuid
from collections.abc import Iterable
from typing import Any

import orjson
from msgspec import Struct

from aiperf.common.control_structs import (
    Command,
    CommandAck,
    CommandErr,
    CommandOk,
    CommandResponse,
)
from aiperf.common.environment import Environment
from aiperf.common.hooks import AIPerfHook
from aiperf.common.models import ErrorDetails


class SystemControllerCommandMixin:
    """Control-channel dispatch + send helpers for :class:`SystemController`.

    Covers encode/decode of @on_command results and the fan-out send-and-wait
    patterns used to coordinate services during configure / profile / shutdown.
    """

    @staticmethod
    def _encode_command_payload(result: Any) -> bytes:
        """Encode a command-hook result to a payload byte-string."""
        from pydantic import BaseModel

        if isinstance(result, BaseModel):
            return result.model_dump_json().encode()
        if isinstance(result, bytes):
            return result
        return orjson.dumps(result)

    async def _dispatch_control_command(
        self, identity: str, message: Command
    ) -> Struct | None:
        """Dispatch an incoming Command from a service to local @on_command hooks.

        Returns a CommandAck/CommandOk/CommandErr response struct.
        """
        for hook in self.get_hooks(AIPerfHook.ON_COMMAND):
            resolved = hook.resolve_params(self)
            if not (isinstance(resolved, Iterable) and message.cmd in resolved):
                continue
            try:
                result = await hook.func(message)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - service cmd dispatch boundary
                tb = traceback.format_exc()
                self.error(
                    f"Failed to handle command {message.cmd} from {identity}: {e}"
                )
                return CommandErr(
                    cid=message.cid,
                    sid=self.service_id,
                    error=str(e),
                    traceback=tb,
                )

            if result is None:
                return CommandAck(cid=message.cid, sid=self.service_id)
            return CommandOk(
                cid=message.cid,
                sid=self.service_id,
                payload=self._encode_command_payload(result),
            )

        self.debug(f"No handler for command {message.cmd} from {identity}")
        return CommandAck(cid=message.cid, sid=self.service_id)

    async def _send_control_command(
        self,
        identity: str,
        cmd: str,
        payload: bytes = b"",
        timeout: float = Environment.SERVICE.COMMAND_RESPONSE_TIMEOUT,
    ) -> CommandResponse:
        """Send a command to a specific service via ROUTER and wait for response."""
        command = Command(cid=uuid.uuid4().hex, cmd=cmd, payload=payload)
        return await self.control_router.request_to(identity, command, timeout)

    async def _send_control_command_to_all(
        self,
        cmd: str,
        service_ids: list[str],
        payload: bytes = b"",
        timeout: float = Environment.SERVICE.COMMAND_RESPONSE_TIMEOUT,
    ) -> list[CommandResponse | ErrorDetails]:
        """Send a command to all specified services and wait for all responses."""
        tasks = {
            sid: asyncio.create_task(
                self._send_control_command(sid, cmd, payload, timeout)
            )
            for sid in service_ids
        }
        results: list[CommandResponse | ErrorDetails] = []
        for sid, task in tasks.items():
            try:
                results.append(await task)
            except asyncio.TimeoutError:
                results.append(
                    ErrorDetails(
                        type="TimeoutError",
                        message=f"Command {cmd} timed out for {sid}",
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - service cmd dispatch boundary
                results.append(ErrorDetails.from_exception(e))
        return results

    async def _send_control_command_to_all_fail_fast(
        self,
        cmd: str,
        service_ids: list[str],
        payload: bytes = b"",
        timeout: float = Environment.SERVICE.COMMAND_RESPONSE_TIMEOUT,
    ) -> list[CommandResponse | ErrorDetails]:
        """Send command to all services, aborting on first error."""
        tasks = {
            sid: asyncio.create_task(
                self._send_control_command(sid, cmd, payload, timeout)
            )
            for sid in service_ids
        }
        results: list[CommandResponse | ErrorDetails] = []
        try:
            for coro in asyncio.as_completed(tasks.values()):
                try:
                    response = await coro
                    results.append(response)
                    if isinstance(response, CommandErr):
                        self.debug(
                            f"Received error from {response.sid}, aborting wait for "
                            f"remaining {len(service_ids) - len(results)} service(s)"
                        )
                        break
                except asyncio.TimeoutError:
                    results.append(
                        ErrorDetails(
                            type="TimeoutError", message=f"Command {cmd} timed out"
                        )
                    )
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 - service cmd dispatch boundary
                    results.append(ErrorDetails.from_exception(e))
                    break
        finally:
            for task in tasks.values():
                task.cancel()
        return results

    @staticmethod
    def _force_exit(code: int) -> None:
        """Flush stdio and exit. Falls back to os._exit if sys.exit hangs
        (e.g. ZMQ context blocking in atexit)."""
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
