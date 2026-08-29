# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-command dispatch and send helpers for the SystemController."""

from __future__ import annotations

import asyncio
import traceback
import uuid
from collections.abc import Iterable

import zmq
from msgspec import Struct

from aiperf.common.control_structs import (
    Command,
    CommandAck,
    CommandErr,
    CommandOk,
    CommandResponse,
    CommandUnhandled,
    encode_command_payload,
)
from aiperf.common.environment import Environment
from aiperf.common.hooks import AIPerfHook
from aiperf.common.models import ErrorDetails


def command_error_details(e: Exception) -> ErrorDetails:
    """Build ErrorDetails for a failed command send, preserving a ZMQ errno.

    ``ErrorDetails.from_exception`` only populates ``code`` from an
    ``error_code`` attribute, which ``zmq.ZMQError`` does not have -- it carries
    ``errno``. Without this the errno is dropped and callers are left matching on
    the exception's message text, which is not stable across pyzmq/libzmq
    versions. Relay callers use ``code`` to tell "that peer has departed" from
    "that peer answered with a fault".
    """
    details = ErrorDetails.from_exception(e)
    if isinstance(e, zmq.ZMQError) and e.errno is not None:
        details.code = e.errno
    return details


class SystemControllerCommandMixin:
    """Control-channel dispatch + send helpers for :class:`SystemController`.

    Covers encode/decode of @on_command results and the fan-out send-and-wait
    patterns used to coordinate services during configure / profile / shutdown.
    """

    async def _dispatch_control_command(
        self, identity: str, message: Command
    ) -> Struct | None:
        """Dispatch an incoming Command from a service to local @on_command hooks.

        Returns the response struct to send back to ``identity``: CommandAck for
        a handler that returned nothing, CommandOk when it returned a result,
        CommandErr when it raised, and CommandUnhandled when no hook matched.
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
                    cmd=message.cmd,
                    sid=self.service_id,
                    error=str(e),
                    traceback=tb,
                )

            if result is None:
                return CommandAck(cid=message.cid, cmd=message.cmd, sid=self.service_id)
            return CommandOk(
                cid=message.cid,
                cmd=message.cmd,
                sid=self.service_id,
                payload=encode_command_payload(result),
            )

        # Distinct from CommandAck on purpose: callers treat "this peer does not
        # implement the command" as a failure, an ack as success.
        self.debug(f"No handler for command {message.cmd} from {identity}")
        return CommandUnhandled(cid=message.cid, cmd=message.cmd, sid=self.service_id)

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

    async def _broadcast_control_command(
        self,
        cmd: str,
        service_ids: Iterable[str],
        payload: bytes = b"",
    ) -> None:
        """Fire a command at every listed service without awaiting any response.

        Replaces an un-targeted ``publish(SomeCommand(...))``, which every
        service received and which the controller never waited on. ROUTER
        delivery is per-peer, so the fan-out is explicit; each send is best
        effort because a service that already exited makes ROUTER_MANDATORY
        raise EHOSTUNREACH, and that must not stop delivery to peers still
        alive.
        """
        for sid in service_ids:
            try:
                await self.control_router.send_to(
                    sid, Command(cid=uuid.uuid4().hex, cmd=cmd, payload=payload)
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - best-effort broadcast, one dead peer must not block the rest
                self.debug(
                    lambda e=e, sid=sid: f"Failed to send {cmd} to '{sid}': {e!r}"
                )

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
        # Awaited in insertion order, not completion order: two callers do
        # zip(service_ids, responses, strict=True). All tasks are already
        # running, so ordered awaiting costs no wall-clock.
        for sid, task in tasks.items():
            try:
                results.append(await task)
            except TimeoutError:
                results.append(
                    ErrorDetails(
                        type="TimeoutError",
                        message=f"Command {cmd} timed out for {sid}",
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - service cmd dispatch boundary
                results.append(command_error_details(e))
        return results

    async def _send_control_command_to_all_fail_fast(
        self,
        cmd: str,
        service_ids: list[str],
        payload: bytes = b"",
        timeout: float = Environment.SERVICE.COMMAND_RESPONSE_TIMEOUT,
    ) -> list[CommandResponse | ErrorDetails]:
        """Send command to all services, aborting on first error.

        Completion-ordered on purpose -- aborting early is the whole point, and
        no caller zips this result against ``service_ids``.

        CommandUnhandled is deliberately NOT an abort condition. Both callers
        fan PROFILE_CONFIGURE / PROFILE_START at *every* registered service, and
        several legitimately implement neither (RecordsManager has no
        PROFILE_CONFIGURE hook). Breaking on "no handler" abandons the fan-out
        at the first such service and cancels the rest in the ``finally`` below,
        so the TimingManager never finishes configuring and the run dies at
        PROFILE_START with "No phase orchestrator configured". The pub/sub
        predecessor broke on the error response only; keep that.
        """
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
                except TimeoutError:
                    results.append(
                        ErrorDetails(
                            type="TimeoutError", message=f"Command {cmd} timed out"
                        )
                    )
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 - service cmd dispatch boundary
                    results.append(command_error_details(e))
                    break
        finally:
            for task in tasks.values():
                task.cancel()
        return results
