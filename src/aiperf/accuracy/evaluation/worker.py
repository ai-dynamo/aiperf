# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict supervised evaluator-worker protocol v2 bootstrap.

Production protocol traffic uses dedicated inherited one-way file descriptors;
stdin/stdout mode exists only behind an explicit ``--stdio`` compatibility flag
for standalone conformance tests.  Provider imports occur only after ``hello``
and installed-closure attestation.
"""

from __future__ import annotations

import argparse
import asyncio
import enum
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from aiperf.accuracy.evaluation.canonical import canonical_dumps, canonical_loads
from aiperf.accuracy.evaluation.contracts import (
    EvaluationError,
    EvaluationHostBinding,
    EvaluationPlanRequest,
    HostOperationEvent,
    ResolvedAsset,
    ScopedProxyBinding,
    UnitOccurrenceRequest,
    require_non_negative_int,
    require_positive_int,
    strict_object,
)
from aiperf.accuracy.evaluation.distributions import (
    DistributionEvidence,
    StockDistributionDescriptor,
    selected_descriptor,
)
from aiperf.accuracy.evaluation.providers.base import EvaluationProviderAdapter
from aiperf.accuracy.evaluation.session import SessionRuntime

_BOOTSTRAP_MAX_MESSAGE_BYTES = 8 * 1024 * 1024
_BOOTSTRAP_MAX_COLLECTION_ITEMS = 16_384


class WorkerState(enum.StrEnum):
    """Protocol-visible worker lifecycle states."""

    SPAWNED = "spawned"
    NEGOTIATED = "negotiated"
    PLANNED = "planned"
    READY = "ready"
    RUNNING = "running"
    DRAINED = "drained"
    MANIFEST_CANDIDATE = "manifest_candidate"
    QUIESCING = "quiescing"
    EXITED = "exited"


class WorkerProtocolError(ValueError):
    """Strict framing, envelope, operation, or lifecycle violation."""


AdapterFactory = Callable[[Any], EvaluationProviderAdapter]


class EvaluatorWorker:
    """One strictly ordered evaluator-provider session worker."""

    def __init__(
        self,
        descriptor: StockDistributionDescriptor,
        staging_root: Path,
        *,
        adapter_factory: AdapterFactory | None = None,
        evidence: DistributionEvidence | None = None,
    ) -> None:
        self.descriptor = descriptor
        self.staging_root = staging_root
        self._adapter_factory = adapter_factory
        self._evidence = evidence
        self.state = WorkerState.SPAWNED
        self.max_message_bytes = _BOOTSTRAP_MAX_MESSAGE_BYTES
        self.max_collection_items = _BOOTSTRAP_MAX_COLLECTION_ITEMS
        self._seen_request_ids: set[int] = set()
        self._last_request_id = -1
        self._adapter: EvaluationProviderAdapter | None = None
        self._runtime: SessionRuntime | None = None
        self._shutdown = False

    async def dispatch(self, envelope: Any) -> dict[str, Any]:
        """Strictly dispatch one request envelope and return its result value."""
        if not isinstance(envelope, dict):
            raise WorkerProtocolError("evaluator-worker request must be an object")
        operation = envelope.get("op")
        request_id = envelope.get("id")
        if not isinstance(operation, str):
            raise WorkerProtocolError("evaluator-worker request omitted string op")
        request_id = require_non_negative_int(request_id, "request id")
        if request_id in self._seen_request_ids or request_id <= self._last_request_id:
            raise WorkerProtocolError(
                "duplicate or regressing evaluator-worker request id"
            )
        self._seen_request_ids.add(request_id)
        self._last_request_id = request_id
        handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "hello": self._hello,
            "plan_session": self._plan_session,
            "bind_assets": self._bind_assets,
            "next_units": self._next_units,
            "instantiate_units": self._instantiate_units,
            "start_units": self._start_units,
            "poll_events": self._poll_events,
            "submit_host_events": self._submit_host_events,
            "cancel_units": self._cancel_units,
            "finalize_session": self._finalize_session,
            "shutdown": self._shutdown_worker,
        }
        handler = handlers.get(operation)
        if handler is None:
            raise WorkerProtocolError(
                f"unknown evaluator-worker operation {operation!r}"
            )
        result = handler(envelope)
        if asyncio.iscoroutine(result):
            result = await result
        if not isinstance(result, dict):
            raise RuntimeError("worker operation handler returned a non-object result")
        return result

    def _hello(self, envelope: dict[str, Any]) -> dict[str, Any]:
        self._require_state(WorkerState.SPAWNED)
        item = strict_object(
            envelope,
            field_name="hello request",
            allowed={
                "op",
                "id",
                "protocol",
                "max_message_bytes",
                "max_collection_items",
                "launch_nonce",
            },
            required={
                "op",
                "id",
                "protocol",
                "max_message_bytes",
                "max_collection_items",
                "launch_nonce",
            },
        )
        if item["protocol"] != 2:
            raise WorkerProtocolError(
                "unsupported evaluator-worker protocol (expected v2)"
            )
        maximum = require_positive_int(item["max_message_bytes"], "max_message_bytes")
        collections = require_positive_int(
            item["max_collection_items"], "max_collection_items"
        )
        if maximum > _BOOTSTRAP_MAX_MESSAGE_BYTES:
            raise WorkerProtocolError(
                "requested message bound exceeds worker hard ceiling"
            )
        if collections > _BOOTSTRAP_MAX_COLLECTION_ITEMS:
            raise WorkerProtocolError(
                "requested collection bound exceeds worker hard ceiling"
            )
        nonce = item["launch_nonce"]
        if not isinstance(nonce, str) or len(nonce.encode("utf-8")) < 32:
            raise WorkerProtocolError("launch_nonce must contain at least 32 bytes")
        evidence = self._evidence or self.descriptor.verify_installed_closure()
        self._evidence = evidence
        identity = self.descriptor.worker_identity(nonce, evidence)
        self._adapter = (
            self._adapter_factory(identity)
            if self._adapter_factory
            else _stock_adapter(self.descriptor.provider_id, identity)
        )
        self.max_message_bytes = maximum
        self.max_collection_items = collections
        self.state = WorkerState.NEGOTIATED
        return identity.to_wire()

    def _plan_session(self, envelope: dict[str, Any]) -> dict[str, Any]:
        self._require_state(WorkerState.NEGOTIATED)
        item = strict_object(
            envelope,
            field_name="plan_session request",
            allowed={"op", "id", "request"},
            required={"op", "id", "request"},
        )
        assert self._adapter is not None
        plan = self._adapter.plan_session(
            EvaluationPlanRequest.from_wire(item["request"])
        )
        self.state = WorkerState.PLANNED
        return plan.to_wire()

    async def _bind_assets(self, envelope: dict[str, Any]) -> dict[str, Any]:
        self._require_state(WorkerState.PLANNED)
        item = strict_object(
            envelope,
            field_name="bind_assets request",
            allowed={"op", "id", "assets", "proxy", "host_binding"},
            required={"op", "id", "assets", "host_binding"},
        )
        raw_assets = _bounded_array(item["assets"], "assets", self.max_collection_items)
        assets = tuple(ResolvedAsset.from_wire(asset) for asset in raw_assets)
        if len({asset.asset_id for asset in assets}) != len(assets):
            raise WorkerProtocolError("bind_assets contains duplicate asset IDs")
        proxy = (
            None
            if item.get("proxy") is None
            else ScopedProxyBinding.from_wire(item["proxy"])
        )
        host_binding = EvaluationHostBinding.from_wire(item["host_binding"])
        assert self._adapter is not None
        session = await self._adapter.bind_assets(
            assets=assets,
            proxy=proxy,
            host_binding=host_binding,
            staging_root=self.staging_root,
        )
        self._runtime = SessionRuntime(session)
        self.state = WorkerState.READY
        return {"identity": session.identity.to_wire()}

    async def _next_units(self, envelope: dict[str, Any]) -> dict[str, Any]:
        runtime = self._require_runtime_state()
        item = strict_object(
            envelope,
            field_name="next_units request",
            allowed={"op", "id", "offset", "limit"},
            required={"op", "id", "offset", "limit"},
        )
        page = await runtime.next_units(
            require_non_negative_int(item["offset"], "offset"),
            require_positive_int(item["limit"], "limit"),
        )
        return page.to_wire()

    async def _instantiate_units(self, envelope: dict[str, Any]) -> dict[str, Any]:
        runtime = self._require_runtime_state()
        item = strict_object(
            envelope,
            field_name="instantiate_units request",
            allowed={"op", "id", "requests"},
            required={"op", "id", "requests"},
        )
        raw_requests = _bounded_array(
            item["requests"], "requests", self.max_collection_items
        )
        requests = tuple(
            UnitOccurrenceRequest.from_wire(value) for value in raw_requests
        )
        units = await runtime.instantiate_units(requests)
        return {"items": [unit.to_wire() for unit in units]}

    async def _start_units(self, envelope: dict[str, Any]) -> dict[str, Any]:
        runtime = self._require_runtime_state()
        item = strict_object(
            envelope,
            field_name="start_units request",
            allowed={"op", "id", "unit_ids"},
            required={"op", "id", "unit_ids"},
        )
        unit_ids = _opaque_id_array(
            item["unit_ids"], "unit_ids", self.max_collection_items
        )
        started = await runtime.start_units(unit_ids)
        self.state = WorkerState.RUNNING
        return {"started": list(started)}

    async def _poll_events(self, envelope: dict[str, Any]) -> dict[str, Any]:
        runtime = self._require_runtime_state()
        item = strict_object(
            envelope,
            field_name="poll_events request",
            allowed={"op", "id", "limit", "wait_ms"},
            required={"op", "id", "limit", "wait_ms"},
        )
        events, next_sequence, drained, credits = await runtime.poll_events(
            require_positive_int(item["limit"], "limit"),
            require_non_negative_int(item["wait_ms"], "wait_ms"),
        )
        if drained:
            self.state = WorkerState.DRAINED
        return {
            "events": list(events),
            "next_sequence": next_sequence,
            "drained": drained,
            "remaining_credits": credits.to_wire(),
        }

    async def _submit_host_events(self, envelope: dict[str, Any]) -> dict[str, Any]:
        runtime = self._require_runtime_state()
        item = strict_object(
            envelope,
            field_name="submit_host_events request",
            allowed={"op", "id", "events"},
            required={"op", "id", "events"},
        )
        raw_events = _bounded_array(item["events"], "events", self.max_collection_items)
        events = tuple(HostOperationEvent.from_wire(value) for value in raw_events)
        accepted = await runtime.submit_host_events(events)
        if runtime.is_drained:
            self.state = WorkerState.DRAINED
        return {"accepted": list(accepted)}

    async def _cancel_units(self, envelope: dict[str, Any]) -> dict[str, Any]:
        runtime = self._require_runtime_state()
        item = strict_object(
            envelope,
            field_name="cancel_units request",
            allowed={"op", "id", "unit_ids"},
            required={"op", "id", "unit_ids"},
        )
        unit_ids = _opaque_id_array(
            item["unit_ids"], "unit_ids", self.max_collection_items
        )
        cancelled = await runtime.cancel_units(unit_ids)
        return {"cancelled": list(cancelled)}

    async def _finalize_session(self, envelope: dict[str, Any]) -> dict[str, Any]:
        self._require_state(WorkerState.DRAINED)
        strict_object(
            envelope,
            field_name="finalize_session request",
            allowed={"op", "id"},
            required={"op", "id"},
        )
        assert self._runtime is not None
        candidate = await self._runtime.finalize()
        self.state = WorkerState.MANIFEST_CANDIDATE
        return candidate.to_wire()

    async def _shutdown_worker(self, envelope: dict[str, Any]) -> dict[str, Any]:
        self._require_state(WorkerState.MANIFEST_CANDIDATE)
        strict_object(
            envelope,
            field_name="shutdown request",
            allowed={"op", "id"},
            required={"op", "id"},
        )
        assert self._runtime is not None
        self.state = WorkerState.QUIESCING
        await self._runtime.close()
        self._shutdown = True
        self.state = WorkerState.EXITED
        return {"shutdown": True}

    def _require_runtime_state(self) -> SessionRuntime:
        if self.state not in {
            WorkerState.READY,
            WorkerState.RUNNING,
            WorkerState.DRAINED,
        }:
            raise WorkerProtocolError(
                f"operation is invalid in evaluator-worker state {self.state.value}"
            )
        assert self._runtime is not None
        return self._runtime

    def _require_state(self, state: WorkerState) -> None:
        if self.state is not state:
            raise WorkerProtocolError(
                f"operation requires evaluator-worker state {state.value}; current={self.state.value}"
            )


async def serve_worker(
    reader: BinaryIO,
    writer: BinaryIO,
    worker: EvaluatorWorker,
) -> None:
    """Serve strict bounded JSONL until graceful shutdown or first violation."""
    loop = asyncio.get_running_loop()
    stream_reader = asyncio.StreamReader(limit=_BOOTSTRAP_MAX_MESSAGE_BYTES + 2)
    protocol = asyncio.StreamReaderProtocol(stream_reader)
    transport, _ = await loop.connect_read_pipe(lambda: protocol, reader)
    try:
        while not worker._shutdown:
            try:
                raw = await stream_reader.readline()
            except ValueError as error:
                raise WorkerProtocolError(
                    "evaluator-worker request exceeded the hard framing bound"
                ) from error
            if raw == b"":
                if worker.state is not WorkerState.EXITED:
                    raise WorkerProtocolError(
                        "evaluator control pipe closed before shutdown"
                    )
                return
            if len(raw) > worker.max_message_bytes + 1 or not raw.endswith(b"\n"):
                raise WorkerProtocolError(
                    "evaluator-worker request exceeded bound or lacked newline"
                )
            request_id: int | None = None
            try:
                request = canonical_loads(raw[:-1], max_bytes=worker.max_message_bytes)
                _validate_collection_bounds(request, worker.max_collection_items)
                if isinstance(request, dict):
                    candidate_id = request.get("id")
                    if isinstance(candidate_id, int) and not isinstance(
                        candidate_id, bool
                    ):
                        request_id = candidate_id
                result = await worker.dispatch(request)
                response: dict[str, Any] = {
                    "id": request_id,
                    "ok": True,
                    "result": result,
                }
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                response = {
                    "id": request_id,
                    "ok": False,
                    "error": _safe_error(error, worker.state).to_wire(),
                }
                await _write_response(writer, response, worker.max_message_bytes)
                raise
            await _write_response(writer, response, worker.max_message_bytes)
    finally:
        transport.close()


async def _write_response(
    writer: BinaryIO, response: dict[str, Any], max_message_bytes: int
) -> None:
    payload = canonical_dumps(response) + b"\n"
    if len(payload) > max_message_bytes:
        raise WorkerProtocolError("evaluator-worker response exceeds negotiated bound")
    remaining = memoryview(payload)
    while remaining:
        written = writer.write(remaining)
        if not isinstance(written, int) or written <= 0:
            raise WorkerProtocolError("evaluator-worker control pipe write failed")
        remaining = remaining[written:]
    writer.flush()


def _stock_adapter(provider_id: str, worker_identity: Any) -> EvaluationProviderAdapter:
    if provider_id == "nemo_evaluator":
        from aiperf.accuracy.evaluation.providers.nemo_evaluator import (
            NemoEvaluatorAdapter,
        )

        return NemoEvaluatorAdapter(worker_identity)
    if provider_id == "openbench":
        from aiperf.accuracy.evaluation.providers.openbench import OpenBenchAdapter

        return OpenBenchAdapter(worker_identity)
    raise WorkerProtocolError("selected stock provider has no adapter factory")


def _safe_error(error: BaseException, state: WorkerState) -> EvaluationError:
    text = str(error)
    lowered = text.lower()
    markers = (
        "http://",
        "https://",
        "authorization",
        "bearer ",
        "api_key",
        "token=",
        "password",
        "credential",
        "secret",
    )
    if any(marker in lowered for marker in markers):
        text = "[redacted sensitive evaluator diagnostic]"
    text = text.encode("utf-8")[:2048].decode("utf-8", errors="ignore")
    return EvaluationError(
        stage=state.value,
        error_kind="protocol_error"
        if isinstance(error, (ValueError, TypeError))
        else "provider_error",
        retryable=False,
        message=text or type(error).__name__,
    )


def _bounded_array(value: Any, name: str, maximum: int) -> list[Any]:
    if not isinstance(value, list) or len(value) > maximum:
        raise WorkerProtocolError(f"{name} must be an array within negotiated bounds")
    return value


def _opaque_id_array(value: Any, name: str, maximum: int) -> tuple[str, ...]:
    items = _bounded_array(value, name, maximum)
    if not items:
        raise WorkerProtocolError(f"{name} must not be empty")
    resolved = tuple(
        item
        for item in items
        if isinstance(item, str) and item and item == item.strip()
    )
    if len(resolved) != len(items) or len(set(resolved)) != len(resolved):
        raise WorkerProtocolError(f"{name} contains malformed or duplicate IDs")
    return resolved


def _validate_collection_bounds(value: Any, maximum: int) -> None:
    if isinstance(value, dict):
        if len(value) > maximum:
            raise WorkerProtocolError("request object exceeds collection bound")
        for child in value.values():
            _validate_collection_bounds(child, maximum)
    elif isinstance(value, list):
        if len(value) > maximum:
            raise WorkerProtocolError("request array exceeds collection bound")
        for child in value:
            _validate_collection_bounds(child, maximum)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIPerf evaluator-provider worker")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--read-fd", type=int, default=3)
    parser.add_argument("--write-fd", type=int, default=4)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="explicit standalone/conformance mode; forbidden for production launches",
    )
    return parser.parse_args(argv)


def _open_control_streams(args: argparse.Namespace) -> tuple[BinaryIO, BinaryIO]:
    if args.stdio:
        return sys.stdin.buffer, sys.stdout.buffer
    if (
        args.read_fd == args.write_fd
        or args.read_fd in {0, 1, 2}
        or args.write_fd in {0, 1, 2}
    ):
        raise WorkerProtocolError(
            "production control descriptors must be distinct from stdio"
        )
    for descriptor in (args.read_fd, args.write_fd):
        os.set_inheritable(descriptor, False)
    reader = os.fdopen(os.dup(args.read_fd), "rb", buffering=0)
    writer = os.fdopen(os.dup(args.write_fd), "wb", buffering=0)
    os.set_inheritable(reader.fileno(), False)
    os.set_inheritable(writer.fileno(), False)
    # Reserve stdout at descriptor level before any provider import.
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1, inheritable=False)
    finally:
        os.close(devnull)
    return reader, writer


def _validate_staging_root(path: Path) -> Path:
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise WorkerProtocolError(
            "staging root must be an existing absolute real directory"
        )
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise WorkerProtocolError("staging root must already be normalized")
    return resolved


def main(argv: list[str] | None = None) -> None:
    """Run one evaluator worker until quiescent shutdown."""
    args = _parse_args(argv)
    descriptor = selected_descriptor(args.provider, args.distribution)
    staging_root = _validate_staging_root(args.staging_root)
    reader, writer = _open_control_streams(args)
    worker = EvaluatorWorker(descriptor, staging_root)
    asyncio.run(serve_worker(reader, writer, worker))


if __name__ == "__main__":
    main()
