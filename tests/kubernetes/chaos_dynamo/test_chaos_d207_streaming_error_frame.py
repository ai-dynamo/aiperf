# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D207 -- assert Dynamo emits an SSE error frame before [DONE].

D-series catalog reference: D2xx (frontend/HTTP streaming faults).

Targets the mid-stream error path in ``lib/llm/src/http/service/disconnect.rs``:
when an upstream decode worker dies after a streaming request has started, the
OpenAI-compatible SSE stream must emit one JSON error frame followed by
``data: [DONE]``. A raw disconnect, a silent EOF, or ``[DONE]`` without the
error JSON frame is a contract break for clients that rely on structured
streaming errors.
"""

from __future__ import annotations

import asyncio

import aiohttp
import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


_CLIENT_ERROR_BUDGET_S: float = 35.0
"""Seconds allowed for the stream to emit the structured error after pod kill."""

_STREAM_START_TIMEOUT_S: float = 10.0
"""Seconds allowed for the first SSE data frame before fault injection."""

_DECODE_POD_SELECTOR: str = "nvidia.com/dynamo-sub-component-type=decode"
"""Label selector for decode-role worker pods in Dynamo disaggregated topology."""


async def _resolve_decode_pod(kubectl: KubectlClient, namespace: str) -> str:
    """Return the first decode pod selected by the Dynamo operator label.

    Raises:
        RuntimeError: When no decode pod is present in ``namespace``.
    """
    pod_res = await kubectl.run(
        "get",
        "pod",
        "-n",
        namespace,
        "-l",
        _DECODE_POD_SELECTOR,
        "-o",
        "jsonpath={.items[0].metadata.name}",
        check=True,
    )
    decode_pod = pod_res.stdout.strip()
    if not decode_pod:
        raise RuntimeError(
            f"D207: no decode pod found in {namespace!r} matching "
            f"{_DECODE_POD_SELECTOR!r}; cannot inject mid-stream worker death"
        )
    return decode_pod


def _append_sse_data_frames(
    frames: list[str],
    buffered_text: str,
    first_frame_seen: asyncio.Event,
) -> str:
    """Append complete SSE ``data:`` lines and return the incomplete suffix."""
    while "\n" in buffered_text:
        raw_line, buffered_text = buffered_text.split("\n", 1)
        line = raw_line.rstrip("\r")
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        frames.append(payload)
        if payload != "[DONE]":
            first_frame_seen.set()
    return buffered_text


async def _read_sse_frames(
    session: aiohttp.ClientSession,
    url: str,
    request_body: dict[str, object],
    frames: list[str],
    first_frame_seen: asyncio.Event,
) -> tuple[int | None, str | None]:
    """Read SSE data frames until EOF and return ``(status, exception_repr)``."""
    buffered_text = ""
    try:
        async with session.post(url, json=request_body) as resp:
            status = resp.status
            if status != 200:
                body = await resp.text()
                frames.append(f"<HTTP {status}: {body}>")
                return status, None
            async for chunk in resp.content.iter_any():
                buffered_text += chunk.decode("utf-8", errors="replace")
                buffered_text = _append_sse_data_frames(
                    frames,
                    buffered_text,
                    first_frame_seen,
                )
            if buffered_text.strip():
                frames.append(f"<TRAILING_BYTES: {buffered_text.strip()}>")
            return status, None
    except (
        aiohttp.ClientError,
        aiohttp.ServerDisconnectedError,
        asyncio.TimeoutError,
    ) as exc:
        return None, repr(exc)


def _format_observed_sequence(frames: list[str], exception_repr: str | None) -> str:
    """Return the exact observed stream sequence for assertion messages."""
    suffix = f", exception={exception_repr}" if exception_repr else ""
    return f"frames={frames!r}{suffix}"


def _error_payload(frame: str) -> dict[str, object] | None:
    """Decode one SSE data payload and return its ``error`` object if present."""
    try:
        payload = orjson.loads(frame)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    return error if isinstance(error, dict) else None


def _assert_error_frame_then_done(
    frames: list[str],
    exception_repr: str | None,
) -> None:
    """Assert the D207 two-frame terminal contract and include observed frames."""
    observed = _format_observed_sequence(frames, exception_repr)
    assert exception_repr is None, (
        "D207: stream raised before receiving the required error JSON frame "
        f"followed by [DONE]; observed {observed}"
    )

    error_index = next(
        (idx for idx, frame in enumerate(frames) if _error_payload(frame) is not None),
        None,
    )
    assert error_index is not None, (
        "D207: stream did not contain an error JSON SSE frame before [DONE]; "
        f"observed {observed}"
    )
    assert error_index + 1 < len(frames) and frames[error_index + 1] == "[DONE]", (
        "D207: error JSON SSE frame was not immediately followed by [DONE]; "
        f"observed {observed}"
    )

    error = _error_payload(frames[error_index])
    assert error is not None
    assert isinstance(error.get("message"), str) and error["message"], (
        f"D207: error JSON frame missing non-empty error.message; observed {observed}"
    )
    assert error.get("type") == "internal_server_error", (
        "D207: error JSON frame has wrong error.type; expected "
        f"'internal_server_error'; observed {observed}"
    )
    assert error.get("code") == 500, (
        "D207: error JSON frame has wrong error.code; expected 500; "
        f"observed {observed}"
    )


async def test_d207_streaming_error_frame_after_decode_kill(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Kill a decode worker mid-stream and assert error JSON then ``[DONE]``.

    Steps:
        1. Open one streaming POST to ``/chat/completions`` with enough output
           budget to stay in-flight.
        2. Wait for the first SSE data frame to prove decode generation began.
        3. Select and kill a decode pod via the same label path used by D401.
        4. Drain the stream and assert the terminal sequence is the Dynamo
           structured error frame followed immediately by ``[DONE]``.
    """
    request_body: dict[str, object] = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a detailed, 800-word incident report about a "
                    "database failover and include numbered remediation steps."
                ),
            }
        ],
        "max_tokens": 512,
        "stream": True,
        "temperature": 0.0,
    }
    frames: list[str] = []
    first_frame_seen = asyncio.Event()
    timeout = aiohttp.ClientTimeout(total=_CLIENT_ERROR_BUDGET_S + 30.0)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        stream_task = asyncio.create_task(
            _read_sse_frames(
                session,
                f"{dynamo_endpoint_url}/chat/completions",
                request_body,
                frames,
                first_frame_seen,
            )
        )
        try:
            await asyncio.wait_for(
                first_frame_seen.wait(),
                timeout=_STREAM_START_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            stream_task.cancel()
            pytest.fail(
                "D207: no SSE data frame received within "
                f"{_STREAM_START_TIMEOUT_S}s; observed "
                f"{_format_observed_sequence(frames, None)}"
            )

        try:
            decode_pod = await _resolve_decode_pod(kubectl, dynamo_deployment_namespace)
        except RuntimeError as exc:
            stream_task.cancel()
            pytest.fail(str(exc))

        async with faults.inject(
            "pod.kill",
            target={"ns": dynamo_deployment_namespace, "pod": decode_pod},
        ):
            logger.info(
                lambda p=decode_pod, ns=dynamo_deployment_namespace: (
                    f"D207: killed decode pod {ns}/{p} mid-stream"
                )
            )
            try:
                _status, exception_repr = await asyncio.wait_for(
                    stream_task,
                    timeout=_CLIENT_ERROR_BUDGET_S + 5.0,
                )
            except asyncio.TimeoutError:
                stream_task.cancel()
                pytest.fail(
                    "D207: stream did not terminate with error JSON then [DONE] "
                    f"within {_CLIENT_ERROR_BUDGET_S + 5.0}s of decode-pod kill; "
                    f"observed {_format_observed_sequence(frames, None)}"
                )

    _assert_error_frame_then_done(frames, exception_repr)
